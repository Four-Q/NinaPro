"""在独立重复次数验证集上训练 NinaPro T=40 分类模型。"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import build_model


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(
    "/root/autodl-tmp/NinaPro/ninapro_data/"
    "neurophic_system_spikes/offset_128/T_40"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tune", "final"), default="tune")
    parser.add_argument("--model", choices=("hybrid", "temporal"), default="hybrid")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input-field", choices=("X", "Vin"), default="X")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--temporal-width", type=int, default=128)
    parser.add_argument("--spatial-width", type=int, default=48)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.1)
    parser.add_argument("--max-time-shift", type=int, default=2)
    parser.add_argument("--spike-dropout", type=float, default=0.01)
    parser.add_argument("--val-repetition", type=int, default=6)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz(path, input_field):
    with np.load(path, allow_pickle=False) as data:
        features = torch.from_numpy(
            data[input_field].astype(np.float32, copy=True)
        )
        labels = torch.from_numpy(data["y"].astype(np.int64, copy=True))
        repetitions = torch.from_numpy(
            data["repetition"].astype(np.int64, copy=True)
        )
    return features, labels, repetitions


def make_loader(features, labels, batch_size, shuffle, num_workers, seed):
    dataset = TensorDataset(features, labels)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def build_loaders(args):
    train_x, train_y, train_repetitions = load_npz(
        args.data_dir / "train.npz", args.input_field
    )
    if args.mode == "tune":
        validation_mask = train_repetitions == args.val_repetition
        if not validation_mask.any() or validation_mask.all():
            raise ValueError("验证重复次数必须同时留下非空训练集和验证集")
        fit_x = train_x[~validation_mask]
        fit_y = train_y[~validation_mask]
        validation_x = train_x[validation_mask]
        validation_y = train_y[validation_mask]
        validation_loader = make_loader(
            validation_x,
            validation_y,
            args.batch_size,
            False,
            args.num_workers,
            args.seed + 1,
        )
        evaluation_name = f"validation_repetition_{args.val_repetition}"
    else:
        fit_x = train_x
        fit_y = train_y
        test_x, test_y, _ = load_npz(
            args.data_dir / "test.npz", args.input_field
        )
        validation_loader = make_loader(
            test_x,
            test_y,
            args.batch_size,
            False,
            args.num_workers,
            args.seed + 1,
        )
        evaluation_name = "test_repetitions_2_5"

    fit_loader = make_loader(
        fit_x,
        fit_y,
        args.batch_size,
        True,
        args.num_workers,
        args.seed,
    )
    return fit_loader, validation_loader, fit_y, evaluation_name


def make_class_weights(labels, num_classes, device):
    counts = torch.bincount(labels, minlength=num_classes).float()
    # 平方根逆频率修正轻度类别不均衡，同时避免小类权重过强。
    weights = torch.sqrt(counts.mean() / counts.clamp_min(1))
    return weights.to(device)


def shift_time(inputs, max_shift):
    if max_shift <= 0:
        return inputs
    batch_size, channels, time_steps = inputs.shape
    shifts = torch.randint(
        -max_shift,
        max_shift + 1,
        (batch_size, 1),
        device=inputs.device,
    )
    source = torch.arange(time_steps, device=inputs.device).view(1, -1) - shifts
    valid = (source >= 0) & (source < time_steps)
    source = source.clamp(0, time_steps - 1)
    source = source.unsqueeze(1).expand(-1, channels, -1)
    shifted = torch.gather(inputs, dim=2, index=source)
    return shifted * valid.unsqueeze(1)


def augment_inputs(inputs, max_time_shift, spike_dropout):
    inputs = shift_time(inputs, max_time_shift)
    if spike_dropout > 0:
        keep = torch.rand_like(inputs) >= spike_dropout
        inputs = inputs * keep
    return inputs


def apply_mixup(inputs, targets, alpha):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    coefficient = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(targets.shape[0], device=targets.device)
    mixed_inputs = (
        coefficient * inputs + (1.0 - coefficient) * inputs[permutation]
    )
    return mixed_inputs, targets, targets[permutation], coefficient


def mixup_loss(logits, targets_a, targets_b, coefficient, criterion):
    return (
        coefficient * criterion(logits, targets_a)
        + (1.0 - coefficient) * criterion(logits, targets_b)
    )


def update_confusion(confusion, predictions, targets, num_classes):
    flat = targets * num_classes + predictions
    confusion += torch.bincount(flat, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def metrics_from_confusion(confusion):
    confusion = confusion.float()
    correct = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    recall = correct / support.clamp_min(1)
    precision = correct / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": float(correct.sum() / confusion.sum().clamp_min(1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_accuracy": recall.tolist(),
        "confusion_matrix": confusion.to(torch.int64).tolist(),
    }


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    args,
):
    model.train()
    confusion = torch.zeros(12, 12, dtype=torch.int64, device=device)
    total_loss = 0.0
    total_samples = 0
    amp_enabled = scaler.is_enabled()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        inputs = augment_inputs(inputs, args.max_time_shift, args.spike_dropout)
        inputs, targets_a, targets_b, coefficient = apply_mixup(
            inputs, targets, args.mixup_alpha
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(inputs)
            loss = mixup_loss(
                logits, targets_a, targets_b, coefficient, criterion
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = targets.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        update_confusion(confusion, logits.argmax(dim=1), targets, 12)

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / total_samples
    return metrics


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp_enabled):
    model.eval()
    confusion = torch.zeros(12, 12, dtype=torch.int64, device=device)
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(inputs)
            loss = criterion(logits, targets)
        batch_size = targets.shape[0]
        total_loss += float(loss) * batch_size
        total_samples += batch_size
        update_confusion(confusion, logits.argmax(dim=1), targets, 12)

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / total_samples
    return metrics


def save_json(value, path):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def model_options(args):
    options = {
        "input_channels": 16,
        "num_classes": 12,
        "temporal_width": args.temporal_width,
        "dropout_rate": args.dropout_rate,
    }
    if args.model == "hybrid":
        options["spatial_width"] = args.spatial_width
    return options


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要 CUDA GPU")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    run_name = args.run_name or (
        f"{args.mode}_{args.model}_seed{args.seed}_width{args.temporal_width}"
    )
    output_dir = HERE / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args) | {"data_dir": str(args.data_dir)}, output_dir / "config.json")

    fit_loader, evaluation_loader, fit_labels, evaluation_name = build_loaders(args)
    options = model_options(args)
    model = build_model(args.model, **options).to(device)
    fit_features = fit_loader.dataset.tensors[0]
    input_mean = fit_features.mean(dim=(0, 2)).to(device)
    input_std = fit_features.std(dim=(0, 2)).clamp_min(1e-6).to(device)
    model.set_input_normalization(input_mean, input_std)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    class_weights = make_class_weights(fit_labels, 12, device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(fit_loader),
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    amp_enabled = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(
        f"run={run_name} mode={args.mode} model={args.model} "
        f"input_field={args.input_field} "
        f"parameters={parameter_count:,} device={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    print(
        f"fit_samples={len(fit_loader.dataset):,} "
        f"evaluation_samples={len(evaluation_loader.dataset):,} "
        f"evaluation={evaluation_name}",
        flush=True,
    )

    history = []
    best_accuracy = -math.inf
    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_name = "best.pt" if args.mode == "tune" else "final.pt"
    checkpoint_path = output_dir / checkpoint_name

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            fit_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            args,
        )
        evaluation_metrics = None
        if args.mode == "tune":
            evaluation_metrics = evaluate(
                model, evaluation_loader, criterion, device, amp_enabled
            )
        elapsed = time.perf_counter() - started
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "evaluation_loss": (
                evaluation_metrics["loss"] if evaluation_metrics is not None else None
            ),
            "evaluation_accuracy": (
                evaluation_metrics["accuracy"]
                if evaluation_metrics is not None
                else None
            ),
            "evaluation_macro_f1": (
                evaluation_metrics["macro_f1"]
                if evaluation_metrics is not None
                else None
            ),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        }
        history.append(record)
        save_json(history, output_dir / "history.json")

        improved = bool(
            evaluation_metrics is not None
            and (
                evaluation_metrics["accuracy"] > best_accuracy
                or (
                    evaluation_metrics["accuracy"] == best_accuracy
                    and evaluation_metrics["loss"] < best_loss
                )
            )
        )
        if improved:
            best_accuracy = evaluation_metrics["accuracy"]
            best_loss = evaluation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "model_options": options,
                    "epoch": epoch,
                    "evaluation_name": evaluation_name,
                    "evaluation_metrics": evaluation_metrics,
                    "config": vars(args) | {"data_dir": str(args.data_dir)},
                },
                checkpoint_path,
            )
            save_json(
                {
                    "best_epoch": best_epoch,
                    "selection_split": evaluation_name,
                    **evaluation_metrics,
                },
                output_dir / "best_metrics.json",
            )
        elif args.mode == "tune":
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        message = (
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train {train_metrics['loss']:.4f}/"
            f"{train_metrics['accuracy'] * 100:.2f}%"
        )
        if evaluation_metrics is not None:
            message += (
                f" | {evaluation_name} {evaluation_metrics['loss']:.4f}/"
                f"{evaluation_metrics['accuracy'] * 100:.2f}%"
            )
        message += f" | {elapsed:.1f}s{marker}"
        print(message, flush=True)

        if args.mode == "tune" and epochs_without_improvement >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if args.mode == "tune":
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # 最终模式固定训练轮数，结束前从未读取测试集指标。
        best_epoch = len(history)
    best_metrics = evaluate(model, evaluation_loader, criterion, device, amp_enabled)
    result = {
        "run_name": run_name,
        "mode": args.mode,
        "model": args.model,
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "evaluation_name": evaluation_name,
        **best_metrics,
    }
    if args.mode == "final":
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "model_options": options,
                "epoch": best_epoch,
                "evaluation_name": evaluation_name,
                "evaluation_metrics": best_metrics,
                "config": vars(args) | {"data_dir": str(args.data_dir)},
            },
            checkpoint_path,
        )
    result_name = "validation_result.json" if args.mode == "tune" else "test_result.json"
    save_json(result, output_dir / result_name)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
