"""模型训练、逐轮测试和单批次诊断。"""

import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .metrics import ClassificationMeter
from .reproducibility import resolve_device
from .visualization import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_training_history,
    save_json,
)


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    num_classes,
    gradient_clip=1.0,
    mixed_precision=False,
    scaler=None,
    progress=True,
    description="Train",
):
    """训练一个 epoch 并返回训练指标。"""

    model.train()
    meter = ClassificationMeter(num_classes)
    amp_enabled = bool(mixed_precision and device.type == "cuda")

    progress_bar = tqdm(
        data_loader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
        disable=not progress,
    )
    for inputs, targets in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _clip_gradients(model, gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            _clip_gradients(model, gradient_clip)
            optimizer.step()

        meter.update(logits, targets, loss)
        progress_bar.set_postfix(_running_postfix(meter))

    return meter.compute()


def evaluate(
    model,
    data_loader,
    criterion,
    device=None,
    num_classes=12,
    mixed_precision=False,
    progress=True,
    description="Test",
):
    """在指定数据集上计算损失、分类指标和混淆矩阵。"""

    device = resolve_device(device)
    model.eval()
    meter = ClassificationMeter(num_classes)
    amp_enabled = bool(mixed_precision and device.type == "cuda")

    with torch.no_grad():
        progress_bar = tqdm(
            data_loader,
            desc=description,
            leave=False,
            dynamic_ncols=True,
            disable=not progress,
        )
        for inputs, targets in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with _autocast_context(device, amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
            meter.update(logits, targets, loss)
            progress_bar.set_postfix(_running_postfix(meter))

    return meter.compute()


def fit(
    model,
    train_loader,
    test_loader,
    output_dir,
    epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    gradient_clip=1.0,
    device=None,
    mixed_precision=False,
    normalization_state=None,
    config=None,
    resume_from=None,
    class_names=None,
    verbose=True,
):
    """训练模型，每个 epoch 测试并按测试准确率保存最佳检查点。

    测试集参与最佳模型选择，因此 ``best.pt`` 是 test-selected checkpoint，
    其测试指标不能解释为完全无偏的最终泛化估计。
    """

    _validate_fit_options(epochs, learning_rate, weight_decay, gradient_clip)
    device = resolve_device(device)
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    amp_enabled = bool(mixed_precision and device.type == "cuda")
    scaler = _create_grad_scaler(amp_enabled)
    num_classes = _infer_num_classes(model)

    training_config = {
        "epochs": epochs,
        "batch_size": getattr(train_loader, "batch_size", None),
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": "CosineAnnealingLR",
        "gradient_clip": gradient_clip,
        "mixed_precision": bool(mixed_precision),
        "device": str(device),
        "selection_split": "test",
        "test_selected_checkpoint": True,
    }
    training_config.update(dict(config or {}))

    start_epoch = 1
    history = []
    best_test_accuracy = float("-inf")
    best_test_loss = float("inf")
    if resume_from is not None:
        resumed = load_checkpoint(
            resume_from,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(resumed.get("epoch", 0)) + 1
        history = list(resumed.get("history") or [])
        best_test_accuracy = _optional_float(
            resumed.get("best_test_accuracy"),
            default=float("-inf"),
        )
        best_test_loss = _optional_float(
            resumed.get("best_test_loss"),
            default=float("inf"),
        )

    if verbose:
        print(f"device={device}, amp={amp_enabled}, epochs={start_epoch}..{epochs}")

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate_now = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            gradient_clip=gradient_clip,
            mixed_precision=amp_enabled,
            scaler=scaler,
            progress=verbose,
            description=f"Train {epoch:03d}/{epochs:03d}",
        )
        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            mixed_precision=amp_enabled,
            progress=verbose,
            description=f"Test  {epoch:03d}/{epochs:03d}",
        )
        elapsed = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "learning_rate": learning_rate_now,
            "epoch_time": elapsed,
        }
        history.append(record)

        is_best = _is_better_test_result(
            test_metrics["accuracy"],
            test_metrics["loss"],
            best_test_accuracy,
            best_test_loss,
        )
        if is_best:
            best_test_accuracy = test_metrics["accuracy"]
            best_test_loss = test_metrics["loss"]

        scheduler.step()
        checkpoint_options = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "history": history,
            "normalization_state": normalization_state,
            "config": training_config,
            "best_test_accuracy": best_test_accuracy,
            "best_test_loss": best_test_loss,
            "test_metrics": test_metrics,
        }
        save_checkpoint(checkpoint_dir / "last.pt", **checkpoint_options)
        if is_best:
            save_checkpoint(checkpoint_dir / "best.pt", **checkpoint_options)

        save_json(history, output_dir / "history.json")
        last_epoch = epoch
        if verbose:
            marker = " *" if is_best else ""
            print(
                f"epoch {epoch:03d}/{epochs:03d} | "
                f"train loss {train_metrics['loss']:.4f}, "
                f"acc {train_metrics['accuracy'] * 100:.2f}% | "
                f"test loss {test_metrics['loss']:.4f}, "
                f"acc {test_metrics['accuracy'] * 100:.2f}% | "
                f"lr {learning_rate_now:.3e} | {elapsed:.2f}s{marker}"
            )

    if last_epoch < 1:
        raise RuntimeError("没有执行任何训练 epoch，请检查 epochs 和恢复检查点")

    save_checkpoint(
        checkpoint_dir / "final.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=last_epoch,
        history=history,
        normalization_state=normalization_state,
        config=training_config,
        best_test_accuracy=best_test_accuracy,
        best_test_loss=best_test_loss,
    )

    best_path = checkpoint_dir / "best.pt"
    best_checkpoint = load_checkpoint(best_path, model, map_location=device)
    best_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        mixed_precision=amp_enabled,
        progress=verbose,
        description="Best test",
    )
    best_metrics.update(
        {
            "best_epoch": int(best_checkpoint["epoch"]),
            "selection_split": "test",
            "test_selected_checkpoint": True,
        }
    )

    save_json(best_metrics, output_dir / "metrics.json")
    plot_training_history(history, figure_dir / "training_curves.png")
    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        figure_dir / "confusion_matrix.png",
        class_names=class_names,
    )
    plot_per_class_accuracy(
        best_metrics["per_class_accuracy"],
        figure_dir / "per_class_accuracy.png",
        class_names=class_names,
    )
    return history


def overfit_one_batch(
    model,
    batch,
    max_steps=500,
    target_accuracy=0.99,
    learning_rate=1e-3,
    weight_decay=0.0,
    gradient_clip=1.0,
    device=None,
    verbose=True,
):
    """反复训练一个批次，诊断数据到梯度的完整链路。"""

    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps 必须是正整数")
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy 必须在 (0, 1] 内")

    device = resolve_device(device)
    model.to(device)
    model.train()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history = []

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        _clip_gradients(model, gradient_clip)
        optimizer.step()

        accuracy = float((logits.argmax(dim=1) == targets).float().mean())
        history.append({"step": step, "loss": float(loss.detach()), "accuracy": accuracy})
        if verbose and (step == 1 or step % 25 == 0 or accuracy >= target_accuracy):
            print(
                f"step {step:03d}/{max_steps:03d} | "
                f"loss {float(loss.detach()):.4f} | acc {accuracy * 100:.2f}%"
            )
        if accuracy >= target_accuracy:
            break
    return history


def _validate_fit_options(epochs, learning_rate, weight_decay, gradient_clip):
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs 必须是正整数")
    if learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")
    if weight_decay < 0:
        raise ValueError("weight_decay 不能小于 0")
    if gradient_clip is not None and gradient_clip <= 0:
        raise ValueError("gradient_clip 必须大于 0 或为 None")


def _infer_num_classes(model):
    num_classes = getattr(model, "num_classes", None)
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise AttributeError("model 必须提供正整数 num_classes 属性")
    return num_classes


def _clip_gradients(model, gradient_clip):
    if gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)


def _autocast_context(device, enabled):
    if enabled:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return nullcontext()


def _create_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _is_better_test_result(accuracy, loss, best_accuracy, best_loss):
    if accuracy > best_accuracy:
        return True
    return accuracy == best_accuracy and loss < best_loss


def _optional_float(value, default):
    return default if value is None else float(value)


def _running_postfix(meter):
    if meter.sample_count == 0:
        return {"loss": "-", "accuracy": "-"}
    correct = int(meter.confusion_matrix.diag().sum())
    return {
        "loss": f"{meter.loss_sum / meter.sample_count:.4f}",
        "accuracy": f"{correct / meter.sample_count * 100:.2f}%",
    }
