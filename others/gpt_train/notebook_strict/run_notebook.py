"""严格复用 src/main notebook 数据与训练流程的命令行入口。"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path("/root/autodl-tmp/NinaPro")
HERE = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import NinaProWindowDataset
from src.training import fit, resolve_device, seed_everything

from models import build_model


CLASS_NAMES = [
    "Index flexion",
    "Index extension",
    "Middle flexion",
    "Middle extension",
    "Ring flexion",
    "Ring extension",
    "Little flexion",
    "Little extension",
    "Thumb adduction",
    "Thumb abduction",
    "Thumb flexion",
    "Thumb extension",
]
DATA_DIR = (
    PROJECT_ROOT
    / "ninapro_data"
    / "neurophic_system_spikes"
    / "offset_128"
    / "T_40"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("adaptive_moe_tcn", "wide_resnet2d"),
        default="adaptive_moe_tcn",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--dropout-rate", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--eval-smoothing-radius", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, deterministic=False)
    device = resolve_device()

    # 以下数据代码逐项保持原 notebook 设置，仅输出目录位于本实验子目录。
    train_dataset = NinaProWindowDataset(DATA_DIR / "train.npz", transform=None)
    test_dataset = NinaProWindowDataset(DATA_DIR / "test.npz", transform=None)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
        **loader_options,
    )

    model_options = {
        "input_channels": 16,
        "num_classes": 12,
        "dropout_rate": args.dropout_rate,
        "eval_smoothing_radius": args.eval_smoothing_radius,
    }
    if args.width is not None:
        model_options["width"] = args.width
    model = build_model(args.model, **model_options)
    output_dir = HERE / "runs" / args.run_name
    config = {
        "strict_notebook_compatible": True,
        "unchanged_input_field": "X",
        "unchanged_input_layout": "[B, C, T]",
        "unchanged_data_dir": str(DATA_DIR),
        "model_name": args.model,
        "model_options": model_options,
        "hyperparameters": vars(args),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strict_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(model)
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")

    fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        output_dir=output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        device=device,
        mixed_precision=True,
        normalization_state=None,
        config=config,
        resume_from=None,
        class_names=CLASS_NAMES,
        warmup_epochs=args.warmup_epochs,
        eval_interval=1,
        checkpoint_interval=5,
        fast_mode=True,
        prefer_cupy=True,
        gpu_resident_data=True,
    )


if __name__ == "__main__":
    main()
