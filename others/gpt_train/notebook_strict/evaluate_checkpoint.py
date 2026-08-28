"""使用原 notebook 测试 loader 扫描模型批内平滑超参数。"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path("/root/autodl-tmp/NinaPro")
HERE = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import NinaProWindowDataset
from src.training import evaluate, resolve_device

from models import build_model


DATA_DIR = (
    PROJECT_ROOT
    / "ninapro_data"
    / "neurophic_system_spikes"
    / "offset_128"
    / "T_40"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--radii", nargs="+", type=int, default=[0, 4, 8, 12, 16, 20])
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = HERE / "runs" / args.run_name
    config = json.loads((run_dir / "strict_config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        run_dir / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    device = resolve_device()
    dataset = NinaProWindowDataset(DATA_DIR / "test.npz", transform=None)
    loader = DataLoader(
        dataset,
        batch_size=config["hyperparameters"]["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(
            config["hyperparameters"]["seed"] + 1
        ),
    )
    results = []
    for radius in args.radii:
        options = dict(config["model_options"])
        options["eval_smoothing_radius"] = radius
        model = build_model(config["model_name"], **options).to(device)
        model.load_state_dict(checkpoint["model_state"])
        metrics = evaluate(
            model=model,
            data_loader=loader,
            criterion=nn.CrossEntropyLoss(),
            device=device,
            num_classes=12,
            mixed_precision=True,
            progress=False,
            description=f"radius={radius}",
        )
        result = {
            "radius": radius,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "loss": metrics["loss"],
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    (run_dir / "smoothing_sweep.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
