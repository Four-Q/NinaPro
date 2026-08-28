"""只读检查 X 的数值范围和训练/测试分布。"""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/root/autodl-tmp/NinaPro")
DATA_DIR = (
    PROJECT_ROOT
    / "ninapro_data"
    / "neurophic_system_spikes"
    / "offset_128"
    / "T_40"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def summarize(name):
    with np.load(DATA_DIR / f"{name}.npz", allow_pickle=False) as data:
        x = data["X"]
        y = data["y"]
        flat = x.reshape(x.shape[0], -1)
        print(
            {
                "split": name,
                "shape": x.shape,
                "dtype": str(x.dtype),
                "min": float(x.min()),
                "max": float(x.max()),
                "mean": float(x.mean()),
                "sample_activity_quantiles": np.quantile(
                    np.abs(flat).mean(axis=1), [0, 0.1, 0.5, 0.9, 1]
                ).tolist(),
                "class_counts": np.bincount(y.astype(np.int64), minlength=12).tolist(),
            }
        )


if __name__ == "__main__":
    summarize("train")
    summarize("test")
