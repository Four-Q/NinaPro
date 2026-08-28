"""只读检查 T=40 神经形态脉冲数据与训练环境。"""

from pathlib import Path

import numpy as np


DATA_DIR = Path(
    "/root/autodl-tmp/NinaPro/ninapro_data/"
    "neurophic_system_spikes/offset_128/T_40"
)


def format_counts(values):
    unique, counts = np.unique(values, return_counts=True)
    return {int(key): int(count) for key, count in zip(unique, counts)}


def inspect_split(name):
    path = DATA_DIR / f"{name}.npz"
    with np.load(path, allow_pickle=False) as data:
        print(f"\n[{name}] path={path}")
        print(f"fields={data.files}")
        for key in data.files:
            array = data[key]
            print(
                f"{key}: shape={array.shape}, dtype={array.dtype}, "
                f"min={array.min()}, max={array.max()}, mean={array.mean():.6f}"
            )

        print(f"class_counts={format_counts(data['y'])}")
        print(f"subject_counts={format_counts(data['subject'])}")
        print(f"repetition_counts={format_counts(data['repetition'])}")

        # 类别-通道发放率能快速判断简单统计特征是否已有较强可分性。
        channel_rates = data["X"].mean(axis=2)
        class_centroids = np.stack(
            [channel_rates[data["y"] == label].mean(axis=0) for label in range(12)]
        )
        print("class_channel_rate_centroids=")
        print(np.array2string(class_centroids, precision=4, suppress_small=True))

        identifiers = np.stack(
            [data["subject"], data["repetition"], data["start_sample"]], axis=1
        )
        return {
            "subjects": set(map(int, np.unique(data["subject"]))),
            "repetitions": set(map(int, np.unique(data["repetition"]))),
            "identifiers": {tuple(map(int, row)) for row in identifiers},
        }


def inspect_environment():
    import torch
    import spikingjelly

    print("[environment]")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"numpy={np.__version__}")
    print(f"spikingjelly={getattr(spikingjelly, '__version__', 'unknown')}")


def main():
    inspect_environment()
    train_info = inspect_split("train")
    test_info = inspect_split("test")
    print("\n[split_relationship]")
    print(f"shared_subjects={sorted(train_info['subjects'] & test_info['subjects'])}")
    print(
        "shared_repetitions="
        f"{sorted(train_info['repetitions'] & test_info['repetitions'])}"
    )
    print(
        "exact_identifier_overlap="
        f"{len(train_info['identifiers'] & test_info['identifiers'])}"
    )


if __name__ == "__main__":
    main()
