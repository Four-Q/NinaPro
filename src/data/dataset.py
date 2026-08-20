"""NinaPro 滑动窗口数据集。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NinaProWindowDataset(Dataset):
    """从一个 NPZ 文件读取 NinaPro 滑动窗口样本。

    每个样本的输入形状为 ``[C, T]``，标签为标量。DataLoader 拼接后
    得到 ``[B, C, T]``，可同时供 ANN 和 SNN 使用。
    """

    def __init__(self, npz_path, transform=None):
        self.path = Path(npz_path).expanduser().resolve()
        self.transform = transform

        if not self.path.is_file():
            raise FileNotFoundError(f"找不到数据文件：{self.path}")

        with np.load(self.path, allow_pickle=False) as data:
            missing_keys = {"X", "y"}.difference(data.files)
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise KeyError(f"{self.path} 缺少必需数组：{missing}")

            # 拷贝出 NPZ 容器，避免文件关闭后仍持有归档内部数组。
            features = np.asarray(data["X"], dtype=np.float32).copy()
            labels = np.asarray(data["y"], dtype=np.int64).copy()

        self._validate_arrays(features, labels)
        self.features = torch.from_numpy(features).contiguous()
        self.labels = torch.from_numpy(labels).contiguous()

    @staticmethod
    def _validate_arrays(features, labels):
        if features.ndim != 3:
            raise ValueError(
                "X 必须具有 [样本数, 通道数, 时间点数] 三个维度，"
                f"实际形状为 {features.shape}"
            )
        if labels.ndim != 1:
            raise ValueError(f"y 必须是一维标签数组，实际形状为 {labels.shape}")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                "X 与 y 的样本数不一致："
                f"{features.shape[0]} != {labels.shape[0]}"
            )
        if features.shape[0] == 0:
            raise ValueError("数据集不能为空")
        if not np.isfinite(features).all():
            raise ValueError("X 中存在 NaN 或无穷值")

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        # clone 可防止未来的器件映射原地修改底层训练数据。
        x = self.features[index].clone()
        y = self.labels[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    @property
    def num_channels(self):
        return self.features.shape[1]

    @property
    def window_size(self):
        return self.features.shape[2]
