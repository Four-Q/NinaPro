"""训练集与测试集 DataLoader 构造入口。"""

import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import NinaProWindowDataset
from .transforms import (
    ChannelwiseZScore,
    ComposeTransforms,
    PhysicalDeviceMapping,
    compute_channel_stats,
)


class NinaProDataPipeline:
    """集中保存 DataLoader 及可复用的归一化状态。"""

    def __init__(self, train_loader, test_loader, normalization):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.normalization = normalization

    def normalization_state_dict(self):
        return self.normalization.state_dict()


def resolve_num_workers(num_workers=None):
    """解析跨平台 DataLoader worker 数量。

    Windows 默认使用主进程，避免 spawn 带来的数据复制和入口限制。Linux
    使用可用 CPU 核心数的一半并最多启用 8 个 worker；这是适合轻量内存
    数据集的保守起点，仍可通过显式参数按机器实测结果覆盖。
    """

    if num_workers is not None:
        if (
            not isinstance(num_workers, int)
            or isinstance(num_workers, bool)
            or num_workers < 0
        ):
            raise ValueError("num_workers 必须是非负整数或 None")
        return num_workers

    if platform.system() == "Windows":
        return 0

    try:
        # Linux 容器或任务调度器可能只分配部分核心，优先遵守 CPU affinity。
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 1

    if cpu_count <= 2:
        return 0
    return min(8, max(1, cpu_count // 2))


def seed_worker(worker_id):
    """让 Python 与 NumPy 的 worker 随机状态跟随 PyTorch 种子。"""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_datasets(
    data_dir,
    physical_mapping=None,
    normalization_state=None,
    eps=1e-6,
    stats_chunk_size=1024,
):
    """创建训练、测试 Dataset，并返回二者共享的归一化 Transform。"""

    data_dir = Path(data_dir).expanduser().resolve()
    train_dataset = NinaProWindowDataset(data_dir / "train.npz")

    if physical_mapping is None:
        physical_mapping = PhysicalDeviceMapping()
    if not callable(physical_mapping):
        raise TypeError("physical_mapping 必须是可调用的 Transform")

    if normalization_state is None:
        mean, std = compute_channel_stats(
            train_dataset.features,
            pre_transform=physical_mapping,
            chunk_size=stats_chunk_size,
        )
        normalization = ChannelwiseZScore(mean, std, eps=eps)
    else:
        normalization = ChannelwiseZScore.from_state_dict(normalization_state)

    # 顺序不可交换：未来的物理器件输出也必须使用其对应的训练集统计量。
    input_transform = ComposeTransforms([physical_mapping, normalization])
    train_dataset.transform = input_transform
    test_dataset = NinaProWindowDataset(
        data_dir / "test.npz",
        transform=input_transform,
    )
    return train_dataset, test_dataset, normalization


def create_data_pipeline(
    data_dir,
    batch_size=64,
    num_workers=None,
    pin_memory=None,
    seed=42,
    drop_last=False,
    physical_mapping=None,
    normalization_state=None,
    eps=1e-6,
    stats_chunk_size=1024,
):
    """创建包含 train/test DataLoader 与归一化状态的数据管道。"""

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed 必须是整数")

    worker_count = resolve_num_workers(num_workers)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_dataset, test_dataset, normalization = create_datasets(
        data_dir=data_dir,
        physical_mapping=physical_mapping,
        normalization_state=normalization_state,
        eps=eps,
        stats_chunk_size=stats_chunk_size,
    )

    train_generator = torch.Generator().manual_seed(seed)
    test_generator = torch.Generator().manual_seed(seed + 1)
    shared_options = {
        "batch_size": batch_size,
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory),
        "worker_init_fn": seed_worker,
    }
    if worker_count > 0:
        shared_options["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=bool(drop_last),
        generator=train_generator,
        **shared_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        generator=test_generator,
        **shared_options,
    )
    return NinaProDataPipeline(train_loader, test_loader, normalization)


def create_dataloaders(*args, **kwargs):
    """便捷接口：仅返回 ``(train_loader, test_loader)``。"""

    pipeline = create_data_pipeline(*args, **kwargs)
    return pipeline.train_loader, pipeline.test_loader
