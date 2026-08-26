"""训练集与测试集 DataLoader 构造入口。"""

import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

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


class DeviceBatchLoader:
    """直接从设备常驻张量产生批次，避免逐样本拼接和主机传输。"""

    def __init__(
        self,
        features,
        labels,
        source_loader,
        shuffle,
        seed,
    ):
        if source_loader.batch_size is None:
            raise ValueError("设备常驻加载器要求 DataLoader 提供 batch_size")

        self.features = features
        self.labels = labels
        self.dataset = source_loader.dataset
        self.batch_size = source_loader.batch_size
        self.drop_last = bool(source_loader.drop_last)
        self.shuffle = bool(shuffle)
        self.device = features.device
        self.num_workers = 0
        self.pin_memory = False
        self.persistent_workers = False

        generator_device = self.device if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device).manual_seed(seed)

    def __len__(self):
        sample_count = self.labels.shape[0]
        if self.drop_last:
            return sample_count // self.batch_size
        return (sample_count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        sample_count = self.labels.shape[0]
        if self.shuffle:
            # 索引和数据位于同一设备，避免每轮打乱触发主机到设备传输。
            indices = torch.randperm(
                sample_count,
                device=self.device,
                generator=self.generator,
            )
        else:
            indices = None

        stop = sample_count
        if self.drop_last:
            stop -= sample_count % self.batch_size

        for start in range(0, stop, self.batch_size):
            end = min(start + self.batch_size, sample_count)
            if indices is None:
                yield self.features[start:end], self.labels[start:end]
            else:
                batch_indices = indices[start:end]
                yield (
                    self.features.index_select(0, batch_indices),
                    self.labels.index_select(0, batch_indices),
                )

    @property
    def resident_bytes(self):
        return (
            self.features.numel() * self.features.element_size()
            + self.labels.numel() * self.labels.element_size()
        )


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

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
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
    print(
        f"DataLoader 构建完成：train={len(train_loader)} batches, "
        f"test={len(test_loader)} batches, "
        f"num_workers={worker_count}, pin_memory={pin_memory}"
    )
    return NinaProDataPipeline(train_loader, test_loader, normalization)


def create_dataloaders(*args, **kwargs):
    """便捷接口：仅返回 ``(train_loader, test_loader)``。"""

    pipeline = create_data_pipeline(*args, **kwargs)
    return pipeline.train_loader, pipeline.test_loader


def create_device_data_loader(
    data_loader,
    device,
    dtype=None,
    materialize_chunk_size=4096,
):
    """将 NinaPro Dataset 一次性向量化变换并常驻到目标设备。

    返回对象保持训练引擎依赖的 ``dataset``、``batch_size``、``drop_last``
    和可迭代接口，因此 Notebook 无需改写训练调用。该优化只适用于公开
    ``features``、``labels`` 的内存数据集；其他 Dataset 会显式报错并由
    调用方回退到原 DataLoader。
    """

    device = torch.device(device)
    if dtype is not None and dtype not in {
        torch.float16,
        torch.float32,
        torch.bfloat16,
    }:
        raise ValueError("dtype 必须是 float16、float32、bfloat16 或 None")
    if (
        not isinstance(materialize_chunk_size, int)
        or isinstance(materialize_chunk_size, bool)
        or materialize_chunk_size <= 0
    ):
        raise ValueError("materialize_chunk_size 必须是正整数")

    dataset = getattr(data_loader, "dataset", None)
    if (
        dataset is None
        or not hasattr(dataset, "features")
        or not hasattr(dataset, "labels")
    ):
        raise TypeError("仅支持公开 features 和 labels 的内存 Dataset")
    if data_loader.batch_size is None:
        raise ValueError("仅支持显式设置 batch_size 的 DataLoader")

    source_features = dataset.features
    source_labels = dataset.labels
    sample_count = source_labels.shape[0]
    materialized = None

    for start in range(0, sample_count, materialize_chunk_size):
        end = min(start + materialize_chunk_size, sample_count)
        batch = source_features[start:end]
        if dataset.transform is not None:
            batch = dataset.transform(batch)
        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch)
        if not batch.is_floating_point():
            batch = batch.to(torch.float32)

        target_dtype = dtype or batch.dtype
        if materialized is None:
            materialized = torch.empty(
                (sample_count, *batch.shape[1:]),
                device=device,
                dtype=target_dtype,
            )
        elif tuple(materialized.shape[1:]) != tuple(batch.shape[1:]):
            raise ValueError("Dataset transform 在不同批次产生了不同形状")

        materialized[start:end].copy_(
            batch.to(device=device, dtype=target_dtype),
        )

    labels = source_labels.to(device=device, dtype=torch.int64)
    sampler = getattr(data_loader, "sampler", None)
    shuffle = isinstance(sampler, RandomSampler)
    source_generator = getattr(sampler, "generator", None)
    if source_generator is None:
        source_generator = getattr(data_loader, "generator", None)
    seed = (
        source_generator.initial_seed()
        if source_generator is not None
        else torch.initial_seed()
    )

    return DeviceBatchLoader(
        features=materialized,
        labels=labels,
        source_loader=data_loader,
        shuffle=shuffle,
        seed=seed,
    )
