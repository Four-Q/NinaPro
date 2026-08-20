from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import (
    ChannelwiseZScore,
    NinaProWindowDataset,
    PhysicalDeviceMapping,
    create_data_pipeline,
    create_datasets,
)
from src.data import loaders


class AddConstantMapping:
    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        return x + self.value


def write_dataset(path, train_x, train_y, test_x=None, test_y=None):
    path = Path(path)
    if test_x is None:
        test_x = train_x.copy()
    if test_y is None:
        test_y = train_y.copy()

    np.savez_compressed(path / "train.npz", X=train_x, y=train_y)
    np.savez_compressed(path / "test.npz", X=test_x, y=test_y)


def test_dataset_returns_only_x_and_y_with_expected_types(tmp_path):
    x = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    np.savez_compressed(tmp_path / "train.npz", X=x, y=y)

    dataset = NinaProWindowDataset(tmp_path / "train.npz")
    sample, label = dataset[1]

    assert len(dataset) == 4
    assert sample.shape == (2, 3)
    assert sample.dtype == torch.float32
    assert label.ndim == 0
    assert label.dtype == torch.int64


def test_dataset_copy_prevents_transform_from_mutating_source(tmp_path):
    x = np.ones((2, 2, 3), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    np.savez_compressed(tmp_path / "train.npz", X=x, y=y)

    def mutate_in_place(sample):
        sample.add_(10)
        return sample

    dataset = NinaProWindowDataset(tmp_path / "train.npz", transform=mutate_in_place)
    transformed, _ = dataset[0]

    assert torch.all(transformed == 11)
    assert torch.all(dataset.features[0] == 1)


def test_physical_mapping_precedes_normalization_and_stats_use_train_only(tmp_path):
    train_x = np.array(
        [
            [[0, 2], [10, 14]],
            [[2, 4], [14, 18]],
        ],
        dtype=np.float32,
    )
    test_x = np.full((1, 2, 2), 100, dtype=np.float32)
    train_y = np.array([0, 1], dtype=np.int64)
    test_y = np.array([1], dtype=np.int64)
    write_dataset(tmp_path, train_x, train_y, test_x, test_y)
    mapping = AddConstantMapping(7)

    train_dataset, test_dataset, normalization = create_datasets(
        tmp_path,
        physical_mapping=mapping,
        stats_chunk_size=1,
    )

    assert train_dataset.transform.transforms[0] is mapping
    assert train_dataset.transform.transforms[1] is normalization
    assert test_dataset.transform is train_dataset.transform

    normalized_train = train_dataset.transform(train_dataset.features)
    assert torch.allclose(normalized_train.mean(dim=(0, 2)), torch.zeros(2), atol=1e-6)
    assert torch.allclose(
        normalized_train.std(dim=(0, 2), correction=0),
        torch.ones(2),
        atol=1e-6,
    )

    # 测试集明显偏离零均值，证明其没有使用自己的统计量重新拟合。
    normalized_test, _ = test_dataset[0]
    assert torch.all(normalized_test > 10)


def test_normalization_state_can_be_exported_and_reused(tmp_path):
    x = np.arange(6 * 2 * 4, dtype=np.float32).reshape(6, 2, 4)
    y = np.arange(6, dtype=np.int64) % 2
    write_dataset(tmp_path, x, y)

    first = create_data_pipeline(
        tmp_path,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    state = first.normalization_state_dict()
    second = create_data_pipeline(
        tmp_path,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        normalization_state=state,
    )

    assert torch.equal(first.normalization.mean, second.normalization.mean)
    assert torch.equal(first.normalization.std, second.normalization.std)
    assert state["mean"].data_ptr() != first.normalization.mean.data_ptr()


def test_dataloaders_have_ann_and_snn_neutral_batch_layout(tmp_path):
    x = np.arange(8 * 3 * 5, dtype=np.float32).reshape(8, 3, 5)
    y = np.arange(8, dtype=np.int64) % 4
    write_dataset(tmp_path, x, y)

    pipeline = create_data_pipeline(
        tmp_path,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        seed=123,
    )
    train_x, train_y = next(iter(pipeline.train_loader))
    test_x, test_y = next(iter(pipeline.test_loader))

    assert train_x.shape == (4, 3, 5)
    assert train_y.shape == (4,)
    assert test_x.shape == (4, 3, 5)
    assert test_y.shape == (4,)
    assert pipeline.train_loader.num_workers == 0
    assert isinstance(pipeline.normalization, ChannelwiseZScore)


def test_physical_device_mapping_is_currently_identity():
    x = torch.randn(2, 3, 4)
    mapping = PhysicalDeviceMapping()

    assert mapping(x) is x


def test_num_workers_defaults_are_platform_aware(monkeypatch):
    monkeypatch.setattr(loaders.platform, "system", lambda: "Windows")
    assert loaders.resolve_num_workers() == 0

    monkeypatch.setattr(loaders.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        loaders.os,
        "sched_getaffinity",
        lambda process_id: set(range(16)),
        raising=False,
    )
    assert loaders.resolve_num_workers() == 8

    monkeypatch.setattr(loaders.os, "sched_getaffinity", lambda process_id: set(range(6)))
    assert loaders.resolve_num_workers() == 3

    monkeypatch.setattr(loaders.os, "sched_getaffinity", lambda process_id: set(range(2)))
    assert loaders.resolve_num_workers() == 0

    assert loaders.resolve_num_workers(5) == 5


@pytest.mark.parametrize("invalid_value", [-1, 1.5, True, "2"])
def test_num_workers_rejects_invalid_explicit_values(invalid_value):
    with pytest.raises(ValueError):
        loaders.resolve_num_workers(invalid_value)
