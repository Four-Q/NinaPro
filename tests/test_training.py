import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    ClassificationMeter,
    fit,
    load_checkpoint,
    metrics_from_confusion_matrix,
    overfit_one_batch,
    save_checkpoint,
    seed_everything,
)
from src.training.engine import _create_learning_rate_scheduler


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        return self.classifier(x.flatten(start_dim=1))


def make_loaders():
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(32, 2, 2, generator=generator)
    y = (x.flatten(start_dim=1).sum(dim=1) > 0).to(torch.int64)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    return train_loader, test_loader


def test_metrics_from_confusion_matrix_are_correct():
    confusion = torch.tensor([[3, 1], [2, 4]])

    metrics = metrics_from_confusion_matrix(confusion)

    assert metrics["accuracy"] == pytest.approx(0.7)
    assert metrics["macro_precision"] == pytest.approx((3 / 5 + 4 / 5) / 2)
    assert metrics["macro_recall"] == pytest.approx((3 / 4 + 4 / 6) / 2)
    assert metrics["per_class_accuracy"] == pytest.approx([3 / 4, 4 / 6])
    assert metrics["confusion_matrix"] == [[3, 1], [2, 4]]
    assert metrics["samples"] == 10


def test_classification_meter_accumulates_loss_and_predictions():
    meter = ClassificationMeter(num_classes=3)
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    targets = torch.tensor([0, 2])

    meter.update(logits, targets, torch.tensor(0.5))
    metrics = meter.compute()

    assert metrics["loss"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 0, 0], [0, 1, 0]]


def test_checkpoint_restores_model_optimizer_scheduler_and_metadata(tmp_path):
    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        history=[{"epoch": 3, "train_loss": 0.5}],
        normalization_state={"mean": torch.zeros(2), "std": torch.ones(2)},
        config={"selection_split": "test"},
        best_test_accuracy=0.75,
        best_test_loss=0.6,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)

    loaded = load_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    assert loaded["epoch"] == 3
    assert loaded["history"][0]["train_loss"] == 0.5
    assert loaded["config"]["selection_split"] == "test"
    assert torch.equal(loaded["normalization_state"]["std"], torch.ones(2))


def test_learning_rate_uses_five_epoch_warmup_then_ninety_five_epoch_cosine():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-2)
    scheduler, effective_warmup_epochs = _create_learning_rate_scheduler(
        optimizer,
        epochs=100,
        warmup_epochs=5,
        warmup_start_factor=0.2,
    )
    learning_rates = []

    for _ in range(100):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert effective_warmup_epochs == 5
    assert learning_rates[:6] == pytest.approx(
        [0.002, 0.0036, 0.0052, 0.0068, 0.0084, 0.01]
    )
    assert all(
        left > right for left, right in zip(learning_rates[5:], learning_rates[6:])
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_fit_returns_history_selects_best_and_saves_outputs(tmp_path):
    seed_everything(9)
    train_loader, test_loader = make_loaders()
    model = TinyClassifier()
    normalization_state = {"mean": torch.zeros(2), "std": torch.ones(2)}

    history = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        output_dir=tmp_path,
        epochs=3,
        learning_rate=0.05,
        device="cpu",
        normalization_state=normalization_state,
        config={"seed": 9},
        class_names=["negative", "positive"],
        verbose=False,
    )

    assert len(history) == 3
    assert set(history[0]) == {
        "epoch",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "learning_rate",
        "epoch_time",
    }
    expected_best = max(
        history,
        key=lambda record: (record["test_accuracy"], -record["test_loss"]),
    )
    best = load_checkpoint(tmp_path / "checkpoints" / "best.pt", model)
    assert best["epoch"] == expected_best["epoch"]
    assert best["config"]["test_selected_checkpoint"] is True

    for filename in ("last.pt", "best.pt", "final.pt"):
        assert (tmp_path / "checkpoints" / filename).is_file()
    for filename in ("history.json", "metrics.json"):
        assert (tmp_path / filename).is_file()
    for filename in (
        "training_curves.png",
        "confusion_matrix.png",
        "per_class_accuracy.png",
    ):
        figure_path = tmp_path / "figures" / filename
        assert figure_path.is_file()
        assert figure_path.stat().st_size > 0

    saved_history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    saved_metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert len(saved_history) == 3
    assert saved_metrics["best_epoch"] == expected_best["epoch"]
    assert saved_metrics["test_selected_checkpoint"] is True

    # fit 返回前会将调用方模型恢复为 best.pt 权重。
    for name, value in model.state_dict().items():
        assert torch.equal(value, best["model_state"][name])


def test_fit_can_resume_only_when_checkpoint_is_explicit(tmp_path):
    train_loader, test_loader = make_loaders()
    first_model = TinyClassifier()
    fit(
        first_model,
        train_loader,
        test_loader,
        output_dir=tmp_path,
        epochs=1,
        device="cpu",
        verbose=False,
    )

    resumed_model = TinyClassifier()
    history = fit(
        resumed_model,
        train_loader,
        test_loader,
        output_dir=tmp_path,
        epochs=2,
        device="cpu",
        resume_from=tmp_path / "checkpoints" / "last.pt",
        verbose=False,
    )

    assert [record["epoch"] for record in history] == [1, 2]
    last = load_checkpoint(tmp_path / "checkpoints" / "last.pt", resumed_model)
    assert last["epoch"] == 2


def test_fit_can_reduce_evaluation_and_checkpoint_frequency(tmp_path):
    train_loader, test_loader = make_loaders()
    model = TinyClassifier()

    history = fit(
        model,
        train_loader,
        test_loader,
        output_dir=tmp_path,
        epochs=3,
        learning_rate=0.05,
        device="cpu",
        eval_interval=2,
        checkpoint_interval=2,
        verbose=False,
    )

    assert history[0]["test_loss"] is None
    assert history[0]["test_accuracy"] is None
    assert history[1]["test_accuracy"] is not None
    # 最后一轮始终评估和保存，避免间隔不能整除 epochs 时丢失最终状态。
    assert history[2]["test_accuracy"] is not None
    last = load_checkpoint(tmp_path / "checkpoints" / "last.pt", model)
    assert last["epoch"] == 3


def test_overfit_one_batch_reaches_target_accuracy():
    seed_everything(17)
    model = TinyClassifier()
    x = torch.tensor(
        [
            [[-2.0, -1.0], [-1.0, -2.0]],
            [[-1.0, -0.5], [-0.5, -1.0]],
            [[1.0, 0.5], [0.5, 1.0]],
            [[2.0, 1.0], [1.0, 2.0]],
        ]
    )
    y = torch.tensor([0, 0, 1, 1])

    history = overfit_one_batch(
        model,
        (x, y),
        max_steps=100,
        target_accuracy=1.0,
        learning_rate=0.05,
        device="cpu",
        verbose=False,
    )

    assert history[-1]["accuracy"] == 1.0
    assert len(history) <= 100


def test_seed_everything_repeats_torch_and_numpy_sequences():
    seed_everything(123)
    first_torch = torch.rand(4)
    first_numpy = __import__("numpy").random.rand(4)

    seed_everything(123)
    second_torch = torch.rand(4)
    second_numpy = __import__("numpy").random.rand(4)

    assert torch.equal(first_torch, second_torch)
    assert (first_numpy == second_numpy).all()
