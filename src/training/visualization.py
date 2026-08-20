"""训练历史与分类结果的文件输出。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def save_json(data, path):
    """以 UTF-8 JSON 保存训练历史或指标。"""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(data), file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)
    return path


def plot_training_history(history, path):
    """绘制训练/测试损失、准确率和学习率曲线。"""

    if not history:
        raise ValueError("history 不能为空")
    epochs = [record["epoch"] for record in history]

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(
        epochs,
        [record["train_loss"] for record in history],
        marker="o",
        markersize=3,
        label="Train",
    )
    axes[0].plot(
        epochs,
        [record["test_loss"] for record in history],
        marker="o",
        markersize=3,
        label="Test",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [record["train_accuracy"] * 100.0 for record in history],
        marker="o",
        markersize=3,
        label="Train",
    )
    axes[1].plot(
        epochs,
        [record["test_accuracy"] * 100.0 for record in history],
        marker="o",
        markersize=3,
        label="Test",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percent")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        epochs,
        [record["learning_rate"] for record in history],
        marker="o",
        markersize=3,
    )
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    return _save_figure(figure, path)


def plot_confusion_matrix(confusion_matrix, path, class_names=None):
    """保存真实标签为行、预测标签为列的混淆矩阵。"""

    confusion = np.asarray(confusion_matrix, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion_matrix 必须是方阵")
    num_classes = confusion.shape[0]
    names = _resolve_class_names(class_names, num_classes)

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=names,
        yticklabels=names,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = confusion.max() / 2.0 if confusion.size else 0.0
    for row in range(num_classes):
        for column in range(num_classes):
            axis.text(
                column,
                row,
                str(confusion[row, column]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if confusion[row, column] > threshold else "black",
            )

    figure.tight_layout()
    return _save_figure(figure, path)


def plot_per_class_accuracy(per_class_accuracy, path, class_names=None):
    """保存每类准确率柱状图。"""

    accuracy = np.asarray(per_class_accuracy, dtype=np.float64)
    if accuracy.ndim != 1 or accuracy.size == 0:
        raise ValueError("per_class_accuracy 必须是一维非空数组")
    names = _resolve_class_names(class_names, accuracy.size)

    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(accuracy.size)
    axis.bar(positions, accuracy * 100.0)
    axis.set(
        title="Per-class Accuracy",
        xlabel="Class",
        ylabel="Accuracy (%)",
        xticks=positions,
        xticklabels=names,
        ylim=(0, 100),
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return _save_figure(figure, path)


def _resolve_class_names(class_names, num_classes):
    if class_names is None:
        return [str(index) for index in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError("class_names 数量必须与类别数相同")
    return [str(name) for name in class_names]


def _save_figure(figure, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path
