"""训练检查点的原子保存与恢复。"""

from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch=0,
    history=None,
    normalization_state=None,
    config=None,
    best_test_accuracy=None,
    best_test_loss=None,
    test_metrics=None,
):
    """保存模型与恢复训练所需的完整状态。"""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "history": list(history or []),
        "normalization_state": normalization_state,
        "config": dict(config or {}),
        "best_test_accuracy": best_test_accuracy,
        "best_test_loss": best_test_loss,
        "test_metrics": test_metrics,
    }

    # 同目录临时文件配合 replace，避免训练中断留下半写入检查点。
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
    return path


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
):
    """加载检查点，并按需恢复优化器、scheduler 和 AMP scaler。"""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到检查点：{path}")

    # PyTorch 2.6 起 weights_only 默认为 True，完整训练状态需要显式关闭。
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    if "model_state" not in checkpoint:
        raise KeyError(f"检查点缺少 model_state：{path}")
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint
