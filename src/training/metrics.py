"""不依赖 scikit-learn 的分类指标。"""

import torch


def metrics_from_confusion_matrix(confusion_matrix):
    """由真实标签为行、预测标签为列的混淆矩阵计算分类指标。"""

    confusion = torch.as_tensor(confusion_matrix, dtype=torch.int64).cpu()
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion_matrix 必须是方阵")
    if confusion.shape[0] == 0:
        raise ValueError("confusion_matrix 不能为空")
    if torch.any(confusion < 0):
        raise ValueError("confusion_matrix 不能包含负数")

    confusion_float = confusion.to(torch.float64)
    true_positive = confusion_float.diag()
    predicted_count = confusion_float.sum(dim=0)
    target_count = confusion_float.sum(dim=1)
    total = confusion_float.sum()

    precision = torch.where(
        predicted_count > 0,
        true_positive / predicted_count,
        torch.zeros_like(true_positive),
    )
    recall = torch.where(
        target_count > 0,
        true_positive / target_count,
        torch.zeros_like(true_positive),
    )
    f1_denominator = precision + recall
    f1 = torch.where(
        f1_denominator > 0,
        2.0 * precision * recall / f1_denominator,
        torch.zeros_like(true_positive),
    )
    accuracy = true_positive.sum() / total if total > 0 else torch.tensor(0.0)

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        # 分类任务中的每类准确率等价于该类 recall。
        "per_class_accuracy": recall.tolist(),
        "confusion_matrix": confusion.tolist(),
        "samples": int(total),
    }


class ClassificationMeter:
    """累计批次损失和混淆矩阵。"""

    def __init__(self, num_classes):
        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes <= 0
        ):
            raise ValueError("num_classes 必须是正整数")
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.loss_sum = None
        self.sample_count = 0
        self.confusion_matrix = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.int64,
        )

    def update(self, logits, targets, loss):
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                "logits 必须具有 [B, num_classes] 形状，"
                f"实际为 {tuple(logits.shape)}"
            )
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets 必须是与 logits 批次大小相同的一维张量")
        if targets.numel() == 0:
            raise ValueError("批次不能为空")

        targets_local = targets.detach().to(device=logits.device, dtype=torch.int64)
        predictions = logits.detach().argmax(dim=1).to(dtype=torch.int64)
        if targets_local.device.type == "cpu" and (
            torch.any(targets_local < 0) or torch.any(targets_local >= self.num_classes)
        ):
            raise ValueError("targets 中存在超出类别范围的标签")

        flat_indices = targets_local * self.num_classes + predictions
        batch_confusion = torch.bincount(
            flat_indices,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        if self.confusion_matrix.device != batch_confusion.device:
            # 首个 CUDA 批次后始终在设备端累计，整轮只在 compute 时同步一次。
            self.confusion_matrix = self.confusion_matrix.to(batch_confusion.device)
        self.confusion_matrix += batch_confusion

        batch_size = targets_local.numel()
        batch_loss_sum = loss.detach().to(dtype=torch.float32) * batch_size
        if self.loss_sum is None:
            self.loss_sum = batch_loss_sum
        else:
            self.loss_sum += batch_loss_sum
        self.sample_count += batch_size

    def compute(self):
        if self.sample_count == 0:
            raise RuntimeError("尚未累计任何样本")
        metrics = metrics_from_confusion_matrix(self.confusion_matrix.cpu())
        metrics["loss"] = float(self.loss_sum.cpu()) / self.sample_count
        return metrics
