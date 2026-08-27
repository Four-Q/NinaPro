"""汇总 outputs 中的训练结果并生成报告可视化。"""

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
ASSET_DIR = ROOT / "docs" / "assets" / "training_results"

EXPERIMENT_PATTERN = re.compile(
    r"BATCH_SIZE(?P<batch>\d+)_LR(?P<lr>[0-9.eE+-]+)_"
    r"HIDDEN(?P<hidden>\d+)_DROPOUT(?P<dropout>[0-9.]+)_"
    r"TAU(?P<tau>[0-9.]+)_WD(?P<weight_decay>[0-9.eE+-]+)"
)

CLASS_NAMES = [
    "食指屈曲",
    "食指伸展",
    "中指屈曲",
    "中指伸展",
    "无名指屈曲",
    "无名指伸展",
    "小指屈曲",
    "小指伸展",
    "拇指内收",
    "拇指外展",
    "拇指屈曲",
    "拇指伸展",
]

RAW_COLOR = "#2878B5"
SPIKE_COLOR = "#F28E2B"
ACCENT_COLOR = "#2A9D8F"
LOSS_COLOR = "#C44E52"
GRID_COLOR = "#D9E1E8"


def configure_style():
    """统一图表样式，并优先使用系统内可显示中文的字体。"""

    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#9AA6B2",
            "axes.labelcolor": "#25313C",
            "xtick.color": "#44515C",
            "ytick.color": "#44515C",
            "text.color": "#25313C",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.size": 10,
        }
    )


def parse_experiment(metrics_path):
    """从目录名和 JSON 中提取单次实验及派生指标。"""

    match = EXPERIMENT_PATTERN.search(metrics_path.parent.name)
    if match is None:
        raise ValueError(f"无法解析实验目录名：{metrics_path.parent.name}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history_path = metrics_path.parent / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    representation = metrics_path.parts[-4].replace("_ninapro_snn", "")
    time_steps = int(metrics_path.parts[-3].split("_")[-1])
    values = match.groupdict()
    best_epoch = int(metrics["best_epoch"])
    best_history = history[best_epoch - 1]
    test_accuracies = np.asarray(
        [item["test_accuracy"] for item in history], dtype=float
    )

    # “接近峰值”定义为首次达到峰值减 1 个百分点，便于观察收敛速度。
    near_peak = np.flatnonzero(test_accuracies >= metrics["accuracy"] - 0.01)
    near_peak_epoch = int(near_peak[0] + 1) if near_peak.size else None
    parameter_count = (
        int(values["hidden"]) ** 2
        + 30 * int(values["hidden"])
        + 14
    )

    return {
        "representation": representation,
        "T": time_steps,
        "batch_size": int(values["batch"]),
        "learning_rate": float(values["lr"]),
        "hidden": int(values["hidden"]),
        "dropout": float(values["dropout"]),
        "tau": float(values["tau"]),
        "weight_decay": float(values["weight_decay"]),
        "parameters": parameter_count,
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "loss": float(metrics["loss"]),
        "samples": int(metrics["samples"]),
        "best_epoch": best_epoch,
        "near_peak_epoch": near_peak_epoch,
        "train_accuracy_at_best": float(best_history["train_accuracy"]),
        "generalization_gap": float(
            best_history["train_accuracy"] - metrics["accuracy"]
        ),
        "final_test_accuracy": float(history[-1]["test_accuracy"]),
        "peak_to_final_drop": float(
            metrics["accuracy"] - history[-1]["test_accuracy"]
        ),
        "last10_test_std": float(np.std(test_accuracies[-10:])),
        "total_epoch_time_seconds": float(
            sum(item["epoch_time"] for item in history)
        ),
        "mean_epoch_time_seconds": float(
            np.mean([item["epoch_time"] for item in history])
        ),
        "per_class_precision": metrics["per_class_precision"],
        "per_class_recall": metrics["per_class_recall"],
        "per_class_f1": metrics["per_class_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "history": history,
        "metrics_path": metrics_path.relative_to(ROOT).as_posix(),
    }


def load_experiments():
    experiments = [
        parse_experiment(path)
        for path in sorted(OUTPUTS_DIR.rglob("metrics.json"))
    ]
    if not experiments:
        raise RuntimeError("outputs 中未找到 metrics.json")
    return experiments


def is_standard(row):
    return (
        row["hidden"] in {64, 128, 256}
        and math.isclose(row["dropout"], 0.1)
        and math.isclose(row["weight_decay"], 1e-4)
    )


def find_row(experiments, representation, time_steps, hidden, dropout, weight_decay):
    matches = [
        row
        for row in experiments
        if row["representation"] == representation
        and row["T"] == time_steps
        and row["hidden"] == hidden
        and math.isclose(row["dropout"], dropout)
        and math.isclose(row["weight_decay"], weight_decay)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "期望唯一实验但得到 "
            f"{len(matches)} 个：{representation}, T={time_steps}, H={hidden}, "
            f"dropout={dropout}, wd={weight_decay}"
        )
    return matches[0]


def add_panel_label(ax, label):
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )


def plot_heatmap(ax, values, title, vmin, vmax, cmap):
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3), ["T=40", "T=80", "T=120"])
    ax.set_yticks(range(3), ["H=64", "H=128", "H=256"])
    ax.set_title(title, pad=12)
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            text_color = "white" if value > (vmin + vmax) / 2 else "#1E2A33"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}%",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )
    return image


def make_standard_heatmaps(experiments):
    standard = [row for row in experiments if is_standard(row)]
    hidden_values = [64, 128, 256]
    time_values = [40, 80, 120]
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)

    panels = []
    for representation in ["raw", "spike"]:
        for metric in ["accuracy", "macro_f1"]:
            matrix = np.full((3, 3), np.nan)
            for row in standard:
                if row["representation"] != representation:
                    continue
                row_index = hidden_values.index(row["hidden"])
                col_index = time_values.index(row["T"])
                matrix[row_index, col_index] = row[metric] * 100
            panels.append(matrix)

    titles = [
        "Raw 输入 · Accuracy",
        "Raw 输入 · Macro-F1",
        "Spike 输入 · Accuracy",
        "Spike 输入 · Macro-F1",
    ]
    for index, (ax, matrix, title) in enumerate(zip(axes.flat, panels, titles)):
        image = plot_heatmap(ax, matrix, title, 35, 68, "YlGnBu")
        add_panel_label(ax, chr(ord("A") + index))
        figure.colorbar(image, ax=ax, shrink=0.78, label="百分比（%）")

    figure.suptitle(
        "标准配置性能全景（Dropout=0.1，WD=1e-4）",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(ASSET_DIR / "01_standard_grid_heatmaps.png", dpi=180)
    plt.close(figure)


def make_representation_gap(experiments):
    hidden_values = [64, 128, 256]
    time_values = [40, 80, 120]
    pairs = []
    delta_matrix = np.zeros((3, 3))
    for hidden_index, hidden in enumerate(hidden_values):
        for time_index, time_steps in enumerate(time_values):
            raw = find_row(experiments, "raw", time_steps, hidden, 0.1, 1e-4)
            spike = find_row(experiments, "spike", time_steps, hidden, 0.1, 1e-4)
            delta = (raw["accuracy"] - spike["accuracy"]) * 100
            delta_matrix[hidden_index, time_index] = delta
            pairs.append((raw, spike, delta))

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), constrained_layout=True)
    ax = axes[0]
    marker_by_time = {40: "o", 80: "s", 120: "^"}
    for raw, spike, _ in pairs:
        ax.scatter(
            spike["accuracy"] * 100,
            raw["accuracy"] * 100,
            s=55 + raw["hidden"] * 0.18,
            marker=marker_by_time[raw["T"]],
            color=RAW_COLOR,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
        )
        ax.annotate(
            f"T{raw['T']}/H{raw['hidden']}",
            (spike["accuracy"] * 100, raw["accuracy"] * 100),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    limits = [35, 68]
    ax.plot(limits, limits, linestyle="--", color="#7B8794", linewidth=1.2)
    ax.fill_between(limits, limits, limits[1], color=RAW_COLOR, alpha=0.05)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Spike Accuracy（%）")
    ax.set_ylabel("Raw Accuracy（%）")
    ax.set_title("9 组严格配对：全部位于 Raw 优势区")
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    add_panel_label(ax, "A")

    ax = axes[1]
    image = plot_heatmap(
        ax,
        delta_matrix,
        "Raw − Spike Accuracy 差值",
        10,
        20,
        "YlGn",
    )
    ax.set_title("Raw − Spike Accuracy（百分点）", pad=12)
    for text_item in ax.texts:
        row_index = int(round(text_item.get_position()[1]))
        col_index = int(round(text_item.get_position()[0]))
        text_item.set_text(f"+{delta_matrix[row_index, col_index]:.2f} pp")
    figure.colorbar(image, ax=ax, shrink=0.82, label="Raw 优势（百分点）")
    add_panel_label(ax, "B")

    figure.suptitle(
        "原始 sEMG 与神经形态二值脉冲的配对比较",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(ASSET_DIR / "02_raw_vs_spike_gap.png", dpi=180)
    plt.close(figure)


def plot_grouped_metrics(ax, rows, labels, title):
    x = np.arange(len(rows))
    width = 0.35
    accuracies = [row["accuracy"] * 100 for row in rows]
    macro_f1 = [row["macro_f1"] * 100 for row in rows]
    bars_accuracy = ax.bar(
        x - width / 2,
        accuracies,
        width,
        label="Accuracy",
        color=RAW_COLOR,
    )
    bars_f1 = ax.bar(
        x + width / 2,
        macro_f1,
        width,
        label="Macro-F1",
        color=ACCENT_COLOR,
    )
    ax.bar_label(bars_accuracy, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(bars_f1, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(max(30, min(accuracies + macro_f1) - 7), max(accuracies + macro_f1) + 6)
    ax.set_ylabel("百分比（%）")
    ax.set_title(title)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="upper left")


def make_hyperparameter_ablations(experiments):
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)

    raw_regularization = [
        find_row(experiments, "raw", 40, 256, 0.0, 1e-4),
        find_row(experiments, "raw", 40, 256, 0.1, 1e-4),
        find_row(experiments, "raw", 40, 256, 0.3, 1e-4),
        find_row(experiments, "raw", 40, 256, 0.3, 1e-3),
    ]
    plot_grouped_metrics(
        axes[0, 0],
        raw_regularization,
        ["D0\nWD1e-4", "D0.1\nWD1e-4", "D0.3\nWD1e-4", "D0.3\nWD1e-3"],
        "Raw · T40/H256 正则化消融",
    )

    spike_dropout = [
        find_row(experiments, "spike", 40, 256, 0.0, 1e-4),
        find_row(experiments, "spike", 40, 256, 0.1, 1e-4),
    ]
    plot_grouped_metrics(
        axes[0, 1],
        spike_dropout,
        ["D0\nWD1e-4", "D0.1\nWD1e-4"],
        "Spike · T40/H256 Dropout 消融",
    )

    raw_capacity = [
        find_row(experiments, "raw", 40, hidden, 0.1, 1e-4)
        for hidden in [64, 128, 256]
    ] + [find_row(experiments, "raw", 40, 320, 0.1, 1e-4)]
    plot_grouped_metrics(
        axes[1, 0],
        raw_capacity,
        ["H64", "H128", "H256", "H320"],
        "Raw · T40 容量扩展",
    )

    spike_capacity = [
        find_row(experiments, "spike", 40, hidden, 0.1, 1e-4)
        for hidden in [64, 128, 256]
    ]
    plot_grouped_metrics(
        axes[1, 1],
        spike_capacity,
        ["H64", "H128", "H256"],
        "Spike · T40 容量扩展",
    )

    for index, ax in enumerate(axes.flat):
        add_panel_label(ax, chr(ord("A") + index))
    figure.suptitle("超参数与模型容量：收益并不均匀", fontsize=16, fontweight="bold")
    figure.savefig(ASSET_DIR / "03_hyperparameter_ablations.png", dpi=180)
    plt.close(figure)


def make_learning_dynamics(best_raw, best_spike):
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
    for row_index, (row, label, color) in enumerate(
        [(best_raw, "最佳 Raw", RAW_COLOR), (best_spike, "最佳 Spike", SPIKE_COLOR)]
    ):
        history = row["history"]
        epochs = [item["epoch"] for item in history]

        accuracy_ax = axes[row_index, 0]
        accuracy_ax.plot(
            epochs,
            [item["train_accuracy"] * 100 for item in history],
            color="#8896A5",
            linewidth=1.4,
            label="Train",
        )
        accuracy_ax.plot(
            epochs,
            [item["test_accuracy"] * 100 for item in history],
            color=color,
            linewidth=2.0,
            label="Test",
        )
        accuracy_ax.axvline(
            row["best_epoch"], color=LOSS_COLOR, linestyle="--", linewidth=1.1
        )
        accuracy_ax.scatter(
            [row["best_epoch"]],
            [row["accuracy"] * 100],
            color=LOSS_COLOR,
            zorder=4,
        )
        accuracy_ax.annotate(
            f"峰值 {row['accuracy'] * 100:.2f}%\nEpoch {row['best_epoch']}",
            (row["best_epoch"], row["accuracy"] * 100),
            xytext=(-50, -35 if row_index == 0 else 12),
            textcoords="offset points",
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": LOSS_COLOR},
        )
        accuracy_ax.set_ylabel("Accuracy（%）")
        accuracy_ax.set_title(
            f"{label} · T{row['T']}/H{row['hidden']}/D{row['dropout']:g}/WD{row['weight_decay']:g}"
        )
        accuracy_ax.legend(frameon=False, ncol=2)
        accuracy_ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)

        loss_ax = axes[row_index, 1]
        loss_ax.plot(
            epochs,
            [item["train_loss"] for item in history],
            color="#8896A5",
            linewidth=1.4,
            label="Train",
        )
        loss_ax.plot(
            epochs,
            [item["test_loss"] for item in history],
            color=color,
            linewidth=2.0,
            label="Test",
        )
        loss_ax.axvline(
            row["best_epoch"], color=LOSS_COLOR, linestyle="--", linewidth=1.1
        )
        loss_ax.set_ylabel("Cross-entropy Loss")
        loss_ax.set_title(f"{label} · 损失曲线")
        loss_ax.legend(frameon=False, ncol=2)
        loss_ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)

    for index, ax in enumerate(axes.flat):
        ax.set_xlabel("Epoch")
        add_panel_label(ax, chr(ord("A") + index))
    figure.suptitle("最佳模型的训练动态与泛化间隙", fontsize=16, fontweight="bold")
    figure.savefig(ASSET_DIR / "04_best_learning_dynamics.png", dpi=180)
    plt.close(figure)


def make_per_class_comparison(best_raw, best_spike):
    x = np.arange(len(CLASS_NAMES))
    width = 0.36
    raw_values = np.asarray(best_raw["per_class_recall"]) * 100
    spike_values = np.asarray(best_spike["per_class_recall"]) * 100

    figure, ax = plt.subplots(figsize=(13.5, 6.3), constrained_layout=True)
    raw_bars = ax.bar(
        x - width / 2,
        raw_values,
        width,
        color=RAW_COLOR,
        label="最佳 Raw",
    )
    spike_bars = ax.bar(
        x + width / 2,
        spike_values,
        width,
        color=SPIKE_COLOR,
        label="最佳 Spike",
    )
    ax.bar_label(raw_bars, fmt="%.1f", padding=2, fontsize=7.5, rotation=90)
    ax.bar_label(spike_bars, fmt="%.1f", padding=2, fontsize=7.5, rotation=90)
    ax.set_xticks(x, CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylim(20, 90)
    ax.set_ylabel("Recall / 类别准确率（%）")
    ax.set_title("逐动作召回率：优势集中在部分易混淆手势", pad=12)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, ncol=2)
    figure.savefig(ASSET_DIR / "05_best_per_class_recall.png", dpi=180)
    plt.close(figure)


def make_confusion_matrices(best_raw, best_spike):
    figure, axes = plt.subplots(1, 2, figsize=(14.6, 6.4), constrained_layout=True)
    last_image = None
    for index, (ax, row, title) in enumerate(
        [
            (axes[0], best_raw, "最佳 Raw · 行归一化混淆矩阵"),
            (axes[1], best_spike, "最佳 Spike · 行归一化混淆矩阵"),
        ]
    ):
        confusion = np.asarray(row["confusion_matrix"], dtype=float)
        normalized = confusion / confusion.sum(axis=1, keepdims=True) * 100
        last_image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=85)
        ax.set_xticks(range(12), range(1, 13))
        ax.set_yticks(range(12), range(1, 13))
        ax.set_xlabel("预测类别编号")
        ax.set_ylabel("真实类别编号")
        ax.set_title(title, pad=12)
        for row_index in range(12):
            for col_index in range(12):
                value = normalized[row_index, col_index]
                if row_index == col_index or value >= 10:
                    ax.text(
                        col_index,
                        row_index,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value >= 43 else "#25313C",
                    )
        add_panel_label(ax, chr(ord("A") + index))
    figure.colorbar(last_image, ax=axes, shrink=0.84, label="真实类别内占比（%）")
    figure.suptitle("错误结构对比（仅标注对角线及 ≥10% 的混淆）", fontsize=16, fontweight="bold")
    figure.savefig(ASSET_DIR / "06_best_confusion_matrices.png", dpi=180)
    plt.close(figure)


def write_summary_csv(experiments):
    fields = [
        "representation",
        "T",
        "batch_size",
        "learning_rate",
        "hidden",
        "dropout",
        "tau",
        "weight_decay",
        "parameters",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "loss",
        "samples",
        "best_epoch",
        "near_peak_epoch",
        "train_accuracy_at_best",
        "generalization_gap",
        "final_test_accuracy",
        "peak_to_final_drop",
        "last10_test_std",
        "total_epoch_time_seconds",
        "mean_epoch_time_seconds",
        "metrics_path",
    ]
    ranked = sorted(experiments, key=lambda row: row["accuracy"], reverse=True)
    with (ASSET_DIR / "experiment_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)


def normalized_confusions(row):
    confusion = np.asarray(row["confusion_matrix"], dtype=float)
    return confusion / confusion.sum(axis=1, keepdims=True)


def top_confusions(row, count=8):
    normalized = normalized_confusions(row)
    np.fill_diagonal(normalized, 0)
    flat_indices = np.argsort(normalized.ravel())[::-1][:count]
    result = []
    for flat_index in flat_indices:
        true_index, predicted_index = np.unravel_index(flat_index, normalized.shape)
        result.append(
            {
                "true_class": int(true_index),
                "predicted_class": int(predicted_index),
                "true_name": CLASS_NAMES[true_index],
                "predicted_name": CLASS_NAMES[predicted_index],
                "rate": float(normalized[true_index, predicted_index]),
            }
        )
    return result


def write_derived_summary(experiments, best_raw, best_spike):
    paired_deltas = []
    paired_by_time = {}
    for time_steps in [40, 80, 120]:
        deltas = []
        for hidden in [64, 128, 256]:
            raw = find_row(experiments, "raw", time_steps, hidden, 0.1, 1e-4)
            spike = find_row(experiments, "spike", time_steps, hidden, 0.1, 1e-4)
            delta = raw["accuracy"] - spike["accuracy"]
            paired_deltas.append(delta)
            deltas.append(delta)
        paired_by_time[str(time_steps)] = {
            "mean": float(np.mean(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
        }

    raw_standard = find_row(experiments, "raw", 40, 256, 0.1, 1e-4)
    spike_standard = find_row(experiments, "spike", 40, 256, 0.1, 1e-4)
    raw_d0 = find_row(experiments, "raw", 40, 256, 0.0, 1e-4)
    raw_d03 = find_row(experiments, "raw", 40, 256, 0.3, 1e-4)
    raw_d03_wd = find_row(experiments, "raw", 40, 256, 0.3, 1e-3)
    spike_d0 = find_row(experiments, "spike", 40, 256, 0.0, 1e-4)
    raw_h320 = find_row(experiments, "raw", 40, 320, 0.1, 1e-4)

    summary = {
        "experiment_count": len(experiments),
        "sample_counts": sorted({row["samples"] for row in experiments}),
        "best_raw": {key: value for key, value in best_raw.items() if key not in {"history", "confusion_matrix", "per_class_precision", "per_class_recall", "per_class_f1"}},
        "best_spike": {key: value for key, value in best_spike.items() if key not in {"history", "confusion_matrix", "per_class_precision", "per_class_recall", "per_class_f1"}},
        "best_raw_vs_best_spike_accuracy_delta": best_raw["accuracy"] - best_spike["accuracy"],
        "fair_t40_h256_accuracy_delta": raw_standard["accuracy"] - spike_standard["accuracy"],
        "paired_standard_accuracy_delta": {
            "mean": float(np.mean(paired_deltas)),
            "min": float(np.min(paired_deltas)),
            "max": float(np.max(paired_deltas)),
            "by_time": paired_by_time,
        },
        "ablations": {
            "raw_dropout_0_to_0_1": raw_standard["accuracy"] - raw_d0["accuracy"],
            "raw_dropout_0_1_to_0_3": raw_d03["accuracy"] - raw_standard["accuracy"],
            "raw_wd_1e_4_to_1e_3_at_dropout_0_3": raw_d03_wd["accuracy"] - raw_d03["accuracy"],
            "spike_dropout_0_to_0_1": spike_standard["accuracy"] - spike_d0["accuracy"],
            "raw_h256_to_h320": raw_h320["accuracy"] - raw_standard["accuracy"],
            "raw_h256_to_h320_parameter_ratio": raw_h320["parameters"] / raw_standard["parameters"],
        },
        "best_raw_top_confusions": top_confusions(best_raw),
        "best_spike_top_confusions": top_confusions(best_spike),
    }
    (ASSET_DIR / "derived_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def validate_experiments(experiments):
    if len(experiments) != 23:
        raise RuntimeError(f"预期 23 组实验，实际为 {len(experiments)} 组")
    if {row["samples"] for row in experiments} != {8787}:
        raise RuntimeError("实验的测试样本数不一致")
    if any(len(row["history"]) != 100 for row in experiments):
        raise RuntimeError("存在非 100 epoch 的训练历史")
    if len([row for row in experiments if is_standard(row)]) != 18:
        raise RuntimeError("标准配置网格不完整")


def main():
    configure_style()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments()
    validate_experiments(experiments)
    best_raw = max(
        (row for row in experiments if row["representation"] == "raw"),
        key=lambda row: row["accuracy"],
    )
    best_spike = max(
        (row for row in experiments if row["representation"] == "spike"),
        key=lambda row: row["accuracy"],
    )

    make_standard_heatmaps(experiments)
    make_representation_gap(experiments)
    make_hyperparameter_ablations(experiments)
    make_learning_dynamics(best_raw, best_spike)
    make_per_class_comparison(best_raw, best_spike)
    make_confusion_matrices(best_raw, best_spike)
    write_summary_csv(experiments)
    write_derived_summary(experiments, best_raw, best_spike)
    print(f"已生成 {len(experiments)} 组实验的汇总与可视化：{ASSET_DIR}")


if __name__ == "__main__":
    main()
