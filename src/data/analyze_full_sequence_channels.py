"""统计 NinaPro full_sequence 原始通道分布并生成可视化。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D


DEFAULT_DATA_DIR = Path("ninapro_data/processed/exerciseA/full_sequence")
DEFAULT_OUTPUT_DIR = Path("outputs/ninapro_db5_exerciseA_channel_ranges")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--violin-samples",
        type=int,
        default=60000,
        help="每个通道用于估计小提琴密度的最大样本数",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def load_split(path, expected_channels):
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    with np.load(path, allow_pickle=False) as data:
        if "X" not in data.files:
            raise KeyError(f"{path} 缺少 X 数组")
        features = np.asarray(data["X"])

    if features.ndim != 2:
        raise ValueError(f"{path} 的 X 应为 [时间点, 通道]，实际为 {features.shape}")
    if features.shape[1] != expected_channels:
        raise ValueError(
            f"{path} 的通道数应为 {expected_channels}，实际为 {features.shape[1]}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"{path} 的 X 中存在 NaN 或无穷值")
    return features


def summarize_channels(features, split_name):
    percentiles = np.percentile(
        features,
        [0.1, 1, 25, 50, 75, 99, 99.9],
        axis=0,
    )
    rows = []
    for index in range(features.shape[1]):
        values = features[:, index]
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        rows.append(
            {
                "split": split_name,
                "channel": index + 1,
                "channel_name": f"CH{index + 1:02d}",
                "sample_count": int(values.size),
                "min": minimum,
                "max": maximum,
                "range": maximum - minimum,
                "mean": float(np.mean(values, dtype=np.float64)),
                "std": float(np.std(values, dtype=np.float64)),
                "p0.1": float(percentiles[0, index]),
                "p1": float(percentiles[1, index]),
                "q1": float(percentiles[2, index]),
                "median": float(percentiles[3, index]),
                "q3": float(percentiles[4, index]),
                "p99": float(percentiles[5, index]),
                "p99.9": float(percentiles[6, index]),
            }
        )
    return pd.DataFrame(rows)


def configure_plot_style():
    # 优先选择含中文字形的字体，保证不同 Windows 环境下标题可读。
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans SC",
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#94A3B8",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.55,
            "svg.fonttype": "none",
        }
    )


def make_violin_sample(features, max_samples, seed):
    if max_samples <= 0:
        raise ValueError("violin_samples 必须大于 0")
    if features.shape[0] <= max_samples:
        return features

    # 所有通道复用同一批时间点，使通道间的密度比较不受抽样位置差异影响。
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(features.shape[0], size=max_samples, replace=False))
    return features[indices]


def plot_channel_distributions(features, overall_stats, output_dir, sample_size, seed):
    configure_plot_style()
    sampled = make_violin_sample(features, sample_size, seed)
    channels = overall_stats["channel_name"].tolist()
    positions = np.arange(1, len(channels) + 1)
    standard_deviations = overall_stats["std"].to_numpy()
    color_scale = colors.Normalize(
        vmin=float(standard_deviations.min()),
        vmax=float(standard_deviations.max()),
    )
    palette = plt.colormaps["viridis"](color_scale(standard_deviations))

    figure = plt.figure(figsize=(16, 10), layout="constrained")
    grid = figure.add_gridspec(2, 1, height_ratios=[2.15, 1], hspace=0.08)
    violin_axis = figure.add_subplot(grid[0])
    range_axis = figure.add_subplot(grid[1])

    violin = violin_axis.violinplot(
        [sampled[:, index] for index in range(sampled.shape[1])],
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=160,
        bw_method="scott",
    )
    for body, color in zip(violin["bodies"], palette):
        body.set_facecolor(color)
        body.set_edgecolor("#334155")
        body.set_linewidth(0.7)
        body.set_alpha(0.88)

    q1 = overall_stats["q1"].to_numpy()
    median = overall_stats["median"].to_numpy()
    q3 = overall_stats["q3"].to_numpy()
    p1 = overall_stats["p1"].to_numpy()
    p99 = overall_stats["p99"].to_numpy()
    violin_axis.vlines(
        positions,
        p1,
        p99,
        color="#64748B",
        linewidth=1.0,
        alpha=0.75,
        zorder=3,
    )
    violin_axis.scatter(
        positions,
        p1,
        marker="_",
        s=210,
        color="#DC2626",
        linewidth=2.2,
        zorder=5,
    )
    violin_axis.scatter(
        positions,
        p99,
        marker="_",
        s=210,
        color="#2563EB",
        linewidth=2.2,
        zorder=5,
    )
    violin_axis.vlines(positions, q1, q3, color="#0F172A", linewidth=3.0, zorder=4)
    violin_axis.scatter(
        positions,
        median,
        s=28,
        facecolor="white",
        edgecolor="#0F172A",
        linewidth=0.8,
        zorder=5,
    )
    violin_axis.axhline(0, color="#64748B", linewidth=1.0, linestyle="--", alpha=0.7)
    violin_axis.set_xlim(0.35, len(channels) + 0.65)
    violin_axis.set_ylim(
        float(overall_stats["min"].min()) - 8,
        float(overall_stats["max"].max()) + 8,
    )
    violin_axis.set_xticks(positions, channels)
    violin_axis.set_ylabel("原始 sEMG 数值")
    violin_axis.set_title(
        "NinaPro DB5 Exercise A：16 通道原始数值分布",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
        pad=16,
    )
    violin_axis.text(
        0.0,
        1.01,
        (
            f"训练集 + 测试集：{features.shape[0]:,} 个时间点；"
            f"小提琴密度固定抽样 {sampled.shape[0]:,} 点/通道；"
            "红线为 P1，蓝线为 P99"
        ),
        transform=violin_axis.transAxes,
        fontsize=10.5,
        color="#475569",
        va="bottom",
    )
    violin_axis.grid(axis="y", linewidth=0.8)
    legend_handles = [
        Line2D([0], [0], color="#DC2626", marker="_", markersize=12, linewidth=0, markeredgewidth=2.2, label="P1（1%）"),
        Line2D([0], [0], color="#2563EB", marker="_", markersize=12, linewidth=0, markeredgewidth=2.2, label="P99（99%）"),
        Line2D([0], [0], color="#0F172A", linewidth=3.0, label="Q1–Q3"),
        Line2D([0], [0], marker="o", markersize=5.5, markerfacecolor="white", markeredgecolor="#0F172A", linewidth=0, label="中位数"),
    ]
    violin_axis.legend(
        handles=legend_handles,
        loc="upper right",
        ncols=4,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=9.5,
    )

    minimum = overall_stats["min"].to_numpy()
    maximum = overall_stats["max"].to_numpy()
    value_range = overall_stats["range"].to_numpy()
    range_axis.bar(
        positions,
        value_range,
        bottom=minimum,
        width=0.64,
        color=palette,
        edgecolor="#334155",
        linewidth=0.7,
        alpha=0.9,
    )
    range_axis.scatter(positions, minimum, color="#0F172A", s=13, zorder=3)
    range_axis.scatter(positions, maximum, color="#0F172A", s=13, zorder=3)
    for x_position, low, high, span in zip(positions, minimum, maximum, value_range):
        range_axis.text(
            x_position,
            high + 7,
            f"{span:g}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#334155",
        )
        range_axis.text(
            x_position,
            low - 7,
            f"{low:g} / {high:g}",
            ha="center",
            va="top",
            fontsize=7.5,
            color="#64748B",
            rotation=24,
        )
    range_axis.set_xlim(0.35, len(channels) + 0.65)
    range_axis.set_ylim(minimum.min() - 35, maximum.max() + 32)
    range_axis.set_xticks(positions, channels)
    range_axis.set_ylabel("精确 Min–Max")
    range_axis.set_title(
        "各通道精确原始范围（柱顶为跨度；柱下为 Min / Max）",
        fontsize=12,
        color="#334155",
        pad=8,
    )
    range_axis.grid(axis="y", linewidth=0.8)

    # 色条编码标准差，补充展示通道信号离散程度。
    scalar_map = plt.cm.ScalarMappable(norm=color_scale, cmap="viridis")
    color_bar = figure.colorbar(
        scalar_map,
        ax=[violin_axis, range_axis],
        location="right",
        shrink=0.72,
        pad=0.015,
    )
    color_bar.set_label("总体标准差", color="#334155")

    png_path = output_dir / "channel_ranges_violin.png"
    svg_path = output_dir / "channel_ranges_violin.svg"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, svg_path, sampled.shape[0]


def write_summary(
    output_dir,
    metadata,
    combined,
    overall_stats,
    all_stats,
    violin_sample_count,
):
    overall_csv = output_dir / "channel_statistics_overall.csv"
    split_csv = output_dir / "channel_statistics_by_split.csv"
    overall_stats.to_csv(overall_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    all_stats.to_csv(split_csv, index=False, encoding="utf-8-sig", float_format="%.6f")

    full_span_channels = overall_stats.loc[
        overall_stats["range"] == overall_stats["range"].max(), "channel_name"
    ].tolist()
    narrowest = overall_stats.loc[overall_stats["range"].idxmin()]
    report = {
        "dataset": metadata.get("dataset", "NinaPro DB5"),
        "exercise": metadata.get("exercise", "A / E1"),
        "normalized": metadata.get("normalized", False),
        "channels": int(combined.shape[1]),
        "total_time_points": int(combined.shape[0]),
        "all_values_finite": bool(np.isfinite(combined).all()),
        "violin_density_sample_count_per_channel": int(violin_sample_count),
        "largest_range": float(overall_stats["range"].max()),
        "channels_with_largest_range": full_span_channels,
        "narrowest_range_channel": str(narrowest["channel_name"]),
        "narrowest_range": float(narrowest["range"]),
        "files": {
            "overall_statistics": overall_csv.name,
            "split_statistics": split_csv.name,
            "figure_png": "channel_ranges_violin.png",
            "figure_svg": "channel_ranges_violin.svg",
        },
    }
    summary_path = output_dir / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return overall_csv, split_csv, summary_path


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_channels = int(metadata.get("channels", 16))

    train = load_split(data_dir / "train.npz", expected_channels)
    test = load_split(data_dir / "test.npz", expected_channels)
    combined = np.concatenate([train, test], axis=0)

    overall_stats = summarize_channels(combined, "overall")
    train_stats = summarize_channels(train, "train")
    test_stats = summarize_channels(test, "test")
    all_stats = pd.concat([overall_stats, train_stats, test_stats], ignore_index=True)

    png_path, svg_path, violin_sample_count = plot_channel_distributions(
        combined,
        overall_stats,
        output_dir,
        args.violin_samples,
        args.seed,
    )
    overall_csv, split_csv, summary_path = write_summary(
        output_dir,
        metadata,
        combined,
        overall_stats,
        all_stats,
        violin_sample_count,
    )

    print(overall_stats[["channel_name", "min", "max", "range"]].to_string(index=False))
    print(f"\n图像：{png_path}")
    print(f"矢量图：{svg_path}")
    print(f"总体统计：{overall_csv}")
    print(f"分割统计：{split_csv}")
    print(f"摘要：{summary_path}")


if __name__ == "__main__":
    main()
