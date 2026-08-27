"""绘制 NinaPro 原始 sEMG 与两种泊松脉冲编码的对比图。"""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


OUTPUT_DIR = Path(__file__).resolve().parent
SELECTED_LABELS = (0, 1, 4, 5, 8, 9)
T_VALUES = (40, 80, 120)
RANDOM_SEED = 42
WINDOW_MS = 200.0

RAW_COLOR = "#334155"
OFFSET_COLOR = "#7c3aed"
POSITIVE_COLOR = "#dc2626"
NEGATIVE_COLOR = "#2563eb"
GRID_COLOR = "#d9e2ec"


def find_project_root():
    candidates = [OUTPUT_DIR, *OUTPUT_DIR.parents]
    for candidate in candidates:
        raw_path = (
            candidate
            / "ninapro_data"
            / "processed"
            / "exerciseA"
            / "slide_window"
            / "test.npz"
        )
        encoded_root = candidate / "ninapro_data" / "raw_data_poisson_spikes"
        if raw_path.is_file() and encoded_root.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("无法定位 NinaPro 原始数据与泊松编码数据")


def configure_plot_style():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = (
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    )
    selected_font = next(
        (font for font in preferred_fonts if font in available_fonts),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": selected_font,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfdff",
            "axes.edgecolor": "#94a3b8",
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def load_array(path, key):
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{path} 缺少字段 {key}")
        return np.asarray(data[key]).copy()


def load_selected_encoded(encoded_root, encoding_name, target_steps, indices):
    path = encoded_root / encoding_name / f"T_{target_steps}" / "test.npz"
    encoded = load_array(path, "X")
    selected = encoded[indices].copy()
    del encoded
    return selected


def select_sample_indices(labels):
    rng = np.random.default_rng(RANDOM_SEED)
    indices = []
    for label in SELECTED_LABELS:
        candidates = np.flatnonzero(labels == label)
        if candidates.size == 0:
            raise RuntimeError(f"测试集中不存在标签 {label}")
        indices.append(int(rng.choice(candidates)))
    return np.asarray(indices, dtype=np.int64)


def time_centers(time_steps):
    return (np.arange(time_steps, dtype=np.float64) + 0.5) * WINDOW_MS / time_steps


def style_time_axis(ax, show_xlabel=True):
    ax.set_xlim(0.0, WINDOW_MS)
    ax.set_xticks((0, 50, 100, 150, 200))
    if show_xlabel:
        ax.set_xlabel("时间 (ms)")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)


def plot_raw_traces(ax, raw, show_channel_labels=True, title=None):
    times = time_centers(raw.shape[1])
    # 仅为防止叠放曲线互相遮挡而缩放显示，不改变保存数据或编码输入。
    display_scale = max(float(np.percentile(np.abs(raw), 95)), 1.0)
    normalized = np.clip(raw / display_scale, -1.25, 1.25)
    offsets = np.arange(raw.shape[0], dtype=np.float64)

    for channel in range(raw.shape[0]):
        ax.plot(
            times,
            offsets[channel] + normalized[channel] * 0.34,
            color=RAW_COLOR,
            linewidth=0.8,
        )

    ax.set_ylim(raw.shape[0] - 0.5, -0.5)
    ax.set_yticks(offsets)
    if show_channel_labels:
        ax.set_yticklabels([f"C{channel + 1}" for channel in range(raw.shape[0])])
    else:
        ax.set_yticklabels([])
    ax.set_ylabel("原始通道" if show_channel_labels else "")
    style_time_axis(ax)
    if title:
        ax.set_title(title)


def plot_offset_raster(ax, spikes, show_channel_labels=True, title=None):
    times = time_centers(spikes.shape[1])
    events = [times[np.flatnonzero(channel)] for channel in spikes]
    offsets = np.arange(spikes.shape[0], dtype=np.float64)
    ax.eventplot(
        events,
        lineoffsets=offsets,
        linelengths=0.72,
        linewidths=0.85,
        colors=OFFSET_COLOR,
    )
    ax.set_ylim(spikes.shape[0] - 0.5, -0.5)
    ax.set_yticks(offsets)
    if show_channel_labels:
        ax.set_yticklabels([f"C{channel + 1}" for channel in range(spikes.shape[0])])
    else:
        ax.set_yticklabels([])
    ax.set_ylabel("加 128 通道" if show_channel_labels else "")
    style_time_axis(ax)
    if title:
        ax.set_title(title)


def plot_split_raster(ax, spikes, show_channel_labels=True, title=None):
    if spikes.shape[0] != 32:
        raise ValueError(f"正负分离脉冲应有 32 个通道，实际为 {spikes.shape}")

    times = time_centers(spikes.shape[1])
    events = []
    colors = []
    labels = []
    for channel in range(16):
        events.append(times[np.flatnonzero(spikes[channel])])
        events.append(times[np.flatnonzero(spikes[16 + channel])])
        colors.extend((POSITIVE_COLOR, NEGATIVE_COLOR))
        labels.extend((f"C{channel + 1}+", f"C{channel + 1}-"))

    offsets = np.arange(32, dtype=np.float64)
    ax.eventplot(
        events,
        lineoffsets=offsets,
        linelengths=0.72,
        linewidths=0.8,
        colors=colors,
    )
    for boundary in np.arange(1.5, 31.5, 2.0):
        ax.axhline(boundary, color="#edf2f7", linewidth=0.45, zorder=0)
    ax.set_ylim(31.5, -0.5)
    ax.set_yticks(offsets)
    if show_channel_labels:
        ax.set_yticklabels(labels, fontsize=6.4)
    else:
        ax.set_yticklabels([])
    ax.set_ylabel("正/负极性通道" if show_channel_labels else "")
    style_time_axis(ax)
    if title:
        ax.set_title(title)


def plot_individual_samples(raw_samples, encoded, selections):
    generated = []
    for position, selection in enumerate(selections):
        raw = raw_samples[position]
        offset = encoded["offset_128"][40][position]
        split = encoded["polarity_split"][40][position]

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(18, 9.2),
            gridspec_kw={"width_ratios": (1.05, 1.0, 1.25)},
        )
        plot_raw_traces(
            axes[0],
            raw,
            title=f"原始 sEMG  [16, 40]\n范围 {raw.min():.0f}～{raw.max():.0f}",
        )
        plot_offset_raster(
            axes[1],
            offset,
            title=(
                "加 128 泊松编码  [16, 40]\n"
                f"密度 {offset.mean():.2%} · 脉冲 {int(offset.sum())}"
            ),
        )
        plot_split_raster(
            axes[2],
            split,
            title=(
                "正负分离泊松编码  [32, 40]\n"
                f"密度 {split.mean():.2%} · 脉冲 {int(split.sum())}"
            ),
        )
        figure.suptitle(
            f"测试样本 {selection['index']} · 标签 {selection['label']} "
            f"· {selection['label_name']}",
            fontsize=15,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.015,
            "注：原始曲线只为叠放显示进行了统一缩放；编码读取的是未归一化原始值。",
            ha="center",
            fontsize=9,
            color="#475569",
        )
        figure.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.08, wspace=0.28)

        output_path = OUTPUT_DIR / (
            f"sample_test_{selection['index']:05d}_label_{selection['label']:02d}.png"
        )
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        generated.append(output_path.name)
    return generated


def plot_overview(raw_samples, encoded, selections):
    row_count = len(selections)
    figure, axes = plt.subplots(
        row_count,
        3,
        figsize=(18, 3.35 * row_count),
        squeeze=False,
        gridspec_kw={"width_ratios": (1.05, 1.0, 1.2)},
    )

    for row, selection in enumerate(selections):
        plot_raw_traces(axes[row, 0], raw_samples[row], show_channel_labels=False)
        plot_offset_raster(
            axes[row, 1],
            encoded["offset_128"][40][row],
            show_channel_labels=False,
        )
        plot_split_raster(
            axes[row, 2],
            encoded["polarity_split"][40][row],
            show_channel_labels=False,
        )
        axes[row, 0].set_ylabel(
            f"#{selection['index']}\n{selection['label_name']}",
            fontsize=9,
            fontweight="semibold",
        )

    axes[0, 0].set_title("原始 sEMG · 16 通道")
    axes[0, 1].set_title("加 128 泊松脉冲 · 16 通道")
    axes[0, 2].set_title("正负分离泊松脉冲 · 32 通道")
    figure.suptitle(
        "NinaPro 原始数据与两种泊松编码对比（测试集，T=40）",
        fontsize=16,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.035, hspace=0.45, wspace=0.2)
    output_path = OUTPUT_DIR / "comparison_overview_T40.png"
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output_path.name


def plot_time_resolution(encoded, selection):
    figure, axes = plt.subplots(
        len(T_VALUES),
        2,
        figsize=(15, 11.5),
        squeeze=False,
        gridspec_kw={"width_ratios": (1.0, 1.2)},
    )
    for row, target_steps in enumerate(T_VALUES):
        offset = encoded["offset_128"][target_steps][0]
        split = encoded["polarity_split"][target_steps][0]
        plot_offset_raster(
            axes[row, 0],
            offset,
            title=(
                f"加 128 · T={target_steps} · 密度 {offset.mean():.2%} "
                f"· {int(offset.sum())} 脉冲"
            ),
        )
        plot_split_raster(
            axes[row, 1],
            split,
            title=(
                f"正负分离 · T={target_steps} · 密度 {split.mean():.2%} "
                f"· {int(split.sum())} 脉冲"
            ),
        )

    figure.suptitle(
        f"同一 200 ms 窗口的时间分辨率对比 · 测试样本 {selection['index']} "
        f"· {selection['label_name']}",
        fontsize=15,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.055, hspace=0.38, wspace=0.23)
    output_path = OUTPUT_DIR / "time_resolution_comparison.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path.name


def build_manifest(selections, raw_samples, encoded, encoding_metadata):
    records = []
    for position, selection in enumerate(selections):
        record = {
            **selection,
            "raw_min": float(raw_samples[position].min()),
            "raw_max": float(raw_samples[position].max()),
            "raw_mean_absolute": float(np.abs(raw_samples[position]).mean()),
        }
        for encoding_name in ("offset_128", "polarity_split"):
            for target_steps in T_VALUES:
                spikes = encoded[encoding_name][target_steps][position]
                prefix = f"{encoding_name}_T{target_steps}"
                record[f"{prefix}_density"] = float(spikes.mean())
                record[f"{prefix}_spike_count"] = int(spikes.sum())
        records.append(record)

    payload = {
        "split": "test",
        "selection_seed": RANDOM_SEED,
        "selected_labels": list(SELECTED_LABELS),
        "window_ms": WINDOW_MS,
        "visualization_only_raw_trace_scaling": True,
        "encoding_parameters": {
            name: data["parameters"]
            for name, data in encoding_metadata["encodings"].items()
        },
        "samples": records,
    }
    json_path = OUTPUT_DIR / "selection_manifest.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = OUTPUT_DIR / "selection_manifest.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def write_readme(records, generated_files):
    rows = []
    for record in records:
        rows.append(
            "| {index} | {label} | {label_name} | {raw_mean_absolute:.2f} | "
            "{offset_128_T40_density:.2%} | {polarity_split_T40_density:.2%} |".format(
                **record
            )
        )

    body = [
        "# NinaPro 原始数据与泊松脉冲编码对比",
        "",
        "从测试集固定抽取 6 个样本，比较未归一化原始 sEMG、加 128 泊松编码和正负极性分离泊松编码。主对比使用 T=40；时间分辨率图对同一样本比较 T=40、80、120。",
        "",
        "原始波形在图中仅为避免 16 条曲线重叠而统一缩放，编码过程直接读取未归一化原始值。",
        "",
        "| 测试索引 | 标签 | 动作 | 原始平均绝对幅值 | 加128 T40密度 | 正负分离 T40密度 |",
        "|---:|---:|---|---:|---:|---:|",
        *rows,
        "",
        "## 文件",
        "",
        *[f"- `{name}`" for name in generated_files],
        "- `selection_manifest.csv`：便于表格查看的样本与脉冲统计。",
        "- `selection_manifest.json`：包含编码参数的完整机器可读记录。",
        "- `plot_poisson_spike_comparison.py`：可复现绘图脚本。",
        "",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(body), encoding="utf-8")


def main():
    configure_plot_style()
    project_root = find_project_root()
    raw_root = project_root / "ninapro_data" / "processed" / "exerciseA" / "slide_window"
    encoded_root = project_root / "ninapro_data" / "raw_data_poisson_spikes"

    raw_features = load_array(raw_root / "test.npz", "X")
    labels = load_array(raw_root / "test.npz", "y")
    metadata = json.loads((raw_root / "metadata.json").read_text(encoding="utf-8"))
    encoding_metadata = json.loads(
        (encoded_root / "metadata.json").read_text(encoding="utf-8")
    )
    label_mapping = metadata["label_mapping"]

    indices = select_sample_indices(labels)
    raw_samples = raw_features[indices].copy()
    del raw_features
    selections = [
        {
            "split": "test",
            "index": int(index),
            "label": int(labels[index]),
            "label_name": label_mapping[str(int(labels[index]))]["name_zh"],
        }
        for index in indices
    ]

    encoded = {"offset_128": {}, "polarity_split": {}}
    for encoding_name in encoded:
        for target_steps in T_VALUES:
            encoded[encoding_name][target_steps] = load_selected_encoded(
                encoded_root,
                encoding_name,
                target_steps,
                indices,
            )

    generated_files = []
    generated_files.append(plot_overview(raw_samples, encoded, selections))
    generated_files.extend(plot_individual_samples(raw_samples, encoded, selections))
    generated_files.append(plot_time_resolution(encoded, selections[0]))
    records = build_manifest(
        selections,
        raw_samples,
        encoded,
        encoding_metadata,
    )
    write_readme(records, generated_files)

    print(f"输出目录：{OUTPUT_DIR}")
    for name in generated_files:
        print(f"- {name}")


if __name__ == "__main__":
    main()
