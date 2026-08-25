"""生成 V3：按通道 p1/p99 对称电压区间的分段线性 LTspice PWL。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import generate_pwl as base


VERSION_NAME = "v3_p1p99_symmetric_threshold_validation"
VERSION_ROOT = base.SAMPLE_ROOT / VERSION_NAME
STATISTICS_PATH = (
    base.SAMPLE_ROOT.parent
    / "ninapro_db5_exerciseA_channel_ranges"
    / "channel_statistics_overall.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--window-count", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=float, default=200.0)
    parser.add_argument("--zero-voltage", type=float, default=1.8639)
    parser.add_argument("--transition-us", type=float, default=1.0)
    return parser.parse_args()


def load_overall_percentiles(path, channel_count):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"].strip().lower() == "overall":
                rows.append(row)

    rows.sort(key=lambda item: int(item["channel"]))
    channels = [int(row["channel"]) for row in rows]
    if channels != list(range(1, channel_count + 1)):
        raise ValueError(
            f"overall 统计的通道应为 1..{channel_count}，实际为 {channels}"
        )

    p1 = np.asarray([float(row["p1"]) for row in rows], dtype=np.float64)
    p99 = np.asarray([float(row["p99"]) for row in rows], dtype=np.float64)
    if np.any(p1 >= 0) or np.any(p99 <= 0):
        raise ValueError("V3 要求每个通道满足 p1 < 0 < p99")
    return p1, p99


def map_to_voltage(samples, p1, p99, zero_voltage):
    samples = np.asarray(samples, dtype=np.float64)
    maximum_voltage = 2.0 * zero_voltage

    # 两个半轴各占 zero_voltage 的宽度；p1/p99 不对称时零点处允许斜率改变。
    negative = zero_voltage * (samples - p1) / (-p1)
    positive = zero_voltage * (1.0 + samples / p99)
    voltage = np.where(samples < 0.0, negative, positive)
    return np.clip(voltage, 0.0, maximum_voltage)


def mapping_diagnostics(samples, voltage, p1, p99, zero_voltage):
    samples = np.asarray(samples, dtype=np.float64)
    voltage = np.asarray(voltage, dtype=np.float64)
    maximum_voltage = 2.0 * zero_voltage

    diagnostics = []
    for channel_index in range(samples.shape[1]):
        raw = samples[:, channel_index]
        vin = voltage[:, channel_index]
        diagnostics.append(
            {
                "channel": channel_index + 1,
                "sample_count": int(raw.size),
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "vin_min": float(vin.min()),
                "vin_max": float(vin.max()),
                "fraction_raw_at_or_below_p1": float(
                    np.mean(raw <= p1[channel_index])
                ),
                "fraction_raw_at_or_above_p99": float(
                    np.mean(raw >= p99[channel_index])
                ),
                "fraction_raw_negative": float(np.mean(raw < 0.0)),
                "fraction_raw_zero": float(np.mean(raw == 0.0)),
                "fraction_raw_positive": float(np.mean(raw > 0.0)),
                "fraction_vin_at_0v": float(np.mean(np.isclose(vin, 0.0))),
                "fraction_vin_below_zero_voltage": float(
                    np.mean(vin < zero_voltage)
                ),
                "fraction_vin_at_zero_voltage": float(
                    np.mean(np.isclose(vin, zero_voltage))
                ),
                "fraction_vin_above_zero_voltage": float(
                    np.mean(vin > zero_voltage)
                ),
                "fraction_vin_between_threshold_and_1p90v": float(
                    np.mean((vin >= zero_voltage) & (vin < 1.90))
                ),
                "fraction_vin_between_threshold_and_2p05v": float(
                    np.mean((vin >= zero_voltage) & (vin < 2.05))
                ),
                "fraction_vin_at_maximum": float(
                    np.mean(np.isclose(vin, maximum_voltage))
                ),
            }
        )
    return diagnostics


def write_readme(path, manifest):
    selection = manifest["selection"]
    window_starts = [
        str(window["source_start_sample"])
        for window in manifest["random_windows"]["windows"]
    ]
    content = f"""# V3 p1/p99 对称分段线性映射验证

本目录集中保存 V3 的 PWL 输入、LTspice 仿真图像和验证统计。

- 数据：NinaPro DB5 Exercise A
- 样本：S{selection['subject']:02d}，标签 {selection['label']}（{selection['action_name_zh']}），重复 {selection['repetition']}
- 完整序列：{manifest['full_sequence']['sample_count']} 点，{manifest['full_sequence']['duration_seconds']:.3f} s
- 随机窗口起点：{', '.join(window_starts)}
- 采样：200 Hz，每点以近似零阶保持维持 5 ms
- 统计：整个数据集（训练集与测试集）的每通道 overall p1/p99

映射锚点为 `p1 -> 0 V`、`0 -> 1.8639 V`、`p99 -> 3.7278 V`，
并在 p1 以下和 p99 以上限幅。详细参数和逐通道输入诊断见 `manifest.json`；
仿真完成后，图像和输出统计位于 `simulation_results/`。
"""
    path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    if args.window_count <= 0:
        raise ValueError("window-count 必须为正整数")
    if args.zero_voltage <= 0:
        raise ValueError("zero-voltage 必须大于 0")

    full_train_path = base.DATA_ROOT / "full_sequence" / "train.npz"
    window_train_path = base.DATA_ROOT / "slide_window" / "train.npz"
    metadata_path = base.DATA_ROOT / "full_sequence" / "metadata.json"
    full_train = base.load_npz(full_train_path)
    window_train = base.load_npz(window_train_path)
    dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    channel_count = int(full_train["X"].shape[1])
    p1, p99 = load_overall_percentiles(STATISTICS_PATH, channel_count)
    full_selection = base.select_full_sequence(
        full_train,
        args.subject,
        args.label,
        args.repetition,
    )
    windows = base.select_random_windows(
        window_train,
        full_selection,
        args.subject,
        args.label,
        args.repetition,
        args.window_count,
        args.random_seed,
    )

    transition_seconds = args.transition_us * 1e-6
    maximum_voltage = 2.0 * args.zero_voltage
    full_voltage = map_to_voltage(
        full_selection["samples"],
        p1,
        p99,
        args.zero_voltage,
    )

    full_output = VERSION_ROOT / "full_sequence_pwl"
    full_files = []
    for channel in range(1, channel_count + 1):
        filename = base.channel_filename(
            args.subject,
            args.label,
            args.repetition,
            channel,
        )
        output_path = full_output / filename
        base.write_pwl(
            output_path,
            full_voltage[:, channel - 1],
            args.sample_rate,
            transition_seconds,
        )
        full_files.append(str(output_path.relative_to(VERSION_ROOT)).replace("\\", "/"))

    window_records = []
    windows_output = VERSION_ROOT / "windows_sequence_pwl"
    for order, window in enumerate(windows, start=1):
        window_voltage = map_to_voltage(
            window["samples"],
            p1,
            p99,
            args.zero_voltage,
        )
        window_dir = windows_output / (
            f"window_{order:02d}_start_{window['start_sample']:06d}"
        )
        files = []
        for channel in range(1, channel_count + 1):
            filename = base.channel_filename(
                args.subject,
                args.label,
                args.repetition,
                channel,
            )
            output_path = window_dir / filename
            base.write_pwl(
                output_path,
                window_voltage[:, channel - 1],
                args.sample_rate,
                transition_seconds,
            )
            files.append(str(output_path.relative_to(VERSION_ROOT)).replace("\\", "/"))

        window_records.append(
            {
                "order": order,
                "slide_window_dataset_index": window["dataset_index"],
                "source_start_sample": window["start_sample"],
                "local_start_in_full_sequence": window["local_start"],
                "sample_count": int(window["samples"].shape[0]),
                "duration_seconds": window["samples"].shape[0] / args.sample_rate,
                "input_diagnostics_by_channel": mapping_diagnostics(
                    window["samples"],
                    window_voltage,
                    p1,
                    p99,
                    args.zero_voltage,
                ),
                "files": files,
            }
        )

    label_info = dataset_metadata["label_mapping"][str(args.label)]
    manifest = {
        "version": VERSION_NAME,
        "dataset": "NinaPro DB5 Exercise A",
        "selection": {
            "split": "train",
            "subject": args.subject,
            "label": args.label,
            "raw_label": label_info["raw_label"],
            "action_name_zh": label_info["name_zh"],
            "repetition": args.repetition,
        },
        "mapping": {
            "formula": (
                "Vin=0 if x<=p1_c; "
                "Vin=V0*(x-p1_c)/(-p1_c) if p1_c<x<0; "
                "Vin=V0*(1+x/p99_c) if 0<=x<p99_c; "
                "Vin=2*V0 if x>=p99_c"
            ),
            "statistics_source": str(
                STATISTICS_PATH.relative_to(base.PROJECT_ROOT)
            ).replace("\\", "/"),
            "statistics_scope": "整个数据集（训练集+测试集）；按通道；overall p1/p99",
            "channel_p1": p1.tolist(),
            "channel_p99": p99.tolist(),
            "negative_slope_volts_per_raw_unit": (
                args.zero_voltage / (-p1)
            ).tolist(),
            "positive_slope_volts_per_raw_unit": (
                args.zero_voltage / p99
            ).tolist(),
            "zero_voltage": args.zero_voltage,
            "maximum_voltage": maximum_voltage,
            "clipping_rule": "x<=p1_c 固定为 0 V；x>=p99_c 固定为最大电压",
        },
        "pwl": {
            "sample_rate_hz": args.sample_rate,
            "sample_interval_seconds": 1.0 / args.sample_rate,
            "hold_mode": "近似零阶保持",
            "transition_seconds": transition_seconds,
            "duration_rule": "N 个样本的最后时间点为 N/sample_rate",
        },
        "full_sequence": {
            "full_sequence_dataset_index": full_selection["sequence_index"],
            "source_start_sample": full_selection["source_start"],
            "source_end_sample": full_selection["source_end"],
            "sample_count": int(full_selection["samples"].shape[0]),
            "duration_seconds": full_selection["samples"].shape[0]
            / args.sample_rate,
            "input_diagnostics_by_channel": mapping_diagnostics(
                full_selection["samples"],
                full_voltage,
                p1,
                p99,
                args.zero_voltage,
            ),
            "files": full_files,
        },
        "random_windows": {
            "seed": args.random_seed,
            "sampling": "同一完整非静息动作内，无放回随机抽取，输出按时间排序",
            "count": args.window_count,
            "windows": window_records,
        },
    }
    VERSION_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = VERSION_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(VERSION_ROOT / "README.md", manifest)

    print(
        json.dumps(
            {
                "version": VERSION_NAME,
                "zero_voltage": args.zero_voltage,
                "maximum_voltage": maximum_voltage,
                "full_sequence_samples": manifest["full_sequence"]["sample_count"],
                "full_sequence_duration_seconds": manifest["full_sequence"][
                    "duration_seconds"
                ],
                "random_window_starts": [
                    item["source_start_sample"] for item in window_records
                ],
                "full_pwl_files": len(full_files),
                "window_pwl_files": sum(len(item["files"]) for item in window_records),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
