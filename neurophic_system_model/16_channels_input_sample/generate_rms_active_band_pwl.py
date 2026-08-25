"""生成采用因果 RMS 包络和活动区映射的 V2 LTspice PWL。"""

import argparse
import json
from pathlib import Path

import numpy as np

import generate_pwl as base


VERSION_ROOT = base.SAMPLE_ROOT / "rms_active_band"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--window-count", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=float, default=200.0)
    parser.add_argument("--rms-window-samples", type=int, default=5)
    parser.add_argument("--rms-quantile", type=float, default=0.95)
    parser.add_argument("--activity-threshold", type=float, default=0.20)
    parser.add_argument("--active-min-voltage", type=float, default=2.05)
    parser.add_argument("--active-max-voltage", type=float, default=2.35)
    parser.add_argument("--hard-max-voltage", type=float, default=2.35)
    parser.add_argument("--transition-us", type=float, default=1.0)
    return parser.parse_args()


def compute_causal_rms(samples, offsets, mean, window_samples):
    samples = np.asarray(samples, dtype=np.float64)
    envelope = np.empty_like(samples)
    for sequence_index in range(len(offsets) - 1):
        start = int(offsets[sequence_index])
        end = int(offsets[sequence_index + 1])
        centered_squared = (samples[start:end] - mean) ** 2
        cumulative = np.vstack(
            [np.zeros((1, samples.shape[1])), np.cumsum(centered_squared, axis=0)]
        )
        indices = np.arange(end - start)
        lower = np.maximum(0, indices - window_samples + 1)
        counts = (indices - lower + 1)[:, None]
        envelope[start:end] = np.sqrt(
            (cumulative[indices + 1] - cumulative[lower]) / counts
        )
    return envelope


def map_envelope_to_voltage(
    envelope,
    q_rms,
    activity_threshold,
    active_min_voltage,
    active_max_voltage,
    hard_max_voltage,
):
    normalized = np.clip(envelope / q_rms, 0.0, 1.0)
    active_scale = np.clip(
        (normalized - activity_threshold) / (1.0 - activity_threshold),
        0.0,
        1.0,
    )
    voltage = np.where(
        normalized < activity_threshold,
        0.0,
        active_min_voltage
        + (active_max_voltage - active_min_voltage) * active_scale,
    )
    return np.clip(voltage, 0.0, hard_max_voltage), normalized


def validate_args(args):
    if args.window_count <= 0:
        raise ValueError("window-count 必须为正整数")
    if args.rms_window_samples <= 0:
        raise ValueError("rms-window-samples 必须为正整数")
    if not 0 < args.rms_quantile < 1:
        raise ValueError("rms-quantile 必须位于 0 和 1 之间")
    if not 0 <= args.activity_threshold < 1:
        raise ValueError("activity-threshold 必须位于 [0, 1) 内")
    if not 0 < args.active_min_voltage <= args.active_max_voltage:
        raise ValueError("活动区电压范围无效")
    if args.active_max_voltage > args.hard_max_voltage:
        raise ValueError("活动区最大电压不能超过硬限幅")


def main():
    args = parse_args()
    validate_args(args)

    full_train_path = base.DATA_ROOT / "full_sequence" / "train.npz"
    window_train_path = base.DATA_ROOT / "slide_window" / "train.npz"
    metadata_path = base.DATA_ROOT / "full_sequence" / "metadata.json"
    full_train = base.load_npz(full_train_path)
    window_train = base.load_npz(window_train_path)
    dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # 先在完整训练序列上计算包络，确保滑动窗口切片保留此前的 RMS 历史。
    training_samples = np.asarray(full_train["X"], dtype=np.float64)
    channel_mean = training_samples.mean(axis=0)
    training_envelope = compute_causal_rms(
        training_samples,
        full_train["offsets"],
        channel_mean,
        args.rms_window_samples,
    )
    channel_q_rms = np.quantile(training_envelope, args.rms_quantile, axis=0)
    if np.any(channel_q_rms <= 0):
        raise ValueError("训练集存在 RMS 分位数不大于 0 的通道")

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
    selected_envelope = training_envelope[
        full_selection["offset_start"] : full_selection["offset_end"]
    ]
    full_voltage, full_normalized = map_envelope_to_voltage(
        selected_envelope,
        channel_q_rms,
        args.activity_threshold,
        args.active_min_voltage,
        args.active_max_voltage,
        args.hard_max_voltage,
    )

    transition_seconds = args.transition_us * 1e-6
    full_output = VERSION_ROOT / "full_sequence_pwl"
    full_files = []
    for channel in range(1, full_voltage.shape[1] + 1):
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
        local_start = window["local_start"]
        local_end = local_start + window["samples"].shape[0]
        window_voltage = full_voltage[local_start:local_end]
        window_normalized = full_normalized[local_start:local_end]
        window_dir = windows_output / (
            f"window_{order:02d}_start_{window['start_sample']:06d}"
        )
        files = []
        for channel in range(1, window_voltage.shape[1] + 1):
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
                "local_start_in_full_sequence": local_start,
                "sample_count": int(window_voltage.shape[0]),
                "duration_seconds": window_voltage.shape[0] / args.sample_rate,
                "active_fraction_by_channel": np.mean(
                    window_normalized >= args.activity_threshold,
                    axis=0,
                ).tolist(),
                "files": files,
            }
        )

    label_info = dataset_metadata["label_mapping"][str(args.label)]
    manifest = {
        "version": "rms_active_band_v2",
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
                "e=causal_rms(x-mu,L=5); a=clip(e/Q95_rms,0,1); "
                "Vin=0 if a<0.20 else 2.05+0.30*(a-0.20)/0.80"
            ),
            "statistics_source": str(full_train_path.relative_to(base.PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "statistics_scope": "全部训练完整序列；按通道；不包含测试集",
            "rms_history_rule": "先在完整动作上计算因果 RMS，再切滑动窗口",
            "rms_window_samples": args.rms_window_samples,
            "rms_window_seconds": args.rms_window_samples / args.sample_rate,
            "rms_quantile": args.rms_quantile,
            "channel_mean": channel_mean.tolist(),
            "channel_q_rms": channel_q_rms.tolist(),
            "activity_threshold": args.activity_threshold,
            "active_min_voltage": args.active_min_voltage,
            "active_max_voltage": args.active_max_voltage,
            "hard_max_voltage": args.hard_max_voltage,
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
            "sample_count": int(full_voltage.shape[0]),
            "duration_seconds": full_voltage.shape[0] / args.sample_rate,
            "active_fraction_by_channel": np.mean(
                full_normalized >= args.activity_threshold,
                axis=0,
            ).tolist(),
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
    print(
        json.dumps(
            {
                "version": manifest["version"],
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
