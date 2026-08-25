"""从 NinaPro DB5 训练数据生成 16 通道 LTspice PWL 激励。"""

import argparse
import json
from pathlib import Path

import numpy as np


SAMPLE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = SAMPLE_ROOT / "q99_linear"
PROJECT_ROOT = SAMPLE_ROOT.parents[1]
DATA_ROOT = PROJECT_ROOT / "ninapro_data" / "processed" / "exerciseA"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--window-count", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=float, default=200.0)
    parser.add_argument("--nominal-max-voltage", type=float, default=2.30)
    parser.add_argument("--hard-max-voltage", type=float, default=2.35)
    parser.add_argument("--transition-us", type=float, default=1.0)
    return parser.parse_args()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def compute_training_mapping_stats(full_train):
    # 使用未重复的完整训练序列，避免重叠窗口让同一采样点被反复计权。
    samples = np.asarray(full_train["X"], dtype=np.float64)
    mean = samples.mean(axis=0)
    q99 = np.quantile(np.abs(samples - mean), 0.99, axis=0)
    if np.any(q99 <= 0):
        raise ValueError("训练集存在 Q99 不大于 0 的通道，无法进行电压映射")
    return mean, q99


def map_to_voltage(samples, mean, q99, nominal_max, hard_max):
    samples = np.asarray(samples, dtype=np.float64)
    normalized = np.clip(np.abs(samples - mean) / q99, 0.0, 1.0)
    # hard_max 是末级保护；按当前公式正常结果不会超过 nominal_max。
    return np.clip(nominal_max * normalized, 0.0, hard_max)


def write_pwl(path, values, sample_rate, transition_seconds):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"不能为 {path} 写入空波形")

    dt = 1.0 / sample_rate
    if not 0 < transition_seconds < dt:
        raise ValueError("跳变过渡时间必须大于 0 且小于采样间隔")

    lines = [f"{0.0:.9f} {values[0]:.9f}"]
    for index, value in enumerate(values):
        interval_end = (index + 1) * dt
        hold_end = interval_end - transition_seconds
        lines.append(f"{hold_end:.9f} {value:.9f}")

        # 最后一个值保持到 N/fs，使 PWL 总时长严格等于实际采样时长。
        next_value = values[index + 1] if index + 1 < values.size else value
        lines.append(f"{interval_end:.9f} {next_value:.9f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def select_full_sequence(full_train, subject, label, repetition):
    mask = (
        (full_train["subject"] == subject)
        & (full_train["y"] == label)
        & (full_train["repetition"] == repetition)
    )
    indices = np.flatnonzero(mask)
    if indices.size != 1:
        raise ValueError(
            "完整序列选择必须唯一："
            f"subject={subject}, label={label}, repetition={repetition}, "
            f"实际找到 {indices.size} 条"
        )

    sequence_index = int(indices[0])
    offset_start = int(full_train["offsets"][sequence_index])
    offset_end = int(full_train["offsets"][sequence_index + 1])
    return {
        "sequence_index": sequence_index,
        "offset_start": offset_start,
        "offset_end": offset_end,
        "source_start": int(full_train["start_sample"][sequence_index]),
        "source_end": int(full_train["end_sample"][sequence_index]),
        "samples": full_train["X"][offset_start:offset_end],
    }


def select_random_windows(
    window_train,
    full_selection,
    subject,
    label,
    repetition,
    count,
    seed,
):
    source_start = full_selection["source_start"]
    source_end = full_selection["source_end"]
    window_size = int(window_train["X"].shape[-1])

    mask = (
        (window_train["subject"] == subject)
        & (window_train["y"] == label)
        & (window_train["repetition"] == repetition)
        & (window_train["start_sample"] >= source_start)
        & (window_train["start_sample"] + window_size <= source_end)
    )
    candidates = np.flatnonzero(mask)
    if candidates.size < count:
        raise ValueError(f"候选窗口只有 {candidates.size} 个，少于请求的 {count} 个")

    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=count, replace=False)
    # 抽样本身是随机的；输出按原始时间排序，便于检查和复用。
    selected = selected[np.argsort(window_train["start_sample"][selected])]

    windows = []
    for dataset_index in selected:
        dataset_index = int(dataset_index)
        start_sample = int(window_train["start_sample"][dataset_index])
        local_start = start_sample - source_start
        window_samples = window_train["X"][dataset_index].T
        full_slice = full_selection["samples"][local_start : local_start + window_size]
        if not np.array_equal(window_samples, full_slice):
            raise ValueError(f"窗口 {dataset_index} 与完整序列中的对应片段不一致")
        windows.append(
            {
                "dataset_index": dataset_index,
                "start_sample": start_sample,
                "local_start": local_start,
                "samples": window_samples,
            }
        )
    return windows


def channel_filename(subject, label, repetition, channel):
    return (
        f"S{subject:02d}_label{label:02d}_rep{repetition:02d}_"
        f"ch{channel:02d}.pwl"
    )


def main():
    args = parse_args()
    if args.window_count <= 0:
        raise ValueError("window-count 必须为正整数")
    if args.nominal_max_voltage > args.hard_max_voltage:
        raise ValueError("标称最大电压不能超过硬限幅")

    full_train_path = DATA_ROOT / "full_sequence" / "train.npz"
    window_train_path = DATA_ROOT / "slide_window" / "train.npz"
    metadata_path = DATA_ROOT / "full_sequence" / "metadata.json"
    full_train = load_npz(full_train_path)
    window_train = load_npz(window_train_path)
    dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    mean, q99 = compute_training_mapping_stats(full_train)
    full_selection = select_full_sequence(
        full_train,
        args.subject,
        args.label,
        args.repetition,
    )
    windows = select_random_windows(
        window_train,
        full_selection,
        args.subject,
        args.label,
        args.repetition,
        args.window_count,
        args.random_seed,
    )

    transition_seconds = args.transition_us * 1e-6
    full_voltage = map_to_voltage(
        full_selection["samples"],
        mean,
        q99,
        args.nominal_max_voltage,
        args.hard_max_voltage,
    )

    full_output = OUTPUT_ROOT / "full_sequence_pwl"
    full_files = []
    for channel in range(1, full_voltage.shape[1] + 1):
        filename = channel_filename(
            args.subject,
            args.label,
            args.repetition,
            channel,
        )
        output_path = full_output / filename
        write_pwl(
            output_path,
            full_voltage[:, channel - 1],
            args.sample_rate,
            transition_seconds,
        )
        full_files.append(str(output_path.relative_to(OUTPUT_ROOT)).replace("\\", "/"))

    window_records = []
    windows_output = OUTPUT_ROOT / "windows_sequence_pwl"
    for order, window in enumerate(windows, start=1):
        window_voltage = map_to_voltage(
            window["samples"],
            mean,
            q99,
            args.nominal_max_voltage,
            args.hard_max_voltage,
        )
        window_dir = windows_output / (
            f"window_{order:02d}_start_{window['start_sample']:06d}"
        )
        files = []
        for channel in range(1, window_voltage.shape[1] + 1):
            filename = channel_filename(
                args.subject,
                args.label,
                args.repetition,
                channel,
            )
            output_path = window_dir / filename
            write_pwl(
                output_path,
                window_voltage[:, channel - 1],
                args.sample_rate,
                transition_seconds,
            )
            files.append(str(output_path.relative_to(OUTPUT_ROOT)).replace("\\", "/"))

        window_records.append(
            {
                "order": order,
                "slide_window_dataset_index": window["dataset_index"],
                "source_start_sample": window["start_sample"],
                "local_start_in_full_sequence": window["local_start"],
                "sample_count": int(window["samples"].shape[0]),
                "duration_seconds": window["samples"].shape[0] / args.sample_rate,
                "files": files,
            }
        )

    label_info = dataset_metadata["label_mapping"][str(args.label)]
    manifest = {
        "version": "q99_linear_v1",
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
            "formula": "Vin=clip(2.30*clip(abs(x-mu_c)/Q99_c,0,1),0,2.35)",
            "statistics_source": str(full_train_path.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "statistics_scope": "全部训练完整序列；按通道；不包含测试集",
            "channel_mean": mean.tolist(),
            "channel_q99_abs_centered": q99.tolist(),
            "nominal_max_voltage": args.nominal_max_voltage,
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
            "sample_count": int(full_selection["samples"].shape[0]),
            "duration_seconds": full_selection["samples"].shape[0] / args.sample_rate,
            "files": full_files,
        },
        "random_windows": {
            "seed": args.random_seed,
            "sampling": "同一完整非静息动作内，无放回随机抽取，输出按时间排序",
            "count": args.window_count,
            "windows": window_records,
        },
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
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
