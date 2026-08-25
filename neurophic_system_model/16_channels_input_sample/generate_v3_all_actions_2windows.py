"""为 S1 的全部12个动作各抽取2个窗口并生成 V3 LTspice PWL。"""

import argparse
import json
from pathlib import Path

import numpy as np

import generate_pwl as base
import generate_v3_p1p99_symmetric_pwl as v3


VERSION_NAME = "v3_all_actions_2windows_encoding"
VERSION_ROOT = base.SAMPLE_ROOT / VERSION_NAME


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--windows-per-action", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=float, default=200.0)
    parser.add_argument("--zero-voltage", type=float, default=1.8639)
    parser.add_argument("--transition-us", type=float, default=1.0)
    return parser.parse_args()


def select_action_windows(
    full_train,
    window_train,
    subject,
    label,
    count,
    rng,
):
    action_mask = (full_train["subject"] == subject) & (full_train["y"] == label)
    repetitions = np.unique(full_train["repetition"][action_mask])
    if repetitions.size < count:
        raise ValueError(
            f"label={label} 只有 {repetitions.size} 个 repetition，少于请求的 {count} 个"
        )

    # 先抽不同 repetition，再在各自序列中抽一个窗口，避免两个窗口过度相似。
    selected_repetitions = np.sort(
        rng.choice(repetitions, size=count, replace=False).astype(int)
    )
    selected_windows = []
    for repetition in selected_repetitions:
        full_selection = base.select_full_sequence(
            full_train,
            subject,
            label,
            repetition,
        )
        window_size = int(window_train["X"].shape[-1])
        mask = (
            (window_train["subject"] == subject)
            & (window_train["y"] == label)
            & (window_train["repetition"] == repetition)
            & (window_train["start_sample"] >= full_selection["source_start"])
            & (
                window_train["start_sample"] + window_size
                <= full_selection["source_end"]
            )
        )
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            raise ValueError(
                f"subject={subject}, label={label}, repetition={repetition} 没有候选窗口"
            )

        dataset_index = int(rng.choice(candidates))
        start_sample = int(window_train["start_sample"][dataset_index])
        local_start = start_sample - full_selection["source_start"]
        samples = window_train["X"][dataset_index].T
        full_slice = full_selection["samples"][
            local_start : local_start + window_size
        ]
        if not np.array_equal(samples, full_slice):
            raise ValueError(f"窗口 {dataset_index} 与完整序列切片不一致")

        selected_windows.append(
            {
                "slide_window_dataset_index": dataset_index,
                "repetition": int(repetition),
                "source_start_sample": start_sample,
                "local_start_in_full_sequence": local_start,
                "samples": samples,
            }
        )
    return selected_windows


def write_readme(path, manifest):
    content = f"""# V3 全动作双窗口脉冲编码

本目录保存 S{manifest['selection']['subject']:02d} 在 NinaPro DB5 Exercise A
全部12个非静息动作上的 V3 编码数据。每个动作从不同 repetition 随机抽取2个窗口，
每个窗口包含40个原始样本、16个通道，持续0.200 s。

- 随机种子：{manifest['selection']['random_seed']}
- V3锚点：`p1 -> 0 V`、`0 -> 1.8639 V`、`p99 -> 3.7278 V`
- p1/p99：整个数据集（训练集+测试集）的每通道 overall 统计量
- PWL：200 Hz，1 µs过渡近似零阶保持，每个样本保持5 ms
- 窗口仿真：每个通道独立从擦除后的初始状态启动

`pwl/` 中包含24个窗口、共384个PWL。LTspice运行后的脉冲时间戳、SNN张量、
动作汇总图及报告保存在 `encoding_results/`。
"""
    path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    if args.windows_per_action != 2:
        raise ValueError("当前任务固定要求每个动作抽取2个窗口")
    if args.zero_voltage <= 0:
        raise ValueError("zero-voltage 必须大于0")

    full_train_path = base.DATA_ROOT / "full_sequence" / "train.npz"
    window_train_path = base.DATA_ROOT / "slide_window" / "train.npz"
    metadata_path = base.DATA_ROOT / "full_sequence" / "metadata.json"
    full_train = base.load_npz(full_train_path)
    window_train = base.load_npz(window_train_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    labels = sorted(
        int(value)
        for value in np.unique(
            full_train["y"][full_train["subject"] == args.subject]
        )
    )
    if labels != list(range(12)):
        raise ValueError(f"预期12个动作标签0..11，实际为 {labels}")

    channel_count = int(full_train["X"].shape[1])
    p1, p99 = v3.load_overall_percentiles(v3.STATISTICS_PATH, channel_count)
    maximum_voltage = 2.0 * args.zero_voltage
    transition_seconds = args.transition_us * 1e-6
    rng = np.random.default_rng(args.random_seed)

    actions = []
    global_window_index = 0
    for label in labels:
        selected = select_action_windows(
            full_train,
            window_train,
            args.subject,
            label,
            args.windows_per_action,
            rng,
        )
        action_info = metadata["label_mapping"][str(label)]
        window_records = []
        for order, window in enumerate(selected, start=1):
            voltage = v3.map_to_voltage(
                window["samples"],
                p1,
                p99,
                args.zero_voltage,
            )
            output_dir = (
                VERSION_ROOT
                / "pwl"
                / f"label_{label:02d}"
                / (
                    f"window_{order:02d}_rep_{window['repetition']:02d}_"
                    f"start_{window['source_start_sample']:06d}"
                )
            )
            files = []
            for channel in range(1, channel_count + 1):
                filename = base.channel_filename(
                    args.subject,
                    label,
                    window["repetition"],
                    channel,
                )
                output_path = output_dir / filename
                base.write_pwl(
                    output_path,
                    voltage[:, channel - 1],
                    args.sample_rate,
                    transition_seconds,
                )
                files.append(
                    str(output_path.relative_to(VERSION_ROOT)).replace("\\", "/")
                )

            window_records.append(
                {
                    "global_window_index": global_window_index,
                    "order_within_action": order,
                    "slide_window_dataset_index": window[
                        "slide_window_dataset_index"
                    ],
                    "repetition": window["repetition"],
                    "source_start_sample": window["source_start_sample"],
                    "local_start_in_full_sequence": window[
                        "local_start_in_full_sequence"
                    ],
                    "sample_count": int(window["samples"].shape[0]),
                    "duration_seconds": window["samples"].shape[0]
                    / args.sample_rate,
                    "input_diagnostics_by_channel": v3.mapping_diagnostics(
                        window["samples"],
                        voltage,
                        p1,
                        p99,
                        args.zero_voltage,
                    ),
                    "files": files,
                }
            )
            global_window_index += 1

        actions.append(
            {
                "label": label,
                "raw_label": action_info["raw_label"],
                "action_name_zh": action_info["name_zh"],
                "windows": window_records,
            }
        )

    manifest = {
        "version": VERSION_NAME,
        "dataset": "NinaPro DB5 Exercise A",
        "selection": {
            "split": "train",
            "subject": args.subject,
            "labels": labels,
            "action_count": len(labels),
            "windows_per_action": args.windows_per_action,
            "different_repetition_per_action": True,
            "random_seed": args.random_seed,
        },
        "mapping": {
            "formula": (
                "Vin=0 if x<=p1_c; "
                "Vin=V0*(x-p1_c)/(-p1_c) if p1_c<x<0; "
                "Vin=V0*(1+x/p99_c) if 0<=x<p99_c; "
                "Vin=2*V0 if x>=p99_c"
            ),
            "statistics_source": str(
                v3.STATISTICS_PATH.relative_to(base.PROJECT_ROOT)
            ).replace("\\", "/"),
            "statistics_scope": "整个数据集（训练集+测试集）；按通道；overall p1/p99",
            "channel_p1": p1.tolist(),
            "channel_p99": p99.tolist(),
            "zero_voltage": args.zero_voltage,
            "maximum_voltage": maximum_voltage,
        },
        "pwl": {
            "sample_rate_hz": args.sample_rate,
            "sample_interval_seconds": 1.0 / args.sample_rate,
            "hold_mode": "近似零阶保持",
            "transition_seconds": transition_seconds,
            "duration_rule": "40个样本的最后时间点为0.200 s",
        },
        "encoding": {
            "final_out_event_rule": "V(final_out) 向上越过1.60 V",
            "window_state_rule": "每个通道窗口独立仿真，等效于窗口前完成时钟擦除",
        },
        "total_windows": global_window_index,
        "total_channel_pwl_files": global_window_index * channel_count,
        "actions": actions,
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
                "actions": len(actions),
                "windows": manifest["total_windows"],
                "pwl_files": manifest["total_channel_pwl_files"],
                "selections": [
                    {
                        "label": action["label"],
                        "windows": [
                            {
                                "repetition": window["repetition"],
                                "start_sample": window["source_start_sample"],
                            }
                            for window in action["windows"]
                        ],
                    }
                    for action in actions
                ],
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
