"""运行 V3 全动作窗口仿真并导出 final_out 脉冲编码数据。"""

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulate_and_plot as simulation


SAMPLE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SAMPLE_ROOT.parents[1]
DEFAULT_INPUT_ROOT = SAMPLE_ROOT / "v3_all_actions_2windows_encoding"
EVENT_THRESHOLD = 1.60


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltspice", type=Path, default=simulation.DEFAULT_LTSPICE)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def build_tasks(manifest, input_root):
    tasks = []
    subject = int(manifest["selection"]["subject"])
    for action in manifest["actions"]:
        for window in action["windows"]:
            for channel, relative_file in enumerate(window["files"], start=1):
                tasks.append(
                    {
                        "subject": subject,
                        "label": int(action["label"]),
                        "raw_label": int(action["raw_label"]),
                        "action_name_zh": action["action_name_zh"],
                        "global_window_index": int(window["global_window_index"]),
                        "order_within_action": int(window["order_within_action"]),
                        "repetition": int(window["repetition"]),
                        "source_start_sample": int(window["source_start_sample"]),
                        "channel": channel,
                        "duration_seconds": float(window["duration_seconds"]),
                        "pwl": input_root / relative_file,
                    }
                )
    return tasks


def rising_crossing_times(time, values, threshold):
    indices = np.flatnonzero(
        (values[:-1] < threshold) & (values[1:] >= threshold)
    )
    if indices.size == 0:
        return np.empty(0, dtype=np.float64)

    left_time = time[indices]
    right_time = time[indices + 1]
    left_value = values[indices]
    right_value = values[indices + 1]
    denominator = right_value - left_value
    fraction = np.divide(
        threshold - left_value,
        denominator,
        out=np.zeros_like(left_value),
        where=denominator != 0,
    )
    return left_time + fraction * (right_time - left_time)


def simulate_task(task, executable, work_parent, timeout):
    with tempfile.TemporaryDirectory(prefix="run_", dir=work_parent) as temp_dir:
        netlist_path = Path(temp_dir) / "simulation.net"
        netlist_path.write_text(
            simulation.make_netlist(task["pwl"], task["duration_seconds"]),
            encoding="ascii",
        )
        raw_path, log_text = simulation.run_ltspice(
            executable,
            netlist_path,
            timeout,
        )
        signals = simulation.read_ltspice_raw(raw_path)

    time = signals["time"]
    vin = signals["v(n001)"]
    vout = signals["v(vout)"]
    final_out = signals["v(final_out)"]
    vout_spikes = rising_crossing_times(time, vout, EVENT_THRESHOLD)
    final_spikes = rising_crossing_times(time, final_out, EVENT_THRESHOLD)
    return {
        **task,
        "ltspice_completed": "Total elapsed time:" in log_text,
        "raw_point_count": int(time.size),
        "time_end_seconds": float(time[-1]),
        "vin_range": [float(vin.min()), float(vin.max())],
        "vout_range": [float(vout.min()), float(vout.max())],
        "final_out_range": [float(final_out.min()), float(final_out.max())],
        "vout_spike_count": int(vout_spikes.size),
        "vout_rate_hz": float(vout_spikes.size / task["duration_seconds"]),
        "final_out_spike_count": int(final_spikes.size),
        "final_out_rate_hz": float(
            final_spikes.size / task["duration_seconds"]
        ),
        "final_out_spike_times_seconds": final_spikes.tolist(),
    }


def relative_path(path):
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def write_spike_events(path, results):
    fieldnames = [
        "subject",
        "label",
        "raw_label",
        "action_name_zh",
        "global_window_index",
        "order_within_action",
        "repetition",
        "source_start_sample",
        "channel",
        "spike_index",
        "spike_time_seconds",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for spike_index, spike_time in enumerate(
                result["final_out_spike_times_seconds"],
                start=1,
            ):
                writer.writerow(
                    {
                        "subject": result["subject"],
                        "label": result["label"],
                        "raw_label": result["raw_label"],
                        "action_name_zh": result["action_name_zh"],
                        "global_window_index": result["global_window_index"],
                        "order_within_action": result["order_within_action"],
                        "repetition": result["repetition"],
                        "source_start_sample": result["source_start_sample"],
                        "channel": result["channel"],
                        "spike_index": spike_index,
                        "spike_time_seconds": f"{spike_time:.9f}",
                    }
                )


def build_spike_tensors(results, window_count, channel_count, duration):
    bins_1ms = int(round(duration / 0.001))
    bins_5ms = int(round(duration / 0.005))
    counts_1ms = np.zeros(
        (window_count, channel_count, bins_1ms),
        dtype=np.uint8,
    )
    counts_5ms = np.zeros(
        (window_count, channel_count, bins_5ms),
        dtype=np.uint8,
    )
    for result in results:
        window_index = result["global_window_index"]
        channel_index = result["channel"] - 1
        times = np.asarray(
            result["final_out_spike_times_seconds"],
            dtype=np.float64,
        )
        indices_1ms = np.clip((times / 0.001).astype(int), 0, bins_1ms - 1)
        indices_5ms = np.clip((times / 0.005).astype(int), 0, bins_5ms - 1)
        np.add.at(counts_1ms[window_index, channel_index], indices_1ms, 1)
        np.add.at(counts_5ms[window_index, channel_index], indices_5ms, 1)
    return counts_1ms, counts_5ms


def window_summaries(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result["global_window_index"], []).append(result)

    summaries = []
    for window_index in sorted(grouped):
        items = sorted(grouped[window_index], key=lambda item: item["channel"])
        counts = np.asarray(
            [item["final_out_spike_count"] for item in items],
            dtype=np.int64,
        )
        summaries.append(
            {
                "global_window_index": window_index,
                "label": items[0]["label"],
                "raw_label": items[0]["raw_label"],
                "action_name_zh": items[0]["action_name_zh"],
                "order_within_action": items[0]["order_within_action"],
                "repetition": items[0]["repetition"],
                "source_start_sample": items[0]["source_start_sample"],
                "total_spikes": int(counts.sum()),
                "active_channel_count": int(np.sum(counts > 0)),
                "mean_spikes_per_channel": float(counts.mean()),
                "minimum_spikes_per_channel": int(counts.min()),
                "maximum_spikes_per_channel": int(counts.max()),
                "mean_rate_hz": float(
                    counts.mean() / items[0]["duration_seconds"]
                ),
            }
        )
    return summaries


def write_window_summary(path, summaries):
    fieldnames = list(summaries[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def plot_action_rasters(output_root, manifest, results):
    plot_root = output_root / "action_summary_plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for action in manifest["actions"]:
        label = int(action["label"])
        action_results = [item for item in results if item["label"] == label]
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for axis, window in zip(axes, action["windows"]):
            window_results = sorted(
                [
                    item
                    for item in action_results
                    if item["global_window_index"]
                    == window["global_window_index"]
                ],
                key=lambda item: item["channel"],
            )
            spike_sequences = [
                item["final_out_spike_times_seconds"] for item in window_results
            ]
            axis.eventplot(
                spike_sequences,
                lineoffsets=np.arange(1, 17),
                linelengths=0.75,
                linewidths=0.8,
                colors="#dc2626",
            )
            axis.set_ylim(0.5, 16.5)
            axis.set_yticks(np.arange(1, 17))
            axis.set_ylabel("Channel")
            axis.set_title(
                f"Window {window['order_within_action']} | "
                f"rep {window['repetition']} | start {window['source_start_sample']}"
            )
            axis.grid(True, axis="x", color="#d1d5db", linewidth=0.5)

        axes[-1].set_xlim(0.0, 0.2)
        axes[-1].set_xlabel("Time (s)")
        figure.suptitle(
            f"S{action_results[0]['subject']:02d} label {label:02d} "
            f"raw label {action['raw_label']} | "
            "final_out rising crossings"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        output_path = plot_root / f"label_{label:02d}_spike_raster.png"
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(relative_path(output_path))
    return paths


def write_report(path, manifest, summaries, results, plot_paths):
    total_spikes = sum(item["final_out_spike_count"] for item in results)
    active_results = sum(item["final_out_spike_count"] > 0 for item in results)
    mean_rate = np.mean([item["final_out_rate_hz"] for item in results])
    lines = [
        "# V3 全动作双窗口编码报告",
        "",
        f"- 动作数：{manifest['selection']['action_count']}",
        f"- 窗口数：{manifest['total_windows']}",
        f"- 通道仿真数：{len(results)}",
        f"- 成功完成：{sum(item['ltspice_completed'] for item in results)}",
        f"- final_out 总脉冲数：{total_spikes}",
        f"- 有脉冲的通道窗口：{active_results}/{len(results)}",
        f"- 通道窗口平均频率：{mean_rate:.2f} Hz",
        "",
        "## 每个动作的两个窗口",
        "",
        "| 动作标签 | 动作 | 窗口 | repetition | 起点 | 总脉冲 | 活动通道 | 平均频率/Hz |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['label']} | {summary['action_name_zh']} | "
            f"{summary['order_within_action']} | {summary['repetition']} | "
            f"{summary['source_start_sample']} | {summary['total_spikes']} | "
            f"{summary['active_channel_count']} | {summary['mean_rate_hz']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `spike_events.csv`：每个 final_out 上升沿的完整时间戳。",
            "- `snn_spike_tensors.npz`：1 ms和5 ms分箱的SNN脉冲计数张量。",
            "- `window_encoding_summary.csv`：24个窗口的汇总统计。",
            "- `encoding_manifest.json`：逐通道仿真结果和脉冲时间戳。",
            f"- `action_summary_plots/`：{len(plot_paths)}张动作级16通道脉冲栅格图。",
            "",
            "所有窗口均独立从电路初始状态运行，等效于窗口开始前完成时钟擦除。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not args.ltspice.is_file():
        raise FileNotFoundError(f"找不到LTspice：{args.ltspice}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit 必须为正整数")

    input_root = args.input_root.resolve()
    manifest_path = input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = build_tasks(manifest, input_root)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    output_root = input_root / "encoding_results"
    output_root.mkdir(parents=True, exist_ok=True)
    work_parent = input_root / "ltspice_work"
    work_parent.mkdir(parents=True, exist_ok=True)

    results = []
    for index, task in enumerate(tasks, start=1):
        result = simulate_task(task, args.ltspice, work_parent, args.timeout)
        result["pwl"] = relative_path(result["pwl"])
        results.append(result)
        print(
            f"[{index:03d}/{len(tasks):03d}] label={task['label']:02d} "
            f"window={task['order_within_action']} ch={task['channel']:02d} "
            f"spikes={result['final_out_spike_count']}",
            flush=True,
        )

    if len(results) != manifest["total_channel_pwl_files"]:
        print(
            "Smoke run completed; limit 模式不写入最终编码文件。",
            flush=True,
        )
        return

    spike_events_path = output_root / "spike_events.csv"
    write_spike_events(spike_events_path, results)
    summaries = window_summaries(results)
    summary_path = output_root / "window_encoding_summary.csv"
    write_window_summary(summary_path, summaries)

    duration = float(manifest["actions"][0]["windows"][0]["duration_seconds"])
    counts_1ms, counts_5ms = build_spike_tensors(
        results,
        manifest["total_windows"],
        16,
        duration,
    )
    ordered_summaries = sorted(
        summaries,
        key=lambda item: item["global_window_index"],
    )
    tensor_path = output_root / "snn_spike_tensors.npz"
    np.savez_compressed(
        tensor_path,
        spike_counts_1ms=counts_1ms,
        spike_binary_1ms=(counts_1ms > 0).astype(np.uint8),
        spike_counts_5ms=counts_5ms,
        spike_binary_5ms=(counts_5ms > 0).astype(np.uint8),
        label=np.asarray([item["label"] for item in ordered_summaries]),
        raw_label=np.asarray([item["raw_label"] for item in ordered_summaries]),
        repetition=np.asarray(
            [item["repetition"] for item in ordered_summaries]
        ),
        source_start_sample=np.asarray(
            [item["source_start_sample"] for item in ordered_summaries]
        ),
        order_within_action=np.asarray(
            [item["order_within_action"] for item in ordered_summaries]
        ),
        bin_size_1ms_seconds=np.asarray(0.001),
        bin_size_5ms_seconds=np.asarray(0.005),
        window_duration_seconds=np.asarray(duration),
    )

    plot_paths = plot_action_rasters(output_root, manifest, results)
    encoding_manifest = {
        "version": manifest["version"],
        "source_manifest": relative_path(manifest_path),
        "source_circuit": relative_path(
            SAMPLE_ROOT.parent / "System_with_TIA.asc"
        ),
        "ltspice_executable": str(args.ltspice),
        "event_threshold_voltage": EVENT_THRESHOLD,
        "state_rule": "每个通道窗口独立仿真，等效于窗口前时钟擦除",
        "task_count": len(results),
        "total_final_out_spikes": int(
            sum(item["final_out_spike_count"] for item in results)
        ),
        "files": {
            "spike_events_csv": relative_path(spike_events_path),
            "window_summary_csv": relative_path(summary_path),
            "snn_tensor_npz": relative_path(tensor_path),
            "action_summary_plots": plot_paths,
        },
        "window_summaries": summaries,
        "results": results,
    }
    encoding_manifest_path = output_root / "encoding_manifest.json"
    encoding_manifest_path.write_text(
        json.dumps(encoding_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_root / "ENCODING_REPORT.md",
        manifest,
        summaries,
        results,
        plot_paths,
    )
    print(f"Encoding manifest: {encoding_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
