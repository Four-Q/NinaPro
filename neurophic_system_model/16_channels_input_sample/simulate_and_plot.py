"""批量运行 LTspice PWL 样本并绘制 Vin、Vout 和 final_out。"""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = SAMPLE_ROOT.parent
PROJECT_ROOT = MODEL_ROOT.parent
INPUT_ROOT = SAMPLE_ROOT / "q99_linear"
PICTURES_ROOT = MODEL_ROOT / "pictures" / "q99_linear"
DEFAULT_LTSPICE = Path(r"D:\Software\LTspice\LTspice.exe")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltspice", type=Path, default=DEFAULT_LTSPICE)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--pictures-root", type=Path, default=PICTURES_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def build_tasks(manifest, input_root, pictures_root):
    tasks = []
    vin_reference_voltage = manifest.get("mapping", {}).get("zero_voltage")
    full = manifest["full_sequence"]
    for relative_file in full["files"]:
        tasks.append(
            {
                "group": "full_sequence_pwl",
                "pwl": input_root / relative_file,
                "duration": float(full["duration_seconds"]),
                "vin_reference_voltage": vin_reference_voltage,
                "output": pictures_root
                / "full_sequence_pwl"
                / f"{Path(relative_file).stem}.png",
            }
        )

    for window in manifest["random_windows"]["windows"]:
        for relative_file in window["files"]:
            relative_path = Path(relative_file)
            tasks.append(
                {
                    "group": "windows_sequence_pwl",
                    "window_order": int(window["order"]),
                    "window_start_sample": int(window["source_start_sample"]),
                    "pwl": input_root / relative_path,
                    "duration": float(window["duration_seconds"]),
                    "vin_reference_voltage": vin_reference_voltage,
                    "output": pictures_root
                    / "windows_sequence_pwl"
                    / relative_path.parent.name
                    / f"{relative_path.stem}.png",
                }
            )
    return tasks


def ltspice_path(path):
    # LTspice 网表接受 Windows 绝对路径；双引号可避免未来目录名含空格时出错。
    return str(path.resolve()).replace("/", "\\")


def make_netlist(pwl_path, duration):
    nbox_model = ltspice_path(MODEL_ROOT / "NbOx_OSC_stable.lib")
    synapse_model = ltspice_path(MODEL_ROOT / "synapse_advanced_v2.sub")
    pwl_file = ltspice_path(pwl_path)
    return f"""* Batch netlist derived from System_with_TIA.asc
XU1 N001 Vout 0 NbOx_OSC VH=1.676 VL=1.127 Rin=89213.44 Rme=806 Rload=10k Cparal=1u
Vin N001 0 PWL file=\"{pwl_file}\"
XSynapse Vout N002 SYN_S SYNAPSE_ADV_V2
VD N002 0 1v
XU2 VDRIVE final_out 0 NbOx_OSC VH=1.676 VL=1.127 Rin=89213.44 Rme=806 Rload=10k Cparal=1u
XU_TIA 0 SYN_S VCC VEE VTIA level2 Avol=1Meg GBW=10Meg Slew=10Meg Ilimit=25m Rail=0 Vos=0 En=0 Enk=0 In=0 Ink=0 Rin=500Meg
VCC VCC 0 5V
VEE VEE 0 -5V
Rf1 VTIA SYN_S 2.5k
Cf1 VTIA SYN_S 10n
V_REF VREF 0 0.75V
XU_SHIFT VREF SHIFT_N VCC VEE VDRIVE level2 Avol=1Meg GBW=10Meg Slew=10Meg Ilimit=25m Rail=0 Vos=0 En=0 Enk=0 In=0 Ink=0 Rin=500Meg
Rf2 VDRIVE SHIFT_N 10k
Rin2 SHIFT_N VTIA 10k
.include \"{nbox_model}\"
.lib \"{synapse_model}\"
.lib UniversalOpAmp2.lib
.tran 0 {duration:.9f} 0 20u uic
.options plotwinsize=0
.save V(N001) V(Vout) V(final_out)
.end
"""


def run_ltspice(executable, netlist_path, timeout):
    command = [str(executable), "-b", str(netlist_path)]
    completed = subprocess.run(
        command,
        cwd=netlist_path.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    raw_path = netlist_path.with_suffix(".raw")
    log_path = netlist_path.with_suffix(".log")
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if completed.returncode != 0 or not raw_path.exists():
        details = "\n".join(
            part
            for part in [completed.stdout, completed.stderr, log_text]
            if part.strip()
        )
        raise RuntimeError(
            f"LTspice 仿真失败，退出码 {completed.returncode}：\n{details}"
        )
    return raw_path, log_text


def read_ltspice_raw(path):
    content = path.read_bytes()
    marker = "Binary:\n".encode("utf-16le")
    marker_index = content.find(marker)
    if marker_index < 0:
        raise ValueError(f"{path} 不是受支持的 LTspice 二进制 RAW 文件")

    data_offset = marker_index + len(marker)
    header = content[:data_offset].decode("utf-16le")
    point_match = re.search(r"No\. Points:\s+(\d+)", header)
    variable_match = re.search(r"No\. Variables:\s+(\d+)", header)
    if point_match is None or variable_match is None:
        raise ValueError(f"{path} 的 RAW 头缺少点数或变量数")

    point_count = int(point_match.group(1))
    variable_count = int(variable_match.group(1))
    names = [
        match.group(1)
        for match in re.finditer(r"^\s*\d+\s+([^\t]+)\t", header, re.MULTILINE)
    ]
    if len(names) != variable_count or names[0].lower() != "time":
        raise ValueError(f"{path} 的变量表结构异常：{names}")

    record_dtype = np.dtype(
        [("time", "<f8"), ("values", "<f4", (variable_count - 1,))]
    )
    expected_size = data_offset + point_count * record_dtype.itemsize
    if expected_size != len(content):
        raise ValueError(
            f"{path} 的二进制长度不匹配：expected={expected_size}, actual={len(content)}"
        )

    records = np.frombuffer(
        content,
        dtype=record_dtype,
        count=point_count,
        offset=data_offset,
    )
    signals = {"time": records["time"].astype(np.float64, copy=False)}
    for index, name in enumerate(names[1:]):
        signals[name.lower()] = records["values"][:, index].astype(
            np.float64,
            copy=False,
        )
    return signals


def select_plot_points(time, signals, max_points=50000):
    if time.size <= max_points:
        return time, signals
    indices = np.linspace(0, time.size - 1, max_points, dtype=np.int64)
    return time[indices], [signal[indices] for signal in signals]


def plot_signals(task, raw_signals):
    time = raw_signals["time"]
    vin = raw_signals["v(n001)"]
    vout = raw_signals["v(vout)"]
    final_out = raw_signals["v(final_out)"]
    plot_time, plotted = select_plot_points(time, [vin, vout, final_out])

    task["output"].parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    vin_upper = max(2.40, float(np.ceil((vin.max() + 0.05) * 10.0) / 10.0))
    series = [
        (plotted[0], "Vin", "#2563eb", (-0.05, vin_upper)),
        (plotted[1], "Vout", "#f97316", (-0.05, 1.80)),
        (plotted[2], "Vfinal_out", "#dc2626", (-0.05, 1.80)),
    ]
    for axis, (values, label, color, limits) in zip(axes, series):
        axis.plot(plot_time, values, color=color, linewidth=0.8)
        axis.set_ylabel(f"{label} (V)")
        axis.set_ylim(*limits)
        axis.grid(True, color="#d1d5db", linewidth=0.5, alpha=0.8)

    if task.get("vin_reference_voltage") is not None:
        axes[0].axhline(
            float(task["vin_reference_voltage"]),
            color="#111827",
            linewidth=0.8,
            linestyle="--",
            alpha=0.8,
            label="raw zero / oscillation boundary",
        )
        axes[0].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(0.0, task["duration"])
    figure.suptitle(task["pwl"].stem)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(task["output"], dpi=160, bbox_inches="tight")
    plt.close(figure)

    return {
        "raw_point_count": int(time.size),
        "time_start_seconds": float(time[0]),
        "time_end_seconds": float(time[-1]),
        "vin_range": [float(vin.min()), float(vin.max())],
        "vout_range": [float(vout.min()), float(vout.max())],
        "final_out_range": [float(final_out.min()), float(final_out.max())],
        "vout_rising_crossings_1p60": int(
            np.sum((vout[:-1] < 1.60) & (vout[1:] >= 1.60))
        ),
        "final_out_rising_crossings_1p60": int(
            np.sum((final_out[:-1] < 1.60) & (final_out[1:] >= 1.60))
        ),
        "vout_crossing_rate_hz": float(
            np.sum((vout[:-1] < 1.60) & (vout[1:] >= 1.60))
            / task["duration"]
        ),
        "final_out_crossing_rate_hz": float(
            np.sum((final_out[:-1] < 1.60) & (final_out[1:] >= 1.60))
            / task["duration"]
        ),
    }


def main():
    args = parse_args()
    if not args.ltspice.is_file():
        raise FileNotFoundError(f"找不到 LTspice：{args.ltspice}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit 必须为正整数")

    input_root = args.input_root.resolve()
    pictures_root = args.pictures_root.resolve()
    manifest_path = input_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到输入清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = build_tasks(manifest, input_root, pictures_root)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    work_parent = input_root / "ltspice_work"
    work_parent.mkdir(parents=True, exist_ok=True)
    results = []
    for index, task in enumerate(tasks, start=1):
        with tempfile.TemporaryDirectory(prefix="run_", dir=work_parent) as temp_dir:
            netlist_path = Path(temp_dir) / "simulation.net"
            netlist_path.write_text(
                make_netlist(task["pwl"], task["duration"]),
                encoding="ascii",
            )
            raw_path, log_text = run_ltspice(args.ltspice, netlist_path, args.timeout)
            signal_summary = plot_signals(task, read_ltspice_raw(raw_path))

        result = {
            "index": index,
            "group": task["group"],
            "pwl": str(task["pwl"].relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "picture": str(task["output"].relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "requested_duration_seconds": task["duration"],
            "ltspice_completed": "Total elapsed time:" in log_text,
            **signal_summary,
        }
        if "window_order" in task:
            result["window_order"] = task["window_order"]
            result["window_start_sample"] = task["window_start_sample"]
        results.append(result)
        print(
            f"[{index:03d}/{len(tasks):03d}] {task['pwl'].name} -> "
            f"{task['output'].relative_to(pictures_root)}",
            flush=True,
        )

    summary = {
        "version": manifest.get("version", "unspecified"),
        "source_circuit": str(
            (MODEL_ROOT / "System_with_TIA.asc").relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "ltspice_executable": str(args.ltspice),
        "ltspice_model": "System_with_TIA.asc 对应批处理网表；器件参数和节点连接保持一致",
        "signals": ["V(N001) as Vin", "V(Vout)", "V(final_out)"],
        "oscillation_count_rule": "1.60 V rising crossings",
        "input_manifest": str(manifest_path.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "task_count": len(results),
        "results": results,
    }
    pictures_root.mkdir(parents=True, exist_ok=True)
    summary_path = pictures_root / "simulation_manifest.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Simulation summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
