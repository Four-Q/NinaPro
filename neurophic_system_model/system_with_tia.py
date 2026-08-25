"""使用纯 Python 近似求解 System_with_TIA.asc 电路。"""

from dataclasses import dataclass
from math import ceil, exp, log
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _SystemModelParameters:
    """原理图及其子电路中的固定参数。"""

    maximum_step: float = 20e-6

    # 两个 NbOx 振荡器使用相同参数。
    vh: float = 1.676
    vl: float = 1.127
    rin_nbox: float = 89213.44
    rme_nbox: float = 806.0
    nbox_rload: float = 10e3
    cparal: float = 1e-6

    # SYNAPSE_ADV_V2 参数。
    synapse_vth: float = 0.679973290013
    synapse_vnorm: float = 2.0
    synapse_pdrive: float = 1.1281797291
    ron1: float = 651321.797132
    roff1: float = 58783.4118027
    ron2: float = 6453.3547737
    roff2: float = 7225.92691131
    ron3: float = 54391.4170977
    roff3: float = 22092.0764053
    g11: float = 0.000663574393119
    g12: float = 0.00122698089109
    g13: float = 4.42225017007e-05
    g23: float = 0.000671314326731
    gv: float = 6.561726644e-06
    synapse_vds: float = 1.0
    state_capacitance: float = 1e-6

    # TIA 和反相电平移位电路参数。
    rf1: float = 2.5e3
    cf1: float = 10e-9
    vref: float = 0.75
    rin2: float = 10e3
    rf2: float = 10e3
    negative_rail: float = -5.0
    positive_rail: float = 5.0

    # final_out 脉冲使用滞回武装、下降沿触发，避免振荡边沿附近重复计数。
    spike_arm_voltage: float = 1.55
    spike_fire_voltage: float = 1.25
    spike_refractory: float = 0.5e-3


_PARAMETERS = _SystemModelParameters()


def _validate_inputs(vin_time, vin_values, T):
    """校验 Vin PWL 数据并统一为 float64 数组。"""
    sample_time = np.asarray(vin_time, dtype=np.float64)
    sample_vin = np.asarray(vin_values, dtype=np.float64)

    if sample_time.ndim != 1 or sample_vin.ndim != 1:
        raise ValueError("vin_time 和 vin_values 必须是一维序列")
    if sample_time.size == 0:
        raise ValueError("Vin 时间序列不能为空")
    if sample_time.shape != sample_vin.shape:
        raise ValueError("vin_time 和 vin_values 的长度必须相同")
    if not np.all(np.isfinite(sample_time)):
        raise ValueError("vin_time 包含 NaN 或无穷大")
    if not np.all(np.isfinite(sample_vin)):
        raise ValueError("vin_values 包含 NaN 或无穷大")
    if np.any(np.diff(sample_time) <= 0.0):
        raise ValueError("vin_time 必须严格递增")
    if sample_time[-1] <= 0.0:
        raise ValueError("vin_time 的最后一个时刻必须大于 0")
    if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)):
        raise TypeError("T 必须是正整数")
    if T <= 0:
        raise ValueError("T 必须是正整数")

    return sample_time, sample_vin, int(T)


def _build_simulation_time(sample_time, duration, maximum_step):
    """生成最大步长受限且包含 PWL 拐点的仿真时间轴。"""
    step_count = max(int(ceil(duration / maximum_step)), 1)
    regular_time = np.linspace(
        0.0,
        duration,
        step_count + 1,
        dtype=np.float64,
    )
    pwl_breakpoints = sample_time[
        (sample_time > 0.0) & (sample_time < duration)
    ]
    if pwl_breakpoints.size == 0:
        return regular_time

    combined = np.sort(np.concatenate((regular_time, pwl_breakpoints)))
    # 合并浮点舍入造成的近重复时刻，避免出现没有物理意义的极小时间步。
    tolerance = max(np.spacing(duration) * 8.0, maximum_step * 1e-10)
    keep = np.r_[True, np.diff(combined) > tolerance]
    time = combined[keep]
    time[0] = 0.0
    time[-1] = duration
    return time


def _advance_nbox_oscillator(
    voltage,
    high_resistance,
    input_voltage,
    dt,
    parameters,
):
    """推进 RC-NbOx 振荡器，并解析一个时间步内的阈值切换。"""
    nbox_resistance = (
        parameters.rin_nbox if high_resistance else parameters.rme_nbox
    )
    target = (
        input_voltage
        * nbox_resistance
        / (parameters.nbox_rload + nbox_resistance)
    )
    tau = parameters.cparal / (
        1.0 / parameters.nbox_rload + 1.0 / nbox_resistance
    )
    next_voltage = target + (voltage - target) * exp(-dt / tau)

    crossed_high = high_resistance and next_voltage >= parameters.vh
    crossed_low = not high_resistance and next_voltage <= parameters.vl
    if not (crossed_high or crossed_low):
        return next_voltage, high_resistance

    # 先定位阈值交点，再用切换后的电阻求解剩余时间。
    threshold = parameters.vh if high_resistance else parameters.vl
    ratio = (threshold - target) / (voltage - target)
    ratio = min(max(ratio, np.finfo(np.float64).tiny), 1.0)
    crossing_time = min(max(-tau * log(ratio), 0.0), dt)
    remaining_time = dt - crossing_time
    high_resistance = not high_resistance

    nbox_resistance = (
        parameters.rin_nbox if high_resistance else parameters.rme_nbox
    )
    target = (
        input_voltage
        * nbox_resistance
        / (parameters.nbox_rload + nbox_resistance)
    )
    tau = parameters.cparal / (
        1.0 / parameters.nbox_rload + 1.0 / nbox_resistance
    )
    next_voltage = target + (threshold - target) * exp(
        -remaining_time / tau
    )
    return next_voltage, high_resistance


def _advance_synapse_state(
    state,
    target,
    gate_is_on,
    on_resistance,
    off_resistance,
    dt,
    parameters,
):
    """按子电路中的一阶状态方程推进单个突触状态。"""
    resistance = on_resistance if gate_is_on else off_resistance
    equilibrium = target if gate_is_on else 0.0
    decay = exp(-dt / (resistance * parameters.state_capacitance))
    return equilibrium + (state - equilibrium) * decay


def _simulate_voltage(time, vin, parameters):
    """求解各关键节点电压和突触电流。"""
    point_count = time.size
    vout = np.zeros(point_count, dtype=np.float64)
    synapse_current = np.zeros(point_count, dtype=np.float64)
    vtia = np.zeros(point_count, dtype=np.float64)
    vdrive = np.zeros(point_count, dtype=np.float64)
    final_out = np.zeros(point_count, dtype=np.float64)

    u1_high_resistance = True
    u2_high_resistance = True
    state_x1 = 0.0
    state_x2 = 0.0
    state_x3 = 0.0

    gain = parameters.rf2 / parameters.rin2
    vdrive[0] = (1.0 + gain) * parameters.vref

    for index in range(1, point_count):
        dt = time[index] - time[index - 1]
        # 时间轴包含所有 PWL 拐点，因此区间端点均值就是该段 Vin 的平均值。
        interval_vin = 0.5 * (vin[index - 1] + vin[index])
        vout[index], u1_high_resistance = _advance_nbox_oscillator(
            voltage=vout[index - 1],
            high_resistance=u1_high_resistance,
            input_voltage=interval_vin,
            dt=dt,
            parameters=parameters,
        )

        normalized_gate = max(
            (vout[index] - parameters.synapse_vth)
            / (parameters.synapse_vnorm - parameters.synapse_vth),
            0.0,
        ) ** parameters.synapse_pdrive
        gate_is_on = vout[index] > parameters.synapse_vth

        state_x1 = _advance_synapse_state(
            state_x1,
            normalized_gate,
            gate_is_on,
            parameters.ron1,
            parameters.roff1,
            dt,
            parameters,
        )
        state_x2 = _advance_synapse_state(
            state_x2,
            normalized_gate,
            gate_is_on,
            parameters.ron2,
            parameters.roff2,
            dt,
            parameters,
        )
        state_x3 = _advance_synapse_state(
            state_x3,
            normalized_gate,
            gate_is_on,
            parameters.ron3,
            parameters.roff3,
            dt,
            parameters,
        )

        conductance = max(
            parameters.g11 * state_x1 * state_x1
            + parameters.g12 * state_x1 * state_x2
            + parameters.g13 * state_x1 * state_x3
            + parameters.g23 * state_x2 * state_x3
            + parameters.gv * normalized_gate,
            0.0,
        )
        synapse_current[index] = parameters.synapse_vds * conductance

        tia_target = -parameters.rf1 * synapse_current[index]
        tia_decay = exp(-dt / (parameters.rf1 * parameters.cf1))
        vtia[index] = tia_target + (
            vtia[index - 1] - tia_target
        ) * tia_decay
        vtia[index] = np.clip(
            vtia[index],
            parameters.negative_rail,
            parameters.positive_rail,
        )

        vdrive[index] = (
            (1.0 + gain) * parameters.vref - gain * vtia[index]
        )
        vdrive[index] = np.clip(
            vdrive[index],
            parameters.negative_rail,
            parameters.positive_rail,
        )

        final_out[index], u2_high_resistance = _advance_nbox_oscillator(
            voltage=final_out[index - 1],
            high_resistance=u2_high_resistance,
            input_voltage=vdrive[index],
            dt=dt,
            parameters=parameters,
        )

    return vout, synapse_current, vtia, vdrive, final_out


def _detect_spike_times(time, voltage, parameters):
    """按滞回武装和下降沿规则检测 final_out 脉冲。"""
    armed = False
    last_spike_time = -np.inf
    spike_times = []

    for index in range(1, time.size):
        if voltage[index] >= parameters.spike_arm_voltage:
            armed = True

        falling_crossing = (
            voltage[index - 1] > parameters.spike_fire_voltage
            and voltage[index] <= parameters.spike_fire_voltage
        )
        outside_refractory = (
            time[index] - last_spike_time >= parameters.spike_refractory
        )
        if armed and falling_crossing and outside_refractory:
            spike_times.append(time[index])
            last_spike_time = time[index]
            armed = False

    return np.asarray(spike_times, dtype=np.float64)


def _bin_spike_times(spike_times, duration, T):
    """将连续脉冲时刻编码成长度为 T 的二进制数组。"""
    spikes = np.zeros(T, dtype=np.uint8)
    valid_spikes = spike_times[
        (spike_times >= 0.0) & (spike_times < duration)
    ]
    indices = np.floor(valid_spikes * T / duration).astype(np.int64)
    spikes[indices] = 1
    return spikes


def simulate_system(vin_time, vin_values, T=40):
    """仿真 Vin 驱动的完整神经形态电路。

    Parameters
    ----------
    vin_time : array-like
        Vin PWL 拐点的时间，单位为秒，必须严格递增。允许包含负时刻，
        但仿真从 0 秒开始，最后一个时刻决定仿真时长。
    vin_values : array-like
        与 ``vin_time`` 一一对应的 Vin，单位为伏。相邻点之间按 PWL
        线性插值，超出首个时刻时沿用首个电压值。
    T : int, default=40
        ``final_out`` 脉冲编码的时间分箱数量。

    Returns
    -------
    dict
        ``time``、``vin``、``vout``、``synapse_current``、``vtia``、
        ``vdrive`` 和 ``final_out`` 为完整仿真波形；``spike_times`` 使用
        1.55 V 武装、下降穿过 1.25 V 触发及 0.5 ms 不应期进行检测；
        ``spikes`` 为长度 T 的 uint8 二进制编码。

    Notes
    -----
    本实现使用阈值切换的解析 RC-NbOx 模型和理想闭环运放近似，运放
    输出限制在正负 5 V。它适合快速批量编码，不保证与 LTspice 的
    平滑开关和有限带宽运放逐点完全一致。
    """
    sample_time, sample_vin, T = _validate_inputs(
        vin_time,
        vin_values,
        T,
    )
    duration = float(sample_time[-1])
    time = _build_simulation_time(
        sample_time,
        duration,
        _PARAMETERS.maximum_step,
    )
    vin = np.interp(time, sample_time, sample_vin)

    vout, synapse_current, vtia, vdrive, final_out = _simulate_voltage(
        time,
        vin,
        _PARAMETERS,
    )
    spike_times = _detect_spike_times(
        time,
        final_out,
        _PARAMETERS,
    )
    spikes = _bin_spike_times(spike_times, duration, T)

    return {
        "time": time,
        "vin": vin,
        "vout": vout,
        "synapse_current": synapse_current,
        "vtia": vtia,
        "vdrive": vdrive,
        "final_out": final_out,
        "spike_times": spike_times,
        "spikes": spikes,
    }


def plot_system_result(image_path, result, show_image=False):
    """绘制并保存 ``simulate_system`` 的主要输入、节点及脉冲波形。"""
    import matplotlib.pyplot as plt

    required_keys = {
        "time",
        "vin",
        "vout",
        "synapse_current",
        "vtia",
        "vdrive",
        "final_out",
        "spikes",
    }
    if not isinstance(result, dict):
        raise TypeError("result 必须是 simulate_system 返回的字典")
    missing_keys = required_keys.difference(result)
    if missing_keys:
        missing_text = ", ".join(sorted(missing_keys))
        raise KeyError(f"result 缺少必要字段：{missing_text}")
    if not isinstance(show_image, (bool, np.bool_)):
        raise TypeError("show_image 必须是布尔值")

    time = np.asarray(result["time"], dtype=np.float64)
    waveform_names = [
        "vin",
        "vout",
        "synapse_current",
        "vtia",
        "vdrive",
        "final_out",
    ]
    waveforms = {
        name: np.asarray(result[name], dtype=np.float64)
        for name in waveform_names
    }
    spikes = np.asarray(result["spikes"])

    if time.ndim != 1 or time.size < 2:
        raise ValueError("time 必须是至少包含两个点的一维数组")
    if np.any(np.diff(time) <= 0.0) or not np.all(np.isfinite(time)):
        raise ValueError("time 必须有限且严格递增")
    for name, values in waveforms.items():
        if values.ndim != 1 or values.shape != time.shape:
            raise ValueError(f"{name} 必须是与 time 等长的一维数组")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} 不能包含 NaN 或无穷大")
    if spikes.ndim != 1 or spikes.size == 0:
        raise ValueError("spikes 必须是非空的一维数组")

    output_path = Path(image_path)
    if not output_path.name:
        raise ValueError("image_path 必须包含图片文件名")
    if not output_path.suffix:
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bin_edges = np.linspace(time[0], time[-1], spikes.size + 1)
    spikes_for_plot = np.r_[spikes, spikes[-1]]
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        dpi=200,
        sharex=True,
    )
    try:
        axes[0].plot(time, waveforms["vin"], label="Vin", linewidth=0.9)
        axes[0].plot(time, waveforms["vout"], label="Vout", linewidth=0.8)
        axes[0].set_ylabel("Voltage (V)")
        axes[0].legend(loc="upper right")

        axes[1].plot(time, waveforms["vtia"], label="VTIA", linewidth=0.9)
        current_axis = axes[1].twinx()
        current_axis.plot(
            time,
            waveforms["synapse_current"] * 1e3,
            color="#2ca02c",
            label="Isyn",
            linewidth=0.8,
        )
        axes[1].set_ylabel("VTIA (V)")
        current_axis.set_ylabel("Isyn (mA)", color="#2ca02c")
        axes[1].legend(loc="upper left")
        current_axis.legend(loc="upper right")

        axes[2].plot(
            time,
            waveforms["vdrive"],
            label="VDRIVE",
            linewidth=0.9,
        )
        axes[2].plot(
            time,
            waveforms["final_out"],
            label="final_out",
            linewidth=0.8,
        )
        axes[2].axhline(
            _PARAMETERS.spike_arm_voltage,
            color="#9467bd",
            linestyle="--",
            linewidth=0.8,
            label="1.55 V arm",
        )
        axes[2].axhline(
            _PARAMETERS.spike_fire_voltage,
            color="#d62728",
            linestyle="--",
            linewidth=0.8,
            label="1.25 V fire",
        )
        axes[2].set_ylabel("Voltage (V)")
        axes[2].legend(loc="upper right")

        axes[3].step(
            bin_edges,
            spikes_for_plot,
            where="post",
            color="#d62728",
            linewidth=1.2,
        )
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("Spike")
        axes[3].set_yticks([0, 1])
        axes[3].set_ylim(-0.05, 1.15)
        axes[3].set_xlim(time[0], time[-1])

        for axis in axes:
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        figure.suptitle("Pure-Python System_with_TIA simulation")
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        if show_image:
            plt.show()
    finally:
        plt.close(figure)

    return output_path


__all__ = ["simulate_system", "plot_system_result"]
