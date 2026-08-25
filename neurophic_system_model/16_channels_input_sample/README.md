# NinaPro 16 通道 LTspice 输入样本

本目录保存从 NinaPro DB5 Exercise A 数据生成的16通道PWL激励及映射验证。

- `q99_linear/`：原始Q99线性映射V1的PWL、清单和说明。
- `rms_active_band/`：RMS活动区映射V2的PWL和清单。
- `v3_p1p99_symmetric_threshold_validation/`：V3逐通道p1/p99对称分段映射的PWL、96项LTspice仿真图和验证报告。
- `v3_all_actions_2windows_encoding/`：S1全部12个动作、每动作2个不同repetition窗口的384通道仿真与SNN脉冲编码数据。
- `generate_pwl.py`：生成 `q99_linear/` 中的V1数据。
- `generate_rms_active_band_pwl.py`：生成 `rms_active_band/` 中的V2数据。
- `generate_v3_p1p99_symmetric_pwl.py`：生成V3验证输入。
- `generate_v3_all_actions_2windows.py`：生成全动作双窗口V3输入。
- `simulate_v3_all_actions_encoding.py`：运行384项窗口仿真并导出脉冲时间戳、SNN张量和动作汇总图。
- `simulate_and_plot.py`：调用LTspice批量仿真指定版本；默认使用V1并输出到 `../pictures/q99_linear/`。

每个 PWL 文件只对应一个通道，包含两列：时间（秒）和 Vin（伏特）。NinaPro DB5 的采样率为 200 Hz，因此每个原始采样持续 5 ms。PWL 使用 1 µs 过渡近似零阶保持，并将最后一个采样保持到 `N/200` 秒：完整序列的波形长度与实际采样持续时间相等，每个 40 点滑动窗口恰好为 0.200 s。

V1电压映射为：

```text
Vin = clip(2.30 * clip(abs(x - μc) / Q99,c, 0, 1), 0, 2.35)
```

`μc` 和 `Q99,c` 使用全部训练完整序列按通道计算，不使用测试集。详细数值见 `q99_linear/manifest.json`。V2公式和参数见 `rms_active_band/manifest.json`。

在本目录运行以下命令可以重新生成默认样本：

```powershell
python generate_pwl.py
```

运行全部 LTspice 仿真并绘图：

```powershell
python simulate_and_plot.py
```

仿真V2时显式指定版本目录：

```powershell
python simulate_and_plot.py `
  --input-root rms_active_band `
  --pictures-root ..\pictures\rms_active_band
```
