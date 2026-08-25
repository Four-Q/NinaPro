# Q99 线性映射 V1

本目录保存原始映射V1的PWL输入和生成清单，与 `../rms_active_band/` 平级保留，用作映射方案对照。

V1使用训练集每通道均值和中心化绝对值第99百分位：

```text
a = clip(abs(x - μc) / Q99,c, 0, 1)
Vin = clip(2.30 * a, 0, 2.35)
```

- `full_sequence_pwl/`：S1、动作0（食指屈曲）、重复1的16通道完整序列；
- `windows_sequence_pwl/`：从同一动作随机抽取的5个窗口，每窗口16通道；
- `manifest.json`：训练统计量、样本来源、时间规则和文件清单；
- `ltspice_work/`：V1批处理网表准备阶段的工作副本。

对应图片位于 `../../pictures/q99_linear/`。重新生成V1可在上一级目录运行：

```powershell
python generate_pwl.py
```
