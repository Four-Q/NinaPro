# RMS 活动区映射 V2

本目录保留原始V1结果之外的新映射版本。处理流程为：训练集通道均值去中心、5点（25 ms）因果RMS、训练集每通道RMS第95百分位归一化、0.20活动门控、2.05～2.35 V活动区映射。

滑动窗口的RMS在完整动作序列上连续计算后再切片，因此窗口开头保留前序采样历史。PWL采样率仍为200 Hz，完整序列持续3.935 s，每个窗口持续0.200 s。

- `full_sequence_pwl/`：完整动作的16通道PWL；
- `windows_sequence_pwl/`：5个随机窗口、每窗口16通道；
- `manifest.json`：映射参数、训练统计量、活动比例和文件清单。

生成命令：

```powershell
python ..\generate_rms_active_band_pwl.py
```
