# Q99 线性映射 V1 仿真结果

本目录保存原始Q99线性映射V1的96张LTspice波形图，与 `../rms_active_band/` 平级保留。

- `full_sequence_pwl/`：16张完整序列图片；
- `windows_sequence_pwl/`：5个随机窗口、共80张图片；
- `simulation_manifest.json`：逐次仿真的输入、图片、节点范围和完成状态。

V1的96个 `final_out` 均未达到1.60 V振荡判定阈值，因此该版本仅作为失败基线和V2对照，不用于后续脉冲数据生成。
