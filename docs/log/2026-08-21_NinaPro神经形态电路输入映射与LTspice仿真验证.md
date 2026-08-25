# 2026-08-21 NinaPro 神经形态电路输入映射与 LTspice 仿真验证日志

## 1. 工作概览

本轮工作围绕 `neurophic_system_model/System_with_TIA.asc` 展开，目标是把 NinaPro DB5 的16通道sEMG数据转换为适合该LTspice神经形态电路的Vin输入，并验证电路输出是否能进入有效振荡状态。

本轮完成内容：

- 解析 `System_with_TIA.asc`、NbOx振荡器和突触子电路；
- 计算第一级NbOx、突触和第二级NbOx的关键输入门限；
- 统计NinaPro DB5训练数据各通道幅度范围；
- 设计并实现Q99线性映射V1；
- 从S1的一条完整非静息动作中生成16通道完整序列PWL；
- 从同一动作随机抽取5个滑动窗口并生成80个窗口PWL；
- 保证PWL持续时间与200 Hz NinaPro实际采样时间严格一致；
- 实现LTspice命令行批处理、RAW解析和三路波形绘图；
- 完成V1共96次LTspice仿真，确认V1无法驱动 `final_out` 振荡；
- 分析V1失败原因，设计5点因果RMS活动区映射V2；
- 使用临时LTspice诊断验证V2参数；
- 生成V2共96个PWL并完成96次正式LTspice仿真；
- 验证V2能使活跃通道起振，同时让低活动通道保持静默；
- 将V1和V2的输入、图片及清单整理为平级版本目录。

本轮没有修改 `System_with_TIA.asc` 及其器件参数。V1与V2之间的差异仅来自输入预处理和映射方式。

## 2. 数据与样本约定

使用数据：

```text
ninapro_data/processed/exerciseA/
├── full_sequence/
│   ├── train.npz
│   ├── test.npz
│   └── metadata.json
└── slide_window/
    ├── train.npz
    ├── test.npz
    └── metadata.json
```

NinaPro DB5当前约定：

- 16通道双Myo sEMG；
- 采样率200 Hz；
- 原始值为Myo任意数字单位，不是直接的物理电压；
- 原始范围约为−128～127；
- 滑动窗口长度200 ms，即40个采样点；
- 窗口步长100 ms，即20个采样点；
- 训练重复为1、3、4、6；
- 测试重复为2、5；
- 当前处理数据不包含静息类。

本轮展示样本固定为：

| 字段 | 值 |
| --- | --- |
| 受试者 | S1 |
| 数据划分 | train |
| 标签 | 0 |
| 原始动作标签 | 1 |
| 动作 | 食指屈曲 |
| repetition | 1 |
| 完整序列长度 | 787点 |
| 完整序列持续时间 | 3.935 s |

从该完整动作中使用固定随机种子42、无放回抽取5个滑动窗口。窗口原始起点为：

```text
1063, 1323, 1463, 1543, 1743
```

每个窗口保持40点和0.200 s，不改变NinaPro的真实采样时间尺度。

## 3. 神经形态电路拓扑

当前电路的主要信号链为：

```text
Vin
  ↓
第一级 NbOx 松弛振荡器 U1
  ↓ Vout
SYNAPSE_ADV_V2 突触模型
  ↓ Isyn
TIA 跨阻放大器
  ↓ VTIA
电平平移运放
  ↓ VDRIVE
第二级 NbOx 松弛振荡器 U2
  ↓
final_out
```

主要文件：

```text
neurophic_system_model/
├── System_with_TIA.asc
├── NbOx_OSC_stable.lib
├── NbOx_OSC.asy
├── synapse_advanced_v2.sub
└── SYNAPSE_ADV_V2.asy
```

## 4. Vin工作区间计算

第一级NbOx模型参数：

```text
VH    = 1.676 V
VL    = 1.127 V
Rin   = 89.21344 kΩ
Rme   = 806 Ω
Rload = 10 kΩ
Cparal = 1 µF
```

高阻态分压系数：

\[
\alpha_H=
\frac{R_{in}}{R_{load}+R_{in}}
=0.899207
\]

低阻态分压系数：

\[
\alpha_L=
\frac{R_{me}}{R_{load}+R_{me}}
=0.074588
\]

突触门限为：

\[
V_{TH}=0.679973\ \mathrm{V}
\]

因此第一级未起振时，突触开始响应所需Vin约为：

\[
V_{in,syn}
=\frac{V_{TH}}{\alpha_H}
\approx0.7562\ \mathrm{V}
\]

**第一级NbOx起振门限为：**
\[
V_{in,on}
=\frac{V_H}{\alpha_H}
\approx1.8639\ \mathrm{V}
\]

数学模型给出的振荡上限约为：

\[
V_{in,stop}
=\frac{V_L}{\alpha_L}
\approx15.1096\ \mathrm{V}
\]

15.11 V只是模型方程上限，不作为当前工程输入范围。当前实验仍将Vin硬限制在2.35 V以内。

后级满足：

\[
V_{TIA}=-I_{syn}R_{f1}
\]

\[
V_{drive}=2V_{REF}-V_{TIA}
=1.5+2500I_{syn}
\]

第二级同样需要约1.8639 V才能起振，因此所需突触电流约为：

\[
I_{syn,on}
=\frac{1.8639-1.5}{2500}
\approx145.5\ \mu\mathrm{A}
\]

**第二级未起振时，`final_out` 的静态基线约为：**
\[
0.899207\times1.5\approx1.349\ \mathrm{V}
\]

因此不能用 `final_out > 0` 判断脉冲。本轮统一使用 `final_out` 上穿1.60 V作为一次振荡事件。

## 5. Q99线性映射V1

### 5.1 映射方法

V1先使用全部训练完整序列计算每通道均值：

\[
x'_c(t)=x_c(t)-\mu_c
\]

再使用中心化绝对值第99百分位：

\[
a_c(t)=
\operatorname{clip}
\left(
\frac{|x_c(t)-\mu_c|}{Q_{99,c}},
0,1
\right)
\]

最后映射为：

\[
V_{in,c}(t)=
\operatorname{clip}(2.30a_c(t),0,2.35)
\]

采用每通道统计量是因为当前训练数据各通道标准差约为4.80～24.45，相差超过5倍。如果统一除以128，弱通道将无法充分利用输入范围。

### 5.2 PWL时间格式

所有PWL均使用两列格式：

```text
时间（秒） Vin（伏特）
```

NinaPro采样间隔：

\[
\Delta t=1/200=5\ \mathrm{ms}
\]

采用1 µs过渡近似零阶保持。对于N个采样，最后一个值保持到：

\[
t_{end}=N/200
\]

因此：

- 完整序列787点严格持续3.935 s；
- 窗口40点严格持续0.200 s；
- 不使用错误的 \((N-1)/200\) 作为总时长。

### 5.3 V1生成结果

```text
neurophic_system_model/16_channels_input_sample/q99_linear/
├── full_sequence_pwl/       # 16个PWL
├── windows_sequence_pwl/    # 5×16=80个PWL
├── manifest.json
├── README.md
└── ltspice_work/
```

V1共生成96个PWL，全部通过以下检查：

- 文件数正确；
- 时间轴严格递增；
- 完整序列终点为3.935 s；
- 窗口终点为0.200 s；
- Vin位于0～2.30 V，未超过2.35 V保护上限。

## 6. LTspice批处理与可视化

本机使用：

```text
LTspice 26.0.2 for Windows
D:\Software\LTspice\LTspice.exe
```

实现脚本：

```text
neurophic_system_model/16_channels_input_sample/simulate_and_plot.py
```

批处理网表保持 `System_with_TIA.asc` 的器件参数和节点连接不变，仅在每次仿真中替换：

- Vin对应的PWL文件；
- `.tran` 仿真终止时间；
- `.save` 保存节点。

每次仿真仅保存：

```text
V(N001)      → Vin
V(Vout)      → 第一级输出
V(final_out) → 最终输出
```

LTspice RAW完成解析和绘图后立即删除，避免96次仿真产生大量临时数据。每个PWL对应一张三联PNG图，从上到下依次为Vin、Vout和Vfinal_out。

## 7. V1失败分析

V1完成96次正式LTspice仿真后发现：

```text
final_out最大值：1.467 V
达到1.60 V的仿真：0/96
达到1.676 V的仿真：0/96
```

虽然54个样本曾有Vin超过第一级相关门限，但所有样本中超过1.8639 V的平均比例只有约0.63%，最长连续时间只有10 ms。

V1失败的主要原因：

1. Q99基于瞬时绝对值，只有极少量尖峰能映射到接近2.30 V；
2. NinaPro每5 ms更新一次输入，超过门限的高值通常不连续；
3. 第一级在Vin=2.30 V时，从低阈值充电到高阈值约需7.9 ms；
4. 孤立的5 ms高值难以稳定完成第一级振荡；
5. 突触模型还需要时间积累状态和电流；
6. 突触电流最终未达到约145.5 µA的第二级起振要求。

因此，V1不是简单的“电压最大值不够”，而是输入幅度、持续时间和电路动态不匹配。

## 8. RMS活动区映射V2

### 8.1 因果RMS包络

V2使用5点因果RMS，对应25 ms：

\[
e_c[n]=
\sqrt{
\frac{1}{L_n}
\sum_{k=\max(0,n-4)}^n
(x_c[k]-\mu_c)^2
}
\]

每条完整动作独立计算RMS，不跨动作边界。动作开头不足5点时只使用当前已有历史。

RMS必须先在完整动作上连续计算，再切出滑动窗口。这样窗口开头保留前序采样历史，不会因为窗口切分重新丢失前4个采样。

### 8.2 RMS归一化

使用全部训练完整序列的每通道RMS第95百分位：

\[
a_c[n]=
\operatorname{clip}
\left(
\frac{e_c[n]}{Q^{RMS}_{95,c}},
0,1
\right)
\]

全部统计量只由训练集计算，测试集和展示样本不参与参数选择。

### 8.3 活动区映射

V2采用静默区和活动区分段映射：

\[
V_{in,c}[n]=
\begin{cases}
0, & a_c[n]<0.20\\[4pt]
2.05+
0.30\dfrac{a_c[n]-0.20}{0.80},
& a_c[n]\ge0.20
\end{cases}
\]

最终硬限制：

\[
0\le V_{in}\le2.35\ \mathrm{V}
\]

设计意图：

- 低幅噪声映射到0 V，不形成持续基线振荡；
- 有效肌电从2.05 V开始，显著高于1.8639 V门限；
- 25 ms包络持续时间长于第一级约7～13 ms的主要充电时间；
- 活动区2.05～2.35 V仍可通过频率和状态变化保留一定幅度信息；
- 不需要提高Vin到3 V，也不需要修改VREF、Rf或NbOx参数。

### 8.4 参数诊断

正式批量生成前，使用临时文件对代表性通道进行LTspice诊断。结果如下：

| 通道 | 活动占比 | `final_out` 最大值 | 1.60 V上升沿 |
| ---: | ---: | ---: | ---: |
| 2 | 76.0% | 1.676 V | 559 |
| 3 | 69.9% | 1.676 V | 455 |
| 6 | 0.64% | 1.444 V | 0 |
| 10 | 76.7% | 1.676 V | 467 |

该结果说明V2可以使活跃通道起振，同时不会强迫低活动通道振荡。

另测试2.30～3.00 V活动区，没有获得明显收益，因此最终保持2.05～2.35 V。

## 9. V2正式仿真结果

V2目录：

```text
neurophic_system_model/16_channels_input_sample/rms_active_band/
├── full_sequence_pwl/       # 16个PWL
├── windows_sequence_pwl/    # 80个PWL
├── manifest.json
└── README.md
```

V2的96个PWL满足：

```text
Vin最小值：0 V
非零Vin最小值：约2.050007 V
Vin最大值：2.35 V
完整序列终点：3.935 s
窗口终点：0.200 s
```

全部96次LTspice仿真成功，无图片或清单缺失。

### 9.1 完整序列振荡统计

16个完整序列通道中有13个产生 `final_out` 振荡，共检测到2220次1.60 V上升沿。

| 通道 | `final_out` 上升沿次数 |
| ---: | ---: |
| 1 | 112 |
| 2 | 559 |
| 3 | 455 |
| 4 | 263 |
| 5 | 0 |
| 6 | 0 |
| 7 | 7 |
| 8 | 180 |
| 9 | 19 |
| 10 | 467 |
| 11 | 100 |
| 12 | 20 |
| 13 | 0 |
| 14 | 15 |
| 15 | 18 |
| 16 | 5 |

通道5、6、13保持静默或只产生亚阈值响应。这是合理结果，因为一个手势不应强迫16个电极通道全部起振。

### 9.2 窗口振荡统计

80个通道窗口中有36个产生振荡，共检测到229次上升沿。

| 窗口 | 原始起点 | 起振通道数 | 上升沿总数 |
| ---: | ---: | ---: | ---: |
| 1 | 1063 | 5 | 16 |
| 2 | 1323 | 12 | 78 |
| 3 | 1463 | 7 | 62 |
| 4 | 1543 | 7 | 43 |
| 5 | 1743 | 5 | 30 |

V2证明当前电路在不修改器件参数的情况下可以把经过适当时间包络和活动区标定的NinaPro信号转换为最终振荡输出。

## 10. 最终目录结构

V1和V2已整理为平级版本：

```text
neurophic_system_model/
├── 16_channels_input_sample/
│   ├── q99_linear/                 # V1输入、清单和说明
│   │   ├── full_sequence_pwl/
│   │   ├── windows_sequence_pwl/
│   │   ├── manifest.json
│   │   └── README.md
│   ├── rms_active_band/            # V2输入、清单和说明
│   │   ├── full_sequence_pwl/
│   │   ├── windows_sequence_pwl/
│   │   ├── manifest.json
│   │   └── README.md
│   ├── generate_pwl.py
│   ├── generate_rms_active_band_pwl.py
│   ├── simulate_and_plot.py
│   └── README.md
└── pictures/
    ├── q99_linear/                 # V1的96张图片和清单
    │   ├── full_sequence_pwl/
    │   ├── windows_sequence_pwl/
    │   ├── simulation_manifest.json
    │   └── README.md
    ├── rms_active_band/            # V2的96张图片和清单
    │   ├── full_sequence_pwl/
    │   ├── windows_sequence_pwl/
    │   ├── simulation_manifest.json
    │   └── README.md
    └── README.md
```

版本标识：

```text
V1：q99_linear_v1
V2：rms_active_band_v2
```

两版各包含：

```text
96个PWL
96张PNG
1份输入manifest
1份仿真manifest
```

## 11. 新增与修改的脚本

### 11.1 `generate_pwl.py`

作用：

- 计算训练集每通道均值和Q99；
- 生成V1完整序列和窗口PWL；
- 输出到 `q99_linear/`；
- 保存样本来源和映射统计量。

### 11.2 `generate_rms_active_band_pwl.py`

作用：

- 在训练完整序列上计算5点因果RMS；
- 计算每通道RMS Q95；
- 执行0.20门控和2.05～2.35 V活动区映射；
- 先处理完整动作，再切窗口；
- 输出到 `rms_active_band/`。

### 11.3 `simulate_and_plot.py`

作用：

- 接收指定输入版本目录；
- 调用LTspice运行每个PWL；
- 解析二进制RAW；
- 绘制Vin、Vout和Vfinal_out；
- 统计Vout及final_out的1.60 V上升沿；
- 保存逐次仿真清单；
- 默认仿真V1，也可通过参数仿真V2。

## 12. 当前结论

1. NinaPro DB5的数字值不能直接当作电压输入LTspice电路；
2. 当前电路的有效输入不仅受幅度门限约束，也受5 ms数据更新速度和器件时间常数约束；
3. 瞬时绝对值Q99线性映射虽然数值范围安全，但持续时间不足，V1无法驱动 `final_out` 振荡；
4. 5点因果RMS能够把双极性sEMG转换成与电路时间尺度更匹配的幅度包络；
5. 0 V静默区与2.05～2.35 V活动区能够同时实现噪声门控和可靠起振；
6. V2让13个活跃完整通道起振，同时保留3个低活动通道的静默状态；
7. V1应保留为失败基线，V2作为当前后续电路输出和SNN脉冲编码的候选方案；
8. 后续不能只比较“原始数据直接训练”和“V2电路输出训练”，还应加入“相同RMS预处理但绕过电路”的消融组，以分离预处理贡献和电路贡献。

## 13. 后续建议

下一步建议按以下顺序推进：

1. 从 `final_out` 上穿1.60 V的位置提取事件脉冲；
2. 将LTspice高分辨率事件归并到与SNN输入一致的时间步；
3. 对完整连续动作先运行电路，再按原始窗口边界切分输出，避免反复重置器件状态；
4. 建立三组严格对照：原始数据、RMS旁路数据、RMS＋电路数据；
5. 所有映射统计量继续只使用训练集；
6. 使用相同数据划分、SNN结构、随机种子和训练预算比较分类性能；
7. 报告准确率之外的脉冲稀疏度、每通道事件率、延迟和对噪声/通道失配的鲁棒性；
8. 在扩大到全部受试者前，对不同动作和重复次数检查V2是否出现过度振荡或过度静默。
