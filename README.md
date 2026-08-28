# SNN for NinaPro

基于 PyTorch 和 SpikingJelly 的 NinaPro DB5 表面肌电（sEMG）手势分类项目。项目围绕 Exercise A（E1）的 12 类手指动作，比较原始 sEMG、软件泊松编码和神经形态电路编码三种输入表示，并提供全连接 SNN、带 Dropout 的 PLIF SNN 以及卷积 SNN（CSNN）训练流程。

## 数据集与任务

- 数据集：NinaPro DB5 Exercise A（E1）
- 受试者：10 人，每人使用两只 Myo 臂环采集 16 通道 sEMG
- 采样率：200 Hz
- 分类目标：12 类动作；原始标签 `1～12` 在训练数据中映射为 `0～11`
- 数据划分：重复次数 `1、3、4、6` 用于训练，`2、5` 用于测试
- 预处理约束：移除休息段，不在数据准备 Notebook 中滤波或归一化；原始 sEMG 的默认训练数据管道只使用训练集统计量做通道归一化

12 类动作依次为：食指屈曲/伸展、中指屈曲/伸展、无名指屈曲/伸展、小指屈曲/伸展、拇指内收/外展、拇指屈曲/伸展。

## 数据流

```text
NinaPro 官方 ZIP
└── raw/archives/*.zip
    └── raw/exerciseA/*.mat
        ├── processed/exerciseA/full_sequence/       连续完整动作
        └── processed/exerciseA/slide_window/        200 ms 滑动窗口（核心数据源）
            ├── 直接输入 SNN，或在训练时在线泊松采样
            ├── raw_data_poisson_spikes/             离线软件泊松脉冲
            └── neurophic_system_spikes/             神经形态电路仿真脉冲
```

## `ninapro_data` 中的数据

> `.gitignore` 会忽略原始数据、解压数据和中间产物；仓库只提交 `neurophic_system_spikes` 下各方案的 ZIP 包。以下目录是当前本地数据准备流程产生的完整结构。

### 目录概览

| 路径 | 内容与作用 | 产生方式 |
| --- | --- | --- |
| `raw/archives/s1.zip` ～ `s10.zip` | NinaPro DB5 每名受试者的官方预处理压缩包，作为可复现的数据源缓存。 | 两个 `data_prepare_*.ipynb` 在本地文件缺失或损坏时，从 NinaPro 官方地址或 Zenodo 备用地址下载。 |
| `raw/exerciseA/S*_E1_A1.mat` | 从每个 ZIP 中提取的 Exercise A MATLAB 数据。主要使用 `emg`、校正后的 `restimulus` 和 `rerepetition` 字段。 | 由数据准备 Notebook 从对应 ZIP 解压并校验。 |
| `processed/exerciseA/full_sequence/` | 保留每次动作的完整连续 sEMG，适合序列长度统计、通道幅值分析和电路输入研究，不直接供当前固定窗口训练管道使用。 | 运行 `src/data_prepare_full_sequence(simple).ipynb`，按动作标签或重复编号的变化切分连续区段。 |
| `processed/exerciseA/slide_window/` | 当前训练和所有脉冲编码的核心源数据；每个样本是固定长度的原始 sEMG 窗口。 | 运行 `src/data_prepare_slide_window(simple).ipynb`，在每个连续动作内部使用 200 ms 窗口、100 ms 步长切片。 |
| `raw_data_poisson_spikes/{offset_128,polarity_split}/T_{40,80,120}/` | 从滑动窗口预先采样得到的二值泊松脉冲，用于离线脉冲训练和不同编码/时间步的对照实验。 | 运行 `src/poisson_spike_encoding.ipynb`，固定随机种子 42，一次生成两种编码和三个 `T`。 |
| `neurophic_system_spikes/{offset_128,polarity_split}/T_{40,80,120}/` | 将 sEMG 映射为输入电压，经 `neurophic_system_model/system_with_tia.py` 的神经形态电路模型仿真、检测放电脉冲并分箱后的数据，用于软硬件映射输入和 SNN 训练。 | 分别运行 `src/(offset_128)neurophic_system_spike_encoding.ipynb` 或 `src/(polarity_split)neurophic_system_spike_encoding.ipynb` 完整编码；对应的 `prepare_*.ipynb` 只负责安全解压并验证已有 ZIP。 |

### 预处理 sEMG

`slide_window` 是默认训练输入，文件为 `train.npz`、`test.npz` 和 `metadata.json`：

| 划分 | `X` 形状 | `X` 类型 | 样本数 |
| --- | --- | --- | ---: |
| train | `[17839, 16, 40]` | `float32` | 17839 |
| test | `[8787, 16, 40]` | `float32` | 8787 |

每个窗口覆盖 200 ms（40 个采样点），相邻窗口间隔 100 ms（20 个采样点）。NPZ 中还保存 `y`、`subject`、`repetition` 和 `start_sample`，便于追踪样本来源。

`full_sequence` 使用拼接存储以支持变长序列：第 `i` 条序列为 `X[offsets[i]:offsets[i + 1]]`，形状为 `[时间点, 16]`。训练集有 480 条序列，测试集有 240 条；`lengths`、`y`、`subject`、`repetition`、`start_sample` 和 `end_sample` 保存对应索引信息。

### 软件泊松脉冲

`raw_data_poisson_spikes` 中的每个 NPZ 只包含二值 `X`（`uint8`）和标签 `y`。两种编码均从未归一化的滑动窗口产生：

- `offset_128`：将原始 sEMG 加 128 后按固定满量程 255 线性映射到最高 100 Hz 的发放率，保持 16 通道。
- `polarity_split`：将每个 sEMG 通道拆为正、负两路，按参考幅值 64 和指数 0.57 做压缩幅值映射，最高发放率 200 Hz；输出顺序为 16 个正通道后接 16 个负通道，共 32 通道。

两者都按 `p = 1 - exp(-rate_hz × bin_seconds)` 采样。`T=40/80/120` 均表示把同一个 200 ms 窗口划分成相应数量的时间步，因此单步时长分别为 5、2.5、约 1.67 ms，而不是改变原始窗口时长。

### 神经形态系统脉冲

`neurophic_system_spikes` 的 `train.npz` 和 `test.npz` 包含：

- `X`：分箱后的二值脉冲，`uint8`，直接作为 SNN 输入；
- `spike_counts`：每个时间箱内的实际脉冲数，`uint16`，用于分析二值化前的计数信息；
- `Vin`：由 40 点 sEMG 映射得到的电路输入电压，`float32`；
- `y`、`subject`、`repetition`、`start_sample`：标签及源窗口索引。

两种映射方案为：

- `offset_128`：虽然目录使用该名称，当前文件实际采用元数据所写的 **V3 按通道 p1/p99 映射**，并非上述软件泊松编码的“原始值 + 128”。每个通道的 sEMG 分位点被映射到 `0～3.7278 V`，零值对应 `1.8639 V`，输出 16 通道。
- `polarity_split`：当前采用 **V5 空闲基线极性分离映射**。每个源通道拆成正、负两路；零幅值使用 `0.7562 V` 空闲电压，非零幅值从 `1.8639 V` 活跃基线向 `3.7278 V` 映射，输出 32 通道。

电路模型输出按下降沿和阈值检测脉冲，再将 200 ms 响应聚合为 `T=40/80/120` 个时间箱。每个 `T_*/metadata.json` 记录映射参数、阈值、字段、shape、源文件哈希和完整性标记；同级 ZIP 是可提交和分发的打包副本。以 `.` 开头的 `*-building` 目录只是可恢复的生成中间状态，不应作为训练输入。

## 环境

建议使用 Python 3.8 或更高版本。在项目环境中安装主要依赖：

```bash
pip install torch spikingjelly numpy scipy pandas requests matplotlib tqdm jupyter pytest numba
```

RTX 50 系列与 CUDA 12.x 环境可额外安装 SpikingJelly 的融合 CUDA 后端：

```bash
pip install -r requirements-fast.txt
```

未安装 CuPy 或 CuPy 不可用时，训练引擎会回退到 PyTorch eager 模式。训练引擎还会为旧版 SpikingJelly 补充新版 NumPy 已移除的 `np.int` 兼容别名。

## 使用

在项目根目录启动 Jupyter：

```bash
jupyter notebook
```

推荐按以下顺序使用：

1. 若没有预处理数据，先运行 `src/data_prepare_slide_window(simple).ipynb`；需要变长完整序列时再运行 `src/data_prepare_full_sequence(simple).ipynb`。
2. 若需要离线软件泊松脉冲，运行 `src/poisson_spike_encoding.ipynb`。
3. 选择与输入表示相对应的训练 Notebook；其中 `T` 可在参数单元中设置为 40、80 或 120。

| 训练 Notebook | 输入 |
| --- | --- |
| `src/main/raw_ninapro_snn_T.ipynb` | `processed/.../slide_window`；用零阶保持扩展到指定 `T` |
| `src/main/online_poisson_spikes_ninapro_snn_T.ipynb` | `slide_window`；每轮在线泊松采样 |
| `src/main/poisson_spike_ninapro_snn_T.ipynb` | `raw_data_poisson_spikes` 离线脉冲 |
| `src/main/neurophic_system_spike_ninapro_snn_T.ipynb` | `neurophic_system_spikes` 电路脉冲 |
| `src/main/online_poisson_spikes_csnn_T.ipynb` | `slide_window`；在线泊松采样后输入 CSNN |
| `src/main/poisson_spikes_csnn_T.ipynb` | `raw_data_poisson_spikes/polarity_split` 离线脉冲 |

原始滑动窗口数据管道的基本用法：

```python
from src.data import create_data_pipeline
from src.models import NinaProSNN

pipeline = create_data_pipeline(
    "ninapro_data/processed/exerciseA/slide_window",
    batch_size=128,
)
model = NinaProSNN(input_channels=16, num_classes=12)

x, y = next(iter(pipeline.train_loader))
logits = model(x)
```

训练输出默认保存在 `outputs/<输入或模型方案>/T_<时间步>/<实验名>/`，其中包括检查点、训练历史、指标、混淆矩阵、分类召回率和训练曲线。汇总对比见 `docs/训练结果详细对比报告.md`。

## 测试

```bash
python -m pytest -q
```

## 项目结构

```text
ninapro_data/              原始、预处理及编码后的 NinaPro 数据
src/data/                  Dataset、归一化、变换和 DataLoader
src/models/snn/            全连接 SNN 与 Dropout PLIF SNN
src/models/csnn/           卷积 SNN
src/training/              训练、评估、指标、可视化和检查点工具
src/main/                  各类输入表示对应的训练 Notebook
neurophic_system_model/    神经形态电路模型、映射脚本和仿真结果
test_notebooks/            数据管道与脉冲编码验证 Notebook
tests/                     自动化测试
docs/                      数据说明、实验报告和开发记录
outputs/                   已完成实验的指标、图表与模型检查点
deprecated/                已废弃的 Notebook 与旧实验结果
```
