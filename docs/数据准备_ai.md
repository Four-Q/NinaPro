# NinaPro DB5 Exercise A 数据准备

## 1. 统一处理规则

- 数据集：NinaPro DB5 Exercise A；
- 受试者：10 名；
- 动作：12 类，每类重复 6 次，不包含静息类；
- 信号：双 Myo 的 16 通道 sEMG，采样率 200 Hz；
- 标签：使用修正后的 `restimulus` 和 `rerepetition`；
- 划分：重复 1、3、4、6 用于训练，重复 2、5 用于测试；
- 标签映射：原始动作编号 1～12 转换为分类标签 0～11；
- 数据类型：sEMG 为 `float32`，标签为 `int64`；
- 不在数据准备阶段滤波、归一化或裁剪幅值。

原始 ZIP 和 MAT 文件由两种数据准备方式共享：

```text
ninapro_data/
└── raw/
    ├── archives/
    └── exerciseA/
```

## 2. 滑动窗口数据

Notebook：

```text
code/ninapro_db5_exerciseA_slide_window.ipynb
```

输出目录：

```text
ninapro_data/processed/exerciseA/slide_window/
├── train.npz
├── test.npz
└── metadata.json
```

配置：

- 窗口长度：200 ms，即 40 个采样点；
- 步长：100 ms，即 20 个采样点；
- 相邻窗口重叠 50%；
- 单个样本形状：`[16通道, 40时间点]`。

当前输出：

- 训练集 `X`：`[17839, 16, 40]`；
- 测试集 `X`：`[8787, 16, 40]`；
- `y`：每个窗口对应一个动作标签。

该形式适合实时分类：模型每次根据最近 200 ms 的信号预测动作，每 100 ms 更新一次结果。

## 3. 完整动作序列数据

Notebook：

```text
code/ninapro_db5_exerciseA_full_sequence.ipynb
```

输出目录：

```text
ninapro_data/processed/exerciseA/full_sequence/
├── train.npz
├── test.npz
└── metadata.json
```

每名受试者的每类动作、每次重复构成一个完整序列：

```text
10 名受试者 × 12 类动作 × 6 次重复 = 720 个序列
```

其中：

- 训练集：480 个完整动作序列，每类 40 个；
- 测试集：240 个完整动作序列，每类 20 个；
- 序列长度：422～2779 个采样点，中位数 736；
- 对应持续时间约为 2.11～13.90 秒。

序列长度不同，因此使用紧凑的变长存储格式：

- `X`：所有序列连续保存，形状为 `[全部时间点, 16通道]`；
- `offsets`：每个序列在 `X` 中的起止位置；
- `lengths`：每个完整动作的实际长度；
- `y`：动作标签；
- `subject`：受试者编号；
- `repetition`：重复编号；
- `start_sample`、`end_sample`：序列在原始 MAT 文件中的位置。

第 `i` 个完整动作可按以下方式取出：

```python
sequence = X[offsets[i]:offsets[i + 1]]
```

返回形状为 `[时间点, 16通道]`。该格式不截断、不补零；训练时应在每个批次内动态补齐，并使用 `lengths` 生成掩码。

## 4. 使用建议

- 实时手势识别、低延迟 SNN：优先使用滑动窗口数据；
- LSTM、GRU、Transformer 或完整动作级 SNN：使用完整动作序列数据；
- 两种数据严格使用相同的受试者、标签和重复编号划分，可以进行公平对比。
