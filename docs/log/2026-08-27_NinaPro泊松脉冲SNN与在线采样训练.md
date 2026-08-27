# 2026-08-27 NinaPro 泊松脉冲 SNN 与在线采样训练日志

## 1. 工作目标

本轮工作围绕 NinaPro DB5 Exercise A 的泊松脉冲分类展开，目标是建立可重复比较的卷积 SNN 与全连接 SNN 训练流程，并针对固定泊松脉冲训练效果不理想的问题实现在线重采样方案。

主要工作如下：

1. 使用 `ninapro_data/raw_data_poisson_spikes` 中的预编码数据训练 CSNN。
2. 为 `offset_128` 与 `polarity_split` 数据提供基于 `DropoutPLIFSNN` 的统一训练 Notebook。
3. 根据已有训练结果分析准确率明显低于预期的原因。
4. 将 CSNN 调整为沿完整 16 电极轴执行 `Conv1d`，不再使用二维卷积。
5. 从原始滑窗数据在线重新采样泊松脉冲，分别训练 CSNN 与 `DropoutPLIFSNN`。
6. 保持现有数据管道、训练函数和评估接口不变。

目标测试准确率最初设为 80%，但本轮已有泊松实验均未达到该目标。日志会区分正式训练结果与仅用于接口验证的 smoke test。

## 2. 数据约定

原始数据来自：

```text
ninapro_data/processed/exerciseA/slide_window/
├── train.npz
└── test.npz
```

数据规模：

| 划分 | 样本数 | 原始形状 | 数据类型 |
|---|---:|---|---|
| train | 17,839 | `(16, 40)` | `float32` |
| test | 8,787 | `(16, 40)` | `float32` |

泊松编码保持同一个 200 ms 物理窗口，通过参数 `T` 兼容 `40、80、120` 三种时间步。两种编码的数据布局为：

| 编码 | 模型输入 | 通道含义 |
|---|---|---|
| `offset_128` | `(B, 16, T)` | 原始 16 通道平移到非负范围 |
| `polarity_split` | `(B, 32, T)` | 前 16 通道为正极性，后 16 通道为负极性 |

二值泊松脉冲不执行 Z-score 或其他归一化，模型直接接收 `0/1` 浮点张量。

## 3. 泊松编码定义

在线编码与 `src/poisson_spike_encoding.ipynb` 中的离线编码采用相同公式。

目标时间步对应的原始索引采用零阶保持：

\[
i_{source}=\left\lfloor\frac{i_{target}\times40}{T}\right\rfloor
\]

`offset_128` 的泊松率为：

\[
u=\operatorname{clip}(x+128,0,255)
\]

\[
\lambda=100\frac{u}{255}\ \mathrm{Hz}
\]

`polarity_split` 先分离正负幅值：

\[
x^+=\max(x,0),\qquad x^-=\max(-x,0)
\]

再使用固定参考幅值与幂次映射：

\[
a=\left[\operatorname{clip}\left(\frac{|x|}{64},0,1\right)\right]^{0.57}
\]

\[
\lambda=200a\ \mathrm{Hz}
\]

每个二值时间箱记录泊松过程中是否至少出现一次事件：

\[
p=1-\exp\left(-\lambda\frac{0.2}{T}\right)
\]

\[
s\sim\operatorname{Bernoulli}(p)
\]

## 4. CSNN 模型

模型实现位于：

```text
src/models/csnn/spatiotemporal_csnn.py
```

当前 `NinaProCSNN` 接收现有数据管道的 `(B, C, T)` 输入，并在模型内部解释通道结构：

```text
offset_128:     (B, 16, T) -> (B, 1, 16, T)
polarity_split: (B, 32, T) -> (B, 2, 16, T)
```

每个 SNN 时间步均在完整 16 电极轴上执行 `Conv1d(kernel_size=3)`。SNN 时间维 `T` 不参与普通卷积，而是用于更新 LIF/PLIF 膜电位。每个特征块由以下结构组成：

```text
Conv1d -> BatchNorm1d -> LIF/PLIF -> time-consistent Dropout
```

模型在中间特征块中不池化电极位置，只在连续分类读出前汇聚 16 电极轴，最后对时间步 logits 求均值。不同 mini-batch 前自动重置膜电位与 Dropout 状态。

在线 CSNN 默认使用固定 `LIFNode(tau=2.0)`，即 `learnable_tau=False`，避免可学习时间常数退化到接近 1。仍保留 `learnable_tau=True` 参数，用于后续 PLIF 对照实验。

两组主要配置的参数量为：

| 配置 | 特征通道 | 参数量 |
|---|---|---:|
| 预编码 CSNN | `(64, 128, 256)` | 127,244 |
| 在线编码 CSNN | `(32, 64, 128)` | 32,908 |

注意：`src/main/poisson_spikes_csnn_T.ipynb` 的开头说明仍保留早期“Conv2d/PLIF”描述，但它当前导入的 `NinaProCSNN` 和已有检查点参数均对应上述 `Conv1d` 实现。后续应同步修正文档单元，避免误解。

## 5. 训练 Notebook

### 5.1 预编码泊松 CSNN

文件：

```text
src/main/poisson_spikes_csnn_T.ipynb
```

主要配置：

```python
T = 40
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
FEATURE_CHANNELS = (64, 128, 256)
DROPOUT_RATE = 0.2
```

该 Notebook 使用 `raw_data_poisson_spikes/polarity_split/T_{T}` 中的固定二值脉冲，数据可以驻留 GPU。当前仅支持预编码的 `polarity_split` 输入。

### 5.2 预编码泊松 DropoutPLIFSNN

文件：

```text
src/main/poisson_spike_ninapro_snn_T.ipynb
```

该 Notebook 兼容：

```text
ENCODING = offset_128 / polarity_split
T = 40 / 80 / 120
```

模型直接使用 `(B, 16, T)` 或 `(B, 32, T)`，不进行二维重塑。当前默认配置为 `polarity_split、T=40、EPOCHS=120、HIDDEN_SIZE=256、DROPOUT_RATE=0.1`。

模型参数量：

| 编码 | 输入通道 | 参数量 |
|---|---:|---:|
| `offset_128` | 16 | 73,230 |
| `polarity_split` | 32 | 77,326 |

### 5.3 在线泊松 CSNN

文件：

```text
src/main/online_poisson_spikes_csnn_T.ipynb
```

训练集在每次 `__getitem__` 时重新采样泊松脉冲，测试集使用 `TEST_ENCODING_SEED=20260827` 固定物化一次。Notebook 同时支持两种编码与三个 `T`，默认配置为：

```python
ENCODING = "polarity_split"
T = 40
EPOCHS = 100
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-3
FEATURE_CHANNELS = (32, 64, 128)
DROPOUT_RATE = 0.3
TAU = 2.0
LEARNABLE_TAU = False
GPU_RESIDENT_DATA = False
```

`GPU_RESIDENT_DATA=False` 是在线重采样的必要条件，否则训练数据会被提前固定为同一组脉冲。

### 5.4 在线泊松 DropoutPLIFSNN

新增文件：

```text
src/main/online_poisson_spikes_ninapro_snn_T.ipynb
```

Notebook 参考预编码 `DropoutPLIFSNN` 训练流程，并复用在线 CSNN 的泊松采样定义。默认配置为：

```python
ENCODING = "polarity_split"
T = 40
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 256
DROPOUT_RATE = 0.1
TAU = 2.0
GPU_RESIDENT_DATA = False
```

输出目录为：

```text
outputs/online_poisson_spikes_ninapro_snn/{ENCODING}/T_{T}/{EXPERIMENT_NAME}/
```

## 6. 在线采样数据集行为

在线 Notebook 中的 `PoissonSpikeDataset` 有两种工作模式：

1. 训练集设置 `resample=True`，每次访问原始窗口时重新计算发放概率并独立采样。
2. 测试集设置 `resample=False`，使用独立固定种子分块物化为 `uint8`，读取时转换为 `float32`。

这样既能使训练模型面对同一个原始窗口的多个随机实现，也能保证不同 epoch 的测试指标来自完全相同的测试脉冲。DataLoader worker 通过 `torch.initial_seed()` 获得独立随机流，并同步 Python 与 NumPy 随机种子。

在线采样不改变现有公共接口：

```text
NinaProWindowDataset -> PoissonSpikeDataset -> DataLoader -> fit()
```

模型、`fit()`、每 epoch 测试、检查点与可视化接口均保持不变。

## 7. 已有正式训练结果

以下指标均来自 `outputs` 中已有的 `metrics.json`，并使用测试集选择最佳 epoch：

| 实验 | 编码 | T | Test Accuracy | Macro F1 | 最佳 epoch |
|---|---|---:|---:|---:|---:|
| 预编码 Conv1d CSNN | `polarity_split` | 40 | 26.4026% | 25.5866% | 66 |
| 预编码 DropoutPLIFSNN | `offset_128` | 40 | 10.6521% | 4.6263% | 22 |
| 预编码 DropoutPLIFSNN | `polarity_split` | 40 | 30.7386% | 30.8176% | 12 |
| 在线 Conv1d CSNN | `polarity_split` | 40 | 24.3314% | 22.4880% | 83 |
| 原始 sEMG DropoutPLIFSNN 对照 | 未泊松编码 | 40 | 68.8290% | 68.9793% | 70 |

结论：

- 当前最佳泊松结果为预编码 `polarity_split + DropoutPLIFSNN` 的 30.7386%，明显低于原始 sEMG 对照的 68.8290%，也没有达到 80% 目标。
- `polarity_split` 明显优于 `offset_128`，说明保留正负极性并消除零值基线发放更适合当前任务。
- 在线重采样 CSNN 的一次完整训练没有超过固定预编码 CSNN，说明“避免记忆固定泊松噪声”本身不足以解决分类信息损失。
- 新增的在线 `DropoutPLIFSNN` Notebook 尚未执行完整 100 epoch，因此没有正式准确率可记录。

所有结果都是 **test-selected checkpoint**：每个 epoch 在测试集上评估并以测试准确率选择 `best.pt`。因此这些指标适合当前项目内对比，但不能解释为使用独立最终测试集得到的无偏泛化结果。

## 8. 效果不理想的原因分析

结合结果与编码定义，当前主要问题可能包括：

1. **泊松采样造成明显信息损失。** 原始 sEMG 对照约为 68.83%，而同一数据经泊松二值化后降到 10%～31%，说明主要瓶颈不仅是网络容量，而是幅值和短时结构在随机二值化中丢失。
2. **`offset_128` 存在较高背景发放。** 原始 `x=0` 会映射到约 50.2 Hz，使静态基线也产生大量脉冲，类别相关的小幅变化容易被背景噪声淹没。
3. **`T=40` 下单次随机实现方差较大。** 一个 200 ms 窗口只有 40 个二值时间箱，模型看到的瞬时脉冲图与期望发放率之间仍可能存在较大偏差。
4. **在线重采样增加训练难度。** 它能减少对固定噪声的记忆，但也要求模型从不断变化的输入中估计稳定统计量；若学习率、正则化或网络读出不匹配，准确率可能先下降。
5. **空间卷积的归纳偏置有限。** 16 个电极并不是普通图像轴。`Conv1d` 比跨极性与时间的二维卷积更符合数据结构，但局部平移共享仍不一定完全符合双臂环电极与动作之间的关系。
6. **当前超参数沿用原始数据或首轮 CSNN 设置。** 泊松脉冲密度、时间常数、阈值、读出方式、学习率与 Dropout 尚未进行系统联合搜索。

以上分析是基于现有实验的诊断假设，尚不能替代编码率、发放统计和消融实验。

## 9. 验证记录

本轮针对当前代码与新增在线 `DropoutPLIFSNN` Notebook 完成了以下检查：

1. `src/models/csnn` 专项测试：`23 passed in 2.58s`。
2. 新 Notebook 是合法 JSON，共 19 个单元、10 个代码单元，所有代码单元均可编译。
3. 原始 train/test 数据形状分别验证为 `(17839, 16, 40)` 和 `(8787, 16, 40)`。
4. `offset_128` 与 `polarity_split` 在 `T=40/80/120` 的六种组合全部通过形状、有限值与二值性检查。
5. 六种组合均完成 `DropoutPLIFSNN` 前向传播、交叉熵计算和反向传播，梯度为有限值。
6. 训练样本重复访问会产生不同脉冲，固定测试样本重复访问完全一致。
7. PyTorch 在线概率与编码文档中的 NumPy 公式对照，六种组合最大绝对误差为 `5.96e-08`。
8. 使用极小数据集调用现有 `fit()` 完成 1 epoch smoke test，证明训练接口无需修改。

smoke test 仅用于验证运行链路，不属于正式准确率实验。

## 10. 后续建议

建议按以下顺序继续：

1. 首先完整训练 `online_poisson_spikes_ninapro_snn_T.ipynb` 的默认 `polarity_split、T=40`，与已有 30.7386% 的固定预编码 `DropoutPLIFSNN` 直接比较。
2. 在相同模型与种子下比较 `T=40、80、120`，判断增加时间箱是否能降低采样方差并改善分类。
3. 统计每类、每通道和每窗口的期望/实测脉冲数，检查当前 200 Hz 与 0.57 次幂映射是否造成过稀、饱和或类别间密度偏差。
4. 对在线训练单独搜索学习率、Dropout、权重衰减和 `tau`，不要直接假设离线预编码的最优参数仍适用。
5. 增加期望率输入、重复采样投票或多次泊松实现平均等对照，量化性能损失究竟来自随机方差还是二值率编码本身。
6. 若需要可解释的最终泛化指标，应增加验证集，并仅在训练完成后对测试集评估一次。
7. 修正 `poisson_spikes_csnn_T.ipynb` 中遗留的 Conv2d/PLIF 文字说明，使其与当前 Conv1d/LIF 实现一致。

## 11. 本轮涉及的主要文件

```text
src/models/csnn/__init__.py
src/models/csnn/spatiotemporal_csnn.py
src/models/__init__.py
src/main/poisson_spikes_csnn_T.ipynb
src/main/poisson_spike_ninapro_snn_T.ipynb
src/main/online_poisson_spikes_csnn_T.ipynb
src/main/online_poisson_spikes_ninapro_snn_T.ipynb
tests/test_csnn_model.py
docs/log/2026-08-27_NinaPro泊松脉冲SNN与在线采样训练.md
```

本轮整理日志时只新增本文件，没有修改模型、训练代码或现有 Notebook。
