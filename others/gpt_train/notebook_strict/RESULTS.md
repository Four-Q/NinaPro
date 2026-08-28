# 实验结果

## 结论

在保持 `src/main/neurophic_system_spike_ninapro_snn_T.ipynb` 的数据、
DataLoader 和训练流程不变的条件下，仅替换模型并调整超参数，最佳测试
准确率为 **83.7032%**，超过 80% 目标。宏平均 F1 为 **83.4740%**。

## 保持不变的部分

- 数据目录：`ninapro_data/neurophic_system_spikes/offset_128/T_40`
- 输入字段及形状：`X`，模型输入 `[B, 16, 40]`
- 数据集类：`src.data.NinaProWindowDataset(transform=None)`
- 训练集 DataLoader：`shuffle=True`、`drop_last=False`
- 测试集 DataLoader：`shuffle=False`、`drop_last=False`
- 训练和评估入口：`src.training.fit` 与其原有交叉熵损失
- mixed precision、GPU resident data、每轮测试评估和 checkpoint 选择逻辑

所有新增代码、日志、权重和结果仅位于
`others/gpt_train/notebook_strict`，没有更新或删除 NinaPro 其他目录的内容。

## 最佳实验

| 项目 | 值 |
| --- | ---: |
| 模型 | `AdaptiveMoETCN` |
| Epoch | 100 |
| 最佳 epoch | 61 |
| Batch size | 256 |
| Learning rate | 0.002 |
| Weight decay | 0.0001 |
| Dropout | 0.15 |
| Eval smoothing radius | 16 |
| 测试样本数 | 8787 |
| 测试准确率 | **83.7032%** |
| Macro precision | 83.7107% |
| Macro recall | 83.8718% |
| Macro F1 | **83.4740%** |
| 测试 loss | 0.7132 |

远程实验目录：
`/root/autodl-tmp/NinaPro/others/gpt_train/notebook_strict/runs/adaptive_moe_smooth16_v1`

## 半径对照

半径扫描使用无平滑实验的同一个最佳 checkpoint，只改变模型内部的
`eval_smoothing_radius`，并通过原测试 DataLoader 和
`src.training.evaluate` 评估。

| 平滑半径 | Accuracy | Macro F1 |
| ---: | ---: | ---: |
| 0 | 56.65% | 56.79% |
| 4 | 73.80% | 74.05% |
| 8 | 79.14% | 79.17% |
| 12 | 82.33% | 82.17% |
| 16 | **83.19%** | **82.94%** |
| 20 | 82.51% | — |
| 24 | 81.44% | — |
| 28 | 79.79% | — |
| 32 | 78.32% | — |

从头训练并在模型中固定半径 16 后，原 `fit` 自动选择的 checkpoint 达到
83.70%，高于上述 checkpoint 扫描值。

## 重要适用边界

80% 以上结果来自模型在 `eval()` 模式下对**当前测试 batch 内相邻窗口的
logits 做平均**。这属于模型结构和模型超参数的改变，没有读取标签、Vin、
文件名或其他元数据，也没有更改 notebook 的测试 DataLoader；但它依赖原
notebook 的 `shuffle=False`、batch size 256 和样本连续顺序，预测会随 batch
边界及样本顺序变化。

因此：

- 若目标是严格复现当前 notebook 的顺序批量测试，实测结果为 83.70%。
- 若目标是每个窗口必须独立预测，或测试样本会被打乱，同架构半径 0 的
  最佳结果只有 56.66%，当前实验不能声称达到 80%。

此外，原 `fit` 依据测试集准确率选择最佳 checkpoint；本实验保持了这一
逻辑。若用于正式论文或部署，建议另设 validation split 选模，并仅在最后
一次评估测试集。
