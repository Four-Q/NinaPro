# 实验结果

## 最终结果

冻结方案在 `test.npz` 上只评估一次：

| 指标 | 单窗口 | 时间段决策分数平均 |
|---|---:|---:|
| Accuracy | 67.24% | **89.80%** |
| Macro F1 | 67.27% | **89.71%** |
| Macro Recall | 66.95% | **89.56%** |

最终方案使用 `Vin`、每位受试者一个 RBF-SVM 专家，以及根据
`subject/repetition/start_sample` 时间间隔自动得到的连续片段。分段过程不读取
标签；验证集重复 6 上的 120 个自动片段均为纯标签片段。固定整段平均后，
验证准确率为 84.46%，最终测试准确率为 89.80%。

完整指标：
`runs/final_subject_svm_vin_c10_segment_mean/test_result.json`

## 模型选择记录

| 方案 | 输入 | 验证准确率 |
|---|---|---:|
| HybridConvNet | 二值 X，单窗口 | 49.41% |
| HybridConvNet | Vin，单窗口 | 51.13% |
| 受试者 RBF-SVM | Vin，单窗口 | 62.59% |
| 受试者 RBF-SVM | Vin，整段平均 | **84.46%** |
| 全局 RBF-SVM | Vin，整段平均 | 69.70% |

`HybridConvNet` 和 `MultiScaleTemporalNet` 遵循现有 notebook 的
`[B, C, T] -> [B, 12]` PyTorch 接口。最终高准确率方案额外依赖 NPZ 元数据，
因此需要使用本目录的最终训练脚本，不能在丢弃元数据的原 notebook 中原样运行。

## 复现

```bash
cd /root/autodl-tmp/NinaPro/others/gpt_train
/root/miniconda3/bin/python train_final_svm.py
```

已保存的十个专家位于：
`runs/final_subject_svm_vin_c10_segment_mean/models/`。
