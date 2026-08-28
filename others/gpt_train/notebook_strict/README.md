# 严格 notebook 兼容实验

本目录只改变模型或训练超参数。数据字段 `X`、`NinaProWindowDataset`、
DataLoader、训练器 `src.training.fit`、损失函数、测试集及测试顺序均与
`src/main/neurophic_system_spike_ninapro_snn_T.ipynb` 一致。

模型接口保持 `[B, 16, 40] -> [B, 12]`。

已完成的主实验使用 `AdaptiveMoETCN`，测试准确率达到 **83.70%**，
宏平均 F1 为 **83.47%**。精确配置、对照实验和结果适用边界见
[`RESULTS.md`](RESULTS.md)。

复现实验：

```bash
/root/miniconda3/bin/python -u \
  /root/autodl-tmp/NinaPro/others/gpt_train/notebook_strict/run_notebook.py \
  --model adaptive_moe_tcn \
  --run-name adaptive_moe_smooth16_v1 \
  --epochs 100 \
  --batch-size 256 \
  --learning-rate 0.002 \
  --weight-decay 0.0001 \
  --dropout-rate 0.15 \
  --eval-smoothing-radius 16
```
