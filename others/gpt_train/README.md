# NinaPro T=40 独立训练实验

本目录中的模型均接收 `[B, 16, 40]` 浮点张量并输出 `[B, 12]` logits，
可替换现有 notebook 中的 `DropoutPLIFSNN`。所有训练产物保存在 `runs/`。

最终实验数字、模型选择过程和适用限制见 `RESULTS.md`。

调参阶段只使用 `train.npz`，固定将重复次数 6 作为验证集：

```bash
/root/miniconda3/bin/python train.py --mode tune --model hybrid
```

使用同一 NPZ 中的连续神经形态系统电压 `Vin` 时，模型输入形状和输出接口
保持不变，归一化统计会作为 buffer 一起保存在模型权重中：

```bash
/root/miniconda3/bin/python train.py --mode tune --model hybrid \
  --input-field Vin --spike-dropout 0
```

选定架构和 epoch 后，最终阶段使用全部训练数据拟合，并且仅在结束时评估
`test.npz`：

```bash
/root/miniconda3/bin/python train.py --mode final --model hybrid --epochs BEST_EPOCH
```

从检查点恢复模型：

```python
import torch
from others.gpt_train.models import build_model

checkpoint = torch.load("best.pt", map_location="cpu", weights_only=False)
model = build_model(checkpoint["model_name"], **checkpoint["model_options"])
model.load_state_dict(checkpoint["model_state_dict"])
```

最终经典模型方案先用 `classical_baseline.py` 在重复 6 上冻结超参数，再运行：

```bash
/root/miniconda3/bin/python train_final_svm.py
```

该方案按 NPZ 已有的 `subject` 选择专家，并按 `subject/repetition/start_sample`
的连续时间段平均决策分数。测试集只在最终脚本中评估一次。
