# SNN for NinaPro

基于 PyTorch 和 SpikingJelly 的 NinaPro DB5 表面肌电（sEMG）手势分类项目。项目使用 Exercise A 的 12 类手指动作，将双 Myo 臂环采集的 16 通道信号送入全连接脉冲神经网络（SNN）。

## 数据格式

默认使用滑动窗口数据：

- 输入形状：`[样本数, 16 通道, 40 时间点]`
- 窗口长度：200 ms（40 个采样点）
- 窗口步长：100 ms（20 个采样点）
- 标签范围：`0～11`

处理后的数据应放在：

```text
ninapro_data/processed/exerciseA/slide_window/
├── train.npz
├── test.npz
└── metadata.json
```

数据下载和划分规则参见 [数据准备说明](docs/数据准备_ai.md)。也可以运行 `src/data_prepare_slide_window(simple).ipynb` 生成滑动窗口数据。

## 环境

建议使用 Python 3.8 或更高版本，主要依赖：

```bash
pip install torch spikingjelly numpy matplotlib tqdm jupyter pytest
```

RTX 50 系列与 CUDA 12.x 环境建议额外安装 SpikingJelly 的融合 CUDA 后端：

```bash
pip install -r requirements-fast.txt
```

训练引擎会为旧版 SpikingJelly 补充新 NumPy 已移除的 `np.int` 兼容别名。
未安装 CuPy 或 CuPy 仍不可用时，会直接回退到 PyTorch eager 模式；训练
流程不会调用 `torch.compile`。

## 使用

在项目根目录启动 Jupyter：

```bash
jupyter notebook
```

- `src/train/raw_ninapro_snn.ipynb`：训练和评估 SNN
- `test_notebooks/data_pipeline_test.ipynb`：检查数据管道并绘制 16 通道样本

数据管道的基本用法：

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

训练输出和模型检查点默认保存在 `outputs/` 中。

主训练 Notebook 默认启用快速配置：batch size 固定为 512、FP16 混合
精度、GPU 常驻数据和每 5 轮评估/保存一次。若需要与旧实验逐轮对比，
可将 `BATCH_SIZE` 改回 128，并将 `EVAL_INTERVAL`、
`CHECKPOINT_INTERVAL` 改回 1；`fit(...)` 的其他调用方式不变。

## 测试

```bash
pytest -q
```

## 项目结构

```text
src/data/       数据集、归一化和 DataLoader
src/models/     SNN 模型
src/training/   训练、评估、指标和检查点工具
src/main/      训练 Notebook
tests/          自动化测试
docs/           数据集与数据准备文档
```
