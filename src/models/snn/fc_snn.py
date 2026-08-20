"""面向 NinaPro 滑动窗口的全连接 SNN 分类器。"""

import torch
from torch import nn

from spikingjelly.activation_based import neuron, surrogate


class NinaProSNN(nn.Module):
    """直接使用 sEMG 时间序列训练的多步 SNN。

    输入采用数据管道的 ``[B, C, T]`` 布局。模型内部将其转换为
    ``[T, B, C]``，把每个 sEMG 采样点视作一个 SNN 时间步。两个隐藏层
    使用 LIF 神经元，输出层保持为连续 logits 并在时间维上取平均。
    """

    def __init__(
        self,
        input_channels=16,
        hidden_size=128,
        num_classes=12,
        tau=2.0,
        auto_reset=True,
        backend="torch",
    ):
        super().__init__()
        self._validate_config(
            input_channels=input_channels,
            hidden_size=hidden_size,
            num_classes=num_classes,
            tau=tau,
            auto_reset=auto_reset,
            backend=backend,
        )

        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.tau = float(tau)
        self.auto_reset = auto_reset
        self.backend = backend
        self.step_mode = "m"

        self.input_projection = nn.Linear(input_channels, hidden_size)
        self.lif1 = self._make_lif_node()
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.lif2 = self._make_lif_node()
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.reset_parameters()

    @staticmethod
    def _validate_config(
        input_channels,
        hidden_size,
        num_classes,
        tau,
        auto_reset,
        backend,
    ):
        integer_options = {
            "input_channels": input_channels,
            "hidden_size": hidden_size,
            "num_classes": num_classes,
        }
        for name, value in integer_options.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if not isinstance(tau, (int, float)) or isinstance(tau, bool) or tau <= 1.0:
            raise ValueError("tau 必须是大于 1 的数值")
        if not isinstance(auto_reset, bool):
            raise TypeError("auto_reset 必须是 bool")
        if backend not in {"torch", "cupy"}:
            raise ValueError("backend 必须是 'torch' 或 'cupy'")

    def _make_lif_node(self):
        return neuron.LIFNode(
            tau=self.tau,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode=self.step_mode,
            backend=self.backend,
            store_v_seq=False,
        )

    def reset_parameters(self):
        """初始化线性层，并为稀疏脉冲传播提供合适的初始权重尺度。"""

        self.input_projection.reset_parameters()
        self.hidden_projection.reset_parameters()
        self.classifier.reset_parameters()

        with torch.no_grad():
            # PyTorch 默认尺度会使第二层 LIF 在初始阶段完全静默。
            # 提升权重尺度可产生稀疏但非零的初始脉冲，并保持原有拓扑。
            self.input_projection.weight.mul_(2.0)
            self.hidden_projection.weight.mul_(4.0)

    def reset(self):
        """清空全部脉冲神经元的膜电位状态。"""

        # 不调用 functional.reset_net(self)，否则它会再次发现本方法并递归。
        self.lif1.reset()
        self.lif2.reset()

    def forward(self, x):
        self._validate_input(x)

        if self.auto_reset:
            # 各滑动窗口批次互相独立，前向前重置可避免膜电位跨批次泄漏。
            self.reset()

        x_seq = x.permute(2, 0, 1).contiguous()
        spike_seq = self.lif1(self.input_projection(x_seq))
        spike_seq = self.lif2(self.hidden_projection(spike_seq))
        logits_seq = self.classifier(spike_seq)

        # 连续读出层不发放脉冲，以保留适合 CrossEntropyLoss 的无界 logits。
        return logits_seq.mean(dim=0)

    def _validate_input(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("模型输入必须是 torch.Tensor")
        if x.ndim != 3:
            raise ValueError(
                "模型输入必须具有 [B, C, T] 三个维度，"
                f"实际形状为 {tuple(x.shape)}"
            )
        if x.shape[0] == 0:
            raise ValueError("批次不能为空")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                "输入通道数与模型配置不一致："
                f"{x.shape[1]} != {self.input_channels}"
            )
        if x.shape[2] == 0:
            raise ValueError("时间维不能为空")
        if not x.is_floating_point():
            raise TypeError("模型输入必须是浮点张量")

    def extra_repr(self):
        return (
            f"input_channels={self.input_channels}, hidden_size={self.hidden_size}, "
            f"num_classes={self.num_classes}, tau={self.tau}, "
            f"step_mode='{self.step_mode}', backend='{self.backend}', "
            f"auto_reset={self.auto_reset}"
        )
