"""沿电极通道轴卷积的 NinaPro 多步 SNN。"""

import torch
from torch import nn

from spikingjelly.activation_based import layer, neuron, surrogate


class _ChannelConvSpikeBlock(nn.Module):
    """逐时间步执行 Conv1d，再沿时间轴更新脉冲神经元。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        tau,
        dropout_rate,
        learnable_tau,
        backend,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.normalization = nn.BatchNorm1d(out_channels)
        self.neuron = self._make_neuron(tau, learnable_tau, backend)
        self.dropout = layer.Dropout(p=dropout_rate, step_mode="m")

    @staticmethod
    def _make_neuron(tau, learnable_tau, backend):
        shared_options = {
            "decay_input": True,
            "v_threshold": 1.0,
            "v_reset": 0.0,
            "surrogate_function": surrogate.ATan(),
            "detach_reset": True,
            "step_mode": "m",
            "backend": backend,
            "store_v_seq": False,
        }
        if learnable_tau:
            return neuron.ParametricLIFNode(init_tau=tau, **shared_options)
        return neuron.LIFNode(tau=tau, **shared_options)

    def reset(self):
        """清除膜电位和时间一致 Dropout 掩码。"""

        self.neuron.reset()
        self.dropout.reset()

    def forward(self, x_seq):
        time_steps, batch_size, input_channels, electrode_channels = x_seq.shape

        # Conv1d 不识别 SNN 时间维，先合并 T 与 B，再恢复为多步序列。
        current = x_seq.reshape(
            time_steps * batch_size,
            input_channels,
            electrode_channels,
        )
        current = self.normalization(self.conv(current))
        current_seq = current.reshape(
            time_steps,
            batch_size,
            current.shape[1],
            electrode_channels,
        )
        return self.dropout(self.neuron(current_seq))


class NinaProCSNN(nn.Module):
    """在完整 16 电极通道轴上使用 Conv1d 的 NinaPro SNN。

    模型接收数据集原有的 ``[B, C, T]``。``offset_128`` 使用
    ``C=16``，内部布局为 ``[B, 1, 16, T]``；``polarity_split`` 使用
    ``C=32``，内部布局为 ``[B, 2, 16, T]``。每个 SNN 时间步都在完整
    16 电极轴上执行 Conv1d，脉冲神经元状态则沿 ``T`` 演化。模型不在
    中间层池化电极位置，只在连续读出前汇聚通道轴。
    """

    def __init__(
        self,
        input_channels=32,
        polarity_channels=2,
        electrode_channels=16,
        feature_channels=(32, 64, 128),
        num_classes=12,
        tau=2.0,
        dropout_rate=0.3,
        learnable_tau=False,
        auto_reset=True,
        backend="torch",
    ):
        super().__init__()
        self._validate_config(
            input_channels=input_channels,
            polarity_channels=polarity_channels,
            electrode_channels=electrode_channels,
            feature_channels=feature_channels,
            num_classes=num_classes,
            tau=tau,
            dropout_rate=dropout_rate,
            learnable_tau=learnable_tau,
            auto_reset=auto_reset,
            backend=backend,
        )

        self.input_channels = input_channels
        self.polarity_channels = polarity_channels
        self.electrode_channels = electrode_channels
        self.feature_channels = tuple(feature_channels)
        self.num_classes = num_classes
        self.tau = float(tau)
        self.dropout_rate = float(dropout_rate)
        self.learnable_tau = learnable_tau
        self.auto_reset = auto_reset
        self.backend = backend
        self.step_mode = "m"

        channel_pairs = zip(
            (self.polarity_channels, *self.feature_channels[:-1]),
            self.feature_channels,
        )
        self.blocks = nn.ModuleList(
            [
                _ChannelConvSpikeBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    tau=self.tau,
                    dropout_rate=self.dropout_rate,
                    learnable_tau=self.learnable_tau,
                    backend=self.backend,
                )
                for in_channels, out_channels in channel_pairs
            ]
        )
        self.classifier = nn.Linear(self.feature_channels[-1], self.num_classes)
        self.reset_parameters()

    @staticmethod
    def _validate_config(
        input_channels,
        polarity_channels,
        electrode_channels,
        feature_channels,
        num_classes,
        tau,
        dropout_rate,
        learnable_tau,
        auto_reset,
        backend,
    ):
        integer_options = {
            "input_channels": input_channels,
            "polarity_channels": polarity_channels,
            "electrode_channels": electrode_channels,
            "num_classes": num_classes,
        }
        for name, value in integer_options.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if input_channels != polarity_channels * electrode_channels:
            raise ValueError(
                "input_channels 必须等于 polarity_channels * electrode_channels"
            )
        if (
            not isinstance(feature_channels, (tuple, list))
            or len(feature_channels) != 3
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in feature_channels
            )
        ):
            raise ValueError("feature_channels 必须包含三个正整数")
        if not isinstance(tau, (int, float)) or isinstance(tau, bool) or tau <= 1.0:
            raise ValueError("tau 必须是大于 1 的数值")
        if (
            not isinstance(dropout_rate, (int, float))
            or isinstance(dropout_rate, bool)
            or not 0.0 <= dropout_rate < 1.0
        ):
            raise ValueError("dropout_rate 必须是 [0, 1) 范围内的数值")
        if not isinstance(learnable_tau, bool):
            raise TypeError("learnable_tau 必须是 bool")
        if not isinstance(auto_reset, bool):
            raise TypeError("auto_reset 必须是 bool")
        if backend not in {"torch", "cupy"}:
            raise ValueError("backend 必须是 'torch' 或 'cupy'")

    def reset_parameters(self):
        """初始化卷积、归一化和连续读出层。"""

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def reset(self):
        """清空全部 LIF/PLIF 膜电位和 Dropout 状态。"""

        for block in self.blocks:
            block.reset()

    def forward(self, x):
        self._validate_input(x)
        if self.auto_reset:
            # 不同滑动窗口批次相互独立，禁止膜电位跨批次泄漏。
            self.reset()

        batch_size, _, time_steps = x.shape
        features = x.reshape(
            batch_size,
            self.polarity_channels,
            self.electrode_channels,
            time_steps,
        )
        feature_seq = features.permute(3, 0, 1, 2).contiguous()
        for block in self.blocks:
            feature_seq = block(feature_seq)

        # 保留 SNN 时间轴，只在连续分类前汇聚 16 个电极位置。
        logits_seq = self.classifier(feature_seq.mean(dim=-1))
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
            f"input_channels={self.input_channels}, "
            f"input_layout=({self.polarity_channels}, {self.electrode_channels}), "
            f"feature_channels={self.feature_channels}, "
            f"num_classes={self.num_classes}, tau={self.tau}, "
            f"dropout_rate={self.dropout_rate}, learnable_tau={self.learnable_tau}, "
            f"step_mode='{self.step_mode}', backend='{self.backend}', "
            f"auto_reset={self.auto_reset}"
        )
