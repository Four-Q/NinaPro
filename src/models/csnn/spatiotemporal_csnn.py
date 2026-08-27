"""面向极性拆分泊松脉冲的时空二维卷积 SNN。"""

import torch
from torch import nn

from spikingjelly.activation_based import layer, neuron, surrogate


class _ConvPLIFBlock(nn.Module):
    """二维卷积、归一化、PLIF 与空间池化组成的特征块。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        tau,
        dropout_rate,
        backend,
        spatial_pool,
    ):
        super().__init__()
        padding = tuple(size // 2 for size in kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.neuron = neuron.ParametricLIFNode(
            init_tau=tau,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode="m",
            backend=backend,
            store_v_seq=False,
        )
        self.dropout = layer.Dropout(p=dropout_rate, step_mode="m")
        self.pool = nn.AvgPool2d(kernel_size=(spatial_pool, 1))

    def reset(self):
        """清除膜电位和时间一致 Dropout 掩码。"""

        self.neuron.reset()
        self.dropout.reset()

    def forward(self, x):
        current = self.normalization(self.conv(x))

        # Conv2d 使用 [B, C, electrode, time]；多步 PLIF 要求时间维在最前。
        current_seq = current.movedim(-1, 0).contiguous()
        spike_seq = self.dropout(self.neuron(current_seq))
        spikes = spike_seq.movedim(0, -1).contiguous()
        return self.pool(spikes)


class NinaProCSNN(nn.Module):
    """使用二维时空卷积学习 NinaPro 极性脉冲的分类器。

    输入保持数据文件中的 ``[B, 32, T]`` 布局。前 16 个通道是正极性，
    后 16 个通道是负极性；模型内部将其重塑为 ``[B, 2, 16, T]``。
    普通 ``Conv2d`` 同时提取电极与时间局部特征，PLIF 神经元沿卷积结果的
    时间轴演化。最后仅汇聚电极轴，并平均全部时间步的连续分类 logits。
    """

    def __init__(
        self,
        input_channels=32,
        polarity_channels=2,
        electrode_channels=16,
        feature_channels=(64, 128, 256),
        num_classes=12,
        tau=2.0,
        dropout_rate=0.2,
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
        self.auto_reset = auto_reset
        self.backend = backend
        self.step_mode = "m"

        kernels = ((3, 5), (3, 5), (3, 3))
        channel_pairs = zip(
            (self.polarity_channels, *self.feature_channels[:-1]),
            self.feature_channels,
        )
        self.blocks = nn.ModuleList(
            [
                _ConvPLIFBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    tau=self.tau,
                    dropout_rate=self.dropout_rate,
                    backend=self.backend,
                    spatial_pool=2,
                )
                for (in_channels, out_channels), kernel_size in zip(
                    channel_pairs,
                    kernels,
                )
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
        if electrode_channels < 8:
            raise ValueError("electrode_channels 必须至少为 8，以支持三次空间池化")
        if not isinstance(tau, (int, float)) or isinstance(tau, bool) or tau <= 1.0:
            raise ValueError("tau 必须是大于 1 的数值")
        if (
            not isinstance(dropout_rate, (int, float))
            or isinstance(dropout_rate, bool)
            or not 0.0 <= dropout_rate < 1.0
        ):
            raise ValueError("dropout_rate 必须是 [0, 1) 范围内的数值")
        if not isinstance(auto_reset, bool):
            raise TypeError("auto_reset 必须是 bool")
        if backend not in {"torch", "cupy"}:
            raise ValueError("backend 必须是 'torch' 或 'cupy'")

    def reset_parameters(self):
        """初始化卷积、归一化与连续读出层。"""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def reset(self):
        """清空全部 PLIF 膜电位和 Dropout 状态。"""

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
        for block in self.blocks:
            features = block(features)

        # 保留 SNN 时间轴，仅汇聚电极位置，然后逐时间步进行连续分类。
        feature_seq = features.mean(dim=2).permute(2, 0, 1).contiguous()
        logits_seq = self.classifier(feature_seq)
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
            f"dropout_rate={self.dropout_rate}, step_mode='{self.step_mode}', "
            f"backend='{self.backend}', auto_reset={self.auto_reset}"
        )
