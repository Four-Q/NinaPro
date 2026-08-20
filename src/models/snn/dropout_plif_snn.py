"""带时间一致 Dropout 和可学习膜时间常数的 NinaPro SNN。"""

from spikingjelly.activation_based import layer, neuron, surrogate

from .fc_snn import NinaProSNN


class DropoutPLIFSNN(NinaProSNN):
    """在全连接 SNN 基础上使用两层 PLIF 和脉冲序列 Dropout。

    Dropout 掩码在一个样本窗口的全部时间步内保持一致，并在下一次前向
    开始前随网络状态一起重置。这样可以避免逐时间步随机掩码被时间平均
    稀释，同时保持不同 mini-batch 之间的随机正则化。
    """

    def __init__(
        self,
        input_channels=16,
        hidden_size=128,
        num_classes=12,
        tau=2.0,
        dropout_rate=0.3,
        auto_reset=True,
        backend="torch",
    ):
        self._validate_dropout_rate(dropout_rate)
        super().__init__(
            input_channels=input_channels,
            hidden_size=hidden_size,
            num_classes=num_classes,
            tau=tau,
            auto_reset=auto_reset,
            backend=backend,
        )
        self.dropout_rate = float(dropout_rate)
        self.dropout1 = layer.Dropout(p=self.dropout_rate, step_mode=self.step_mode)
        self.dropout2 = layer.Dropout(p=self.dropout_rate, step_mode=self.step_mode)

    @staticmethod
    def _validate_dropout_rate(dropout_rate):
        if (
            not isinstance(dropout_rate, (int, float))
            or isinstance(dropout_rate, bool)
            or not 0.0 <= dropout_rate < 1.0
        ):
            raise ValueError("dropout_rate 必须是 [0, 1) 范围内的数值")

    def _make_lif_node(self):
        return neuron.ParametricLIFNode(
            init_tau=self.tau,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode=self.step_mode,
            backend=self.backend,
            store_v_seq=False,
        )

    def reset(self):
        """清空 PLIF 膜电位以及两个 Dropout 层保存的时间一致掩码。"""

        self.lif1.reset()
        self.dropout1.reset()
        self.lif2.reset()
        self.dropout2.reset()

    def forward(self, x):
        self._validate_input(x)

        if self.auto_reset:
            # 每个窗口使用独立膜电位和掩码，但窗口内全部时间步共享掩码。
            self.reset()

        x_seq = x.permute(2, 0, 1).contiguous()
        spike_seq = self.lif1(self.input_projection(x_seq))
        spike_seq = self.dropout1(spike_seq)
        spike_seq = self.lif2(self.hidden_projection(spike_seq))
        spike_seq = self.dropout2(spike_seq)
        logits_seq = self.classifier(spike_seq)
        return logits_seq.mean(dim=0)

    def extra_repr(self):
        return f"{super().extra_repr()}, dropout_rate={self.dropout_rate}"
