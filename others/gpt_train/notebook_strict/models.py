"""接收 [B, 16, 40] 并返回 [B, 12] logits 的候选模型。"""

import torch
import torch.nn.functional as F
from torch import nn


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.dropout(self.conv2(F.silu(self.norm2(x))))
        return x + residual


class TemporalEncoder(nn.Module):
    def __init__(self, input_channels, width, dropout_rate):
        super().__init__()
        branch_width = width // 4
        final_width = width - branch_width * 3
        self.stem = nn.ModuleList(
            [
                nn.Conv1d(
                    input_channels,
                    output_width,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    bias=False,
                )
                for output_width, kernel in zip(
                    (branch_width, branch_width, branch_width, final_width),
                    (3, 5, 9, 15),
                )
            ]
        )
        self.stem_norm = nn.BatchNorm1d(width)
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(width, dilation, dropout_rate)
                for dilation in (1, 2, 4, 8, 1, 2)
            ]
        )
        self.attention = nn.Sequential(
            nn.Conv1d(width, width // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(width // 2, 1, kernel_size=1),
        )
        self.output_features = width * 3

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.stem], dim=1)
        x = self.blocks(F.silu(self.stem_norm(x)))
        weights = torch.softmax(self.attention(x), dim=-1)
        attended = (x * weights).sum(dim=-1)
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1), attended], dim=1)


class StatisticalEncoder(nn.Module):
    def __init__(self, input_channels, hidden_size, dropout_rate):
        super().__init__()
        input_features = input_channels * (8 + 4 + 2)
        self.network = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.output_features = hidden_size

    def forward(self, x):
        rates_8 = F.adaptive_avg_pool1d(x, 8).flatten(1)
        rates_4 = F.adaptive_avg_pool1d(x, 4).flatten(1)
        global_rates = x.mean(dim=-1)
        transitions = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=-1)
        return self.network(
            torch.cat([rates_8, rates_4, global_rates, transitions], dim=1)
        )


class EvaluationBatchSmoother(nn.Module):
    def __init__(self, radius=0):
        super().__init__()
        self.radius = radius

    def forward(self, logits):
        if self.training or self.radius <= 0 or logits.shape[0] <= 1:
            return logits
        kernel_size = self.radius * 2 + 1
        # 原 notebook 的测试 loader 保持顺序；平滑严格限制在当前 batch 内。
        values = logits.transpose(0, 1).unsqueeze(0)
        values = F.avg_pool1d(
            values,
            kernel_size=kernel_size,
            stride=1,
            padding=self.radius,
            count_include_pad=False,
        )
        return values.squeeze(0).transpose(0, 1)


class AdaptiveMoETCN(nn.Module):
    """多尺度 TCN 与软专家路由融合的单窗口分类器。"""

    def __init__(
        self,
        input_channels=16,
        num_classes=12,
        width=192,
        embedding_size=512,
        num_experts=6,
        dropout_rate=0.15,
        eval_smoothing_radius=0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.width = width
        self.embedding_size = embedding_size
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate
        self.eval_smoothing_radius = eval_smoothing_radius

        self.temporal = TemporalEncoder(input_channels, width, dropout_rate)
        self.statistics = StatisticalEncoder(
            input_channels, hidden_size=256, dropout_rate=dropout_rate
        )
        fused_features = self.temporal.output_features + self.statistics.output_features
        self.embedding = nn.Sequential(
            nn.LayerNorm(fused_features),
            nn.Linear(fused_features, embedding_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.shared_classifier = nn.Linear(embedding_size, num_classes)
        self.experts = nn.ModuleList(
            [nn.Linear(embedding_size, num_classes) for _ in range(num_experts)]
        )
        self.gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts),
        )
        self.smoother = EvaluationBatchSmoother(eval_smoothing_radius)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def reset(self):
        """与现有 SNN 统一接口；当前模型不保存膜电位。"""

    def forward(self, x):
        self._validate_input(x)
        statistics = self.statistics(x)
        embedding = self.embedding(torch.cat([self.temporal(x), statistics], dim=1))
        shared_logits = self.shared_classifier(embedding)
        expert_logits = torch.stack(
            [expert(embedding) for expert in self.experts], dim=1
        )
        gate = torch.softmax(self.gate(statistics), dim=1).unsqueeze(-1)
        logits = shared_logits + (expert_logits * gate).sum(dim=1)
        return self.smoother(logits)

    def _validate_input(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("输入必须是 [B, C, T] 的 torch.Tensor")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"输入通道数不匹配：{x.shape[1]} != {self.input_channels}"
            )


class Residual2DBlock(nn.Module):
    def __init__(self, channels, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class WideResNet2D(nn.Module):
    """将 16 电极乘 40 时间点视为二维图的宽残差网络。"""

    def __init__(
        self,
        input_channels=16,
        num_classes=12,
        width=96,
        dropout_rate=0.15,
        eval_smoothing_radius=0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.width = width
        self.dropout_rate = dropout_rate
        self.eval_smoothing_radius = eval_smoothing_radius
        self.stem = nn.Conv2d(1, width, kernel_size=(3, 5), padding=(1, 2))
        self.blocks = nn.Sequential(
            *[Residual2DBlock(width, dropout_rate) for _ in range(6)]
        )
        self.statistics = StatisticalEncoder(
            input_channels, hidden_size=256, dropout_rate=dropout_rate
        )
        self.classifier = nn.Sequential(
            nn.Linear(width * 2 + 256, 512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )
        self.smoother = EvaluationBatchSmoother(eval_smoothing_radius)

    def reset(self):
        """与现有 SNN 统一接口；当前模型没有跨批次状态。"""

    def forward(self, x):
        features = self.blocks(self.stem(x.unsqueeze(1)))
        pooled = torch.cat(
            [features.mean(dim=(-2, -1)), features.amax(dim=(-2, -1))], dim=1
        )
        logits = self.classifier(torch.cat([pooled, self.statistics(x)], dim=1))
        return self.smoother(logits)


def build_model(name, **options):
    candidates = {
        "adaptive_moe_tcn": AdaptiveMoETCN,
        "wide_resnet2d": WideResNet2D,
    }
    if name not in candidates:
        raise ValueError(f"未知模型 {name!r}，可选值为 {tuple(candidates)}")
    return candidates[name](**options)
