"""兼容现有 NinaPro notebook 输入输出约定的分类模型。"""

import torch
import torch.nn.functional as F
from torch import nn


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(self.norm2(self.conv2(x)))
        return F.gelu(x + residual)


class SpatialResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(self.norm2(self.conv2(x)))
        return F.gelu(x + residual)


class TemporalFeatureEncoder(nn.Module):
    def __init__(self, input_channels, width=128, dropout_rate=0.15):
        super().__init__()
        branch_width = width // 4
        last_width = width - branch_width * 3
        branch_widths = (branch_width, branch_width, branch_width, last_width)
        kernels = (3, 5, 9, 15)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    input_channels,
                    output_width,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    bias=False,
                )
                for output_width, kernel in zip(branch_widths, kernels)
            ]
        )
        self.stem_norm = nn.BatchNorm1d(width)
        self.blocks = nn.Sequential(
            TemporalResidualBlock(width, dilation=1, dropout_rate=dropout_rate),
            TemporalResidualBlock(width, dilation=2, dropout_rate=dropout_rate),
            TemporalResidualBlock(width, dilation=4, dropout_rate=dropout_rate),
            TemporalResidualBlock(width, dilation=8, dropout_rate=dropout_rate),
        )
        self.attention = nn.Conv1d(width, 1, kernel_size=1)
        self.output_features = width * 3

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = F.gelu(self.stem_norm(x))
        x = self.blocks(x)

        # 同时保留平均、峰值和可学习时间注意力，兼顾持续与瞬时激活。
        weights = torch.softmax(self.attention(x), dim=-1)
        attention_pool = (x * weights).sum(dim=-1)
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1), attention_pool], dim=1)


class SpatialFeatureEncoder(nn.Module):
    def __init__(self, width=48, dropout_rate=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            SpatialResidualBlock(width, dropout_rate=dropout_rate),
            SpatialResidualBlock(width, dropout_rate=dropout_rate),
        )
        self.output_features = width * 2

    def forward(self, x):
        x = self.blocks(self.stem(x.unsqueeze(1)))
        return torch.cat(
            [x.mean(dim=(-2, -1)), x.amax(dim=(-2, -1))], dim=1
        )


class StatisticalFeatureEncoder(nn.Module):
    def __init__(self, input_channels=16, segments=8):
        super().__init__()
        self.input_channels = input_channels
        self.segments = segments
        self.output_features = input_channels * (segments + 2)

    def forward(self, x):
        # 分段发放率保留粗粒度时间位置；翻转率描述脉冲边沿密度。
        segment_rates = F.adaptive_avg_pool1d(x, self.segments).flatten(1)
        global_rates = x.mean(dim=-1)
        transition_rates = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=-1)
        return torch.cat([segment_rates, global_rates, transition_rates], dim=1)


class HybridConvNet(nn.Module):
    """融合时间、通道邻域和显式发放率统计的 ANN 分类器。"""

    def __init__(
        self,
        input_channels=16,
        num_classes=12,
        temporal_width=128,
        spatial_width=48,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.temporal_width = temporal_width
        self.spatial_width = spatial_width
        self.dropout_rate = dropout_rate
        self.register_buffer(
            "input_mean", torch.zeros(1, input_channels, 1), persistent=True
        )
        self.register_buffer(
            "input_std", torch.ones(1, input_channels, 1), persistent=True
        )

        self.temporal_encoder = TemporalFeatureEncoder(
            input_channels,
            width=temporal_width,
            dropout_rate=dropout_rate * 0.75,
        )
        self.spatial_encoder = SpatialFeatureEncoder(
            width=spatial_width,
            dropout_rate=dropout_rate * 0.5,
        )
        self.statistical_encoder = StatisticalFeatureEncoder(input_channels)
        fused_features = (
            self.temporal_encoder.output_features
            + self.spatial_encoder.output_features
            + self.statistical_encoder.output_features
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_features),
            nn.Linear(fused_features, 384),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def reset(self):
        """与现有 SNN 接口对齐；无状态 ANN 不需要执行操作。"""

    def set_input_normalization(self, mean, std):
        self.input_mean.copy_(mean.reshape(1, self.input_channels, 1))
        self.input_std.copy_(std.reshape(1, self.input_channels, 1))

    def forward(self, x):
        self._validate_input(x)
        x = (x - self.input_mean) / self.input_std.clamp_min(1e-6)
        features = torch.cat(
            [
                self.temporal_encoder(x),
                self.spatial_encoder(x),
                self.statistical_encoder(x),
            ],
            dim=1,
        )
        return self.classifier(features)

    def _validate_input(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("输入必须是形状为 [B, C, T] 的 torch.Tensor")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"输入通道数不匹配：{x.shape[1]} != {self.input_channels}"
            )


class MultiScaleTemporalNet(nn.Module):
    """用于消融对比的纯时间多尺度卷积网络。"""

    def __init__(
        self,
        input_channels=16,
        num_classes=12,
        temporal_width=160,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.temporal_width = temporal_width
        self.dropout_rate = dropout_rate
        self.register_buffer(
            "input_mean", torch.zeros(1, input_channels, 1), persistent=True
        )
        self.register_buffer(
            "input_std", torch.ones(1, input_channels, 1), persistent=True
        )
        self.temporal_encoder = TemporalFeatureEncoder(
            input_channels,
            width=temporal_width,
            dropout_rate=dropout_rate * 0.75,
        )
        self.statistical_encoder = StatisticalFeatureEncoder(input_channels)
        fused_features = (
            self.temporal_encoder.output_features
            + self.statistical_encoder.output_features
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_features),
            nn.Linear(fused_features, 320),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(320, num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def reset(self):
        """与现有 SNN 接口对齐；无状态 ANN 不需要执行操作。"""

    def set_input_normalization(self, mean, std):
        self.input_mean.copy_(mean.reshape(1, self.input_channels, 1))
        self.input_std.copy_(std.reshape(1, self.input_channels, 1))

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std.clamp_min(1e-6)
        features = torch.cat(
            [self.temporal_encoder(x), self.statistical_encoder(x)], dim=1
        )
        return self.classifier(features)


def build_model(name, **options):
    models = {
        "hybrid": HybridConvNet,
        "temporal": MultiScaleTemporalNet,
    }
    if name not in models:
        raise ValueError(f"未知模型 {name!r}，可选值为 {tuple(models)}")
    return models[name](**options)
