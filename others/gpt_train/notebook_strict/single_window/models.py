"""单窗口模型：输出严格只依赖当前样本的 [16, 40] 输入。"""

import torch
import torch.nn.functional as F
from torch import nn


def validate_input(x, input_channels):
    if not isinstance(x, torch.Tensor) or x.ndim != 3:
        raise ValueError("输入必须是 [B, C, T] 的 torch.Tensor")
    if x.shape[1] != input_channels:
        raise ValueError(f"输入通道数不匹配：{x.shape[1]} != {input_channels}")


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout_rate):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return x + self.network(x)


class MultiResolutionStatistics(nn.Module):
    """在模型内部提取单窗口多尺度统计量，不接触任何元数据。"""

    def __init__(self, input_channels, output_size, dropout_rate):
        super().__init__()
        self.pool_bins = (1, 2, 4, 5, 8, 10, 20, 40)
        self.autocorrelation_lags = (1, 2, 4, 8)
        channel_features = (
            sum(self.pool_bins)
            + 8
            + len(self.autocorrelation_lags)
            + 2
            + 2
        )
        input_features = input_channels * channel_features + input_channels**2
        self.network = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, 768),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(768, output_size),
            nn.SiLU(),
        )

    def forward(self, x):
        multiscale = [F.adaptive_avg_pool1d(x, bins).flatten(1) for bins in self.pool_bins]
        differences = (x[:, :, 1:] - x[:, :, :-1]).abs()
        difference_profile = F.adaptive_avg_pool1d(differences, 8).flatten(1)

        centered = x - x.mean(dim=-1, keepdim=True)
        covariance = torch.bmm(centered, centered.transpose(1, 2)) / x.shape[-1]
        autocorrelations = [
            (centered[:, :, :-lag] * centered[:, :, lag:]).mean(dim=-1)
            for lag in self.autocorrelation_lags
        ]

        # 事件重心与离散程度能保留窗口内位置，同时不引入相邻窗口信息。
        positions = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype)
        weights = x.abs() + 1e-4
        weight_sum = weights.sum(dim=-1).clamp_min(1e-4)
        center = (weights * positions).sum(dim=-1) / weight_sum
        spread = (
            weights * (positions.view(1, 1, -1) - center.unsqueeze(-1)).square()
        ).sum(dim=-1) / weight_sum
        global_statistics = torch.cat(
            [x.mean(dim=-1), x.std(dim=-1, unbiased=False)], dim=1
        )

        features = torch.cat(
            multiscale
            + [
                difference_profile,
                covariance.flatten(1),
                *autocorrelations,
                center,
                spread,
                global_statistics,
            ],
            dim=1,
        )
        return self.network(features)


class MultiViewInceptionNet(nn.Module):
    """融合原始、样本归一化、多尺度卷积与统计分支。"""

    def __init__(
        self,
        input_channels=16,
        num_classes=12,
        width=200,
        dropout_rate=0.3,
    ):
        super().__init__()
        if width % 5:
            raise ValueError("width 必须能被 5 整除")
        self.input_channels = input_channels
        self.num_classes = num_classes
        branch_width = width // 5
        self.stems = nn.ModuleList(
            [
                nn.Conv1d(
                    input_channels * 2,
                    branch_width,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    bias=False,
                )
                for kernel in (1, 3, 5, 9, 15)
            ]
        )
        self.stem_norm = nn.BatchNorm1d(width)
        self.blocks = nn.Sequential(
            *[
                ResidualTemporalBlock(width, dilation, dropout_rate)
                for dilation in (1, 2, 4, 8, 1, 2, 4, 8)
            ]
        )
        self.attention = nn.Sequential(
            nn.Conv1d(width, width // 2, 1),
            nn.SiLU(),
            nn.Conv1d(width // 2, 1, 1),
        )
        self.statistics = MultiResolutionStatistics(
            input_channels, output_size=384, dropout_rate=dropout_rate
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(width * 3 + 384),
            nn.Linear(width * 3 + 384, 512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def reset(self):
        """兼容现有训练器；模型没有跨样本状态。"""

    def forward(self, x):
        validate_input(x, self.input_channels)
        # 第二个视图仅使用当前样本的活动尺度，降低被试间幅值漂移。
        activity_scale = x.abs().mean(dim=(1, 2), keepdim=True).clamp_min(0.05)
        views = torch.cat([x, x / activity_scale], dim=1)
        temporal = torch.cat([stem(views) for stem in self.stems], dim=1)
        temporal = self.blocks(F.silu(self.stem_norm(temporal)))
        attention = torch.softmax(self.attention(temporal), dim=-1)
        pooled = torch.cat(
            [
                temporal.mean(dim=-1),
                temporal.amax(dim=-1),
                (temporal * attention).sum(dim=-1),
            ],
            dim=1,
        )
        return self.classifier(torch.cat([pooled, self.statistics(x)], dim=1))


class Residual2DBlock(nn.Module):
    def __init__(self, channels, dropout_rate, stride=1):
        super().__init__()
        output_channels = channels * stride
        self.network = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
        )
        self.skip = (
            nn.Identity()
            if stride == 1
            else nn.Conv2d(channels, output_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x):
        return self.skip(x) + self.network(x)


class CompactStatistics(nn.Module):
    def __init__(self, input_channels, output_size, dropout_rate):
        super().__init__()
        input_features = input_channels * (8 + 4 + 2)
        self.network = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, output_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_size, output_size),
            nn.SiLU(),
        )

    def forward(self, x):
        rates_8 = F.adaptive_avg_pool1d(x, 8).flatten(1)
        rates_4 = F.adaptive_avg_pool1d(x, 4).flatten(1)
        global_rates = x.mean(dim=-1)
        transitions = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=-1)
        return self.network(
            torch.cat([rates_8, rates_4, global_rates, transitions], dim=1)
        )


class FlatResidual2DBlock(nn.Module):
    def __init__(self, channels, dropout_rate):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.network(x)


class WideResNet2DIndependent(nn.Module):
    """已验证二维残差基线的纯单窗口版本，不包含平滑模块。"""

    def __init__(
        self,
        input_channels=16,
        num_classes=12,
        width=96,
        dropout_rate=0.25,
        tta_time_radius=0,
        tta_channel_radius=0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.tta_time_radius = tta_time_radius
        self.tta_channel_radius = tta_channel_radius
        self.stem = nn.Conv2d(1, width, kernel_size=(3, 5), padding=(1, 2))
        self.blocks = nn.Sequential(
            *[FlatResidual2DBlock(width, dropout_rate) for _ in range(6)]
        )
        self.statistics = CompactStatistics(
            input_channels, output_size=256, dropout_rate=dropout_rate
        )
        self.classifier = nn.Sequential(
            nn.Linear(width * 2 + 256, 512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def reset(self):
        """兼容现有训练器；模型没有跨样本状态。"""

    def forward(self, x):
        validate_input(x, self.input_channels)
        if self.training or (self.tta_time_radius <= 0 and self.tta_channel_radius <= 0):
            return self._forward_once(x)

        # 所有视图都由同一个窗口生成；不同 batch 样本之间没有信息交换。
        views = [x]
        for offset in range(1, self.tta_time_radius + 1):
            views.extend([self._shift_time(x, offset), self._shift_time(x, -offset)])
        for offset in range(1, self.tta_channel_radius + 1):
            views.extend(
                [torch.roll(x, offset, dims=1), torch.roll(x, -offset, dims=1)]
            )
        logits = self._forward_once(torch.cat(views, dim=0))
        return logits.view(len(views), x.shape[0], self.num_classes).mean(dim=0)

    def _forward_once(self, x):
        features = self.blocks(self.stem(x.unsqueeze(1)))
        pooled = torch.cat(
            [features.mean(dim=(-2, -1)), features.amax(dim=(-2, -1))], dim=1
        )
        return self.classifier(torch.cat([pooled, self.statistics(x)], dim=1))

    def _shift_time(self, x, offset):
        shifted = torch.zeros_like(x)
        if offset > 0:
            shifted[:, :, offset:] = x[:, :, :-offset]
        else:
            shifted[:, :, :offset] = x[:, :, -offset:]
        return shifted


class DualAxisResNet(nn.Module):
    """沿电极轴和时间轴联合建模的单窗口二维残差网络。"""

    def __init__(
        self,
        input_channels=16,
        num_classes=12,
        width=64,
        dropout_rate=0.25,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.stem = nn.Conv2d(2, width, kernel_size=(3, 5), padding=(1, 2), bias=False)
        self.blocks = nn.Sequential(
            Residual2DBlock(width, dropout_rate),
            Residual2DBlock(width, dropout_rate),
            Residual2DBlock(width, dropout_rate, stride=2),
            Residual2DBlock(width * 2, dropout_rate),
            Residual2DBlock(width * 2, dropout_rate, stride=2),
            Residual2DBlock(width * 4, dropout_rate),
        )
        self.statistics = MultiResolutionStatistics(
            input_channels, output_size=384, dropout_rate=dropout_rate
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(width * 8 + 384),
            nn.Linear(width * 8 + 384, 512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def reset(self):
        """兼容现有训练器；模型没有跨样本状态。"""

    def forward(self, x):
        validate_input(x, self.input_channels)
        activity_scale = x.abs().mean(dim=(1, 2), keepdim=True).clamp_min(0.05)
        views = torch.stack([x, x / activity_scale], dim=1)
        features = self.blocks(self.stem(views))
        pooled = torch.cat(
            [features.mean(dim=(-2, -1)), features.amax(dim=(-2, -1))], dim=1
        )
        return self.classifier(torch.cat([pooled, self.statistics(x)], dim=1))


class AxisTransformer(nn.Module):
    """分别对时间 token 和电极 token 建模，再与窗口统计特征融合。"""

    def __init__(
        self,
        input_channels=16,
        num_classes=12,
        width=128,
        dropout_rate=0.25,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.time_projection = nn.Linear(input_channels, width)
        self.channel_projection = nn.Linear(40, width)
        self.time_position = nn.Parameter(torch.zeros(1, 40, width))
        self.channel_position = nn.Parameter(torch.zeros(1, input_channels, width))
        layer_options = {
            "d_model": width,
            "nhead": 8,
            "dim_feedforward": width * 4,
            "dropout": dropout_rate,
            "activation": "gelu",
            "batch_first": True,
            "norm_first": True,
        }
        self.time_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_options), num_layers=4
        )
        self.channel_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_options), num_layers=3
        )
        self.statistics = MultiResolutionStatistics(
            input_channels, output_size=384, dropout_rate=dropout_rate
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(width * 4 + 384),
            nn.Linear(width * 4 + 384, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )
        nn.init.trunc_normal_(self.time_position, std=0.02)
        nn.init.trunc_normal_(self.channel_position, std=0.02)

    def reset(self):
        """兼容现有训练器；模型没有跨样本状态。"""

    def forward(self, x):
        validate_input(x, self.input_channels)
        time_tokens = self.time_projection(x.transpose(1, 2)) + self.time_position
        channel_tokens = self.channel_projection(x) + self.channel_position
        time_tokens = self.time_encoder(time_tokens)
        channel_tokens = self.channel_encoder(channel_tokens)
        pooled = torch.cat(
            [
                time_tokens.mean(dim=1),
                time_tokens.amax(dim=1),
                channel_tokens.mean(dim=1),
                channel_tokens.amax(dim=1),
            ],
            dim=1,
        )
        return self.classifier(torch.cat([pooled, self.statistics(x)], dim=1))


def build_model(name, **options):
    candidates = {
        "multiview_inception": MultiViewInceptionNet,
        "wide_resnet2d": WideResNet2DIndependent,
        "dual_axis_resnet": DualAxisResNet,
        "axis_transformer": AxisTransformer,
    }
    if name not in candidates:
        raise ValueError(f"未知模型 {name!r}，可选值为 {tuple(candidates)}")
    return candidates[name](**options)
