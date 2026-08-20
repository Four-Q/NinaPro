"""NinaPro 输入变换和训练集统计量计算。"""

import torch


class PhysicalDeviceMapping:
    """物理器件映射的预留接口。

    当前是恒等变换。未来实现器件映射时，应同时支持 ``[C, T]`` 单样本
    和 ``[B, C, T]`` 批数据，以便使用完全相同的映射计算归一化统计量。
    """

    def __call__(self, x):
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ChannelwiseZScore:
    """使用训练集统计量对每个通道执行 Z-score 归一化。"""

    def __init__(self, mean, std, eps=1e-6):
        if eps <= 0:
            raise ValueError("eps 必须大于 0")

        self.mean = torch.as_tensor(mean, dtype=torch.float32).flatten().clone()
        self.std = torch.as_tensor(std, dtype=torch.float32).flatten().clone()
        self.eps = float(eps)

        if self.mean.numel() == 0:
            raise ValueError("mean 和 std 不能为空")
        if self.mean.shape != self.std.shape:
            raise ValueError(
                f"mean 与 std 的形状必须相同：{self.mean.shape} != {self.std.shape}"
            )
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("mean 或 std 中存在 NaN 或无穷值")
        if torch.any(self.std < 0):
            raise ValueError("std 不能包含负数")

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("ChannelwiseZScore 的输入必须是 torch.Tensor")
        if x.ndim < 2:
            raise ValueError(f"输入至少需要 [C, T] 两个维度，实际形状为 {x.shape}")
        if x.shape[-2] != self.mean.numel():
            raise ValueError(
                "输入通道数与归一化统计量不一致："
                f"{x.shape[-2]} != {self.mean.numel()}"
            )

        if not x.is_floating_point():
            x = x.to(torch.float32)

        # 将统计量放在倒数第二维，兼容 [C, T] 和 [..., C, T]。
        stats_shape = [1] * x.ndim
        stats_shape[-2] = self.mean.numel()
        mean = self.mean.to(device=x.device, dtype=x.dtype).view(stats_shape)
        std = self.std.to(device=x.device, dtype=x.dtype).view(stats_shape)
        return (x - mean) / (std + self.eps)

    def state_dict(self):
        """返回可随模型检查点保存的归一化统计量。"""

        return {
            "mean": self.mean.detach().cpu().clone(),
            "std": self.std.detach().cpu().clone(),
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state):
        required_keys = {"mean", "std"}
        missing_keys = required_keys.difference(state)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise KeyError(f"归一化状态缺少字段：{missing}")
        return cls(state["mean"], state["std"], state.get("eps", 1e-6))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_channels={self.mean.numel()}, "
            f"eps={self.eps})"
        )


class ComposeTransforms:
    """不依赖 torchvision 的可序列化 Transform 组合器。"""

    def __init__(self, transforms):
        self.transforms = tuple(transforms)
        if not self.transforms:
            raise ValueError("transforms 不能为空")
        if any(not callable(transform) for transform in self.transforms):
            raise TypeError("transforms 中的每个对象都必须可调用")

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def __repr__(self):
        body = ",\n  ".join(repr(transform) for transform in self.transforms)
        return f"{self.__class__.__name__}([\n  {body}\n])"


def compute_channel_stats(features, pre_transform=None, chunk_size=1024):
    """计算 ``[N, C, T]`` 数据在 N、T 维度上的总体均值和标准差。

    ``pre_transform`` 会先作用于每个批次，因此物理器件映射始终位于
    归一化之前。使用分块和 float64 累加，避免一次性创建大型临时数组。
    """

    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim != 3:
        raise ValueError(f"features 必须具有 [N, C, T] 形状，实际为 {features.shape}")
    if features.shape[0] == 0:
        raise ValueError("features 不能为空")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size 必须是正整数")

    value_sum = None
    squared_sum = None
    value_count = 0

    with torch.no_grad():
        for start in range(0, features.shape[0], chunk_size):
            batch = features[start : start + chunk_size]
            if pre_transform is not None:
                batch = pre_transform(batch)
            if not isinstance(batch, torch.Tensor):
                batch = torch.as_tensor(batch)
            if batch.ndim != 3:
                raise ValueError(
                    "pre_transform 必须保持 [N, C, T] 三维结构，"
                    f"实际输出为 {batch.shape}"
                )
            if not torch.isfinite(batch).all():
                raise ValueError("用于计算统计量的数据中存在 NaN 或无穷值")

            batch = batch.to(dtype=torch.float64)
            batch_sum = batch.sum(dim=(0, 2))
            batch_squared_sum = batch.square().sum(dim=(0, 2))

            if value_sum is None:
                value_sum = torch.zeros_like(batch_sum)
                squared_sum = torch.zeros_like(batch_squared_sum)
            elif value_sum.shape != batch_sum.shape:
                raise ValueError("pre_transform 在不同批次产生了不同的通道数")

            value_sum += batch_sum
            squared_sum += batch_squared_sum
            value_count += batch.shape[0] * batch.shape[2]

    mean = value_sum / value_count
    # 浮点舍入可能产生极小负数，截断后再开方。
    variance = (squared_sum / value_count - mean.square()).clamp_min(0.0)
    std = variance.sqrt()
    return mean.to(torch.float32), std.to(torch.float32)
