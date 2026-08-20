"""训练设备选择与随机状态控制。"""

import random

import numpy as np
import torch


def seed_everything(seed=42, deterministic=False):
    """设置 Python、NumPy、PyTorch 和 CUDA 的随机种子。"""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed 必须是整数")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic 必须是 bool")

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        # 非确定性模式允许 cuDNN 根据输入形状选择更快的实现。
        torch.backends.cudnn.benchmark = not deterministic
    return seed


def resolve_device(device=None):
    """自动选择设备，同时检查显式 CUDA 请求是否可用。"""

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 PyTorch 环境没有可用 CUDA 设备")
    return resolved
