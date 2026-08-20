"""直接训练的 NinaPro 脉冲神经网络。"""

from .dropout_plif_snn import DropoutPLIFSNN
from .fc_snn import NinaProSNN

__all__ = ["DropoutPLIFSNN", "NinaProSNN"]
