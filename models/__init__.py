"""models 包：模型结构定义与改进模块（CBAM / WIoU）。

导入本包时会尝试把 CBAM 注册进 ultralytics 的模块命名空间，
这样 `YOLO("models/yolo11m-cbam.yaml")` 才能解析到 CBAM 层。
（没装 ultralytics 时静默跳过，不影响纯 PyTorch 使用）
"""

from .cbam import CBAM, ChannelAttention, SpatialAttention, register_to_ultralytics
from .wiou import WIoULoss
from .wiou_patch import apply_wiou, applied_version

# 注册是幂等的，失败也不抛异常（例如 ultralytics 尚未安装）
REGISTERED = register_to_ultralytics()

__all__ = [
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "WIoULoss",
    "apply_wiou",
    "applied_version",
    "register_to_ultralytics",
    "REGISTERED",
]
