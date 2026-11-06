"""CBAM: Convolutional Block Attention Module (ECCV 2018)

论文: https://arxiv.org/abs/1807.06521

结构（串行）:
    x -> ChannelAttention -> SpatialAttention -> out

用途（本项目）:
    替换 YOLOv11 尾部的 C2PSA 模块，让模型在「通道维 + 空间维」上更关注食物主体、
    抑制盘边/桌布/餐具等背景干扰 —— 特别是中餐「一盘多菜、菜色相近」的场景。

⚠ 当前仅「代码就位」，暂不启用：等中餐数据集准备好后再训练使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力：全局平均池化 + 全局最大池化 -> 共享 MLP -> Sigmoid"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction, 8)  # 通道很少时避免降维到 0
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享 MLP 用 1x1 卷积实现（等价于两层 FC，但要保持 NCHW 形状）
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力：通道维的平均 + 最大 -> 7x7 卷积 -> Sigmoid"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size 只能是 3 或 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(feat))


class CBAM(nn.Module):
    """CBAM 主模块：先通道注意力，再空间注意力（保持通道数不变，可即插即用）"""

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


def register_to_ultralytics() -> bool:
    """把 CBAM 注册进 ultralytics 的模块命名空间。

    yolo11m-cbam.yaml 里写的是 `CBAM`，ultralytics 的 parse_model 会用
    globals()[m] 查找模块，因此必须先注入，否则会 KeyError: 'CBAM'。

    返回 True 表示注册成功，False 表示环境里没有 ultralytics（不影响纯 PyTorch 使用）。
    """
    try:
        import ultralytics.nn.tasks as tasks
    except Exception:  # pragma: no cover - 没装 ultralytics 时静默跳过
        return False
    tasks.CBAM = CBAM
    return True


if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)
    m = CBAM(64)
    y = m(x)
    print(f"输入 {tuple(x.shape)} -> 输出 {tuple(y.shape)}，参数量 {sum(p.numel() for p in m.parameters())}")
