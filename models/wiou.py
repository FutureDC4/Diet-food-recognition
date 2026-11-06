"""Wise-IoU (WIoU) 损失函数

论文: Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism
      (https://arxiv.org/abs/2301.10051)

背景:
    传统 CIoU 对所有样本一视同仁，低质量样本（远离目标的预测框）会产生
    巨大且有害的梯度。WIoU 用「动态非单调聚焦机制」：
      - 质量好的样本给更高权重（放大有效梯度）
      - 离群样本（outlier）按 β = L_IoU / L_IoU_mean 动态降权，避免有害梯度

三个版本:
    v1 (monotonous=None)  : L = R_WIoU * L_IoU               固定权重
    v2 (monotonous=True)  : L = sqrt(β) * R_WIoU * L_IoU     单调聚焦
    v3 (monotonous=False) : L = α·γ^(β-δ) 反向权重           非单调聚焦（默认）

⚠ 当前仅「代码就位」，暂不启用：等中餐数据集准备好后，替换 YOLO 默认的
   BboxLoss(CIoU) 再训练。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class WIoULoss(nn.Module):
    """WIoU 损失。

    参数
    ----
    monotonous : None | True | False
        None -> v1；True -> v2（单调）；False -> v3（非单调，推荐）
    momentum : float
        滑动平均动量，用于估计 L_IoU 的均值（默认按 7000 步的 EMA）
    gamma : float
        聚焦系数 γ（v3 用）
    delta : float
        阈值 δ（v3 用）
    reduction : str
        "mean" | "sum" | "none"
    eps : float
        数值稳定项

    forward(pred, target)
    ---------------------
    pred, target : Tensor (N, 4)，xyxy 像素坐标（同 YOLO 的 BboxLoss 输入格式）
    返回：标量 loss（reduction="none" 时返回逐框 loss）
    """

    # 类级状态：整个训练过程共享 L_IoU 均值（WIoU 论文的 running mean 做法）
    iou_mean: float = 1.0
    _momentum: float = 1 - 0.5 ** (1 / 7000)
    _updating: bool = True

    def __init__(
        self,
        monotonous: Optional[bool] = False,
        momentum: float = 1 - 0.5 ** (1 / 7000),
        gamma: float = 1.9,
        delta: float = 3.0,
        reduction: str = "mean",
        eps: float = 1e-7,
    ):
        super().__init__()
        self.monotonous = monotonous
        self.momentum = momentum
        self.gamma = gamma
        self.delta = delta
        self.reduction = reduction
        self.eps = eps
        WIoULoss._momentum = momentum

    # ---------------- 基础几何量 ----------------
    @staticmethod
    def _iou_and_geometry(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
        """返回 IoU 与 WIoU 需要的几何量（中心距平方、最小外接框宽高）。"""
        b1_x1, b1_y1, b1_x2, b1_y2 = pred.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target.unbind(-1)

        inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0)
        inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
        inter = inter_w * inter_h

        area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
        area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
        union = area1 + area2 - inter
        iou = inter / (union + eps)

        # 中心点距离平方（ρ²）
        center_dist = (b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 / 4 + (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2 / 4
        # 最小外接框宽高（Wg, Hg）
        cw = (torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)).clamp(min=eps)
        ch = (torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)).clamp(min=eps)
        return iou, center_dist, cw, ch

    def _focus_weight(self, loss_iou: torch.Tensor) -> torch.Tensor:
        """动态聚焦权重 r：β 越小（质量越好）权重越高；离群点按版本反向压权。"""
        if self.monotonous is None:  # v1：不聚焦
            return torch.ones_like(loss_iou)

        beta = loss_iou.detach() / (WIoULoss.iou_mean + self.eps)
        if self.monotonous:  # v2：单调，sqrt(β)
            return beta.sqrt()
        # v3：非单调，β/ (δ·γ^(β-δ)) —— β 远大于 δ 时权重急剧下降
        alpha = self.delta * torch.pow(self.gamma, beta - self.delta)
        return beta / alpha

    # ---------------- 前向 ----------------
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.sum() * 0.0

        iou, center_dist, cw, ch = self._iou_and_geometry(pred, target, self.eps)
        loss_iou = 1.0 - iou

        # R_WIoU：尺度不变的距离权重。梯度分离设计 —— 距离项与 Wg/Hg 全部 detach，
        # 不参与反向传播，避免「拉大框」这种捷径解。
        # 指数上界 1.5（r ≤ 4.48）：训练早期预测框可能离目标很远，比值能到 50+，
        # 那样 exp 会放大到 1e20 量级，把整条 box_loss 顶到上千、梯度爆掉。
        # 1.5 对应「最难的样本最多放大 4.5 倍」，与 CIoU 的量级可比。
        with torch.no_grad():
            r_wiou = torch.exp(
                (center_dist / (cw ** 2 + ch ** 2).clamp(min=self.eps)).clamp(max=1.5)
            )

        # 更新 L_IoU 均值（训练期用，推理/评估期不要更新）
        if self.training and WIoULoss._updating:
            WIoULoss.iou_mean = (
                (1 - self.momentum) * WIoULoss.iou_mean
                + self.momentum * float(loss_iou.detach().mean().item())
            )
            WIoULoss.iou_mean = max(WIoULoss.iou_mean, 1e-3)

        loss = self._focus_weight(loss_iou) * r_wiou * loss_iou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    # ---------------- 工具 ----------------
    @classmethod
    def reset_running_mean(cls, value: float = 1.0) -> None:
        """重置 L_IoU 滑动均值（换数据集/重新训练时调用）。"""
        cls.iou_mean = value

    @classmethod
    def set_updating(cls, flag: bool) -> None:
        """是否允许更新滑动均值（验证集评估时建议关闭）。"""
        cls._updating = flag


if __name__ == "__main__":
    torch.manual_seed(0)
    pred = torch.tensor([[10.0, 10.0, 60.0, 60.0], [0.0, 0.0, 30.0, 30.0]])
    target = torch.tensor([[12.0, 12.0, 58.0, 58.0], [100.0, 100.0, 130.0, 130.0]])
    for mode, name in [(None, "v1"), (True, "v2"), (False, "v3")]:
        loss_fn = WIoULoss(monotonous=mode)
        print(f"WIoU-{name}: loss = {float(loss_fn(pred, target)):.4f}")
    print(f"参考：同输入下 1-IoU 均值 = {float((1 - WIoULoss._iou_and_geometry(pred, target)[0]).mean()):.4f}")
