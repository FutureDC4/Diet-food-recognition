"""把 WIoU 接进 ultralytics 的训练循环（猴补丁）。

ultralytics 的检测损失是 v8DetectionLoss，其中 bbox 回归项由 BboxLoss.forward
用 CIoU 计算。要换成 WIoU 而不改 site-packages，做法是：

    BboxLoss.__init__   —— 额外挂一个 WIoULoss 实例
    BboxLoss.forward    —— IoU 项换成 WIoU，DFL 项保持官方实现

因为 v8DetectionLoss 内部是 `self.bbox_loss = BboxLoss(reg_max)`，
补丁打在类上，任何新建的损失实例都会自动用 WIoU。

用法（训练脚本已封装 --wiou）：
    import models
    from models.wiou_patch import apply_wiou
    apply_wiou("v3")
"""

from __future__ import annotations

import os

import torch

from .wiou import WIoULoss

_APPLIED: str | None = None
_WARMUP_STEPS: int = 0
_STEP: int = 0
_DEBUG: bool = os.getenv("WIOU_DEBUG", "") == "1"
_DBG_N: int = 0


def _wiou_bbox_forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
):
    """替换 BboxLoss.forward：IoU 项用 WIoU，其余（归一化方式、DFL）与官方一致。

    前 _WARMUP_STEPS 次前向走官方 CIoU：结构被改过（CBAM 换了 C2PSA）的模型
    在最初几十步里预测框还很乱，WIoU 的距离权重会把这类离群框放大，先让
    回归头用 CIoU 站稳，再切 WIoU 更稳。
    """
    global _STEP, _DBG_N
    from ultralytics.utils.loss import bbox2dist  # 延迟导入，避免模型文件强依赖 ultralytics

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    warmup = self.training and _STEP < _WARMUP_STEPS
    if self.training:
        _STEP += 1

    if int(fg_mask.sum()) > 0:
        if warmup:
            from ultralytics.utils.metrics import bbox_iou

            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        else:
            self.wiou.train(self.training)
            # 注意：WIoULoss 返回 (N,) 的一维逐框损失，官方 weight 是 (N,1)，
            # 直接相乘会被广播成 (N,N)，把损失放大 N 倍（实测 box_loss 会飙到 2000+）。
            # 统一压平成一维再乘。
            per_box = self.wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask]).reshape(-1)
            w_flat = weight.reshape(-1)
            loss_iou = (per_box * w_flat).sum() / target_scores_sum
            if _DEBUG:
                if _DBG_N < 4:
                    _DBG_N += 1
                    iou, cd, cw, ch = self.wiou._iou_and_geometry(
                        pred_bboxes[fg_mask], target_bboxes[fg_mask])
                    ratio = cd / (cw ** 2 + ch ** 2).clamp(min=1e-7)
                    print(f"[WIoU-dbg] fg={int(fg_mask.sum())} tss={float(target_scores_sum):.3f} "
                          f"w_sum={float(weight.sum()):.3f} w_max={float(weight.max()):.3f}")
                    print(f"[WIoU-dbg] per_box min/max/mean={float(per_box.min()):.4f}/"
                          f"{float(per_box.max()):.4f}/{float(per_box.mean()):.4f} "
                          f"iou_mean={self.wiou.iou_mean:.5f} loss_iou={float(loss_iou):.4f}")
                    print(f"[WIoU-dbg] ratio 中位/90%/max={float(ratio.median()):.3f}/"
                          f"{float(ratio.quantile(0.9)):.3f}/{float(ratio.max()):.3f} "
                          f"iou min/mean={float(iou.min()):.4f}/{float(iou.mean()):.4f}")
    else:
        loss_iou = pred_bboxes.sum() * 0.0

    if getattr(self, "dfl_loss", None) is not None:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(
            pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
        ) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        loss_dfl = torch.tensor(0.0).to(pred_dist.device)

    return loss_iou, loss_dfl


def apply_wiou(version: str = "v3", warmup_steps: int = 0) -> str:
    """启用 WIoU 回归损失。

    version      : v1（固定权重）/ v2（单调聚焦）/ v3（非单调聚焦，推荐）
    warmup_steps : 前多少个训练步先用 CIoU 热身（结构改动过的模型建议 300 左右）
    返回实际生效的版本名；重复调用是幂等的。
    """
    global _APPLIED, _WARMUP_STEPS, _STEP
    version = str(version).lower().strip()
    if version in ("none", "off", ""):
        return "none"

    monotonous = {"v1": None, "v2": True, "v3": False}.get(version)
    if version not in ("v1", "v2", "v3"):
        raise ValueError(f"WIoU 版本只能是 v1/v2/v3，收到: {version}")

    from ultralytics.utils import loss as uloss

    _WARMUP_STEPS = int(warmup_steps)
    _STEP = 0
    if _APPLIED == version and _WARMUP_STEPS == int(warmup_steps):
        return version

    original_init = uloss.BboxLoss.__init__

    def patched_init(self, reg_max: int = 16):
        original_init(self, reg_max)
        self.wiou = WIoULoss(monotonous=monotonous, reduction="none")

    uloss.BboxLoss.__init__ = patched_init
    uloss.BboxLoss.forward = _wiou_bbox_forward
    WIoULoss.reset_running_mean(1.0)
    _APPLIED = version
    return version


def applied_version() -> str | None:
    return _APPLIED
