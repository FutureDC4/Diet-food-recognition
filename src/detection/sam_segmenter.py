"""SAM（Segment Anything）分割器 —— 可选精度升级。

用途：默认流程用 bbox 面积估重（快、误差中等）。启用 SAM 后，
对每个食物 bbox 做一次分割，用「掩膜像素数」代替 bbox 面积，
面积更接近真实食物轮廓，重量估计明显更准（代价：CPU 上每张图数秒）。

依赖：segment_anything（已安装 1.0）+ 官方 checkpoint sam_vit_h_4b8939.pth（2.4GB）
配置：config.yaml → segmentation.enabled

注意事项：
- 只在需要时 import，未安装 segment_anything 不会影响主流程
- 掩膜面积为「裁剪图（与原始像素同密度）」上的像素数，可直接用 px/cm 换算
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..utils.config_loader import get, resolve_path
from ..utils.logger import get_logger

log = get_logger("检测.SAM")


def _resolve_checkpoint(checkpoint: Optional[str | Path] = None) -> Path:
    if checkpoint:
        p = Path(checkpoint)
        if not p.is_absolute():
            p = resolve_path("paths.weights_dir", "models/weights").parent.parent / p
        if p.exists():
            return p
        raise FileNotFoundError(f"SAM checkpoint 不存在：{p}")

    raw = get("segmentation.sam_checkpoint", "../food_calories/sam_vit_h_4b8939.pth")
    p = Path(str(raw))
    if not p.is_absolute():
        from ..utils.config_loader import PROJECT_ROOT

        p = (PROJECT_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"SAM checkpoint 不存在：{p}\n"
            f"请下载 sam_vit_h_4b8939.pth（2.4GB）放到该路径，或在 config.yaml 里改 "
            f"segmentation.sam_checkpoint"
        )
    return p


class SamSegmenter:
    """SAM 封装：一次 set_image，多次框提示出掩膜。"""

    _instance: Optional["SamSegmenter"] = None

    def __init__(
        self,
        checkpoint: Optional[str | Path] = None,
        model_type: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from segment_anything import SamPredictor, sam_model_registry

        self.checkpoint = _resolve_checkpoint(checkpoint)
        self.model_type = str(model_type or get("segmentation.sam_model_type", "vit_h"))
        self.device = str(device or get("segmentation.device", get("detection.device", "cpu")))
        log.info(f"加载 SAM：{self.checkpoint.name}（{self.model_type}, {self.device}）—— 首次加载约 10-30 秒")
        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint))
        sam.to(self.device).eval()
        self.predictor = SamPredictor(sam)
        self._image_set = False
        log.info("SAM 就绪")

    # ------------------------------------------------------------------
    @classmethod
    def get(cls, **kwargs) -> "SamSegmenter":
        """进程内单例，避免重复加载 2.4GB 权重。"""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    # ------------------------------------------------------------------
    def set_image(self, image_bgr: np.ndarray) -> None:
        import cv2

        self.predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        self._image_set = True

    def mask_area_in_box(self, bbox: Sequence[float], min_ratio: float = 0.02) -> int:
        """用 bbox 作为提示，返回框内掩膜像素数（自动选得分最高的掩膜）。"""
        if not self._image_set:
            return 0
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        if masks is None or len(masks) == 0:
            return 0
        best = int(np.argmax(scores))
        mask = masks[best].astype(np.uint8)

        h, w = mask.shape[:2]
        bx1, by1 = max(int(x1), 0), max(int(y1), 0)
        bx2, by2 = min(int(x2), w - 1), min(int(y2), h - 1)
        if bx2 <= bx1 or by2 <= by1:
            return 0

        region = mask[by1:by2 + 1, bx1:bx2 + 1]
        count = int(np.count_nonzero(region))
        box_area = float((bx2 - bx1 + 1) * (by2 - by1 + 1))
        if box_area <= 0 or count / box_area < min_ratio:
            return 0
        return count

    # ------------------------------------------------------------------
    def refine(self, image_bgr: np.ndarray, detections: List[dict]) -> List[dict]:
        """为一组检测结果补充 `mask_area_px`（原地修改并返回）。"""
        if not detections:
            return detections
        self.set_image(image_bgr)
        for det in detections:
            area = self.mask_area_in_box(det["bbox"])
            if area > 0:
                det["mask_area_px"] = float(area)
                det["bbox_area_px"] = det.get("area_px")
        log.info(
            "SAM 精修面积：" + ", ".join(
                f"{d['name']} {d.get('mask_area_px', 0):.0f}/{d.get('area_px', 0):.0f}px²" for d in detections
            )
        )
        return detections


if __name__ == "__main__":
    import sys

    from ..utils.image_utils import load_image

    path = sys.argv[1] if len(sys.argv) > 1 else "data/images/test.jpg"
    img = load_image(path)
    seg = SamSegmenter.get()
    seg.set_image(img)
    print("框内掩膜面积：", seg.mask_area_in_box([100, 100, 500, 400]))
