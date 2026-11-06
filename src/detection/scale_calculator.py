"""尺度计算：由参考物（叉/刀/勺/盘）的像素尺寸推算 像素/厘米 比例。

思路与 DidntRead 的 calculate_scale_obb 一致：OBB 最长边(px) / 真实尺寸(cm)，
多个参考物取中位数（抗单点误检）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..utils.config_loader import get, resolve_path
from ..utils.logger import get_logger

log = get_logger("检测.尺度")


def load_reference_sizes(path: Optional[str | Path] = None) -> Dict[str, float]:
    """读取参考物真实尺寸表（cm）。"""
    p = Path(path) if path else resolve_path("paths.reference_sizes")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        sizes = {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}
        if sizes:
            return sizes
        log.warning(f"参考尺寸表为空，使用配置内默认值：{p}")
    return {str(k): float(v) for k, v in (get("reference_sizes") or {}).items()}


class ScaleCalculator:
    """把像素换算成厘米（以及平方厘米）。"""

    def __init__(
        self,
        ref_sizes: Optional[Dict[str, float]] = None,
        fallback_image_width_cm: Optional[float] = None,
    ):
        self.ref_sizes = ref_sizes or load_reference_sizes()
        self.fallback_image_width_cm = float(
            fallback_image_width_cm
            if fallback_image_width_cm is not None
            else get("plate.fallback_image_width_cm", 40.0)
        )

    # ------------------------------------------------------------------
    def calculate_scale(
        self,
        detections: Sequence[dict],
        image_width: Optional[int] = None,
    ) -> dict:
        """返回 {px_per_cm, source, samples, reliable}。

        样本来源优先级：餐具 > 盘子（餐具尺寸更标准，盘子大小差异大）。
        """
        samples: List[Tuple[str, float, float]] = []   # (名称, 像素长, 实际cm)
        for det in detections:
            name = str(det.get("name", ""))
            if name not in self.ref_sizes:
                continue
            long_px = det.get("long_px")
            if long_px is None:
                obb = det.get("obb")
                if obb is None:
                    continue
                pts = np.asarray(obb, dtype=np.float32).reshape(4, 2)
                sides = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
                long_px = max(sides)
            if not long_px or long_px <= 1:
                continue
            samples.append((name, float(long_px), self.ref_sizes[name]))

        if samples:
            scales = [px / cm for _, px, cm in samples if cm > 0]
            px_per_cm = float(np.median(scales))
            utensils = [s for s in samples if s[0] != "Plate"]
            source = "餐具" if utensils else "盘子"
            reliable = True
            detail = ", ".join(f"{n}:{px / cm:.2f}px/cm" for n, px, cm in samples)
            log.info(f"尺度计算（{source}，{len(samples)} 个参考物）→ {px_per_cm:.2f} px/cm [{detail}]")
            return {
                "px_per_cm": px_per_cm,
                "source": source,
                "samples": [{"name": n, "long_px": px, "real_cm": cm, "px_per_cm": px / cm} for n, px, cm in samples],
                "reliable": reliable,
            }

        # 兜底：一个参考物都没有
        if image_width:
            px_per_cm = float(image_width) / self.fallback_image_width_cm
            log.warning(
                f"未找到任何参考物（叉/刀/勺/盘），按画面宽度≈{self.fallback_image_width_cm}cm 估算 "
                f"{px_per_cm:.2f} px/cm —— 结果误差较大"
            )
            return {"px_per_cm": px_per_cm, "source": "兜底（画面宽度假设）", "samples": [], "reliable": False}

        raise RuntimeError("未找到可用参考物，且无法获得图像宽度，无法估算尺度")

    # ------------------------------------------------------------------
    def px_area_to_cm2(self, area_px: float, px_per_cm: float) -> float:
        if px_per_cm <= 0:
            raise ValueError(f"尺度非法：{px_per_cm}")
        return float(area_px) / (px_per_cm ** 2)

    def px_to_cm(self, px: float, px_per_cm: float) -> float:
        if px_per_cm <= 0:
            raise ValueError(f"尺度非法：{px_per_cm}")
        return float(px) / px_per_cm


if __name__ == "__main__":
    import sys

    from ..utils.image_utils import load_image, image_size
    from .plate_detector import PlateDetector

    if len(sys.argv) < 2:
        raise SystemExit("用法: python -m src.detection.scale_calculator <图片路径>")
    img = load_image(sys.argv[1])
    dets = PlateDetector().detect(img)
    w, _ = image_size(img)
    print(ScaleCalculator().calculate_scale(dets, image_width=w))
