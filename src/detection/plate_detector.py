"""盘子 + 餐具 OBB 检测（4 类：Fork / Knife / Plate / Spoon）。

模型来自 DidntRead/food_calories 的 yolo11m-obb-plate.pt（借用，不训练）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

import models  # noqa: F401  —— 注册 CBAM，为将来换成中餐 CBAM 模型预留

from ..utils.config_loader import get, resolve_path
from ..utils.device import is_cpu, resolve_device
from ..utils.image_utils import quad_center, quad_long_short_sides
from ..utils.logger import get_logger

log = get_logger("检测.盘子")


class PlateDetector:
    """封装盘子/餐具 OBB 检测。"""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO  # 延迟导入，加快非检测场景启动

        self.model_path = Path(model_path) if model_path else resolve_path("paths.plate_model")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"盘子检测模型不存在：{self.model_path}\n"
                f"请从 food_calories 复制 yolo11m-obb-plate.pt 到 models/weights/"
            )

        self.conf = float(conf if conf is not None else get("detection.plate_conf_threshold", 0.25))
        self.iou = float(iou if iou is not None else get("detection.iou_threshold", 0.45))
        self.imgsz = int(imgsz if imgsz is not None else get("detection.imgsz", 640))
        self.device = resolve_device(device if device is not None else get("detection.device", "cpu"))
        self.half = bool(get("detection.half", False)) and not is_cpu(self.device)
        self.max_det = int(get("detection.max_det", 30))

        log.info(
            f"加载盘子检测模型：{self.model_path.name}"
            f"（device={self.device}, half={self.half}, imgsz={self.imgsz}）"
        )
        self.model = YOLO(str(self.model_path))
        self.names = dict(self.model.names)
        log.info(f"盘子模型类别：{list(self.names.values())}")

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[dict]:
        """返回 [{name, conf, obb(4x2 ndarray), long_px, short_px, center}]。"""
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return []
        res = results[0]
        detections: List[dict] = []

        obb = getattr(res, "obb", None)
        if obb is None or obb.xyxyxyxy is None or len(obb) == 0:
            log.warning("未检测到任何盘子/餐具（模型输出为空）")
            return detections

        xyxyxyxy = obb.xyxyxyxy.cpu().numpy()          # (N, 4, 2)
        confs = obb.conf.cpu().numpy().reshape(-1)
        clss = obb.cls.cpu().numpy().astype(int).reshape(-1)

        for quad, cf, cid in zip(xyxyxyxy, confs, clss):
            name = self.names.get(int(cid), f"class_{int(cid)}")
            long_px, short_px = quad_long_short_sides(quad)
            cx, cy = quad_center(quad)
            detections.append(
                {
                    "name": name,
                    "conf": float(cf),
                    "obb": np.asarray(quad, dtype=np.float32),
                    "long_px": float(long_px),
                    "short_px": float(short_px),
                    "center": (cx, cy),
                }
            )

        detections.sort(key=lambda d: d["conf"], reverse=True)
        log.info("盘子/餐具检测到 " + ", ".join(f"{d['name']}({d['conf']:.2f})" for d in detections))
        return detections

    # ------------------------------------------------------------------
    @staticmethod
    def get_plate(detections: Sequence[dict], names: Sequence[str] = ("Plate", "plate", "PLATE")) -> Optional[dict]:
        """从检测结果中挑出盘子（取置信度最高的一个）。"""
        plates = [d for d in detections if d["name"] in names]
        return max(plates, key=lambda d: d["conf"]) if plates else None

    @property
    def class_names(self) -> List[str]:
        return list(self.names.values())


if __name__ == "__main__":
    import sys

    from ..utils.image_utils import load_image

    if len(sys.argv) < 2:
        raise SystemExit("用法: python -m src.detection.plate_detector <图片路径>")
    img = load_image(sys.argv[1])
    for d in PlateDetector().detect(img):
        print(f"{d['name']:8s} conf={d['conf']:.3f} long={d['long_px']:.1f}px")
