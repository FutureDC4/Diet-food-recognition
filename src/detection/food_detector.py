"""食物检测（bbox）封装。

当前权重说明（重要）：
    仓库自带的 `yolo11m.pt` 实际是 **COCO-80 官方预训练权重**（不是 9 类食物模型），
    因此本模块新增「营养表白名单过滤」：只保留 food_reference.yaml 里存在的食物类别，
    像 person / chair / dining table 这类非食物/未收录类别会被丢弃并记录到 rejected。

    可选 SAM 精修：`sam=True` 时对每个食物框跑一次 SAM 分割，用掩膜像素数代替 bbox 面积，
    重量估计更准（CPU 上每张图多花几秒）。

    后续换成自训练的中餐模型时：只要把类别 key 追加到 food_reference.yaml，
    白名单会自动生效，代码无需改动。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import cv2
import numpy as np
import yaml

import models  # noqa: F401  —— 注册 CBAM，兼容将来用 CBAM 结构训练的中餐模型

from ..utils.config_loader import get, resolve_path
from ..utils.image_utils import extract_rotated_region
from ..utils.logger import get_logger

log = get_logger("检测.食物")


def _norm_name(name: str) -> str:
    """类别名归一化：小写 + 空格/连字符转下划线（'hot dog' -> 'hot_dog'）。"""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def load_food_whitelist(path: Optional[str | Path] = None) -> Set[str]:
    """从食物营养表读取「认可的食物类别」集合。"""
    p = Path(path) if path else resolve_path("paths.food_reference")
    if not p.exists():
        return set()
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {_norm_name(k) for k in data.keys() if str(k).lower() != "default"}


class FoodDetector:
    """封装食物 bbox 检测（+ 可选白名单过滤 + 可选 SAM 面积精修）。"""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
        min_area_ratio: float = 0.002,
        filter_unknown: Optional[bool] = None,
        allowed_classes: Optional[Iterable[str]] = None,
        use_sam: Optional[bool] = None,
    ):
        from ultralytics import YOLO

        self.model_path = Path(model_path) if model_path else resolve_path("paths.food_model")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"食物检测模型不存在：{self.model_path}\n"
                f"请从 food_calories 复制 yolo11m.pt 到 models/weights/"
            )

        self.conf = float(conf if conf is not None else get("detection.conf_threshold", 0.25))
        self.iou = float(iou if iou is not None else get("detection.iou_threshold", 0.45))
        self.imgsz = int(imgsz if imgsz is not None else get("detection.imgsz", 640))
        self.device = str(device if device is not None else get("detection.device", "cpu"))
        self.half = bool(get("detection.half", False))
        self.max_det = int(get("detection.max_det", 30))
        self.min_area_ratio = float(min_area_ratio)

        self.filter_unknown = bool(
            filter_unknown if filter_unknown is not None else get("detection.filter_unknown_food", True)
        )
        self.allowed: Set[str] = (
            {_norm_name(n) for n in allowed_classes} if allowed_classes is not None else load_food_whitelist()
        )
        self.use_sam = bool(use_sam if use_sam is not None else get("segmentation.enabled", False))
        self._segmenter = None
        self.rejected: List[dict] = []

        log.info(f"加载食物检测模型：{self.model_path.name}")
        self.model = YOLO(str(self.model_path))
        self.names = dict(self.model.names)
        log.info(
            f"食物模型共 {len(self.names)} 类；白名单过滤={'开' if self.filter_unknown else '关'}"
            f"（营养表收录 {len(self.allowed)} 类）"
        )
        if self.use_sam:
            log.info("SAM 面积精修：已启用（config.segmentation.enabled=true）")

    # ------------------------------------------------------------------
    @property
    def class_names(self) -> List[str]:
        return list(self.names.values())

    # ------------------------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        region_quad: Optional[Sequence[Sequence[float]]] = None,
        region_ratio: Optional[float] = None,
        fallback_full: bool = True,
        sam: Optional[bool] = None,
    ) -> dict:
        """检测食物。

        返回 {"detections": [...], "rejected": [...], "mode": "plate"|"full", "crop": ndarray|None}
        每个 detection：{name, conf, bbox[xyxy], area_px, image_area_px, mode, [mask_area_px]}
        """
        sam_on = self.use_sam if sam is None else bool(sam)

        if region_quad is not None:
            ratio = float(region_ratio if region_ratio is not None else get("plate.crop_shrink_ratio", 0.95))
            crop, matrix = extract_rotated_region(image, region_quad, ratio=ratio, return_matrix=True)
            dets = self._detect_on(crop, mode="plate", sam=sam_on)
            if dets:
                # 裁剪图是「摆正」过的独立坐标系，画框和对标注都必须映射回原图，
                # 否则框会整体偏向画面一角。像素密度 1:1，area_px 无需换算。
                self._restore_boxes(dets, matrix, image.shape[1], image.shape[0])
                return {"detections": dets, "rejected": self.rejected, "mode": "plate", "crop": crop}
            if not fallback_full:
                return {"detections": [], "rejected": self.rejected, "mode": "plate", "crop": crop}
            log.info("盘内未检到（白名单内的）食物，回退整图检测")

        dets = self._detect_on(image, mode="full", sam=sam_on)
        return {"detections": dets, "rejected": self.rejected, "mode": "full", "crop": None}

    # ------------------------------------------------------------------
    def _detect_on(self, image: np.ndarray, mode: str, sam: bool = False) -> List[dict]:
        self.rejected = []
        h, w = image.shape[:2]
        if h < 16 or w < 16:
            return []

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
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            log.info(f"食物检测（{mode}）：未检出")
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy().reshape(-1)
        clss = boxes.cls.cpu().numpy().astype(int).reshape(-1)
        image_area = float(h * w)

        detections: List[dict] = []
        for box, cf, cid in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            area_px = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            if image_area > 0 and area_px / image_area < self.min_area_ratio:
                continue
            name = self.names.get(int(cid), f"class_{int(cid)}")
            item = {
                "name": name,
                "conf": float(cf),
                "bbox": [x1, y1, x2, y2],
                "area_px": float(area_px),
                "image_area_px": image_area,
                "mode": mode,
            }
            if self.filter_unknown and _norm_name(name) not in self.allowed:
                self.rejected.append(item)
                continue
            detections.append(item)

        if self.rejected:
            top = sorted(self.rejected, key=lambda x: -x["conf"])[:6]
            log.info("白名单外丢弃：" + ", ".join(f"{d['name']}({d['conf']:.2f})" for d in top))

        detections.sort(key=lambda d: d["area_px"], reverse=True)

        if sam and detections:
            detections = self._refine_with_sam(image, detections)

        log.info(
            f"食物检测（{mode}）："
            + (", ".join(f"{d['name']}({d['conf']:.2f}, {self._area_of(d):.0f}px²)" for d in detections) or "无")
        )
        return detections

    # ------------------------------------------------------------------
    @staticmethod
    def _restore_boxes(detections: List[dict], matrix: np.ndarray,
                       width: int, height: int) -> None:
        """把裁剪图坐标的 bbox 映射回原图（原地修改）。

        area_px / mask_area_px 是裁剪图上的真实面积（透视变换为 1:1），保持不变；
        旋转后的轴对齐外接框会比原框大，但只影响画框，不参与估重。
        """
        try:
            inv = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
        except np.linalg.LinAlgError:
            return
        for det in detections:
            box = det.get("bbox")
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
            pts = cv2.perspectiveTransform(corners, inv.astype(np.float32))[0]
            det["bbox_crop"] = [x1, y1, x2, y2]
            det["bbox"] = [
                float(np.clip(pts[:, 0].min(), 0, width - 1)),
                float(np.clip(pts[:, 1].min(), 0, height - 1)),
                float(np.clip(pts[:, 0].max(), 0, width - 1)),
                float(np.clip(pts[:, 1].max(), 0, height - 1)),
            ]

    # ------------------------------------------------------------------
    @staticmethod
    def _area_of(det: dict) -> float:
        return float(det.get("mask_area_px") or det.get("area_px") or 0.0)

    def _refine_with_sam(self, image: np.ndarray, detections: List[dict]) -> List[dict]:
        try:
            from .sam_segmenter import SamSegmenter

            if self._segmenter is None:
                self._segmenter = SamSegmenter.get(device=self.device)
            self._segmenter.refine(image, detections)
        except Exception as exc:  # SAM 失败不影响主流程
            log.warning(f"SAM 精修失败，回退 bbox 面积：{exc.__class__.__name__}: {exc}")
        return detections

    # ------------------------------------------------------------------
    @staticmethod
    def _union_area(boxes: Sequence[Sequence[float]], scale: float = 8.0) -> float:
        """多个 bbox 的并集面积（栅格近似）。

        同一盘菜常被重复检出多个互相重叠的框，直接把面积相加会严重高估；
        这里用 1/scale 分辨率的栅格求并集，重叠部分只算一次。
        """
        if not boxes:
            return 0.0
        xs = [float(b[0]) for b in boxes] + [float(b[2]) for b in boxes]
        ys = [float(b[1]) for b in boxes] + [float(b[3]) for b in boxes]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        gw = max(int((x1 - x0) / scale) + 1, 1)
        gh = max(int((y1 - y0) / scale) + 1, 1)
        if gw * gh > 4_000_000:      # 极端情况保护，避免大内存
            return float(sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes))
        grid = np.zeros((gh, gw), dtype=bool)
        for b in boxes:
            c1 = int((float(b[0]) - x0) / scale)
            c2 = max(int((float(b[2]) - x0) / scale), c1 + 1)
            r1 = int((float(b[1]) - y0) / scale)
            r2 = max(int((float(b[3]) - y0) / scale), r1 + 1)
            grid[r1:min(r2, gh), c1:min(c2, gw)] = True
        return float(grid.sum()) * (scale ** 2)

    @staticmethod
    def merged_detections(detections: Sequence[dict]) -> List[dict]:
        """同类别的多个框合并：面积取并集（避免重叠重复计数），置信度取最大。"""
        grouped: dict[str, dict] = {}
        for det in detections:
            key = _norm_name(det["name"])
            if key not in grouped:
                grouped[key] = {
                    "name": det["name"],
                    "conf": det["conf"],
                    "boxes": [list(det["bbox"])],
                    "area_px": det["area_px"],
                    "mask_area_px": det.get("mask_area_px"),
                    "count": 1,
                }
            else:
                g = grouped[key]
                g["conf"] = max(g["conf"], det["conf"])
                g["boxes"].append(list(det["bbox"]))
                g["count"] += 1
                if det.get("mask_area_px"):
                    g["mask_area_px"] = (g.get("mask_area_px") or 0) + det["mask_area_px"]

        merged: List[dict] = []
        for g in grouped.values():
            b1, b2 = g["boxes"][0], g["boxes"][-1]
            if len(g["boxes"]) > 1:
                area = FoodDetector._union_area(g["boxes"])
                union_bbox = [
                    min(b[0] for b in g["boxes"]), min(b[1] for b in g["boxes"]),
                    max(b[2] for b in g["boxes"]), max(b[3] for b in g["boxes"]),
                ]
            else:
                area = float(g["area_px"])
                union_bbox = list(b1)
            merged.append(
                {
                    "name": g["name"],
                    "conf": g["conf"],
                    "area_px": float(area),
                    "mask_area_px": g.get("mask_area_px"),
                    "count": g["count"],
                    "bbox": union_bbox,
                }
            )
        return sorted(merged, key=lambda d: d["area_px"], reverse=True)


if __name__ == "__main__":
    import sys

    from ..utils.image_utils import load_image
    from .plate_detector import PlateDetector

    path = sys.argv[1] if len(sys.argv) > 1 else "data/images/test.jpg"
    img = load_image(path)
    plates = PlateDetector().detect(img)
    plate = PlateDetector.get_plate(plates)
    out = FoodDetector().detect(img, region_quad=plate["obb"] if plate else None)
    print("mode:", out["mode"])
    for d in out["detections"]:
        print(f"  {d['name']:12s} conf={d['conf']:.3f} area={d['area_px']:.0f}px²")
    if out["rejected"]:
        print("被白名单丢弃：", [d["name"] for d in out["rejected"]])
