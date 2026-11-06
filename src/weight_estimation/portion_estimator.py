"""重量估计：bbox 面积 × 假设高度 × 密度（替代原项目的 SAM 分割方案）。

原项目：SAM 分割出食物像素 -> 占盘子面积百分比 -> 乘密度和高度
本项目：YOLO bbox 面积 -> 像素面积换算 cm² -> × 高度 × 密度
        （不下 2.4GB SAM checkpoint，精度略降但可视化/可解释性更好）

公式：
    实际面积 cm² = bbox 像素面积 / (px_per_cm)²
    实际面积 cm² *= area_fill_factor      # bbox 是外接矩形，食物一般填不满
    体积 cm³     = 面积 cm² × 高度 cm
    重量 g       = 体积 cm³ × 密度 g/cm³
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from ..utils.config_loader import get, resolve_path
from ..utils.logger import get_logger

log = get_logger("重量估计")


def load_density_table(path: Optional[str | Path] = None) -> Dict[str, float]:
    """读取密度表（g/cm³）。同时兼容密度表与营养参考表两处来源。"""
    table: Dict[str, float] = {}
    for key, default in (("paths.food_reference", "data/food_reference.yaml"),
                         ("paths.density_table", "data/density_table.yaml")):
        p = Path(path) if path else resolve_path(key, default)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for name, value in data.items():
            if isinstance(value, dict) and "density" in value:
                table.setdefault(str(name).lower(), float(value["density"]))
            elif isinstance(value, (int, float)):
                table.setdefault(str(name).lower(), float(value))
    return table


class PortionEstimator:
    """bbox → 重量。"""

    def __init__(
        self,
        density_table: Optional[Dict[str, float]] = None,
        default_height_cm: Optional[float] = None,
        area_fill_factor: Optional[float] = None,
        min_weight_g: Optional[float] = None,
        max_weight_g: Optional[float] = None,
        height_by_food: Optional[Dict[str, float]] = None,
        mask_fill_factor: Optional[float] = None,
    ):
        self.density_table = density_table if density_table is not None else load_density_table()
        self.default_height_cm = float(
            default_height_cm if default_height_cm is not None else get("weight_estimation.default_height_cm", 3.0)
        )
        self.area_fill_factor = float(
            area_fill_factor if area_fill_factor is not None else get("weight_estimation.area_fill_factor", 0.80)
        )
        # 有 SAM 掩膜时面积几乎就是真实轮廓，填充系数取接近 1
        self.mask_fill_factor = float(
            mask_fill_factor if mask_fill_factor is not None else get("segmentation.mask_fill_factor", 0.95)
        )
        self.min_weight_g = float(
            min_weight_g if min_weight_g is not None else get("weight_estimation.min_weight_g", 10.0)
        )
        self.max_weight_g = float(
            max_weight_g if max_weight_g is not None else get("weight_estimation.max_weight_g", 1500.0)
        )
        cfg_heights = get("weight_estimation.height_by_food", {}) or {}
        self.height_by_food = {
            **{str(k).lower(): float(v) for k, v in cfg_heights.items() if isinstance(v, (int, float))},
            **(height_by_food or {}),
        }
        # 单份重量上限（按食物覆写）：防止 COCO 误检/大框把一份食物算成 1kg+
        cfg_max = get("weight_estimation.max_weight_by_food", {}) or {}
        self.max_weight_by_food = {
            str(k).lower().replace(" ", "_"): float(v) for k, v in cfg_max.items() if isinstance(v, (int, float))
        }
        self.max_plate_fill = float(get("weight_estimation.max_plate_fill_ratio", 0.85))


    # ------------------------------------------------------------------
    def density_of(self, food_name: str) -> float:
        """查密度：精确 -> 归一化 -> 默认 0.80。"""
        key = str(food_name).strip().lower()
        if key in self.density_table:
            return float(self.density_table[key])
        norm = key.replace(" ", "_").replace("-", "_")
        if norm in self.density_table:
            return float(self.density_table[norm])
        return float(self.density_table.get("default", 0.80))

    def height_of(self, food_name: str, override: Optional[float] = None) -> float:
        if override is not None:
            return float(override)
        key = str(food_name).strip().lower()
        if key in self.height_by_food and key != "default":
            return float(self.height_by_food[key])
        return self.default_height_cm

    def max_weight_for(self, food_name: str) -> float:
        key = str(food_name).strip().lower().replace(" ", "_").replace("-", "_")
        return float(self.max_weight_by_food.get(key, self.max_weight_g))

    # ------------------------------------------------------------------
    def estimate(
        self,
        px_per_cm: float,
        food_detections: Sequence[dict],
        default_height_cm: Optional[float] = None,
        scale_reliable: bool = True,
        plate_area_cm2: Optional[float] = None,
    ) -> List[dict]:
        """对每个检测框估算重量。

        返回 [{name, conf, bbox, area_px, area_cm2, height_cm, volume_cm3, density, weight_g, ...}]

        plate_area_cm2：盘子面积（cm²）。给出时会做物理约束：
            所有食物面积之和不超过盘子面积的 max_plate_fill 倍，超出按比例缩小。
        """
        if px_per_cm is None or px_per_cm <= 0:
            raise ValueError(f"无效的尺度：{px_per_cm}")

        # ---- 第一遍：面积换算（cm²）----
        raw: List[dict] = []
        for det in food_detections:
            name = str(det.get("name", "unknown"))
            use_mask = bool(det.get("mask_area_px"))
            area_px = float(
                det.get("mask_area_px") or det.get("area_px") or self._bbox_area(det.get("bbox"))
            )
            if area_px <= 0:
                continue
            fill = self.mask_fill_factor if use_mask else self.area_fill_factor
            raw.append(
                {
                    "det": det,
                    "name": name,
                    "use_mask": use_mask,
                    "area_px": area_px,
                    "area_cm2": area_px / (px_per_cm ** 2) * fill,
                }
            )

        # ---- 物理约束：所有食物面积之和 ≤ 盘子面积 × max_plate_fill ----
        shrink = 1.0
        if plate_area_cm2 and plate_area_cm2 > 0 and raw:
            total_area = sum(r["area_cm2"] for r in raw)
            limit = float(plate_area_cm2) * self.max_plate_fill
            if total_area > limit > 0:
                shrink = limit / total_area
                log.info(
                    f"食物总面积 {total_area:.0f} cm² 超过盘子可用面积 {limit:.0f} cm²，"
                    f"按 {shrink:.2f} 比例整体收缩"
                )
                for r in raw:
                    r["area_cm2"] *= shrink

        # ---- 第二遍：体积 → 重量（含单份上限）----
        results: List[dict] = []
        for r in raw:
            det, name = r["det"], r["name"]
            area_cm2 = r["area_cm2"]
            height_cm = self.height_of(name, default_height_cm or det.get("height_cm"))
            volume_cm3 = area_cm2 * height_cm
            density = self.density_of(name)
            weight_g = volume_cm3 * density

            item_max = self.max_weight_for(name)
            capped = False
            if weight_g > item_max:
                weight_g, capped = item_max, True
            elif weight_g < self.min_weight_g:
                weight_g, capped = self.min_weight_g, True

            results.append(
                {
                    "name": name,
                    "conf": float(det.get("conf", 0.0)),
                    "bbox": det.get("bbox"),
                    "count": int(det.get("count", 1)),
                    "area_px": r["area_px"],
                    "area_source": "sam_mask" if r["use_mask"] else "bbox",
                    "area_cm2": round(area_cm2, 2),
                    "height_cm": round(height_cm, 2),
                    "volume_cm3": round(volume_cm3, 2),
                    "density": round(density, 3),
                    "weight_g": round(weight_g, 1),
                    "capped": capped,
                    "area_shrink": round(shrink, 3),
                    "scale_reliable": bool(scale_reliable),
                }
            )

        results.sort(key=lambda d: d["weight_g"], reverse=True)
        total = sum(d["weight_g"] for d in results)
        log.info(
            f"重量估计（{px_per_cm:.2f} px/cm）："
            + ("、".join(f"{d['name']} {d['weight_g']:.0f}g" for d in results) if results else "无食物")
            + f"；合计 {total:.0f}g"
        )
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_area(bbox: Optional[Sequence[float]]) -> float:
        if not bbox or len(bbox) < 4:
            return 0.0
        x1, y1, x2, y2 = bbox[:4]
        return max(float(x2) - float(x1), 0.0) * max(float(y2) - float(y1), 0.0)

    @staticmethod
    def total_weight(items: Sequence[dict]) -> float:
        return round(sum(float(i.get("weight_g", 0.0)) for i in items), 1)


if __name__ == "__main__":
    table = load_density_table()
    print(f"密度表条目数：{len(table)}，示例：{list(table.items())[:3]}")
    est = PortionEstimator()
    demo = [{"name": "rice", "conf": 0.92, "bbox": [100, 100, 300, 250], "area_px": 30000}]
    print(est.estimate(px_per_cm=12.0, food_detections=demo))
