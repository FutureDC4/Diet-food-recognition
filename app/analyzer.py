# -*- coding: utf-8 -*-
"""上位机的分析引擎：一次调用完成 检测 → 估重 → 营养 → 配额/结构评估。

与 main.py 的区别：
  - 不打印、不交互，只返回结构化结果给界面；
  - 模型只加载一次（界面连续分析多张图时不必反复加载权重）；
  - 增加「米饭三级处理」：米饭被菜覆盖时框会明显偏小，按面积占比分三级处理
    （≥20% 用检测值 / 4%-20% 按占食物面积 1/3 反推 / <4% 判为误检丢弃）；
  - 增加「食物高度档位」：单图无法判断堆叠高度，由用户在界面上选薄摊/常规/堆高；
  - 增加「餐次配额 + 结构评估」：把数值判断全部在这里算死，再交给大模型组织语言。
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import FoodDetector, PlateDetector, ScaleCalculator  # noqa: E402
from src.health import BMRCalculator  # noqa: E402
from src.utils.config_loader import get  # noqa: E402
from src.utils.food_labels import zh_name  # noqa: E402
from src.utils.image_utils import load_image  # noqa: E402
from src.weight_estimation import NutritionCalculator, PortionEstimator  # noqa: E402

from .calibration import calibrate  # noqa: E402
from .detection_rules import (  # noqa: E402
    DEFAULT_HEIGHT_LEVEL, HEIGHT_LEVELS, apply_rice_rule, height_multiplier,
)
from .diet_rules import build_budget  # noqa: E402
from .render import draw_annotations  # noqa: E402

MEAL_TYPES = ("早餐", "午餐", "晚餐", "加餐")


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


class MealAnalyzer:
    """带缓存的单图分析器。"""

    def __init__(self, use_sam: bool = False):
        self.use_sam = use_sam
        self._plate_det: Optional[PlateDetector] = None
        self._food_det: Optional[FoodDetector] = None
        self._scale = ScaleCalculator()
        self._portion = PortionEstimator()
        self._nutrition = NutritionCalculator()
        self._bmr = BMRCalculator()

    # ------------------------------------------------------------------
    def _detectors(self) -> tuple[PlateDetector, FoodDetector]:
        if self._plate_det is None:
            self._plate_det = PlateDetector()
        if self._food_det is None:
            self._food_det = FoodDetector()
        return self._plate_det, self._food_det

    def warmup(self) -> None:
        """预加载两个模型，避免界面上第一次点击时卡好几秒。"""
        self._detectors()

    # ------------------------------------------------------------------
    def analyze(
        self,
        image_path: str | Path,
        user: Dict[str, Any],
        meal_type: str = "午餐",
        rice_rule: bool = True,
        height_level: str = DEFAULT_HEIGHT_LEVEL,
    ) -> Dict[str, Any]:
        """完整分析一张图。

        height_level: 薄摊 / 常规 / 堆高，影响体积→重量的高度系数。
        """
        t0 = time.time()
        warnings: List[str] = []

        image = load_image(str(image_path))
        h, w = image.shape[:2]

        plate_det, food_det = self._detectors()

        # ---------- 1. 盘子与尺度 ----------
        plate_dets = plate_det.detect(image)
        plate = PlateDetector.get_plate(plate_dets)
        scale_info = self._scale.calculate_scale(plate_dets, image_width=w)
        px_per_cm = float(scale_info["px_per_cm"])
        if not plate:
            warnings.append("未检测到餐盘，重量估计误差会偏大")
        if not scale_info.get("reliable", True):
            warnings.append("缺少餐具/餐盘参考，尺度按画面宽度估算")

        # ---------- 2. 食物检测 ----------
        # 默认全图检测（与训练/评估一致）；盘内裁剪会让框偏大且切掉盘边食物，
        # 实测召回低 25 个点，所以盘子只用来算尺度。
        region_quad = None
        if plate is not None and bool(get("plate.crop_for_detection", False)):
            region_quad = plate["obb"]
        food_out = food_det.detect(image, region_quad=region_quad, sam=self.use_sam)
        raw_dets = food_out["detections"]
        merged = food_det.merged_detections(raw_dets)

        # ---------- 3. 米饭三级处理 ----------
        rice_info: Dict[str, Any] = {"applied": False, "level": 0, "note": "已关闭"}
        rice_on = bool(rice_rule) and bool(get("rice_rule.enabled", True))
        if rice_on:
            merged, rice_info = apply_rice_rule(
                merged,
                trust_ratio=float(get("rice_rule.trust_ratio", 0.20)),
                reject_ratio=float(get("rice_rule.reject_ratio", 0.04)),
                target_ratio=float(get("rice_rule.target_ratio", 1.0 / 3.0)),
            )

        # ---------- 4. 高度档位 ----------
        mult, level_name = height_multiplier(height_level)

        # ---------- 5. 重量 / 热量 ----------
        plate_area_cm2 = None
        if plate:
            diameter_cm = plate["long_px"] / px_per_cm
            plate_area_cm2 = math.pi * (diameter_cm / 2.0) ** 2

        weight_items = self._portion.estimate(
            px_per_cm=px_per_cm,
            food_detections=merged,
            scale_reliable=bool(scale_info.get("reliable", True)),
            plate_area_cm2=plate_area_cm2,
        )

        # 高度档位：在 PortionEstimator 查到的「按食物类型的默认高度」上乘系数
        if abs(mult - 1.0) > 1e-6:
            for item in weight_items:
                base_h = float(item.get("height_cm", 0) or 0)
                item["height_cm"] = round(base_h * mult, 2)
                item["weight_g"] = round(float(item["weight_g"]) * mult, 1)
                item["height_level"] = level_name
        for item in weight_items:
            item.setdefault("height_level", level_name)
            item["height_multiplier"] = mult

        # 米饭规则反推的面积会随「其他菜（偏大的）标注框」一起放大，
        # 这里按现实单份上限收一次，避免出现 400g 一碗饭这种数字。
        rice_cap = float(get("weight_estimation.rice_rule_max_g", 250) or 0)
        if rice_info.get("applied") and rice_cap > 0:
            for item in weight_items:
                if _norm(item.get("name", "")).startswith("rice") and item["weight_g"] > rice_cap:
                    item["weight_g"] = rice_cap
                    item["capped"] = True
                    item["rule_capped"] = True
                    rice_info["note"] += f"；再按单份上限收到 {rice_cap:.0f} g"

        dishes = self._nutrition.calculate_many(weight_items)
        for d in dishes:
            d["name_cn"] = zh_name(d["name"])

        # 装盘营养 → 入口营养：扣掉菜帮/盘底汤汁、表面浮油与随汁流失的盐。
        # 只做一次折算，参数在 config.yaml → nutrition_calibration，可整体关闭。
        dishes, calibration = calibrate(dishes)
        totals = self._nutrition.totals(dishes)

        # ---------- 6. 健康指标 ----------
        health = self._bmr.calculate(
            age=int(user["age"]),
            gender=str(user["gender"]),
            height_cm=float(user["height_cm"]),
            weight_kg=float(user["weight_kg"]),
            activity_level=int(user.get("activity_level", 1)),
        )
        meal_share = self._bmr.meal_share(totals["calories"], health["tdee"])

        # ---------- 7. 餐次配额 + 结构评估 + 该加什么 ----------
        try:
            budget = build_budget(dishes, totals, user, health, meal_type)
        except Exception as exc:                        # noqa: BLE001
            budget = {"error": f"{exc.__class__.__name__}: {exc}", "checks": [], "additions": []}
            warnings.append("膳食规则库加载失败，结构评估不可用")

        # ---------- 8. 标注图 ----------
        ratio_zoom = 1 if max(h, w) < 1400 else 2
        annotated = draw_annotations(image, dishes=dishes, plate_quad=region_quad, zoom=ratio_zoom)

        return {
            "image_path": str(image_path),
            "image": image,
            "annotated": annotated,
            "image_size": (w, h),
            "plate": plate,
            "plate_area_cm2": round(plate_area_cm2, 1) if plate_area_cm2 else None,
            "scale": scale_info,
            "px_per_cm": round(px_per_cm, 2),
            "food_detections": raw_dets,
            "rejected": food_out.get("rejected", []),
            "detection_mode": food_out.get("mode", "full"),
            "merged": merged,
            "rice_rule": rice_info,
            "height_level": level_name,
            "height_multiplier": mult,
            "height_levels": dict(HEIGHT_LEVELS),
            "dishes": dishes,
            "totals": totals,
            "calibration": calibration,
            "health": health,
            "meal_share": meal_share,
            "meal_type": meal_type,
            "budget": budget,
            "user": dict(user),
            "warnings": warnings,
            "elapsed_s": round(time.time() - t0, 2),
        }
