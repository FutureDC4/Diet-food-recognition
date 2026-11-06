"""营养计算：重量 → 热量 / 三大营养素 / 膳食纤维 / 钠 / 微量元素。

数据来自 ``data/food_reference.yaml``（《中国食物成分表》第 6 版典型值）：
  - 早期版本只有「热量」和一组合成的供能比，蛋白/脂肪/碳水**是反推出来的**；
  - 现在直接用表里的实测三大营养素，并补充膳食纤维、钠、钾、钙、铁、锌、硒、
    镁与维生素 A/B1/B2/C/E，以及烹调方式与促炎倾向。

没有实测数值的食物（COCO 旧类别等）仍回退到「热量 × 供能比」的粗略拆分，
保证老接口和 CLI 不报错。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from ..utils.config_loader import resolve_path
from ..utils.logger import get_logger

log = get_logger("营养计算")

DEFAULT_ENTRY = {
    "density": 0.80,
    "calories_per_100g": 200.0,
    "protein_g": 8.0,
    "fat_g": 8.0,
    "carb_g": 24.0,
    "fiber_g": 1.0,
    "sodium_mg": 300.0,
}

# 微量元素字段（per 100g），顺序决定界面/报告里的展示顺序
MICRO_FIELDS = (
    "calcium_mg", "iron_mg", "zinc_mg", "selenium_ug", "potassium_mg", "magnesium_mg",
    "vitamin_a_ug", "vitamin_b1_mg", "vitamin_b2_mg", "vitamin_c_mg", "vitamin_e_mg",
)

# 三大营养素供能比（仅在缺实测值时兜底）
_MACRO_PROFILE = {
    "chicken": (0.45, 0.30, 0.25),
    "rice": (0.10, 0.05, 0.85),
    "pasta": (0.15, 0.10, 0.75),
    "egg": (0.35, 0.45, 0.20),
    "cheese": (0.25, 0.70, 0.05),
    "cake": (0.07, 0.35, 0.58),
    "apple": (0.03, 0.03, 0.94),
    "banana": (0.05, 0.03, 0.92),
    "broccoli": (0.30, 0.10, 0.60),
}
_DEFAULT_MACRO = (0.20, 0.25, 0.55)


def _num(value, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_food_reference(path: Optional[str | Path] = None) -> Dict[str, dict]:
    """读取食物营养参考表（含微量元素与烹调方式）。"""
    p = Path(path) if path else resolve_path("paths.food_reference")
    if not p.exists():
        raise FileNotFoundError(f"食物营养参考表不存在：{p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    table: Dict[str, dict] = {}
    for name, value in data.items():
        key = str(name).lower()
        if key.startswith("_"):        # _meta 等说明块
            continue
        if isinstance(value, dict):
            entry = {
                "density": _num(value.get("density"), DEFAULT_ENTRY["density"]),
                "calories_per_100g": _num(value.get("calories_per_100g"),
                                         DEFAULT_ENTRY["calories_per_100g"]),
                "protein_g": _num(value.get("protein_g"), DEFAULT_ENTRY["protein_g"]),
                "fat_g": _num(value.get("fat_g"), DEFAULT_ENTRY["fat_g"]),
                "carb_g": _num(value.get("carb_g"), DEFAULT_ENTRY["carb_g"]),
                "fiber_g": _num(value.get("fiber_g"), DEFAULT_ENTRY["fiber_g"]),
                "sodium_mg": _num(value.get("sodium_mg"), DEFAULT_ENTRY["sodium_mg"]),
                "name_cn": value.get("name_cn"),
                "category": value.get("category", "其他"),
                "cooking": value.get("cooking", "未知"),
                "inflammation": value.get("inflammation", "中"),
                "risk": value.get("risk"),
                "note": value.get("note", ""),
                "micro": {k: _num(v, 0.0) for k, v in (value.get("micro") or {}).items()},
            }
            table[key] = entry
        elif isinstance(value, (int, float)):
            table[key] = {
                **DEFAULT_ENTRY,
                "density": float(value),
                "name_cn": None, "category": "其他", "cooking": "未知",
                "inflammation": "中", "risk": None, "note": "", "micro": {},
            }
    return table


class NutritionCalculator:
    """重量 → 热量 / 三大营养素 / 膳食纤维 / 钠 / 微量元素。"""

    def __init__(self, reference: Optional[Dict[str, dict]] = None):
        self.reference = reference if reference is not None else load_food_reference()
        self.default = self.reference.get("default", dict(DEFAULT_ENTRY))

    # ------------------------------------------------------------------
    def find_food(self, food_name: str) -> tuple[dict, bool]:
        """查营养参考，返回 (条目, 是否命中)。

        匹配顺序：精确 → 去空格/下划线/连字符 → 包含匹配。
        """
        key = str(food_name).strip().lower()
        if key in self.reference:
            return {**self.reference[key], "matched_key": key}, True

        norm = key.replace(" ", "_").replace("-", "_")
        if norm in self.reference:
            return {**self.reference[norm], "matched_key": norm}, True

        flat = norm.replace("_", "")
        for name, entry in self.reference.items():
            if name == "default":
                continue
            if flat and (flat in name.replace("_", "") or name.replace("_", "") in flat):
                return {**entry, "matched_key": name}, True

        return {**self.default, "matched_key": "default"}, False

    # ------------------------------------------------------------------
    def calculate(self, food_name: str, weight_g: float) -> dict:
        """单个食物的营养计算（返回每份的实际克数）。"""
        entry, matched = self.find_food(food_name)
        weight_g = float(weight_g)
        ratio = weight_g / 100.0
        calories = weight_g * float(entry["calories_per_100g"]) / 100.0

        # 有实测三大营养素就用实测值；否则按供能比反推（老数据兼容）
        if entry.get("protein_g") is not None and entry.get("carb_g") is not None \
                and float(entry.get("protein_g", 0)) + float(entry.get("carb_g", 0)) > 0:
            protein_g = round(float(entry["protein_g"]) * ratio, 1)
            fat_g = round(float(entry["fat_g"]) * ratio, 1)
            carb_g = round(float(entry["carb_g"]) * ratio, 1)
            macro_source = "table"
        else:
            profile = _MACRO_PROFILE.get(entry.get("matched_key", "").replace("_", ""), _DEFAULT_MACRO)
            protein_g, fat_g, carb_g = self._split_macros(calories, profile)
            macro_source = "estimated"

        if not matched:
            log.debug(f"食物 `{food_name}` 不在营养表中，使用默认值（200 kcal/100g）")

        micro = {
            k: round(float((entry.get("micro") or {}).get(k, 0.0)) * ratio, 2)
            for k in MICRO_FIELDS
        }

        return {
            "name": food_name,
            "weight_g": round(weight_g, 1),
            "calories": round(calories, 1),
            "calories_per_100g": float(entry["calories_per_100g"]),
            "density": float(entry["density"]),
            "protein_g": protein_g,
            "fat_g": fat_g,
            "carb_g": carb_g,
            "fiber_g": round(float(entry.get("fiber_g", 0)) * ratio, 1),
            "sodium_mg": round(float(entry.get("sodium_mg", 0)) * ratio, 1),
            "micro": micro,
            "category": entry.get("category", "其他"),
            "cooking": entry.get("cooking", "未知"),
            "inflammation": entry.get("inflammation", "中"),
            "risk": entry.get("risk"),
            "food_note": entry.get("note", ""),
            "macro_source": macro_source,
            "matched": matched,
            "matched_key": entry.get("matched_key", "default"),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _split_macros(calories: float, profile: Sequence[float]) -> tuple[float, float, float]:
        """按供能比反推三大营养素克数（仅在缺实测值时使用）。"""
        p_ratio, f_ratio, c_ratio = profile
        total = p_ratio * 4 + f_ratio * 9 + c_ratio * 4
        if total <= 0:
            return 0.0, 0.0, 0.0
        scale = calories / total
        return round(p_ratio * scale, 1), round(f_ratio * scale, 1), round(c_ratio * scale, 1)

    # ------------------------------------------------------------------
    def calculate_many(self, weight_items: Sequence[dict]) -> List[dict]:
        """批量计算：输入 PortionEstimator 的输出，输出带营养的结果。"""
        out: List[dict] = []
        for item in weight_items:
            nutrition = self.calculate(item["name"], item["weight_g"])
            merged = {**item, **nutrition, "name": item["name"]}
            out.append(merged)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def totals(nutrition_items: Sequence[dict]) -> dict:
        """汇总总重量 / 总热量 / 三大营养素 / 纤维 / 钠。"""
        return {
            "weight_g": round(sum(float(i.get("weight_g", 0)) for i in nutrition_items), 1),
            "calories": round(sum(float(i.get("calories", 0)) for i in nutrition_items), 1),
            "protein_g": round(sum(float(i.get("protein_g", 0)) for i in nutrition_items), 1),
            "fat_g": round(sum(float(i.get("fat_g", 0)) for i in nutrition_items), 1),
            "carb_g": round(sum(float(i.get("carb_g", 0)) for i in nutrition_items), 1),
            "fiber_g": round(sum(float(i.get("fiber_g", 0)) for i in nutrition_items), 1),
            "sodium_mg": round(sum(float(i.get("sodium_mg", 0)) for i in nutrition_items), 1),
        }


if __name__ == "__main__":
    calc = NutritionCalculator()
    print(calc.calculate("rice", 180))
    print(calc.calculate("fried_chicken", 150))
    print(calc.totals(calc.calculate_many([
        {"name": "rice", "weight_g": 180},
        {"name": "water_spinach", "weight_g": 120},
    ])))
