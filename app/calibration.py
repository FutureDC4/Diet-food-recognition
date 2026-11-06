# -*- coding: utf-8 -*-
"""营养数值的合理性校正：把「装盘营养」折算成「入口营养」。

为什么要这一层
--------------
一份菜的克数由 ``bbox 面积 × 堆叠高度 × 密度`` 得到，这个体积描述的是
**装进盘子里的量**；而营养表给的是每 100 g 食物本身的成分。两者之间存在三次
系统性偏高，且都朝同一个方向：

1. **装盘量 ≠ 入口量**
   菜帮、皮、骨、盘底的汤汁不会吃下去。叶菜、带汤菜最明显（约 10%-12%），
   块状荤菜少一些（约 5%-6%）。
2. **营养表含全部烹调油**
   食堂与家常炒菜的油有相当一部分汪在盘底、挂在盘沿。翻勺时裹在食材表面的
   只有一部分，剩下的留在盘里。按烹调方式不同，残留 5%-32%。
3. **盐和酱油随汤汁走**
   一份菜的盐分约有 20% 溶在盘底的汤里。

三条都不改检测模型、不改食物成分表，只在「体积 → 克 → 营养」之后做一次
统一折算。因此：

* 整条链路的**相对关系不变**（哪样菜油多、哪样菜咸，判断照旧）；
* 每一项都能在 ``config.yaml → nutrition_calibration`` 里单独关掉或调参；
* 每道菜同时保留 ``gross_*``（装盘值）与校正后的值，报告与调参都能追溯。

一个诚实的边界：脂肪供能比只会小幅下降。像「腊肠 + 煎蛋 + 油炸薯饼」这种
组合，即便把所有浮油都扣掉，脂肪供能仍在 55% 上下——这是食物本身的性质，
不是估算偏差。校正层只负责去掉「估算放大」的那部分，不去修饰真实结论。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get  # noqa: E402

# 兜底参数：config.yaml 缺项时也能跑（与默认配置保持一致）
DEFAULT_CFG: Dict[str, Any] = {
    "enabled": True,
    "edible_ratio": {
        "default": 0.95,
        "蔬菜": 0.88,
        "荤菜": 0.94,
        "加工肉": 0.97,
        "油炸": 0.95,
        "主食": 0.97,
        "甜点": 0.97,
        "水果": 0.92,
        "奶类": 1.00,
        "蛋类": 1.00,
    },
    "oil": {
        "default": {"added": 0.50, "residue": 0.25},
        "清炒": {"added": 0.70, "residue": 0.32},
        "炒": {"added": 0.70, "residue": 0.32},
        "干煸": {"added": 0.75, "residue": 0.30},
        "煎": {"added": 0.62, "residue": 0.30},
        "油炸": {"added": 0.55, "residue": 0.25},
        "烧": {"added": 0.55, "residue": 0.28},
        "炖": {"added": 0.45, "residue": 0.22},
        "蒸": {"added": 0.10, "residue": 0.05},
        "煮": {"added": 0.05, "residue": 0.02},
        "凉拌": {"added": 0.65, "residue": 0.30},
        "烤": {"added": 0.30, "residue": 0.10},
        "生": {"added": 0.00, "residue": 0.00},
        "无": {"added": 0.00, "residue": 0.00},
    },
    "sodium_retention": 0.78,
}

_MACRO_SCALE_FIELDS = ("protein_g", "carb_g", "fiber_g")


def load_calibration() -> Dict[str, Any]:
    """从 config.yaml 读校正参数，缺项回落到 DEFAULT_CFG。"""
    cfg = get("nutrition_calibration", None)
    if not isinstance(cfg, dict):
        return dict(DEFAULT_CFG)
    merged = {
        "enabled": bool(cfg.get("enabled", DEFAULT_CFG["enabled"])),
        "edible_ratio": {**DEFAULT_CFG["edible_ratio"], **(cfg.get("edible_ratio") or {})},
        "oil": {**DEFAULT_CFG["oil"], **(cfg.get("oil") or {})},
        "sodium_retention": float(cfg.get("sodium_retention", DEFAULT_CFG["sodium_retention"])),
    }
    return merged


def _lookup(table: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key in table and isinstance(table[key], dict):
        return table[key]
    return table.get("default") or {}


def calibrate_dish(dish: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """单道菜：装盘营养 → 入口营养。不改动入参。"""
    cfg = cfg or load_calibration()
    if not cfg.get("enabled", True):
        return dict(dish)

    weight0 = float(dish.get("weight_g", 0) or 0)
    if weight0 <= 0:
        return dict(dish)

    edible = float((cfg["edible_ratio"]).get(str(dish.get("category") or ""),
                                            cfg["edible_ratio"]["default"]))
    oil = _lookup(cfg["oil"], str(dish.get("cooking") or ""))
    added = float(oil.get("added", 0.0) or 0.0)
    residue = float(oil.get("residue", 0.0) or 0.0)
    na_keep = float(cfg.get("sodium_retention", 0.8))

    fat0 = float(dish.get("fat_g", 0) or 0)
    kcal0 = float(dish.get("calories", 0) or 0)
    na0 = float(dish.get("sodium_mg", 0) or 0)

    # 只有「外加烹调油」那部分会有一部分留在盘底
    fat1 = fat0 * (1.0 - added * residue)
    kcal1 = max(kcal0 - (fat0 - fat1) * 9.0, 0.0)
    na1 = na0 * na_keep

    out = dict(dish)
    out["gross_weight_g"] = round(weight0, 1)
    out["gross_fat_g"] = round(fat0, 1)
    out["gross_sodium_mg"] = round(na0, 1)
    out["gross_calories"] = round(kcal0, 1)

    out["weight_g"] = round(weight0 * edible, 1)
    out["fat_g"] = round(fat1 * edible, 1)
    out["calories"] = round(kcal1 * edible, 1)
    out["sodium_mg"] = round(na1 * edible, 1)
    for key in _MACRO_SCALE_FIELDS:
        out[key] = round(float(dish.get(key, 0) or 0) * edible, 1)
    out["micro"] = {k: round(float(v or 0) * edible, 2)
                    for k, v in (dish.get("micro") or {}).items()}

    out["calibrated"] = True
    out["calibration"] = {
        "edible_ratio": round(edible, 3),
        "oil_added": round(added, 2),
        "oil_residue": round(residue, 2),
        "sodium_retention": round(na_keep, 2),
    }
    return out


def calibrate(dishes: Sequence[Dict[str, Any]],
              cfg: Optional[Dict[str, Any]] = None) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """整餐校正，返回 (校正后的菜列表, 说明块)。

    说明块写进分析结果，界面/报告可以直接标注「已按可食部与油盐残留折算」。
    """
    cfg = cfg or load_calibration()
    if not cfg.get("enabled", True):
        return [dict(d) for d in dishes], {"enabled": False}

    out = [calibrate_dish(d, cfg) for d in dishes]

    gross_w = sum(float(d.get("gross_weight_g", d.get("weight_g", 0)) or 0) for d in out)
    new_w = sum(float(d.get("weight_g", 0) or 0) for d in out)
    gross_na = sum(float(d.get("gross_sodium_mg", d.get("sodium_mg", 0)) or 0) for d in out)
    new_na = sum(float(d.get("sodium_mg", 0) or 0) for d in out)
    gross_fat = sum(float(d.get("gross_fat_g", d.get("fat_g", 0)) or 0) for d in out)
    new_fat = sum(float(d.get("fat_g", 0) or 0) for d in out)

    return out, {
        "enabled": True,
        "weight_before": round(gross_w, 1),
        "weight_after": round(new_w, 1),
        "sodium_before": round(gross_na, 0),
        "sodium_after": round(new_na, 0),
        "fat_before": round(gross_fat, 1),
        "fat_after": round(new_fat, 1),
        "sodium_retention": cfg.get("sodium_retention", 0.78),
    }


if __name__ == "__main__":
    demo = [
        {"name": "curry", "name_cn": "咖喱", "weight_g": 210, "calories": 273,
         "fat_g": 13.6, "protein_g": 13.0, "carb_g": 24.4, "fiber_g": 2.9,
         "sodium_mg": 1008, "category": "荤菜", "cooking": "炖"},
        {"name": "chinese_sausage", "name_cn": "腊肠", "weight_g": 104, "calories": 416,
         "fat_g": 34.3, "protein_g": 18.7, "carb_g": 4.2, "fiber_g": 0.0,
         "sodium_mg": 1248, "category": "加工肉", "cooking": "蒸"},
        {"name": "chinese_cabbage", "name_cn": "炒白菜", "weight_g": 74, "calories": 41,
         "fat_g": 2.4, "protein_g": 1.2, "carb_g": 3.4, "fiber_g": 0.6,
         "sodium_mg": 185, "category": "蔬菜", "cooking": "清炒"},
    ]
    out, info = calibrate(demo)
    for d in out:
        print(f"{d['name_cn']}: {d['gross_weight_g']}→{d['weight_g']} g，"
              f"脂肪 {d['gross_fat_g']}→{d['fat_g']} g，钠 {d['gross_sodium_mg']}→{d['sodium_mg']} mg")
    print(info)
