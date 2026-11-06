# -*- coding: utf-8 -*-
"""检测结果的几何后处理规则。

两个规则：
  1. **米饭三级处理**——米饭在中餐里最容易被菜压住，检测框会明显偏小。
     按「米饭框面积 / 全部食物面积」分三级：
        一级  ≥ 20%        框够大，直接用检测值
        二级  4% ~ 20%     被遮挡，按「占食物总面积 1/3」反推面积
        三级  < 4%         正常人不会只打这么一点饭，判为误检，直接丢弃
  2. **食物高度档位**——单张照片无法判断堆得多高，交给用户选「薄摊/常规/堆高」，
     把 PortionEstimator 里按食物类型查到的默认高度乘上对应系数。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

RICE_KEYS = {"rice", "white_rice", "steamed_rice", "米饭", "rice_cooked"}

# ---- 米饭三级阈值 ----
RICE_TRUST_RATIO = 0.20      # ≥ 20% 视为检测可信
RICE_REJECT_RATIO = 0.04     # < 4% 视为误检
RICE_TARGET_RATIO = 1.0 / 3.0  # 被遮挡时的目标占比

# ---- 食物高度档位 ----
HEIGHT_LEVELS: Dict[str, float] = {
    "薄摊": 0.75,   # 平盘摊开、汤菜、凉拌
    "常规": 1.00,   # 食堂一格菜的正常高度
    "堆高": 1.35,   # 大份、堆起来的一盘
}
DEFAULT_HEIGHT_LEVEL = "常规"


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def is_rice(name: str) -> bool:
    return _norm(name) in RICE_KEYS or "rice" in _norm(name)


def height_multiplier(level: Optional[str]) -> Tuple[float, str]:
    """档位 → 高度系数。未知档位回落到常规。

    ``config.yaml`` 的 ``height_levels`` 可以覆盖系数；档位名称以本模块为准
    （界面按钮由 HEIGHT_LEVELS 生成，改名字要同步改配置）。
    """
    if level not in HEIGHT_LEVELS:
        return HEIGHT_LEVELS[DEFAULT_HEIGHT_LEVEL], DEFAULT_HEIGHT_LEVEL
    mult = HEIGHT_LEVELS[level]
    try:
        from src.utils.config_loader import get
        cfg = get("height_levels", {}) or {}
        if isinstance(cfg.get(level), (int, float)):
            mult = float(cfg[level])
    except Exception:                                 # noqa: BLE001
        pass
    return mult, str(level)


def apply_rice_rule(
    merged: List[dict],
    trust_ratio: float = RICE_TRUST_RATIO,
    reject_ratio: float = RICE_REJECT_RATIO,
    target_ratio: float = RICE_TARGET_RATIO,
) -> Tuple[List[dict], Dict[str, Any]]:
    """米饭三级处理。返回 (处理后的列表, 说明信息)。

    面积一律用 bbox 面积（`area_px`）；如果检测带了 SAM 掩膜，占比计算也统一用
    bbox，避免掩膜与 bbox 口径不一致导致占比失真。
    """
    info: Dict[str, Any] = {
        "applied": False, "level": 0, "ratio": target_ratio,
        "detected_ratio": None, "note": "未检测到米饭",
    }
    if not merged:
        return merged, info

    idx = next((i for i, d in enumerate(merged) if is_rice(d.get("name", ""))), None)
    if idx is None:
        return merged, info

    def area(d: dict) -> float:
        return float(d.get("area_px") or 0)

    total = sum(area(d) for d in merged)
    rice_area = area(merged[idx])
    if total <= 0 or rice_area <= 0:
        info["note"] = "米饭面积为 0，跳过规则"
        return merged, info

    ratio = rice_area / total
    info["detected_ratio"] = round(ratio, 4)

    # ---------------- 一级：可信 ----------------
    if ratio >= trust_ratio:
        info.update({"level": 1, "note": f"米饭占食物面积 {ratio:.0%}，检测可信，按检测值计算"})
        return merged, info

    # ---------------- 三级：判为误检 ----------------
    if ratio < reject_ratio:
        info.update({
            "level": 3, "applied": False, "rejected": True,
            "note": f"米饭只占食物面积 {ratio:.1%}（<{reject_ratio:.0%}），判为误检，已剔除",
        })
        out = [dict(d) for i, d in enumerate(merged) if i != idx]
        return out, info

    # ---------------- 二级：按 1/3 反推 ----------------
    others = total - rice_area
    if others <= 0:
        info.update({"level": 2, "note": "只有米饭，无需修正"})
        return merged, info

    target = others * target_ratio / max(1.0 - target_ratio, 1e-6)
    out = [dict(d) for d in merged]
    out[idx]["area_px"] = target
    out[idx]["mask_area_px"] = None      # 面积已改，掩膜不再代表实际
    out[idx]["rule_adjusted"] = True
    out[idx]["rice_level"] = 2
    info.update({
        "applied": True, "level": 2,
        "others_area_px": round(others), "detected_area_px": round(rice_area),
        "target_area_px": round(target),
        "note": (f"米饭被遮挡（占 {ratio:.0%}，低于 {trust_ratio:.0%}），"
                 f"按占食物总面积 1/3 反推修正（{rice_area:,.0f} → {target:,.0f} px²）"),
    })
    return out, info
