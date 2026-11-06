# -*- coding: utf-8 -*-
"""单餐膳食规则：餐次配额（B）+ 结构评估（C）+「该加什么」。

设计原则
--------
**这一层不依赖大模型。** 本地 Ollama 是 1.5B 的小模型，让它自己算「蛋白质差几克」
会经常算错。所以先把所有数字算准、判断做死，再把结论交给模型去组织语言。

依据（见 ``data/dri_cn.yaml``）：
  * 餐次能量分配：早餐 25% / 午餐 40% / 晚餐 35%（加餐由午晚各匀 5%）
  * 《中国居民膳食营养素参考摄入量》2023 版：蛋白质 RNI、膳食纤维 AI 25g/d、
    钙/铁/锌/硒/钾/镁与维生素 A/B1/B2/C/E 的 RNI
  * 《中国居民膳食指南（2022）》：蔬菜 300-500g/d、奶 300-500g/d、盐 <5g/d、油 25-30g/d

输出分三块：
  ``targets``    本餐的能量区间与各营养素目标
  ``checks``     结构评估条目（达标 / 偏低 / 偏高），带一句话说明
  ``additions``  具体的「该加什么」——食物名 + 分量 + 能补多少
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 食物归类（用于蔬菜占比 / 主食供能比 / 油炸占比）
VEG_CATEGORIES = {"蔬菜"}
PROTEIN_CATEGORIES = {"荤菜", "蛋类", "奶类", "加工肉"}
STAPLE_CATEGORIES = {"主食", "甜点"}

# 状态 → 界面颜色语义
STATUS_OK = "ok"
STATUS_LOW = "low"
STATUS_HIGH = "high"
STATUS_WARN = "warn"


# ====================================================================== 载入
def load_dri(path: Optional[str | Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else PROJECT_ROOT / "data" / "dri_cn.yaml"
    if not p.exists():
        raise FileNotFoundError(f"DRIs 参考表不存在：{p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def _age_band(age: int) -> str:
    if age < 50:
        return "18-49"
    if age < 65:
        return "50-64"
    if age < 80:
        return "65-79"
    return "80+"


MAIN_MEALS = ("早餐", "午餐", "晚餐")


def _meal_share_map(dri: Dict[str, Any]) -> Dict[str, float]:
    """餐次占比。

    三餐 早 25% / 午 40% / 晚 35% 加起来正好是 100%，**不做归一化**，
    否则午餐会被压到 36%，和「午餐按 TDEE 40% 算」的口径对不上。
    加餐不额外占一天的能量，而是从午、晚各匀出 5%，所以单独取 0.10。
    """
    raw = {k: float(v) for k, v in (dri.get("meal_share") or {}).items()
           if isinstance(v, (int, float))}
    out = {k: raw.get(k, 0.0) for k in MAIN_MEALS}
    out["加餐"] = raw.get("加餐", 0.10)
    return out


# ====================================================================== 目标
def meal_targets(user: Dict[str, Any], health: Dict[str, Any],
                 meal_type: str, dri: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """本餐的能量区间与营养素目标。

    蛋白质取「RNI 按餐次分摊」与「体重法 1.0 g/kg/d 按餐次分摊」的较大值：
    前者照顾矮小/轻体重人群，后者照顾健身人群，避免任一口径太偏。
    """
    dri = dri or load_dri()
    shares = _meal_share_map(dri)
    share = shares.get(meal_type, shares.get("午餐", 0.4))

    tdee = float(health.get("tdee", 0) or 0)
    kcal_target = tdee * share
    # ±10% 作为「合理区间」，比给一个点值更符合真实饮食
    kcal_min, kcal_max = kcal_target * 0.90, kcal_target * 1.10

    age = int(user.get("age", 30) or 30)
    gender_key = "male" if str(user.get("gender", "男")) == "男" else "female"
    weight = float(user.get("weight_kg", 60) or 60)

    macros = dri.get("macros") or {}
    protein_cfg = macros.get("protein") or {}
    rni_table = (protein_cfg.get("rni_g_per_day") or {}).get(gender_key, {})
    protein_daily = float(rni_table.get(_age_band(age), 65 if gender_key == "male" else 55))
    protein_daily = max(protein_daily, weight * float(protein_cfg.get("g_per_kg_day", 1.0)))

    fiber_daily = float((macros.get("fiber") or {}).get("ai_g_per_day", 25))
    structure = dri.get("structure") or {}

    veg_ratio_range = structure.get("veg_weight_ratio") or [0.30, 0.60]
    staple_range = structure.get("staple_energy_ratio") or [0.25, 0.55]

    # 蔬菜：以膳食指南 300-500 g/d 的中位 400 g 折算到本餐，
    # 再与「占本餐重量 ≥30%」互为校验，取较大值（见 evaluate_structure）
    veg_daily = 400.0

    return {
        "meal_type": meal_type,
        "meal_share": round(share, 4),
        "meal_share_text": f"{share:.0%}",
        "tdee": round(tdee, 0),
        "kcal_target": round(kcal_target, 0),
        "kcal_min": round(kcal_min, 0),
        "kcal_max": round(kcal_max, 0),
        "protein_g": round(protein_daily * share, 1),
        "protein_daily_g": round(protein_daily, 1),
        "fiber_g": round(fiber_daily * share, 1),
        "fiber_daily_g": round(fiber_daily, 1),
        "veg_g": round(veg_daily * share, 0),
        "veg_daily_g": round(veg_daily, 0),
        "veg_ratio_min": float(veg_ratio_range[0]),
        "veg_ratio_max": float(veg_ratio_range[1]),
        "staple_ratio_min": float(staple_range[0]),
        "staple_ratio_max": float(staple_range[1]),
        "fried_ratio_max": float(structure.get("fried_energy_ratio_max", 0.30)),
        "sodium_mg": float(structure.get("sodium_mg_per_meal", 667)),
        "fat_ratio": [float(x) for x in ((macros.get("fat") or {}).get("energy_ratio") or [0.20, 0.30])],
        "carb_ratio": [float(x) for x in ((macros.get("carb") or {}).get("energy_ratio") or [0.50, 0.65])],
    }


# ====================================================================== 结构评估
def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def evaluate_structure(dishes: List[dict], totals: Dict[str, Any],
                       targets: Dict[str, Any]) -> Dict[str, Any]:
    """B+C 里的 C：不看总量，看结构。

    返回 {"checks": [...], "groups": {...}, "score": 0-100}
    每个 check: key / label / actual / actual_text / target_text / status / hint
    """
    kcal = float(totals.get("calories", 0) or 0)
    protein = float(totals.get("protein_g", 0) or 0)
    fiber = float(totals.get("fiber_g", 0) or 0)
    fat = float(totals.get("fat_g", 0) or 0)
    sodium = float(totals.get("sodium_mg", 0) or 0)
    weight = float(totals.get("weight_g", 0) or 0)

    # ---- 分组 ----
    veg_kcal = veg_g = staple_kcal = fried_kcal = protein_kcal = 0.0
    fried_names: List[str] = []
    for d in dishes:
        cat = str(d.get("category", ""))
        c = float(d.get("calories", 0) or 0)
        w = float(d.get("weight_g", 0) or 0)
        if cat in VEG_CATEGORIES:
            veg_kcal += c
            veg_g += w
        if cat in STAPLE_CATEGORIES:
            staple_kcal += c
        if cat in PROTEIN_CATEGORIES:
            protein_kcal += c
        if d.get("risk") in ("油炸", "加工肉") or cat in ("油炸",):
            fried_kcal += c
            fried_names.append(str(d.get("name_cn") or d.get("name")))

    groups = {
        "veg_g": round(veg_g, 0),
        "veg_kcal": round(veg_kcal, 0),
        "veg_weight_ratio": round(_ratio(veg_g, weight), 3),
        "staple_kcal": round(staple_kcal, 0),
        "staple_energy_ratio": round(_ratio(staple_kcal, kcal), 3),
        "fried_kcal": round(fried_kcal, 0),
        "fried_energy_ratio": round(_ratio(fried_kcal, kcal), 3),
        "fried_names": fried_names,
        "fat_energy_ratio": round(_ratio(fat * 9, kcal), 3),
        "protein_energy_ratio": round(_ratio(protein * 4, kcal), 3),
    }

    checks: List[Dict[str, Any]] = []

    # 1. 能量配额
    kmin, kmax = float(targets["kcal_min"]), float(targets["kcal_max"])
    if kcal < kmin:
        st, hint = STATUS_LOW, f"比建议区间少 {kmin - kcal:.0f} kcal，可能吃不饱或漏识别"
    elif kcal > kmax:
        st, hint = STATUS_HIGH, f"比建议区间多 {kcal - kmax:.0f} kcal，注意烹调油与主食量"
    else:
        st, hint = STATUS_OK, "落在建议区间内"
    checks.append({
        "key": "kcal", "label": "本餐热量",
        "actual": round(kcal, 0), "actual_text": f"{kcal:.0f} kcal",
        "target_text": f"{kmin:.0f}-{kmax:.0f} kcal（TDEE 的 {targets['meal_share_text']}）",
        "status": st, "hint": hint,
    })

    # 2. 蛋白质
    p_target = float(targets["protein_g"])
    pr = _ratio(protein, p_target)
    if pr >= 1.0:
        st, hint = STATUS_OK, "蛋白质充足"
    elif pr >= float(0.8):
        st, hint = STATUS_OK, f"接近目标，还差 {p_target - protein:.0f} g"
    elif pr >= 0.5:
        st, hint = STATUS_LOW, f"偏低，还差 {p_target - protein:.0f} g（约一个鸡蛋 + 一杯牛奶）"
    else:
        st, hint = STATUS_LOW, f"明显不足，差 {p_target - protein:.0f} g，下一餐要重点补"
    checks.append({
        "key": "protein", "label": "蛋白质",
        "actual": round(protein, 1), "actual_text": f"{protein:.0f} g",
        "target_text": f"目标 {p_target:.0f} g",
        "status": st, "hint": hint, "ratio": round(min(pr, 1.5), 2),
    })

    # 3. 膳食纤维
    f_target = float(targets["fiber_g"])
    fr = _ratio(fiber, f_target)
    if fr >= 0.9:
        st, hint = STATUS_OK, "纤维充足"
    elif fr >= 0.6:
        st, hint = STATUS_LOW, f"略低，还差 {f_target - fiber:.1f} g，加一份绿叶菜就够"
    else:
        st, hint = STATUS_LOW, f"明显偏低，只到目标的 {fr:.0%}，蔬菜与全谷物都要加"
    checks.append({
        "key": "fiber", "label": "膳食纤维",
        "actual": round(fiber, 1), "actual_text": f"{fiber:.1f} g",
        "target_text": f"目标 {f_target:.1f} g",
        "status": st, "hint": hint, "ratio": round(min(fr, 1.5), 2),
    })

    # 4. 蔬菜占比（重量）
    vmin = float(targets["veg_ratio_min"])
    vr = groups["veg_weight_ratio"]
    veg_target_g = max(float(targets.get("veg_g", 0) or 0), vmin * weight)
    if vr >= vmin:
        st, hint = STATUS_OK, f"蔬菜 {groups['veg_g']:.0f} g，量达标"
    elif vr >= vmin * 0.5:
        st, hint = STATUS_LOW, f"蔬菜偏少（占 {vr:.0%}），建议 ≥{vmin:.0%}，约 {veg_target_g:.0f} g"
    else:
        st, hint = STATUS_LOW, f"蔬菜明显不足（占 {vr:.0%}），这一餐几乎全是主食和荤菜"
    checks.append({
        "key": "veg", "label": "蔬菜占比",
        "actual": round(vr * 100, 0), "actual_text": f"{vr:.0%}",
        "target_text": f"≥{vmin:.0%}（约 {veg_target_g:.0f} g）",
        "status": st, "hint": hint,
    })

    # 5. 主食供能比
    smax, smin = float(targets["staple_ratio_max"]), float(targets["staple_ratio_min"])
    sr = groups["staple_energy_ratio"]
    if sr > smax:
        st, hint = STATUS_HIGH, f"主食供能 {sr:.0%} 偏高（建议 ≤{smax:.0%}），可减 1/4 饭量"
    elif sr < smin and kcal > 0:
        st, hint = STATUS_LOW, f"主食供能仅 {sr:.0%}，若在减重可接受，否则易饿"
    else:
        st, hint = STATUS_OK, "主食比例合理"
    checks.append({
        "key": "staple", "label": "主食供能比",
        "actual": round(sr * 100, 0), "actual_text": f"{sr:.0%}",
        "target_text": f"建议 {smin:.0%}-{smax:.0%}",
        "status": st, "hint": hint,
    })

    # 6. 脂肪供能比
    fr_min, fr_max = targets["fat_ratio"]
    far = groups["fat_energy_ratio"]
    if kcal <= 0:
        st, hint = STATUS_OK, "—"
    elif far > fr_max:
        st, hint = STATUS_HIGH, f"脂肪供能 {far:.0%} 偏高（建议 {fr_min:.0%}-{fr_max:.0%}），油偏多"
    elif far < fr_min:
        st, hint = STATUS_LOW, f"脂肪供能仅 {far:.0%}，可能过素，注意必需脂肪酸"
    else:
        st, hint = STATUS_OK, "脂肪比例合理"
    checks.append({
        "key": "fat", "label": "脂肪供能比",
        "actual": round(far * 100, 0), "actual_text": f"{far:.0%}",
        "target_text": f"建议 {fr_min:.0%}-{fr_max:.0%}",
        "status": st, "hint": hint,
    })

    # 7. 钠
    na_max = float(targets["sodium_mg"])
    if sodium > na_max * 1.25:
        st, hint = STATUS_HIGH, f"钠 {sodium:.0f} mg 偏高（一餐上限约 {na_max:.0f} mg），下一餐吃清淡些"
    elif sodium > na_max:
        st, hint = STATUS_HIGH, f"钠略超 {na_max:.0f} mg，注意酱油与腌制食品"
    else:
        st, hint = STATUS_OK, "钠在合理范围"
    checks.append({
        "key": "sodium", "label": "钠",
        "actual": round(sodium, 0), "actual_text": f"{sodium:.0f} mg",
        "target_text": f"单餐建议 ≤{na_max:.0f} mg",
        "status": st, "hint": hint,
    })

    # 8. 油炸 / 加工肉
    fmax = float(targets["fried_ratio_max"])
    frr = groups["fried_energy_ratio"]
    fried_txt = "、".join(fried_names)
    if fried_names and frr > fmax:
        st, hint = STATUS_WARN, f"{fried_txt} 属油炸/加工类，占本餐供能 {frr:.0%}，促炎负荷偏高"
    elif fried_names:
        st, hint = STATUS_OK, f"有 {fried_txt}，占比 {frr:.0%}，偶尔吃没问题"
    else:
        st, hint = STATUS_OK, "没有油炸或加工肉类"
    checks.append({
        "key": "fried", "label": "促炎负荷",
        "actual": round(frr * 100, 0), "actual_text": f"{frr:.0%}",
        "target_text": f"油炸/加工肉 ≤{fmax:.0%}",
        "status": st, "hint": hint, "names": fried_names,
    })

    # ---- 进度条数值（0-100，供界面画条；超上限时截到 100）----
    bar_ref = {
        "kcal": max(kmax, 1.0),
        "protein": max(p_target, 0.1),
        "fiber": max(f_target, 0.1),
        "veg": max(vmin * 100, 1.0),
        "staple": max(smax * 100, 1.0),
        "fat": max(fr_max * 100, 1.0),
        "sodium": max(na_max, 1.0),
        "fried": max(fmax * 100, 1.0),
    }
    for c in checks:
        ref = bar_ref.get(c["key"], 100.0)
        c["bar"] = int(max(0, min(100, round(float(c["actual"]) / ref * 100))))

    # ---- 评分：达标项 / 总项 ----
    scored = [c for c in checks if c["key"] in ("kcal", "protein", "fiber", "veg", "staple", "fat", "sodium", "fried")]
    good = sum(1 for c in scored if c["status"] == STATUS_OK)
    score = round(good / len(scored) * 100) if scored else 0

    return {"checks": checks, "groups": groups, "score": score,
            "good": good, "total": len(scored)}


# ====================================================================== 微量元素
def _user_overrides(user: Dict[str, Any], dri: Dict[str, Any]) -> Dict[str, float]:
    """性别 / 年龄段对 RNI 的覆盖（女性铁、锌、镁、维 A，50 岁以上钙）。"""
    overrides: Dict[str, float] = {}
    if str(user.get("gender", "男")) == "女":
        overrides.update((dri.get("micronutrient_overrides") or {}).get("female") or {})
    if int(user.get("age", 30) or 30) >= 50:
        overrides.update((dri.get("micronutrient_overrides") or {}).get("age_50_plus") or {})
        if str(user.get("gender", "男")) == "女":
            overrides.update((dri.get("micronutrient_overrides") or {}).get("female_age_50_plus") or {})
    return overrides


def micro_daily_text(user: Dict[str, Any], dri: Optional[Dict[str, Any]] = None) -> str:
    """该用户每日微量元素参考摄入量，一行文字，直接写进 prompt。

    让 1.5B 的本地模型「预先知道不同人群需要多少」——不必自己记忆 DRIs，
    只要照抄这一行即可，避免它把成人铁 RNI 说成 8mg 这类常见错误。
    """
    dri = dri or load_dri()
    meta = dri.get("micronutrients") or {}
    overrides = _user_overrides(user, dri)
    items = []
    for key, info in meta.items():
        if info.get("tracked", True) is False:
            continue
        rni = float(overrides.get(key, info.get("rni", 0)) or 0)
        if rni <= 0:
            continue
        items.append(f"{info.get('name', key)} {rni:g}{info.get('unit', '')}")
    return "、".join(items)


def micronutrient_status(dishes: List[dict], targets: Dict[str, Any],
                         user: Dict[str, Any], dri: Optional[Dict[str, Any]] = None
                         ) -> List[Dict[str, Any]]:
    """把本餐微量元素加总，与「每日 RNI × 餐次占比」比较。

    单餐本来只占一天的一部分，看的是「这一餐有没有贡献」，所以只标出
    明显偏低（<30% 目标）与表现突出的项目，避免把正常现象报成问题。
    """
    dri = dri or load_dri()
    meta = dri.get("micronutrients") or {}
    share = float(targets.get("meal_share", 0.4))
    overrides = _user_overrides(user, dri)

    totals: Dict[str, float] = {}
    for d in dishes:
        for k, v in (d.get("micro") or {}).items():
            totals[k] = totals.get(k, 0.0) + float(v or 0)

    out: List[Dict[str, Any]] = []
    for key, info in meta.items():
        if info.get("tracked", True) is False:      # 碘、维生素 D 无法由菜品估算
            continue
        rni = float(overrides.get(key, info.get("rni", 0)) or 0)
        if rni <= 0:
            continue
        per_meal = rni * share
        actual = round(totals.get(key, 0.0), 2)
        pct = round(_ratio(actual, per_meal) * 100)
        out.append({
            "key": key,
            "name": info.get("name", key),
            "actual": actual,
            "unit": info.get("unit", ""),
            "target": round(per_meal, 1),
            "pct": pct,
            "low": pct < 30,
            "note": info.get("note", ""),
        })
    out.sort(key=lambda x: x["pct"])
    return out


# ====================================================================== 该加什么
def suggest_additions(dishes: List[dict], totals: Dict[str, Any], checks: List[Dict[str, Any]],
                      targets: Dict[str, Any], micro: List[Dict[str, Any]],
                      groups: Optional[Dict[str, Any]] = None,
                      max_items: int = 4) -> List[Dict[str, str]]:
    """根据缺口生成具体建议：食物名 + 分量 + 能补多少。

    每条都是可执行的加法；「减」类建议单独标 action=reduce，界面上用另一种颜色。
    """
    by_key = {c["key"]: c for c in checks}
    groups = groups or {}
    kcal = float(totals.get("calories", 0) or 0)
    protein = float(totals.get("protein_g", 0) or 0)
    fiber = float(totals.get("fiber_g", 0) or 0)
    weight = float(totals.get("weight_g", 0) or 0)
    p_target = float(targets["protein_g"])
    f_target = float(targets["fiber_g"])

    out: List[Dict[str, str]] = []

    # 蛋白质缺口最大 → 优先
    p_gap = p_target - protein
    if p_gap >= 8:
        if p_gap >= 18:
            out.append({"action": "add", "text": "加一份 150g 的豆腐或 100g 鸡胸肉（约 +16-23g 蛋白）"})
        else:
            out.append({"action": "add", "text": "加一个水煮蛋（约 50g，+6.5g 蛋白）或 200ml 无糖酸奶（+7g 蛋白）"})

    # 蔬菜缺口：目标取「每日 400g 按餐次折算」与「占本餐重量 30%」中的较大值
    veg_now = float(groups.get("veg_g", 0) or 0)
    veg_target = max(float(targets.get("veg_g", 0) or 0),
                     float(targets["veg_ratio_min"]) * weight)
    veg_need = veg_target - veg_now
    if veg_need >= 60:
        add_g = int(min(300, max(100, round(veg_need / 50) * 50)))
        out.append({"action": "add",
                    "text": f"加一份 {add_g}g 清炒时蔬或白灼菜（约 +{add_g * 0.015:.1f}g 纤维），"
                            f"把这一餐蔬菜补到 {veg_now + add_g:.0f}g"})
    elif by_key.get("fiber", {}).get("status") == STATUS_LOW:
        out.append({"action": "add", "text": "蔬菜量够了但纤维还差，主食换一半成杂粮饭/蒸红薯，能多 3-4g 纤维"})

    # 主食偏多 → 减
    if by_key.get("staple", {}).get("status") == STATUS_HIGH:
        staple_g = float(groups.get("staple_kcal", 0) or 0)
        staple_weight = sum(float(d.get("weight_g", 0)) for d in dishes
                            if str(d.get("category")) in STAPLE_CATEGORIES)
        cut = round(staple_weight * 0.25 / 10) * 10
        if cut >= 30:
            out.append({"action": "reduce",
                        "text": f"主食减约 {cut:.0f}g（约 -{staple_g * 0.25:.0f} kcal），空出的胃口用蔬菜补"})

    # 油炸/加工肉
    if by_key.get("fried", {}).get("status") == STATUS_WARN:
        names = "、".join(by_key["fried"].get("names") or []) or "油炸类"
        out.append({"action": "reduce",
                    "text": f"{names}这类一周控制在 1-2 次，下次换成清蒸、白灼或炖的做法"})

    # 钠超
    if by_key.get("sodium", {}).get("status") == STATUS_HIGH:
        out.append({"action": "reduce", "text": "这餐偏咸，下一餐少放酱油/少吃腌制食品，多喝一杯水"})

    # 钙不足（中式饮食最常见）
    ca = next((m for m in micro if m["key"] == "calcium_mg"), None)
    if ca and ca["pct"] < 25:
        out.append({"action": "add", "text": "加 300ml 牛奶或无糖酸奶（+310mg 钙，约占全天推荐量 40%）"})

    # 维C 与铁：一起说才有意义
    vc = next((m for m in micro if m["key"] == "vitamin_c_mg"), None)
    fe = next((m for m in micro if m["key"] == "iron_mg"), None)
    if vc and vc["pct"] < 25 and fe and fe["pct"] < 40:
        out.append({"action": "add", "text": "加一个橙子或猕猴桃：维生素 C 能把这一餐的铁吸收率提高 2-3 倍"})

    # 能量明显不足
    if kcal < float(targets["kcal_min"]) * 0.75:
        out.append({"action": "add", "text": "整餐热量偏低，若不是在减重，建议再加半份主食或一份蛋白"})

    # 去重 + 截断（add 优先，其次 reduce）
    seen, picked = set(), []
    for item in sorted(out, key=lambda x: 0 if x["action"] == "add" else 1):
        if item["text"] in seen:
            continue
        seen.add(item["text"])
        picked.append(item)
    return picked[:max_items]


# ====================================================================== 汇总
def _join_cn(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "、".join(items[:-1]) + "和" + items[-1]


def build_summary(checks: List[Dict[str, Any]], targets: Dict[str, Any],
                  totals: Dict[str, Any]) -> str:
    """由规则层写一句「总评」，让大模型照着展开，而不是自己下判断。

    1.5B 的小模型经常把「膳食纤维偏低」复述成「偏高」，这类事实判断不能交给它，
    所以这里先写死一句正确的中文总评，prompt 要求第一节以它开头。
    """
    kcal = float(totals.get("calories", 0) or 0)
    kmin = float(targets.get("kcal_min", 0) or 0)
    kmax = float(targets.get("kcal_max", 0) or 0)
    if kcal < kmin:
        energy = f"热量偏低（{kcal:.0f} kcal，建议 {kmin:.0f}-{kmax:.0f}）"
    elif kcal > kmax:
        energy = f"热量偏高（{kcal:.0f} kcal，建议 {kmin:.0f}-{kmax:.0f}）"
    else:
        energy = f"热量适中（{kcal:.0f} kcal，在建议的 {kmin:.0f}-{kmax:.0f} 之间）"

    # 能量已经单独说过了，避免出现「热量偏低；本餐热量…不足」的重复
    rest = [c for c in checks if c["key"] != "kcal"]
    low = [c["label"] for c in rest if c["status"] == STATUS_LOW]
    high = [c["label"] for c in rest if c["status"] == STATUS_HIGH]
    warn = [c["label"] for c in rest if c["status"] == STATUS_WARN]

    parts = [energy]
    if low:
        parts.append(f"{_join_cn(low)}不足")
    if high:
        parts.append(f"{_join_cn(high)}偏高")
    if warn:
        parts.append(f"{_join_cn(warn)}需要控制")
    if not (low or high or warn):
        parts.append("各项结构指标都达标")
    return "；".join(parts) + "。"


def build_tip(checks: List[Dict[str, Any]], health: Dict[str, Any]) -> str:
    """第 4 节的个性化提示素材（按 BMI + 本次最主要的问题生成）。"""
    bmi_label = str(health.get("bmi_label", ""))
    bmi = health.get("bmi", 0)
    lead = {
        "偏瘦": "体重偏轻，优先保证总能量与蛋白质，不必过度限制主食",
        "正常": "体重在正常范围，重点是吃得均衡而不是少吃",
        "超重": "体重略超，控制烹调油与主食量比单纯节食更有效",
        "肥胖": "体重偏高，建议先减少油炸与含糖饮料，再逐步减少精制主食",
    }.get(bmi_label, "保持规律三餐")

    bad = [c for c in checks if c["status"] != "ok"]
    if not bad:
        return f"BMI {bmi}（{bmi_label}）。{lead}，这一餐的搭配可以照这个思路继续。"
    names = "、".join(c["label"] for c in bad[:2])
    return f"BMI {bmi}（{bmi_label}）。{lead}；这次最需要改的是{names}，下一餐先从这里入手。"


def build_budget(dishes: List[dict], totals: Dict[str, Any], user: Dict[str, Any],
                 health: Dict[str, Any], meal_type: str) -> Dict[str, Any]:
    """一步到位：targets + checks + additions + micro。"""
    dri = load_dri()
    targets = meal_targets(user, health, meal_type, dri)
    ev = evaluate_structure(dishes, totals, targets)
    micro = micronutrient_status(dishes, targets, user, dri)
    additions = suggest_additions(dishes, totals, ev["checks"], targets, micro, ev["groups"])
    return {
        "targets": targets,
        "checks": ev["checks"],
        "groups": ev["groups"],
        "score": ev["score"],
        "good": ev["good"],
        "total": ev["total"],
        "summary": build_summary(ev["checks"], targets, totals),
        "tip": build_tip(ev["checks"], health),
        "micro": micro,
        "micro_daily": micro_daily_text(user, dri),
        "additions": additions,
        "key_points": dri.get("key_points") or [],
        "source": dri.get("_meta", {}).get("source", "中国居民膳食营养素参考摄入量（2023版）"),
    }


if __name__ == "__main__":
    user = {"name": "张三", "age": 22, "gender": "男", "height_cm": 175, "weight_kg": 65}
    health = {"tdee": 2200, "bmr": 1600}
    demo = [
        {"name": "rice", "name_cn": "米饭", "weight_g": 200, "calories": 232, "protein_g": 5.2,
         "fat_g": 0.6, "carb_g": 51.8, "fiber_g": 0.6, "sodium_mg": 4, "category": "主食",
         "micro": {"calcium_mg": 14, "iron_mg": 0.6, "vitamin_c_mg": 0}},
        {"name": "fried_chicken", "name_cn": "炸鸡排", "weight_g": 150, "calories": 420,
         "protein_g": 28.5, "fat_g": 24, "carb_g": 19.5, "fiber_g": 0.9, "sodium_mg": 1050,
         "category": "油炸", "risk": "油炸", "micro": {"calcium_mg": 33, "iron_mg": 1.5}},
    ]
    totals = {
        "weight_g": 350, "calories": 652, "protein_g": 33.7, "fat_g": 24.6, "carb_g": 71.3,
        "fiber_g": 1.5, "sodium_mg": 1054,
    }
    b = build_budget(demo, totals, user, health, "午餐")
    print(f"目标：{b['targets']}")
    print(f"评分：{b['score']} ({b['good']}/{b['total']})")
    for c in b["checks"]:
        print(f"  [{c['status']}] {c['label']}: {c['actual_text']} / {c['target_text']} — {c['hint']}")
    print("该加什么：")
    for a in b["additions"]:
        print(f"  ({a['action']}) {a['text']}")
    print("微量元素最低三项：")
    for m in b["micro"][:3]:
        print(f"  {m['name']}: {m['actual']}{m['unit']} / {m['target']}{m['unit']} ({m['pct']}%)")
