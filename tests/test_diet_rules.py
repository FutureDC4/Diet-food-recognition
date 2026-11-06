"""米饭三级处理 + 餐次配额 + 结构评估 测试。

运行：python tests/test_diet_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.detection_rules import apply_rice_rule, height_multiplier  # noqa: E402
from app.diet_rules import build_budget, meal_targets              # noqa: E402

USER = {"name": "测试", "age": 22, "gender": "男", "height_cm": 175, "weight_kg": 65}
HEALTH = {"tdee": 2400, "bmr": 1650, "bmi": 21.2, "bmi_label": "正常"}


def _dets(rice_area: float, other_area: float):
    return ([{"name": "rice", "area_px": rice_area}]
            + [{"name": f"dish{i}", "area_px": other_area / 2} for i in range(2)])


# ---------------------------------------------------------------- 米饭三级
def test_rice_level1_trusted():
    """占比 40% ≥ 20%：一级，直接用检测值，不改面积。"""
    out, info = apply_rice_rule(_dets(40, 60))
    assert info["level"] == 1 and info["applied"] is False, info
    assert len(out) == 3
    assert abs(out[0]["area_px"] - 40) < 1e-6, out[0]
    print(f"[OK] 一级（{info['detected_ratio']:.0%}）：{info['note']}")


def test_rice_level2_adjusted():
    """占比 15%（4%~20%）：二级，修正后应正好占食物总面积 1/3。"""
    out, info = apply_rice_rule(_dets(15, 85))
    assert info["level"] == 2 and info["applied"] is True, info
    total = sum(d["area_px"] for d in out)
    ratio = out[0]["area_px"] / total
    assert abs(ratio - 1 / 3) < 0.01, f"修正后占比 {ratio:.3f}，应约为 1/3"
    assert out[0].get("rule_adjusted") is True
    print(f"[OK] 二级（15% → {ratio:.0%}）：{info['note']}")


def test_rice_level3_rejected():
    """占比 3% < 4%：三级，判为误检并剔除。"""
    out, info = apply_rice_rule(_dets(3, 97))
    assert info["level"] == 3 and info.get("rejected") is True, info
    assert all(d["name"] != "rice" for d in out), out
    print(f"[OK] 三级（3%）：{info['note']}")


def test_rice_absent():
    out, info = apply_rice_rule([{"name": "curry", "area_px": 100}])
    assert info["level"] == 0 and len(out) == 1
    print(f"[OK] 无米饭：{info['note']}")


def test_height_levels():
    assert height_multiplier("常规")[0] == 1.0
    assert height_multiplier("薄摊")[0] < 1.0
    assert height_multiplier("堆高")[0] > 1.0
    assert height_multiplier("乱填")[1] == "常规"           # 未知回落到常规
    print(f"[OK] 高度档位：薄摊={height_multiplier('薄摊')[0]}, "
          f"常规={height_multiplier('常规')[0]}, 堆高={height_multiplier('堆高')[0]}")


# ---------------------------------------------------------------- 餐次配额
def test_meal_share_exact():
    """早 25% / 午 40% / 晚 35%，且三者之和正好 100%。"""
    shares = [meal_targets(USER, HEALTH, m)["meal_share"] for m in ("早餐", "午餐", "晚餐")]
    assert shares == [0.25, 0.40, 0.35], shares
    assert abs(sum(shares) - 1.0) < 1e-9
    print(f"[OK] 餐次占比：{shares}，合计 {sum(shares):.0%}")


def test_meal_kcal_range():
    """午餐 2400 kcal × 40% = 960，区间 ±10%。"""
    t = meal_targets(USER, HEALTH, "午餐")
    assert abs(t["kcal_target"] - 960) < 1, t
    assert abs(t["kcal_min"] - 864) < 1 and abs(t["kcal_max"] - 1056) < 1, t
    print(f"[OK] 午餐预算：{t['kcal_min']:.0f}-{t['kcal_max']:.0f} kcal")


def test_protein_target_matches_guide():
    """65 kg 男性：全天 1.0 g/kg = 65 g，午餐 40% → 26 g（指南建议 25-30g）。"""
    t = meal_targets(USER, HEALTH, "午餐")
    assert 25 <= t["protein_g"] <= 30, t
    assert abs(t["fiber_g"] - 10.0) < 0.5, t      # 25 g/d × 40%
    print(f"[OK] 午餐目标：蛋白 {t['protein_g']} g，纤维 {t['fiber_g']} g，蔬菜 {t['veg_g']:.0f} g")


# ---------------------------------------------------------------- 结构评估
def _meal(kind: str):
    """两种典型餐：油炸单一餐 / 搭配合理的餐。"""
    if kind == "fried":
        dishes = [
            {"name": "rice", "name_cn": "米饭", "weight_g": 200, "calories": 232,
             "protein_g": 5.2, "fat_g": 0.6, "carb_g": 51.8, "fiber_g": 0.6,
             "sodium_mg": 4, "category": "主食",
             "micro": {"calcium_mg": 14, "iron_mg": 0.6, "vitamin_c_mg": 0}},
            {"name": "fried_chicken", "name_cn": "炸鸡排", "weight_g": 150, "calories": 420,
             "protein_g": 28.5, "fat_g": 24, "carb_g": 19.5, "fiber_g": 0.9,
             "sodium_mg": 1050, "category": "油炸", "risk": "油炸",
             "micro": {"calcium_mg": 33, "iron_mg": 1.5}},
        ]
        totals = {"weight_g": 350, "calories": 652, "protein_g": 33.7, "fat_g": 24.6,
                  "carb_g": 71.3, "fiber_g": 1.5, "sodium_mg": 1054}
    else:
        dishes = [
            {"name": "rice", "name_cn": "米饭", "weight_g": 180, "calories": 208,
             "protein_g": 4.7, "fat_g": 0.5, "carb_g": 46.6, "fiber_g": 0.5,
             "sodium_mg": 4, "category": "主食", "micro": {"calcium_mg": 12}},
            {"name": "water_spinach", "name_cn": "炒空心菜", "weight_g": 200, "calories": 110,
             "protein_g": 4.4, "fat_g": 6, "carb_g": 6, "fiber_g": 2.8,
             "sodium_mg": 560, "category": "蔬菜", "micro": {"calcium_mg": 198}},
            {"name": "fried_eggs", "name_cn": "炒蛋", "weight_g": 100, "calories": 180,
             "protein_g": 12, "fat_g": 13.5, "carb_g": 1.5, "fiber_g": 0,
             "sodium_mg": 350, "category": "荤菜", "micro": {"calcium_mg": 55}},
            {"name": "yogurt", "name_cn": "无糖酸奶", "weight_g": 150, "calories": 93,
             "protein_g": 5.3, "fat_g": 4.5, "carb_g": 7.5, "fiber_g": 0,
             "sodium_mg": 68, "category": "奶类", "micro": {"calcium_mg": 177}},
        ]
        totals = {"weight_g": 630, "calories": 591, "protein_g": 26.4, "fat_g": 24.5,
                  "carb_g": 61.6, "fiber_g": 3.3, "sodium_mg": 982}
    return dishes, totals


def test_fried_meal_flagged():
    dishes, totals = _meal("fried")
    b = build_budget(dishes, totals, USER, HEALTH, "午餐")
    by = {c["key"]: c for c in b["checks"]}
    assert by["fiber"]["status"] == "low", by["fiber"]
    assert by["veg"]["status"] == "low", by["veg"]
    assert by["fried"]["status"] == "warn", by["fried"]
    assert by["sodium"]["status"] == "high", by["sodium"]
    assert len(b["additions"]) >= 2
    assert "不足" in b["summary"] or "偏高" in b["summary"]
    print(f"[OK] 油炸餐：得分 {b['score']}，问题 = {b['summary']}")


def test_balanced_meal_scores_higher():
    bad = build_budget(*_meal("fried"), USER, HEALTH, "午餐")
    good = build_budget(*_meal("balanced"), USER, HEALTH, "午餐")
    assert good["score"] > bad["score"], (good["score"], bad["score"])
    print(f"[OK] 搭配合理的餐得分 {good['score']} > 油炸餐 {bad['score']}")


def test_summary_never_contradicts():
    """总评里不能同时出现「纤维不足」和「纤维偏高」。"""
    for kind in ("fried", "balanced"):
        b = build_budget(*_meal(kind), USER, HEALTH, "午餐")
        s = b["summary"]
        assert not ("膳食纤维" in s and "膳食纤维不足" in s and "膳食纤维偏高" in s), s
    print("[OK] 总评判断自洽")


def test_veg_ratio_uses_weight():
    dishes, totals = _meal("balanced")
    b = build_budget(dishes, totals, USER, HEALTH, "午餐")
    g = b["groups"]
    assert 0 < g["veg_weight_ratio"] < 1
    assert abs(g["veg_weight_ratio"] - 200 / 630) < 0.01, g
    print(f"[OK] 蔬菜重量占比 {g['veg_weight_ratio']:.0%}（200/630）")


def main() -> int:
    tests = [
        test_rice_level1_trusted, test_rice_level2_adjusted,
        test_rice_level3_rejected, test_rice_absent, test_height_levels,
        test_meal_share_exact, test_meal_kcal_range, test_protein_target_matches_guide,
        test_fried_meal_flagged, test_balanced_meal_scores_higher,
        test_summary_never_contradicts, test_veg_ratio_uses_weight,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {t.__name__}: {exc.__class__.__name__}: {exc}")
    print(f"\n膳食规则测试完成：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
