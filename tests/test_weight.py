"""重量估计 + 营养计算测试。

运行：python tests/test_weight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weight_estimation import NutritionCalculator, PortionEstimator  # noqa: E402


def test_density_lookup():
    est = PortionEstimator()
    assert est.density_of("rice") == 0.85
    assert est.density_of("Rice") == 0.85
    assert est.density_of("红烧肉") == 0.80            # 中文名未收录 -> 默认
    assert est.density_of("hongshaorou") == 1.05       # 中餐预留条目可命中
    assert est.density_of("unknown_food_xyz") == 0.80  # 未知 -> default
    print(f"[OK] 密度查表：rice={est.density_of('rice')}, unknown={est.density_of('unknown_food_xyz')}")


def test_weight_estimate_reasonable():
    """约 80cm² 的米饭（≈9cm×9cm），估算应落在 100-300 g。"""
    est = PortionEstimator()
    px_per_cm = 12.0
    det = [{"name": "rice", "conf": 0.9, "bbox": [0, 0, 12 * 9, 12 * 9], "area_px": 80 * px_per_cm ** 2}]
    out = est.estimate(px_per_cm=px_per_cm, food_detections=det)
    assert len(out) == 1
    w = out[0]["weight_g"]
    assert 100 <= w <= 300, f"米饭重量不合理: {w}"
    print(f"[OK] 米饭重量估计：{w} g（面积 {out[0]['area_cm2']} cm²，高度 {out[0]['height_cm']} cm，密度 {out[0]['density']}）")


def test_weight_clamped():
    """极小/极大面积要被 min/单份上限夹住，避免爆表。"""
    est = PortionEstimator()
    tiny = est.estimate(12.0, [{"name": "rice", "conf": 0.5, "area_px": 4.0}])
    huge = est.estimate(12.0, [{"name": "rice", "conf": 0.5, "area_px": 10_000_000}])
    assert tiny[0]["weight_g"] >= est.min_weight_g
    assert huge[0]["weight_g"] <= est.max_weight_for("rice")
    assert est.max_weight_for("cake") < est.max_weight_for("pizza")
    print(f"[OK] 夹取：tiny={tiny[0]['weight_g']}g, huge={huge[0]['weight_g']}g"
          f"（rice 上限 {est.max_weight_for('rice')}g / cake 上限 {est.max_weight_for('cake')}g）")


def test_plate_area_constraint():
    """食物面积总和超过盘子面积时，应整体收缩。"""
    est = PortionEstimator()
    dets = [
        {"name": "apple", "conf": 0.9, "area_px": 100 * 100},
        {"name": "banana", "conf": 0.9, "area_px": 100 * 100},
    ]
    free = est.estimate(10.0, dets)
    capped = est.estimate(10.0, dets, plate_area_cm2=50.0)   # 盘子很小，必然触发收缩
    assert capped[0]["area_shrink"] < 1.0, capped
    assert capped[0]["area_cm2"] < free[0]["area_cm2"]
    print(f"[OK] 盘子面积约束：无约束 {free[0]['area_cm2']}cm² → 收缩 {capped[0]['area_cm2']}cm²"
          f"（factor {capped[0]['area_shrink']}）")


def test_scale_effect():
    """尺度翻倍（同样像素面积）→ 重量变为 1/4。"""
    est = PortionEstimator()
    det = [{"name": "chicken", "conf": 0.9, "area_px": 20000}]
    w1 = est.estimate(10.0, det)[0]["weight_g"]
    w2 = est.estimate(20.0, det)[0]["weight_g"]
    assert abs(w1 / w2 - 4) < 0.05, f"尺度平方关系不成立: {w1} vs {w2}"
    print(f"[OK] 尺度平方关系：10px/cm -> {w1}g，20px/cm -> {w2}g")


def test_nutrition_calc():
    calc = NutritionCalculator()
    # 白米饭按《中国食物成分表》第 6 版 116 kcal/100g（早期版本曾用 130）
    r = calc.calculate("rice", 180)
    assert abs(r["calories"] - 208.8) < 1.0, r
    assert abs(r["protein_g"] - 4.7) < 0.2, r
    assert abs(r["fiber_g"] - 0.5) < 0.2, r
    assert r["category"] == "主食" and r["cooking"] == "蒸", r
    c = calc.calculate("chicken", 150)
    assert abs(c["calories"] - 247.5) < 1.0, c
    unknown = calc.calculate("totally_unknown", 100)
    assert unknown["matched"] is False and unknown["calories"] == 200.0
    print(f"[OK] 营养：rice 180g={r['calories']} kcal（蛋白 {r['protein_g']}g）, "
          f"chicken 150g={c['calories']} kcal, 未知食物={unknown['calories']} kcal")


def test_nutrition_totals():
    calc = NutritionCalculator()
    items = calc.calculate_many([{"name": "rice", "weight_g": 180}, {"name": "chicken", "weight_g": 150}])
    t = calc.totals(items)
    assert abs(t["calories"] - 456.3) < 2, t
    assert t["protein_g"] > 0 and t["carb_g"] > 0
    assert "fiber_g" in t and "sodium_mg" in t
    print(f"[OK] 汇总：{t}")


def main() -> int:
    tests = [
        test_density_lookup,
        test_weight_estimate_reasonable,
        test_weight_clamped,
        test_plate_area_constraint,
        test_scale_effect,
        test_nutrition_calc,
        test_nutrition_totals,
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
    print(f"\n重量/营养测试完成：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
