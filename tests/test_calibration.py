"""营养合理性校正层 测试。

运行：python tests/test_calibration.py

关注三件事：
  1. 折算只降不升，且幅度在预设区间内（不是随便打折）；
  2. 蒸煮类（腊肠）几乎不扣油，清炒类扣得最多——方向要符合常识；
  3. 关闭开关后必须逐字回到原始值，保证可回退。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.calibration import calibrate, calibrate_dish, load_calibration  # noqa: E402

CFG = load_calibration()

SAUSAGE = {"name": "chinese_sausage", "name_cn": "腊肠", "category": "加工肉", "cooking": "蒸",
           "weight_g": 100.0, "calories": 400.0, "fat_g": 33.0, "protein_g": 18.0,
           "carb_g": 4.0, "fiber_g": 0.0, "sodium_mg": 1200.0, "micro": {"calcium_mg": 20}}
CABBAGE = {"name": "chinese_cabbage", "name_cn": "炒白菜", "category": "蔬菜", "cooking": "清炒",
           "weight_g": 100.0, "calories": 55.0, "fat_g": 3.2, "protein_g": 1.5,
           "carb_g": 3.0, "fiber_g": 1.0, "sodium_mg": 250.0, "micro": {"vitamin_c_mg": 20}}


def test_disabled_returns_original():
    """关掉校正 → 数值一字不改。"""
    out, info = calibrate([SAUSAGE], {**CFG, "enabled": False})
    assert info["enabled"] is False
    for key in ("weight_g", "calories", "fat_g", "sodium_mg"):
        assert out[0][key] == SAUSAGE[key], (key, out[0][key], SAUSAGE[key])
    print("[OK] 关闭开关后数值与原始一致")


def test_monotonic_decrease():
    """所有校正后的值都 ≤ 原始值，且不为负。"""
    out = calibrate_dish(SAUSAGE, CFG)
    for key in ("weight_g", "calories", "fat_g", "sodium_mg", "protein_g", "fiber_g"):
        assert 0 <= out[key] <= SAUSAGE[key] + 1e-6, (key, out[key])
    print(f"[OK] 腊肠 100g → {out['weight_g']}g / {out['calories']}kcal / 脂肪 {out['fat_g']}g")


def test_steam_barely_trims_oil():
    """蒸制食物脂肪几乎不扣（油不在表面），只扣可食部。"""
    out = calibrate_dish(SAUSAGE, CFG)
    assert out["fat_g"] >= SAUSAGE["fat_g"] * 0.95, out["fat_g"]
    print(f"[OK] 蒸制脂肪仅降 {100 - out['fat_g'] / SAUSAGE['fat_g'] * 100:.1f}%")


def test_stirfry_trims_more_than_steam():
    """清炒的脂肪折算比例应大于蒸制（浮油多）。"""
    stir = calibrate_dish(CABBAGE, CFG)
    steam = calibrate_dish(SAUSAGE, CFG)
    r_stir = stir["fat_g"] / CABBAGE["fat_g"]
    r_steam = steam["fat_g"] / SAUSAGE["fat_g"]
    assert r_stir < r_steam, (r_stir, r_steam)
    print(f"[OK] 清炒保留 {r_stir:.0%} 脂肪 < 蒸制保留 {r_steam:.0%}")


def test_sodium_kept_inside_food():
    """蒸制/腌制：盐在食材里，几乎不该扣（腊肠咸是腌进去的）。"""
    out = calibrate_dish(SAUSAGE, CFG)
    keep = out["sodium_mg"] / SAUSAGE["sodium_mg"]
    assert keep >= 0.90, keep
    print(f"[OK] 蒸腊肠的钠保留 {keep:.0%}")


def test_sodium_trimmed_in_stirfry():
    """清炒：盐在汤汁里，会随盘底汤水留下 20% 以上。"""
    out = calibrate_dish(CABBAGE, CFG)
    keep = out["sodium_mg"] / CABBAGE["sodium_mg"]
    want = CFG["sodium_retention"]["清炒"] * CFG["edible_ratio"][CABBAGE["category"]]
    assert keep < 0.75 and abs(keep - want) < 0.02, (keep, want)
    print(f"[OK] 清炒白菜的钠保留 {keep:.0%}（期望 {want:.0%}）")


def test_energy_consistent_with_fat_change():
    """热量只按去掉的脂肪能量下降，不多扣。"""
    out = calibrate_dish(SAUSAGE, CFG)
    edible = out["calibration"]["edible_ratio"]
    fat_gross = out["fat_g"] / edible                       # 还原成装盘口径再比
    removed_kcal = (SAUSAGE["fat_g"] - fat_gross) * 9
    expect = (SAUSAGE["calories"] - removed_kcal) * edible
    assert abs(out["calories"] - expect) < 2.0, (out["calories"], expect)
    print(f"[OK] 热量 {SAUSAGE['calories']:.0f} → {out['calories']:.0f} kcal（其中去油 "
          f"{removed_kcal:.1f} kcal，其余为可食部折算）")


def test_gross_values_kept():
    """装盘值必须保留，便于报告追溯与调参。"""
    out = calibrate_dish(SAUSAGE, CFG)
    assert out["gross_weight_g"] == 100.0 and out["gross_fat_g"] == 33.0
    assert out["gross_sodium_mg"] == 1200.0 and out["calibration"]["edible_ratio"] > 0
    print(f"[OK] 保留装盘值：{out['gross_weight_g']}g / 脂肪 {out['gross_fat_g']}g")


def main() -> int:
    tests = [
        test_disabled_returns_original, test_monotonic_decrease, test_steam_barely_trims_oil,
        test_stirfry_trims_more_than_steam, test_sodium_kept_inside_food,
        test_sodium_trimmed_in_stirfry,
        test_energy_consistent_with_fat_change, test_gross_values_kept,
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
    print(f"\n营养校正测试完成：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
