"""BMR / TDEE 计算测试。

运行：python tests/test_health.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.health import BMRCalculator, normalize_gender  # noqa: E402


def test_bmr_male():
    """男 25 岁 175cm 70kg → 1673.8 kcal/天（Mifflin-St Jeor）。"""
    calc = BMRCalculator()
    bmr = calc.calculate_bmr(25, "男", 175, 70)
    assert abs(bmr - 1673.8) < 1.0, bmr
    print(f"[OK] 男性 BMR = {bmr} kcal/天")


def test_bmr_female():
    """女 30 岁 162cm 55kg → 10*55+6.25*162-5*30-161 = 1251.5。"""
    calc = BMRCalculator()
    bmr = calc.calculate_bmr(30, "女", 162, 55)
    assert abs(bmr - 1251.5) < 1.0, bmr
    print(f"[OK] 女性 BMR = {bmr} kcal/天")


def test_tdee_levels():
    """活动系数：1.375 / 1.55 / 1.725 / 1.9。"""
    calc = BMRCalculator()
    expected = {1: 1.375, 2: 1.55, 3: 1.725, 4: 1.9}
    for level, factor in expected.items():
        tdee = calc.calculate_tdee(1673.8, level)
        assert abs(tdee - 1673.8 * factor) < 0.5, (level, tdee)
    print(f"[OK] TDEE：{ {lv: calc.calculate_tdee(1673.8, lv) for lv in expected} }")


def test_full_calculate():
    calc = BMRCalculator()
    r = calc.calculate(25, "男", 175, 70, 1)
    assert abs(r["tdee"] - 2301.5) < 1.0, r
    assert abs(r["bmi"] - 22.9) < 0.1, r
    assert r["bmi_label"] == "正常"
    assert r["activity_name"] == "轻度"
    print(f"[OK] 完整结果：BMR={r['bmr']}, TDEE={r['tdee']}, BMI={r['bmi']}({r['bmi_label']})")


def test_meal_share():
    calc = BMRCalculator()
    assert calc.meal_share(482, 2301.4) == 20.9
    print(f"[OK] 本餐占比：{calc.meal_share(482, 2301.4)}%")


def test_gender_alias():
    assert normalize_gender("男") == "男"
    assert normalize_gender("Male") == "男"
    assert normalize_gender("f") == "女"
    assert normalize_gender("女") == "女"
    try:
        normalize_gender("x")
        raise AssertionError("非法性别应抛 ValueError")
    except ValueError:
        pass
    print("[OK] 性别别名归一")


def test_invalid_input():
    calc = BMRCalculator()
    for args in [(200, "男", 175, 70), (25, "男", 20, 70), (25, "男", 175, 1)]:
        try:
            calc.calculate_bmr(*args)
            raise AssertionError(f"非法输入未报错: {args}")
        except ValueError:
            continue
    try:
        calc.calculate_tdee(1673.8, 9)
        raise AssertionError("非法活动量未报错")
    except ValueError:
        pass
    print("[OK] 非法输入已拦截")


def main() -> int:
    tests = [
        test_bmr_male,
        test_bmr_female,
        test_tdee_levels,
        test_full_calculate,
        test_meal_share,
        test_gender_alias,
        test_invalid_input,
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
    print(f"\n健康计算测试完成：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
