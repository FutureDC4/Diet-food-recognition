"""BMR / TDEE 计算（Mifflin-St Jeor 公式）。

男性 BMR = 10×体重(kg) + 6.25×身高(cm) - 5×年龄 + 5
女性 BMR = 10×体重(kg) + 6.25×身高(cm) - 5×年龄 - 161

TDEE = BMR × 活动系数
    1 轻度 1.375 / 2 中度 1.55 / 3 重度 1.725 / 4 极重 1.9
"""

from __future__ import annotations

from typing import Dict, Optional

from ..utils.config_loader import activity_levels
from ..utils.logger import get_logger

log = get_logger("健康计算")

_MALE_ALIASES = {"男", "男性", "m", "male", "man", "1"}
_FEMALE_ALIASES = {"女", "女性", "f", "female", "woman", "0"}


def normalize_gender(gender: str) -> str:
    """把各种写法统一成 '男' / '女'。"""
    g = str(gender).strip().lower()
    if g in _MALE_ALIASES:
        return "男"
    if g in _FEMALE_ALIASES:
        return "女"
    raise ValueError(f"无法识别的性别：{gender}（请输入 男/女）")


class BMRCalculator:
    """基础代谢与每日总消耗。"""

    def __init__(self, levels: Optional[Dict[int, dict]] = None):
        self.levels = levels or activity_levels()

    # ------------------------------------------------------------------
    @staticmethod
    def calculate_bmr(age: int, gender: str, height_cm: float, weight_kg: float) -> float:
        """Mifflin-St Jeor 基础代谢率（kcal/天）。"""
        age = int(age)
        height_cm = float(height_cm)
        weight_kg = float(weight_kg)
        if not (0 < age < 130):
            raise ValueError(f"年龄不合理：{age}")
        if not (80 <= height_cm <= 250):
            raise ValueError(f"身高不合理：{height_cm} cm")
        if not (10 <= weight_kg <= 350):
            raise ValueError(f"体重不合理：{weight_kg} kg")

        g = normalize_gender(gender)
        base = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age
        return round(base + (5.0 if g == "男" else -161.0), 1)

    # ------------------------------------------------------------------
    def calculate_tdee(self, bmr: float, activity_level: int) -> float:
        """每日总能量消耗（kcal/天）。"""
        level = int(activity_level)
        if level not in self.levels:
            raise ValueError(f"活动量只能是 {sorted(self.levels)} 之一，收到：{activity_level}")
        return round(float(bmr) * float(self.levels[level]["factor"]), 1)

    # ------------------------------------------------------------------
    def calculate(
        self,
        age: int,
        gender: str,
        height_cm: float,
        weight_kg: float,
        activity_level: int = 1,
    ) -> dict:
        """一次性算出 BMR / TDEE / BMI 等派生指标。"""
        g = normalize_gender(gender)
        bmr = self.calculate_bmr(age, g, height_cm, weight_kg)
        tdee = self.calculate_tdee(bmr, activity_level)
        height_m = float(height_cm) / 100.0
        bmi = round(float(weight_kg) / (height_m ** 2), 1)
        level = self.levels[int(activity_level)]

        result = {
            "age": int(age),
            "gender": g,
            "height_cm": float(height_cm),
            "weight_kg": float(weight_kg),
            "activity_level": int(activity_level),
            "activity_name": level["name"],
            "activity_factor": float(level["factor"]),
            "bmr": bmr,
            "tdee": tdee,
            "bmi": bmi,
            "bmi_label": self._bmi_label(bmi),
        }
        log.info(
            f"健康计算：{g} {age}岁 {height_cm}cm {weight_kg}kg 活动量{activity_level}({level['name']}) "
            f"→ BMR {bmr} kcal/天，TDEE {tdee} kcal/天，BMI {bmi}"
        )
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _bmi_label(bmi: float) -> str:
        if bmi < 18.5:
            return "偏瘦"
        if bmi < 24:
            return "正常"
        if bmi < 28:
            return "超重"
        return "肥胖"

    @staticmethod
    def meal_share(meal_calories: float, tdee: float) -> float:
        """本餐热量占每日总消耗的百分比。"""
        if not tdee or tdee <= 0:
            return 0.0
        return round(float(meal_calories) / float(tdee) * 100.0, 1)


if __name__ == "__main__":
    calc = BMRCalculator()
    print(calc.calculate(25, "男", 175, 70, 1))
    print(calc.calculate(25, "女", 162, 55, 2))
