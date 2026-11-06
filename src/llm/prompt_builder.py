"""Prompt 构建：把「识别结果 + 营养估算 + 个人健康数据」组装成中文营养师提示词。"""

from __future__ import annotations

from typing import Sequence

SYSTEM_PROMPT = (
    "你是一位专业的中国注册营养师，擅长根据用户的个人信息和单餐食物记录给出实用、"
    "可执行的膳食建议。回答要具体、温和、不说教，避免绝对化表述，不做疾病诊断。"
)


def _fmt_food_lines(dishes: Sequence[dict]) -> str:
    lines = []
    for idx, d in enumerate(dishes, 1):
        name = d.get("name_cn") or d.get("name", "未知食物")
        weight = float(d.get("weight_g", 0) or 0)
        calories = float(d.get("calories", 0) or 0)
        extra = ""
        if d.get("matched") is False:
            extra = "（营养库未收录，按同类均值估算）"
        lines.append(f"{idx}. {name} — 约 {weight:.0f}g，{calories:.0f} kcal{extra}")
    return "\n".join(lines) if lines else "（未识别到明确食物）"


def build_prompt(
    detections: Sequence[dict],
    weights: Sequence[dict],
    calories: dict,
    user_info: dict,
    bmr_tdee: dict,
    meal_share: float = 0.0,
    extra_context: str = "",
) -> str:
    """构建最终 Prompt（纯文本，便于日志排查）。

    detections : 检测结果（可空，名字会兜底取 weights 里的）
    weights    : PortionEstimator 输出
    calories   : NutritionCalculator.totals() 输出 {weight_g, calories, protein_g, fat_g, carb_g}
    user_info  : {age, gender, height_cm, weight_kg, activity_level}
    bmr_tdee   : BMRCalculator.calculate() 输出
    meal_share : 本餐热量占 TDEE 百分比
    """
    dishes = list(weights) or list(detections)
    total_weight = float(calories.get("weight_g", 0) or 0)
    total_kcal = float(calories.get("calories", 0) or 0)

    tdee = float(bmr_tdee.get("tdee", 0) or 0)
    bmr = float(bmr_tdee.get("bmr", 0) or 0)
    share_text = f"{meal_share:.1f}%"
    remain_text = f"{max(tdee - total_kcal, 0):.0f} kcal" if tdee else "未知"

    prompt = f"""你是专业营养师。请根据以下信息给出一份单餐健康膳食建议。

【用户信息】
- 年龄：{user_info.get('age')} 岁
- 性别：{user_info.get('gender')}
- 身高：{user_info.get('height_cm')} cm
- 体重：{user_info.get('weight_kg')} kg
- 活动量：{bmr_tdee.get('activity_name', user_info.get('activity_level', ''))}（活动系数 {bmr_tdee.get('activity_factor', '')}）
- BMI：{bmr_tdee.get('bmi', '')}（{bmr_tdee.get('bmi_label', '')}）
- BMR（基础代谢）：{bmr:.0f} kcal/天
- TDEE（每日消耗）：{tdee:.0f} kcal/天

【本餐识别结果】
{_fmt_food_lines(dishes)}

- 总重量：{total_weight:.0f} g
- 总热量：{total_kcal:.0f} kcal
- 估算蛋白质 {float(calories.get('protein_g', 0) or 0):.0f} g / 脂肪 {float(calories.get('fat_g', 0) or 0):.0f} g / 碳水 {float(calories.get('carb_g', 0) or 0):.0f} g
- 本餐热量占 TDEE：{share_text}
- 按 TDEE 计算，本餐后剩余可摄入约：{remain_text}
{extra_context}

【请完成以下四项】
1. 本餐营养评估：热量、蛋白质、脂肪、碳水配比是否合理，指出最突出的 1-2 个问题。
2. 与用户每日需求的匹配度：结合 BMR/TDEE 和 BMI，判断这一餐的分量是否合适（偏多/偏少/合适）。
3. 食用搭配建议：给出 2-3 条具体可操作的搭配调整（例如加什么、减什么、替换什么）。
4. 针对该用户的健康建议：结合年龄、性别、体重与 BMI 给出 2-3 条个性化提示。

【输出要求】
- 用 Markdown 输出，四个小节标题分别为「一、本餐营养评估」「二、与每日需求的匹配」「三、搭配建议」「四、个性化健康建议」
- 语言简洁，中文，总长度控制在 450 字以内
- 数字带单位，建议要具体（不要只说“多吃蔬菜”，要说“加一份 100g 的清炒青菜”）
- 直接输出正文：不要用 ``` 代码块包裹，不要输出 YAML/front-matter（title、date、layout 等），
  不要输出免责声明之外的前言，直接从「一、本餐营养评估」开始
"""
    return prompt


if __name__ == "__main__":
    demo_weights = [
        {"name": "chicken", "weight_g": 150, "calories": 248},
        {"name": "rice", "weight_g": 180, "calories": 234},
    ]
    demo_user = {"age": 25, "gender": "男", "height_cm": 175, "weight_kg": 70, "activity_level": 1}
    demo_health = {"bmr": 1673.8, "tdee": 2301.4, "activity_name": "轻度", "activity_factor": 1.375,
                   "bmi": 22.9, "bmi_label": "正常"}
    print(build_prompt([], demo_weights, {"weight_g": 330, "calories": 482}, demo_user, demo_health, 20.9))
