# -*- coding: utf-8 -*-
"""上位机使用的膳食建议 Prompt。

与原 prompt_builder 的差别：
  1. 带入「餐次」（早餐/午餐/晚餐）和**该餐次的能量配额**，让建议有明确基准；
  2. 主体是**单餐建议**——这一餐缺什么、马上补什么；
  3. 结构判断（蛋白/纤维/蔬菜/主食/钠/促炎）已经在 :mod:`app.diet_rules` 里算死，
     prompt 只把这套结论交给模型组织语言，不让小模型自己算数；
  4. 补充每道菜的烹调方式、促炎倾向、关键微量元素，以及中国居民膳食指南的依据条目，
     让建议能落到「为什么」。

为什么要「算死再交给模型」：本机跑的是 1.5B 模型（CPU 推理），
让它自己做减法（「目标 28g，实际 18g，还差多少」）错误率很高，
所以数值结论一律由规则层给出，模型只负责表达与取舍。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

SYSTEM_PROMPT = (
    "你是一位专业的中国注册营养师，熟悉食堂与家常菜，熟悉《中国居民膳食指南（2022）》"
    "和《中国居民膳食营养素参考摄入量（2023版）》。你的建议要具体、可执行、语气温和，"
    "不说教、不做疾病诊断。你只依据给出的数据说话，不臆造未检测到的食物。"
    "输入中【已判定的结论】由系统依据膳食指南算出，你必须原样采用其中的"
    "「偏高／偏低／不足／达标」判断和所有数字，不得自行重新判断或给出相反结论；"
    "你的任务是把这些结论解释清楚、变成能马上执行的建议。"
)


def _food_lines(dishes: Sequence[dict]) -> str:
    if not dishes:
        return "（未识别到明确食物）"
    lines = []
    for i, d in enumerate(dishes, 1):
        line = (f"{i}. {d.get('name_cn') or d.get('name')}（{d.get('cooking', '未知')}）"
                f" {float(d.get('weight_g', 0)):.0f}g · {float(d.get('calories', 0)):.0f}kcal"
                f" · 蛋白 {float(d.get('protein_g', 0)):.0f}g"
                f" 脂肪 {float(d.get('fat_g', 0)):.0f}g"
                f" 碳水 {float(d.get('carb_g', 0)):.0f}g"
                f" 纤维 {float(d.get('fiber_g', 0)):.1f}g"
                f" 钠 {float(d.get('sodium_mg', 0)):.0f}mg")
        flags = []
        if d.get("rule_adjusted"):
            flags.append("米饭面积按占比规则修正")
        if d.get("inflammation") == "高":
            flags.append("促炎负荷高")
        if d.get("risk"):
            flags.append(f"属{d['risk']}类")
        if flags:
            line += "（" + "，".join(flags) + "）"
        lines.append(line)
    return "\n".join(lines)


def _check_lines(checks: Sequence[dict]) -> str:
    if not checks:
        return "（结构评估不可用）"
    mark = {"ok": "达标", "low": "偏低", "high": "偏高", "warn": "注意"}
    out = []
    for c in checks:
        out.append(f"- {c['label']}：{c['actual_text']}，参考 {c['target_text']}"
                   f" → {mark.get(c['status'], c['status'])}。{c.get('hint', '')}")
    return "\n".join(out)


def _micro_lines(micro: Sequence[dict], limit: int = 4) -> str:
    if not micro:
        return "（无微量元素数据）"
    low = [m for m in micro if m["pct"] < 40][:limit]
    if not low:
        return "这一餐的微量元素覆盖较全面，无需特别补充。"
    return "；".join(f"{m['name']} {m['actual']}{m['unit']}（本餐目标 {m['target']}{m['unit']}，"
                    f"完成 {m['pct']}%）" for m in low)


def _addition_lines(additions: Sequence[dict]) -> str:
    if not additions:
        return "（没有明显缺口）"
    out = []
    for a in additions:
        prefix = "加" if a.get("action") == "add" else "减"
        out.append(f"- {prefix}：{a.get('text')}")
    return "\n".join(out)


def _verdict_lines(checks: Sequence[dict]) -> str:
    """把非达标项压成「必须原样采用」的结论清单。

    1.5B 的小模型很容易把「纤维 6.2g / 目标 10g」读成「基本达标」，
    所以这里单独列一份结论，并要求它照抄判断、不许改判。
    """
    bad = [c for c in checks if c.get("status") != "ok"]
    if not bad:
        return "这一餐各项结构指标均达标（热量、蛋白质、纤维、蔬菜、主食、脂肪、钠、促炎负荷）。"
    mark = {"low": "偏低", "high": "偏高", "warn": "需要控制", "ok": "达标"}
    return "\n".join(
        f"- {c['label']}：{mark.get(c['status'], c['status'])}"
        f"（实际 {c['actual_text']}，参考 {c['target_text']}）"
        for c in bad
    )


def build_meal_prompt(
    result: Dict[str, Any],
    day: Dict[str, Any] | None = None,
    rice_note: str = "",
) -> str:
    """result 为 MealAnalyzer.analyze() 的输出。"""
    user = result.get("user", {})
    health = result.get("health", {})
    totals = result.get("totals", {})
    dishes = result.get("dishes", [])
    meal_type = result.get("meal_type", "本餐")
    budget = result.get("budget") or {}
    targets = budget.get("targets") or {}
    day = day or {}

    tdee = float(health.get("tdee", 0) or 0)
    kcal = float(totals.get("calories", 0) or 0)
    protein = float(totals.get("protein_g", 0) or 0)
    fat = float(totals.get("fat_g", 0) or 0)
    carb = float(totals.get("carb_g", 0) or 0)
    fiber = float(totals.get("fiber_g", 0) or 0)
    sodium = float(totals.get("sodium_mg", 0) or 0)

    kcal_text = (f"{targets.get('kcal_min', 0):.0f}-{targets.get('kcal_max', 0):.0f} kcal"
                 f"（TDEE {tdee:.0f} × {targets.get('meal_share_text', '')}）"
                 if targets else "—")

    day_lines = ""
    if day.get("count"):
        meals = day.get("meals", {})
        meal_txt = "、".join(f"{k} {v:.0f} kcal" for k, v in meals.items()) or "无"
        day_lines = (
            f"- 今日已记录 {day['count']} 餐：{meal_txt}\n"
            f"- 今日累计热量约 {day['calories']:.0f} kcal"
            f"（占每日需求 {day['calories'] / tdee * 100:.0f}%），"
            f"蛋白质累计约 {day.get('protein_g', 0):.0f} g\n"
        )
    else:
        day_lines = "- 今日暂无其他记录（这是今天的第一餐记录）\n"

    rice_line = (f"\n- 数据处理说明（仅用于你理解数字，**不要写进建议**）：{rice_note}"
                 if rice_note else "")

    return f"""你是营养师。下面是用户「{user.get('name', '')}」的{meal_type}识别与营养估算结果。
这些数字已经由系统算好，请直接引用，不要重新计算。

【用户】
- {user.get('gender')}，{user.get('age')} 岁，{user.get('height_cm')} cm，{user.get('weight_kg')} kg
- BMI {health.get('bmi')}（{health.get('bmi_label')}）；BMR {float(health.get('bmr', 0)):.0f} kcal，TDEE {tdee:.0f} kcal/天
- 活动量：{health.get('activity_name', '')}
- 每日微量元素参考（DRIs 2023，已按性别年龄调整）：{budget.get('micro_daily') or '—'}

【这一餐（{meal_type}）的参考配额】
- 热量：{kcal_text}
- 蛋白质：约 {targets.get('protein_g', 0):.0f} g（按全天 {targets.get('protein_daily_g', 0):.0f} g 折算）
- 膳食纤维：约 {targets.get('fiber_g', 0):.1f} g
- 蔬菜：够 {targets.get('veg_g', 0):.0f} g、占本餐重量 {float(targets.get('veg_ratio_min', 0.3)) * 100:.0f}% 以上
- 主食供能比：{float(targets.get('staple_ratio_min', 0.25)) * 100:.0f}%-{float(targets.get('staple_ratio_max', 0.55)) * 100:.0f}%

【这一餐识别结果】
{_food_lines(dishes)}

合计：{float(totals.get('weight_g', 0)):.0f} g，{kcal:.0f} kcal；
蛋白 {protein:.0f} g / 脂肪 {fat:.0f} g / 碳水 {carb:.0f} g / 纤维 {fiber:.1f} g / 钠 {sodium:.0f} mg{rice_line}

【已判定的结论——必须原样采用，不许改判】
总评：{budget.get('summary') or '（无）'}
{_verdict_lines(budget.get('checks') or [])}

【各项明细（供你解释原因时引用）】
{_check_lines(budget.get('checks') or [])}

【本餐相对薄弱的微量元素】
{_micro_lines(budget.get('micro') or [])}

【系统给出的补充建议（可改写措辞，不要改变食物与分量）】
{_addition_lines(budget.get('additions') or [])}

【今日整体】
{day_lines}
【判断依据（可引用其中一条来支撑建议，不要整段照抄）】
- 每天蔬菜 300-500g，深色蔬菜占一半以上
- 每天奶及奶制品 300-500g，是钙的第一来源；中式饮食普遍钙不足
- 每天大豆及坚果 25-35g，提供植物蛋白与不饱和脂肪酸
- 烹调油 25-30g/天、盐 <5g/天，食堂菜普遍超标
- 油炸食品产生丙烯酰胺与 AGEs，与慢性低度炎症相关，每周 1-2 次为宜
- 加工肉制品（腊肠等）被 IARC 列为 1 类致癌物，尽量少吃
- 维生素 C 与含铁食物同食可把非血红素铁吸收率提高 2-3 倍
- 精白米面损失 70% 以上 B 族维生素与膳食纤维，可用全谷物/薯类替代一部分

【请按以下四节输出，每节 2-4 句】
「一、这餐怎么样」：**第一句直接用下面这句总评**，一字不改：
  「{budget.get('summary') or ''}」
  然后**只补一句**，解释其中**最主要那一个**问题对健康的影响
  （例如长期高钠与血压、油炸与炎症、纤维不足与肠道），
  **不要逐项解释每一个指标**，也不要把总评里的各项再复述一遍。
  **严禁出现与结论清单相反的判断**，例如清单说膳食纤维不足就不能说"纤维达标/偏高"。
「二、现在补什么」：2-3 条具体建议，写清食物名 + 分量（例如"加一份 150g 的清炒油麦菜"），
  优先补标为「偏低」的项目。
「三、下一餐怎么安排」：结合今日累计，给下一餐 1-2 条方向性建议（不必精确到克）。
  不要复述本餐的补充建议。
「四、一句话提醒」：参考这句素材，用自己的话写一句：
  「{budget.get('tip') or ''}」
  不要说"总体均衡""继续保持"这类与结论相反的客套话。

严格要求：
- 中文，总长度 350 字以内，数字带单位；
- 不要输出代码块、不要 YAML 头、不要开场白，直接从「一、这餐怎么样」开始；
- 不要提及"检测框""识别""系统""数据"等实现细节，也不要把米饭处理说明当成建议。
"""
