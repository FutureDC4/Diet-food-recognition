# -*- coding: utf-8 -*-
"""建议文本的清洗测试：提示词残留剔除 / 版式整理 / 判定兜底纠正。

运行：python tests/test_advice_text.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main_window import format_advice  # noqa: E402
from app.prompt import enforce_verdict  # noqa: E402

# 1.5B 模型实际抄出来的样子（取自 docs/screenshots/02_gui_advice.png）
COPIED = (
    "一、这餐怎么样\n"
    "第一句直接用下面这句总评：热量适中（1007 kcal，在建议的 934-1142 之间）…"
    "然后只补一句**，解释其中**最主要那一个**问题对健康的影响**：钠偏高。\n"
    "二、现在补什么\n"
    "优先补标为「偏低」的项目。加一份 150g 的清炒油麦菜，再配 200g 无糖酸奶。\n"
    "四、一句话提醒\n"
    "参考这句素材，用自己的话写一句：这餐不缺热量，缺的是菜。\n"
    "不要写「总体均衡」「继续保持」这类与结论相反的客套话。\n"
)

LOW_CHECKS = [
    {"key": "fiber", "label": "膳食纤维", "status": "low",
     "actual_text": "4.8 g", "target_text": "10.0 g"},
    {"key": "veg", "label": "蔬菜占比", "status": "low",
     "actual_text": "14%", "target_text": "30%"},
    {"key": "sodium", "label": "钠", "status": "high",
     "actual_text": "2652 mg", "target_text": "≤667 mg"},
    {"key": "kcal", "label": "本餐热量", "status": "ok",
     "actual_text": "1007 kcal", "target_text": "934-1142 kcal"},
    {"key": "protein", "label": "蛋白质", "status": "ok",
     "actual_text": "47 g", "target_text": "27 g"},
]

def test_meta_fragments_removed():
    """「第一句直接用…」「不要写…客套话」这类要求不许出现在正文里。"""
    out = format_advice(COPIED)
    for bad in ("第一句直接", "一字不改", "只补一句", "最主要那一个",
                "客套话", "总体均衡", "参考这句素材", "**"):
        assert bad not in out, f"正文残留指令片段：{bad}\n{out}"
    print("[OK] 提示词片段已剔除")


def test_real_content_kept():
    """清洗不能连真正写出来的判断和建议一起删掉。"""
    out = format_advice(COPIED)
    assert "热量适中" in out and "1007 kcal" in out
    assert "钠偏高" in out
    assert "清炒油麦菜" in out and "150g" in out
    assert "这餐不缺热量，缺的是菜" in out
    print("[OK] 正文内容与数字完整保留")


def test_headings_rendered():
    """「一、xxx」要变成 h4 标题，正文另起」。"""
    out = format_advice(COPIED)
    assert out.startswith("#### 一、这餐怎么样"), out[:40]
    assert "#### 二、现在补什么" in out and "#### 四、一句话提醒" in out
    print("[OK] 四节标题层级正常")


def test_no_leading_punctuation():
    """摘掉指令后不能留下「，：」这样的开头。"""
    for line in format_advice(COPIED).split("\n"):
        s = line.strip()
        if s and not s.startswith("#"):
            assert s[0] not in "，,、；;：:。", f"行首有残留标点：{s}"
    print("[OK] 无行首残留标点")


def test_verdict_guard_fixes_contradiction():
    """规则层判「偏低」的项目，被写成「达标」要纠回来。"""
    text = "膳食纤维达标，蔬菜占比充足，下一餐正常吃即可。"
    out = enforce_verdict(text, LOW_CHECKS)
    assert "膳食纤维偏低" in out and "蔬菜占比偏低" in out, out
    assert "达标" not in out and "充足" not in out
    print(f"[OK] 判定兜底生效：{out}")


def test_verdict_guard_keeps_high_item():
    """标为偏高的项目不归这一层管，原样留着。"""
    text = "钠偏高，注意少喝汤。"
    assert enforce_verdict(text, LOW_CHECKS) == text
    print("[OK] 偏高项不被改动")


def test_verdict_guard_fixes_ok_item():
    """已达标的热量被写成「不足」要纠回来（模型会把达标项一并说成未达标）。"""
    text = "热量不足，蛋白质偏低，下一餐多吃点。"
    out = enforce_verdict(text, LOW_CHECKS)
    assert "热量达标" in out and "蛋白质达标" in out, out
    print(f"[OK] 达标项被误判时已纠正：{out}")


def test_verdict_guard_no_checks():
    """没有结论清单时不做任何改写。"""
    text = "膳食纤维达标。"
    assert enforce_verdict(text, []) == text
    print("[OK] 无结论清单时不改写")


def main() -> int:
    tests = [
        test_meta_fragments_removed, test_real_content_kept, test_headings_rendered,
        test_no_leading_punctuation, test_verdict_guard_fixes_contradiction,
        test_verdict_guard_keeps_high_item, test_verdict_guard_fixes_ok_item,
        test_verdict_guard_no_checks,
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
    print(f"\n建议文本测试完成：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
