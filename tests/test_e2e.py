"""端到端测试：图片 → 盘子 → 食物 → 重量 → 热量 → BMR/TDEE → Prompt → Ollama。

不依赖交互输入，全部用预设数据。
运行：python tests/test_e2e.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as app  # noqa: E402

TEST_IMAGE = PROJECT_ROOT / "data" / "images" / "test.jpg"
PRESET_USER = {
    "image_path": str(TEST_IMAGE),
    "age": 25,
    "gender": "男",
    "height_cm": 175.0,
    "weight_kg": 70.0,
    "activity_level": 1,
}


def test_pipeline_no_llm():
    """完整流水线（不走 LLM）：结果结构完整、数值合理。"""
    result = app.run_pipeline(
        image_path=str(TEST_IMAGE),
        user=PRESET_USER,
        use_llm=False,
        save_annotated=True,
        save_db=False,
    )

    for key in [
        "plate_detections", "scale", "food_detections", "dishes",
        "totals", "health", "meal_share", "prompt", "annotated_path",
    ]:
        assert key in result, f"结果缺少字段: {key}"

    assert result["px_per_cm"] > 0
    assert result["health"]["bmr"] > 0
    assert result["health"]["tdee"] > result["health"]["bmr"]
    assert 0 <= result["meal_share"] < 200
    assert "本餐识别结果" in result["prompt"]

    if result["dishes"]:
        for d in result["dishes"]:
            assert d["weight_g"] > 0
            assert d["calories"] >= 0
            assert 10 <= d["weight_g"] <= 1500
        assert abs(result["totals"]["weight_g"] - sum(d["weight_g"] for d in result["dishes"])) < 1
        print(f"[OK] 识别 {len(result['dishes'])} 种食物，合计 {result['totals']['weight_g']}g / "
              f"{result['totals']['calories']} kcal；本餐占 TDEE {result['meal_share']}%")
    else:
        print("[WARN] 该图片未识别出食物（可能是模型不认识的类别），流水线本身正常")

    if result["annotated_path"]:
        assert Path(result["annotated_path"]).exists(), "标注图未生成"
        print(f"[OK] 标注图：{result['annotated_path']}")
    print(f"[OK] 用时 {result['elapsed_s']}s")


def test_pipeline_with_llm():
    """完整流水线（含 LLM）：Ollama 在线则应有建议文本；离线则应有降级提示。"""
    from src.llm import OllamaClient

    result = app.run_pipeline(
        image_path=str(TEST_IMAGE),
        user=PRESET_USER,
        use_llm=True,
        save_annotated=False,
        save_db=False,
    )
    text = result["llm_text"]
    assert text, "LLM 输出为空（应有建议或至少降级提示）"

    client = OllamaClient()
    online = client.is_available() and client.has_model()
    if online:
        assert len(text) > 80, f"在线建议过短: {text[:100]}"
        print(f"[OK] Ollama 在线，建议 {len(text)} 字符：{text[:50].strip()}...")
    else:
        assert "Ollama" in text, text[:120]
        print("[OK] Ollama 离线，正确输出降级提示")


def test_demo_user_consistency():
    """spec 里的示例：男 25 岁 175cm 70kg / 活动量 1 → BMR≈1674，TDEE≈2302，330g/482kcal → 20.9%。"""
    from src.health import BMRCalculator

    calc = BMRCalculator()
    r = calc.calculate(25, "男", 175, 70, 1)
    assert abs(r["bmr"] - 1674) < 2
    assert abs(r["tdee"] - 2302) < 2
    assert calc.meal_share(482, r["tdee"]) == 20.9
    print(f"[OK] 示例数据一致：BMR {r['bmr']}, TDEE {r['tdee']}, 占比 {calc.meal_share(482, r['tdee'])}%")


def main() -> int:
    tests = [test_demo_user_consistency, test_pipeline_no_llm, test_pipeline_with_llm]
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
    print(f"\n端到端测试完成：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
