"""Ollama 集成测试：在线判定 / 流式输出 / 离线降级 / Prompt 完整性。

运行：python tests/test_llm.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import FALLBACK_TEXT, OllamaClient, build_prompt  # noqa: E402

DEMO_WEIGHTS = [
    {"name": "chicken", "weight_g": 150, "calories": 247.5, "matched": True},
    {"name": "rice", "weight_g": 180, "calories": 234.0, "matched": True},
]
DEMO_USER = {"age": 25, "gender": "男", "height_cm": 175, "weight_kg": 70, "activity_level": 1}
DEMO_HEALTH = {
    "bmr": 1673.8,
    "tdee": 2301.5,
    "bmi": 22.9,
    "bmi_label": "正常",
    "activity_name": "轻度",
    "activity_factor": 1.375,
}
DEMO_TOTALS = {"weight_g": 330, "calories": 481.5, "protein_g": 40, "fat_g": 30, "carb_g": 50}


def test_prompt_content():
    prompt = build_prompt([], DEMO_WEIGHTS, DEMO_TOTALS, DEMO_USER, DEMO_HEALTH, meal_share=20.9)
    for kw in ["chicken", "rice", "150g", "180g", "1674", "2302", "20.9%", "年龄：25", "男", "BMI"]:
        assert kw in prompt, f"Prompt 缺少关键信息: {kw}"
    assert "一、本餐营养评估" in prompt
    assert "Markdown" in prompt
    print(f"[OK] Prompt 构建正常（{len(prompt)} 字符），含食物/重量/热量/健康数据")
    return prompt


def test_availability():
    client = OllamaClient()
    available = client.is_available()
    print(f"[{'OK' if available else 'WARN'}] Ollama {client.host} {'在线' if available else '离线'}；"
          f"模型列表={client.list_models()}；目标模型 {client.model} 存在={client.has_model()}")
    if not available:
        print("     （离线属于可接受状态，降级逻辑会在下一个用例验证）")


def test_stream_online():
    client = OllamaClient(num_predict=120, timeout=120)
    if not client.is_available() or not client.has_model():
        print("[SKIP] Ollama 不可用或模型缺失，跳过在线流式测试")
        return
    t0 = time.time()
    chunks = []
    for chunk in client.stream_generate("用一句话说明米饭的主要营养特点。"):
        chunks.append(chunk)
        if len("".join(chunks)) > 40:
            break
    text = "".join(chunks).strip()
    assert len(chunks) >= 1, "没有收到任何流式数据块"
    assert text and "【注意】" not in text, f"流式输出异常: {text[:120]}"
    print(f"[OK] 流式输出 {len(chunks)} 块 / {len(text)} 字符 / {time.time() - t0:.1f}s：{text[:60]}...")


def test_fallback_offline():
    """指向不存在的端口，验证降级提示。"""
    client = OllamaClient(host="http://127.0.0.1:59999", timeout=3)
    assert client.is_available(timeout=2) is False
    text = "".join(client.stream_generate("你好"))
    assert text.strip(), "降级时应有兜底文本"
    assert "Ollama" in text, f"降级文本异常: {text[:120]}"
    print(f"[OK] 离线降级：{text.splitlines()[0]}")


def test_missing_model_fallback():
    """模型名不存在时，应提示 ollama pull。"""
    client = OllamaClient(model="not-exist-model:0b", timeout=5)
    if not client.is_available():
        print("[SKIP] Ollama 离线，跳过缺失模型测试")
        return
    text = "".join(client.stream_generate("你好"))
    assert "ollama pull" in text, f"缺失模型提示异常: {text[:150]}"
    print(f"[OK] 缺失模型提示：{text.splitlines()[0]}")


def main() -> int:
    tests = [
        test_prompt_content,
        test_availability,
        test_stream_online,
        test_fallback_offline,
        test_missing_model_fallback,
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
    print(f"\nLLM 测试完成：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
