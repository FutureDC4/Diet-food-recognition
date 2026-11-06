# -*- coding: utf-8 -*-
"""推理设备解析测试：配置写 cuda 但机器没有显卡时，必须回退 cpu 而不是报错。

运行：python tests/test_device.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import is_cpu, resolve_device  # noqa: E402


def test_cuda_falls_back_when_unavailable():
    """没有 CUDA 设备时，cuda:0 要变成 cpu；有设备时保持原样。"""
    import torch

    got = resolve_device("cuda:0")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        assert got == "cuda:0", f"有显卡却没用上：{got}"
        print(f"[OK] 有 CUDA 设备，沿用 {got}")
    else:
        assert got == "cpu", f"无 CUDA 设备却没有回退：{got}"
        print("[OK] 无 CUDA 设备，已回退 cpu")


def test_cpu_passthrough():
    """配置明确写 cpu 时不做任何改动。"""
    assert resolve_device("cpu") == "cpu"
    print("[OK] cpu 原样返回")


def test_empty_defaults_cpu():
    """没配置时默认 cpu，不留空串给 ultralytics。"""
    assert resolve_device("") == "cpu" and resolve_device(None) == "cpu"
    print("[OK] 空配置默认 cpu")


def test_is_cpu():
    """fp16 只在 cuda 下开，这里提供判断依据。"""
    assert is_cpu("cpu") and is_cpu("") and is_cpu(None)
    assert not is_cpu("cuda:0") and not is_cpu("cuda")
    print("[OK] CPU 判定正确")


def main() -> int:
    tests = [
        test_cuda_falls_back_when_unavailable, test_cpu_passthrough,
        test_empty_defaults_cpu, test_is_cpu,
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
    print(f"\n设备解析测试完成：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
