# -*- coding: utf-8 -*-
"""推理设备解析。

配置里写的是 `cuda:0`，但笔记本切到集显模式、外接显卡没插、驱动重装之后，
torch 里根本没有可用的 CUDA 设备。这时 ultralytics 会直接抛出
`ValueError: Invalid CUDA 'device=0' requested`，整条分析链路断在第一步。

这里把「配置值」翻译成「这台机器此刻真的能用的设备」：能用就用，不能用退回 CPU。
"""

from __future__ import annotations

from .logger import get_logger

log = get_logger(__name__)


def resolve_device(preferred: str | None = None) -> str:
    """返回当前可用的推理设备名。

    - `preferred` 为 `cuda` / `cuda:0` 且 torch 能看到显卡 → 原样返回；
    - 否则返回 `cpu`（同时打一条日志，便于排查为什么变慢了）。
    """
    want = str(preferred or "").strip()
    if not want:
        return "cpu"
    if not want.lower().startswith("cuda"):
        return want
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return want
        log.warning(f"配置要求 {want}，但当前没有可用的 CUDA 设备，已回退 cpu 推理")
    except Exception as exc:                     # noqa: BLE001
        log.warning(f"检测 CUDA 失败（{exc}），已回退 cpu 推理")
    return "cpu"


def is_cpu(device: str | None) -> bool:
    """判断设备是否走 CPU（CPU 下不能开 fp16）。"""
    return not str(device or "").lower().startswith("cuda")
