"""日志配置：loguru，同时输出到控制台与文件。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from .config_loader import PROJECT_ROOT, ensure_dir, get

_CONFIGURED = False
_FORMAT = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"


def setup_logger(level: Optional[str] = None, log_file: Optional[str | Path] = None) -> "logger.__class__":
    """初始化全局 logger（幂等，可重复调用）。"""
    global _CONFIGURED
    if _CONFIGURED:
        return logger

    level = (level or get("logging.level", "INFO")).upper()
    if log_file is None:
        log_file = get("logging.file", "logs/app.log")
    path = Path(str(log_file))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    ensure_dir(path.parent)

    logger.remove()
    logger.add(sys.stderr, level=level, format=_FORMAT, colorize=True, backtrace=False, diagnose=False)
    logger.add(
        str(path),
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=get("logging.rotation", "10 MB"),
        retention=get("logging.retention", "14 days"),
        encoding="utf-8",
        enqueue=True,
    )
    _CONFIGURED = True
    return logger


def get_logger(name: str = "膳食识别系统"):
    """获取带上下文标签的 logger。"""
    setup_logger()
    return logger.bind(name=name)


__all__ = ["setup_logger", "get_logger", "logger"]
