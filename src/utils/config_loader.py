"""配置加载：读取 config.yaml，提供全局配置与路径解析。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# 项目根目录：src/utils/config_loader.py -> 上溯两级
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_FILE: Path = PROJECT_ROOT / "config.yaml"

_DEFAULT_ACTIVITY = {
    1: {"name": "轻度", "factor": 1.375},
    2: {"name": "中度", "factor": 1.550},
    3: {"name": "重度", "factor": 1.725},
    4: {"name": "极重", "factor": 1.900},
}


@lru_cache(maxsize=1)
def _load(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误（顶层应为字典）：{path}")
    return data


def get_config(reload: bool = False, config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """返回全局配置字典（带缓存）。"""
    path = Path(config_path) if config_path else CONFIG_FILE
    if reload:
        _load.cache_clear()
    return _load(str(path))


def get(key_path: str, default: Any = None) -> Any:
    """按 `a.b.c` 路径读取配置项。"""
    node: Any = get_config()
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_path(key_path: str, default: Optional[str] = None) -> Path:
    """读取某个路径类配置项，并解析成绝对路径（相对项目根目录）。"""
    raw = get(key_path, default)
    if raw is None:
        raise KeyError(f"配置项缺失：{key_path}")
    p = Path(str(raw))
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def activity_levels() -> Dict[int, Dict[str, Any]]:
    """活动量系数表，配置缺失时回退到内置标准值。"""
    levels = get("health.activity_levels") or {}
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in levels.items():
        try:
            out[int(k)] = {"name": v.get("name", ""), "factor": float(v.get("factor", 1.375))}
        except (TypeError, ValueError):
            continue
    return out or _DEFAULT_ACTIVITY


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    cfg = get_config()
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"顶层配置项: {list(cfg.keys())}")
    print(f"食物模型: {resolve_path('paths.food_model')}")
