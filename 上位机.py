# -*- coding: utf-8 -*-
"""膳食识别上位机 —— 启动入口。

双击「启动上位机.bat」或在项目根目录执行：
    python 上位机.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    try:
        from app.main_window import run
    except ModuleNotFoundError as exc:
        print("缺少依赖：", exc)
        print("请先执行：pip install PySide6-Essentials")
        return 1
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
