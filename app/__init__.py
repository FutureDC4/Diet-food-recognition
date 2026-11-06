# -*- coding: utf-8 -*-
"""桌面版上位机（PySide6）。

模块划分：
    analyzer.py     分析引擎：检测 → 估重 → 营养 → 健康指标（含米饭占比规则）
    prompt.py       膳食建议提示词（单餐视角）
    main_window.py  界面与线程
    render.py       中文标签可视化
    user_store.py   用户档案本地存储（data/users.json）
    theme.py        界面样式
"""

__all__ = ["analyzer", "prompt", "render", "theme", "user_store"]
