# -*- coding: utf-8 -*-
"""上位机的小组件：结构评估面板。

「餐次配额 + 结构评估」（B+C 方案）在界面上的呈现：
面板顶部是这一餐的能量预算与实际值，下面是 8 个检查项，
每项一条进度条 + 状态色，鼠标悬停显示具体说明。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget,
)

# 状态 → 颜色
STATUS_COLOR = {
    "ok": "#2f8f5b",
    "low": "#c98a2b",
    "high": "#c4453a",
    "warn": "#c98a2b",
}
STATUS_TEXT = {"ok": "达标", "low": "偏低", "high": "偏高", "warn": "注意"}


class CheckRow(QWidget):
    """一行检查项：标签 + 实际值 + 进度条。"""

    def __init__(self, label: str, unit_hint: str = "", parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(6)

        self.lb = QLabel(label)
        self.lb.setFixedWidth(58)
        self.lb.setStyleSheet("color:#6b7280; font-size:12px;")

        self.val = QLabel("--")
        self.val.setFixedWidth(64)
        self.val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.val.setStyleSheet("color:#1f2329; font-size:12px; font-weight:500;")

        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 100)
        self.bar.setFixedHeight(6)
        self.bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lay.addWidget(self.lb)
        lay.addWidget(self.val)
        lay.addWidget(self.bar, 1)

    def set_check(self, check: Dict[str, Any]) -> None:
        color = STATUS_COLOR.get(check.get("status", "ok"), "#9aa2ad")
        self.val.setText(str(check.get("actual_text", "--")))
        self.bar.setValue(int(check.get("bar", 0)))
        self.bar.setStyleSheet(
            "QProgressBar { background:#eef0f3; border:none; border-radius:3px; }"
            f"QProgressBar::chunk {{ background:{color}; border-radius:3px; }}"
        )
        tip = (f"{check.get('label')}：{check.get('actual_text')}，"
               f"参考 {check.get('target_text')}\n{check.get('hint', '')}")
        for w in (self, self.lb, self.val, self.bar):
            w.setToolTip(tip)


class StructurePanel(QFrame):
    """餐次配额 + 结构评估面板。"""

    ROWS = [
        ("kcal", "热量"),
        ("protein", "蛋白质"),
        ("fiber", "纤维"),
        ("veg", "蔬菜"),
        ("staple", "主食"),
        ("fat", "脂肪"),
        ("sodium", "钠"),
        ("fried", "促炎"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumWidth(298)
        self.setMaximumWidth(322)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(6)
        title = QLabel("餐次配额与结构")
        title.setObjectName("CardTitle")
        self.badge = QLabel("")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.badge.setStyleSheet("color:#8b939e; font-size:12px;")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.badge)
        lay.addLayout(head)

        # 能量预算块
        self.box = QFrame()
        self.box.setStyleSheet("background:#f7f9fc; border:none; border-radius:8px;")
        box_lay = QVBoxLayout(self.box)
        box_lay.setContentsMargins(10, 8, 10, 8)
        box_lay.setSpacing(2)

        self.lb_meal = QLabel("—")
        self.lb_meal.setStyleSheet("color:#6b7280; font-size:11px; border:none;")
        self.lb_kcal = QLabel("—")
        self.lb_kcal.setStyleSheet("color:#14181f; font-size:15px; font-weight:600; border:none;")
        self.lb_share = QLabel("—")
        self.lb_share.setStyleSheet("color:#8b939e; font-size:11px; border:none;")
        box_lay.addWidget(self.lb_meal)
        box_lay.addWidget(self.lb_kcal)
        box_lay.addWidget(self.lb_share)
        lay.addWidget(self.box)

        # 检查项
        self.rows: Dict[str, CheckRow] = {}
        for key, label in self.ROWS:
            row = CheckRow(label)
            self.rows[key] = row
            lay.addWidget(row)

        lay.addStretch(1)

        self.foot = QLabel("依据《中国居民膳食指南（2022）》\n"
                           "与 DRIs（2023 版）折算")
        self.foot.setStyleSheet("color:#a8b0bb; font-size:10.5px; border:none;")
        self.foot.setWordWrap(True)
        lay.addWidget(self.foot)

        self.clear()

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self.badge.setText("等待分析")
        self.lb_meal.setText("—")
        self.lb_kcal.setText("—")
        self.lb_share.setText("—")
        for row in self.rows.values():
            row.val.setText("--")
            row.bar.setValue(0)
            row.bar.setStyleSheet(
                "QProgressBar { background:#eef0f3; border:none; border-radius:3px; }"
                "QProgressBar::chunk { background:#d5d9e0; border-radius:3px; }"
            )

    def show_budget(self, result: Dict[str, Any]) -> None:
        budget = result.get("budget") or {}
        targets = budget.get("targets") or {}
        checks: List[Dict[str, Any]] = budget.get("checks") or []
        if not checks:
            self.clear()
            self.badge.setText("评估不可用")
            return

        meal = result.get("meal_type", "")
        tdee = float(targets.get("tdee", 0) or 0)
        self.lb_meal.setText(f"{meal} · 每日需求 {tdee:.0f} kcal 的 "
                             f"{targets.get('meal_share_text', '')}")
        kcal = float(result["totals"].get("calories", 0) or 0)
        kmin = float(targets.get("kcal_min", 0) or 0)
        kmax = float(targets.get("kcal_max", 0) or 0)
        self.lb_kcal.setText(f"实际 {kcal:.0f} kcal")
        self.lb_share.setText(f"建议区间 {kmin:.0f}-{kmax:.0f} kcal")

        score = int(budget.get("score", 0))
        good = budget.get("good", 0)
        total = budget.get("total", 0)
        color = "#2f8f5b" if score >= 70 else ("#c98a2b" if score >= 40 else "#c4453a")
        self.badge.setText(f"达标 {good}/{total}")
        self.badge.setStyleSheet(f"color:{color}; font-size:12px; font-weight:600;")

        by_key = {c["key"]: c for c in checks}
        for key, row in self.rows.items():
            if key in by_key:
                row.set_check(by_key[key])

    def additions_text(self, result: Dict[str, Any]) -> str:
        budget = result.get("budget") or {}
        items = budget.get("additions") or []
        if not items:
            return ""
        lines = []
        for a in items:
            mark = "＋" if a.get("action") == "add" else "－"
            lines.append(f"{mark} {a.get('text')}")
        return "\n".join(lines)
