# -*- coding: utf-8 -*-
"""界面样式（QSS）。

浅色主题：白色卡片 + 浅灰底 + 单一主色（蓝），避免花哨，
答辩投影时对比度高、距离远也看得清。
"""

MAIN_QSS = """
QWidget { font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif; font-size: 13px; color: #1f2329; }
QWidget#Root { background: #f2f3f5; }

QFrame#Card { background: #ffffff; border: 1px solid #e4e7ec; border-radius: 12px; }
QLabel#CardTitle { font-size: 13px; color: #6b7280; font-weight: 500; }
QLabel#Hint { color: #8b939e; font-size: 12px; }
QLabel#AppTitle { font-size: 19px; font-weight: 600; color: #14181f; }
QLabel#AppSub { color: #8b939e; font-size: 12px; }

QLabel#MetricValue { font-size: 26px; font-weight: 600; color: #14181f; }
QLabel#MetricUnit  { font-size: 12px; color: #8b939e; }
QLabel#MetricLabel { font-size: 12px; color: #6b7280; }

QPushButton {
    background: #ffffff; border: 1px solid #d7dbe2; border-radius: 8px;
    padding: 7px 14px; color: #2b3138;
}
QPushButton:hover { background: #f6f8fb; border-color: #b9c0cb; }
QPushButton:pressed { background: #eceff4; }
QPushButton:disabled { color: #a8b0bb; background: #f4f5f7; border-color: #e4e7ec; }

QPushButton#Primary {
    background: #2f6fed; border: 1px solid #2f6fed; color: #ffffff;
    font-size: 14px; font-weight: 500; padding: 11px 16px; border-radius: 9px;
}
QPushButton#Primary:hover { background: #2a63d4; border-color: #2a63d4; }
QPushButton#Primary:pressed { background: #2456ba; }
QPushButton#Primary:disabled { background: #b8c7e6; border-color: #b8c7e6; color: #ffffff; }

QPushButton#Meal {
    background: #f4f5f7; border: 1px solid #e4e7ec; border-radius: 8px;
    padding: 9px 0px; color: #4b5563; font-size: 13px;
}
QPushButton#Meal:hover { background: #eef1f6; }
QPushButton#Meal:checked {
    background: #e8f0fe; border: 1px solid #2f6fed; color: #1a4fbb; font-weight: 500;
}

QComboBox {
    background: #ffffff; border: 1px solid #d7dbe2; border-radius: 8px;
    padding: 7px 10px; min-height: 18px;
}
QComboBox:hover { border-color: #b9c0cb; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #ffffff; border: 1px solid #e4e7ec; border-radius: 8px;
    selection-background-color: #e8f0fe; selection-color: #1a4fbb; outline: none;
}

QLineEdit, QSpinBox, QDoubleSpinBox {
    background: #ffffff; border: 1px solid #d7dbe2; border-radius: 8px;
    padding: 7px 10px; min-height: 18px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #2f6fed; }

QTabWidget::pane { border: none; background: transparent; }
QTabBar::tab {
    background: transparent; border: none; padding: 9px 18px;
    color: #6b7280; font-size: 13px; margin-right: 4px;
}
QTabBar::tab:hover { color: #1f2329; }
QTabBar::tab:selected { color: #1a4fbb; font-weight: 500; border-bottom: 2px solid #2f6fed; }

QTableWidget {
    background: #ffffff; border: none; gridline-color: #f0f2f5;
    selection-background-color: #e8f0fe; selection-color: #1a4fbb;
}
QTableWidget::item { padding: 6px 4px; }
QHeaderView::section {
    background: #fafbfc; border: none; border-bottom: 1px solid #e4e7ec;
    padding: 8px 4px; color: #6b7280; font-weight: 500;
}

QTextBrowser {
    background: #ffffff; border: 1px solid #e4e7ec; border-radius: 10px;
    padding: 14px; font-size: 13.5px; line-height: 1.7;
}

QProgressBar {
    background: #eef0f3; border: none; border-radius: 4px; height: 8px; text-align: center;
}
QProgressBar::chunk { background: #2f6fed; border-radius: 4px; }

QFrame#DropArea {
    background: #fbfcfd; border: 1.5px dashed #c7ccd6; border-radius: 12px;
}
QFrame#DropArea[dragging="true"] {
    background: #eef4ff; border: 1.5px dashed #2f6fed;
}

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #d5d9e0; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #bfc5cf; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #d5d9e0; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

QStatusBar { background: #f2f3f5; color: #6b7280; }
QSplitter::handle { background: transparent; }
QToolTip { background: #2b3138; color: #ffffff; border: none; padding: 6px 8px; border-radius: 6px; }
"""
