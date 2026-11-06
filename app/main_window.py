# -*- coding: utf-8 -*-
"""膳食识别上位机（PySide6）。

界面结构：
    左侧：就餐人选择 / 餐次选择 / 图片上传（拖拽或点选）/ 开始分析
    右侧：「识别结果」「膳食建议」「历史记录」三个页签

线程模型：
    模型预热、图片分析、Ollama 流式生成各跑在独立 QThread，界面不卡。
"""

from __future__ import annotations

import random
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QColor, QFont, QImage, QPixmap, QTextBlockFormat, QTextCharFormat, QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSizePolicy,
    QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem, QTextBrowser,
    QVBoxLayout, QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.database import DBManager                                    # noqa: E402
from src.llm import OllamaClient                                     # noqa: E402
from src.utils.logger import get_logger, setup_logger                # noqa: E402

from .analyzer import MEAL_TYPES, MealAnalyzer                       # noqa: E402
from .detection_rules import DEFAULT_HEIGHT_LEVEL, HEIGHT_LEVELS     # noqa: E402
from .prompt import SYSTEM_PROMPT, build_meal_prompt                 # noqa: E402
from .theme import MAIN_QSS                                          # noqa: E402
from .user_store import ACTIVITY_NAMES, UserStore                    # noqa: E402
from .widgets import StructurePanel                                  # noqa: E402

log = get_logger("上位机")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ====================================================================== 工具
_MARK_RE = re.compile(r"^[「【]?\s*([一二三四五六七八]、)\s*([^」】\s　，。：:]{2,16})(.*)$")
_BULLET_RE = re.compile(r"^[ \t]*[•·\-*]\s+")


def format_advice(text: str) -> str:
    """把模型输出的「一、xxx」小节整理成 Markdown 标题。

    模型时常用「一、这餐怎么样 正文…」这种「标题+正文同一行」的写法，
    这里按行切开：标题单独成行做 h4，剩余正文另起一段，渲染出来才有层级。
    """
    if not text:
        return text
    # 先把「一、xxx」这类包裹符拆成独立行
    t = re.sub(r"[「【]\s*([一二三四五六七八]、[^」】\n]{2,26}?)\s*[」】][：:]?", r"\n\1\n", text)

    out: List[str] = []
    for raw in t.replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            out.append("")
            continue
        m = _MARK_RE.match(line)
        if m:
            title = f"{m.group(1)}{m.group(2)}"
            rest = m.group(3).strip(" 　，,：:、")
            if out and out[-1] != "":
                out.append("")
            out.append(f"#### {title}")
            if rest:
                out.append("")
                out.append(rest)
        else:
            out.append(_BULLET_RE.sub("- ", line))

    txt = "\n".join(out)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def ndarray_to_pixmap(bgr: np.ndarray) -> QPixmap:
    """OpenCV BGR ndarray → QPixmap。"""
    arr = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, int(arr.strides[0]), QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


# ====================================================================== 组件
class MetricCard(QFrame):
    """一个指标卡：标题 + 大数字 + 单位。"""

    def __init__(self, label: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(84)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)

        self.lbl = QLabel(label)
        self.lbl.setObjectName("MetricLabel")
        row = QHBoxLayout()
        row.setSpacing(3)
        row.setContentsMargins(0, 0, 0, 0)
        self.value = QLabel("--")
        self.value.setObjectName("MetricValue")
        self.unit = QLabel(unit)
        self.unit.setObjectName("MetricUnit")
        self.unit.setAlignment(Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self.value)
        row.addWidget(self.unit)
        row.addStretch(1)
        lay.addWidget(self.lbl)
        lay.addLayout(row)

    def set_value(self, text: str) -> None:
        self.value.setText(str(text))


class ImageView(QLabel):
    """按窗口大小自适应显示图片。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#fafbfc; border:1px solid #e4e7ec; border-radius:10px;")
        self._pm: Optional[QPixmap] = None
        self.setText("识别结果会显示在这里")
        self.setFont(QFont("Microsoft YaHei UI", 12))
        self.setStyleSheet(
            "background:#fafbfc; border:1px solid #e4e7ec; border-radius:10px; color:#9aa2ad;"
        )

    def set_image(self, pm: Optional[QPixmap]) -> None:
        self._pm = pm
        self._rescale()

    def clear_image(self, text: str = "识别结果会显示在这里") -> None:
        self._pm = None
        super().setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pm is None or self._pm.isNull():
            return
        target = self.size() - QSize(10, 10)
        self.setPixmap(self._pm.scaled(
            target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))


class DropArea(QFrame):
    """拖拽 / 点击选择图片的区域。"""

    picked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropArea")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(168)
        self.setProperty("dragging", "false")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        self.thumb = QLabel("点这里选择图片\n或把图片拖进来")
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet("color:#9aa2ad; font-size:12px; border:none;")
        self.thumb.setMinimumHeight(112)
        lay.addWidget(self.thumb)
        self.name = QLabel("")
        self.name.setObjectName("Hint")
        self.name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name.setWordWrap(True)
        lay.addWidget(self.name)

    # ---- 交互 ----
    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择食物照片", str(PROJECT_ROOT / "data"),
                "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)",
            )
            if path:
                self.picked.emit(path)
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_dragging(True)

    def dragLeaveEvent(self, event):  # noqa: N802
        self._set_dragging(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):  # noqa: N802
        self._set_dragging(False)
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in IMAGE_SUFFIXES:
                self.picked.emit(str(p))
                break
        event.acceptProposedAction()

    def _set_dragging(self, on: bool) -> None:
        self.setProperty("dragging", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_preview(self, path: str, pm: QPixmap) -> None:
        self.thumb.setPixmap(pm.scaled(
            148, 112, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        name = Path(path).name
        if len(name) > 30:
            name = name[:26] + "…" + Path(path).suffix
        self.name.setText(name)


class UserDialog(QDialog):
    """新建 / 编辑用户档案。"""

    def __init__(self, profile: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建用户" if profile is None else "编辑用户")
        self.setMinimumWidth(360)
        profile = profile or {}

        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.ed_name = QLineEdit(str(profile.get("name", "")))
        self.ed_name.setPlaceholderText("例如：张三")

        self.sp_age = QSpinBox()
        self.sp_age.setRange(1, 129)
        self.sp_age.setValue(int(profile.get("age", 21)))

        self.cb_gender = QComboBox()
        self.cb_gender.addItems(["男", "女"])
        if profile.get("gender") == "女":
            self.cb_gender.setCurrentIndex(1)

        self.dp_height = QDoubleSpinBox()
        self.dp_height.setRange(80, 250)
        self.dp_height.setDecimals(1)
        self.dp_height.setSuffix(" cm")
        self.dp_height.setValue(float(profile.get("height_cm", 175)))

        self.dp_weight = QDoubleSpinBox()
        self.dp_weight.setRange(10, 350)
        self.dp_weight.setDecimals(1)
        self.dp_weight.setSuffix(" kg")
        self.dp_weight.setValue(float(profile.get("weight_kg", 65)))

        self.cb_activity = QComboBox()
        for level, name in ACTIVITY_NAMES.items():
            self.cb_activity.addItem(name, level)
        idx = max(0, int(profile.get("activity_level", 1)) - 1)
        self.cb_activity.setCurrentIndex(idx)

        rows = [
            ("姓名", self.ed_name), ("年龄", self.sp_age), ("性别", self.cb_gender),
            ("身高", self.dp_height), ("体重", self.dp_weight), ("日常活动量", self.cb_activity),
        ]
        for i, (label, widget) in enumerate(rows):
            lb = QLabel(label)
            lb.setObjectName("Hint")
            form.addWidget(lb, i, 0)
            form.addWidget(widget, i, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(14)
        title = QLabel("身体信息用于计算基础代谢与每日热量需求，只保存在本机。")
        title.setObjectName("Hint")
        title.setWordWrap(True)
        lay.addWidget(title)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def values(self) -> Dict[str, Any]:
        return {
            "name": self.ed_name.text().strip(),
            "age": self.sp_age.value(),
            "gender": self.cb_gender.currentText(),
            "height_cm": self.dp_height.value(),
            "weight_kg": self.dp_weight.value(),
            "activity_level": int(self.cb_activity.currentData() or 1),
        }


# ====================================================================== 线程
class WarmupWorker(QThread):
    ready = Signal()
    failed = Signal(str)

    def __init__(self, analyzer: MealAnalyzer):
        super().__init__()
        self.analyzer = analyzer

    def run(self):
        try:
            self.analyzer.warmup()
            self.ready.emit()
        except Exception as exc:                        # noqa: BLE001
            log.exception("模型预热失败")
            self.failed.emit(f"{exc.__class__.__name__}: {exc}")


class AnalyzeWorker(QThread):
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, analyzer: MealAnalyzer, image_path: str, user: dict,
                 meal_type: str, rice_rule: bool, height_level: str = DEFAULT_HEIGHT_LEVEL):
        super().__init__()
        self.analyzer = analyzer
        self.image_path = image_path
        self.user = user
        self.meal_type = meal_type
        self.rice_rule = rice_rule
        self.height_level = height_level

    def run(self):
        try:
            result = self.analyzer.analyze(
                self.image_path, self.user, meal_type=self.meal_type,
                rice_rule=self.rice_rule, height_level=self.height_level,
            )
            self.done.emit(result)
        except Exception as exc:                        # noqa: BLE001
            log.error("分析失败\n" + traceback.format_exc())
            self.failed.emit(f"{exc.__class__.__name__}: {exc}")


class LlmWorker(QThread):
    chunk = Signal(str)
    finished_text = Signal(str)

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        client = OllamaClient()
        buf: List[str] = []
        try:
            for piece in client.stream_generate(self.prompt, system=SYSTEM_PROMPT):
                if self._stop:
                    break
                buf.append(piece)
                self.chunk.emit(piece)
        except Exception as exc:                        # noqa: BLE001
            log.warning(f"建议生成中断：{exc}")
        self.finished_text.emit("".join(buf))


class OllamaStatusWorker(QThread):
    status = Signal(bool, str)

    def run(self):
        try:
            client = OllamaClient()
            ok = client.is_available(timeout=2.5)
            text = f"Ollama {client.model} 已连接" if ok else "Ollama 未连接，建议将只显示检测结果"
            self.status.emit(ok, text)
        except Exception:                               # noqa: BLE001
            self.status.emit(False, "Ollama 状态未知")


# ====================================================================== 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        setup_logger()
        self.setWindowTitle("膳食识别与营养建议系统")
        self.resize(1320, 840)
        self.setMinimumSize(1100, 720)
        self.setAcceptDrops(True)

        self.store = UserStore()
        self.db = DBManager()
        self.analyzer = MealAnalyzer()
        self.client = OllamaClient()

        self._image_path: Optional[str] = None
        self._result: Optional[Dict[str, Any]] = None
        self._llm_worker: Optional[LlmWorker] = None
        self._llm_buffer: List[str] = []
        self._saved_llm: str = ""
        self._record_id: Optional[int] = None
        self._warm = False

        self._build_ui()
        self._refresh_users()
        self._set_busy(False)

        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(120)
        self._flush_timer.timeout.connect(self._flush_llm)

        QTimer.singleShot(300, self._start_warmup)
        QTimer.singleShot(600, self._check_ollama)

    # ------------------------------------------------------------ 构建界面
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 10)
        outer.setSpacing(14)

        outer.addWidget(self._build_left(), 0)
        outer.addWidget(self._build_right(), 1)

        self.status = self.statusBar()
        self.status.showMessage("就绪")
        self.status.setSizeGripEnabled(True)

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("Card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        lb = QLabel(title)
        lb.setObjectName("CardTitle")
        lay.addWidget(lb)
        return frame, lay

    def _build_left(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(330)

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        title = QLabel("膳食识别与营养建议")
        title.setObjectName("AppTitle")
        sub = QLabel("中餐菜品识别 · 分量估算 · 个性化建议")
        sub.setObjectName("AppSub")
        lay.addWidget(title)
        lay.addWidget(sub)

        # ---- 就餐人 ----
        card_user, lay_user = self._card("就餐人")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.cb_user = QComboBox()
        self.cb_user.setMinimumWidth(150)
        self.cb_user.currentIndexChanged.connect(self._on_user_changed)
        btn_new = QPushButton("新建")
        btn_new.setFixedWidth(56)
        btn_new.clicked.connect(self._new_user)
        row.addWidget(self.cb_user, 1)
        row.addWidget(btn_new)
        lay_user.addLayout(row)

        self.lb_user = QLabel("还没有用户，点「新建」录入身体信息")
        self.lb_user.setObjectName("Hint")
        self.lb_user.setWordWrap(True)
        lay_user.addWidget(self.lb_user)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.btn_edit = QPushButton("编辑资料")
        self.btn_del = QPushButton("删除")
        self.btn_edit.clicked.connect(self._edit_user)
        self.btn_del.clicked.connect(self._delete_user)
        row2.addWidget(self.btn_edit, 1)
        row2.addWidget(self.btn_del)
        lay_user.addLayout(row2)
        lay.addWidget(card_user)

        # ---- 餐次 + 食物高度 ----
        card_meal, lay_meal = self._card("这一餐")
        grid = QHBoxLayout()
        grid.setSpacing(6)
        self.meal_group = QButtonGroup(self)
        self.meal_group.setExclusive(True)
        for i, name in enumerate(MEAL_TYPES):
            btn = QPushButton(name)
            btn.setObjectName("Meal")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setMinimumWidth(1)
            self.meal_group.addButton(btn, i)
            grid.addWidget(btn)
            if name == "午餐":
                btn.setChecked(True)
        lay_meal.addLayout(grid)

        self.lb_meal_hint = QLabel("决定本餐热量预算：早 25% / 午 40% / 晚 35%")
        self.lb_meal_hint.setObjectName("Hint")
        self.lb_meal_hint.setWordWrap(True)
        lay_meal.addWidget(self.lb_meal_hint)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#eef0f3;")
        lay_meal.addWidget(sep)

        lb_h = QLabel("食物堆得多高")
        lb_h.setObjectName("Hint")
        lay_meal.addWidget(lb_h)

        hrow = QHBoxLayout()
        hrow.setSpacing(6)
        self.height_group = QButtonGroup(self)
        self.height_group.setExclusive(True)
        height_tips = {
            "薄摊": "平盘摊开、汤菜、凉拌，约按常规高度的 0.75 折算",
            "常规": "食堂一格菜的正常高度（默认）",
            "堆高": "大份、堆起来的一盘，约按常规高度的 1.35 折算",
        }
        for i, name in enumerate(HEIGHT_LEVELS):
            btn = QPushButton(name)
            btn.setObjectName("Meal")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(height_tips.get(name, ""))
            self.height_group.addButton(btn, i)
            hrow.addWidget(btn)
            if name == DEFAULT_HEIGHT_LEVEL:
                btn.setChecked(True)
        lay_meal.addLayout(hrow)

        self.lb_height_hint = QLabel("照片看不出堆多高，按倍数换算重量")
        self.lb_height_hint.setObjectName("Hint")
        self.lb_height_hint.setWordWrap(True)
        lay_meal.addWidget(self.lb_height_hint)
        lay.addWidget(card_meal)

        # ---- 图片 ----
        card_img, lay_img = self._card("食物照片")
        self.drop = DropArea()
        self.drop.picked.connect(self.load_image)
        lay_img.addWidget(self.drop)
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        btn_pick = QPushButton("选择图片")
        btn_pick.clicked.connect(self._pick_image)
        btn_sample = QPushButton("示例")
        btn_sample.clicked.connect(self._load_sample)
        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self._clear_image)
        row3.addWidget(btn_pick, 2)
        row3.addWidget(btn_sample, 1)
        row3.addWidget(btn_clear, 1)
        lay_img.addLayout(row3)
        lay.addWidget(card_img)

        # ---- 开始分析 ----
        self.btn_run = QPushButton("开始分析")
        self.btn_run.setObjectName("Primary")
        self.btn_run.setMinimumHeight(44)
        self.btn_run.clicked.connect(self._start_analyze)
        lay.addWidget(self.btn_run)

        self.lb_ollama = QLabel("Ollama 状态检测中…")
        self.lb_ollama.setObjectName("Hint")
        self.lb_ollama.setWordWrap(True)
        lay.addWidget(self.lb_ollama)

        lay.addStretch(1)
        return panel

    def _build_right(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_result(), "识别结果")
        self.tabs.addTab(self._build_tab_advice(), "膳食建议")
        self.tabs.addTab(self._build_tab_history(), "历史记录")
        return self.tabs

    def _build_tab_result(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(10)

        # 上排：识别图 + 餐次配额与结构评估
        top = QHBoxLayout()
        top.setSpacing(10)
        self.view = ImageView()
        self.panel = StructurePanel()
        top.addWidget(self.view, 1)
        top.addWidget(self.panel, 0)
        lay.addLayout(top, 1)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.card_kcal = MetricCard("总热量", "kcal")
        self.card_weight = MetricCard("总重量", "g")
        self.card_protein = MetricCard("蛋白质", "g")
        self.card_fiber = MetricCard("膳食纤维", "g")
        self.card_share = MetricCard("占每日需求", "%")
        for c in (self.card_kcal, self.card_weight, self.card_protein,
                  self.card_fiber, self.card_share):
            row.addWidget(c)
        lay.addLayout(row)

        self.tb_dishes = QTableWidget(0, 7)
        self.tb_dishes.setHorizontalHeaderLabels(
            ["菜品", "烹调", "置信度", "重量", "热量", "蛋白质", "纤维"]
        )
        self.tb_dishes.verticalHeader().setVisible(False)
        self.tb_dishes.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tb_dishes.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tb_dishes.setShowGrid(False)
        self.tb_dishes.setFixedHeight(232)
        self.tb_dishes.verticalHeader().setDefaultSectionSize(28)
        head = self.tb_dishes.horizontalHeader()
        # 数值列定宽（保证表头不被裁），菜品列吃掉剩余宽度
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 7):
            head.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
            self.tb_dishes.setColumnWidth(i, 76)
        head.setStretchLastSection(False)
        self.tb_dishes.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay.addWidget(self.tb_dishes)

        self.lb_detail = QLabel("")
        self.lb_detail.setObjectName("Hint")
        self.lb_detail.setWordWrap(True)
        lay.addWidget(self.lb_detail)
        return page

    def _build_tab_advice(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(10)

        # 系统按规则算出的「该加什么」——不依赖大模型，先给结论
        self.box_add = QFrame()
        self.box_add.setObjectName("Card")
        add_lay = QVBoxLayout(self.box_add)
        add_lay.setContentsMargins(14, 10, 14, 10)
        add_lay.setSpacing(4)
        lb_add = QLabel("这一餐的缺口与补充建议（按膳食指南规则生成）")
        lb_add.setObjectName("CardTitle")
        self.lb_additions = QLabel("完成一次分析后显示。")
        self.lb_additions.setWordWrap(True)
        self.lb_additions.setStyleSheet("font-size:13px; line-height:170%;")
        add_lay.addWidget(lb_add)
        add_lay.addWidget(self.lb_additions)
        self.box_add.setVisible(False)
        lay.addWidget(self.box_add)

        row = QHBoxLayout()
        self.lb_advice_title = QLabel("AI 膳食建议")
        self.lb_advice_title.setObjectName("CardTitle")
        row.addWidget(self.lb_advice_title)
        row.addStretch(1)
        self.btn_regen = QPushButton("重新生成")
        self.btn_regen.clicked.connect(self._restart_llm)
        self.btn_export = QPushButton("导出报告")
        self.btn_export.clicked.connect(self._export_report)
        row.addWidget(self.btn_regen)
        row.addWidget(self.btn_export)
        lay.addLayout(row)

        self.txt_advice = QTextBrowser()
        self.txt_advice.setOpenExternalLinks(True)
        self.txt_advice.setPlaceholderText("完成一次分析后，这里会显示针对这一餐的建议。")
        # 注意：Qt 的 setMarkdown 不会套用 document 的默认样式表，
        # 标题会渲染成黑色、零间距（实测 blockFormat 的 topMargin 为 0），
        # 所以标题样式在 _render_advice() 里逐块刷一遍。
        base_font = self.txt_advice.font()
        base_font.setPointSizeF(10.5)
        self.txt_advice.setFont(base_font)
        lay.addWidget(self.txt_advice, 1)
        return page

    # ------------------------------------------------------------ 建议排版
    def _render_advice(self, markdown: str) -> None:
        self.txt_advice.setMarkdown(markdown)
        self._style_advice()

    def _style_advice(self) -> None:
        """给 Markdown 渲染出来的标题/正文补上颜色、字号与行距。"""
        doc = self.txt_advice.document()
        cursor = QTextCursor(doc)
        blk = doc.begin()
        while blk.isValid():
            pos = blk.position()
            bf = blk.blockFormat()
            heading = bf.headingLevel()
            if heading:
                bf.setTopMargin(0 if blk.blockNumber() == 0 else 20)
                bf.setBottomMargin(6)
                bf.setLineHeight(130, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
                cursor.setPosition(pos)
                cursor.setBlockFormat(bf)

                fmt = QTextCharFormat()
                fmt.setForeground(QColor(0x1A, 0x4F, 0xBB))
                fmt.setFontPointSize(13.5)
                fmt.setFontWeight(QFont.Weight.DemiBold)
                cursor.setPosition(pos)
                cursor.setPosition(pos + len(blk.text()), QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(fmt)
            else:
                bf.setBottomMargin(5)
                bf.setLineHeight(172, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
                cursor.setPosition(pos)
                cursor.setBlockFormat(bf)
            blk = blk.next()

    def _build_tab_history(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(10)

        row = QHBoxLayout()
        self.lb_today = QLabel("今日暂无记录")
        self.lb_today.setObjectName("CardTitle")
        row.addWidget(self.lb_today)
        row.addStretch(1)
        btn = QPushButton("刷新")
        btn.clicked.connect(self._refresh_history)
        row.addWidget(btn)
        lay.addLayout(row)

        self.tb_history = QTableWidget(0, 5)
        self.tb_history.setHorizontalHeaderLabels(["时间", "餐次", "菜品", "热量", "重量"])
        self.tb_history.verticalHeader().setVisible(False)
        self.tb_history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tb_history.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tb_history.setShowGrid(False)
        head = self.tb_history.horizontalHeader()
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for i in (0, 1, 3, 4):
            head.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self.tb_history, 1)
        return page

    # ------------------------------------------------------------ 用户
    def _refresh_users(self, select: Optional[str] = None) -> None:
        names = self.store.names()
        target = select or self.store.active_name
        self.cb_user.blockSignals(True)
        self.cb_user.clear()
        self.cb_user.addItems(names)
        if names:
            idx = names.index(target) if target in names else 0
            self.cb_user.setCurrentIndex(idx)
        self.cb_user.blockSignals(False)
        self._on_user_changed(self.cb_user.currentIndex())

    def _current_user_name(self) -> str:
        return self.cb_user.currentText().strip()

    def _current_user(self) -> Optional[Dict[str, Any]]:
        name = self._current_user_name()
        return self.store.get(name) if name else None

    def _on_user_changed(self, _idx: int) -> None:
        user = self._current_user()
        if not user:
            self.lb_user.setText("还没有用户，点「新建」录入身体信息")
            self.btn_run.setEnabled(False)
            return
        self.store.set_active(user["name"])
        self.lb_user.setText(
            f"{user['gender']} · {user['age']} 岁 · {user['height_cm']:.0f} cm · "
            f"{user['weight_kg']:.0f} kg\n活动量：{ACTIVITY_NAMES.get(int(user['activity_level']), '')}"
        )
        self.btn_run.setEnabled(self._image_path is not None)

    def _new_user(self) -> None:
        dlg = UserDialog(None, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            profile = self.store.upsert(dlg.values())
        except ValueError as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._refresh_users(profile["name"])
        self.status.showMessage(f"已保存用户：{profile['name']}")

    def _edit_user(self) -> None:
        user = self._current_user()
        if not user:
            return
        dlg = UserDialog(user, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            profile = self.store.upsert(dlg.values())
        except ValueError as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._refresh_users(profile["name"])

    def _delete_user(self) -> None:
        name = self._current_user_name()
        if not name:
            return
        if QMessageBox.question(self, "删除用户", f"确定删除用户「{name}」？历史记录会保留。") \
                != QMessageBox.StandardButton.Yes:
            return
        self.store.delete(name)
        self._refresh_users()

    # ------------------------------------------------------------ 图片
    def load_image(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(self, "打不开", f"文件不存在：\n{p}")
            return
        self._image_path = str(p)
        pm = QPixmap(str(p))
        if pm.isNull():
            QMessageBox.warning(self, "打不开", "这张图片无法读取，请换一张。")
            self._image_path = None
            return
        self.drop.set_preview(str(p), pm)
        self.view.clear_image("已选择图片，点左侧「开始分析」")
        self.btn_run.setEnabled(bool(self._current_user()))
        self.status.showMessage(f"已选择：{p.name}")

    def _clear_image(self) -> None:
        self._image_path = None
        self.drop.thumb.setPixmap(QPixmap())
        self.drop.thumb.setText("点这里选择图片\n或把图片拖进来")
        self.drop.name.setText("")
        self.view.clear_image()
        self.panel.clear()
        self.box_add.setVisible(False)
        self.btn_run.setEnabled(False)

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择食物照片", str(PROJECT_ROOT / "data"),
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)",
        )
        if path:
            self.load_image(path)

    def _load_sample(self) -> None:
        test_dir = PROJECT_ROOT.parent / "数据" / "test" / "images"
        if not test_dir.exists():
            test_dir = PROJECT_ROOT / "data" / "images"
        if not test_dir.exists():
            QMessageBox.information(self, "没有示例", "没有找到示例图片目录。")
            return
        files = [p for p in test_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        if not files:
            QMessageBox.information(self, "没有示例", f"{test_dir} 里没有图片。")
            return
        self.load_image(str(random.choice(files)))

    # ------------------------------------------------------------ 分析
    def _meal_type(self) -> str:
        btn = self.meal_group.checkedButton()
        return btn.text() if btn else "午餐"

    def _height_level(self) -> str:
        btn = self.height_group.checkedButton()
        return btn.text() if btn else DEFAULT_HEIGHT_LEVEL

    def _set_busy(self, busy: bool, text: str = "") -> None:
        self.btn_run.setEnabled(not busy and self._image_path is not None and bool(self._current_user()))
        self.btn_run.setText("分析中…" if busy else "开始分析")
        if text:
            self.status.showMessage(text)

    def _start_analyze(self) -> None:
        user = self._current_user()
        if not user:
            QMessageBox.information(self, "缺少用户", "请先新建一个用户，录入年龄、身高、体重。")
            return
        if not self._image_path:
            QMessageBox.information(self, "缺少图片", "请先选择或拖入一张食物照片。")
            return

        self._set_busy(True, "正在识别…（首次运行需要加载模型，约 5-10 秒）")
        self.tb_dishes.setRowCount(0)
        self.txt_advice.setPlainText("")
        self.panel.clear()
        self.box_add.setVisible(False)
        self._llm_buffer = []
        self._saved_llm = ""

        self.worker = AnalyzeWorker(self.analyzer, self._image_path, user,
                                    self._meal_type(), True, self._height_level())
        self.worker.done.connect(self._on_analyzed)
        self.worker.failed.connect(self._on_analyze_failed)
        self.worker.start()

    def _on_analyze_failed(self, msg: str) -> None:
        self._set_busy(False, "分析失败")
        QMessageBox.critical(self, "分析失败", msg + "\n\n详细日志见 logs/app.log")

    def _on_analyzed(self, result: Dict[str, Any]) -> None:
        self._result = result
        self._set_busy(False, f"识别完成，用时 {result['elapsed_s']} 秒")
        self.tabs.setCurrentIndex(0)

        self.view.set_image(ndarray_to_pixmap(result["annotated"]))

        totals = result["totals"]
        self.card_kcal.set_value(f"{totals['calories']:.0f}")
        self.card_weight.set_value(f"{totals['weight_g']:.0f}")
        self.card_protein.set_value(f"{totals['protein_g']:.0f}")
        self.card_fiber.set_value(f"{totals.get('fiber_g', 0):.1f}")
        self.card_share.set_value(f"{result['meal_share']:.0f}")

        # 餐次配额与结构
        self.panel.show_budget(result)

        dishes = result["dishes"]
        self.tb_dishes.setRowCount(len(dishes))
        for r, d in enumerate(dishes):
            cells = [
                d.get("name_cn") or d.get("name", ""),
                d.get("cooking", "—"),
                f"{float(d.get('conf', 0)):.2f}",
                f"{float(d.get('weight_g', 0)):.0f} g",
                f"{float(d.get('calories', 0)):.0f}",
                f"{float(d.get('protein_g', 0)):.0f} g",
                f"{float(d.get('fiber_g', 0)):.1f} g",
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if c == 1 and d.get("inflammation") == "高":
                    item.setForeground(Qt.GlobalColor.darkRed)
                if c == 0:
                    tip = [f"烹调方式：{d.get('cooking', '未知')}",
                           f"类别：{d.get('category', '')}",
                           f"钠 {float(d.get('sodium_mg', 0)):.0f} mg / 脂肪 {float(d.get('fat_g', 0)):.1f} g"]
                    if d.get("inflammation") == "高":
                        tip.append("促炎负荷高（油炸或加工肉）")
                    if d.get("food_note"):
                        tip.append(d["food_note"])
                    item.setToolTip("\n".join(tip))
                self.tb_dishes.setItem(r, c, item)

        bits = []
        if not dishes:
            bits.append("没有识别到菜品：请确认照片里食物在盘内、光线充足。")
        rice = result.get("rice_rule") or {}
        if rice.get("level"):
            bits.append("米饭：" + rice.get("note", ""))
        bits.append(f"高度档位：{result.get('height_level', '常规')}"
                    f"（×{result.get('height_multiplier', 1.0):.2f}）")
        if result.get("warnings"):
            bits.extend(result["warnings"])
        if result.get("rejected"):
            names = "、".join(sorted({d["name"] for d in result["rejected"]})[:5])
            bits.append(f"已过滤非食物类别：{names}")
        self.lb_detail.setText("　·　".join(bits))

        # 缺口与补充建议（规则层，不依赖大模型）
        add_items = (result.get("budget") or {}).get("additions") or []
        if add_items:
            html = []
            for a in add_items:
                add = a.get("action") == "add"
                color = "#1a6b43" if add else "#a8412f"
                mark = "＋" if add else "－"
                html.append(
                    f'<div style="margin:2px 0;">'
                    f'<span style="color:{color};font-weight:600;">{mark}</span> '
                    f'<span style="color:#2b3138;">{a.get("text", "")}</span></div>'
                )
            self.lb_additions.setText("".join(html))
            self.box_add.setVisible(True)
        else:
            self.box_add.setVisible(False)

        if not dishes:
            self._render_advice("这一餐没有识别到菜品，暂时无法给出营养建议。")
            return

        self._save_record(result)
        self._start_llm()

    def _save_record(self, result: Dict[str, Any]) -> None:
        try:
            self._record_id = self.db.save_record(
                image_path=result["image_path"],
                user_info=result["user"],
                foods=result["dishes"],
                totals=result["totals"],
                health=result["health"],
                llm_output="",
                user_name=result["user"].get("name", ""),
                meal_type=result.get("meal_type", ""),
            )
        except Exception as exc:                        # noqa: BLE001
            log.warning(f"历史记录保存失败：{exc}")

    # ------------------------------------------------------------ 建议
    def _start_llm(self) -> None:
        if not self._result:
            return
        user = self._result["user"]
        try:
            # 排除刚写入的本餐记录，否则「今日已吃」会把这一餐算两遍
            day = self.db.day_summary(user.get("name", ""), exclude_id=self._record_id)
        except Exception:                               # noqa: BLE001
            day = {}

        rice = self._result.get("rice_rule") or {}
        level = int(rice.get("level", 0) or 0)
        if level == 2:
            rice_note = ("米饭被菜遮挡、检测框偏小，已按「占食物总面积 1/3」反推分量"
                         f"（原始占比 {float(rice.get('detected_ratio') or 0):.0%}）")
        elif level == 1:
            rice_note = f"米饭检测面积占食物的 {float(rice.get('detected_ratio') or 0):.0%}，按检测值计算，未做修正"
        elif level == 3:
            rice_note = "图片中米饭占比过小，判为误检已剔除，这一餐按无主食处理"
        else:
            rice_note = ""

        prompt = build_meal_prompt(self._result, day=day, rice_note=rice_note)
        self._llm_buffer = []
        self._render_advice("_正在生成建议…_")

        self._llm_worker = LlmWorker(prompt)
        self._llm_worker.chunk.connect(self._on_llm_chunk)
        self._llm_worker.finished_text.connect(self._on_llm_done)
        self._llm_worker.start()
        self._flush_timer.start()

    def _restart_llm(self) -> None:
        if not self._result:
            QMessageBox.information(self, "还没有结果", "请先分析一张图片。")
            return
        if self._llm_worker and self._llm_worker.isRunning():
            self._llm_worker.stop()
        self._start_llm()

    def _on_llm_chunk(self, _piece: str) -> None:
        self._llm_buffer.append(_piece)

    def _flush_llm(self) -> None:
        if not self._llm_buffer:
            return
        text = format_advice("".join(self._llm_buffer))
        self._render_advice(text)
        bar = self.txt_advice.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_llm_done(self, text: str) -> None:
        self._flush_timer.stop()
        self._flush_llm()
        self._saved_llm = text.strip()
        if self._saved_llm:
            self._render_advice(format_advice(self._saved_llm))
            self._update_llm_text()
        else:
            self._render_advice("_没有取到建议内容，请检查 Ollama 是否已启动。_")

    def _update_llm_text(self) -> None:
        if self._record_id and self._saved_llm:
            try:
                with self.db._connect() as conn:        # noqa: SLF001
                    conn.execute("UPDATE history SET llm_output = ? WHERE id = ?",
                                 (self._saved_llm, self._record_id))
            except Exception as exc:                    # noqa: BLE001
                log.debug(f"建议写回数据库失败：{exc}")
        self._refresh_history()

    # ------------------------------------------------------------ 历史
    def _refresh_history(self) -> None:
        name = self._current_user_name()
        if not name:
            self.tb_history.setRowCount(0)
            self.lb_today.setText("今日暂无记录")
            return
        try:
            day = self.db.day_summary(name)
            records = self.db.user_records(name, limit=50)
        except Exception as exc:                        # noqa: BLE001
            log.warning(f"读取历史失败：{exc}")
            return

        if day.get("count"):
            meals = "、".join(f"{k} {v:.0f}" for k, v in day.get("meals", {}).items())
            self.lb_today.setText(
                f"今日已记录 {day['count']} 餐（{meals} kcal）　合计 {day['calories']:.0f} kcal　"
                f"蛋白质 {day['protein_g']:.0f} g"
            )
        else:
            self.lb_today.setText("今日暂无记录")

        self.tb_history.setRowCount(len(records))
        for r, rec in enumerate(records):
            foods = rec.get("foods") or []
            names = "、".join(str(f.get("name_cn") or f.get("name")) for f in foods[:4])
            if len(foods) > 4:
                names += f" 等 {len(foods)} 样"
            cells = [
                str(rec.get("created_at", ""))[5:16],
                rec.get("meal_type") or "-",
                names or "-",
                f"{float(rec.get('total_calories') or 0):.0f}",
                f"{float(rec.get('total_weight_g') or 0):.0f}",
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c >= 3:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.tb_history.setItem(r, c, item)

    # ------------------------------------------------------------ 报告
    def _export_report(self) -> None:
        if not self._result:
            QMessageBox.information(self, "还没有结果", "请先分析一张图片。")
            return
        r = self._result
        totals, health = r["totals"], r["health"]
        budget = r.get("budget") or {}
        targets = budget.get("targets") or {}
        lines = [
            f"# 膳食分析报告（{r['meal_type']}）",
            "",
            f"- 就餐人：{r['user'].get('name')}（{r['user'].get('gender')}，{r['user'].get('age')} 岁，"
            f"{r['user'].get('height_cm')} cm，{r['user'].get('weight_kg')} kg）",
            f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"- 图片：{Path(r['image_path']).name}",
            f"- 食物高度档位：{r.get('height_level', '常规')}（×{r.get('height_multiplier', 1.0):.2f}）",
            "",
            "## 识别结果",
            "",
            "| 菜品 | 烹调 | 置信度 | 重量(g) | 热量(kcal) | 蛋白(g) | 脂肪(g) | 碳水(g) | 纤维(g) | 钠(mg) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for d in r["dishes"]:
            lines.append(
                f"| {d.get('name_cn')} | {d.get('cooking', '')} | {float(d.get('conf', 0)):.2f} "
                f"| {float(d.get('weight_g', 0)):.0f} | {float(d.get('calories', 0)):.0f} "
                f"| {float(d.get('protein_g', 0)):.1f} | {float(d.get('fat_g', 0)):.1f} "
                f"| {float(d.get('carb_g', 0)):.1f} | {float(d.get('fiber_g', 0)):.1f} "
                f"| {float(d.get('sodium_mg', 0)):.0f} |"
            )
        lines += [
            "",
            f"合计：{totals['weight_g']:.0f} g，{totals['calories']:.0f} kcal；"
            f"蛋白质 {totals['protein_g']:.0f} g / 脂肪 {totals['fat_g']:.0f} g / "
            f"碳水 {totals['carb_g']:.0f} g / 纤维 {totals.get('fiber_g', 0):.1f} g / "
            f"钠 {totals.get('sodium_mg', 0):.0f} mg",
            "",
            f"BMR {health['bmr']:.0f} kcal，TDEE {health['tdee']:.0f} kcal，BMI {health['bmi']}"
            f"（{health['bmi_label']}），本餐占每日需求 {r['meal_share']:.0f}%",
        ]

        # ---- 餐次配额与结构评估 ----
        if targets:
            lines += [
                "",
                f"## 餐次配额与结构评估（得分 {budget.get('score', 0)}，"
                f"达标 {budget.get('good', 0)}/{budget.get('total', 0)}）",
                "",
                f"这一餐按「{r['meal_type']}占每日需求 {targets.get('meal_share_text')}」"
                f"折算，建议热量 {targets['kcal_min']:.0f}-{targets['kcal_max']:.0f} kcal，"
                f"实际 {totals['calories']:.0f} kcal。",
                "",
                "| 项目 | 实际 | 参考 | 判断 |",
                "|---|---|---|---|",
            ]
            mark = {"ok": "达标", "low": "偏低", "high": "偏高", "warn": "注意"}
            for c in budget.get("checks") or []:
                lines.append(f"| {c['label']} | {c['actual_text']} | {c['target_text']} "
                             f"| {mark.get(c['status'], c['status'])} |")

        # ---- 该加什么 ----
        adds = budget.get("additions") or []
        if adds:
            lines += ["", "## 这一餐的缺口与补充建议", ""]
            for a in adds:
                prefix = "＋" if a.get("action") == "add" else "－"
                lines.append(f"- {prefix} {a.get('text')}")

        # ---- 微量元素 ----
        micro = budget.get("micro") or []
        if micro:
            lines += ["", "## 微量元素（本餐 / 本餐目标）", ""]
            for m in micro[:8]:
                lines.append(f"- {m['name']}：{m['actual']}{m['unit']} / "
                             f"{m['target']}{m['unit']}（{m['pct']}%）")

        rice = r.get("rice_rule") or {}
        if rice.get("level"):
            lines += ["", f"> 米饭处理：{rice.get('note')}"]
        if r.get("warnings"):
            lines += ["", "> 提示：" + "；".join(r["warnings"])]

        if self._saved_llm:
            lines += ["", "## AI 膳食建议", "", self._saved_llm]

        lines += ["", "---", "",
                  f"> 营养数据来源：{budget.get('source') or '《中国食物成分表》第 6 版'}；"
                  "本报告为估算结果，不构成医疗建议。"]

        out_dir = PROJECT_ROOT / "data" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        self.status.showMessage(f"报告已保存：{path}")
        QMessageBox.information(self, "已导出", f"报告已保存到：\n{path}")

    # ------------------------------------------------------------ 其他
    def _start_warmup(self) -> None:
        self.warm = WarmupWorker(self.analyzer)
        self.warm.ready.connect(self._on_warm)
        self.warm.failed.connect(self._on_warm_failed)
        self.warm.start()

    def _on_warm(self) -> None:
        self._warm = True
        self.status.showMessage("模型已就绪")

    def _on_warm_failed(self, msg: str) -> None:
        self.status.showMessage("模型加载失败")
        QMessageBox.critical(self, "模型加载失败",
                             msg + "\n\n请确认 models/weights/chinese_food.pt 存在，"
                                   "且显卡驱动正常（或把 config.yaml 的 detection.device 改成 cpu）。")

    def _check_ollama(self) -> None:
        self.ollama_status = OllamaStatusWorker()
        self.ollama_status.status.connect(self._on_ollama_status)
        self.ollama_status.start()

    def _on_ollama_status(self, ok: bool, text: str) -> None:
        self.lb_ollama.setText(text)
        self.lb_ollama.setStyleSheet("color:#2f8f5b;" if ok else "color:#b57b1e;")

    # 拖拽到整个窗口也能收图
    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in IMAGE_SUFFIXES:
                self.load_image(str(p))
                break

    def closeEvent(self, event):  # noqa: N802
        for w in (getattr(self, "_llm_worker", None), getattr(self, "worker", None)):
            if w is not None and w.isRunning():
                if hasattr(w, "stop"):
                    w.stop()
                w.wait(1500)
        super().closeEvent(event)


def run() -> int:
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("膳食识别与营养建议系统")
    app.setStyleSheet(MAIN_QSS)
    win = MainWindow()
    win.show()
    return app.exec()
