# -*- coding: utf-8 -*-
"""上位机冒烟测试：自动建档 → 载图 → 分析 → 截图，用于人工核对界面。

用法：
    python tools/gui_smoke_test.py [--image 路径] [--wait-llm 25]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QTimer                      # noqa: E402
from PySide6.QtWidgets import QApplication             # noqa: E402

from app.main_window import MainWindow                 # noqa: E402
from app.theme import MAIN_QSS                         # noqa: E402

OUT_DIR = PROJECT_ROOT / "runs" / "gui"


def pick_sample() -> str:
    root = PROJECT_ROOT.parent / "数据"
    cand: list[Path] = []
    for split in ("test", "valid"):
        d = root / split / "images"
        if d.exists():
            cand += [p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not cand:
        sys.exit("没有找到示例图片")
    # 优先选标注框最多的（多菜餐盘，最能体现效果）
    best, best_n = None, -1
    for p in cand[:120]:
        lp = root / "test" / "labels" / f"{p.stem}.txt"
        if not lp.exists():
            lp = root / "valid" / "labels" / f"{p.stem}.txt"
        n = len([l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]) if lp.exists() else 0
        if n > best_n:
            best, best_n = p, n
    return str(best or random.choice(cand))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--wait-llm", type=float, default=25.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(MAIN_QSS)

    win = MainWindow()
    win.resize(1360, 880)
    win.show()

    if not win.store.names():
        win.store.upsert({"name": "测试用户", "age": 21, "gender": "男",
                          "height_cm": 175, "weight_kg": 68, "activity_level": 2})
    win._refresh_users("测试用户")

    image = args.image or pick_sample()
    print("样例图片:", image)
    win.load_image(image)

    shots: list[str] = []

    def shoot(name: str) -> str:
        p = OUT_DIR / f"{name}.png"
        win.grab().save(str(p))
        shots.append(str(p))
        print("截图:", p)
        return str(p)

    def after_result(round_no: int = 0) -> None:
        if win._result is None:
            if round_no > 120:
                print("分析超时")
                app.quit()
                return
            QTimer.singleShot(500, lambda: after_result(round_no + 1))
            return
        print("识别到:", [(d.get("name_cn"), round(float(d["weight_g"]))) for d in win._result["dishes"]])
        print("总计:", win._result["totals"], "占比", round(win._result["meal_share"], 1))
        QTimer.singleShot(600, lambda: (shoot("01_识别结果"), QTimer.singleShot(
            int(args.wait_llm * 1000), after_llm)))

    def after_llm() -> None:
        win.tabs.setCurrentIndex(1)
        QTimer.singleShot(500, lambda: (shoot("02_膳食建议"), QTimer.singleShot(400, after_history)))

    def after_history() -> None:
        win.tabs.setCurrentIndex(2)
        QTimer.singleShot(500, lambda: (shoot("03_历史记录"), app.quit()))

    QTimer.singleShot(6000, win._start_analyze)      # 等模型预热完成
    QTimer.singleShot(6500, after_result)

    app.exec()
    print("\n完成，截图：")
    for s in shots:
        print("  ", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
