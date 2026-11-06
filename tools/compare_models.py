# -*- coding: utf-8 -*-
"""两个模型在 test 上的逐类 AP 配对比较 + 显著性判断。"""
from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

import models  # noqa: E402,F401
from ultralytics import YOLO  # noqa: E402

WEIGHTS = {
    "base": "runs/chinese/cn11s_base_18/weights/best.pt",
    "cbam": "runs/chinese/cn11s_cbam_wiou_18/weights/best.pt",
}


def main() -> int:
    cfg = yaml.safe_load((PROJECT_ROOT / "configs/chinese_food_18.yaml").read_text(encoding="utf-8"))
    names = [cfg["names"][i] for i in range(cfg["nc"])]

    res = {}
    for tag, w in WEIGHTS.items():
        m = YOLO(str(PROJECT_ROOT / w))
        r = m.val(data=str(PROJECT_ROOT / "configs/chinese_food_18.yaml"), split="test",
                  imgsz=640, batch=16, device=0, conf=0.001, iou=0.6,
                  verbose=False, plots=False)
        ap = r.box.ap
        res[tag] = (ap if getattr(ap, "ndim", 1) == 1 else ap.mean(1), r.box.ap50)
        print(f"[done] {tag}: mAP50-95={r.box.map:.4f}  mAP50={r.box.map50:.4f}")

    print()
    print(f"{'class':<20}{'base_5095':>11}{'cbam_5095':>11}{'diff':>9}"
          f"{'base_50':>10}{'cbam_50':>10}")
    print("-" * 72)
    diffs = []
    for i, nm in enumerate(names):
        b, c = float(res["base"][0][i]), float(res["cbam"][0][i])
        diffs.append(b - c)
        print(f"{nm:<20}{b:>11.4f}{c:>11.4f}{b - c:>+9.4f}"
              f"{float(res['base'][1][i]):>10.4f}{float(res['cbam'][1][i]):>10.4f}")

    n = len(diffs)
    md, sd = st.mean(diffs), st.stdev(diffs)
    t = md / (sd / math.sqrt(n))
    win = sum(1 for d in diffs if d > 0)
    print("-" * 72)
    print(f"基线更好 {win}/{n} 类 | 平均差 {md:+.4f} | 标准差 {sd:.4f}")
    print(f"配对 t = {t:.2f} (df={n - 1}) | Cohen's d = {md / sd:.2f}")
    print()
    # 单侧 t 临界值（df=17, alpha=0.05）约 1.74
    print("判读：|t| < 1.74 视为无显著差异；1.74~2.5 弱证据；>2.5 较强证据")
    return 0


if __name__ == "__main__":
    sys.exit(main())
