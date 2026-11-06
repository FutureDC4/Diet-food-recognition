# -*- coding: utf-8 -*-
"""统计各类标注框的尺寸与遮挡程度，用于解释「漏检」的成因。

用法：
    python tools/box_stats.py --data configs/chinese_food_20.yaml --split val

输出每类：目标数 / 中位边长(px) / 小目标占比(面积<32²) / 被同类或其他框遮挡的中位比例
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(p: Path) -> dict:
    import yaml

    return yaml.safe_load(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/chinese_food_20.yaml")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    cfg = load_yaml(PROJECT_ROOT / args.data)
    names = [cfg["names"][i] for i in range(len(cfg["names"]))]
    root = Path(cfg["path"])
    img_rel = str(cfg[args.split])          # 形如 valid/images
    img_dir = root / img_rel
    lbl_dir = root / img_rel.replace("images", "labels")

    per_cls: dict[int, list[tuple[float, float, float]]] = {i: [] for i in range(len(names))}
    for lp in sorted(lbl_dir.glob("*.txt")):
        rows = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            cx, cy, w, h = (float(v) for v in p[1:5])
            rows.append((c, cx, cy, w, h))
        if not rows:
            continue
        # 用像素换算（数据集已统一 resize 到 640）
        boxes = [(c, (cx - w / 2) * 640, (cy - h / 2) * 640, (cx + w / 2) * 640, (cy + h / 2) * 640, w * 640, h * 640)
                 for c, cx, cy, w, h in rows]
        for i, (c, x1, y1, x2, y2, bw, bh) in enumerate(boxes):
            area = max(bw, 1) * max(bh, 1)
            # 遮挡代理：与其他框交集占自身面积的最大比例
            occ = 0.0
            for j, (c2, a1, b1, a2, b2, _, _) in enumerate(boxes):
                if i == j:
                    continue
                iw = max(0.0, min(x2, a2) - max(x1, a1))
                ih = max(0.0, min(y2, b2) - max(y1, b1))
                if iw > 0 and ih > 0:
                    occ = max(occ, iw * ih / area)
            per_cls[c].append((min(bw, bh), area, occ))

    print(f"{'类别':<22}{'目标数':>6}{'中位短边px':>11}{'小目标占比':>10}{'中位遮挡':>9}{'遮挡>30%占比':>12}")
    print("-" * 74)
    rows = []
    for c, vals in per_cls.items():
        if not vals:
            rows.append((names[c], 0, 0, 0, 0, 0))
            continue
        a = np.array(vals, dtype=float)
        short = a[:, 0]
        area = a[:, 1]
        occ = a[:, 2]
        small = float((area < 32 * 32).mean())
        rows.append((names[c], len(vals), float(np.median(short)), small, float(np.median(occ)), float((occ > 0.3).mean())))
    for n, cnt, med, small, occ, occ30 in sorted(rows, key=lambda r: r[2]):
        print(f"{n:<22}{cnt:>6}{med:>11.0f}{small:>9.0%}{occ:>9.2f}{occ30:>11.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
