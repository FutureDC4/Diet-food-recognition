# -*- coding: utf-8 -*-
"""把 YOLO 标签画到图片上，用于目视核对标注质量。

用法：
    python tools/preview_labels.py "D:/Work/膳食模型/数据" --split valid -n 6
"""
import argparse
import os
import random

import cv2
import numpy as np
import yaml


def imread_unicode(path):
    """cv2.imread 在 Windows 中文路径下会失败，用 imdecode 绕开。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok

COLORS = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--split", default="valid")
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--out", default="data/output/preview")
    ap.add_argument("--yaml", default=None, help="类别定义文件（默认 <root>/data.yaml）")
    ap.add_argument("--filter", default=None, help="只画文件名含该子串的图片")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    cfg = yaml.safe_load(open(args.yaml or os.path.join(root, "data.yaml"), encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    idir = os.path.join(root, args.split, "images")
    ldir = os.path.join(root, args.split, "labels")
    stems = sorted(
        os.path.splitext(f)[0] for f in os.listdir(idir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if args.filter:
        stems = [s for s in stems if args.filter in s]
    random.seed(0)
    picks = random.sample(stems, min(args.n, len(stems)))
    os.makedirs(args.out, exist_ok=True)

    for stem in picks:
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(idir, stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        img = imread_unicode(img_path)
        if img is None:
            print("!! 读不了:", img_path)
            continue
        h, w = img.shape[:2]
        lp = os.path.join(ldir, stem + ".txt")
        n = 0
        if os.path.exists(lp):
            for line in open(lp, encoding="utf-8"):
                f = line.split()
                if len(f) != 5:
                    continue
                c = int(float(f[0]))
                cx, cy, bw, bh = (float(x) for x in f[1:])
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                col = COLORS[c % len(COLORS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                label = names[c] if c < len(names) else str(c)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
                cv2.putText(img, label, (x1 + 2, max(10, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                n += 1
        out = os.path.join(args.out, f"{stem[:40]}_{n}.jpg")
        imwrite_unicode(out, img)
        print(f"{os.path.basename(out)}  boxes={n}  size={w}x{h}")


if __name__ == "__main__":
    main()
