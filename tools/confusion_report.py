# -*- coding: utf-8 -*-
"""统计检测模型的易混类对与弱类。

用法：
    python tools/confusion_report.py --weights runs/chinese/cn11s_20class/weights/best.pt
    python tools/confusion_report.py --weights xxx/best.pt --split test --device cpu

输出：
    1) 每类的 支持数 / 检出率(recall) / 在本类上的误判来源(FP) / 漏检去向(FN)
    2) 全局 top-N 混淆对（真实类 -> 预测成什么）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/chinese_food_20.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="混淆矩阵使用的置信度阈值")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cpu", help="分析建议用 cpu，不抢训练显存")
    ap.add_argument("--topk", type=int, default=3, help="每类列出前几个混淆对象")
    args = ap.parse_args()

    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import YAML

    data = YAML.load(str(PROJECT_ROOT / args.data))
    names = [data["names"][i] for i in range(len(data["names"]))]

    v = DetectionValidator(
        args=dict(
            model=str(Path(args.weights).resolve()),
            data=str((PROJECT_ROOT / args.data).resolve()),
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            workers=2,
            plots=True,  # ultralytics 只有在 plots=True 时才会累积混淆矩阵
            verbose=False,
            project=str(PROJECT_ROOT / "runs" / "analysis"),
            name=f"cm_{args.split}",
            exist_ok=True,
        )
    )
    v()
    cm = v.confusion_matrix
    # ultralytics ConfusionMatrix: matrix[pred, true]，最后一行/列是 background
    m = cm.matrix.astype(float)
    nc = len(names)
    bg = nc  # background 索引

    tp = np.diag(m)[:nc]
    support = m[:nc, :nc].sum(axis=0) + m[bg, :nc]  # 每个真实类的目标数
    recall = tp / np.maximum(support, 1e-9)

    print("=" * 88)
    print(f"分类别：{args.split} 集，conf={args.conf}  iou={args.iou}")
    print("=" * 88)
    print(f"{'类别':<22}{'真值数':>7}{'检出率':>9}   误判来源(FP) 与 漏检去向(FN)")
    print("-" * 88)
    rows = []
    for i in range(nc):
        # FP: 预测成 i，但真实是别的类
        fp_src = [(names[j], m[i, j]) for j in range(nc) if j != i and m[i, j] > 0]
        fp_src.sort(key=lambda x: -x[1])
        # FN: 真实是 i，但被预测成别的类
        fn_dst = [(names[j], m[j, i]) for j in range(nc) if j != i and m[j, i] > 0]
        fn_dst.sort(key=lambda x: -x[1])
        missed_bg = m[bg, i]
        rows.append((i, support[i], recall[i], fp_src, fn_dst, missed_bg))

    for i, sup, rec, fp_src, fn_dst, mbg in sorted(rows, key=lambda r: r[2]):
        fptxt = ", ".join(f"{n}×{int(c)}" for n, c in fp_src[: args.topk]) or "-"
        fntxt = ", ".join(f"{n}×{int(c)}" for n, c in fn_dst[: args.topk]) or "-"
        print(f"{names[i]:<22}{int(sup):>7}{rec:>8.1%}   被误当成-> {fptxt}")
        print(f"{'':<22}{'':>7}{'':>9}   漏检成-> {fntxt}  | 未检出(背景)×{int(mbg)}")

    print("\n" + "=" * 88)
    print("全局混淆对（真实类 -> 被预测成，双向合并计数）")
    print("=" * 88)
    pairs = []
    for i in range(nc):
        for j in range(i + 1, nc):
            c = m[i, j] + m[j, i]
            if c > 0:
                pairs.append((c, names[j], names[i], m[j, i], m[i, j]))
    pairs.sort(key=lambda x: -x[0])
    for c, a, b, ab, ba in pairs[:12]:
        print(f"  {a:<22} <-> {b:<22} 共 {int(c):>3}   ({a}->{b}: {int(ab)}, {b}->{a}: {int(ba)})")

    print("\n" + "=" * 88)
    print("汇总：最弱 5 类（按检出率）")
    print("=" * 88)
    for i, sup, rec, *_ in sorted(rows, key=lambda r: r[2])[:5]:
        print(f"  {names[i]:<22} 真值 {int(sup):>3} 个，检出率 {rec:>6.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
