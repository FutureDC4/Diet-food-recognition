# -*- coding: utf-8 -*-
"""数据集画像：用于对比不同来源数据集的「域差异」，直接出论文表格。

统计项：
  - 图片数 / 框数 / 每图框数分布
  - 框面积占比（相对整图）→ 反映「单菜特写」还是「餐盘分格」
  - 框中心到画面中心的距离 → 反映「摆盘居中」程度
  - 每类实例数（长尾分析）

用法：
    python tools/dataset_stats.py "D:/Work/datasets/chinese_food" --name cafeteria
    python tools/dataset_stats.py "D:/Work/datasets/s_workspace" --name mixed --csv out.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics
import sys
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_cfg(root: Path) -> dict:
    for cand in (root / "data.yaml", root / "dataset.yaml"):
        if cand.exists():
            import yaml

            cfg = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
            names = cfg.get("names")
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            cfg["names"] = names or []
            return cfg
    sys.exit(f"[错误] 找不到 data.yaml: {root}")


def resolve_split(root: Path, cfg: dict, split: str) -> Path | None:
    val = cfg.get(split) or (cfg.get("val") if split == "valid" else None)
    if not val:
        return None
    p = Path(val)
    if not p.is_absolute():
        p = (root / p).resolve()
    if not p.exists():
        # Roboflow 导出常见：yaml 里写 ../train/images
        alt = (root / split / "images").resolve()
        return alt if alt.exists() else None
    return p


def split_stats(root: Path, cfg: dict, split: str, names: list[str]):
    idir = resolve_split(root, cfg, split)
    if idir is None:
        return None
    ldir = Path(str(idir).replace(f"{os.sep}images", f"{os.sep}labels"))
    if not ldir.exists():
        ldir = idir.parent.parent / "labels"

    stems = [os.path.splitext(f)[0] for f in os.listdir(idir)
             if f.lower().endswith(IMG_EXT)]
    boxes_per_img = []
    area_ratios = []
    center_dist = []
    cls_cnt = collections.Counter()
    total = 0

    for stem in stems:
        lp = ldir / (stem + ".txt")
        if not lp.exists():
            boxes_per_img.append(0)
            continue
        n = 0
        for line in lp.read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) != 5:
                continue
            try:
                c = int(float(f[0]))
                cx, cy, w, h = (float(x) for x in f[1:])
            except ValueError:
                continue
            n += 1
            total += 1
            area_ratios.append(w * h)
            # 中心偏移（归一化，0=正中）
            center_dist.append(((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5)
            if 0 <= c < len(names):
                cls_cnt[names[c]] += 1
        boxes_per_img.append(n)

    def med(x):
        return round(statistics.median(x), 4) if x else 0.0

    return {
        "split": split,
        "images": len(stems),
        "boxes": total,
        "boxes_per_img_median": med(boxes_per_img),
        "boxes_per_img_mean": round(total / max(1, len(stems)), 2),
        "single_obj_ratio": round(sum(1 for b in boxes_per_img if b == 1) / max(1, len(stems)), 3),
        "multi_obj_ratio": round(sum(1 for b in boxes_per_img if b >= 2) / max(1, len(stems)), 3),
        "empty_ratio": round(sum(1 for b in boxes_per_img if b == 0) / max(1, len(stems)), 3),
        "area_ratio_median": med(area_ratios),
        "area_ratio_mean": round(sum(area_ratios) / max(1, len(area_ratios)), 4),
        "center_dist_median": med(center_dist),
        "classes_present": len(cls_cnt),
        "cls": cls_cnt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--name", default=None)
    ap.add_argument("--split", default="all", choices=["all", "train", "valid", "test"])
    ap.add_argument("--csv", default=None, help="把每类实例数写到 csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        sys.exit(f"[错误] 目录不存在: {root}")
    cfg = find_cfg(root)
    names = cfg["names"]
    tag = args.name or root.name

    print("=" * 68)
    print(f"数据集: {tag}   路径: {root}")
    print(f"声明类别数: {len(names)}")
    print("=" * 68)

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    results = []
    for sp in splits:
        r = split_stats(root, cfg, sp, names)
        if r:
            results.append(r)

    hdr = (f"{'split':6s} {'图':>6s} {'框':>7s} {'框/图':>6s} {'单目标占比':>9s} "
           f"{'多目标占比':>9s} {'面积中位':>8s} {'中心偏移':>8s} {'出现类别':>8s}")
    print(hdr)
    print("-" * 68)
    for r in results:
        print(f"{r['split']:6s} {r['images']:6d} {r['boxes']:7d} "
              f"{r['boxes_per_img_mean']:6.2f} {r['single_obj_ratio']*100:8.1f}% "
              f"{r['multi_obj_ratio']*100:8.1f}% {r['area_ratio_median']*100:7.1f}% "
              f"{r['center_dist_median']:8.3f} {r['classes_present']:8d}")

    total = collections.Counter()
    for r in results:
        total.update(r["cls"])
    print(f"\n--- 每类实例数（{len(total)} 类有样本 / 声明 {len(names)} 类）---")
    for name, cnt in total.most_common():
        bar = "#" * min(60, int(cnt / max(1, max(total.values())) * 60))
        print(f"  {name:38s} {cnt:5d} {bar}")
    absent = [n for n in names if total.get(n, 0) == 0]
    if absent:
        print(f"  未出现类别 ({len(absent)}): {absent}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["class", "instances", "ratio"])
            s = sum(total.values()) or 1
            for name, cnt in total.most_common():
                w.writerow([name, cnt, round(cnt / s, 4)])
            for n in absent:
                w.writerow([n, 0, 0])
        print(f"\n已写出 CSV: {args.csv}")

    out_dir = Path(__file__).resolve().parents[1] / "runs" / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / f"{tag}.json"
    json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细 JSON: {json_out}")


if __name__ == "__main__":
    main()
