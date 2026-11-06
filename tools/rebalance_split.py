# -*- coding: utf-8 -*-
"""把 train 里少量的「整组源图」移到 valid/test，保证每个类别在验证集里有足够样本。

为什么需要：
    Roboflow 导出的数据同一张源图有 3-4 个增强副本（文件名只差 .rf.<hash>）。
    如果按文件随机抽样做验证集，很容易出现「某类验证集只有 1 个框」——
    mAP 完全不可信。正确做法是按「源图组」整组移动，且整组不能同时出现在
    train 与 valid（否则就是数据泄漏）。

特性：
    * 只按源图组移动，移动后该源图完全离开 train，不产生泄漏
    * 只处理指定类别（默认新增的 8 类），不动原有 12 类的官方划分
    * --dry-run 先看要动多少
    * 移动是 shutil.move（不是删除），文件只是换了目录

用法：
    python tools/rebalance_split.py --root "D:/Work/膳食模型/数据" \
        --yaml "D:/Work/膳食模型/数据/data_20class.yaml" \
        --ids 12 13 14 15 16 17 18 19 --valid-target 12 --test-target 8 --dry-run
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm_name(p: str | Path) -> str:
    b = Path(p).stem
    b = re.sub(r"\.rf\.[0-9a-f]+$", "", b)
    for suf in ("_jpg", "_png", "_jpeg"):
        b = re.sub(re.escape(suf) + "$", "", b, flags=re.I)
    return b.lower()


def count_split_boxes(root: Path, split: str, ids: Set[int]) -> Counter:
    c: Counter = Counter()
    for lp in (root / split / "labels").glob("*.txt"):
        for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
            f = line.split()
            if not f:
                continue
            try:
                cid = int(f[0])
            except ValueError:
                continue
            if cid in ids:
                c[cid] += 1
    return c


def build_groups(root: Path, ids: Set[int]) -> Dict[str, List[Path]]:
    """train 里含目标类别的图片，按归一化源图名分组。"""
    groups: Dict[str, List[Path]] = defaultdict(list)
    for img in (root / "train" / "images").iterdir():
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lp = root / "train" / "labels" / (img.stem + ".txt")
        if not lp.exists():
            continue
        has = False
        for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
            f = line.split()
            if f and int(f[0]) in ids:
                has = True
                break
        if has:
            groups[norm_name(img)].append(img)
    return groups


def group_class_counts(root: Path, files: Sequence[Path]) -> Counter:
    c: Counter = Counter()
    for img in files:
        lp = root / "train" / "labels" / (img.stem + ".txt")
        if not lp.exists():
            continue
        for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
            f = line.split()
            if f:
                c[int(f[0])] += 1
    return c


def move_group(root: Path, files: Sequence[Path], dst_split: str, dry: bool) -> int:
    moved = 0
    for img in files:
        src_lbl = root / "train" / "labels" / (img.stem + ".txt")
        dst_img = root / dst_split / "images" / img.name
        dst_lbl = root / dst_split / "labels" / (img.stem + ".txt")
        if dry:
            moved += 1
            continue
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img), dst_img)
        if src_lbl.exists():
            shutil.move(str(src_lbl), dst_lbl)
        moved += 1
    return moved


def rebalance(root: Path, ids: Set[int], split: str, target: int, groups, used: Set[str],
              max_imgs: int, dry: bool) -> int:
    cur = count_split_boxes(root, split, ids)
    need = {cid: max(0, target - cur[cid]) for cid in ids}
    if not any(need.values()):
        print(f"  [{split}] 各类已达标：{dict(cur)}")
        return 0

    print(f"  [{split}] 现状 {dict(sorted(cur.items()))}  目标每类 ≥{target}")
    print(f"        缺口 {dict(sorted((k, v) for k, v in need.items() if v))}")

    order = list(groups.keys())
    random.Random(42).shuffle(order)
    # 优先移动「含缺口类别最多」的组，少浪费图片
    def score(name: str) -> int:
        gc = group_class_counts(root, groups[name])
        return sum(min(gc[cid], need[cid]) for cid in ids if need[cid] > 0)

    order.sort(key=score, reverse=True)

    moved_imgs = 0
    moved_boxes: Counter = Counter()
    for name in order:
        if not any(v > 0 for v in need.values()):
            break
        if moved_imgs >= max_imgs:
            print(f"        已达单次移动上限 {max_imgs} 张，停止")
            break
        if name in used:
            continue
        gc = group_class_counts(root, groups[name])
        # 这组至少对某个缺口类有贡献才移
        if not any(gc[cid] > 0 and need[cid] > 0 for cid in ids):
            continue
        n = move_group(root, groups[name], split, dry)
        used.add(name)
        moved_imgs += n
        for cid in ids:
            if gc[cid] and need[cid] > 0:
                take = min(gc[cid], need[cid])
                need[cid] -= take
                moved_boxes[cid] += gc[cid]

    print(f"        移动 {moved_imgs} 张图片，新增框 {dict(sorted(moved_boxes.items()))}")
    return moved_imgs


def main() -> int:
    ap = argparse.ArgumentParser(description="按源图组重划分 train→valid/test")
    ap.add_argument("--root", required=True)
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--ids", nargs="+", type=int, required=True, help="要补的类别 ID")
    ap.add_argument("--valid-target", type=int, default=12)
    ap.add_argument("--test-target", type=int, default=8)
    ap.add_argument("--max-ratio", type=float, default=0.12, help="最多移动 train 目标类图片的比例")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    ids = set(args.ids)
    if not (root / "train" / "images").is_dir():
        sys.exit(f"目录结构不对：{root}/train/images 不存在")

    groups = build_groups(root, ids)
    total_imgs = sum(len(v) for v in groups.values())
    max_imgs = int(total_imgs * args.max_ratio)
    print(f"train 中含目标类别的源图组 {len(groups)} 个，共 {total_imgs} 张图片")
    print(f"单次最多移动 {max_imgs} 张（{args.max_ratio:.0%}）")

    used: Set[str] = set()
    print("\n== 补 test ==")
    rebalance(root, ids, "test", args.test_target, groups, used, max_imgs, args.dry_run)
    print("\n== 补 valid ==")
    rebalance(root, ids, "valid", args.valid_target, groups, used, max_imgs, args.dry_run)

    print("\n最终各 split 目标类框数：")
    for sp in ("train", "valid", "test"):
        c = count_split_boxes(root, sp, ids)
        print(f"  {sp:6s} {dict(sorted(c.items()))}")
    if args.dry_run:
        print("\n[干跑] 未移动任何文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
