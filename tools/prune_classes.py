# -*- coding: utf-8 -*-
"""从数据集中永久剔除若干类别，并把剩余类别 ID 重新紧凑编号。

为什么需要：YOLO 要求类别 ID 从 0 连续，删掉中间两类后，
原本的 17/18/19 必须整体前移 2 位，否则训练时 names 对不上。

对「删完一个框都不剩」的图片做的是【连图带标签一起删】而不是保留成空标签：
一张明明有菜却没有框的图，会被当成负样本教模型「这里没东西」，是有害的。

用法：
    python tools/prune_classes.py --data configs/chinese_food_20.yaml --drop braised_egg braised_meat --dry-run
    python tools/prune_classes.py --data configs/chinese_food_20.yaml --drop braised_egg braised_meat --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/chinese_food_20.yaml")
    ap.add_argument("--drop", nargs="+", required=True, help="要剔除的类别名")
    ap.add_argument("--apply", action="store_true", help="真正执行（默认只统计）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不修改（默认行为，写出来更直观）")
    ap.add_argument("--no-backup", action="store_true", help="跳过标签备份（不推荐）")
    args = ap.parse_args()

    import yaml

    cfg_path = PROJECT_ROOT / args.data
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    old_names = [cfg["names"][i] for i in range(len(cfg["names"]))]

    unknown = [c for c in args.drop if c not in old_names]
    if unknown:
        raise SystemExit(f"[错误] 数据集里没有这些类: {unknown}\n现有: {old_names}")
    drop_ids = {old_names.index(c) for c in args.drop}
    keep_names = [n for i, n in enumerate(old_names) if i not in drop_ids]
    # 旧 ID -> 新 ID
    remap = {}
    new_id = 0
    for i, n in enumerate(old_names):
        if i in drop_ids:
            continue
        remap[i] = new_id
        new_id += 1

    root = Path(cfg["path"])
    print(f"数据集根目录: {root}")
    print(f"剔除: {args.drop} -> 剩余 {len(keep_names)} 类")
    print(f"ID 重映射: " + ", ".join(f"{old_names[i]}->{remap[i]}" for i in sorted(remap)))
    print("-" * 78)

    total_stats = {}
    plan = []  # (split, label_path, img_path, kept_lines, removed_boxes)
    for split, cfg_key in (("train", "train"), ("valid", "val"), ("test", "test")):
        if cfg_key not in cfg:
            print(f"[跳过] {split}: 配置里没有 {cfg_key} 字段")
            continue
        img_dir = root / str(cfg[cfg_key])
        lbl_dir = Path(str(img_dir).replace("images", "labels"))
        if not lbl_dir.exists():
            print(f"[跳过] {split}: 找不到 {lbl_dir}")
            continue
        n_img_before = len([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        n_boxes_before = n_removed = 0
        n_empty_after = 0
        for lp in sorted(lbl_dir.glob("*.txt")):
            out_lines = []
            for line in lp.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(float(p[0]))
                n_boxes_before += 1
                if c in drop_ids:
                    n_removed += 1
                    continue
                out_lines.append(f"{remap[c]} " + " ".join(p[1:]))
            if not out_lines:
                n_empty_after += 1
            plan.append((split, lp, None, out_lines, n_removed))
        total_stats[split] = dict(images=n_img_before, boxes_before=n_boxes_before,
                                  removed=n_removed, emptied_imgs=n_empty_after)
        print(f"{split:6s}: 图片 {n_img_before:5d}  框 {n_boxes_before:6d}  "
              f"删框 {n_removed:5d}  剩 {n_boxes_before - n_removed:6d}  "
              f"删完无框的图片 {n_empty_after:4d}")

    print("-" * 78)
    if not args.apply:
        print("这是预览（dry-run），没有修改任何文件。确认无误后加 --apply 执行。")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not args.no_backup:
        bak = root / f"_labels_backup_{stamp}"
        for split, cfg_key in (("train", "train"), ("valid", "val"), ("test", "test")):
            if cfg_key not in cfg:
                continue
            src = Path(str(root / str(cfg[cfg_key])).replace("images", "labels"))
            if src.exists():
                shutil.copytree(src, bak / split)
        print(f"标签已备份到: {bak}")

    manifest = root / f"_prune_manifest_{stamp}.csv"
    removed_imgs = 0
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "action", "boxes_removed"])
        for split, lp, _, out_lines, removed in plan:
            if out_lines:
                lp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
                w.writerow([split, lp.stem, "relabeled", removed])
            else:
                ip = next((p for p in (Path(str(lp.parent).replace("labels", "images"))).iterdir()
                           if p.stem == lp.stem and p.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
                if ip and ip.exists():
                    ip.unlink()
                lp.unlink()
                removed_imgs += 1
                w.writerow([split, lp.stem, "deleted_image_and_label", removed])

    # 更新 yaml：写新 names + 新文件
    new_cfg = dict(cfg)
    new_cfg["nc"] = len(keep_names)
    new_cfg["names"] = {i: n for i, n in enumerate(keep_names)}
    new_path = cfg_path.with_name(cfg_path.stem.replace("20", str(len(keep_names))) + ".yaml")
    header = (f"# {len(keep_names)} 类中餐菜品检测配置（由 tools/prune_classes.py 于 {stamp} 生成）\n"
              f"# 已剔除: {', '.join(args.drop)}\n")
    body = yaml.safe_dump({k: v for k, v in new_cfg.items() if k not in ("nc", "names")},
                          allow_unicode=True, sort_keys=False)
    names_txt = "nc: %d\nnames:\n" % len(keep_names) + "".join(
        f"  {i}: {n}\n" for i, n in enumerate(keep_names))
    new_path.write_text(header + body + names_txt, encoding="utf-8")
    print(f"新配置: {new_path}")

    stats_path = root / f"_prune_stats_{stamp}.json"
    stats_path.write_text(json.dumps({"dropped": args.drop, "kept": keep_names,
                                      "splits": total_stats, "deleted_images": removed_imgs},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"统计: {stats_path}")
    print(f"清单: {manifest}")
    print(f"完成：删掉 {removed_imgs} 张只剩被剔类的图片，其余标签已重编号。")
    print("注意：各 split 下的 labels.cache 已失效，训练前会自动重建（或手动删除）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
