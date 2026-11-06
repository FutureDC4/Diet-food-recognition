# -*- coding: utf-8 -*-
"""把源数据集（分割/polygon 或 bbox 标注）中的指定类别，抽取合并进目标检测数据集。

特性：
1. polygon 标注自动转 bbox（取点集外接矩形），原始 polygon 另存一份，方便日后训练分割模型。
2. 类别按「名称」映射到目标数据集的 ID：已有的沿用原 ID，新类追加在后面。
3. 去重：跳过源数据里与目标数据集同源（归一化文件名相同）的图片，避免重复。
4. 防泄漏：源 split 若与 train 中同源图重名，降级到 train；valid/test 每个源图只保留一份
   增强副本，避免 mAP 虚高。
5. 只增不删：不修改、不删除目标数据集里的任何已有文件；data.yaml 改动前先备份。

用法：
    # 先干跑看统计
    python tools/merge_dataset.py --src "D:/Work/膳食模型/数据2" --dst "D:/Work/膳食模型/数据" \
        --classes pumpkin "sweet potato" "fried string Beans" "Braised egg" "Braised meat" \
                  "F chicken legs" "Fried Noodles" "loofah" --dry-run

    # 确认无误后真正执行
    python tools/merge_dataset.py ...（去掉 --dry-run）
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
FILE_PREFIX = "s2_"  # 加进目标数据集的图片统一前缀，避免命名冲突


# ----------------------------------------------------------------------
def norm_name(p: str | Path) -> str:
    """归一化文件名：去掉 roboflow 增强哈希、扩展名与 _jpg/_png 后缀。"""
    b = Path(p).stem
    b = re.sub(r"\.rf\.[0-9a-f]+$", "", b)
    for suf in ("_jpg", "_png", "_jpeg"):
        b = re.sub(re.escape(suf) + "$", "", b, flags=re.I)
    return b.lower()


def slug(name: str) -> str:
    """类名规范化为小写蛇形：'Braised meat' -> 'braised_meat'。

    YOLO 标签按 ID 存储，改类名不影响 .txt，纯为可读性与跨平台安全。
    """
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def read_names(yaml_path: Path) -> List[str]:
    import yaml

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = cfg.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    raise ValueError(f"{yaml_path} 里找不到 names")


def index_images(split_dir: Path) -> Dict[str, Path]:
    """split 下 stem -> 图片路径。"""
    out: Dict[str, Path] = {}
    if not split_dir.is_dir():
        return out
    for p in split_dir.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            out[p.stem] = p
    return out


def parse_label(path: Path) -> List[Tuple[int, List[float]]]:
    """解析一行标注：(class_id, 坐标列表)。坐标可能是 4 个(bbox) 或 2n 个(polygon)。"""
    out: List[Tuple[int, List[float]]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(coords) == 4:
            out.append((cid, coords))
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            out.append((cid, coords))
    return out


def to_bbox(coords: Sequence[float]) -> Optional[List[float]]:
    """4 点坐标 -> 归一化 xywh；polygon -> 外接矩形 xywh。"""
    if len(coords) == 4:
        cx, cy, w, h = coords
    else:
        xs = coords[0::2]
        ys = coords[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    # 夹到 [0,1]，裁掉退化框
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    if w <= 1e-4 or h <= 1e-4:
        return None
    return [cx, cy, w, h]


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="抽取指定类别合并进目标数据集")
    ap.add_argument("--src", required=True, help="源数据集根目录（含 data.yaml）")
    ap.add_argument("--dst", required=True, help="目标数据集根目录（含 data.yaml）")
    ap.add_argument("--classes", nargs="+", required=True, help="要抽取的源类别名（按要追加的顺序）")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写任何文件")
    ap.add_argument("--keep-all-aug", action="store_true", help="valid/test 也保留全部增强副本（默认每源图 1 份）")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    for d in (src, dst):
        if not (d / "data.yaml").exists():
            sys.exit(f"缺少 data.yaml：{d}")

    src_names = read_names(src / "data.yaml")
    dst_names = read_names(dst / "data.yaml")
    print(f"源 {src.name}: {len(src_names)} 类")
    print(f"目标 {dst.name}: {len(dst_names)} 类 → {dst_names}")

    # ---- 构建 源ID -> 目标ID 映射 ----
    dst_lower = {slug(n): i for i, n in enumerate(dst_names)}
    src2dst_new: Dict[int, int] = {}
    new_names: List[str] = []
    missing: List[str] = []
    for name in args.classes:
        if name not in src_names:
            missing.append(name)
            continue
        sid = src_names.index(name)
        clean = slug(name)
        if clean in dst_lower:
            print(f"  [跳过] {name} 已存在于目标数据集（ID {dst_lower[clean]}）")
            continue
        new_id = len(dst_names) + len(new_names)
        src2dst_new[sid] = new_id
        new_names.append(clean)
        print(f"  [追加] {name:28s} 源ID {sid:>2} → 目标ID {new_id}  ({clean})")

    if missing:
        print(f"\n[警告] 源数据里不存在这些类，已忽略：{missing}")
    if not src2dst_new:
        sys.exit("没有可追加的类别，退出。")

    final_names = dst_names + new_names
    target_ids = set(src2dst_new.keys())

    # ---- 目标数据集已有图片（归一化名），用于去重 ----
    dst_norm: Set[str] = set()
    for sp in SPLITS:
        for p in (dst / sp / "images").glob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                dst_norm.add(norm_name(p))
    print(f"\n目标数据集已有唯一源图 {len(dst_norm)} 张")

    # ---- 源数据先扫一遍，规划 split 与去重 ----
    plan: List[dict] = []          # 待复制的条目
    skipped_dup = 0                # 与目标重复
    skipped_empty = 0              # 过滤后无标注
    dropped_cls: Counter = Counter()
    norm_in_train: Set[str] = set()

    # 第一遍：收集 train 的归一化名（用于防泄漏降级）
    for lp in (src / "train" / "labels").glob("*.txt"):
        norm_in_train.add(norm_name(lp))

    used_in_eval: Set[str] = set()  # valid/test 已用过的源图名
    split_counter: Dict[str, Counter] = {sp: Counter() for sp in SPLITS}

    for sp in SPLITS:
        lbl_dir = src / sp / "labels"
        img_dir = src / sp / "images"
        if not lbl_dir.is_dir():
            continue
        img_map = index_images(img_dir)
        for lp in sorted(lbl_dir.glob("*.txt")):
            stem = lp.stem
            if stem not in img_map:
                continue
            nn = norm_name(lp)
            if nn in dst_norm:
                skipped_dup += 1
                continue

            items = parse_label(lp)
            if not any(cid in target_ids for cid, _ in items):
                continue  # 不含目标类别，跳过

            # 类别映射 + polygon 转 bbox
            kept_bbox: List[Tuple[int, List[float]]] = []
            kept_poly: List[Tuple[int, List[float]]] = []
            for cid, coords in items:
                if cid in src2dst_new:
                    did = src2dst_new[cid]
                elif slug(src_names[cid]) in dst_lower:
                    did = dst_lower[slug(src_names[cid])]
                else:
                    dropped_cls[src_names[cid]] += 1
                    continue
                bb = to_bbox(coords)
                if bb is None:
                    continue
                kept_bbox.append((did, bb))
                if len(coords) > 4:
                    kept_poly.append((did, coords))

            if not kept_bbox:
                skipped_empty += 1
                continue

            # 决定目标 split：防泄漏 + 验证集去增强副本
            dst_split = sp
            if sp in ("valid", "test") and nn in norm_in_train:
                dst_split = "train"
            if dst_split in ("valid", "test") and not args.keep_all_aug:
                if nn in used_in_eval:
                    continue
                used_in_eval.add(nn)

            plan.append(
                {
                    "src_img": img_map[stem],
                    "src_label": lp,
                    "split": dst_split,
                    "bbox": kept_bbox,
                    "poly": kept_poly,
                    "norm": nn,
                }
            )
            split_counter[dst_split]["images"] += 1
            split_counter[dst_split]["boxes"] += len(kept_bbox)

    # ---- 汇总 ----
    print("\n" + "=" * 74)
    print(f"待复制图片：{len(plan)}")
    for sp in SPLITS:
        c = split_counter[sp]
        print(f"  {sp:6s} 图片 {c['images']:>5}  标注框 {c['boxes']:>5}")
    print(f"跳过（与目标数据集同源，防重复）：{skipped_dup}")
    print(f"跳过（过滤后无可用标注）：{skipped_empty}")
    if dropped_cls:
        print("丢弃的非目标类别标注（Top 10，会成为负样本空缺，属正常损耗）：")
        for n, c in dropped_cls.most_common(10):
            print(f"    {n:40s} {c}")
    kept_ids = Counter()
    for e in plan:
        for did, _ in e["bbox"]:
            kept_ids[did] += 1
    print("\n合并后各类别标注数（仅本次新增）：")
    for did in range(len(final_names)):
        if kept_ids.get(did):
            print(f"  ID {did:>2} {final_names[did]:26s} +{kept_ids[did]}")

    if args.dry_run:
        print("\n[干跑] 未写入任何文件。去掉 --dry-run 即执行。")
        return 0

    # ---- 执行写入 ----
    print("\n开始写入 …")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) 备份 data.yaml
    bak = dst / f"data.yaml.bak_{stamp}"
    shutil.copy2(dst / "data.yaml", bak)
    print(f"  已备份原 data.yaml → {bak.name}")

    # 2) 复制图片与标签
    poly_root = dst / "_polygon_labels"
    manifest: List[dict] = []
    for e in plan:
        sp = e["split"]
        (dst / sp / "images").mkdir(parents=True, exist_ok=True)
        (dst / sp / "labels").mkdir(parents=True, exist_ok=True)

        new_stem = FILE_PREFIX + e["src_img"].stem
        new_img = dst / sp / "images" / (new_stem + e["src_img"].suffix.lower())
        new_lbl = dst / sp / "labels" / (new_stem + ".txt")
        shutil.copy2(e["src_img"], new_img)
        label_lines = [
            f"{cid} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}" for cid, bb in e["bbox"]
        ]
        new_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        # 原始 polygon 另存
        if e["poly"]:
            pd = poly_root / sp
            pd.mkdir(parents=True, exist_ok=True)
            (pd / (new_stem + ".txt")).write_text(
                "\n".join(f"{cid} " + " ".join(f"{v:.6f}" for v in coords) for cid, coords in e["poly"]) + "\n",
                encoding="utf-8",
            )
        manifest.append(
            {
                "split": sp,
                "image": str(new_img.relative_to(dst)),
                "src_image": str(e["src_img"]),
                "boxes": len(e["bbox"]),
                "classes": ";".join(final_names[cid] for cid, _ in e["bbox"]),
                "has_polygon": "yes" if e["poly"] else "",
            }
        )

    # 3) 写清单
    man_path = dst / f"_merge_manifest_{stamp}.csv"
    with open(man_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"  合并清单 → {man_path.name}（{len(manifest)} 行）")

    # 4) 写 20 类配置（不覆盖原 data.yaml）
    yaml_path = dst / f"data_{len(final_names)}class.yaml"
    lines = [
        f"# {len(final_names)} 类中餐食堂菜品检测数据集",
        "# 由 tools/merge_dataset.py 生成",
        f"# 原 12 类来自 Chinese-Cafeteria-Food-Recognition v3",
        f"# 追加 {len(new_names)} 类来自 s-workspace/chinese-food v20（polygon 已转 bbox）",
        f"path: {dst.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(final_names)}",
        "names:",
    ]
    for i, n in enumerate(final_names):
        lines.append(f"  {i}: {n}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  数据集配置 → {yaml_path.name}（nc={len(final_names)}）")

    # 5) 统计 JSON
    stat = {
        "generated_at": stamp,
        "src": str(src),
        "dst": str(dst),
        "classes_added": new_names,
        "final_classes": final_names,
        "images_added": {sp: split_counter[sp]["images"] for sp in SPLITS},
        "boxes_added": {sp: split_counter[sp]["boxes"] for sp in SPLITS},
        "skipped_duplicate": skipped_dup,
        "skipped_empty": skipped_empty,
        "dropped_other_classes": dict(dropped_cls),
        "per_class_added": {final_names[k]: v for k, v in kept_ids.items()},
    }
    (dst / f"_merge_stats_{stamp}.json").write_text(
        json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  统计 JSON  → _merge_stats_{stamp}.json")
    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
