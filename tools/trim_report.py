# -*- coding: utf-8 -*-
"""按行号裁剪 test_plan_c.py 生成的报告（只改展示，不重跑模型）。

用途：写报告/答辩时想拿掉某几张图，直接按表格行号删，图片文件保留。

用法：
    python tools/trim_report.py --dir runs/plan_c_test --drop 3,5,6,10,11,12
    python tools/trim_report.py --dir runs/plan_c_test --keep 1,2,4,7
    python tools/trim_report.py --dir runs/plan_c_test --drop 3 --dry   # 只看会删哪几张

顶部汇总会按剩下的图片重算（否则百分比和列出来的图对不上）。
原报告自动备份为 report.html.bak。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from test_plan_c import html_report  # noqa: E402


def recompute(rows: list[dict], tags: list[str]) -> dict:
    """按保留下来的行重算召回/精确/菜品级命中。"""
    summary: dict[str, dict] = {}
    for tag in tags:
        gt = hit = pred = gt_cls = hit_cls = 0
        for row in rows:
            rec = row["runs"].get(tag)
            if not rec:
                continue
            gt += rec.get("gt_count", 0)
            hit += len(rec.get("hit") or [])
            pred += len(rec.get("preds") or [])
            gt_cls += rec.get("gt_cls", 0)
            hit_cls += rec.get("hit_cls", 0)
        recall = hit / gt if gt else 0.0
        prec = hit / pred if pred else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
        summary[tag] = {
            "recall": round(recall, 4), "precision": round(prec, 4), "f1": round(f1, 4),
            "class_recall": round(hit_cls / gt_cls, 4) if gt_cls else 0.0,
            "gt": gt, "pred": pred, "hit": hit, "gt_cls": gt_cls, "hit_cls": hit_cls,
        }
    return summary


def parse_ids(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/plan_c_test")
    ap.add_argument("--drop", default="", help="要删掉的行号，如 3,5,6 或 3-6")
    ap.add_argument("--keep", default="", help="只保留这些行号（与 --drop 二选一）")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--dry", action="store_true", help="只打印结果，不写文件")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / args.dir
    data_file = out_dir / "result.json"
    if not data_file.exists():
        sys.exit(f"找不到 {data_file}，请先跑 tools/test_plan_c.py")

    payload = json.loads(data_file.read_text(encoding="utf-8"))
    rows: list[dict] = payload["images"]
    total = len(rows)

    if not args.drop and not args.keep:
        sys.exit("请给 --drop 或 --keep")

    if args.keep:
        keep = parse_ids(args.keep)
        bad = [i for i in keep if i < 1 or i > total]
        if bad:
            sys.exit(f"行号越界：{bad}（共 {total} 行）")
        kept = [r for i, r in enumerate(rows, 1) if i in keep]
        dropped = [i for i in range(1, total + 1) if i not in keep]
    else:
        drop = parse_ids(args.drop)
        bad = [i for i in drop if i < 1 or i > total]
        if bad:
            sys.exit(f"行号越界：{bad}（共 {total} 行）")
        kept = [r for i, r in enumerate(rows, 1) if i not in drop]
        dropped = sorted(drop)

    if not kept:
        sys.exit("不能把所有行都删掉")

    tags = list(payload.get("summary", {}).keys())
    if not tags:
        tags = [t for t in (rows[0]["runs"].keys() if rows else [])]

    summary = recompute(kept, tags)

    print(f"共 {total} 行 → 保留 {len(kept)} 行，删除 {len(dropped)} 行")
    for i in dropped:
        print(f"  - 第 {i:2d} 行  {rows[i - 1]['name'][:56]}")
    print("\n重算后的汇总：")
    for tag in tags:
        s = summary[tag]
        print(f"  {tag:6s} 框级召回 {s['recall']:.1%}  精确 {s['precision']:.1%}  "
              f"F1 {s['f1']:.1%} | 菜品级命中 {s['class_recall']:.1%}（{s['hit_cls']}/{s['gt_cls']}）")

    if args.dry:
        print("\n(--dry：未写文件)")
        return 0

    target = out_dir / args.out
    if target.exists():
        shutil.copy2(target, target.with_suffix(target.suffix + ".bak"))
        print(f"\n原报告已备份：{target.name}.bak")

    report = html_report(kept, out_dir, tags, summary)
    print(f"已写入：{report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
