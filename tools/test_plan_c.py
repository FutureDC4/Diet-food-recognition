# -*- coding: utf-8 -*-
"""方案C（Neck 端 CBAM + WIoU）批量实测：多图检测 + 端到端营养链路。

与 eval_three.py 的区别：
  eval_three 只看 mAP（模型层面）；这里看「上位机实际跑出来什么」——
  检测框与标注对齐情况、漏检/误检、米饭规则触发、热量与结构评分。

用法：
    # 12 张 test 图，方案C，同时跑基线对照
    python tools/test_plan_c.py --count 12

    # 只看方案C，指定图片
    python tools/test_plan_c.py --images a.jpg b.jpg

输出：
    runs/plan_c_test/<tag>/<图名>.png   带框结果
    runs/plan_c_test/report.html        图文对照表
    runs/plan_c_test/result.json        原始数据
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

PLAN_C = "runs/chinese/cn11s_cbamneck_wiou_18/weights/best.pt"
BASE = "runs/chinese/cn11s_base_18/weights/best.pt"
USER = {"name": "测试", "age": 22, "gender": "男",
        "height_cm": 175, "weight_kg": 66, "activity_level": 2}


# ---------------------------------------------------------------- 图片挑选
def gt_boxes(lbl: Path, names: Dict[int, str], w: int, h: int) -> List[dict]:
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cid = int(float(p[0]))
        cx, cy, bw, bh = [float(v) for v in p[1:5]]
        out.append({
            "name": names.get(cid, str(cid)),
            "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h],
        })
    return out


def pick_images(img_dir: Path, lbl_dir: Path, count: int, seed: int = 7) -> List[Path]:
    """按标注框数量分层抽样：多菜餐盘优先，同时保留单菜/双菜，避免只看好样本。"""
    files = []
    for ip in sorted(img_dir.glob("*.jpg")):
        lp = lbl_dir / (ip.stem + ".txt")
        if not lp.exists():
            continue
        n = len([l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()])
        if n > 0:
            files.append((n, ip))
    if not files:
        return []

    buckets = {"many": [], "mid": [], "few": []}
    for n, ip in files:
        if n >= 5:
            buckets["many"].append((n, ip))
        elif n >= 3:
            buckets["mid"].append((n, ip))
        else:
            buckets["few"].append((n, ip))

    rng = random.Random(seed)
    quota = [("many", 0.5), ("mid", 0.3), ("few", 0.2)]
    picked: List[Path] = []
    for key, share in quota:
        pool = sorted(buckets[key], key=lambda t: t[1].name)
        k = min(int(round(count * share)), len(pool))
        rng.shuffle(pool)
        picked += [ip for _, ip in pool[:k]]
    if len(picked) < count:                     # 配额不足时从剩余里补
        rest = [ip for n, ip in files if ip not in picked]
        rng.shuffle(rest)
        picked += rest[: count - len(picked)]
    return picked[:count]


# ---------------------------------------------------------------- 匹配
def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds: List[dict], gts: List[dict], thr: float = 0.5) -> Dict[str, Any]:
    """类别感知的贪心匹配（按置信度从高到低）。"""
    used = [False] * len(gts)
    hit, wrong_class = [], []
    for p in sorted(preds, key=lambda d: -float(d.get("conf", 0))):
        best, best_iou = -1, 0.0
        for i, g in enumerate(gts):
            if used[i]:
                continue
            v = iou(p["bbox"], g["bbox"])
            if v > best_iou:
                best, best_iou = i, v
        if best >= 0 and best_iou >= thr:
            used[best] = True
            if str(p["name"]) == str(gts[best]["name"]):
                hit.append(p["name"])
            else:
                wrong_class.append({"pred": p["name"], "gt": gts[best]["name"]})
    missed = [g["name"] for i, g in enumerate(gts) if not used[i]]
    return {"hit": hit, "missed": missed, "wrong": wrong_class}


# ---------------------------------------------------------------- 跑模型
def make_analyzer(weights: Path):
    from app.analyzer import MealAnalyzer
    from src.detection import FoodDetector

    a = MealAnalyzer()
    a._food_det = FoodDetector(model_path=str(weights), filter_unknown=True)
    return a


def run_one(analyzer, path: Path) -> Dict[str, Any]:
    r = analyzer.analyze(path, USER, "午餐", True, "常规")
    return r


# ---------------------------------------------------------------- 报告
def html_report(rows: List[dict], out_dir: Path, tags: List[str],
                summary: Optional[dict] = None) -> Path:
    def cell(rec: Optional[dict]) -> str:
        if not rec:
            return "<td class='empty'>—</td>"
        n_gt, n_pr = rec["gt_count"], len(rec["preds"])
        rec_ok = "".join(f"<span class='chip'>{x}</span>" for x in rec["hit"][:6])
        bad = rec["missed"] + [f"{w['pred']}→{w['gt']}" for w in rec["wrong"]]
        bad_html = "".join(f"<span class='chip bad'>{x}</span>" for x in bad[:8])
        return (
            "<td>"
            f"<div class='num'>{rec['kcal']:.0f} kcal · 达标 {rec['good']}/{rec['total']}</div>"
            f"<div class='meta'>预测 {n_pr} / 真值 {n_gt} · 命中 {len(rec['hit'])}"
            f" · 菜品 {rec['hit_cls']}/{rec['gt_cls']}</div>"
            f"<div class='chips'>{rec_ok}{bad_html}</div>"
            f"<div class='note'>{rec['rice']}</div>"
            "</td>"
        )

    cards = ""
    if summary:
        parts = []
        for tag in tags:
            s = summary.get(tag) or {}
            if not s:
                continue
            parts.append(
                "<div class='card'>"
                f"<div class='cname'>{tag}</div>"
                f"<div class='big'>{s.get('class_recall', 0):.1%}</div>"
                f"<div class='cl'>菜品级命中（{s.get('hit_cls', 0)}/{s.get('gt_cls', 0)}）</div>"
                f"<div class='cl'>框级召回 {s.get('recall', 0):.1%} · 精确 {s.get('precision', 0):.1%} · F1 {s.get('f1', 0):.1%}</div>"
                "</div>"
            )
        cards = "<div class='cards'>" + "".join(parts) + "</div>"

    head = "".join(f"<th>{t}</th>" for t in tags)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td class='imgcell'><img src='{row['thumb']}'>"
            f"<div class='fname'>{row['name'][:34]}</div></td>"
            + "".join(cell(row["runs"].get(t)) for t in tags)
            + "</tr>"
        )

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>方案C 批量实测</title>
<style>
 body{{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#f5f6f8;color:#1f2430;margin:0;padding:28px}}
 h1{{font-size:21px;margin:0 0 6px}}
 .sub{{color:#6b7280;font-size:13px;margin-bottom:18px}}
 table{{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden;
        box-shadow:0 1px 4px rgba(0,0,0,.08)}}
 th{{background:#eef1f6;font-size:13px;text-align:left;padding:10px 12px;border-bottom:1px solid #e2e6ee}}
 td{{padding:10px 12px;border-bottom:1px solid #eef0f4;vertical-align:top;font-size:13px}}
 .imgcell{{width:300px}} .imgcell img{{width:280px;border-radius:8px;display:block}}
 .fname{{color:#9aa1ad;font-size:11px;margin-top:4px}}
 .num{{font-weight:600}} .meta{{color:#6b7280;font-size:12px;margin:3px 0}}
 .note{{color:#8a909c;font-size:11px;margin-top:4px;line-height:1.5}}
 .chip{{display:inline-block;background:#e8f2ea;color:#2f6b3d;border-radius:4px;
        padding:1px 5px;margin:2px 3px 0 0;font-size:11px}}
 .chip.bad{{background:#fbeaea;color:#a13a34}}
 .empty{{color:#c3c8d1;text-align:center}}
 .legend{{font-size:12px;color:#6b7280;margin-top:10px}}
 .cards{{display:flex;gap:14px;margin:16px 0 6px}}
 .card{{background:#fff;border-radius:10px;padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,.08);min-width:230px}}
 .cname{{font-size:13px;color:#6b7280}} .big{{font-size:26px;font-weight:700;color:#1a4fbb;margin:2px 0 4px}}
 .cl{{font-size:12px;color:#6b7280}}
</style></head><body>
<h1>方案C（Neck-CBAM + WIoU）批量实测</h1>
{cards}
<div class="sub">同一批图、同一套端到端链路（整图检测 → 米饭三级 → 估重 → 结构评估）；
绿色标签为命中，红色为漏检或错类。</div>
<table><thead><tr><th>原图</th>{head}</tr></thead>
<tbody>{''.join(body)}</tbody></table>
<div class="legend">注：命中判定为 IoU ≥ 0.5 且类别一致；「x 项达标」为八项结构指标中达标项数。</div>
</body></html>"""
    p = out_dir / "report.html"
    p.write_text(html, encoding="utf-8")
    return p


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/chinese_food_18.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--plan-c", default=PLAN_C)
    ap.add_argument("--baseline", default=BASE)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--out", default="runs/plan_c_test")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cfg["names"].items()}
    root = Path(cfg["path"])
    key = {"train": "train", "val": "valid", "test": "test"}[args.split]
    img_dir, lbl_dir = root / key / "images", root / key / "labels"

    if args.images:
        imgs = [Path(p) for p in args.images]
    else:
        imgs = pick_images(img_dir, lbl_dir, args.count, args.seed)
    if not imgs:
        sys.exit("没有可用图片")

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [("方案C", PROJECT_ROOT / args.plan_c)]
    if not args.no_baseline:
        models.append(("基线", PROJECT_ROOT / args.baseline))

    print(f"图片 {len(imgs)} 张 | 模型 {', '.join(m[0] for m in models)}\n")

    analyzers = {tag: make_analyzer(w) for tag, w in models}
    for a in analyzers.values():
        a.warmup()

    rows, all_stats = [], {tag: {"gt": 0, "hit": 0, "pred": 0, "missed": 0,
                                 "gt_cls": 0, "hit_cls": 0} for tag, _ in models}

    from PIL import Image

    for i, ip in enumerate(imgs, 1):
        from src.utils.image_utils import load_image
        img = load_image(str(ip))
        h, w = img.shape[:2]
        gts = gt_boxes(lbl_dir / (ip.stem + ".txt"), names, w, h)

        row = {"name": ip.name, "gt_count": len(gts),
               "gt_names": sorted({g["name"] for g in gts}), "runs": {}}
        print(f"[{i}/{len(imgs)}] {ip.name}  真值 {len(gts)} 个框："
              + ", ".join(row["gt_names"]))

        for tag, _ in models:
            r = run_one(analyzers[tag], ip)
            preds = [{"name": d["name"], "conf": d["conf"], "bbox": d["bbox"]}
                     for d in r["merged"]]
            m = match(preds, gts)
            tag_dir = out_dir / tag.replace("方案", "plan").replace("基线", "base")
            tag_dir.mkdir(parents=True, exist_ok=True)
            save = tag_dir / f"{ip.stem}.png"
            Image.fromarray(r["annotated"][:, :, ::-1]).save(save)

            st = all_stats[tag]
            st["gt"] += len(gts)
            st["hit"] += len(m["hit"])
            st["pred"] += len(preds)
            st["missed"] += len(m["missed"])

            # 菜品级命中：同一类多份被合并成一个框，逐实例 IoU 会低估；
            # 对一餐的建议来说「这盘菜认出来没有」才是关键指标。
            gt_cls = {g["name"] for g in gts}
            dp_cls = {p["name"] for p in preds}
            st["gt_cls"] += len(gt_cls)
            st["hit_cls"] += len(gt_cls & dp_cls)

            row["runs"][tag] = {
                "gt_count": len(gts),
                "preds": preds,
                "hit": m["hit"],
                "missed": m["missed"],
                "wrong": m["wrong"],
                "hit_cls": len(gt_cls & dp_cls),
                "gt_cls": len(gt_cls),
                "kcal": r["totals"]["calories"],
                "weight_g": r["totals"]["weight_g"],
                "good": r["budget"].get("good", 0),
                "total": r["budget"].get("total", 8),
                "rice": (r["rice_rule"].get("note") or "")[:40],
                "image": str(save.relative_to(out_dir)).replace("\\", "/"),
            }
            print(f"    {tag}: 预测 {len(preds):2d} 命中 {len(m['hit']):2d} "
                  f"漏检 {len(m['missed']):2d} 错类 {len(m['wrong'])} | "
                  f"{r['totals']['calories']:.0f} kcal | {r['elapsed_s']}s")
            if m["missed"]:
                print(f"      漏: {', '.join(m['missed'])}")

        # 缩略图用方案C的标注图
        first = models[0][0]
        row["thumb"] = row["runs"][first]["image"]
        rows.append(row)

    # ---- 汇总
    print("\n" + "=" * 62)
    summary = {}
    for tag, st in all_stats.items():
        recall = st["hit"] / st["gt"] if st["gt"] else 0
        prec = st["hit"] / st["pred"] if st["pred"] else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
        cls_recall = st["hit_cls"] / st["gt_cls"] if st["gt_cls"] else 0
        summary[tag] = {"recall": round(recall, 4), "precision": round(prec, 4),
                        "f1": round(f1, 4), "class_recall": round(cls_recall, 4),
                        "gt": st["gt"], "pred": st["pred"], "hit": st["hit"],
                        "gt_cls": st["gt_cls"], "hit_cls": st["hit_cls"]}
        print(f"{tag:6s} 框级召回 {recall:6.1%}  精确 {prec:6.1%}  F1 {f1:6.1%} | "
              f"菜品级命中 {cls_recall:6.1%}（{st['hit_cls']}/{st['gt_cls']}）")
        print(f"        真值框 {st['gt']} / 预测框 {st['pred']} / 命中 {st['hit']}")

    payload = {"images": rows, "summary": summary,
               "models": {t: str(w) for t, w in models}}
    (out_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = html_report(rows, out_dir, [t for t, _ in models], summary)
    print(f"\n报告：{report}")
    print(f"数据：{out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
