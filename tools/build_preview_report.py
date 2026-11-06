# -*- coding: utf-8 -*-
"""生成检测结果汇总 HTML（图片 + 逐图真实/预测对照）。

用法：
    python tools/build_preview_report.py --weights runs/chinese/xxx/weights/best.pt
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import yaml  # noqa: E402

from src.utils.food_labels import FOOD_LABELS_ZH  # noqa: E402
from tools.preview_predict import pick_images  # noqa: E402


def zh(name: str) -> str:
    return FOOD_LABELS_ZH.get(name, name)


def read_gt(img_path: Path, names: dict) -> list[str]:
    lbl = Path(str(img_path.parent).replace("images", "labels")) / (img_path.stem + ".txt")
    out = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5:
                out.append(names[int(float(p[0]))])
    return out


def b64(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/chinese_food_18.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--out", default="runs/preview/report.html")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cfg["names"].items()}

    from ultralytics import YOLO

    model = YOLO(args.weights)
    jobs = [("test", p) for p in pick_images(cfg, "test", args.n_test)]
    jobs += [("train", p) for p in pick_images(cfg, "train", args.n_train)]

    cards, tot_gt, tot_pred, hit_cls, tot_cls = [], 0, 0, 0, 0
    for split, ip in jobs:
        r = model.predict(str(ip), imgsz=640, conf=args.conf, iou=0.5,
                          device=0, verbose=False)[0]
        preds = [(names[int(b.cls.item())], float(b.conf.item())) for b in r.boxes]
        gt = read_gt(ip, names)
        tot_gt += len(gt)
        tot_pred += len(preds)

        gt_set = set(gt)
        pr_set = {n for n, _ in preds}
        missed = gt_set - pr_set
        extra = pr_set - gt_set
        tot_cls += len(gt_set)
        hit_cls += len(gt_set & pr_set)

        tags_gt = "".join(
            f'<span class="tag miss">{zh(c)}</span>' if c in missed
            else f'<span class="tag ok">{zh(c)}</span>'
            for c in sorted(gt_set)
        )
        counts: dict[str, int] = {}
        for n, _ in sorted(preds, key=lambda t: -t[1]):
            counts[n] = counts.get(n, 0) + 1
        tags_pr = "".join(
            f'<span class="tag {"extra" if n in extra else "ok"}">{zh(n)}'
            f'<i>{"×" + str(c) if c > 1 else ""} {max(cf for m, cf in preds if m == n):.2f}</i></span>'
            for n, c in counts.items()
        ) or '<span class="tag none">无检出</span>'

        annotated = r.plot(labels=False, conf=False, line_width=2)
        cards.append(f"""
    <div class="card">
      <div class="imgbox"><img src="{b64(annotated)}" alt=""></div>
      <div class="meta">
        <div class="row"><span class="badge {split}">{split}</span>
          <span class="fname">{ip.name[:46]}</span></div>
        <div class="line"><b>真实</b> {len(gt)} 框<br>{tags_gt}</div>
        <div class="line"><b>预测</b> {len(preds)} 框<br>{tags_pr}</div>
      </div>
    </div>""")

    rec = hit_cls / max(tot_cls, 1) * 100
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>中餐检测结果汇总</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:32px 28px 56px; background:#f4f5f7; color:#1a1d23;
         font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif; }}
  h1 {{ font-size:22px; margin:0 0 6px; font-weight:600; letter-spacing:.3px; }}
  .sub {{ color:#6b7280; font-size:13px; margin-bottom:22px; }}
  .stats {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:26px; }}
  .stat {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px;
           padding:12px 20px; min-width:128px; }}
  .stat b {{ display:block; font-size:24px; font-weight:650; color:#111827; line-height:1.3; }}
  .stat span {{ font-size:12px; color:#6b7280; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:18px; }}
  .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:12px;
           overflow:hidden; display:flex; flex-direction:column; }}
  .imgbox {{ background:#0f1115; }}
  .imgbox img {{ width:100%; display:block; }}
  .meta {{ padding:12px 14px 14px; font-size:12.5px; }}
  .row {{ display:flex; align-items:center; gap:8px; margin-bottom:8px; }}
  .badge {{ font-size:10px; padding:2px 7px; border-radius:4px; font-weight:600;
            text-transform:uppercase; letter-spacing:.4px; }}
  .badge.test {{ background:#dcfce7; color:#15803d; }}
  .badge.train {{ background:#dbeafe; color:#1d4ed8; }}
  .fname {{ font-family:ui-monospace,Consolas,monospace; font-size:10.5px; color:#9ca3af;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .line {{ margin:9px 0; color:#4b5563; line-height:2.0; }}
  .line b {{ color:#111827; font-weight:600; }}
  .tag {{ display:inline-block; padding:1px 7px; border-radius:5px; margin:2px 3px 2px 0;
          font-size:11.5px; background:#f3f4f6; color:#374151; }}
  .tag.ok {{ background:#eef7ee; color:#2f6b3a; }}
  .tag.miss {{ background:#fdecec; color:#b42318; }}
  .tag.extra {{ background:#fff6dd; color:#946200; }}
  .tag.none {{ background:#f3f4f6; color:#9ca3af; }}
  .tag i {{ font-style:normal; opacity:.62; font-size:10.5px; margin-left:3px; }}
  .legend {{ font-size:12px; color:#6b7280; margin-bottom:18px; }}
  .legend em {{ font-style:normal; padding:1px 7px; border-radius:5px; margin:0 4px 0 10px; }}
</style></head>
<body>
  <h1>中餐菜品检测 · 结果汇总</h1>
  <div class="sub">{Path(args.weights).parent.parent.name} &nbsp;·&nbsp; 18 类 &nbsp;·&nbsp;
    imgsz 640 &nbsp;·&nbsp; conf {args.conf} &nbsp;·&nbsp; iou 0.50</div>

  <div class="stats">
    <div class="stat"><b>{len(jobs)}</b><span>抽样图片</span></div>
    <div class="stat"><b>{tot_gt}</b><span>真实标注框</span></div>
    <div class="stat"><b>{tot_pred}</b><span>模型检出框</span></div>
    <div class="stat"><b>{rec:.0f}%</b><span>类别命中率</span></div>
  </div>

  <div class="legend">图例：
    <em style="background:#eef7ee;color:#2f6b3a">检出正确</em>
    <em style="background:#fdecec;color:#b42318">漏检</em>
    <em style="background:#fff6dd;color:#946200">多余/误检</em>
  </div>

  <div class="grid">{''.join(cards)}
  </div>
</body></html>"""

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"已生成: {out}")
    print(f"图片 {len(jobs)} 张 | 真实框 {tot_gt} | 检出框 {tot_pred} | 类别命中率 {rec:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
