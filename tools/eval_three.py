# -*- coding: utf-8 -*-
"""三方对照评估：基线 / 旧CBAM(替换C2PSA) / 新CBAM(Neck端)。

每个模型单独进程跑，结果追加到 logs/eval_<tag>.json，避免一次超时全丢。
用法:
    python tools/eval_three.py --tag base
    python tools/eval_three.py --report      # 汇总三个 json 出表
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RUNS = {
    "base": "runs/chinese/cn11s_base_18/weights/best.pt",
    "cbam_old": "runs/chinese/cn11s_cbam_wiou_18/weights/best.pt",
    "cbam_neck": "runs/chinese/cn11s_cbamneck_wiou_18/weights/best.pt",
}
LABEL = {
    "base": "基线 yolo11s+CIoU",
    "cbam_old": "旧CBAM(替换C2PSA)+WIoU",
    "cbam_neck": "新CBAM(Neck端)+WIoU",
}


def eval_one(tag: str, split: str) -> dict:
    import models  # noqa: F401
    import yaml
    from ultralytics import YOLO

    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs/chinese_food_18.yaml", encoding="utf-8"))
    names = [cfg["names"][i] for i in range(cfg["nc"])]
    m = YOLO(str(PROJECT_ROOT / RUNS[tag]))
    r = m.val(data=str(PROJECT_ROOT / "configs/chinese_food_18.yaml"), split=split,
              imgsz=640, batch=16, device=0, conf=0.001, iou=0.6,
              verbose=False, plots=False)
    ap_raw = r.box.ap
    ap = ap_raw if getattr(ap_raw, "ndim", 1) == 1 else ap_raw.mean(1)
    return {
        "tag": tag,
        "split": split,
        "names": names,
        "map50": float(r.box.map50),
        "map": float(r.box.map),
        "mp": float(r.box.mp),
        "mr": float(r.box.mr),
        "ap": [float(x) for x in ap],
        "ap50": [float(x) for x in r.box.ap50],
    }


def report() -> None:
    out_dir = PROJECT_ROOT / "logs"
    recs = {}
    for tag in RUNS:
        for sp in ("val", "test"):
            p = out_dir / f"eval_{tag}_{sp}.json"
            if p.exists():
                recs[(tag, sp)] = json.loads(p.read_text(encoding="utf-8"))

    print("=== 指标对照（18 类，test 94 张 / val 131 张）===")
    print(f"{'实验':<26}{'val mAP50':>10}{'val 50-95':>10}{'test mAP50':>11}{'test 50-95':>11}{'test P':>8}{'test R':>8}")
    for tag in RUNS:
        v = recs.get((tag, "val"))
        t = recs.get((tag, "test"))
        if not v or not t:
            print(f"{LABEL[tag]:<26}  (缺数据)")
            continue
        print(f"{LABEL[tag]:<26}{v['map50']:>10.4f}{v['map']:>10.4f}"
              f"{t['map50']:>11.4f}{t['map']:>11.4f}{t['mp']:>8.4f}{t['mr']:>8.4f}")

    base, old, neck = (recs.get(("base", "test")), recs.get(("cbam_old", "test")), recs.get(("cbam_neck", "test")))
    if not all([base, old, neck]):
        return
    names = neck["names"]
    print("\n=== 逐类 test AP50-95 ===")
    print(f"{'类别':<22}{'base':>9}{'cbam_old':>10}{'cbam_neck':>11}{'neck-base':>11}{'neck-old':>10}")
    d_base, d_old = [], []
    for i, n in enumerate(names):
        b, o, c = base["ap"][i], old["ap"][i], neck["ap"][i]
        d_base.append(c - b)
        d_old.append(c - o)
        print(f"{n:<22}{b:>9.4f}{o:>10.4f}{c:>11.4f}{c-b:>+11.4f}{c-o:>+10.4f}")

    print("\n=== 配对统计 ===")
    for tag, d in (("新 - 基线", d_base), ("新 - 旧CBAM", d_old)):
        n = len(d)
        md = st.mean(d)
        sd = st.stdev(d)
        t = md / (sd / math.sqrt(n))
        win = sum(1 for x in d if x > 0)
        print(f"{tag:<12} 平均 {md:+.4f}  胜 {win}/{n}  配对t={t:.2f} (df={n-1})  Cohen's d={md/sd:+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=list(RUNS))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        report()
        return 0
    if not args.tag:
        ap.error("需要 --tag 或 --report")

    import os
    os.chdir(PROJECT_ROOT)
    rec = eval_one(args.tag, args.split)
    p = PROJECT_ROOT / "logs" / f"eval_{args.tag}_{args.split}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{args.tag}/{args.split}] mAP50={rec['map50']:.4f} mAP50-95={rec['map']:.4f} "
          f"P={rec['mp']:.4f} R={rec['mr']:.4f}  -> {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
