# -*- coding: utf-8 -*-
"""用指定权重在挑出来的图片上跑推理，保存带框结果并打印逐图对照。

用法：
    python tools/preview_predict.py --weights runs/chinese/xxx/weights/best.pt --split test --top 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402


def pick_images(cfg: dict, split: str, top: int, seed: int = 0) -> list[Path]:
    """按标签框数量从多到少挑选，优先多菜餐盘图。"""
    root = Path(cfg["path"])
    key = {"train": "train", "val": "val", "test": "test"}[split]
    img_dir = root / str(cfg[key])
    lbl_dir = Path(str(img_dir).replace("images", "labels"))

    scored = []
    for lp in lbl_dir.glob("*.txt"):
        n = len([l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()])
        if n == 0:
            continue
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            ip = img_dir / (lp.stem + ext)
            if ip.exists():
                scored.append((n, ip))
                break
    scored.sort(key=lambda t: (-t[0], t[1].name))
    return [p for _, p in scored[:top]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/chinese_food_18.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="runs/preview")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = cfg["names"]
    imgs = pick_images(cfg, args.split, args.top)
    if not imgs:
        sys.exit("没找到可用图片")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out_dir = PROJECT_ROOT / args.out / f"{args.split}_conf{args.conf}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"权重: {args.weights}")
    print(f"图片: {len(imgs)} 张（{args.split}，按标注框数降序）  conf={args.conf}\n")

    rows = []
    for ip in imgs:
        r = model.predict(str(ip), imgsz=640, conf=args.conf, iou=0.5, device=0,
                          verbose=False, save=False)[0]
        preds = []
        for b in r.boxes:
            cid = int(b.cls.item())
            preds.append((names[cid], float(b.conf.item())))

        lbl = Path(str(ip.parent).replace("images", "labels")) / (ip.stem + ".txt")
        gt = []
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) >= 5:
                    gt.append(names[int(float(p[0]))])

        # 保存带框图
        import cv2
        annotated = r.plot(labels=True, conf=True, line_width=2)
        save = out_dir / f"{ip.stem}_pred.jpg"
        cv2.imencode(".jpg", annotated)[1].tofile(str(save))

        print(f"[{ip.name}]")
        print(f"  真实 {len(gt):2d}: {', '.join(sorted(set(gt)))}")
        if preds:
            s = sorted(preds, key=lambda t: -t[1])
            print(f"  预测 {len(preds):2d}: " + ", ".join(f"{n}({c:.2f})" for n, c in s))
        else:
            print("  预测  0: （无检出）")
        print(f"  -> {save}")
        rows.append((ip.name, len(gt), len(preds)))

    print("\n" + "=" * 60)
    tot_gt = sum(r[1] for r in rows)
    tot_pr = sum(r[2] for r in rows)
    print(f"合计: 真实框 {tot_gt}  预测框 {tot_pr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
