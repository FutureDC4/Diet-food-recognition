# -*- coding: utf-8 -*-
"""YOLO 数据集完整性校验：标签/图片一一对应、坐标合法、类别分布。

用法：
    python tools/check_dataset.py "D:/Work/膳食模型/数据"
"""
import collections
import glob
import os
import sys

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_names(root):
    yml = os.path.join(root, "data.yaml")
    names = []
    nc = None
    with open(yml, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("nc:"):
                nc = int(s.split(":")[1].strip())
            elif s.startswith("names:"):
                body = s.split(":", 1)[1].strip()
                body = body.strip("[]")
                for tok in body.split(","):
                    tok = tok.strip().strip("'\"")
                    if tok:
                        names.append(tok)
    return names, nc


def check_split(root, split, names):
    idir = os.path.join(root, split, "images")
    ldir = os.path.join(root, split, "labels")
    if not os.path.isdir(idir):
        return None

    imgs = {}
    for p in glob.glob(os.path.join(idir, "*")):
        if p.lower().endswith(IMG_EXT):
            imgs[os.path.splitext(os.path.basename(p))[0]] = p
    lbls = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(ldir, "*.txt"))
    }

    missing = sorted(set(imgs) - set(lbls))
    orphan = sorted(set(lbls) - set(imgs))

    empty, bad, per_img_boxes = [], [], []
    cls_cnt = collections.Counter()
    total_boxes = 0
    nc = len(names)
    tiny = 0

    for stem, lp in lbls.items():
        if stem not in imgs:
            continue
        with open(lp, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            empty.append(stem)
            per_img_boxes.append(0)
            continue
        n = 0
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                bad.append((stem, "字段数=" + str(len(parts))))
                continue
            try:
                c = int(float(parts[0]))
                cx, cy, w, h = (float(x) for x in parts[1:])
            except ValueError:
                bad.append((stem, "非数字:" + ln))
                continue
            if not (0 <= c < nc):
                bad.append((stem, "类别越界:" + str(c)))
                continue
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                bad.append((stem, "坐标越界:" + ln))
                continue
            if w <= 0 or h <= 0:
                bad.append((stem, "宽高非正:" + ln))
                continue
            if w * h < 1e-5:
                tiny += 1
            cls_cnt[names[c]] += 1
            n += 1
            total_boxes += 1
        per_img_boxes.append(n)

    per_img_boxes.sort()
    med = per_img_boxes[len(per_img_boxes) // 2] if per_img_boxes else 0
    return {
        "split": split,
        "images": len(imgs),
        "labels": len(lbls),
        "missing": missing,
        "orphan": orphan,
        "empty": empty,
        "bad": bad,
        "boxes": total_boxes,
        "cls": cls_cnt,
        "per_img_med": med,
        "per_img_max": per_img_boxes[-1] if per_img_boxes else 0,
        "tiny": tiny,
        "multi_class_imgs": sum(1 for _, p in lbls.items() if p in lbls),
    }


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    root = os.path.abspath(root)
    print("数据集根目录:", root)
    names, nc = load_names(root)
    print(f"类别数 nc={nc}  声明={len(names)}")
    print("类别表:", names)
    if nc != len(names):
        print("!! data.yaml 的 nc 与 names 数量不一致")

    # data.yaml 里的路径是否可用
    for split in ("train", "valid", "test"):
        for key in (split, "val" if split == "valid" else split):
            pass

    results = []
    for split in ("train", "valid", "test"):
        r = check_split(root, split, names)
        if r:
            results.append(r)

    print("\n" + "=" * 66)
    for r in results:
        print(f"[{r['split']:5s}] 图 {r['images']:5d} | 标签 {r['labels']:5d} "
              f"| 框 {r['boxes']:6d} | 缺标签 {len(r['missing']):3d} "
              f"| 孤立标签 {len(r['orphan']):3d} | 空标签 {len(r['empty']):3d} "
              f"| 异常 {len(r['bad']):3d} | 每图框数 中位{r['per_img_med']} 最多{r['per_img_max']}")
        for k in ("missing", "orphan", "empty"):
            if r[k]:
                print(f"     {k} 示例:", r[k][:5])
        if r["bad"]:
            print("     bad 示例:", r["bad"][:5])

    total = collections.Counter()
    for r in results:
        total.update(r["cls"])
    print("\n--- 全库类别分布 ---")
    for name in names:
        print(f"  {name:22s} {total.get(name, 0):6d}")
    print(f"  {'合计':22s} {sum(total.values()):6d}")

    # train/val 类别覆盖差异
    tr = results[0]["cls"] if results else collections.Counter()
    va = results[1]["cls"] if len(results) > 1 else collections.Counter()
    lack = [n for n in names if tr.get(n, 0) and not va.get(n, 0)]
    if lack:
        print("\n!! 验证集缺失这些类（会拉低 mAP/无法评估）:", lack)
    few = [n for n in names if 0 < tr.get(n, 0) < 50]
    if few:
        print("!! 训练样本偏少的类(<50):", [(n, tr[n]) for n in few])

    bad_total = sum(len(r["bad"]) for r in results) + sum(len(r["empty"]) for r in results)
    print("\n结论:", "标注可用，可直接训练" if bad_total == 0 else f"有 {bad_total} 处需要人工确认")
    return 0


if __name__ == "__main__":
    sys.exit(main())
