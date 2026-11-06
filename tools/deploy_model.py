# -*- coding: utf-8 -*-
"""把训练好的 best.pt 部署进项目：校验类别 → 复制权重 → 可选改 config。

用法：
    python tools/deploy_model.py --weights runs/chinese/cn11s/weights/best.pt --validate
    python tools/deploy_model.py --weights ... --patch-config      # 同时改 config.yaml
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "models" / "weights" / "chinese_food.pt"


def load_ref_keys() -> set[str]:
    import yaml

    p = PROJECT_ROOT / "data" / "food_reference.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {str(k).lower() for k in data}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--validate", action="store_true", help="在测试集上评估并打印每类指标")
    ap.add_argument("--patch-config", action="store_true", help="把 config.yaml 的 food_model 指向新权重")
    ap.add_argument("--copy", action="store_true", default=True)
    args = ap.parse_args()

    src = Path(args.weights)
    if not src.is_absolute():
        src = (PROJECT_ROOT / src).resolve()
    if not src.exists():
        sys.exit(f"[错误] 权重不存在: {src}")

    from ultralytics import YOLO

    model = YOLO(str(src))
    names = model.names
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    print(f"权重: {src}")
    print(f"类别数: {len(names)}")
    print("类别:", list(names))

    ref = load_ref_keys()
    missing = [n for n in names if str(n).lower() not in ref]
    if missing:
        print("\n[警告] 这些类别在 data/food_reference.yaml 里没有营养数据：", missing)
        print("       它们会被检测白名单过滤掉（等于永远检不出来）。")
        print("       请先把这些 key 补进 food_reference.yaml 和 density_table.yaml，再运行。")
    else:
        print("[OK] 所有类别都能在 food_reference.yaml 中查到营养数据")

    lack_zh = [n for n in names if n not in _zh_keys()]
    if lack_zh:
        print("[提示] 这些类别没有中文名映射（只影响显示）:", lack_zh)

    # 复制
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, TARGET)
    print(f"\n已复制到: {TARGET}")

    # 可选：改 config
    if args.patch_config:
        cfg_path = PROJECT_ROOT / "config.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        old = "food_model: models/weights/yolo11m.pt"
        new = "food_model: models/weights/chinese_food.pt"
        if new in text:
            print("config.yaml 已指向新权重，无需修改")
        elif old in text:
            cfg_path.write_text(text.replace(old, new), encoding="utf-8")
            print("config.yaml 的 food_model 已改为 models/weights/chinese_food.pt")
        else:
            print("[警告] 没在 config.yaml 里找到 food_model 那一行，请手动改")

    # 可选：评估
    if args.validate:
        print("\n在测试集上评估...")
        m = model.val(data=str(PROJECT_ROOT / "configs" / "chinese_food.yaml"),
                      split="test", imgsz=640, batch=8, device="cpu", plots=False, verbose=False)
        print(f"\nmAP50    : {m.box.map50:.4f}")
        print(f"mAP50-95 : {m.box.map:.4f}")
        print("\n每类 mAP50：")
        for i, name in enumerate(names):
            if i < len(m.box.ap50):
                print(f"  {name:22s} {m.box.ap50[i]:.3f}")
    return 0


def _zh_keys() -> set[str]:
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.food_labels import FOOD_LABELS_ZH

        return set(FOOD_LABELS_ZH)
    except Exception:
        return set()


if __name__ == "__main__":
    sys.exit(main())
