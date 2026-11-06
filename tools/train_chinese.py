# -*- coding: utf-8 -*-
"""中餐 12 类检测模型训练脚本（YOLOv11）。

用法（GPU，插电后）：
    python tools/train_chinese.py --model models/weights/yolo11s.pt --epochs 120 --batch 16

用法（纯 CPU 兜底）：
    python tools/train_chinese.py --model models/weights/yolo11n.pt --epochs 100 --batch 8

训练完的权重在 runs/chinese/<name>/weights/best.pt，
再用 tools/deploy_model.py 一条命令替换进项目。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # 让 `import models`（CBAM / WIoU）可用
    sys.path.insert(0, str(PROJECT_ROOT))


def detect_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def check_dataset(data_yaml: Path) -> dict:
    import yaml

    if not data_yaml.exists():
        sys.exit(f"[错误] 数据集配置不存在: {data_yaml}")
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", ""))
    if not root.exists():
        sys.exit(
            f"[错误] 数据根目录不存在: {root}\n"
            f"       请先创建 ASCII 联接：\n"
            f"       powershell -Command \"New-Item -ItemType Junction -Path 'D:\\Work\\datasets\\chinese_food' "
            f"-Target 'D:\\Work\\膳食模型\\数据'\""
        )
    for split in ("train", "val"):
        p = root / cfg[split]
        if not p.exists():
            sys.exit(f"[错误] {split} 图片目录不存在: {p}")
        n = len([f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"  {split:5s}: {n:5d} 张  ({p})")
    print(f"  类别数 : {cfg.get('nc', len(cfg['names']))}")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="训练中餐菜品检测模型")
    ap.add_argument("--model", default="models/weights/yolo11s.pt", help="预训练权重")
    ap.add_argument("--data", default="configs/chinese_food.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=0, help="0=按设备自动（GPU 16 / CPU 8）")
    ap.add_argument("--workers", type=int, default=None, help="0=自动（GPU 8 / CPU 2）")
    ap.add_argument("--device", default=None, help="cuda 用 0，CPU 用 cpu")
    ap.add_argument("--patience", type=int, default=30, help="mAP 连续不涨多少轮后早停")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true", help="从上次中断处继续")
    ap.add_argument("--cache", default="none", choices=["none", "ram", "disk"])
    ap.add_argument("--no-close-mosaic", action="store_true")
    ap.add_argument("--pretrained", default=None,
                    help="迁移学习权重（--model 给 .yaml 时用，例如 models/weights/yolo11s.pt）")
    ap.add_argument("--wiou", default="none", choices=["none", "v1", "v2", "v3"],
                    help="用 WIoU 替换 bbox 回归损失（v3 推荐）")
    ap.add_argument("--wiou-warmup", type=int, default=300,
                    help="WIoU 生效前先用 CIoU 热身多少个训练步（结构改动过建议 300）")
    ap.add_argument("--seed", type=int, default=0,
                    help="随机种子。单 seed 对比不可靠时，用 1/2/3 复跑取均值")
    args = ap.parse_args()

    os.chdir(PROJECT_ROOT)
    device = detect_device(args.device)
    is_gpu = device != "cpu"

    model_path = Path(args.model)
    is_yaml = model_path.suffix.lower() in (".yaml", ".yml")
    if is_yaml and not model_path.exists():
        sys.exit(f"[错误] 模型结构文件不存在: {model_path}")
    if not is_yaml and not model_path.exists() and not model_path.stem.lower().startswith("yolo"):
        sys.exit(f"[错误] 预训练权重不存在: {model_path}")

    batch = args.batch or (16 if is_gpu else 8)
    workers = args.workers if args.workers is not None else (8 if is_gpu else 2)
    name = args.name or model_path.stem.replace("yolo11", "cn11")

    print("=" * 62)
    print("中餐菜品检测训练")
    print("=" * 62)
    print(f"  设备    : {device}  ({'GPU' if is_gpu else 'CPU —— 会很慢，建议接 NVIDIA 显卡或用云 GPU'})")
    print(f"  权重    : {model_path}{'  （结构定义 yaml）' if is_yaml else ''}")
    if is_yaml:
        print(f"  迁移    : {args.pretrained or '无 —— 从零训练'}")
    print(f"  bbox损失: {'官方 CIoU' if args.wiou == 'none' else 'WIoU-' + args.wiou}")
    print(f"  数据集  : {args.data}")
    check_dataset(Path(args.data))
    print(f"  epochs  : {args.epochs}   imgsz={args.imgsz}  batch={batch}  workers={workers}  seed={args.seed}")
    print(f"  输出    : runs/chinese/{name}/")
    print("=" * 62)

    import torch

    if not is_gpu and not torch.cuda.is_available():
        print("\n[提示] torch 当前是 CPU 版（" + torch.__version__ + "）。")
        print("       即使机器有 NVIDIA 显卡也用不了，需要先装 CUDA 版 torch：")
        print("       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("       装完用 python -c \"import torch;print(torch.cuda.is_available())\" 确认输出 True\n")

    from ultralytics import YOLO

    t0 = time.time()
    if is_yaml:
        import models  # noqa: F401  —— 把 CBAM 注册进 ultralytics 的模块命名空间
        model = YOLO(str(model_path))
        if args.pretrained:
            ck = Path(args.pretrained)
            if not ck.exists():
                sys.exit(f"[错误] 迁移权重不存在: {ck}")
            model.load(str(ck))
            print(f"  [迁移] 已载入 {ck.name}，未匹配的层（如 CBAM）随机初始化")
    else:
        model = YOLO(str(model_path))

    if args.wiou != "none":
        from models.wiou_patch import apply_wiou

        ver = apply_wiou(args.wiou, warmup_steps=args.wiou_warmup)
        print(f"  [损失] BboxLoss 已替换为 WIoU-{ver}（前 {args.wiou_warmup} 步用 CIoU 热身）")
    kwargs = dict(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "chinese"),
        name=name,
        exist_ok=True,
        pretrained=not is_yaml,  # yaml 结构用 model.load() 显式迁移，避免重复下载官方权重
        optimizer="auto",
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        val=True,
        plots=True,
        cache=(args.cache == "ram") if args.cache != "disk" else "disk",
        amp=is_gpu,
        cos_lr=True,
        close_mosaic=0 if args.no_close_mosaic else 10,
        # 轻度增强：食堂菜品是实拍，不需要过强几何变换
        degrees=5.0,
        translate=0.1,
        scale=0.35,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.35,
    )
    if args.resume:
        kwargs["resume"] = True

    model.train(**kwargs)

    best = PROJECT_ROOT / "runs" / "chinese" / name / "weights" / "best.pt"
    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    if best.exists():
        print(f"训练完成，用时 {elapsed/3600:.2f} 小时")
        print(f"最佳权重: {best}")
        print("\n下一步（验证 + 替换进项目）：")
        print(f"  python tools/deploy_model.py --weights \"{best}\" --validate")
    else:
        print(f"未找到 best.pt，检查 runs/chinese/{name}/weights/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
