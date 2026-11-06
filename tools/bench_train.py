# -*- coding: utf-8 -*-
"""CPU 训练速度基准：测每种模型在 640 输入下的单步耗时，推算每 epoch 时间。

用法：python tools/bench_train.py
"""
import time

import torch
from ultralytics import YOLO


def bench(weight, imgsz=640, batch=8, iters=4):
    model = YOLO(weight).model
    model.train()
    model.args = model.args if hasattr(model, "args") else None
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(batch, 3, imgsz, imgsz)
    # 训练模式下的前向（含 loss 计算）需要 targets，这里只测 backbone+head 前向与反传成本
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad()
        try:
            preds = model(x)
            loss = sum(p.sum() for p in preds) if isinstance(preds, (list, tuple)) else preds.sum()
        except Exception:
            # 部分版本 train 模式必须走 loss，退回 eval 前向只测推理成本
            model.eval()
            with torch.no_grad():
                out = model(x)
            model.train()
            o = out[0] if isinstance(out, (list, tuple)) else out
            loss = o.sum()
        loss.backward()
        opt.step()
    dt = (time.time() - t0) / iters
    return dt


def main():
    n_params = {}
    for w in ("models/weights/yolo11n.pt", "models/weights/yolo11s.pt", "models/weights/yolo11m.pt"):
        try:
            m = YOLO(w)
        except Exception as e:
            print(f"{w}: 无法加载 ({e})")
            continue
        n_params[w] = sum(p.numel() for p in m.model.parameters()) / 1e6
        for batch in (8,):
            try:
                dt = bench(w, batch=batch)
            except Exception as e:
                print(f"{w} 基准失败: {e}")
                continue
            per_img = dt / batch
            ep = per_img * 1597
            print(f"{w:32s} {n_params[w]:5.2f}M  batch={batch}  "
                  f"{dt*1000:6.0f} ms/step  {per_img*1000:5.0f} ms/img  "
                  f"≈ {ep/60:5.1f} min/epoch (1597 图)")


if __name__ == "__main__":
    main()
