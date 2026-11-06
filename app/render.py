# -*- coding: utf-8 -*-
"""检测结果可视化（中文标签）。

项目原有的 draw_detections 用 cv2.putText，只能画英文；
上位机需要中文菜名，这里改用 PIL 绘制，字体走 Windows 微软雅黑。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
)

# 与 GUI 主题一致的 8 色调色板（同一类别永远是同一颜色）
_PALETTE = [
    (214, 69, 65), (220, 133, 45), (196, 164, 46), (71, 156, 88),
    (56, 150, 160), (61, 111, 190), (128, 96, 190), (190, 87, 141),
]


@lru_cache(maxsize=4)
def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _color_for(name: str) -> tuple[int, int, int]:
    return _PALETTE[hash(str(name)) % len(_PALETTE)]


def draw_annotations(
    image_bgr: np.ndarray,
    dishes: Iterable[dict] = (),
    plate_quad: Optional[Sequence[Sequence[float]]] = None,
    zoom: int = 1,
) -> np.ndarray:
    """在 BGR 图上画食物框 + 中文标签（菜名 / 置信度 / 重量）。

    dishes: 至少包含 name、bbox；可选 conf、weight_g、name_cn、rule_adjusted
    返回 BGR ndarray，可直接给 Qt 显示。
    """
    rgb = Image.fromarray(image_bgr[:, :, ::-1].copy())
    draw = ImageDraw.Draw(rgb, "RGBA")

    if plate_quad is not None:
        pts = [(float(x), float(y)) for x, y in plate_quad]
        draw.polygon(pts, outline=(0, 200, 120, 220), width=max(2, zoom))

    size = max(14, int(round(min(rgb.size) * 0.026))) + 2 * (zoom - 1)
    font = _font(size)

    for det in dishes:
        box = det.get("bbox")
        if not box:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        color = _color_for(det.get("name", "?"))
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=max(2, zoom))

        label = det.get("name_cn") or str(det.get("name", ""))
        parts = [label]
        if det.get("conf") is not None:
            parts.append(f"{float(det['conf']):.2f}")
        if det.get("weight_g"):
            parts.append(f"{float(det['weight_g']):.0f}g")
        if det.get("rule_adjusted"):
            parts.append("占比修正")
        text = "  ".join(parts)

        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(3, size // 4)
        ty = max(y1 - th - 2 * pad, 0)
        draw.rectangle([x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad], fill=color + (235,))
        draw.text((x1 + pad, ty + pad - tb[1]), text, font=font, fill=(255, 255, 255, 255))

    return np.asarray(rgb)[:, :, ::-1].copy()
