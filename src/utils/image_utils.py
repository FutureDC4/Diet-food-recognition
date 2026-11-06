"""图片工具：本地/URL 加载、旋转区域裁剪、结果绘制与保存。

注意：Windows 下中文路径用 cv2.imread / imwrite 会失败，这里统一走
np.fromfile + cv2.imdecode 与 cv2.imencode + tofile。
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_url(source: str) -> bool:
    return str(source).lower().startswith(("http://", "https://"))


def load_image(source: str | Path) -> np.ndarray:
    """加载图片为 BGR ndarray。自动区分 URL 与本地路径。"""
    s = str(source).strip().strip('"').strip("'")
    if not s:
        raise ValueError("图片路径为空")
    if is_url(s):
        return load_image_from_url(s)
    return load_image_from_file(s)


def load_image_from_file(path: str | Path) -> np.ndarray:
    p = Path(str(path).strip().strip('"').strip("'"))
    if not p.exists():
        raise FileNotFoundError(f"图片不存在：{p}")
    if p.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图片格式：{p.suffix}（支持 {sorted(IMAGE_SUFFIXES)}）")

    data = np.fromfile(str(p), dtype=np.uint8)  # 兼容中文路径
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片解码失败（文件可能损坏）：{p}")
    return image


def load_image_from_url(url: str, timeout: int = 20) -> np.ndarray:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    pil = Image.open(BytesIO(resp.content)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """保存 BGR 图片（兼容中文路径）。"""
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"图片编码失败：{p}")
    buf.tofile(str(p))
    return p


# ---------------------------------------------------------------- 几何
def quad_long_short_sides(quad: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """返回四边形（4×2）的最长边与最短边长度（像素）。"""
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    sides = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
    return max(sides), min(sides)


def quad_center(quad: Sequence[Sequence[float]]) -> Tuple[float, float]:
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def expand_quad(quad: Sequence[Sequence[float]], ratio: float = 1.1) -> np.ndarray:
    """以最小外接旋转矩形为基础，按比例外扩四边形（保持旋转角）。"""
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    box = cv2.boxPoints(((cx, cy), (w * ratio, h * ratio), angle))
    return np.asarray(box, dtype=np.float32)


def quad_to_bbox(quad: Sequence[Sequence[float]]) -> list[float]:
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


def extract_rotated_region(
    image: np.ndarray,
    quad: Sequence[Sequence[float]],
    ratio: float = 1.0,
    return_matrix: bool = False,
):
    """把旋转四边形区域「摆正」后裁剪出来（1:1 像素密度，不缩放）。

    像素/厘米比例在裁剪后保持不变，因此裁剪图上的 bbox 面积可直接用原始 scale 换算。

    return_matrix=True 时额外返回 原图→裁剪图 的 3×3 透视矩阵，
    调用方可用它的逆矩阵把裁剪图上的 bbox 映射回原图坐标（画框、对标注都需要）。
    """
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if ratio != 1.0:
        pts = expand_quad(pts, ratio)

    # 排序为 左上/右上/右下/左下，保证透视变换朝向正确
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = _order_quad(box)

    width = int(round(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3]))))
    height = int(round(max(np.linalg.norm(box[1] - box[2]), np.linalg.norm(box[3] - box[0]))))
    width, height = max(width, 2), max(height, 2)

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    if return_matrix:
        return crop, matrix
    return crop


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """把 4 个点排序成 左上、右上、右下、左下。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # 左上：x+y 最小
    ordered[2] = pts[np.argmax(s)]      # 右下：x+y 最大
    ordered[1] = pts[np.argmin(diff)]   # 右上：y-x 最小
    ordered[3] = pts[np.argmax(diff)]   # 左下：y-x 最大
    return ordered


# ---------------------------------------------------------------- 可视化
def draw_detections(
    image: np.ndarray,
    food_detections: Iterable[dict] = (),
    plate_detections: Iterable[dict] = (),
    region_quad: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    """在图上画 OBB（盘子/餐具）与 bbox（食物），用于人工核对。"""
    canvas = image.copy()

    if region_quad is not None:
        pts = np.asarray(region_quad, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (255, 160, 0), 2)

    for det in plate_detections:
        quad = det.get("obb")
        if quad is None:
            continue
        pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 200, 0), 2)
        cx, cy = quad_center(quad)
        cv2.putText(canvas, f"{det['name']} {det.get('conf', 0):.2f}", (int(cx) - 40, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

    for det in food_detections:
        box = det.get("bbox")
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det['name']} {det.get('conf', 0):.2f}"
        if det.get("weight_g"):
            label += f" {det['weight_g']:.0f}g"
        cv2.putText(canvas, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

    return canvas


def image_size(image: np.ndarray) -> Tuple[int, int]:
    h, w = image.shape[:2]
    return w, h


if __name__ == "__main__":
    print("image_utils 自检：", sorted(IMAGE_SUFFIXES))
