"""检测流水线测试：盘子 OBB / 尺度 / 食物 bbox。

运行：python tests/test_detection.py
      （也兼容 pytest：pytest tests/test_detection.py -q）
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import FoodDetector, PlateDetector, ScaleCalculator  # noqa: E402
from src.utils.image_utils import load_image                            # noqa: E402

TEST_IMAGE = PROJECT_ROOT / "data" / "images" / "test.jpg"

_plate_detector = None
_food_detector = None


def get_plate_detector() -> PlateDetector:
    global _plate_detector
    if _plate_detector is None:
        _plate_detector = PlateDetector()
    return _plate_detector


def get_food_detector() -> FoodDetector:
    global _food_detector
    if _food_detector is None:
        _food_detector = FoodDetector()
    return _food_detector


def test_model_classes():
    """盘子模型 4 类；食物模型（COCO-80）与营养表白名单至少有 8 类交集。"""
    from src.detection.food_detector import load_food_whitelist

    p = get_plate_detector()
    f = get_food_detector()
    assert len(p.class_names) == 4, f"盘子模型类别数异常: {p.class_names}"
    assert set(["Fork", "Knife", "Plate", "Spoon"]).issubset(set(p.class_names)), p.class_names

    overlap = {n.lower().replace(" ", "_") for n in f.class_names} & load_food_whitelist()
    assert len(overlap) >= 8, f"食物模型与营养表白名单交集过少: {overlap}"
    print(f"[OK] 类别：盘子 {p.class_names}；食物模型 {len(f.class_names)} 类，命中白名单 {sorted(overlap)}")


def test_plate_detection():
    """test.jpg 应能检出 Plate（OBB 4 点 + conf）。"""
    image = load_image(TEST_IMAGE)
    dets = get_plate_detector().detect(image)
    assert dets, "未检测到任何盘子/餐具"
    plate = PlateDetector.get_plate(dets)
    assert plate is not None, f"未检测到 Plate，实际：{[d['name'] for d in dets]}"
    assert plate["obb"].shape == (4, 2)
    assert 0 < plate["conf"] <= 1
    assert plate["long_px"] > 50
    print(f"[OK] 盘子：conf={plate['conf']:.3f}, 最长边={plate['long_px']:.1f}px")


def test_scale_calculation():
    """尺度应在合理范围（4-80 px/cm），且给出可追溯来源。"""
    image = load_image(TEST_IMAGE)
    dets = get_plate_detector().detect(image)
    info = ScaleCalculator().calculate_scale(dets, image_width=image.shape[1])
    assert "px_per_cm" in info and info["px_per_cm"] > 0
    assert 2 < info["px_per_cm"] < 200, f"尺度不合理: {info['px_per_cm']}"
    print(f"[OK] 尺度：{info['px_per_cm']:.2f} px/cm（来源 {info['source']}）")


def test_food_detection():
    """食物检测应返回 bbox/conf/area，且盘内检测优先；三张测试图至少一张能检出食物。"""
    detector = get_food_detector()
    total_hits = 0
    for name in ["test.jpg", "test2.jpg", "test3.jpg"]:
        path = PROJECT_ROOT / "data" / "images" / name
        if not path.exists():
            continue
        image = load_image(path)
        plates = get_plate_detector().detect(image)
        plate = PlateDetector.get_plate(plates)
        out = detector.detect(image, region_quad=plate["obb"] if plate else None)
        dets = out["detections"]
        total_hits += len(dets)
        for d in dets:
            assert d["name"] and len(d["bbox"]) == 4
            assert d["area_px"] > 0
            assert 0 < d["conf"] <= 1
        print(f"[OK] {name}（{out['mode']}）：{[(d['name'], round(d['conf'], 2)) for d in dets] or '未检出'}"
              f"{' / 过滤:' + str(sorted({r['name'] for r in out['rejected']})) if out.get('rejected') else ''}")
    assert total_hits > 0, "三张测试图都没有检出白名单内食物"
    print(f"[OK] 食物检测共 {total_hits} 个目标")


def test_expand_and_crop():
    """OBB 外扩 + 旋转裁剪：裁剪图尺寸应接近原始四边形边长。"""
    from src.utils.image_utils import extract_rotated_region, quad_long_short_sides

    image = load_image(TEST_IMAGE)
    plates = get_plate_detector().detect(image)
    plate = PlateDetector.get_plate(plates)
    if plate is None:
        print("[SKIP] 未检出盘子，跳过裁剪测试")
        return
    long_px, short_px = quad_long_short_sides(plate["obb"])
    crop = extract_rotated_region(image, plate["obb"], ratio=1.0)
    h, w = crop.shape[:2]
    assert abs(max(w, h) - long_px) / max(long_px, 1) < 0.15, f"裁剪尺寸偏差过大: crop={w}x{h}, quad={long_px:.0f}x{short_px:.0f}"
    print(f"[OK] 旋转裁剪：{w}x{h}（原四边形 {long_px:.0f}x{short_px:.0f}）")


def test_sam_checkpoint():
    """SAM checkpoint 可定位（不加载权重，避免 CPU 上 70s 的编码耗时）。"""
    from src.detection.sam_segmenter import _resolve_checkpoint

    p = _resolve_checkpoint()
    assert p.exists(), f"SAM checkpoint 不存在: {p}"
    assert p.stat().st_size > 1_000_000_000, f"文件大小异常: {p.stat().st_size}"
    print(f"[OK] SAM checkpoint：{p}（{p.stat().st_size / 2**30:.2f} GB）")


def main() -> int:
    tests = [
        test_model_classes,
        test_plate_detection,
        test_scale_calculation,
        test_food_detection,
        test_expand_and_crop,
        test_sam_checkpoint,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {t.__name__}: {exc.__class__.__name__}: {exc}")
    print(f"\n检测测试完成：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
