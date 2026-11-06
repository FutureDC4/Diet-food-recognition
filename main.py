#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""膳食识别系统（命令行版）

用法：
    python main.py                        # 交互式输入
    python main.py --image data/images/test.jpg --age 25 --gender 男 \
                   --height 175 --weight 70 --activity 1     # 非交互（便于批量/测试）

流程：图片 -> 盘子/餐具 OBB 检测 -> 像素/厘米尺度 -> 食物 bbox 检测
      -> 重量估计 -> 热量计算 -> BMR/TDEE -> Ollama 膳食建议（流式）
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

# 允许从任意目录运行（保证 `import src.*` / `import models` 可用）
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import FoodDetector, PlateDetector, ScaleCalculator          # noqa: E402
from src.health import BMRCalculator, normalize_gender                          # noqa: E402
from src.llm import SYSTEM_PROMPT, OllamaClient, build_prompt                   # noqa: E402
from src.utils.config_loader import ensure_dir, get, get_config, resolve_path   # noqa: E402
from src.utils.food_labels import label_with_zh, zh_name                        # noqa: E402
from src.utils.image_utils import draw_detections, image_size, load_image, save_image  # noqa: E402
from src.utils.logger import get_logger, setup_logger                           # noqa: E402
from src.weight_estimation import NutritionCalculator, PortionEstimator          # noqa: E402

log = get_logger("主程序")

LINE = "━" * 28
THIN = "─" * 28
BANNER = f"""================================
  膳食识别系统（命令行版）
================================"""


# ============================================================ 输出工具
def dwidth(text: str) -> int:
    """计算字符串在终端里的显示宽度（中文占 2 列）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


def pad(text: str, width: int, align: str = "left") -> str:
    gap = max(width - dwidth(text), 0)
    return (" " * gap + str(text)) if align == "right" else (str(text) + " " * gap)


def hr(char: str = "-", width: int = 64) -> None:
    print(char * width)


def step(msg: str) -> None:
    print(f"  · {msg}")


# ============================================================ 输入收集
def ask(prompt: str, default: Optional[str] = None, cast=None, validator=None, trials: int = 5):
    """带校验的 input（最多重试 trials 次）。"""
    for _ in range(trials):
        raw = input(prompt).strip()
        if not raw:
            if default is not None:
                raw = str(default)
                print(f"    （使用默认值：{raw}）")
            else:
                print("    ! 不能为空，请重新输入")
                continue
        try:
            value = cast(raw) if cast else raw
        except (TypeError, ValueError):
            print("    ! 输入格式有误，请重新输入")
            continue
        if validator is not None:
            ok, msg = validator(value)
            if not ok:
                print(f"    ! {msg}")
                continue
        return value
    raise SystemExit("输入错误次数过多，已退出")


def ask_image_path() -> str:
    """交互式获取图片路径（支持拖拽文件到终端）。"""
    while True:
        raw = input("请输入食物图片路径: ").strip().strip('"').strip("'")
        if not raw:
            print("    ! 路径不能为空")
            continue
        if raw.lower().startswith(("http://", "https://")):
            return raw
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists() and p.is_file():
            return str(p)
        print(f"    ! 找不到图片：{p}")


def activity_prompt() -> str:
    return "请输入活动量(1-4, 1=轻度 2=中度 3=重度 4=极重): "


def collect_user_input_from_console() -> Dict[str, Any]:
    print()
    image_path = ask_image_path()
    age = ask("请输入年龄: ", cast=int, validator=lambda v: (0 < v < 130, "年龄需在 1-129 之间"))
    gender = ask(
        "请输入性别(男/女): ",
        validator=lambda v: (_is_gender(v), "只能输入 男 或 女"),
    )
    gender = normalize_gender(gender)
    height = ask("请输入身高(cm): ", cast=float, validator=lambda v: (80 <= v <= 250, "身高需在 80-250 cm"))
    weight = ask("请输入体重(kg): ", cast=float, validator=lambda v: (10 <= v <= 350, "体重需在 10-350 kg"))
    activity = ask(
        activity_prompt(),
        cast=int,
        validator=lambda v: (v in (1, 2, 3, 4), "活动量只能是 1/2/3/4"),
    )
    return {
        "image_path": image_path,
        "age": int(age),
        "gender": gender,
        "height_cm": float(height),
        "weight_kg": float(weight),
        "activity_level": int(activity),
    }


def _is_gender(value: str) -> bool:
    try:
        normalize_gender(value)
        return True
    except ValueError:
        return False


# ============================================================ 主流程
def run_pipeline(
    image_path: str,
    user: Dict[str, Any],
    use_llm: bool = True,
    save_annotated: bool = True,
    save_db: bool = True,
    model_overrides: Optional[Dict[str, str]] = None,
    verbose: bool = False,
    use_sam: bool = False,
) -> Dict[str, Any]:
    """完整流程：图片 → 检测 → 重量 → 热量 → 健康 → LLM。"""
    model_overrides = model_overrides or {}
    t0 = time.time()

    print()
    print("正在识别食物...")
    image = load_image(image_path)
    width_px, height_px = image_size(image)
    step(f"图片已加载：{width_px}×{height_px} px")

    # ---------- 1. 盘子 + 餐具（OBB） ----------
    plate_detector = PlateDetector(model_path=model_overrides.get("plate_model"))
    plate_dets = plate_detector.detect(image)
    plate = PlateDetector.get_plate(plate_dets)
    if plate:
        step(f"盘子已定位：置信度 {plate['conf']:.2f}，直径约 {plate['long_px']:.0f} px")
    else:
        print("  ! 未检测到盘子，将直接在整图上识别食物（重量估计误差较大）")

    # ---------- 2. 尺度（像素/厘米） ----------
    scale_calc = ScaleCalculator()
    scale_info = scale_calc.calculate_scale(plate_dets, image_width=width_px)
    px_per_cm = scale_info["px_per_cm"]
    if plate:
        plate_cm = plate["long_px"] / px_per_cm
        step(f"尺度：{px_per_cm:.2f} px/cm（来源：{scale_info['source']}）→ 盘子直径约 {plate_cm:.1f} cm")
    else:
        step(f"尺度：{px_per_cm:.2f} px/cm（来源：{scale_info['source']}）")

    # ---------- 3. 食物检测 ----------
    food_detector = FoodDetector(model_path=model_overrides.get("food_model"))
    # 默认全图检测（与训练/评估一致）；盘内裁剪会让框偏大并切掉盘边食物
    region_quad = plate["obb"] if (plate is not None and get("plate.crop_for_detection", False)) else None
    food_out = food_detector.detect(image, region_quad=region_quad, sam=use_sam)
    raw_dets: List[dict] = food_out["detections"]
    rejected: List[dict] = food_out.get("rejected", [])
    det_mode = food_out["mode"]
    if det_mode == "full":
        step("食物检测区域：整图（未使用盘子裁剪）")
    else:
        step("食物检测区域：盘子 OBB 旋转裁剪（已摆正）")
    if use_sam:
        step("面积来源：SAM 掩膜（精度更高）")
    if rejected:
        names = ", ".join(sorted({d["name"] for d in rejected})[:6])
        step(f"已过滤非食物/未收录类别：{names}")

    merged = food_detector.merged_detections(raw_dets)

    # ---------- 4. 重量 + 5. 热量 ----------
    plate_area_cm2 = None
    if plate is not None:
        plate_diameter_cm = plate["long_px"] / px_per_cm
        plate_area_cm2 = math.pi * (plate_diameter_cm / 2.0) ** 2

    portion = PortionEstimator()
    weight_items = portion.estimate(
        px_per_cm=px_per_cm,
        food_detections=merged,
        scale_reliable=scale_info["reliable"],
        plate_area_cm2=plate_area_cm2,
    )
    nutrition = NutritionCalculator()
    dishes = nutrition.calculate_many(weight_items)
    totals = nutrition.totals(dishes)

    # ---------- 6. 健康数据 ----------
    bmr_calc = BMRCalculator()
    health = bmr_calc.calculate(
        age=user["age"],
        gender=user["gender"],
        height_cm=user["height_cm"],
        weight_kg=user["weight_kg"],
        activity_level=user["activity_level"],
    )
    meal_share = bmr_calc.meal_share(totals["calories"], health["tdee"])

    # ---------- 7. 打印结果 ----------
    print_detection_table(dishes, raw_dets, rejected)
    print_health_block(health, meal_share, totals)

    # ---------- 8. 保存标注图 ----------
    annotated_path = None
    if save_annotated:
        out_dir = ensure_dir(resolve_path("paths.output_dir", "data/output"))
        annotated_path = out_dir / f"annotated_{Path(image_path).stem}.jpg"
        try:
            canvas = draw_detections(image, food_detections=dishes, plate_detections=plate_dets, region_quad=region_quad)
            save_image(canvas, annotated_path)
            step(f"标注图已保存：{annotated_path}")
        except Exception as exc:  # 保存失败不影响主流程
            log.warning(f"标注图保存失败：{exc}")
            annotated_path = None

    # ---------- 9. LLM 膳食建议 ----------
    prompt = build_prompt(
        detections=raw_dets,
        weights=dishes,
        calories=totals,
        user_info=user,
        bmr_tdee=health,
        meal_share=meal_share,
    )
    llm_text = ""
    if use_llm:
        print()
        print("正在调用 Ollama 生成膳食建议...")
        print(LINE)
        client = OllamaClient(
            host=model_overrides.get("host"),
            model=model_overrides.get("model"),
        )
        buffer: List[str] = []
        try:
            for chunk in client.stream_generate(prompt, system=SYSTEM_PROMPT):
                print(chunk, end="", flush=True)
                buffer.append(chunk)
        except KeyboardInterrupt:
            print("\n（已中断建议生成）")
        llm_text = "".join(buffer).strip()
        if llm_text:
            print()
        print(LINE)
    else:
        step("已跳过 Ollama 建议（--no-llm）")

    # ---------- 10. 写历史记录 ----------
    record_id = None
    if save_db and get("database.enabled", True):
        try:
            from src.database import DBManager

            record_id = DBManager().save_record(
                image_path=str(image_path),
                user_info=user,
                foods=dishes,
                totals=totals,
                health=health,
                llm_output=llm_text,
            )
            step(f"历史记录已保存（id={record_id}）")
        except Exception as exc:
            log.warning(f"历史记录保存失败：{exc}")

    elapsed = time.time() - t0
    print(f"\n完成，用时 {elapsed:.1f} 秒。")

    return {
        "image_path": str(image_path),
        "image_size": (width_px, height_px),
        "plate_detections": plate_dets,
        "plate": plate,
        "scale": scale_info,
        "px_per_cm": px_per_cm,
        "food_detections": raw_dets,
        "detection_mode": det_mode,
        "dishes": dishes,
        "totals": totals,
        "health": health,
        "meal_share": meal_share,
        "prompt": prompt,
        "llm_text": llm_text,
        "annotated_path": str(annotated_path) if annotated_path else None,
        "record_id": record_id,
        "elapsed_s": round(elapsed, 2),
    }


# ============================================================ 结果打印
def print_detection_table(dishes: List[dict], raw_dets: List[dict], rejected: Optional[List[dict]] = None) -> None:
    print(LINE)
    if not dishes:
        if raw_dets:
            print("检测到食物但重量估计失败，请检查尺度是否合理。")
        else:
            print("未检测到任何食物。")
            print("可能原因：当前模型是 COCO-80，只认识约 10 类常见食物（苹果/香蕉/披萨/三明治…）；")
            print("          中餐菜品基本识别不了；也可能是图片过暗、过小或食物不在盘内。")
            if rejected:
                names = ", ".join(sorted({d["name"] for d in rejected})[:8])
                print(f"          本次识别到但被过滤的类别：{names}")
        print(LINE)
        return

    print("检测到以下食物：")
    header = f"{pad('序号', 6)}| {pad('食物', 20)}| {pad('置信度', 10)}| {pad('重量', 12)}| 热量"
    print(header)
    print("-" * 66)
    for idx, dish in enumerate(dishes, 1):
        name = label_with_zh(dish["name"])
        conf = f"{dish.get('conf', 0):.2f}"
        weight = f"~{dish['weight_g']:.0f} g"
        kcal = f"{dish['calories']:.0f} kcal"
        print(f"{pad(idx, 6)}| {pad(name, 20)}| {pad(conf, 10)}| {pad(weight, 12)}| {kcal}")

    total_weight = sum(d["weight_g"] for d in dishes)
    total_kcal = sum(d["calories"] for d in dishes)
    print("-" * 66)
    print(f"总重量: {total_weight:.0f} g")
    print(f"总热量: {total_kcal:.0f} kcal")
    print(LINE)


def print_health_block(health: dict, meal_share: float, totals: dict) -> None:
    print("用户健康数据：")
    print(f"  BMR（基础代谢）: {health['bmr']:.0f} kcal/天")
    print(f"  TDEE（每日消耗）: {health['tdee']:.0f} kcal/天  （活动量 {health['activity_level']} {health['activity_name']}）")
    print(f"  本餐热量占比: {meal_share:.1f}% TDEE")
    print(f"  BMI: {health['bmi']}（{health['bmi_label']}）")
    print(
        f"  估算三大营养素: 蛋白质 {totals['protein_g']:.0f} g / "
        f"脂肪 {totals['fat_g']:.0f} g / 碳水 {totals['carb_g']:.0f} g"
    )
    if totals["calories"] > health["tdee"] * 0.5:
        print("  ! 提示：单餐热量已超过每日需求的一半，注意控制。")
    print(LINE)


# ============================================================ 入口
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="膳食识别系统（命令行版）")
    parser.add_argument("--image", help="图片路径或 URL（不给则进入交互模式）")
    parser.add_argument("--age", type=int, help="年龄")
    parser.add_argument("--gender", help="性别：男/女")
    parser.add_argument("--height", type=float, help="身高 cm")
    parser.add_argument("--weight", type=float, help="体重 kg")
    parser.add_argument("--activity", type=int, choices=[1, 2, 3, 4], help="活动量 1-4")
    parser.add_argument("--no-llm", action="store_true", help="不调用 Ollama")
    parser.add_argument("--no-save", action="store_true", help="不保存标注图与历史记录")
    parser.add_argument("--no-db", action="store_true", help="不写 SQLite 历史记录")
    parser.add_argument("--sam", action="store_true", help="启用 SAM 分割精修面积（更准但更慢）")
    parser.add_argument("--model", help="覆盖 Ollama 模型名")
    parser.add_argument("--host", help="覆盖 Ollama 地址，如 http://localhost:11434")
    parser.add_argument("--food-model", help="覆盖食物模型路径")
    parser.add_argument("--plate-model", help="覆盖盘子模型路径")
    parser.add_argument("--verbose", action="store_true", help="输出更详细的日志")
    return parser


def _user_from_args(args) -> Optional[Dict[str, Any]]:
    if all(v is not None for v in (args.image, args.age, args.gender, args.height, args.weight)):
        return {
            "image_path": str(Path(args.image)) if not str(args.image).lower().startswith("http") else args.image,
            "age": int(args.age),
            "gender": normalize_gender(args.gender),
            "height_cm": float(args.height),
            "weight_kg": float(args.weight),
            "activity_level": int(args.activity or 1),
        }
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logger("DEBUG" if args.verbose else None)

    # 老式 Windows 控制台（GBK）遇到无法编码的字符时替换而非崩溃
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    print(BANNER)
    log.debug(f"配置项：{list(get_config().keys())}")

    try:
        user = _user_from_args(args)
        if user is None:
            user = collect_user_input_from_console()

        result = run_pipeline(
            image_path=user["image_path"],
            user=user,
            use_llm=not args.no_llm,
            save_annotated=not args.no_save,
            save_db=not (args.no_save or args.no_db),
            model_overrides={
                "model": args.model,
                "host": args.host,
                "food_model": args.food_model,
                "plate_model": args.plate_model,
            },
            verbose=args.verbose,
            use_sam=bool(args.sam or get("segmentation.enabled", False)),
        )
        return 0 if result["dishes"] else 2
    except KeyboardInterrupt:
        print("\n已取消。")
        return 130
    except FileNotFoundError as exc:
        print(f"\n[文件错误] {exc}")
        return 1
    except ValueError as exc:
        print(f"\n[输入错误] {exc}")
        return 1
    except RuntimeError as exc:
        print(f"\n[运行错误] {exc}")
        return 1
    except ModuleNotFoundError as exc:
        print(f"\n[依赖缺失] {exc}\n请先执行：pip install -r requirements.txt")
        return 1
    except Exception as exc:  # 兜底，避免堆栈直接把用户吓走
        log.exception("未预期的错误")
        print(f"\n[未预期错误] {exc.__class__.__name__}: {exc}")
        print("详细堆栈见 logs/app.log")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
