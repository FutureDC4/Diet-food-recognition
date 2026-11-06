# -*- coding: utf-8 -*-
"""数据集类别审计：统计每类实例数、检测重名/近重复、给出是否值得纳入训练的建议。

用法：
    python tools/class_audit.py "D:/Work/datasets/s_workspace"
    python tools/class_audit.py "D:/Work/datasets/s_workspace" --csv 类别审计.csv
    # 指定只关心某几个候选类（其余不逐个展开）
    python tools/class_audit.py "D:/Work/datasets/s_workspace" --focus pumpkin loofah "Braised meat"

设计要点：
1. 实例数比图片数更有意义 —— 一类可能有 40 张图 300 个框，也可能 1 张图 50 个框。
2. 重名检测做「归一化后完全相等」，近重复检测做「归一化后相似度高」，
   专门用于抓 Fchicken legs / chicken leg、Fried green beans / fried string Beans 这种同菜异名。
3. 只读，不改任何文件。
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 现有 Chinese-Cafeteria-Food-Recognition v3 的 12 类（归一化后）
EXISTING_12 = [
    "carrot_eggs",
    "chicken_nuggets",
    "chinese_cabbage",
    "chinese_sausage",
    "curry",
    "fried_chicken",
    "fried_dumplings",
    "fried_eggs",
    "mung_bean_sprouts",
    "rice",
    "triangle_hash_brown",
    "water_spinach",
]

# 阈值：实例数低于此值不建议纳入
THRESH_GOOD = 150
THRESH_OK = 80

# 已知语义同义词组：组内任意两个视为同一道菜，不能同时作为两个类
SYNONYM_GROUPS = [
    {"fried green beans", "fried string beans", "fried green bean", "stir fried green beans"},
    {"fried noodles", "fried rice noodles", "fried noodle"},
    {"chicken leg", "fchicken legs", "fried chicken legs", "chicken legs"},
    {"fried rice", "rice"},
    {"scrambled eggs", "scrambled egg", "fried eggs", "steamed eggs in three color"},
    {"chinese cabbage", "friend chinese bok-choy", "bok choy", "bok-choy"},
    {"braised meat", "braised pork", "braised pork belly"},
    {"sweet and sour pork ribs", "fried ribs"},
    {"cucumber", "cold cucumber", "stir-fried cucumber balls"},
    {"eggplant", "fried eggplant", "eggplant bean bread"},
]


def norm(name: str) -> str:
    """类名归一化：小写、去空格/下划线/连字符、合并复数 s。"""
    s = str(name).strip().lower()
    s = re.sub(r"[\s_\-]+", "", s)
    s = re.sub(r"(?<=[a-z])s$", "", s)  # 去尾 s（legs -> leg）
    return s


def read_names(dataset: Path, yaml_file: Optional[str] = None) -> Dict[int, str]:
    """从 data.yaml 读 names；支持 list 和 dict 两种写法。"""
    try:
        import yaml
    except ImportError:
        sys.exit("缺少 pyyaml：pip install pyyaml")

    cands = [Path(yaml_file)] if yaml_file else [dataset / "data.yaml", dataset / "dataset.yaml", dataset / "data.yml"]
    for p in cands:
        if Path(p).exists():
            cfg = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
            names = cfg.get("names")
            if isinstance(names, list):
                return {i: str(v) for i, v in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            sys.exit(f"{p} 里找不到 names 字段")
    sys.exit(f"在 {dataset} 下找不到 data.yaml / dataset.yaml")


def find_label_dirs(dataset: Path) -> List[Path]:
    """找出所有含 .txt 的 labels 目录（兼容 train/valid/test 和扁平结构）。"""
    dirs: List[Path] = []
    for p in dataset.rglob("labels"):
        if p.is_dir() and any(p.glob("*.txt")):
            dirs.append(p)
    if not dirs and any(dataset.glob("*.txt")):
        dirs.append(dataset)
    return sorted(dirs)


def count_instances(label_dirs: List[Path], n_classes: int) -> Tuple[Counter, Counter, int]:
    """返回 (实例数 Counter, 图片数 Counter, 异常行数)。"""
    inst: Counter = Counter()
    imgs: Counter = Counter()
    bad = 0
    for d in label_dirs:
        for txt in d.glob("*.txt"):
            seen: set[int] = set()
            for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    bad += 1
                    continue
                try:
                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    bad += 1
                    continue
                if not (0 <= cid < n_classes):
                    bad += 1
                    continue
                if any(v < 0 or v > 1 for v in coords):
                    bad += 1
                    continue
                if coords[2] <= 0 or coords[3] <= 0:
                    bad += 1
                    continue
                inst[cid] += 1
                seen.add(cid)
            for cid in seen:
                imgs[cid] += 1
    return inst, imgs, bad


def find_duplicates(names: Dict[int, str]) -> List[Tuple[int, int, str, float]]:
    """找出源数据内部近重复的类对，以及同义词组内冲突。"""
    by_norm: Dict[str, List[int]] = defaultdict(list)
    for cid, n in names.items():
        by_norm[norm(n)].append(cid)

    hits: List[Tuple[int, int, str, float]] = []
    ids = sorted(names)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            na, nb = norm(names[a]), norm(names[b])
            if na == nb:
                hits.append((a, b, "完全同名（归一化后）", 1.0))
                continue
            ratio = difflib.SequenceMatcher(None, na, nb).ratio()
            # 同义词组判断：组内任意两项互为同菜
            same_group = False
            for g in SYNONYM_GROUPS:
                gn = {norm(x) for x in g}
                if na in gn and nb in gn:
                    same_group = True
                    break
            # 包含关系也算（如 "fried noodle" 与 "fried rice noodle"）
            contain = (na in nb or nb in na) and min(len(na), len(nb)) >= 6
            if same_group or contain or ratio >= 0.86:
                reason = "同义词组" if same_group else ("名称包含" if contain else "名称高度相似")
                hits.append((a, b, reason, round(ratio, 2)))
    return sorted(hits, key=lambda x: -x[3])


def overlap_with_existing(names: Dict[int, str]) -> Dict[int, str]:
    """源数据的类是否与现有 12 类重名（归一化后）。"""
    ex = {norm(x) for x in EXISTING_12}
    return {cid: n for cid, n in names.items() if norm(n) in ex}


def verdict(inst: int) -> str:
    if inst >= THRESH_GOOD:
        return "可纳入"
    if inst >= THRESH_OK:
        return "勉强（会成为弱项）"
    return "太少，建议放弃或自己补拍"


def main() -> int:
    ap = argparse.ArgumentParser(description="数据集类别审计")
    ap.add_argument("dataset", help="数据集根目录（含 data.yaml）")
    ap.add_argument("--csv", default=None, help="把明细导出为 CSV")
    ap.add_argument("--focus", nargs="*", default=None, help="只详细展开这些类名（模糊匹配）")
    ap.add_argument("--yaml", default=None, help="类别定义文件（默认 <dataset>/data.yaml）")
    args = ap.parse_args()

    root = Path(args.dataset)
    if not root.is_dir():
        sys.exit(f"目录不存在：{root}")

    names = read_names(root, args.yaml)
    n = max(names) + 1
    label_dirs = find_label_dirs(root)
    if not label_dirs:
        sys.exit(f"{root} 下找不到任何 labels 目录")

    print("=" * 84)
    print(f"数据集：{root}")
    print(f"类别数：{len(names)}   labels 目录：{[str(d.relative_to(root)) for d in label_dirs]}")
    print("=" * 84)

    inst, imgs, bad = count_instances(label_dirs, n)
    total_inst = sum(inst.values())
    print(f"标注框总数：{total_inst}   异常行：{bad}\n")

    # ---- 1. 全类别按实例数排序 ----
    print("【一】各类实例数（降序）")
    print(f"{'ID':>3}  {'类名':38s} {'实例':>6} {'图片':>6} {'评估'}")
    print("-" * 84)
    for cid in sorted(names, key=lambda c: -inst.get(c, 0)):
        if args.focus and not any(norm(f) in norm(names[cid]) or norm(names[cid]) in norm(f) for f in args.focus):
            continue
        print(
            f"{cid:>3}  {names[cid][:38]:38s} {inst.get(cid, 0):>6} {imgs.get(cid, 0):>6} "
            f"{verdict(inst.get(cid, 0))}"
        )
    if args.focus:
        print("（只显示 --focus 命中的类）")

    # ---- 2. 与现有 12 类重叠 ----
    ov = overlap_with_existing(names)
    print("\n【二】与现有 12 类重名（这些不要重复加入）")
    if ov:
        for cid, nm in ov.items():
            print(f"  ID {cid:>2}  {nm}  → 已有，跳过（实例 {inst.get(cid, 0)}）")
    else:
        print("  无重名")

    # ---- 3. 源数据内部近重复 ----
    dups = find_duplicates(names)
    print("\n【三】源数据内部近重复（合并前必须处理，否则同类互相打架）")
    if dups:
        shown = 0
        for a, b, reason, ratio in dups:
            # 只展示两个都还有实例的
            if inst.get(a, 0) == 0 and inst.get(b, 0) == 0:
                continue
            if shown >= 25:
                print(f"  ... 还有 {len(dups) - shown} 组未显示")
                break
            print(
                f"  {names[a]!r:34s} <-> {names[b]!r:34s} [{reason} {ratio}] "
                f"({inst.get(a,0)} / {inst.get(b,0)} 框)"
            )
            shown += 1
    else:
        print("  未发现近重复")

    # ---- 4. 导出 CSV ----
    if args.csv:
        out = Path(args.csv)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "name", "instances", "images", "verdict", "in_existing_12"])
            ex = {norm(x) for x in EXISTING_12}
            for cid in sorted(names, key=lambda c: -inst.get(c, 0)):
                w.writerow(
                    [
                        cid,
                        names[cid],
                        inst.get(cid, 0),
                        imgs.get(cid, 0),
                        verdict(inst.get(cid, 0)),
                        "yes" if norm(names[cid]) in ex else "",
                    ]
                )
        print(f"\n明细已导出：{out.resolve()}")

    print("\n判定标准：实例 ≥150 可纳入；80–150 会成为全表弱项；<80 建议放弃或自己补拍。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
