"""食物类别中文名映射（仅用于界面展示，不影响计算逻辑）。

换成中餐模型后，只要类别 key 与这里/或 yaml 中一致，就会自动显示中文名；
没收录的类别直接显示模型原始类别名，不会报错。
"""

from __future__ import annotations

FOOD_LABELS_ZH = {
    # ---- 中餐食堂 12 类（主模型 chinese_food.pt 的类别）----
    "carrot_eggs": "胡萝卜炒蛋",
    "chicken_nuggets": "鸡米花",
    "chinese_cabbage": "炒白菜",
    "chinese_sausage": "腊肠",
    "curry": "咖喱",
    "fried_chicken": "炸鸡排",
    "fried_dumplings": "煎饺",
    "fried_eggs": "炒蛋",
    "mung_bean_sprouts": "炒豆芽",
    "water_spinach": "炒空心菜",
    "triangle_hash_brown": "三角薯饼",
    # ---- 中餐扩充类中保留的 6 类（s Workspace v20 数据，广东食堂常见）----
    # 红烧肉 / 卤蛋 已从数据集中剔除（互混严重），不再保留映射
    "fried_chicken_legs": "炸鸡腿",
    "fried_noodles": "炒面",
    "fried_string_beans": "炒豆角",
    "loofah": "丝瓜",
    "pumpkin": "蒸南瓜",
    "sweet_potato": "蒸红薯",
    # ---- COCO 里能命中的食物（旧模型 yolo11m.pt 用）----
    "apple": "苹果",
    "banana": "香蕉",
    "broccoli": "西兰花",
    "orange": "橙子",
    "carrot": "胡萝卜",
    "sandwich": "三明治",
    "hot_dog": "热狗",
    "pizza": "披萨",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    # ---- 原仓库 9 类中的其余 5 类 ----
    "chicken": "鸡肉",
    "rice": "米饭",
    "pasta": "意面",
    "egg": "鸡蛋",
    "cheese": "奶酪",
    # ---- 中餐预留 ----
    "rice_cooked": "米饭",
    "noodles": "面条",
    "mantou": "馒头",
    "hongshaorou": "红烧肉",
    "tangculiji": "糖醋里脊",
    "gongbaojiding": "宫保鸡丁",
    "huiguorou": "回锅肉",
    "yuxiangrousi": "鱼香肉丝",
    "qingshaoshucai": "清炒时蔬",
    "disanxian": "地三鲜",
    "mapodoufu": "麻婆豆腐",
    "ganbiandoujiao": "干煸豆角",
    "fanqiedantang": "番茄蛋汤",
    "chaodan": "炒蛋",
    "liangbancai": "凉拌菜",
}

# 餐具/盘子
UTENSIL_LABELS_ZH = {
    "Fork": "叉子",
    "Knife": "刀",
    "Plate": "盘子",
    "Spoon": "勺子",
}


def zh_name(name: str) -> str:
    """返回中文名；没有映射时返回原名。"""
    key = str(name).strip()
    if key in FOOD_LABELS_ZH:
        return FOOD_LABELS_ZH[key]
    norm = key.lower().replace(" ", "_").replace("-", "_")
    if norm in FOOD_LABELS_ZH:
        return FOOD_LABELS_ZH[norm]
    for k, v in FOOD_LABELS_ZH.items():
        if k.replace("_", "") == norm.replace("_", ""):
            return v
    return key


def label_with_zh(name: str) -> str:
    """'chicken' -> 'chicken（鸡肉）'，无映射时只返回原名。"""
    zh = zh_name(name)
    return f"{name}（{zh}）" if zh != name else str(name)


def zh_utensil(name: str) -> str:
    return UTENSIL_LABELS_ZH.get(str(name), str(name))


__all__ = ["FOOD_LABELS_ZH", "UTENSIL_LABELS_ZH", "zh_name", "label_with_zh", "zh_utensil"]
