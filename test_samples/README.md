# 测试实例图片

本目录存放模型测试用的样例图片与对应识别结果，供复现和效果核对。

## 目录结构

```
test_samples/
├── input/     模型测试原图（输入）
└── results/   已测试的识别结果（输出，含检测框 + 类别 + 重量）
```

## input/ — 模型测试原图

| 文件 | 场景 | 说明 |
|---|---|---|
| `01_western_plate.jpg` | 西式餐盘 | 水煮蛋、胡萝卜、橙子、燕麦球，盘子边缘清晰（尺度参考测试） |
| `02_smiley_plate.jpg` | 创意摆盘 | 香肠、橙子、火腿、芝士条摆成笑脸，测试非典型摆盘 |
| `03_western_lunch.jpg` | 西式午餐 | 三明治、水果、蔬菜，**带叉子和牛奶杯**（餐具尺度参考测试） |
| `04_spoon_scale.jpg` | 尺度参考 | 勺子单独测试，验证餐具 → 像素/cm 换算 |
| `05_chinese_oval_plate.jpg` | **中餐椭圆盘** | 炸鸡排、绿豆芽、中式香肠、煎饺、米饭（中餐模型主测试图） |
| `06_chinese_bento.jpg` | **中餐便当盒** | 炸鱼排、青菜、土豆、黄瓜片、米饭（遮挡场景测试） |

## results/ — 已测试识别结果

结果图为模型推理后的可视化输出：绿色框 = 盘子 OBB，红色框 = 食物 bbox，标签格式为 `类别 置信度 估计重量`。

| 文件 | 对应原图 | 识别到的食物（类别 / 置信度 / 估计重量） |
|---|---|---|
| `01_western_plate_result.jpg` | `input/01_western_plate.jpg` | carrot 0.89 / 106g、orange 0.54 / 161g、plate 0.88 |
| `02_chinese_oval_plate_result.jpg` | `input/05_chinese_oval_plate.jpg` | fried_chicken 0.91 / 201g、mung_bean_sprouts 0.92 / 88g、fried_dumplings 0.80 / 190g、chinese_sausage 0.84 / 59g、rice 0.70 / 65g |
| `03_chinese_bento_result.jpg` | `input/06_chinese_bento.jpg` | 中餐便当盒多菜品（含遮挡场景） |

## 复现命令

```bash
# 单图推理（命令行）
py -3.10 main.py --image test_samples/input/05_chinese_oval_plate.jpg \
    --age 25 --gender 男 --height 175 --weight 70 --activity 1

# 加 SAM 分割精修（重量更准，CPU 约 +55 秒/图）
py -3.10 main.py --image test_samples/input/05_chinese_oval_plate.jpg \
    --age 25 --gender 男 --height 175 --weight 70 --activity 1 --sam
```

> 说明：`01`、`02`、`03` 使用西餐 9 类模型（`yolo11m.pt`）测试尺度换算与重量估计流程；
> `05`、`06` 使用自训练中餐模型（`chinese_food.pt`）测试类别识别效果。
> 模型权重体积较大未纳入仓库，获取方式见根目录 README。

## 结果说明

- **重量估计为近似值**：采用「bbox 面积 × 形状填充系数 × 假设高度 × 密度」方案，
  误差约 15–25%，用于营养估算而非精确计量。
- **米饭遮挡处理**：中餐场景米饭常被菜品遮盖，代码内置三级处理规则
  （正常检测 → 面积过小按总食物面积 1/3 估算 → 未检测到则不处理），详见 `docs/上位机说明.md`。
- **框偏大的影响**：bbox 对圆形食物天然偏大约 27%（π/4），已通过填充系数补偿。
