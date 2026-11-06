# 膳食识别系统

> 食物识别 + 重量估计 + Ollama 健康膳食建议。
> 思路源自 [DidntRead/food_calories](https://github.com/DidntRead/food_calories)，重构为可维护的工程：
> **YOLOv11 检测（盘子 OBB + 食物 bbox）→ 盘子/餐具参考尺度 → 面积估重 → 密度/热量换算 → BMR/TDEE → Ollama 流式建议**。

两个入口：

| 入口 | 命令 | 适用 |
|---|---|---|
| **图形界面（上位机）** | 双击 `启动上位机.bat` 或 `py -3.10 上位机.py` | 演示、日常使用 |
| 命令行 | `py -3.10 main.py` | 批量跑图、单图快速验证 |

图界面包含「餐次配额 + 结构评估」「该加什么」「米饭三级处理」「食物高度档位」等逻辑，
详见 **[docs/上位机说明.md](docs/上位机说明.md)**。

---

## 1. 快速开始

```bash
# 依赖已在本机 Python 3.10 环境装好，重装命令：
py -3.10 -m pip install -r requirements.txt

# 交互式运行
py -3.10 main.py

# 一次性传参（脚本/测试用）
py -3.10 main.py --image data/images/test.jpg --age 25 --gender 男 --height 175 --weight 70 --activity 1

# 加 SAM 分割精修（重量更准，CPU 约 +55 秒/图）
py -3.10 main.py --image data/images/test.jpg --age 25 --gender 男 --height 175 --weight 70 --sam
```

实际运行输出（`data/images/test.jpg`）：

```
================================
  膳食识别系统（命令行版）
================================

请输入食物图片路径: data/images/test.jpg
请输入年龄: 25
请输入性别(男/女): 男
请输入身高(cm): 175
请输入体重(kg): 70
请输入活动量(1-4, 1=轻度 2=中度 3=重度 4=极重): 1

正在识别食物...
  · 图片已加载：204×246 px
  · 盘子已定位：置信度 0.88，直径约 201 px
  · 尺度：8.04 px/cm（来源：盘子）→ 盘子直径约 25.0 cm
  · 食物检测区域：整图（盘子只用于换算尺度）
  · 已过滤非食物/未收录类别：dining table, person
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
检测到以下食物：
序号  | 食物                | 置信度    | 重量        | 热量
------------------------------------------------------------------
1     | orange（橙子）      | 0.54      | ~204 g      | 96 kcal
2     | carrot（胡萝卜）    | 0.89      | ~153 g      | 63 kcal
------------------------------------------------------------------
总重量: 357 g
总热量: 159 kcal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
用户健康数据：
  BMR（基础代谢）: 1674 kcal/天
  TDEE（每日消耗）: 2302 kcal/天  （活动量 1 轻度）
  本餐热量占比: 6.9% TDEE
  BMI: 22.9（正常）
  估算三大营养素: 蛋白质 6 g / 脂肪 8 g / 碳水 17 g
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
正在调用 Ollama 生成膳食建议...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 一、本餐营养评估
（流式 Markdown 建议……）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 2. 环境说明（本机实测）

| 项 | 值 |
|---|---|
| 解释器 | `C:\Users\hua\AppData\Local\Programs\Python\Python310\python.exe`（3.10.11） |
| 关键依赖 | ultralytics 8.3.158 / torch 2.1.0+cpu / torchvision 0.16.0 / opencv-python 4.11 / numpy 1.26.4 / segment-anything 1.0 / loguru |
| 设备 | **CPU**（无可用 NVIDIA GPU，`config.yaml: detection.device = cpu`） |
| Ollama | `http://localhost:11434`，已装 `qwen2.5:1.5b`（默认） |
| SAM | `D:\Work\膳食模型\food_calories\sam_vit_h_4b8939.pth`（2.4GB，可选） |

> 直接敲 `python` 可能指向别的解释器（无依赖），建议统一用 `py -3.10 main.py`。

---

## 3. ⚠ 关于食物模型的重要说明

原始仓库里的 `yolo11m.pt` **不是** 9 类食物模型，而是 **COCO-80 官方预训练权重**（`person/car/dog/pizza…` 80 类）。
仓库里的 `yolo_food.pt` 虽是 498 类，但类别名是纯数字（无名称映射），无法映射营养数据。

所以本系统做了两件事：

1. **营养表白名单过滤**（`detection.filter_unknown_food: true`）
   只保留 `data/food_reference.yaml` 里存在的类别，`person / chair / dining table` 等自动丢弃并提示。
   当前 COCO 能命中的食物共 10 类：`apple / banana / orange / broccoli / carrot / sandwich / hot_dog / pizza / donut / cake`。
2. **换模型零改代码**
   训练好中餐模型后覆盖 `models/weights/yolo11m.pt`，把类别 key 追加到 `food_reference.yaml` /
   `density_table.yaml`，白名单与营养查表自动生效。

---

## 4. 目录结构

```
膳食模型x/
├── 上位机.py                   # 图形界面入口（PySide6）
├── 启动上位机.bat              # 双击启动
├── main.py                     # 命令行入口：交互输入 → 调各模块 → 打印 + 流式建议
├── requirements.txt
├── config.yaml                 # 全局配置（含 rice_rule / height_levels / ollama）
├── README.md
├── app/                        # 上位机
│   ├── main_window.py          # 主窗口：三页签、拖拽上传、三线程
│   ├── analyzer.py             # 单图分析：检测 → 估重 → 营养 → 配额/结构
│   ├── detection_rules.py      # 米饭三级处理 + 食物高度档位
│   ├── diet_rules.py           # 餐次配额(B) + 结构评估(C) + 该加什么
│   ├── widgets.py              # 结构评估面板
│   ├── prompt.py               # 膳食建议 prompt（结论写死，模型只组织语言）
│   ├── render.py               # PIL 中文标签绘制
│   ├── user_store.py           # 用户档案（data/users.json）
│   └── theme.py                # QSS
├── models/
│   ├── cbam.py                 # CBAM 注意力模块
│   ├── wiou.py / wiou_patch.py # WIoU v1/v2/v3 损失
│   ├── yolo11s-cbam-neck.yaml  # 方案C：Neck 端 P3/P4 后加 CBAM
│   └── weights/
│       ├── yolo11m-obb-plate.pt  # 盘子+餐具 OBB（4 类，41MB）
│       └── chinese_food.pt       # 18 类中餐检测模型（已部署基线）
├── data/
│   ├── food_reference.yaml     # 营养参考（=检测白名单）：能量/三大营养素/纤维/钠/微量元素/烹调方式
│   ├── dri_cn.yaml             # 中国居民 DRIs 2023 + 膳食指南 2022（餐次占比、结构阈值）
│   ├── density_table.yaml      # 密度表（g/cm³）
│   ├── reference_sizes.yaml    # 参考物真实尺寸（叉/刀/勺/盘，cm）
│   ├── users.json              # 用户档案
│   ├── images/                 # 示例图
│   ├── output/ reports/        # 标注图 / 导出报告
│   └── history.db              # SQLite 历史（首次运行自动创建）
├── src/
│   ├── detection/              # plate_detector / scale_calculator / food_detector / sam_segmenter
│   ├── weight_estimation/      # portion_estimator / nutrition_calculator
│   ├── health/                 # bmr_calculator（BMR/TDEE/BMI）
│   ├── llm/                    # ollama_client（流式+降级）/ prompt_builder
│   ├── database/               # db_manager（SQLite 历史 + 当日汇总 + CSV 导出）
│   └── utils/                  # config_loader / logger / image_utils / food_labels
├── configs/                    # 数据集 yaml（chinese_food_18.yaml 等）
├── tools/                      # 训练 / 评估 / 数据清洗 / GUI 冒烟测试
├── tests/                      # detection / weight / health / llm / diet_rules / e2e
├── docs/
│   ├── 上位机说明.md            # 餐次配额、结构评估、米饭三级、营养数据来源
│   └── 模型改进详解.md          # CBAM + WIoU：位置、原因、启用清单、消融实验
├── logs/                       # loguru 日志（自动轮转）
└── _upstream/food_calories/    # 原始项目副本（只读参考，已剔除 .git 与 .pt）
```

---

## 5. 核心算法

### 5.1 尺度（像素 → 厘米）
OBB 检测叉/刀/勺/盘，`OBB 最长边(px) ÷ 真实尺寸(cm)`，多参考物取**中位数**：

```
px_per_cm = median([fork/18.5, knife/21.0, spoon/17.0, plate/25.0])
```
优先用餐具（尺寸标准），只有盘子时按 25cm 标准盘；完全没有参考物时退化为「画面宽 40cm」并告警。

### 5.2 重量（bbox 面积 → 克）
```
整图 → YOLO 食物 bbox（盘子 OBB 只用来定尺度）
面积(cm²) = bbox 像素面积 / px_per_cm² × area_fill_factor(0.80)
体积(cm³) = 面积 × 高度（默认 3cm，可按食物覆写）
重量(g)   = 体积 × 密度
```
> 曾试过「盘子 OBB 外扩 10% → 旋转裁剪摆正 → 再检测」，实测明显更差：
> 25 张 test 图框级召回 51.4%（全图 77.0%），精确 60.3%（全图 82.6%）。
> 旋转后的裁剪图会让模型给出更大的框，盘边食物还会被切掉；训练与评估都是全图。
> 该路径保留在 `plate.crop_for_detection`，默认 `false`。
三重保护，避免「一份菜 1kg」这种离谱值：

| 约束 | 配置 | 作用 |
|---|---|---|
| bbox 填充率 | `area_fill_factor: 0.80` | bbox 是外接矩形，食物填不满 |
| 单份上限 | `max_weight_by_food`（cake 300g / pizza 600g …） | 抑制误检大框 |
| 盘子面积约束 | `max_plate_fill_ratio: 0.85` | 所有食物面积之和 ≤ 盘子面积×0.85，超出按比例收缩 |

同类多框自动合并（一块鸡胸 + 一块鸡腿 → 一个 chicken）。

### 5.3 SAM 分割精修（可选，`--sam`）
对每个食物 bbox 用 SAM 做框提示分割，用**掩膜像素数**代替 bbox 面积（`mask_fill_factor=0.95`）。
实测同一张图：bbox 方案 204g+153g → SAM 方案 119g+79g（掩膜剔除了方框里的空隙，更接近真实）。

CPU 上每张图约 **55-80 秒**（`vit_h` 编码 1024×1024 是瓶颈），所以默认关闭。

### 5.4 热量与健康
```
热量(kcal) = 重量 × kcal_per_100g / 100
BMR  = 10×体重 + 6.25×身高 - 5×年龄 + (男 +5 / 女 -161)   # Mifflin-St Jeor
TDEE = BMR × 活动系数(1.375 / 1.55 / 1.725 / 1.9)
本餐占比 = 热量 / TDEE
```
示例（男 25 岁 175cm 70kg，活动量 1）：BMR 1673.8、TDEE 2301.5。

### 5.5 LLM 建议
`prompt_builder` 组装「个人信息 + BMR/TDEE/BMI + 食物清单 + 总重量热量 + 三大营养素 + TDEE 占比」，
要求输出 4 节 Markdown（营养评估 / 需求匹配 / 搭配建议 / 个性化建议），`ollama_client` 走
`POST /api/generate` **流式**打印。

**降级兜底**：服务未启动 → 打印排查步骤；模型未拉取 → 提示 `ollama pull`；中途断流 → 保留已输出内容。
检测与营养结果始终可用。

---

## 6. 配置说明（config.yaml）

| 配置项 | 默认 | 说明 |
|---|---|---|
| `detection.device` | `cpu` | 有 GPU 改 `cuda:0`（可同时 `half: true`） |
| `detection.conf_threshold` | `0.25` | 食物置信度；误检多就抬到 0.35-0.45 |
| `detection.filter_unknown_food` | `true` | 营养表白名单过滤（COCO 必需） |
| `plate.expand_ratio` | `1.10` | 盘子 OBB 外扩，圈住盘边食物 |
| `plate.crop_for_detection` | `false` | 是否先把盘子摆正裁出来再检测食物；实测裁剪会明显掉点，默认全图 |
| `segmentation.enabled` | `false` | 是否默认启用 SAM（也可 `--sam` 临时开） |
| `segmentation.sam_checkpoint` | `../food_calories/sam_vit_h_4b8939.pth` | 2.4GB，相对项目根 |
| `weight_estimation.default_height_cm` | `3.0` | 食物堆叠高度（汤菜 2.0，馒头/蛋糕 5-6） |
| `weight_estimation.area_fill_factor` | `0.80` | bbox 填充率；估重偏高就降到 0.65 |
| `weight_estimation.height_by_food` | 见文件 | 按食物覆写高度 |
| `weight_estimation.max_weight_by_food` | 见文件 | 单份重量上限 |
| `weight_estimation.max_plate_fill_ratio` | `0.85` | 食物面积/盘子面积上限 |
| `ollama.model` | `qwen2.5:1.5b` | 换 `qwen2.5:7b` 建议质量明显更好 |
| `ollama.timeout` | `180` | CPU 推理 1.5b 约 10-40s，7b 更慢 |
| `database.enabled` | `true` | 是否写 SQLite 历史 |

---

## 7. 命令行参数

```
--image 路径/URL   --age 25   --gender 男   --height 175   --weight 70   --activity 1
--sam               启用 SAM 分割精修（更准、更慢）
--no-llm            跳过 Ollama，只输出检测与营养
--no-save           不保存标注图
--no-db             不写 SQLite 历史
--model / --host    覆盖 Ollama 模型/地址
--food-model / --plate-model   覆盖模型路径（试新权重用）
--verbose           输出 DEBUG 日志
```

返回码：`0` 成功、`2` 未识别到食物、`1` 参数/运行错误、`130` 手动中断。

---

## 8. 测试

```bash
py -3.10 tests/test_detection.py   # 盘子/尺度/食物检测/旋转裁剪/SAM checkpoint
py -3.10 tests/test_weight.py      # 密度/重量合理性/上限/盘子约束/尺度平方律/热量
py -3.10 tests/test_health.py      # BMR/TDEE/BMI/性别别名/非法输入
py -3.10 tests/test_llm.py         # Prompt 完整性/在线流式/离线降级/缺失模型
py -3.10 tests/test_e2e.py         # 端到端（含 LLM）
```
用例名均为 `test_*`，也可直接用 pytest 跑。

---

## 9. 已知限制

1. **食物模型是 COCO-80**，只能识别 10 类常见食物；中餐菜品基本识别不了，常被误判成 `cake/bowl`
   （这正是单份上限与盘子面积约束存在的原因）。
2. **重量是粗估**：bbox 面积 + 固定高度 + 密度表，误差通常 ±30%~50%；汤汁、堆叠类菜品偏差更大，
   只适合「量级判断」，不适合精确记账。开 `--sam` 可明显改善（掩膜更贴轮廓）。
3. **遮挡**：一盘多菜互相压盖时，被压住的菜面积会被高估（bbox 重叠）。
4. **拍摄角度**：非俯拍（45° 斜拍）时盘子 OBB 变梯形，尺度与面积都会偏低，建议俯拍。
5. **CPU 推理**：检测 1-4s；SAM +55-80s；Ollama 1.5b 生成 10-40s。
6. **小模型建议质量有限**：`qwen2.5:1.5b` 偶有事实性错误（把 BMR 当热量上限之类），
   建议 `ollama pull qwen2.5:7b` 后改 `config.yaml: ollama.model`。

---

## 10. 后续升级路线

| 目标 | 做法 | 代码改动 |
|---|---|---|
| 换中餐模型 | 训练 `chinese_food.pt` 覆盖 `models/weights/yolo11m.pt`，类别 key 追加到两个 yaml | **0 行** |
| 提升检测精度 | 启用 CBAM + WIoU 训练，见 `docs/模型改进详解.md` | 仅训练侧 |
| 更准重量 | 默认开 `segmentation.enabled: true`（SAM 掩膜） | 改配置 |
| 历史管理 | `db_manager` 已支持保存/查询/导出 CSV，可加 `--history` 命令 | 约 10 行 |
| 换大模型 | `ollama pull qwen2.5:7b` + 改配置 | 0 行 |

---

## 11. 常见问题

**Q：`ModuleNotFoundError: No module named 'ultralytics'`**
A：用错解释器了。装依赖和运行都统一用 `py -3.10`。

**Q：中文路径图片读不进来？**
A：已处理（`np.fromfile + cv2.imdecode`），支持 `.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`。

**Q：没检测到盘子？**
A：照片最好带完整盘子或刀叉勺；否则退到整图检测 + 画面宽度假设，误差变大。

**Q：识别不出中餐？**
A：当前模型是 COCO-80，中餐不在类别里，属预期行为，需要自训练中餐模型。

**Q：Ollama 说没有模型？**
A：`ollama pull qwen2.5:1.5b`，或把 `config.yaml` 的 `ollama.model` 改成 `ollama list` 里已有的。

**Q：只想看检测结果？**
A：加 `--no-llm`。
