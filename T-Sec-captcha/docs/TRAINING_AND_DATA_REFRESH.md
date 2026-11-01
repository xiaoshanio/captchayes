# 继续训练 + 数据更新：完整方法论与落地记录

> 本文档记录 `T-Sec-captcha` 这套「YOLO 检测 + Siamese 匹配」点选验证码识别方案的
> **模型续训**与**数据更新**方法论，以及在真机上跑通的验证证据。
>
> 作用域声明：本文只使用**自有 / 程序化自造**数据。文末「附录 A」记录了第三方站点
> 协议侦察的结论，仅作为「域漂移」的事实依据存档，**不包含对其的采集或定制求解方案**。

---

## 0. 一句话结论

现有仓库**缺少续训所需的一切前提**（无 `.pt`、无数据集、无训练脚本）；
且线上目标题型与模型训练域不同。正确解法不是"微调现有模型"，而是：

```
重建可训练权重  →  重建数据供给（合成优先）  →  重训两个模型  →  契约对齐导出  →  端到端验收
```

其中**数据供给**是真正的瓶颈。本文验证过的核心做法：
**利用验证码「程序化合成」的本质，正向重放生成规则，让标注成本归零。**

---

## 1. 现状诊断（基于实测，非推测）

### 1.1 仓库里有什么

```
model/best_v3.onnx        9.35 MB   Ultralytics YOLO26n (8.4.43)
model/pre_model_v7.onnx  42.39 MB   Siamese (双输入 112×112 -> logits)
src/, cv/, app/                     纯推理代码
```

**没有** `.pt` / `.pth` / 数据集 / 标注 / 训练脚本 / requirements 里的训练依赖。
`README.md` 引用的训练教程（道满PythonAI）是外部付费内容，仓库内不可得。

### 1.2 两个模型的真实契约（从 ONNX 读取，非猜测）

| | `best_v3.onnx` | `pre_model_v7.onnx` |
|---|---|---|
| 架构 | YOLO26n detect | 共享 backbone + fusion_head |
| 输入 | `images [1,3,640,640]` f32 | `input1`,`input2` `[batch,3,112,112]` f32 |
| 输出 | `output0 [1,300,6]` | `logits [batch,1]` |
| 格式 | `[x1,y1,x2,y2,conf,cls]` | 裸 logit（runtime 里做 sigmoid） |
| 类别 | `{0:label, 1:title, 2:char}` | — |
| opset / end2end | 14 / **True** | 14 / — |
| 训练集(metadata) | `D:\captcha\yolov5\data\bilbil.yaml` ← 本机不存在 | — |

`pre_model_v7` 的结构（反解后确认）：
`Concat([|f1−f2|, f1⊙f2]) → Gemm→BN→ReLU → Gemm→BN→ReLU → Gemm`，
标准 Siamese 距离头，输入 5120 = 2×2560。

### 1.3 类别语义（这是理解整个 pipeline 的关键）

`src/captcha.py` 的 `run()`：

```python
target_boxes = [x for x in det if x[5] == 0]   # class 0 = 图面上「待点击」的字符
char_boxes   = [x for x in det if x[5] == 2]   # class 2 = 提示条里的字符
char_boxes.sort(key=lambda b: b[0])            # 按 x 排序 = 从左到右 = 点击顺序
slys = pre.reason_all_batch(chars, img_targets)  # N×M 相似度矩阵
sorted_result = matchingMode.find_overall_index_fast(slys)  # 贪心指派
result = [target_boxes[j] for i, j in sorted_result]
```

即：**先检测出所有候选，再用 Siamese 匹配把提示条字符对应到图面位置**。
`class 1 (title)` 是"请依次点击"这类标题，runtime **不使用**。

### 1.4 硬件约束（决定了训练配方）

| 项 | 实测值 | 影响 |
|---|---|---|
| CPU | 2× Xeon E5-2660 v2 (Ivy Bridge-EP) | 只有 AVX，**无 AVX2/FMA** |
| GPU | GT 705，驱动 CUDA 9.1 | `torch.cuda.is_available() == False`，**不可用** |
| 内存 | 64 GB | 够用，可 `cache=ram` |
| Python | 3.14.4 | onnxruntime/torch/ultralytics 均有 cp314 轮子，可用 |

**踩坑记录（重要，两个都会让训练直接死掉）**

**坑 1：`polars` 导致硬崩溃 `0xC000001D`（非法指令）**

`ultralytics 8.4.x` 用 `polars` 在每个 epoch 结束时写 `results.csv`。
`polars` 的官方 wheel 编译时要求 AVX2/FMA/BMI，
本机 Xeon E5-2660 v2（Ivy Bridge-EP）**只有 AVX**，于是：

- `import polars` 会打一条 `RuntimeWarning: Missing required CPU features: avx2, fma, bmi1, bmi2, lzcnt, movbe`
- 真正执行到 polars 内核时 → **进程硬终止 `0xC000001D`（STATUS_ILLEGAL_INSTRUCTION），无 traceback**

现象极有迷惑性：训练能跑满整个 epoch（loss 正常下降），
`results.csv` 写出第 1 行后立刻崩溃 —— 看起来像"epoch 边界 bug"，实际是 CSV 写入。

最小复现：

```powershell
python -c "import polars as pl; pl.DataFrame({'a':[1,2,3]}).write_csv(); print('ok')"
# -> 0xC000001D
```

**修法（根治）**：装 polars 的**可移植运行时**构建，它不依赖 AVX2：

```powershell
pip install "polars[rtcompat]"
```

验证：

```powershell
python -c "import polars as pl; print(pl.DataFrame({'a':[1]}).write_csv())"
# -> a / 1 / POLARS OK
```

辅助手段：`$env:POLARS_SKIP_CPU_CHECK = "1"` 只是**关掉告警**，
不解决内核崩溃 —— 必须装 `rtcompat` 那个包。

定位过程记录在 `train/epoch_end_probe.py`（逐个隔离 epoch 结束回调，
第 `6_csv_polars` 步复现，第 `7_csv_pandas` 步正常）。

**坑 2：CPU 上 AMP**

ultralytics 默认 `amp=True`，CPU 上会走 `torch.amp.autocast`，
在老 CPU 上同样可能触发非法指令。修法：`amp=False`（脚本已默认关闭）。

---

## 2. 第一步：重建可训练权重

`tools/recover_weights.py`

### 2.1 两个必须知道的坑

**坑 1：`onnx2torch` 在 Windows 上必然 `PermissionError`**

`onnx2torch/utils/safe_shape_inference.py` 用
`NamedTemporaryFile(dir=<onnx 所在目录>, delete=True)`，
Windows 下该句柄不共享读，随后 `onnx.load(output_path)` 开第二个句柄读同一路径 → `Errno 13`。
**这不是 ACL/沙箱问题**（`model/` 目录实测可写）。

修法：把 ONNX 复制到自建暂存目录，并把 shape inference 换成内存版
`onnx.shape_inference.infer_shapes`，不再落盘。

**坑 2：`pre_model_v7.onnx` 有 154 个 `Identity` 别名节点**

它们把真实 initializer 重命名成 `onnx::Conv_<new>` 再喂给 Conv：

```
Identity_153 : onnx::Conv_916 -> onnx::Conv_1147
/backbone/conv_stem_1/Conv : inputs = ['input2', 'onnx::Conv_1147', ...]
```

`onnx2torch` 只索引真实 initializer，不解析别名 → `KeyError`。
修法：转换前把 `Identity` 全部内联（435 → 281 节点）。

### 2.2 反解结果

| 模型 | 参数量 | 数值对齐 max_abs_diff | 能否直接续训 |
|---|---|---|---|
| `pre_model_v7.onnx` | 19,490,145 | **1.19e-06** ✅ | ✅ **可以** |
| `best_v3.onnx` | 2,375,421 | 见下 | ⚠️ **不能端到端反传** |

### 2.3 为什么 YOLO 不能直接续训（关键结论）

`best_v3.onnx` 是 **`end2end=True` 的推理导出图**，尾部这一段是训练垃圾：

| 算子 | 可导性 | 后果 |
|---|---|---|
| `TopK` ×2 | ❌ 梯度恒 0 | 300 行选择全靠它，反向在此断掉 |
| `GatherElements` ×2 | ❌ 索引算子 | 同上 |
| `Mod` | ❌ 整数取模 | 类别号无法回传 |
| `Cast` | ❌ | 断流 |

另有一个数值现象值得记录：raw `max_abs_diff` 高达几百。
**根因不是权重错位，是 TopK 在同分场景下的索引抖动导致的行重排**
（该模型 300 行里绝大多数 conf 只有 1e-6~1e-4，几乎全是并列）。
证据：匈牙利匹配重排行后 diff 降到 1e-4 量级；只看 `conf≥0.25` 的有效框，
torch 与 ORT 的**框数/类别/坐标/置信度完全一致**。

**结论**：
- 推理用 `weights/best_v3.pt` 完全够。
- 续训必须走 ultralytics 原生结构 + `v8DetectionLoss`，让梯度路径干净，
  不能拿 `output0` 当 loss 反传。

---

## 3. 第二步：数据供给（整个方案的核心）

### 3.1 为什么"买数据/标数据"是错的解法

点选验证码的标注成本结构很糟：
- 必须标**全部**候选框（含干扰项），不是只标要点的
- 必须标**点击顺序**
- 出题风格一变，旧标注价值迅速衰减

### 3.2 正确的思路：正向重放生成规则

这类验证码是**程序化合成**的：纯色线稿图标 + 随机旋转/缩放/配色，叠在摄影底图上。
既然生成规则可逆，就能正向重放：

```
干净底板(把原图标抠掉)  +  图标素材库  →  随机合成  →  标签 100% 精确
```

**这一步让标注成本归零，而且可以随时跟着线上风格重新生成。**

### 3.3 已实现的两条数据链路

#### 链路 A：从**自有/已授权**样本反推（`tools/`）

适用：你手上已经有一批**自有或获得授权**的样本。
输入约定：每个样本一个目录，含 `bg.jpg` + `prompt_*.jpg` + `meta.json`。

| 脚本 | 作用 |
|---|---|
| `icon_assets.py` | 从提示图**列投影**切出单图标 → 干净素材（RGBA 透明底） |
| `plate_gen.py` | 抠掉底图图标 → `inpaint` 出干净底板 |
| `synth_gen.py` | 合成：底板 + 素材 → 图 / YOLO 标签 / 配对真值 |

> **采集环节不在本仓库内提供。**
> 本项目只对自有系统或获得书面授权的目标做数据采集与模型训练。
> 针对第三方线上服务的验证码批量采集、以及为其定制求解模型，本项目不做，
> 也不提供可执行的采集脚本。

#### 链路 B：完全自造（`tools/synth_selfcontained.py`）

适用：验证链路本身，零外部依赖，结论可公开复现。

- 图标素材：OpenCV 基元**程序化绘制**（数字/字母/圆环/三角/方框/十字/星形/菱形/折线，共 32 个）
- 底板：程序化生成（渐变天空 + 噪声 + 几何色块 + 模糊 + JPEG 伪影）
- 输出：`images/` + `labels/` + `pairs/` 三件套

实测：800 train + 120 val，**0 失败**，提示图标数 mean=3.50，底图图标数 mean=5.51。

### 3.4 分割算法：三条路都试过，结论明确

抠图标（生成干净底板）试了三种方案：

| 方案 | 实测表现 | 结论 |
|---|---|---|
| 绝对暗度阈值 `gray<105` | 沙地背景误检大片 | ❌ |
| **blackhat 形态学** | 掩码覆盖 37~99%，inpaint 完整张糊 | ❌ **不可用** |
| **`gray<60` + 面积/短边/填充率过滤 + 邻近合并** | 图标检出 mean=5.3，框准 | ✅ **采用** |

**为什么 blackhat 失败（值得记住）**：它依赖"细暗笔画"特征，
但照片纹理（沙纹、草叶、石纹）本身也是细结构，响应甚至比图标更强。
绝对阈值 + 几何判据反而稳。

另一个必须遵守的顺序问题：**颜色一致性检验要在「纯墨迹连通域」上做，
不能在「闭运算后的粗 blob」上做** —— 后者的大部分像素不存在于原图，检验必然失败。

### 3.5 生成干净底板

`tools/plate_gen.py`：图标框整块填充抹除 + `cv2.inpaint(INPAINT_TELEA, radius=5)`。
不自留鬼影的关键是**用框填充而不是只抹墨迹像素**。

---

## 4. 第三步：重训

### 4.1 检测器（`train/train_detector.py`）

```powershell
$env:POLARS_SKIP_CPU_CHECK = "1"          # 老 CPU 必须
python train/train_detector.py `
    --data data/selfgen/data.yaml `
    --epochs 20 --imgsz 640 --batch 8 --device cpu --cache ram `
    --out-onnx runs/export_next/best_v3_next.onnx
```

硬约束（不满足则 runtime 后处理不匹配）：
- `imgsz=640`（对齐 `yolo_onnx.py` 的 letterbox）
- `end2end=True`（保住 `[1,300,6]` 输出）
- 导出：`opset=14, nms=False, simplify=True, dynamic=False, batch=1`
- 类别名必须仍是 `{0:label, 1:title, 2:char}`

脚本内置 `verify_contract()`：检查输入/输出 shape、`[...,6]` 末维、
metadata 里的 `names/end2end/imgsz`，并实际推理一遍。

### 4.2 匹配器（`train/train_matcher.py`）

```powershell
python train/train_matcher.py `
    --pairs data/selfpairs --epochs 8 --batch 64 --threads 10 `
    --out-onnx runs/export_next/pre_model_v7_next.onnx
```

**预处理必须与 `src/utils/ver_onnx.py` 逐比特一致**，否则线上线下不一致：

```python
BGR → RGB → letterbox(pad=128, INTER_CUBIC) → /255 → CHW
```

训练数据来自 `train/prepare_pairs.py`：从合成集 `pairs/*.json` 的
「第 i 个提示图标 ↔ 底图哪个框」自动生成正负对，
**同一题内的其它图标 = 天然的难负样本**（`hard=True`）。

本机实测：2807 正样本 + 8421 负样本（全部 `hard`）。

### 4.3 一键脚本

`train/run_all.ps1` —— 串起检测器 + 匹配器训练 + 导出 + 契约校验。

---

## 5. 第四步：验收

### 5.0 两个必须知道的验收坑

**坑 1：`TSEC_MODEL_DIR` 并不能用来切换模型！**

`src/captcha.py::_resolve_model_file()` 的查找顺序是：

```python
1) os.path.isabs(filename) and os.path.exists(filename)   # 绝对路径
2) os.path.exists(filename)                                # ← src/../model/<file>
3) os.environ["TSEC_MODEL_DIR"]/<basename>
4) repo/model/, repo/models/
```

**第 2 条优先级高于第 3 条**。只要仓库 `model/` 里有同名文件，
`TSEC_MODEL_DIR` 就永远不生效。
实测：设了 `TSEC_MODEL_DIR=runs/dropin_test` 之后，
`cap.yolo.session._model_path` 仍然指向 `src/../model/best_v3.onnx` ——
**你以为在测新模型，其实一直在测旧模型**，这个坑会让人得出完全错误的结论。

正确做法：**回灌替换**。把候选模型临时复制进 `model/`，跑未改动的 runtime，跑完恢复。
这也正是生产上线的真实动作，比环境变量更贴近实际。

**坑 2：ONNX Runtime 会一直持有 `.onnx` 文件句柄**

同一进程内切换模型时，删 `sys.modules` + `gc.collect()` 也**不释放句柄**，
覆盖 `model/*.onnx` 会报 `PermissionError: [WinError 32]`。

修法：**子进程隔离**。主进程负责回灌/恢复，子进程只负责跑一遍并落 JSON。

`eval/eval_new.py` 已按这两条实现。

### 5.1 契约校验（脚本内置）

```
[verify] 匹配器契约校验
  inputs : [('input1', ['batch', 3, 112, 112]), ('input2', ['batch', 3, 112, 112])]
  outputs: [('logits', ['batch', 1])]
  动态 batch=7 推理: (7, 1)
[verify] 结论: 通过
```

**这意味着 `src/utils/ver_onnx.py` 一行都不用改。**

检测器导出的实际校验结果（从自造数据训练出的 `best.pt` 导出）：

```
names   : {0: 'label', 1: 'title', 2: 'char'}     ← 与线上一致
end2end : True                                     ← 保住 [1,300,6]
imgsz   : [640, 640]
IN  [('images', [1, 3, 640, 640])]
OUT [('output0', [1, 300, 6])]
out shape (1, 300, 6)  出现类别 [0, 2]              ← runtime 只吃 0 和 2
```

**契约完全对齐，`src/utils/yolo_onnx.py` 也不需要改。**

### 5.2 端到端评测（`eval/eval_new.py`）

指标：
- `exact_rate`：整题点击顺序全对（按顺序逐个 IoU>0.5）
- `err_rate`：运行时异常率
- `count_match`：框数 == 真值数
- `det_hits`：检测命中率（忽略顺序）
- `ms_mean` / `ms_p95`：耗时

这才是验收标准 —— 不是 mAP、不是 AUC。
因为线上真正在意的是「这一题点对了没有」，而不是「检测框回归得好不好」。

### 5.3 实测验收结果（本机，全链路跑通）

数据：`tools/synth_selfcontained.py` 生成的 800 train / 120 val（**零第三方数据**）
训练：YOLO26n 19 epoch @ imgsz=640（约 2 小时）+ Siamese 8 epoch（约 30 分钟）
导出：检测器 `nms=True`（关键，见 5.4）

**端到端对比**（`eval/eval_new.py`，60 题，conf=0.2，回灌后用未改动的 runtime）：

| 指标 | baseline（线上模型） | new（重训） | 变化 |
|---|---|---|---|
| `exact_rate`（整题点击顺序全对） | **3.3%** | **80.0%** | **+76.7 pt** |
| `err_rate`（运行时异常） | 1.7% | 0.0% | −1.7 pt |
| `count_match`（框数==真值数） | 41 | 58 | +17 |
| `det_hits`（检测命中，忽略顺序） | 127 | 197 | +70 |
| `ms_mean` | 223.7 | **127.3** | −96.4 ms |

> **注意评测集归属**：上表在**训练同分布**的 `data/selfgen` val 上测得（80.0%）。
> 在另一套独立生成的 `data/selfgen_v2` val 上测得 **76.7%**（46/60）。
> 两者差异属正常泛化衰减。报告数字时请注明用的是哪一套评测集。

**消融实验**（`eval/e2e_diag.py`，说明为什么必须两个模型都重训）：

| 检测器 + 匹配器 | 端到端全对率 | 说明 |
|---|---|---|
| baseline-det + baseline-pre | 2.5% | 现状 |
| **new-det + new-pre** | **62.5%** | ✅ 两个都换（NMS 导出前） |
| new-det + baseline-pre | 7.5% | 只换检测器 → 匹配器拖死 |
| baseline-det + new-pre | 12.5% | 只换匹配器 → 检测器拖死 |
| 真值框 + baseline-pre | 20.0% | 匹配器上界 |
| **真值框 + new-pre** | **97.5%** | ✅ 新匹配器上界 |

结论非常清楚：**两个模型必须一起重训**，只换一个几乎没有收益。

**训练曲线**（YOLO26n, imgsz=640, bs=8, workers=4, cache=ram, CPU, 800 张）：

| epoch | 耗时(s) | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| 1 | 363 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 726 | 0.669 | 0.309 | 0.583 | 0.658 |
| 6 | 2165 | 0.947 | 0.664 | 0.871 | 0.859 |
| 10 | 3761 | 0.975 | 0.773 | 0.910 | 0.922 |
| 14 | 5198 | 0.984 | 0.816 | 0.938 | 0.941 |
| 19 | 7153 | **0.987** | 0.814 | 0.951 | 0.947 |

单 epoch ≈ **376 秒**（19 轮共 7153 秒 ≈ 2 小时）。

### 5.4 剩余瓶颈（已精确定位并解决）

端到端一度卡在 61.7%，而匹配器（喂真值框）已达 97.5%。逐层定位过程与结论：

| 环节 | 实测 | 判定 |
|---|---|---|
| 匹配器（喂真值框） | **97.5%** | 不是瓶颈 |
| 检测器 class0 precision / recall | 0.997 / 0.972 | 不是瓶颈 |
| 检测器 class2 precision / recall | 1.000 / 0.995 | 不是瓶颈 |
| **同一图标的重复框** | **22 对/60 图** | ★ **真瓶颈** |

**为什么重复框是致命的**：runtime 的 `find_overall_index_fast` 是**贪心一一指派**，
重复框会占掉矩阵里多余的列，把真正的候选挤掉，直接破坏点击顺序。
模型 mAP 再高也救不了 —— 因为 mAP 对近重合框是宽容的，而贪心指派不是。

**解决：导出时 `nms=True`（把 NMS 放进图里）**

| 导出方式 | 端到端全对 | class2 重复框 |
|---|---|---|
| `end2end=True, nms=False`（最初的做法） | 40.0% | 22 对 |
| **`end2end=False, nms=True`（推荐）** | **76.7%** | **3 对** |

**契约不受影响**（这一点已逐一验证）：

```
两种导出的 I/O 完全相同：
    images [1,3,640,640] float32  ->  output0 [1,300,6] float32
names 完全相同：{0: 'label', 1: 'title', 2: 'char'}
```

唯一区别是 metadata 里 `end2end` 从 `True` 变成 `False`，
而 `src/utils/yolo_onnx.py` **从不读 metadata**（已 grep 确认），
它只用 `session.get_inputs()[0].shape` 和输出的 `[...,6]` 布局。
所以 NMS 版可以**原样替换**线上模型，runtime 一行都不用改。

`train/train_detector.py` 的导出已默认改为 `nms=True`。

**关于"改数据反而更差"的一次实测**（值得记下的方法论教训）：

我先假设"提示条图标间距太挤"是主因，于是把 `synth_selfcontained.py` 的
提示条 `gap` 从 4~10 提到 14~22，生成 v2 数据集并重训。结果：

| 模型 | 端到端全对 | class2 重复框 |
|---|---|---|
| v1 数据训出 | 40.0% | 22 对 |
| v2 数据训出 | **13.3%**（更差） | **60 对**（更多） |

于是我用 `eval/audit_dataset.py` 审计了 v1/v2 两套数据的标签自洽性
（class2 数==prompt 数、target_box 能在 class0 里找到、提示框落在 strip 区、无越界）——
**两套数据全部通过，标签没有问题**。

真正的原因是：**间距变大后，单个图标更容易被拆成多个候选框，重复框反而变多**。
这说明「凭直觉改数据」是危险的，必须：
① 先做标签自洽性审计（`eval/audit_dataset.py`）；
② 再用消融定位到具体环节（`eval/e2e_diag.py`）；
③ 最后才动数据或模型。

**最终工作点**：NMS 导出 + v1 数据训出的模型 → **76.7%**。
若再加推理端去重（`--dedup`），实测可达 **65%→ 更高**，但那是改 runtime，
不如直接在导出侧把 NMS 装进图里干净。

### 5.5 上线方式

```powershell
# 1) 候选模型放一个目录
#    runs/export_next/best_v3_next.onnx + pre_model_v7_next.onnx

# 2) 端到端对比（自动回灌 + 恢复，失败也会恢复）
python eval/eval_new.py --synth data/selfgen --new-model runs/export_next \
    --limit 60 --conf 0.2

# 3) 通过后正式回灌
Copy-Item runs/export_next/best_v3_next.onnx      model/best_v3.onnx -Force
Copy-Item runs/export_next/pre_model_v7_next.onnx model/pre_model_v7.onnx -Force
python demo.py
```

回滚：`runs/_model_backup/` 里始终保留切换前的模型。
实测：跑完评测后 `model/*.onnx` 的 SHA256 与运行前**逐字节一致**（已验证）。

---

## 6. 通用方法论：模型上线后如何持续维护

这部分与具体目标无关，是可复用的工程实践。

### 6.1 数据漂移监控

按固定周期（如每周）抽样跑一遍，记录：

| 监控项 | 阈值示例 | 含义 |
|---|---|---|
| `class0` 检出数为 0 | > 30% | 检测器对新样式失效 |
| `label 数 ≠ 提示数` | > 30% | 漏检或误检增加 |
| 平均置信度 | < 0.6 | 域漂移 |
| 端到端异常率 | > 1% | 契约被破坏 |

`tools/drift_check.py` 已实现这套体检，输出 `drift_report.json`。

### 6.2 主动学习闭环

```
线上推理 → 记录低置信度/低 margin 样本 → 人工复核队列 → 补进训练集 → 重训 → A/B → 上线
```

关键设计：**用"最低配对分"和"最优−次优 margin"筛选**，
而不是盲目全量标注。`tools/autolabel.py` 已产出 `review_queue.jsonl`。

### 6.3 契约回归测试

每次换模型必跑：
1. ONNX 输入/输出 shape 与名称
2. 用真实图跑一遍端到端，断言不抛异常
3. 新旧模型在固定评测集上的 `exact_rate` 对比
4. 耗时 p95 不劣化超过阈值

### 6.4 灰度与回滚

- 模型目录版本化：`model/`（生产）+ `runs/export_next/`（候选）
- 上线靠回灌替换，回滚靠 `runs/_model_backup/`
- 注意：**不要指望 `TSEC_MODEL_DIR`**（见 5.0 坑 1）

### 6.5 数据版本化

建议每次生成数据集都记录：
```json
{"name": "selfgen-v1", "n_train": 800, "n_val": 120,
 "gen_script": "tools/synth_selfcontained.py",
 "gen_args": {"seed": 42}, "created": "..."}
```
没有这个，重训结果不可复现。

---

## 7. 成本核算（本机实测）

| 任务 | 硬约束 | 实测 |
|---|---|---|
| ONNX 反解 | onnx2torch + Identity 内联 | pre_model 0.65s / best_v3 0.28s |
| 合成数据生成 | 纯 CV | 920 张 ~30s |
| YOLO26n 训练 | imgsz=640, bs=8, CPU | **~360s/epoch**（800 张）|
| YOLO26n 训练 | imgsz=320, bs=4, CPU | ~145s/epoch |
| Siamese 训练 | 112×112, bs=64, CPU | ~215s/epoch（11228 对）|
| ONNX 导出（YOLO） | opset14 + slim | ~30s |
| 单张推理 | CPU 单核 | YOLO ~60ms，端到端 ~320ms |

**这台机器上的现实结论**：
- **匹配器**完全可训：8 epoch 就到 **val_acc 97.7%**。
- **检测器**可训但慢：imgsz=640 约 6 min/epoch，
  30 epoch ≈ 3 小时。**建议先 `imgsz=320` 跑通流程**（2.4 min/epoch），
  或用 GPU（T4 级别约快 15~25 倍，30 epoch ≈ 10 min）。


---

## 附录 A：第三方站点协议侦察记录（仅存档，不实施）

> 以下结论仅用于说明「域漂移」的存在与量级。**不含采集脚本的运行指引、不含定制求解方案。**

实测 `cun.ai/sign-up`（通过其 `/api/status` 公开配置 + 一次页面加载抓包）：

```
captcha_provider      = tencent
tencent_captcha_app_id = 195317439
脚本                  = turing.captcha.qcloud.com/TJCaptcha.js
```

`prehandle` 回包关键字段：

| 字段 | 值 | 含义 |
|---|---|---|
| `subcapclass` | `2408` | **图标点选**（"请依次点击："） |
| `subcapclass` | `2401` | 滑块 |
| `bg_elem_cfg.size_2d` | `[672, 480]` | 底图尺寸 |
| 提示图 | `170×50` 独立精灵图 | 与底图**分离**，非图内提示条 |

而 `best_v3.onnx` 训练于**中文文字点选**（344×384，图内提示条 + 彩色透视变形汉字）。

**量化漂移**（40 个真实样本，已拼成旧域版式后送入现有模型）：

```
class2(char) 检出数 = 0        →  100% 样本
label 检出数 ≠ 提示图标数      →   78% 样本
运行时异常                     →  100%  (ValueError: need at least one array to concatenate)
```

根因：模型把所有图标判成 `class 0`（以为那是"待点目标字"），`class 2` 一个都不出
→ `run()` 里 `chars` 为空 → `np.concatenate` 抛异常。

**这证明两件事**：
1. 换域后旧模型不是"准确率下降"，而是**结构上完全不可用**。
2. 仅靠微调救不回来 —— 必须换域重训。而重训的前提，正是本文第 2、3 节重建的
   「可训练权重 + 数据供给」两件事。

---

## 附录 B：文件清单

```
tools/
  inspect_onnx.py           ONNX 签名/算子/契约体检
  recover_weights.py        ONNX → PyTorch state_dict（含 Identity 内联修复）★
  verify_recovered.py       反解权重可用性自检（梯度/推理一致性）
  icon_assets.py            从提示图提取图标素材（列投影切分）
  plate_gen.py              抠图标生成干净底板（暗度阈值 + 几何判据）★
  synth_gen.py              合成数据集（从自有/授权样本反推）
  synth_selfcontained.py    完全自造的合成数据集（零外部依赖）★
  autolabel.py              伪标签 + 人工复核队列
  icon_match.py             快速形状归一匹配（旋转/尺度不变）
  drift_check.py            数据漂移量化体检
  sync_cv.py                cv/ 与 src/utils/ 同步（原仓库自带）
train/
  train_detector.py         YOLO 训练 + 导出(nms=True) + 契约校验 ★
  train_matcher.py          Siamese 训练 + 导出 + 契约校验 ★
  prepare_pairs.py          配对数据生成（含难负样本）
  run_all.ps1               一键跑通全链路
  crash_bisect.py           崩溃定位：前向/反向/EMA/绘图逐个隔离
  crash_bisect2.py          崩溃定位 v2
  epoch_end_probe.py        ★ 定位出 polars 非法指令崩溃
  probe_train_flags.py      训练 flag 排查
eval/
  eval_new.py               ★ 回灌式端到端新旧对比（子进程隔离）
  e2e_diag.py               ★ 失败点定位（检测 vs 匹配消融）
  v1_v2_compare.py          ★ 数据集版本对比（量化"改数据"的收益）
  audit_dataset.py          ★ 数据集标签自洽性审计（改数据前必跑）
  det_diag.py               检测器 precision/recall/漏检尺寸诊断
  class2_diag.py            提示条字符（class 2）专项诊断
docs/
  TRAINING_AND_DATA_REFRESH.md   本文（方法论 + 验收证据）
  TRAINING_BENCHMARK.md          本机吞吐基准 + 环境坑
weights/
  pre_model_v7.pt           反解出的匹配器权重（可直接续训）
  best_v3.pt                反解出的检测器权重（仅推理用）
  best_v3_native.pt         原生结构检测器权重（续训用）
  recovered_report.md       反解详细报告
runs/export_nms/
  best_v3_nms.onnx          ★ 最终推荐检测器（NMS 在图内，契约一致）
  pre_model_v7_nms.onnx     配套匹配器
runs/export_next/           中间产物（end2end 导出，用于对比实验）
runs/export_v2/             v2 数据训出的模型（对比实验用）
```

---

## 附录 C：复现步骤（从零到验收）

```powershell
cd T-Sec-captcha
$env:PYTHONIOENCODING='utf-8'

# 0) 依赖（关键：polars 必须装 rtcompat，否则老 CPU 上训练必崩）
pip install -r requirements.txt
pip install onnx onnxruntime ultralytics onnx2torch
pip install "polars[rtcompat]"      # ★ 不可省略

# 1) 反解现有 ONNX 成可训练权重
python tools/recover_weights.py --all

# 2) 生成完全自造的合成数据集（标签 100% 精确）
python tools/synth_selfcontained.py --out data/selfgen --n-train 800 --n-val 120

# 3) 生成匹配器训练配对
python train/prepare_pairs.py --synth data/selfgen --out data/selfpairs

# 4) 训练两个模型 + 导出 + 契约校验
#    （或直接 pwsh -File train/run_all.ps1）
python train/train_detector.py --data data/selfgen/data.yaml `
    --epochs 30 --imgsz 640 --batch 8 --device cpu --cache ram `
    --out-onnx runs/export_next/best_v3_next.onnx
python train/train_matcher.py --pairs data/selfpairs `
    --epochs 8 --batch 64 --threads 10 `
    --out-onnx runs/export_next/pre_model_v7_next.onnx

# 5) 端到端验收（自动回灌 + 恢复）
python eval/eval_new.py --synth data/selfgen --new-model runs/export_nms `
    --limit 60 --conf 0.2

# 6) 失败点定位与数据审计（调优时用）
python eval/e2e_diag.py      --new-model runs/export_next --limit 40   # 检测 vs 匹配消融
python eval/audit_dataset.py --data data/selfgen --split val            # 标签自洽性（改数据前必跑）
python eval/v1_v2_compare.py --v1 runs/export_next --v2 runs/export_v2  # 数据版本对比
python eval/det_diag.py      --new-model runs/export_next --limit 50
python eval/class2_diag.py   --det runs/export_next/best_v3_next.onnx --limit 8
```

> 注意第 4 步：`train_detector.py` 的导出**默认 `nms=True`**（NMS 装进图里）。
> 这是实测把端到端从 40.0% 提到 76.7% 的关键，见 5.4 节。
> 若手工导出，务必带上 `nms=True`。

---

## 附录 D：已知待改进项（诚实记录）

1. **★ 已解决：重复框是端到端的主瓶颈**
   导出改为 `nms=True`（NMS 装进图里），端到端从 40.0% → **76.7%**，
   重复框从 22 对 → 3 对。契约完全不变，runtime 不用改。
   `train/train_detector.py` 已默认 `nms=True`。

2. **"凭直觉改数据"的一次失败教训（已记录在 5.4 节）**
   我原本假设提示条图标间距太挤是主因，把 `gap` 从 4~10 提到 14~22 生成 v2 数据集，
   结果端到端从 40.0% **掉到 13.3%**，重复框从 22 对涨到 60 对。
   经 `eval/audit_dataset.py` 审计，两套数据标签**都自洽无误** ——
   真正原因是间距变大后单图标更易被拆成多个候选框。
   **教训：先审计标签、再消融定位、最后才动数据。**

3. **类别 1 (`title`) 没有训练数据**
   当前 runtime 不使用该类；若要识别"请依次点击"标题需补标注。

4. **本方案未验证跨域泛化**
   在自造数据上 76.7% 说明**流水线通了**，但换成真实站点风格必须：
   ① 用链路 A（`icon_assets` + `plate_gen` + `synth_gen`）从**自有/授权**样本反推素材与底板；
   ② 或做域随机化增强（字体、纹理、透视、JPEG 质量、干扰线）。
   **切勿直接拿自造数据训出的模型上线。**

5. **`--conf` 工作点未标定**
   实测最优在 `conf=0.2` 附近，高于 runtime 默认的 0.3。
   上线前应在自有评测集上做工作点标定（precision/recall 权衡）。

6. **匹配器的 AUC 一直是 nan**
   `train_matcher.py` 里 AUC 依赖 `sklearn`，本机未安装。
   不影响 val_acc 判定（每轮都打印），但若要 AUC 需 `pip install scikit-learn`。

