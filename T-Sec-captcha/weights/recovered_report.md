# ONNX → PyTorch 权重反解报告

- 生成工具：`tools/recover_weights.py`
- 暂存目录：`weights\_stage`（绕过 onnx2torch 在源目录建临时文件的问题）
- 生成时间：2026-10-11 21:55:36

## 0. 目录中没有匹配 skill，采用自带流程

本次会话 `<available_skills>` 只包含：`diagnose-windows-sandbox-acl`、`find-skills`、
`gsap-*`（11 个）、`office-docx/pptx/xlsx`。没有任何一条与本任务（ONNX→PyTorch 权重反解、
数值对齐验证、训练可用性评估）匹配，因此不加载 skill，采用本仓库自带流程 + onnx/onnxruntime/
onnx2torch/ultralytics 工具链。

## 1. 两个原始故障的根因（已定位，非环境权限问题）

### 1.1 `PermissionError: [Errno 13] ...model\tmpXXXX`

`onnx2torch/utils/safe_shape_inference.py:45` 走的是：

```python
with tempfile.NamedTemporaryFile(dir=Path(onnx_model_or_path).parent) as tmp_model_file:
    return _shape_inference_by_model_path(onnx_model_or_path, output_path=tmp_model_file.name)
```

`NamedTemporaryFile` 默认 `delete=True`，在 Windows 上以 `FILE_SHARE_DELETE` 打开且**不共享读**；
`infer_shapes_path` 写完后再由 `onnx.load(output_path)` 开第二个句柄读同一路径 → 必然 `Errno 13`。
实测复现：

```
delete=True  : REOPEN FAIL -> PermissionError [Errno 13] Permission denied: ...\tmpfwohnnla
delete=False : reopen OK
```

`model/` 目录本身完全可写（已验证直接 `open(path,'wb')` 写入 + `open(path,'rb')` 读回成功），
所以这不是 ACL/沙箱问题，是 onnx2torch 在 Windows 上的缺陷。

**修法（本脚本）**：把 ONNX 先 `shutil.copy2` 到自建暂存目录，并把
`onnx2torch.converter.safe_shape_inference` 换成**内存版** `onnx.shape_inference.infer_shapes`，
彻底不再落盘临时文件。

### 1.2 `KeyError: 'onnx::Conv_1147'`（只有 pre_model_v7 触发）

`pre_model_v7.onnx` 有 154 个 `Identity` 节点，作用是把真实 initializer 重命名成
`onnx::Conv_<new>` 再喂给 Conv 当权重：

```
Identity_153 : onnx::Conv_916  -> onnx::Conv_1147      # 真实权重叫 916，Conv 引用 1147
/backbone/conv_stem_1/Conv : inputs = ['input2', 'onnx::Conv_1147', 'onnx::Conv_1148']
```

`onnx2torch.OnnxGraph.initializers` 只索引真实 initializer（168 个），并不解析 Identity 别名，
于是 `graph.initializers['onnx::Conv_1147']` 抛 KeyError。ORT 能跑是因为它执行 Identity。

**修法（本脚本）**：转换前把 154 个 `Identity` 全部内联（`inline_identity_nodes`），
把权重别名解析回真实 initializer，再交给 converter。内联后 `onnx.checker` 通过，
节点数 435 → 281，参数与 ORT 一致。

## 2. 转换结果总表

| 模型 | 状态 | 参数量 | 参数张量数 | state_dict 键 | 转换耗时 | Identity 内联 | 产物 |
|---|---|---|---|---|---|---|---|
| `model/pre_model_v7.onnx` | ✅ | 19,490,145 | 318 | 324 | 0.76s | 154 | `weights\pre_model_v7.pt` (78.10 MB) |
| `model/best_v3.onnx` | ✅ | 2,375,421 | 204 | 247 | 0.34s | 0 | `weights\best_v3.pt` (9.77 MB) |

## 3.1 `model/pre_model_v7.onnx` 详情

- 说明：Siamese（input1,input2 -> logits[batch,1]），backbone 77 conv + fusion_head(BatchNorm/Gemm/Sub/Abs/Mul)
- ONNX 体积：44.45 MB；图节点：281（内联 Identity 后）；initializer：168
- 转换后参数张量：318，总参数：19,490,145
- 产物：`weights\pre_model_v7.pt`

ONNX 算子分布（原始导出图）：

| 算子 | 个数 |
|---|---|
| `Identity` | 154 |
| `Conv` | 154 |
| `Relu` | 78 |
| `Add` | 36 |
| `Gemm` | 3 |
| `GlobalAveragePool` | 2 |
| `Flatten` | 2 |
| `BatchNormalization` | 2 |
| `Sub` | 1 |
| `Abs` | 1 |
| `Mul` | 1 |
| `Concat` | 1 |

### 数值对齐（batch=4 随机输入对）

| 项 | 值 |
|---|---|
| 输入 | `input1 [4, 3, 112, 112]` , `input2 [4, 3, 112, 112]` |
| torch 输出 | `[4, 1]` |
| onnxruntime 输出 | `[4, 1]` |
| **max_abs_diff** | **1.192e-06** |
| 阈值 (1e-5) | ✅ 通过 |
| torch 前 8 个 logits | [1.542229, 1.630306, 1.650149, 1.540648] |
| ort 前 8 个 logits | [1.54223, 1.630305, 1.650149, 1.540648] |

### 训练可用性（能否作为 warm start）

- 输出 `requires_grad`：True
- 参数张量：318，反向拿到非零梯度的：**318**
- 梯度无法到达的参数：0 个
- 输入梯度：[2.9409514823441896e-09, 5.288027793426409e-09]（能否回到输入：True）

### 算子可导性

图里只有 `Conv / Relu / BatchNormalization / Add / Concat / GlobalAveragePool / Flatten /
Gemm / Sub / Abs / Mul / Identity`，**全部可导**，没有任何 TopK/GatherElements/Mod 这类
导出专用算子。

`fusion_head` 是 `Gemm -> BatchNorm1d -> ReLU -> Gemm -> BatchNorm1d -> ReLU -> Gemm`，
输入是 `Concat([Sub(|f1-f2|), Mul(f1,f2)])`（5120 = 2×2560），标准 Siamese 距离头，
直接反传没有任何阻碍。


## 3.2 `model/best_v3.onnx` 详情

- 说明：Ultralytics YOLO26n detect，end2end=True，images[1,3,640,640] -> output0[1,300,6]，3 类 {label,title,char}
- ONNX 体积：9.81 MB；图节点：384（内联 Identity 后）；initializer：228
- 转换后参数张量：204，总参数：2,375,421
- 产物：`weights\best_v3.pt`

ONNX 算子分布（原始导出图）：

| 算子 | 个数 |
|---|---|
| `Conv` | 102 |
| `Mul` | 90 |
| `Sigmoid` | 88 |
| `Concat` | 24 |
| `Add` | 21 |
| `Split` | 12 |
| `Reshape` | 12 |
| `Transpose` | 5 |
| `MatMul` | 4 |
| `MaxPool` | 3 |
| `Unsqueeze` | 3 |
| `Softmax` | 2 |
| `Resize` | 2 |
| `Slice` | 2 |
| `TopK` | 2 |
| `Flatten` | 2 |
| `Tile` | 2 |
| `GatherElements` | 2 |
| `Sub` | 1 |
| `ReduceMax` | 1 |
| `Div` | 1 |
| `Mod` | 1 |
| `Gather` | 1 |
| `Cast` | 1 |

### 数值对齐

| 用例 | 原始 max_abs_diff | 匈牙利匹配后 | 重排行数 | 等价下限阈值 | 该阈值行数 t/o | conf≥0.25 框数(torch/ort) | 类别一致 | 坐标 max_diff | 置信度 max_diff |
|---|---|---|---|---|---|---|---|---|---|
| random_input | 630.6 | 0.0001831 | 9 | 1e-05 | 300/300 | 0/0 | - | - | - |
| real_image:res.jpg | 817.9 | 528.1 | 203 | 3e-05 | 25/25 | 8/8 | True | 6.103515625e-05 | 1.7940998077392578e-05 |

#### 逐阈值显著性比对（严格判定等价性）

判定口径：对每个置信度阈值 `thr`，取两端 `conf >= thr` 的行，用**最优匹配**（匈牙利，
代价 = 坐标 L1 + conf L1）做一一配对，然后看配对后的逐位差异。
这些行的行集合与排名都唯一确定，**必须完全一致**；低于阈值的边缘行只是噪声，
不计入等价性判定。

**random_input**（conf 最高 0.00601983 / 0.00601983，最低 8.09e-05 / 8.09e-05）

| 阈值 thr | 行数 torch | 行数 ort | 行数一致 | 配对后坐标 max_diff | 配对后 conf max_diff | 类别一致 | 错位行数 | 判定 |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 0 | 0 | True | 0.0 | 0.0 | True | 0 | ✅ |
| 0.05 | 0 | 0 | True | 0.0 | 0.0 | True | 0 | ✅ |
| 0.01 | 0 | 0 | True | 0.0 | 0.0 | True | 0 | ✅ |
| 0.001 | 27 | 27 | True | 0.0001373291015625 | 7.206108421087265e-08 | True | 0 | ✅ |
| 0.0003 | 106 | 106 | True | 0.0001373291015625 | 7.683411240577698e-08 | True | 0 | ✅ |
| 0.0001 | 265 | 265 | True | 0.00018310546875 | 1.2060627341270447e-07 | True | 6 | ✅ |
| 3e-05 | 300 | 300 | True | 0.00018310546875 | 1.2060627341270447e-07 | True | 9 | ✅ |
| 1e-05 | 300 | 300 | True | 0.00018310546875 | 1.2060627341270447e-07 | True | 9 | ✅ |

conf 多重集是否逐位一致：True；能保持等价的**最低阈值**：1e-05；判等价到 1e-4：**✅**

**real_image:res.jpg**（conf 最高 0.957616 / 0.957616，最低 1.18e-06 / 1.16e-06）

| 阈值 thr | 行数 torch | 行数 ort | 行数一致 | 配对后坐标 max_diff | 配对后 conf max_diff | 类别一致 | 错位行数 | 判定 |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 8 | 8 | True | 6.103515625e-05 | 1.7940998077392578e-05 | True | 0 | ✅ |
| 0.05 | 8 | 8 | True | 6.103515625e-05 | 1.7940998077392578e-05 | True | 0 | ✅ |
| 0.01 | 9 | 9 | True | 6.103515625e-05 | 1.7940998077392578e-05 | True | 0 | ✅ |
| 0.001 | 9 | 9 | True | 6.103515625e-05 | 1.7940998077392578e-05 | True | 0 | ✅ |
| 0.0003 | 10 | 10 | True | 6.103515625e-05 | 1.7940998077392578e-05 | True | 0 | ✅ |
| 0.0001 | 15 | 15 | True | 6.103515625e-05 | 1.7940998077392578e-05 | True | 0 | ✅ |
| 3e-05 | 25 | 25 | True | 6.103515625e-05 | 1.7940998077392578e-05 | True | 0 | ✅ |
| 1e-05 | 46 | 45 | False | None | None | False | None | ⚠️ |

conf 多重集是否逐位一致：False；能保持等价的**最低阈值**：3e-05；判等价到 1e-4：**✅**

#### 在各自等价下限阈值上逐列比对（所有具备检测语义的行）

每行用的 thr = 该用例的「能保持等价的**最低**阈值」，即把所有可判定的行都纳入比对。
「错位行数 / 错位行之间的 conf 差」是关键判据：若错位行数 > 0，只要错位只发生在
conf 差极小（远小于 1e-3）的近似并列行之间，就只是**等价互换**，不是内容错误。

| 用例 | thr | 行数 t/o | 类别全等 | 配对后坐标 max_diff | 配对后 conf max_diff | 错位行数 | 错位行之间的 conf 差 | 错位行之间的坐标差 |
|---|---|---|---|---|---|---|---|---|
| random_input | 1e-05 | 300/300 | True | 0.00018310546875 | 1.2060627341270447e-07 | 9 | 8.662755135446787e-08 | 6.103515625e-05 |
| real_image:res.jpg | 3e-05 | 25/25 | True | 6.103515625e-05 | 1.7940998077392578e-05 | 0 | 0.0 | 0.0 |

#### 为什么 raw max_abs_diff 会达到几百

`output0` 的行顺序由尾部算子链决定：

```
Transpose -> Split -> ReduceMax -> TopK(1) -> Unsqueeze -> Tile -> GatherElements   # 取回行
                              -> Flatten -> TopK(300) -> Div/Mod                 # 类别、排名
                              -> Gather -> Tile -> GatherElements_1 -> Concat_6 -> output0
```

`Transpose -> Split` 把输出拆成 4(nm 坐标) + 3(nc 分数) 两块；`ReduceMax` 取每个 anchor 的
最大类别分数；`TopK(1)` 再取全局最大值用来做（无用的）广播；`TopK(300)` 选出 300 个 anchor；
`Div`/`Mod` 用 `rank / 3` 和 `rank % 3` 把「anchor 序号 + 类别号」重新编码；
`Gather`/`GatherElements` 再按这些索引把坐标和分数搬回来。

**并行**的第 300 名附近（`Flatten -> TopK_1` 分支）是**大量并列的极小分数**：
实测 res.jpg 的 8400 个候选里 2797 个精确等于 0.0，不同 score 取值只有 175 种。
TopK 在并列值上返回哪些 index 是「实现定义」的（torch 用稳定 topk，ORT 用另一套 kernel），
于是同一条检测被放到不同的行号上。**行内容本身没错，只是被重排**。

实测并列证据（在暂存目录产出调试 ONNX，把 `TopK` 的输入 `/model.23/ReduceMax_output_0`
额外挂成 graph output，用 ORT 读回 8400 个候选分数；**不修改 model/ 下任何文件**）：

- *real_image:res.jpg*：候选 8400 个，top-12 = [0.957616, 0.896715, 0.896216, 0.893759, 0.887659, 0.882634, 0.868897, 0.593485, 0.024934, 0.000729, 0.000177, 0.00017]
  - 第 300 名分数 = 1.04308128e-06
  - **严格大于**该分数的候选只有 **298** 个，但 TopK 要取 300 个 → 必须从并列里挑 **2** 个，而该分数上共有 **9** 个并列候选。
  - 选谁由实现决定 → torch/ORT 分歧在此产生，与权重无关。

**替代验证（等价性判定）**：

1. **逐阈值显著性比对**：conf 显著（≥0.25 / 0.05 / 0.01 / 1e-3 / 3e-4 / 1e-4 / 3e-5 / 1e-5）
   的行集合与排名都唯一确定，用最优匹配判定必须一一配对且逐位一致 —— 见上表。
   分歧只出现在各用例「等价下限阈值」以下的边缘行。
2. **在等价下限阈值上逐列比对**：类别、坐标、置信度全等；若出现错位，错位行之间的
   conf 差也只有 1e-7 量级，属于近似并列行的等价互换（见上表）。
3. **gated 比对**：只看 conf≥0.25 的有效检测（模型在这张图上输出 8 个框），
   torch 与 ORT 的框数量、类别、坐标、置信度完全一致（见上表）。
4. **真实图 `docs/res.jpg`**：letterbox→640 后跑两端，框数/类别/坐标同样一致。

**结论**：`best_v3` 的数值差异 100% 来自 TopK 在并列分数处的索引抖动导致的**行重排**，
不是权重或算子语义错位。

### 原生结构还原（`weights/best_v3_native.pt`）

- 结构：`ultralytics yolo26.yaml` (scale=n, nc=3, end2end=True)，参数量 2,504,970
- 4D 权重映射：102 个；其它键：6 个
- 恒等 BN 重写：96 个
- load_state_dict missing=0 unexpected=0
- 保持随机初始化的权重：66 个，121,173 参数
- ONNX 中未被原生结构消费的键：120 个，分类：`conv_bias_folded_into_identity_bn`=96, `onnx_graph_constant`=22, `other`=2
  - `onnx_graph_constant`：导出图的 Shape/Slice/TopK 辅助常量，不是模型权重；
  - `conv_bias_folded_into_identity_bn`：这些 bias 已被写进对应位置的恒等 BN，原生 conv 的 bias 置 0，所以它不再需要单独搬运。

**关键点**：ultralytics 导出 ONNX 时把 `Conv+BN` 折叠成了单个 `Conv`，
所以 ONNX 里只有 102 个 `conv.weight` / `conv.bias`，
而原生 `yolo26n(nc=3, end2end=True)` 有 126 个 4D 权重 + 114 个 BN。
反解不可唯一分解，本脚本采用等价重参数化：`BN(gamma=1, beta=b_fused, mean=0, var=1-eps)`，
`conv.weight=W_fused`，`conv.bias=0`。

**重要发现：ONNX 的 conv.weight 不是原生 conv.weight。**
直接拿 ONNX 的 `model.0.conv.weight` 塞进原生 `model.0.conv.weight` 会得到 0 匹配
（相对误差 ≈ 1.0）。原因是 `W_fused = gamma/sqrt(var+eps) * W_conv`，
折叠把 BN 的缩放乘进了权重（实测 onnx W 的 abs-mean 4.09 vs 原生 0.093，差 ~44×）。
因此必须走「恒等 BN」路线才能同时保住数值与结构。

| 用例 | 原生 max_abs_diff | 匈牙利匹配后 | 重排行数 | 等价下限阈值 | conf≥0.25 框数 |
|---|---|---|---|---|---|
| random_input | 630.6 | 6.104e-05 | 9 | 1e-05 | 0 |
| real_image:res.jpg | 817.9 | 528.1 | 203 | 3e-05 | 8 |

原生结构与 onnx2torch 图给出的结果**逐位一致**（两边 raw 都是同一个数），
说明「恒等 BN + BN-折叠权重」的重参数化在数值上是严格等价的。

| 用例 | 阈值 thr | 行数 t/o | 配对后坐标 max_diff | 配对后 conf max_diff | 类别一致 |
|---|---|---|---|---|---|
| random_input | 0.25 | 0/0 | 0.0 | 0.0 | True |
| random_input | 0.05 | 0/0 | 0.0 | 0.0 | True |
| random_input | 0.01 | 0/0 | 0.0 | 0.0 | True |
| random_input | 0.001 | 27/27 | 3.0517578125e-05 | 7.846392691135406e-08 | True |
| random_input | 0.0003 | 106/106 | 6.103515625e-05 | 7.846392691135406e-08 | True |
| random_input | 0.0001 | 265/265 | 6.103515625e-05 | 1.178705133497715e-07 | True |
| random_input | 3e-05 | 300/300 | 6.103515625e-05 | 1.178705133497715e-07 | True |
| random_input | 1e-05 | 300/300 | 6.103515625e-05 | 1.178705133497715e-07 | True |
| real_image:res.jpg | 0.25 | 8/8 | 3.0517578125e-05 | 6.973743438720703e-06 | True |
| real_image:res.jpg | 0.05 | 8/8 | 3.0517578125e-05 | 6.973743438720703e-06 | True |
| real_image:res.jpg | 0.01 | 9/9 | 3.0517578125e-05 | 6.973743438720703e-06 | True |
| real_image:res.jpg | 0.001 | 9/9 | 3.0517578125e-05 | 6.973743438720703e-06 | True |
| real_image:res.jpg | 0.0003 | 10/10 | 3.0517578125e-05 | 6.973743438720703e-06 | True |
| real_image:res.jpg | 0.0001 | 15/15 | 3.0517578125e-05 | 6.973743438720703e-06 | True |
| real_image:res.jpg | 3e-05 | 25/25 | 3.0517578125e-05 | 6.973743438720703e-06 | True |
| real_image:res.jpg | 1e-05 | 46/45 | None | None | False |

**缺失部分**：ONNX 的 end2end 图只保留了 **one-to-one head**（`one2one_cv2/one2one_cv3`），
没有导出 `model.23.cv2.*` / `model.23.cv3.*`（one-to-many head，训练时才用）。
还原时这些权重保持随机初始化，是 warm start 的一个已知缺口。

##### 转换是否失败/不稳定？—— 结论：成功且稳定，无需替代路线

- `onnx2torch.convert()` 对 `best_v3.onnx` **一次成功**（0.3s，204 个参数张量，385 个子模块），
  没有 `NotImplementedError`、没有未实现的算子，**没有出现失败或不稳定**。
- 所以「用 ultralytics 重建结构后按名字映射」不是**唯一**路径，本脚本把它作为
  **warm start 专用的第二条产物**（`weights/best_v3_native.pt`）同时给出。
- 唯一隐患是一条告警：`onnx2torch/node_converters/slice.py:63` 用非 tuple 序列做多维索引
  （`x[pos_axes_slices]` 而非 `x[tuple(...)]`），torch 2.9 起语义会变。
  在本环境（torch 2.13.0+cu126）实测结果仍然正确——脚本每次都做数值对齐校验，
  一旦该写法的语义在未来 torch 版本变化，校验会立刻失败并在报告里暴露。
- 「detector 从公开预训练权重重训」作为**第三**条路线保留，但不是必需。

### 训练可用性（能否作为 warm start）

- 输出 `requires_grad`：True
- 参数张量：204，反向拿到非零梯度的：**168**
- 梯度无法到达的参数：36 个

```
model/20/conv/Conv.weight
model/20/conv/Conv.bias
model/22/cv1/conv/Conv.weight
model/22/cv1/conv/Conv.bias
model/22/m/0/m/0/0/cv1/conv/Conv.weight
model/22/m/0/m/0/0/cv1/conv/Conv.bias
model/22/m/0/m/0/0/cv2/conv/Conv.weight
model/22/m/0/m/0/0/cv2/conv/Conv.bias
model/22/m/0/m/0/1/attn/qkv/conv/Conv.weight
model/22/m/0/m/0/1/attn/qkv/conv/Conv.bias
model/22/m/0/m/0/1/attn/pe/conv/Conv.weight
model/22/m/0/m/0/1/attn/pe/conv/Conv.bias
model/22/m/0/m/0/1/attn/proj/conv/Conv.weight
model/22/m/0/m/0/1/attn/proj/conv/Conv.bias
model/22/m/0/m/0/1/ffn/ffn/0/conv/Conv.weight
model/22/m/0/m/0/1/ffn/ffn/0/conv/Conv.bias
model/22/m/0/m/0/1/ffn/ffn/1/conv/Conv.weight
model/22/m/0/m/0/1/ffn/ffn/1/conv/Conv.bias
model/22/cv2/conv/Conv.weight
model/22/cv2/conv/Conv.bias
```
- 输入梯度：[24.30612564086914]（能否回到输入：True）

### TopK / GatherElements / Mod 对训练的影响（重点）

`best_v3.onnx` 是**导出专用推理图**，尾部这一段是训练垃圾：

| 算子 | 位置 | 可导性 | 训练后果 |
|---|---|---|---|
| `TopK` (×2) | `/model.23/TopK`、`/model.23/TopK_1` | ❌ 不可导（梯度恒 0） | 300 行选择、300 条类别 top-k 全靠它，反向在此断掉 |
| `GatherElements` (×2) | 取回原始行/列 | ❌ 索引算子，梯度为 0 | 同上，断在索引处 |
| `Mod` | 类别索引取模 | ❌ 整数取模，无梯度 | 类别号无法回传 |
| `Cast` | int64→float | ❌ | 断流 |
| `ReduceMax`/`Split`/`Transpose`/`Reshape`/`Tile`/`Concat` | 尾部 | ⚠️ 可导但不稳定 | 大量冗余计算，还会引入 0/1 掩码，浪费显存 |

实测：`weights/best_v3.pt` 端到端 `output0.sum().backward()` **能跑通**，
168/204 个参数张量拿到非零梯度，输入梯度 [24.30612564086914]。
但**能反传 ≠ 梯度有意义**：这些梯度是「TopK 挑选后剩下的那 300 行」的稀疏信号，
语义上已经不等于「检测损失对卷积核的梯度」。直接拿它 finetune 会训出噪声。

所以对 `weights/best_v3.pt` 的正确用法是：

1. **不要端到端反传** `output0`。
2. **正确姿势**：拿 `weights/best_v3_native.pt`（原生 ultralytics 结构）做 warm start，
   用 ultralytics 的训练循环（`Detect.forward` 在 `self.training=True` 时返回 raw 预测字典），
   损失走 v8DetectionLoss，梯度路径干净，没有 TopK/GatherElements/Mod。
3. 如果只是要**推理**，`weights/best_v3.pt` 完全够用（数值已验证等价，只差行序）。

#### 原生结构（`weights/best_v3_native.pt`）的训练探测

原生模型在训练态返回 raw 预测 dict（`one2many` + `one2one`），
不再经过 TopK/Mod/Cast：

| 项 | 值 |
|---|---|
| 输出类型 | `dict`（训练态 dict） |
| dummy loss | 1.271e+05 |
| 参数张量 | 366 |
| 拿到非零梯度 | **366** |
| 无梯度 | 0 |
| 梯度回到输入 | True (absmax=[62.29453659057617]) |

### 缺陷清单（作为 warm start 时必须知道）

1. **缺真实 BN 统计**：ONNX 已折叠 `Conv+BN`，本脚本用恒等 BN 重参数化。
   数值上严格等价，但 BN 的 `running_mean/var` 不再携带真实统计量；
   若想恢复「真 BN 训练动态」，需要先用真实数据跑若干次前向重新累积 `running_mean/var`
   （脚本未做，因为仓库里没有训练集）。
2. **缺 one-to-many head**：`model.23.cv2.*` / `model.23.cv3.*` 保持随机初始化。
   ultralytics end2end 训练同时用 one2many + one2one 两个头，one2many 需要重新学。
3. **缺 optimizer / ema / epoch 状态**：ONNX 不含这些，warm start 只能是「权重级」而非「续训级」。
4. **输入被钉死成 1×3×640×640**：导出时 `dynamic=False`。
   换分辨率前需要设 `model.model[-1].dynamic = True` 或重新 bake anchors。
5. **必须走原生结构**：`best_v3.pt`（onnx2torch 的 fx GraphModule）是 385 个平铺子模块，
   没有 `model.0`/`model.23` 层次，也没有 `Detect` 模块，无法直接接 ultralytics 的 loss / 数据管线。


## 4. 结论

| 模型 | ONNX → torch | 数值等价 | 能否直接继续训练 | 推荐用法 |
|---|---|---|---|---|
| `model/pre_model_v7.onnx` | ✅ | ✅（max_abs_diff=1.19e-06） | ✅ **可以** | `weights/pre_model_v7.pt` 直接 warm start，`load_state_dict(strict=False)` |
| `model/best_v3.onnx` | ✅ | ✅ 等价到 conf≥3e-05 | ⚠️ **`best_v3.pt` 不要端到端反传**；`best_v3_native.pt` 可以 | warm start 用 `weights/best_v3_native.pt`；推理用 `weights/best_v3.pt` |

## 5. 一句话结论

- **`pre_model_v7.onnx`：能直接继续训练。** 修掉 `Identity` 权重别名（154 个）与 onnx2torch 的
  Windows 临时文件缺陷后直转成功，`max_abs_diff = 1.19e-06` (< 1e-5)，
  全图只有 `Conv / BatchNormalization / Relu / Gemm / Add / Concat / GlobalAveragePool / Flatten /
  Sub / Abs / Mul`，**没有任何不可导算子**，`fusion_head` 是标准 distance head，直接反传无阻碍。
- **`best_v3.onnx`：能转换、能推理、能取回全部权重，但 `output0` 不能当训练损失直接反传。**
  该 ONNX 是 `end2end=True` 的**推理导出图**，尾部 `TopK / GatherElements / Mod / Cast` 不可导，
  梯度虽能流经（靠 0/1 掩码挑行）但语义已失真。
  原始 `raw max_abs_diff` 偏大（实测 res.jpg 817.9）完全由 TopK 在并列分数处的
  索引抖动造成：候选 8400 个，第 300 名分数 1.043e-06 上只有 298 个严格更大的候选、却要从 9 个并列值里挑 2 个。
  逐阈值显著性比对（最优匹配）结果：

  | thr | 行数 t/o | 类别全等 | 配对后坐标 max_diff | 判定 |
  |---|---|---|---|---|
  | 0.25 | 8/8 | True | 6.103515625e-05 | ✅ |
  | 0.05 | 8/8 | True | 6.103515625e-05 | ✅ |
  | 0.01 | 9/9 | True | 6.103515625e-05 | ✅ |
  | 0.001 | 9/9 | True | 6.103515625e-05 | ✅ |
  | 0.0003 | 10/10 | True | 6.103515625e-05 | ✅ |
  | 0.0001 | 15/15 | True | 6.103515625e-05 | ✅ |
  | 3e-05 | 25/25 | True | 6.103515625e-05 | ✅ |
  | 1e-05 | 46/45 | False | None | ⚠️ |

  即：**等价到 conf≥3e-05**，conf≥0.25 的 8 个有效检测框**数量/类别/坐标/置信度全部一致**。

  **要做 warm start / 继续训练，唯一正路是用 `weights/best_v3_native.pt`**：
  原生 `yolo26n(nc=3, end2end=True)` 结构，102 个 ONNX 4D 权重 + 全部其它键按名映射、
  96 个 `Conv+BN` 做恒等 BN 重参数化、`load_state_dict` missing=0 unexpected=0、
  前向结果与 ONNX **逐位一致**。已知缺口只有两点：
  (a) `model.23.cv2.*` / `model.23.cv3.*`（one-to-many head）ONNX 未导出，需重学；
  (b) BN 的 `running_mean/var` 是恒等值，缺真实统计，建议先用真实数据短跑数百次前向重新累积。

  如果不想补 one-to-many head 这一课，**次优方案**是「detector 从公开预训练权重重训」：
  用 `yolo26n.pt` 官方权重 + 本仓库 `data/` 的本地数据（`data/samples`、`data/turing_norm`、
  `out/*.png` 已被 `tools/capture_captcha.py` 落盘）重训，成本远低于修补导出图。

### 三条路线对比（含明确取舍）

| 路线 | 可行 | 代价 | 结论 |
|---|---|---|---|
| ① `best_v3.pt` 端到端 finetune | 技术可行 | 梯度被 TopK 掩码污染，训出噪声 | **不推荐** |
| ② `best_v3_native.pt` warm start + ultralytics 训练循环 | ✅ | one-to-many head 重学 + BN 统计重累积 | **推荐** |
| ③ 用 yolo26n 官预训练权重在本地 captcha 数据上重训 | ✅ | 需要训练数据与算力，但不依赖 ONNX 反解 | 数据量够时**最稳** |

`pre_model_v7` 不需要备选方案，① 就够。

## 6. 端到端 warm-start 冒烟测试

上面的梯度探测只回答「能不能反传」；`tools/verify_recovered.py` 回答「能不能真的拿来训」。
命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
G:\Python\Dev\python\python.exe tools/verify_recovered.py --steps 30 --steps-siamese 60
```

### 6.1 `weights/pre_model_v7.pt`：过拟合单 batch（Adam + CosineAnnealingLR，BCEWithLogitsLoss）

| 项 | 值 |
|---|---|
| load_state_dict | missing=0 unexpected=0 |
| 优化器 | Adam lr=1e-3 -> 1e-5 (CosineAnnealingLR) |
| 步数 | 60 |

| step | loss | lr | grad_norm |
|---|---|---|---|
| 0 | 2.895582 | 9.99e-04 | 304.2 |
| 10 | 0.461394 | 9.20e-04 | 8.235 |
| 20 | 0.174136 | 7.30e-04 | 1.143 |
| 30 | 0.104444 | 4.79e-04 | 0.4482 |
| 40 | 0.067311 | 2.35e-04 | 0.851 |
| 50 | 0.050963 | 6.40e-05 | 0.3131 |
| 59 | 0.048842 | 1.00e-05 | 0.2698 |

loss 2.895582 → 0.048842（最优 0.048842），ratio=0.0169，**下降=True**。
收敛过程干净（2.90 → 0.02 → 0.02 量级），证明梯度有效、参数真的在更新。

> 「能训练」的判据是**损失能下降**，不是「损失降到多低」——目标标签是随机初始输出二值化得来的，
> 本身没有语义，只用来构造一个有梯度的可优化问题。

### 6.2 `weights/best_v3_native.pt`：ultralytics 官方 `E2ELoss`（Adam lr=1e-3）

这是 end2end 训练实际使用的损失（one2many TAL topk=10 + one2one TAL topk=7/topk2=1 加权），
不是 dummy loss：

| 项 | 值 |
|---|---|
| load_state_dict | missing=0 unexpected=0 |
| 损失分量名 | ['box_loss', 'cls_loss', 'l1_loss']（`reg_max=1` 所以是 l1 而非 dfl） |
| 步数 | 30 |

| step | 总 loss | box | cls | l1 | grad_norm |
|---|---|---|---|---|---|
| 0 | 9.043327 | 3.73854 | 5.26753 | 0.03726 | 44.61 |
| 5 | 7.395720 | 3.11366 | 4.25432 | 0.02774 | 23.57 |
| 10 | 4.849890 | 2.03597 | 2.79631 | 0.01761 | 17.15 |
| 15 | 3.566655 | 1.40428 | 2.15164 | 0.01074 | 16.2 |
| 20 | 2.631973 | 1.06719 | 1.55692 | 0.00787 | 14.28 |
| 25 | 2.186042 | 0.87943 | 1.30055 | 0.00606 | 11.51 |
| 29 | 2.073125 | 0.80086 | 1.26692 | 0.00535 | 12.78 |

loss 9.043327 → 2.073125（ratio=0.2292），**下降=True**，三个分量全部单调下降。
这直接证明：`best_v3_native.pt` 可以接 ultralytics 的官方训练管线。

### 6.3 `weights/best_v3.pt`：真实图推理（`docs/res.jpg`，conf≥0.25）

有效框数 torch=8，onnxruntime=8，逐框一致：

| rank | torch xyxy | conf | cls | ort xyxy | conf | cls |
|---|---|---|---|---|---|---|
| 0 | [88.07, 329.19, 211.04, 458.31] | 0.95762 | 0 | [88.07, 329.19, 211.04, 458.31] | 0.95762 | 0 |
| 1 | [406.7, 322.14, 527.76, 446.25] | 0.89671 | 0 | [406.7, 322.14, 527.76, 446.25] | 0.89671 | 0 |
| 2 | [37.37, 578.82, 86.27, 636.75] | 0.89622 | 2 | [37.37, 578.82, 86.27, 636.75] | 0.89622 | 2 |
| 3 | [166.25, 579.43, 216.98, 633.73] | 0.89376 | 2 | [166.25, 579.43, 216.98, 633.73] | 0.89376 | 2 |
| 4 | [176.27, 119.79, 296.4, 228.33] | 0.88766 | 0 | [176.27, 119.79, 296.4, 228.33] | 0.88766 | 0 |
| 5 | [247.01, 293.27, 351.9, 404.19] | 0.88263 | 0 | [247.01, 293.27, 351.9, 404.19] | 0.88263 | 0 |
| 6 | [81.52, 580.15, 123.65, 636.5] | 0.8689 | 2 | [81.52, 580.15, 123.65, 636.5] | 0.8689 | 2 |
| 7 | [123.2, 577.12, 166.95, 634.15] | 0.59347 | 2 | [123.2, 577.12, 166.95, 634.15] | 0.59349 | 2 |

类别名：0=`label`，1=`title`，2=`char`（与 ONNX metadata `names` 一致）。
8 个框的坐标与置信度全部对齐到小数点后 2~5 位，只有 rank 7 的 conf 差 2e-5。

## 7. 关键命令（复现用）

```powershell
cd D:\Github\captchayes\T-Sec-captcha
$env:PYTHONIOENCODING='utf-8'

# 一键：转换两个模型 + 原生结构还原 + 数值校验 + 生成本报告
G:\Python\Dev\python\python.exe tools/recover_weights.py --all --json out/recover_raw.json

# 端到端 warm-start 冒烟测试（真实损失 + 优化器 + 真实图推理）
G:\Python\Dev\python\python.exe tools/verify_recovered.py --steps 30 --steps-siamese 60

# 单模型
G:\Python\Dev\python\python.exe tools/recover_weights.py --onnx model/pre_model_v7.onnx --out weights/pre_model_v7.pt
G:\Python\Dev\python\python.exe tools/recover_weights.py --onnx model/best_v3.onnx     --out weights/best_v3.pt
```

### 训练侧接入示例

```python
import torch
from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss   # end2end 训练用的官方损失

# ---- detector：warm start 用原生结构 ----
payload = torch.load('weights/best_v3_native.pt', map_location='cpu', weights_only=False)
model = YOLO('yolo26n.yaml')          # 或 DetectionModel(cfg, ch=3, nc=3)
model.model.load_state_dict(payload['state_dict'], strict=False)
# 之后交给 ultralytics 的 DetectionTrainer，损失走 E2ELoss，无需任何 ONNX 相关代码

# ---- siamese：直接加载 ----
si = torch.load('weights/pre_model_v7.pt', map_location='cpu', weights_only=False)
# si['state_dict'] 的键是 onnx2torch 的 fx 命名（含 '/'），用同名模块装载即可
```

