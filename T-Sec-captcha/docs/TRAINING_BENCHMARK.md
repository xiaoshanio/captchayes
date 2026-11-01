# 本机训练吞吐基准（Xeon E5-2660 v2 ×2，无 GPU）

## 1. 硬件事实（实测）

| 项 | 值 | 影响 |
|---|---|---|
| CPU | 2× Intel Xeon E5-2660 v2 @ 2.20GHz | **只有 AVX，无 AVX2/FMA/BMI** |
| 核心 | 10C/20T ×2 | `torch.get_num_threads()` = 20 |
| 内存 | 64 GB | 可 `cache=ram` |
| GPU | NVIDIA GeForce GT 705，驱动 23.21.13.9135 (CUDA 9.1) | `torch.cuda.is_available()` = **False** |
| OS | Windows Server 2025 Datacenter | — |
| Python | 3.14.4 | onnxruntime/torch/ultralytics 均有 cp314 轮子 |

**关键约束：这台机器没有可用 GPU，且 CPU 缺少 AVX2。**

---

## 2. 三个必须知道的坑（都会让训练失败）

### 坑 1：polars 非法指令崩溃 `0xC000001D`

`ultralytics` 每个 epoch 结束用 `polars` 写 `results.csv`。
polars 官方 wheel 要求 AVX2，本机执行到内核即 **硬终止**（无 traceback）。

**修法**：`pip install "polars[rtcompat]"`（可移植运行时，不依赖 AVX2）。
定位脚本：`train/epoch_end_probe.py`。

### 坑 2：CPU 上的 AMP

`amp=True` 会走 `torch.amp.autocast`，在老 CPU 上同样有非法指令风险。
**修法**：`amp=False`（`train_detector.py` 已默认关闭）。

### 坑 3：`POLARS_SKIP_CPU_CHECK` 只是关告警

它不解决内核崩溃，必须装 `rtcompat`。

---

## 3. 实测吞吐

### 3.1 YOLO26n 检测器（`runs/detect/selfgen/results.csv`）

配置：`imgsz=640, batch=8, workers=4, device=cpu, cache=ram`，800 张训练图。

| epoch | 耗时(s) | box_loss | cls_loss | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|---|---|
| 1 | 363 | 2.674 | 4.091 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 726 | 1.931 | 2.526 | 0.669 | 0.309 | 0.583 | 0.658 |
| 3 | 1086 | 1.611 | 1.798 | 0.837 | 0.463 | 0.752 | 0.708 |
| 4 | 1443 | 1.490 | 1.458 | 0.887 | 0.565 | 0.814 | 0.758 |
| 5 | 1805 | 1.379 | 1.267 | 0.924 | 0.638 | 0.798 | 0.843 |
| 6 | 2165 | 1.277 | 1.083 | 0.947 | 0.664 | 0.871 | 0.859 |
| 8 | 2941 | 1.155 | 0.887 | 0.972 | 0.727 | 0.911 | 0.917 |
| 10 | 3761 | 1.129 | 0.776 | 0.975 | 0.773 | 0.910 | 0.922 |
| 14 | 5198 | 1.031 | 0.654 | 0.984 | 0.816 | 0.938 | 0.941 |
| 19 | 7153 | 0.935 | 0.558 | **0.987** | 0.814 | 0.951 | 0.947 |

**结论**：
- 单 epoch ≈ **376 秒（6.3 分钟）**（19 轮累计 7153 秒）
- 15 epoch ≈ **1 小时 35 分钟**
- 19 epoch ≈ **2 小时**
- mAP50 在 epoch 6 就到 0.947，epoch 19 到 0.987 —— **收敛很快，不必跑满 30**
- ⚠️ 但注意：**mAP 继续涨，端到端准确率不涨**（见 `TRAINING_AND_DATA_REFRESH.md` 5.4 节），
  瓶颈在数据版式而非训练轮数。

另一档实测（`bench/`，已清理）：
- `imgsz=320, batch=16`：约 12.6 s/epoch（小样本）
- `imgsz=640, batch=8`：约 30.4 s/epoch（小样本）

### 3.2 Siamese 匹配器（`train_matcher.py`）

配置：`112×112 双输入, batch=64, threads=10, workers=0, device=cpu`。
数据：11228 训练对 / 1640 验证对。

| epoch | 耗时(s) | train loss | val_acc |
|---|---|---|---|
| 1 | 358.6 | 0.5270 | 0.2500 |
| 2 | 253.5 | 0.2365 | 0.8927 |
| 3 | 258.7 | 0.1194 | 0.9646 |
| 5 | 224.5 | 0.0484 | 0.9750 |
| 6 | 213.7 | 0.0328 | 0.9787 |
| 8 | 156.8 | 0.0197 | **0.9774** |

**结论**：
- 单 epoch ≈ **200~260 秒**（3.5~4.5 分钟）
- 8 epoch ≈ **30 分钟**，val_acc 达 **97.7%**
- **匹配器在这台机器上完全可训**

### 3.3 推理（CPU 单进程）

| 项 | 耗时 |
|---|---|
| YOLO 单张 | ~60 ms |
| 端到端（检测+匹配） | ~205~320 ms |

---

## 4. 结论与建议

| 问题 | 答案 |
|---|---|
| 匹配器能否在本机训练？ | ✅ 能。8 epoch / 30 分钟到 97.7% |
| 检测器能否在本机训练？ | ✅ 能，但慢。15 epoch / 1.5 小时到 mAP50 0.984 |
| 必须上 GPU 吗？ | 不必须。但 30+ epoch 或大数据集（>5000 张）建议上 GPU |
| GPU 能快多少？ | T4 级别约快 **15~25 倍**；30 epoch 从 3 小时降到 ~10 分钟 |

### 本机调优建议

1. `pip install "polars[rtcompat]"` —— 必须
2. `$env:POLARS_SKIP_CPU_CHECK = "1"` —— 关告警
3. `amp=False` —— 必须
4. `cache=ram` —— 64G 内存够用，省掉重复解码
5. `workers=4` —— Windows 下别超过 8（DataLoader 复制进程，句柄易炸）
6. `torch.set_num_threads(10)` —— 20 逻辑核全给反而抢核
7. `--no-val` 可省时间，但会丢失训练中验证（不推荐，mAP 曲线是判断收敛的唯一依据）
8. 想快速跑通流程：`imgsz=320`（约 2.4 min/epoch）
9. 想拿最终精度：`imgsz=640`（必须，对齐 runtime letterbox）

### 时间预算参考（本机 CPU）

| 目标 | 配置 | 预计耗时 |
|---|---|---|
| 跑通流程验证 | imgsz=320, 10 epoch, 800 张 | ~25 分钟 |
| 检测器可用 | imgsz=640, 6 epoch, 800 张（mAP50 0.947） | ~36 分钟 |
| 检测器收敛 | imgsz=640, 19 epoch, 800 张（mAP50 0.987） | ~2 小时 |
| 匹配器可用 | 112×112, 8 epoch, 11k 对（val_acc 0.977） | ~30 分钟 |
| **全链路（本次实测）** | 检测器 19 epoch + 匹配器 8 epoch | **约 2.5 小时** |

---

## 5. 子代理补充发现（ultralytics 静默改写参数）

bench 子任务独立复现并补充了两条**会让人调参调不出结论**的坑：

1. **`device=cpu` 时强制 `workers=0`**
   `engine/trainer.py:179` 硬写 `self.args.workers = 0`（注释称 "faster CPU training"）。
   你传的 `--workers 4` 在 CPU 上**根本不生效**。

2. **`select_device()` 覆盖线程数**
   内部执行 `torch.set_num_threads(min(8, cpu_count - 1))` = **8**，
   覆盖用户设置的任何 `torch.set_num_threads` 值。

**规避方法**：子类化 `DetectionTrainer`，在 `_setup_train()` 前后重新赋值
（bench 子任务的 `bench/bench_train.py` 已实现并验证 workers 0/4/8、threads 5/10/20 均能实际生效）。

⚠️ 注意：本报告第 3 节的时间数据是**默认配置**下的实测（即 workers 被强制为 0、
threads 被设为 8），这也解释了为什么 640/bs8 的 epoch 时间偏长。
若要按自己的 workers/threads 跑，需先做上述子类化。

3. **polars 崩溃的替代修法**
   子任务用 `sys.modules` 注入 stub 规避；本仓库推荐更干净的官方修法：
   `pip install "polars[rtcompat]"`（可移植运行时，不依赖 AVX2）。
   两者都可，stub 作为兜底。

4. **同配置重复测量有约 40% 方差**
   疑似双路 NUMA 跨节点线程调度抖动。做基准对比时应取 min/median 而非单次值。
