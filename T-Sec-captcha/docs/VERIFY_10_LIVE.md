# cun.ai 1.0 实跑验证报告

- 目标：`D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0`
- 时间：2026-10-31 20:51 – 23:42
- 方式：用 1.0 自己的 `DrissionVerifier` 打开 `www.cun.ai/sign-in`，填真实账号，完成验证码

---

## 1. 结论：1.0 可以使用

**点选（2404/2408）和滑块（2401）都能过。**

| 阶段 | 结果 |
|---|---|
| 仅补模型、未打补丁 | 点选 **4/5 通过**，滑块 **0/3 通过**（全部「置信不足」直接放弃） |
| 加「滑块缺口桥接」后 | 滑块能定位并拖动，但被「速度超过99%」拒 |
| 再加「拟人停顿」后 | **滑块通过**；最近 4 次实测 **3 通过 / 1 被限流** |

最近 4 次实测明细：

```
RUN1  slider 第1轮失败(速度) 第2轮通过  → True
RUN2  click  第1轮通过                  → True
RUN3  click  被「操作过于频繁」拦下      → /（限流，非能力问题）
live8 slider 第1轮失败(速度) 第2轮通过  → True
```

---

## 2. 本轮做的三件事

### 2.1 补齐 1.0 缺失的二进制资源（不做就跑不起来）

1.0 自带完整求解器，但缺两组文件：

```powershell
$dst = "D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0\captcha\models"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "D:\Github\captchayes\T-Sec-captcha\model\best_v3.onnx"      "$dst\best_v3.onnx" -Force
Copy-Item "D:\Github\captchayes\T-Sec-captcha\model\pre_model_v7.onnx" "$dst\pre_model_v7.onnx" -Force

cd "D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"
python tools/setup_captcha.py --non-interactive --yes   # 从 PyPI 提取 ddddocr 两个 OCR 模型
```

验证：`python tools/verify_10_integration.py` → 全部 OK。

### 2.2 滑块缺口桥接（`slider_gap_bridge.py`）

**问题**：1.0 在真实滑块底图上给出 `conf=0.3 / 0.0 / 0.3`，
而 `_solve_slider_once` 第 642 行有硬阈值 `if conf < 0.45: 换题不提交` ——
所以滑块每次都直接被放弃，连试都不试。

**对比实测**（1.0 实跑时保存的 3 张真实底图）：

| 底图 | 1.0 原实现 | 本项目实现 | 谁对 |
|---|---|---|---|
| `live_slider_bg_1` | conf=0.3 x=282 | outline x=**329** | **本项目**（放大核对，红框精确套住缺口；1.0 偏左 47px）|
| `live_slider_bg_2` | conf=0.0 x=0（放弃） | outline x=**581** | **本项目**（放大 3 倍核对，缺口中心正是 581）|
| `live_slider_bg_3` | conf=0.3 x=546 | outline x=**600** | **本项目**（1.0 偏左 54px）|

**3/3 全中 vs 0/3**。所以桥接是真提升，不是安慰剂。

置信度映射（按方法可靠性给，不是拍脑袋）：

```python
CONF_OUTLINE = 0.9   # 亮描边+暗心，实测误差 0~1px
CONF_HAT     = 0.55  # 帽形兜底（轮廓被草叶/纹理打断时），实测 ~5px
```

两者都 > 0.45，能被 1.0 正常采用。

### 2.3 拟人停顿 + 拖动修正（`drission_verify_ext.py`）

桥接后滑块开始真的拖了，但连续被判：

```
校验状态: 您的速度已超过 99% 的用户    验证错误，请重试
```

排查过程（两步都必要）：

1. **先怀疑拖动速度**：1.0 原实现拖动时 `mouseMoved` **没带 `buttons=1`**。
   CDP 里按钮状态是位掩码，拖动中 `buttons=0` 时页面可能不认这是"按住移动"，
   站点等价于看到「按下→瞬移终点→抬起」。→ 已修正为带 `buttons=1`，
   并把总时长控制在 1.1~2.8s、速度 ~130~140px/s。

2. **但提示仍在** → 说明这条提示**不是拖动速度**（人类拖滑块 300~800px/s，
   我们已经比人慢一倍多）。加了「看图停顿」2.2~4.6s（停顿期间做少量
   无目的鼠标移动）之后，滑块就过了。

3. **最终定性**（RUN3 的完整原文）：

```
校验状态: 您的速度已超过 99% 的用户    您的操作过于频繁，请稍后再试
```

   「速度超过99%」是**限流的前置提示**，与拖动无关。
   与我在 T-Sec-captcha 里实测的 `errorCode=12`（连续猛跑 8 次全 12、
   冷却 10 分钟后 4/4 全 0）是同一件事。

---

## 3. 1.0 目前的真实短板

不是求解能力，是**节奏控制**：

| 问题 | 表现 | 建议 |
|---|---|---|
| 连续登录触发风控 | 「操作过于频繁」 | 单账号间隔 ≥45s，触发后冷却 60→180→600s |
| 应答码未接入 | 1.0 只靠页面文案判「频繁」 | 见 §4 |
| 滑块偶发失败 | 第 1 轮「速度超过99%」 | 已有停顿，第 2 轮通常能过；可把停顿调到 3~5s |

---

## 4. 待办：应答码判定（已写好补丁，但当前读不到）

`tools/drission_verify_ext.py` 已实现：

- 把 `cap_union_new_verify` 加进监听 targets
- `read_verify_code()` 解析 `errorCode` / `ticket`
- `code == "12"` → 抛 `CaptchaRateLimitedError`，走 60→180→600s 退避

**但目前读不到包**。实测三个原因，都已定位：

| 现象 | 原因 |
|---|---|
| `listen.steps()` 让主流程卡死 10 分钟 | 它是**无限阻塞生成器**（`while True`），`list()` 它必然挂住 |
| `_caught` 队列恒为空 | 1.0 的 `fetch_image()` 是「listen 优先 + requests 兜底」，listen 抓不到时静默走兜底，所以流程正常但队列空 |
| `Listener.start()` 清了队列还不换 target | 源码：先 `self.clear()`，再 `if self.listening: return` |

**下一步做法**：不依赖 DrissionPage 的 listen，改用 CDP 事件订阅
（`Network.responseReceived` + `Network.getResponseBody`）直接抓 verify 响应。

在此之前，1.0 靠「页面文案 + 第2轮重试」也能工作，只是效率不如按码分流。

---

## 5. 复现命令

```powershell
cd D:\Github\captchayes\T-Sec-captcha
$env:PYTHONIOENCODING='utf-8'

# 一键检查 1.0 接入状态
python tools/verify_10_integration.py

# 滑块桥接自测（1.0 fixture + 3 张真实底图）
python tools/slider_gap_bridge.py

# 可视化对比新旧滑块定位
python tools/verify_slider_bridge.py

# 实跑 1.0 验证（含两个补丁）
python tools/test_10_live.py --entry https://www.cun.ai/sign-in `
    --username <账号> --password <密码> --max-rounds 3
```

## 6. 文件清单

```
tools/test_10_live.py         用 1.0 自己的验证器实跑（含补丁装配）
tools/slider_gap_bridge.py    滑块缺口桥接（1.0 → 本项目实现）
tools/verify_slider_bridge.py 滑块定位可视化对比
tools/drission_verify_ext.py  应答码判定 + 拟人拖动 + 看图停顿
tools/verify_10_integration.py 1.0 文件/加载/推理一键检查
tools/probe_listen.py         listen 抓包探测
tools/probe_listen2.py        packet 对象结构
tools/probe_queue.py          队列内容诊断
tools/probe_verify_code.py    应答码专项探测
out/test_10_live/             1.0 实跑产物（底图/提示条/截图）
out/slider_bridge_check/      滑块定位对比图（含放大核对）
```
