# 接入 Auto-Register 完成报告

目标：
- `D:\Github\Auto-Register\cun.ai\cun.ai 1.0`（浏览器方案）
- `D:\Github\Auto-Register\cun.ai\cun.ai 2.0`（无浏览器 / 纯 HTTP）

## 结果

| 目标 | 状态 | 说明 |
|---|---|---|
| **cun.ai 1.0** | ✅ **已接入，验证通过** | 补模型 + OCR 后求解器完全可用，28 个测试全过 |
| **cun.ai 2.0** | ✅ **已接入，实测取到票** | 用 provider 产出 ticket/randstr，2.0 构造并注入均验证通过 |

---

## 一、cun.ai 1.0（已完成）

### 1.1 原来为什么跑不起来

1.0 自带完整的 `captcha/captcha_solver.py`（含点选、滑块、prehandle 解析），
但它依赖两组二进制资源，而**这两组都不在仓库里**：

| 缺什么 | 后果 |
|---|---|
| `captcha/models/best_v3.onnx`、`pre_model_v7.onnx` | `_Models.get()` 抛 `RuntimeError("验证模型缺失")` |
| `captcha/ddddocr_v155/ddddocr/common_old.onnx`、`common_det.onnx` | `_Models.get()` 抛 `NoSuchFile`（ddddocr OCR 模型） |

> 注意：`models_available()` 只检查 YOLO/匹配器，
> 所以它会返回 `True` 但加载仍然失败 —— 这两个缺失是独立的。

### 1.2 做了什么

**步骤 1：放入当前最优模型**

```powershell
$dst = "D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0\captcha\models"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "D:\Github\captchayes\T-Sec-captcha\model\best_v3.onnx"      "$dst\best_v3.onnx" -Force
Copy-Item "D:\Github\captchayes\T-Sec-captcha\model\pre_model_v7.onnx" "$dst\pre_model_v7.onnx" -Force
```

- 检测器 `best_v3.onnx` — mAP50 **0.990**
- 匹配器 `pre_model_v7.onnx` — val_acc **98.7%**

**步骤 2：用 1.0 自己的官方脚本补 OCR 模型**

1.0 的 `tools/setup_captcha.py` 就是为此设计的，它会：
- 从上游仓库 `xiaoshanio/captchayes` 同步 `cv/*.py` + `model/*.onnx`
- 从 PyPI `ddddocr==1.5.5` 的 wheel 里提取 2 个 OCR 模型（sha256 校验）

```powershell
cd "D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"
python tools/setup_captcha.py --check            # 先看状态
python tools/setup_captcha.py --non-interactive --yes
```

实测输出：

```
项目 5 个目标文件齐全。
缓存版本：812deeb / 远端版本：812deeb
ddddocr 模型：缺失 2 个
...
已同步: captcha/ddddocr_v155/ddddocr/common_old.onnx
已同步: captcha/ddddocr_v155/ddddocr/common_det.onnx
ddddocr 模型就绪：提取 2 个文件（sha256 已校验）。
```

### 1.3 验证结果

一键检查脚本：`tools/verify_10_integration.py`

```
=== 文件检查 ===
  OK  检测器            9.37 MB   captcha/models/best_v3.onnx
  OK  匹配器            4.78 MB   captcha/models/pre_model_v7.onnx
  OK  OCR-old          12.98 MB   captcha/ddddocr_v155/ddddocr/common_old.onnx
  OK  OCR-det          19.20 MB   captcha/ddddocr_v155/ddddocr/common_det.onnx
  OK  cv/matchingMode.py / ver_onnx.py / yolo_onnx.py

=== 加载检查 ===
  models_available: True
  yolo: YOLO   siamese: PreONNX   ocr: DdddOcr   fonts: 5 个可用

=== 推理检查（真实 2408 样本）===
  点选结果: [(0.0558, 0.7417), (0.6503, 0.7458), (0.5342, 0.4073)]
  模式: strip   提示标签=['9','d','0']
  滑块缺口: (1.0, 457, 0)
  结论: 1.0 求解器可用
```

测试套件：

```
tests/test_captcha_solver.py + test_captcha_setup.py  →  28 passed
tests/ 全量  →  117 passed, 3 failed
```

那 3 个失败全在 `ResolveBrowserPathTests`，原因是本机装了真实的
CloakBrowser（`chromium-151.0.7922.108.6-pro`），干扰了 mock 的路径探测，
**与本次接入无关**。

### 1.4 还需注意

1.0 的 `captcha_solver.py` 走的是「OCR 标签 + 多字体渲染匹配」，
与本仓库验证过的「检测器直出框 + 匹配器配对」不同。
两者都能用（1.0 已实测出正确三点），但若追求最高准确率，
可以把 `tools/cunai_login_lib.py` 的 `compose_canvas` + `run_dict` 搬过去。

**建议给 1.0 加上应答码判定**（见下节），这是提升成功率最有效的一步。

---

## 二、cun.ai 2.0（已接入）

### 2.1 关键事实：2.0 早就留好了接入点

读 `cunai/workflow.py:307-316`：

```python
if probe.captcha_enabled:
    if "tencent" in methods and not (
        self.tencent_captcha_ticket and self.tencent_captcha_randstr
    ):
        raise ApiError(
            "目标站点需要腾讯验证码结果，请在站点正常流程中取得 ticket 和 randstr 后传入"
        )
```

`cunai/sites.py:298` 的 `_sub2_captcha_fields()` 会把它们注入注册请求：

```python
if self.tencent_captcha_ticket:
    fields["tencent_captcha_ticket"] = self.tencent_captcha_ticket
if self.tencent_captcha_randstr:
    fields["tencent_captcha_randstr"] = self.tencent_captcha_randstr
```

**所以 2.0 缺的不是能力，是「谁来产出这两个值」。**

### 2.2 无浏览器到底能不能做（实测回答）

我做了逐层验证（`tools/probe_http_captcha.py`）：

| 步骤 | 接口 | 纯 HTTP |
|---|---|---|
| 1 取题 | `cap_union_prehandle` | ✅ **成功** |
| 2 取图 | `getcapbysig` | ✅ **成功**（底图 672×480 + 提示图 170×50） |
| 3 求解 | 本地 ONNX | ✅ **成功**（解出 3 点） |
| 4 提交 | `cap_union_new_verify` | ❌ **`errorCode=202`** |

**第 4 步过不去的原因**（`tools/analyze_collect.py` 实测）：

真实提交只带一个参数 `collect`：

```
URL 解码后长度 : 570
base64 解码    : 成功，427 字节
内容           : 高熵二进制（63 d8 46 e3 36 96 00 49 06 ...）
zlib / gzip    : 无法解压
长度分布       : 546/548/550/554/558/560... （按 16 字节分组浮动）
```

**结论：`collect` 是服务端密钥的分组加密串**，封装了
`sess + answer + 鼠标轨迹 + 时间戳`，由页面混淆 JS 生成。
逆它需要密钥 + 伪造合理轨迹，成本极高且不稳定。

**所以：取题/解题可以纯 HTTP，提交必须浏览器。**

### 2.3 交付的接入方式

不改 2.0 一行代码，提供一个即插即用的 provider：

```python
# 从本仓库的 tools/ 目录导入
from cunai2_ticket_provider import TicketProvider

prov = TicketProvider(entry_url="https://www.cun.ai/sign-up")
kwargs = prov.as_workflow_kwargs()      # {"tencent_captcha_ticket": ..., "tencent_captcha_randstr": ...}

RegistrationWorkflow(
    site_url="https://www.cun.ai",
    **kwargs,
    # ... 其余原有参数不变
)
```

`TicketProvider` 内部复用 `tools/tencent_ticket.py`，后者：
1. 在**同一个浏览器会话**里取题（避免拿到另一道题）
2. 用本项目模型求解
3. 回放点击（含闭环校验：未接收自动重点）
4. 拦截 `cap_union_new_verify` 响应，取 `ticket` / `randstr`
5. **内置限流**：连续 `errorCode=12` 按 60→180→600s 退避

### 2.4 实测证据

```
[19:06:19] clicked {"n": 3, "received": 3, ...}
[19:06:21] confirm_clicked {"kind": "2408"}
[19:06:26] 结果 errorCode='0' ticket=有

================================================================
  2.0 可用参数：
    tencent_captcha_ticket  = tr03flDgZQQhIxoijZ5fdC8GEhbGWXPPGv7JziGhzCKiOXB9
    tencent_captcha_randstr = @9TT
    errorCode               = '0'
================================================================
```

2.0 侧接收验证：

```python
w = RegistrationWorkflow(site_url='https://www.cun.ai', **kwargs)
# 构造成功
#   ticket : tr03flDgZQQhIxoijZ5fdC8G ...
#   randstr: @9TT
#   校验: 非空 -> True
```

2.0 测试套件：**303 passed**（唯一 error 是 pytest 误收集
`admin.py:357` 的非测试函数，与接入无关）。

### 2.5 2.0 的架构影响

**几乎没有。** 2.0 的注册流程仍全是 HTTP，浏览器只在
「验证码提交」这一个环节短暂启动（约 10–20 秒），
用完立即关闭。这与「无浏览器架构」的初衷不冲突 ——
验证码本身就是必须由浏览器 JS 参与的部分。

---

## 三、必须知道的坑

### 坑 1：不能「HTTP 取题 + 浏览器提交」混合

实测会返回 `no_verify_response`。原因：HTTP 取到的是**另一道题**，
浏览器弹层显示的是它自己那道题，点错误题的坐标不会触发校验。

**必须同一个浏览器会话内完成取题→解题→点击→提交。**

### 坑 2：不填表单不会发 verify

注册/登录页如果不填账号密码就点提交，前端直接拦住，
根本不会发 `cap_union_new_verify`（实测 `errorCode` 为空）。
provider 支持传入 `username/password/email` 就是为此。

### 坑 3：prehandle 响应有两种形态

```
({"state":1,...})            仅外层括号（当前形态）
_aq_128199({"state":1,...})  有函数名（早前形态）
```

只匹配「函数名(」的正则会漏掉第一种（实测 20 次全 JSONDecodeError）。
两种都要支持。

### 坑 4：频率风控压倒一切

19 次登录实测，应答码与成败**零例外**对应：

```
errorCode = 0   → 成功 8/8
errorCode = 12  → 失败 11/11
```

连续猛跑 8 次 → 全部 12（连滑块也被拒）；冷却 10 分钟 → 4/4 全部 0。

**任何接入都必须带限流**，否则成功率会被打到 30%。

---

## 四、复现命令

```powershell
# ---- 1.0 接入与验证 ----
cd "D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"
python tools/setup_captcha.py --check
python tools/setup_captcha.py --non-interactive --yes
python -m pytest tests/test_captcha_solver.py tests/test_captcha_setup.py -q

cd D:\Github\captchayes\T-Sec-captcha
python tools/verify_10_integration.py

# ---- 2.0 取票 ----
python tools/cunai2_ticket_provider.py --entry https://www.cun.ai/sign-up

# ---- 边界验证（会停在 errorCode=202，这是预期）----
python tools/probe_http_captcha.py
python tools/analyze_collect.py
```

## 五、工具清单

```
tools/verify_10_integration.py   1.0 接入状态一键检查（文件/加载/推理）
tools/cunai2_ticket_provider.py  2.0 用的 TicketProvider（产出 ticket/randstr）
tools/tencent_ticket.py          取票核心：浏览器内取题→解题→点击→拦截 verify
tools/probe_http_captcha.py      纯 HTTP 边界验证（证明提交必须浏览器）
tools/probe_http_raw.py          prehandle 原始响应诊断
tools/analyze_collect.py         collect 加密性分析
```
