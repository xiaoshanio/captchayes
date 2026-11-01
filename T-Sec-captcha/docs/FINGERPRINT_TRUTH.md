# 人机判定的真实决定因素（更正）

> 本文更正前一版文档里的错误结论。

---

## 1. 更正：什么不是人机判定

前一份文档把下面这条当成了「AI 人机判定」的结论，**这是错的**：

```
校验状态: 您的操作过于频繁，请稍后再试
```

这条是**限流**（同一 IP / 账号短时间请求过多），
不是「你是机器人」的判定。它和指纹、行为特征无关，
只反映请求频率。处理方式就是退避，不需要改任何伪装。

**结论：不要把「操作过于频繁」当作人机判定结果。**

---

## 2. 反证：那条「速度」提示也不是真按速度算的

我之前还怀疑「您的速度已超过 99% 的用户」是拖动速度问题，也不成立：

| 拖动速度 | 人类参考 | 站点反馈 |
|---|---|---|
| 136 px/s | 人类 300~800 px/s（我们**比人慢一倍多**） | 「速度已超过 99% 的用户」 |

如果真按 px/s 判定，比人慢一倍的拖动不可能「超过 99% 用户」。

所以这条提示是**被识别为非人之后返回的固定话术**，
与拖动快慢无关 —— 这也符合你说的：判定看的是浏览器与 IP，不是速度。

（另外它常和限流同时出现：
`您的速度已超过 99% 的用户    您的操作过于频繁，请稍后再试`
说明这两句是一组风控话术，不是独立的速度评判。）

---

## 3. 真正决定通过的因素（实测证据）

### 3.1 浏览器指纹

`tools/fingerprint_audit.py` 对比了两个引擎：

| 字段 | DrissionPage（1.0 用） | CloakBrowser | 判断 |
|---|---|---|---|
| `webdriver` | False | False | ✅ 都没暴露 |
| `cdc_` 残留键 | 无 | 无 | ✅ 干净 |
| `plugins` / `mimeTypes` | 5 / 2 | 5 / 2 | ✅ 正常 |
| `platform` | Win32 | Win32 | ✅ |
| `languages` | zh-CN,zh | zh-CN | ✅ |
| **`screen`** | **800×600** ❌ | 1366×768 | **明显破绽** |
| `outer` | 1920×1080（改后） | 1050×700 | — |
| `hardwareConcurrency` | 40 | 12 | ⚠️ 40 核偏服务器 |
| `deviceMemory` | 32 GB | 16 GB | ⚠️ |
| `webglVendor` | NVIDIA | Intel | 都可信 |
| `timezone` | Asia/Shanghai | Asia/Shanghai | ✅ |

**关键破绽：无头模式 `screen = 800×600`。**
这是 Chromium 无头虚拟显示器的默认值，真实用户几乎不会用这个分辨率，
反爬系统一眼就能识别。

### 3.2 已修复

`drission_verify_ext.harden_fingerprint()`：

| 模式 | 修复前 | 修复后 |
|---|---|---|
| 无头 | 800×600 | **1920×1080**（CDP `Emulation.setDeviceMetricsOverride`）|
| 有头 | 2560×1440（真实） | **保留真实值**（覆盖反而变假）|

实测日志：

```
已覆盖 screen -> 1920x1080（原始 800x600 是反爬特征）
```

修复后点选一轮即过（`test_10_live9.log`：`点选验证通过，弹窗已关闭`）。

### 3.3 出口 IP

`tools/fingerprint_audit.py` 实测两个引擎出口 IP 相同，均为：

```
152.175.65.134
国家: Singapore    ASN: AS209554 ISIF OU
proxy=false  hosting=false  mobile=false
```

问题：
- **地理位置与目标站不符**（cun.ai 面向中国大陆用户，
  新加坡 IP 长期登录会被加权）
- 该 ASN 属机房/云服务段，ip-api 虽标 `hosting=false`，
  但同段 IP 通常用于服务器

这一项**无法靠改代码解决**，只能换更合适的出口
（同城家宽 / 移动网络）。它是目前最可能的隐性扣分项。

---

## 4. 修正后的优先级

| 优先级 | 事项 | 状态 | 说明 |
|---|---|---|---|
| **1** | `screen` 800×600 | ✅ 已修 | CDP 覆盖，实测生效 |
| **2** | 出口 IP 地理位置 | ❌ 待处理 | 新加坡机房 IP，与站点用户群不符 |
| 3 | `hardwareConcurrency=40` | 可选 | 40 核像服务器，可覆盖为 8~16 |
| 4 | `deviceMemory=32` | 可选 | 同上，覆盖为 8/16 |
| 5 | 限流退避 | ✅ 已有 | 与指纹无关，纯频率问题 |
| — | ~~拖动速度~~ | 不必再调 | 已证明不是判定依据 |

---

## 5. 别再做的两件事

1. **不要继续调拖动速度/加速度曲线。**
   已降到 136px/s 仍被拒，说明这不是评分子项。
   真实的拖拽只需要「有轨迹、有加速减速」即可，过度拟人没有收益。

2. **不要把「操作过于频繁」当作能力不足。**
   它是纯频率限制，等一等就好；改指纹并不能绕过它。

---

## 6. 复现命令

```powershell
cd D:\Github\captchayes\T-Sec-captcha
$env:PYTHONIOENCODING='utf-8'

# 指纹审计（对比 DrissionPage 与 CloakBrowser）
python tools/fingerprint_audit.py

# 只测 DrissionPage
python tools/fingerprint_audit.py --only dp

# 试各种 screen 正常化手段
python tools/probe_screen.py

# 带指纹加固的 1.0 实跑
python tools/test_10_live.py --entry https://www.cun.ai/sign-in `
    --username <账号> --password <密码> --max-rounds 3
```

## 7. 相关文件

```
tools/fingerprint_audit.py          指纹审计（两个引擎对比 + IP 性质）
tools/probe_screen.py               screen 正常化手段验证
tools/drission_verify_ext.py        含 harden_fingerprint() 指纹加固
tools/test_10_live.py               1.0 实跑（已接入加固）
out/fingerprint/fingerprint.json    指纹明细
```
