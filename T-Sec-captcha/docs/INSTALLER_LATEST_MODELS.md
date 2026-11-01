# 保证 1.0 安装程序装到「最新模型」

目标：`D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0` 的安装流程必须能装上最新模型。

---

## 1. 原来的三个问题

### 问题 1：install.bat 根本不装验证码

```bat
echo [3/4] 验证码模型：当前站点注册/登录无验证，本步灰色跳过（不下载）。
set "CUN_SKIP_CAPTCHA=1"
"%PYTHON_EXE%" tools\setup_captcha.py --skip
```

注释理由是「当前站点无验证」，但这个前提**已经过时**——
cun.ai 现在明确下发 2404 中文点选 / 2408 图标点选 / 2401 滑块。

### 问题 2：默认通道只能拿到旧模型

即使不 skip，走 GitHub 也装不到最新模型：

| 位置 | best_v3.onnx | pre_model_v7.onnx |
|---|---|---|
| 本地 `main`（e23266c） | 9,827,931 | 5,012,923 |
| **origin/main（812deeb）** | **9,806,720** | **44,452,568** |
| 上游缓存 `~/.cun-captcha-upstream` | 9,806,720 | 44,452,568 |

```
git branch -vv
* main e23266c [origin/main: ahead 1]
```

新模型只在**本地未推送**的提交里，所以从 GitHub 拉到的必然是旧的
（`pre_model_v7` 甚至差 8 倍大小）。

### 问题 3：光看「文件齐全」判断不出新旧

原脚本的判断只有：

```python
def missing_targets(root):   # 文件在不在
```

文件存在就认为没问题 —— 装的是 9.8MB 还是 44MB 的模型，它看不出来。
这就是「静默装旧模型」的根源。

---

## 2. 改了什么

### 2.1 安装脚本增加「本地源」通道（推荐路径）

新增 `--source`，直接从本地开发目录复制，**不联网、不走 git**：

```powershell
python tools\setup_captcha.py --source D:\Github\captchayes\T-Sec-captcha
```

支持两种写法（自动识别）：

```
D:\Github\captchayes\T-Sec-captcha     ← 直接指向子目录
D:\Github\captchayes                   ← 指向仓库根，自动补 T-Sec-captcha
```

实测输出：

```
从本地源安装验证码逻辑（不联网）
  源：D:\Github\captchayes\T-Sec-captcha
  目标：D:\Github\Auto-Register\cun.ai\cun.ai 1.0
  已安装: captcha/cv/matchingMode.py (1871 字节)
  已安装: captcha/cv/ver_onnx.py (4888 字节)
  已安装: captcha/cv/yolo_onnx.py (4605 字节)
  已安装: captcha/models/best_v3.onnx (9827931 字节)
  已安装: captcha/models/pre_model_v7.onnx (5012923 字节)
共安装 5 个文件。

模型指纹核对：
  [OK  ] best_v3.onnx: 9827931 字节  sha256=f009f3b392f0…
  [OK  ] pre_model_v7.onnx: 5012923 字节  sha256=ba230a287e7d…
```

### 2.2 增加模型指纹登记与校验

脚本里新增 `LATEST_MODELS`，登记最新模型的字节数与 sha256：

```python
LATEST_MODELS = {
    "captcha/models/best_v3.onnx": (
        9827931,
        "f009f3b392f0962dbd02e5de08876d4c53c7cc0f0a35fb2388ca010222b31d24"),
    "captcha/models/pre_model_v7.onnx": (
        5012923,
        "ba230a287e7df110437dab923fd757290a4aadf9b376c2e73ef8480ea6fdd338"),
}
```

配套三个函数：

| 函数 | 作用 |
|---|---|
| `model_report(root)` | 逐个模型比对指纹，返回「是/否最新」 |
| `models_are_latest(root)` | 布尔汇总 |
| `cache_models_stale(cache)` | 检查**上游缓存**是否过期（路径布局不同，单独实现）|

`--check` 现在会打印：

```
模型指纹：
  [OK  ] best_v3.onnx: 9827931 字节  sha256=f009f3b392f0…
  [OK  ] pre_model_v7.onnx: 5012923 字节  sha256=ba230a287e7d…
```

缺失时：

```
  [WARN] best_v3.onnx: 缺失
  [WARN] pre_model_v7.onnx: 缺失
  → 当前不是最新模型。装最新请用：
     python tools/setup_captcha.py --source <T-Sec-captcha 路径>
```

### 2.3 上游过期时告警，绝不静默

`--non-interactive`（默认通道）在同步**之前**先比对缓存：

```
警告：上游仓库里的模型不是最新版本：
  best_v3.onnx: 9806720 字节 sha256=a9804980ad25… (最新应为 9827931 / f009f3b392f0…)
  pre_model_v7.onnx: 44452568 字节 sha256=885e01b20592… (最新应为 5012923 / ba230a287e7d…)
  —— 安装后仍是旧模型。要装最新请改用本地源：
    python tools/setup_captcha.py --source <T-Sec-captcha 路径>
非交互：文件齐全，跳过。
```

注意它同时**不会覆盖已有的新模型**（走「文件齐全，跳过」分支）。

### 2.4 install.bat 第 3 步改为真安装 + 源优先级

```bat
echo [3/4] 验证码模型：安装最新模型（站点注册/登录需要人机验证）。
if defined CUN_CAPTCHA_SOURCE (
  "%PYTHON_EXE%" tools\setup_captcha.py --source "%CUN_CAPTCHA_SOURCE%"
) else if exist "%~dp0..\..\..\captchayes\T-Sec-captcha\model\best_v3.onnx" (
  "%PYTHON_EXE%" tools\setup_captcha.py --source "%~dp0..\..\..\captchayes\T-Sec-captcha"
) else (
  "%PYTHON_EXE%" tools\setup_captcha.py --non-interactive
)
...
echo 模型指纹核对：
"%PYTHON_EXE%" tools\setup_captcha.py --check
```

优先级：`CUN_CAPTCHA_SOURCE` 显式指定 → 本机自动探测 → GitHub 回退。
装完自动打印指纹，一眼能看出是不是最新。

> `install.bat` 是 **GBK** 编码，改动必须按 GBK 回写，否则中文乱码。
> 已备份为 `install.bat.bak`。

---

## 3. 验证结果

### 3.1 新旧模型能否区分（`tools/verify_latest_models.py`）

```
=== 装的是最新模型（应判 是） ===
  [OK  ] best_v3.onnx: 9827931 字节  sha256=f009f3b392f0…
  [OK  ] pre_model_v7.onnx: 5012923 字节  sha256=ba230a287e7d…
  models_are_latest -> True

=== 装的是旧模型（应判 否，并给出期望值） ===
  [WARN] best_v3.onnx: 9806720 字节  sha256=a9804980ad25… (期望 9827931 / f009f3b392f0…)
  [WARN] pre_model_v7.onnx: 44452568 字节  sha256=885e01b20592… (期望 5012923 / ba230a287e7d…)
  models_are_latest -> False

=== 没装模型（应判 否 + 缺失） ===
  [WARN] best_v3.onnx: 缺失
  [WARN] pre_model_v7.onnx: 缺失
  models_are_latest -> False

结论: 通过 —— 新旧模型能被正确区分
```

### 3.2 全新安装演练

把 `captcha/models/*.onnx` 移走 → `--check` 报缺失 → 用 `--source` 装回：

```
[WARN] best_v3.onnx: 缺失          ← 移走后
（执行 --source）
[OK  ] best_v3.onnx: 9827931 字节  sha256=f009f3b392f0…   ← 装回后
```

### 3.3 功能验证

`python tools/verify_10_integration.py`：

```
models_available: True
yolo: YOLO   siamese: PreONNX   ocr: DdddOcr   fonts: 5 个可用
点选结果: [(0.0558, 0.7417), (0.6503, 0.7458), (0.5342, 0.4073)]
模式: strip  提示标签=['9','d','0']
滑块缺口: (1.0, 457, 0)
结论: 1.0 求解器可用
全部通过。
```

### 3.4 回归

```
tests/test_captcha_setup.py          15 passed
```

（改的过程中曾因在告警里调用 `local_rev()` 而挂掉
`test_non_interactive_skips_when_complete` —— 它会起 git 子进程，
而该测试断言不得调用子进程。已改为不取版本号。）

---

## 4. 还剩一件事：远端仍然落后

`--source` 和 install.bat 自动探测只解决**本机**。
若要把安装包发给别人、或让默认 GitHub 通道也装到最新，
必须把本地提交推上去：

```
main e23266c [origin/main: ahead 1]   ← 待推送
origin/main 812deeb                   ← 旧模型
```

这一步会改动远端仓库，需要你确认后再执行：

```powershell
cd D:\Github\captchayes\T-Sec-captcha
git push origin main
```

推送后，`--non-interactive` 与 `--check` 都会自动认为模型是最新的，
install.bat 也不再打印过期告警。

---

## 5. 复现命令

```powershell
cd D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0
$env:PYTHONIOENCODING='utf-8'

# 看当前模型新旧（含指纹）
python tools\setup_captcha.py --check

# 装最新（本地源，推荐）
python tools\setup_captcha.py --source D:\Github\captchayes\T-Sec-captcha

# 走 GitHub（会告警是否过期）
python tools\setup_captcha.py --non-interactive

# 验证新旧识别逻辑
cd D:\Github\captchayes\T-Sec-captcha
python tools\verify_latest_models.py
```

## 6. 文件清单

```
cun.ai 1.0/tools/setup_captcha.py   +LATEST_MODELS / --source / 指纹校验
cun.ai 1.0/install.bat              第 3 步真安装 + 源优先级 + 装完自检
cun.ai 1.0/install.bat.bak          改动前备份
tools/patch_install_bat.py          重打 install.bat 的脚本（GBK 安全、按行替换）
tools/verify_latest_models.py       新旧模型识别验证（三场景）
```
