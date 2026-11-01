# captchayes 仓库 git 公开面清理方案

> 目标：公开仓库只保留 **模型文件 + 训练/推理/评测代码**，去掉系统环境信息、测试站点信息、账号凭据与抓取数据。

---

## 0. 现状实测

| 项 | 数值 |
|---|---|
| 远端 | https://github.com/xiaoshanio/captchayes.git（main = 812deeb） |
| 提交数 | 3（c328bd9 / b50b4ce / 812deeb） |
| 已跟踪文件 | 37 个 |
| .git 体积 | 47.8 MB（1 个 pack，55 对象） |
| HEAD 内最大 blob | `model/pre_model_v7.onnx` 42.39 MB、`model/best_v3.onnx` 9.35 MB |
| 工作区未跟踪且**未被忽略** | 905 个文件（out 733 / runs 65 / tools 63 / eval 14 / docs 9 / weights 9 / train 8 / bench 3 / model 1） |
| 工作区已忽略 | data/ 838.9 MB（1147 个样本目录）、runs/ 496.3 MB、weights/ 206.5 MB、out/ 99.9 MB |
| 提交作者 | XIAOSHAN \<2162419098@qq.com\>（全部 3 个提交） |

已跟踪文件清单（37）：README.md、INTEGRATION_CUNAI.md、.gitignore、.gitattributes、requirements.txt、
demo.py、service.py、service_demo.py、verify_cunai.py、__init__.py、
src/{captcha,drawing}.py + src/utils/{matchingMode,ver_onnx,yolo_onnx}.py、
cv/{matchingMode,ver_onnx,yolo_onnx}.py、app/（9 个 py）、model/{best_v3,pre_model_v7}.onnx、
tools/sync_cv.py。

---

## 1. 已公开内容的问题清单

### A. 系统 / 本机环境信息（P1）

| 位置 | 内容 |
|---|---|
| `INTEGRATION_CUNAI.md` L3, L35-37, L43, L48, L64 | `F:\Kaifa\cun.ai`、`F:/Kaifa/captchayes/...` |
| `verify_cunai.py` L67（HEAD 版同） | 默认参数 `--cunai F:\Kaifa\cun.ai` |
| `README.md` L91 | `set TSEC_MODEL_DIR=F:\Kaifa\cun.ai\captcha\models` |
| 提交作者 | 真实 QQ 邮箱 `2162419098@qq.com` |

> 工作区未跟踪的 `docs/INTEGRATE_AUTO_REGISTER.md`、`bench/run_all.py`（`--python G:\Python\Dev\python\python.exe`）、
> `docs/TRAINING_AND_DATA_REFRESH.md`（`D:\captcha\yolov5\data\bilbil.yaml`）含更多本机路径与内网工程名，一旦 `git add -A` 即进公开面。

### B. 测试站点 / 目标业务信息（P1）

| 位置 | 内容 |
|---|---|
| `README.md` L164 | `page.goto("https://example.com/login")`（尚可） |
| `README.md` L10-11, L130-202 | CloakBrowser（cloakbrowser.dev）整节软广 + 联合过验证码示例 |
| `app/models/input.py` L25-26 | 真实极验挑战图 URL（`static.geetest.com/.../474b661c0...jpg?challenge=894235a0...`）作为默认值/示例 |
| 未跟踪但高危：`tools/` 64 个脚本 | `tools/cunai_login_lib.py`（1105 行，含 cun.ai 登录全流程 + 天御 DOM 选择器 + 8 条"踩坑"）、`login_cunai.py`、`login_10x.py`、`recon_cunai.py`、`probe_turing.py`、`collect_2408.py`、`analyze_collect.py`、`dump_net.py`、`tap_verify.py`、`summarize_login.py` 等，全部硬编码 `https://www.cun.ai`、`https://turing.captcha.qcloud.com`、appid `195317439` |
| 未跟踪：`docs/RUN_2408.md`(39KB) / `docs/INTEGRATE_AUTO_REGISTER.md` / `docs/LOGIN_SUCCESS_RATE.md` | 天御弹层 DOM 全量选择器、题型 2401/2404/2408 判定、submit 接口 `cap_union_new_verify`、errorCode=202 分析、图标点选提示条几何（672×480 + 56 strip） |

### C. 账号凭据 / 隐私（P0）

| 位置 | 内容 |
|---|---|
| `docs/LOGIN_SUCCESS_RATE.md` L4 | `账号：zanjianwan25946（凭据有效，已实测可登录）` |
| `docs/LOGIN_SUCCESS_RATE.md` L174 | `--username zanjianwan25946 --password 60c0cbdb37de` |
| `tools/probe_captcha_dom.py` L68-69 | 默认 `lindabaker57252` / `lindabaker572` |
| `tools/probe_popup_buttons.py` L64-65 | 同上 |
| `tools/recon_cunai.py` L434-436 | `tsecprobe001` / `Probe!2026#tsec` / `tsecprobe001@example.com` |
| `out/login/login_report.json` L3 | `"username": "lindabaker57252"` |
| `out/login10x/report_*.json`（16 个） | `"username": "zanjianwan25946"`，单个文件 75~82 处事件含登录态、URL、点击坐标 |

### D. 抓取数据 / 运行产物（P1）

| 位置 | 内容 |
|---|---|
| `out/`（733 文件） | 337 png / 293 jpg / 75 json / 58 log / 4 jsonl / 2 bin / 1 html：站点验证码原图、弹层截图、点击前后快照、事件日志 |
| `out/dump_net/resp_*.txt` | 站点真实响应体：`_aq_128199({...})` prehandle JSONP 原文、混淆 JS、混淆多语言包（658KB） |
| `runs/`（65 文件） | 34 onnx + 19 pt + csv/yaml 训练产物 |
| `weights/`（9 文件） | `.pt` 权重 9.32 / 9.88 / 74.48 MB + `_stage/*.onnx` 42.39 MB |
| `data/`（1147 个样本目录） | 站点 2408 题型原始底图/提示图/label（已被 .gitignore 忽略，但策略只是 `data/`） |

### E. 仓库结构 / 可用性缺陷（P2，但影响"clone 即用"承诺）

1. **clone 路径断链**：仓库根目录是 `captchayes/`，内容全部在 `T-Sec-captcha/`。README L40-41 写
   `git clone ... T-Sec-captcha && cd T-Sec-captcha`，实际得到 `T-Sec-captcha/T-Sec-captcha/demo.py`，`python demo.py` 必失败。
2. **`docs/res.jpg` 在 HEAD 中被删除**（812deeb 删除，b50b4ce 才刚加回来），但 README L68/73/117 与 `demo.py` 仍以它为默认输入 → clone 后 `demo.py`、`service_demo.py` 直接报文件缺失。
3. **模型与工作区不一致**：HEAD 提交的 `pre_model_v7.onnx` 是 42.39 MB 旧域模型，工作区已是 5.01 MB 的 2408 域重训版且未提交（`git status` 显示 ` M`）；`src/captcha.py` 有未提交的 `_filter_char_boxes` 修复（+9.0pt）。即"公开的东西 ≠ 真正在用的东西"。
4. **重复代码**：`cv/` 与 `src/utils/` 三个文件同内容双份（`tools/sync_cv.py` 维护），只为 Windows 下免 symlink。
5. **依赖过重**：`requirements.txt` 14 项，含 `playwright / aiohttp / cryptography / matplotlib / Cython`，与 README 声称的核心依赖（numpy/pillow/opencv/onnxruntime）严重不符。
6. **README 过度承诺**：96% 准确率、"1 核 2G 可跑"，无 benchmark 出处；`docs/TRAINING_BENCHMARK.md` 未公开。
7. **无 LICENSE**；`app/models/input.py` 有一段重复 import（pydantic 声明两次）。
8. **`.gitignore` 用负向规则 `!model/*.onnx`** 导致 `model/best_v3_next.onnx`（9.37 MB，未跟踪）实际**不被忽略**，任何 `git add -A` 会带进第 3 个模型。

---

## 2. 目标结构（公开面）

```text
captchayes/                      # 仓库根 = 包根（修掉双层目录）
├── LICENSE                      # 新增（建议 MIT 或自定义"仅研究"）
├── README.md                    # 重写：无站点、无广告、无本机路径
├── requirements.txt             # 只留推理核心
├── requirements-service.txt     # fastapi/uvicorn/python-multipart
├── requirements-train.txt       # torch/ultralytics（可选）
├── .gitignore / .gitattributes
├── model/
│   ├── best_v3.onnx             # 点选检测（YOLO）
│   └── pre_model_v7.onnx        # 孪生比对
├── src/
│   ├── captcha.py
│   ├── drawing.py
│   └── utils/{yolo_onnx,ver_onnx,matchingMode}.py
├── cv/                          # 兼容 from cv import（软链或保持双份，见 4.6）
├── app/                         # FastAPI 服务
├── train/
│   ├── train_detector.py        # 参数化：--data、--imgsz、--epochs
│   ├── train_matcher.py
│   └── prepare_pairs.py         # 只接受通用目录约定
├── eval/
│   ├── eval_new.py
│   └── e2e_diag.py
├── bench/
│   └── bench_train.py
├── tools/
│   └── sync_cv.py               # 唯一保留的 tools
├── demo.py / service.py / service_demo.py
├── docs/
│   ├── TRAINING.md              # 由 TRAINING_*.md 脱敏合并
│   ├── BENCHMARK.md
│   └── sample.jpg               # 合成样本（替代 res.jpg，去掉站点来源）
└── samples/                     # 可选：2~3 张自制合成图，证明可跑
```

**不进公开仓库**：`data/`、`out/`、`runs/`、`weights/`、`tools/` 下全部站点脚本、
`docs/{LOGIN_SUCCESS_RATE,RUN_2408,INTEGRATE_AUTO_REGISTER,DRIFT_EXPERIMENT}.md`、
`INTEGRATION_CUNAI.md`（或脱敏后改名 `docs/INTEGRATION.md`）、`model/best_v3_next.onnx`。

---

## 3. 脱敏规则（逐条可执行）

### 3.1 站点标识 → 占位符

| 原文 | 替换 |
|---|---|
| `www.cun.ai` / `cun.ai` | `TARGET_HOST` |
| `turing.captcha.qcloud.com` | `CAPTCHA_HOST` |
| appid `195317439` | `APP_ID` |
| 题型 `2408 / 2404 / 2401`（站点内部编号） | `ICON_TAP / TEXT_TAP / SLIDER` |
| `.tencent-captcha-dy__*` 选择器 | 删除整段 |
| `cap_union_prehandle` / `cap_union_new_verify` / `getcapbysig` | 删除整段 |

### 3.2 本机路径 → 相对路径

| 原文 | 替换 |
|---|---|
| `F:\Kaifa\cun.ai`、`F:/Kaifa/captchayes/...` | `./` 或 `--cunai` 无默认值 |
| `G:\Python\Dev\python\python.exe` | `sys.executable` |
| `D:\captcha\yolov5\data\bilbil.yaml` | `--data dataset.yaml` |
| `D:\Github\Auto-Register\...` | 删除 |
| `C:\Windows\Fonts\*.ttc` | 保留但加注释"仅 Windows 兜底，可用 `--font` 覆盖" |

### 3.3 凭据与个人标识 → 全删

- `zanjianwan25946` / `60c0cbdb37de` / `lindabaker57252` / `lindabaker572` / `tsecprobe001` / `Probe!2026#tsec`
- `username` 默认值一律改为 `None` + `required=True`：`probe_captcha_dom.py`、`probe_popup_buttons.py`、`recon_cunai.py`
- 报告 JSON 里的 `username` 字段在生成侧改为写入 `sha256(user)[:8]`
- 提交流作者改为 `xiaoshanio <xiaoshanio@users.noreply.github.com>`（GitHub noreply）

### 3.4 抓取物与产物

- `app/models/input.py` 的极验示例 URL → `https://example.com/captcha.jpg`（或删除 default，改 `Field(..., example="...")` 空值）
- 所有 `out/`、`runs/`、`weights/`、`data/` 内容不入库；如需发布权重，走 **GitHub Releases**（zip 附 sha256），不要进 git 历史。

### 3.5 README 重写要点

1. 删掉 CloakBrowser 整节（L10-11、L130-202）与所有外链软广。
2. 删掉"撞库/批量注册"语境的免责声明段（L215-217），换成中性"研究与自有系统测试"一句。
3. 准确率改为可复现数字 + 指向 `docs/BENCHMARK.md`，不写"约 96%"。
4. 运行说明按 `--data/--model` 参数写，不出现任何真实域名与本机盘符。
5. 补 `LICENSE`。

---

## 4. 执行方案（按序）

### 4.1 先备份（必做）

```powershell
cd D:\Github
git -C captchayes bundle create captchayes-full-backup.bundle --all
Copy-Item captchayes captchayes-private-snapshot -Recurse   # 全量工作区（含 data/out/runs/weights）
```

### 4.2 建白名单文件

`D:\Github\keep-paths.txt`：

```text
T-Sec-captcha/README.md
T-Sec-captcha/LICENSE
T-Sec-captcha/.gitignore
T-Sec-captcha/.gitattributes
T-Sec-captcha/requirements.txt
T-Sec-captcha/requirements-service.txt
T-Sec-captcha/requirements-train.txt
T-Sec-captcha/__init__.py
T-Sec-captcha/src/
T-Sec-captcha/cv/
T-Sec-captcha/app/
T-Sec-captcha/model/best_v3.onnx
T-Sec-captcha/model/pre_model_v7.onnx
T-Sec-captcha/train/
T-Sec-captcha/eval/
T-Sec-captcha/bench/
T-Sec-captcha/tools/sync_cv.py
T-Sec-captcha/demo.py
T-Sec-captcha/service.py
T-Sec-captcha/service_demo.py
T-Sec-captcha/docs/
```

> 注：此时 `train/eval/bench/docs` 尚未被跟踪，白名单只作用于**历史裁剪**；新代码在 4.4 脱敏后再 `git add`。
> 若只想裁历史、不想动结构，也可跳过 `--path-rename`。

### 4.3 裁历史 + 拉平目录

```powershell
pip install git-filter-repo
cd D:\Github\captchayes
git filter-repo --force `
  --paths-from-file ..\keep-paths.txt `
  --path-rename T-Sec-captcha/: `
  --mailmap ..\.mailmap
```

`.mailmap`：

```text
xiaoshanio <xiaoshanio@users.noreply.github.com> XIAOSHAN <2162419098@qq.com>
```

> 这一步会永久丢弃 `docs/res.jpg` 与 42.39 MB 旧 `pre_model_v7.onnx` 的历史 blob，`.git` 从 47.8 MB 降到 ~15 MB。
> `--path-rename` 修掉 §1.E.1 的双层目录问题。

### 4.4 写入脱敏后的新内容

顺序：先改 `src/captcha.py`（带上工作区里那版 `_filter_char_boxes`，这是真实修复）、
提交工作区 5.01 MB 的 `pre_model_v7.onnx` 与 9.83 MB 的 `best_v3.onnx`（保证公开模型 = 实际在用的模型），
再按 §3 脱敏 `README.md`、`INTEGRATION_CUNAI.md → docs/INTEGRATION.md`、`app/models/input.py`、
`verify_cunai.py`（默认值改 `--cunai` 必填、`--image` 必填，去掉 `F:\Kaifa`）。

然后新增（从工作区脱敏后复制）：

| 源 | 目标 | 处理 |
|---|---|---|
| `train/train_detector.py` | `train/train_detector.py` | `--recovered` 默认改 `weights/best_v3.pt`（相对路径），`--arch` 保留 |
| `train/train_matcher.py` | `train/train_matcher.py` | 数据路径参数化 |
| `train/prepare_pairs.py` | `train/prepare_pairs.py` | 只保留通用 `--src/--out` |
| `train/crash_bisect*.py`、`probe_train_flags.py`、`epoch_end_probe.py` | 丢弃 | 调试残留 |
| `eval/eval_new.py`、`e2e_diag.py`、`v1_v2_compare.py`、`matcher_score_diag.py` | `eval/` | 去掉 2408/站点常量 |
| `eval/*_drift*`、`domain_gap.py`、`class2_diag.py`、`crop_source_test.py`、`audit_dataset.py` | 丢弃 | 依赖站点样本域 |
| `bench/bench_train.py` | `bench/` | `--python` 默认改 `sys.executable` |
| `bench/make_toy_dataset.py`、`run_all.py` | 丢弃 | 引本机路径 |
| `docs/TRAINING_BENCHMARK.md` | `docs/BENCHMARK.md` | 去域名，补复现命令 |
| `docs/CONTINUE_TRAINING.md`(36KB)、`TRAINING_AND_DATA_REFRESH.md` | `docs/TRAINING.md` | 去 2408/站点/本机路径，保留方法论 |
| `docs/badcase_example.png` | `docs/badcase.png` | 需肉眼确认不含站点水印/URL |
| 自制合成图 1 张 | `docs/sample.jpg` | 替代 `res.jpg`，数据来源可公开 |

### 4.5 新 `.gitignore`（比现状严格）

```gitignore
# ---- Python
__pycache__/
*.py[cod]
.venv/ venv/
.pytest_cache/

# ---- 运行产物 / 抓取数据（一律不入库）
data/
datasets/
out/
runs/
weights/
artifacts/
browser_profiles/
*.log
*.pid
res2.jpg

# ---- 站点相关脚本与文档（留在私有仓）
tools/login_*.py
tools/probe_*.py
tools/recon_*.py
tools/collect_*.py
tools/dump_net.py
tools/tap_verify.py
tools/summarize_login.py
tools/cunai_login_lib.py
tools/analyze_collect.py
tools/show_report.py
docs/LOGIN_*.md
docs/RUN_*.md
docs/INTEGRATE_AUTO_REGISTER.md

# ---- 凭据
.env
*.secret
secrets/

# ---- 模型：显式白名单，避免 !model/*.onnx 误伤
model/*
!model/best_v3.onnx
!model/pre_model_v7.onnx

# ---- 系统
.DS_Store
Thumbs.db
```

> 关键修正：`!model/*.onnx` 改成 `model/*` + 两个精确 `!` 白名单，
> 这样 `model/best_v3_next.onnx` 不会再被误纳入。

### 4.6 `cv/` 与 `src/utils/` 去重（可选但建议）

Linux/macOS 保留 `cv/` → `src/utils/` 的相对软链，Windows 上由 `tools/sync_cv.py --check` 在 CI 里校验哈希；
或干脆只留 `src/utils/`，`cv/__init__.py` 里做 re-export：

```python
from src.utils import matchingMode, ver_onnx, yolo_onnx  # noqa: F401
```

体积从 6 份文件降到 3 份，且保留 `from cv import ...` 兼容。

### 4.7 提交与推送

```powershell
git add -A
git status --short          # 必须只剩白名单内文件，且无 out/runs/data/weights
git commit -m "chore: 公开面收敛为模型 + 训练/推理代码，移除站点信息与本机路径"

# 重写历史后需强制推送（旧提交哈希将失效）
git push --force-with-lease origin main
git -C . gc --prune=now --aggressive
```

### 4.8 推送后收尾

1. GitHub 仓库 Settings → 确认无 Actions/Pages 引用旧路径；删除可能残留的 fork/cache（旧对象在 GitHub 侧会保留一段时间，但本次只涉及模型 blob，不含密钥，无需轮换）。
2. 旧提交里出现过的**字符串凭据**（`zanjianwan25946` / `60c0cbdb37de` / `lindabaker57252` / `tsecprobe001`）：
   本次实测**未进入 git 历史**（全在未跟踪文件中），若历史上曾经 `add` 过，务必**直接改密**，不要指望历史清理。
3. 发布模型：若后续模型 >50 MB，改走 Releases：

```powershell
Compress-Archive model\*.onnx captchayes-models-v1.zip
# 上传 Release，README 写下载 + sha256 校验
```

4. 私有仓：把 `data/`、`out/`、`runs/`、`weights/`、站点 `tools/`、`docs/LOGIN_*` 整体转到一个私有仓库或本地裸仓存档，避免丢失复现能力。

---

## 5. 验收清单

- [ ] `git ls-files | grep -Ei 'cun\.ai|turing|cunai|zanjian|lindabaker|tsecprobe|195317439|F:\\\\|G:\\\\|D:\\\\'` 无输出
- [ ] `git log --format='%ae' | sort -u` 只剩 `xiaoshanio@users.noreply.github.com`
- [ ] `.git` 体积 < 20 MB（模型 15 MB + 代码）
- [ ] 全新目录 `git clone` 后 `cd captchayes && pip install -r requirements.txt && python demo.py` 一次通过
- [ ] `git ls-files | grep -E '^(out|runs|weights|data)/'` 无输出
- [ ] `python tools/sync_cv.py --check` 返回 PASS
- [ ] README 中不再出现任何真实域名、盘符、第三方产品推广
