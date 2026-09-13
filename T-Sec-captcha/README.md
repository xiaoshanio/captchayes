# T-Sec-captcha 文字点选验证码识别

基于深度学习的高性能文字点选验证码识别方案，技术栈：**YOLO 目标检测 + Siamese 孪生网络 + ONNX CPU 推理**。

- 平均推理耗时：300~500ms
- 综合准确率：约 96%
- 仅需 Python 3.8+，1 核 2G 机器可跑
- 支持本地 Python 调用 / 命令行 / FastAPI HTTP 服务三种用法

> 最佳实践：搭配隐身浏览器使用，减少被风控触发验证码的概率，触发后用本项目自动点选通过。
> 推荐：**[CloakBrowser — https://cloakbrowser.dev](https://cloakbrowser.dev)**

---

## 1. 项目结构

```text
T-Sec-captcha/
├── src/captcha.py      # 核心识别类 TextSelectCaptcha
├── model/              # ONNX 模型 (best_v3.onnx + pre_model_v7.onnx)
├── cv/ / src/utils/    # 底层推理代码 (yolo_onnx / ver_onnx / matchingMode)
├── demo.py             # 本地命令行演示
├── service.py          # FastAPI 服务启动入口
├── service_demo.py     # HTTP 调用示例
├── app/                # FastAPI 实现 (POST /api/v1/identify)
└── requirements.txt
```

核心类只有 3 个方法：

- `detection(image_path)`：只做 YOLO 检测
- `run(image_path)`：返回按点击顺序排列的坐标 `[[x1,y1,x2,y2], ...]`
- `run_dict(image_path)`：返回 `{"imgW, imgH, point, corp"}`，直接给自动化点击用

---

## 2. 安装

```bash
git clone https://github.com/xiaoshanio/captchayes.git T-Sec-captcha
cd T-Sec-captcha

pip install -r requirements.txt
```

核心依赖：`numpy / pillow / opencv-python-headless / onnxruntime`。
`fastapi / uvicorn / python-multipart` 仅 HTTP 服务模式需要。

验证安装是否成功：

```bash
python demo.py
python verify_cunai.py
```

`demo.py` 正常会输出 4 个点的坐标 + `res2.jpg` 可视化结果。

---

## 3. 使用方式一：本地 Python API（最常用）

```python
from src.captcha import TextSelectCaptcha

cap = TextSelectCaptcha()  # 启动时加载两个 ONNX 模型

# 方式 A：拿原始坐标框，顺序即点击顺序
boxes = cap.run("docs/res.jpg")
print(boxes)
# [[119, 174, 193, 244], [223, 189, 298, 267], ...]

# 方式 B：拿自动化友好的字典
result = cap.run_dict("docs/res.jpg")
print(result)
# {
#   "imgW": 344, "imgH": 384,
#   "point": [{"x_rel": 159.0, "y_rel": 209.0}, ...],  # 点击中心点
#   "corp": [{"x1":126,"y1":174,"x2":192,"y2":244}, ...]
# }
```

自定义模型路径 / 环境变量覆盖模型目录：

```python
# 默认加载 model/best_v3.onnx + model/pre_model_v7.onnx
cap = TextSelectCaptcha(per_path="pre_model_v7.onnx", yolo_path="best_v3.onnx")
```

```bash
# 用环境变量指向别的模型目录（例如 cun.ai 的 captcha/models）
set TSEC_MODEL_DIR=F:\Kaifa\cun.ai\captcha\models
python demo.py
```

---

## 4. 使用方式二：HTTP 服务（适合跨语言 / 分布式）

启动：

```bash
python service.py
# 服务地址: http://127.0.0.1:8000
# API 文档: http://127.0.0.1:8000/docs
```

接口：

- `POST /api/v1/identify`：返回 JSON 坐标
- `POST /api/v1/show`：返回标注后的 JPEG 图片

调用示例（见 `service_demo.py`）：

```python
import base64, requests

with open("docs/res.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://127.0.0.1:8000/api/v1/identify", json={
    "dataType": 2,
    "imageSource": img_b64
})
print(resp.json())
# {"code":200,"msg":"成功","data":{"imgW":344,"imgH":384,"point":[...],"corp":[...]}}
```

---

## 5. 推荐搭配：CloakBrowser（https://cloakbrowser.dev）

### 为什么搭配？

做自动化时完整的链路是两层：

1. **尽量不触发验证**：普通 Playwright / Puppeteer 指纹太假（`navigator.webdriver=true`、CDP 泄露、TLS 指纹对不上），很容易被 reCAPTCHA v3 打低分、被 Cloudflare Turnstile 拦住。
2. **触发后自动解掉**：真遇到文字点选 / 消消乐验证码，就用本项目识别坐标并自动点击。

[CloakBrowser](https://cloakbrowser.dev) 就是专门解决第 1 层的：

- 不是 JS 注入、不是改启动参数，而是在 **Chromium C++ 源码级打了 73 个补丁**后重新编译的真实浏览器
- Canvas / WebGL / Audio / 字体 / GPU / `navigator.webdriver` / TLS 指纹都在二进制层面和真机一致
- **Playwright / Puppeteer 原生 API，改一行 import 就能用**，支持 Python / JS / .NET / Docker
- `humanize=True` 自带贝塞尔鼠标、真人打字节奏、拟真滚动
- 一句话总结：CloakBrowser 负责**防拦截**，本项目负责**解点选**，两者互补。它本身不做打码服务，所以点选识别正好由 T-Sec-captcha 补上。

官网、文档、下载：**https://cloakbrowser.dev**

```bash
pip install cloakbrowser
# 或 npm install cloakbrowser
# 或 docker run --rm cloakhq/cloakbrowser cloaktest
```

### 联合示例：CloakBrowser + T-Sec-captcha 全自动过点选

```python
from cloakbrowser import launch
from src.captcha import TextSelectCaptcha

# 1. 用隐身浏览器打开目标页（和 Playwright 用法一样）
browser = launch(humanize=True)
page = browser.new_page()
page.goto("https://example.com/login")

# 2. 截取验证码图（按你页面的选择器改）
page.locator("#captcha-img").screenshot(path="captcha.jpg")

# 3. 本地识别，拿到按顺序的点击点
cap = TextSelectCaptcha()
result = cap.run_dict("captcha.jpg")

# 4. 把相对图片的坐标换算成页面绝对坐标并点击
box = page.locator("#captcha-img").bounding_box()
for p in result["point"]:
    x = box["x"] + p["x_rel"] * box["width"] / result["imgW"]
    y = box["y"] + p["y_rel"] * box["height"] / result["imgH"]
    page.mouse.click(x, y)

browser.close()
```

迁移老代码只需要改一行：

```python
# before:
# from playwright.sync_api import sync_playwright
# browser = sync_playwright().start().chromium.launch()

# after:
from cloakbrowser import launch
browser = launch()
```

如果是分布式部署，让 CloakBrowser 端通过 HTTP 调本项目的识别服务即可：

```python
# 爬虫机（跑 CloakBrowser） -> 识别机（跑 python service.py）
# POST http://识别机IP:8000/api/v1/identify
```

更多 CloakBrowser 用法（代理、持久化 Profile、Docker 部署）见官网：https://cloakbrowser.dev

---

## 6. 常见问题

- `FileNotFoundError: 验证模型缺失`：确认 `model/*.onnx` 已随 git clone 下来，或用 `TSEC_MODEL_DIR` 指定目录。
- 推理慢：首次加载模型需要几秒，之后每次 300~500ms；不要每次识别都 `TextSelectCaptcha()`，全局单例复用。
- 坐标点不准：返回顺序即点击顺序，直接用 `point[i].x_rel / y_rel`（图片内中心点），注意按图片缩放比例换算到页面。
- 只想验证对接：跑 `python service_demo.py`，前提是 `service.py` 已启动。

---

## 7. 免责声明

本项目仅供学术研究和安全测试（自己的账号、自己的系统、公开站点）使用。请遵守法律法规和目标站点 ToS，不得用于批量注册、撞库、绕过付费/风控等非法用途。因滥用造成的后果由使用者承担。
