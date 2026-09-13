# T-Sec-captcha 在 cun.ai 中的集成（只读 cun.ai，不修改它）

本仓库 `git clone` 后可直接应用到 `F:\Kaifa\cun.ai`，无需修改 cun.ai 原项目。

```bash
git clone https://github.com/xiaoshanio/captchayes.git T-Sec-captcha
cd T-Sec-captcha
pip install -r requirements.txt
python demo.py            # 应输出 4 点坐标
python verify_cunai.py    # 应全 PASS（只读比对 cun.ai，不写 cun.ai）
```

## 文件映射（已逐字节校验一致）

| 本仓库（git 源） | 复制到 cun.ai 的目标 | 说明 |
|---|---|---|
| `model/best_v3.onnx` | `captcha/models/best_v3.onnx` | YOLO 点选检测，哈希一致 |
| `model/pre_model_v7.onnx` | `captcha/models/pre_model_v7.onnx` | Siamese 比对，哈希一致 |
| `cv/matchingMode.py` | `captcha/cv/matchingMode.py` | 与 `src/utils/matchingMode.py` 同内容 |
| `cv/ver_onnx.py` | `captcha/cv/ver_onnx.py` | 与 `src/utils/ver_onnx.py` 同内容 |
| `cv/yolo_onnx.py` | `captcha/cv/yolo_onnx.py` | 与 `src/utils/yolo_onnx.py` 同内容 |

`cv/` 与 `src/utils/` 在本仓库内保持同内容双份（Windows 不用 symlink），
就是为了“复制即用”。改一处请同步另一处，或运行：

```bash
python tools/sync_cv.py --check   # 检查一致性
python tools/sync_cv.py           # 以 src/utils 为准同步到 cv/
```

## 集成方式（三种，按需选一）

### A. 文件复制（推荐，与现有 cun.ai 结构完全一致）
```powershell
Copy-Item "T-Sec-captcha\model\best_v3.onnx"  "F:\Kaifa\cun.ai\captcha\models\best_v3.onnx" -Force
Copy-Item "T-Sec-captcha\model\pre_model_v7.onnx" "F:\Kaifa\cun.ai\captcha\models\pre_model_v7.onnx" -Force
Copy-Item "T-Sec-captcha\cv\*.py" "F:\Kaifa\cun.ai\captcha\cv\" -Force
```

### B. 直接引用（不复制，适合联调）
```python
import sys
sys.path.insert(0, "F:/Kaifa/captchayes/T-Sec-captcha")
from src.captcha import TextSelectCaptcha  # 与 cun.ai 内 solver 底层同一套 onnx
cap = TextSelectCaptcha()
print(cap.run_dict("docs/res.jpg"))
# 模型目录也可用环境变量覆盖：
#   set TSEC_MODEL_DIR=F:\Kaifa\cun.ai\captcha\models
```

### C. 独立服务（cun.ai 经 HTTP 调用，不碰 Python 依赖）
```bash
python service.py
# POST http://127.0.0.1:8000/api/v1/identify
python service_demo.py
```

## 依赖交集
cun.ai 已有 `numpy / opencv-python-headless / onnxruntime / Pillow`，
覆盖本仓库核心推理依赖。`fastapi / uvicorn / python-multipart / aiohttp`
仅独立服务模式需要；`playwright` 仅旧 `bilbil.py` 演示需要（本仓库已不含）。

## 维护约束
- 不要修改 `F:\Kaifa\cun.ai`，验证脚本只读它。
- `model/*.onnx` 必须提交到 git，否则 clone 后不可用。
- `cv/` 与 `src/utils/` 保持同哈希。
