# -*- coding: utf-8 -*-
"""crash_bisect: 定位 0xC000001D（非法指令）到底发生在哪一步。

背景：本机 CPU 是 Xeon E5-2660 v2（Ivy Bridge-EP，仅 AVX，无 AVX2/FMA/BMI）。
ultralytics 训练在 optimizer 建好之后、第一个 step 之前直接以 0xC000001D 终止（硬崩溃，无 traceback）。

本脚本把"训练"拆成可独立运行的最小步骤，逐步定位：
  S1 模型构建
  S2 前向推理（imgsz 320 / 640）
  S3 前向 + 反向
  S4 损失计算（v8DetectionLoss 路径）
  S5 完整 train step（forward+loss+backward+step）
  S6 我们自己的合成图 resize（OpenCV 路径）

用法（每个步骤独立子进程，崩溃不会带走整个脚本）：
    python train/crash_bisect.py
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARM = os.path.join(REPO, "bench", "weights", "yolo26n.pt")

COMMON = r"""
import os, sys, traceback
os.environ.setdefault('POLARS_SKIP_CPU_CHECK','1')
sys.path.insert(0, r'{repo}')
import torch, numpy as np
torch.set_num_threads(4)
""".format(repo=REPO)

STEPS = {
    "S1_model_build": r"""
from ultralytics import YOLO
m = YOLO(r'{warm}')
print('MARK build ok, params', sum(p.numel() for p in m.model.parameters()))
""",
    "S1b_new_from_yaml": r"""
from ultralytics import YOLO
m = YOLO('yolo26n.yaml')
print('MARK yaml build ok')
""",
    "S2_forward_320": r"""
from ultralytics import YOLO
m = YOLO(r'{warm}').model.eval()
x = torch.randn(1,3,320,320)
with torch.no_grad():
    y = m(x)
print('MARK forward320 ok', type(y))
""",
    "S2_forward_640": r"""
from ultralytics import YOLO
m = YOLO(r'{warm}').model.eval()
x = torch.randn(1,3,640,640)
with torch.no_grad():
    y = m(x)
print('MARK forward640 ok', type(y))
""",
    "S3_forward_backward": r"""
from ultralytics import YOLO
m = YOLO(r'{warm}').model.train()
x = torch.randn(2,3,320,320)
out = m(x)
loss = sum(v.float().sum() for v in (out.values() if isinstance(out,dict) else out)) if isinstance(out,(dict,list)) else out.float().sum()
loss.backward()
print('MARK fwd+bwd ok', float(loss))
""",
    "S4_loss_module": r"""
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
m = YOLO(r'{warm}')
model = m.model.train()
crit = v8DetectionLoss(model)
preds = model(torch.randn(2,3,320,320))
batch = {
  'batch_idx': torch.tensor([0,0,1], dtype=torch.float32),
  'cls': torch.tensor([[0.],[2.],[0.]]),
  'bboxes': torch.tensor([[0.5,0.5,0.1,0.1],[0.3,0.3,0.1,0.1],[0.6,0.6,0.1,0.1]]),
  'img': torch.randn(2,3,320,320),
}
l, items = crit(preds, batch)
print('MARK loss ok', float(l))
""",
    "S5_full_train_step": r"""
from ultralytics import YOLO
m = YOLO(r'{warm}')
model = m.model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(2,3,320,320)
for i in range(2):
    out = model(x)
    loss = sum(v.float().sum() for v in out.values()) if isinstance(out,dict) else out.float().sum()
    opt.zero_grad(); loss.backward(); opt.step()
print('MARK train step ok')
""",
    "S6_cv2_resize": r"""
import cv2, numpy as np
img = (np.random.rand(536,672,3)*255).astype('uint8')
r = cv2.resize(img, (320,320), interpolation=cv2.INTER_LINEAR)
b = cv2.imdecode(cv2.imencode('.jpg', img,[int(cv2.IMWRITE_JPEG_QUALITY),90])[1], cv2.IMREAD_COLOR)
print('MARK cv2 ok', r.shape, b.shape, 'cv2', cv2.__version__)
""",
    "S7_numpy_matmul": r"""
import numpy as np
a = np.random.rand(1024,1024).astype('float32')
b = a @ a
print('MARK numpy ok', float(b.sum()))
""",
    "S8_torch_conv_bench": r"""
import torch
c = torch.nn.Conv2d(64,64,3,padding=1)
x = torch.randn(4,64,160,160)
with torch.no_grad():
    for _ in range(5):
        y = c(x)
print('MARK torch conv ok', tuple(y.shape))
""",
}


def main():
    print("=" * 88)
    print("崩溃定位：0xC000001D = STATUS_ILLEGAL_INSTRUCTION（CPU 缺少 AVX2/FMA）")
    print("=" * 88)
    results = {}
    for name, body in STEPS.items():
        # 不用 str.format（Step 里含大量花括号），直接替换占位符
        code = COMMON + body.replace("{warm}", WARM)
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["POLARS_SKIP_CPU_CHECK"] = "1"
        try:
            p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, timeout=600, env=env, cwd=REPO)
            rc = p.returncode
            out = (p.stdout or "") + (p.stderr or "")
            ok = "MARK" in out
            if rc == 0 and ok:
                status = "OK"
            elif rc == 0:
                status = "NO_MARK"
            elif (rc & 0xFFFFFFFF) == 0xC000001D:
                status = "ILLEGAL_INSTRUCTION"
            else:
                status = f"rc={rc} (0x{rc & 0xFFFFFFFF:08X})"
            mark = [l for l in out.splitlines() if "MARK" in l]
            note = mark[0][:80] if mark else ""
            if not ok:
                for l in out.splitlines():
                    if "Error" in l or "error" in l:
                        note = l.strip()[:90]
                        break
        except subprocess.TimeoutExpired:
            status, note = "TIMEOUT", ""
        results[name] = status
        print(f"  {name:24s} -> {status:22s} {note}")

    print("\n=== 结论 ===")
    bad = [k for k, v in results.items() if v != "OK"]
    if not bad:
        print("  所有步骤都通过 —— 崩溃在 ultralytics Trainer 的更上层（数据加载/AMP/绘图）")
    else:
        print(f"  失败步骤: {bad}")
        print("  第一个失败步骤就是根因所在；请据此规避（换 flag / 降 imgsz / 关掉该模块）")


if __name__ == "__main__":
    main()
