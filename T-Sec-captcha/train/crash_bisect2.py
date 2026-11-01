# -*- coding: utf-8 -*-
"""crash_bisect2: 精确定位 0xC000001D —— 前向没问题，重点查反向 / 优化器 / AMP。

S2 已证明 forward(320) 与 forward(640) 都 OK。
本脚本继续拆分：
  A. 纯 backward（简单 sum loss）
  B. 真实 v8DetectionLoss（正确构造 batch）
  C. optimizer.step()
  D. AMP/GradScaler 路径（ultralytics 默认 amp=True）
  E. ModelEMA 更新
  F. imgsz 640 下的完整 step

用法：
    python train/crash_bisect2.py
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARM = os.path.join(REPO, "bench", "weights", "yolo26n.pt")

COMMON = (
    "import os, sys, traceback\n"
    "os.environ.setdefault('POLARS_SKIP_CPU_CHECK','1')\n"
    f"sys.path.insert(0, r'{REPO}')\n"
    "import torch, numpy as np\n"
    "torch.set_num_threads(4)\n"
    "from ultralytics import YOLO\n"
    f"m = YOLO(r'{WARM}')\n"
)

STEPS = {
    "A_backward_simple": """
model = m.model.train()
x = torch.randn(2,3,320,320)
out = model(x)
tot = 0.0
for t in (out if isinstance(out,(list,tuple)) else [out]):
    if isinstance(t, dict):
        for v in t.values():
            if torch.is_tensor(v): tot = tot + v.float().sum()
    elif torch.is_tensor(t):
        tot = tot + t.float().sum()
tot.backward()
print('MARK backward simple ok', float(tot))
""",
    "B_backward_640": """
model = m.model.train()
x = torch.randn(2,3,640,640)
out = model(x)
tot = 0.0
for t in (out if isinstance(out,(list,tuple)) else [out]):
    if isinstance(t, dict):
        for v in t.values():
            if torch.is_tensor(v): tot = tot + v.float().sum()
    elif torch.is_tensor(t):
        tot = tot + t.float().sum()
tot.backward()
print('MARK backward 640 ok', float(tot))
""",
    "C_real_loss": """
from ultralytics.utils.loss import v8DetectionLoss
model = m.model.train()
crit = v8DetectionLoss(model)
x = torch.randn(2,3,320,320)
preds = model(x)
batch = {
  'batch_idx': torch.tensor([0,0,1], dtype=torch.float32),
  'cls': torch.tensor([[0.],[2.],[0.]]),
  'bboxes': torch.tensor([[0.5,0.5,0.1,0.1],[0.3,0.3,0.1,0.1],[0.6,0.6,0.1,0.1]]),
  'img': x,
}
l, items = crit(preds, batch)
l.backward()
print('MARK real loss ok', float(l))
""",
    "D_optimizer_step": """
model = m.model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(2,3,320,320)
for i in range(2):
    out = model(x)
    tot = 0.0
    for t in (out if isinstance(out,(list,tuple)) else [out]):
        if isinstance(t, dict):
            for v in t.values():
                if torch.is_tensor(v): tot = tot + v.float().sum()
        elif torch.is_tensor(t):
            tot = tot + t.float().sum()
    opt.zero_grad(); tot.backward(); opt.step()
print('MARK opt step ok')
""",
    "E_amp_scaler": """
try:
    scaler = torch.amp.GradScaler('cpu')
except Exception as e:
    print('MARK amp unsupported:', e); raise SystemExit(0)
model = m.model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(2,3,320,320)
with torch.amp.autocast('cpu'):
    out = model(x)
    tot = 0.0
    for t in (out if isinstance(out,(list,tuple)) else [out]):
        if isinstance(t, dict):
            for v in t.values():
                if torch.is_tensor(v): tot = tot + v.float().sum()
        elif torch.is_tensor(t):
            tot = tot + t.float().sum()
scaler.scale(tot).backward(); scaler.step(opt); scaler.update()
print('MARK amp ok')
""",
    "F_ema": """
from ultralytics.utils.torch_utils import ModelEMA
model = m.model.train()
ema = ModelEMA(model)
ema.update(model)
print('MARK ema ok')
""",
    "G_plots": """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1,2,3],[3,2,1]); fig.savefig(os.path.join(os.environ.get('TEMP','.'),'_t.png')); plt.close(fig)
print('MARK plots ok')
""",
}


def main():
    print("=" * 88)
    print("崩溃定位 v2：前向已排除，重点查反向 / 优化器 / AMP / EMA / 绘图")
    print("=" * 88)
    results = {}
    for name, body in STEPS.items():
        code = COMMON + body
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["POLARS_SKIP_CPU_CHECK"] = "1"
        try:
            p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, timeout=900, env=env, cwd=REPO)
            rc = p.returncode
            out = (p.stdout or "") + (p.stderr or "")
            if rc == 0 and "MARK" in out:
                status = "OK"
            elif rc == 0:
                status = "NO_MARK"
            elif (rc & 0xFFFFFFFF) == 0xC000001D:
                status = "ILLEGAL_INSTRUCTION"
            else:
                status = f"rc=1"
            mark = [l for l in out.splitlines() if "MARK" in l]
            note = mark[0][:70] if mark else ""
            if rc != 0:
                for l in out.splitlines():
                    if "Error" in l:
                        note = l.strip()[:80]
                        break
        except subprocess.TimeoutExpired:
            status, note = "TIMEOUT", ""
        results[name] = status
        print(f"  {name:22s} -> {status:22s} {note}")

    print("\n=== 结论 ===")
    bad = [k for k, v in results.items() if v not in ("OK", "NO_MARK")]
    print(f"  失败: {bad if bad else '无'}")
    if "D_optimizer_step" in bad:
        print("  -> 反向/优化器路径触发了非法指令：本机 CPU 缺 AVX2，PyTorch 的某些 kernel 无法执行")
    if bad and "D_optimizer_step" not in bad:
        print("  -> 反向本身可用，问题在被标记的那一步")


if __name__ == "__main__":
    main()
