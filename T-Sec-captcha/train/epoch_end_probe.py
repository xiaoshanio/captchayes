# -*- coding: utf-8 -*-
"""epoch_end_probe: 定位「第 1 个 epoch 跑完、写 results.csv 之后」的 0xC000001D。

已知事实：
  - forward(320/640) OK，backward OK，EMA OK，绘图 OK
  - 训练循环能跑满 200/200 个 batch，loss 正常下降
  - results.csv 写出了 epoch 1 行，随后硬崩溃 0xC000001D

因此崩溃发生在 **epoch 结束回调**里。本脚本逐个测这些回调：
  1. torch.save 检查点（含 ema/optimizer/model）
  2. strip_optimizer（ultralytics 保存 best.pt 时会做）
  3. ModelEMA.update_attr
  4. model.info()（FLOPs 统计，走 ultralytics-thop）
  5. 保存 CSV（polars / pandas 路径）
  6. copy.deepcopy(model)
  7. 学习率调度器

用法：
    python train/epoch_end_probe.py
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARM = os.path.join(REPO, "bench", "weights", "yolo26n.pt")
TMP = os.path.join(REPO, "out", "epochprobe")

COMMON = (
    "import os, sys, copy, tempfile\n"
    "os.environ.setdefault('POLARS_SKIP_CPU_CHECK','1')\n"
    f"sys.path.insert(0, r'{REPO}')\n"
    "os.makedirs(r'{tmp}', exist_ok=True)\n"
    "import torch\n"
    "torch.set_num_threads(4)\n"
    "from ultralytics import YOLO\n"
    f"m = YOLO(r'{WARM}')\n"
    "model = m.model\n"
).replace("{tmp}", TMP)

STEPS = {
    "1_torch_save_ckpt": """
ck = {'epoch': 0, 'best_fitness': None, 'model': None, 'ema': None,
      'updates': 0, 'optimizer': None, 'train_args': vars(getattr(m,'overrides',{})) and {} }
torch.save({'model': model.float().state_dict()}, os.path.join(r'%s','ck.pt'))
print('MARK save ok')
""" % TMP,
    "2_deepcopy_model": """
mc = copy.deepcopy(model)
print('MARK deepcopy ok', type(mc).__name__)
""",
    "3_strip_optimizer": """
from ultralytics.utils.torch_utils import strip_optimizer
p = os.path.join(r'%s','s.pt')
torch.save({'model': model.float(), 'optimizer': None, 'ema': None}, p)
strip_optimizer(p)
print('MARK strip ok')
""" % TMP,
    "4_ema_update_attr": """
from ultralytics.utils.torch_utils import ModelEMA
ema = ModelEMA(model)
ema.update(model)
ema.update_attr(model, include=['yaml','nc','names','stride','class_weights'])
print('MARK ema update_attr ok')
""",
    "5_model_info_flops": """
try:
    info = model.info(verbose=False)
    print('MARK info ok', info)
except Exception as e:
    print('MARK info raised', type(e).__name__, e)
""",
    "6_csv_polars": """
import polars as pl
df = pl.DataFrame({'epoch':[1,2],'loss':[1.0,0.5]})
df.write_csv(os.path.join(r'%s','t.csv'))
print('MARK polars ok', pl.__version__)
""" % TMP,
    "7_csv_pandas": """
import pandas as pd
df = pd.DataFrame({'epoch':[1,2],'loss':[1.0,0.5]})
df.to_csv(os.path.join(r'%s','t2.csv'), index=False)
print('MARK pandas ok', pd.__version__)
""" % TMP,
    "8_lr_scheduler": """
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for sched in ('linear','cosine'):
    try:
        s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        for _ in range(3): s.step()
        print('MARK sched ok', sched)
    except Exception as e:
        print('MARK sched raised', type(e).__name__, e)
""",
    "9_validator_small": """
from ultralytics.models.yolo.detect import DetectionValidator
print('MARK validator import ok')
""",
}


def main():
    print("=" * 88)
    print("epoch 结束回调定位（崩溃在 results.csv 写出之后）")
    print("=" * 88)
    results = {}
    for name, body in STEPS.items():
        code = COMMON + body
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["POLARS_SKIP_CPU_CHECK"] = "1"
        try:
            p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, timeout=600, env=env, cwd=REPO)
            rc = p.returncode
            out = (p.stdout or "") + (p.stderr or "")
            if rc == 0 and "MARK" in out:
                status = "OK"
            elif rc == 0:
                status = "NO_MARK"
            elif (rc & 0xFFFFFFFF) == 0xC000001D:
                status = "ILLEGAL_INSTRUCTION"
            else:
                status = "rc=1"
            mark = [l for l in out.splitlines() if "MARK" in l]
            note = mark[0][:75] if mark else ""
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
    bad = [k for k, v in results.items() if v not in ("OK",)]
    print(f"  非 OK: {bad if bad else '无'}")


if __name__ == "__main__":
    main()
