# -*- coding: utf-8 -*-
"""probe_train_flags: 定位 YOLO 训练在本机崩溃（exit 0xC000001D 非法指令）的根因。

现象：
    bench 子任务用 {imgsz=640,batch=8,workers=4,cache=False,amp=False,plots=False,val=False}
    能正常跑完 1 epoch；
    而 train_detector.py 用 {cache=ram, amp=True, plots=True, val=True, end2end=True}
    在 optimizer 建好之后、打印第一轮之前以 0xC000001D 终止（= 执行了本 CPU 不支持的指令，
    本机 Xeon E5-2660 v2 只有 AVX，没有 AVX2/FMA）。

本脚本逐项开关，找出到底是哪个 flag 引入了 AVX2 代码路径。

用法：
    python train/probe_train_flags.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "selfgen", "data.yaml")
WARM = os.path.join(REPO, "bench", "weights", "yolo26n.pt")

BASE = dict(model=WARM, data=DATA, epochs=1, imgsz=320, batch=4, workers=2,
            device="cpu", cache=False, amp=False, plots=False, val=False,
            exist_ok=True, project=os.path.join(REPO, "runs", "detect"),
            name="flagprobe", seed=0, deterministic=False)

CASES = [
    ("known-good (bench 同款)", {}),
    ("+ end2end=True", {"end2end": True}),
    ("+ nms=False", {"nms": False}),
    ("+ cache=ram", {"cache": "ram"}),
    ("+ amp=True", {"amp": True}),
    ("+ plots=True", {"plots": True}),
    ("+ val=True", {"val": True}),
    ("+ cache=ram+plots+val", {"cache": "ram", "plots": True, "val": True}),
]

SNIPPET = r"""
import os, sys, yaml
cfg = yaml.safe_load(sys.argv[1])
from ultralytics import YOLO
m = YOLO(cfg.pop('model'))
m.train(**cfg)
print('TRAIN_OK')
"""


def run_case(name, extra):
    cfg = dict(BASE)
    cfg.update(extra)
    cfg = {k: v for k, v in cfg.items() if v is not None}
    # 让 ultralytics 用绝对路径，避免嵌套
    cfg["project"] = os.path.abspath(cfg["project"])
    cfg["data"] = os.path.abspath(cfg["data"])
    cfg["model"] = os.path.abspath(cfg["model"])
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["POLARS_SKIP_CPU_CHECK"] = "1"
    tmp = os.path.join(REPO, "out", "_flagcfg.yaml")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    try:
        p = subprocess.run([sys.executable, "-c", SNIPPET, tmp],
                           capture_output=True, text=True, timeout=900, env=env,
                           cwd=REPO)
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode == 0 and "TRAIN_OK" in out:
            status = "OK"
        elif p.returncode == 0:
            status = "NO_MARKER"
        else:
            status = f"CRASH rc={p.returncode} (0x{p.returncode & 0xFFFFFFFF:08X})"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
        out = ""
    # 抓关键报错行
    key = ""
    for line in out.splitlines():
        low = line.lower()
        if any(k in low for k in ("error", "invalid", "illegal", "avx", "traceback")):
            key = line.strip()[:120]
            break
    print(f"  {name:32s} -> {status:28s} {key}")
    return status


if __name__ == "__main__":
    if not os.path.exists(WARM):
        print(f"缺少预训练权重 {WARM}（bench 子任务下载的 yolo26n.pt）")
        sys.exit(1)
    print("=" * 92)
    print("YOLO 训练 flag 崩溃定位（本机 CPU 无 AVX2）")
    print("=" * 92)
    results = {}
    for name, extra in CASES:
        results[name] = run_case(name, extra)
    print("\n=== 结论 ===")
    bad = [k for k, v in results.items() if not v.startswith("OK")]
    good = [k for k, v in results.items() if v.startswith("OK")]
    print(f"  可用配置: {good}")
    print(f"  崩溃配置: {bad}")
