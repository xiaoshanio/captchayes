#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bench/bench_train.py

本机（2x Xeon E5-2660 v2 / 无可用 GPU）CPU 训练吞吐基准。

模式：
  --mode ultralytics   走 ultralytics YOLO26n 的真实 model.train() 训练循环，
                       用 on_train_batch_start/end 回调把每个 optimizer step 的
                       「纯计算时间」和「等 DataLoader 的时间」拆开统计，并按 epoch 分组，
                       用来量化同配置内的方差（NUMA / 超线程漂移）。
  --mode torch         纯 torch 微基准，隔离 dataloader，比较 threads / channels_last / AMP。
  --mode infer         纯推理吞吐。

关键坑（本脚本已内建规避，详见 README 段注释）：
  1. ultralytics 8.4.175 在 device=cpu 时强制 self.args.workers = 0
     （engine/trainer.py:179）。要测 workers 必须子类化 DetectionTrainer 改回去。
  2. select_device() 会执行 torch.set_num_threads(min(8, cpu_count-1)) = 8
     （utils/torch_utils.py:366），静默覆盖调用方的线程数设置。
  3. 本机 CPU 无 AVX2/FMA/BMI2，原生 polars wheel 会 SIGILL 崩掉整个进程
     （0xC000001D）。推荐 pip install "polars[rtcompat]"；本脚本在启动时
     子进程探测一次，探测失败就自动注入 stub 兜底。

用法：
  $env:PYTHONIOENCODING='utf-8'
  G:\\Python\\Dev\\python\\python.exe bench/bench_train.py --mode ultralytics \
      --config y640_bs8_w4 --epochs 5 --warmup-steps 3 --json-out bench/results/y640_bs8_w4.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

T_START = time.perf_counter()


# ---------------------------------------------------------------- polars 崩溃规避
def probe_polars_ok() -> bool:
    """在子进程里试 import polars。原生 wheel 在本机是 SIGILL，Python 捕不到，
    只能在子进程里探测，用退出码判断。"""
    try:
        p = subprocess.run(
            [sys.executable, "-c", "import polars; polars.DataFrame({'a':[1]})"],
            capture_output=True, timeout=90,
        )
        return p.returncode == 0
    except Exception:
        return False


def install_polars_stub() -> None:
    import types

    pl = types.ModuleType("polars")
    pl.__doc__ = "bench stub: native polars SIGILLs on this AVX2-less CPU"

    def _boom(*a, **k):
        raise RuntimeError("polars stubbed out: CPU lacks AVX2/FMA/BMI2")

    for n in ("read_csv", "DataFrame", "Series", "col", "concat", "scan_csv"):
        setattr(pl, n, _boom)
    sel = types.ModuleType("polars.selectors")
    sel.numeric = _boom
    sel.cs = _boom
    pl.selectors = sel
    sys.modules["polars"] = pl
    sys.modules["polars.selectors"] = sel


if os.environ.get("BENCH_POLARS_STUB") == "1":
    _POLARS_MODE = "stub-forced"
    install_polars_stub()
elif probe_polars_ok():
    _POLARS_MODE = "real-ok"
else:
    _POLARS_MODE = "stub-auto"
    install_polars_stub()


import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    import psutil

    _PROC = psutil.Process()
except Exception:
    _PROC = None


def rss_mb() -> float:
    return (_PROC.memory_info().rss / 1024 / 1024) if _PROC else float("nan")


# ---------------------------------------------------------------- 命名档位
# 所有 imgsz 的数据像素尺寸都对齐 stride 32，避免额外 padding 混入测量。
DET_CFGS: dict[str, dict] = {
    "y640_bs8_w4": dict(imgsz=640, batch=8, workers=4),
    "y640_bs8_w0": dict(imgsz=640, batch=8, workers=0),
    "y640_bs8_w8": dict(imgsz=640, batch=8, workers=8),
    "y640_bs8_ram": dict(imgsz=640, batch=8, workers=4, cache="ram"),
    "y640_bs8_nomosaic": dict(imgsz=640, batch=8, workers=4, mosaic=0.0),
    "y640_bs8_t10": dict(imgsz=640, batch=8, workers=4, threads=10),
    "y640_bs8_t5": dict(imgsz=640, batch=8, workers=4, threads=5),
    "y640_bs16_w4": dict(imgsz=640, batch=16, workers=4),
    "y640_bs16_w8": dict(imgsz=640, batch=16, workers=8),
    "y512_bs16_w4": dict(imgsz=512, batch=16, workers=4),
    "y480_bs16_w4": dict(imgsz=480, batch=16, workers=4),
    "y416_bs16_w4": dict(imgsz=416, batch=16, workers=4),
    "y384_bs16_w4": dict(imgsz=384, batch=16, workers=4),
    "y320_bs16_w4": dict(imgsz=320, batch=16, workers=4),
    "y320_bs32_w4": dict(imgsz=320, batch=32, workers=4),
    "y256_bs16_w4": dict(imgsz=256, batch=16, workers=4),
    "y224_bs16_w4": dict(imgsz=224, batch=16, workers=4),
}


def resolve_cfg(args) -> dict:
    base = dict(DET_CFGS[args.config]) if args.config else dict(imgsz=640, batch=8, workers=4)
    for k in ("imgsz", "batch", "workers", "cache", "mosaic", "threads"):
        v = getattr(args, k, None)
        if v is not None:
            base[k] = v
    base.setdefault("cache", False)
    base.setdefault("mosaic", 1.0)
    base.setdefault("threads", None)
    return base


# ---------------------------------------------------------------- 统计工具
def quantiles(xs: list[float]) -> dict:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return {}
    n = len(xs)

    def q(p: float) -> float:
        return xs[min(n - 1, max(0, int(round(p * (n - 1)))))]

    mean = statistics.fmean(xs)
    return {
        "n": n, "min": xs[0], "p10": q(0.10), "p50": statistics.median(xs),
        "mean": mean, "p90": q(0.90), "max": xs[-1],
        "cv_pct": (statistics.pstdev(xs) / mean * 100) if len(xs) > 1 and mean else 0.0,
    }


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "cpu_count_logical": os.cpu_count(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "numpy": np.__version__,
        "polars_mode": _POLARS_MODE,
    }
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    try:
        import polars

        info["polars"] = getattr(polars, "__version__", "stub")
    except Exception:
        info["polars"] = "unavailable"
    return info


# ---------------------------------------------------------------- ultralytics 模式
_BENCH_WORKERS: int | None = None
_BENCH_THREADS: int | None = None
_BENCH_STATE: dict = {}


def _make_bench_trainer():
    from ultralytics.models.yolo.detect import DetectionTrainer

    class BenchTrainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 坑 1：device=cpu 时 ultralytics 硬写 workers=0。
            # train() -> _setup_train() 才建 DataLoader，所以这里改回去有效。
            if _BENCH_WORKERS is not None:
                self.args.workers = int(_BENCH_WORKERS)
            _BENCH_STATE["workers_actual"] = int(self.args.workers)

        def _setup_train(self):
            out = super()._setup_train()
            # 坑 2：select_device() 会把线程数重置为 min(8, cpu_count-1)=8，
            # 静默覆盖调用方设置，导致 threads 对比实验全部失效。这里纠正回来。
            if _BENCH_THREADS is not None:
                torch.set_num_threads(int(_BENCH_THREADS))
            _BENCH_STATE["threads_actual"] = torch.get_num_threads()
            return out

        def preprocess_batch(self, batch):
            try:
                _BENCH_STATE["batch_sizes"].append(int(batch["img"].shape[0]))
                _BENCH_STATE["imgsz_actual"].append(int(batch["img"].shape[-1]))
            except Exception:
                pass
            return super().preprocess_batch(batch)

    return BenchTrainer


def run_ultralytics(args) -> dict:
    global _BENCH_WORKERS, _BENCH_THREADS
    from ultralytics import YOLO

    cfg = resolve_cfg(args)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    _BENCH_THREADS = cfg.get("threads")
    _BENCH_WORKERS = cfg["workers"]

    _BENCH_STATE.clear()
    _BENCH_STATE.update({"batch_sizes": [], "imgsz_actual": [],
                         "workers_actual": None, "threads_actual": None})

    t0 = time.perf_counter()
    model = YOLO(args.weights)
    load_s = time.perf_counter() - t0

    starts: list[float] = []
    ends: list[float] = []
    epoch_of: list[int] = []
    cur = {"epoch": 0}

    def on_epoch_start(trainer):
        cur["epoch"] = int(getattr(trainer, "epoch", 0))

    def on_batch_start(trainer):
        starts.append(time.perf_counter())
        epoch_of.append(cur["epoch"])

    def on_batch_end(trainer):
        ends.append(time.perf_counter())

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_batch_start", on_batch_start)
    model.add_callback("on_train_batch_end", on_batch_end)

    overrides = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device="cpu",
        workers=cfg["workers"],
        cache=cfg.get("cache", False),
        mosaic=cfg.get("mosaic", 1.0),
        project=str(args.project),
        name=args.tag or args.config,
        exist_ok=True,
        plots=False,
        val=args.val,
        save=False,
        verbose=False,
        deterministic=False,
        seed=0,
        amp=False,          # 坑 3：老 CPU 上 autocast 有非法指令风险
        fraction=args.fraction,
        close_mosaic=0,
        max_det=100,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, erasing=0.0, mixup=0.0, copy_paste=0.0,
    )

    t0 = time.perf_counter()
    model.train(trainer=_make_bench_trainer(), **overrides)
    wall_s = time.perf_counter() - t0

    n = min(len(starts), len(ends))
    steps = [ends[i] - starts[i] for i in range(n)]
    waits = [starts[i + 1] - ends[i] for i in range(n - 1)]
    ep_of = epoch_of[:n]

    # 预热台阶：实测 epoch 0 的 p50 比 epoch 1 高约 2.3 倍
    # （320/bs16：ep0=4.23s vs ep1=1.82s）。根因是首轮 forward/backward 里的
    # oneDNN primitive JIT + 内存分配器预热 + 线程亲和收敛。所以默认丢掉
    # 前 --drop-epochs 个 epoch 的全部步，只统计稳态。
    keep = [i for i in range(n) if ep_of[i] >= args.drop_epochs]
    if not keep:
        keep = list(range(n))
    steps_kept = [steps[i] for i in keep]
    ep_kept = [ep_of[i] for i in keep]

    warm = args.warmup_steps
    steps_s = steps_kept[warm:]
    waits_s = waits[min(warm, max(0, len(waits) - 1)):]

    # 按 epoch 分组（含被丢弃的预热 epoch，方便报告里量化台阶大小）
    per_epoch = []
    for e in sorted(set(ep_of)):
        grp = [steps[i] for i in range(n) if ep_of[i] == e]
        if not grp:
            continue
        med = statistics.median(grp)
        per_epoch.append({
            "epoch": e, "n_steps": len(grp), "dropped": e < args.drop_epochs,
            "p50": med, "min": min(grp), "max": max(grp),
            "cv_pct": (statistics.pstdev(grp) / statistics.fmean(grp) * 100) if len(grp) > 1 else 0.0,
        })

    bs = int(statistics.median(_BENCH_STATE["batch_sizes"])) if _BENCH_STATE["batch_sizes"] else cfg["batch"]
    med_step = statistics.median(steps_s) if steps_s else float("nan")
    med_wall = (statistics.median([starts[i + 1] - starts[i] for i in range(n - 1)][warm:])
                if n > warm + 1 else med_step)

    # 「乐观值」= 取各 epoch 内最快的 20% 步的中位数，代表无跨 socket 抖动时的本机能力
    fast = sorted(steps_s)[: max(1, len(steps_s) // 5)]
    opt_step = statistics.median(fast) if fast else float("nan")

    return {
        "mode": "ultralytics",
        "config": args.config,
        "tag": args.tag or args.config,
        "cfg": cfg,
        "data": str(args.data),
        "weights": args.weights,
        "epochs": args.epochs,
        "fraction": args.fraction,
        "warmup_steps_dropped": warm,
        "drop_epochs": args.drop_epochs,
        "model_load_s": load_s,
        "train_wall_s": wall_s,
        "steps_per_epoch": (n // max(1, args.epochs)) if args.epochs else n,
        "steps_measured": len(steps_s),
        "batch_median": bs,
        "workers_requested": cfg["workers"],
        "workers_actual": _BENCH_STATE["workers_actual"],
        "threads_requested": cfg.get("threads"),
        "threads_actual": _BENCH_STATE["threads_actual"],
        "imgsz_actual_median": (statistics.median(_BENCH_STATE["imgsz_actual"])
                                if _BENCH_STATE["imgsz_actual"] else None),
        "compute_s": quantiles(steps_s),
        "compute_s_optimistic": quantiles(fast),
        "dataload_wait_s": quantiles(waits_s),
        "step_s_median": med_step,
        "step_s_optimistic": opt_step,
        "wall_step_s_median": med_wall,
        "img_per_s": (bs / med_step) if med_step == med_step and med_step > 0 else float("nan"),
        "img_per_s_optimistic": (bs / opt_step) if opt_step == opt_step and opt_step > 0 else float("nan"),
        "img_per_s_wall": (bs / med_wall) if med_wall == med_wall and med_wall > 0 else float("nan"),
        "per_epoch": per_epoch,
        "rss_mb_end": rss_mb(),
        "env": env_info(),
        "raw": {"compute_s": [round(x, 4) for x in steps_s],
                "wait_s": [round(x, 4) for x in waits_s],
                "epoch_of": ep_kept[warm:],
                "per_epoch_all": per_epoch,
                "batch_sizes": _BENCH_STATE["batch_sizes"]},
    }


# ---------------------------------------------------------------- torch 微基准
def make_synth_batch(batch: int, imgsz: int, ncls: int = 3, nbox: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(batch, 3, imgsz, imgsz, generator=g)
    boxes, cls, idx = [], [], []
    for b in range(batch):
        for _ in range(nbox):
            x1, y1 = torch.rand(2, generator=g).tolist()
            w = torch.rand(1, generator=g).item() * 0.3
            h = torch.rand(1, generator=g).item() * 0.3
            boxes.append([x1, y1, min(1.0, x1 + w), min(1.0, y1 + h)])
            cls.append(int(torch.randint(0, ncls, (1,), generator=g).item()))
            idx.append(b)
    return {
        "img": img,
        "bboxes": torch.tensor(boxes, dtype=torch.float32),
        "cls": torch.tensor(cls, dtype=torch.float32).view(-1, 1),
        "batch_idx": torch.tensor(idx, dtype=torch.float32),
    }


def build_torch_model(imgsz: int, weights: str):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    model = YOLO(weights).model
    ns = get_cfg()
    ns.imgsz = imgsz
    ns.box, ns.cls, ns.dfl = 7.5, 0.5, 1.5
    ns.epochs = 1
    model.args = ns
    model.criterion = None
    model.train()
    # YOLO(pt) 加载时会把 requires_grad 全置 False（推理优化），必须显式打开
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def run_torch(args) -> dict:
    cfg = resolve_cfg(args)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))

    model = build_torch_model(cfg["imgsz"], args.weights)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    bs = cfg["batch"]
    batch = make_synth_batch(bs, cfg["imgsz"], seed=args.seed)
    if args.channels_last:
        batch["img"] = batch["img"].to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    amp_mode = None if args.amp == "off" else args.amp

    def one_step():
        for p in model.parameters():
            p.grad = None
        t0 = time.perf_counter()
        if amp_mode:
            with torch.autocast(device_type="cpu", dtype=getattr(torch, amp_mode)):
                loss, _ = model.loss(batch)
            loss.float().sum().backward()
        else:
            loss, _ = model.loss(batch)
            loss.sum().backward()
        opt.step()
        return time.perf_counter() - t0, float(loss.detach().sum())

    steps, losses, err = [], [], None
    for i in range(args.steps + args.warmup_steps):
        try:
            dt, lv = one_step()
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
            break
        if i >= args.warmup_steps:
            steps.append(dt)
            losses.append(lv)

    med = statistics.median(steps) if steps else float("nan")
    return {
        "mode": "torch", "config": args.config, "cfg": cfg,
        "batch": bs, "imgsz": cfg["imgsz"],
        "channels_last": bool(args.channels_last), "amp": amp_mode or "off",
        "error": err,
        "step_s": quantiles(steps), "step_s_median": med,
        "img_per_s": (bs / med) if med == med and med > 0 else float("nan"),
        "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)] if losses else None,
        "rss_mb_end": rss_mb(), "env": env_info(),
        "raw": {"steps_s": [round(x, 4) for x in steps]},
    }


def run_infer(args) -> dict:
    cfg = resolve_cfg(args)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))

    model = build_torch_model(cfg["imgsz"], args.weights)
    model.eval()
    bs = cfg["batch"]
    batch = make_synth_batch(bs, cfg["imgsz"], seed=args.seed)["img"]

    times = []
    with torch.no_grad():
        for i in range(args.steps + args.warmup_steps):
            t0 = time.perf_counter()
            model(batch)
            dt = time.perf_counter() - t0
            if i >= args.warmup_steps:
                times.append(dt)
    med = statistics.median(times) if times else float("nan")
    return {
        "mode": "infer", "config": args.config, "batch": bs, "imgsz": cfg["imgsz"],
        "step_s": quantiles(times), "step_s_median": med,
        "img_per_s": (bs / med) if med == med and med > 0 else float("nan"),
        "env": env_info(), "raw": {"steps_s": [round(x, 4) for x in times]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ultralytics", "torch", "infer"], default="ultralytics")
    ap.add_argument("--config", default="", choices=[""] + sorted(DET_CFGS))
    ap.add_argument("--weights", default="bench/weights/yolo26n.pt")
    ap.add_argument("--data", default="bench/toy_dataset/data.yaml")
    ap.add_argument("--project", default="bench/results/runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--val", action="store_true", help="训练中执行 val（默认关闭，只测训练步）")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--drop-epochs", type=int, default=1,
                    help="丢弃前 N 个 epoch 的步（首轮有 ~2.3x 预热台阶，默认丢 1）")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--amp", default="off", choices=["off", "bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="")
    # 覆盖档位参数
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--mosaic", type=float, default=None)
    args = ap.parse_args()

    if args.mode == "ultralytics":
        res = run_ultralytics(args)
    elif args.mode == "torch":
        res = run_torch(args)
    else:
        res = run_infer(args)

    res["wall_total_s"] = time.perf_counter() - T_START
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=====BENCH_RESULT_BEGIN=====")
    print(json.dumps(res, ensure_ascii=False))
    print("=====BENCH_RESULT_END=====")


if __name__ == "__main__":
    main()
