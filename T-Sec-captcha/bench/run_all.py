#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bench/run_all.py

批量执行 bench_train.py 的各档配置，每档独立子进程 + 独立超时，
结果落 bench/results/*.json，汇总落 bench/results/summary.json。

设计要点：
  - 每档一个子进程：避免 ultralytics 全局状态（线程数、EMA、trainer 单例）串味。
  - 每档独立超时：ultralytics 在 CPU 上有卡死风险，卡住就 kill 并标记 timeout。
  - epochs 按「目标步数」自动算，保证每档都测到 >= target_steps 个有效步；
    再加上 --drop-epochs 1 丢掉首轮预热台阶。
  - 关键档位可 --repeats 重复测量，暴露同配置内的方差（本机存在 2x 台阶）。

用法:
  $env:PYTHONIOENCODING='utf-8'
  python bench/run_all.py --target-steps 30 --warmup 3 --drop-epochs 1 --timeout 1200
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# (config, n_images, repeats)
PLAN = [
    ("y640_bs8_w4", 368, 1),      # 主档：部署契约 640 / bs8
    ("y320_bs16_w4", 368, 1),     # 主档：低分辨率对照
    ("y640_bs16_w4", 368, 1),
    ("y416_bs16_w4", 368, 1),
    ("y512_bs16_w4", 368, 1),
    ("y256_bs16_w4", 368, 1),
    ("y640_bs8_w0", 368, 1),
    ("y640_bs8_w8", 368, 1),
    ("y640_bs16_w8", 368, 1),
    ("y640_bs8_t10", 368, 1),
    ("y640_bs8_t5", 368, 1),
    ("y640_bs8_ram", 368, 1),
    ("y640_bs8_nomosaic", 368, 1),
    ("y320_bs32_w4", 368, 1),
    ("y224_bs16_w4", 368, 1),
]

# 关键档位额外重复测量（放在 PLAN 之后单独跑）
REPEAT_KEYS = ["y640_bs8_w4", "y320_bs16_w4"]


def run_one(cfg: str, n_images: int, args, tag_suffix: str = "") -> dict:
    batch = int(cfg.split("_bs")[1].split("_")[0])
    steps_per_epoch = max(1, n_images // batch)
    need = args.target_steps + args.warmup
    epochs = max(1 + args.drop_epochs,
                 args.drop_epochs + math.ceil(need / steps_per_epoch))

    tag = cfg + tag_suffix
    out_json = HERE / "results" / f"{tag}.json"
    log = HERE / "logs" / f"{tag}.log"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python), str(HERE / "bench_train.py"),
        "--mode", "ultralytics",
        "--config", cfg,
        "--epochs", str(epochs),
        "--warmup-steps", str(args.warmup),
        "--drop-epochs", str(args.drop_epochs),
        "--tag", tag,
        "--json-out", str(out_json),
        "--data", str(args.data),
        "--weights", str(args.weights),
        "--project", str(HERE / "results" / "runs"),
    ]
    t0 = time.perf_counter()
    status, err = "ok", ""
    try:
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=args.timeout, cwd=str(ROOT))
        if p.returncode != 0:
            status = f"exit_{p.returncode}"
    except subprocess.TimeoutExpired:
        status, err = "timeout", f"killed after {args.timeout}s"
    wall = time.perf_counter() - t0

    res = {"config": cfg, "tag": tag, "epochs": epochs,
           "steps_per_epoch": steps_per_epoch, "status": status,
           "error": err, "process_wall_s": round(wall, 1)}
    if out_json.exists():
        try:
            res.update(json.loads(out_json.read_text(encoding="utf-8")))
        except Exception as e:
            res["parse_error"] = repr(e)
    else:
        if log.exists():
            res["log_tail"] = "\n".join(
                log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=r"G:\Python\Dev\python\python.exe")
    ap.add_argument("--data", default="bench/toy_dataset/data.yaml")
    ap.add_argument("--weights", default="bench/weights/yolo26n.pt")
    ap.add_argument("--target-steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--drop-epochs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--only", default="")
    ap.add_argument("--repeats", type=int, default=0, help="对 REPEAT_KEYS 额外重复次数")
    args = ap.parse_args()

    plan = PLAN
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        plan = [p for p in PLAN if p[0] in want]

    results = []
    jobs = [(c, n, "") for c, n, _ in plan]
    if args.repeats:
        for k in REPEAT_KEYS:
            if any(c == k for c, _, _ in plan):
                n = next(n for c, n, _ in plan if c == k)
                for r in range(1, args.repeats + 1):
                    jobs.append((k, n, f"_r{r}"))

    for i, (cfg, n_img, sfx) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {cfg}{sfx} ...", flush=True)
        r = run_one(cfg, n_img, args, sfx)
        sig = ""
        if r.get("status") == "ok":
            sig = (f" step_p50={r.get('step_s_median'):.3f}s opt={r.get('step_s_optimistic'):.3f}s"
                   f" img/s={r.get('img_per_s'):.2f} th={r.get('threads_actual')} wk={r.get('workers_actual')}")
        print(f"    -> {r['status']}{sig}  ({r['process_wall_s']}s)", flush=True)
        results.append(r)
        (HERE / "results" / "summary.partial.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    out = HERE / "results" / "summary.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 关键档位的重复测量汇总
    agg = {}
    for r in results:
        if r.get("status") != "ok" or not r.get("step_s_median"):
            continue
        agg.setdefault(r["config"], []).append({
            "tag": r["tag"], "p50": r["step_s_median"],
            "opt": r.get("step_s_optimistic"), "min": r["compute_s"]["min"],
            "max": r["compute_s"]["max"], "cv": r["compute_s"]["cv_pct"],
        })
    repeat_summary = {}
    for k, v in agg.items():
        if len(v) > 1:
            p50s = [x["p50"] for x in v]
            opt = [x["opt"] for x in v if x["opt"]]
            repeat_summary[k] = {
                "runs": v, "p50_median": statistics.median(p50s),
                "p50_min": min(p50s), "p50_max": max(p50s),
                "opt_median": statistics.median(opt) if opt else None,
                "spread_pct": (max(p50s) - min(p50s)) / min(p50s) * 100 if min(p50s) else 0,
            }
    (HERE / "results" / "repeat_summary.json").write_text(
        json.dumps(repeat_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n[run_all] ok={len(ok)}/{len(results)} -> {out}")
    if repeat_summary:
        print("[run_all] repeat spread:")
        for k, v in repeat_summary.items():
            print(f"  {k}: p50 {v['p50_min']:.3f}~{v['p50_max']:.3f}s "
                  f"(spread {v['spread_pct']:.0f}%), opt_p50={v['opt_median']:.3f}s")


if __name__ == "__main__":
    main()
