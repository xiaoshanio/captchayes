# -*- coding: utf-8 -*-
"""tune_seg: 用「配对覆盖率」而不是「图标数」来标定分割参数。

为什么指标要换：
    凭图标数调参会走错方向 —— 干扰图标多检几个对训练无害（模型该学会区分），
    但**漏掉一个目标图标**就整题废掉（配对不完整，标签错）。
    所以真正的目标函数是：**每题 3 个提示图标都能在底图找到对应**的比例。

本脚本跑一组参数网格，输出：
    exact：3/3 全配上的题占比（主指标）
    recall：平均配对率（辅助）
    平均底图图标数（次要，用于观察噪声）
    ms/题（性能）

用法：
    python tools/tune_seg.py --samples data/samples --limit 60
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from seg_2408 import segment_icons_2408, imread_u          # noqa: E402
from autolabel_2408 import split_prompt_icons            # noqa: E402
from icon_match import normalize_shape, assign_icons      # noqa: E402


def evaluate(dirs, params, pair_thr=0.42):
    ex = 0
    rates = []
    ns = []
    t0 = time.time()
    detail = []
    for mp in dirs:
        d = os.path.dirname(mp)
        sid = os.path.basename(d)
        try:
            bg = imread_u(os.path.join(d, "bg.jpg"))
            meta = json.load(open(mp, encoding="utf-8"))
            pr = imread_u(os.path.join(d, meta["prompts"][0]["file"]))
            if bg is None or pr is None:
                continue
            icons = split_prompt_icons(pr)
            seg, _ = segment_icons_2408(bg, **params)
            p_norms = [normalize_shape(it["img"])[0] for it in icons]
            b_norms = [normalize_shape(bg[y1:y2, x1:x2])[0]
                       for (x1, y1, x2, y2) in [s["box"] for s in seg]]
            pairs, scores, _m = assign_icons(p_norms, b_norms)
            good = sum(1 for _i, _j, s in pairs if s >= pair_thr)
            rates.append(good / max(len(icons), 1))
            ns.append(len(seg))
            if good == len(icons) and len(icons) > 0:
                ex += 1
            detail.append({"sid": sid, "n_prompt": len(icons),
                           "n_bg": len(seg), "good": good,
                           "scores": [round(float(s), 3) for s in
                                      [p[2] for p in pairs]]})
        except Exception as e:
            detail.append({"sid": sid, "err": f"{type(e).__name__}: {e}"})
    ms = (time.time() - t0) * 1000 / max(len(dirs), 1)
    return {
        "params": params,
        "n": len(rates),
        "exact": ex,
        "exact_rate": round(ex / max(len(rates), 1), 4),
        "pair_rate": round(float(np.mean(rates)) if rates else 0, 4),
        "mean_bg_icons": round(float(np.mean(ns)) if ns else 0, 2),
        "ms_per_item": round(ms, 1),
        "detail": detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="out/tune_seg.json")
    args = ap.parse_args()

    root = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    dirs = sorted(glob.glob(os.path.join(root, "*", "meta.json")))[:args.limit]
    print(f"[scan] {len(dirs)} 题")

    grid = []
    for min_side in (18, 22, 26):
        for core in (0.62, 0.70, 0.78):
            grid.append({"min_side": min_side, "core_ratio_min": core,
                         "p25_max": 26.0, "min_ink": 250})
    for gap in (10, 14, 20):
        grid.append({"min_side": 22, "core_ratio_min": 0.70,
                     "p25_max": 26.0, "min_ink": 250, "gap": gap})
    grid.append({"min_side": 22, "core_ratio_min": 0.70,
                 "p25_max": 32.0, "min_ink": 250})

    results = []
    for i, params in enumerate(grid):
        r = evaluate(dirs, params)
        results.append(r)
        print(f"  [{i+1}/{len(grid)}] {params} -> exact={r['exact_rate']:.1%} "
              f"pair={r['pair_rate']:.1%} bg={r['mean_bg_icons']} "
              f"{r['ms_per_item']:.0f}ms")

    results.sort(key=lambda r: (-r["exact_rate"], -r["pair_rate"]))
    best = results[0]
    print("\n=== 最优参数 ===")
    print(f"  {best['params']}")
    print(f"  3/3 全配 : {best['exact_rate']:.1%}  ({best['exact']}/{best['n']})")
    print(f"  平均配率 : {best['pair_rate']:.1%}")
    print(f"  底图图标 : {best['mean_bg_icons']}")
    print(f"  耗时     : {best['ms_per_item']:.0f} ms/题")

    with open(os.path.join(REPO, args.out), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n明细 -> {args.out}")


if __name__ == "__main__":
    main()
