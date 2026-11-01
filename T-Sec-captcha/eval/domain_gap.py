# -*- coding: utf-8 -*-
"""domain_gap: 量化两个数据域之间的差异 —— 用数字证明"漂移"确实存在。

为什么要这个：
    "数据过于老了"在真机上的表现是出题方改版（换字形/描边/尺寸/背景）。
    如果只在单一域上验证，就永远测不出"老化"这件事。
    本脚本给出**可复现的域距离指标**，作为漂移实验的前提证据。

指标（都不需要模型，纯统计）：
    1. 图像尺寸 / 提示条高度
    2. 图标框的尺寸分布（w,h）与数量分布
    3. 灰度直方图的 JS 散度
    4. 边缘密度（Canny）均值
    5. 平均饱和度 / 亮度
    6. 提示条图标高度占比
    7. 结构相似性代理：把两域的均值图对齐后算 SSIM（skimage 不可用时用相关）

用法：
    python eval/domain_gap.py --a data/drift_v1 --b data/drift_v2 --split val
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def load_split(d, split):
    img_dir = os.path.join(d, "images", split)
    lab_dir = os.path.join(d, "labels", split)
    pr_dir = os.path.join(d, "pairs", split)
    out = []
    for f in sorted(os.listdir(lab_dir)):
        if not f.endswith(".txt"):
            continue
        base = f[:-4]
        ip = os.path.join(img_dir, base + ".jpg")
        pp = os.path.join(pr_dir, base + ".json")
        if os.path.exists(ip) and os.path.exists(pp):
            out.append((ip, os.path.join(lab_dir, f), pp))
    return out


def hist_of(gray):
    h = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
    return h / max(h.sum(), 1)


def js_div(p, q, eps=1e-9):
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def describe(d, split="val", limit=200):
    items = load_split(d, split)[:limit]
    if not items:
        return None
    sizes, strip_hs = [], []
    bw_list, bh_list, n_box = [], [], []
    hist_acc = None
    edge_density, sat_mean, val_mean, prompt_h = [], [], [], []
    for ip, lp, pp in items:
        img = imread_u(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        sizes.append((W, H))
        truth = json.load(open(pp, encoding="utf-8"))
        strip_hs.append(truth.get("strip_h", 56))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = hist_of(gray)
        hist_acc = h if hist_acc is None else hist_acc + h
        edges = cv2.Canny(gray, 60, 160)
        edge_density.append(float((edges > 0).mean()))
        sat_mean.append(float(hsv[:, :, 1].mean()))
        val_mean.append(float(hsv[:, :, 2].mean()))
        for line in open(lp, encoding="utf-8"):
            v = line.split()
            if len(v) != 5:
                continue
            c = int(float(v[0]))
            cx, cy, bw, bh = [float(t) for t in v[1:]]
            if c == 0:
                bw_list.append(bw * W)
                bh_list.append(bh * H)
            n_box.append(c)
        for q in truth.get("pairs", []):
            pb = q["prompt_box"]
            prompt_h.append(pb[3] - pb[1])
    hist_acc = hist_acc / max(len(items), 1)
    return {
        "n": len(items),
        "size_mean": [round(st.mean([s[0] for s in sizes]), 1),
                      round(st.mean([s[1] for s in sizes]), 1)],
        "strip_h_mean": round(st.mean(strip_hs), 1),
        "box_w_med": round(st.median(bw_list), 1) if bw_list else None,
        "box_h_med": round(st.median(bh_list), 1) if bh_list else None,
        "box_area_med": round(st.median([w * h for w, h in zip(bw_list, bh_list)]), 0) if bw_list else None,
        "n_cls0_mean": round(st.mean([n_box.count(0) for _ in [0]]) if False else
                             sum(1 for c in n_box if c == 0) / max(len(items), 1), 2),
        "n_cls2_mean": round(sum(1 for c in n_box if c == 2) / max(len(items), 1), 2),
        "edge_density": round(st.mean(edge_density), 4),
        "sat_mean": round(st.mean(sat_mean), 1),
        "val_mean": round(st.mean(val_mean), 1),
        "prompt_h_med": round(st.median(prompt_h), 1) if prompt_h else None,
        "_hist": hist_acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/drift_v1", help="域 A（旧域）")
    ap.add_argument("--b", default="data/drift_v2", help="域 B（漂移域）")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="out/domaingap")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    A = describe(args.a, args.split, args.limit)
    B = describe(args.b, args.split, args.limit)
    if not A or not B:
        raise SystemExit("数据不足，先生成两个域")

    ha, hb = A.pop("_hist"), B.pop("_hist")
    js = js_div(ha, hb)

    print("=" * 92)
    print(f"域差异量化：A={args.a}  vs  B={args.b}   (split={args.split})")
    print("=" * 92)
    keys = ["n", "size_mean", "strip_h_mean", "box_w_med", "box_h_med", "box_area_med",
            "n_cls0_mean", "n_cls2_mean", "edge_density", "sat_mean", "val_mean",
            "prompt_h_med"]
    print(f"  {'指标':<18}{'域A(旧)':>16}{'域B(漂移)':>16}{'相对变化':>14}")
    for k in keys:
        va, vb = A.get(k), B.get(k)
        if va is None or vb is None:
            continue
        if isinstance(va, list):
            rel = f"[{vb[0]-va[0]:+.0f},{vb[1]-va[1]:+.0f}]"
            sa, sb = f"{va}", f"{vb}"
        elif va == 0:
            rel = "n/a"
            sa, sb = f"{va}", f"{vb}"
        else:
            rel = f"{(vb-va)/abs(va)*100:+.1f}%"
            sa, sb = f"{va}", f"{vb}"
        print(f"  {k:<18}{sa:>16}{sb:>16}{rel:>14}")

    print(f"\n  灰度直方图 JS 散度 : {js:.5f}   (0=完全相同, 越大差异越大)")
    if js > 0.02:
        print("    -> 背景灰度分布差异明显（域漂移成立）")
    elif js > 0.005:
        print("    -> 背景灰度分布有可测差异")
    else:
        print("    -> 背景灰度分布接近（漂移主要来自字形/几何）")

    # 结论：漂移维度清单
    drift = []
    if A["box_area_med"] and B["box_area_med"]:
        r = B["box_area_med"] / A["box_area_med"]
        if r < 0.75 or r > 1.33:
            drift.append(f"图标尺寸 {r:.2f}x（{'变小' if r < 1 else '变大'}）")
    if A["strip_h_mean"] != B["strip_h_mean"]:
        drift.append(f"提示条高度 {A['strip_h_mean']} -> {B['strip_h_mean']}")
    if A["n_cls0_mean"] != B["n_cls2_mean"]:
        pass
    if abs(B["edge_density"] - A["edge_density"]) / max(A["edge_density"], 1e-6) > 0.15:
        drift.append(f"边缘密度 {A['edge_density']} -> {B['edge_density']}")
    if abs(B["sat_mean"] - A["sat_mean"]) / max(A["sat_mean"], 1e-6) > 0.15:
        drift.append(f"饱和度 {A['sat_mean']} -> {B['sat_mean']}")
    if js > 0.005:
        drift.append(f"背景灰度分布 (JS={js:.4f})")
    if A["n_cls0_mean"] != B["n_cls0_mean"]:
        drift.append(f"底图图标数 {A['n_cls0_mean']} -> {B['n_cls0_mean']}")

    print("\n  漂移维度清单：")
    for d in drift:
        print(f"    - {d}")
    if not drift:
        print("    - （未检测到显著漂移，两域过近，不适合做漂移实验）")

    rep = {"A": A, "B": B, "js_divergence": round(js, 6), "drift_dims": drift}
    json.dump(rep, open(os.path.join(args.out, "gap.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n报告 -> {args.out}/gap.json")


if __name__ == "__main__":
    main()
