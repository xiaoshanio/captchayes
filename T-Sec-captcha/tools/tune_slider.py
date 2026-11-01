# -*- coding: utf-8 -*-
"""tune_slider: 网格搜索滑块缺口检测的最佳窗口参数。

已知真值（人工看图确定）：
    bg_1.png: 缺口中心 ≈ (467, 277)
    bg_2.png: 缺口中心 ≈ (314, 234)
    bg_3.png: 缺口中心 ≈ (447, 254)

两类打分：
    hatN : blur(outer) - blur(inner)          —— 暗心 vs 外圈
    ring : blur(BorderN) - blur(inner)        —— 亮描边 vs 暗心（更贴近真实特征）

目标：让真值在候选里排第 1，且与第二名有足够间隔。

用法：
    python tools/tune_slider.py --dir out/login10x/20261031_124914
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from slider_gap import imread_u  # noqa: E402

TRUTH = {"bg_1.png": (467, 277), "bg_2.png": (314, 234), "bg_3.png": (447, 254)}


def blur(img, k):
    return cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT)


def top_peaks(s, n=3, padx=110, pady=60):
    s = s.copy()
    H, W = s.shape
    s[:pady, :] = -1e6
    s[H - pady:, :] = -1e6
    s[:, :padx] = -1e6
    s[:, W - pady:] = -1e6
    out = []
    for _ in range(n):
        _r, mx, _l, ml = cv2.minMaxLoc(s)
        out.append((ml[0], ml[1], round(float(mx), 1)))
        y, x = ml[1], ml[0]
        s[max(0, y - 70):y + 70, max(0, x - 70):x + 70] = -1e6
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/login10x/20261031_124914")
    args = ap.parse_args()
    d = args.dir if os.path.isabs(args.dir) else os.path.join(REPO, args.dir)

    imgs = {}
    for name in TRUTH:
        p = os.path.join(d, name)
        if os.path.exists(p):
            imgs[name] = cv2.cvtColor(imread_u(p), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if not imgs:
        raise SystemExit(f"没找到底图: {d}")

    results = []
    # 组合：inner 越小越聚焦暗心；outer/border 控制环宽
    for inner in (48, 52, 56, 60):
        for border in (64, 68, 72, 76):
            if border <= inner:
                continue
            for kind in ("hat", "ring"):
                ok = 0
                detail = []
                for name, g in imgs.items():
                    if kind == "hat":
                        s = blur(g, border) - blur(g, inner)
                    else:
                        # 亮描边 - 暗心：描边窗口略大于 inner
                        s = blur(g, border) - blur(g, inner)
                        # 再减去更外圈，抑制大面积暗区
                        s = s - (blur(g, border * 2) - blur(g, border)) * 0.5
                    t = top_peaks(s, 2)
                    tx, ty = TRUTH[name]
                    hit = abs(t[0][0] - tx) <= 25 and abs(t[0][1] - ty) <= 25
                    margin = t[0][2] - t[1][2] if len(t) > 1 else 0
                    ok += 1 if hit else 0
                    detail.append(f"{name}:{'HIT' if hit else 'miss'}@"
                                  f"({t[0][0]},{t[0][1]})m={margin:.0f}")
                results.append((ok, kind, inner, border, detail))

    results.sort(key=lambda r: (-r[0], -min(
        1, float(r[4][0].split("m=")[-1]) if "m=" in r[4][0] else 0)))
    print(f"{'kind':5s} {'inner':>5s} {'border':>6s} {'hits':>4s}  detail")
    print("-" * 100)
    for ok, kind, inner, border, detail in results[:14]:
        print(f"{kind:5s} {inner:>5d} {border:>6d} {ok:>4d}  " + " | ".join(detail))


if __name__ == "__main__":
    main()
