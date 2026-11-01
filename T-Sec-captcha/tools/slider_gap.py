# -*- coding: utf-8 -*-
"""slider_gap: 天御滑块缺口检测。

════════════════════════════════════════════════════════════════
为什么不能用模板匹配 fg-item：
    滑块底图上的缺口是**挖空的暗块**，而 fg-item 拼图块是**亮的**（真实图像内容）。
    用 TM_CCOEFF_NORMED 拿亮块去匹配暗口，相关性极低
    （实测 score 仅 0.37~0.69，位置全错）。

实测出来的两个有效特征（在 3 张真实底图上标定）：
    A) 缺口有一圈**浅灰亮描边**（灰度 150~175），内部暗（50~75）
       → 用「亮掩码 + 轮廓 + 内部暗」精确定位，误差 0~1px（bg_2/bg_3）
    B) 暗方块嵌在较亮背景里 → 帽形滤波 blur(outer)-blur(inner)
       → 窗口 56/68 最优；轮廓被草叶打断时兜底（bg_1）
       窗口不能太大：outer=92 时被大面积暗区带偏（实测错到 cx=139）

用法：
    python tools/slider_gap.py --dir out/login10x/XXX
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# 标定出来的默认值
HAT_INNER = 56
HAT_OUTER = 68
BRIGHT_THR = 130
INNER_DARK_MAX = 108


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def find_gap_outline(bg, bright_thr=BRIGHT_THR, min_side=50, max_side=95,
                     min_x=120, inner_dark_max=INNER_DARK_MAX, verbose=False):
    """路径 A：亮描边 + 轮廓。返回 dict 或 None。"""
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    m = (gray > bright_thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        if abs(w - h) > 18:
            continue
        if x + w // 2 < min_x:
            continue
        inner = gray[y + 8:y + h - 8, x + 8:x + w - 8]
        if inner.size < 200:
            continue
        im_ = float(inner.mean())
        if im_ > inner_dark_max:
            continue
        cand = {"cx": x + w / 2.0, "cy": y + h / 2.0, "w": w, "h": h,
                "inner": round(im_, 1), "src": "outline",
                "score": round(inner_dark_max - im_, 1)}
        # 内部越暗越像缺口
        if best is None or cand["score"] > best["score"]:
            best = cand
    if verbose and best:
        print(f"  A/outline -> cx={best['cx']:.0f} cy={best['cy']:.0f} "
              f"{best['w']}x{best['h']} inner={best['inner']}")
    return best


def find_gap_hat(bg, inner=HAT_INNER, outer=HAT_OUTER, min_x=120,
                 verbose=False):
    """路径 B：帽形滤波。返回 dict 或 None。"""
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    def bm(img, k):
        return cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT)

    m_in, m_out = bm(gray, inner), bm(gray, outer)
    s = m_out - m_in
    pad = max(inner, outer) // 2
    s[:pad, :] = -1e6
    s[H - pad:, :] = -1e6
    s[:, :max(min_x, pad)] = -1e6
    s[:, W - pad:] = -1e6
    _mn, mx, _ml, ml = cv2.minMaxLoc(s)
    cx, cy = int(ml[0]), int(ml[1])
    if verbose:
        print(f"  B/hat -> ({cx},{cy}) score={mx:.1f} "
              f"inner={m_in[cy, cx]:.1f} outer={m_out[cy, cx]:.1f}")
    if mx <= 3.0:
        return None
    return {"cx": float(cx), "cy": float(cy), "w": 60.0, "h": 60.0,
            "inner": round(float(m_in[cy, cx]), 1),
            "src": "hat", "score": round(float(mx), 1)}


def find_gap(bg, verbose=False):
    """返回 dict(cx, cy, w, h, src, score) 或 None。"""
    if bg is None:
        return None
    a = find_gap_outline(bg, verbose=verbose)
    b = find_gap_hat(bg, verbose=verbose)
    if a is None and b is None:
        return None
    if a is not None and b is not None:
        # 两者接近 → 用轮廓的精确尺寸；否则信任轮廓（它要求内部暗，误检率低）
        if abs(a["cx"] - b["cx"]) < 45:
            return a
        return a
    return a or b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    paths = []
    if args.bg:
        paths = [args.bg if os.path.isabs(args.bg) else os.path.join(REPO, args.bg)]
    elif args.dir:
        d = args.dir if os.path.isabs(args.dir) else os.path.join(REPO, args.dir)
        paths = sorted(glob.glob(os.path.join(d, "bg_*.png")))
    else:
        for p in sorted(glob.glob(os.path.join(REPO, "out", "login10x", "*",
                                               "bg_*.png"))):
            im = imread_u(p)
            if im is not None and im.shape[0] != 480:
                paths.append(p)
    if not paths:
        raise SystemExit("没找到滑块底图")

    for p in paths:
        im = imread_u(p)
        if im is None:
            continue
        print(f"\n=== {os.path.relpath(p, REPO)}  {im.shape[1]}x{im.shape[0]} ===")
        r = find_gap(im, verbose=True)
        if r is None:
            print("  未找到缺口")
            continue
        print(f"  -> ({r['cx']:.0f}, {r['cy']:.0f}) {r['w']}x{r['h']} "
              f"src={r['src']} score={r['score']}")
        vis = im.copy()
        cv2.rectangle(vis, (int(r["cx"] - r["w"] / 2), int(r["cy"] - r["h"] / 2)),
                      (int(r["cx"] + r["w"] / 2), int(r["cy"] + r["h"] / 2)),
                      (0, 0, 255), 2)
        cv2.line(vis, (int(r["cx"]), 0), (int(r["cx"]), vis.shape[0]), (0, 255, 0), 1)
        out = args.dump or os.path.join(
            REPO, "out", "slider_dbg",
            os.path.basename(os.path.dirname(p)) + "_" + os.path.basename(p))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imencode(".png", vis)[1].tofile(out)
        print(f"  可视化 -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
