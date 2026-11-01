# -*- coding: utf-8 -*-
"""inspect_sample: 看清一个训练样本的版式（底图/提示条/prompts 元数据）。

为什么要看：
    登录弹层里 2408 的「提示图标」不在底图上，而在弹层 header 区。
    要把它们合成到底部提示条，必须知道训练时的精确版式：
        提示条高度、图标尺寸、起始 x、间距。
    否则合成图与训练分布不一致，检测器/匹配器都会失效。

用法：
    python tools/inspect_sample.py
    python tools/inspect_sample.py --sid t2408_1810855608033804288
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default=None)
    ap.add_argument("--samples", default="data/samples")
    args = ap.parse_args()

    root = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    if args.sid:
        sids = [args.sid]
    else:
        sids = sorted(d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d)))[:1]

    for sid in sids:
        d = os.path.join(root, sid)
        mp = os.path.join(d, "meta.json")
        main_p = os.path.join(d, "main.jpg")
        print(f"=== {sid} ===")
        if os.path.exists(mp):
            meta = json.load(open(mp, encoding="utf-8"))
            print(json.dumps(meta, ensure_ascii=False, indent=2)[:1200])
        if not os.path.exists(main_p):
            print("  无 main.jpg")
            continue
        im = imread_u(main_p)
        H, W = im.shape[:2]
        print(f"\nmain.jpg 尺寸: {W}x{H}")

        # 探测底部白条高度：从下往上找整行接近纯白的行
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        strip_h = 0
        for y in range(H - 1, -1, -1):
            row = g[y]
            if float((row > 245).mean()) > 0.55:
                strip_h += 1
            else:
                break
        print(f"底部近白行数(估算提示条高): {strip_h}")

        if strip_h > 5:
            strip = im[H - strip_h:, :]
            gp = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            cols = (gp < 200).sum(axis=0)          # 每列的非白像素数
            on = cols > 0
            # 找连续列区间 = 每个图标的水平范围
            segs = []
            s = None
            for x, v in enumerate(on):
                if v and s is None:
                    s = x
                elif not v and s is not None:
                    if x - s >= 6:
                        segs.append((s, x - 1))
                    s = None
            if s is not None and len(on) - s >= 6:
                segs.append((s, len(on) - 1))
            print(f"提示条内图标水平区间: {segs}")
            for i, (x1, x2) in enumerate(segs):
                sub = gp[:, x1:x2 + 1]
                rows = np.where((sub < 200).any(axis=1))[0]
                print(f"  图标{i}: x={x1}..{x2} (w={x2-x1+1}) "
                      f"y={rows.min()}..{rows.max()} (h={rows.max()-rows.min()+1}) "
                      f"(提示条内相对坐标)")
            out = os.path.join(REPO, "out", "inspect")
            os.makedirs(out, exist_ok=True)
            p = os.path.join(out, f"{sid}_strip.png")
            cv2.imencode(".png", strip)[1].tofile(p)
            print(f"  提示条截图 -> {os.path.relpath(p, REPO)}")

        # 同时看看有没有单独的 prompt 文件
        for f in sorted(os.listdir(d)):
            if f.endswith((".jpg", ".png")):
                fp = os.path.join(d, f)
                i2 = imread_u(fp)
                if i2 is not None:
                    print(f"  文件 {f}: {i2.shape[1]}x{i2.shape[0]}")


if __name__ == "__main__":
    main()
