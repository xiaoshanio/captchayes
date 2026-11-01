# -*- coding: utf-8 -*-
"""check_live_2408: 在真实登录的合成图上跑模型，把预测框画出来核对。

目的：
    区分「模型答错」和「点击/时序问题」。
    离线评测 100%，但线上 2408 通过率只有 ~19%，
    必须肉眼确认模型在**线上图**上给出的框对不对。

用法：
    python tools/check_live_2408.py --dir out/login10x/<run>/try08
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.dir if os.path.isabs(args.dir) else os.path.join(REPO, args.dir)
    out = args.out or os.path.join(REPO, "out", "check_live",
                                   os.path.basename(os.path.dirname(d)) + "_" +
                                   os.path.basename(d))
    os.makedirs(out, exist_ok=True)

    from src.captcha import TextSelectCaptcha
    cap = TextSelectCaptcha()
    cap.yolo.conf_threshold = args.conf

    imgs = sorted(glob.glob(os.path.join(d, "c2408_*.jpg")))
    if not imgs:
        raise SystemExit(f"{d} 里没有 c2408_*.jpg")
    for p in imgs:
        im = imread_u(p)
        if im is None:
            continue
        try:
            res = cap.run_dict(p)
        except Exception as e:
            print(f"{os.path.basename(p)}: 推理失败 {type(e).__name__}: {e}")
            continue
        corps = res.get("corp") or []
        vis = im.copy()
        H = im.shape[0]
        bg_h = H - 56
        for i, c in enumerate(corps):
            x1, y1 = int(c["x1"]), int(c["y1"])
            x2, y2 = int(c["x2"]), int(c["y2"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, str(i + 1), (x1 + 2, max(16, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                        cv2.LINE_AA)
        # 顺便把全部检测框（含未选中）用细线画出来，便于看有没有漏检
        det = cap.yolo.inference(im)
        for x in det:
            if int(x[5]) == 0:
                cv2.rectangle(vis, (int(x[0]), int(x[1])),
                              (int(x[2]), int(x[3])), (0, 200, 0), 1)
        dst = os.path.join(out, os.path.basename(p).replace(".jpg", "_pred.jpg"))
        cv2.imencode(".jpg", vis)[1].tofile(dst)
        pts = [(round((c["x1"] + c["x2"]) / 2), round((c["y1"] + c["y2"]) / 2))
               for c in corps]
        print(f"{os.path.basename(p)}: {len(corps)} 点 {pts} -> {dst}")

    print(f"\n产物 -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
