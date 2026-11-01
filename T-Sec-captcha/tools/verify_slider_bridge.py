# -*- coding: utf-8 -*-
"""verify_slider_bridge: 把桥接后的缺口预测画到 1.0 放弃的真实底图上。

目的：
    1.0 在这 3 张图上给出 conf=0.3/0.0/0.3（低于 0.45 阈值，直接放弃）。
    我的实现声称能找到缺口。必须**肉眼验证**：如果预测错，桥接就是帮倒忙。

用法：
    python tools/verify_slider_bridge.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from slider_gap import find_gap  # noqa: E402

SRC_DIR = os.path.join(REPO, "out", "test_10_live")
OUT_DIR = os.path.join(REPO, "out", "slider_bridge_check")

PROJ_10 = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in (PROJ_10, os.path.join(PROJ_10, "captcha")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import captcha.captcha_solver as cs
    except Exception:
        cs = None

    names = ["live_slider_bg_1.png", "live_slider_bg_2.png",
             "live_slider_bg_3.png"]
    for n in names:
        p = os.path.join(SRC_DIR, n)
        if not os.path.exists(p):
            print("缺文件 " + n)
            continue
        im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        H, W = im.shape[:2]
        vis = im.copy()

        old = None
        if cs is not None:
            try:
                old = cs.find_slider_gap(im)
            except Exception:
                old = None
        r = find_gap(im, verbose=False)

        # 画 1.0 的预测（蓝）与我的预测（红）
        if old:
            try:
                ox = int(old[1])
                if ox > 0:
                    cv2.line(vis, (ox, 0), (ox, H), (255, 0, 0), 2)
                    cv2.putText(vis, "1.0 conf=" + str(old[0]) + " x=" + str(ox),
                                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 0, 0), 2, cv2.LINE_AA)
            except Exception:
                pass
        if r:
            cx, cy, w, h = int(r["cx"]), int(r["cy"]), int(r["w"]), int(r["h"])
            cv2.rectangle(vis, (cx - w // 2, cy - h // 2),
                          (cx + w // 2, cy + h // 2), (0, 0, 255), 2)
            cv2.line(vis, (cx, 0), (cx, H), (0, 0, 255), 2)
            cv2.putText(vis, "mine " + r["src"] + " x=" + str(cx),
                        (6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2, cv2.LINE_AA)

        out = os.path.join(OUT_DIR, n.replace(".png", "_cmp.jpg"))
        cv2.imencode(".jpg", vis)[1].tofile(out)
        print(n + " -> " + os.path.relpath(out, REPO))
        if old:
            print("   1.0 : conf=" + str(old[0]) + " x=" + str(old[1]))
        if r:
            print("   mine: " + r["src"] + " x=" + str(int(r["cx"]))
                  + " y=" + str(int(r["cy"]))
                  + " w=" + str(int(r["w"])) + " h=" + str(int(r["h"])))

    print("")
    print("产物目录: " + os.path.relpath(OUT_DIR, REPO))


if __name__ == "__main__":
    main()
