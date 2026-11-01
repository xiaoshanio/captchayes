# -*- coding: utf-8 -*-
"""demo_2408: 上线后的冒烟测试 —— 用真实 2408 样本验证 runtime。

为什么不用 demo.py：
    demo.py 读 docs/res.jpg，那是**旧域**（中文文字点选 344x384）样本。
    换域重训后新模型不认识旧域，会抛
    `ValueError: need at least one array to concatenate`
    —— 这是预期行为（模型是域专用的），不是 bug。
    但它不能用来验证"上线是否成功"，所以单独写这个脚本。

用法：
    python tools/demo_2408.py
    python tools/demo_2408.py --n 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="out/demo2408")
    args = ap.parse_args()

    from src.captcha import TextSelectCaptcha

    root = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    dirs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d))
                  and os.path.exists(os.path.join(root, d, "meta.json")))
    if not dirs:
        raise SystemExit(f"没有样本: {root}")

    t0 = time.time()
    cap = TextSelectCaptcha()
    print(f"[load] 模型加载 {time.time()-t0:.2f}s")
    print(f"[load] det  = {cap.yolo.session._model_path}")
    print(f"[load] pre  = {cap.pre.session._model_path}")
    cap.yolo.conf_threshold = args.conf

    ok = 0
    for sid in dirs[:args.n]:
        img_p = os.path.join(root, sid, "main.jpg")
        if not os.path.exists(img_p):
            continue
        t = time.time()
        try:
            r = cap.run_dict(img_p)
            ms = (time.time() - t) * 1000
            ok += 1
            print(f"\n[{sid}] {ms:.0f} ms  imgW={r['imgW']} imgH={r['imgH']} "
                  f"点数={len(r['point'])}")
            for i, p in enumerate(r["point"]):
                print(f"    #{i+1}  相对坐标 ({p['x_rel']:.1f}, {p['y_rel']:.1f})")
            # 可视化
            img = cv2.imdecode(np.fromfile(img_p, dtype=np.uint8), cv2.IMREAD_COLOR)
            for i, c in enumerate(r["corp"]):
                cv2.rectangle(img, (c["x1"], c["y1"]), (c["x2"], c["y2"]),
                              (0, 0, 255), 2)
                cv2.putText(img, str(i + 1), (c["x1"] + 2, max(14, c["y1"] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                            cv2.LINE_AA)
            vp = os.path.join(out, sid + ".jpg")
            cv2.imencode(".jpg", img)[1].tofile(vp)
            print(f"    可视化 -> {os.path.relpath(vp, REPO)}")
        except Exception as e:
            print(f"\n[{sid}] FAIL {type(e).__name__}: {e}")

    print(f"\n=== 冒烟结果: {ok}/{min(args.n, len(dirs))} 通过 ===")


if __name__ == "__main__":
    main()
