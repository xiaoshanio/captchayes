# -*- coding: utf-8 -*-
"""class2_diag: 专查提示条字符（class 2）的检测质量。

端到端诊断显示 class2 检出数不符率高达 75~95%，这是顺序错的直接原因。
本脚本逐图打印：真值框数 / 检出框数 / 每个检出框的位置与置信度，
并可视化，用来判断是「漏检」「多检」还是「框位置错」。

用法：
    python eval/class2_diag.py --det runs/export_next/best_v3_next.onnx --limit 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="data/selfgen")
    ap.add_argument("--det", default="runs/export_next/best_v3_next.onnx")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", default="out/class2diag")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from src.utils import yolo_onnx
    det = yolo_onnx.YOLO(args.det)
    det.conf_threshold = args.conf

    d = args.synth
    bases = [f[:-5] for f in sorted(os.listdir(os.path.join(d, "pairs", "val")))][:args.limit]
    print(f"检测器: {args.det}  conf={args.conf}")
    print("=" * 100)
    for base in bases:
        img = cv2.imdecode(np.fromfile(os.path.join(d, "images", "val", base + ".jpg"),
                                       dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        truth = json.load(open(os.path.join(d, "pairs", "val", base + ".json"),
                              encoding="utf-8"))
        gt = sorted(truth["pairs"], key=lambda q: q["click_order"])
        dets = det.inference(img)
        d0 = [x for x in dets if int(x[5]) == 0]
        d2 = [x for x in dets if int(x[5]) == 2]

        # 真值 class2 框（提示条，在底部 strip 区）
        gt2 = [q["prompt_box"] for q in gt]
        strip_top = H - truth.get("strip_h", 56)

        print(f"\n{base}: 真值 提示={len(gt)} 底图目标={len(gt)} | "
              f"检出 class2={len(d2)} class0={len(d0)}")
        print(f"   strip_top={strip_top} (图高 {H})")
        for i, b in enumerate(sorted(gt2, key=lambda x: x[0])):
            print(f"   GT2[{i}] x=[{b[0]:6.1f},{b[2]:6.1f}] y=[{b[1]:6.1f},{b[3]:6.1f}]")
        for i, x in enumerate(sorted(d2, key=lambda t: t[0])):
            inside = "strip" if x[1] > strip_top else "BG!"
            print(f"   D2 [{i}] x=[{x[0]:6.1f},{x[2]:6.1f}] y=[{x[1]:6.1f},{x[3]:6.1f}] "
                  f"conf={x[4]:.3f} {inside}")
        for i, x in enumerate(sorted(d0, key=lambda t: t[0])):
            inside = "BG" if x[1] < strip_top else "STRIP!"
            print(f"   D0 [{i}] x=[{x[0]:6.1f},{x[2]:6.1f}] y=[{x[1]:6.1f},{x[3]:6.1f}] "
                  f"conf={x[4]:.3f} {inside}")

        vis = img.copy()
        for b in gt2:
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (0, 165, 255), 1)
        for q in gt:
            b = q["target_box"]
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (0, 255, 0), 1)
        for x in d2:
            cv2.rectangle(vis, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])),
                          (255, 0, 0), 2)
        for x in d0:
            cv2.rectangle(vis, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])),
                          (0, 0, 255), 2)
        cv2.imwrite(os.path.join(args.out, base + ".jpg"), vis)

    print(f"\n可视化 -> {args.out}/")
    print("图例: 橙细框=GT提示条 绿细框=GT底图目标 蓝粗框=检出class2 红粗框=检出class0")


if __name__ == "__main__":
    main()
