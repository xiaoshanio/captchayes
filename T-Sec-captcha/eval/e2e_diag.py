# -*- coding: utf-8 -*-
"""e2e_diag: 端到端失败点定位 —— 检测错 还是 匹配错？

背景：
    用真值框喂匹配器时，新匹配器 97.5% 全对（baseline 只有 20%）。
    但端到端 exact_rate 只有 2.5%。说明瓶颈在**检测环节**。

本脚本拆开统计每一环：
    A. class2（提示条字符）检出数 vs 真值数   -> 多检/少检都会破坏顺序
    B. class0（图面候选）检出数 vs 真值数
    C. 用「检测到的框」喂匹配器，端到端顺序正确率
    D. 用「真值框」喂匹配器，端到端顺序正确率（上界）
    E. 不同 conf 阈值下的端到端正确率（找最优工作点）

用法：
    python eval/e2e_diag.py --new-model runs/export_next --limit 40
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


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def greedy(slys):
    mat = np.array(slys, dtype=float)
    res = []
    for _ in range(min(mat.shape[0], mat.shape[1])):
        i, j = np.unravel_index(int(np.argmax(mat)), mat.shape)
        res.append((int(i), int(j)))
        mat[i, :] = -1e9
        mat[:, j] = -1e9
    res.sort(key=lambda t: t[0])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="data/selfgen")
    ap.add_argument("--new-model", default="runs/export_next")
    ap.add_argument("--baseline", default="model")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="out/e2ediag")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from src.utils import yolo_onnx, ver_onnx
    d = args.synth
    bases = [f[:-5] for f in sorted(os.listdir(os.path.join(d, "pairs", "val")))][:args.limit]

    combos = [
        ("baseline-det + baseline-pre",
         os.path.join(args.baseline, "best_v3.onnx"),
         os.path.join(args.baseline, "pre_model_v7.onnx")),
        ("new-det + new-pre",
         os.path.join(args.new_model, "best_v3_next.onnx"),
         os.path.join(args.new_model, "pre_model_v7_next.onnx")),
        ("new-det + baseline-pre",
         os.path.join(args.new_model, "best_v3_next.onnx"),
         os.path.join(args.baseline, "pre_model_v7.onnx")),
        ("baseline-det + new-pre",
         os.path.join(args.baseline, "best_v3.onnx"),
         os.path.join(args.new_model, "pre_model_v7_next.onnx")),
    ]

    print("=" * 96)
    print("端到端失败点定位")
    print("=" * 96)
    for tag, det_p, pre_p in combos:
        if not (os.path.exists(det_p) and os.path.exists(pre_p)):
            print(f"[skip] {tag}")
            continue
        det = yolo_onnx.YOLO(det_p)
        pre = ver_onnx.PreONNX(pre_p)
        print(f"\n--- {tag} ---")
        for conf in (0.2, 0.3, 0.4, 0.5):
            det.conf_threshold = conf
            n = ok = over2 = under0 = 0
            for base in bases:
                img = cv2.imdecode(np.fromfile(
                    os.path.join(d, "images", "val", base + ".jpg"), dtype=np.uint8),
                    cv2.IMREAD_COLOR)
                if img is None:
                    continue
                truth = json.load(open(os.path.join(d, "pairs", "val", base + ".json"),
                                       encoding="utf-8"))
                gt = sorted(truth["pairs"], key=lambda q: q["click_order"])
                dets = det.inference(img)
                tb = [x[:4] for x in dets if int(x[5]) == 0]
                cb = [x[:4] for x in dets if int(x[5]) == 2]
                cb.sort(key=lambda b: b[0])
                n += 1
                if len(cb) != len(gt):
                    over2 += 1
                if len(tb) < len(gt):
                    under0 += 1
                if not cb or not tb:
                    continue
                chars = [img[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in cb]
                tg = [img[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in tb]
                if any(c.size == 0 for c in chars + tg):
                    continue
                res = greedy(pre.reason_all_batch(chars, tg))
                out = [tb[j] for i, j in res]
                if len(out) == len(gt) and all(
                        iou(out[k], gt[k]["target_box"]) > 0.5 for k in range(len(gt))):
                    ok += 1
            print(f"   conf={conf:.2f}  端到端全对={ok}/{n} ({ok/max(n,1)*100:5.1f}%)   "
                  f"class2数不符={over2/max(n,1)*100:4.0f}%   class0不足={under0/max(n,1)*100:4.0f}%")

    # 真值框上界
    print("\n--- 真值框 + 各匹配器（上界） ---")
    for tag, pre_p in (("baseline-pre", os.path.join(args.baseline, "pre_model_v7.onnx")),
                       ("new-pre", os.path.join(args.new_model, "pre_model_v7_next.onnx"))):
        if not os.path.exists(pre_p):
            continue
        pre = ver_onnx.PreONNX(pre_p)
        ok = tot = 0
        for base in bases:
            img = cv2.imdecode(np.fromfile(
                os.path.join(d, "images", "val", base + ".jpg"), dtype=np.uint8),
                cv2.IMREAD_COLOR)
            if img is None:
                continue
            truth = json.load(open(os.path.join(d, "pairs", "val", base + ".json"),
                                   encoding="utf-8"))
            prs = sorted(truth["pairs"], key=lambda q: q["click_order"])
            if len(prs) < 2:
                continue
            pc = [img[int(q["prompt_box"][1]):int(q["prompt_box"][3]),
                      int(q["prompt_box"][0]):int(q["prompt_box"][2])] for q in prs]
            tc = [img[int(q["target_box"][1]):int(q["target_box"][3]),
                      int(q["target_box"][0]):int(q["target_box"][2])] for q in prs]
            if any(c.size == 0 for c in pc + tc):
                continue
            res = greedy(pre.reason_all_batch(pc, tc))
            tot += 1
            if all(j == i for i, j in res):
                ok += 1
        print(f"   {tag:14s} 全对={ok}/{tot} ({ok/max(tot,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
