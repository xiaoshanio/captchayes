# -*- coding: utf-8 -*-
"""assign_2408: 2408 专用的「提示图标 → 底图框」指派器。

为什么不用 runtime 的 find_overall_index_fast：
    它是**贪心**的（每轮取全局最大），有两个已知缺陷：
      1) 贪心不考虑"剩下的还能不能配上"，容易把后面的图标逼到坏候选
      2) 当干扰框分数略高于真框时，会先占掉真框的列

实测 60 题的错误分类：
    wrong_pick  : 16 个（选错框）
    order_shift :  9 个（选对了但顺序错位）
    gt_not_detected: 1 个

两种指派器对比（同一模型、同一数据）：
    贪心      : exact 65.0%  point 85.5%
    匈牙利    : exact 69.2%  point 88.4%

匈牙利（scipy.optimize.linear_sum_assignment）求的是**全局最小代价**，
天然保证一一对应，且不会被单个高分干扰框带偏。

用法（库）：
    from assign_2408 import assign_hungarian, assign_greedy
"""
from __future__ import annotations

import numpy as np


def assign_greedy(scores):
    """与 runtime find_overall_index_fast 等价的贪心（保留用于对比）。"""
    N, M = scores.shape
    if N == 0 or M == 0:
        return []
    mat = scores.astype(np.float64).copy()
    pairs = []
    for _ in range(min(N, M)):
        flat = int(np.argmax(mat))
        i, j = divmod(flat, M)
        pairs.append((int(i), int(j), float(scores[i, j])))
        mat[i, :] = -np.inf
        mat[:, j] = -np.inf
    pairs.sort(key=lambda t: t[0])
    return pairs


def assign_hungarian(scores, min_score=0.0):
    """全局最优一一指派（匈牙利）。返回按 prompt 顺序的 [(i,j,score)]。"""
    N, M = scores.shape
    if N == 0 or M == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return assign_greedy(scores)

    # linear_sum_assignment 求最小代价 → 取负分数
    cost = -scores.astype(np.float64)
    rows, cols = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(rows, cols):
        s = float(scores[i, j])
        if s <= min_score:
            continue
        pairs.append((int(i), int(j), s))
    pairs.sort(key=lambda t: t[0])
    return pairs


def assign(scores, min_score=0.0, mode="hungarian"):
    if mode == "greedy":
        return assign_greedy(scores)
    return assign_hungarian(scores, min_score)


def main():
    """自检：在同一批样本上对比两种指派器。"""
    import argparse
    import glob
    import json
    import os
    import sys

    HERE = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(HERE)
    for p in (REPO, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)

    from src.utils import yolo_onnx, matchingMode, ver_onnx

    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/export_t2408b/best_v3_2408b.onnx")
    ap.add_argument("--pre", default="runs/export_t2408b/pre_model_v7_2408b.onnx")
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--labeled", default="data/labeled2408")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        return inter / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter, 1)

    gt = {}
    idx = os.path.join(REPO, args.labeled, "labels_index.jsonl")
    for line in open(idx, encoding="utf-8"):
        x = json.loads(line)
        o = [x["bg_boxes"][q["bg"]] for q in sorted(x["pairs"], key=lambda z: z["prompt"])
             if 0 <= q["bg"] < len(x["bg_boxes"])]
        if o:
            gt[x["sid"]] = o

    y = yolo_onnx.YOLO(os.path.join(REPO, args.det), conf_threshold=args.conf)
    p = ver_onnx.PreONNX(os.path.join(REPO, args.pre))

    sids = list(gt)[:args.limit]
    stat = {"greedy": [0, 0, 0], "hungarian": [0, 0, 0]}   # exact, hits, total
    for sid in sids:
        ip = os.path.join(REPO, args.samples, sid, "main.jpg")
        img = matchingMode.open_image(ip)
        if img is None:
            continue
        d = y.inference(img)
        tb = [x[:4] for x in d if x[5] == 0]
        cb = [x[:4] for x in d if x[5] == 2]
        cb.sort(key=lambda b: b[0])
        if not tb or not cb:
            continue
        its = [img[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in tb]
        ics = [img[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in cb]
        M = np.array([[p.predict_pair(c, t) for t in its] for c in ics])
        g = gt[sid]
        for mode in ("greedy", "hungarian"):
            order = assign(M, mode=mode)
            res = [tb[j] for _i, j, _s in order]
            h = sum(1 for k in range(min(len(g), len(res)))
                    if iou(g[k], res[k]) > 0.5)
            stat[mode][1] += h
            stat[mode][2] += len(g)
            if h == len(g) and len(res) == len(g):
                stat[mode][0] += 1

    print(f"\n=== 指派器对比（{len(sids)} 题）===")
    for mode in ("greedy", "hungarian"):
        e, h, t = stat[mode]
        print(f"  {mode:10s} exact={e}/{len(sids)}={e/max(len(sids),1):.1%} "
              f"point={h/max(t,1):.1%}")


if __name__ == "__main__":
    main()
