# -*- coding: utf-8 -*-
"""drift_diag: 定位漂移域上端到端失败的具体环节。

背景：
    drift_recover.py 测出 旧模型@旧域 80% -> 旧模型@漂移域 6.7%（老化代价 -73pt），
    但 新模型@漂移域 只有 3.3% —— 这不合常理，说明有别的因素在拖后腿。

    分项诊断已确认：
      - 检测器在漂移域上很好（class0 P=0.97/R=0.99，class2 P=0.96/R=0.96）
      - 匹配器（喂真值框）在漂移域上 87.5% 全对
    所以问题一定出在「检测框 -> 裁剪 -> 指派」这条链上。

本脚本逐题打印检测框与真值框，找出具体失效点。

用法：
    python eval/drift_diag.py --det runs/drift_rec/export/detB.onnx \
        --mat runs/drift_rec/export/matB.onnx --data data/driftB --limit 6
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
    m = np.array(slys, dtype=float)
    res = []
    for _ in range(min(m.shape[0], m.shape[1])):
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        res.append((int(i), int(j)))
        m[i, :] = -1e9
        m[:, j] = -1e9
    res.sort(key=lambda t: t[0])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/drift_rec/export/detB.onnx")
    ap.add_argument("--mat", default="runs/drift_rec/export/matB.onnx")
    ap.add_argument("--data", default="data/driftB")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()

    from src.utils import yolo_onnx, ver_onnx
    det = yolo_onnx.YOLO(os.path.join(REPO, args.det))
    det.conf_threshold = args.conf
    mat = ver_onnx.PreONNX(os.path.join(REPO, args.mat))

    d = os.path.join(REPO, args.data)
    pdir = os.path.join(d, "pairs", "val")
    bases = [f[:-5] for f in sorted(os.listdir(pdir))][:args.limit]

    for b in bases:
        ip = os.path.join(d, "images", "val", b + ".jpg")
        img = cv2.imdecode(np.fromfile(ip, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        truth = json.load(open(os.path.join(pdir, b + ".json"), encoding="utf-8"))
        gt = sorted(truth["pairs"], key=lambda q: q["click_order"])
        strip_h = truth.get("strip_h", 56)
        strip_top = H - strip_h

        dets = det.inference(img)
        tb = [x[:4] for x in dets if int(x[5]) == 0]
        cb_items = [(x[:4], float(x[4])) for x in dets if int(x[5]) == 2]
        cb_items.sort(key=lambda z: z[0][0])
        cb = [z[0] for z in cb_items]

        print(f"\n{b}: img={W}x{H} strip_h={strip_h} strip_top={strip_top}")
        print(f"  真值: prompt={len(gt)}  |  检出: cls0={len(tb)} cls2={len(cb)}")

        # 1) class2 是否落在提示条区域
        bad_region = [i for i, z in enumerate(cb) if z[1] < strip_top - 8]
        print(f"  class2 落在提示条外的: {bad_region if bad_region else '无'}")

        # 2) class2 与真值 prompt_box 的 IoU 匹配
        for i, (z, cf) in enumerate(cb_items):
            best = max((iou(z, q["prompt_box"]) for q in gt), default=0.0)
            flag = "OK " if best > 0.5 else "MISS"
            print(f"   D2[{i}] x=[{z[0]:5.0f},{z[2]:5.0f}] y=[{z[1]:5.0f},{z[3]:5.0f}] "
                  f"conf={cf:.2f} bestIoU={best:.2f} {flag}")

        # 3) 每个真值 prompt 是否被覆盖
        for i, q in enumerate(gt):
            pb = q["prompt_box"]
            cov = any(iou(z, pb) > 0.5 for z in cb)
            print(f"   GT2[{i}] x=[{pb[0]:5.0f},{pb[2]:5.0f}] y=[{pb[1]:5.0f},{pb[3]:5.0f}] "
                  f"{'covered' if cov else 'MISSING'}")

        # 4) 端到端
        if cb and tb:
            chars = [img[int(z[1]):int(z[3]), int(z[0]):int(z[2])] for z in cb]
            tg = [img[int(z[1]):int(z[3]), int(z[0]):int(z[2])] for z in tb]
            if all(x.size > 0 for x in chars + tg):
                res = greedy(mat.reason_all_batch(chars, tg))
                out = [tb[j] for i, j in res]
                hits = sum(1 for k in range(min(len(out), len(gt)))
                           if iou(out[k], gt[k]["target_box"]) > 0.5)
                print(f"  端到端: 输出{len(out)}个点, 按序命中 {hits}/{len(gt)}")
                # 若命中数 == len(gt) 但长度不等，说明多输出/少输出
                if len(out) != len(gt):
                    print(f"    !! 长度不等: out={len(out)} gt={len(gt)}")


if __name__ == "__main__":
    main()
