# -*- coding: utf-8 -*-
"""v1_v2_compare: 验证「数据版式修复」是否真的解决了端到端瓶颈。

背景（v1 的实测结论）：
    端到端卡在 61.7%，而匹配器（喂真值框）已达 97.5%。
    用 eval/e2e_diag.py 定位到主因是 **class2（提示条字符）框数不符 10~15%**，
    且加训到 19 epoch（mAP50 0.947 -> 0.987）端到端**完全没变** ——
    说明瓶颈是数据版式（提示条图标间距太挤导致粘连/重复），不是训练轮数。

本脚本做的事：
    在同一套评测集上，对比 v1 数据训出的模型 与 v2 数据训出的模型，
    逐项给出 class2 框数不符率 / class0 召回 / 端到端全对率，
    用来判定「修数据」这个动作到底值多少。

用法：
    python eval/v1_v2_compare.py --v1 runs/export_next --v2 runs/export_v2 --limit 60
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


def find_onnx(d, kinds=("best_v3", "best"), prefer=None):
    if not os.path.isdir(d):
        return None
    fs = [f for f in os.listdir(d) if f.endswith(".onnx")]
    if prefer:
        for f in fs:
            if prefer in f:
                return os.path.join(d, f)
    for f in sorted(fs):
        if any(k in f for k in kinds):
            return os.path.join(d, f)
    return os.path.join(d, fs[0]) if fs else None


def find_matcher(d):
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if f.endswith(".onnx") and "pre" in f:
            return os.path.join(d, f)
    return None


def eval_combo(tag, det_p, pre_p, cases, synth_dir, conf=0.2):
    from src.utils import yolo_onnx, ver_onnx
    det = yolo_onnx.YOLO(det_p)
    det.conf_threshold = conf
    pre = ver_onnx.PreONNX(pre_p)

    n = ok = n2_bad = n0_short = 0
    confs0, confs2 = [], []
    for c in cases:
        img = cv2.imdecode(np.fromfile(c["img"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gt = sorted(c["truth"]["pairs"], key=lambda q: q["click_order"])
        dets = det.inference(img)
        tb = [x[:4] for x in dets if int(x[5]) == 0]
        cb = [x[:4] for x in dets if int(x[5]) == 2]
        cb.sort(key=lambda b: b[0])
        confs0 += [float(x[4]) for x in dets if int(x[5]) == 0]
        confs2 += [float(x[4]) for x in dets if int(x[5]) == 2]
        n += 1
        if len(cb) != len(gt):
            n2_bad += 1
        if len(tb) < len(gt):
            n0_short += 1
        if not cb or not tb:
            continue
        chars = [img[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in cb]
        tg = [img[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in tb]
        if any(x.size == 0 for x in chars + tg):
            continue
        res = greedy(pre.reason_all_batch(chars, tg))
        out = [tb[j] for i, j in res]
        if len(out) == len(gt) and all(iou(out[k], gt[k]["target_box"]) > 0.5
                                       for k in range(len(gt))):
            ok += 1
    r = {"tag": tag, "n": n, "exact": ok,
         "exact_rate": round(ok / max(n, 1), 4),
         "class2_mismatch": round(n2_bad / max(n, 1), 4),
         "class0_short": round(n0_short / max(n, 1), 4),
         "conf0_med": round(float(np.median(confs0)), 3) if confs0 else None,
         "conf2_med": round(float(np.median(confs2)), 3) if confs2 else None}
    print(f"  {tag:34s} exact={ok:3d}/{n:<3d} ({r['exact_rate']*100:5.1f}%)  "
          f"class2不符={r['class2_mismatch']*100:4.0f}%  class0不足={r['class0_short']*100:3.0f}%  "
          f"conf0={r['conf0_med']} conf2={r['conf2_med']}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="data/selfgen_v2", help="评测集（用 v2 的 val）")
    ap.add_argument("--v1", default="runs/export_next")
    ap.add_argument("--v2", default="runs/export_v2")
    ap.add_argument("--baseline", default="model")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--out", default="out/v1v2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = args.synth
    pdir = os.path.join(d, "pairs", "val")
    bases = [f[:-5] for f in sorted(os.listdir(pdir))][:args.limit]
    cases = []
    for b in bases:
        ip = os.path.join(d, "images", "val", b + ".jpg")
        tp = os.path.join(pdir, b + ".json")
        if os.path.exists(ip) and os.path.exists(tp):
            cases.append({"id": b, "img": ip,
                          "truth": json.load(open(tp, encoding="utf-8"))})
    print(f"[cases] {len(cases)} 张（来自 {d}/pairs/val）  conf={args.conf}")
    print("=" * 108)

    combos = [
        ("baseline-det + baseline-pre",
         find_onnx(args.baseline, prefer="best_v3"), find_matcher(args.baseline)),
        ("v1-det + v1-pre",
         find_onnx(args.v1, prefer="best_v3"), find_matcher(args.v1)),
        ("v2-det + v2-pre",
         find_onnx(args.v2, prefer="best_v3"), find_matcher(args.v2)),
        ("v2-det + v1-pre",
         find_onnx(args.v2, prefer="best_v3"), find_matcher(args.v1)),
        ("v1-det + v2-pre",
         find_onnx(args.v1, prefer="best_v3"), find_matcher(args.v2)),
    ]
    rows = []
    for tag, dp, pp in combos:
        if not dp or not pp or not os.path.exists(dp) or not os.path.exists(pp):
            print(f"  {tag:34s} [skip] 缺模型  det={dp}  pre={pp}")
            continue
        try:
            rows.append(eval_combo(tag, dp, pp, cases, d, args.conf))
        except Exception as e:
            print(f"  {tag:34s} [FAIL] {type(e).__name__}: {e}")

    print("=" * 108)
    if len(rows) >= 3:
        b = rows[0]
        v1 = next((r for r in rows if r["tag"].startswith("v1-det")), None)
        v2 = next((r for r in rows if r["tag"].startswith("v2-det + v2")), None)
        if v1 and v2:
            print(f"  数据版式修复的效果：")
            print(f"    端到端全对率 : v1={v1['exact_rate']*100:.1f}%  ->  v2={v2['exact_rate']*100:.1f}%"
                  f"   ({(v2['exact_rate']-v1['exact_rate'])*100:+.1f} pt)")
            print(f"    class2 框数不符: v1={v1['class2_mismatch']*100:.0f}%  ->  v2={v2['class2_mismatch']*100:.0f}%"
                  f"   ({(v2['class2_mismatch']-v1['class2_mismatch'])*100:+.0f} pt)")
    json.dump(rows, open(os.path.join(args.out, "compare.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n报告 -> {args.out}/compare.json")


if __name__ == "__main__":
    main()
