# -*- coding: utf-8 -*-
"""e2e_ablate: 端到端消融 —— 对比不同匹配器在真实推理路径下的表现。

用于验证 docs/DRIFT_EXPERIMENT.md 6.8 提出的修复方向是否真的有效：
    - matB          : 用真值框训练的匹配器（基线）
    - matB_detpw25  : 用检测器输出框对齐 + pos_weight 训练的匹配器

用法：
    python eval/e2e_ablate.py --data data/driftB --det runs/drift_rec/export/detB.onnx \
        --mats "gt=runs/drift_rec/export/matB.onnx" "det=runs/drift_rec/export/matB_detpw25.onnx"
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


def greedy(s):
    m = np.array(s, dtype=float)
    res = []
    for _ in range(min(m.shape[0], m.shape[1])):
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        res.append((int(i), int(j)))
        m[i, :] = -1e9
        m[:, j] = -1e9
    res.sort(key=lambda t: t[0])
    return res


def dd(items, thr=0.45):
    out = []
    for b, c in sorted(items, key=lambda t: -t[1]):
        if all(iou(b, o[0]) <= thr for o in out):
            out.append((b, c))
    return out


def run(det, mat, d, bases, use_dd, conf):
    ok = n = err = n2 = 0
    pdir = os.path.join(d, "pairs", "val")
    for b in bases:
        ip = os.path.join(d, "images", "val", b + ".jpg")
        img = cv2.imdecode(np.fromfile(ip, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        truth = json.load(open(os.path.join(pdir, b + ".json"), encoding="utf-8"))
        gt = sorted(truth["pairs"], key=lambda q: q["click_order"])
        n += 1
        try:
            det.conf_threshold = conf
            ds = det.inference(img)
            r0 = [(x[:4], float(x[4])) for x in ds if int(x[5]) == 0]
            r2 = [(x[:4], float(x[4])) for x in ds if int(x[5]) == 2]
            if use_dd:
                r0, r2 = dd(r0), dd(r2)
            tb = [z[0] for z in r0]
            cb = sorted([z[0] for z in r2], key=lambda z: z[0])
            if len(cb) != len(gt):
                n2 += 1
            if not cb or not tb:
                raise ValueError("nobox")
            ch = [img[int(z[1]):int(z[3]), int(z[0]):int(z[2])] for z in cb]
            tg = [img[int(z[1]):int(z[3]), int(z[0]):int(z[2])] for z in tb]
            if any(x.size == 0 for x in ch + tg):
                raise ValueError("crop")
            res = greedy(mat.reason_all_batch(ch, tg))
            out = [tb[j] for i, j in res]
            if len(out) == len(gt) and all(iou(out[k], gt[k]["target_box"]) > 0.5
                                           for k in range(len(gt))):
                ok += 1
        except Exception:
            err += 1
    return {"n": n, "exact": ok, "exact_rate": round(ok / max(n, 1), 4),
            "err_rate": round(err / max(n, 1), 4),
            "class2_mismatch": round(n2 / max(n, 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/driftB")
    ap.add_argument("--det", default="runs/drift_rec/export/detB.onnx")
    ap.add_argument("--mats", nargs="+", required=True, help="name=path")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--out", default="out/e2e_ablate")
    args = ap.parse_args()
    os.makedirs(os.path.join(REPO, args.out), exist_ok=True)

    from src.utils import yolo_onnx, ver_onnx
    det = yolo_onnx.YOLO(os.path.join(REPO, args.det))
    d = os.path.join(REPO, args.data)
    pdir = os.path.join(d, "pairs", "val")
    bases = [f[:-5] for f in sorted(os.listdir(pdir))][:args.limit]

    print("=" * 92)
    print(f"端到端消融  数据={args.data}  det={os.path.basename(args.det)}  "
          f"limit={args.limit}  conf={args.conf}")
    print("=" * 92)
    rows = []
    for spec in args.mats:
        name, path = spec.split("=", 1)
        p = os.path.join(REPO, path)
        if not os.path.exists(p):
            print(f"  [skip] {name}: 缺 {path}")
            continue
        mat = ver_onnx.PreONNX(p)
        for use_dd in (False, True):
            r = run(det, mat, d, bases, use_dd, args.conf)
            r.update({"matcher": name, "dedup": use_dd})
            rows.append(r)
            tag = "去重" if use_dd else "原始"
            print(f"  {name:14s} {tag}  端到端={r['exact']:2d}/{r['n']:<2d} "
                  f"({r['exact_rate']*100:5.1f}%)  err={r['err_rate']*100:4.1f}%  "
                  f"class2不符={r['class2_mismatch']*100:3.0f}%")

    print("\n" + "=" * 92)
    print("对比（都取「去重后」，因为去重是已证明有效的后处理）")
    print("=" * 92)
    dd_rows = [r for r in rows if r["dedup"]]
    for r in dd_rows:
        print(f"  {r['matcher']:14s} {r['exact_rate']*100:5.1f}%")
    if len(dd_rows) >= 2:
        base, best = dd_rows[0], max(dd_rows, key=lambda r: r["exact_rate"])
        gain = (best["exact_rate"] - base["exact_rate"]) * 100
        print(f"\n  最佳: {best['matcher']}  相对 {base['matcher']} 变化 {gain:+.1f} pt")
        if gain > 10:
            print("  ✓ 检测框对齐训练有效")
        elif gain > 0:
            print("  · 有正向收益但幅度有限")
        else:
            print("  · 未见收益 —— 需重新审视该修复方向")

    json.dump(rows, open(os.path.join(REPO, args.out, "ablate.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n报告 -> {os.path.join(args.out, 'ablate.json')}")


if __name__ == "__main__":
    main()
