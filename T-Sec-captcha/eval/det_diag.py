# -*- coding: utf-8 -*-
"""det_diag: 检测器专项诊断 —— 定位"新模型检出框变少"的原因。

对比 baseline / new 两个检测器在自造验证集上的：
  - 每张图 class0 / class2 检出数 vs 真值数
  - 置信度分布
  - 不同 conf 阈值下的召回
  - 漏检框的尺寸分布（是不是小目标漏了）

用法：
    python eval/det_diag.py --new-model runs/export_next --limit 60
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def load_det(onnx_path):
    from src.utils import yolo_onnx
    return yolo_onnx.YOLO(onnx_path)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def gt_boxes(synth_dir, base, W, H):
    """从 labels 读真值（归一化 -> 像素）。"""
    p = os.path.join(synth_dir, "labels", "val", base + ".txt")
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        v = line.split()
        if len(v) != 5:
            continue
        c = int(float(v[0]))
        cx, cy, bw, bh = [float(t) for t in v[1:]]
        out.append((c, [(cx - bw / 2) * W, (cy - bh / 2) * H,
                        (cx + bw / 2) * W, (cy + bh / 2) * H]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="data/selfgen")
    ap.add_argument("--new-model", default="runs/export_next")
    ap.add_argument("--baseline", default="model")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="out/detdiag")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = os.path.join(args.synth, "pairs", "val")
    bases = [f[:-5] for f in sorted(os.listdir(d))][:args.limit]
    print(f"[cases] {len(bases)} 张")

    models = {
        "baseline": os.path.join(args.baseline, "best_v3.onnx"),
        "new": os.path.join(args.new_model, "best_v3_next.onnx"),
    }
    report = {}
    for tag, path in models.items():
        if not os.path.exists(path):
            print(f"[skip] {tag}: 缺 {path}")
            continue
        det = load_det(path)
        det.conf_threshold = 0.05          # 低阈值，看原始输出
        n0 = n2 = g0 = g2 = 0
        confs0, confs2 = [], []
        missed = []
        for base in bases:
            img_p = os.path.join(args.synth, "images", "val", base + ".jpg")
            img = cv2.imdecode(np.fromfile(img_p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]
            dets = det.inference(img)
            d0 = [x for x in dets if int(x[5]) == 0]
            d2 = [x for x in dets if int(x[5]) == 2]
            n0 += len(d0)
            n2 += len(d2)
            confs0 += [float(x[4]) for x in d0]
            confs2 += [float(x[4]) for x in d2]
            gt = gt_boxes(args.synth, base, W, H)
            g0 += sum(1 for c, b in gt if c == 0)
            g2 += sum(1 for c, b in gt if c == 2)
            # 漏检统计（class 0）
            for c, b in gt:
                if c != 0:
                    continue
                if not any(iou(x[:4], b) > 0.5 for x in d0):
                    missed.append((b[2] - b[0], b[3] - b[1]))
        rep = {"n_images": len(bases),
               "det_class0": n0, "gt_class0": g0,
               "det_class2": n2, "gt_class2": g2,
               "recall0": round(n0 / max(g0, 1), 3),
               "recall2": round(n2 / max(g2, 1), 3),
               "conf0_med": round(st.median(confs0), 3) if confs0 else None,
               "conf0_p10": round(float(np.percentile(confs0, 10)), 3) if confs0 else None,
               "conf2_med": round(st.median(confs2), 3) if confs2 else None,
               "missed0": len(missed),
               "missed_wh_med": [round(st.median([m[0] for m in missed]), 1),
                                 round(st.median([m[1] for m in missed]), 1)] if missed else None,
               "extra0": n0 - g0}
        report[tag] = rep
        print(f"\n=== {tag} ({os.path.basename(path)}) ===")
        for k, v in rep.items():
            print(f"  {k:16s}: {v}")

    print("\n" + "=" * 70)
    print("诊断结论")
    print("=" * 70)
    b, n = report.get("baseline"), report.get("new")
    if b and n:
        print(f"  class0 召回: baseline={b['recall0']}  new={n['recall0']}")
        print(f"  class2 召回: baseline={b['recall2']}  new={n['recall2']}")
        print(f"  平均置信度 : baseline={b['conf0_med']}  new={n['conf0_med']}")
        print(f"  漏检数     : baseline={b['missed0']}  new={n['missed0']}")
        if n["recall0"] < 0.9:
            print("\n  >> 新模型 class0 召回不足 -> 训练不充分，需更多 epoch 或降 conf 阈值")
        if n["conf0_med"] and b["conf0_med"] and n["conf0_med"] < b["conf0_med"] - 0.15:
            print("  >> 新模型置信度显著偏低 -> 仍是欠训练")
    json.dump(report, open(os.path.join(args.out, "detdiag.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n报告: {args.out}/detdiag.json")


if __name__ == "__main__":
    main()
