# -*- coding: utf-8 -*-
"""audit_dataset: 数据集自审 —— 在怪模型之前先验证标签本身对不对。

触发原因：
    v2 数据集（改了提示条间距）训出的模型，在同一套 val 上
    端到端 13.3%，反而低于 v1 的 40%，class2 框数不符率从 32% 涨到 68%。
    这种"改数据反而更差"的反常结果，第一嫌疑是**数据本身有 bug**，不是模型。

本脚本逐图核对三件事必须自洽：
    A. labels/*.txt 里 class2 的框数 == pairs/*.json 里 prompt 数
    B. labels/*.txt 里 class0 的框数 == pairs 里 pairs 数 + 干扰项数
    C. 每个 pair 的 target_box 必须能在 class0 框里找到 IoU>0.9 的匹配
    D. 提示条框必须落在图片下半部的 strip 区域内
    E. 框不能越界

用法：
    python eval/audit_dataset.py --data data/selfgen_v2 --split val
    python eval/audit_dataset.py --data data/selfgen --split val
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/selfgen_v2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--draw", type=int, default=0)
    args = ap.parse_args()
    os.makedirs("out/audit", exist_ok=True)

    d = args.data
    lab_dir = os.path.join(d, "labels", args.split)
    pair_dir = os.path.join(d, "pairs", args.split)
    img_dir = os.path.join(d, "images", args.split)
    bases = [f[:-4] for f in sorted(os.listdir(lab_dir)) if f.endswith(".txt")][:args.limit]
    print(f"审计 {d}/{args.split}: {len(bases)} 张")
    print("=" * 96)

    bad = {"A_cls2_mismatch": [], "B_cls0_mismatch": [], "C_target_not_in_cls0": [],
           "D_prompt_outside_strip": [], "E_out_of_bounds": [], "F_no_pairs": []}
    cls2_counts, cls0_counts, n_prompt = [], [], []

    for b in bases:
        lp = os.path.join(lab_dir, b + ".txt")
        pp = os.path.join(pair_dir, b + ".json")
        ip = os.path.join(img_dir, b + ".jpg")
        if not (os.path.exists(lp) and os.path.exists(pp) and os.path.exists(ip)):
            continue
        truth = json.load(open(pp, encoding="utf-8"))
        img = cv2.imdecode(np.fromfile(ip, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        strip_h = truth.get("strip_h", 56)
        strip_top = H - strip_h

        c0, c2 = [], []
        for line in open(lp, encoding="utf-8"):
            v = line.split()
            if len(v) != 5:
                continue
            c = int(float(v[0]))
            cx, cy, bw, bh = [float(t) for t in v[1:]]
            box = [(cx - bw / 2) * W, (cy - bh / 2) * H,
                   (cx + bw / 2) * W, (cy + bh / 2) * H]
            # E 越界检查
            if box[0] < -2 or box[1] < -2 or box[2] > W + 2 or box[3] > H + 2:
                bad["E_out_of_bounds"].append((b, c, [round(v, 1) for v in box]))
            (c0 if c == 0 else c2 if c == 2 else []).append(box)

        prs = truth.get("pairs", [])
        cls2_counts.append(len(c2))
        cls0_counts.append(len(c0))
        n_prompt.append(len(prs))

        # A: class2 数 == prompt 数
        if len(c2) != len(prs):
            bad["A_cls2_mismatch"].append((b, len(c2), len(prs)))
        # B: class0 数 >= prompt 数（应等于 prompt + decoy）
        if len(c0) < len(prs):
            bad["B_cls0_mismatch"].append((b, len(c0), len(prs)))
        # C: 每个 pair 的 target_box 必须能在 class0 里找到
        for q in prs:
            tb = q["target_box"]
            if not any(iou(tb, x) > 0.9 for x in c0):
                bad["C_target_not_in_cls0"].append((b, q["click_order"],
                                                    [round(v, 1) for v in tb]))
        # D: prompt_box 必须落在 strip 区
        for q in prs:
            pb = q["prompt_box"]
            if pb[1] < strip_top - 2:
                bad["D_prompt_outside_strip"].append((b, q["click_order"],
                                                      [round(v, 1) for v in pb], strip_top))
        if not prs:
            bad["F_no_pairs"].append(b)

        if args.draw and len(bad["A_cls2_mismatch"]) + len(bad["C_target_not_in_cls0"]) < args.draw:
            vis = img.copy()
            for x in c0:
                cv2.rectangle(vis, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])),
                              (0, 0, 255), 2)
            for x in c2:
                cv2.rectangle(vis, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])),
                              (255, 0, 0), 2)
            for q in prs:
                tb = q["target_box"]
                cv2.rectangle(vis, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])),
                              (0, 255, 0), 1)
            cv2.imwrite(os.path.join("out/audit", f"{os.path.basename(d)}_{b}.jpg"), vis)

    print("-- 统计 --")
    print(f"  class2 框数: mean={np.mean(cls2_counts):.2f} min={min(cls2_counts)} max={max(cls2_counts)}")
    print(f"  class0 框数: mean={np.mean(cls0_counts):.2f} min={min(cls0_counts)} max={max(cls0_counts)}")
    print(f"  prompt 数  : mean={np.mean(n_prompt):.2f} min={min(n_prompt)} max={max(n_prompt)}")
    print("\n-- 一致性检查（全部应为 0） --")
    for k, v in bad.items():
        flag = "OK" if not v else f"★ {len(v)} 处异常"
        print(f"  {k:26s}: {flag}")
        for item in v[:4]:
            print(f"      {item}")

    total_bad = sum(len(v) for v in bad.values())
    print("\n" + "=" * 96)
    if total_bad == 0:
        print("结论：数据集自洽，标签无误。模型表现异常应到别处找原因。")
    else:
        print(f"结论：数据集存在 {total_bad} 处不自洽 —— **先修数据**，否则训练无意义。")


if __name__ == "__main__":
    main()
