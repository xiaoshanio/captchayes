# -*- coding: utf-8 -*-
"""relabel_by_det_2408: 用**已训练检测器**的输出重建标注（自举/蒸馏标注）。

为什么这样做（这是本轮的方法论修正）：

    分割器（阈值法）在「图标与背景同色」的子域上物理上做不到 ——
    实测 t2408_1810856471406067712：底图 5 个图标，
    漏掉的「收音机」「建筑」压在深色色块上，灰度差只有 10~20，
    Otsu 阈值 79.2 时图标与背景区间完全重叠，任何全局阈值都分不开。

    而**检测器**（YOLO，在 305 题上学过）已经能出 5.37 框/题（真值约 5），
    它学的是"图标长什么样"，不是"哪里比背景暗" —— 所以能处理同色情况。

    于是正确顺序是：
        检测器（自举）→ 用它重建 class0 标注 → 重训检测器 → 迭代
    这是标准的**自训练 / 知识蒸馏**闭环，也是"数据标注搞不定"时的通用解法。

    注意：必须用**独立于目标集**的检测器，否则会自我强化错误。
    这里用 runs/export_t2408b（在 305 题 v1 标注上训的），
    对 v2 的 1038 题做标注 —— v2 里有 733 题是新的，检测器没见过。

流程：
    1) 检测器出 class0（底图图标）+ class2（提示条图标）
    2) class2 用**精确切分**覆盖（提示图列投影，100% 准）
    3) 配对：用匹配器算 class2 × class0 分数矩阵 + 匈牙利指派
    4) 只保留「配对分 >= min_score」的题（宁缺毋滥）
    5) 输出成与 autolabel_2408 相同的目录结构

用法：
    python tools/relabel_by_det_2408.py --det runs/export_t2408b/best_v3_2408b.onnx \
        --pre runs/export_t2408b/pre_model_v7_2408b.onnx \
        --samples data/samples --out data/labeled_det
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from autolabel_2408 import split_prompt_icons, compose, yolo_line, imwrite_u  # noqa: E402


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1)


def nms_same_class(dets, thr):
    keep = []
    for d in sorted(dets, key=lambda z: -z[4]):
        if all(d[5] != k[5] or iou(d[:4], k[:4]) <= thr for k in keep):
            keep.append(d)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/export_t2408b/best_v3_2408b.onnx")
    ap.add_argument("--pre", default="runs/export_t2408b/pre_model_v7_2408b.onnx")
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--out", default="data/labeled_det")
    ap.add_argument("--conf", type=float, default=0.12,
                    help="检测阈值。自举标注宁可多召回（干扰项多几个无害），"
                         "所以比推理时的 0.25 更低")
    ap.add_argument("--dedup", type=float, default=0.5)
    ap.add_argument("--min-score", type=float, default=0.55,
                    help="配对分下限，低于此值的题整题丢弃（宁缺毋滥）")
    ap.add_argument("--min-pairs", type=float, default=1.0,
                    help="至少配对成功的提示图标比例（1.0=必须全配上）")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strip-h", type=int, default=56)
    args = ap.parse_args()

    from src.utils import yolo_onnx, matchingMode, ver_onnx

    root = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "review"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    y = yolo_onnx.YOLO(os.path.join(REPO, args.det), conf_threshold=args.conf)
    pre = ver_onnx.PreONNX(os.path.join(REPO, args.pre))

    dirs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d))
                  and os.path.exists(os.path.join(root, d, "meta.json")))
    print(f"[scan] {len(dirs)} 题")

    rng = random.Random(args.seed)
    rows, review = [], []
    dropped = {"no_prompt": 0, "no_det": 0, "low_score": 0, "incomplete": 0}
    t_ok = 0

    for i, sid in enumerate(dirs):
        d = os.path.join(root, sid)
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        bg = matchingMode.open_image(os.path.join(d, "bg.jpg"))
        prs = meta.get("prompts") or []
        if bg is None or not prs:
            dropped["no_prompt"] += 1
            continue
        pr = matchingMode.open_image(os.path.join(d, prs[0]["file"]))
        if pr is None:
            dropped["no_prompt"] += 1
            continue
        icons = split_prompt_icons(pr)
        if not icons:
            dropped["no_prompt"] += 1
            continue

        # 用与训练/推理一致的版式合成（关键：与 autolabel_2408.compose 相同）
        canvas, bg_h, off_x = compose(bg, pr, args.strip_h, "left")
        H, W = canvas.shape[:2]

        det = y.inference(canvas)
        if args.dedup > 0:
            det = nms_same_class(det, args.dedup)
        det_t = [x for x in det if x[5] == 0 and x[3] <= bg_h]
        if len(det_t) < len(icons):
            dropped["no_det"] += 1
            continue

        # class2：用精确切分覆盖（提示图列投影 100% 准）
        char_boxes = []
        for it in icons:
            x1, y1, x2, y2 = it["box"]
            char_boxes.append((off_x + x1, bg_h + y1, off_x + x2, bg_h + y2))
        char_boxes.sort(key=lambda b: b[0])

        # 配对：匹配器打分 + 匈牙利
        ics = [canvas[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in char_boxes]
        its = [canvas[int(x[1]):int(x[3]), int(x[0]):int(x[2])] for x in det_t]
        if not ics or not its:
            dropped["no_det"] += 1
            continue
        M = np.array([[pre.predict_pair(c, t) for t in its] for c in ics])
        try:
            from scipy.optimize import linear_sum_assignment
            rr, cc = linear_sum_assignment(-M)
            pairs = sorted([(int(a), int(b), float(M[a, b]))
                            for a, b in zip(rr, cc)], key=lambda z: z[0])
        except Exception:
            mat = M.astype(float).copy()
            pairs = []
            for _ in range(min(len(ics), len(its))):
                a, b = divmod(int(np.argmax(mat)), mat.shape[1])
                pairs.append((int(a), int(b), float(M[a, b])))
                mat[a, :] = -np.inf
                mat[:, b] = -np.inf
            pairs.sort(key=lambda z: z[0])

        n_ok = sum(1 for _a, _b, s in pairs if s >= args.min_score)
        if n_ok < len(icons) * args.min_pairs:
            dropped["low_score"] += 1
            continue

        labels = []
        for (x1, y1, x2, y2) in char_boxes:
            labels.append(yolo_line(2, (x1, y1, x2, y2), W, H))
        for x in det_t:
            labels.append(yolo_line(0, (x[0], x[1], x[2], x[3]), W, H))

        split = "val" if rng.random() < args.val_ratio else "train"
        img_p = os.path.join(out, "images", split, sid + ".jpg")
        lab_p = os.path.join(out, "labels", split, sid + ".txt")
        imwrite_u(img_p, canvas)
        with open(lab_p, "w", encoding="utf-8") as f:
            f.write("\n".join(labels) + "\n")

        rows.append({
            "sid": sid, "split": split,
            "n_prompt": len(icons), "n_bg_icons": len(det_t),
            "min_pair": round(min(s for _a, _b, s in pairs), 4),
            "pairs": [{"prompt": a, "bg": b, "score": round(s, 4)}
                      for a, b, s in pairs],
            "bg_boxes": [[int(v) for v in x[:4]] for x in det_t],
            "need_review": bool(n_ok < len(icons)),
            "image": os.path.relpath(img_p, out).replace("\\", "/"),
            "label": os.path.relpath(lab_p, out).replace("\\", "/"),
        })
        if n_ok < len(icons):
            review.append({"sid": sid, "min_pair": rows[-1]["min_pair"]})
        t_ok += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(dirs)}] 已处理 {t_ok}  丢弃 {sum(dropped.values())}")

    with open(os.path.join(out, "labels_index.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "review_queue.jsonl"), "w", encoding="utf-8") as f:
        for r in review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import yaml
    cfg = {"path": out.replace("\\", "/"), "train": "images/train",
           "val": "images/val", "names": {0: "label", 1: "title", 2: "char"}}
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    n_tr = sum(1 for r in rows if r["split"] == "train")
    rep = {"samples": root, "out": out, "det": args.det, "pre": args.pre,
           "conf": args.conf, "min_score": args.min_score,
           "n_total": len(dirs), "n_kept": len(rows),
           "train": n_tr, "val": len(rows) - n_tr,
           "dropped": dropped, "n_review": len(review)}
    with open(os.path.join(out, "relabel_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print("\n=== 自举标注完成 ===")
    print(f"  输入题数   : {len(dirs)}")
    print(f"  保留       : {len(rows)}  (train={n_tr}, val={len(rows)-n_tr})")
    print(f"  丢弃       : {dropped}")
    print(f"  复核队列   : {len(review)}")
    if rows:
        mp = [r["min_pair"] for r in rows]
        nb = [r["n_bg_icons"] for r in rows]
        print(f"  配对分     : mean={np.mean(mp):.3f} min={min(mp):.3f}")
        print(f"  底图图标数 : mean={np.mean(nb):.2f}")
    print(f"  输出       : {out}/")


if __name__ == "__main__":
    main()
