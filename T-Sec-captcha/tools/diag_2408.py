# -*- coding: utf-8 -*-
"""diag_2408: 把端到端 68% 拆解到每个环节，量化各自的**上界**。

为什么必须先做这个：
    上一轮只知道 exact=68.0% / point_hit=86.4%，但不知道损失分布在哪。
    "再训一轮"是盲目优化；必须先回答三个问题：

        Q1 检测器给的是不是好框？        -> 框 IoU 分布、召回、重复框
        Q2 匹配器在**真值框**上有多强？  -> 匹配能力上界（与检测无关）
        Q3 匹配器在**检测框**上掉多少？  -> 训练-推理的框来源差异代价

    这三者的答案直接决定下一步该训哪个模型、怎么训。

四组对照（与 eval/crop_source_test.py 同思路，但针对 2408 + 新模型）：

    A  真值提示框 + 真值目标框   -> 纯匹配上界
    B  真值提示框 + 检测目标框   -> 只有目标侧用检测框
    C  检测提示框 + 真值目标框   -> 只有提示侧用检测框
    D  检测提示框 + 检测目标框   -> 真实推理路径（= 端到端）

    若 A >> D，说明**框来源**是主要损失，修法是「用检测框裁剪训练匹配器」，
    而不是继续加强检测器。

用法：
    python tools/diag_2408.py --det runs/export_t2408b/best_v3_2408b.onnx \
        --pre runs/export_t2408b/pre_model_v7_2408b.onnx --limit 80
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


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


def crop(img, box, pad=0, norm_short=None):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    c = img[y1:y2, x1:x2]
    if c.size == 0:
        return None
    if norm_short:
        h, w = c.shape[:2]
        s = norm_short / float(max(min(h, w), 1))
        if abs(s - 1.0) > 0.02:
            c = cv2.resize(c, (max(1, int(round(w*s))), max(1, int(round(h*s)))),
                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return c


def score_matrix(pre, ics, its):
    return np.array([[pre.predict_pair(c, t) for t in its] for c in ics])


def greedy(m):
    mat = m.astype(float).copy()
    N, M = mat.shape
    out = []
    for _ in range(min(N, M)):
        i, j = divmod(int(np.argmax(mat)), M)
        out.append((i, j, float(m[i, j])))
        mat[i, :] = -np.inf
        mat[:, j] = -np.inf
    out.sort(key=lambda t: t[0])
    return out


def hungarian(m, min_score=0.0):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return greedy(m)
    rows, cols = linear_sum_assignment(-m.astype(float))
    out = [(int(i), int(j), float(m[i, j])) for i, j in zip(rows, cols)
           if float(m[i, j]) > min_score]
    out.sort(key=lambda t: t[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/export_t2408b/best_v3_2408b.onnx")
    ap.add_argument("--pre", default="runs/export_t2408b/pre_model_v7_2408b.onnx")
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--labeled", default="data/labeled2408")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dedup", type=float, default=0.5)
    ap.add_argument("--norm-short", type=int, default=64)
    ap.add_argument("--out", default="out/diag2408")
    args = ap.parse_args()

    from src.utils import yolo_onnx, matchingMode, ver_onnx

    root = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    lab = args.labeled if os.path.isabs(args.labeled) else os.path.join(REPO, args.labeled)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    # ---- 真值：提示图标框（来自 meta 的 ins 切分）+ 目标框（来自 pairs）
    gt = {}
    idx = os.path.join(lab, "labels_index.jsonl")
    for line in open(idx, encoding="utf-8"):
        r = json.loads(line)
        boxes = r.get("bg_boxes") or []
        order = []
        for p in sorted(r.get("pairs") or [], key=lambda z: z["prompt"]):
            j = p["bg"]
            if 0 <= j < len(boxes):
                order.append(boxes[j])
        if order and len(order) == r.get("n_prompt"):
            gt[r["sid"]] = {"targets": order, "n_prompt": r["n_prompt"]}

    # 真值提示框：从标注图的 label 里读 class2（等比放置，坐标精确）
    def gt_prompt_boxes(sid):
        for split in ("train", "val"):
            p = os.path.join(lab, "labels", split, sid + ".txt")
            if not os.path.exists(p):
                continue
            img_p = os.path.join(lab, "images", split, sid + ".jpg")
            im = cv2.imdecode(np.fromfile(img_p, np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            H, W = im.shape[:2]
            out_ = []
            for line in open(p):
                v = line.split()
                if len(v) != 5 or int(float(v[0])) != 2:
                    continue
                cx, cy, bw, bh = [float(x) for x in v[1:]]
                out_.append([(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H])
            out_.sort(key=lambda b: b[0])
            return out_
        return []

    y = yolo_onnx.YOLO(os.path.join(REPO, args.det), conf_threshold=args.conf)
    pre = ver_onnx.PreONNX(os.path.join(REPO, args.pre))

    sids = [s for s in gt if os.path.exists(os.path.join(root, s, "main.jpg"))][:args.limit]
    print(f"[diag] {len(sids)} 题  det={os.path.basename(args.det)}  "
          f"pre={os.path.basename(args.pre)}")

    # 统计容器
    box_ious = []          # 真值目标框 vs 最佳检测框
    det_dup = 0
    det_total = 0
    miss = 0
    extra = 0
    mode_stat = {k: {"exact": 0, "hits": 0, "tot": 0, "n": 0}
                 for k in ("A", "B", "C", "D")}
    t0 = time.time()

    for sid in sids:
        ip = os.path.join(root, sid, "main.jpg")
        img = matchingMode.open_image(ip)
        if img is None:
            continue
        d = y.inference(img)
        if args.dedup > 0:
            d = nms_same_class(d, args.dedup)
        det_t = [x[:4] for x in d if x[5] == 0]
        det_c = [x[:4] for x in d if x[5] == 2]
        det_c.sort(key=lambda b: b[0])
        g = gt[sid]
        gt_t = g["targets"]
        gt_c = gt_prompt_boxes(sid)

        # --- Q1 检测框质量
        det_total += len(det_t)
        for b in gt_t:
            best = max([iou(b, x) for x in det_t], default=0.0)
            box_ious.append(best)
            if best < 0.5:
                miss += 1
        for i in range(len(det_t)):
            for j in range(i + 1, len(det_t)):
                if iou(det_t[i], det_t[j]) > 0.5:
                    det_dup += 1
        extra += max(0, len(det_t) - len(gt_t))

        # --- 四组对照
        # A: 真值提示 + 真值目标
        if gt_c and gt_t:
            ics = [crop(img, b) for b in gt_c]
            its = [crop(img, b) for b in gt_t]
            ics = [c for c in ics if c is not None]
            its = [t for t in its if t is not None]
            if ics and its:
                m = score_matrix(pre, ics, its)
                for mode, M_, tb in (("A", m, gt_t),):
                    order = greedy(M_)
                    res = [tb[j] for _i, j, _s in order]
                    h = sum(1 for k in range(min(len(gt_t), len(res)))
                            if iou(gt_t[k], res[k]) > 0.5)
                    st = mode_stat[mode]
                    st["hits"] += h
                    st["tot"] += len(gt_t)
                    st["n"] += 1
                    if h == len(gt_t) and len(res) == len(gt_t):
                        st["exact"] += 1

        # B: 真值提示 + 检测目标
        if gt_c and det_t:
            ics = [crop(img, b) for b in gt_c]
            its = [crop(img, b, norm_short=args.norm_short or None) for b in det_t]
            ics = [c for c in ics if c is not None]
            its = [t for t in its if t is not None]
            if ics and its and len(its) == len(det_t):
                m = score_matrix(pre, ics, its)
                order = greedy(m)
                res = [det_t[j] for _i, j, _s in order]
                h = sum(1 for k in range(min(len(gt_t), len(res)))
                        if iou(gt_t[k], res[k]) > 0.5)
                st = mode_stat["B"]
                st["hits"] += h
                st["tot"] += len(gt_t)
                st["n"] += 1
                if h == len(gt_t) and len(res) == len(gt_t):
                    st["exact"] += 1

        # C: 检测提示 + 真值目标
        if det_c and gt_t:
            ics = [crop(img, b, norm_short=args.norm_short or None) for b in det_c]
            its = [crop(img, b) for b in gt_t]
            ics = [c for c in ics if c is not None]
            its = [t for t in its if t is not None]
            if ics and its:
                m = score_matrix(pre, ics, its)
                order = greedy(m)
                res = [gt_t[j] for _i, j, _s in order]
                h = sum(1 for k in range(min(len(gt_t), len(res)))
                        if iou(gt_t[k], res[k]) > 0.5)
                st = mode_stat["C"]
                st["hits"] += h
                st["tot"] += len(gt_t)
                st["n"] += 1
                if h == len(gt_t) and len(res) == len(gt_t):
                    st["exact"] += 1

        # D: 检测提示 + 检测目标（真实路径）
        if det_c and det_t:
            ics = [crop(img, b, norm_short=args.norm_short or None) for b in det_c]
            its = [crop(img, b, norm_short=args.norm_short or None) for b in det_t]
            ics = [c for c in ics if c is not None]
            its = [t for t in its if t is not None]
            if ics and its:
                m = score_matrix(pre, ics, its)
                order = greedy(m)
                res = [det_t[j] for _i, j, _s in order]
                h = sum(1 for k in range(min(len(gt_t), len(res)))
                        if iou(gt_t[k], res[k]) > 0.5)
                st = mode_stat["D"]
                st["hits"] += h
                st["tot"] += len(gt_t)
                st["n"] += 1
                if h == len(gt_t) and len(res) == len(gt_t):
                    st["exact"] += 1

    bi = np.array(box_ious) if box_ious else np.array([0.0])
    rep = {
        "det": args.det, "pre": args.pre, "limit": len(sids),
        "conf": args.conf, "dedup": args.dedup, "norm_short": args.norm_short,
        "Q1_detector": {
            "gt_boxes": len(box_ious),
            "recall_iou50": round(float((bi > 0.5).mean()), 4),
            "iou_mean": round(float(bi.mean()), 4),
            "iou_p10": round(float(np.percentile(bi, 10)), 4),
            "iou_p50": round(float(np.percentile(bi, 50)), 4),
            "dup_pairs": det_dup,
            "extra_boxes": extra,
            "det_per_img": round(det_total / max(len(sids), 1), 2),
        },
        "Q2_Q3_ablation": {},
        "elapsed_s": round(time.time() - t0, 1),
    }
    for k, st in mode_stat.items():
        rep["Q2_Q3_ablation"][k] = {
            "n": st["n"],
            "exact_rate": round(st["exact"] / max(st["n"], 1), 4),
            "point_hit": round(st["hits"] / max(st["tot"], 1), 4),
        }

    with open(os.path.join(out, "diag_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print("\n=== Q1 检测器 ===")
    q = rep["Q1_detector"]
    print(f"  真值框召回 IoU>0.5 : {q['recall_iou50']:.1%}  ({q['gt_boxes']} 框)")
    print(f"  IoU 分布           : mean={q['iou_mean']:.3f} "
          f"p10={q['iou_p10']:.3f} p50={q['iou_p50']:.3f}")
    print(f"  重复框对           : {q['dup_pairs']}")
    print(f"  多余框             : {q['extra_boxes']}  "
          f"(平均 {q['det_per_img']} 框/题)")

    print("\n=== Q2/Q3 四组对照 ===")
    names = {"A": "真值提示+真值目标（纯匹配上界）",
             "B": "真值提示+检测目标",
             "C": "检测提示+真值目标",
             "D": "检测提示+检测目标（真实路径）"}
    for k in ("A", "B", "C", "D"):
        v = rep["Q2_Q3_ablation"][k]
        print(f"  {k} {names[k]:<32} exact={v['exact_rate']:.1%} "
              f"point={v['point_hit']:.1%}  (n={v['n']})")

    a = rep["Q2_Q3_ablation"]["A"]["exact_rate"]
    dd = rep["Q2_Q3_ablation"]["D"]["exact_rate"]
    print(f"\n  A-D 差距（框来源代价）: {(a-dd)*100:+.1f} pt")
    print(f"\n报告 -> {os.path.join(out, 'diag_report.json')}")


if __name__ == "__main__":
    main()
