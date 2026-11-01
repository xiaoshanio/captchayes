# -*- coding: utf-8 -*-
"""crop_source_test: 关键假设验证 —— 匹配器对「裁剪框来源」有多敏感？

动机：
    漂移域上出现了矛盾数据：
        - 检测器分项很好      : class0 P=0.97/R=0.99, class2 P=0.96/R=0.96
        - 匹配器喂**真值框**很好: 87.5% 全对
        - 但端到端只有 5~15%
    这三者不可能同时成立，除非**中间环节把信息破坏了**。

    唯一没测过的环节：检测框 vs 真值框的**几何抖动**。
    匹配器是在真值框裁出的干净 crop 上训练的；
    推理时喂给它的是**检测框**裁出的 crop —— 如果检测框有偏移/尺度误差，
    裁出来的图标会包含更多背景或切掉笔画，匹配器的相似度就会退化。

本脚本做四组对照，把「框来源」这一个变量单独拿出来：

    A. 真值框裁剪 + 真值框裁剪      -> 上界（纯匹配能力）
    B. 检测框裁剪 + 检测框裁剪      -> 真实推理路径
    C. 真值提示框 + 检测目标框      -> 只有目标侧抖动
    D. 检测提示框 + 真值目标框      -> 只有提示侧抖动

若 A 高而 B 低，就证明**匹配器对框抖动敏感**，
那么正确的修复方向不是"再训检测器"，而是：
    ① 训练匹配器时就用**检测框裁剪**（训练-推理对齐）
    ② 或给 crop 加 padding / 用中心点采样，降低框误差敏感性

用法：
    python eval/crop_source_test.py --det runs/drift_rec/export/detB.onnx \
        --mat runs/drift_rec/export/matB.onnx --data data/driftB --limit 40
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


def dedup(items, thr=0.45):
    out = []
    for b, c in sorted(items, key=lambda t: -t[1]):
        if all(iou(b, o[0]) <= thr for o in out):
            out.append((b, c))
    return out


def crop(img, box, pad=0):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    c = img[y1:y2, x1:x2]
    return c if c.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/drift_rec/export/detB.onnx")
    ap.add_argument("--mat", default="runs/drift_rec/export/matB.onnx")
    ap.add_argument("--data", default="data/driftB")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--pad", type=int, default=0, help="crop 外扩像素（测试容错性）")
    ap.add_argument("--out", default="out/crop_src")
    args = ap.parse_args()
    os.makedirs(os.path.join(REPO, args.out), exist_ok=True)

    from src.utils import yolo_onnx, ver_onnx
    det = yolo_onnx.YOLO(os.path.join(REPO, args.det))
    det.conf_threshold = args.conf
    mat = ver_onnx.PreONNX(os.path.join(REPO, args.mat))

    d = os.path.join(REPO, args.data)
    pdir = os.path.join(d, "pairs", "val")
    bases = [f[:-5] for f in sorted(os.listdir(pdir))][:args.limit]

    stats = {k: {"ok": 0, "tot": 0} for k in ("A_gt_gt", "B_det_det", "C_gtp_dett", "D_detp_gtt")}
    box_err = []   # 检测框 vs 真值框的 IoU 分布

    for b in bases:
        ip = os.path.join(d, "images", "val", b + ".jpg")
        img = cv2.imdecode(np.fromfile(ip, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        truth = json.load(open(os.path.join(pdir, b + ".json"), encoding="utf-8"))
        gt = sorted(truth["pairs"], key=lambda q: q["click_order"])
        if len(gt) < 2:
            continue

        # 真值 crop
        gt_p = [crop(img, q["prompt_box"], args.pad) for q in gt]
        gt_t = [crop(img, q["target_box"], args.pad) for q in gt]
        if any(x is None for x in gt_p + gt_t):
            continue

        # 检测框：按与真值的 IoU 一一对齐，只保留能对上的
        dets = det.inference(img)
        r0 = dedup([(x[:4], float(x[4])) for x in dets if int(x[5]) == 0])
        r2 = dedup([(x[:4], float(x[4])) for x in dets if int(x[5]) == 2])
        det_t_boxes = [z[0] for z in r0]
        det_p_boxes = sorted([z[0] for z in r2], key=lambda z: z[0])

        # 提示侧：检测框按 x 排序后，必须与 gt 顺序一一对应（数量也要对）
        if len(det_p_boxes) != len(gt):
            continue
        det_p = [crop(img, bb, args.pad) for bb in det_p_boxes]
        # 目标侧：为每个 gt target 找 IoU 最大的检测框
        det_t = []
        ok_targets = True
        for q in gt:
            best, bi = 0.0, -1
            for i, bb in enumerate(det_t_boxes):
                v = iou(bb, q["target_box"])
                if v > best:
                    best, bi = v, i
            if bi < 0 or best < 0.5:
                ok_targets = False
                break
            box_err.append(best)
            det_t.append(crop(img, det_t_boxes[bi], args.pad))
        if not ok_targets or any(x is None for x in det_p + det_t):
            continue

        for key, P, T in (("A_gt_gt", gt_p, gt_t),
                          ("B_det_det", det_p, det_t),
                          ("C_gtp_dett", gt_p, det_t),
                          ("D_detp_gtt", det_p, gt_t)):
            s = mat.reason_all_batch(P, T)
            res = greedy(s)
            stats[key]["tot"] += 1
            if all(j == i for i, j in res):
                stats[key]["ok"] += 1

    print("=" * 90)
    print(f"裁剪框来源对照   det={os.path.basename(args.det)}  mat={os.path.basename(args.mat)}")
    print(f"数据={args.data}  limit={args.limit}  pad={args.pad}px")
    print("=" * 90)
    labels = {
        "A_gt_gt": "A 真值框 → 真值框（上界）",
        "B_det_det": "B 检测框 → 检测框（真实推理）",
        "C_gtp_dett": "C 真值提示 + 检测目标",
        "D_detp_gtt": "D 检测提示 + 真值目标",
    }
    rows = []
    for k in ("A_gt_gt", "B_det_det", "C_gtp_dett", "D_detp_gtt"):
        s = stats[k]
        rate = s["ok"] / max(s["tot"], 1)
        rows.append({"case": k, "label": labels[k], "n": s["tot"],
                     "ok": s["ok"], "rate": round(rate, 4)})
        print(f"  {labels[k]:28s} {s['ok']:2d}/{s['tot']:<2d}  ({rate*100:5.1f}%)")

    if box_err:
        be = np.array(box_err)
        print(f"\n  检测框 vs 真值框 IoU: mean={be.mean():.3f} "
              f"p10={np.percentile(be,10):.3f} p50={np.percentile(be,50):.3f}")

    print("\n" + "=" * 90)
    print("判读")
    print("=" * 90)
    a = stats["A_gt_gt"]["ok"] / max(stats["A_gt_gt"]["tot"], 1)
    bb_ = stats["B_det_det"]["ok"] / max(stats["B_det_det"]["tot"], 1)
    c = stats["C_gtp_dett"]["ok"] / max(stats["C_gtp_dett"]["tot"], 1)
    dd = stats["D_detp_gtt"]["ok"] / max(stats["D_detp_gtt"]["tot"], 1)
    gap = (a - bb_) * 100
    print(f"  A - B = {gap:+.1f} pt   （框抖动造成的总损失）")
    print(f"  A - C = {(a-c)*100:+.1f} pt   （仅目标侧抖动）")
    print(f"  A - D = {(a-dd)*100:+.1f} pt   （仅提示侧抖动）")
    if gap > 15:
        print("\n  ✓ 匹配器对**框抖动高度敏感** —— 这是漂移域端到端失败的关键机制")
        print("    修复方向：")
        print("      1) 训练匹配器时就用**检测框裁剪**（训练-推理对齐），而不是真值框")
        print("      2) crop 加 padding（--pad 8~12）让匹配器见到更多上下文")
        print("      3) 提示侧用 x 中心点 + 固定宽高采样，降低框误差影响")
    else:
        print("\n  · 框抖动影响有限，瓶颈在别处")

    json.dump({"rows": rows,
               "box_iou": {"mean": float(np.mean(box_err)) if box_err else None,
                           "p10": float(np.percentile(box_err, 10)) if box_err else None},
               "args": vars(args)},
              open(os.path.join(REPO, args.out, "crop_source.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n报告 -> {os.path.join(args.out, 'crop_source.json')}")


if __name__ == "__main__":
    main()
