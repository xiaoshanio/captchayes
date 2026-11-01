# -*- coding: utf-8 -*-
"""prepare_pairs_det: 用**检测器真实输出框**生成匹配器训练配对（训练-推理对齐）。

为什么需要它
------------
`eval/crop_source_test.py` 实测（漂移域，34 题）：

    匹配器喂真值框裁剪 : 91.2% 全对
    匹配器喂检测框裁剪 : 61.8% 全对     <- 掉 29.4 pt
    检测框 vs 真值框 IoU: mean=0.81 p10=0.69

原因：匹配器训练时只见"完美真值框"裁出的干净图标，
推理时拿到的却是检测框裁出的图（多带背景 / 切掉笔画），
**训练分布与推理分布不一致**。

`train/prepare_pairs.py --jitter` 曾试图用**随机抖动**模拟这种误差，
但实测失败（模型塌缩成常数输出，见 docs/DRIFT_EXPERIMENT.md 6.5）。
随机抖动太"野"，会破坏正样本。

本脚本改用**真实检测框**：先把已训好的检测器在训练集上跑一遍，
用它实际输出的框去裁剪。这样匹配器看到的分布
**与推理时逐点一致**，且不会引入人为破坏。

与 prepare_pairs.py 的区别
--------------------------
    prepare_pairs.py      : 框来自 pairs/*.json 的 target_box / prompt_box（真值）
    prepare_pairs_det.py  : 框来自检测器实际输出，再用 IoU 对齐回真值拿标签

用法
----
    python tools/prepare_pairs_det.py \
        --synth data/driftB \
        --det runs/drift_rec/export/detB.onnx \
        --out data/driftB_pairs_det \
        --conf 0.2

产物
----
    <out>/train.jsonl, <out>/val.jsonl     每行 {"a":crop,"b":crop,"label":0|1,"hard":bool}
    <out>/_crops/                           裁剪缓存
    <out>/align_report.json                 对齐统计（有多少题能完整对齐等）
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


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img, q=95):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        buf.tofile(p)
    return ok


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def dedup(items, thr=0.45):
    """items: [(box, conf)]，按 conf 降序保留，抑制近重合框。"""
    out = []
    for b, c in sorted(items, key=lambda t: -t[1]):
        if all(iou(b, o[0]) <= thr for o in out):
            out.append((b, c))
    return out


def crop_to(img, box, cache_dir, key, pad=0):
    fn = os.path.join(cache_dir, key + ".jpg")
    if os.path.exists(fn):
        return fn
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
    imwrite_u(fn, c)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="data/driftB")
    ap.add_argument("--det", default="runs/drift_rec/export/detB.onnx")
    ap.add_argument("--out", default="data/driftB_pairs_det")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--pad", type=int, default=4, help="裁剪外扩像素（与推理口径一致）")
    ap.add_argument("--dd-thr", type=float, default=0.45, help="去重 IoU 阈值")
    ap.add_argument("--align-iou", type=float, default=0.5,
                    help="检测框与真值框对齐所需的最小 IoU")
    ap.add_argument("--neg-per-pos", type=int, default=3)
    args = ap.parse_args()

    from src.utils import yolo_onnx
    det = yolo_onnx.YOLO(os.path.join(REPO, args.det))
    det.conf_threshold = args.conf

    synth = os.path.join(REPO, args.synth)
    out = os.path.join(REPO, args.out)
    cache = os.path.join(out, "_crops")
    os.makedirs(cache, exist_ok=True)

    report = {"synth": args.synth, "det": args.det, "conf": args.conf,
              "splits": {}}

    for split in ("train", "val"):
        img_dir = os.path.join(synth, "images", split)
        pr_dir = os.path.join(synth, "pairs", split)
        if not os.path.isdir(pr_dir):
            print(f"[skip] {split}: 缺 {pr_dir}")
            continue

        bases = sorted(f[:-5] for f in os.listdir(pr_dir) if f.endswith(".json"))
        rows = []
        n_pos = n_neg = n_hard = 0
        n_img = n_aligned = 0
        missing_prompt = missing_target = 0

        for base in bases:
            ip = os.path.join(img_dir, base + ".jpg")
            if not os.path.exists(ip):
                continue
            img = imread_u(ip)
            if img is None:
                continue
            n_img += 1
            truth = json.load(open(os.path.join(pr_dir, base + ".json"), encoding="utf-8"))
            gts = sorted(truth.get("pairs", []), key=lambda q: q["click_order"])
            if len(gts) < 2:
                continue

            dets = det.inference(img)
            r0 = dedup([(x[:4], float(x[4])) for x in dets if int(x[5]) == 0], args.dd_thr)
            r2 = dedup([(x[:4], float(x[4])) for x in dets if int(x[5]) == 2], args.dd_thr)
            if not r0 or not r2:
                missing_prompt += 1
                continue

            # --- 提示侧：为每个真值 prompt 找 IoU 最大的检测框
            prompt_crops = []
            ok = True
            for q in gts:
                best, bi = 0.0, -1
                for i, (bb, _c) in enumerate(r2):
                    v = iou(bb, q["prompt_box"])
                    if v > best:
                        best, bi = v, i
                if bi < 0 or best < args.align_iou:
                    ok = False
                    break
                key = f"{split}_{base}_dp{q['click_order']}"
                p = crop_to(img, r2[bi][0], cache, key, args.pad)
                if p is None:
                    ok = False
                    break
                prompt_crops.append((q, p))
            if not ok:
                missing_prompt += 1
                continue

            # --- 目标侧：为每个真值 target 找 IoU 最大的检测框
            tgt_crops = []
            for q in gts:
                best, bi = 0.0, -1
                for i, (bb, _c) in enumerate(r0):
                    v = iou(bb, q["target_box"])
                    if v > best:
                        best, bi = v, i
                if bi < 0 or best < args.align_iou:
                    ok = False
                    break
                key = f"{split}_{base}_dt{q['click_order']}"
                p = crop_to(img, r0[bi][0], cache, key, args.pad)
                if p is None:
                    ok = False
                    break
                tgt_crops.append(p)
            if not ok:
                missing_target += 1
                continue

            n_aligned += 1
            # --- 正样本
            for k, (q, pp) in enumerate(prompt_crops):
                rows.append({"a": pp, "b": tgt_crops[k], "label": 1, "hard": False})
                n_pos += 1
            # --- 难负样本：同题内其它目标检测框（这才是推理时真正会混淆的）
            for k, (q, pp) in enumerate(prompt_crops):
                for j, tp in enumerate(tgt_crops):
                    if j == k:
                        continue
                    rows.append({"a": pp, "b": tp, "label": 0, "hard": True})
                    n_neg += 1
                    n_hard += 1

        outp = os.path.join(out, f"{split}.jsonl")
        with open(outp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rate = n_aligned / max(n_img, 1)
        print(f"[{split}] 图 {n_img}  完整对齐 {n_aligned} ({rate*100:.1f}%)  "
              f"pos={n_pos} neg={n_neg} (hard={n_hard})")
        if missing_prompt:
            print(f"  提示侧对齐失败 {missing_prompt} 张")
        if missing_target:
            print(f"  目标侧对齐失败 {missing_target} 张")
        report["splits"][split] = {
            "n_img": n_img, "n_aligned": n_aligned,
            "align_rate": round(rate, 4),
            "pos": n_pos, "neg": n_neg, "hard": n_hard,
            "missing_prompt": missing_prompt, "missing_target": missing_target,
        }

    with open(os.path.join(out, "align_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告 -> {os.path.join(args.out, 'align_report.json')}")
    print("\n下一步：用这份配对训练匹配器（框口径与推理一致）")
    print(f"  python train/train_matcher.py --pairs {args.out} --epochs 6 \\")
    print("      --batch 64 --workers 0 --threads 10 --out-onnx <路径>")


if __name__ == "__main__":
    main()
