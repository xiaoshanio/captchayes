# -*- coding: utf-8 -*-
"""badcase_2408: 抓出错配案例，输出并排对比图，定位"正确框分数≈0"的原因。

动机（来自 diag 的关键数据）：
    错配时**正确框的分数中位数只有 0.048**，而命中时是 0.956。
    这不是"两个候选分数接近"，而是"正确框根本没被认出来"。

    两种可能：
      (a) 提示图标切分错了（切出来的不是那个图标）
      (b) 底图检测框裁到了错误区域（框里不是那个图标）
      (c) 匹配器对某种形变/尺寸组合失效

    本脚本把 [提示图标 | 正确底图框 | 实际选中的框] 并排拼图，
    人眼一看就知道是哪种。

用法：
    python tools/badcase_2408.py --limit 60 --max-cases 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from diag_2408 import (iou, crop, score_matrix, greedy,   # noqa: E402
                       nms_same_class)
from autolabel_2408 import split_prompt_icons             # noqa: E402


def pad_to(img, h, w):
    out = np.full((h, w, 3), 255, np.uint8)
    ih, iw = img.shape[:2]
    out[:min(ih, h), :min(iw, w)] = img[:min(ih, h), :min(iw, w)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/export_t2408b/best_v3_2408b.onnx")
    ap.add_argument("--pre", default="runs/export_t2408b/pre_model_v7_2408b.onnx")
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--labeled", default="data/labeled2408")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--max-cases", type=int, default=8)
    ap.add_argument("--score-thr", type=float, default=0.4,
                    help="正确框分数低于此值视为错配案例")
    ap.add_argument("--out", default="out/badcase2408")
    args = ap.parse_args()

    from src.utils import yolo_onnx, matchingMode, ver_onnx

    root = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    lab = args.labeled if os.path.isabs(args.labeled) else os.path.join(REPO, args.labeled)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    gt = {}
    for line in open(os.path.join(lab, "labels_index.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        boxes = r.get("bg_boxes") or []
        order = [boxes[p["bg"]] for p in sorted(r.get("pairs") or [],
                                                key=lambda z: z["prompt"])
                 if 0 <= p["bg"] < len(boxes)]
        if order and len(order) == r.get("n_prompt"):
            gt[r["sid"]] = order

    y = yolo_onnx.YOLO(os.path.join(REPO, args.det), conf_threshold=0.25)
    pre = ver_onnx.PreONNX(os.path.join(REPO, args.pre))

    shown = 0
    recs = []
    for sid in list(gt)[:args.limit]:
        img = matchingMode.open_image(os.path.join(root, sid, "main.jpg"))
        if img is None:
            continue
        d = nms_same_class(y.inference(img), 0.5)
        dt = [x[:4] for x in d if x[5] == 0]
        dc = [x[:4] for x in d if x[5] == 2]
        dc.sort(key=lambda b: b[0])
        meta_p = os.path.join(root, sid, "meta.json")
        if not os.path.exists(meta_p):
            continue
        meta = json.load(open(meta_p, encoding="utf-8"))
        prs = meta.get("prompts") or []
        if not prs:
            continue
        pr = cv2.imdecode(np.fromfile(os.path.join(root, sid, prs[0]["file"]),
                                      np.uint8), cv2.IMREAD_COLOR)
        if pr is None:
            continue
        icons = split_prompt_icons(pr)
        if not icons or len(dc) < len(icons):
            continue

        ics = [crop(icons[i]["img"],
                    (0, 0, icons[i]["img"].shape[1], icons[i]["img"].shape[0]),
                    norm_short=64) for i in range(len(icons))]
        its = [crop(img, b, norm_short=64) for b in dt]
        keep = [i for i, c in enumerate(its) if c is not None]
        its = [its[i] for i in keep]
        dt = [dt[i] for i in keep]
        ics = [c for c in ics if c is not None]
        if not ics or not its:
            continue

        m = score_matrix(pre, ics, its)
        g = gt[sid]
        for i, gb in enumerate(g):
            if i >= m.shape[0]:
                break
            bi, bs = -1, 0.0
            for j, b in enumerate(dt):
                v = iou(gb, b)
                if v > 0.5 and v > bs:
                    bs, bi = v, j
            if bi < 0 or m[i][bi] > args.score_thr:
                continue
            top = int(np.argmax(m[i]))
            ih = icons[i]["img"].shape
            rec = {
                "sid": sid, "prompt": i,
                "score_correct": round(float(m[i][bi]), 4),
                "score_top": round(float(m[i][top]), 4),
                "top_is_correct": bool(top == bi),
                "prompt_wh": [int(ih[1]), int(ih[0])],
                "gt_box": [int(v) for v in gb],
                "det_box": [int(v) for v in dt[bi]],
                "det_wh": [int(dt[bi][2]-dt[bi][0]), int(dt[bi][3]-dt[bi][1])],
                "det_conf": None,
            }
            recs.append(rec)

            # 并排图：提示 | 正确底图框 | 实际选中框
            H = 80
            W = 80
            tile = np.full((H + 18, W * 3 + 20, 3), 255, np.uint8)
            tile[:80, :80] = pad_to(ics[i], 80, 80)
            tile[:80, 90:170] = pad_to(its[bi], 80, 80)
            tile[:80, 180:260] = pad_to(its[top], 80, 80)
            cv2.putText(tile, f"prompt {i}", (2, 95), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 1)
            cv2.putText(tile, f"GT {m[i][bi]:.2f}", (92, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(tile, f"PICKED {m[i][top]:.2f}", (182, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            fn = os.path.join(out, f"{sid}_p{i}.jpg")
            cv2.imencode(".jpg", tile)[1].tofile(fn)
            shown += 1
            print(f"  {sid} p{i}: correct={m[i][bi]:.3f} top={m[i][top]:.3f} "
                  f"prompt={ih[1]}x{ih[0]} det={rec['det_wh'][0]}x{rec['det_wh'][1]}")
            if shown >= args.max_cases:
                break
        if shown >= args.max_cases:
            break

    with open(os.path.join(out, "badcases.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)

    print(f"\n=== {shown} 个错配案例 -> {out}/ ===")
    if recs:
        pw = [r["prompt_wh"][0] for r in recs]
        dw = [r["det_wh"][0] for r in recs]
        print(f"  提示图标宽 mean={np.mean(pw):.0f}  底图框宽 mean={np.mean(dw):.0f}")
        print(f"  正确分 mean={np.mean([r['score_correct'] for r in recs]):.3f}")


if __name__ == "__main__":
    main()
