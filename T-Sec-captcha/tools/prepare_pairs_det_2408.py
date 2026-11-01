# -*- coding: utf-8 -*-
"""prepare_pairs_det_2408: 用**检测框**裁剪生成匹配器训练配对（训练-推理对齐）。

为什么必须换成检测框裁剪（这是本轮最关键的一步）：

    tools/diag_2408.py 的四组对照给出了决定性证据（60 题）：

        A 真值提示 + 真值目标    exact 95.0%  point 96.7%   ← 纯匹配上界
        B 真值提示 + 检测目标    exact 70.0%  point 86.7%   ← 目标侧换检测框
        C 检测提示 + 真值目标    exact 85.0%  point 89.4%   ← 提示侧换检测框
        D 检测提示 + 检测目标    exact 68.3%  point 87.8%   ← 真实路径

    结论：
      · 检测器召回 99.4%、IoU mean 0.876 —— **检测器不是瓶颈**
      · 匹配器在干净框上 95%，喂检测框掉到 70% —— **匹配器对框来源敏感**
      · 几何偏差其实很小（宽 +2.0% / 高 +1.6% / 中心偏 3%），
        但训练分布里从来没有这种扰动，所以模型没学会忽略它

    上一版用 `--jitter 0.12` 在标注框上加噪声，但那是**均匀随机扰动**，
    与检测框的真实偏差形态（系统性内缩 + 边缘抖动）不同。
    正确做法：**直接拿检测器的输出当训练裁剪框**，训练-推理完全对齐。

配对构造（与 prepare_pairs_2408 相同的标签语义，只是框来源变了）：

    正样本   : 提示图标 ↔ 与真值框 IoU>0.5 的那个检测框
    难负样本 : 同题的其它检测框（同图同风格，最难）
    易负样本 : 跨题随机检测框

    同时保留一部分「标注框裁剪」的正样本（--gt-ratio），
    让模型既见过干净框也见过检测框 —— 相当于分布插值，比纯检测框更稳。

产出：
    <out>/train.jsonl, val.jsonl, crops/, pairs_report.json

用法：
    python tools/prepare_pairs_det_2408.py --det runs/export_t2408b/best_v3_2408b.onnx \
        --labeled data/labeled2408 --samples data/samples --out data/pairsdet2408
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


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1)


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img, q=95):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        buf.tofile(p)
    return ok


def crop(img, box, cache_dir, key, norm_short=None, pad=0):
    fn = os.path.join(cache_dir, key + ".jpg")
    if os.path.exists(fn):
        return fn
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
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
    imwrite_u(fn, c)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/export_t2408b/best_v3_2408b.onnx")
    ap.add_argument("--labeled", default="data/labeled2408")
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--out", default="data/pairsdet2408")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dedup", type=float, default=0.5)
    ap.add_argument("--norm-short", type=int, default=64)
    ap.add_argument("--hard-neg", type=int, default=2)
    ap.add_argument("--easy-neg", type=int, default=1)
    ap.add_argument("--gt-ratio", type=float, default=0.35,
                    help="正样本里用标注框裁剪的比例（其余用检测框）。"
                         "0=纯检测框，1=纯标注框")
    ap.add_argument("--min-iou", type=float, default=0.5,
                    help="检测框与真值框的 IoU 下限，低于此视为该目标未检出")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from src.utils import yolo_onnx, matchingMode

    lab = args.labeled if os.path.isabs(args.labeled) else os.path.join(REPO, args.labeled)
    sroot = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    crops_dir = os.path.join(out, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    recs = [json.loads(l) for l in
            open(os.path.join(lab, "labels_index.jsonl"), encoding="utf-8") if l.strip()]
    print(f"[scan] {len(recs)} 题标注")

    y = yolo_onnx.YOLO(os.path.join(REPO, args.det), conf_threshold=args.conf)
    rng = random.Random(args.seed)

    # 预跑检测，缓存每题检测框
    det_cache = {}
    for r in recs:
        sid = r["sid"]
        p = os.path.join(sroot, sid, "main.jpg")
        if not os.path.exists(p):
            continue
        img = matchingMode.open_image(p)
        if img is None:
            continue
        d = y.inference(img)
        if args.dedup > 0:
            keep = []
            for x in sorted(d, key=lambda z: -z[4]):
                if all(x[5] != k[5] or iou(x[:4], k[:4]) <= args.dedup for k in keep):
                    keep.append(x)
            d = keep
        det_cache[sid] = {"t": [x[:4] for x in d if x[5] == 0],
                          "c": [x[:4] for x in d if x[5] == 2],
                          "img": img}
    print(f"[det] 已检测 {len(det_cache)} 题")

    # 跨题易负样本池
    all_det_boxes = [(sid, b) for sid, v in det_cache.items() for b in v["t"]]
    if not all_det_boxes:
        raise SystemExit("没有任何检测框，检查 --det / --conf")

    rows = []
    n_pos_gt = n_pos_det = n_hard = n_easy = 0
    skipped = 0

    for r in recs:
        sid = r["sid"]
        if sid not in det_cache:
            skipped += 1
            continue
        v = det_cache[sid]
        img = v["img"]
        det_t = v["t"]
        det_c = v["c"]
        gt_t = r.get("bg_boxes") or []
        pairs = r.get("pairs") or []
        n_prompt = r.get("n_prompt") or 0

        # 提示图标裁剪（与 autolabel_2408 同一切法，保证与推理一致）
        from autolabel_2408 import split_prompt_icons
        meta_p = os.path.join(sroot, sid, "meta.json")
        if not os.path.exists(meta_p):
            skipped += 1
            continue
        meta = json.load(open(meta_p, encoding="utf-8"))
        prs = meta.get("prompts") or []
        if not prs:
            skipped += 1
            continue
        pr = imread_u(os.path.join(sroot, sid, prs[0]["file"]))
        if pr is None:
            skipped += 1
            continue
        icons = split_prompt_icons(pr)
        if not icons:
            skipped += 1
            continue

        a_paths = {}
        for i, it in enumerate(icons):
            fn = crop(it["img"], (0, 0, it["img"].shape[1], it["img"].shape[0]),
                      crops_dir, f"{sid}_p{i}", norm_short=args.norm_short or None)
            a_paths[i] = fn

        # 真值目标框 -> 对应的检测框索引（IoU 最大者）
        # 注意：这里必须保存**索引**而不是框对象。
        # 早期写成 `next(j for j,b in enumerate(det_t) if b is dbox)`，
        # 依赖对象身份 —— 而 det_t 的元素是 `x[:4]` 切片，每次访问都新建列表，
        # `is` 比较必然失败，导致 dj 恒为 None、整个配对集为空（静默失败）。
        gt2det = {}
        for k, g in enumerate(gt_t):
            best_j, bi = None, 0.0
            for j, b in enumerate(det_t):
                s = iou(g, b)
                if s > bi:
                    bi, best_j = s, j
            if best_j is not None and bi >= args.min_iou:
                gt2det[k] = best_j

        # 标注框裁剪缓存（用于 gt-ratio）
        gt_crops = {}
        for k, g in enumerate(gt_t):
            fn = crop(img, g, crops_dir, f"{sid}_gt{k}",
                      norm_short=args.norm_short or None)
            gt_crops[k] = fn

        matched_det_idx = set()
        for p in pairs:
            i, k = p["prompt"], p["bg"]
            if i >= len(icons):
                continue
            if k not in gt2det:
                continue
            dj = gt2det[k]
            dbox = det_t[dj]
            matched_det_idx.add(dj)

            # 正样本：按 gt-ratio 决定用标注框还是检测框
            use_gt = rng.random() < args.gt_ratio and gt_crops.get(k)
            if use_gt:
                bpath, src = gt_crops[k], "gt"
            else:
                bpath, src = crop(img, dbox, crops_dir, f"{sid}_d{dj}",
                                  norm_short=args.norm_short or None), "det"
            if a_paths.get(i) and bpath:
                rows.append({"a": a_paths[i], "b": bpath, "label": 1,
                             "hard": False, "sid": sid, "src": src})

            # 难负样本：同题其它检测框
            others = [j for j in range(len(det_t)) if j != dj]
            rng.shuffle(others)
            for j in others[:max(1, args.hard_neg)]:
                fn = crop(img, det_t[j], crops_dir, f"{sid}_d{j}",
                          norm_short=args.norm_short or None)
                if a_paths.get(i) and fn:
                    rows.append({"a": a_paths[i], "b": fn, "label": 0,
                                 "hard": True, "sid": sid, "src": "det"})
                    n_hard += 1

            # 易负样本：跨题随机检测框
            for _ in range(max(0, args.easy_neg)):
                osid, obox = rng.choice(all_det_boxes)
                if osid == sid:
                    continue
                ov = det_cache.get(osid)
                if ov is None:
                    continue
                fn = crop(ov["img"], obox, crops_dir, f"{osid}_e{abs(hash(tuple(obox)))%99999}",
                          norm_short=args.norm_short or None)
                if a_paths.get(i) and fn:
                    rows.append({"a": a_paths[i], "b": fn, "label": 0,
                                 "hard": False, "sid": sid, "src": "det"})
                    n_easy += 1

    if not rows:
        raise SystemExit("没有生成任何配对")

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    val, train = rows[:n_val], rows[n_val:]
    for name, part in (("train", train), ("val", val)):
        with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for x in part:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    n_pos = sum(1 for x in train if x["label"] == 1)
    rep = {
        "det": args.det, "labeled": lab, "samples": sroot, "out": out,
        "conf": args.conf, "dedup": args.dedup, "norm_short": args.norm_short,
        "gt_ratio": args.gt_ratio, "min_iou": args.min_iou,
        "n_items": len(rows), "train": len(train), "val": len(val),
        "train_pos": n_pos,
        "train_pos_from_gt": sum(1 for x in train if x["label"] == 1 and x["src"] == "gt"),
        "train_pos_from_det": sum(1 for x in train if x["label"] == 1 and x["src"] == "det"),
        "train_hard_neg": sum(1 for x in train if x["label"] == 0 and x["hard"]),
        "train_easy_neg": sum(1 for x in train if x["label"] == 0 and not x["hard"]),
        "skipped": skipped,
        "pos_weight_suggest": round(
            (len(train) - n_pos) / max(n_pos, 1), 3),
    }
    with open(os.path.join(out, "pairs_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print("\n=== 配对生成完成（检测框裁剪）===")
    print(f"  总配对     : {len(rows)}  (train={len(train)}, val={len(val)})")
    print(f"  正样本     : {n_pos}  (标注框 {rep['train_pos_from_gt']} / "
          f"检测框 {rep['train_pos_from_det']})")
    print(f"  难负样本   : {rep['train_hard_neg']}")
    print(f"  易负样本   : {rep['train_easy_neg']}")
    print(f"  跳过题     : {skipped}")
    print(f"  pos_weight : {rep['pos_weight_suggest']}")
    print(f"  输出       : {out}/")


if __name__ == "__main__":
    main()
