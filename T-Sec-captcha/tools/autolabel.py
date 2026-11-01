# -*- coding: utf-8 -*-
"""autolabel: 半自动标注 —— 用「多角度掩码边缘模板匹配」给新域样本生成 YOLO 伪标签。

核心思路（关键设计决策）：
    现有 runtime (src/captcha.py) 期望**单张图**同时含「提示字条」和「待点图」：
      class 2 = 'char'  = 提示条里的字（按 x 从左到右 = 点击顺序）
      class 0 = 'label' = 图面上可点击的字
      class 1 = 'title' = "请依次点击" 之类标题（runtime 不用）
    天御把提示图单独给了（170x50 精灵图）。为了**不改 runtime**，
    本脚本把「底图 + 提示条」拼成一张图（下边粘提示条），
    与旧域 344x384 的版式完全一致 → 模型可直接 drop-in 替换。

标签怎么来的：
    提示条里的图标 = 精确框（我们自己拼的，100% 准）  → class 2
    底图里的图标   = 模板匹配定位（有分数，可能错）    → class 0，低分进人工复核队列

用法：
    python tools/autolabel.py --samples data/samples --out data/labeled
    python tools/autolabel.py --samples data/samples --out data/labeled --review-thr 0.45
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time

import cv2
import numpy as np

STRIP_H = 56          # 底部提示条高度
TITLE_H = 0           # 标题条（暂不合成，保持简单）


# ------------------------------------------------------------------ 图像工具
def imread_u(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path, img, q=95):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        buf.tofile(path)
    return ok


def rotate_bound(img, angle):
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def edge_map(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.Canny(g, 60, 160)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


# ------------------------------------------------------------------ 提示图切图标
def split_prompt_icons(strip_bgr, ink_thr=150, min_w=8, gap=3):
    """按列投影把提示条切成 N 个图标，从左到右即点击顺序。"""
    g = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)
    ink = (g < ink_thr).astype(np.uint8)
    col = ink.sum(axis=0)
    runs, start = [], None
    for x, v in enumerate(col):
        if v > 0 and start is None:
            start = x
        elif v == 0 and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(col)))
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    out = []
    for a, b in merged:
        if b - a < min_w:
            continue
        sub = ink[:, a:b]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        y1, y2 = int(rows[0]), int(rows[-1]) + 1
        out.append({"x1": a, "y1": y1, "x2": b, "y2": y2,
                    "img": strip_bgr[y1:y2, a:b].copy()})
    return out


# ------------------------------------------------------------------ 模板匹配定位
def locate_icon(bg_edge, bg_shape, tpl, angles, scales, topk=8):
    """掩码边缘相关匹配，返回候选 [{score, angle, scale, box}]（已去重）。"""
    H, W = bg_shape[:2]
    cands = []
    for ang in angles:
        rt = rotate_bound(tpl, ang)
        for sc in scales:
            t = cv2.resize(rt, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR) if sc != 1.0 else rt
            g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if t.ndim == 3 else t
            mask = (g < 170).astype(np.uint8) * 255
            if mask.sum() == 0:
                continue
            th, tw = g.shape[:2]
            if th >= H or tw >= W or th < 10 or tw < 10:
                continue
            e = edge_map(t)
            if e.sum() == 0:
                continue
            try:
                res = cv2.matchTemplate(bg_edge, e, cv2.TM_CCORR_NORMED, mask=mask)
            except cv2.error:
                continue
            res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
            flat = res.ravel()
            k = min(topk, flat.size)
            for p in np.argpartition(-flat, k - 1)[:k]:
                y, x = divmod(int(p), res.shape[1])
                cands.append({"score": float(flat[p]), "angle": float(ang),
                              "scale": float(sc), "box": (x, y, x + tw, y + th)})

    # 去重：IoU 或中心距过近
    keep = []
    for c in sorted(cands, key=lambda d: -d["score"]):
        dup = False
        for k in keep:
            if iou(c["box"], k["box"]) > 0.3:
                dup = True
                break
            cx1, cy1 = (c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2
            cx2, cy2 = (k["box"][0] + k["box"][2]) / 2, (k["box"][1] + k["box"][3]) / 2
            if abs(cx1 - cx2) < 14 and abs(cy1 - cy2) < 14:
                dup = True
                break
        if not dup:
            keep.append(c)
    return keep


# ------------------------------------------------------------------ 合成训练图
def compose(bg, strip, canvas_w=672):
    """底图 + 底部提示条 → 单张训练图（版式对齐旧域 344x384）。"""
    h, w = bg.shape[:2]
    strip_r = cv2.resize(strip, (w, STRIP_H), interpolation=cv2.INTER_LINEAR) \
        if strip.shape[0] != STRIP_H else strip
    canvas = np.full((h + STRIP_H, w, 3), 255, dtype=np.uint8)
    canvas[:h, :w] = bg
    canvas[h:h + STRIP_H, :w] = strip_r
    return canvas, h


def yolo_line(cls, box, W, H):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    bw, bh = (x2 - x1) / W, (y2 - y1) / H
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ------------------------------------------------------------------ 主流程
def process_sample(sdir, angles, scales, review_thr, verbose=True):
    meta = json.load(open(os.path.join(sdir, "meta.json"), encoding="utf-8"))
    bg_p = os.path.join(sdir, "bg.jpg")
    pr = meta["prompts"][0]
    pr_p = os.path.join(sdir, pr["file"])
    bg = imread_u(bg_p)
    strip = imread_u(pr_p)
    if bg is None or strip is None:
        return None

    icons = split_prompt_icons(strip)
    if not icons:
        return None

    canvas, bg_h = compose(bg, strip)
    H, W = canvas.shape[:2]
    bg_edge = edge_map(canvas)          # 匹配在整张合成图上做（提示条区域也算，天然给干扰）

    labels = []
    # class 2：提示条图标（精确，比例映射到 672 宽）
    sx = W / strip.shape[1]
    sy = STRIP_H / strip.shape[0]
    strip_boxes = []
    for i, it in enumerate(icons):
        b = (it["x1"] * sx, bg_h + it["y1"] * sy, it["x2"] * sx, bg_h + it["y2"] * sy)
        # 轻微内缩，避免把提示条边界算进去
        b = (max(0, b[0] - 1), max(bg_h, b[1] - 1), min(W, b[2] + 1), min(H, b[3] + 1))
        strip_boxes.append({"idx": i, "box": b})
        labels.append(yolo_line(2, b, W, H))

    # class 0：底图里定位每个提示图标
    used = []
    detail = []
    for i, it in enumerate(icons):
        # 用原始提示图标（未缩放）当模板，scale 覆盖底图侧常见的放大
        cands = locate_icon(bg_edge, canvas.shape, it["img"], angles, scales)
        # 排除落在提示条区域的候选
        cands = [c for c in cands if c["box"][3] < bg_h]
        pick = None
        for c in cands:
            if all(iou(c["box"], u) < 0.3 for u in used):
                pick = c
                break
        if pick is None and cands:
            pick = cands[0]
        if pick:
            used.append(pick["box"])
            labels.append(yolo_line(0, pick["box"], W, H))
            detail.append({"icon": i, "score": round(pick["score"], 4),
                           "angle": pick["angle"], "scale": pick["scale"],
                           "box": [round(v, 1) for v in pick["box"]],
                           "top3": [{"score": round(c["score"], 4), "angle": c["angle"],
                                     "scale": round(c["scale"], 2)} for c in cands[:3]]})
        else:
            detail.append({"icon": i, "score": 0.0, "box": None})

    scores = [d["score"] for d in detail if d["box"]]
    min_score = min(scores) if scores else 0.0
    need_review = (len(detail) < len(icons)) or (min_score < review_thr)

    return {
        "sid": meta["sid"], "canvas": canvas, "labels": labels, "detail": detail,
        "n_prompt": len(icons), "n_found": len([d for d in detail if d["box"]]),
        "min_score": round(min_score, 4), "need_review": need_review,
        "W": W, "H": H, "subcapclass": meta.get("subcapclass"),
        "instruction": meta.get("instruction"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--out", default="data/labeled")
    ap.add_argument("--review-thr", type=float, default=0.45,
                    help="最低匹配分低于此值则进人工复核队列")
    ap.add_argument("--angle-step", type=int, default=6)
    ap.add_argument("--scale-min", type=float, default=1.2)
    ap.add_argument("--scale-max", type=float, default=3.0)
    ap.add_argument("--scale-step", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draw", action="store_true", help="输出可视化，便于人工复核")
    args = ap.parse_args()

    import random
    random.seed(args.seed)

    dirs = sorted(d for d in
                  (os.path.join(args.samples, x) for x in os.listdir(args.samples))
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta.json")))
    if not dirs:
        raise SystemExit(f"没有找到样本目录: {args.samples}")
    print(f"[scan] {len(dirs)} 个样本")

    angles = list(range(0, 360, args.angle_step))
    scales = [round(s, 2) for s in np.arange(args.scale_min, args.scale_max + 1e-6, args.scale_step)]

    for sub in ["images/train", "images/val", "labels/train", "labels/val", "review"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    rows, review = [], []
    t0 = time.time()
    for i, d in enumerate(dirs):
        try:
            r = process_sample(d, angles, scales, args.review_thr)
        except Exception as e:
            print(f"  [FAIL] {os.path.basename(d)}: {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        split = "val" if random.random() < args.val_ratio else "train"
        base = r["sid"]
        img_p = os.path.join(args.out, "images", split, base + ".jpg")
        lab_p = os.path.join(args.out, "labels", split, base + ".txt")
        imwrite_u(img_p, r["canvas"])
        with open(lab_p, "w", encoding="utf-8") as f:
            f.write("\n".join(r["labels"]) + "\n")

        if args.draw or r["need_review"]:
            vis = r["canvas"].copy()
            for ln in r["labels"]:
                c, cx, cy, bw, bh = ln.split()
                x1 = int((float(cx) - float(bw) / 2) * r["W"])
                y1 = int((float(cy) - float(bh) / 2) * r["H"])
                x2 = int((float(cx) + float(bw) / 2) * r["W"])
                y2 = int((float(cy) + float(bh) / 2) * r["H"])
                col = (0, 0, 255) if c == "0" else (0, 180, 0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                cv2.putText(vis, c, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col, 1, cv2.LINE_AA)
            vp = os.path.join(args.out, "review", base + (".jpg" if r["need_review"] else "_vis.jpg"))
            imwrite_u(vp, vis)
            if r["need_review"]:
                review.append({"sid": base, "min_score": r["min_score"],
                               "n_prompt": r["n_prompt"], "n_found": r["n_found"],
                               "image": os.path.relpath(img_p, args.out).replace("\\", "/"),
                               "vis": os.path.relpath(vp, args.out).replace("\\", "/")})

        row = {"sid": base, "split": split, "n_prompt": r["n_prompt"], "n_found": r["n_found"],
               "min_score": r["min_score"], "need_review": r["need_review"],
               "detail": r["detail"], "image": os.path.relpath(img_p, args.out).replace("\\", "/"),
               "label": os.path.relpath(lab_p, args.out).replace("\\", "/")}
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(dirs)}] 已标注 {len(rows)}  需复核 {len(review)}")

    with open(os.path.join(args.out, "labels_index.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "review_queue.jsonl"), "w", encoding="utf-8") as f:
        for r in review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_tr = sum(1 for r in rows if r["split"] == "train")
    print("\n=== 标注完成 ===")
    print(f"  样本总数   : {len(rows)}  (train={n_tr}, val={len(rows)-n_tr})")
    print(f"  需人工复核 : {len(review)}  ({len(review)/max(len(rows),1)*100:.1f}%)")
    if rows:
        ms = [r["min_score"] for r in rows]
        print(f"  匹配分     : min={min(ms):.3f} mean={sum(ms)/len(ms):.3f} max={max(ms):.3f}")
        full = sum(1 for r in rows if r["n_found"] == r["n_prompt"])
        print(f"  图标全定位 : {full}/{len(rows)} ({full/len(rows)*100:.1f}%)")
    print(f"  输出       : {args.out}/")
    print(f"  复核队列   : {os.path.join(args.out, 'review_queue.jsonl')}")
    print(f"  耗时       : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
