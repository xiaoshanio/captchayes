# -*- coding: utf-8 -*-
"""autolabel_2408: 针对天御 2408（图标点选）的自动标注器。

与 tools/autolabel.py 的区别（为什么不直接复用）：
    autolabel.py 把 170x50 的提示图 **横向拉伸** 到 672 宽（3.95 倍），
    图标笔画被插值糊掉，class2 框和实际字形都对不上，模型学不到东西。
    本脚本改为 **等比放置**：提示图按原像素尺寸贴到提示条左侧并垂直居中，
    其余区域填白 —— 与线上真实渲染一致（天御弹层也是原尺寸渲染提示图）。

标注来源（成本结构最优）：
    class 2（提示图标）: 提示图按列投影切分 → 100% 精确，无需人工
    class 0（底图图标）: blackhat 分割 + 形状归一匹配 → 有分数，低分进人工复核

产出（与 train_detector.py 期望的 YOLO 目录一致）：
    <out>/images/{train,val}/<sid>.jpg
    <out>/labels/{train,val}/<sid>.txt      cls cx cy w h (归一化)
    <out>/data.yaml
    <out>/labels_index.jsonl                每题明细（含配对分、margin）
    <out>/review_queue.jsonl                需人工复核的题
    <out>/review/<sid>.jpg                  可视化（框 + 点击序号）

用法：
    python tools/autolabel_2408.py --samples data/samples --out data/labeled2408
    python tools/autolabel_2408.py --samples data/samples --out data/labeled2408 \
        --review-thr 0.42 --draw
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from seg_2408 import segment_icons_auto                 # noqa: E402
from match_2408 import match_prompt_to_boxes, assign as match_assign  # noqa: E402

CLASS_TARGET = 0   # 底图里可点的图标（含干扰项）
CLASS_TITLE = 1    # 标题（2408 无此元素，保留位）
CLASS_CHAR = 2     # 提示条里的图标（从左到右 = 点击顺序）

STRIP_H = 56       # 提示条高度（与 synth_gen / autolabel 保持一致）


# ------------------------------------------------------------------ IO
def imread_u(p, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), flags)


def imwrite_u(p, img, q=95):
    ext = os.path.splitext(p)[1].lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), q] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(p)
    return ok


# ------------------------------------------------------------------ 提示图切分
def split_prompt_icons(strip, ink_thr=150, min_w=6, gap=3, pad=1):
    """列投影切开提示图 → [{img, box}]，从左到右即点击顺序。"""
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
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

    H, W = strip.shape[:2]
    out = []
    for a, b in merged:
        if b - a < min_w:
            continue
        sub = ink[:, a:b]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        x1, x2 = max(0, a - pad), min(W, b + pad)
        y1, y2 = max(0, int(rows[0]) - pad), min(H, int(rows[-1]) + 1 + pad)
        out.append({"img": strip[y1:y2, x1:x2].copy(),
                    "box": [int(x1), int(y1), int(x2), int(y2)]})
    return out


# ------------------------------------------------------------------ 版式合成
def compose(bg, prompt, strip_h=STRIP_H, align="left", margin=6):
    """底图 + 提示条（等比放置，不拉伸）→ (canvas, bg_h, prompt_offset_x)。

    为什么等比：线上天御弹层里提示图是原尺寸渲染的，
    横向拉伸会改变字形宽高比，与推理时的真实输入分布不一致。
    """
    h, w = bg.shape[:2]
    canvas = np.full((h + strip_h, w, 3), 255, np.uint8)
    canvas[:h, :w] = bg

    ph, pw = prompt.shape[:2]
    # 竖直居中
    ty = h + max(0, (strip_h - ph) // 2)
    if align == "left":
        tx = margin
    elif align == "center":
        tx = max(margin, (w - pw) // 2)
    else:
        tx = max(margin, w - pw - margin)
    tx2, ty2 = min(w, tx + pw), min(h + strip_h, ty + ph)
    canvas[ty:ty2, tx:tx2] = prompt[:ty2 - ty, :tx2 - tx]
    return canvas, h, tx


def yolo_line(cls, box, W, H):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H
    bw, bh = (x2 - x1) / W, (y2 - y1) / H
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ------------------------------------------------------------------ 单题处理
def process_one(sdir, review_thr, strip_h=STRIP_H, align="left",
                min_side=22, min_ink=250):
    meta = json.load(open(os.path.join(sdir, "meta.json"), encoding="utf-8"))
    sid = meta.get("sid") or os.path.basename(sdir)
    bg = imread_u(os.path.join(sdir, "bg.jpg"))
    prompts = meta.get("prompts") or []
    if bg is None or not prompts:
        return None
    prompt = imread_u(os.path.join(sdir, prompts[0]["file"]))
    if prompt is None:
        return None

    icons = split_prompt_icons(prompt)
    if not icons:
        return None

    canvas, bg_h, off_x = compose(bg, prompt, strip_h, align)
    H, W = canvas.shape[:2]

    # ---- class 0: 底图图标（2408 专用分割，自动判纯黑/彩色子域）
    seg, _dbg = segment_icons_auto(bg, min_side=min_side, min_ink=min_ink,
                                   min_expected=max(4, len(icons) + 1))
    seg = [s for s in seg if s["box"][3] <= bg_h]
    bg_boxes = [s["box"] for s in seg]

    # ---- 提示图标 ↔ 底图候选：多角度多尺度软墨迹模板匹配
    scores, _mmeta = match_prompt_to_boxes(icons, bg, bg_boxes)
    pairs = match_assign(scores)

    labels = []
    detail = []
    # class 2: 提示条图标（精确框，位置 = 提示图内框 + 偏移）
    for i, it in enumerate(icons):
        x1, y1, x2, y2 = it["box"]
        box = (off_x + x1, bg_h + y1, off_x + x2, bg_h + y2)
        labels.append(yolo_line(CLASS_CHAR, box, W, H))
        detail.append({"icon": i, "prompt_box": [int(v) for v in box]})

    # class 0: 底图全部图标（含干扰项）
    for j, box in enumerate(bg_boxes):
        labels.append(yolo_line(CLASS_TARGET, box, W, H))

    # 配对质量 → 复核判据
    pair_scores = [s for _i, _j, s in pairs]
    min_pair = min(pair_scores) if pair_scores else 0.0
    margins = []
    if scores.size:
        for i in range(scores.shape[0]):
            row = np.sort(scores[i])[::-1]
            margins.append(float(row[0] - row[1]) if len(row) > 1 else float(row[0]))
    margin_min = min(margins) if margins else 0.0
    complete = len(pairs) == len(icons)
    need_review = (not complete) or (min_pair < review_thr)

    return {
        "sid": sid,
        "canvas": canvas,
        "labels": labels,
        "W": W, "H": H, "bg_h": bg_h,
        "n_prompt": len(icons),
        "n_bg_icons": len(bg_boxes),
        "pairs": [{"prompt": int(i), "bg": int(j), "score": round(float(s), 4)}
                  for i, j, s in pairs],
        "bg_boxes": [[int(v) for v in b] for b in bg_boxes],
        "min_pair": round(float(min_pair), 4),
        "margin_min": round(float(margin_min), 4),
        "need_review": bool(need_review),
        "subcapclass": meta.get("subcapclass"),
        "instruction": meta.get("instruction"),
        "detail": detail,
    }


def draw(r):
    vis = r["canvas"].copy()
    # class 0 灰框
    for box in r["bg_boxes"]:
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (170, 170, 170), 1)
    # class 2 绿框
    for d in r["detail"]:
        b = d["prompt_box"]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 170, 0), 2)
    # 点击序号
    for rank, p in enumerate(r["pairs"]):
        b = r["bg_boxes"][p["bg"]]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cv2.putText(vis, f"{rank+1}", (b[0] + 2, max(14, b[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--out", default="data/labeled2408")
    ap.add_argument("--review-thr", type=float, default=0.42,
                    help="最低配对分低于此值进人工复核队列")
    ap.add_argument("--strip-h", type=int, default=STRIP_H)
    ap.add_argument("--align", choices=["left", "center", "right"], default="left")
    ap.add_argument("--min-side", type=int, default=20)
    ap.add_argument("--min-ink", type=int, default=120)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)

    dirs = sorted(d for d in glob.glob(os.path.join(
        args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples),
        "*")) if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta.json")))
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        raise SystemExit(f"没有样本: {args.samples}")
    print(f"[scan] {len(dirs)} 个样本")

    for sub in ("images/train", "images/val", "labels/train", "labels/val", "review"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    rows, review = [], []
    t0 = time.time()
    fail = 0
    for i, d in enumerate(dirs):
        try:
            r = process_one(d, args.review_thr, args.strip_h, args.align,
                            args.min_side, args.min_ink)
        except Exception as e:
            fail += 1
            print(f"  [FAIL] {os.path.basename(d)}: {type(e).__name__}: {e}")
            continue
        if r is None:
            fail += 1
            continue

        split = "val" if random.random() < args.val_ratio else "train"
        img_p = os.path.join(out, "images", split, r["sid"] + ".jpg")
        lab_p = os.path.join(out, "labels", split, r["sid"] + ".txt")
        imwrite_u(img_p, r["canvas"])
        with open(lab_p, "w", encoding="utf-8") as f:
            f.write("\n".join(r["labels"]) + "\n")

        if args.draw or r["need_review"]:
            imwrite_u(os.path.join(out, "review", r["sid"] + ".jpg"), draw(r))
        if r["need_review"]:
            review.append({"sid": r["sid"], "min_pair": r["min_pair"],
                           "margin_min": r["margin_min"],
                           "n_prompt": r["n_prompt"],
                           "n_bg_icons": r["n_bg_icons"],
                           "pairs": r["pairs"],
                           "image": os.path.relpath(img_p, out).replace("\\", "/"),
                           "vis": f"review/{r['sid']}.jpg"})

        rows.append({"sid": r["sid"], "split": split,
                     "n_prompt": r["n_prompt"], "n_bg_icons": r["n_bg_icons"],
                     "min_pair": r["min_pair"], "margin_min": r["margin_min"],
                     "need_review": r["need_review"], "pairs": r["pairs"],
                     "bg_boxes": r["bg_boxes"],
                     "subcapclass": r["subcapclass"],
                     "image": os.path.relpath(img_p, out).replace("\\", "/"),
                     "label": os.path.relpath(lab_p, out).replace("\\", "/")})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(dirs)}] 已标 {len(rows)}  复核 {len(review)}")

    with open(os.path.join(out, "labels_index.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "review_queue.jsonl"), "w", encoding="utf-8") as f:
        for r in review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # data.yaml（train_detector.py 读它）
    import yaml
    cfg = {
        "path": out.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "label", 1: "title", 2: "char"},
    }
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    n_tr = sum(1 for r in rows if r["split"] == "train")
    print("\n=== 标注完成 ===")
    print(f"  样本总数   : {len(rows)}  (train={n_tr}, val={len(rows)-n_tr})")
    print(f"  失败       : {fail}")
    print(f"  需人工复核 : {len(review)} "
          f"({len(review)/max(len(rows),1)*100:.1f}%)")
    if rows:
        mp = [r["min_pair"] for r in rows]
        np_ = [r["n_prompt"] for r in rows]
        nb = [r["n_bg_icons"] for r in rows]
        full = sum(1 for r in rows if len(r["pairs"]) == r["n_prompt"])
        print(f"  配对分     : min={min(mp):.3f} "
              f"mean={sum(mp)/len(mp):.3f} max={max(mp):.3f}")
        print(f"  提示图标数 : mean={sum(np_)/len(np_):.2f} "
              f"min={min(np_)} max={max(np_)}")
        print(f"  底图图标数 : mean={sum(nb)/len(nb):.2f} "
              f"min={min(nb)} max={max(nb)}")
        print(f"  全配对成功 : {full}/{len(rows)} "
              f"({full/len(rows)*100:.1f}%)")
    print(f"  输出       : {out}/")
    print(f"  data.yaml  : {os.path.join(out, 'data.yaml')}")
    print(f"  耗时       : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
