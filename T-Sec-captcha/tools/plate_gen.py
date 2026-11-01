# -*- coding: utf-8 -*-
"""plate_gen: 生成干净背景底板 v2 —— 用「暗度阈值 + 线稿几何判据」精确抠掉图标。

v1 失败原因：blackhat 会把照片纹理（沙纹/草叶）当成笔画，掩码覆盖 37~99%，
inpaint 后整张糊掉。

v2 策略（对「黑线稿图标压在照片上」这个具体题型最优且简单）：
    1) 暗度阈值 gray < T (默认 60) —— 图标是近黑线稿，照片极少有纯黑
    2) 连通域过滤：面积 >= 25，短边 >= 5，填充率 < 0.80（排除实心暗块）
    3) 邻近合并（gap=14）→ 图标级框
    4) 只保留整体尺寸合理（短边 >= 22）的框；**并要求框内墨迹颜色一致性**
    5) inpaint 半径小(3)，避免大面积糊

用法：
    python tools/plate_gen.py --samples data/samples --out data/plates --draw
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img, q=95):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        buf.tofile(p)
    return ok


def merge_boxes(boxes, gap=14):
    out = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if (a[0] - gap < b[2] and b[0] < a[2] + gap and
                        a[1] - gap < b[3] and b[1] < a[3] + gap):
                    out[i] = [min(a[0], b[0]), min(a[1], b[1]),
                              max(a[2], b[2]), max(a[3], b[3])]
                    out.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in out]


def icon_mask(bgr, thr=60, min_area=25, min_side=5, max_fill=0.80,
              min_icon_side=20, gap=14, dilate=3, keep_border=False):
    """返回 (掩码, 图标框列表, 统计)。"""
    H, W = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = ((g < thr) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        if min(w, h) < min_side:
            continue
        if w > W * 0.5 or h > H * 0.5:
            continue
        if a / float(w * h) > max_fill:
            continue
        if not keep_border and (x <= 0 or y <= 0 or x + w >= W or y + h >= H):
            continue
        boxes.append((x, y, x + w, y + h))

    merged = merge_boxes(boxes, gap=gap)
    icons = [b for b in merged if (b[2] - b[0]) >= min_icon_side and (b[3] - b[1]) >= min_icon_side]

    keep = np.zeros_like(m)
    for (x1, y1, x2, y2) in icons:
        keep[y1:y2, x1:x2] = m[y1:y2, x1:x2]
    if dilate > 0 and keep.sum() > 0:
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))

    st = {"thr": thr, "components": len(boxes), "icons": len(icons),
          "cover": round(float((keep > 0).mean()), 4),
          "boxes": [list(b) for b in icons]}
    return keep, icons, st


def plate_mask_from_boxes(shape, icons, dilate=5):
    """用图标框直接构造掩码：框内全部抹掉（比只抹墨迹干净，不留鬼影）。"""
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    for (x1, y1, x2, y2) in icons:
        m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 255
    if dilate > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    return m


def make_plate(bgr, radius=3, box_fill=True, box_dilate=5, **kw):
    """box_fill=True: 用图标框整块填充抹除（推荐，不留鬼影）。
    box_fill=False: 只抹墨迹像素（保留框内照片细节，但可能残留抗锯齿边）。"""
    mask, icons, st = icon_mask(bgr, **kw)
    if mask.sum() == 0:
        return bgr, mask, st
    if box_fill:
        mask = plate_mask_from_boxes(bgr.shape, icons, dilate=box_dilate)
        st["cover"] = round(float((mask > 0).mean()), 4)
        st["mode"] = "box_fill"
    else:
        st["mode"] = "ink_only"
    plate = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    return plate, mask, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--out", default="data/plates")
    ap.add_argument("--thr", type=int, default=60)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.draw:
        os.makedirs(os.path.join(args.out, "_vis"), exist_ok=True)

    index = os.path.join(args.out, "plates_index.jsonl")
    done = set()
    if not args.force and os.path.exists(index):
        for line in open(index, encoding="utf-8"):
            try:
                done.add(json.loads(line)["sid"])
            except Exception:
                pass
    if args.force:
        open(index, "w").close()

    metas = sorted(glob.glob(os.path.join(args.samples, "*", "meta.json")))
    n_ok = covers = 0
    ni = []
    for mp in metas:
        d = os.path.dirname(mp)
        sid = os.path.basename(d)
        if sid in done:
            continue
        bg = imread_u(os.path.join(d, "bg.jpg"))
        if bg is None:
            continue
        plate, mask, st = make_plate(bg, radius=args.radius, thr=args.thr,
                                     dilate=args.dilate)
        imwrite_u(os.path.join(args.out, f"{sid}.jpg"), plate)
        with open(index, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sid": sid, "file": f"{sid}.jpg",
                                "w": plate.shape[1], "h": plate.shape[0],
                                "icons": st["icons"], "cover": st["cover"]},
                               ensure_ascii=False) + "\n")
        n_ok += 1
        covers += st["cover"]
        ni.append(st["icons"])
        if args.draw:
            vis = np.concatenate([bg, np.full((bg.shape[0], 6, 3), 255, np.uint8), plate], axis=1)
            for (x1, y1, x2, y2) in st["boxes"]:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            imwrite_u(os.path.join(args.out, "_vis", f"{sid}.jpg"), vis)

    print(f"[done] 底板 {n_ok} 张 -> {args.out}/")
    if ni:
        print(f"  检出图标: mean={np.mean(ni):.1f} min={min(ni)} max={max(ni)}")
        print(f"  掩码覆盖: mean={covers/len(ni)*100:.2f}% max={max(ni) and 0 or 0}")
    if args.draw:
        print(f"  对比图  : {args.out}/_vis/ (左原图 右底板)")


if __name__ == "__main__":
    main()
