# -*- coding: utf-8 -*-
"""make_plates: 从采集样本生成「干净背景底板」—— 把图标抠掉，留纯照片。

为什么需要：
    采集到的 bg.jpg 里已经有 5 个图标了。如果直接拿它当合成底板，
    就会和后来贴上去的图标混在一起，标签必然错。
    用 inpaint 把原图标抹掉 -> 得到无图标的纯照片底板，
    之后合成出来的每一张图，图标位置都是我们自己的 -> 标签 100% 准确。

产物：
    data/plates/<sid>.jpg       干净底板
    data/plates_index.jsonl     底板索引

用法：
    python tools/make_plates.py --samples data/samples --out data/plates
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icon_match import ink_mask, _merge_boxes  # noqa: E402


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img, q=95):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        buf.tofile(p)
    return ok


def icon_mask_for_inpaint(bgr, dilate=9):
    """给 inpaint 用的掩码：图标墨迹 + 膨胀（必须完全覆盖，否则留残影）。"""
    m, _ = ink_mask(bgr)
    # 连笔画
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    # 过滤掉过小的噪声块（保留图标级连通域）
    cnts, labels = cv2.connectedComponents(m, 8)[:2]
    # connectedComponents 返回 (n, labels)；用 stats 版本
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 120:
            keep[lab == i] = 255
    # 膨胀覆盖描边外缘与抗锯齿
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    return cv2.dilate(keep, k, iterations=1)


def make_plate(bg, dilate=9, radius=5):
    mask = icon_mask_for_inpaint(bg, dilate=dilate)
    if mask.sum() == 0:
        return bg, mask
    # INPAINT_TELEA 对细长结构比 NS 更自然
    out = cv2.inpaint(bg, mask, radius, cv2.INPAINT_TELEA)
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--out", default="data/plates")
    ap.add_argument("--dilate", type=int, default=9)
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--draw", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.draw:
        os.makedirs(os.path.join(args.out, "_vis"), exist_ok=True)

    metas = sorted(glob.glob(os.path.join(args.samples, "*", "meta.json")))
    index = os.path.join(args.out, "plates_index.jsonl")
    done = set()
    if os.path.exists(index):
        for line in open(index, encoding="utf-8"):
            try:
                done.add(json.loads(line)["sid"])
            except Exception:
                pass

    n_ok = 0
    for mp in metas:
        d = os.path.dirname(mp)
        sid = os.path.basename(d)
        if sid in done:
            continue
        bg = imread_u(os.path.join(d, "bg.jpg"))
        if bg is None:
            continue
        plate, mask = make_plate(bg, dilate=args.dilate, radius=args.radius)
        fn = f"{sid}.jpg"
        imwrite_u(os.path.join(args.out, fn), plate)
        cover = float(mask.mean() / 255.0)
        rec = {"sid": sid, "file": fn, "w": plate.shape[1], "h": plate.shape[0],
               "mask_cover": round(cover, 4)}
        with open(index, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_ok += 1
        if args.draw:
            vis = np.concatenate([bg, np.full((bg.shape[0], 8, 3), 255, np.uint8), plate], axis=1)
            imwrite_u(os.path.join(args.out, "_vis", f"{sid}.jpg"), vis)
        print(f"  {sid}: {plate.shape[1]}x{plate.shape[0]} 覆盖={cover*100:.1f}%")
    print(f"\n[done] 新生成底板 {n_ok} 张 -> {args.out}/")


if __name__ == "__main__":
    main()
