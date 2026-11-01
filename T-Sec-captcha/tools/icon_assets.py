# -*- coding: utf-8 -*-
"""icon_assets: 图标资产库构建 —— 从提示精灵图自动提取「单图标素材」（带精确标签）。

为什么这是整个流水线的支点：
    提示精灵图里的每个图标都是**独立裁切的干净素材**（无背景、笔画完整）。
    按列投影切开后，我们免费得到「一张图标图 + 它在提示条里的精确框」。
    有了素材库 + 随机合成器，就能无限生成**标签 100% 准确**的训练数据，
    彻底绕开「人工标注底图图标」这个成本大头。

产出：
    data/assets/icons/<hash>.png      单图标素材（RGBA，透明底，正立）
    data/assets/icons_index.jsonl     素材索引（hash、尺寸、墨迹像素、颜色、来源）

用法：
    python tools/icon_assets.py --samples data/samples --out data/assets
    python tools/icon_assets.py --samples data/samples --out data/assets --report
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import cv2
import numpy as np


def imread_u(p, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), flags)


def imwrite_u(p, img):
    ext = os.path.splitext(p)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(p)
    return ok


def split_prompt_icons(strip_bgr, ink_thr=150, min_w=7, gap=3, pad=1):
    """列投影切开提示条 → [{img, box}]，从左到右即点击顺序。"""
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
    H, W = strip_bgr.shape[:2]
    for a, b in merged:
        if b - a < min_w:
            continue
        sub = ink[:, a:b]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        x1 = max(0, a - pad)
        x2 = min(W, b + pad)
        y1 = max(0, int(rows[0]) - pad)
        y2 = min(H, int(rows[-1]) + 1 + pad)
        out.append({"img": strip_bgr[y1:y2, x1:x2].copy(),
                    "box": [int(x1), int(y1), int(x2), int(y2)]})
    return out


def to_sprite(bgr, trim=True):
    """BGR 图标 -> RGBA 透明底素材（用 Otsu 抠墨迹，其余透明）。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    _, alpha = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 去掉孤立噪点
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    if cv2.countNonZero(alpha) < 20:
        return None
    if trim:
        ys, xs = np.nonzero(alpha)
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        alpha = alpha[y1:y2, x1:x2]
        bgr = bgr[y1:y2, x1:x2]
    b, g_, r = cv2.split(bgr)
    return cv2.merge([b, g_, r, alpha])


def dominant_color(rgba):
    """素材的主色（图标通常是单色线稿），用于合成时做颜色扰动。"""
    a = rgba[:, :, 3]
    m = a > 128
    if m.sum() == 0:
        return [0, 0, 0]
    px = rgba[:, :, :3][m].astype(np.float32)
    med = np.median(px, axis=0)
    return [int(v) for v in med]


def build(samples_dir, out_dir, min_side=8, max_side=120):
    icons_dir = os.path.join(out_dir, "icons")
    os.makedirs(icons_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "icons_index.jsonl")

    seen = set()
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["hash"])
                except Exception:
                    pass

    metas = sorted(glob.glob(os.path.join(samples_dir, "*", "meta.json")))
    print(f"[scan] {len(metas)} 个样本")
    added = kept = skipped = 0
    stats = {}

    for mp in metas:
        d = os.path.dirname(mp)
        meta = json.load(open(mp, encoding="utf-8"))
        for pr in meta.get("prompts", []):
            p = os.path.join(d, pr["file"])
            if not os.path.exists(p):
                continue
            strip = imread_u(p)
            if strip is None:
                continue
            icons = split_prompt_icons(strip)
            for k, it in enumerate(icons):
                h, w = it["img"].shape[:2]
                if min(h, w) < min_side or max(h, w) > max_side:
                    skipped += 1
                    continue
                sprite = to_sprite(it["img"])
                if sprite is None:
                    skipped += 1
                    continue
                key = hashlib.sha1(np.ascontiguousarray(sprite).tobytes()).hexdigest()[:16]
                if key in seen:
                    kept += 1
                    continue
                seen.add(key)
                fn = f"{key}.png"
                imwrite_u(os.path.join(icons_dir, fn), sprite)
                col = dominant_color(sprite)
                ink = int((sprite[:, :, 3] > 128).sum())
                rec = {"hash": key, "file": fn, "w": sprite.shape[1], "h": sprite.shape[0],
                       "ink": ink, "color_bgr": col,
                       "src": os.path.basename(d), "idx_in_prompt": k,
                       "prompt_size": [strip.shape[1], strip.shape[0]]}
                with open(index_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                added += 1
                stats[sprite.shape] = stats.get(sprite.shape, 0) + 1
    print(f"[done] 新增素材 {added}，重复跳过 {kept}，过滤 {skipped}")
    print(f"[size] 尺寸分布 top10: "
          f"{sorted(stats.items(), key=lambda kv: -kv[1])[:10]}")
    print(f"[path] {icons_dir}/  (索引: {index_path})")
    return added


def report(out_dir):
    index_path = os.path.join(out_dir, "icons_index.jsonl")
    if not os.path.exists(index_path):
        print("没有索引")
        return
    recs = [json.loads(l) for l in open(index_path, encoding="utf-8")]
    print(f"素材总数: {len(recs)}")
    import collections
    ws = collections.Counter(r["w"] for r in recs)
    hs = collections.Counter(r["h"] for r in recs)
    ar = collections.Counter(round(r["w"] / max(r["h"], 1), 1) for r in recs)
    print("宽度分布:", sorted(ws.items())[:12])
    print("高度分布:", sorted(hs.items())[:12])
    print("宽高比分布:", ar.most_common(10))
    ink = [r["ink"] for r in recs]
    print(f"墨迹像素: min={min(ink)} mean={sum(ink)//len(ink)} max={max(ink)}")
    srcs = collections.Counter(r["src"] for r in recs)
    print(f"来源样本数: {len(srcs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--out", default="data/assets")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(args.out)
    else:
        build(args.samples, args.out)
