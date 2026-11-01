# -*- coding: utf-8 -*-
"""match_2408: 2408 专用图标匹配器（替代 icon_match.shape_score）。

为什么换掉 shape_score：
    实测 50 题里，3/3 全配只有 56%，配对分普遍落在 0.20~0.45，
    大量真配对被打成 0.2x。根因是 shape_score 的「面积归一 + 主轴对齐」
    对**非对称图标**（如云朵、猫头、房形）会把不同图标归一到很像的形状，
    而同一图标旋转后主轴又可能落到 180° 之外的分支上。

本模块的做法（利用「框已给出」这个强先验）：
    A. 提示图图标本身就是**正立原图**，不需要归一
    B. 底图候选框已由 seg_2408 给出，尺寸已知
    C. 于是匹配退化成「小范围旋转 + 小范围缩放」的模板匹配：
         for angle in ±180 step 10
           for scale in [0.9, 1.0, 1.1]
             在候选框内（外扩 12px）做 TM_CCOEFF_NORMED（掩码版）
       取最大分
    D. 再用「墨迹 IoU」交叉校验，两者加权

成本：单对 ~ (36 角度 × 3 尺度) 次小图匹配 ≈ 3~6 ms，
      3 提示 × 5 候选 = 15 对 ≈ 50~90 ms/题，可接受（离线标注）。

用法（库）：
    from match_2408 import match_prompt_to_boxes
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def _ink(bgr, thr=None):
    """二值墨迹（图标是纯黑线稿）。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    if thr is None:
        thr = 100
    return (g < thr).astype(np.uint8) * 255


def _soft_ink(bgr, lo=40, hi=140):
    """软墨迹：把「黑→白」映射成连续置信度，抗 JPEG 振铃与抗锯齿。

    为什么需要软墨迹：底图是 JPEG 压缩的摄影图，图标边缘有振铃，
    硬阈值会让同一图标的边缘在提示图与底图里厚度不同，
    TM_CCOEFF_NORMED 对厚度差很敏感。软墨迹把这个差异摊平。
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    g = g.astype(np.float32)
    v = np.clip((hi - g) / float(hi - lo), 0.0, 1.0)
    return (v * 255).astype(np.uint8)


def _rot(img, angle):
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _best_in_box(tpl_ink, box_ink, angles, scales, pad=14):
    """在候选框（外扩 pad）内做多角度多尺度掩码模板匹配，返回最高分。"""
    H, W = box_ink.shape[:2]
    # 候选区域外扩
    y1 = max(0, -pad)
    region = np.zeros((H + 2 * pad, W + 2 * pad), np.uint8)
    region[pad:pad + H, pad:pad + W] = box_ink
    if region.sum() == 0:
        return 0.0, 0.0, 0.0

    best = 0.0
    best_a, best_s = 0.0, 1.0
    for ang in angles:
        t = _rot(tpl_ink, ang) if ang else tpl_ink
        for sc in scales:
            tt = (cv2.resize(t, None, fx=sc, fy=sc,
                             interpolation=cv2.INTER_NEAREST)
                  if sc != 1.0 else t)
            th, tw = tt.shape[:2]
            if th > region.shape[0] or tw > region.shape[1]:
                continue
            if th < 8 or tw < 8:
                continue
            try:
                res = cv2.matchTemplate(region, tt, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            v = float(res.max())
            if v > best:
                best, best_a, best_s = v, float(ang), float(sc)
    return best, best_a, best_s


def _iou_at_best(tpl_ink, box_ink, angle, scale, pad=14):
    """在最优角度/尺度下，按「居中放置」算墨迹 IoU（对 TM 分数的交叉校验）。"""
    t = _rot(tpl_ink, angle) if angle else tpl_ink
    if scale != 1.0:
        t = cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    H, W = box_ink.shape[:2]
    canvas = np.zeros((H, W), np.uint8)
    th, tw = t.shape[:2]
    oy, ox = (H - th) // 2, (W - tw) // 2
    ys, ye = max(0, oy), min(H, oy + th)
    xs, xe = max(0, ox), min(W, ox + tw)
    if ye <= ys or xe <= xs:
        return 0.0
    canvas[ys:ye, xs:xe] = t[ys - oy:ye - oy, xs - ox:xe - ox]
    inter = np.logical_and(canvas > 0, box_ink > 0).sum()
    union = np.logical_or(canvas > 0, box_ink > 0).sum()
    return float(inter / union) if union else 0.0


def match_prompt_to_boxes(prompt_icons, bg, boxes, angle_step=10,
                          scales=None, pad=14, w_tm=0.75, w_iou=0.25,
                          verbose=False, scale_jitter=(0.55, 0.73, 0.91,
                                                       1.09, 1.27, 1.46,
                                                       1.64, 1.82, 2.0),
                          soft=True):
    """返回 (scores[N,M], meta[N][M]) 。

    scores[i][j] = 第 i 个提示图标 与 第 j 个底图候选框 的相似度 ∈ [0,1]

    【关键修正 1】尺度不再用固定列表，而是由**尺寸比**推出：
        提示图标在 170x50 精灵图里约 36~44 px，
        底图图标由天御按 ~67~77 px 渲染，真实尺度比 ≈ 1.6~2.0。
        固定 scales=(0.85..1.2) 覆盖不到，实测会把「电池」错配成「房子」。

    【关键修正 2】软墨迹（soft=True）：
        底图是 JPEG 摄影图，图标边缘有振铃/抗锯齿，硬阈值二值化后
        同一图标的笔画厚度在提示图与底图里不一致，TM_CCOEFF_NORMED 很敏感。
        改用连续置信度后实测：3/3 全配 60% → 75%，平均配率 85% → 92%。
    """
    ink_fn = _soft_ink if soft else _ink
    N, M = len(prompt_icons), len(boxes)
    scores = np.zeros((N, M), np.float32)
    meta = [[None] * M for _ in range(N)]
    angles = list(range(-180, 180, angle_step))

    box_inks = []
    for (x1, y1, x2, y2) in boxes:
        box_inks.append(ink_fn(bg[y1:y2, x1:x2]))

    for i, it in enumerate(prompt_icons):
        tpl = ink_fn(it["img"])
        if tpl.sum() == 0:
            continue
        ys, xs = np.nonzero(tpl > 30)
        if len(ys) == 0:
            continue
        tpl_h = max(int(ys.max() - ys.min() + 1), 1)
        tpl_w = max(int(xs.max() - xs.min() + 1), 1)
        tpl_short = min(tpl_h, tpl_w)

        for j in range(M):
            if box_inks[j].sum() == 0:
                continue
            bys, bxs = np.nonzero(box_inks[j] > 30)
            if len(bys) == 0:
                continue
            bh = max(int(bys.max() - bys.min() + 1), 1)
            bw = max(int(bxs.max() - bxs.min() + 1), 1)
            ratio = min(bh, bw) / float(tpl_short)

            if scales is None:
                sc_list = [max(0.5, min(3.0, ratio * k)) for k in scale_jitter]
            else:
                sc_list = list(scales)

            tm, ang, sc = _best_in_box(tpl, box_inks[j], angles, sc_list, pad)
            iou = _iou_at_best(tpl, box_inks[j], ang, sc, pad)
            s = w_tm * tm + w_iou * iou
            scores[i, j] = s
            meta[i][j] = {"tm": round(tm, 3), "iou": round(iou, 3),
                          "angle": ang, "scale": round(sc, 3),
                          "ratio": round(ratio, 3)}
            if verbose:
                print(f"    p{i}->b{j}: tm={tm:.3f} iou={iou:.3f} "
                      f"ang={ang:.0f} sc={sc:.2f} ratio={ratio:.2f} => {s:.3f}")
    return scores, meta


def assign(scores, min_score=0.0):
    """贪心一一指派（与 runtime find_overall_index_fast 同思路）。"""
    N, M = scores.shape
    if N == 0 or M == 0:
        return []
    mat = scores.copy()
    pairs = []
    for _ in range(min(N, M)):
        i, j = np.unravel_index(int(np.argmax(mat)), mat.shape)
        if mat[i, j] <= min_score:
            break
        pairs.append((int(i), int(j), float(mat[i, j])))
        mat[i, :] = -1
        mat[:, j] = -1
    pairs.sort(key=lambda t: t[0])
    return pairs


def main():
    """自检：在 data/samples 上前 N 题跑一遍，报告配对质量。"""
    import argparse
    import glob
    import json
    import time

    from seg_2408 import segment_icons_2408, imread_u
    from autolabel_2408 import split_prompt_icons

    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    dirs = sorted(glob.glob(os.path.join(root, "*", "meta.json")))[:args.limit]

    ex = 0
    rates = []
    t0 = time.time()
    for mp in dirs:
        d = os.path.dirname(mp)
        sid = os.path.basename(d)
        meta = json.load(open(mp, encoding="utf-8"))
        bg = imread_u(os.path.join(d, "bg.jpg"))
        pr = imread_u(os.path.join(d, meta["prompts"][0]["file"]))
        if bg is None or pr is None:
            continue
        icons = split_prompt_icons(pr)
        seg, _ = segment_icons_2408(bg)
        boxes = [s["box"] for s in seg]
        sc, _m = match_prompt_to_boxes(icons, bg, boxes, verbose=args.verbose)
        pairs = assign(sc)
        good = sum(1 for _i, _j, s in pairs if s >= args.thr)
        rates.append(good / max(len(icons), 1))
        if good == len(icons) and len(icons) > 0:
            ex += 1
        print(f"  {sid}: 提示={len(icons)} 候选={len(boxes)} 配对={len(pairs)} "
              f"good={good} 分={[round(p[2],2) for p in pairs]}")

    print(f"\n=== 汇总 ({args.limit} 题) ===")
    print(f"  3/3 全配 : {ex}/{len(rates)} = {ex/max(len(rates),1):.1%}")
    print(f"  平均配率 : {np.mean(rates):.1%}")
    print(f"  耗时     : {(time.time()-t0)*1000/max(len(rates),1):.0f} ms/题")


if __name__ == "__main__":
    main()
