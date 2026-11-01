# -*- coding: utf-8 -*-
"""icon_match: 图标标定核心库 —— 快速、旋转/尺度不变的线稿图标匹配。

为什么不用暴力滑窗模板匹配：
    672x480 底图 × 36 角度 × 10 尺度 × masked matchTemplate ≈ 每张 10~30 秒。
    几百张样本要跑几小时，完全不可用于数据流水线。

本库的做法（两阶段）：
    A. **分割**：黑帽形态学(blackhat) 提取细线笔画 -> 闭运算连笔画 -> 连通域 -> 框
       —— 底图里所有可点图标（含干扰项）都能拿到框，不需要匹配。
    B. **形状归一定匹配**：把图标裁剪块转成「旋转+尺度归一」的 64x64 二值图
       （质心 + 主轴方向 + 面积归一），两两比较做 IoU/相关；
       180° 歧义用翻转双向取最大解决。每对比较 < 1ms。

这两个能力合起来就满足自动标注的全部需求：
    class 0 = 底图全部图标（来自 A）
    class 2 = 提示条图标（来自精确裁剪）
    点击顺序 + 置信度 = 来自 B

用法（作为库）：
    from tools.icon_match import segment_icons, normalize_shape, assign_icons
用法（自检）：
    python tools/icon_match.py --samples data/samples --limit 12
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

NORM = 64          # 归一化画布边长


# ==================================================================== 分割
def ink_mask(bgr, bh_kernel=21, bh_thr=28, s_strong=90):
    """黑帽形态学提取「压在照片上的细线稿」。返回 (掩码, 调试信息)。

    blackhat = closing(img,k) - img
    细笔画被 closing 抹掉 -> blackhat 亮；大片暗背景闭运算抹不掉 -> blackhat 近似 0。
    这就是为什么它比「绝对暗度阈值」强：能区分图标笔画与深色背景。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_kernel, bh_kernel))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    dark = (bh > bh_thr).astype(np.uint8) * 255

    # 彩色描边（部分题型的图标是彩色线稿）
    colored = ((s > s_strong) * 255).astype(np.uint8)
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    m = cv2.bitwise_or(dark, colored)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    return m, {"blackhat": bh, "dark": dark, "colored": colored}


def _merge_boxes(boxes, gap=18):
    """把同一图标被切开的多个部件并成一个框（并查集式迭代合并）。"""
    out = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                # 扩张后相交 => 合并
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


def segment_icons(bgr, close_k=13, min_side=22, min_ink=140, max_ratio=0.6,
                  gap=18, merge=True):
    """返回底图中所有线稿图标的候选框 [{box:(x1,y1,x2,y2), ink, fill, w, h}]。

    参数按天御图标点选标定：图标本体约 40~140 px，线稿笔画较细。
    """
    m, dbg = ink_mask(bgr)
    if close_k:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    H, W = bgr.shape[:2]
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) < min_side:
            continue
        if w > W * max_ratio or h > H * max_ratio:
            continue
        sub = m[y:y + h, x:x + w]
        ink = int((sub > 0).sum())
        if ink < min_ink:
            continue
        fill = ink / float(w * h)
        if fill > 0.75:                       # 大实心块 = 背景，不是线稿
            continue
        boxes.append((x, y, x + w, y + h))

    if merge:
        boxes = _merge_boxes(boxes, gap=gap)

    out = []
    for (x1, y1, x2, y2) in boxes:
        w, h = x2 - x1, y2 - y1
        if w > W * max_ratio or h > H * max_ratio:
            continue
        sub = m[y1:y2, x1:x2]
        ink = int((sub > 0).sum())
        if ink < min_ink:
            continue
        out.append({"box": (int(x1), int(y1), int(x2), int(y2)),
                    "ink": ink, "w": int(w), "h": int(h),
                    "fill": round(ink / float(max(w * h, 1)), 3)})
    out.sort(key=lambda d: -d["ink"])
    return out, {"mask": m, **dbg}


# ============================================================ 形状归一
def shape_mask(patch, size=NORM, ink_thr=None):
    """裁剪块 -> 二值墨迹掩码（用于形状比较）。"""
    if patch is None or patch.size == 0:
        return np.zeros((size, size), np.uint8)
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    # 图标是深色线稿：用 Otsu 自适应，避免不同题型阈值不同
    if ink_thr is None:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        bw = (gray < ink_thr).astype(np.uint8) * 255
    return bw


def normalize_shape(patch, size=NORM, pad=0.06):
    """旋转+尺度+平移归一：返回 (二值图 size×size, 方向角, 质量)。

    步骤：
      1) Otsu 取墨迹
      2) 最大连通域为主，取其像素集合
      3) 质心平移 + 主轴(二阶矩方向)旋转 -> 正立
      4) 面积归一（等价于尺度归一）后放到 size×size 画布上
    """
    bw = shape_mask(patch)
    if cv2.countNonZero(bw) < 30:
        return np.zeros((size, size), np.uint8), 0.0, 0

    # 取最大连通域 + 其邻近部件（同一图标的笔画可能分离）
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return np.zeros((size, size), np.uint8), 0.0, 0
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mx, my, mw, mh, ma = stats[main]
    keep = np.zeros_like(bw)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        # 与主体框有重叠(带余量)的部件一起保留
        if (x - 6 < mx + mw and mx < x + w + 6 and
                y - 6 < my + mh and my < y + h + 6):
            keep[labels == i] = 255

    ys, xs = np.nonzero(keep)
    if len(xs) < 30:
        return np.zeros((size, size), np.uint8), 0.0, 0
    cx, cy = xs.mean(), ys.mean()
    x0, y0 = xs - cx, ys - cy
    mu20 = float((x0 ** 2).mean())
    mu02 = float((y0 ** 2).mean())
    mu11 = float((x0 * y0).mean())
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)      # 主轴角(弧度)

    # 尺度归一：用「面积」而不是包围盒，抗笔画粗细差异
    scale = 42.0 / max(np.sqrt(float(len(xs))), 1.0)

    M = cv2.getRotationMatrix2D((cx, cy), np.degrees(theta), scale)
    # 平移到目标画布中心
    M[0, 2] += size / 2.0 - cx
    M[1, 2] += size / 2.0 - cy
    norm = cv2.warpAffine(keep, M, (size, size), flags=cv2.INTER_NEAREST)
    # 轻微内缩去掉插值毛边
    p = int(size * pad)
    if p > 0:
        norm[:p] = 0
        norm[-p:] = 0
        norm[:, :p] = 0
        norm[:, -p:] = 0
    return norm, float(np.degrees(theta) % 180.0), int(len(xs))


def flip180(norm):
    return cv2.rotate(norm, cv2.ROTATE_180)


def shape_score(a, b):
    """两个归一形状的相似度 (0~1)，自动处理 180 度主轴歧义。"""
    if a is None or b is None or a.sum() == 0 or b.sum() == 0:
        return 0.0
    best = 0.0
    for cand in (b, flip180(b)):
        inter = np.logical_and(a > 0, cand > 0).sum()
        union = np.logical_or(a > 0, cand > 0).sum()
        iou = inter / float(union) if union else 0.0
        # 叠加归一化相关，IoU 对细线稿偏严，两者加权更稳
        aa = (a > 0).astype(np.float32) - 0.5
        bb = (cand > 0).astype(np.float32) - 0.5
        denom = (np.linalg.norm(aa) * np.linalg.norm(bb))
        corr = float((aa * bb).sum() / denom) if denom > 0 else 0.0
        best = max(best, 0.6 * iou + 0.4 * corr)
    return float(best)


# ============================================================ 指派
def assign_icons(prompt_norms, bg_norms, scores=None):
    """把每个提示图标指派到底图图标（一一对应，贪心 + 匈牙利式回退）。

    返回 (pairs, all_scores, margin)
        pairs     : [(prompt_idx, bg_idx, score), ...] 按 prompt 顺序 = 点击顺序
        all_scores: N×M 分数矩阵
        margin    : 每个 prompt 的「最优分 - 次优分」，用于判断是否可信
    """
    N, M = len(prompt_norms), len(bg_norms)
    if N == 0 or M == 0:
        return [], np.zeros((N, M)), []
    if scores is None:
        scores = np.zeros((N, M), np.float32)
        for i in range(N):
            for j in range(M):
                scores[i, j] = shape_score(prompt_norms[i], bg_norms[j])

    # 贪心全局最大（与 runtime 的 find_overall_index_fast 同思路，但用形状分）
    mat = scores.copy()
    pairs = []
    used_cols = set()
    order = []
    for _ in range(min(N, M)):
        i, j = np.unravel_index(int(np.argmax(mat)), mat.shape)
        if mat[i, j] <= 0:
            break
        pairs.append((int(i), int(j), float(mat[i, j])))
        order.append((int(i), int(j)))
        mat[i, :] = -1
        mat[:, j] = -1
        used_cols.add(j)
    pairs.sort(key=lambda t: t[0])

    margins = []
    for i in range(N):
        row = np.sort(scores[i])[::-1]
        margins.append(float(row[0] - row[1]) if len(row) > 1 else float(row[0]) if len(row) else 0.0)
    return pairs, scores, margins


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


# ============================================================ 自检
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--out", default="out/iconmatch")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import glob
    import time
    from autolabel import split_prompt_icons, imread_u

    metas = sorted(glob.glob(os.path.join(args.samples, "*", "meta.json")))[:args.limit]
    report = []
    for mp in metas:
        d = os.path.dirname(mp)
        sid = os.path.basename(d)
        meta = json.load(open(mp, encoding="utf-8"))
        bg = imread_u(os.path.join(d, "bg.jpg"))
        pr = imread_u(os.path.join(d, meta["prompts"][0]["file"]))
        if bg is None or pr is None:
            continue

        t0 = time.time()
        seg, dbg = segment_icons(bg)
        t_seg = (time.time() - t0) * 1000

        icons = split_prompt_icons(pr)
        p_norms = [normalize_shape(it["img"])[0] for it in icons]
        b_norms = [normalize_shape(bg[y1:y2, x1:x2])[0] for x1, y1, x2, y2 in
                   [b["box"] for b in seg]]
        t1 = time.time()
        pairs, scores, margins = assign_icons(p_norms, b_norms)
        t_match = (time.time() - t1) * 1000

        vis = bg.copy()
        for k, b in enumerate(seg):
            x1, y1, x2, y2 = b["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (160, 160, 160), 1)
            cv2.putText(vis, f"c{k}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (160, 160, 160), 1, cv2.LINE_AA)
        for rank, (i, j, sc) in enumerate(pairs):
            x1, y1, x2, y2 = seg[j]["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(vis, f"#{rank+1} {sc:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out, f"{sid}.jpg"), vis)

        # 归一形状拼图（便于看归一质量）
        tile = np.zeros((NORM * max(len(p_norms), 1), NORM * (len(p_norms) + len(b_norms)), 3),
                        np.uint8)
        for i, n in enumerate(p_norms):
            tile[i % max(len(p_norms), 1) * NORM:(i % max(len(p_norms), 1) + 1) * NORM,
                 0:NORM] = cv2.cvtColor(n, cv2.COLOR_GRAY2BGR)
        for j, n in enumerate(b_norms):
            col = len(p_norms) + j
            r = j % max(len(b_norms), 1)
            tile[r * NORM:(r + 1) * NORM, col * NORM:(col + 1) * NORM] = \
                cv2.cvtColor(n, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(args.out, f"{sid}_shapes.png"), tile)

        pair_scores = [p[2] for p in pairs]
        rec = {"sid": sid, "n_prompt": len(icons), "n_bg_icons": len(seg),
               "pairs": [{"prompt": i, "bg": j, "score": round(s, 3)} for i, j, s in pairs],
               "min_pair_score": round(min(pair_scores), 3) if pair_scores else 0.0,
               "margins": [round(m, 3) for m in margins],
               "t_seg_ms": round(t_seg, 1), "t_match_ms": round(t_match, 1)}
        report.append(rec)
        print(f"  {sid}: 提示={len(icons)} 底图图标={len(seg)} 配对={len(pairs)} "
              f"minscore={rec['min_pair_score']:.3f} "
              f"seg={t_seg:.0f}ms match={t_match:.0f}ms")

    if report:
        import statistics as s
        print("\n=== 汇总 ===")
        print(f"  底图图标数: mean={s.mean([r['n_bg_icons'] for r in report]):.1f} "
              f"min={min(r['n_bg_icons'] for r in report)} max={max(r['n_bg_icons'] for r in report)}")
        print(f"  提示图标数: mean={s.mean([r['n_prompt'] for r in report]):.1f}")
        print(f"  最低配对分: mean={s.mean([r['min_pair_score'] for r in report]):.3f}")
        print(f"  分割耗时  : {s.mean([r['t_seg_ms'] for r in report]):.0f} ms/张")
        print(f"  配准耗时  : {s.mean([r['t_match_ms'] for r in report]):.0f} ms/张")
        json.dump(report, open(os.path.join(args.out, "report.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    print(f"\n可视化 -> {args.out}/")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
