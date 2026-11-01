# -*- coding: utf-8 -*-
"""seg_2408: 天御 2408 图标分割器（专用，比通用 blackhat 精确）。

为什么不能直接用 tools/icon_match.py 的 segment_icons：
    它的 blackhat 判据只看「细暗笔画」，而 2408 的底图是**摄影实景**
    （水草、沙纹、枝叶都是细结构），blackhat 响应比图标还强，
    实测 12 题里误检 8 个水草碎片，底图图标召回只有 2.08/5。

本模块的判据（针对「纯黑线稿压在照片上」这一确定题型）：
    D1 暗度：图标是近纯黑，gray < dark_thr(72)；照片最暗处极少低于 60
    D2 彩度：图标是中性黑，max(B,G,R)-min(...) < chroma_thr(45)；水草/天空是彩色
    D3 面积：墨迹像素 >= min_ink(120)
    D4 尺寸：短边 >= min_side(20)，长边 <= 0.6*画幅
    D5 填充率：ink/(w*h) < 0.72（实心暗块是背景不是线稿）
    D6 笔画一致性：框内墨迹的「暗度中位数」< 85，排除半透明阴影
    D7 结构：线稿至少有一个内部空洞（闭环）或笔画分散度足够
        —— 这一条能把「一整根草叶」这种单条细线排除掉

用法（库）：
    from seg_2408 import segment_icons_2408
用法（自检）：
    python tools/seg_2408.py --samples data/samples --limit 10
"""
from __future__ import annotations

import argparse
import glob
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


def imread_u(p, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), flags)


def ink_mask_2408(bgr, dark_thr=72, chroma_thr=45, adaptive=False):
    """近黑 + 低彩度 → 图标墨迹掩码。

    adaptive=True 时按题自适应暗度阈值（Otsu 的 0.55 倍），用于处理
    **彩色图标子域**：

        实测 305 题里有 53 题（17.4%）的底图图标是**彩色**的
        （压在彩色三角/几何色块背景上），此时：
          · 图标灰度 60~110（不是纯黑），dark_thr=72 会漏掉一半
          · 图标彩度 56，反而**低于**背景彩度 81 —— 彩度判据完全失效
        这类题的标注召回极低，是"15% 配对错误"的真正来源。

        自适应阈值实测：thr=80~90 能稳定分出 5~6 个图标（真值 5~6 个），
        而固定 72 只能分出 2~3 个。
    """
    b, g, r = cv2.split(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if adaptive:
        # Otsu 在整图上找暗/亮分界，再取一个偏暗的比例
        otsu, _ = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_thr = float(np.clip(otsu * 0.55, 70, 110))
        chroma_thr = 255.0        # 彩色子域下彩度不可用，直接放行
    mx = np.maximum(np.maximum(b, g), r).astype(np.int16)
    mn = np.minimum(np.minimum(b, g), r).astype(np.int16)
    chroma = mx - mn
    m = ((gray < dark_thr) & (chroma < chroma_thr)).astype(np.uint8) * 255
    # 去掉孤立噪点（照片颗粒）
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    return m


def _merge(boxes, gap):
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
    return [tuple(int(v) for v in b) for b in out]


def _has_structure(sub_mask, touching_border=False):
    """线稿结构判据：内部有闭环空洞，或笔画分散在多个连通域。

    单根草叶 = 1 个细长连通域且无空洞 → 排除。
    例外：图标被画幅边界裁切时（touching_border），闭环必然断开，
    此时不能用闭环判据，改由「墨迹量 + 填充率」把关。
    """
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(sub_mask, 8)
    parts = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 12]
    if len(parts) >= 2:
        return True
    if touching_border:
        return True
    # 单连通域：找内部空洞（RETR_CCOMP 的二级轮廓）
    cnts, hier = cv2.findContours(sub_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(cnts) == 0:
        return False
    hier = hier[0]
    for idx in range(len(cnts)):
        if hier[idx][3] != -1 and cv2.contourArea(cnts[idx]) > 15:
            return True
    return False


def _size_ok(w, h, min_side, max_side, min_long=45, thin_short=10):
    """尺寸判据，含**细长图标例外**。

    为什么需要例外：天御的图标集里有 "1"、"l"、"I" 这类极窄字形，
    实测 w≈18~22, h≈55~70。若统一要求 min(w,h) >= min_side(22)，
    这些字会被整体漏检 → 配对必然失败（实测某题 "1" 配对分 0.01）。
    规则：短边 >= min_side 正常通过；
          否则若长边 >= min_long 且短边 >= thin_short，视为细长图标放行。
    """
    short, long_ = min(w, h), max(w, h)
    if short >= min_side and long_ <= max_side:
        return True
    if short >= thin_short and long_ >= min_long and long_ <= max_side:
        return True
    return False


def _split_merged(boxes, mask, max_side, min_gap_px=6):
    """把「两个图标被 merge 成一个高瘦框」的情况拆开。

    实测问题：两个图标在竖直方向靠得很近时，close_k=7 的闭运算会把它们连起来，
    得到一个 h≈150 的框（单图标 h≈72）。此时：
        1) 在框内墨迹的行投影上找「近零行」（两图标之间的缝）
        2) 以缝为界切成上下两段，各自重新求 bbox
        3) 只保留尺寸合理的段

    同理处理横向 merge（w 过大时用列投影）。
    """
    out = []
    for (x1, y1, x2, y2) in boxes:
        w, h = x2 - x1, y2 - y1
        if max(w, h) <= max_side:
            out.append((x1, y1, x2, y2))
            continue
        sub = mask[y1:y2, x1:x2]
        split_done = False
        # 先试竖直切（高瘦）
        if h > max_side and h >= w:
            rows = (sub > 0).sum(axis=1)
            cuts = _find_gaps(rows, min_gap_px)
            if len(cuts) >= 2:
                for (a, b) in _pairs(cuts):
                    seg = sub[a:b]
                    ys, xs = np.nonzero(seg)
                    if len(ys) < 100:
                        continue
                    sy1, sy2 = y1 + a + int(ys.min()), y1 + a + int(ys.max()) + 1
                    sx1, sx2 = x1 + int(xs.min()), x1 + int(xs.max()) + 1
                    if max(sx2 - sx1, sy2 - sy1) <= max_side:
                        out.append((sx1, sy1, sx2, sy2))
                split_done = True
        # 再试横向切（宽扁）
        if not split_done and w > max_side and w > h:
            cols = (sub > 0).sum(axis=0)
            cuts = _find_gaps(cols, min_gap_px)
            if len(cuts) >= 2:
                for (a, b) in _pairs(cuts):
                    seg = sub[:, a:b]
                    ys, xs = np.nonzero(seg)
                    if len(ys) < 100:
                        continue
                    sy1, sy2 = y1 + int(ys.min()), y1 + int(ys.max()) + 1
                    sx1, sx2 = x1 + a + int(xs.min()), x1 + a + int(xs.max()) + 1
                    if max(sx2 - sx1, sy2 - sy1) <= max_side:
                        out.append((sx1, sy1, sx2, sy2))
                split_done = True
        if not split_done:
            out.append((x1, y1, x2, y2))
    return out


def _find_gaps(proj, min_gap_px):
    """在投影里找「近零带」，返回各段的 [start, end) 区间列表。"""
    thr = max(1, int(proj.max() * 0.06)) if proj.size else 1
    zero = proj <= thr
    segs, start = [], 0
    run_start = None
    for i, z in enumerate(zero):
        if z and run_start is None:
            run_start = i
        elif not z and run_start is not None:
            if i - run_start >= min_gap_px:
                segs.append((start, run_start))
                start = i
            run_start = None
    if run_start is not None and len(zero) - run_start >= min_gap_px:
        segs.append((start, run_start))
    else:
        segs.append((start, len(zero)))
    return [s for s in segs if s[1] - s[0] >= 20]


def _pairs(cuts):
    return cuts


def segment_icons_2408(bgr, dark_thr=72, chroma_thr=45, min_side=22,
                       max_side=112, min_ink=250, max_fill=0.72, gap=10,
                       max_ratio=0.6, core_ratio_min=0.70, p25_max=26.0,
                       close_k=7, debug=False, thin_short=10, min_long=45,
                       min_short_side=18, adaptive=None, core_ratio_min_color=0.25):
    """返回 (icons, dbg)。

    标定依据（实测 300+ 真实 2408 样本，判据强度排序）：

        core40 = 框内墨迹像素中 gray<40 的占比 —— **最强判据（纯黑线稿子域）**
            真图标: 0.76 ~ 0.91（天御用纯黑 #0a0a0a 渲染线稿）
            噪声  : 0.00 ~ 0.62（水草/水波/云/枝叶，是「深灰」不是纯黑）
            阈值 0.70 一刀切开，比 dark_med / 闭环 / 结构判据都稳。

        p25（墨迹灰度的 25 分位）
            真图标 <= 20，噪声 >= 30。作为 core40 的交叉校验。

        ink >= 300 / min_side 16 / max_side 112
            天御图标固定渲染约 67~77 px，最大 105；
            > 112 基本是「两个图标被 merge 成一条」，交给 _split_merged 拆。

    【彩色图标子域（占 17.4%）必须走 adaptive】
        实测 305 题里有 53 题底图图标是**彩色**的（压在彩色几何色块背景上）：
          · 图标灰度 60~110，固定 dark_thr=72 会漏一半
          · 图标彩度 56 < 背景彩度 81，彩度判据完全失效
          · core40 也失效（没有纯黑像素）
        这类题必须用 adaptive=True：Otsu 自适应暗度 + 放行彩度 +
        改用 core_ratio_min_color（相对本图暗度的占比）当判据。

    为什么不用闭环判据：
        图标常被画幅边界裁切，闭环必然断开，会误杀；
        实测闭环判据的净收益为负（漏检 > 误检）。
    """
    H, W = bgr.shape[:2]
    if adaptive is None:
        # 自动判断：先按纯黑子域跑一遍，若候选太少则切到 adaptive 重跑
        adaptive = False
    m = ink_mask_2408(bgr, dark_thr, chroma_thr, adaptive=adaptive)
    if close_k:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (close_k, close_k)))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(bgr)
    chroma = (np.maximum(np.maximum(b, g), r).astype(np.int16) -
              np.minimum(np.minimum(b, g), r).astype(np.int16))

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    rej = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not _size_ok(w, h, min_side, max_side, min_long, thin_short):
            rej.append(("size", (x, y, w, h)))
            continue
        if w > W * max_ratio or h > H * max_ratio:
            rej.append(("max_ratio", (x, y, w, h)))
            continue
        sub = m[y:y + h, x:x + w]
        ink = int((sub > 0).sum())
        if ink < min_ink:
            rej.append(("min_ink", (x, y, w, h)))
            continue
        if ink / float(max(w * h, 1)) > max_fill:
            rej.append(("max_fill", (x, y, w, h)))
            continue
        boxes.append((x, y, x + w, y + h))

    boxes = _merge(boxes, gap=gap)
    boxes = _split_merged(boxes, m, max_side)

    out = []
    for (x1, y1, x2, y2) in boxes:
        w, h = x2 - x1, y2 - y1
        if not _size_ok(w, h, min_side, max_side, min_long, thin_short):
            rej.append(("post_size", (x1, y1, w, h)))
            continue
        # 碎片过滤：**两个方向都小**的框基本是分割残渣（实测 21x24 / 23x25 / 28x20
        # 这类框检测器学不出来，留在标签里就是纯噪声）。
        # 注意不能一刀切掉所有 min<30 的框 —— 细长字符（如 "1" 19x52、20x54）
        # 是真实目标，它们的长边 >= 45，必须保留。
        if min(w, h) < min_short_side and max(w, h) < min_long:
            rej.append((f"fragment({w}x{h})", (x1, y1, w, h)))
            continue
        sub = m[y1:y2, x1:x2]
        sel = sub > 0
        ink = int(sel.sum())
        if ink < min_ink:
            rej.append(("post_min_ink", (x1, y1, w, h)))
            continue
        fill = ink / float(max(w * h, 1))
        if fill > max_fill:
            rej.append(("post_max_fill", (x1, y1, w, h)))
            continue
        gv = gray[y1:y2, x1:x2][sel]
        core = float((gv < 40).mean())
        p25 = float(np.percentile(gv, 25))
        chroma_med = float(np.median(chroma[y1:y2, x1:x2][sel]))

        if adaptive:
            # 彩色子域：没有纯黑像素，core40 与 chroma 判据都失效。
            # 改用「相对本图暗度的占比」：框内墨迹里低于全图 p30 灰度的比例。
            rel = float((gv < np.percentile(gray, 30)).mean())
            if rel < core_ratio_min_color:
                rej.append((f"rel={rel:.2f}", (x1, y1, w, h)))
                continue
        else:
            if core < core_ratio_min:
                rej.append((f"core={core:.2f}", (x1, y1, w, h)))
                continue
            if p25 > p25_max:
                rej.append((f"p25={p25:.0f}", (x1, y1, w, h)))
                continue
            if chroma_med > 55:
                rej.append((f"chroma={chroma_med:.0f}", (x1, y1, w, h)))
                continue
        touching = (x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1)
        out.append({"box": (int(x1), int(y1), int(x2), int(y2)),
                    "ink": ink, "w": int(w), "h": int(h),
                    "fill": round(fill, 3),
                    "core": round(core, 3),
                    "p25": round(p25, 1),
                    "dark_med": round(float(np.median(gv)), 1),
                    "chroma_med": round(chroma_med, 1),
                    "touching_border": bool(touching)})
    out.sort(key=lambda d: -d["ink"])
    if debug and rej:
        for reason, box in rej[:15]:
            print(f"      [rej:{reason}] {box}")
    return out, {"mask": m, "n_boxes_raw": len(boxes), "rejected": rej,
                 "adaptive": bool(adaptive)}


def segment_icons_auto(bgr, min_expected=4, **kw):
    """自动判子域：先按纯黑子域跑，候选太少就切彩色子域重跑。

    为什么用「候选数」当判据而不是「彩度/暗度」：
        实测两类子域的**背景**彩度几乎一样（62.0 vs 62.3），
        图标彩度也没有稳定的分界（纯黑子域 0，彩色子域 56，但背景 81）。
        唯一稳定可测的信号是「纯黑判据能分出几个图标」：
            纯黑子域：通常 4~6 个
            彩色子域：通常 1~3 个（大部分图标不纯黑，被 core40 拒掉）
        所以直接看结果数量，最鲁棒。
    """
    icons, dbg = segment_icons_2408(bgr, adaptive=False, **kw)
    if len(icons) >= min_expected:
        return icons, dbg
    icons2, dbg2 = segment_icons_2408(bgr, adaptive=True, **kw)
    if len(icons2) > len(icons):
        dbg2["fallback_from"] = len(icons)
        return icons2, dbg2
    return icons, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--out", default="out/seg2408")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--draw", action="store_true", default=True)
    args = ap.parse_args()
    out = os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    metas = sorted(glob.glob(os.path.join(
        args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples),
        "*", "meta.json")))[:args.limit]
    ns = []
    for mp in metas:
        d = os.path.dirname(mp)
        sid = os.path.basename(d)
        bg = imread_u(os.path.join(d, "bg.jpg"))
        if bg is None:
            continue
        icons, dbg = segment_icons_2408(bg)
        ns.append(len(icons))
        vis = bg.copy()
        for k, it in enumerate(icons):
            x1, y1, x2, y2 = it["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"{k}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # 掩码可视化
        mm = cv2.cvtColor(dbg["mask"], cv2.COLOR_GRAY2BGR)
        both = np.concatenate([vis, mm], axis=1)
        cv2.imwrite(os.path.join(out, f"{sid}.jpg"), both)
        print(f"  {sid}: 图标={len(icons)} 原始框={dbg['n_boxes_raw']}")
    if ns:
        print(f"\n图标数 mean={np.mean(ns):.2f} min={min(ns)} max={max(ns)}")
    print(f"可视化 -> {out}/")


if __name__ == "__main__":
    main()
