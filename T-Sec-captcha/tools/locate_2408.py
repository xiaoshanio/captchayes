# -*- coding: utf-8 -*-
"""locate_2408: 用提示图标做模板匹配，在底图上定位所有图标（含与背景同色的）。

为什么必须补这一步（这是本轮最关键的认知修正）：

    tools/seg_2408.py 是**基于暗度/彩度阈值**的分割器。它在两类题上失效：

        ① 纯黑线稿 + 摄影背景（82.6%）     → 有效，召回 ~99%
        ② 彩色/低对比图标 + 彩色色块背景（17.4%）→ 失效

    子域 ② 的失败原因实测（以 t2408_1810856471406067712 为例）：

        底图 5 个图标：收音机、定位、建筑、笑脸、房子
        分割器只出 3 个（定位、笑脸、房子）
        漏掉的「收音机」「建筑」压在**深色色块**上，图标本身也是深色，
        灰度差只有 10~20 —— **任何全局阈值都分不开**
        （实测 Otsu×0.55 = 79.2，图标 60~110，背景 60~90，区间重叠）

    也就是说：**靠"把图标从背景里分出来"这条路在子域 ② 走不通。**
    必须换成"用已知的图标形状去找它"—— 即模板匹配。

    这正好是我们已有的信息：提示图里就有 3 个目标图标的干净图像！
    用它们做多角度多尺度模板匹配，能直接定位到目标图标，
    完全绕过"分割"这一步。

本模块做两件事：
    1) locate_by_template(prompt_icons, bg)  → 定位**目标**图标（提示里有的）
    2) detect_extra_blobs(bg, exclude)       → 用阈值法补**干扰项**框
       （干扰项不参与点击，框不准影响小；漏了会少一列，影响也小）

用法（库）：
    from locate_2408 import locate_all
    boxes, info = locate_all(prompt_icons, bg)
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


def _soft_ink(bgr, lo=40, hi=140):
    """软墨迹：黑→白映射成连续置信度（抗 JPEG 振铃/抗锯齿）。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    v = np.clip((hi - g.astype(np.float32)) / float(hi - lo), 0.0, 1.0)
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


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1)


def locate_one(icon_bgr, bg_ink, angle_step=15, scale_range=(1.3, 2.4),
               scale_steps=6, score_thr=0.45, max_hits=2):
    """用单个提示图标在底图软墨迹上做多角度多尺度匹配。

    返回 [{box, score, angle, scale}]，按分数降序。

    尺度范围为什么是 1.3~2.4：
        提示图标在 170x50 精灵图里 36~48 px，
        底图图标渲染 67~80 px，比值 1.4~2.2。留余量取 1.3~2.4。
    """
    tpl0 = _soft_ink(icon_bgr)
    if tpl0.sum() == 0:
        return []
    ys, xs = np.nonzero(tpl0 > 30)
    if len(ys) == 0:
        return []
    tpl0 = tpl0[ys.min():ys.max()+1, xs.min():xs.max()+1]

    H, W = bg_ink.shape[:2]
    hits = []
    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    for ang in range(-180, 180, angle_step):
        t = _rot(tpl0, ang) if ang else tpl0
        for sc in scales:
            tt = cv2.resize(t, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
            th, tw = tt.shape[:2]
            if th >= H or tw >= W or th < 20 or tw < 20:
                continue
            try:
                res = cv2.matchTemplate(bg_ink, tt, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            # 取局部极大（不止全局最大，可能有多个同图标）
            k = min(max_hits * 3, res.size)
            flat = res.ravel()
            idxs = np.argpartition(-flat, k - 1)[:k]
            for p in idxs:
                v = float(flat[p])
                if v < score_thr:
                    continue
                y, x = divmod(int(p), res.shape[1])
                hits.append({"box": (int(x), int(y), int(x + tw), int(y + th)),
                             "score": v, "angle": float(ang),
                             "scale": float(sc)})
    if not hits:
        return []
    hits.sort(key=lambda z: -z["score"])
    # 去重：IoU>0.3 视为同一个
    keep = []
    for h in hits:
        if all(iou(h["box"], k["box"]) <= 0.3 for k in keep):
            keep.append(h)
        if len(keep) >= max_hits:
            break
    return keep


def detect_blobs(bg, exclude=(), dark_thr=None, min_side=20, min_ink=200,
                 max_side=140, exclude_iou=0.25):
    """阈值法补干扰项框（子域 ① 有效；子域 ② 只能补到一部分）。

    exclude 里的框会被跳过（已由模板匹配定位的目标）。
    """
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    if dark_thr is None:
        otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_thr = float(np.clip(otsu * 0.55, 60, 110))
    m = (g < dark_thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    H, W = g.shape[:2]
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if min(w, h) < min_side or a < min_ink:
            continue
        if max(w, h) > max_side or w > W * 0.5 or h > H * 0.5:
            continue
        if a / float(w * h) > 0.75:
            continue
        box = (int(x), int(y), int(x + w), int(y + h))
        if any(iou(box, e) > exclude_iou for e in exclude):
            continue
        out.append({"box": box, "score": 0.0, "angle": 0.0, "scale": 1.0,
                    "src": "blob", "ink": int(a)})
    return out


def locate_all(prompt_icons, bg, score_thr=0.45, angle_step=15,
               scale_range=(1.3, 2.4), scale_steps=6, want_blobs=True):
    """定位底图上所有图标：目标用模板匹配，干扰项用阈值补。

    返回 (boxes, info)
        boxes: [(x1,y1,x2,y2), ...]  去重后的所有候选框
        info : {"targets": [...], "blobs": [...], "n_prompt": N}
    """
    bg_ink = _soft_ink(bg)
    targets = []
    per_icon = []
    for i, ic in enumerate(prompt_icons):
        hits = locate_one(ic["img"] if isinstance(ic, dict) else ic,
                          bg_ink, angle_step=angle_step,
                          scale_range=scale_range, scale_steps=scale_steps,
                          score_thr=score_thr)
        per_icon.append({"icon": i, "hits": hits})
        for h in hits:
            h2 = dict(h)
            h2["src"] = f"tpl{i}"
            targets.append(h2)

    # 目标之间去重（不同提示图标可能匹配到同一位置，取高分那个）
    targets.sort(key=lambda z: -z["score"])
    tkeep = []
    for t in targets:
        if all(iou(t["box"], k["box"]) <= 0.3 for k in tkeep):
            tkeep.append(t)

    blobs = detect_blobs(bg, exclude=[t["box"] for t in tkeep]) if want_blobs else []
    allb = tkeep + blobs
    # 统一去重
    allb.sort(key=lambda z: -z["score"])
    final = []
    for b in allb:
        if all(iou(b["box"], f["box"]) <= 0.25 for f in final):
            final.append(b)
    return [f["box"] for f in final], {"targets": tkeep, "blobs": blobs,
                                       "per_icon": per_icon,
                                       "n_prompt": len(prompt_icons)}


def main():
    """自检：对比「分割器」与「模板定位」在目标图标上的召回。"""
    import argparse
    import glob
    import json

    from seg_2408 import segment_icons_auto, imread_u
    from autolabel_2408 import split_prompt_icons

    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--labeled", default="data/labeled2408v2")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--score-thr", type=float, default=0.45)
    args = ap.parse_args()

    root = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    lab = args.labeled if os.path.isabs(args.labeled) else os.path.join(REPO, args.labeled)

    gt = {}
    idx = os.path.join(lab, "labels_index.jsonl")
    if os.path.exists(idx):
        for line in open(idx, encoding="utf-8"):
            r = json.loads(line)
            boxes = r.get("bg_boxes") or []
            order = [boxes[p["bg"]] for p in sorted(r.get("pairs") or [],
                                                    key=lambda z: z["prompt"])
                     if 0 <= p["bg"] < len(boxes)]
            if order:
                gt[r["sid"]] = order

    sids = [s for s in gt][:args.limit]
    seg_hit = seg_tot = tpl_hit = tpl_tot = 0
    for sid in sids:
        d = os.path.join(root, sid)
        bg = imread_u(os.path.join(d, "bg.jpg"))
        meta_p = os.path.join(d, "meta.json")
        if bg is None or not os.path.exists(meta_p):
            continue
        meta = json.load(open(meta_p, encoding="utf-8"))
        prs = meta.get("prompts") or []
        if not prs:
            continue
        pr = imread_u(os.path.join(d, prs[0]["file"]))
        if pr is None:
            continue
        icons = split_prompt_icons(pr)
        if not icons:
            continue

        seg, _ = segment_icons_auto(bg)
        seg_boxes = [s["box"] for s in seg]
        tpl_boxes, _info = locate_all(icons, bg, score_thr=args.score_thr)

        for g in gt[sid]:
            seg_tot += 1
            tpl_tot += 1
            if max([iou(g, b) for b in seg_boxes], default=0) > 0.5:
                seg_hit += 1
            if max([iou(g, b) for b in tpl_boxes], default=0) > 0.5:
                tpl_hit += 1

    print(f"\n=== 目标图标召回（{len(sids)} 题）===")
    print(f"  分割器     : {seg_hit}/{seg_tot} = {seg_hit/max(seg_tot,1):.1%}")
    print(f"  模板定位   : {tpl_hit}/{tpl_tot} = {tpl_hit/max(tpl_tot,1):.1%}")


if __name__ == "__main__":
    main()
