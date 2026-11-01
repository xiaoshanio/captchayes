# -*- coding: utf-8 -*-
"""hardneg_2408: 跨题难负样本挖掘 —— 按「形状相似度」找最容易混淆的图标。

为什么需要（这是单点命中卡在 87.8% 的直接原因）：

    实测每题候选框 5 个、提示图标 3 个 → **2 个干扰项**。
    天御的干扰项不是随机噪声，而是**与目标外形相似的其它图标**
    （同类线稿、相近笔画密度、相近尺寸）。

    当前配对集的难负样本只有「同题其它框」（prepare_pairs_det_2408 的 hard-neg），
    它覆盖了同题干扰，但**没有覆盖跨题的形近图标**：
    模型在训练时从没见过"另一个题里的相似图标"，推理时遇到就会给高分。

    这解释了为什么：
        A 组（真值框，候选=3，全是目标）     点命中 96.7%
        B 组（检测框，候选=5，含 2 干扰）    点命中 86.1%
        差 10.6 pt 全部来自"形近干扰项"。

本脚本的做法：
    1) 对每个提示图标，算它与**全局所有底图图标**的相似度
       （用 tools/match_2408.py 的软墨迹多角度匹配，已验证 3/3 全配 75%）
    2) 取 top-K 最相似的「非本题」图标作为额外难负样本
    3) 输出增量配对文件，与主配对集合并训练

    为什么不用随机跨题负样本：
        随机负样本太容易（不同图标一眼可分），对决策边界几乎无贡献。
        实测易负样本 758 个，但模型对它们的 loss 早已接近 0。

产出：
    <out>/hardneg_train.jsonl / hardneg_val.jsonl
    <out>/hardneg_report.json

用法：
    python tools/hardneg_2408.py --det runs/export_t2408b/best_v3_2408b.onnx \
        --out data/pairsdet2408
    # 然后与主配对集合并
    python tools/merge_pairs.py --dirs data/pairsdet2408 --out data/pairsfinal
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


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img, q=95):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        buf.tofile(p)
    return ok


def crop(img, box, cache_dir, key, norm_short=None):
    fn = os.path.join(cache_dir, key + ".jpg")
    if os.path.exists(fn):
        return fn
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
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


def shape_sig(img, size=32):
    """快速形状指纹：归一化二值墨迹的 DCT 低频（用于粗筛，不是最终判据）。

    为什么要粗筛：
        全量两两匹配（N 提示图标 × M 底图图标）代价太高。
        先用 32x32 形状指纹做一次粗筛，只对 top-C 候选跑精确匹配。
    """
    if img is None or img.size == 0:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bw = (g < 110).astype(np.float32)
    if bw.sum() < 20:
        return None
    ys, xs = np.nonzero(bw)
    y1, y2, x1, x2 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    bw = bw[y1:y2, x1:x2]
    bw = cv2.resize(bw, (size, size), interpolation=cv2.INTER_AREA)
    v = cv2.dct(bw)[:8, :8].flatten()
    v = v / (np.linalg.norm(v) + 1e-6)
    return v.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/export_t2408b/best_v3_2408b.onnx")
    ap.add_argument("--labeled", default="data/labeled2408")
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--out", default="data/pairsdet2408")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dedup", type=float, default=0.5)
    ap.add_argument("--norm-short", type=int, default=64)
    ap.add_argument("--topk", type=int, default=2,
                    help="每个提示图标取 top-K 最相似的跨题图标作难负样本")
    ap.add_argument("--coarse", type=int, default=12,
                    help="形状指纹粗筛保留的候选数")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from src.utils import yolo_onnx, matchingMode
    from autolabel_2408 import split_prompt_icons
    from match_2408 import _soft_ink, _rot, _best_in_box

    lab = args.labeled if os.path.isabs(args.labeled) else os.path.join(REPO, args.labeled)
    sroot = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    crops_dir = os.path.join(out, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    recs = [json.loads(l) for l in
            open(os.path.join(lab, "labels_index.jsonl"), encoding="utf-8") if l.strip()]
    y = yolo_onnx.YOLO(os.path.join(REPO, args.det), conf_threshold=args.conf)

    # ---- 建全局图标池（每题：检测框 + 裁剪 + 形状指纹）
    pool = []      # [{sid, box, img, sig}]
    per_q = {}     # sid -> {t: [boxes], c: [boxes], img}
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
                if all(x[5] != k[5] or
                       (lambda a, b: (max(0, min(a[2], b[2])-max(a[0], b[0])) *
                                      max(0, min(a[3], b[3])-max(a[1], b[1]))) /
                                     max((a[2]-a[0])*(a[3]-a[1]) +
                                         (b[2]-b[0])*(b[3]-b[1]) -
                                         max(0, min(a[2], b[2])-max(a[0], b[0])) *
                                         max(0, min(a[3], b[3])-max(a[1], b[1])), 1)
                        )(x[:4], k[:4]) <= args.dedup for k in keep):
                    keep.append(x)
            d = keep
        dt = [x[:4] for x in d if x[5] == 0]
        dc = [x[:4] for x in d if x[5] == 2]
        dc.sort(key=lambda b: b[0])
        per_q[sid] = {"t": dt, "c": dc, "img": img, "rec": r}
        for j, b in enumerate(dt):
            cpath = crop(img, b, crops_dir, f"{sid}_pool{j}",
                         norm_short=args.norm_short or None)
            if cpath is None:
                continue
            cimg = imread_u(cpath)
            if cimg is None:
                continue
            sig = shape_sig(cimg)
            if sig is None:
                continue
            pool.append({"sid": sid, "j": j, "box": b, "sig": sig,
                         "path": cpath, "img": img})

    print(f"[pool] {len(pool)} 个底图图标（来自 {len(per_q)} 题）")
    if not pool:
        raise SystemExit("图标池为空")

    sigs = np.stack([p["sig"] for p in pool])
    rng = random.Random(args.seed)
    rows = []
    n_hard = 0
    angles = list(range(-180, 180, 20))     # 粗匹配用 20° 步长（精匹配留给主训练）

    for qi, (sid, q) in enumerate(per_q.items()):
        rec = q["rec"]
        meta_p = os.path.join(sroot, sid, "meta.json")
        if not os.path.exists(meta_p):
            continue
        meta = json.load(open(meta_p, encoding="utf-8"))
        prs = meta.get("prompts") or []
        if not prs:
            continue
        pr = imread_u(os.path.join(sroot, sid, prs[0]["file"]))
        if pr is None:
            continue
        icons = split_prompt_icons(pr)
        if not icons:
            continue

        for i, it in enumerate(icons):
            a = crop(it["img"], (0, 0, it["img"].shape[1], it["img"].shape[0]),
                     crops_dir, f"{sid}_p{i}", norm_short=args.norm_short or None)
            if a is None:
                continue
            a_sig = shape_sig(it["img"])
            if a_sig is None:
                continue

            # 粗筛：余弦相似 top-C（排除本题）
            sims = sigs @ a_sig
            order = np.argsort(-sims)
            cands = []
            for idx in order:
                p = pool[int(idx)]
                if p["sid"] == sid:
                    continue
                cands.append((float(sims[idx]), p))
                if len(cands) >= args.coarse:
                    break

            # 精筛：软墨迹多角度模板匹配
            tpl = _soft_ink(it["img"])
            scored = []
            for cs, p in cands:
                box_ink = _soft_ink(p["img"][int(p["box"][1]):int(p["box"][3]),
                                             int(p["box"][0]):int(p["box"][2])])
                if box_ink.size == 0 or box_ink.sum() == 0:
                    continue
                best = 0.0
                for ang in angles:
                    t = _rot(tpl, ang) if ang else tpl
                    th, tw = t.shape[:2]
                    bh, bw_ = box_ink.shape[:2]
                    # 尺度对齐（与 match_2408 同思路：按短边比）
                    ys, xs = np.nonzero(t > 30)
                    if len(ys) == 0:
                        continue
                    ts = max(min(ys.max()-ys.min()+1, xs.max()-xs.min()+1), 1)
                    ys2, xs2 = np.nonzero(box_ink > 30)
                    if len(ys2) == 0:
                        continue
                    bs = max(min(ys2.max()-ys2.min()+1, xs2.max()-xs2.min()+1), 1)
                    sc = bs / float(ts)
                    tt = cv2.resize(t, None, fx=sc, fy=sc,
                                    interpolation=cv2.INTER_LINEAR)
                    th, tw = tt.shape[:2]
                    if th > bh or tw > bw_ or th < 8 or tw < 8:
                        continue
                    try:
                        r_ = cv2.matchTemplate(box_ink, tt, cv2.TM_CCOEFF_NORMED)
                    except cv2.error:
                        continue
                    best = max(best, float(np.nan_to_num(r_, nan=0.0).max()))
                scored.append((best, p))

            scored.sort(key=lambda z: -z[0])
            for sc, p in scored[:args.topk]:
                rows.append({"a": a, "b": p["path"], "label": 0, "hard": True,
                             "sid": sid, "src": "crossq", "sim": round(sc, 4)})
                n_hard += 1
        if (qi + 1) % 40 == 0:
            print(f"  [{qi+1}/{len(per_q)}] 已挖 {n_hard} 个跨题难负样本")

    if not rows:
        raise SystemExit("没有挖到任何跨题难负样本")

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    val, train = rows[:n_val], rows[n_val:]
    for name, part in (("hardneg_train", train), ("hardneg_val", val)):
        with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for x in part:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    sims_all = [x["sim"] for x in rows]
    rep = {
        "det": args.det, "pool": len(pool), "questions": len(per_q),
        "topk": args.topk, "coarse": args.coarse,
        "n_hardneg": len(rows), "train": len(train), "val": len(val),
        "sim_mean": round(float(np.mean(sims_all)), 4),
        "sim_p90": round(float(np.percentile(sims_all, 90)), 4),
        "sim_max": round(float(np.max(sims_all)), 4),
        "note": "sim 越高说明越难（越像目标图标）",
    }
    with open(os.path.join(out, "hardneg_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print("\n=== 跨题难负样本挖掘完成 ===")
    print(f"  图标池     : {len(pool)}")
    print(f"  挖到难负样本: {len(rows)}  (train={len(train)}, val={len(val)})")
    print(f"  相似度     : mean={rep['sim_mean']:.3f} "
          f"p90={rep['sim_p90']:.3f} max={rep['sim_max']:.3f}")
    print(f"  输出       : {out}/")


if __name__ == "__main__":
    main()
