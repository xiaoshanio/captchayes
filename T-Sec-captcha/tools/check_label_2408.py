# -*- coding: utf-8 -*-
"""check_label_2408: 量化标注自洽率（用匹配器当裁判）。

为什么需要这个独立脚本：
    §1.4 里的自洽率检查一直是内联在临时命令里的，没法复跑、没法对比多代标注。
    本轮要比较 labeled_det（第一代自举）vs labeled_det2（第二代自举），
    需要一个固定口径的工具。

判据：
    对每个 `pairs[i]`（提示图标 i ↔ 底图框 bg），用匹配器算相似度。
    自洽 = 分数 >= 0.5。
    干净的标注应该是**双峰**：绝大多数 > 0.9，极少数 < 0.1。
    如果出现大量 0.3~0.7，说明标注有系统性问题。

用法：
    python tools/check_label_2408.py --labeled data/labeled_det2
    python tools/check_label_2408.py --compare data/labeled_det data/labeled_det2
"""
from __future__ import annotations

import argparse
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

from autolabel_2408 import split_prompt_icons      # noqa: E402


def crop(img, box, norm_short=None):
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
    return c


def score_labeled(labeled, pre, samples, norm_short=64, limit=0):
    lab = labeled if os.path.isabs(labeled) else os.path.join(REPO, labeled)
    root = samples if os.path.isabs(samples) else os.path.join(REPO, samples)
    idx = os.path.join(lab, "labels_index.jsonl")
    if not os.path.exists(idx):
        return None
    recs = [json.loads(l) for l in open(idx, encoding="utf-8") if l.strip()]
    if limit:
        recs = recs[:limit]

    scores = []
    n_q = 0
    for r in recs:
        sid = r["sid"]
        bb = r.get("bg_boxes") or []
        pairs = r.get("pairs") or []
        if not bb or not pairs:
            continue
        mp = os.path.join(root, sid, "meta.json")
        ip = os.path.join(root, sid, "main.jpg")
        if not os.path.exists(mp) or not os.path.exists(ip):
            continue
        meta = json.load(open(mp, encoding="utf-8"))
        prs = meta.get("prompts") or []
        if not prs:
            continue
        img = cv2.imdecode(np.fromfile(ip, np.uint8), cv2.IMREAD_COLOR)
        pr = cv2.imdecode(np.fromfile(os.path.join(root, sid, prs[0]["file"]),
                                      np.uint8), cv2.IMREAD_COLOR)
        if img is None or pr is None:
            continue
        icons = split_prompt_icons(pr)
        if not icons:
            continue
        ic_c = [crop(it["img"], (0, 0, it["img"].shape[1], it["img"].shape[0]),
                     norm_short) for it in icons]
        bx_c = [crop(img, b, norm_short) for b in bb]
        n_q += 1
        for p in pairs:
            i, k = p["prompt"], p["bg"]
            if i >= len(ic_c) or ic_c[i] is None:
                continue
            if k >= len(bx_c) or bx_c[k] is None:
                continue
            scores.append(float(pre.predict_pair(ic_c[i], bx_c[k])))

    if not scores:
        return None
    a = np.array(scores)
    return {
        "labeled": labeled, "questions": n_q, "pairs": len(a),
        "consistent": int((a >= 0.5).sum()),
        "consistency": round(float((a >= 0.5).mean()), 4),
        "mean": round(float(a.mean()), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p50": round(float(np.percentile(a, 50)), 4),
        "low_ratio": round(float((a < 0.5).mean()), 4),
        # 双峰程度：高分与低分的间隔
        "bimodal_gap": round(float(np.percentile(a, 75) -
                                   np.percentile(a, 25)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/labeled_det2")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="多个标注目录，一次对比")
    ap.add_argument("--pre", default="runs/export_v3/pre_model_v7_v3.onnx")
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--norm-short", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="out/labelcheck2408.json")
    args = ap.parse_args()

    from src.utils import ver_onnx
    pre = ver_onnx.PreONNX(os.path.join(REPO, args.pre))

    targets = args.compare if args.compare else [args.labeled]
    results = []
    print(f"{'标注目录':<24} {'题数':>5} {'配对':>6} {'自洽率':>8} "
          f"{'mean':>7} {'p10':>7} {'双峰间隔':>9}")
    print("-" * 76)
    for t in targets:
        r = score_labeled(t, pre, args.samples, args.norm_short, args.limit)
        if r is None:
            print(f"{t:<24}  (无数据)")
            continue
        results.append(r)
        print(f"{t:<24} {r['questions']:>5} {r['pairs']:>6} "
              f"{r['consistency']:>7.1%} {r['mean']:>7.3f} {r['p10']:>7.3f} "
              f"{r['bimodal_gap']:>9.3f}")

    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n报告 -> {os.path.relpath(out, REPO)}")

    if len(results) >= 2:
        a, b = results[0], results[-1]
        print(f"\n对比 {a['labeled']} -> {b['labeled']}:")
        print(f"  自洽率  {a['consistency']:.1%} -> {b['consistency']:.1%} "
              f"({(b['consistency']-a['consistency'])*100:+.1f} pt)")
        print(f"  题数    {a['questions']} -> {b['questions']} "
              f"({b['questions']-a['questions']:+d})")
        print(f"  p10     {a['p10']:.3f} -> {b['p10']:.3f}")


if __name__ == "__main__":
    main()
