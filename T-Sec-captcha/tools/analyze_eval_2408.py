# -*- coding: utf-8 -*-
"""analyze_eval_2408: 分析验收结果的失败模式（框数不符 / 部分正确 / 无真值）。

用法：
    python tools/analyze_eval_2408.py --eval out/eval2408/new.json \
        --labeled data/labeled_det --show 3
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="out/eval2408/new.json")
    ap.add_argument("--labeled", default="data/labeled_det")
    ap.add_argument("--show", type=int, default=3, help="打印几个案例明细")
    args = ap.parse_args()

    ev = args.eval if os.path.isabs(args.eval) else os.path.join(REPO, args.eval)
    lab = args.labeled if os.path.isabs(args.labeled) else os.path.join(REPO, args.labeled)

    gt = {}
    idx = os.path.join(lab, "labels_index.jsonl")
    for line in open(idx, encoding="utf-8"):
        x = json.loads(line)
        boxes = x.get("bg_boxes") or []
        o = [boxes[p["bg"]] for p in sorted(x.get("pairs") or [],
                                            key=lambda z: z["prompt"])
             if 0 <= p["bg"] < len(boxes)]
        if o:
            gt[x["sid"]] = o

    recs = json.load(open(ev, encoding="utf-8"))
    st = collections.Counter()
    shown = 0
    for rec in recs:
        if not rec.get("ok"):
            st["err"] += 1
            continue
        g = gt.get(rec["sid"])
        c = [[x["x1"], x["y1"], x["x2"], x["y2"]] for x in rec.get("corp") or []]
        if not g:
            st["no_gt"] += 1
            continue
        if len(c) != len(g):
            st[f"count_{len(c)}vs{len(g)}"] += 1
            if shown < args.show:
                shown += 1
                print(f"\n[{rec['sid']}] pred={len(c)} gt={len(g)}")
                for i, x in enumerate(c):
                    best = max([(iou(x, gg), k) for k, gg in enumerate(g)],
                               default=(0, -1))
                    print(f"   pred{i} {x}  best_gt={best[1]} iou={best[0]:.2f}")
                print(f"   gt = {g}")
            continue
        hit = sum(1 for k in range(len(g)) if iou(g[k], c[k]) > 0.5)
        st["exact" if hit == len(g) else f"partial{hit}"] += 1

    print("\n=== 失败模式 ===")
    tot = sum(st.values())
    for k, v in sorted(st.items(), key=lambda z: -z[1]):
        print(f"  {k:16s}: {v:4d}  ({v/max(tot,1):.1%})")
    print(f"\n  总题数: {tot}")
    print(f"  exact : {st.get('exact',0)}/{tot} = "
          f"{st.get('exact',0)/max(tot,1):.1%}")


if __name__ == "__main__":
    main()
