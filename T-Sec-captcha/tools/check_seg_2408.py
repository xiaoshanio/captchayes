# -*- coding: utf-8 -*-
"""check_seg_2408: 检查单题分割结果（调试用）。

用法：
    python tools/check_seg_2408.py --sid t2408_xxx
    python tools/check_seg_2408.py --sid t2408_xxx --draw
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

from seg_2408 import (segment_icons_2408, segment_icons_auto,  # noqa: E402
                      imread_u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True)
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--labeled", default="data/labeled2408v2")
    ap.add_argument("--out", default="out/segcheck")
    ap.add_argument("--draw", action="store_true", default=True)
    args = ap.parse_args()

    root = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    d = os.path.join(root, args.sid)
    bg = imread_u(os.path.join(d, "bg.jpg"))
    if bg is None:
        raise SystemExit(f"读不到 {d}/bg.jpg")
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    a, da = segment_icons_2408(bg, adaptive=False)
    b, db = segment_icons_auto(bg, min_expected=4)

    print(f"[纯黑子域] {len(a)} 框  (原始连通域 {da['n_boxes_raw']})")
    for x in a:
        print(f"   box={x['box']} ink={x['ink']} core={x.get('core')} "
              f"p25={x.get('p25')} chroma={x.get('chroma_med')}")
    print(f"[auto]     {len(b)} 框  (fallback_from={db.get('fallback_from')})")
    for x in b:
        print(f"   box={x['box']} ink={x['ink']} core={x.get('core')}")

    # 标注结果
    idx = os.path.join(REPO, args.labeled, "labels_index.jsonl")
    if os.path.exists(idx):
        for line in open(idx, encoding="utf-8"):
            r = json.loads(line)
            if r["sid"] == args.sid:
                print(f"\n[标注] bg_boxes={r.get('bg_boxes')}")
                print(f"       pairs={r.get('pairs')}")
                print(f"       n_prompt={r.get('n_prompt')} "
                      f"n_bg_icons={r.get('n_bg_icons')}")
                break

    if args.draw:
        vis = bg.copy()
        for i, x in enumerate(a):
            x1, y1, x2, y2 = x["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"A{i}", (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        vis2 = bg.copy()
        for i, x in enumerate(b):
            x1, y1, x2, y2 = x["box"]
            cv2.rectangle(vis2, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(vis2, f"B{i}", (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)
        both = np.concatenate([vis, np.full((bg.shape[0], 6, 3), 255, np.uint8),
                               vis2], axis=1)
        p = os.path.join(out, args.sid + ".jpg")
        cv2.imencode(".jpg", both)[1].tofile(p)
        print(f"\n可视化 -> {os.path.relpath(p, REPO)}")


if __name__ == "__main__":
    main()
