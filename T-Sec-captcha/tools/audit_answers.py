# -*- coding: utf-8 -*-
"""audit_answers: 逐题核对模型答案，并对比「成功 vs 失败」的差异。

为什么要做这件事：
    闭环点击已做到 100% 接收（62/62），点击时序不再是瓶颈。
    但仍约 70% 失败，且失败表现是「判错、弹层仍在」。
    必须回答：失败的那些题，模型答案是不是真的错了？

    做法：把每次 2408 的合成图 + 模型选中框画出来，
    人工/半自动核对「提示条顺序」与「选中图标」是否一致。

补充：同时统计「提示图标数 vs 检出 class2 数」是否吻合 ——
    不吻合说明答案不完整，这是最可能的系统性问题。

用法：
    python tools/audit_answers.py --report out/login10x/report_XXXX.json
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


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None)
    ap.add_argument("--max-cases", type=int, default=6)
    args = ap.parse_args()

    if args.report:
        p = args.report if os.path.isabs(args.report) else os.path.join(REPO, args.report)
    else:
        p = sorted(glob.glob(os.path.join(REPO, "out", "login10x",
                                          "report_*.json")))[-1]
    d = json.load(open(p, encoding="utf-8"))
    run_id = os.path.basename(p).replace("report_", "").replace(".json", "")
    base = os.path.join(REPO, "out", "login10x", run_id)

    from src.captcha import TextSelectCaptcha
    cap = TextSelectCaptcha()
    cap.yolo.conf_threshold = 0.25

    out = os.path.join(REPO, "out", "audit", run_id)
    os.makedirs(out, exist_ok=True)

    rows = []
    shown_ok = 0
    shown_bad = 0
    for r in d.get("runs", []):
        tid = r.get("attempt")
        tdir = os.path.join(base, f"try{tid:02d}")
        for img in sorted(glob.glob(os.path.join(tdir, "c2408_*.jpg"))):
            stem = os.path.splitext(os.path.basename(img))[0]
            idx = stem.split("_")[-1]
            im = imread_u(img)
            if im is None:
                continue
            try:
                res = cap.run_dict(img)
            except Exception:
                continue
            corps = res.get("corp") or []
            # 检测数量（判断提示图标数是否吻合）
            det = cap.yolo.inference(im)
            n0 = sum(1 for x in det if int(x[5]) == 0)
            n2 = sum(1 for x in det if int(x[5]) == 2)
            rows.append({"try": tid, "idx": idx, "ok": bool(r.get("ok")),
                         "n_click": len(corps), "n_class0": n0,
                         "n_class2": n2})

            vis = im.copy()
            for i, c in enumerate(corps):
                x1, y1 = int(c["x1"]), int(c["y1"])
                x2, y2 = int(c["x2"]), int(c["y2"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, str(i + 1), (x1 + 2, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                            cv2.LINE_AA)
            tag = "OK" if r.get("ok") else "FAIL"
            # 只为前若干张出图，避免刷屏
            if r.get("ok") and shown_ok < args.max_cases:
                shown_ok += 1
            elif not r.get("ok") and shown_bad < args.max_cases:
                shown_bad += 1
            else:
                continue
            dst = os.path.join(out, f"{tag}_t{tid:02d}_{idx}.jpg")
            cv2.imencode(".jpg", vis)[1].tofile(dst)

    print(f"共 {len(rows)} 道 2408，人工核对图 -> {os.path.relpath(out, REPO)}")
    print(f"{'try':>4s} {'idx':>4s} {'ok':>5s} {'clicks':>6s} "
          f"{'class0':>6s} {'class2':>6s} {'吻合':>4s}")
    print("-" * 52)
    mismatch = 0
    for x in rows:
        match = "Y" if x["n_click"] == x["n_class2"] else "N"
        if match == "N":
            mismatch += 1
        print(f"{x['try']:>4d} {x['idx']:>4s} {str(x['ok']):>5s} "
              f"{x['n_click']:>6d} {x['n_class0']:>6d} {x['n_class2']:>6d} "
              f"{match:>4s}")
    if rows:
        okn = sum(1 for x in rows if x["ok"])
        print(f"\n成功 {okn}/{len(rows)}")
        print(f"点击数与提示图标数不吻合: {mismatch}/{len(rows)}")


if __name__ == "__main__":
    main()
