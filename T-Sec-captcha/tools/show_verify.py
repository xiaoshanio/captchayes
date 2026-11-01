# -*- coding: utf-8 -*-
"""show_verify: 查看已抓到的验证应答明细，定位 errorCode 含义。

用法：
    python tools/show_verify.py
    python tools/show_verify.py --only-fail
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="out/verify_tap/verify_log.jsonl")
    ap.add_argument("--only-fail", action="store_true")
    ap.add_argument("--body", type=int, default=300)
    args = ap.parse_args()

    p = args.file if os.path.isabs(args.file) else os.path.join(REPO, args.file)
    if not os.path.exists(p):
        raise SystemExit(f"缺文件 {p}")
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

    rets = Counter()
    ecs = Counter()
    pair = Counter()
    for r in recs:
        for t in r["verify"]:
            ec = str(t.get("errorCode"))
            rt = str(t.get("ret"))
            rets[rt] += 1
            ecs[ec] += 1
            pair[(rt, ec, r["ok"])] += 1

    print(f"记录 {len(recs)} 次登录")
    print(f"ret       分布: {dict(rets)}")
    print(f"errorCode 分布: {dict(ecs)}")
    print("\n(ret, errorCode, 登录成功) 组合:")
    for k, v in pair.most_common():
        print(f"  ret={k[0]:>5s} errorCode={k[1]:>4s} ok={k[2]}  x{v}")

    print("\n明细:")
    for r in recs:
        if args.only_fail and r["ok"]:
            continue
        ecs_r = [str(t.get("errorCode")) for t in r["verify"]]
        print(f"  #{r['attempt']} ok={r['ok']} kinds={r['kinds']} "
              f"ec={ecs_r}")
        for t in r["verify"]:
            if str(t.get("errorCode")) in ("None", "0"):
                continue
            b = (t.get("body_head") or "").replace("\n", " ")
            pd = (t.get("post_head") or "").replace("\n", " ")
            print(f"     body: {b[:args.body]}")
            print(f"     post: {pd[:220]}")


if __name__ == "__main__":
    main()
