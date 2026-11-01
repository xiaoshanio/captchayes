# -*- coding: utf-8 -*-
"""show_report: 打印 login_10x 报告摘要与每轮明细。

用法：
    python tools/show_report.py
    python tools/show_report.py --file out/login10x/report_XXXX.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--events", action="store_true", help="打印每轮事件")
    args = ap.parse_args()

    if args.file:
        p = args.file if os.path.isabs(args.file) else os.path.join(REPO, args.file)
    else:
        cands = sorted(glob.glob(os.path.join(REPO, "out", "login10x",
                                              "report_*.json")))
        if not cands:
            raise SystemExit("没有报告文件")
        p = cands[-1]

    d = json.load(open(p, encoding="utf-8"))
    print(f"报告: {os.path.relpath(p, REPO)}")
    print(f"账号: {d.get('username')}  目标次数: {d.get('times')}")
    print(f"开始: {d.get('started')}  结束: {d.get('finished', '-')}")
    print()
    print(f"{'#':>2s} {'ok':>4s} {'kinds':<34s} {'ms':>8s}  note")
    print("-" * 78)
    for r in d.get("runs", []):
        kinds = ",".join(r.get("kinds") or [])
        print(f"{r.get('attempt'):>2} {str(r.get('ok')):>4s} {kinds:<34s} "
              f"{r.get('ms', 0):>8}  {r.get('note', '')}")

    s = d.get("summary") or {}
    print()
    print("=== 汇总 ===")
    print(f"  尝试次数      : {s.get('attempts')}")
    print(f"  登录成功      : {s.get('success')}  ({s.get('success_rate')})")
    print(f"  验证码通过    : {s.get('captcha_passed_any')}  "
          f"({s.get('captcha_pass_rate')})")
    print(f"  首题题型分布  : {s.get('first_kind_counts')}")
    print(f"  各题型成功数  : {s.get('first_kind_success')}")
    print(f"  平均耗时      : {s.get('ms_mean')} ms")

    if args.events:
        print()
        for r in d.get("runs", []):
            print(f"\n--- #{r.get('attempt')} ok={r.get('ok')} ---")
            for e in r.get("events") or []:
                ev = e.get("ev")
                rest = {k: v for k, v in e.items() if k != "ev"}
                print(f"  {ev}: {json.dumps(rest, ensure_ascii=False)[:200]}")


if __name__ == "__main__":
    main()
