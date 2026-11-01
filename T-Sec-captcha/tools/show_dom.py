# -*- coding: utf-8 -*-
"""show_dom: 打印 probe_captcha_dom.py 采集到的 DOM 节点（含 id/cls）。

用法：
    python tools/show_dom.py
    python tools/show_dom.py --json out/probe_captcha/dom.json --filter refresh
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/probe_captcha/dom.json")
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()

    p = args.json if os.path.isabs(args.json) else os.path.join(REPO, args.json)
    if not os.path.exists(p):
        raise SystemExit(f"缺文件: {p}（先跑 tools/probe_captcha_dom.py）")
    d = json.load(open(p, encoding="utf-8"))
    nodes = d.get("nodes") or d
    print(f"共 {len(nodes)} 节点\n")
    print(f"{'tag':6s} {'rect':26s} {'id':34s} cls")
    print("-" * 110)
    for n in nodes:
        ident = n.get("id") or ""
        cls = (n.get("cls") or "")[:60]
        line = f"{n['tag']:6s} {str(n.get('rect')):26s} {ident[:32]:34s} {cls}"
        if args.filter and args.filter.lower() not in line.lower():
            continue
        print(line)
        bg = n.get("bg") or n.get("src") or ""
        if bg:
            print(f"       bg/src: {bg[:100]}")


if __name__ == "__main__":
    main()
