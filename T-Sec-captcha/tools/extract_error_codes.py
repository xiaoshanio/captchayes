# -*- coding: utf-8 -*-
"""extract_error_codes: 从天御 JS 源码里抽取报错码与判定逻辑。

为什么能从源码拿到：
    天御的校验脚本 `tgJCap.*.js` 是公开可下发的（浏览器会加载），
    里面就写着 errorCode 的分支含义。抓下来 grep 即可，
    不需要靠猜或靠抓包猜码。

用法：
    python tools/extract_error_codes.py
    python tools/extract_error_codes.py --file out/dump_net/resp_4.txt
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def ctx(s, i, before=110, after=160):
    a = max(0, i - before)
    b = min(len(s), i + after)
    return s[a:b].replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="out/dump_net/resp_4.txt")
    ap.add_argument("--max", type=int, default=12)
    args = ap.parse_args()

    p = args.file if os.path.isabs(args.file) else os.path.join(REPO, args.file)
    if not os.path.exists(p):
        raise SystemExit(f"缺文件 {p}（先跑 tools/dump_net.py）")
    s = open(p, encoding="utf-8", errors="replace").read()
    print(f"=== {os.path.relpath(p, REPO)}  {len(s)} 字符 ===\n")

    # 1) 校验接口名
    print("--- 校验接口 ---")
    for m in re.finditer(r"cap_union_new_verify", s):
        print(f"  @{m.start()}: ...{ctx(s, m.start(), 80, 120)}...")
    print()

    # 2) errorCode 出现位置
    print("--- errorCode 相关 ---")
    n = 0
    for m in re.finditer(r"errorCode", s):
        n += 1
        print(f"  [{n}] @{m.start()}: ...{ctx(s, m.start())}...")
        if n >= args.max:
            break
    print()

    # 3) 常见码值分支：形如 8===xxx 或 case 8: / e.errcode
    print("--- 可能的码值分支 ---")
    pats = [
        (r"errorCode\"?\s*\]?\s*[=:]{1,2}\s*[\"']?(\d{1,3})", "errorCode=NN"),
        (r"err?code\"?\s*\]?\s*[=:]{1,2}\s*[\"']?(\d{1,3})", "errcode=NN"),
        (r"case\s*(\d{1,3})\s*:", "case NN:"),
    ]
    for pat, label in pats:
        found = re.findall(pat, s)
        if found:
            from collections import Counter
            c = Counter(found)
            print(f"  {label}: {dict(c.most_common(12))}")
    print()

    # 4) 验证成功/失败的提示文案（可反推状态机）
    print("--- 状态文案（含验证字样，取前 20 条）---")
    seen = set()
    cnt = 0
    for m in re.finditer(r"[\"']([^\"']{2,40}(?:验证|失败|重试|通过|错误)[^\"']{0,20})[\"']", s):
        t = m.group(1)
        if t in seen:
            continue
        seen.add(t)
        print(f"  {t}")
        cnt += 1
        if cnt >= 20:
            break


if __name__ == "__main__":
    main()
