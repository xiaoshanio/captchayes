# -*- coding: utf-8 -*-
"""decode_verify_codes: 从天御混淆 JS 里还原 errorCode 的码值含义。

背景：
    tgJCap.*.js 被混淆过，码值不在明面上（都是 _0x5bf64d(0x98d) 这种查表）。
    但码值在源码里会以**字符串字面量**出现（'7' / '1' / '8' …），
    且出现在 errorCode 分支附近。把它们连同上下文抓出来，
    再配合 wording（提示文案），就能反推出码值含义。

用法：
    python tools/decode_verify_codes.py
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    p = os.path.join(REPO, "out", "dump_net", "resp_4.txt")
    if not os.path.exists(p):
        raise SystemExit(f"缺文件 {p}（先跑 tools/dump_net.py）")
    s = open(p, encoding="utf-8", errors="replace").read()

    print("=== 1) errorCode 附近的字符串数字（疑似码值）===")
    allhits = []
    for m in re.finditer(r"errorCode", s):
        seg = s[max(0, m.start() - 600):m.start() + 600]
        nums = re.findall(r"'(\d{1,2})'", seg)
        if nums:
            c = Counter(nums)
            allhits.append((m.start(), c.most_common(6)))
    for pos, nums in allhits:
        print(f"  @{pos}: {nums}")
    print("\n=== 2) 带数字比较的分支（可能是码值判定）===")
    # 形如 '7'===e 或 e==='7'
    pats = [r"'(\d{1,2})'===_?[\w$]+", r"_?[\w$]+==='(\d{1,2})'"]
    cnt = Counter()
    for pat in pats:
        for m in re.finditer(pat, s):
            cnt[m.group(1)] += 1
    print(f"  {dict(cnt.most_common(15))}")

    print("\n=== 3) wording 相关（错误文案 key）===")
    keys = set()
    for m in re.finditer(r"wording ['\"]?[,:=] ?['\"]?([a-z0-9\-]{3,40})", s):
        keys.add(m.group(1))
    for k in sorted(keys)[:40]:
        print(f"  {k}")

    print("\n=== 4) 验证成功后的关键字段 ===")
    for m in re.finditer(r"(randstr|ticket)", s):
        seg = s[max(0, m.start() - 90):m.start() + 90].replace("\n", " ")
        # 只打印含 ret / ok / success 的片段
        if any(k in seg for k in ("ret", "success", "onSuccess")):
            print(f"  ...{seg}...")
            break


if __name__ == "__main__":
    main()
