# -*- coding: utf-8 -*-
"""analyze_collect: 分析天御 verify 的 collect 参数，判断纯 HTTP 可行性。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
现状
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
纯 HTTP 提交 verify 时只带 aid/sess/answer → 返回 errorCode=202。
真实浏览器只 POST 一个 `collect=<长加密串>`，里面封装了 sess/answer/
行为数据（鼠标轨迹、时间戳等）。

所以关键问题是：collect 是**可离线复现的编码**，还是**服务端密钥加密**？

判断依据：
    1) 长度是否固定（固定 → 定长加密/哈希；变化 → 变长序列化）
    2) 解码后是否为 base64 → 明文可读
    3) 是否含时间戳/鼠标轨迹等高熵内容

用法：
    python tools/analyze_collect.py
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.parse
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def try_decode(s: str):
    """尝试多层解码，打印每层结果。"""
    print(f"  原始长度: {len(s)}")
    print(f"  前缀: {s[:40]}…")

    # 1) URL 解码
    u = urllib.parse.unquote(s)
    print(f"  URL 解码后长度: {len(u)}  前缀: {u[:48]}")

    # 2) base64
    for name, raw in (("URL解码后", u), ("原始", s)):
        pad = raw + "=" * (-len(raw) % 4)
        try:
            b = base64.b64decode(pad, validate=False)
        except Exception as e:
            print(f"  base64({name}) 失败: {type(e).__name__}")
            continue
        print(f"  base64({name}) 成功: {len(b)} 字节  head={b[:32]!r}")
        # 3) zlib / gzip
        for tag, fn in (("zlib", zlib.decompress),
                        ("gzip", lambda x: zlib.decompress(x, 16 + zlib.MAX_WBITS))):
            try:
                d = fn(b)
                print(f"    {tag} 解压成功: {len(d)} 字节")
                print(f"    {tag} 明文: {d[:200]!r}")
                return d
            except Exception:
                pass
        # 4) 看是否本身是可读文本
        try:
            t = b.decode("utf-8")
            if sum(c.isprintable() for c in t[:80]) > 60:
                print(f"    可读文本: {t[:200]!r}")
                return b
        except Exception:
            pass


def main():
    p = os.path.join(REPO, "out", "verify_tap", "verify_log.jsonl")
    if not os.path.exists(p):
        raise SystemExit(f"缺文件 {p}（先跑 tools/tap_verify.py）")
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

    collects = []
    for r in recs:
        for t in r["verify"]:
            pd = t.get("post_head") or ""
            m = re.search(r"collect=([^&]+)", pd)
            if m:
                collects.append(urllib.parse.unquote(m.group(1)))

    print(f"共收集到 {len(collects)} 个 collect 样例\n")
    if not collects:
        raise SystemExit("没有 collect 样例")

    lens = [len(c) for c in collects]
    print(f"长度统计: min={min(lens)} max={max(lens)} "
          f"mean={sum(lens) / len(lens):.0f}")
    print(f"  长度集合: {sorted(set(lens))[:12]}")
    print()

    print("=== 解码尝试（取最长的一个）===")
    s = max(collects, key=len)
    try_decode(s)

    print("\n=== 长度是否随答案变化（关键判断）===")
    print("  若长度随不同题/不同答案变化 → 变长序列化（可能可复现）")
    print("  若固定长度 → 定长加密（需密钥，纯 HTTP 难）")
    for i, c in enumerate(collects[:6]):
        print(f"  样例{i}: 长度={len(c)}  head={c[:24]}…")


if __name__ == "__main__":
    main()
