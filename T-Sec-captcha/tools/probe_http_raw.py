# -*- coding: utf-8 -*-
"""probe_http_raw: 打印 prehandle 的原始响应，找出解析失败的真正原因。

背景：
    之前 tools/probe_http_captcha.py 连跑 20 次全 JSONDecodeError，
    但更早（第 4 轮）用 requests 直连 prehandle 是成功的。
    说明现在响应形态变了 —— 必须看原始字节，不能靠猜。

用法：
    python tools/probe_http_raw.py
"""
from __future__ import annotations

import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "out", "http_captcha")

CAP = "https://turing.captcha.qcloud.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


def main():
    os.makedirs(OUT, exist_ok=True)
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Referer": "https://www.cun.ai/sign-up",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    q = {
        "aid": "195317439", "protocol": "https", "accver": "1",
        "showtype": "popup", "ua": UA, "noheader": "1", "fb": "1",
        "aged": "0", "enableAged": "1", "enableDarkMode": "0",
        "grayscale": "1", "dyeid": "1", "clientype": "1",
        "cap_cd": "", "uid": "", "lang": "zh-cn",
        "entry_url": "https://www.cun.ai/sign-up", "elder_captcha": "0",
        "login_appid": "", "wb": "2", "subsid": "1",
    }
    r = s.get(f"{CAP}/cap_union_prehandle", params=q, timeout=20)
    print(f"HTTP {r.status_code}")
    print(f"content-type: {r.headers.get('content-type')}")
    print(f"content-length: {r.headers.get('content-length')}")
    print(f"实际字节数: {len(r.content)}")
    print(f"encoding: {r.encoding}   apparent: {r.apparent_encoding}")
    print("\n--- 原始前 600 字节（repr）---")
    print(repr(r.content[:600]))
    print("\n--- 按 utf-8 解码前 600 字符 ---")
    try:
        txt = r.content.decode("utf-8", "replace")
        print(txt[:600])
    except Exception as e:
        print(f"解码失败: {e}")

    p = os.path.join(OUT, "prehandle_raw.bin")
    open(p, "wb").write(r.content)
    print(f"\n原始响应 -> {os.path.relpath(p, REPO)}")

    # 尝试几种解析方式
    print("\n--- 解析尝试 ---")
    txt = r.content.decode("utf-8", "replace").strip()
    for name, fn in (
        ("json.loads 直解", lambda t: json.loads(t)),
        ("剥外层括号", lambda t: json.loads(t[t.find("(") + 1:t.rfind(")")])),
        ("取首尾花括号", lambda t: json.loads(t[t.find("{"):t.rfind("}") + 1])),
    ):
        try:
            j = fn(txt)
            print(f"  {name}: OK  subcapclass={j.get('subcapclass')}")
        except Exception as e:
            print(f"  {name}: {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
