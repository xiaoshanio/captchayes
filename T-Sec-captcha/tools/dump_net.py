# -*- coding: utf-8 -*-
"""dump_net: 打印登录触发验证码时**所有** turing/qcloud 相关请求与响应体开头。

目的：
    probe_popup_session 里只拿到了 3 个响应，且 prehandle JSON 解析失败。
    需要看清：
        1) 究竟有哪些 URL（prehandle 可能带不同参数/域名）
        2) 响应体是 JSON、JSONP 还是别的包装
        3) 哪张图是 sprite（含提示图），哪张是 bg

用法：
    python tools/dump_net.py --username X --password Y
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "out", "dump_net")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://www.cun.ai/sign-in")
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    from cloakbrowser import launch
    from probe_popup_session import close_banners

    browser = launch(headless=not args.headed, humanize=True,
                     locale="zh-CN", timezone="Asia/Shanghai")
    try:
        ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                                  locale="zh-CN", timezone_id="Asia/Shanghai")
        page = ctx.new_page()
        recs = []

        def on_resp(r):
            u = r.url
            if any(k in u for k in ("turing.captcha", "cap_union", "getcapbysig",
                                    "qcloud.com")):
                recs.append(r)

        page.on("response", on_resp)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3200)
        close_banners(page)
        page.locator("input[name='username']").first.fill(args.username, timeout=5000)
        page.locator("input[name='password']").first.fill(args.password, timeout=5000)
        for sel in ["[role='checkbox']", "input[type='checkbox']"]:
            try:
                el = page.locator(sel).first
                if el.count():
                    el.click(timeout=3000)
                    break
            except Exception:
                pass
        for sel in ["button[type='submit']", "button:has-text('登录')"]:
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    el.click(timeout=6000)
                    break
            except Exception:
                pass
        # 等弹层 + 再等图片
        for _ in range(30):
            try:
                if page.locator(".tencent-captcha__transform").first.is_visible():
                    break
            except Exception:
                pass
            page.wait_for_timeout(400)
        page.wait_for_timeout(2500)

        print(f"\n=== 共 {len(recs)} 个相关响应 ===")
        idx = 0
        for r in recs:
            idx += 1
            u = r.url
            short = u.split("?")[0]
            q = u.split("?", 1)[1][:120] if "?" in u else ""
            ct = ""
            try:
                ct = r.headers.get("content-type", "")
            except Exception:
                pass
            try:
                body = r.body()
                blen = len(body)
            except Exception as e:
                body, blen = b"", -1
            print(f"\n[{idx}] {short}")
            print(f"    query: {q}")
            print(f"    ct={ct} len={blen}")

            if blen > 0 and ("prehandle" in u or "json" in ct or "javascript" in ct):
                try:
                    txt = body.decode("utf-8", "replace")
                except Exception:
                    txt = ""
                print(f"    body[:400]: {txt[:400]}")
                p = os.path.join(OUT, f"resp_{idx}.txt")
                open(p, "w", encoding="utf-8").write(txt)
            elif blen > 1000:
                ext = ".jpg" if body[:2] == b"\xff\xd8" else \
                      ".png" if body[:8] == b"\x89PNG\r\n\x1a\n" else ".bin"
                p = os.path.join(OUT, f"resp_{idx}{ext}")
                open(p, "wb").write(body)
                print(f"    -> saved {p}")

        # 同时把弹层 header 的 HTML dump 出来（看提示图怎么渲染）
        html = page.evaluate(r"""
        () => {
          const w = document.querySelector('.tencent-captcha-dy__header-wrap');
          return w ? w.outerHTML : '(no header)';
        }
        """)
        p = os.path.join(OUT, "header.html")
        open(p, "w", encoding="utf-8").write(html or "")
        print(f"\nheader HTML -> {p}")
        print((html or "")[:1200])
    finally:
        try:
            browser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
