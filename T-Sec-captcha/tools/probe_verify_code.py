# -*- coding: utf-8 -*-
"""probe_verify_code: 专测「应答码捕获」是否生效。

背景：
    补丁装上后，遇到滑块题时 1.0 自己因「缺口置信不足」不提交，
    所以没有 verify 包，无法验证补丁是否正确读到 errorCode。

    本脚本循环换题直到拿到**点选**题（click），确保会走到提交，
    然后打印补丁读到的 errorCode。

用法：
    python tools/probe_verify_code.py --entry https://www.cun.ai/sign-in \
        --username X --password Y
"""
from __future__ import annotations

import argparse
import os
import sys

PROJ_10 = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"
OUT = r"D:\Github\captchayes\T-Sec-captcha\out\probe_verify_code"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="https://www.cun.ai/sign-in")
    ap.add_argument("--username", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    for p in (PROJ_10, os.path.join(PROJ_10, "captcha")):
        sys.path.insert(0, p)
    # 补丁在 T-Sec-captcha/tools 下
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import drission_verify_ext as ext
    ext.install()

    from DrissionPage import ChromiumOptions, ChromiumPage
    from captcha.drission_verify import DrissionVerifier

    co = ChromiumOptions()
    co.auto_port()
    co.headless()
    co.set_argument("--no-sandbox")
    co.set_user_agent(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

    page = ChromiumPage(co)
    try:
        page.get(args.entry)
        page.wait.doc_loaded(timeout=30)
        page.wait(2.5)
        if args.username:
            try:
                e = page.ele("@name=username", timeout=2)
                if e:
                    e.input(args.username)
            except Exception:
                pass
        if args.password:
            try:
                e = page.ele("@name=password", timeout=2)
                if e:
                    e.input(args.password)
            except Exception:
                pass
        try:
            e = page.ele("@type=checkbox", timeout=2)
            if e:
                e.click()
        except Exception:
            pass
        for txt in ("登录", "注册"):
            hit = False
            try:
                for el in page.eles(f"text:{txt}"):
                    if el and el.states.is_displayed:
                        el.click()
                        hit = True
                        break
            except Exception:
                pass
            if hit:
                print("点击 " + txt)
                break
        page.wait(3.0)

        v = DrissionVerifier(page, artifact_dir=OUT, max_rounds=args.rounds,
                             stage=lambda s, m: print("  [s] " + m))
        ok = v.ensure_verified()
        print("")
        print("ensure_verified -> " + str(ok))

        # 直接问补丁要应答码
        vj = None
        try:
            vj = v.read_verify_code()
        except Exception as e:
            print("read_verify_code 异常: " + type(e).__name__ + ": " + str(e))
        print("")
        print("=== 应答码 ===")
        if isinstance(vj, dict):
            print("  errorCode : " + repr(vj.get("errorCode")))
            print("  ticket    : " + str(vj.get("ticket"))[:40])
            print("  randstr   : " + str(vj.get("randstr"))[:20])
        else:
            print("  未读到（可能本轮未提交，或被限频）")

        # 队列里到底有什么
        pkts = ext._iter_packets(page)
        print("")
        print("队列内 " + str(len(pkts)) + " 个包：")
        for i, p in enumerate(pkts):
            u = ext._packet_url(p)
            mark = " ★VERIFY" if "cap_union_new_verify" in u else ""
            print("  [" + str(i) + "]" + mark + " " + u[:100])
    finally:
        try:
            page.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
