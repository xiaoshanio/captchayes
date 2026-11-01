# -*- coding: utf-8 -*-
"""probe_listen: 验证 DrissionPage 的 listen 能否抓到 verify 响应体。

为什么需要单独验证：
    给 1.0 打的补丁加了 targets 包含 cap_union_new_verify，
    但实测日志里没有出现「应答码 errorCode=...」，说明没读到。

    必须分清两种可能：
        (a) target 没匹配上（DrissionPage 匹配规则问题）
        (b) 匹配上了但 p.response.body 对 JSON 返回空
            （1.0 的 _packet_body_bytes 只认图片，JSON 会返回 None）

本脚本把「收到哪些包」「body 能不能取到」全打印出来。

用法：
    python tools/probe_listen.py --entry https://www.cun.ai/sign-in \
        --username X --password Y
"""
from __future__ import annotations

import argparse
import os
import sys

PROJ_10 = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"
OUT = r"D:\Github\captchayes\T-Sec-captcha\out\probe_listen"


def fill_form(page, username, password):
    if username:
        try:
            e = page.ele("@name=username", timeout=2)
            if e:
                e.input(username)
        except Exception:
            pass
    if password:
        try:
            e = page.ele("@name=password", timeout=2)
            if e:
                e.input(password)
        except Exception:
            pass
    try:
        e = page.ele("@type=checkbox", timeout=2)
        if e:
            e.click()
    except Exception:
        pass


def click_trigger(page):
    for txt in ("登录", "登 录", "注册"):
        try:
            for el in page.eles(f"text:{txt}"):
                if el and el.states.is_displayed:
                    el.click()
                    print(f"点击「{txt}」")
                    return True
        except Exception:
            continue
    return False


def describe_body(packet):
    try:
        body = packet.response.body
    except Exception as e:
        return f"ERR:{type(e).__name__}"
    if body is None:
        return "None"
    if isinstance(body, (bytes, bytearray)):
        return f"bytes[{len(body)}]"
    return f"{type(body).__name__}[{len(str(body))}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="https://www.cun.ai/sign-in")
    ap.add_argument("--username", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    for p in (PROJ_10, os.path.join(PROJ_10, "captcha")):
        sys.path.insert(0, p)

    from DrissionPage import ChromiumOptions, ChromiumPage
    from captcha.drission_verify import DrissionVerifier

    co = ChromiumOptions()
    co.auto_port()
    if not args.headed:
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
        fill_form(page, args.username, args.password)
        if not click_trigger(page):
            print("找不到触发按钮")
            return
        page.wait(3.0)

        targets = ["prehandle", "getcapbysig", "cap_union_new_verify"]
        print("订阅 targets = " + str(targets))
        page.listen.start(targets=targets)

        v = DrissionVerifier(page, artifact_dir=OUT, max_rounds=2)
        try:
            ok = v.ensure_verified()
            print("ensure_verified -> " + str(ok))
        except Exception as e:
            print("验证异常: " + type(e).__name__ + ": " + str(e))

        try:
            steps = page.listen.steps()
        except Exception as e:
            steps = None
            print("取 steps 失败: " + str(e))

        packets = []
        if steps:
            if isinstance(steps, (list, tuple)):
                packets = list(steps)
            else:
                packets = [steps]

        print("")
        print("共收到 " + str(len(packets)) + " 个包：")
        for i, p in enumerate(packets):
            try:
                url = str(p.url or "")
            except Exception:
                url = "?"
            mark = " ★VERIFY" if "cap_union_new_verify" in url else ""
            print("  [" + str(i) + "] body=" + describe_body(p) + mark)
            print("       " + url[:100])

        for p in packets:
            try:
                u = str(p.url or "")
            except Exception:
                continue
            if "cap_union_new_verify" not in u:
                continue
            try:
                b = p.response.body
            except Exception as e:
                print("")
                print("取 verify body 失败: " + type(e).__name__ + ": " + str(e))
                continue
            print("")
            print("verify body 类型=" + type(b).__name__)
            if isinstance(b, (bytes, bytearray)):
                print("  文本: " + bytes(b).decode("utf-8", "replace")[:400])
            else:
                print("  内容: " + str(b)[:400])

        page.listen.stop()
    finally:
        try:
            page.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
