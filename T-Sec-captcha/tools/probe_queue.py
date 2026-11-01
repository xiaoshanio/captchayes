# -*- coding: utf-8 -*-
"""probe_queue: 诊断 DrissionPage 监听队列里到底有没有 verify 包。

背景：
    补丁修好了「卡死」，但 read_verify_code() 仍返回 None，
    日志里没有出现「应答码 errorCode=...」。必须分清：

        (a) verify 包根本没进队列（targets 没匹配上）
        (b) 进队列了但 response.body 取不到（JSON 响应没被缓存）

做法：验证结束后，直接把 queue 里的每个包的 url / body 状态打出来。

用法：
    python tools/probe_queue.py --entry https://www.cun.ai/sign-in \
        --username X --password Y
"""
from __future__ import annotations

import argparse
import os
import sys

PROJ_10 = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"
OUT = r"D:\Github\captchayes\T-Sec-captcha\out\probe_queue"


def dump_queue(page, tag):
    print("")
    print("--- 队列快照 [" + tag + "] ---")
    lst = None
    try:
        lst = page.listen
    except Exception as e:
        print("  取 listen 失败: " + type(e).__name__)
        return
    # 队列对象
    q = getattr(lst, "_caught", None)
    print("  _caught 类型: " + str(type(q)))
    if q is None:
        return
    items = []
    try:
        items = list(getattr(q, "queue", []))
    except Exception as e:
        print("  读 queue 失败: " + type(e).__name__)
    print("  qsize=" + str(q.qsize()) + "  窥视到 " + str(len(items)) + " 个")
    for i, p in enumerate(items):
        url = ""
        try:
            url = str(getattr(p, "url", "") or "")
        except Exception:
            pass
        if not url:
            try:
                url = str(p.request.url or "")
            except Exception:
                url = "?"
        bodystate = "?"
        try:
            b = p.response.body
            if b is None:
                bodystate = "None"
            elif isinstance(b, (bytes, bytearray)):
                bodystate = "bytes[" + str(len(b)) + "]"
            else:
                bodystate = type(b).__name__ + "[" + str(len(str(b))) + "]"
        except Exception as e:
            bodystate = "ERR:" + type(e).__name__
        mark = " ★VERIFY" if "cap_union_new_verify" in url else ""
        print("    [" + str(i) + "] body=" + bodystate + mark)
        print("         " + url[:110])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="https://www.cun.ai/sign-in")
    ap.add_argument("--username", default="")
    ap.add_argument("--password", default="")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    for p in (PROJ_10, os.path.join(PROJ_10, "captcha")):
        sys.path.insert(0, p)

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

        targets = ["prehandle", "getcapbysig", "cap_union_new_verify"]
        print("订阅 targets = " + str(targets))
        page.listen.start(targets=targets)

        v = DrissionVerifier(page, artifact_dir=OUT, max_rounds=2)
        try:
            ok = v.ensure_verified()
            print("ensure_verified -> " + str(ok))
        except Exception as e:
            print("验证异常: " + type(e).__name__ + ": " + str(e))

        dump_queue(page, "验证后立即")
        page.wait(1.5)
        dump_queue(page, "再等1.5s")

        page.listen.stop()
    finally:
        try:
            page.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
