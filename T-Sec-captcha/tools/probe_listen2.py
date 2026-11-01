# -*- coding: utf-8 -*-
"""probe_listen2: 看清 DrissionPage listen 返回的 packet 对象真实结构。

背景：
    probe_listen 只收到 1 个包，且访问 p.url 直接抛 AttributeError，
    说明 1.0 代码里 `str(packet.url or "")` 的写法在当前 DrissionPage
    版本上不成立。必须先看清对象有什么属性，才能正确取 URL 和 body。

做法：
    打印 packet 的类型、__dict__、dir() 里非下划线属性，
    以及 postData / response 的实际类型与可用字段。

用法：
    python tools/probe_listen2.py --entry https://www.cun.ai/sign-in \
        --username X --password Y
"""
from __future__ import annotations

import argparse
import os
import sys

PROJ_10 = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"
OUT = r"D:\Github\captchayes\T-Sec-captcha\out\probe_listen"


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
            clicked = False
            try:
                for el in page.eles(f"text:{txt}"):
                    if el and el.states.is_displayed:
                        el.click()
                        clicked = True
                        break
            except Exception:
                pass
            if clicked:
                print("点击 " + txt)
                break
        page.wait(3.0)

        # 不设 target 限制，全收，便于看清单个包的结构
        print("listen.start(targets=True) —— 全量收集")
        page.listen.start(targets=True)

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
            packets = list(steps) if isinstance(steps, (list, tuple)) else [steps]

        print("")
        print("共收到 " + str(len(packets)) + " 个包")
        for i, p in enumerate(packets[:3]):
            print("")
            print("=== 包 " + str(i) + " ===")
            print("  type: " + str(type(p)))
            attrs = [a for a in dir(p) if not a.startswith("_")]
            print("  attrs: " + str(attrs))
            for a in ("url", "request", "response", "frameId", "target"):
                try:
                    v_ = getattr(p, a)
                    print("  " + a + ": " + str(type(v_)))
                except Exception as e:
                    print("  " + a + ": GET-ERR " + type(e).__name__)

            # request / response 的字段
            for holder in ("request", "response"):
                try:
                    h = getattr(p, holder)
                except Exception:
                    continue
                sub = [a for a in dir(h) if not a.startswith("_")]
                print("  " + holder + " attrs: " + str(sub[:24]))
                for a in ("url", "method", "postData", "body", "status",
                          "headers", "rawBody"):
                    try:
                        val = getattr(h, a)
                        s = str(val)
                        print("    " + holder + "." + a + " = " + s[:120])
                    except Exception as e:
                        print("    " + holder + "." + a + " ERR "
                              + type(e).__name__)

        # 尝试找 verify 包并打印 body
        print("")
        print("=== 查找 verify 包 ===")
        found = 0
        for p in packets:
            url = ""
            for path in ("url",):
                try:
                    url = str(getattr(p, path) or "")
                except Exception:
                    pass
            if not url:
                try:
                    url = str(p.request.url or "")
                except Exception:
                    pass
            if "cap_union_new_verify" in url:
                found += 1
                print("找到 verify: " + url[:120])
                try:
                    b = p.response.body
                    print("  body type=" + type(b).__name__)
                    if isinstance(b, (bytes, bytearray)):
                        print("  文本: " + bytes(b).decode("utf-8",
                                                          "replace")[:400])
                    else:
                        print("  内容: " + str(b)[:400])
                except Exception as e:
                    print("  取 body 失败: " + type(e).__name__ + ": " + str(e))
        print("verify 包数量: " + str(found))

        page.listen.stop()
    finally:
        try:
            page.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
