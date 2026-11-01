# -*- coding: utf-8 -*-
"""test_10_live: 用 cun.ai 1.0 自己的验证器实跑一次验证码。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测什么
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
不测 1.0 的完整注册流程（那条路依赖 remail 邮箱服务），
只测**用户真正关心的那一环**：

    打开注册/登录页 → 触发弹层 → DrissionVerifier.ensure_verified()
    → 拿到 errorCode / ticket → 判定 1.0 是否真的可用

这样能把「验证码能不能过」与「邮箱/建号流程」解耦，
结论更干净。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
为什么用 DrissionPage 而不是 CloakBrowser
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.0 的验证器内部用 `page.run_js` / CDP `Network` 抓包
（`start_capture` / `_packets` / `prehandle_texts`），
这些都是 DrissionPage 的 API。换浏览器引擎它就跑不起来，
所以必须用它自己的 page。

用法：
    python tools/test_10_live.py --entry https://www.cun.ai/sign-in \
        --username xxx --password yyy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

TZ = timezone(timedelta(hours=8))
PROJ_10 = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"
OUT = r"D:\Github\captchayes\T-Sec-captcha\out\test_10_live"


def log(m):
    print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="https://www.cun.ai/sign-in")
    ap.add_argument("--username", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--headed", action="store_true", help="显示浏览器窗口")
    ap.add_argument("--max-rounds", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if not os.path.isdir(PROJ_10):
        raise SystemExit(f"1.0 目录不存在: {PROJ_10}")
    sys.path.insert(0, os.path.join(PROJ_10, "captcha"))
    sys.path.insert(0, PROJ_10)

    # ---- 预检：模型是否就绪
    import captcha.captcha_solver as solver
    log(f"模型可用: {solver.models_available()}")
    if not solver.models_available():
        raise SystemExit("1.0 模型缺失，先跑 tools/setup_captcha.py")
    try:
        solver._Models.get()
        log("模型加载 OK")
    except Exception as e:
        raise SystemExit(f"模型加载失败: {type(e).__name__}: {e}")

    # ---- 启动 DrissionPage（1.0 验证器依赖它的 run_js / CDP）
    from DrissionPage import ChromiumOptions, ChromiumPage
    from captcha.drission_verify import (CaptchaError, DrissionVerifier,
                                         CaptchaRateLimitedError)

    # 安装增强补丁：抓 cap_union_new_verify + 应答码判定
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import drission_verify_ext
    drission_verify_ext.install()

    # 滑块桥接：用本项目更好的缺口检测替换 1.0 的实现
    # 实测依据：1.0 在 3 张真实滑块底图上给出 conf=0.3/0.0/0.3（全部放弃），
    # 本项目实现 3/3 精确命中（放大核对 bg_2 缺口中心 x=581，预测 581）
    import slider_gap_bridge
    slider_gap_bridge.install()

    co = ChromiumOptions()
    co.auto_port()
    if not args.headed:
        co.headless()
    # 【指纹】必须显式给窗口/屏幕尺寸。
    # 实测（tools/fingerprint_audit.py）：无头默认 screen=800x600，
    # 这是反爬一眼能识别的特征（真实用户普遍 1366x768 以上）。
    co.set_argument("--window-size=1920,1080")
    co.set_argument("--force-device-scale-factor=1")
    co.set_argument("--no-sandbox")
    co.set_argument("--disable-blink-features=AutomationControlled")
    co.set_user_agent(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
    co.set_argument("--lang=zh-CN")

    log("启动 DrissionPage")
    page = ChromiumPage(co)
    report = {"entry": args.entry, "verify_ok": None, "result": None,
              "error": None}

    try:
        log(f"打开 {args.entry}")
        page.get(args.entry)
        page.wait.doc_loaded(timeout=30)
        # 修正指纹破绽：无头下 screen 默认 800x600，是明显的反爬特征
        # （实测 data：无头 800x600 / 有头 2560x1440，CDP 可修正无头值）
        drission_verify_ext.harden_fingerprint(page,
                                               headless=not args.headed)
        page.wait(2.5)

        # 关营销横幅（点了才知道会不会挡住）
        for txt in ("知道了", "我知道了"):
            try:
                el = page.ele(f"text:{txt}", timeout=1)
                if el:
                    el.click()
                    page.wait(0.6)
                    log(f"关闭横幅: {txt}")
                    break
            except Exception:
                pass

        # 填表单（不填则前端会拦住，不发 verify）
        if args.username:
            for sel in ("@name=username", "tag:input@@type=text"):
                try:
                    el = page.ele(sel, timeout=2)
                    if el:
                        el.clear()
                        el.input(args.username)
                        log(f"填入用户名 ({sel})")
                        break
                except Exception:
                    pass
        if args.password:
            for sel in ("@name=password", "@type=password"):
                try:
                    el = page.ele(sel, timeout=2)
                    if el:
                        el.clear()
                        el.input(args.password)
                        log("填入密码")
                        break
                except Exception:
                    pass
        # 勾协议
        try:
            el = page.ele("@type=checkbox", timeout=2)
            if el:
                el.click()
                page.wait(0.4)
                log("勾选协议")
        except Exception:
            pass

        # 触发弹层
        triggered = False
        for txt in ("登录", "登 录", "注册"):
            try:
                els = page.eles(f"tag:button@@text():{txt}")
                if not els:
                    els = page.eles(f"text:{txt}")
                for el in els:
                    if el and el.states.is_displayed:
                        el.click()
                        triggered = True
                        log(f"点击「{txt}」触发弹层")
                        break
                if triggered:
                    break
            except Exception:
                continue
        if not triggered:
            report["error"] = "no_trigger_button"
            raise SystemExit("找不到可用于触发的按钮")

        page.wait(3.0)

        # ---- 调用 1.0 自己的验证器
        verifier = DrissionVerifier(
            page,
            stage=lambda s, m: log(f"  [stage] {s}: {m}"),
            artifact_dir=OUT,
            max_rounds=args.max_rounds,
            file_tag="live",
        )
        log("调用 DrissionVerifier.ensure_verified()")
        ok = verifier.ensure_verified()
        report["verify_ok"] = bool(ok)
        log(f"ensure_verified 返回: {ok}")

        # 抓一下状态与 URL
        try:
            report["url_after"] = page.url
        except Exception:
            pass
        try:
            report["status"] = verifier.read_status()
        except Exception:
            pass
        page.get_screenshot(path=os.path.join(OUT, "after.png"),
                            full_page=False)
    except CaptchaRateLimitedError as e:
        report["error"] = f"rate_limited: {e}"
        log(f"被限频: {e}")
    except CaptchaError as e:
        report["error"] = f"captcha_error: {e}"
        log(f"验证错误: {e}")
    except SystemExit:
        raise
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
        log(f"异常: {report['error']}")
    finally:
        try:
            page.quit()
        except Exception:
            pass
        p = os.path.join(OUT, "report.json")
        report["finished"] = datetime.now(TZ).isoformat(timespec="seconds")
        json.dump(report, open(p, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        log(f"报告 -> {p}")

        print("\n" + "=" * 60)
        print(f"  验证通过: {report['verify_ok']}")
        if report["error"]:
            print(f"  错误    : {report['error']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
