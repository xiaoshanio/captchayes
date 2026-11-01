# -*- coding: utf-8 -*-
"""probe_screen: 试出「让 screen 尺寸正常化」的可行手段。

背景：
    tools/fingerprint_audit.py 实测：无头下 `screen.width x height` = 800x600，
    这是反爬极易识别的特征（真实用户普遍 1366×768 以上）。
    加 `--window-size=1920,1080` 只改了 outer 窗口尺寸，screen 仍是 800x600，
    因为 headless 的 screen 来自虚拟显示器。

本脚本对比几种手段，看哪种能让 screen 变成正常值：
    A) 只加 --window-size
    B) + CDP Emulation.setDeviceMetricsOverride（含 screenWidth/screenHeight）
    C) 走非无头（headless=False，用真实显示器）

用法：
    python tools/probe_screen.py
"""
from __future__ import annotations

import sys

PROJ_10 = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"
for p in (PROJ_10, PROJ_10 + r"\captcha"):
    if p not in sys.path:
        sys.path.insert(0, p)

from DrissionPage import ChromiumOptions, ChromiumPage  # noqa: E402

JS = "return screen.width + 'x' + screen.height + ' dpr=' + devicePixelRatio"


def make(headless=True, window="1920,1080"):
    co = ChromiumOptions()
    co.auto_port()
    if headless:
        co.headless()
    co.set_argument("--window-size=" + window)
    co.set_argument("--no-sandbox")
    return ChromiumPage(co)


def probe(label, headless=True):
    page = None
    try:
        page = make(headless=headless)
        page.get("https://www.cun.ai/sign-in")
        page.wait.doc_loaded(timeout=30)
        page.wait(1.0)
        before = page.run_js(JS)
        print(f"[{label}] 启动后 screen = {before}")

        # 尝试 CDP 覆盖
        try:
            page.run_cdp("Emulation.setDeviceMetricsOverride",
                         width=1920, height=1080, deviceScaleFactor=1,
                         mobile=False, screenWidth=1920, screenHeight=1080)
            after = page.run_js(JS)
            print(f"[{label}] CDP 覆盖后 = {after}")
        except Exception as e:
            print(f"[{label}] CDP 覆盖失败: {type(e).__name__}: {e}")
        return True
    except Exception as e:
        print(f"[{label}] 失败: {type(e).__name__}: {e}")
        return False
    finally:
        if page is not None:
            try:
                page.quit()
            except Exception:
                pass


def main():
    print("=== A/B: 无头 + window-size + CDP 覆盖 ===")
    probe("无头")

    print("")
    print("=== C: 非无头（真实显示器）===")
    probe("有头", headless=False)


if __name__ == "__main__":
    main()
