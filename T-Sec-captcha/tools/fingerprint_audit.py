# -*- coding: utf-8 -*-
"""fingerprint_audit: 审计浏览器指纹，找出「被识别为自动化」的证据。

════════════════════════════════════════════════════════════════
为什么做这个
════════════════════════════════════════════════════════════════
之前的判断错了：把「您的操作过于频繁」当成人机判定结论。
实际上那是**限流**（同 IP 请求太多），不是 AI 检测。

真正决定验证码能否通过的，是：
    1) 浏览器指纹是否暴露自动化（webdriver / CDP 残留 / WebGL / 插件）
    2) 伪装是否自洽（UA 与 platform/语言/时区/屏幕是否一致）
    3) IP 信誉（机房 IP、代理、被标记段）

而且有反证：拖动已降到 136px/s（人类 300~800px/s，我们比人慢一倍多），
站点仍报「速度已超过 99% 的用户」——若真按速度算，不该如此。
更像是被判定为非人后的固定话术。

本脚本对比两个引擎的指纹暴露面：
    A) cun.ai 1.0 用的 DrissionPage（ChromiumOptions 默认 + 1.0 的 set_argument）
    B) 本项目 CloakBrowser（主打 stealth）

输出「哪一项暴露了自动化」，作为后续加固清单。

用法：
    python tools/fingerprint_audit.py
    python tools/fingerprint_audit.py --only dp     # 只测 DrissionPage
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "out", "fingerprint")
PROJ_10 = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"

# 一段尽量全的指纹采集脚本
FP_JS = r"""
() => {
  const out = {};
  const n = navigator;

  // --- 最经典的自动化标志 ---
  out.webdriver = n.webdriver;                      // 期望 undefined/false
  out.hasChromeRuntime = typeof window.chrome === 'object'
                          && !!window.chrome.runtime;
  out.pluginsLen = n.plugins ? n.plugins.length : -1;   // 头less 常见 0
  out.mimeTypesLen = n.mimeTypes ? n.mimeTypes.length : -1;
  out.languages = (n.languages || []).join(',');
  out.platform = n.platform;
  out.vendor = n.vendor;
  out.hardwareConcurrency = n.hardwareConcurrency;   // 头less 常见 1~2
  out.deviceMemory = n.deviceMemory;
  out.maxTouchPoints = n.maxTouchPoints;

  // --- UA / 客户端提示 是否自洽 ---
  out.userAgent = n.userAgent;
  out.uaData = null;
  try {
    if (n.userAgentData) {
      out.uaData = {
        brands: (n.userAgentData.brands || []).map(b => b.brand + '/' + b.version),
        mobile: n.userAgentData.mobile,
        platform: n.userAgentData.platform,
      };
    }
  } catch (e) {}

  // --- 渲染层指纹 ---
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl') || c.getContext('experimental-webgl');
    if (gl) {
      const dbg = gl.getExtension('WEBGL_debug_renderer_info');
      out.webglVendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : null;
      out.webglRenderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : null;
    } else {
      out.webglVendor = 'NO_WEBGL';
    }
  } catch (e) { out.webglVendor = 'ERR'; }

  // canvas 指纹（软件渲染/虚拟环境常与真机差异明显）
  try {
    const cv = document.createElement('canvas');
    cv.width = 220; cv.height = 40;
    const ctx = cv.getContext('2d');
    ctx.textBaseline = 'top';
    ctx.font = '14px Arial';
    ctx.fillStyle = '#f60';
    ctx.fillRect(10, 5, 80, 22);
    ctx.fillStyle = '#069';
    ctx.fillText('captcha-fp-测试', 4, 8);
    out.canvasHash = cv.toDataURL().slice(-64);
  } catch (e) { out.canvasHash = 'ERR'; }

  // --- 环境一致性 ---
  out.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  out.tzOffsetMin = new Date().getTimezoneOffset();
  out.screen = screen.width + 'x' + screen.height
               + '@' + (window.devicePixelRatio || 1);
  out.outer = window.outerWidth + 'x' + window.outerHeight;
  out.inner = window.innerWidth + 'x' + window.innerHeight;
  out.colorDepth = screen.colorDepth;

  // --- 权限 / 通知（headless 常见异常值）---
  try { out.notificationPerm = Notification.permission; } catch (e) {}

  // --- CDP / 自动化残留痕迹 ---
  out.cdcKeys = Object.keys(window).filter(k =>
      k.startsWith('cdc_') || k.includes('__cdp') || k.includes('__webdriver'));
  out.docKeys = Object.keys(document).filter(k =>
      k.startsWith('$cdc') || k.includes('webdriver'));

  // --- iframe 里的 webdriver（有的检测走 iframe）---
  try {
    const f = document.createElement('iframe');
    f.style.display = 'none';
    document.body.appendChild(f);
    out.iframeWebdriver = f.contentWindow.navigator.webdriver;
    document.body.removeChild(f);
  } catch (e) {}

  // --- 连接 / 电池 / 媒体设备（真实设备通常有值）---
  try {
    out.connection = n.connection
      ? (n.connection.effectiveType + '/' + n.connection.rtt) : null;
  } catch (e) {}
  out.mediaDevices = !!(n.mediaDevices && n.mediaDevices.enumerateDevices);

  // --- 时区 vs 语言 一致性（代理常见破绽）---
  out.consistency = {
    uaLang: (n.language || ''),
    tz: out.timezone,
    platform: out.platform,
  };
  return out;
}
"""


def audit_drissionpage():
    """审计 cun.ai 1.0 所用的 DrissionPage 浏览器。"""
    for p in (PROJ_10, os.path.join(PROJ_10, "captcha")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from DrissionPage import ChromiumOptions, ChromiumPage

    # 与 test_10_live.py 完全相同的启动参数
    co = ChromiumOptions()
    co.auto_port()
    co.headless()
    co.set_argument("--window-size=1920,1080")
    co.set_argument("--force-device-scale-factor=1")
    co.set_argument("--no-sandbox")
    co.set_argument("--disable-blink-features=AutomationControlled")
    co.set_user_agent(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
    co.set_argument("--lang=zh-CN")

    page = ChromiumPage(co)
    try:
        page.get("https://www.cun.ai/sign-in")
        page.wait.doc_loaded(timeout=30)
        page.wait(1.5)
        fp = page.run_js("return (" + FP_JS + ")()")
        try:
            fp["exitIp"] = page.run_js(
                "return fetch('https://api.ipify.org?format=json')"
                ".then(r=>r.json()).then(j=>j.ip).catch(()=>null)")
        except Exception:
            fp["exitIp"] = None
        return fp
    finally:
        try:
            page.quit()
        except Exception:
            pass


def audit_cloakbrowser():
    """审计本项目 CloakBrowser（stealth 引擎）作为对照。"""
    from cloakbrowser import launch
    browser = launch(headless=True, humanize=True, locale="zh-CN",
                     timezone="Asia/Shanghai")
    try:
        ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                                  locale="zh-CN",
                                  timezone_id="Asia/Shanghai")
        page = ctx.new_page()
        page.goto("https://www.cun.ai/sign-in",
                  wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(1800)
        fp = page.evaluate(FP_JS)
        try:
            fp["exitIp"] = page.evaluate(
                "fetch('https://api.ipify.org?format=json')"
                ".then(r=>r.json()).then(j=>j.ip).catch(()=>null)")
        except Exception:
            fp["exitIp"] = None
        return fp
    finally:
        try:
            browser.close()
        except Exception:
            pass


RISK = [
    ("webdriver", lambda v: v not in (False, None, "false"), "navigator.webdriver 暴露"),
    ("pluginsLen", lambda v: v == 0, "plugins 数量为 0（无头典型特征）"),
    ("hardwareConcurrency", lambda v: isinstance(v, int) and v <= 2,
     "CPU 核心数异常低"),
    ("webglVendor", lambda v: v in ("NO_WEBGL", "ERR", None) or
     (isinstance(v, str) and "SwiftShader" in v), "WebGL 缺失或软件渲染"),
    ("hasChromeRuntime", lambda v: v is False, "window.chrome.runtime 缺失"),
    ("cdcKeys", lambda v: bool(v), "存在 cdc_/_cdp 自动化残留键"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["dp", "cloak"], default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    results = {}

    if args.only in (None, "dp"):
        print("=== 审计 DrissionPage（cun.ai 1.0 用的引擎）===", flush=True)
        try:
            results["drissionpage"] = audit_drissionpage()
            print("  OK", flush=True)
        except Exception as e:
            print(f"  失败: {type(e).__name__}: {e}", flush=True)

    if args.only in (None, "cloak"):
        print("=== 审计 CloakBrowser（对照）===", flush=True)
        try:
            results["cloakbrowser"] = audit_cloakbrowser()
            print("  OK", flush=True)
        except Exception as e:
            print(f"  失败: {type(e).__name__}: {e}", flush=True)

    p = os.path.join(OUT, "fingerprint.json")
    json.dump(results, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    names = list(results.keys())
    print("")
    print("=" * 78)
    print(" 指纹对比")
    print("=" * 78)
    keys = ["webdriver", "hasChromeRuntime", "pluginsLen", "mimeTypesLen",
            "languages", "platform", "hardwareConcurrency", "deviceMemory",
            "maxTouchPoints", "webglVendor", "timezone", "screen", "outer",
            "colorDepth", "notificationPerm", "cdcKeys", "iframeWebdriver",
            "connection", "mediaDevices", "exitIp"]
    w = 26
    print(f"{'字段':<22}" + "".join(f"{n:<{w}}" for n in names))
    print("-" * 78)
    for k in keys:
        row = f"{k:<22}"
        for n in names:
            v = results[n].get(k)
            s = str(v)
            if len(s) > w - 2:
                s = s[:w - 3] + "…"
            row += f"{s:<{w}}"
        print(row)

    print("")
    print("=" * 78)
    print(" 自动化暴露面")
    print("=" * 78)
    for n in names:
        fp = results[n]
        risks = []
        for field, pred, desc in RISK:
            try:
                if pred(fp.get(field)):
                    risks.append(desc)
            except Exception:
                pass
        ua = fp.get("userAgent") or ""
        if "HeadlessChrome" in ua:
            risks.append("UA 含 HeadlessChrome")
        print(f"  {n}: " + ("; ".join(risks) if risks else "未发现明显暴露"))
    print("")
    print(f"明细 -> {os.path.relpath(p, REPO)}")


if __name__ == "__main__":
    main()
