# -*- coding: utf-8 -*-
"""probe_popup_session: 拦截弹层**自己那次** prehandle 响应，看清题型与提示图来源。

为什么必须拦截网络，而不能另发一次 prehandle：
    实测另发 prehandle 是**另一道题**，与弹层显示的不是同一道。
    只有弹层自己请求的那次响应，才与页面上的图对得上。

要搞清的问题：
    1) 2408 图标点选的「提示图」从哪里来？
       训练版式：bg 672x480 + strip 56，提示图 170x50 贴在 (6,3)。
       弹层里提示图应该是 sprite 里裁出来的（sprite_pos + size_2d）。
    2) 2404 中文文字点选的提示「苍 倡 乘」是纯文字还是 sprite 图？
       如果也是 sprite 裁图，就能用真实字形（比 PIL 渲染准得多）。

用法：
    python tools/probe_popup_session.py --rounds 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

TZ = timezone(timedelta(hours=8))
OUT = os.path.join(REPO, "out", "probe_session")
SEL_POPUP = ".tencent-captcha__transform, #tcaptcha_transform_dy"
SEL_BG = ".tencent-captcha-dy__verify-bg-img"
SEL_TITLE = ".tencent-captcha-dy__header-text"


def log(m):
    print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] {m}", flush=True)


def imwrite_u(p, im):
    ok, buf = cv2.imencode(os.path.splitext(p)[1] or ".png", im)
    if ok:
        buf.tofile(p)


def parse_jsonp(txt):
    t = (txt or "").strip()
    if t.startswith("("):
        t = t[1:]
    if t.endswith(")"):
        t = t[:-1]
    return json.loads(t)


def close_banners(page):
    """关掉页面顶部的营销横幅。

    实测 cun.ai 的横幅每次不一样（"首充继续加赠 50%" / "会员等级全新上线"），
    且会盖住登录按钮。它的关闭按钮是横幅右上角的圆形 ×。
    策略：先试 aria-label/文本，再按横幅容器右上角坐标点。
    """
    for sel in ["button:has-text('知道了')", "button:has-text('我知道了')",
                "[aria-label='Close']", "[aria-label='关闭']"]:
        try:
            el = page.locator(sel).first
            if el.count() and el.is_visible():
                el.click(timeout=2000)
                page.wait_for_timeout(500)
                return True
        except Exception:
            pass
    # 按坐标点横幅右上角 ×
    try:
        box = page.evaluate(r"""
        () => {
          // 找页面里宽度 300~700、高度 80~260 且含 close 图标的容器
          const cands = [...document.querySelectorAll('div')].filter(el => {
            const r = el.getBoundingClientRect();
            return r.width > 300 && r.width < 760 && r.height > 70 &&
                   r.height < 300 && r.y < 320 &&
                   (el.className || '').toString().match(/banner|notice|promo|ad|modal|dialog/i);
          });
          if (!cands.length) return null;
          const r = cands[0].getBoundingClientRect();
          return [r.x, r.y, r.width, r.height];
        }
        """)
        if box:
            x, y, w, h = box
            page.mouse.click(x + w - 26, y + 26)
            page.wait_for_timeout(600)
            return True
    except Exception:
        pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://cun.ai/sign-in")
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    from cloakbrowser import launch

    browser = launch(headless=not args.headed, humanize=True,
                     locale="zh-CN", timezone="Asia/Shanghai")
    try:
        ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                                  locale="zh-CN", timezone_id="Asia/Shanghai")
        page = ctx.new_page()

        # 收集与验证码相关的响应对象（先存起来，稍后再读 body，避免 handler 里阻塞）
        hits = []

        def on_resp(r):
            u = r.url
            if ("cap_union_prehandle" in u or "getcapbysig" in u
                    or "cap_union_new_getcapb" in u):
                hits.append(r)

        page.on("response", on_resp)

        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3200)
        close_banners(page)
        for sel, val in (("input[name='username']", args.username),
                         ("input[name='password']", args.password)):
            try:
                page.locator(sel).first.fill(val, timeout=5000)
            except Exception:
                pass
        for sel in ["[role='checkbox']", "input[type='checkbox']"]:
            try:
                el = page.locator(sel).first
                if el.count():
                    el.click(timeout=3000)
                    break
            except Exception:
                pass

        clicked = False
        for sel in ["button[type='submit']", "button:has-text('登录')"]:
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    el.click(timeout=6000)
                    clicked = True
                    break
            except Exception:
                pass
        log(f"click_login={clicked}")

        # 等弹层真的出现（比固定 sleep 可靠）
        appeared = False
        for _ in range(30):
            try:
                if page.locator(SEL_POPUP).first.count() and \
                        page.locator(SEL_POPUP).first.is_visible():
                    appeared = True
                    break
            except Exception:
                pass
            page.wait_for_timeout(400)
        log(f"popup_appeared={appeared}")
        if not appeared:
            page.screenshot(path=os.path.join(OUT, "no_popup.png"))
            raise SystemExit("弹层未出现，检查截图 no_popup.png")
        page.wait_for_timeout(2000)

        summary = []
        for rd in range(1, args.rounds + 1):
            hits.clear()
            # 触发一次新的验证码请求：点 footer 刷新
            try:
                icons = page.locator(".tencent-captcha-dy__footer-icon")
                if icons.count() > 0:
                    icons.nth(0).click(timeout=2500)
            except Exception:
                pass
            page.wait_for_timeout(2600)

            title = ""
            try:
                el = page.locator(SEL_TITLE).first
                if el.count():
                    title = (el.inner_text() or "").strip()
            except Exception:
                pass

            # 从拦截到的响应里找 prehandle
            pre_j = None
            sprite_bytes = None
            bg_bytes = None
            for r in list(hits):
                try:
                    u = r.url
                    if "cap_union_prehandle" in u:
                        pre_j = parse_jsonp(r.text())
                    elif "img_index=0" in u or "getcapbysig" in u:
                        b = r.body()
                        if len(b) > 3000 and sprite_bytes is None:
                            sprite_bytes = b
                        elif len(b) > 3000:
                            # 第二张当 bg
                            if bg_bytes is None:
                                bg_bytes = b
                except Exception:
                    continue

            rec = {"round": rd, "title": title, "n_hits": len(hits)}
            if pre_j:
                dsi = (pre_j.get("data") or {}).get("dyn_show_info") or {}
                rec["subcapclass"] = str(pre_j.get("subcapclass"))
                rec["sess"] = (pre_j.get("data") or {}).get("sess", "")[:20]
                rec["instruction"] = dsi.get("instruction")
                rec["bg_elem_cfg"] = {
                    k: (v[:80] if isinstance(v, str) else v)
                    for k, v in (dsi.get("bg_elem_cfg") or {}).items()
                }
                rec["ins_elem_cfg"] = dsi.get("ins_elem_cfg")
                rec["sprite_url"] = (dsi.get("sprite_url") or "")[:100]
                rec["dyn_keys"] = sorted(dsi.keys())
            log(json.dumps(rec, ensure_ascii=False)[:500])
            summary.append(rec)

            if sprite_bytes:
                p = os.path.join(OUT, f"r{rd}_sprite.png")
                open(p, "wb").write(sprite_bytes)
                im = cv2.imdecode(np.frombuffer(sprite_bytes, np.uint8),
                                  cv2.IMREAD_COLOR)
                if im is not None:
                    log(f"  sprite {im.shape[1]}x{im.shape[0]} -> {p}")
            if bg_bytes:
                p = os.path.join(OUT, f"r{rd}_img2.png")
                open(p, "wb").write(bg_bytes)
                im = cv2.imdecode(np.frombuffer(bg_bytes, np.uint8),
                                  cv2.IMREAD_COLOR)
                if im is not None:
                    log(f"  img2 {im.shape[1]}x{im.shape[0]} -> {p}")
            try:
                page.screenshot(path=os.path.join(OUT, f"r{rd}_page.png"))
            except Exception:
                pass

        with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log(f"汇总 -> {os.path.join(OUT, 'summary.json')}")
    finally:
        try:
            browser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
