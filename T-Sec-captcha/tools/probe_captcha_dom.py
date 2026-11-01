# -*- coding: utf-8 -*-
"""probe_captcha_dom: 精确定位天御弹层内「底图」和「提示图」的 DOM 元素。

为什么要单独探测：
    login_cunai.py 第一版把 `#tcaptcha_transform_dy` 的 rect（360x405）
    当成底图尺寸，但那是**整个弹层**（含标题/提示图/确定按钮/安全验证行）。
    底图只是弹层里的一块，实测约 335x240。

    如果几何读错，模型解出的坐标再准也会点到错误位置。

    更根本的问题是：**必须用页面弹层里那一张真实的题**，
    不能另外发一次 prehandle（那是另一道题）。

本脚本做三件事：
    1) 打开 sign-in，触发弹层
    2) 遍历弹层内所有元素，打印 tag/id/class/rect/背景图，定位底图与提示图
    3) 尝试用 Playwright 的 element.screenshot() 直接截底图，验证能否拿到原图

用法：
    python tools/probe_captcha_dom.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "out", "probe_captcha")

DUMP_JS = r"""
() => {
  const out = [];
  const push = (el, tag) => {
    const r = el.getBoundingClientRect();
    if (r.width < 20 || r.height < 20) return;
    const cs = getComputedStyle(el);
    out.push({
      tag: el.tagName,
      id: el.id || '',
      cls: (el.className || '').toString().slice(0, 80),
      rect: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)],
      bg: (cs.backgroundImage || '').slice(0, 160),
      src: el.tagName === 'IMG' ? (el.src || '').slice(0, 160) : '',
      kind: tag,
    });
  };
  // 弹层根
  const roots = [...document.querySelectorAll(
    '#tcaptcha_transform_dy, .tencent-captcha__transform, [id*=tcaptcha], [class*=tcaptcha]')];
  for (const r of roots) {
    push(r, 'root');
    for (const el of r.querySelectorAll('*')) push(el, 'child');
  }
  return {n: out.length, nodes: out};
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://www.cun.ai/sign-in")
    ap.add_argument("--username", default="lindabaker57252")
    ap.add_argument("--password", default="lindabaker572")
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
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)
        for sel in ["button:has-text('知道了')", "[aria-label='Close']"]:
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    el.click(timeout=2500)
                    page.wait_for_timeout(500)
                    break
            except Exception:
                pass

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
        for sel in ["button[type='submit']", "button:has-text('登录')"]:
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    el.click(timeout=6000)
                    break
            except Exception:
                pass
        page.wait_for_timeout(3500)
        page.screenshot(path=os.path.join(OUT, "01_popup.png"))

        dump = page.evaluate(DUMP_JS)
        with open(os.path.join(OUT, "dom.json"), "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
        print(f"[dom] 共 {dump['n']} 个节点")
        print(f"{'kind':6s} {'tag':5s} {'rect':28s} {'id/cls':40s} bg/src")
        print("-" * 120)
        for n in dump["nodes"]:
            r = n["rect"]
            ident = (n["id"] or n["cls"])[:38]
            mark = ""
            if "getcapbysig" in n["bg"] or "getcapbysig" in n["src"]:
                mark = " ★CAPTCHA-IMG"
            print(f"{n['kind']:6s} {n['tag']:5s} "
                  f"{str(r):28s} {ident:40s} "
                  f"{(n['bg'] or n['src'])[:60]}{mark}")

        # 尝试直接用 element screenshot 拿底图
        cands = [n for n in dump["nodes"]
                 if "getcapbysig" in (n["bg"] or "") or "getcapbysig" in (n["src"] or "")]
        print(f"\n[candidates] 含 getcapbysig 的元素: {len(cands)}")
        for i, n in enumerate(cands):
            sel = f"#{n['id']}" if n["id"] else None
            if not sel:
                continue
            try:
                el = page.locator(sel).first
                p = os.path.join(OUT, f"elem_{i}.png")
                el.screenshot(path=p)
                im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
                if im is not None:
                    print(f"  {sel}: screenshot {im.shape[1]}x{im.shape[0]} -> {p}")
            except Exception as e:
                print(f"  {sel}: screenshot 失败 {type(e).__name__}: {e}")

        # 也把整个弹层截下来，供人眼核对
        for sel in ["#tcaptcha_transform_dy", ".tencent-captcha__transform"]:
            try:
                el = page.locator(sel).first
                if el.count():
                    p = os.path.join(OUT, "popup_elem.png")
                    el.screenshot(path=p)
                    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
                    if im is not None:
                        print(f"  弹层截图 {sel}: {im.shape[1]}x{im.shape[0]} -> {p}")
                    break
            except Exception:
                continue

        print(f"\n产物 -> {OUT}/")
    finally:
        try:
            browser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
