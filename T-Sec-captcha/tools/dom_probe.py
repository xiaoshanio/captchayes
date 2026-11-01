# -*- coding: utf-8 -*-
"""dom_probe: 打印 cun.ai/sign-up 的真实表单结构（input/button/iframe 清单）。

只读，不提交。用于写对采集器的选择器。
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "out", "recon")

JS = """
() => {
  const vis = el => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const pick = el => ({
    tag: el.tagName.toLowerCase(),
    type: el.type || null,
    name: el.name || null,
    id: el.id || null,
    placeholder: el.placeholder || null,
    text: (el.innerText || el.value || '').trim().slice(0, 40),
    cls: (el.className || '').toString().slice(0, 80),
    visible: vis(el),
    rect: (() => { const r = el.getBoundingClientRect();
      return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; })(),
  });
  return {
    url: location.href,
    title: document.title,
    inputs: [...document.querySelectorAll('input,textarea')].map(pick),
    buttons: [...document.querySelectorAll('button,[role=button]')].map(pick),
    iframes: [...document.querySelectorAll('iframe')].map(f => ({
      src: f.src, id: f.id, visible: vis(f),
      rect: (() => { const r = f.getBoundingClientRect();
        return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; })(),
    })),
    scripts: [...document.querySelectorAll('script[src]')].map(s => s.src),
    turing_globals: Object.keys(window).filter(k => /captcha|turing|Tencent/i.test(k)),
  };
}
"""


def main():
    from cloakbrowser import launch
    os.makedirs(OUT, exist_ok=True)
    browser = launch(headless=True, humanize=True, locale="zh-CN",
                     timezone="Asia/Shanghai")
    ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                              locale="zh-CN", timezone_id="Asia/Shanghai")
    page = ctx.new_page()
    page.goto("https://cun.ai/sign-up", wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2500)
    for sel in ["button:has-text('知道了')", "button:has-text('我知道了')"]:
        try:
            el = page.locator(sel).first
            if el.count() and el.is_visible():
                el.click(timeout=3000)
                page.wait_for_timeout(600)
                break
        except Exception:
            pass
    info = page.evaluate(JS)
    print(json.dumps(info, ensure_ascii=False, indent=2)[:9000])
    with open(os.path.join(OUT, "dom_probe.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    browser.close()


if __name__ == "__main__":
    main()
