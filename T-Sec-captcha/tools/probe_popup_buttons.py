# -*- coding: utf-8 -*-
"""probe_popup_buttons: 找出天御弹层里的「刷新」「关闭」可点元素。

为什么需要：
    实测弹层的 × 和刷新图标**没有语义 class**，是背景图 div。
    之前用「弹层 rect 右上角 -18px」盲点坐标，结果点在 header 上
    **把弹层拖走了**（弹层支持拖拽），之后 bounding_box 全变成 -10000。

    本脚本遍历弹层内所有元素，打印它们的位置/尺寸/背景图/cursor 样式，
    从中识别：
        - cursor: pointer 的小方块（通常是按钮）
        - 位于 header 右端 / footer 右端的元素
        - 背景图含 refresh / close 的 url

用法：
    python tools/probe_popup_buttons.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "out", "probe_popup")

# 把弹层内每个元素的几何 + 样式 + 是否可点 全 dump 出来
DUMP_JS = r"""
() => {
  const root = document.querySelector('.tencent-captcha__transform, #tcaptcha_transform_dy');
  if (!root) return {found: false, nodes: []};
  const rr = root.getBoundingClientRect();
  const nodes = [];
  for (const el of [root, ...root.querySelectorAll('*')]) {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    // 相对弹层的坐标，便于判断"右上角/右下角"
    const rel = [Math.round(r.x - rr.x), Math.round(r.y - rr.y),
                 Math.round(r.width), Math.round(r.height)];
    nodes.push({
      tag: el.tagName,
      id: el.id || '',
      cls: (el.className || '').toString().slice(0, 70),
      rel: rel,
      cursor: cs.cursor || '',
      bg: (cs.backgroundImage || '').slice(0, 120),
      op: cs.opacity,
      txt: (el.textContent || '').trim().slice(0, 20),
      isRoot: el === root,
    });
  }
  return {found: true, root: [Math.round(rr.x), Math.round(rr.y),
                              Math.round(rr.width), Math.round(rr.height)],
          nodes: nodes};
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
                    el.click(timeout=2000)
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
        page.wait_for_timeout(4000)
        page.screenshot(path=os.path.join(OUT, "popup_full.png"))

        d = page.evaluate(DUMP_JS)
        if not d.get("found"):
            raise SystemExit("没找到弹层")
        print(f"弹层 root rect = {d['root']}")
        print(f"共 {len(d['nodes'])} 个元素\n")
        print(f"{'tag':5s} {'rel(x,y,w,h)':26s} {'cursor':8s} {'op':4s} "
              f"{'id/cls':46s} txt / bg")
        print("-" * 130)
        for n in d["nodes"]:
            print(f"{n['tag']:5s} {str(n['rel']):26s} {n['cursor']:8s} "
                  f"{n['op']:4s} {(n['id'] or n['cls'])[:44]:46s} "
                  f"{n['txt'][:16]} {n['bg'][:40]}")

        # 候选：cursor=pointer 且尺寸小（<40px）→ 很可能是按钮
        print("\n[candidates] cursor=pointer 且 尺寸<40px:")
        for n in d["nodes"]:
            w, h = n["rel"][2], n["rel"][3]
            if n["cursor"] == "pointer" and 0 < w <= 40 and 0 < h <= 40:
                print(f"  {n['tag']:5s} rel={n['rel']} "
                      f"id={n['id']} cls={n['cls'][:40]}")

        with open(os.path.join(OUT, "popup_nodes.json"), "w",
                  encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        print(f"\n产物 -> {OUT}/popup_nodes.json")
    finally:
        try:
            browser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
