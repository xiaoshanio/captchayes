# -*- coding: utf-8 -*-
"""probe_click_feedback: 量化「点击是否被接收」，为时序优化提供依据。

════════════════════════════════════════════════════════════════
要回答的问题
════════════════════════════════════════════════════════════════
2408 失败全是「点完判错、弹层仍在」。可能是：
    (a) 时序：点击没被接收（早于交互层就绪）
    (b) 答案：点击被接收但选错/顺序错
    (c) 风控：答对但被判机器

判断依据：天御 2408 的 click_cfg.mark_style = inc_number，
**每次有效点击应在图上画一个序号标记**。所以：
    点击后该位置像素应有明显变化 → 说明点击被接收
    点击后像素几乎无变化       → 说明点击被丢弃（时序问题）

════════════════════════════════════════════════════════════════
本脚本做的 A/B 对照
════════════════════════════════════════════════════════════════
    A) 立即点击：解完马上点第 1 个点 → 测像素变化
    B) 稳定后点击：等弹层位置连续不变 + 700ms → 点第 1 个点 → 测像素变化

若 A 的变化量显著小于 B，则「立即点击会被吞」被量化证实，
延迟阈值就有了依据（而不是拍脑袋加等待）。

用法：
    python tools/probe_click_feedback.py --username X --password Y --trials 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(REPO, "out", "click_feedback")
POPUP_SEL = ".tencent-captcha__transform, #tcaptcha_transform_dy"
CHILD_JS = r"""
() => {
  const root = document.querySelector('.tencent-captcha__transform, #tcaptcha_transform_dy');
  if (!root) return {n: 0, sig: ''};
  const items = [];
  for (const el of root.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    items.push(`${el.tagName}|${(el.className||'').toString().slice(0,24)}|` +
               `${Math.round(r.x)},${Math.round(r.y)},${Math.round(r.width)},${Math.round(r.height)}`);
  }
  return {n: items.length, sig: items.join(';')};
}
"""


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def diff_at(a, b, lx, ly, half=55):
    """比较两张弹层截图在 (lx,ly) 附近的差异。坐标是弹层内局部坐标。"""
    if a is None or b is None:
        return None
    h, w = a.shape[:2]
    x1, y1 = max(0, lx - half), max(0, ly - half)
    x2, y2 = min(w, lx + half), min(h, ly + half)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    d = cv2.absdiff(a[y1:y2, x1:x2], b[y1:y2, x1:x2])
    g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return {"mean": round(float(g.mean()), 2), "max": int(g.max()),
            "n_big": int((g > 25).sum()), "px": int(g.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--trials", type=int, default=4, help="每种模式各做几次")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--url", default="https://www.cun.ai/sign-in")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    from src.captcha import TextSelectCaptcha
    from cloakbrowser import launch
    import cunai_login_lib as L

    cap = TextSelectCaptcha()
    cap.yolo.conf_threshold = 0.25

    browser = launch(headless=not args.headed, humanize=True,
                     locale="zh-CN", timezone="Asia/Shanghai")
    results = []
    try:
        ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                                  locale="zh-CN", timezone_id="Asia/Shanghai")
        page = ctx.new_page()
        bot = L.CunaiLogin(page, ctx, cap, None, out_dir=OUT)

        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)
        bot.close_banners()
        bot.fill_form(args.username, args.password)
        bot.net.arm()
        if not bot.click_login() or not bot.wait_popup(20):
            raise SystemExit("弹层未出现")
        bot.net.wait_ready(12)

        def ensure_2408(max_try=8):
            """换题直到拿到 2408。"""
            for _ in range(max_try):
                if not bot.popup_visible():
                    bot.reload_relogin(args.username, args.password)
                    if not bot.popup_visible():
                        continue
                if L.kind_of(bot.read_title()) == "2408":
                    return True
                bot.net.arm()
                bot.refresh_popup()
                bot.net.wait_ready(10)
                page.wait_for_timeout(600)
            return False

        def popup_rect():
            try:
                b = page.locator(POPUP_SEL).first.bounding_box()
                return b
            except Exception:
                return None

        def one_trial(mode):
            """mode: 'immediate' | 'stable'"""
            if not ensure_2408():
                return None
            t_popup = time.time()
            if mode == "stable":
                bot.wait_popup_stable()
            else:
                page.wait_for_timeout(120)      # 仅最小间隔
            t_click = time.time()

            r = bot.solve_current(1, {}, dry_run=True)
            if not r.get("ok") or not r.get("bg_pts"):
                return None
            bg_pts = r["bg_pts"]
            box = r["box"]
            pr = popup_rect()
            if not box or not pr:
                return None

            sx = box["width"] / float(L.BG_W)
            sy = box["height"] / float(L.BG_H)
            # 第 1 个点的页面坐标
            px = box["x"] + bg_pts[0][0] * sx
            py = box["y"] + bg_pts[0][1] * sy
            # 弹层内局部坐标（截图坐标系）
            lx = int(px - pr["x"])
            ly = int(py - pr["y"])

            before_p = os.path.join(OUT, f"{mode}_before_{len(results)}.png")
            after_p = os.path.join(OUT, f"{mode}_after_{len(results)}.png")
            a = None
            try:
                page.locator(POPUP_SEL).first.screenshot(path=before_p)
                a = imread_u(before_p)
            except Exception:
                pass
            sig_before = page.evaluate(CHILD_JS)

            delay_ms = (t_click - t_popup) * 1000
            page.mouse.move(px + 1, py + 1, steps=5)
            page.wait_for_timeout(40)
            page.mouse.click(px, py)
            page.wait_for_timeout(450)          # 等序号标记渲染

            try:
                page.locator(POPUP_SEL).first.screenshot(path=after_p)
                b = imread_u(after_p)
            except Exception:
                b = None
            sig_after = page.evaluate(CHILD_JS)

            ds = diff_at(a, b, lx, ly)
            rec = {"mode": mode, "delay_ms": round(delay_ms),
                   "local": [lx, ly], "page": [round(px), round(py)],
                   "diff": ds,
                   "dom_n": [sig_before.get("n"), sig_after.get("n")],
                   "dom_changed": sig_before.get("sig") != sig_after.get("sig")}
            print(f"  [{mode}] delay={rec['delay_ms']}ms "
                  f"diff_mean={ds['mean'] if ds else '-'} "
                  f"n_big={ds['n_big'] if ds else '-'} "
                  f"dom={rec['dom_n']} changed={rec['dom_changed']}")
            results.append(rec)

            # 收尾：换一道题，避免影响下一次
            bot.net.arm()
            bot.refresh_popup()
            bot.net.wait_ready(8)
            page.wait_for_timeout(500)
            return rec

        for mode in ("immediate", "stable"):
            print(f"\n=== 模式 {mode} ===")
            done = 0
            guard = 0
            while done < args.trials and guard < args.trials * 4:
                guard += 1
                rec = one_trial(mode)
                if rec:
                    done += 1

    finally:
        try:
            browser.close()
        except Exception:
            pass

    # ---- 汇总
    print("\n" + "=" * 66)
    print(" A/B 对照：点击是否被接收（像素变化量）")
    print("=" * 66)
    for mode in ("immediate", "stable"):
        rs = [r for r in results if r["mode"] == mode and r.get("diff")]
        if not rs:
            print(f"  {mode:10s} 无有效样本")
            continue
        mean_ = np.mean([r["diff"]["mean"] for r in rs])
        big = np.mean([r["diff"]["n_big"] for r in rs])
        recv = sum(1 for r in rs if r["diff"]["n_big"] > 20)
        domc = sum(1 for r in rs if r.get("dom_changed"))
        dly = np.mean([r["delay_ms"] for r in rs])
        print(f"  {mode:10s} n={len(rs)}  平均延迟={dly:.0f}ms  "
              f"diff_mean={mean_:.2f}  n_big={big:.0f}  "
              f"被接收={recv}/{len(rs)}  DOM变化={domc}/{len(rs)}")

    out_json = os.path.join(OUT, "feedback.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n明细 -> {os.path.relpath(out_json, REPO)}")


if __name__ == "__main__":
    main()
