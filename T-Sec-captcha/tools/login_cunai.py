# -*- coding: utf-8 -*-
"""login_cunai: 用真实账户在 cun.ai 登录并完成验证码（页面内截图版）。

═══ 三个关键修正（都是踩坑后得到的，写在这里避免再犯）═══

【修正 1】必须截**页面弹层里的那张题**，不能另发 prehandle
    第一版用 ctx.request 另发一次 `cap_union_prehandle` 取题。
    问题：那是**另一道题**，与页面弹层显示的完全不同。
    实测页面显示「请依次点击：舱 稠 安」，而我取到的是别的图，
    解出来的坐标自然全错。

    正确做法：定位弹层里的底图元素，直接 `element.screenshot()`。
    实测元素是 `.tencent-captcha-dy__verify-bg-img`，渲染 330x236。

【修正 2】弹层根的 rect 不是底图的 rect
    实测 `#tcaptcha_transform_dy` 是 360x361（含标题/按钮/安全验证行），
    底图 `__verify-bg-img` 是 330x236 且偏移 (15, 50)。
    第一版把根的 rect 当底图，点击全部偏移 —— 看起来"点了但没反应"。

【修正 3】题型是**混合下发**的，必须路由
    实测 sign-in 15 次抽样：
        2401 滑块         7
        2408 图标点选      6   ← model/ 里的新模型能解
        2404 中文文字点选  2   ← 需要旧域模型（runs/legacy/）
    instruction 字段可以区分：
        2408 → "请依次点击："（无字）
        2404 → "请依次点击：秤 爱 驰"（带中文）
    本脚本对 2408 用新模型、对 2404 用旧模型，其余题型跳过并刷新。

用法：
    python tools/login_cunai.py --username xxx --password yyy
    python tools/login_cunai.py --username xxx --password yyy --headed
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
OUT = os.path.join(REPO, "out", "login")
STRIP_H = 56

# 定位弹层内各元素（实测选择器）
SEL_POPUP = ".tencent-captcha__transform, #tcaptcha_transform_dy"
SEL_BG = ".tencent-captcha-dy__verify-bg-img"
SEL_TITLE = ".tencent-captcha-dy__header-text"


def log(msg):
    print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] {msg}", flush=True)


def imwrite_u(p, img, q=96):
    ext = os.path.splitext(p)[1].lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), q] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(p)
    return ok


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def compose_icon_canvas(bg, strip_h=STRIP_H):
    """2408 图标点选：底图下方留白条（提示图由弹层标题给出，不在图里）。

    注意：2408 的提示图（170x50）是弹层里的独立元素，
    训练时我们把提示图贴在底部 56px 条里（class2 = 提示条图标）。
    但登录场景下提示图是**弹层标题区**，不在底图上。
    这里做一个"空白提示条"版本给检测器用（它主要靠 class0 定位底图图标），
    然后从标题文字里读提示（本脚本暂不依赖提示图匹配，因为 2408 的
    底图图标与提示图是同一套图，模型已能直接配）。

    实际上更稳的做法：只把底图送检测器，取 class0 框；
    class2 从提示图元素单独裁（见 login 主流程）。
    """
    h, w = bg.shape[:2]
    canvas = np.full((h + strip_h, w, 3), 255, np.uint8)
    canvas[:h, :w] = bg
    return canvas


def compose_text_canvas(bg, chars, strip_h=STRIP_H):
    """2404 中文文字点选：把标题里的目标字渲染成底部提示条，拼到底图下方。

    为什么必须这么做：
        旧域模型（runs/legacy/best_v3.onnx）是**单图契约** ——
        图内底部自带提示条，class2 = 提示条里的字，class0 = 图面可点的字。
        登录弹层里提示条是**标题文字**（"请依次点击：苍 倡 乘"），底图上没有。
        直接把裸底图送检测器会得到 n_char=0（实测），配对必然失败。

        把字渲染成提示条后，合成图与旧域训练样本（344x384 单图）同版式，
        模型就能正常工作。

    为什么用 PIL 而不是 cv2.putText：
        OpenCV 的 Hershey 字体**不支持中文**，画出来是 `?` 或空白。
        必须用 TrueType 中文字体（实测 msyh.ttc / simhei.ttf 可用）。
    """
    h, w = bg.shape[:2]
    canvas = np.full((h + strip_h, w, 3), 255, np.uint8)
    canvas[:h, :w] = bg
    if not chars:
        return canvas

    from PIL import Image, ImageDraw, ImageFont
    font_path = None
    for c in (r"C:\Windows\Fonts\msyh.ttc",
              r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.exists(c):
            font_path = c
            break
    if font_path is None:
        return canvas

    n = len(chars)
    cell = min(int(w / max(n, 1)), strip_h)
    fsize = max(10, int(cell * 0.72))
    try:
        font = ImageFont.truetype(font_path, fsize)
    except Exception:
        return canvas

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    drw = ImageDraw.Draw(pil)
    total = cell * n
    x0 = max(4, (w - total) // 2)
    for i, ch in enumerate(chars):
        cx = x0 + i * cell + cell // 2
        cy = h + strip_h // 2
        try:
            bb = drw.textbbox((0, 0), ch, font=font)
            tw, tht = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, tht = fsize, fsize
        drw.text((cx - tw / 2 - (bb[0] if 'bb' in dir() else 0),
                  cy - tht / 2 - (bb[1] if 'bb' in dir() else 0)),
                 ch, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://www.cun.ai/sign-in")
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--max-refresh", type=int, default=12,
                    help="最多刷新多少次去找可解题型")
    ap.add_argument("--max-submit", type=int, default=3,
                    help="最多提交几次登录")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    report = {"url": args.url, "username": args.username,
              "started": datetime.now(TZ).isoformat(timespec="seconds"),
              "steps": []}

    def step(name, **kw):
        rec = {"step": name,
               "t": datetime.now(TZ).isoformat(timespec="seconds"), **kw}
        report["steps"].append(rec)
        log(f"{name}: {json.dumps(kw, ensure_ascii=False)[:400]}")

    # ---- 加载模型
    from src.captcha import TextSelectCaptcha
    from src.utils import yolo_onnx, ver_onnx, matchingMode
    t0 = time.time()
    cap = TextSelectCaptcha()
    cap.yolo.conf_threshold = args.conf
    legacy_dir = os.path.join(REPO, "runs", "legacy")
    legacy_y = legacy_pre = None
    try:
        legacy_y = yolo_onnx.YOLO(os.path.join(legacy_dir, "best_v3.onnx"),
                                  conf_threshold=0.3)
        legacy_pre = ver_onnx.PreONNX(os.path.join(legacy_dir,
                                                   "pre_model_v7.onnx"))
        legacy_ok = True
    except Exception as e:
        legacy_ok = False
        log(f"旧域模型加载失败: {type(e).__name__}: {e}")
    step("load_model", ms=round((time.time() - t0) * 1000, 1),
         det=os.path.basename(cap.yolo.session._model_path),
         pre=os.path.basename(cap.pre.session._model_path),
         legacy_ready=legacy_ok)

    from cloakbrowser import launch
    browser = None
    try:
        log("启动 CloakBrowser")
        browser = launch(headless=not args.headed, humanize=True,
                         locale="zh-CN", timezone="Asia/Shanghai")
        ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                                  locale="zh-CN", timezone_id="Asia/Shanghai")
        page = ctx.new_page()

        log(f"打开 {args.url}")
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)
        for sel in ["button:has-text('知道了')", "button:has-text('我知道了')",
                    "[aria-label='Close']"]:
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    el.click(timeout=2500)
                    log(f"关闭弹窗: {sel}")
                    page.wait_for_timeout(600)
                    break
            except Exception:
                continue

        # ---- 填表单 + 勾协议
        filled = []
        for sel, val, tag in [
            ("input[name='username']", args.username, "username"),
            ("input[placeholder*='用户名']", args.username, "username"),
            ("input[name='password']", args.password, "password"),
            ("input[type='password']", args.password, "password"),
        ]:
            if tag in filled:
                continue
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    el.fill(val, timeout=5000)
                    filled.append(tag)
            except Exception:
                pass
        agreed = False
        for sel in ["[role='checkbox']", "input[type='checkbox']"]:
            try:
                el = page.locator(sel).first
                if el.count():
                    try:
                        el.check(timeout=4000)
                    except Exception:
                        el.click(timeout=4000)
                    agreed = True
                    break
            except Exception:
                continue
        step("form", filled=filled, agreed=agreed)
        if "username" not in filled or "password" not in filled:
            raise SystemExit("找不到用户名/密码输入框")

        def click_login():
            for sel in ["button[type='submit']", "button:has-text('登录')",
                        "button:has-text('登 录')"]:
                try:
                    el = page.locator(sel).first
                    if not (el.count() and el.is_visible()):
                        continue
                    txt = (el.inner_text() or "").strip()
                    if any(k in txt for k in ("GitHub", "LinuxDO")):
                        continue
                    el.click(timeout=6000)
                    return True
                except Exception:
                    continue
            return False

        def popup_visible():
            try:
                el = page.locator(SEL_POPUP).first
                return bool(el.count() and el.is_visible())
            except Exception:
                return False

        def read_title():
            try:
                el = page.locator(SEL_TITLE).first
                if el.count():
                    return (el.inner_text() or "").strip()
            except Exception:
                pass
            return ""

        def close_popup():
            """关闭弹层 —— 用实测出来的真实选择器。

            实测（tools/probe_popup_buttons.py）：
                .tencent-captcha-dy__header-close   16x16  cursor:pointer  ← 关闭
                .tencent-captcha-dy__footer-icon    18x18  cursor:pointer  ← 刷新验证/问题反馈
                .tencent-captcha-dy__verify-confirm-btn          ← 确定（点完点选才 enabled）

            之前用「弹层 rect 右上角 -18px」盲点坐标，点在 header 上
            等于**开始拖动弹层**，之后 bounding_box 全变 (-10165,-10152)。
            这是本项目最隐蔽的一个坑。
            """
            if not popup_visible():
                return True
            try:
                el = page.locator(".tencent-captcha-dy__header-close").first
                if el.count() and el.is_visible():
                    el.click(timeout=2500)
                    page.wait_for_timeout(900)
                    if not popup_visible():
                        return True
            except Exception as e:
                log(f"close_popup: {type(e).__name__}")
            try:
                page.keyboard.press("Escape")
                page.wait_for_timeout(700)
                return not popup_visible()
            except Exception:
                return False

        def refresh_popup():
            """点弹层 footer 的「刷新验证」换一道题 —— 不用关弹层。

            实测 .tencent-captcha-dy__footer-icon 有多个（刷新/反馈/无障碍/常规），
            第一个是「刷新验证」，且它的 IMG 子元素带
            `.tencent-captcha-dy__unselectable`（pointer-events:none），
            所以直接点 DIV 本身。
            """
            try:
                icons = page.locator(".tencent-captcha-dy__footer-icon")
                n = icons.count()
                if n > 0:
                    icons.nth(0).click(timeout=2500)
                    page.wait_for_timeout(1800)
                    return True
            except Exception as e:
                log(f"refresh_popup: {type(e).__name__}")
            return False

        def reopen_popup():
            """换一道题 —— 优先用「刷新验证」按钮（不关弹层，最稳）。

            只有在刷新失败/弹层已失效时才走「关闭 + 重新点登录」。
            """
            # 弹层已失效（被拖出视口）→ 只能刷新页面
            try:
                box = page.locator(SEL_POPUP).first.bounding_box()
                if box and (box["x"] < -1000 or box["y"] < -1000):
                    log("弹层已失效，刷新页面重来")
                    return reload_and_login()
            except Exception:
                pass

            # 优先：点「刷新验证」
            if refresh_popup():
                if popup_visible():
                    return True

            # 次选：关闭后重新点登录
            close_popup()
            page.wait_for_timeout(800)
            if popup_visible():
                return reload_and_login()
            return click_login_and_wait()

        def reload_and_login():
            try:
                page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(3000)
            except Exception as e:
                log(f"reload 失败: {type(e).__name__}: {e}")
                return False
            for sel in ["button:has-text('知道了')", "[aria-label='Close']"]:
                try:
                    el = page.locator(sel).first
                    if el.count() and el.is_visible():
                        el.click(timeout=2000)
                        page.wait_for_timeout(500)
                        break
                except Exception:
                    pass
            refill()
            return click_login_and_wait()

        def refill():
            """重新填账号密码 + 勾协议（刷新后用）。"""
            for sel, val, tag in [
                ("input[name='username']", args.username, "username"),
                ("input[name='password']", args.password, "password"),
            ]:
                try:
                    el = page.locator(sel).first
                    if el.count() and el.is_visible():
                        el.fill(val, timeout=5000)
                except Exception:
                    pass
            for sel in ["[role='checkbox']", "input[type='checkbox']"]:
                try:
                    el = page.locator(sel).first
                    if el.count():
                        try:
                            el.check(timeout=3000)
                        except Exception:
                            el.click(timeout=3000)
                        break
                except Exception:
                    continue

        def click_login_and_wait():
            ok = click_login()
            page.wait_for_timeout(2800)
            return ok and popup_visible()

        def solve_slider(box):
            """解 2401 滑块：用模板匹配找缺口 x，再拖动滑块块。

            天御滑块是「拼图缺口」式：底图上有一个缺口，
            `.tencent-captcha-dy__fg-item` 是待嵌入的拼图块。
            用 TM_CCOEFF_NORMED 在底图上找拼图块的最佳位置 = 缺口位置。
            这是经典解法，不依赖深度学习。
            """
            try:
                fg = page.locator(".tencent-captcha-dy__fg-item").first
                if not (fg.count() and fg.is_visible()):
                    return False
                p_fg = os.path.join(OUT, "slider_fg.png")
                fg.screenshot(path=p_fg)
                fg_im = imread_u(p_fg)
                if fg_im is None:
                    return False
                handle = page.locator(".tencent-captcha-dy__slider-block").first
                if not (handle.count() and handle.is_visible()):
                    return False
                hb = handle.bounding_box()

                bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
                fg_gray = cv2.cvtColor(fg_im, cv2.COLOR_BGR2GRAY)
                fh, fw = fg_gray.shape[:2]
                if fw >= bg_gray.shape[1] or fh >= bg_gray.shape[0]:
                    log(f"  滑块: 拼图块 {fw}x{fh} 过大，跳过")
                    return False
                res = cv2.matchTemplate(bg_gray, fg_gray, cv2.TM_CCOEFF_NORMED)
                _mn, mx, _ml, maxl = cv2.minMaxLoc(res)
                gap_x = maxl[0]
                scale = box["width"] / float(bg.shape[1])
                drag = gap_x * scale
                log(f"  滑块: gap_x={gap_x} score={mx:.3f} drag={drag:.0f}px")
                sx = hb["x"] + hb["width"] / 2
                sy = hb["y"] + hb["height"] / 2
                page.mouse.move(sx, sy, steps=8)
                page.wait_for_timeout(150)
                page.mouse.down()
                page.wait_for_timeout(120)
                steps = 24
                for k in range(1, steps + 1):
                    t = k / float(steps)
                    e = 1 - (1 - t) ** 2.2        # ease-out，像人手
                    page.mouse.move(sx + drag * e, sy + random.uniform(-1, 1))
                    page.wait_for_timeout(random.randint(12, 26))
                page.wait_for_timeout(220)
                page.mouse.up()
                return True
            except Exception as e:
                log(f"solve_slider 失败: {type(e).__name__}: {e}")
                return False

        def grab_bg():
            """取弹层里底图的**原图**，返回 (bgr, rect) 或 (None, None)。

            【关键】不能用 element.screenshot() 截渲染后的小图。
            实测弹层里底图渲染尺寸只有 330x237，图标被缩到 ~35px，
            模型（在 672x480 原图上训的）检测不到任何图标：
                element.screenshot -> 330x237 -> dets=0
                原图 672x480       -> dets=9 (cls2=3, cls0=6)

            正确做法：从 CSS background-image 的 URL 里取原图，
            用浏览器上下文请求（带 cookie），拿到 672x480 原图。

            注意：必须用**当前弹层这一次**的 URL（含 sess/image 参数），
            否则就是另一道题。
            """
            try:
                el = page.locator(SEL_BG).first
                if not (el.count() and el.is_visible()):
                    return None, None
                box = el.bounding_box()
                if box is None:
                    return None, None
                if box["x"] < -1000 or box["y"] < -1000:
                    log(f"  grab_bg: 弹层已移出视口")
                    return None, None

                # 从 background-image 里抠 URL（等图片真正加载出来）
                url = None
                for _ in range(15):
                    url = page.evaluate(r"""
                    () => {
                      const el = document.querySelector(
                        '.tencent-captcha-dy__verify-bg-img');
                      if (!el) return null;
                      const bg = getComputedStyle(el).backgroundImage || '';
                      const m = bg.match(/url\(["']?([^"')]+)["']?\)/);
                      return m ? m[1] : null;
                    }
                    """)
                    if url and "getcapbysig" in url:
                        break
                    page.wait_for_timeout(400)
                if not url or "getcapbysig" not in url:
                    log("  grab_bg: 没拿到底图 URL")
                    return None, None

                r2 = ctx.request.get(url, timeout=25000)
                if not r2.ok:
                    log(f"  grab_bg: 取图失败 HTTP {r2.status}")
                    return None, None
                buf = r2.body()
                im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if im is None:
                    return None, None
                log(f"  grab_bg: 原图 {im.shape[1]}x{im.shape[0]} "
                    f"(渲染 {box['width']:.0f}x{box['height']:.0f})")
                return im, box
            except Exception as e:
                log(f"grab_bg 失败: {type(e).__name__}: {e}")
                return None, None

        # ---- 触发弹层
        if not click_login():
            raise SystemExit("找不到登录按钮")
        page.wait_for_timeout(3000)
        step("click_login", popup=popup_visible())
        page.screenshot(path=os.path.join(OUT, "01_popup.png"))

        solved = False
        for attempt in range(1, args.max_refresh + 1):
            if not popup_visible():
                log("弹层消失，可能已登录或需要重新点登录")
                break
            title = read_title()
            bg, box = grab_bg()
            if bg is None or box is None:
                log(f"--- {attempt}: 取不到底图，刷新 ---")
                reopen_popup()
                continue

            # ---- 题型判定（按标题文字，实测三种）
            #   2401 滑块       : "拖动下方滑块完成拼图"
            #   2408 图标点选   : "请依次点击："（冒号后为空）
            #   2404 文字点选   : "请依次点击：舱 稠 安"（冒号后带中文）
            t = (title or "").strip()
            if "滑块" in t or "拖动" in t:
                kind = "2401-slider"
            elif "：" in t and len(t.split("：", 1)[1].strip()) > 0:
                kind = "2404-text"
            elif "请依次点击" in t:
                kind = "2408-icon"
            else:
                kind = "unknown"
            step("captcha_ready", attempt=attempt, title=title, kind=kind,
                 bg_wh=[bg.shape[1], bg.shape[0]],
                 box={k: round(v, 1) for k, v in (box or {}).items()})
            page.screenshot(path=os.path.join(OUT, f"02_captcha_{attempt}.png"))

            if kind == "2401-slider":
                # 滑块：本项目主攻点选，但滑块是当前高频题型，
                # 用模板匹配兜底解一下（不追求高成功率，能过就行）
                step("try_slider", attempt=attempt)
                ok_sl = solve_slider(box)
                step("slider_attempt", attempt=attempt, ok=ok_sl)
                page.wait_for_timeout(3500)
                page.screenshot(path=os.path.join(OUT, f"03_after_{attempt}.png"))
                cur = page.url
                if any(k in cur for k in ("/console", "/dashboard", "/chat")):
                    solved = True
                    break
                if not popup_visible():
                    page.wait_for_timeout(2500)
                    if any(k in page.url for k in ("/console", "/dashboard", "/chat")):
                        solved = True
                        break
                reopen_popup()
                continue

            if kind == "unknown":
                step("skip_unsupported", attempt=attempt, kind=kind)
                reopen_popup()
                continue

            if kind == "2408-icon":
                # ---- 用新模型解 2408
                canvas = compose_icon_canvas(bg, STRIP_H)
                p_img = os.path.join(OUT, f"icon_{attempt}.jpg")
                imwrite_u(p_img, canvas)
                t1 = time.time()
                try:
                    d = cap.run_dict(p_img)
                except Exception as e:
                    step("solve_fail", attempt=attempt,
                         err=f"{type(e).__name__}: {e}")
                    reopen_popup()
                    continue
                ms = (time.time() - t1) * 1000
                corps = d.get("corp") or []
                step("solved_2408", attempt=attempt, ms=round(ms, 1),
                     n=len(corps),
                     pts=[(round((c["x1"] + c["x2"]) / 2, 1),
                           round((c["y1"] + c["y2"]) / 2, 1)) for c in corps])
                if not corps:
                    reopen_popup()
                    continue
                # 坐标换算：canvas -> 底图元素渲染位置
                #
                # 【坑】box 是**底图元素**的渲染 rect（对应原底图 480 高），
                # 而 canvas 是 480 + 56 = 536 高的合成图。
                # 所以 sy 必须除以 bg_h（=480），不能除以 canvas 高 536，
                # 否则点击 y 会压缩 ~10%，全部偏上。
                W, H = canvas.shape[1], canvas.shape[0]
                bg_h = H - STRIP_H
                sx = box["width"] / float(W)
                sy = box["height"] / float(bg_h)
                pts = []
                for c in corps:
                    cx = (c["x1"] + c["x2"]) / 2.0
                    cy = (c["y1"] + c["y2"]) / 2.0
                    if cy > bg_h:
                        continue
                    pts.append((box["x"] + cx * sx, box["y"] + cy * sy))
            else:
                # ---- 用旧域模型解 2404（中文文字点选）
                #
                # 【关键】旧域模型是「单图」契约：图内底部自带提示条
                #   class2 = 提示条里的字，class0 = 图面可点的字
                # 而登录弹层里**提示条是标题文字**（"请依次点击：苍 倡 乘"），
                # 底图上没有提示条 → 直接送检测器会得到 n_char=0
                #   （实测：legacy_no_boxes n_target=3 n_char=0）
                #
                # 修法：把标题里的字**渲染成提示条**拼到底图下方，
                # 合成出与旧域训练样本同版式的图，再送模型。
                if not legacy_ok:
                    step("skip_2404", attempt=attempt, reason="旧域模型不可用")
                    reopen_popup()
                    continue
                chars = t.split("：", 1)[1].strip().split() if "：" in t else []
                canvas = compose_text_canvas(bg, chars, STRIP_H)
                p_img = os.path.join(OUT, f"text_{attempt}.jpg")
                imwrite_u(p_img, canvas)
                det = legacy_y.inference(canvas)
                tb = [x[:4] for x in det if x[5] == 0]
                cb = [x[:4] for x in det if x[5] == 2]
                cb.sort(key=lambda b: b[0])
                step("legacy_detect", attempt=attempt, chars=chars,
                     n_target=len(tb), n_char=len(cb))
                if not tb or not cb:
                    step("legacy_no_boxes", attempt=attempt,
                         n_target=len(tb), n_char=len(cb))
                    reopen_popup()
                    continue
                its = [canvas[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in tb]
                ics = [canvas[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in cb]
                slys = legacy_pre.reason_all_batch(ics, its)
                order = matchingMode.find_overall_index_fast(slys)
                res = [tb[j] for _i, j in order]
                step("solved_2404", attempt=attempt, n=len(res),
                     boxes=[list(map(int, b)) for b in res])
                if not res:
                    reopen_popup()
                    continue
                # 同样：box 对应底图 480 高，canvas 是 536 高合成图
                W, H = canvas.shape[1], canvas.shape[0]
                bg_h = H - STRIP_H
                sx = box["width"] / float(W)
                sy = box["height"] / float(bg_h)
                pts = []
                for (x1, y1, x2, y2) in res:
                    cy = (y1 + y2) / 2.0
                    if cy > bg_h:             # 提示条内的框不是点击目标
                        continue
                    pts.append((box["x"] + (x1 + x2) / 2.0 * sx,
                                box["y"] + cy * sy))

            # ---- 按顺序点击
            n_click = 0
            for i, (px, py) in enumerate(pts):
                page.mouse.move(px + random.uniform(-2, 2),
                                py + random.uniform(-2, 2),
                                steps=random.randint(6, 12))
                page.wait_for_timeout(random.randint(80, 180))
                page.mouse.click(px, py)
                n_click += 1
                log(f"  点击 #{i+1} ({px:.0f}, {py:.0f})")
                page.wait_for_timeout(random.randint(200, 450))
            step("clicked", attempt=attempt, kind=kind, n=n_click)

            # 2408 点完后通常要按「确定」；2404 一般是自动提交
            if kind == "2408-icon":
                for sel in [".tencent-captcha-dy__verify-btn",
                            "button:has-text('确定')", "[class*='verify-btn']"]:
                    try:
                        el = page.locator(sel).first
                        if el.count() and el.is_visible():
                            el.click(timeout=3000)
                            log("点击「确定」")
                            break
                    except Exception:
                        continue

            page.wait_for_timeout(3000)
            page.screenshot(path=os.path.join(OUT, f"03_after_{attempt}.png"))

            cur = page.url
            body = ""
            try:
                body = page.inner_text("body")[:4000]
            except Exception:
                pass
            if any(k in cur for k in ("/console", "/dashboard", "/chat", "/token")):
                solved = True
                break
            if any(k in body for k in ("退出登录", "退出", "个人设置", "令牌", "额度")):
                solved = True
                break
            still = popup_visible()
            step("after_click", attempt=attempt, url=cur,
                 popup_still_open=still)
            if not still:
                # 弹层关了但没跳转，可能验证通过、正在提交
                page.wait_for_timeout(3000)
                if any(k in page.url for k in ("/console", "/dashboard", "/chat")):
                    solved = True
                    break
                if not popup_visible():
                    # 重新点登录再试
                    click_login()
                    page.wait_for_timeout(2500)
            else:
                reopen_popup()

        # ---- 最终判定
        page.wait_for_timeout(2000)
        final_url = page.url
        try:
            final_body = page.inner_text("body")[:6000]
        except Exception:
            final_body = ""
        page.screenshot(path=os.path.join(OUT, "99_final.png"))
        success = any(k in final_url for k in ("/console", "/dashboard",
                                               "/chat", "/token"))
        if not success:
            success = any(k in final_body for k in ("退出登录", "个人设置",
                                                    "我的账户", "API 密钥",
                                                    "令牌", "额度"))
        report["final_url"] = final_url
        report["success"] = bool(success)
        report["captcha_solved"] = bool(solved)
        report["body_head"] = final_body[:2000]
        step("final", url=final_url, success=report["success"])

        print("\n" + "=" * 66)
        print(f"  最终 URL  : {final_url}")
        print(f"  登录成功  : {'是' if report['success'] else '否'}")
        print(f"  验证码通过: {'是' if solved else '否'}")
        print("=" * 66)

    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
        log(f"异常: {report['error']}")
        try:
            if browser is not None:
                page.screenshot(path=os.path.join(OUT, "99_error.png"))
        except Exception:
            pass
    finally:
        report["finished"] = datetime.now(TZ).isoformat(timespec="seconds")
        with open(os.path.join(OUT, "login_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log(f"报告 -> {os.path.join(OUT, 'login_report.json')}")
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

