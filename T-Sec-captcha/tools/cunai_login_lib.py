# -*- coding: utf-8 -*-
"""cunai_login_lib: cun.ai 登录 + 天御验证码求解的可复用实现。

════════════════════════════════════════════════════════════════════
本文件集中了所有「实测踩出来的坑」，每条都注明原因，避免以后再犯。
════════════════════════════════════════════════════════════════════

【坑 1】不能另发 prehandle 取题
    另发一次 prehandle 拿到的是**另一道题**，与弹层显示的完全不同。
    必须用 page.on("response") 拦截**弹层自己那次**请求。

【坑 2】prehandle 是 JSONP，前缀不是 "("
    实测响应体：`_aq_128199({"state":1,...,"subcapclass":"2408",...})`
    只剥外层括号不够，必须用正则剥掉 `_aq_数字(` 前缀。

【坑 3】弹层底图元素渲染尺寸 ≠ 原图尺寸
    底图原图 672x480，弹层里渲染成 330x236（缩放 0.49）。
    模型在 672x480 上训练，喂渲染小图会**检测不到任何图标**（实测 dets=0）。
    必须从 prehandle 的 bg_elem_cfg.img_url 取原图。

【坑 4】提示图不在底图上，在 header
    2408 的提示图是 `.tencent-captcha-dy__header-answer img`，
    对应 ins_elem_cfg[0] 的 sprite_pos/size_2d（实测 [0,0] + [170,50]）。
    训练版式：canvas = bg(672x480) + strip(56)，提示图 170x50 贴在 (6,3)。
    不把提示图拼进底部提示条 → 检测不到 class2 → 匹配必然失败。

【坑 5】坐标换算 sy 要除以 bg_h(480)，不是 canvas 高(536)
    box 是底图元素的渲染 rect，只对应 480 高的底图。
    用 536 会让点击 y 压缩 ~10%，全部偏上。

【坑 6】关闭弹层不能盲点坐标
    弹层 header 支持拖拽。点在 header 上等于**开始拖动**，
    之后所有 bounding_box 变成 (-10165,-10152)，弹层再也回不来。
    正确选择器：
        .tencent-captcha-dy__header-close        关闭
        .tencent-captcha-dy__footer-icon nth(0)  刷新验证
        .tencent-captcha-dy__verify-confirm-btn  确定

【坑 7】题型是混合下发的
    实测 sign-in 抽样：2401 滑块 / 2404 中文文字点选 / 2408 图标点选。
    2408 用 model/ 新模型；2404 用 runs/legacy/ 旧域模型。

【坑 8】营销横幅会盖住登录按钮
    每次横幅不同，需先关掉再点登录。
"""
from __future__ import annotations

import json
import os
import random
import re
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
CAP_HOST = "https://turing.captcha.qcloud.com"

SEL_POPUP = ".tencent-captcha__transform, #tcaptcha_transform_dy"
SEL_BG = ".tencent-captcha-dy__verify-bg-img"
SEL_TITLE = ".tencent-captcha-dy__header-text"
SEL_ANSWER = ".tencent-captcha-dy__header-answer"
SEL_CLOSE = ".tencent-captcha-dy__header-close"
SEL_FOOTER_ICON = ".tencent-captcha-dy__footer-icon"
SEL_CONFIRM = ".tencent-captcha-dy__verify-confirm-btn"
SEL_FG = ".tencent-captcha-dy__fg-item"
SEL_HANDLE = ".tencent-captcha-dy__slider-block"

# 训练版式常量（来自 data/samples/*/meta.json）
BG_W, BG_H = 672, 480
STRIP_H = 56
PROMPT_W, PROMPT_H = 170, 50
PROMPT_X, PROMPT_Y = 6, 3


def log(msg, tag=""):
    ts = datetime.now(TZ).strftime("%H:%M:%S")
    print(f"[{ts}]{(' ' + tag) if tag else ''} {msg}", flush=True)


def imwrite_u(p, im):
    ext = os.path.splitext(p)[1].lower() or ".png"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 96] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, im, params)
    if ok:
        buf.tofile(p)
    return ok


def imread_u(p):
    if not os.path.exists(p):
        return None
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def parse_jsonp(txt):
    """剥掉 JSONP 包装：`_aq_128199({...})` 或 `({...})` 或纯 JSON。"""
    t = (txt or "").strip()
    m = re.match(r"^\s*[\w$.]+\((.*)\)\s*;?\s*$", t, re.S)
    if m:
        t = m.group(1)
    return json.loads(t)


def compose_canvas(bg, prompt=None):
    """按训练版式合成：bg(672x480) + strip(56)，提示图 170x50 贴在 (6,3)。

    与 data/samples/main.jpg 完全一致（meta: strip_h=56, prompt 170x50 @ (6,3)）。
    """
    if bg is None:
        return None
    h, w = bg.shape[:2]
    canvas = np.full((h + STRIP_H, w, 3), 255, np.uint8)
    canvas[:h, :w] = bg
    if prompt is not None and prompt.size:
        ph, pw = prompt.shape[:2]
        # 缩放到训练尺寸（原图应为 170x50，容错）
        if (pw, ph) != (PROMPT_W, PROMPT_H):
            prompt = cv2.resize(prompt, (PROMPT_W, PROMPT_H),
                                interpolation=cv2.INTER_AREA if pw > PROMPT_W
                                else cv2.INTER_CUBIC)
            ph, pw = PROMPT_H, PROMPT_W
        x1, y1 = PROMPT_X, h + PROMPT_Y
        x2, y2 = min(w, x1 + pw), min(h + STRIP_H, y1 + ph)
        canvas[y1:y2, x1:x2] = prompt[:y2 - y1, :x2 - x1]
    return canvas


def render_text_strip(bg, chars):
    """2404 中文文字点选：把标题里的目标字渲染成底部提示条。

    旧域模型是单图契约（图内底部自带提示条，class2=提示条字，class0=图面字）。
    登录弹层里提示是**纯文字**，底图没有提示条 → 直接送检测器 n_char=0。
    所以要把字渲染进提示条，合成同版式图。

    用 PIL + TrueType 中文字体：OpenCV 的 Hershey 字体不支持中文。
    """
    h, w = bg.shape[:2]
    canvas = np.full((h + STRIP_H, w, 3), 255, np.uint8)
    canvas[:h, :w] = bg
    if not chars:
        return canvas
    from PIL import Image, ImageDraw, ImageFont
    fp = None
    for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.exists(c):
            fp = c
            break
    if fp is None:
        return canvas
    n = len(chars)
    cell = min(int(w / max(n, 1)), STRIP_H)
    fsize = max(10, int(cell * 0.74))
    try:
        font = ImageFont.truetype(fp, fsize)
    except Exception:
        return canvas
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    drw = ImageDraw.Draw(pil)
    total = cell * n
    x0 = max(6, (w - total) // 2)
    for i, ch in enumerate(chars):
        cx = x0 + i * cell + cell / 2
        cy = h + STRIP_H / 2
        try:
            bb = drw.textbbox((0, 0), ch, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            drw.text((cx - tw / 2 - bb[0], cy - th / 2 - bb[1]),
                     ch, font=font, fill=(0, 0, 0))
        except Exception:
            pass
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def kind_of(title):
    """按标题判定题型（实测三种）。"""
    t = (title or "").strip()
    if "滑块" in t or "拖动" in t:
        return "2401"
    if "：" in t and len(t.split("：", 1)[1].strip()) > 0:
        return "2404"
    if "请依次点击" in t:
        return "2408"
    return "unknown"


class NetCapture:
    """拦截验证码相关响应，供当前这一道题使用。

    用法：
        cap = NetCapture(page)
        cap.arm()            # 清空并开始收集
        ...点击刷新/登录...
        cap.wait_ready()     # 等到 prehandle + 图都齐
        data = cap.data      # {"pre": dict, "bg": bytes, "sprite": bytes}
    """

    def __init__(self, page):
        self.page = page
        self.resps = []
        self.page.on("response", self._on)

    def _on(self, r):
        u = r.url
        if "cap_union_prehandle" in u or "getcapbysig" in u:
            self.resps.append(r)

    def arm(self):
        self.resps = []

    def wait_ready(self, timeout=12.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.page.wait_for_timeout(300)
            pre, imgs = self._parse()
            if pre is not None and imgs:
                # 需要 img_index=1（bg）；2408 还需要 img_index=0 或 header 图
                if 1 in imgs:
                    self.data = {"pre": pre, "imgs": imgs}
                    return True
        self.data = None
        return False

    def _parse(self):
        pre = None
        imgs = {}
        for r in list(self.resps):
            u = r.url
            try:
                if "cap_union_prehandle" in u:
                    pre = parse_jsonp(r.text())
                elif "getcapbysig" in u:
                    m = re.search(r"img_index=(\d+)", u)
                    idx = int(m.group(1)) if m else -1
                    b = r.body()
                    if len(b) > 500:
                        imgs.setdefault(idx, b)
            except Exception:
                continue
        return pre, imgs


class CunaiLogin:
    """cun.ai 登录 + 验证码求解。"""

    def __init__(self, page, ctx, cap_model, legacy=None, out_dir=None,
                 max_solve_tries=6, conf=0.25):
        self.page = page
        self.ctx = ctx
        self.cap = cap_model            # src.captcha.TextSelectCaptcha
        self.legacy = legacy            # (yolo, pre) 或 None
        self.out = out_dir or os.path.join(REPO, "out", "login")
        os.makedirs(self.out, exist_ok=True)
        self.max_solve_tries = max_solve_tries
        self.cap.yolo.conf_threshold = conf
        self.net = NetCapture(page)
        self.events = []

    # ---------------- 页面基础操作
    def ev(self, name, **kw):
        rec = {"ev": name, **kw}
        self.events.append(rec)
        log(f"{name} {json.dumps(kw, ensure_ascii=False)[:260]}")

    def close_banners(self):
        for sel in ["button:has-text('知道了')", "button:has-text('我知道了')",
                    "[aria-label='Close']", "[aria-label='关闭']"]:
            try:
                el = self.page.locator(sel).first
                if el.count() and el.is_visible():
                    el.click(timeout=2000)
                    self.page.wait_for_timeout(500)
                    return True
            except Exception:
                pass
        return False

    def popup_visible(self):
        try:
            el = self.page.locator(SEL_POPUP).first
            return bool(el.count() and el.is_visible())
        except Exception:
            return False

    def read_title(self):
        try:
            el = self.page.locator(SEL_TITLE).first
            if el.count():
                return (el.inner_text() or "").strip()
        except Exception:
            pass
        return ""

    def fill_form(self, username, password):
        got = []
        for sel, val, tag in (
            ("input[name='username']", username, "u"),
            ("input[placeholder*='用户名']", username, "u"),
            ("input[name='password']", password, "p"),
            ("input[type='password']", password, "p"),
        ):
            if tag in got:
                continue
            try:
                el = self.page.locator(sel).first
                if el.count() and el.is_visible():
                    el.fill(val, timeout=5000)
                    got.append(tag)
            except Exception:
                pass
        agreed = False
        for sel in ["[role='checkbox']", "input[type='checkbox']"]:
            try:
                el = self.page.locator(sel).first
                if el.count():
                    try:
                        el.check(timeout=3000)
                    except Exception:
                        el.click(timeout=3000)
                    agreed = True
                    break
            except Exception:
                continue
        return {"filled": got, "agreed": agreed}

    def click_login(self):
        for sel in ["button[type='submit']", "button:has-text('登录')",
                    "button:has-text('登 录')"]:
            try:
                el = self.page.locator(sel).first
                if not (el.count() and el.is_visible()):
                    continue
                txt = (el.inner_text() or "").strip()
                if any(k in txt for k in ("GitHub", "LinuxDO", "Google")):
                    continue
                el.click(timeout=6000)
                return True
            except Exception:
                continue
        return False

    def wait_popup(self, timeout=20.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.popup_visible():
                return True
            self.page.wait_for_timeout(300)
        return False

    def close_popup(self):
        if not self.popup_visible():
            return True
        try:
            el = self.page.locator(SEL_CLOSE).first
            if el.count() and el.is_visible():
                el.click(timeout=2500)
                self.page.wait_for_timeout(800)
                return not self.popup_visible()
        except Exception:
            pass
        return False

    def refresh_popup(self):
        try:
            icons = self.page.locator(SEL_FOOTER_ICON)
            if icons.count() > 0:
                icons.nth(0).click(timeout=2500)
                self.page.wait_for_timeout(1500)
                return True
        except Exception:
            pass
        return False

    def bg_element_box(self):
        try:
            el = self.page.locator(SEL_BG).first
            if not (el.count() and el.is_visible()):
                return None
            b = el.bounding_box()
            if b is None or b["x"] < -1000 or b["y"] < -1000:
                return None
            return b
        except Exception:
            return None

    def fetch_bytes(self, url):
        if not url:
            return None
        full = url if url.startswith("http") else CAP_HOST + url
        try:
            r = self.ctx.request.get(full, timeout=25000)
            return r.body() if r.ok else None
        except Exception:
            return None

    # ---------------- 解题
    def get_bg_url(self):
        """从底图元素的 CSS background-image 取当前这道题的底图 URL。

        为什么用 DOM 而不是网络拦截：
            刷新验证码时，若浏览器命中缓存（同 sess 同 image 参数），
            Playwright 的 response 事件**不会触发**，拦截就漏了。
            DOM 里的 background-image 永远指向**当前**这道题，最可靠。
        """
        try:
            u = self.page.evaluate(r"""
            () => {
              const el = document.querySelector(
                '.tencent-captcha-dy__verify-bg-img');
              if (!el) return null;
              const bg = getComputedStyle(el).backgroundImage || '';
              const m = bg.match(/url\(["']?([^"')]+)["']?\)/);
              return m ? m[1] : null;
            }
            """)
            return u if u and "getcapbysig" in u else None
        except Exception:
            return None

    def get_prompt_url(self):
        """2408 的提示图 URL（header-answer 里的 img，对应 img_index=0）。"""
        try:
            u = self.page.evaluate(r"""
            () => {
              const img = document.querySelector(
                '.tencent-captcha-dy__header-answer img');
              return img ? img.src : null;
            }
            """)
            return u if u and "getcapbysig" in u else None
        except Exception:
            return None

    def solve_current(self, attempt, meta, dry_run=False):
        """解当前这一道题。返回 dict(kind, ok, pts, detail)。

        dry_run=True 时只解不点，返回 bg 坐标点位（供时序探测/闭环点击用）。
        """
        title = self.read_title()
        kind = kind_of(title)
        box = self.bg_element_box()
        self.ev("captcha", attempt=attempt, kind=kind, title=title,
                box=(None if not box else {k: round(v, 1) for k, v in box.items()}))
        if box is None:
            return {"kind": kind, "ok": False, "detail": "no_bg_box"}

        # 底图：优先 DOM URL（永远对应当前题），兜底用网络拦截到的字节
        bg = None
        bg_url = self.get_bg_url()
        if bg_url:
            b = self.fetch_bytes(bg_url)
            if b:
                bg = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if bg is None:
            nd = self.net.data or {}
            imgs = nd.get("imgs") or {}
            if 1 in imgs:
                bg = cv2.imdecode(np.frombuffer(imgs[1], np.uint8),
                                  cv2.IMREAD_COLOR)
        if bg is None:
            return {"kind": kind, "ok": False, "detail": "no_bg_img"}
        self.ev("bg_ok", attempt=attempt, wh=[bg.shape[1], bg.shape[0]])
        imwrite_u(os.path.join(self.out, f"bg_{attempt}.png"), bg)

        if kind == "2401":
            ok = self._solve_slider(box, bg, attempt)
            return {"kind": kind, "ok": ok, "detail": "slider"}

        if kind == "2404":
            chars = title.split("：", 1)[1].strip().split() if "：" in title else []
            if not chars or not self.legacy:
                return {"kind": kind, "ok": False, "detail": "no_chars_or_legacy"}
            canvas = render_text_strip(bg, chars)
            return self._solve_legacy(canvas, box, attempt, kind, chars,
                                      dry_run=dry_run)

        # 2408：取提示图（header-answer img，对应 img_index=0）
        prompt = None
        pu = self.get_prompt_url()
        if pu:
            b = self.fetch_bytes(pu)
            if b:
                prompt = cv2.imdecode(np.frombuffer(b, np.uint8),
                                      cv2.IMREAD_COLOR)
        if prompt is None:
            nd = self.net.data or {}
            imgs = nd.get("imgs") or {}
            if 0 in imgs:
                prompt = cv2.imdecode(np.frombuffer(imgs[0], np.uint8),
                                      cv2.IMREAD_COLOR)
        canvas = compose_canvas(bg, prompt)
        p_img = os.path.join(self.out, f"c2408_{attempt}.jpg")
        imwrite_u(p_img, canvas)
        # 记检测数量：class2=提示条图标、class0=底图图标。
        # 若 class2 数与真实提示数不符，说明提示图没取对/漏检，
        # 这时答案必然不完整（线上 2408 失败的主要原因之一）。
        try:
            _d0 = self.cap.yolo.inference(canvas)
            _n0 = sum(1 for x in _d0 if int(x[5]) == 0)
            _n2 = sum(1 for x in _d0 if int(x[5]) == 2)
            self.ev("det_2408", attempt=attempt, n_class0=_n0, n_class2=_n2,
                    has_prompt=prompt is not None)
        except Exception:
            pass
        t0 = time.time()
        try:
            d = self.cap.run_dict(p_img)
        except Exception as e:
            self.ev("solve_err", attempt=attempt, err=f"{type(e).__name__}: {e}")
            return {"kind": kind, "ok": False, "detail": "model_err"}
        ms = (time.time() - t0) * 1000
        corps = d.get("corp") or []
        self.ev("solved_2408", attempt=attempt, ms=round(ms, 1), n=len(corps),
                has_prompt=prompt is not None)
        if not corps:
            return {"kind": kind, "ok": False, "detail": "no_points"}
        bg_pts = []
        for c in corps:
            cx = (c["x1"] + c["x2"]) / 2.0
            cy = (c["y1"] + c["y2"]) / 2.0
            if cy > bg.shape[0]:
                continue
            bg_pts.append((cx, cy))
        pts = self._to_page(corps, box, bg.shape[1], bg.shape[0])
        if dry_run:
            return {"kind": kind, "ok": True, "detail": "dry_run",
                    "n": len(bg_pts), "bg_pts": bg_pts, "box": box}
        return self._click_points(pts, box, attempt, kind, bg_pts=bg_pts)

    def _solve_legacy(self, canvas, box, attempt, kind, chars, dry_run=False):
        y, pre = self.legacy
        p_img = os.path.join(self.out, f"c2404_{attempt}.jpg")
        imwrite_u(p_img, canvas)
        det = y.inference(canvas)
        tb = [x[:4] for x in det if x[5] == 0]
        cb = [x[:4] for x in det if x[5] == 2]
        cb.sort(key=lambda b: b[0])
        self.ev("legacy_detect", attempt=attempt, chars=chars,
                n_target=len(tb), n_char=len(cb))
        if not tb or not cb:
            return {"kind": kind, "ok": False, "detail": "legacy_no_boxes"}
        from src.utils import matchingMode
        its = [canvas[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in tb]
        ics = [canvas[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in cb]
        slys = pre.reason_all_batch(ics, its)
        order = matchingMode.find_overall_index_fast(slys)
        res = [tb[j] for _i, j in order]
        if not res:
            return {"kind": kind, "ok": False, "detail": "legacy_no_pairs"}
        W = canvas.shape[1]
        bg_h = canvas.shape[0] - STRIP_H
        sx = box["width"] / float(W)
        sy = box["height"] / float(bg_h)
        pts = []
        bg_pts = []
        for (x1, y1, x2, y2) in res:
            cy = (y1 + y2) / 2.0
            if cy > bg_h:
                continue
            bx, by = (x1 + x2) / 2.0, cy
            bg_pts.append((bx, by))
            pts.append((box["x"] + bx * sx, box["y"] + by * sy))
        self.ev("solved_2404", attempt=attempt, n=len(pts))
        if dry_run:
            return {"kind": kind, "ok": True, "detail": "dry_run",
                    "n": len(bg_pts), "bg_pts": bg_pts, "box": box}
        return self._click_points(pts, box, attempt, kind, bg_pts=bg_pts)

    def _to_page(self, corps, box, W, bg_h):
        """模型坐标 -> 页面坐标。sy 必须除以 bg_h（坑 5）。"""
        sx = box["width"] / float(W)
        sy = box["height"] / float(bg_h)
        pts = []
        for c in corps:
            cx = (c["x1"] + c["x2"]) / 2.0
            cy = (c["y1"] + c["y2"]) / 2.0
            if cy > bg_h:
                continue
            pts.append((box["x"] + cx * sx, box["y"] + cy * sy))
        return pts

    # ---------------- 点击闭环支持
    def popup_snapshot(self):
        """截弹层当前画面，返回 (ndarray, popup_rect)。

        用途：点击前后对比，判断这次点击是否被站点接收。
        天御 2408 的 click_cfg.mark_style = inc_number，
        有效点击会在图上叠加序号标记 → 对应位置像素必然变化。
        """
        try:
            el = self.page.locator(SEL_POPUP).first
            if not (el.count() and el.is_visible()):
                return None, None
            rect = el.bounding_box()
            p = os.path.join(self.out, "_snap.png")
            el.screenshot(path=p)
            return imread_u(p), rect
        except Exception:
            return None, None

    @staticmethod
    def _diff_at(a, b, lx, ly, half=55):
        """弹层内局部坐标 (lx,ly) 附近的前后差异。"""
        if a is None or b is None:
            return None
        h, w = a.shape[:2]
        x1, y1 = max(0, int(lx) - half), max(0, int(ly) - half)
        x2, y2 = min(w, int(lx) + half), min(h, int(ly) + half)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        d = cv2.absdiff(a[y1:y2, x1:x2], b[y1:y2, x1:x2])
        g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        return {"mean": round(float(g.mean()), 2),
                "n_big": int((g > 25).sum())}

    def human_click(self, px, py, hold_ms=None):
        """拟人点击：先靠近再落点，按下后保持 40~90ms 再抬起。

        为什么不用 page.mouse.click()：
            它是一次瞬时的 down+up，down 与 up 之间几乎无间隔。
            天御会采集 mousedown/mouseup 时间差与落点前的移动轨迹，
            瞬时点击是明显的机器特征。
        """
        # 先移到目标附近（模拟手的一次粗定位）
        self.page.mouse.move(px + random.uniform(-14, 14),
                             py + random.uniform(-14, 14),
                             steps=random.randint(4, 8))
        self.page.wait_for_timeout(random.randint(50, 110))
        # 再精确落到目标
        self.page.mouse.move(px + random.uniform(-1.5, 1.5),
                             py + random.uniform(-1.5, 1.5),
                             steps=random.randint(5, 10))
        self.page.wait_for_timeout(random.randint(30, 70))
        self.page.mouse.down()
        self.page.wait_for_timeout(hold_ms or random.randint(45, 90))
        self.page.mouse.up()

    def _click_points(self, pts, box, attempt, kind, bg_pts=None):
        """按顺序点击。

        【坑】box 不能沿用解题时读到的值。
            实测弹层有入场动画（缩放/位移），解题时读到的 bounding_box
            在 2~3 秒后点击时已经变了 → 点偏 → 明明答对却不通过。
            （实测 try08：模型选中 定位→电池→5，顺序与提示一致，
            但因为坐标是动画期间读的，点击全偏。）

            修法：**点击前重新读一次 box**，用它的宽高重新换算。
        """
        if not pts:
            return {"kind": kind, "ok": False, "detail": "no_pts"}

        # 重新读取实时几何
        fresh = self.bg_element_box()
        if fresh and bg_pts:
            box = fresh
            sx = box["width"] / float(BG_W)
            sy = box["height"] / float(BG_H)
            pts = [(box["x"] + bx * sx, box["y"] + by * sy)
                   for bx, by in bg_pts]
            self.ev("box_refreshed", attempt=attempt,
                    box={k: round(v, 1) for k, v in box.items()},
                    n=len(pts))

        # ---- 闭环点击：每点一次都确认「真的被接收」，没接收就重点
        #
        # 为什么必须闭环：
        #     实测失败样本里模型答案 100% 正确（tools/check_live_2408.py 肉眼核对），
        #     但站点判错 —— 因为早期点击会被交互层丢弃。
        #     只靠"加固定等待"是盲修；闭环才能自适应。
        #
        # 判据：点完等 ~380ms，比较该点附近的像素变化。
        #     天御 mark_style=inc_number，有效点击会叠加序号标记 → 必有变化。
        #     变化太弱（n_big < 12）视为未接收，最多重点 2 次。
        n_recv = 0
        for i, (px, py) in enumerate(pts):
            ok_here = False
            for try_i in range(3):
                before, prect = self.popup_snapshot()
                lx = ly = None
                if prect:
                    lx = px - prect["x"]
                    ly = py - prect["y"]
                self.human_click(px, py)
                self.page.wait_for_timeout(380)
                after, _ = self.popup_snapshot()
                d = self._diff_at(before, after, lx, ly) if (
                    lx is not None and ly is not None) else None
                got = (d is not None and d["n_big"] >= 12)
                if d is not None:
                    self.ev("click_check", attempt=attempt, idx=i + 1,
                            try_i=try_i + 1, n_big=d["n_big"],
                            mean=d["mean"], received=got)
                if got:
                    ok_here = True
                    break
                # 未接收：稍等再重点（这是补救的关键动作）
                self.page.wait_for_timeout(random.randint(180, 320))
            n_recv += 1 if ok_here else 0
            self.page.wait_for_timeout(random.randint(120, 260))

        self.ev("clicked", attempt=attempt, n=len(pts), received=n_recv,
                pts=[(round(x), round(y)) for x, y in pts])

        # 点选类（2404/2408）点完后都要按「确定」
        # 实测：2404 的弹层里同样有蓝色确定按钮，不点它验证不会提交
        # （第一版只给 2408 点了确定，导致 2404 永远停在弹层上不通过）
        #
        # 【坑】确定按钮初始是 disabled（computed cursor = not-allowed），
        # 只有在点够数量的点位后才变 pointer。提前点等于没点，
        # 验证不提交、弹层不关，最终被判失败。
        # 所以必须**等 cursor 变成 pointer** 再点。
        if kind in ("2408", "2404"):
            self.page.wait_for_timeout(700)          # 让最后一次点击先落地
            enabled = False
            for _ in range(14):                      # 最多等 ~4.2s
                try:
                    cur = self.page.evaluate(r"""
                    () => {
                      const el = document.querySelector(
                        '.tencent-captcha-dy__verify-confirm-btn');
                      if (!el) return 'missing';
                      return getComputedStyle(el).cursor;
                    }
                    """)
                except Exception:
                    cur = "err"
                if cur == "pointer":
                    enabled = True
                    break
                if cur == "missing":
                    break                            # 可能已自动提交
                self.page.wait_for_timeout(300)
            self.ev("confirm_state", attempt=attempt, kind=kind,
                    enabled=enabled)
            if enabled:
                for _ in range(3):
                    try:
                        el = self.page.locator(SEL_CONFIRM).first
                        if el.count() and el.is_visible():
                            el.click(timeout=3000)
                            self.ev("confirm_clicked", attempt=attempt,
                                    kind=kind)
                            break
                    except Exception:
                        pass
                    self.page.wait_for_timeout(350)
        self.page.wait_for_timeout(2500)
        # 点击后状态：弹层是否关闭 / 是否有错误提示 / 截图
        still = self.popup_visible()
        body = ""
        try:
            body = self.page.inner_text("body")[:1500]
        except Exception:
            pass
        err_marks = ("验证失败", "请重试", "验证错误", "重新验证", "再试一次")
        has_err = any(k in body for k in err_marks)
        self.ev("after_click", attempt=attempt, popup=still,
                err=has_err, url=self.page.url[-60:])
        try:
            self.page.screenshot(
                path=os.path.join(self.out, f"after_{attempt}.png"))
        except Exception:
            pass
        return {"kind": kind, "ok": True, "detail": "clicked",
                "n": len(pts), "popup_after": still, "err": has_err}

    # ---------------- 滑块
    def _solve_slider(self, box, bg, attempt):
        """解 2401 滑块。

        【坑】拖动距离不能直接用 gap_x。
            拼图块（`.tencent-captcha-dy__fg-item`）一开始就画在底图上，
            起点约在底图 x≈51（实测 DOM: 页面 x=580，底图 x=555，缩放 0.491）。
            所以：drag = (gap_cx - piece_cx_bg) * sx
            第一版直接用 gap_cx * sx，多拖了约 25px，必然失败。
        """
        try:
            from slider_gap import find_gap
            r = find_gap(bg, verbose=False)
            if r is None:
                self.ev("slider_no_gap", attempt=attempt)
                return False
            fg = self.page.locator(SEL_FG).first
            handle = self.page.locator(SEL_HANDLE).first
            if not (fg.count() and handle.count()):
                self.ev("slider_missing", attempt=attempt)
                return False
            fb = fg.bounding_box()
            hb = handle.bounding_box()
            if fb is None or hb is None:
                return False

            sx = box["width"] / float(bg.shape[1])
            # 拼图块中心在「底图坐标系」里的 x
            piece_cx_bg = (fb["x"] + fb["width"] / 2.0 - box["x"]) / sx
            drag = (r["cx"] - piece_cx_bg) * sx
            self.ev("slider", attempt=attempt, gap=round(r["cx"], 1),
                    gap_cy=round(r["cy"], 1), src=r["src"],
                    score=r["score"], piece_bg=round(piece_cx_bg, 1),
                    drag=round(drag, 1))
            if drag <= 1:
                self.ev("slider_drag_too_small", attempt=attempt)
                return False

            cx = hb["x"] + hb["width"] / 2
            cy = hb["y"] + hb["height"] / 2
            self.page.mouse.move(cx, cy, steps=8)
            self.page.wait_for_timeout(140)
            self.page.mouse.down()
            self.page.wait_for_timeout(110)
            steps = 26
            for k in range(1, steps + 1):
                t = k / float(steps)
                e = 1 - (1 - t) ** 2.2          # 先快后慢，像人手
                self.page.mouse.move(cx + drag * e, cy + random.uniform(-1, 1))
                self.page.wait_for_timeout(random.randint(12, 26))
            # 到位后微调 + 停顿（天御会检测轨迹）
            self.page.mouse.move(cx + drag, cy, steps=3)
            self.page.wait_for_timeout(random.randint(180, 320))
            self.page.mouse.up()
            self.page.wait_for_timeout(2600)
            return True
        except Exception as e:
            self.ev("slider_err", attempt=attempt, err=f"{type(e).__name__}: {e}")
            return False

    # ---------------- 登录结果判定
    def login_state(self):
        """返回登录态判定明细，便于定位误判原因。"""
        st = {"url": "", "has_pw": None, "bad_hit": None, "err": None}
        try:
            st["url"] = self.page.url
        except Exception as e:
            st["err"] = f"url:{type(e).__name__}"
            return st
        if "sign-in" in st["url"] or "sign-up" in st["url"]:
            st["reason"] = "url_signin"
            return st
        try:
            pw = self.page.locator("input[type='password']").first
            st["has_pw"] = bool(pw.count() and pw.is_visible())
        except Exception as e:
            st["err"] = f"pw:{type(e).__name__}"
            st["has_pw"] = False
        if st["has_pw"]:
            st["reason"] = "password_field_visible"
            return st
        try:
            body = self.page.inner_text("body")[:6000]
        except Exception as e:
            st["err"] = (st["err"] or "") + f" body:{type(e).__name__}"
            body = ""
        bad = ("验证失败", "验证错误", "用户名或密码", "密码错误", "登录失败",
               "请先勾选")
        hit = [k for k in bad if k in body]
        st["bad_hit"] = hit
        if hit:
            st["reason"] = "error_text"
            return st
        st["reason"] = "ok"
        return st

    def is_logged_in(self):
        """判定是否已登录。

        判据组合：
            1) URL 含 sign-in/sign-up  → 未登录
            2) 页面上仍有可见的密码输入框 → 未登录（还在登录表单）
            3) 页面出现错误文案 → 未登录
            4) 否则视为已登录
        """
        return self.login_state().get("reason") == "ok"

    def logout(self):
        """退出登录：先试 UI，失败则清 cookie（等效登出）。"""
        # 1) 直接访问可能的登出路径
        for path in ("/sign-out", "/logout", "/api/auth/signout"):
            try:
                self.page.goto("https://www.cun.ai" + path,
                               wait_until="domcontentloaded", timeout=20000)
                self.page.wait_for_timeout(1200)
                if "sign-in" in self.page.url:
                    self.ev("logout_by_url", path=path)
                    return True
            except Exception:
                continue
        # 2) UI 上找退出
        try:
            for sel in ["text=退出登录", "text=退出", "[aria-label*='账号']",
                        "[aria-label*='用户']"]:
                el = self.page.locator(sel).first
                if el.count() and el.is_visible():
                    el.click(timeout=2500)
                    self.page.wait_for_timeout(1000)
                    el2 = self.page.locator("text=退出登录").first
                    if el2.count() and el2.is_visible():
                        el2.click(timeout=2500)
                        self.page.wait_for_timeout(1500)
                        if "sign-in" in self.page.url:
                            self.ev("logout_by_ui")
                            return True
                    break
        except Exception:
            pass
        # 3) 清 cookie（保证下一次是未登录态）
        try:
            self.ctx.clear_cookies()
            self.goto_retry("https://www.cun.ai/sign-in", timeout=30000)
            self.page.wait_for_timeout(1500)
            self.ev("logout_by_clear_cookies")
            return True
        except Exception:
            return False

    def goto_retry(self, url, tries=3, wait="domcontentloaded", timeout=60000):
        """带重试的 goto。

        实测偶发 `net::ERR_CONNECTION_CLOSED`（50 次登录里出现 1 次），
        属于瞬断，重试即可；不重试会直接让整次尝试失败。
        """
        last = None
        for i in range(tries):
            try:
                self.page.goto(url, wait_until=wait, timeout=timeout)
                return True
            except Exception as e:
                last = e
                if i < tries - 1:
                    self.page.wait_for_timeout(1200 * (i + 1))
        self.ev("goto_fail", url=url, err=f"{type(last).__name__}")
        return False

    def wait_popup_stable(self, need_same=3, max_wait=8.0):
        """等弹层真正就绪：位置连续 need_same 次不变 + 固定稳定期。

        为什么不能只看"元素存在"：
            弹层出现瞬间 DOM 就有 rect，但交互层（点击事件绑定、
            图片解码）还没好。此时第一次点击会被丢弃，
            天御只收到剩余点击 → 判错并自动换题。
            这就是"模型答对却过不了"的根因。
        """
        last = None
        same = 0
        t0 = time.time()
        while time.time() - t0 < max_wait:
            try:
                b = self.page.locator(SEL_BG).first.bounding_box()
            except Exception:
                b = None
            if b:
                key = (round(b["x"], 1), round(b["y"], 1),
                       round(b["width"], 1), round(b["height"], 1))
                if key == last:
                    same += 1
                    if same >= need_same:
                        break
                else:
                    last = key
                    same = 0
            self.page.wait_for_timeout(220)
        self.page.wait_for_timeout(700)      # 再留一点缓冲给事件绑定

    def reload_relogin(self, username, password):
        """重新打开登录页并再点一次登录 —— 用来「重掷题型」。

        为什么需要：
            实测题型在**同一个弹层会话内是固定的**（点 footer 刷新只会换题、
            不换类型：连续 15 次都是 2401）。只有重新走一次登录流程，
            服务端才会重新下发（可能变成 2408/2404）。
        """
        try:
            if not self.goto_retry("https://www.cun.ai/sign-in"):
                return False
            self.page.wait_for_timeout(2200)
            self.close_banners()
            self.fill_form(username, password)
            self.net.arm()
            if not self.click_login():
                return False
            ok = self.wait_popup(18)
            if ok:
                self.net.wait_ready(10)
                self.wait_popup_stable()
            return ok
        except Exception:
            return False

    # ---------------- 一次完整登录尝试
    def attempt_once(self, username, password, attempt, max_refresh=5):
        """一次登录尝试。返回 dict(ok, kind, tries, ms, note)。"""
        t0 = time.time()
        res = {"attempt": attempt, "ok": False, "kinds": [],
               "solved_any": False, "note": ""}

        if not self.goto_retry("https://www.cun.ai/sign-in"):
            res["note"] = "goto_fail"
            return res
        self.page.wait_for_timeout(2200)

        self.close_banners()
        f = self.fill_form(username, password)
        if len(f["filled"]) < 2:
            res["note"] = "form_incomplete"
            return res

        self.net.arm()
        if not self.click_login():
            res["note"] = "no_login_btn"
            return res
        if not self.wait_popup(20):
            # 没弹验证码 → 可能直接登录成功
            self.page.wait_for_timeout(2500)
            res["ok"] = self.is_logged_in()
            res["note"] = "no_popup"
            res["ms"] = round((time.time() - t0) * 1000)
            return res

        self.net.wait_ready(12)
        # 等弹层完全就绪再动手。
        # 实测依据：模型在线上图上 100% 答对（tools/check_live_2408.py 肉眼核对），
        # 但点完被判错 —— 因为弹层刚出现时交互层尚未绑定，
        # **第一次点击会被吞掉**，站点只收到后两次点击 → 判定错误并换题。
        # 这里给足稳定时间，并把「绘制完成 + 位置不再变化」作为就绪条件。
        self.wait_popup_stable()

        fails = 0
        for t in range(1, max_refresh + 1):
            if not self.popup_visible():
                break
            r = self.solve_current(t, res)
            res["kinds"].append(r["kind"])

            if not r.get("ok"):
                # 压根没解出/没点到；但也可能是上一步已经登录成功
                self.ev("solve_not_ok", attempt=t, detail=r.get("detail"))
                if self.is_logged_in():
                    res["ok"] = True
                    res["captcha_accepted"] = True
                    break
                fails += 1
                self.net.arm()
                if fails >= 2:
                    if self.reload_relogin(username, password):
                        fails = 0
                        continue
                    break
                if not self.refresh_popup():
                    break
                self.net.wait_ready(10)
                self.wait_popup_stable()
                continue

            # 点完了：先看是否已经登录（最可靠，直接看 URL / 密码框）
            #
            # 【坑】登录成功后弹层 DOM 会残留，popup_visible() 仍返回 True。
            # 第一版据此判定"弹层还在=失败"，把已经成功的登录判成失败
            # （实测 URL 已跳到 /dashboard/overview 却记 ok=False）。
            # 所以必须**先判登录态**，再考虑弹层状态。
            self.page.wait_for_timeout(1800)
            st = self.login_state()
            self.ev("login_state", attempt=t, **st)
            if st.get("reason") == "ok":
                res["ok"] = True
                res["captcha_accepted"] = True
                break

            popup_after = r.get("popup_after")
            if popup_after is False:
                self.page.wait_for_timeout(2500)
                if self.is_logged_in():
                    res["ok"] = True
                    res["captcha_accepted"] = True
                    break
                res["note"] = "popup_closed_not_logged_in"
                break

            if r.get("err"):
                res["note"] = "captcha_rejected"

            # 弹层还在 → 换题重试。
            # 题型在同一会话里固定，所以失败 2 次后改为「重开页面重掷题型」。
            fails += 1
            self.net.arm()
            if fails >= 2:
                if self.reload_relogin(username, password):
                    fails = 0
                    continue
                break
            if not self.refresh_popup():
                break
            self.net.wait_ready(10)
            self.wait_popup_stable()

        res["ms"] = round((time.time() - t0) * 1000)
        return res
