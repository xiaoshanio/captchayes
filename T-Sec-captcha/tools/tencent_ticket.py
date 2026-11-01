# -*- coding: utf-8 -*-
"""tencent_ticket: 为 cun.ai 2.0（纯 HTTP 架构）产出天御 ticket/randstr。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
为什么必须是「HTTP 解题 + 浏览器提交」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
实测确认（tools/probe_http_captcha.py）：

    取题 cap_union_prehandle   ✅ 纯 HTTP 可行
    取图 getcapbysig           ✅ 纯 HTTP 可行
    本地 ONNX 求解             ✅ 可行
    提交 cap_union_new_verify  ❌ 纯 HTTP 不可行（errorCode=202）

因为真实提交只带一个参数 `collect`，它是**服务端密钥的分组加密串**：

    URL 解码后 570 字符 → base64 解码 427 字节 → 高熵二进制
    zlib/gzip 均无法解压；长度按 16 字节分组浮动

它封装了 sess/answer/鼠标轨迹/时间戳，由页面混淆 JS 用密钥生成。
逆它需要密钥 + 伪造合理轨迹，成本极高且不稳定。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本模块的做法（方案 A）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1) 纯 HTTP 取题取图（不启动浏览器也能做，省一个浏览器会话）
    2) 本地模型求解出点击坐标
    3) 用「极简浏览器」打开站点 → 回放点击 → 由站点 JS 自己生成 collect
    4) 拦截 verify 响应，取 ticket / randstr 返回

浏览器只承担提交这一件事，2.0 的 HTTP 主体架构不变。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
限流（必须）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
实测应答码与成败零例外对应：
    errorCode="0"  → 通过 8/8
    errorCode="12" → 拒绝 11/11
连续猛跑 8 次全部 12（连滑块也被拒），冷却 10 分钟后 4/4 全部 0。
所以本模块内置「连续 12 → 指数退避」。

用法（作为库）：
    from tencent_ticket import fetch_ticket
    t = fetch_ticket(entry_url="https://www.cun.ai/sign-up", aid="195317439")
    # t = {"ticket": "...", "randstr": "...", "errorCode": "0"}

用法（命令行自测）：
    python tools/tencent_ticket.py --entry https://www.cun.ai/sign-up
"""
from __future__ import annotations

import argparse
import json
import os
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
CAP = "https://turing.captcha.qcloud.com"
DEFAULT_AID = "195317439"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

SEL_POPUP = ".tencent-captcha__transform, #tcaptcha_transform_dy"
SEL_TITLE = ".tencent-captcha-dy__header-text"
SEL_CONFIRM = ".tencent-captcha-dy__verify-confirm-btn"


def log(m):
    print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] {m}", flush=True)


def parse_jsonp(t):
    """天御响应两种形态都要支持（实测都出现过）。

        ({"state":1,...})            仅外层括号
        _aq_128199({"state":1,...})  有函数名
    """
    t = (t or "").strip()
    if not t:
        raise ValueError("empty response")
    m = re.match(r"^\s*[\w$.]+\((.*)\)\s*;?\s*$", t, re.S)
    if m:
        return json.loads(m.group(1))
    if t.startswith("(") and t.endswith(")"):
        return json.loads(t[1:-1])
    return json.loads(t)


# ---------------------------------------------------------------- HTTP 取题
def http_fetch_challenge(entry_url, aid=DEFAULT_AID, want="2408", tries=25,
                         session=None):
    """纯 HTTP 取一道目标题型，返回 (bg, prompt, meta) 或 None。"""
    import requests
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", UA)
    s.headers.setdefault("Referer", entry_url)

    pre = None
    for i in range(tries):
        q = {
            "aid": str(aid), "protocol": "https", "accver": "1",
            "showtype": "popup", "ua": UA, "noheader": "1", "fb": "1",
            "aged": "0", "enableAged": "1", "enableDarkMode": "0",
            "grayscale": "1", "dyeid": "1", "clientype": "1",
            "cap_cd": "", "uid": "", "lang": "zh-cn",
            "entry_url": entry_url, "elder_captcha": "0",
            "login_appid": "", "wb": "2", "subsid": "1",
        }
        try:
            r = s.get(f"{CAP}/cap_union_prehandle", params=q, timeout=20)
            j = parse_jsonp(r.text)
        except Exception:
            continue
        if str(j.get("subcapclass")) == want:
            pre = j
            break
        time.sleep(0.25)
    if pre is None:
        return None

    data = pre.get("data") or {}
    dsi = data.get("dyn_show_info") or {}
    ins = dsi.get("ins_elem_cfg") or []
    bg_cfg = dsi.get("bg_elem_cfg") or {}

    def get(u):
        if not u:
            return None
        full = u if u.startswith("http") else CAP + u
        try:
            r = s.get(full, timeout=25)
            return r.content if r.ok else None
        except Exception:
            return None

    bg_b = get(bg_cfg.get("img_url"))
    bg = cv2.imdecode(np.frombuffer(bg_b, np.uint8), cv2.IMREAD_COLOR) \
        if bg_b else None
    prompt = None
    if ins:
        sp_b = get(dsi.get("sprite_url"))
        sprite = cv2.imdecode(np.frombuffer(sp_b, np.uint8),
                              cv2.IMREAD_COLOR) if sp_b else None
        if sprite is not None:
            pos = ins[0].get("sprite_pos") or [0, 0]
            sz = ins[0].get("size_2d") or [170, 50]
            prompt = sprite[int(pos[1]):int(pos[1]) + int(sz[1]),
                            int(pos[0]):int(pos[0]) + int(sz[0])]
    if bg is None or prompt is None:
        return None
    return bg, prompt, {"sess": data.get("sess") or "",
                        "subcapclass": str(pre.get("subcapclass")),
                        "instruction": dsi.get("instruction")}


# ---------------------------------------------------------------- 本地求解
def solve_local(bg, prompt):
    """用本仓库模型求解，返回底图坐标点列表 [(x, y), ...]。"""
    import cunai_login_lib as L
    from src.captcha import TextSelectCaptcha
    canvas = L.compose_canvas(bg, prompt)
    tmp = os.path.join(REPO, "out", "ticket_tmp.jpg")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    L.imwrite_u(tmp, canvas)
    cap = TextSelectCaptcha()
    cap.yolo.conf_threshold = 0.25
    res = cap.run_dict(tmp)
    return [((c["x1"] + c["x2"]) / 2.0, (c["y1"] + c["y2"]) / 2.0)
            for c in (res.get("corp") or [])]


# ---------------------------------------------------------------- 浏览器提交
def browser_submit(entry_url, bg_pts, aid=DEFAULT_AID, out_dir=None,
                   cooldown_on_12=60):
    """用极简浏览器完成提交，返回 {"ticket","randstr","errorCode"}。

    ⚠️ 已弃用（保留作为负面证据）：
        这个函数接收「外部算好的 bg_pts」再去浏览器里点。
        实测必然失败 —— 因为外部（HTTP 取题）拿到的题与浏览器弹层里的
        题**不是同一道**，点错题坐标不会触发校验，返回 no_verify_response。

        正确做法见 `browser_fetch_ticket()`：取题/解题/点击/提交
        全部在同一个浏览器会话内完成。
    """
    from cloakbrowser import launch
    import cunai_login_lib as L

    out_dir = out_dir or os.path.join(REPO, "out", "ticket")
    os.makedirs(out_dir, exist_ok=True)
    result = {"ticket": "", "randstr": "", "errorCode": "", "sess": ""}

    browser = launch(headless=True, humanize=True, locale="zh-CN",
                     timezone="Asia/Shanghai")
    try:
        ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                                  locale="zh-CN",
                                  timezone_id="Asia/Shanghai")
        page = ctx.new_page()
        caps = []

        def on_resp(r):
            if "cap_union_new_verify" not in r.url:
                return
            try:
                j = parse_jsonp(r.text())
            except Exception:
                return
            caps.append(j)

        page.on("response", on_resp)
        page.goto(entry_url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2600)

        # 关营销横幅
        for sel in ["button:has-text('知道了')", "[aria-label='Close']"]:
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    el.click(timeout=2000)
                    page.wait_for_timeout(500)
                    break
            except Exception:
                pass

        # 触发弹层：点页面上的注册/登录按钮
        triggered = False
        for sel in ["button[type='submit']", "button:has-text('注册')",
                    "button:has-text('登录')", "button:has-text('立即')"]:
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    el.click(timeout=5000)
                    triggered = True
                    break
            except Exception:
                continue
        if not triggered:
            return {**result, "error": "no_trigger_button"}

        # 等弹层稳定
        t0 = time.time()
        stable = 0
        last = None
        while time.time() - t0 < 12:
            try:
                b = page.locator(SEL_POPUP).first.bounding_box()
            except Exception:
                b = None
            if b:
                key = (round(b["x"], 1), round(b["y"], 1),
                       round(b["width"], 1))
                if key == last:
                    stable += 1
                    if stable >= 3:
                        break
                else:
                    last, stable = key, 0
            page.wait_for_timeout(220)
        page.wait_for_timeout(700)

        # 取底图元素几何
        try:
            box = page.locator(L.SEL_BG).first.bounding_box()
        except Exception:
            box = None
        if not box:
            return {**result, "error": "no_bg_box"}

        # 按底图坐标系换算并点击（humanize 在 launch 里已开启）
        n_click = 0
        for bx, by in bg_pts:
            px = box["x"] + bx * box["width"] / float(L.BG_W)
            py = box["y"] + by * box["height"] / float(L.BG_H)
            page.mouse.move(px + 1, py + 1, steps=6)
            page.wait_for_timeout(90)
            page.mouse.down()
            page.wait_for_timeout(60)
            page.mouse.up()
            n_click += 1
            page.wait_for_timeout(280)

        # 等确定按钮可用再点
        page.wait_for_timeout(700)
        for _ in range(14):
            try:
                cur = page.evaluate(
                    "() => { const e=document.querySelector("
                    "'.tencent-captcha-dy__verify-confirm-btn');"
                    "return e?getComputedStyle(e).cursor:'missing';}")
            except Exception:
                cur = "err"
            if cur == "pointer":
                try:
                    page.locator(SEL_CONFIRM).first.click(timeout=3000)
                except Exception:
                    pass
                break
            if cur == "missing":
                break
            page.wait_for_timeout(300)

        # 等 verify 响应
        t1 = time.time()
        while time.time() - t1 < 12 and not caps:
            page.wait_for_timeout(250)

        if caps:
            j = caps[-1]
            result.update({
                "ticket": j.get("ticket") or "",
                "randstr": j.get("randstr") or "",
                "errorCode": str(j.get("errorCode")),
                "sess": (j.get("sess") or "")[:24],
            })
        else:
            result["error"] = "no_verify_response"
        result["n_click"] = n_click
    finally:
        try:
            browser.close()
        except Exception:
            pass
    return result


def browser_fetch_ticket(entry_url, aid=DEFAULT_AID, out_dir=None,
                         username="", password="", email="",
                         max_wait=25):
    """用「同一次浏览器会话」完成取题→解题→提交→取票。

    【关键】不能「HTTP 取题 + 浏览器提交」混合。
        实测混合会返回 no_verify_response —— 因为 HTTP 取到的是**另一道题**，
        浏览器弹层里显示的是它自己那道题，点错误题的坐标自然不会触发校验。

        正确做法：全部在浏览器里完成（取题、取图、求解、点击、提交），
        我们只从 `cap_union_new_verify` 响应里读出 ticket/randstr。

    为什么要填表单：
        不填账号密码直接点提交，站点会在前端就拦住，根本不发 verify 请求
        （实测 errorCode 为空、no_verify_response）。

    返回 {"ticket","randstr","errorCode","sess","n_click",...}
    """
    from cloakbrowser import launch
    import cunai_login_lib as L
    from src.captcha import TextSelectCaptcha
    from src.utils import yolo_onnx, ver_onnx

    out_dir = out_dir or os.path.join(REPO, "out", "ticket")
    os.makedirs(out_dir, exist_ok=True)
    result = {"ticket": "", "randstr": "", "errorCode": "", "sess": ""}

    # 旧域模型（2404 用）
    legacy = None
    try:
        import shutil
        ldir = os.path.join(REPO, "runs", "legacy")
        os.makedirs(ldir, exist_ok=True)
        for s_, d_ in (
            (os.path.join(REPO, "runs", "_model_backup", "best_v3.onnx.bak"),
             os.path.join(ldir, "best_v3.onnx")),
            (os.path.join(REPO, "runs", "_model_backup", "pre_model_v7.onnx.bak"),
             os.path.join(ldir, "pre_model_v7.onnx")),
        ):
            if os.path.exists(s_) and not os.path.exists(d_):
                shutil.copyfile(s_, d_)
        legacy = (yolo_onnx.YOLO(os.path.join(ldir, "best_v3.onnx"),
                                 conf_threshold=0.3),
                  ver_onnx.PreONNX(os.path.join(ldir, "pre_model_v7.onnx")))
    except Exception:
        legacy = None

    cap = TextSelectCaptcha()
    cap.yolo.conf_threshold = 0.25

    browser = launch(headless=True, humanize=True, locale="zh-CN",
                     timezone="Asia/Shanghai")
    try:
        ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                                  locale="zh-CN",
                                  timezone_id="Asia/Shanghai")
        page = ctx.new_page()
        bot = L.CunaiLogin(page, ctx, cap, legacy, out_dir=out_dir)

        caps = []

        def on_resp(r):
            if "cap_union_new_verify" not in r.url:
                return
            try:
                caps.append(parse_jsonp(r.text()))
            except Exception:
                pass

        page.on("response", on_resp)

        if not bot.goto_retry(entry_url):
            return {**result, "error": "goto_fail"}
        page.wait_for_timeout(2600)
        bot.close_banners()

        # 填表单（注册页/登录页字段不同，尽量都试）
        filled = 0
        for sel, val in (("input[name='username']", username),
                         ("input[name='email']", email),
                         ("input[type='email']", email),
                         ("input[name='password']", password),
                         ("input[type='password']", password)):
            if not val:
                continue
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    el.fill(val, timeout=4000)
                    filled += 1
            except Exception:
                pass
        # 勾协议
        for sel in ["[role='checkbox']", "input[type='checkbox']"]:
            try:
                el = page.locator(sel).first
                if el.count():
                    try:
                        el.check(timeout=2500)
                    except Exception:
                        el.click(timeout=2500)
                    break
            except Exception:
                continue
        log(f"  表单已填 {filled} 项")

        # 点提交触发弹层
        bot.net.arm()
        if not bot.click_login() or not bot.wait_popup(max_wait):
            return {**result, "error": "no_popup"}
        bot.net.wait_ready(10)
        bot.wait_popup_stable()

        # 解题并点击（复用库里的完整逻辑）
        title = bot.read_title()
        kind = L.kind_of(title)
        log(f"  题型 {kind}  标题={title!r}")
        r = bot.solve_current(1, {})
        log(f"  解题 -> ok={r.get('ok')} n={r.get('n')} "
            f"detail={r.get('detail')}")
        if not r.get("ok"):
            return {**result, "error": f"solve_{r.get('detail')}",
                    "kind": kind}

        # 等 verify 响应
        t1 = time.time()
        while time.time() - t1 < 15 and not caps:
            page.wait_for_timeout(250)

        if caps:
            j = caps[-1]
            result.update({
                "ticket": j.get("ticket") or "",
                "randstr": j.get("randstr") or "",
                "errorCode": str(j.get("errorCode")),
                "sess": (j.get("sess") or "")[:24],
            })
        else:
            result["error"] = "no_verify_response"
            try:
                page.screenshot(path=os.path.join(out_dir, "no_verify.png"))
            except Exception:
                pass
        result["kind"] = kind
    finally:
        try:
            browser.close()
        except Exception:
            pass
    return result


# ---------------------------------------------------------------- 对外入口
def fetch_ticket(entry_url, aid=DEFAULT_AID, max_attempts=3,
                 cooldown_on_12=60, out_dir=None, username="", password="",
                 email=""):
    """取一张可用的天御票据（ticket + randstr）。

    内置限流：连续遇到 errorCode=12 时按 60→180→600s 退避。
    返回 {"ticket","randstr","errorCode",...}；成功时 errorCode=="0"。
    """
    backoff = [cooldown_on_12, cooldown_on_12 * 3, cooldown_on_12 * 10]
    last = {"ticket": "", "randstr": "", "errorCode": ""}
    consec12 = 0

    for attempt in range(1, max_attempts + 1):
        log(f"第 {attempt}/{max_attempts} 次取票")
        r = browser_fetch_ticket(entry_url, aid=aid, out_dir=out_dir,
                                 username=username, password=password,
                                 email=email)
        log(f"  结果 errorCode={r.get('errorCode')!r} "
            f"ticket={'有' if r.get('ticket') else '无'}"
            f"{' err=' + r['error'] if r.get('error') else ''}")
        last = r
        if r.get("errorCode") == "0" and r.get("ticket"):
            return r

        if r.get("errorCode") == "12":
            consec12 += 1
            wait = backoff[min(consec12 - 1, len(backoff) - 1)]
            log(f"  命中风控（12），冷却 {wait}s")
            time.sleep(wait)
        else:
            time.sleep(3)

    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="https://www.cun.ai/sign-up")
    ap.add_argument("--aid", default=DEFAULT_AID)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--username", default="", help="注册页需要填，否则不发 verify")
    ap.add_argument("--password", default="")
    ap.add_argument("--email", default="")
    args = ap.parse_args()

    r = fetch_ticket(args.entry, aid=args.aid, max_attempts=args.attempts,
                     username=args.username, password=args.password,
                     email=args.email)
    print("\n" + "=" * 60)
    print(f"  errorCode : {r.get('errorCode')!r}")
    print(f"  ticket    : {r.get('ticket', '')[:40]}")
    print(f"  randstr   : {r.get('randstr', '')[:40]}")
    if r.get("error"):
        print(f"  error     : {r['error']}")
    print("=" * 60)
    os.makedirs(os.path.join(REPO, "out", "ticket"), exist_ok=True)
    p = os.path.join(REPO, "out", "ticket", "last.json")
    json.dump(r, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"-> {os.path.relpath(p, REPO)}")


if __name__ == "__main__":
    main()
