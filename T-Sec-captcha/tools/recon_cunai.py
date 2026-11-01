# -*- coding: utf-8 -*-
"""recon_cunai: 侦察 cun.ai/sign-up 的真实验证码类型与取数接口。

做法（只读侦察，不真正完成注册）：
    1) CloakBrowser Pro 打开 https://cun.ai/sign-up
    2) 关掉营销弹窗，填一个一次性邮箱/密码（不提交成功也无所谓）
    3) 点"注册"触发天御验证码弹层，全程拦截 turing.captcha.qcloud.com
    4) 记录 prehandle / getcapbysig / verify 的完整 URL + 回包
    5) 落盘：JSON 原文、精灵图、按 sprite_pos 裁出的底图与提示图

产出：
    out/recon/net.jsonl              全部验证码网络事件（含完整 URL，可复现）
    out/recon/prehandle_NN.json      每次 prehandle 原文
    out/recon/img_NN_sprite.png      整张精灵图
    out/recon/img_NN_bg.png          底图（按 bg_elem_cfg 裁出）
    out/recon/img_NN_fg*.png         前景元素（滑块/提示图）
    out/recon/page_rN.png            页面截图
    out/recon/recon_report.json      结论（题型、尺寸、接口签名）

用法：
    python tools/recon_cunai.py
    python tools/recon_cunai.py --rounds 4 --headed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from urllib.parse import urlparse

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "out", "recon")
CAP_HOST = "turing.captcha.qcloud.com"

SUBCLASS_TABLE = {
    "2401": "滑块拼图 slide",
    "2402": "文字点选 char-click",
    "2403": "语序点选 word-order-click",
    "2404": "行为式（无图）",
    "2405": "空间推理 spatial",
    "2407": "语音",
    "2408": "图标点选 icon-click（请依次点击）",
    "2409": "消消乐/连连看 match",
}


def imwrite_u(path, img, q=95):
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), q] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(path)
    return ok


def imdecode_u(buf):
    return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def parse_jsonp(text):
    t = (text or "").strip()
    if t.startswith("("):
        t = t[1:]
    if t.endswith(")"):
        t = t[:-1]
    try:
        return json.loads(t)
    except Exception:
        return None


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class Recon:
    def __init__(self, args):
        self.args = args
        self.events = []
        self.prehandles = []
        self.saved_imgs = {}
        self.n = 0

    # ------------------------------------------------------------ 网络钩子
    def on_response(self, resp):
        try:
            url = resp.url
        except Exception:
            return
        if CAP_HOST not in url:
            return
        path = urlparse(url).path
        rec = {"t": round(time.time(), 3), "method": resp.request.method,
               "url": url, "path": path, "status": resp.status}
        try:
            rec["req_headers"] = dict(resp.request.headers)
        except Exception:
            pass

        if "prehandle" in path:
            try:
                body = resp.text()
            except Exception:
                body = ""
            j = parse_jsonp(body)
            rec["len"] = len(body)
            if j is not None:
                self.n += 1
                fn = os.path.join(OUT, f"prehandle_{self.n:02d}.json")
                dump(j, fn)
                rec["saved"] = os.path.basename(fn)
                rec["subcapclass"] = j.get("subcapclass")
                rec["capclass"] = j.get("capclass")
                dsi = j.get("dyn_show_info") or {}
                rec["instruction"] = dsi.get("instruction")
                rec["bg_size"] = (dsi.get("bg_elem_cfg") or {}).get("size_2d")
                self.prehandles.append({"file": fn, "n": self.n, "json": j,
                                        "url": url})
            self.events.append(rec)
            return

        if "getcapbysig" in path:
            try:
                buf = resp.body()
            except Exception:
                buf = b""
            img = imdecode_u(buf) if buf else None
            rec["len"] = len(buf)
            rec["shape"] = list(img.shape) if img is not None else None
            if img is not None:
                m = re.search(r"img_index=(\d+)", url)
                idx = m.group(1) if m else "0"
                fn = os.path.join(OUT, f"raw_n{self.n:02d}_idx{idx}.png")
                imwrite_u(fn, img, q=100)
                rec["saved"] = os.path.basename(fn)
                self.saved_imgs.setdefault(self.n, {})[f"idx{idx}"] = fn
            self.events.append(rec)
            return

        if "verify" in path:
            try:
                rec["body"] = (resp.text() or "")[:600]
            except Exception:
                pass
            self.events.append(rec)
            return

        self.events.append(rec)

    # ------------------------------------------------------------ 主流程
    def run(self):
        from cloakbrowser import launch
        os.makedirs(OUT, exist_ok=True)
        print(f"[recon] CloakBrowser Pro, headless={not self.args.headed}")
        browser = launch(headless=not self.args.headed, humanize=True,
                         locale="zh-CN", timezone="Asia/Shanghai")
        ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                                  locale="zh-CN", timezone_id="Asia/Shanghai")
        page = ctx.new_page()
        page.on("response", self.on_response)

        print(f"[recon] goto {self.args.url}")
        page.goto(self.args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3500)

        # 1) 关营销弹窗
        for sel in ["button:has-text('知道了')", "button:has-text('我知道了')",
                    ".modal button:has-text('知道了')", "[aria-label='Close']"]:
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    el.click(timeout=3000)
                    print(f"[recon] 关闭弹窗: {sel}")
                    page.wait_for_timeout(800)
                    break
            except Exception:
                continue

        # 2) 填表单（用一次性占位值；提交失败也无所谓，只要触发验证码）
        fills = [("input[name='username']", self.args.username),
                 ("input[name='password']", self.args.password),
                 ("input[name='confirmPassword']", self.args.password),
                 ("input[name='email']", self.args.email)]
        for sel, val in fills:
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    el.fill(val, timeout=4000)
                    print(f"[recon] 填入 {sel} <- {val[:3]}***")
            except Exception as e:
                print(f"[recon] 填 {sel} 失败: {type(e).__name__}")

        # 2.5) 勾选同意协议（否则按钮不提交）
        for sel in ["input[type='checkbox']", "[role='checkbox']"]:
            try:
                el = page.locator(sel).first
                if el.count():
                    el.check(timeout=3000, force=True)
                    print(f"[recon] 勾选协议: {sel}")
                    break
            except Exception:
                try:
                    el.click(timeout=2000, force=True)
                    print(f"[recon] 点击勾选协议: {sel}")
                    break
                except Exception:
                    continue

        # 3) 点"创建账户"提交（会拉起天御验证码弹层）
        clicked = False
        for sel in ["button:has-text('创建账户')", "button:has-text('创建帐户')",
                    "button[type='submit']:has-text('创建')"]:
            try:
                el = page.locator(sel).first
                if el.count() and el.is_visible():
                    txt = (el.inner_text() or "").strip()
                    el.click(timeout=5000)
                    print(f"[recon] 点击提交按钮: {sel}  text={txt!r}")
                    clicked = True
                    break
            except Exception:
                continue
        if not clicked:
            print("[recon] 未点到提交按钮")

        # 4) 等验证码弹层，多轮刷新拿样本
        for r in range(self.args.rounds):
            page.wait_for_timeout(4500)
            print(f"[recon] 第 {r+1}/{self.args.rounds} 轮，已捕获 prehandle={self.n}")
            try:
                page.screenshot(path=os.path.join(OUT, f"page_r{r+1}.png"))
            except Exception:
                pass
            # 在验证码 iframe 里点刷新
            for fr in page.frames:
                if CAP_HOST in (fr.url or ""):
                    for sel in [".tcaptcha-refresh", ".tcaptcha-reload",
                                "#reload", ".tcaptcha-btn-close"]:
                        try:
                            el = fr.locator(sel).first
                            if el.count() and el.is_visible() and "close" not in sel:
                                el.click(timeout=2500)
                                print(f"[recon] iframe 内刷新: {sel}")
                                break
                        except Exception:
                            continue
            # 兜底：直接在当前页上下文里调 prehandle（拿到干净的回包）
            if r + 1 < self.args.rounds:
                self._call_prehandle_in_page(page)

        # 5) 拉取 prehandle 声明的图片（用浏览器上下文，带完整 cookie/referer）
        for ph in self.prehandles:
            self._fetch_imgs(ctx, ph)

        # 6) 按 sprite_pos 裁剪
        self._crop_from_sprite()

        with open(os.path.join(OUT, "net.jsonl"), "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self._report()
        try:
            browser.close()
        except Exception:
            pass

    def _call_prehandle_in_page(self, page):
        """在页面上下文里直接 fetch 一次 prehandle，产出可复现的接口签名。"""
        js = """
        async (aid) => {
          const ua = navigator.userAgent;
          const p = new URLSearchParams({
            aid: String(aid), protocol: 'https', accver: '1', showtype: 'popup',
            ua: ua, noheader: '1', fb: '1', aged: '0', enableAged: '1',
            enableDarkMode: '0', grayscale: '1', dyeid: '1', clientype: '1',
            cap_cd: '', uid: '', lang: 'zh-cn',
            entry_url: location.origin + '/sign-up', elder_captcha: '0',
            login_appid: '', wb: '2', subsid: '1'
          });
          const r = await fetch('https://turing.captcha.qcloud.com/cap_union_prehandle?' + p,
                                {credentials: 'include'});
          return await r.text();
        }
        """
        try:
            txt = page.evaluate(js, int(self.args.aid))
            j = parse_jsonp(txt)
            if j:
                self.n += 1
                fn = os.path.join(OUT, f"prehandle_{self.n:02d}.json")
                dump(j, fn)
                self.prehandles.append({"file": fn, "n": self.n, "json": j,
                                        "url": "in-page fetch"})
                print(f"[recon] in-page prehandle -> {os.path.basename(fn)} "
                      f"subcapclass={j.get('subcapclass')}")
        except Exception as e:
            print(f"[recon] in-page prehandle 失败: {type(e).__name__}: {e}")

    def _fetch_imgs(self, ctx, ph):
        dsi = ph["json"].get("dyn_show_info") or {}
        pairs = []
        if (dsi.get("bg_elem_cfg") or {}).get("img_url"):
            pairs.append(("bgimg", dsi["bg_elem_cfg"]["img_url"]))
        if dsi.get("sprite_url"):
            pairs.append(("sprite", dsi["sprite_url"]))
        for tag, u in pairs:
            full = u if u.startswith("http") else "https://" + CAP_HOST + u
            try:
                resp = ctx.request.get(full, timeout=30000,
                                       headers={"Referer": "https://cun.ai/"})
                buf = resp.body()
                img = imdecode_u(buf)
                if img is None:
                    continue
                fn = os.path.join(OUT, f"img_{ph['n']:02d}_{tag}.png")
                imwrite_u(fn, img, q=100)
                self.saved_imgs.setdefault(ph["n"], {})[tag] = fn
                print(f"[recon] 存图 {os.path.basename(fn)} {img.shape}")
            except Exception as e:
                print(f"[recon] 拉图失败 {tag}: {type(e).__name__}: {e}")

    def _crop_from_sprite(self):
        for ph in self.prehandles:
            n = ph["n"]
            imgs = self.saved_imgs.get(n) or {}
            sp_path = imgs.get("sprite")
            if not sp_path or not os.path.exists(sp_path):
                continue
            sprite = cv2.imdecode(np.fromfile(sp_path, dtype=np.uint8),
                                  cv2.IMREAD_UNCHANGED)
            if sprite is None:
                continue
            dsi = ph["json"].get("dyn_show_info") or {}
            bgcfg = dsi.get("bg_elem_cfg") or {}
            crops = {}
            if bgcfg.get("sprite_pos") and bgcfg.get("size_2d"):
                x, y = bgcfg["sprite_pos"]
                w, h = bgcfg["size_2d"]
                crops["bg"] = sprite[y:y + h, x:x + w]
            for e in (dsi.get("fg_elem_list") or []):
                pos, size = e.get("sprite_pos"), e.get("size_2d")
                if not pos or not size:
                    continue
                x, y = pos
                w, h = size
                crops[f"fg{e.get('id')}"] = sprite[y:y + h, x:x + w]
            for name, sub in crops.items():
                if sub is None or sub.size == 0:
                    continue
                fn = os.path.join(OUT, f"img_{n:02d}_{name}.png")
                imwrite_u(fn, sub, q=100)
                print(f"[recon] 裁出 {os.path.basename(fn)} {sub.shape}")

    def _report(self):
        rounds = []
        for ph in self.prehandles:
            j = ph["json"]
            dsi = j.get("dyn_show_info") or {}
            bgcfg = dsi.get("bg_elem_cfg") or {}
            fgs = dsi.get("fg_elem_list") or []
            rounds.append({
                "file": os.path.basename(ph["file"]),
                "capclass": j.get("capclass"),
                "subcapclass": j.get("subcapclass"),
                "kind": SUBCLASS_TABLE.get(str(j.get("subcapclass")), "未知"),
                "state": j.get("state"),
                "instruction": dsi.get("instruction"),
                "bg_size_2d": bgcfg.get("size_2d"),
                "bg_sprite_pos": bgcfg.get("sprite_pos"),
                "fg_elem_ids": [e.get("id") for e in fgs],
                "fg_elem_sizes": [e.get("size_2d") for e in fgs],
                "fg_elem_types": [e.get("type") for e in fgs],
                "has_pow": bool(((j.get("data") or {}).get("comm_captcha_cfg") or {})
                                .get("pow_cfg")),
                "sess_len": len(j.get("sess") or ""),
                "sid": j.get("sid"),
                "uip": j.get("uip"),
            })
        hist = {}
        for r in rounds:
            hist[str(r["subcapclass"])] = hist.get(str(r["subcapclass"]), 0) + 1
        clickable = [k for k in hist if k in ("2402", "2403", "2408", "2409")]
        rep = {
            "url": self.args.url,
            "aid": self.args.aid,
            "captcha_host": CAP_HOST,
            "n_prehandle": len(self.prehandles),
            "n_net_events": len(self.events),
            "subcapclass_hist": hist,
            "clickable_subcapclass": clickable,
            "is_click_captcha": bool(clickable),
            "rounds": rounds,
            "endpoints": self._endpoints(),
        }
        dump(rep, os.path.join(OUT, "recon_report.json"))
        print("\n=== 侦察结论 ===")
        print(f"  prehandle 次数 : {rep['n_prehandle']}")
        print(f"  subcapclass    : {hist}")
        print(f"  点选题命中     : {rep['is_click_captcha']} {clickable}")
        for r in rounds[:4]:
            print(f"    - {r['file']}: sub={r['subcapclass']}({r['kind']}) "
                  f"instr={r['instruction']!r} bg={r['bg_size_2d']} "
                  f"fg={r['fg_elem_sizes']}")
        print(f"\n产物: {OUT}")

    def _endpoints(self):
        eps = {}
        for e in self.events:
            if "prehandle" in e.get("path", ""):
                eps.setdefault("prehandle", e["url"])
            elif "getcapbysig" in e.get("path", ""):
                eps.setdefault("getcapbysig", e["url"])
            elif "verify" in e.get("path", ""):
                eps.setdefault("verify", e["url"])
        return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://cun.ai/sign-up")
    ap.add_argument("--aid", default="195317439", help="腾讯验证码 appid")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--username", default="tsecprobe001")
    ap.add_argument("--password", default="Probe!2026#tsec")
    ap.add_argument("--email", default="tsecprobe001@example.com")
    args = ap.parse_args()
    Recon(args).run()


if __name__ == "__main__":
    main()
