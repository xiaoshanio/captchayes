# -*- coding: utf-8 -*-
"""collect_2408: 用 CloakBrowser Pro 持续采集天御 2408（图标点选）样本。

为什么用 CloakBrowser 而不是裸 requests：
    天御 prehandle 会校验 UA/TLS/环境指纹，并对同 IP 高频请求降级成 2404（行为式）。
    裸 HTTP 抽样实测 2408 命中率会掉；走真实 Chromium 网络栈 + 完整 cookie 更稳。
    另外 cun.ai 有 Cloudflare，裸请求容易拿不到 entry_url 关联的会话。

采集链路（全部在浏览器上下文里发起，图片以 base64 回传）：
    1) page.goto(cun.ai/sign-up)                 拿域 cookie / 过 CF
    2) page.evaluate(fetch prehandle)            取题目 JSON（含 subcapclass）
    3) 只保留 subcapclass == 2408
    4) fetch sprite_url (img_index=0)            整张精灵图（含提示图）
       fetch bg_elem_cfg.img_url (img_index=1)   底图 672x480
    5) 按 ins_elem_cfg 从精灵图裁出提示图 170x50
    6) 落盘成下游管线要的 data/samples/<sid>/ 结构

产出（每个样本一个目录）：
    data/samples/<sid>/bg.jpg          底图 672x480（不含提示条）
    data/samples/<sid>/prompt_0.jpg    提示图 170x50
    data/samples/<sid>/main.jpg        合成图（bg + 底部提示条），供人眼核对
    data/samples/<sid>/meta.json       sid/subcapclass/尺寸/时间/接口参数
    data/samples/collect_report.json   汇总

用法：
    python tools/collect_2408.py --target 300
    python tools/collect_2408.py --target 300 --delay 1.2 --headed
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SAMPLES = os.path.join(REPO, "data", "samples")
CAP = "https://turing.captcha.qcloud.com"
TZ = timezone(timedelta(hours=8))

# 在页面里执行：取一次 prehandle，命中 2408 就把两张图一起 base64 回来
FETCH_JS = r"""
async (cfg) => {
  const ua = navigator.userAgent;
  const p = new URLSearchParams({
    aid: String(cfg.aid), protocol: 'https', accver: '1', showtype: 'popup',
    ua: ua, noheader: '1', fb: '1', aged: '0', enableAged: '1',
    enableDarkMode: '0', grayscale: '1', dyeid: '1', clientype: '1',
    cap_cd: '', uid: '', lang: 'zh-cn',
    entry_url: location.origin + '/sign-up', elder_captcha: '0',
    login_appid: '', wb: '2', subsid: '1'
  });
  const r = await fetch('https://turing.captcha.qcloud.com/cap_union_prehandle?' + p,
                        {credentials: 'include'});
  let txt = await r.text();
  txt = txt.trim();
  if (txt.startsWith('(')) txt = txt.slice(1);
  if (txt.endsWith(')')) txt = txt.slice(0, -1);
  const j = JSON.parse(txt);
  const out = {subcapclass: String(j.subcapclass), raw: j};
  if (out.subcapclass !== cfg.want) return out;

  const dsi = (j.data || {}).dyn_show_info || {};
  const bgcfg = dsi.bg_elem_cfg || {};
  const toB64 = async (u) => {
    if (!u) return null;
    const full = u.startsWith('http') ? u : 'https://turing.captcha.qcloud.com' + u;
    const rr = await fetch(full, {credentials: 'include'});
    const ab = await rr.arrayBuffer();
    let s = '';
    const bytes = new Uint8Array(ab);
    const CH = 0x8000;
    for (let i = 0; i < bytes.length; i += CH) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
    }
    return btoa(s);
  };
  out.sprite = await toB64(dsi.sprite_url);
  out.bg = await toB64(bgcfg.img_url);
  return out;
}
"""


def imwrite_u(path, img, q=96):
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), q] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(path)
    return ok


def b64_to_img(s):
    if not s:
        return None
    return cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8),
                        cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ HTTP 通道
# 为什么需要 HTTP 通道：
#   CloakBrowser Pro 有并发席位限制（1 个），席位被占时 launch 直接抛
#   CloakBrowserLicenseError。持续采集必须能降级到纯 HTTP，不占席位也能跑。
#   实测裸 HTTP 的 2408 命中率约 33%（2401/2404/2408 混合），完全够用。
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
]


class HttpChannel:
    """裸 HTTP 取题通道。每个样本换一次 UA，失败自动重连。"""

    def __init__(self, aid, entry):
        import requests
        self.requests = requests
        self.aid = aid
        self.entry = entry
        self.sess = None
        self.ua = None
        self._rotate()

    def _rotate(self):
        if self.sess is not None:
            try:
                self.sess.close()
            except Exception:
                pass
        self.ua = random.choice(UA_POOL)
        self.sess = self.requests.Session()
        self.sess.headers.update({
            "User-Agent": self.ua,
            "Referer": "https://cun.ai/",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        })

    def fetch(self):
        """返回 (subcapclass, raw_json, sprite_bytes, bg_bytes)。"""
        params = {
            "aid": str(self.aid), "protocol": "https", "accver": "1",
            "showtype": "popup", "ua": self.ua, "noheader": "1", "fb": "1",
            "aged": "0", "enableAged": "1", "enableDarkMode": "0",
            "grayscale": "1", "dyeid": "1", "clientype": "1", "cap_cd": "",
            "uid": "", "lang": "zh-cn", "entry_url": self.entry,
            "elder_captcha": "0", "login_appid": "", "wb": "2", "subsid": "1",
        }
        r = self.sess.get(CAP + "/cap_union_prehandle", params=params, timeout=25)
        t = (r.text or "").strip()
        if t.startswith("("):
            t = t[1:]
        if t.endswith(")"):
            t = t[:-1]
        j = json.loads(t)
        sub = str(j.get("subcapclass"))
        if sub != "2408":
            return sub, j, None, None
        dsi = (j.get("data") or {}).get("dyn_show_info") or {}
        bgcfg = dsi.get("bg_elem_cfg") or {}

        def get(u):
            if not u:
                return None
            full = u if u.startswith("http") else CAP + u
            rr = self.sess.get(full, timeout=30)
            return rr.content if rr.status_code == 200 else None

        return sub, j, get(dsi.get("sprite_url")), get(bgcfg.get("img_url"))

    def close(self):
        try:
            self.sess.close()
        except Exception:
            pass


def compose_main(bg, prompt, strip_h=56, margin=6):
    """底图 + 提示条（**等比放置**）→ 与 autolabel_2408.compose 完全一致的版式。

    【必须与 autolabel_2408 一致 —— 这是踩过的坑】
        最初这里把 170x50 提示图 resize 到 672 宽（3.95 倍），
        而 tools/autolabel_2408.py 用的是等比放置（原尺寸贴左侧）。
        结果：同一题在 data/samples/<sid>/main.jpg 与
        data/labeled2408/images/<split>/<sid>.jpg 里提示条位置完全不同，
        检测器在标注图上 class2 召回 100%，在 main.jpg 上只有 1~2/3
        —— 端到端验收直接失效，且极难定位（两张图长得"差不多"）。

        现在两边都用：提示图原像素贴左侧、垂直居中、其余填白。
    """
    h, w = bg.shape[:2]
    canvas = np.full((h + strip_h, w, 3), 255, np.uint8)
    canvas[:h, :w] = bg
    ph, pw = prompt.shape[:2]
    ty = h + max(0, (strip_h - ph) // 2)
    tx = margin
    tx2, ty2 = min(w, tx + pw), min(h + strip_h, ty + ph)
    canvas[ty:ty2, tx:tx2] = prompt[:ty2 - ty, :tx2 - tx]
    return canvas


def save_sample(out_dir, sid_base, j, dsi, sprite, bg, args, channel):
    """落盘一个 2408 样本，返回 sid 或 None。"""
    if bg is None:
        return None
    prompt = None
    for e in (dsi.get("ins_elem_cfg") or []):
        pos, size = e.get("sprite_pos"), e.get("size_2d")
        if pos and size and sprite is not None:
            x, y = pos
            w, h = size
            sub_img = sprite[y:y + h, x:x + w]
            if sub_img.size:
                prompt = sub_img
                break
    if prompt is None:
        return None

    sid = f"t2408_{sid_base}"
    d = os.path.join(out_dir, sid)
    os.makedirs(d, exist_ok=True)
    imwrite_u(os.path.join(d, "bg.jpg"), bg)
    imwrite_u(os.path.join(d, "prompt_0.jpg"), prompt)
    imwrite_u(os.path.join(d, "main.jpg"), compose_main(bg, prompt, args.strip_h))

    meta = {
        "sid": sid,
        "source": "cun.ai/sign-up",
        "aid": args.aid,
        "channel": channel,
        "subcapclass": str(j.get("subcapclass")),
        "instruction": dsi.get("instruction"),
        "capclass": j.get("capclass"),
        "state": j.get("state"),
        "bg_size": (dsi.get("bg_elem_cfg") or {}).get("size_2d"),
        "click_cfg": (dsi.get("bg_elem_cfg") or {}).get("click_cfg"),
        "ins_elem_cfg": dsi.get("ins_elem_cfg"),
        "verify_trigger_cfg": dsi.get("verify_trigger_cfg"),
        "has_sprite": bool(dsi.get("sprite_url")),
        "has_bg": bool((dsi.get("bg_elem_cfg") or {}).get("img_url")),
        "sess_len": len(j.get("sess") or ""),
        "turing_sid": j.get("sid"),
        "uip": j.get("uip"),
        "collected_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "prompt_shape": list(prompt.shape),
        "bg_shape": list(bg.shape),
        "strip_h": args.strip_h,
        # 下游 tools/icon_assets.py 读这个字段切提示图
        "prompts": [{"file": "prompt_0.jpg",
                     "w": int(prompt.shape[1]), "h": int(prompt.shape[0])}],
    }
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aid", default="195317439")
    ap.add_argument("--target", type=int, default=300, help="目标 2408 样本数")
    ap.add_argument("--max-rounds", type=int, default=6000, help="最多请求轮数")
    ap.add_argument("--delay", type=float, default=1.0, help="每轮基础间隔秒")
    ap.add_argument("--jitter", type=float, default=0.8)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--out", default=SAMPLES)
    ap.add_argument("--strip-h", type=int, default=56)
    ap.add_argument("--entry", default="https://cun.ai/sign-up")
    ap.add_argument("--channel", choices=["auto", "browser", "http"],
                    default="auto",
                    help="auto=优先 CloakBrowser，席位被占自动降级 HTTP")
    args = ap.parse_args()

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out_dir, exist_ok=True)

    page = None
    browser = None
    channel_used = "http"
    if args.channel in ("auto", "browser"):
        try:
            from cloakbrowser import launch
            print(f"[collect] 尝试 CloakBrowser Pro, headless={not args.headed}")
            browser = launch(headless=not args.headed, humanize=True,
                             locale="zh-CN", timezone="Asia/Shanghai")
            ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                                      locale="zh-CN",
                                      timezone_id="Asia/Shanghai")
            page = ctx.new_page()
            print(f"[collect] goto {args.entry}")
            page.goto(args.entry, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)
            channel_used = "browser"
        except Exception as e:
            print(f"[collect] CloakBrowser 不可用（{type(e).__name__}: "
                  f"{str(e)[:120]}），降级到 HTTP 通道")
            page = None
            browser = None
            if args.channel == "browser":
                raise

    http_ch = None
    if page is None:
        http_ch = HttpChannel(args.aid, args.entry)
        channel_used = "http"
    print(f"[collect] 采集通道 = {channel_used}")

    kept = 0
    rounds = 0
    hist = {}
    t0 = time.time()
    seen_sig = set()

    while kept < args.target and rounds < args.max_rounds:
        rounds += 1
        try:
            if page is not None:
                res = page.evaluate(FETCH_JS, {"aid": args.aid, "want": "2408"})
                sub = str(res.get("subcapclass"))
                j = res["raw"]
                dsi = (j.get("data") or {}).get("dyn_show_info") or {}
                sprite = b64_to_img(res.get("sprite")) if sub == "2408" else None
                bg = b64_to_img(res.get("bg")) if sub == "2408" else None
            else:
                sub, j, sb, bb = http_ch.fetch()
                dsi = (j.get("data") or {}).get("dyn_show_info") or {}
                sprite = (cv2.imdecode(np.frombuffer(sb, np.uint8),
                                       cv2.IMREAD_COLOR) if sb else None)
                bg = (cv2.imdecode(np.frombuffer(bb, np.uint8),
                                   cv2.IMREAD_COLOR) if bb else None)
        except Exception as e:
            print(f"  [r{rounds}] fetch 失败: {type(e).__name__}: {e}")
            if http_ch is not None:
                http_ch._rotate()
            time.sleep(3)
            continue

        hist[sub] = hist.get(sub, 0) + 1
        if sub != "2408":
            print(f"  [r{rounds}] sub={sub} 跳过 (已收 {kept}/{args.target})")
            time.sleep(args.delay + random.uniform(0, args.jitter))
            continue

        # 指纹去重（同一题重复下发）
        if bg is not None:
            sig = (bg.shape, int(bg[::8, ::8].sum()))
            if sig in seen_sig:
                print(f"  [r{rounds}] 重复样本，跳过")
                time.sleep(args.delay + random.uniform(0, args.jitter))
                continue
            seen_sig.add(sig)

        sid = save_sample(out_dir, j.get("sid") or int(time.time() * 1000),
                          j, dsi, sprite, bg, args, channel_used)
        if sid is None:
            print(f"  [r{rounds}] 样本不完整（缺提示图/底图），跳过")
            time.sleep(args.delay + random.uniform(0, args.jitter))
            continue

        kept += 1
        print(f"  [r{rounds}] ✓ {sid}  bg={bg.shape[1]}x{bg.shape[0]}  "
              f"({kept}/{args.target})")
        time.sleep(args.delay + random.uniform(0, args.jitter))

    report = {
        "aid": args.aid,
        "entry": args.entry,
        "channel": channel_used,
        "target": args.target,
        "kept_2408": kept,
        "rounds": rounds,
        "subcapclass_hist": hist,
        "hit_rate": round(kept / max(rounds, 1), 4),
        "elapsed_s": round(time.time() - t0, 1),
        "out": out_dir,
        "finished_at": datetime.now(TZ).isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "collect_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== 采集完成 ===")
    print(f"  通道      : {channel_used}")
    print(f"  2408 样本 : {kept}")
    print(f"  总轮数    : {rounds}   命中率 {report['hit_rate']:.1%}")
    print(f"  题型分布  : {hist}")
    print(f"  耗时      : {report['elapsed_s']}s")
    print(f"  目录      : {out_dir}")

    if http_ch is not None:
        http_ch.close()
    if browser is not None:
        try:
            browser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
