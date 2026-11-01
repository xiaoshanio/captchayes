# -*- coding: utf-8 -*-
"""probe_turing: 直连天御验证码 SDK，多轮抽样统计题型分布（subcapclass）。

目的：搞清楚 cun.ai 的 aid=195317439 实际会下发哪些题型，
      以及「图标点选 2408」的出现条件与取数接口。

两种取样方式（都会做）：
  A) 纯 HTTP：直接请求 cap_union_prehandle（快，可批量）
  B) 页面内注入 TJCaptcha.js：走真实 SDK 链路，能拿到弹层 DOM 与 iframe

产出：
  out/turing/prehandle_stats.json     题型分布 + 每轮关键字段
  out/turing/raw/prehandle_NNN.json   每轮原文
  out/turing/sdk_probe.json           页面内 SDK 链路结果

用法：
  python tools/probe_turing.py --n 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "out", "turing")
RAW = os.path.join(OUT, "raw")
CAP = "https://turing.captcha.qcloud.com"
PREHANDLE = CAP + "/cap_union_prehandle"

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
]

SUBCLASS = {
    "2401": "slide 滑块拼图",
    "2402": "char-click 文字点选",
    "2403": "word-order 语序点选",
    "2404": "behavior 行为式(无图)",
    "2405": "spatial 空间推理",
    "2407": "voice 语音",
    "2408": "icon-click 图标点选",
    "2409": "match 消消乐",
}


def parse_jsonp(t: str):
    t = (t or "").strip()
    if t.startswith("("):
        t = t[1:]
    if t.endswith(")"):
        t = t[:-1]
    try:
        return json.loads(t)
    except Exception:
        return None


def build_params(aid, ua, entry="https://cun.ai/sign-up", extra=None):
    p = {
        "aid": str(aid), "protocol": "https", "accver": "1",
        "showtype": "popup", "ua": ua, "noheader": "1", "fb": "1",
        "aged": "0", "enableAged": "1", "enableDarkMode": "0",
        "grayscale": "1", "dyeid": "1", "clientype": "1",
        "cap_cd": "", "uid": "", "lang": "zh-cn",
        "entry_url": entry, "elder_captcha": "0",
        "login_appid": "", "wb": "2", "subsid": "1",
    }
    if extra:
        p.update(extra)
    return p


def one_prehandle(sess, aid, ua, extra=None, entry="https://cun.ai/sign-up"):
    params = build_params(aid, ua, entry, extra)
    r = sess.get(PREHANDLE, params=params, timeout=25,
                 headers={"Referer": "https://cun.ai/", "User-Agent": ua,
                          "Accept": "*/*"})
    j = parse_jsonp(r.text)
    return j, r.text, params


def sample_http(aid, n, delay=(0.6, 1.6)):
    os.makedirs(RAW, exist_ok=True)
    sess = requests.Session()
    rows = []
    for i in range(n):
        ua = random.choice(UA_POOL)
        try:
            j, raw, params = one_prehandle(sess, aid, ua)
        except Exception as e:
            rows.append({"i": i, "err": f"{type(e).__name__}: {e}"})
            continue
        if j is None:
            rows.append({"i": i, "err": "jsonp parse fail", "len": len(raw)})
            continue
        dsi = j.get("dyn_show_info") or {}
        bgcfg = dsi.get("bg_elem_cfg") or {}
        row = {
            "i": i,
            "subcapclass": j.get("subcapclass"),
            "kind": SUBCLASS.get(str(j.get("subcapclass")), "?"),
            "capclass": j.get("capclass"),
            "state": j.get("state"),
            "instruction": dsi.get("instruction"),
            "bg_size": bgcfg.get("size_2d"),
            "sprite_pos": bgcfg.get("sprite_pos"),
            "fg_sizes": [e.get("size_2d") for e in (dsi.get("fg_elem_list") or [])],
            "has_pow": bool((((j.get("data") or {}).get("comm_captcha_cfg") or {})
                             .get("pow_cfg"))),
            "sid": j.get("sid"),
            "uip": j.get("uip"),
            "sess_len": len(j.get("sess") or ""),
            "has_sprite": bool(dsi.get("sprite_url")),
            "has_bgimg": bool(bgcfg.get("img_url")),
            "entry_url": params["entry_url"],
        }
        rows.append(row)
        fn = os.path.join(RAW, f"prehandle_{i:03d}.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        print(f"  [{i+1}/{n}] sub={row['subcapclass']}({row['kind']}) "
              f"bg={row['bg_size']} instr={row['instruction']!r}")
        time.sleep(random.uniform(*delay))
    return rows


SDK_JS = """
async (cfg) => {
  // 注入天御 SDK（与 cun.ai 线上同一份脚本）
  const load = () => new Promise((res, rej) => {
    if (window.TencentCaptcha) return res(true);
    const s = document.createElement('script');
    s.src = 'https://turing.captcha.qcloud.com/TJCaptcha.js';
    s.onload = () => res(true);
    s.onerror = () => rej(new Error('sdk load fail'));
    document.head.appendChild(s);
  });
  await load();
  const out = {sdk: typeof window.TencentCaptcha, rounds: []};
  const grab = () => {
    const ifr = [...document.querySelectorAll('iframe')]
      .map(f => ({src: f.src, w: f.getBoundingClientRect().width,
                  h: f.getBoundingClientRect().height}))
      .filter(o => /turing|captcha/.test(o.src));
    return ifr;
  };
  for (let i = 0; i < cfg.rounds; i++) {
    const r = await new Promise((resolve) => {
      const timer = setTimeout(() => resolve({timeout: true}), 12000);
      try {
        new window.TencentCaptcha(cfg.aid, (res) => {
          clearTimeout(timer);
          resolve({cb: res});
        }, {showtype: 'popup', userLanguage: 'zh-cn'}).show();
      } catch (e) {
        clearTimeout(timer);
        resolve({err: String(e)});
      }
    });
    await new Promise(r => setTimeout(r, 1500));
    out.rounds.push({i, result: r, iframes: grab(),
                     tcaptcha: !!document.querySelector('.tcaptcha-transform, #tcaptcha_iframe, .tcaptcha-popup')});
    // 关掉弹层
    const btn = document.querySelector('.tcaptcha-close, .tcaptcha-btn-close');
    if (btn) btn.click();
    await new Promise(r => setTimeout(r, 800));
  }
  return out;
}
"""


def sample_sdk(aid, rounds=3, headed=False):
    from cloakbrowser import launch
    browser = launch(headless=not headed, humanize=True, locale="zh-CN",
                     timezone="Asia/Shanghai")
    ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                              locale="zh-CN", timezone_id="Asia/Shanghai")
    page = ctx.new_page()
    page.goto("https://cun.ai/sign-up", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(2500)
    try:
        out = page.evaluate(SDK_JS, {"aid": str(aid), "rounds": rounds})
    except Exception as e:
        out = {"err": f"{type(e).__name__}: {e}"}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sdk_probe.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[sdk] ->", json.dumps(out, ensure_ascii=False)[:1200])
    browser.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aid", default="195317439")
    ap.add_argument("--n", type=int, default=30, help="HTTP 抽样轮数")
    ap.add_argument("--sdk-rounds", type=int, default=3)
    ap.add_argument("--no-sdk", action="store_true")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print(f"[probe] aid={args.aid}  HTTP 抽样 {args.n} 轮")
    rows = sample_http(args.aid, args.n)

    hist = Counter(str(r.get("subcapclass")) for r in rows if r.get("subcapclass"))
    bg_sizes = Counter(str(r.get("bg_size")) for r in rows if r.get("bg_size"))
    instrs = Counter(str(r.get("instruction")) for r in rows if r.get("instruction"))
    stats = {
        "aid": args.aid,
        "n": len(rows),
        "subcapclass_hist": dict(hist),
        "subcapclass_kind": {k: SUBCLASS.get(k, "?") for k in hist},
        "bg_size_hist": dict(bg_sizes),
        "instruction_hist": dict(instrs),
        "click_captcha_ratio": round(
            sum(v for k, v in hist.items() if k in ("2402", "2403", "2408", "2409"))
            / max(sum(hist.values()), 1), 4),
        "rows": rows,
    }
    with open(os.path.join(OUT, "prehandle_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== HTTP 抽样结论 ===")
    print(f"  样本数      : {len(rows)}")
    print(f"  subcapclass : {dict(hist)}")
    print(f"  题型        : {stats['subcapclass_kind']}")
    print(f"  底图尺寸    : {dict(bg_sizes)}")
    print(f"  提示语      : {dict(instrs)}")
    print(f"  点选题占比  : {stats['click_captcha_ratio']:.1%}")

    if not args.no_sdk:
        print("\n[probe] 页面内注入 TJCaptcha.js 走真实 SDK 链路")
        sample_sdk(args.aid, args.sdk_rounds, args.headed)


if __name__ == "__main__":
    main()
