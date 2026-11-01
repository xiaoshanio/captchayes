# -*- coding: utf-8 -*-
"""probe_http_captcha: 验证「纯 HTTP 能否完成天御 2408 全流程」（无浏览器）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
为什么要先验证
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
要把求解器接到 cun.ai 2.0（无浏览器架构），必须确认天御的
**完整链路**可以只靠 HTTP 完成：

    1) cap_union_prehandle   取题（sess / subcapclass / 图 URL）
    2) getcapbysig           取底图 + 提示图（原图）
    3) 本地模型求解           得到点击坐标
    4) cap_union_new_verify  提交答案 → 拿 ticket + randstr

第 4 步是关键：如果 verify 需要浏览器环境生成的 collect 加密串
（页面里那些乱码参数），纯 HTTP 就做不了，2.0 只能外部传 ticket。

本脚本逐步验证，把每一步的可行性打印出来。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
已知约束
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- prehandle 返回 JSONP：`_aq_128199({...})`，要剥前缀
- 2408 的 answer 格式：点击坐标序列，形如 "x1,y1,x2,y2,..."
- collect 参数是混淆 JS 生成的加密串，纯算很难，但可能可省略/复用

用法：
    python tools/probe_http_captcha.py --entry https://www.cun.ai/sign-up
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

CAP = "https://turing.captcha.qcloud.com"
OUT = os.path.join(REPO, "out", "http_captcha")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


def parse_jsonp(t):
    """解析天御响应。实测有两种形态，都要支持：

        ({"state":1,...})          ← 无函数名，仅外层括号（当前形态）
        _aq_128199({"state":1,...}) ← 有函数名（早前形态）
    """
    t = (t or "").strip()
    if not t:
        raise ValueError("empty")
    # 形态 A：有函数名
    m = re.match(r"^\s*[\w$.]+\((.*)\)\s*;?\s*$", t, re.S)
    if m:
        return json.loads(m.group(1))
    # 形态 B：只有外层括号
    if t.startswith("(") and t.endswith(")"):
        return json.loads(t[1:-1])
    # 形态 C：裸 JSON
    return json.loads(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="https://www.cun.ai/sign-up")
    ap.add_argument("--aid", default="195317439")
    ap.add_argument("--want", default="2408")
    ap.add_argument("--tries", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Referer": args.entry,
        "Accept-Language": "zh-CN,zh;q=0.9",
    })

    # ---- 步骤 1：取题（换到目标题型）
    print("=== 步骤1 prehandle 取题 ===")
    pre = None
    for i in range(args.tries):
        q = {
            "aid": args.aid, "protocol": "https", "accver": "1",
            "showtype": "popup", "ua": UA, "noheader": "1", "fb": "1",
            "aged": "0", "enableAged": "1", "enableDarkMode": "0",
            "grayscale": "1", "dyeid": "1", "clientype": "1",
            "cap_cd": "", "uid": "", "lang": "zh-cn",
            "entry_url": args.entry, "elder_captcha": "0",
            "login_appid": "", "wb": "2", "subsid": "1",
        }
        try:
            r = s.get(f"{CAP}/cap_union_prehandle", params=q, timeout=20)
            j = parse_jsonp(r.text)
        except Exception as e:
            print(f"  try{i+1} 失败 {type(e).__name__}")
            continue
        sub = str(j.get("subcapclass"))
        if sub == args.want:
            pre = j
            print(f"  try{i+1} 命中 {sub}")
            break
        print(f"  try{i+1} 拿到 {sub}（跳过）")
        time.sleep(0.3)
    if pre is None:
        raise SystemExit(f"未取到 {args.want}")

    data = pre.get("data") or {}
    dsi = data.get("dyn_show_info") or {}
    sess = data.get("sess") or ""
    print(f"  sess={sess[:24]}…  subcapclass={pre.get('subcapclass')}")
    print(f"  instruction={dsi.get('instruction')!r}")

    # ---- 步骤 2：取图
    print("\n=== 步骤2 取底图/提示图 ===")
    bg_cfg = dsi.get("bg_elem_cfg") or {}
    ins = dsi.get("ins_elem_cfg") or []
    print(f"  bg size={bg_cfg.get('size_2d')}")
    print(f"  ins={ins}")

    def fetch(u):
        full = u if u.startswith("http") else CAP + u
        r = s.get(full, timeout=25)
        return r.content if r.ok else None

    bg_b = fetch(bg_cfg.get("img_url") or "")
    bg = cv2.imdecode(np.frombuffer(bg_b, np.uint8), cv2.IMREAD_COLOR) if bg_b else None
    print(f"  底图: {'%dx%d' % (bg.shape[1], bg.shape[0]) if bg is not None else 'FAIL'}")

    prompt = None
    if ins:
        sp = fetch(dsi.get("sprite_url") or "")
        sprite = cv2.imdecode(np.frombuffer(sp, np.uint8), cv2.IMREAD_COLOR) if sp else None
        if sprite is not None:
            pos = ins[0].get("sprite_pos") or [0, 0]
            sz = ins[0].get("size_2d") or [170, 50]
            prompt = sprite[int(pos[1]):int(pos[1]) + int(sz[1]),
                            int(pos[0]):int(pos[0]) + int(sz[0])]
            print(f"  提示图: {prompt.shape[1]}x{prompt.shape[0]}")
    if bg is None or prompt is None:
        raise SystemExit("取图失败")

    # ---- 步骤 3：本地模型求解
    print("\n=== 步骤3 模型求解 ===")
    from src.captcha import TextSelectCaptcha
    import cunai_login_lib as L
    canvas = L.compose_canvas(bg, prompt)
    p = os.path.join(OUT, "canvas.jpg")
    L.imwrite_u(p, canvas)
    cap = TextSelectCaptcha()
    cap.yolo.conf_threshold = 0.25
    res = cap.run_dict(p)
    corps = res.get("corp") or []
    pts = [((c["x1"] + c["x2"]) / 2.0, (c["y1"] + c["y2"]) / 2.0)
           for c in corps]
    print(f"  解出 {len(pts)} 点: {[(round(x), round(y)) for x, y in pts]}")
    if not pts:
        raise SystemExit("求解失败")

    # ---- 步骤 4：提交（关键验证点）
    print("\n=== 步骤4 cap_union_new_verify 提交（关键）===")
    answer = ",".join(f"{int(round(x))},{int(round(y))}" for x, y in pts)
    print(f"  answer={answer}")
    body = {
        "aid": args.aid, "sess": sess, "answer": answer,
        "collect": "", "tlg": str(len(answer)), "eks": "",
        "randstr": "", "userLanguage": "zh-cn", "cap_cd": "",
    }
    try:
        r = s.post(f"{CAP}/cap_union_new_verify", data=body, timeout=25)
        print(f"  HTTP {r.status_code}  len={len(r.text)}")
        print(f"  resp: {r.text[:400]}")
        try:
            j = parse_jsonp(r.text)
            print(f"  errorCode={j.get('errorCode')!r} "
                  f"ticket={'有' if j.get('ticket') else '无'} "
                  f"randstr={'有' if j.get('randstr') else '无'}")
        except Exception as e:
            print(f"  解析失败: {type(e).__name__}")
    except Exception as e:
        print(f"  请求失败 {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
