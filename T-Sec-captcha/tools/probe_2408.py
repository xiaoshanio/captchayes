# -*- coding: utf-8 -*-
"""probe_2408: 验证 2408（图标点选）的取图链路，并目视确认版式。

链路（全部来自 prehandle 回包，无需浏览器）：
    prehandle  ->  data.dyn_show_info.sprite_url   (img_index=0)  整张精灵图
               ->  data.dyn_show_info.bg_elem_cfg.img_url (img_index=1) 底图
               ->  data.dyn_show_info.ins_elem_cfg[0]  sprite_pos+size_2d = 提示图在图中的位置

产出 out/turing/v2408/<n>_*.png
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "out", "turing", "v2408")
CAP = "https://turing.captcha.qcloud.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


def imwrite_u(p, img, q=100):
    ok, buf = cv2.imencode(os.path.splitext(p)[1] or ".png", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        buf.tofile(p)
    return ok


def decode(buf):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)


def main():
    os.makedirs(OUT, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Referer": "https://cun.ai/"})
    params = {
        "aid": "195317439", "protocol": "https", "accver": "1",
        "showtype": "popup", "ua": UA, "noheader": "1", "fb": "1",
        "aged": "0", "enableAged": "1", "enableDarkMode": "0",
        "grayscale": "1", "dyeid": "1", "clientype": "1", "cap_cd": "",
        "uid": "", "lang": "zh-cn", "entry_url": "https://cun.ai/sign-up",
        "elder_captcha": "0", "login_appid": "", "wb": "2", "subsid": "1",
    }
    got = 0
    for attempt in range(30):
        if got >= 3:
            break
        r = sess.get(CAP + "/cap_union_prehandle", params=params, timeout=25)
        t = r.text.strip()
        if t.startswith("("):
            t = t[1:]
        if t.endswith(")"):
            t = t[:-1]
        j = json.loads(t)
        if str(j.get("subcapclass")) != "2408":
            print(f"  attempt {attempt}: sub={j.get('subcapclass')} 跳过")
            continue
        dsi = j["data"]["dyn_show_info"]
        got += 1
        print(f"\n=== 2408 sample #{got} ===")
        print("  instruction :", dsi.get("instruction"))
        print("  bg_elem_cfg :", {k: v for k, v in dsi["bg_elem_cfg"].items()
                                  if k != "img_url"})
        print("  ins_elem_cfg:", dsi.get("ins_elem_cfg"))
        print("  verify_trig :", dsi.get("verify_trigger_cfg"))
        print("  fg_elem_list:", dsi.get("fg_elem_list"))
        # 取图
        for tag, key in (("sprite", "sprite_url"), ("bg", "img_url")):
            u = dsi.get("sprite_url") if tag == "sprite" else dsi["bg_elem_cfg"].get("img_url")
            if not u:
                continue
            rr = sess.get(CAP + u if u.startswith("/") else u, timeout=30)
            img = decode(rr.content)
            if img is None:
                print(f"  {tag}: 解码失败 len={len(rr.content)}")
                continue
            p = os.path.join(OUT, f"s{got}_{tag}.png")
            imwrite_u(p, img)
            print(f"  {tag}: shape={img.shape} -> {os.path.basename(p)}")
            if tag == "sprite":
                ins = (dsi.get("ins_elem_cfg") or [{}])[0]
                if ins.get("sprite_pos") and ins.get("size_2d"):
                    x, y = ins["sprite_pos"]
                    w, h = ins["size_2d"]
                    crop = img[y:y + h, x:x + w]
                    pp = os.path.join(OUT, f"s{got}_prompt.png")
                    imwrite_u(pp, crop)
                    print(f"  prompt: shape={crop.shape} -> {os.path.basename(pp)}")
        with open(os.path.join(OUT, f"s{got}_prehandle.json"), "w",
                  encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
