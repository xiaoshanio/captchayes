# -*- coding: utf-8 -*-
"""tap_verify: 抓 `cap_union_new_verify` 的应答码，并把它变成决策依据。

════════════════════════════════════════════════════════════════
「应答码」到底是什么
════════════════════════════════════════════════════════════════
点完确定后，前端会向天御发一个校验请求：

    POST https://turing.captcha.qcloud.com/cap_union_new_verify

响应是 JSON（或 JSONP），关键字段：

    ret        : 0 = 通过；非 0 = 未通过
    ticket     : 通过时下发的票据（后续业务请求要用）
    randstr    : 通过时的随机串
    errorCode  : 未通过时的错误码（源码里就是 errorCode 分支）
    errorMessage: 对应文案
    sess       : 本次会话 id

所以「应答码」= 这次校验请求返回的 ret / errorCode。
它能直接回答我们最关心的问题：
    没通过，是因为**答案错**、**提交没生效**、还是**被风控**。

════════════════════════════════════════════════════════════════
本脚本做的事
════════════════════════════════════════════════════════════════
    1) 监听 cap_union_new_verify 的请求体与响应体
    2) 把「我们点了什么」与「站点收到什么、返回什么」对齐
    3) 落到 out/verify_tap/verify_log.jsonl，供统计分析

有了这份数据，后续所有优化都能变成「按应答分流」而不是盲调：
    - 有请求 + ret!=0 + errorCode 常见值 → 答案/顺序问题
    - 有请求 + ret!=0 + 风控类码         → 行为特征问题
    - 无请求                              → 提交环节没触发（确定按钮无效）

用法：
    python tools/tap_verify.py --username X --password Y --times 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

TZ = timezone(timedelta(hours=8))
OUT = os.path.join(REPO, "out", "verify_tap")


def parse_body(txt):
    """天御响应可能是纯 JSON、JSONP（fn({...})）或带前缀。"""
    t = (txt or "").strip()
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        pass
    import re
    m = re.match(r"^\s*[\w$.]+\((.*)\)\s*;?\s*$", t, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--times", type=int, default=5)
    ap.add_argument("--max-refresh", type=int, default=3)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, "verify_log.jsonl")

    from src.captcha import TextSelectCaptcha
    from cloakbrowser import launch
    import cunai_login_lib as L

    cap = TextSelectCaptcha()
    cap.yolo.conf_threshold = 0.25

    # 旧域模型（2404 中文文字点选）——不加载会让 2404 直接解不了
    legacy = None
    try:
        from src.utils import yolo_onnx, ver_onnx
        import shutil
        ldir = os.path.join(REPO, "runs", "legacy")
        os.makedirs(ldir, exist_ok=True)
        for src, dst in (
            (os.path.join(REPO, "runs", "_model_backup", "best_v3.onnx.bak"),
             os.path.join(ldir, "best_v3.onnx")),
            (os.path.join(REPO, "runs", "_model_backup", "pre_model_v7.onnx.bak"),
             os.path.join(ldir, "pre_model_v7.onnx")),
        ):
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
        legacy = (yolo_onnx.YOLO(os.path.join(ldir, "best_v3.onnx"),
                                 conf_threshold=0.3),
                  ver_onnx.PreONNX(os.path.join(ldir, "pre_model_v7.onnx")))
        print("[init] legacy 模型已加载（2404 可用）", flush=True)
    except Exception as e:
        print(f"[warn] legacy 未加载: {type(e).__name__}: {e}", flush=True)

    browser = launch(headless=not args.headed, humanize=True,
                     locale="zh-CN", timezone="Asia/Shanghai")
    records = []
    try:
        ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                                  locale="zh-CN", timezone_id="Asia/Shanghai")
        page = ctx.new_page()
        bot = L.CunaiLogin(page, ctx, cap, legacy, out_dir=OUT)

        # 抓校验请求/响应（这是「应答码」的来源）
        taps = []

        def on_resp(r):
            if "cap_union_new_verify" not in r.url:
                return
            rec = {"ts": round(time.time(), 3), "url": r.url[:120]}
            try:
                rec["status"] = r.status
            except Exception:
                pass
            try:
                req = r.request
                rec["method"] = req.method
                pd = req.post_data
                if pd:
                    rec["post_head"] = pd[:600]
            except Exception:
                pass
            try:
                body = r.text()
                rec["body_head"] = (body or "")[:600]
                j = parse_body(body)
                if isinstance(j, dict):
                    for k in ("ret", "errorCode", "errorMessage", "errCode",
                              "ticket", "randstr", "sess"):
                        if k in j:
                            v = j[k]
                            if k in ("ticket", "sess") and isinstance(v, str):
                                v = v[:16] + "…"
                            rec[k] = v
            except Exception as e:
                rec["body_err"] = type(e).__name__
            taps.append(rec)
            if "ret" in rec or "errorCode" in rec:
                print(f"    [verify] ret={rec.get('ret')} "
                      f"errorCode={rec.get('errorCode')} "
                      f"msg={rec.get('errorMessage')}", flush=True)

        page.on("response", on_resp)

        for i in range(1, args.times + 1):
            print(f"\n{'='*60}\n  第 {i}/{args.times} 次\n{'='*60}", flush=True)
            bot.out = os.path.join(OUT, f"try{i:02d}")
            os.makedirs(bot.out, exist_ok=True)
            try:
                bot.logout()
            except Exception:
                pass
            time.sleep(0.8)

            taps.clear()
            r = bot.attempt_once(args.username, args.password, i,
                                 max_refresh=args.max_refresh)
            time.sleep(0.6)

            rec = {"attempt": i, "ok": bool(r.get("ok")),
                   "kinds": r.get("kinds") or [], "ms": r.get("ms"),
                   "note": r.get("note") or "",
                   "n_verify": len(taps),
                   "verify": [dict(t) for t in taps],
                   "events": bot.events[-25:]}
            bot.events = []
            records.append(rec)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            rets = [t.get("ret") for t in taps]
            ecs = [t.get("errorCode") for t in taps]
            print(f"  -> ok={rec['ok']} kinds={rec['kinds']} "
                  f"verify={rec['n_verify']} ret={rets} errorCode={ecs}",
                  flush=True)

            if r.get("ok"):
                try:
                    bot.logout()
                except Exception:
                    pass
    finally:
        try:
            browser.close()
        except Exception:
            pass

    # ---- 汇总
    print("\n" + "=" * 66)
    print(" 应答码汇总")
    print("=" * 66)
    okn = sum(1 for r in records if r["ok"])
    nov = sum(1 for r in records if r["n_verify"] == 0)
    print(f"  尝试 {len(records)}  成功 {okn}  "
          f"成功率 {okn/max(len(records),1):.0%}")
    print(f"  没有发出校验请求的次数: {nov}/{len(records)}")
    from collections import Counter
    rets = Counter()
    ecs = Counter()
    for r in records:
        for t in r["verify"]:
            rets[t.get("ret")] += 1
            ecs[t.get("errorCode")] += 1
    print(f"  ret 分布      : {dict(rets)}")
    print(f"  errorCode 分布: {dict(ecs)}")
    print(f"\n  明细 -> {os.path.relpath(log_path, REPO)}")


if __name__ == "__main__":
    main()
