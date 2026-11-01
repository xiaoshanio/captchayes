# -*- coding: utf-8 -*-
"""login_10x: 对 cun.ai 连续做 N 次「退出 + 登录」，统计成功率。

测什么：
    - 端到端登录成功率（含验证码求解）
    - 各题型出现次数与各自通过率
    - 平均耗时

为什么用同一个 browser 反复登录而不是每次新开：
    更接近真人使用（保持 cookie/指纹一致），
    也让"退出"这一步真正被验证。

用法：
    python tools/login_10x.py --username X --password Y --times 10
    python tools/login_10x.py --username X --password Y --times 10 --headed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

TZ = timezone(timedelta(hours=8))
OUT = os.path.join(REPO, "out", "login10x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--times", type=int, default=10)
    ap.add_argument("--url", default="https://www.cun.ai/sign-in")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-refresh", type=int, default=5,
                    help="每道题最多换几次题型")
    ap.add_argument("--cooldown", type=float, default=3.0,
                    help="两次尝试之间的等待秒数")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    run_id = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    report = {"username": args.username, "times": args.times,
              "started": datetime.now(TZ).isoformat(timespec="seconds"),
              "runs": []}

    # ---- 模型
    from src.captcha import TextSelectCaptcha
    from src.utils import yolo_onnx, ver_onnx
    t0 = time.time()
    cap = TextSelectCaptcha()
    legacy_dir = os.path.join(REPO, "runs", "legacy")
    legacy = None
    for src, dst in (
        (os.path.join(REPO, "runs", "_model_backup", "best_v3.onnx.bak"),
         os.path.join(legacy_dir, "best_v3.onnx")),
        (os.path.join(REPO, "runs", "_model_backup", "pre_model_v7.onnx.bak"),
         os.path.join(legacy_dir, "pre_model_v7.onnx")),
    ):
        if os.path.exists(src) and not os.path.exists(dst):
            os.makedirs(legacy_dir, exist_ok=True)
            import shutil
            shutil.copyfile(src, dst)
    try:
        legacy = (
            yolo_onnx.YOLO(os.path.join(legacy_dir, "best_v3.onnx"),
                           conf_threshold=0.3),
            ver_onnx.PreONNX(os.path.join(legacy_dir, "pre_model_v7.onnx")),
        )
    except Exception as e:
        print(f"[warn] 旧域模型不可用: {type(e).__name__}: {e}", flush=True)

    print(f"[init] 模型加载 {round((time.time()-t0)*1000)} ms  "
          f"det={os.path.basename(cap.yolo.session._model_path)} "
          f"pre={os.path.basename(cap.pre.session._model_path)} "
          f"legacy={'ok' if legacy else 'no'}", flush=True)

    from cloakbrowser import launch
    from cunai_login_lib import CunaiLogin

    browser = launch(headless=not args.headed, humanize=True,
                     locale="zh-CN", timezone="Asia/Shanghai")
    try:
        ctx = browser.new_context(viewport={"width": 1440, "height": 980},
                                  locale="zh-CN",
                                  timezone_id="Asia/Shanghai")
        page = ctx.new_page()
        bot = CunaiLogin(page, ctx, cap, legacy,
                         out_dir=os.path.join(OUT, run_id),
                         max_solve_tries=args.max_refresh, conf=args.conf)

        for i in range(1, args.times + 1):
            print(f"\n{'='*64}\n  第 {i}/{args.times} 次\n{'='*64}", flush=True)
            # 每次尝试用独立输出目录，避免文件互相覆盖、便于事后看失败样本
            bot.out = os.path.join(OUT, run_id, f"try{i:02d}")
            os.makedirs(bot.out, exist_ok=True)
            # 确保从登出态开始
            try:
                bot.logout()
            except Exception as e:
                print(f"[warn] logout: {type(e).__name__}", flush=True)
            time.sleep(0.8)

            r = bot.attempt_once(args.username, args.password, i,
                                 max_refresh=args.max_refresh)
            r["events"] = bot.events[-40:]
            bot.events = []
            report["runs"].append(r)
            print(f"  -> ok={r['ok']} kinds={r['kinds']} "
                  f"solved={r['solved_any']} ms={r.get('ms')} "
                  f"note={r.get('note')}", flush=True)

            # 成功后退出，为下一轮准备
            if r["ok"]:
                try:
                    bot.logout()
                except Exception:
                    pass
            time.sleep(args.cooldown)

            # 每轮落盘，防止中断丢数据
            with open(os.path.join(OUT, f"report_{run_id}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

    finally:
        try:
            browser.close()
        except Exception:
            pass

    # ---- 统计
    runs = report["runs"]
    n = len(runs)
    ok = sum(1 for r in runs if r["ok"])
    kinds = Counter()
    kind_ok = Counter()
    for r in runs:
        ks = r.get("kinds") or []
        if ks:
            kinds[ks[0]] += 1
            if r["ok"]:
                kind_ok[ks[0]] += 1
    solved_any = sum(1 for r in runs if r.get("captcha_accepted"))
    ms = [r.get("ms", 0) for r in runs if r.get("ms")]

    report["summary"] = {
        "attempts": n,
        "success": ok,
        "success_rate": round(ok / n, 4) if n else 0.0,
        "captcha_passed_any": solved_any,
        "captcha_pass_rate": round(solved_any / n, 4) if n else 0.0,
        "first_kind_counts": dict(kinds),
        "first_kind_success": dict(kind_ok),
        "ms_mean": round(sum(ms) / len(ms), 1) if ms else None,
        "finished": datetime.now(TZ).isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUT, f"report_{run_id}.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f"  尝试次数      : {n}")
    print(f"  登录成功      : {ok}  ({report['summary']['success_rate']:.1%})")
    print(f"  验证码通过    : {solved_any}  "
          f"({report['summary']['captcha_pass_rate']:.1%})")
    print(f"  题型分布      : {dict(kinds)}")
    print(f"  各题型成功    : {dict(kind_ok)}")
    print(f"  平均耗时      : {report['summary']['ms_mean']} ms")
    print(f"  报告          : out/login10x/report_{run_id}.json")
    print("=" * 64)


if __name__ == "__main__":
    main()
