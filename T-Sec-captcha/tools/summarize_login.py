# -*- coding: utf-8 -*-
"""summarize_login: 汇总所有 login_10x 报告，算成功概率与题型分解。

用法：
    python tools/summarize_login.py
    python tools/summarize_login.py --out docs/LOGIN_SUCCESS_RATE.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def wilson(k, n, z=1.96):
    """Wilson 置信区间 —— 小样本比 正态近似 稳。"""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - m) / d, (c + m) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(REPO, "out", "login10x",
                                          "report_*.json")))
    if not files:
        raise SystemExit("没有报告")

    batches = []
    all_runs = []
    for p in files:
        d = json.load(open(p, encoding="utf-8"))
        runs = d.get("runs") or []
        if not runs:
            continue
        # 只统计完整批次（>=5 次）里的 10 次批次
        if len(runs) < 5:
            continue
        ok = sum(1 for r in runs if r.get("ok"))
        batches.append({
            "file": os.path.basename(p),
            "n": len(runs), "ok": ok,
            "rate": round(ok / len(runs), 3),
            "started": d.get("started"),
        })
        for r in runs:
            r["_batch"] = os.path.basename(p)
            all_runs.append(r)

    n = len(all_runs)
    ok = sum(1 for r in all_runs if r.get("ok"))
    p, lo, hi = wilson(ok, n)

    # 按「首次出现的题型」分解
    first = Counter()
    first_ok = Counter()
    for r in all_runs:
        ks = r.get("kinds") or []
        k0 = ks[0] if ks else "none"
        first[k0] += 1
        if r.get("ok"):
            first_ok[k0] += 1

    # 所有题型出现次数（含重试）
    allk = Counter()
    for r in all_runs:
        for k in (r.get("kinds") or []):
            allk[k] += 1

    ms = [r.get("ms", 0) for r in all_runs if r.get("ms")]

    print("=" * 72)
    print(" cun.ai 登录成功率统计")
    print("=" * 72)
    print(f"\n批次:")
    for b in batches:
        print(f"  {b['file']}  {b['ok']}/{b['n']} = {b['rate']:.0%}"
              f"   ({b.get('started')})")
    print(f"\n合计: {ok}/{n} = {p:.1%}   "
          f"95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"平均单次耗时: {round(sum(ms)/len(ms)) if ms else '-'} ms")

    print(f"\n按首次题型分解:")
    for k, v in first.most_common():
        o = first_ok.get(k, 0)
        print(f"  {k:10s} {o}/{v} = {o/v:.0%}")

    # 逐批次 × 题型
    print(f"\n逐批次 × 首题题型:")
    hdr = f"  {'batch':28s} {'ok/n':>8s}  " + "  ".join(
        f"{k:>10s}" for k in ("2401", "2404", "2408"))
    print(hdr)
    for b in batches:
        f = b["file"].replace("report_", "").replace(".json", "")
        cells = []
        for k in ("2401", "2404", "2408"):
            cnt = 0
            oks = 0
            for r in all_runs:
                if r.get("_batch") != b["file"]:
                    continue
                ks = r.get("kinds") or []
                if ks and ks[0] == k:
                    cnt += 1
                    if r.get("ok"):
                        oks += 1
            cells.append(f"{oks}/{cnt}" if cnt else "-")
        print(f"  {f:28s} {b['ok']:>3d}/{b['n']:<4d}  " +
              "  ".join(f"{c:>10s}" for c in cells))

    # 修复后（最后 2 个完整批次）
    if len(batches) >= 2:
        last = batches[-2:]
        ln = sum(b["n"] for b in last)
        lok = sum(b["ok"] for b in last)
        lp, llo, lhi = wilson(lok, ln)
        print(f"\n最近两组（含交互修复）: {lok}/{ln} = {lp:.1%} "
              f"95% CI [{llo:.1%}, {lhi:.1%}]")

    print(f"\n验证码出现次数（含重试）: {dict(allk)}")

    # 失败原因分布
    notes = Counter()
    for r in all_runs:
        if not r.get("ok"):
            notes[r.get("note") or "popup_still_open"] += 1
    print(f"\n失败原因: {dict(notes)}")

    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        lines = [
            "# cun.ai 登录成功率实测",
            "",
            f"账号：`zanjianwan25946`　目标站点：`https://www.cun.ai/sign-in`",
            "",
            "## 总结论",
            "",
            f"**总成功率 {ok}/{n} = {p:.1%}**（95% CI {lo:.1%}~{hi:.1%}）",
            "",
            "| 批次 | 成功/次数 | 成功率 |",
            "|---|---|---|",
        ]
        for b in batches:
            lines.append(f"| {b['file']} | {b['ok']}/{b['n']} | {b['rate']:.0%} |")
        lines += [
            f"| **合计** | **{ok}/{n}** | **{p:.1%}** |",
            "",
            "## 按题型",
            "",
            "| 首次题型 | 通过/次数 | 通过率 |",
            "|---|---|---|",
        ]
        for k, v in first.most_common():
            o = first_ok.get(k, 0)
            lines.append(f"| {k} | {o}/{v} | {o/v:.0%} |")
        lines += [
            "",
            f"验证码出现次数（含重试）：`{dict(allk)}`",
            "",
            f"失败原因分布：`{dict(notes)}`",
            "",
            f"平均单次耗时：{round(sum(ms)/len(ms)) if ms else '-'} ms",
            "",
        ]
        open(out, "w", encoding="utf-8").write("\n".join(lines))
        print(f"\n文档 -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
