# -*- coding: utf-8 -*-
"""summary_2408: 汇总当前所有实测指标，一条命令看清全貌。

用法：
    python tools/summary_2408.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def n_lines(p):
    if not os.path.exists(p):
        return 0
    return sum(1 for line in open(p, encoding="utf-8") if line.strip())


def read_json(p):
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def last_csv_row(p):
    if not os.path.exists(p):
        return None
    rows = [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    if len(rows) < 2:
        return None
    head = rows[0].split(",")
    last = rows[-1].split(",")
    return dict(zip(head, last))


def main():
    print("=" * 68)
    print(" T-Sec-captcha 2408 状态汇总")
    print("=" * 68)

    # ---- 数据
    n_samples = len([d for d in glob.glob(os.path.join(REPO, "data", "samples", "*"))
                     if os.path.isdir(d)])
    print("\n[数据]")
    print(f"  样本总数        : {n_samples}")
    for name in ("labeled2408", "labeled2408v2", "labeled_det", "labeled_det_full"):
        p = os.path.join(REPO, "data", name, "labels_index.jsonl")
        if os.path.exists(p):
            print(f"  {name:16s}: {n_lines(p)} 题")
    for name in ("pairsdet2408", "pairsdet_v3", "pairsfinal"):
        p = os.path.join(REPO, "data", name, "train.jsonl")
        if os.path.exists(p):
            print(f"  {name:16s}: {n_lines(p)} 训练配对")

    # ---- 标注自洽率
    print("\n[标注自洽率]")
    for name in ("labeled2408", "labeled2408v2", "labeled_det", "labeled_det_full"):
        p = os.path.join(REPO, "data", name, "fix_report.json")
        r = read_json(p)
        if r:
            print(f"  {name:16s}: 保留 {r.get('pairs_kept')} "
                  f"改指向 {r.get('pairs_fixed')} 丢弃 {r.get('pairs_dropped')}")

    # ---- 模型
    print("\n[模型]")
    for d in ("runs/export_t2408b", "runs/export_v3", "model"):
        p = os.path.join(REPO, d)
        if not os.path.isdir(p):
            continue
        print(f"  {d}:")
        for f in sorted(os.listdir(p)):
            if f.endswith(".onnx"):
                sz = os.path.getsize(os.path.join(p, f)) / 1024 / 1024
                print(f"    {f:34s} {sz:6.2f} MB")

    # ---- 训练指标
    print("\n[训练指标]")
    for name in ("t2408", "t2408b", "v3"):
        p = os.path.join(REPO, "runs", "detect", name, "results.csv")
        r = last_csv_row(p)
        if r:
            print(f"  检测器 {name:8s}: ep={r.get('epoch'):>3s} "
                  f"mAP50={float(r.get('metrics/mAP50(B)', 0)):.4f} "
                  f"R={float(r.get('metrics/recall(B)', 0)):.4f}")
    for name in ("t2408", "t2408b", "det2408", "final2408", "v3", "v4"):
        p = os.path.join(REPO, "runs", "matcher", name, "best.pt")
        if os.path.exists(p):
            sz = os.path.getsize(p) / 1024 / 1024
            print(f"  匹配器 {name:8s}: best.pt {sz:.1f} MB")

    # ---- 标注自洽率（如果 check_label_2408 跑过）
    print("\n[标注自洽率]")
    lc = read_json(os.path.join(REPO, "out", "labelcheck2408.json"))
    if lc:
        for r in lc:
            print(f"  {r['labeled']:24s} {r['questions']:>4} 题  "
                  f"自洽 {r['consistency']:.1%}  mean={r['mean']:.3f}  "
                  f"p10={r['p10']:.3f}")
    else:
        print("  （未跑 tools/check_label_2408.py）")

    # ---- 验收
    print("\n[端到端验收]")
    r = read_json(os.path.join(REPO, "out", "eval2408", "report.json"))
    if r and "new" in r:
        n = r["new"]
        print(f"  配置      : limit={r.get('limit')} conf={r.get('conf')} "
              f"dedup={r.get('dedup')}")
        print(f"  总题数    : {n['n']}   有真值: {n['n_scored']}")
        print(f"  exact     : {n['exact']}/{n['n']} = {n['exact_rate']:.1%}")
        if n["n_scored"]:
            print(f"  exact(有效): {n['exact']}/{n['n_scored']} = "
                  f"{n['exact']/n['n_scored']:.1%}")
        print(f"  point_hit : {n['point_hit']:.1%}")
        print(f"  count_match: {n['count_match']}")
        print(f"  err_rate  : {n['err_rate']:.1%}")
        print(f"  ms_mean   : {n['ms_mean']}   ms_p95: {n['ms_p95']}")
    if r and "compare" in r:
        print("  对比 baseline:")
        for k, v in r["compare"].items():
            print(f"    {k:12s} {v['baseline']} -> {v['new']} ({v['delta']:+.4f})")

    # ---- 诊断
    print("\n[四组对照诊断]")
    d = read_json(os.path.join(REPO, "out", "diag2408", "diag_report.json"))
    if d:
        q1 = d.get("Q1_detector") or {}
        if q1:
            print(f"  检测器召回: {q1.get('recall_iou50')}  "
                  f"IoU mean={q1.get('iou_mean')}  重复框={q1.get('dup_pairs')}")
        for k in ("A", "B", "C", "D"):
            v = (d.get("Q2_Q3_ablation") or {}).get(k)
            if v:
                print(f"  {k}: exact={v['exact_rate']:.1%}  "
                      f"point={v['point_hit']:.1%}  (n={v['n']})")

    print("\n" + "=" * 68)


if __name__ == "__main__":
    main()
