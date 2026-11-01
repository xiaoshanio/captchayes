# -*- coding: utf-8 -*-
"""matcher_score_diag: 检查匹配器的打分分布，判断是「训练坏了」还是「阈值/尺度问题」。

背景：
    给匹配器加框抖动增强（jitter=0.12）后，val_acc 从 0.964 掉到 0.728，
    在真值框上的全对率从 91.2% 崩到 11.8%。这不像"增强带来的合理折衷"，
    更像是训练出了问题。

本脚本直接看两个匹配器在**干净真值框**上的相似度矩阵：
    对角（正样本对）应该高、非对角（负样本对）应该低。
    如果对角也低，说明模型根本没学会"同一图标"这个概念（训练坏了）；
    如果对角高但分布重叠，说明是判别力不足（可能需要更多数据/更强 backbone）。

用法：
    python eval/matcher_score_diag.py --data data/driftB --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/driftB")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--models", nargs="*", default=None,
                    help="形如 name=path，默认比较 matB 与 matB_j")
    args = ap.parse_args()

    from src.utils import ver_onnx
    models = args.models or [
        "matB_clean=runs/drift_rec/export/matB.onnx",
        "matB_jitter=runs/drift_rec/export/matB_j.onnx",
        "matA_old=runs/export_nms/pre_model_v7_nms.onnx",
    ]

    d = os.path.join(REPO, args.data)
    pdir = os.path.join(d, "pairs", "val")
    bases = [f[:-5] for f in sorted(os.listdir(pdir))][:args.limit]

    # 预先加载所有 crop（两个模型共用，保证公平对比）
    cases = []
    for b in bases:
        ip = os.path.join(d, "images", "val", b + ".jpg")
        img = cv2.imdecode(np.fromfile(ip, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        truth = json.load(open(os.path.join(pdir, b + ".json"), encoding="utf-8"))
        prs = sorted(truth["pairs"], key=lambda q: q["click_order"])
        if len(prs) < 2:
            continue
        pc, tc = [], []
        ok = True
        for q in prs:
            p1 = img[int(q["prompt_box"][1]):int(q["prompt_box"][3]),
                     int(q["prompt_box"][0]):int(q["prompt_box"][2])]
            p2 = img[int(q["target_box"][1]):int(q["target_box"][3]),
                     int(q["target_box"][0]):int(q["target_box"][2])]
            if p1.size == 0 or p2.size == 0:
                ok = False
                break
            pc.append(p1)
            tc.append(p2)
        if ok:
            cases.append((pc, tc))
    print(f"[cases] {len(cases)} 题（{args.data}）")

    print("=" * 88)
    print("匹配器打分分布（干净真值框，模拟『理想检测』）")
    print("=" * 88)
    rows = []
    for spec in models:
        name, path = spec.split("=", 1)
        p = os.path.join(REPO, path)
        if not os.path.exists(p):
            print(f"  [skip] {name}: 缺 {path}")
            continue
        mat = ver_onnx.PreONNX(p)
        diag, off = [], []
        exact = 0
        for pc, tc in cases:
            m = np.array(mat.reason_all_batch(pc, tc), dtype=float)
            n = min(m.shape)
            diag += [m[i, i] for i in range(n)]
            off += [m[i, j] for i in range(n) for j in range(n) if i != j]
            # 贪心全对？
            mm = m.copy()
            res = []
            for _ in range(min(mm.shape)):
                i, j = np.unravel_index(int(np.argmax(mm)), mm.shape)
                res.append((int(i), int(j)))
                mm[i, :] = -1e9
                mm[:, j] = -1e9
            res.sort(key=lambda t: t[0])
            if all(j == i for i, j in res):
                exact += 1
        d_arr, o_arr = np.array(diag), np.array(off)
        row = {
            "name": name, "n": len(cases), "exact": exact,
            "exact_rate": round(exact / max(len(cases), 1), 4),
            "diag_mean": round(float(d_arr.mean()), 4),
            "diag_p10": round(float(np.percentile(d_arr, 10)), 4),
            "off_mean": round(float(o_arr.mean()), 4),
            "off_p90": round(float(np.percentile(o_arr, 90)), 4),
            # 分离度：对角均值 - 非对角均值（越大越好）
            "separation": round(float(d_arr.mean() - o_arr.mean()), 4),
        }
        rows.append(row)
        print(f"\n  {name}")
        print(f"    全对率      : {exact}/{len(cases)} ({row['exact_rate']*100:.1f}%)")
        print(f"    对角(正样本): mean={row['diag_mean']:.3f}  p10={row['diag_p10']:.3f}")
        print(f"    非对角(负)  : mean={row['off_mean']:.3f}  p90={row['off_p90']:.3f}")
        print(f"    分离度      : {row['separation']:.3f}")

    print("\n" + "=" * 88)
    print("判读")
    print("=" * 88)
    for r in rows:
        if r["diag_mean"] < 0.6:
            print(f"  ! {r['name']}: 对角均值仅 {r['diag_mean']:.3f} —— 模型没学会'同一图标'，训练有问题")
        elif r["separation"] < 0.15:
            print(f"  · {r['name']}: 分离度偏低 ({r['separation']:.3f}) —— 判别力不足")
        else:
            print(f"  ✓ {r['name']}: 分离度 {r['separation']:.3f}，正常")

    out = os.path.join(REPO, "out", "matcher_score.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n报告 -> {out}")


if __name__ == "__main__":
    main()
