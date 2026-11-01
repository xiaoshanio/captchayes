# -*- coding: utf-8 -*-
"""merge_pairs: 把多份配对集合并成一个，并按 label/src 做配比控制。

为什么需要单独一步：
    匹配器训练集由三部分构成，各自的"难度"和"信息量"不同：

        主配对（prepare_pairs_det_2408）
            pos      : 检测框/标注框裁剪的正样本
            hard_neg : 同题其它框（同图同风格）
            easy_neg : 跨题随机框（太容易，loss 早饱和）
        跨题难负（hardneg_2408）
            crossq   : 与提示图标**形状最相似**的跨题图标（最难）

    直接 concat 会被 easy_neg 稀释（758 个易负样本 vs 757 正样本）。
    本脚本按 --easy-ratio 对易负样本降采样，保持难易平衡。

用法：
    python tools/merge_pairs.py --dirs data/pairsdet2408 --out data/pairsfinal
    python tools/merge_pairs.py --dirs data/pairsdet2408 --out data/pairsfinal \
        --easy-ratio 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["data/pairsdet2408"],
                    help="一个或多个配对目录（每个含 train.jsonl / val.jsonl "
                         "及可选的 hardneg_*.jsonl）")
    ap.add_argument("--out", default="data/pairsfinal")
    ap.add_argument("--easy-ratio", type=float, default=0.35,
                    help="易负样本保留比例（相对正样本数）。1.0=全留")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tr, va = [], []
    for d in args.dirs:
        for name, bucket in (("train.jsonl", tr), ("val.jsonl", va),
                             ("hardneg_train.jsonl", tr),
                             ("hardneg_val.jsonl", va)):
            p = os.path.join(d, name)
            part = load(p)
            if part:
                print(f"  [load] {p}: {len(part)}")
                bucket.extend(part)

    if not tr:
        raise SystemExit("没有任何训练配对")

    def balance(items):
        pos = [x for x in items if x["label"] == 1]
        hard = [x for x in items if x["label"] == 0 and x.get("hard")]
        easy = [x for x in items if x["label"] == 0 and not x.get("hard")]
        n_easy_keep = int(len(pos) * args.easy_ratio)
        if args.easy_ratio < 1.0 and len(easy) > n_easy_keep:
            rng.shuffle(easy)
            easy = easy[:n_easy_keep]
        out = pos + hard + easy
        rng.shuffle(out)
        return out, len(pos), len(hard), len(easy)

    tr, p1, h1, e1 = balance(tr)
    va, p2, h2, e2 = balance(va)

    out = args.out if os.path.isabs(args.out) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
    os.makedirs(out, exist_ok=True)
    for name, part in (("train.jsonl", tr), ("val.jsonl", va)):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for x in part:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    n_pos = p1
    rep = {
        "dirs": args.dirs, "out": out, "easy_ratio": args.easy_ratio,
        "train": len(tr), "val": len(va),
        "train_pos": p1, "train_hard": h1, "train_easy": e1,
        "val_pos": p2, "val_hard": h2, "val_easy": e2,
        "pos_weight_suggest": round((h1 + e1) / max(p1, 1), 3),
    }
    with open(os.path.join(out, "merge_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print("\n=== 合并完成 ===")
    print(f"  train : {len(tr)}  (pos={p1} hard={h1} easy={e1})")
    print(f"  val   : {len(va)}  (pos={p2} hard={h2} easy={e2})")
    print(f"  pos_weight : {rep['pos_weight_suggest']}")
    print(f"  输出  : {out}/")


if __name__ == "__main__":
    main()
