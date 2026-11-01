# -*- coding: utf-8 -*-
"""prepare_pairs_2408: 从 autolabel_2408 的产物生成 Siamese 训练配对。

与 train/prepare_pairs.py 的区别：
    那个脚本读合成集 data/synth/pairs/*.json（合成器的配对真值）。
    本脚本读 data/labeled2408/labels_index.jsonl（真实 2408 样本的标注结果），
    直接从**原图**裁剪，配对真值来自 autolabel_2408 的 pairs 字段。

配对策略（关键设计）：
    正样本 : 第 i 个提示图标 ↔ 它在底图对应的框（来自标注 pairs）
    难负样本: 同一题内的**其它**底图框（同图、同风格、同配色 → 最难）
    易负样本: 跨题随机抽取的底图框（扩大决策边界，防过拟合到同题）

    正:难负:易负 = 1 : 2 : 1（可调）
    正样本占比低 → 必须开 --auto-pos-weight，否则 BCE 会塌缩成全判负
    （这是 docs/TRAINING_AND_DATA_REFRESH.md 5.x 记录的既有坑，此处沿用同样的对策）。

几何抖动（关键）：
    训练时裁剪用的是**标注框**，推理时用的是**检测框**，
    两者有系统偏差（实测 IoU 0.81，p10=0.69）。
    所以裁剪时对框做 jitter，让训练分布覆盖推理分布 —— 与
    train/prepare_pairs.py::crop_cached 的 jitter 设计一致。

产出：
    <out>/train.jsonl, <out>/val.jsonl
    <out>/crops/*.jpg          裁剪缓存（避免训练时重复解码整图）
    <out>/pairs_report.json    统计

用法：
    python tools/prepare_pairs_2408.py --labeled data/labeled2408 --out data/pairs2408
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img, q=95):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        buf.tofile(p)
    return ok


def crop(img, box, cache_dir, key, jitter=0.0, rng=None, pad=0,
         norm_short=None):
    """裁剪并缓存。jitter 模拟检测框相对真值框的几何误差。

    norm_short：把裁剪结果的最短边缩放到该值（保持长宽比）。
        为什么需要 —— 实测提示图标裁剪 a 侧 30x39、底图裁剪 b 侧 57x63，
        两者差 1.9 倍。直接 letterbox(112) 后，**尺度差异会盖过形状差异**，
        模型学到的是"大小不同"，而不是"形状是否相同"。
        （验证：val_acc 93.9% 但端到端单点命中只有 83.8%，
          说明它在同尺寸的标注框上表现好，在尺寸不一的检测框上就退化。）

        两侧统一归一到相同最短边后，模型只能靠形状判别。
    """
    fn = os.path.join(cache_dir, key + ".jpg")
    if os.path.exists(fn):
        return fn
    x1, y1, x2, y2 = [float(v) for v in box]
    if jitter and rng is not None:
        w, h = x2 - x1, y2 - y1
        dx = rng.uniform(-jitter, jitter) * w
        dy = rng.uniform(-jitter, jitter) * h
        sx = rng.uniform(-jitter, jitter) * w
        sy = rng.uniform(-jitter, jitter) * h
        x1, x2 = x1 + dx + sx, x2 + dx - sx
        y1, y2 = y1 + dy + sy, y2 + dy - sy
    x1 = max(0, int(round(x1)) - pad)
    y1 = max(0, int(round(y1)) - pad)
    x2 = min(img.shape[1], int(round(x2)) + pad)
    y2 = min(img.shape[0], int(round(y2)) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    c = img[y1:y2, x1:x2]
    if c.size == 0:
        return None
    if norm_short:
        h, w = c.shape[:2]
        s = norm_short / float(max(min(h, w), 1))
        if abs(s - 1.0) > 0.02:
            c = cv2.resize(c, (max(1, int(round(w * s))),
                               max(1, int(round(h * s)))),
                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    imwrite_u(fn, c)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/labeled2408")
    ap.add_argument("--samples", default="data/samples",
                    help="原始样本目录（取 bg.jpg 与 prompt 图做裁剪）")
    ap.add_argument("--out", default="data/pairs2408")
    ap.add_argument("--hard-neg", type=int, default=2, help="每题难负样本数/正样本")
    ap.add_argument("--easy-neg", type=int, default=1, help="每题易负样本数/正样本")
    ap.add_argument("--jitter", type=float, default=0.12,
                    help="裁剪框抖动比例（模拟检测框误差）")
    ap.add_argument("--norm-short", type=int, default=64,
                    help="把两侧裁剪的最短边统一到该像素，0=关闭。"
                         "实测 a 侧 30x39 / b 侧 57x63 差 1.9 倍，"
                         "不归一的话模型学的是尺寸差异而非形状")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-score", type=float, default=0.45,
                    help="标注配对分低于此值的正样本不采用（避免脏标签）")
    args = ap.parse_args()

    root = args.labeled if os.path.isabs(args.labeled) else os.path.join(REPO, args.labeled)
    sroot = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    crops_dir = os.path.join(out, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    index = os.path.join(root, "labels_index.jsonl")
    if not os.path.exists(index):
        raise SystemExit(f"缺少标注索引: {index}（先跑 tools/autolabel_2408.py）")

    recs = [json.loads(l) for l in open(index, encoding="utf-8") if l.strip()]
    print(f"[scan] {len(recs)} 题标注")

    rng = random.Random(args.seed)
    rows = []
    skipped = 0
    for rec in recs:
        sid = rec["sid"]
        sdir = os.path.join(sroot, sid)
        meta_p = os.path.join(sdir, "meta.json")
        if not os.path.exists(meta_p):
            skipped += 1
            continue
        meta = json.load(open(meta_p, encoding="utf-8"))
        bg = imread_u(os.path.join(sdir, "bg.jpg"))
        prompts = meta.get("prompts") or []
        if bg is None or not prompts:
            skipped += 1
            continue

        # 提示图标：从标注产物里重新切（与标注时同一切法，保证一致性）
        from autolabel_2408 import split_prompt_icons
        pr = imread_u(os.path.join(sdir, prompts[0]["file"]))
        if pr is None:
            skipped += 1
            continue
        icons = split_prompt_icons(pr)
        pairs = rec.get("pairs") or []
        bg_boxes = rec.get("bg_boxes") or []
        if not icons or not pairs or not bg_boxes:
            skipped += 1
            continue

        # 提示图标裁剪（a 侧）—— 与 b 侧统一最短边，让模型只看形状
        a_paths = {}
        for i, it in enumerate(icons):
            fn = crop(it["img"], (0, 0, it["img"].shape[1], it["img"].shape[0]),
                      crops_dir, f"{sid}_p{i}", jitter=0.0,
                      norm_short=args.norm_short or None)
            a_paths[i] = fn

        # 底图框裁剪（b 侧）
        b_paths = {}
        for j, box in enumerate(bg_boxes):
            fn = crop(bg, box, crops_dir, f"{sid}_b{j}", jitter=args.jitter, rng=rng,
                      norm_short=args.norm_short or None)
            b_paths[j] = fn

        matched_bg = {}
        for p in pairs:
            i, j, sc = p["prompt"], p["bg"], p["score"]
            if sc < args.min_score:
                continue
            matched_bg[i] = j
            if a_paths.get(i) and b_paths.get(j):
                rows.append({"a": a_paths[i], "b": b_paths[j], "label": 1,
                             "hard": False, "sid": sid, "score": sc})

        # 难负样本：同题其它框
        for i, j in matched_bg.items():
            others = [k for k in range(len(bg_boxes)) if k != j]
            rng.shuffle(others)
            for k in others[:max(1, args.hard_neg)]:
                if a_paths.get(i) and b_paths.get(k):
                    rows.append({"a": a_paths[i], "b": b_paths[k], "label": 0,
                                 "hard": True, "sid": sid, "score": 0.0})

        # 易负样本：跨题随机框（本题之后随机取一个题）
        for i, j in matched_bg.items():
            for _ in range(max(0, args.easy_neg)):
                other = rng.choice(recs)
                ob = other.get("bg_boxes") or []
                if not ob:
                    continue
                k = rng.randrange(len(ob))
                obg = imread_u(os.path.join(sroot, other["sid"], "bg.jpg"))
                if obg is None:
                    continue
                fn = crop(obg, ob[k], crops_dir,
                          f"{other['sid']}_b{k}_e", jitter=args.jitter, rng=rng,
                          norm_short=args.norm_short or None)
                if a_paths.get(i) and fn:
                    rows.append({"a": a_paths[i], "b": fn, "label": 0,
                                 "hard": False, "sid": sid, "score": 0.0})

    if not rows:
        raise SystemExit("没有生成任何配对，检查 --labeled / --samples 是否一致")

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    val, train = rows[:n_val], rows[n_val:]

    for name, part in (("train", train), ("val", val)):
        with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos = sum(1 for r in train if r["label"] == 1)
    n_hard = sum(1 for r in train if r["label"] == 0 and r["hard"])
    n_easy = sum(1 for r in train if r["label"] == 0 and not r["hard"])
    rep = {
        "labeled": root, "samples": sroot, "out": out,
        "n_items": len(rows), "train": len(train), "val": len(val),
        "train_pos": n_pos, "train_hard_neg": n_hard, "train_easy_neg": n_easy,
        "skipped": skipped,
        "pos_weight_suggest": round((n_hard + n_easy) / max(n_pos, 1), 3),
        "jitter": args.jitter, "min_score": args.min_score,
    }
    with open(os.path.join(out, "pairs_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print("\n=== 配对生成完成 ===")
    print(f"  总配对   : {len(rows)}  (train={len(train)}, val={len(val)})")
    print(f"  正样本   : {n_pos}")
    print(f"  难负样本 : {n_hard}")
    print(f"  易负样本 : {n_easy}")
    print(f"  跳过题   : {skipped}")
    print(f"  pos_weight 建议: {rep['pos_weight_suggest']}")
    print(f"  输出     : {out}/")


if __name__ == "__main__":
    main()
