# -*- coding: utf-8 -*-
"""prepare_pairs: 从合成数据集生成 Siamese 训练用的正负配对。

为什么能从合成数据免费得到配对标签：
    synth_gen.py 的 pairs/*.json 记了「第 i 个提示图标 ↔ 底图哪个框」，
    所以：正样本 = 提示图标 vs 它对应的底图裁剪；负样本 = 提示图标 vs 其它框。
    还能自动做难负样本挖掘（同一题的其它图标 = 最难的那些负样本）。

产出：
    data/pairs/train.jsonl / val.jsonl
    每行: {"a": 提示图标裁剪路径, "b": 底图裁剪路径, "label": 1/0, "hard": bool}

用法：
    python train/prepare_pairs.py --synth data/synth --out data/pairs
    python train/prepare_pairs.py --synth data/synth --out data/pairs --neg-per-pos 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import cv2
import numpy as np


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img, q=95):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        buf.tofile(p)
    return ok


def crop_cached(img, box, cache_dir, key, pad=0, jitter=0.0, rng=None):
    """裁剪并缓存到磁盘（避免每次训练重复解码整图）。

    jitter: 框抖动比例（模拟**检测框**相对真值框的几何误差）。
        为什么需要它 —— 实测（eval/crop_source_test.py）：
            匹配器喂真值框裁剪：91.2% 全对
            匹配器喂检测框裁剪：61.8% 全对   <- 掉 29.4 pt
        检测框与真值框的平均 IoU 只有 0.81（p10=0.69），
        裁出来的图会多带背景/切掉笔画，而训练时只见过"完美框"，
        于是训练-推理不一致。加抖动就是让训练分布覆盖推理分布。
    """
    fn = os.path.join(cache_dir, key + ".jpg")
    if os.path.exists(fn):
        return fn
    x1, y1, x2, y2 = [float(v) for v in box]
    if jitter and rng is not None:
        w, h = x2 - x1, y2 - y1
        # 各边独立抖动 + 整体平移（检测框误差的主要形态）
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
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    imwrite_u(fn, crop)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="data/synth")
    ap.add_argument("--out", default="data/pairs")
    ap.add_argument("--neg-per-pos", type=int, default=3, help="每个正样本配几个负样本")
    ap.add_argument("--pad", type=int, default=4, help="裁剪外扩像素")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="框抖动比例，默认 0（关闭）。")
    ap.add_argument("--jitter-target-only", action="store_true",
                    help="抖动只作用于底图目标框，提示条框保持真值。"
                         "提示条图标在纯白条上，抖动会把邻居/留白拉进裁剪，"
                         "直接破坏正样本（实测 jitter=0.04 就让 val_acc 掉到 0.25）。")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cache = os.path.join(args.out, "_crops")
    # 缓存目录带上 pad/jitter 指纹：否则改了抖动参数却复用旧缓存，等于没改
    if args.pad or args.jitter:
        tag = f"_crops_p{args.pad}_j{args.jitter}"
        if args.jitter_target_only:
            tag += "_tgt"
        cache = os.path.join(args.out, tag)
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    for split in ("train", "val"):
        img_dir = os.path.join(args.synth, "images", split)
        pr_dir = os.path.join(args.synth, "pairs", split)
        if not os.path.isdir(pr_dir):
            print(f"[skip] {split}: 没有 pairs 目录")
            continue
        rows = []
        n_pos = n_neg = n_hard = 0
        files = sorted(f for f in os.listdir(pr_dir) if f.endswith(".json"))
        for fn in files:
            base = fn[:-5]
            truth = json.load(open(os.path.join(pr_dir, fn), encoding="utf-8"))
            img_path = os.path.join(img_dir, base + ".jpg")
            img = imread_u(img_path)
            if img is None:
                continue
            pairs = truth.get("pairs", [])
            if not pairs:
                continue

            # 提示图标裁剪（从合成图的底部提示条上切）
            # jitter 只作用于 train 切分：val 必须保持"干净真值框"，
            # 否则验证指标会和推理口径混在一起，看不出真实退化。
            # 默认 --jitter-target-only 时提示侧不抖动（见参数说明）。
            jt = args.jitter if split == "train" else 0.0
            jt_prompt = 0.0 if args.jitter_target_only else jt
            prompt_crops = []
            for q in pairs:
                key = f"{split}_{base}_p{q['click_order']}"
                p = crop_cached(img, q["prompt_box"], cache, key, pad=args.pad,
                                jitter=jt_prompt, rng=rng)
                prompt_crops.append((q, p))
            prompt_crops = [(q, p) for q, p in prompt_crops if p]
            if not prompt_crops:
                continue

            # 底图所有框：正样本用 target_box，负样本用其余的框
            all_targets = []
            for q in pairs:
                all_targets.append(q["target_box"])
            # 干扰项：labels 里 class0 但不在 pairs 里的框
            lab_path = os.path.join(args.synth, "labels", split, base + ".txt")
            if os.path.exists(lab_path):
                H, W = img.shape[:2]
                for i, line in enumerate(open(lab_path, encoding="utf-8")):
                    v = line.split()
                    if len(v) != 5:
                        continue
                    c = int(float(v[0]))
                    if c != 0:
                        continue
                    cx, cy, bw, bh = [float(t) for t in v[1:]]
                    box = [(cx - bw / 2) * W, (cy - bh / 2) * H,
                           (cx + bw / 2) * W, (cy + bh / 2) * H]
                    if not any(_same(box, t) for t in all_targets):
                        all_targets.append(box)

            for q, p_prompt in prompt_crops:
                key = f"{split}_{base}_t{q['click_order']}"
                p_target = crop_cached(img, q["target_box"], cache, key, pad=args.pad,
                                       jitter=jt, rng=rng)
                if not p_target:
                    continue
                rows.append({"a": p_prompt, "b": p_target, "label": 1, "hard": False})
                n_pos += 1
                # 负样本
                neg_pool = []
                for t in all_targets:
                    if _same(t, q["target_box"]):
                        continue
                    neg_pool.append(t)
                rng.shuffle(neg_pool)
                for t in neg_pool[:args.neg_per_pos]:
                    nk = f"{split}_{base}_n{q['click_order']}_{abs(hash(tuple(round(v) for v in t))) % 10**6}"
                    pc = crop_cached(img, t, cache, nk, pad=args.pad,
                                     jitter=jt, rng=rng)
                    if not pc:
                        continue
                    # 同一题内的其它图标 = 难负样本
                    rows.append({"a": p_prompt, "b": pc, "label": 0, "hard": True})
                    n_neg += 1
                    n_hard += 1

        outp = os.path.join(args.out, f"{split}.jsonl")
        with open(outp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{split}] pos={n_pos} neg={n_neg} (hard={n_hard}) -> {outp}")

    print(f"\n[done] 缓存裁剪图 {len(os.listdir(cache))} 张 -> {cache}")


def _same(a, b, tol=2.0):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


if __name__ == "__main__":
    main()
