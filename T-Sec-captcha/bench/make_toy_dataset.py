#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bench/make_toy_dataset.py

生成 toy YOLO 检测数据集，用于本机 CPU 训练吞吐基准（bench/bench_train.py）。

图像规格刻意对齐真实验证码契约：
  - 344x384 / 672x480 / 672x536（截图型）
并保证 h、w 都是 32 的倍数，这样 ultralytics 的 stride-32 padding 不会额外插黑边，
测出来的单步耗时只反映「真实计算量 + letterbox 缩放量」。

类别：3 类，与 data/selfgen 的 {label, title, char} 对齐。

用法:
  python bench/make_toy_dataset.py --n 400 --nval 32 --out bench/toy_dataset
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["label", "title", "char"]
SIZES = [(344, 384), (672, 480), (672, 536)]  # (w, h)，h/w 均为 32 的倍数


def rand_color(rng: random.Random) -> tuple[int, int, int]:
    return (rng.randint(20, 235), rng.randint(20, 235), rng.randint(20, 235))


def make_one(idx: int, rng: random.Random, ncls: int, noise_level: int):
    w, h = rng.choice(SIZES)
    mode = rng.random()
    if mode < 0.45:
        img = np.random.randint(0, noise_level, size=(h, w, 3), dtype=np.uint8)
    elif mode < 0.7:
        img = np.full((h, w, 3), rand_color(rng), dtype=np.uint8)
    else:
        img = np.full((h, w, 3), rand_color(rng), dtype=np.uint8)
        # 稀疏噪声块，制造一点纹理，避免整图完全恒定
        for _ in range(rng.randint(3, 12)):
            x0, y0 = rng.randint(0, w - 1), rng.randint(0, h - 1)
            x1, y1 = min(w, x0 + rng.randint(8, 64)), min(h, y0 + rng.randint(8, 64))
            img[y0:y1, x0:x1] = rand_color(rng)

    labels = []
    for _ in range(rng.randint(1, 4)):
        bw = rng.randint(int(w * 0.05), int(w * 0.30))
        bh = rng.randint(int(h * 0.05), int(h * 0.30))
        x0 = rng.randint(0, max(0, w - bw - 1))
        y0 = rng.randint(0, max(0, h - bh - 1))
        x1, y1 = x0 + bw, y0 + bh
        cls = rng.randint(0, ncls - 1)
        col = rand_color(rng)
        if rng.random() < 0.5:
            cv2.rectangle(img, (x0, y0), (x1, y1), col, rng.choice([-1, 2, 3]))
        else:
            cv2.circle(img, ((x0 + x1) // 2, (y0 + y1) // 2), max(3, min(bw, bh) // 2), col, -1)
        labels.append((cls, (x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h, bw / float(w), bh / float(h)))
    return img, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--nval", type=int, default=32, help="划给 val 的图数（基准只测训练步，val 只需非空）")
    ap.add_argument("--out", type=str, default="bench/toy_dataset")
    ap.add_argument("--ncls", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--noise-level", type=int, default=256)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = max(1, args.n - args.nval)
    for i in range(args.n):
        split = "val" if i >= n_train else "train"
        img, labels = make_one(i, rng, args.ncls, args.noise_level)
        cv2.imwrite(str(out / "images" / split / f"{i:06d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in labels]
        (out / "labels" / split / f"{i:06d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    names = {i: n for i, n in enumerate(CLASSES[: args.ncls])}
    yaml_txt = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {args.ncls}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in names.items())
    )
    (out / "data.yaml").write_text(yaml_txt, encoding="utf-8")

    total = sum(len(p.read_text(encoding="utf-8").splitlines())
                for p in (out / "labels" / "train").glob("*.txt"))
    print(f"[toy_dataset] train={n_train} val={args.n - n_train} train_labels={total} classes={args.ncls}")
    print(f"[toy_dataset] -> {out.resolve()}")
    print(yaml_txt)


if __name__ == "__main__":
    main()
