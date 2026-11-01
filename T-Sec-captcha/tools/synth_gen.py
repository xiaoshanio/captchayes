# -*- coding: utf-8 -*-
"""synth_gen: 合成训练数据生成器 —— 标签 100% 准确的无限数据源。

这是整条「数据更新」流水线的支点。核心洞察：
    天御两个题型（图标点选 2408 / 文字点选）都是**程序化合成**出来的：
    纯色线稿图标 + 随机旋转/缩放/配色 叠在摄影底图上。
    既然生成规则可逆，我们就能**正向重放**这个规则：
        干净底板(plates) + 图标素材(assets) → 随机合成 → 精确标注
    标签由我们自己写入，不存在标注噪声 —— 这是「数据太老」的根本解法，
    不用花钱标数据，也能随时跟着线上出题风格重新生成。

生成产物（与旧域版式完全一致，runtime 不用改）：
    images/<split>/<id>.jpg    底图 + 底部提示条（拼成单图）
    labels/<split>/<id>.txt    YOLO 格式: 0=底图图标(含干扰) 2=提示条图标
    pairs/<split>/<id>.json    配对真值(哪个提示图标对应底图哪个位置) → 训练 Siamese

用法：
    python tools/synth_gen.py --plates data/plates --assets data/assets --out data/synth \
        --n-train 3000 --n-val 300
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STRIP_H = 56
CLASS_TARGET = 0      # 底图里可点的图标（含干扰项）
CLASS_CHAR = 2        # 提示条里的图标（顺序 = 点击顺序）
CLASS_TITLE = 1       # 标题（保留位，暂不合成）


# ------------------------------------------------------------------ IO
def imread_u(p, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), flags)


def imwrite_u(p, img, q=92):
    ext = os.path.splitext(p)[1] or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), q] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(p)
    return ok


# ------------------------------------------------------------------ 几何
def rotate_rgba(img, angle, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(math.ceil(h * sin + w * cos)), int(math.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(img, M, (nw, nh), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def paste_rgba(canvas, sprite, cx, cy):
    """把 RGBA 素材以 (cx,cy) 为中心 alpha 合成到 canvas。返回实际落点框。"""
    h, w = sprite.shape[:2]
    x1, y1 = int(round(cx - w / 2)), int(round(cy - h / 2))
    x2, y2 = x1 + w, y1 + h
    H, W = canvas.shape[:2]
    sx1, sy1 = max(0, -x1), max(0, -y1)
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2, dy2 = min(W, x2), min(H, y2)
    sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)
    if dx2 <= dx1 or dy2 <= dy1:
        return None
    patch = sprite[sy1:sy2, sx1:sx2]
    a = patch[:, :, 3:4].astype(np.float32) / 255.0
    rgb = patch[:, :, :3].astype(np.float32)
    dst = canvas[dy1:dy2, dx1:dx2].astype(np.float32)
    canvas[dy1:dy2, dx1:dx2] = (rgb * a + dst * (1 - a)).astype(np.uint8)
    return (dx1, dy1, dx2, dy2)


def recolor(sprite, color_bgr=None, jitter=18):
    """给素材上色（保留 alpha），模拟题目里图标的多种配色。"""
    s = sprite.copy()
    if color_bgr is None:
        color_bgr = s[:, :, :3][s[:, :, 3] > 128].mean(axis=0) if (s[:, :, 3] > 128).any() else np.array([0, 0, 0])
    col = np.array(color_bgr, dtype=np.float32)
    if jitter:
        col = np.clip(col + np.random.randint(-jitter, jitter + 1, 3), 0, 255)
    # 只改变色相方向：把 rgb 归一化到该颜色的亮度梯度上
    rgb = s[:, :, :3].astype(np.float32)
    gray = rgb.mean(axis=2, keepdims=True)
    base = max(float(gray.mean()), 1.0)
    out = col.reshape(1, 1, 3) * (gray / base)
    s[:, :, :3] = np.clip(out, 0, 255).astype(np.uint8)
    return s


# ------------------------------------------------------------------ 合成一张
def compose_one(plate, assets, rng,
                n_decoy_range=(1, 3), n_target_range=(3, 4),
                scale_range=(0.75, 1.35), angle_jitter=25.0,
                dark_ratio=0.8, jpeg_q=(80, 95)):
    """返回 (合成图, yolo行列表, pairs真值dict)。

    流程（严格按题目反向规则）：
      1) 从素材库随机取 K 个当「提示图标」（= 要点的）
      2) 再取 D 个当干扰项
      3) 把它们随机旋转/缩放/配色，贴到底板随机位置（不越界、不重叠太狠）
      4) 底图图标框 = class 0（K+D 个，全部）
      5) 提示条 = 把 K 个素材按随机顺序横排 → class 2（顺序即点击顺序）
      6) 配对真值 = 第 i 个提示图标 对应 底图第 j 个框
    """
    H, W = plate.shape[:2]
    canvas = plate.copy()

    K = rng.randint(*n_target_range)
    D = rng.randint(*n_decoy_range)
    if len(assets) < K + D:
        K = max(1, min(K, len(assets) // 2))
        D = max(0, min(D, len(assets) - K))

    chosen = rng.sample(range(len(assets)), min(K + D, len(assets)))
    target_idx = chosen[:K]
    decoy_idx = chosen[K:K + D]

    # --- 底图贴图标
    placed = []          # [{box, asset_idx, is_target, order_in_prompt}]
    boxes = []
    attempts = 0
    for ai in target_idx + decoy_idx:
        is_target = ai in target_idx
        asset = assets[ai]
        for _ in range(40):
            attempts += 1
            sc = rng.uniform(*scale_range)
            ang = rng.uniform(-angle_jitter, angle_jitter)
            sp = asset["sprite"]
            sp = cv2.resize(sp, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
            sp = rotate_rgba(sp, ang)
            base_col = [0, 0, 0] if rng.random() < dark_ratio else \
                [rng.randint(0, 200), rng.randint(0, 200), rng.randint(0, 200)]
            sp = recolor(sp, base_col, jitter=12)
            h, w = sp.shape[:2]
            if h >= H - 4 or w >= W - 4:
                continue
            cx = rng.randint(w // 2 + 2, W - w // 2 - 2)
            cy = rng.randint(h // 2 + 2, H - h // 2 - 2)
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            # 与已放的重叠不能太大（真实题里图标基本不重叠）
            ok = True
            for b in boxes:
                if _iou(box, b) > 0.12:
                    ok = False
                    break
            if not ok:
                continue
            got = paste_rgba(canvas, sp, cx, cy)
            if got is None:
                continue
            boxes.append(box)
            placed.append({"box": box, "asset": ai, "is_target": is_target,
                           "sprite": sp})
            break

    if not placed:
        return None

    # 重新整理：真正的目标（贴成功的）
    tgt_placed = [p for p in placed if p["is_target"]]
    if not tgt_placed:
        return None

    # 点击顺序随机化（提示条顺序 ≠ 底图位置顺序）
    rng.shuffle(tgt_placed)
    for order, p in enumerate(tgt_placed):
        p["order"] = order

    # --- 拼提示条（底部）
    strip = np.full((STRIP_H, W, 3), 255, dtype=np.uint8)
    # 提示条上的图标尺寸按真实比例：约 34~44 px 高
    pad_x = rng.randint(8, 20)
    gap = rng.randint(4, 10)
    cur_x = pad_x
    prompt_boxes = []
    for p in tgt_placed:
        src = p["sprite"]
        th = rng.randint(34, 44)
        sc = th / src.shape[0]
        s2 = cv2.resize(src, (max(4, int(src.shape[1] * sc)), th), interpolation=cv2.INTER_LINEAR)
        # 提示条上的图标是纯黑
        s2 = recolor(s2, [0, 0, 0], jitter=8)
        if cur_x + s2.shape[1] + pad_x > W:
            break
        y = (STRIP_H - s2.shape[0]) // 2
        a = s2[:, :, 3:4].astype(np.float32) / 255.0
        rgb = s2[:, :, :3].astype(np.float32)
        dst = strip[y:y + s2.shape[0], cur_x:cur_x + s2.shape[1]].astype(np.float32)
        strip[y:y + s2.shape[0], cur_x:cur_x + s2.shape[1]] = (rgb * a + dst * (1 - a)).astype(np.uint8)
        prompt_boxes.append({"order": p["order"], "box": (cur_x, y, cur_x + s2.shape[1], y + s2.shape[0]),
                             "sprite": s2})
        cur_x += s2.shape[1] + gap

    if not prompt_boxes:
        return None

    # 只保留提示条上真的出现的那些目标（顺序按提示条从左到右重排）
    prompt_boxes.sort(key=lambda d: d["box"][0])
    for i, pb in enumerate(prompt_boxes):
        pb["click_order"] = i

    # --- 拼成最终图（底图 + 白条）
    final = np.full((H + STRIP_H, W, 3), 255, dtype=np.uint8)
    final[:H] = canvas
    final[H:H + STRIP_H] = strip

    # JPEG 压缩伪影（真实图的域特征之一）
    q = rng.randint(*jpeg_q)
    ok, buf = cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        final = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # --- YOLO 标签
    FH, FW = final.shape[:2]
    lines = []
    for p in placed:
        x1, y1, x2, y2 = p["box"]
        lines.append(_yolo(CLASS_TARGET, x1, y1, x2, y2, FW, FH))
    for pb in prompt_boxes:
        x1, y1, x2, y2 = pb["box"]
        lines.append(_yolo(CLASS_CHAR, x1, y1 + H, x2, y2 + H, FW, FH))

    # --- 配对真值：第 i 个提示图标 ↔ 底图哪个框
    pairs = []
    for pb in prompt_boxes:
        # 找到贴上去的那个目标（按 asset + order 匹配）
        cand = [p for p in tgt_placed if p.get("order") == pb["order"]]
        if not cand:
            continue
        p = cand[0]
        x1, y1, x2, y2 = p["box"]
        pairs.append({"click_order": pb["click_order"],
                      "prompt_box": [round(v, 1) for v in (pb["box"][0], pb["box"][1] + H,
                                                           pb["box"][2], pb["box"][3] + H)],
                      "target_box": [round(v, 1) for v in (x1, y1, x2, y2)],
                      "target_center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                      "asset": p["asset"]})

    truth = {
        "imgW": FW, "imgH": FH, "strip_h": STRIP_H,
        "n_prompt": len(prompt_boxes), "n_target_bg": len(placed),
        "n_decoy": sum(1 for p in placed if not p["is_target"]),
        "pairs": pairs,
        "click_centers": [[round((p['box'][0] + p['box'][2]) / 2, 1),
                           round((p['box'][1] + p['box'][3]) / 2, 1)] for p in
                          sorted([c for c in placed if c["is_target"] and
                                  any(q["target_box"] == [round(v, 1) for v in c["box"]] for q in pairs)],
                                 key=lambda c: next(q["click_order"] for q in pairs
                                                    if q["target_box"] == [round(v, 1) for v in c["box"]]))],
    }
    return final, lines, truth


def _yolo(cls, x1, y1, x2, y2, W, H):
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    bw, bh = (x2 - x1) / W, (y2 - y1) / H
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


# ------------------------------------------------------------------ 驱动
def load_assets(assets_dir, min_h=18, max_h=90):
    idx = os.path.join(assets_dir, "icons_index.jsonl")
    if not os.path.exists(idx):
        raise SystemExit(f"缺素材索引: {idx}（先跑 tools/icon_assets.py）")
    out = []
    for line in open(idx, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        p = os.path.join(assets_dir, "icons", r["file"])
        if not os.path.exists(p):
            continue
        sp = imread_u(p)
        if sp is None or sp.shape[2] != 4:
            continue
        if not (min_h <= sp.shape[0] <= max_h):
            continue
        out.append({"hash": r["hash"], "sprite": sp, "w": sp.shape[1], "h": sp.shape[0]})
    if not out:
        raise SystemExit("素材为空")
    return out


def load_plates(plates_dir):
    idx = os.path.join(plates_dir, "plates_index.jsonl")
    plates = []
    if os.path.exists(idx):
        for line in open(idx, encoding="utf-8"):
            r = json.loads(line)
            p = os.path.join(plates_dir, r["file"])
            if os.path.exists(p):
                plates.append(p)
    else:
        plates = sorted(glob.glob(os.path.join(plates_dir, "*.jpg")))
    if not plates:
        raise SystemExit(f"缺底板: {plates_dir}（先跑 tools/make_plates.py）")
    return plates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="data/plates")
    ap.add_argument("--assets", default="data/assets")
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-decoy", type=int, default=1)
    ap.add_argument("--max-decoy", type=int, default=3)
    ap.add_argument("--min-target", type=int, default=3)
    ap.add_argument("--max-target", type=int, default=4)
    ap.add_argument("--scale-min", type=float, default=0.75)
    ap.add_argument("--scale-max", type=float, default=1.35)
    ap.add_argument("--angle", type=float, default=25.0)
    ap.add_argument("--draw", type=int, default=0, help="额外输出 N 张可视化")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    assets = load_assets(args.assets)
    plates = load_plates(args.plates)
    print(f"[data] 素材 {len(assets)} 个, 底板 {len(plates)} 张")

    plate_imgs = []
    for p in plates[:60]:
        im = imread_u(p, cv2.IMREAD_COLOR)
        if im is not None:
            plate_imgs.append(im)
    if not plate_imgs:
        raise SystemExit("底板读取失败")

    for sub in ["images/train", "images/val", "labels/train", "labels/val",
                "pairs/train", "pairs/val"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    if args.draw:
        os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)

    stats = {"ok": 0, "fail": 0, "prompts": [], "targets": []}
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        for i in range(n):
            plate = rng.choice(plate_imgs)
            try:
                r = compose_one(plate, assets, rng,
                                n_decoy_range=(args.min_decoy, args.max_decoy),
                                n_target_range=(args.min_target, args.max_target),
                                scale_range=(args.scale_min, args.scale_max),
                                angle_jitter=args.angle)
            except Exception:
                r = None
            if r is None:
                stats["fail"] += 1
                continue
            final, lines, truth = r
            name = f"{split}_{i:06d}"
            imwrite_u(os.path.join(args.out, "images", split, name + ".jpg"), final)
            with open(os.path.join(args.out, "labels", split, name + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            with open(os.path.join(args.out, "pairs", split, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(truth, f, ensure_ascii=False, indent=1)
            stats["ok"] += 1
            stats["prompts"].append(truth["n_prompt"])
            stats["targets"].append(truth["n_target_bg"])

            if args.draw and i < args.draw:
                vis = final.copy()
                for ln in lines:
                    c, cx, cy, bw, bh = ln.split()
                    x1 = int((float(cx) - float(bw) / 2) * final.shape[1])
                    y1 = int((float(cy) - float(bh) / 2) * final.shape[0])
                    x2 = int((float(cx) + float(bw) / 2) * final.shape[1])
                    y2 = int((float(cy) + float(bh) / 2) * final.shape[0])
                    col = (0, 0, 255) if c == "0" else (0, 170, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                for q in truth["pairs"]:
                    cv2.putText(vis, f"#{q['click_order']+1}",
                                (int(q["target_center"][0]) - 10, int(q["target_center"][1]) + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                imwrite_u(os.path.join(args.out, "vis", name + ".jpg"), vis)
        print(f"[{split}] 生成 {n} 张完成")

    import statistics as st
    print("\n=== 合成统计 ===")
    print(f"  成功 {stats['ok']}  失败 {stats['fail']}")
    if stats["prompts"]:
        print(f"  提示图标数 : mean={st.mean(stats['prompts']):.2f} "
              f"min={min(stats['prompts'])} max={max(stats['prompts'])}")
        print(f"  底图图标数 : mean={st.mean(stats['targets']):.2f} "
              f"min={min(stats['targets'])} max={max(stats['targets'])}")
    # data.yaml
    yml = os.path.join(args.out, "data.yaml")
    with open(yml, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write("names:\n  0: label\n  1: title\n  2: char\n")
    print(f"  数据集配置 : {yml}")
    print(f"  输出       : {args.out}/")
    if args.draw:
        print(f"  可视化     : {args.out}/vis/")


if __name__ == "__main__":
    main()
