# -*- coding: utf-8 -*-
"""ingest_captures: 把**你自己截的图**整理成下游管线要的 data/samples 结构。

为什么需要这个工具：
    tools/plate_gen.py 和 tools/icon_assets.py 都要求这样的目录结构：
        data/samples/<sid>/meta.json
        data/samples/<sid>/<主图>.jpg
        data/samples/<sid>/prompt_0.jpg   （提示条/提示图，可多张）
    但人手工截图通常是"一堆散落的 png/jpg"，名字也不规整。
    这个工具负责把散图整理成规范结构，并做**质量闸门**，避免脏图污染训练集。

重要：这个工具只处理**你自己有权使用的图片**
    （自有服务、已获授权、公开演示页手动截图、或你自己的测试环境）。
    它不是采集器，不做任何网络请求。

两种用法
--------
1) 简单模式：一个目录里放截图，每张图 = 一个样本（图里同时含底图和提示条）
       python tools/ingest_captures.py --src my_shots --out data/samples
   工具会尝试自动切出底部提示条（浅色底 + 深色图标的那条），
   切不出来就把整图当主图、不产出 prompt 图（仍可用于底板挖掘）。

2) 精确模式：你已经按规范摆好了
       my_shots/<sid>/main.jpg
       my_shots/<sid>/prompt_0.jpg
       python tools/ingest_captures.py --src my_shots --out data/samples --presorted

质量闸门（不合格的图会被拒绝并记录原因，不会静默混进去）
    - 最小尺寸 200x150
    - 不能近乎全白/全黑（方差过低）
    - 不能是重复图（感知哈希去重）
    - 提示条切分后必须至少含 2 个图标，否则视为切分失败

产物
    data/samples/<sid>/meta.json   含 source/尺寸/提示条位置/质量指标
    data/samples/ingest_report.json  汇总（收了多少、拒了多少、为什么）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def imread_u(p):
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(p, img, q=95):
    ext = os.path.splitext(p)[1].lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), q] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(p)
    return ok


def phash(img, size=8):
    """简化感知哈希：灰度缩放 + DCT 低频符号。用于去重。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:size, :size]
    med = np.median(d[1:, 1:])
    bits = (d > med).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def find_strip(img):
    """在底部找"浅色提示条"：连续多行接近白、且条内有深色笔画。

    返回 (strip_top, strip_img) 或 (None, None)。
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 每行的"亮像素占比"和"暗像素占比"
    row_light = (gray > 235).mean(axis=1)
    row_dark = (gray < 110).mean(axis=1)

    best = None
    # 提示条一般占底部 6%~30%
    for h in range(max(20, int(H * 0.06)), int(H * 0.30) + 1):
        top = H - h
        band_light = row_light[top:].mean()
        band_dark = row_dark[top:].mean()
        # 条件：整条足够亮，且内部有深色图标（不是纯白空条）
        if band_light > 0.80 and 0.002 < band_dark < 0.35:
            score = band_light + band_dark
            if best is None or score > best[0]:
                best = (score, top, h)
    if best is None:
        return None, None
    _, top, h = best
    return top, img[top:top + h]


def count_icons_in_strip(strip):
    """粗估提示条里有几个图标：暗像素的连通域计数（过滤太小/太扁）。"""
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    m = (g < 110).astype(np.uint8)
    if m.sum() < 30:
        return 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    cnt = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 25 or min(w, h) < 6:
            continue
        if h < strip.shape[0] * 0.25:
            continue
        cnt += 1
    return cnt


def quality_ok(img, min_w=200, min_h=150):
    """返回 (ok, reason)。"""
    H, W = img.shape[:2]
    if W < min_w or H < min_h:
        return False, f"尺寸过小 {W}x{H} < {min_w}x{min_h}"
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.std() < 8:
        return False, f"方差过低(std={g.std():.1f})，近乎纯色"
    if g.mean() > 250:
        return False, "近乎全白"
    if g.mean() < 5:
        return False, "近乎全黑"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="你的截图目录")
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--presorted", action="store_true",
                    help="src 已经是 <sid>/main.jpg + prompt_N.jpg 结构")
    ap.add_argument("--source", default="self-capture",
                    help="来源标注，写进 meta.json（便于日后追溯授权）")
    ap.add_argument("--dedup-dist", type=int, default=4,
                    help="感知哈希汉明距离 <= 该值视为重复图")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的样本")
    ap.add_argument("--dry-run", action="store_true", help="只体检不落盘")
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(REPO, args.src)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    if not os.path.isdir(src):
        raise SystemExit(f"源目录不存在: {src}")
    os.makedirs(out, exist_ok=True)

    report = {"src": src, "out": out, "source": args.source,
              "accepted": [], "rejected": [], "duplicates": []}
    hashes = []

    # 收集候选
    if args.presorted:
        cands = []
        for sid in sorted(os.listdir(src)):
            d = os.path.join(src, sid)
            if not os.path.isdir(d):
                continue
            mains = [f for f in sorted(os.listdir(d))
                     if f.lower().endswith(IMG_EXT) and not f.startswith("prompt")]
            if mains:
                cands.append((sid, os.path.join(d, mains[0]),
                              [os.path.join(d, f) for f in sorted(os.listdir(d))
                               if f.lower().startswith("prompt")]))
    else:
        cands = []
        for f in sorted(os.listdir(src)):
            if f.lower().endswith(IMG_EXT):
                sid = os.path.splitext(f)[0]
                cands.append((sid, os.path.join(src, f), []))

    print(f"[scan] {len(cands)} 个候选样本")

    for sid, main_path, prompt_paths in cands:
        img = imread_u(main_path)
        if img is None:
            report["rejected"].append({"sid": sid, "reason": "无法解码"})
            continue
        ok, reason = quality_ok(img)
        if not ok:
            report["rejected"].append({"sid": sid, "reason": reason})
            continue

        # 去重
        ph = phash(img)
        dup = next((h for h in hashes if hamming(h[0], ph) <= args.dedup_dist), None)
        if dup:
            report["duplicates"].append({"sid": sid, "same_as": dup[1]})
            continue
        hashes.append((ph, sid))

        H, W = img.shape[:2]
        strip_top, strip = (None, None)
        if not prompt_paths:
            strip_top, strip = find_strip(img)

        n_icons = count_icons_in_strip(strip) if strip is not None else 0

        rec = {
            "sid": sid,
            "source": args.source,
            "w": W, "h": H,
            "strip_top": strip_top,
            "strip_h": (H - strip_top) if strip_top is not None else None,
            "n_icons_in_strip": n_icons,
            "phash": ph,
            "note": ("自动切出提示条" if strip is not None else
                     "未切出提示条（整图当主图，仅供底板挖掘）"),
        }
        report["accepted"].append(rec)

        if args.dry_run:
            continue

        d = os.path.join(out, sid)
        if os.path.exists(d) and not args.force:
            continue
        os.makedirs(d, exist_ok=True)
        imwrite_u(os.path.join(d, "main.jpg"), img)

        prompts = []
        if prompt_paths:
            for i, pp in enumerate(prompt_paths):
                pi = imread_u(pp)
                if pi is None:
                    continue
                fn = f"prompt_{i}.jpg"
                imwrite_u(os.path.join(d, fn), pi)
                prompts.append({"file": fn, "w": pi.shape[1], "h": pi.shape[0]})
        elif strip is not None and n_icons >= 2:
            fn = "prompt_0.jpg"
            imwrite_u(os.path.join(d, fn), strip)
            prompts.append({"file": fn, "w": strip.shape[1], "h": strip.shape[0]})

        # bg.jpg = 去掉提示条后的底图。
        # tools/plate_gen.py 明确读 <sid>/bg.jpg（不是 main.jpg），
        # 少了这个文件整条底板挖掘链路会静默跳过所有样本。
        if strip_top is not None and strip_top > 20:
            bg = img[:strip_top]
        else:
            bg = img
        imwrite_u(os.path.join(d, "bg.jpg"), bg)
        rec["bg_h"] = bg.shape[0]

        rec["prompts"] = prompts
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({**rec, "prompts": prompts}, f, ensure_ascii=False, indent=2)

    # 汇总
    n_ok, n_rej, n_dup = (len(report["accepted"]), len(report["rejected"]),
                          len(report["duplicates"]))
    print(f"\n{'='*76}")
    print(f"收下 {n_ok}  拒绝 {n_rej}  重复 {n_dup}")
    print(f"{'='*76}")
    if report["rejected"]:
        print("  拒绝原因：")
        for r in report["rejected"][:12]:
            print(f"    - {r['sid']}: {r['reason']}")
    if report["duplicates"]:
        print("  重复样本：")
        for r in report["duplicates"][:12]:
            print(f"    - {r['sid']} ≈ {r['same_as']}")
    with_strip = sum(1 for r in report["accepted"] if r.get("strip_top") is not None)
    if n_ok:
        print(f"  自动切出提示条: {with_strip}/{n_ok}")
        print(f"  提示条图标数中位: "
              f"{int(np.median([r['n_icons_in_strip'] for r in report['accepted']]))}")

    if not args.dry_run:
        rp = os.path.join(out, "ingest_report.json")
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告 -> {rp}")
        print("\n下一步：")
        print(f"  python tools/plate_gen.py   --samples {os.path.relpath(out, REPO)} --out data/plates")
        print(f"  python tools/icon_assets.py --samples {os.path.relpath(out, REPO)} --out data/assets")
        print(f"  python tools/synth_gen.py   --plates data/plates --assets data/assets --out data/synth")


if __name__ == "__main__":
    main()
