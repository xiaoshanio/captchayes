# -*- coding: utf-8 -*-
"""synth_selfcontained: 完全自造的合成数据生成器（零第三方数据）。

为什么单独写一个：
    tools/synth_gen.py 需要「真实站点的底板 + 图标素材」。
    为了验证「继续训练」这条链路本身，我们需要一份**完全由代码生成**的数据，
    不依赖任何外部站点 —— 这样验证结论才干净、可复现、可公开。

生成内容（与 runtime 期望的版式完全一致）：
    - 图标素材：用 OpenCV 基元程序化绘制线稿图标（数字/字母/简单象形）
    - 底板    ：程序化生成类照片背景（渐变天空 / 噪声纹理 / 几何色块）
    - 合成图  ：底板 + 底部白条提示条（拼成单图，与旧域 344x384 版式同构）
    - 标签    ：class 0 = 底图图标(含干扰) ；class 2 = 提示条图标
    - 配对真值：第 i 个提示图标 ↔ 底图哪个框

用法：
    python tools/synth_selfcontained.py --out data/selfgen --n-train 800 --n-val 120
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import cv2
import numpy as np

STRIP_H = 56
CLASS_TARGET = 0
CLASS_CHAR = 2


# ================================================================ 域风格预设
#
# 为什么要做「域风格」：
#   「数据过于老了」在真机上的典型表现是——出题方换了图标字形、描边粗细、
#   尺寸分布、旋转幅度、背景风格。模型不是"精度掉一点"，而是**结构上失效**。
#   为了能**量化**这个失效、并验证「继续训练 + 数据更新」能不能救回来，
#   生成器需要能产出两个明确不同的域。
#
# v1 = 旧域（基线）；v2 = 漂移域（模拟出题方改版）
STYLE_PRESETS = {
    "v1": dict(
        icon_style="basic", thick=3,
        n_target=(3, 4), n_decoy=(1, 3),
        scale=(0.9, 1.5), angle=18.0, dark_ratio=0.85,
        jpeg_q=(80, 95), min_box=42, min_gap=10,
        strip_h=56, strip_pad=(16, 28), strip_gap=(14, 22),
        plate_style="geometric",
    ),
    "v2": dict(
        # 漂移点：新字形集 + 更粗描边 + 更小尺寸 + 更大旋转 + 更"照片化"背景 + 更高提示条
        icon_style="alt", thick=5,
        n_target=(3, 4), n_decoy=(2, 4),
        scale=(0.62, 1.05), angle=38.0, dark_ratio=0.6,
        jpeg_q=(62, 82), min_box=34, min_gap=8,
        strip_h=68, strip_pad=(20, 34), strip_gap=(16, 26),
        plate_style="texture",
    ),
}


def render_icon_alt(kind: int, idx: int, size=64, thick=5) -> np.ndarray:
    """漂移域的字形集：与 v1 完全不重叠的图形语言（箭头/月牙/闪电/水滴/齿轮…）。"""
    S = size
    img = np.full((S, S, 4), 0, np.uint8)
    c = (S // 2, S // 2)
    col = (0, 0, 0, 255)
    r = S // 2 - 3

    if kind == 0:      # 右箭头
        pts = np.array([[c[0] - r, c[1] - r // 2], [c[0], c[1] - r // 2],
                        [c[0], c[1] - r], [c[0] + r, c[1]],
                        [c[0], c[1] + r], [c[0], c[1] + r // 2],
                        [c[0] - r, c[1] + r // 2]], np.int32)
        cv2.polylines(img, [pts], True, col, thick, cv2.LINE_AA)
    elif kind == 1:    # 月牙（两圆相减）
        m = np.zeros((S, S), np.uint8)
        cv2.circle(m, c, r, 255, -1)
        cv2.circle(m, (c[0] + r // 2, c[1] - r // 3), int(r * 0.82), 0, -1)
        img[m > 0] = col
    elif kind == 2:    # 闪电
        pts = np.array([[c[0] + r // 3, c[1] - r], [c[0] - r // 2, c[1] + r // 6],
                        [c[0] + r // 6, c[1] + r // 6], [c[0] - r // 3, c[1] + r]], np.int32)
        cv2.polylines(img, [pts], False, col, thick, cv2.LINE_AA)
    elif kind == 3:    # 水滴
        pts = np.array([[c[0], c[1] - r], [c[0] + int(r * 0.72), c[1] + int(r * 0.35)],
                        [c[0], c[1] + r], [c[0] - int(r * 0.72), c[1] + int(r * 0.35)]], np.int32)
        cv2.polylines(img, [pts], True, col, thick, cv2.LINE_AA)
    elif kind == 4:    # 齿轮（粗齿）
        n = 8
        pts = []
        for k in range(n * 2):
            ang = k * math.pi / n
            rr = r if k % 2 == 0 else r * 0.66
            pts.append([int(c[0] + rr * math.cos(ang)), int(c[1] + rr * math.sin(ang))])
        cv2.polylines(img, [np.array(pts, np.int32)], True, col, thick, cv2.LINE_AA)
    elif kind == 5:    # 半圆（下半）
        cv2.ellipse(img, c, (r, r), 0, 0, 180, col, thick, cv2.LINE_AA)
    elif kind == 6:    # 双向箭头
        cv2.line(img, (c[0] - r, c[1]), (c[0] + r, c[1]), col, thick, cv2.LINE_AA)
        cv2.polylines(img, [np.array([[c[0] - r, c[1]], [c[0] - r + r // 2, c[1] - r // 2],
                                      [c[0] - r + r // 2, c[1] + r // 2]], np.int32)],
                      True, col, thick, cv2.LINE_AA)
        cv2.polylines(img, [np.array([[c[0] + r, c[1]], [c[0] + r - r // 2, c[1] - r // 2],
                                      [c[0] + r - r // 2, c[1] + r // 2]], np.int32)],
                      True, col, thick, cv2.LINE_AA)
    else:              # 感叹号/棒状
        cv2.line(img, (c[0], c[1] - r), (c[0], c[1] + r // 3), col, thick, cv2.LINE_AA)
        cv2.circle(img, (c[0], c[1] + int(r * 0.78)), max(2, thick // 2 + 1), col, -1, cv2.LINE_AA)
    return img


# ================================================================ 图标渲染
def render_icon(kind: int, idx: int, size=64, thick=3) -> np.ndarray:
    """程序化绘制一个线稿图标，返回 RGBA（透明底、纯黑笔画）。"""
    S = size
    img = np.full((S, S, 4), 0, np.uint8)
    c = (S // 2, S // 2)
    col = (0, 0, 0, 255)
    r = S // 2 - 4

    if kind == 0:      # 数字/字母（0-9, A-H）
        ch = "0123456789ABCDEFGH"[idx % 16]
        font = cv2.FONT_HERSHEY_SIMPLEX
        sc = 1.35
        (tw, th), _ = cv2.getTextSize(ch, font, sc, thick)
        cv2.putText(img, ch, (c[0] - tw // 2, c[1] + th // 2), font, sc, col, thick, cv2.LINE_AA)
    elif kind == 1:    # 圆环
        cv2.circle(img, c, r, col, thick, cv2.LINE_AA)
    elif kind == 2:    # 三角
        pts = np.array([[c[0], c[1] - r], [c[0] - r, c[1] + r], [c[0] + r, c[1] + r]], np.int32)
        cv2.polylines(img, [pts], True, col, thick, cv2.LINE_AA)
    elif kind == 3:    # 方框
        cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), col, thick, cv2.LINE_AA)
    elif kind == 4:    # 十字
        cv2.line(img, (c[0] - r, c[1]), (c[0] + r, c[1]), col, thick, cv2.LINE_AA)
        cv2.line(img, (c[0], c[1] - r), (c[0], c[1] + r), col, thick, cv2.LINE_AA)
    elif kind == 5:    # 五角星
        pts = []
        for k in range(10):
            ang = -math.pi / 2 + k * math.pi / 5
            rr = r if k % 2 == 0 else r * 0.45
            pts.append([int(c[0] + rr * math.cos(ang)), int(c[1] + rr * math.sin(ang))])
        cv2.polylines(img, [np.array(pts, np.int32)], True, col, thick, cv2.LINE_AA)
    elif kind == 6:    # 菱形
        pts = np.array([[c[0], c[1] - r], [c[0] + r, c[1]], [c[0], c[1] + r], [c[0] - r, c[1]]], np.int32)
        cv2.polylines(img, [pts], True, col, thick, cv2.LINE_AA)
    else:              # 折线（类箭头/屋顶）
        pts = np.array([[c[0] - r, c[1] + r // 2], [c[0], c[1] - r], [c[0] + r, c[1] + r // 2]], np.int32)
        cv2.polylines(img, [pts], False, col, thick, cv2.LINE_AA)
    return img


def build_icon_bank(n_per_kind=4, size=64, thick=3, style="basic"):
    """构建图标素材库。style='basic' 为旧域字形，'alt' 为漂移域字形。"""
    bank = []
    renderer = render_icon if style == "basic" else render_icon_alt
    for kind in range(8):
        for i in range(n_per_kind):
            sp = renderer(kind, i, size, thick)
            a = sp[:, :, 3]
            ys, xs = np.nonzero(a)
            if len(xs) < 20:
                continue
            sp = sp[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if min(sp.shape[:2]) < 10:
                continue
            bank.append(sp)
    return bank


# ================================================================ 底板渲染
def make_plate(rng: random.Random, W=672, H=480, style="geometric") -> np.ndarray:
    """程序化生成类照片背景。

    style='geometric'  : 旧域（渐变 + 几何色块，比较"干净"）
    style='texture'    : 漂移域（更接近真实照片：多尺度噪声 + 强纹理 + 更重压缩）
    """
    mode = rng.randint(0, 3)

    # 1) 天空渐变
    top = np.array([rng.randint(120, 255), rng.randint(140, 255), rng.randint(180, 255)], np.float32)
    bot = np.array([rng.randint(60, 200), rng.randint(80, 220), rng.randint(90, 230)], np.float32)
    t = np.linspace(0, 1, H, dtype=np.float32).reshape(-1, 1, 1)
    img = (top.reshape(1, 1, 3) * (1 - t) + bot.reshape(1, 1, 3) * t)
    img = np.repeat(img, W, axis=1)

    # 2) 随机几何色块
    if mode in (1, 2, 3):
        n = rng.randint(3, 7)
        for _ in range(n):
            cx, cy = rng.randint(0, W), rng.randint(0, H)
            rad = rng.randint(40, 150)
            col = [rng.randint(0, 255) for _ in range(3)]
            if mode == 3:
                pts = np.array([[cx + int(rad * math.cos(a)), cy + int(rad * math.sin(a))]
                                for a in np.linspace(0, 2 * math.pi, rng.randint(5, 8), endpoint=False)], np.int32)
                cv2.fillPoly(img, [pts], col)
            else:
                cv2.circle(img, (cx, cy), rad, col, -1)

    if style == "texture":
        # 漂移域专属：叠加多尺度平滑噪声，模拟真实照片的纹理与光照不均
        base = img.astype(np.float32)
        for scale, amp in ((64, 60), (24, 32), (8, 18)):
            small = np.random.rand(max(2, H // scale + 1), max(2, W // scale + 1), 3).astype(np.float32)
            up = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
            base += (up - 0.5) * amp
        # 强暗角
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        base *= np.clip(1.0 - 0.28 * d ** 2, 0.5, 1.0)[:, :, None]
        img = np.clip(base, 0, 255).astype(np.uint8)

    # 3) 纹理噪声（模拟照片颗粒）
    noise = np.random.normal(0, rng.uniform(4, 16), (H, W, 3)).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # 4) 轻微模糊 + JPEG 压缩伪影
    if rng.random() < (0.9 if style == "texture" else 0.7):
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


# ================================================================ 合成
def rotate_rgba(img, angle):
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(math.ceil(h * sin + w * cos)), int(math.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def paste_rgba(canvas, sprite, cx, cy):
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


def recolor(sprite, color_bgr, jitter=12):
    s = sprite.copy()
    col = np.array(color_bgr, np.float32)
    if jitter:
        col = np.clip(col + np.random.randint(-jitter, jitter + 1, 3), 0, 255)
    rgb = s[:, :, :3].astype(np.float32)
    gray = rgb.mean(axis=2, keepdims=True)
    base = max(float(gray.mean()), 1.0)
    s[:, :, :3] = np.clip(col.reshape(1, 1, 3) * (gray / base), 0, 255).astype(np.uint8)
    return s


def _near(a, b, gap=0, iou_thr=0.08):
    """两个框是否太近（净空不足）或重叠过多。"""
    if (a[0] - gap < b[2] and b[0] < a[2] + gap and
            a[1] - gap < b[3] and b[1] < a[3] + gap):
        return True
    return iou(a, b) > iou_thr


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def yolo_line(cls, box, W, H):
    x1, y1, x2, y2 = box
    return (f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
            f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")


def compose_one(bank, rng, canvas_w=672, canvas_h=480, style="v1", **overrides):
    """合成一题。

    style: 'v1'（旧域）或 'v2'（漂移域）。所有风格参数来自 STYLE_PRESETS，
           也可用 **overrides 单独覆盖某一项。

    参数经验值（来自 eval/e2e_diag.py 的端到端诊断）：
      - scale (0.9,1.5)  ：底图图标不能太小，否则检测器召回不足；也不能太密
      - angle 18         ：旋转过大时图标框会互相挤压
      - min_box 42       ：过滤掉过小的贴图（真实题里图标本体都 >= 40px）
      - min_gap 10       ：底图图标之间至少留 10px 净空，避免框粘连
    """
    S = dict(STYLE_PRESETS[style])
    S.update(overrides)
    strip_h = S["strip_h"]
    n_target, n_decoy = S["n_target"], S["n_decoy"]
    scale, angle = S["scale"], S["angle"]
    dark_ratio, jpeg_q = S["dark_ratio"], S["jpeg_q"]
    min_box, min_gap = S["min_box"], S["min_gap"]

    plate = make_plate(rng, canvas_w, canvas_h, style=S["plate_style"])
    canvas = plate.copy()
    H, W = canvas.shape[:2]

    K = rng.randint(*n_target)
    D = rng.randint(*n_decoy)
    K = min(K, len(bank))
    pool = rng.sample(range(len(bank)), min(K + D, len(bank)))
    tgt_ids, decoy_ids = pool[:K], pool[K:]

    placed, boxes = [], []
    for ai in tgt_ids + decoy_ids:
        is_t = ai in tgt_ids
        for _ in range(60):
            sc = rng.uniform(*scale)
            sp = cv2.resize(bank[ai], None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
            sp = rotate_rgba(sp, rng.uniform(-angle, angle))
            base_col = [0, 0, 0] if rng.random() < dark_ratio else \
                [rng.randint(0, 200) for _ in range(3)]
            sp = recolor(sp, base_col)
            h, w = sp.shape[:2]
            if h >= H - 4 or w >= W - 4 or min(h, w) < min_box:
                continue
            cx = rng.randint(w // 2 + 2, W - w // 2 - 2)
            cy = rng.randint(h // 2 + 2, H - h // 2 - 2)
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            # 与已放图标：既要留净空，也不能重叠
            if any(_near(box, b, min_gap) for b in boxes):
                continue
            if paste_rgba(canvas, sp, cx, cy) is None:
                continue
            boxes.append(box)
            placed.append({"box": box, "asset": ai, "is_target": is_t, "sprite": sp})
            break

    tgt_placed = [p for p in placed if p["is_target"]]
    if not tgt_placed:
        return None
    rng.shuffle(tgt_placed)
    for o, p in enumerate(tgt_placed):
        p["order"] = o

    # 提示条
    #
    # 经验教训（来自端到端诊断 eval/e2e_diag.py）：
    #   提示条图标如果排得太挤，检测器会把相邻图标并成一个框，或者一个图标裂成两个框，
    #   导致 class2 检出数 != 提示图标数 -> 贪心指派的矩阵尺寸错 -> 点击顺序必然错。
    #   所以这里强制：较宽的 gap、较大的左右留白、图标之间至少留净空。
    strip = np.full((strip_h, W, 3), 255, np.uint8)
    pad_x = rng.randint(*S["strip_pad"])
    gap = rng.randint(*S["strip_gap"])
    cur_x = pad_x
    prompt_boxes = []
    for p in tgt_placed:
        src = p["sprite"]
        th = rng.randint(int(strip_h * 0.6), int(strip_h * 0.79))
        sc = th / src.shape[0]
        s2 = cv2.resize(src, (max(4, int(src.shape[1] * sc)), th), interpolation=cv2.INTER_LINEAR)
        s2 = recolor(s2, [0, 0, 0], jitter=6)
        if cur_x + s2.shape[1] + pad_x > W:
            break
        y = (strip_h - s2.shape[0]) // 2
        a = s2[:, :, 3:4].astype(np.float32) / 255.0
        rgb = s2[:, :, :3].astype(np.float32)
        dst = strip[y:y + s2.shape[0], cur_x:cur_x + s2.shape[1]].astype(np.float32)
        strip[y:y + s2.shape[0], cur_x:cur_x + s2.shape[1]] = (rgb * a + dst * (1 - a)).astype(np.uint8)
        prompt_boxes.append({"order": p["order"], "box": (cur_x, y, cur_x + s2.shape[1], y + s2.shape[0])})
        cur_x += s2.shape[1] + gap
    if not prompt_boxes:
        return None

    prompt_boxes.sort(key=lambda d: d["box"][0])
    for i, pb in enumerate(prompt_boxes):
        pb["click_order"] = i

    final = np.full((H + strip_h, W, 3), 255, np.uint8)
    final[:H] = canvas
    final[H:H + strip_h] = strip
    ok, buf = cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(*jpeg_q)])
    if ok:
        final = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    FH, FW = final.shape[:2]
    lines = [yolo_line(CLASS_TARGET, b, FW, FH) for b in boxes]
    lines += [yolo_line(CLASS_CHAR, (pb["box"][0], pb["box"][1] + H,
                                     pb["box"][2], pb["box"][3] + H), FW, FH)
              for pb in prompt_boxes]

    pairs = []
    for pb in prompt_boxes:
        cand = [p for p in tgt_placed if p.get("order") == pb["order"]]
        if not cand:
            continue
        x1, y1, x2, y2 = cand[0]["box"]
        pairs.append({"click_order": pb["click_order"],
                      "prompt_box": [round(v, 1) for v in (pb["box"][0], pb["box"][1] + H,
                                                           pb["box"][2], pb["box"][3] + H)],
                      "target_box": [round(v, 1) for v in (x1, y1, x2, y2)],
                      "target_center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                      "asset": cand[0]["asset"]})

    truth = {"imgW": FW, "imgH": FH, "strip_h": strip_h, "style": style,
             "n_prompt": len(prompt_boxes), "n_target_bg": len(placed),
             "n_decoy": sum(1 for p in placed if not p["is_target"]),
             "pairs": pairs}
    return final, lines, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/selfgen")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--style", default="v1", choices=sorted(STYLE_PRESETS),
                    help="域风格：v1=旧域，v2=漂移域（模拟出题方改版）")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    S = STYLE_PRESETS[args.style]
    bank = build_icon_bank(thick=S["thick"], style=S["icon_style"])
    print(f"[icons] 程序化生成 {len(bank)} 个图标素材 (style={args.style}, "
          f"icon_style={S['icon_style']}, thick={S['thick']})")

    for sub in ["images/train", "images/val", "labels/train", "labels/val",
                "pairs/train", "pairs/val"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    if args.draw:
        os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)

    stats = {"ok": 0, "fail": 0, "prompts": [], "targets": []}
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        for i in range(n):
            try:
                r = compose_one(bank, rng, style=args.style)
            except Exception:
                r = None
            if r is None:
                stats["fail"] += 1
                continue
            final, lines, truth = r
            name = f"{split}_{i:06d}"
            ok, buf = cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            buf.tofile(os.path.join(args.out, "images", split, name + ".jpg"))
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
                buf2 = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1]
                buf2.tofile(os.path.join(args.out, "vis", name + ".jpg"))
        print(f"[{split}] {n} 张")

    import statistics as st
    print(f"\n=== 合成统计 (style={args.style}) ===")
    print(f"  成功 {stats['ok']} 失败 {stats['fail']}")
    if stats["prompts"]:
        print(f"  提示图标数 mean={st.mean(stats['prompts']):.2f} "
              f"({min(stats['prompts'])}~{max(stats['prompts'])})")
        print(f"  底图图标数 mean={st.mean(stats['targets']):.2f} "
              f"({min(stats['targets'])}~{max(stats['targets'])})")

    yml = os.path.join(args.out, "data.yaml")
    with open(yml, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write("names:\n  0: label\n  1: title\n  2: char\n")
    # 记录生成参数，保证可复现（数据版本化）
    with open(os.path.join(args.out, "gen_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"style": args.style, "seed": args.seed,
                   "n_train": args.n_train, "n_val": args.n_val,
                   "preset": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in S.items()},
                   "gen_script": "tools/synth_selfcontained.py"}, f,
                  ensure_ascii=False, indent=2)
    print(f"  data.yaml : {yml}")
    print(f"  manifest  : {os.path.join(args.out, 'gen_manifest.json')}")


if __name__ == "__main__":
    main()
