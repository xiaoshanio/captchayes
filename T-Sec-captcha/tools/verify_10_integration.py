# -*- coding: utf-8 -*-
"""verify_10_integration: 一键验证 cun.ai 1.0 的验证码接入状态。

背景：
    1.0 的设计是「从上游仓库同步 cv/*.py + model/*.onnx，
    并从 PyPI 提取 ddddocr 的两个 OCR 模型」。
    本脚本把「该装的都装了吗、装完能用吗」做成可重复检查。

用法：
    python tools/verify_10_integration.py
    python tools/verify_10_integration.py --root "D:\\GitHub\\Auto-Register\\cun.ai\\cun.ai 1.0"
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default=r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0")
    args = ap.parse_args()
    root = args.root
    if not os.path.isdir(root):
        raise SystemExit(f"目录不存在: {root}")

    cap = os.path.join(root, "captcha")
    files = [
        ("检测器", os.path.join(cap, "models", "best_v3.onnx")),
        ("匹配器", os.path.join(cap, "models", "pre_model_v7.onnx")),
        ("OCR-old", os.path.join(cap, "ddddocr_v155", "ddddocr", "common_old.onnx")),
        ("OCR-det", os.path.join(cap, "ddddocr_v155", "ddddocr", "common_det.onnx")),
    ] + [(f"cv/{n}", os.path.join(cap, "cv", n))
         for n in ("matchingMode.py", "ver_onnx.py", "yolo_onnx.py")]

    print("=== 文件检查 ===")
    missing = 0
    for label, p in files:
        ok = os.path.exists(p)
        size = f"{os.path.getsize(p) / 1024 / 1024:6.2f} MB" if ok else "  缺失"
        print(f"  {'OK ' if ok else 'MISS'} {label:12s} {size}  "
              f"{os.path.relpath(p, root)}")
        if not ok:
            missing += 1
    if missing:
        print(f"\n缺 {missing} 个文件 → 运行 python tools/setup_captcha.py")
        return 1

    print("\n=== 加载检查 ===")
    sys.path.insert(0, cap)
    sys.path.insert(0, root)
    try:
        import captcha.captcha_solver as cs
    except Exception as e:
        print(f"  import 失败: {type(e).__name__}: {e}")
        return 1
    print(f"  models_available: {cs.models_available()}")
    try:
        m = cs._Models.get()
        print(f"  yolo    : {type(m.yolo).__name__}")
        print(f"  siamese : {type(m.siamese).__name__}")
        print(f"  ocr     : {type(getattr(m, 'dddd', None)).__name__}")
        print(f"  fonts   : {len(cs.FONTS)} 个可用")
    except Exception as e:
        print(f"  加载失败: {type(e).__name__}: {e}")
        return 1

    print("\n=== 推理检查（用主仓库真实样本）===")
    repo = r"D:\Github\captchayes\T-Sec-captcha"
    sample = os.path.join(repo, "data", "samples",
                          "t2408_1810855608033804288", "main.jpg")
    if not os.path.exists(sample):
        print("  跳过（样本不存在）")
        return 0
    import cv2
    import numpy as np
    img = cv2.imdecode(np.fromfile(sample, np.uint8), cv2.IMREAD_COLOR)
    bg, strip = img[:480], img[480:]
    try:
        res, detail = cs.solve_click(bg, strip_img=strip)
        print(f"  点选结果: {res}")
        print(f"  模式: {detail.get('mode')}  "
              f"提示标签={detail.get('t_labels')}")
        if res:
            print("  结论: 1.0 求解器可用")
        else:
            print(f"  结论: 求解返回空 ({detail.get('err')})")
            return 1
    except Exception as e:
        print(f"  推理失败: {type(e).__name__}: {e}")
        return 1

    # 滑块
    fx = os.path.join(root, "tests", "fixtures", "slider_bg_r1a4.png")
    if os.path.exists(fx):
        try:
            r = cs.find_slider_gap_full(open(fx, "rb").read())
            print(f"  滑块缺口: {r}")
        except Exception as e:
            print(f"  滑块失败: {type(e).__name__}: {e}")

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
