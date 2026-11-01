# -*- coding: utf-8 -*-
"""check_legacy: 验证旧域（中文文字点选 2404）模型是否可用。

为什么需要：
    实测 cun.ai 的验证码题型是**混合下发**的（sign-in 15 次抽样）：
        2401 滑块        7
        2408 图标点选    6
        2404 中文文字点选 2   ← instruction = "请依次点击：秤 爱 驰"
    我们重训的是 2408 域模型，在 2404 上不可用（不同域）。
    但仓库里保留着旧域模型（runs/_model_backup/），它正是中文文字点选。

    本脚本用旧域模型跑旧域样本 docs/res.jpg，确认它还能用，
    这样「混合题型」时可以用**双模型路由**：2408 走新模型，2404 走旧模型。

用法：
    python tools/check_legacy.py
"""
from __future__ import annotations

import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main():
    legacy_dir = os.path.join(REPO, "runs", "legacy")
    os.makedirs(legacy_dir, exist_ok=True)
    for src, dst in (
        (os.path.join(REPO, "runs", "_model_backup", "best_v3.onnx.bak"),
         os.path.join(legacy_dir, "best_v3.onnx")),
        (os.path.join(REPO, "runs", "_model_backup", "pre_model_v7.onnx.bak"),
         os.path.join(legacy_dir, "pre_model_v7.onnx")),
    ):
        if not os.path.exists(src):
            raise SystemExit(f"缺备份模型: {src}")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
    print(f"[legacy] 模型目录就绪: {legacy_dir}")

    img = os.path.join(REPO, "docs", "res.jpg")
    if not os.path.exists(img):
        raise SystemExit(f"缺样本: {img}")
    im = cv2.imdecode(np.fromfile(img, np.uint8), cv2.IMREAD_COLOR)
    print(f"[legacy] 样本 {os.path.basename(img)} 尺寸 {im.shape[1]}x{im.shape[0]}")

    # 用 TSEC_MODEL_DIR 指向 legacy 目录
    # 注意：_resolve_model_file 里 os.path.exists(src/../model/<file>) 优先级更高，
    # 所以必须**直接构造 YOLO/PreONNX**，绕过 TextSelectCaptcha 的路径解析。
    from src.utils import yolo_onnx, ver_onnx, matchingMode

    y = yolo_onnx.YOLO(os.path.join(legacy_dir, "best_v3.onnx"), conf_threshold=0.3)
    pre = ver_onnx.PreONNX(os.path.join(legacy_dir, "pre_model_v7.onnx"))

    det = y.inference(im)
    tb = [x[:4] for x in det if x[5] == 0]
    cb = [x[:4] for x in det if x[5] == 2]
    cb.sort(key=lambda b: b[0])
    print(f"[legacy] class0(图面字)={len(tb)}  class2(提示条字)={len(cb)}")
    if not tb or not cb:
        print("[legacy] FAIL 检测不到足够框，旧模型可能已损坏")
        return 1

    its = [im[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in tb]
    ics = [im[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in cb]
    slys = pre.reason_all_batch(ics, its)
    order = matchingMode.find_overall_index_fast(slys)
    res = [tb[j] for _i, j in order]

    print(f"[legacy] 点击顺序 ({len(res)} 点):")
    for i, (x1, y1, x2, y2) in enumerate(res):
        print(f"    #{i+1} 中心 ({(x1+x2)//2}, {(y1+y2)//2})")

    vis = im.copy()
    for i, (x1, y1, x2, y2) in enumerate(res):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, str(i + 1), (x1 + 2, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    out_dir = os.path.join(REPO, "out", "legacy")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "legacy_res.jpg")
    cv2.imencode(".jpg", vis)[1].tofile(p)
    print(f"[legacy] 可视化 -> {os.path.relpath(p, REPO)}")
    print("[legacy] PASS 旧域模型可用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
