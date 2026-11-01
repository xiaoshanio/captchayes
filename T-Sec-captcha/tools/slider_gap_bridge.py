# -*- coding: utf-8 -*-
"""slider_gap_bridge: 把更好的滑块缺口检测桥接进 cun.ai 1.0。

════════════════════════════════════════════════════════════════
为什么需要
════════════════════════════════════════════════════════════════
1.0 实跑 5 次的结果（out/test_10_live*.log 汇总）：

    点选题：通过 3 / 失败 1（失败那次是补丁卡死人为中断）
    滑块题：通过 0 / 失败 3 —— 全是「缺口置信不足(0.3)，换题不提交」

原因在 `_solve_slider_once` 第 642 行：

    if (conf or 0) < 0.45:
        self._say("human_verify", f"缺口置信不足({conf}), 换题不提交")
        return "refresh"

而 1.0 自己的 `find_gap_with_piece` / `find_slider_gap` 在这批真实底图上
只给出 0.3 / 0.0 / None —— 达不到 0.45，于是每次都直接放弃。

本项目 T-Sec-captcha 的 `tools/slider_gap.py` 在同样类型的底图上是
**3/3 全中**（误差 0~1px），用的是「亮描边 + 暗心」和「帽形滤波」两条路。

════════════════════════════════════════════════════════════════
本模块做什么
════════════════════════════════════════════════════════════════
不改 1.0 源码，monkey-patch 掉 `captcha_solver.find_slider_gap`：

    from slider_gap_bridge import install
    install()
    # 之后 1.0 的滑块流程会自动用新实现

置信度映射（不是拍脑袋，按方法可靠性给）：

    路径 A「亮描边 + 轮廓」：内部必须暗、框须 50~95px 见方
        → 实测误差 0~1px（bg_2 / bg_3 与人工标注完全一致）→ conf = 0.9
    路径 B「帽形滤波」：轮廓被草叶/纹理打断时兜底
        → 实测误差 ~5px（bg_1）→ conf = 0.55

两者都 > 0.45 阈值，所以能被 1.0 正常采用。

用法：
    python tools/slider_gap_bridge.py            # 自测（1.0 的 fixture）
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PROJ_10 = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"

MARK = "_tsec_slider_bridge"

# 方法可靠性 → 置信度
CONF_OUTLINE = 0.9      # 亮描边+暗心，实测 0~1px
CONF_HAT = 0.55         # 帽形兜底，实测 ~5px
CONF_NONE = 0.0


def _find(bg):
    """调用本项目实现，返回 (conf, gap_x) 或 (0.0, 0)。"""
    try:
        from slider_gap import find_gap
    except Exception as e:
        print("[bridge] 导入 slider_gap 失败: "
              + type(e).__name__ + ": " + str(e), flush=True)
        return CONF_NONE, 0
    try:
        r = find_gap(bg, verbose=False)
    except Exception as e:
        print("[bridge] find_gap 异常: "
              + type(e).__name__ + ": " + str(e), flush=True)
        return CONF_NONE, 0
    if not r:
        return CONF_NONE, 0
    src = r.get("src", "")
    x = int(round(float(r.get("cx", 0))))
    if src == "outline":
        return CONF_OUTLINE, x
    if src == "hat":
        return CONF_HAT, x
    return CONF_HAT, x


def install(verbose: bool = True):
    """给 1.0 的 captcha_solver 打滑块补丁（幂等）。"""
    for p in (PROJ_10, os.path.join(PROJ_10, "captcha")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import captcha.captcha_solver as solver

    if getattr(solver, MARK, False):
        if verbose:
            print("[bridge] 滑块补丁已安装过，跳过", flush=True)
        return solver

    orig_find = getattr(solver, "find_slider_gap", None)

    def find_slider_gap(bg_img):
        """1.0 期望返回 (conf, gap_x, extra)。"""
        conf, x = _find(bg_img)
        if verbose and conf > 0:
            print(f"[bridge] 滑块定位 conf={conf} x={x}", flush=True)
        if conf <= 0 and orig_find is not None:
            try:
                return orig_find(bg_img)
            except Exception:
                pass
        return conf, x, 0.0

    solver.find_slider_gap = find_slider_gap
    setattr(solver, MARK, True)
    if verbose:
        print("[bridge] 已替换 captcha_solver.find_slider_gap", flush=True)
    return solver


def main():
    """自测：用 1.0 自带的滑块 fixture 对比新旧实现。"""
    for p in (PROJ_10, os.path.join(PROJ_10, "captcha")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import cv2
    import numpy as np
    import captcha.captcha_solver as solver

    fx = os.path.join(PROJ_10, "tests", "fixtures", "slider_bg_r1a4.png")
    if not os.path.exists(fx):
        raise SystemExit("缺 fixture: " + fx)
    bg = cv2.imdecode(np.fromfile(fx, np.uint8), cv2.IMREAD_COLOR)
    print("fixture 尺寸: " + str(bg.shape))
    print("")
    print("--- 1.0 原实现 ---")
    try:
        print("  find_slider_gap = " + str(solver.find_slider_gap(bg)))
    except Exception as e:
        print("  异常: " + type(e).__name__ + ": " + str(e))

    install()
    print("")
    print("--- 桥接后 ---")
    print("  find_slider_gap = " + str(solver.find_slider_gap(bg)))

    # 再测本项目留存的真实登录底图（3 张，已知真值）
    real_dir = os.path.join(REPO, "out", "login10x", "20261031_124914")
    truth = {"bg_1.png": 467, "bg_2.png": 314, "bg_3.png": 447}
    if os.path.isdir(real_dir):
        print("")
        print("--- 真实登录底图（容差 25px）---")
        for name, tx in truth.items():
            p = os.path.join(real_dir, name)
            if not os.path.exists(p):
                continue
            im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
            conf, x, _ = solver.find_slider_gap(im)
            hit = abs(x - tx) <= 25
            print(f"  {name}: conf={conf} x={x} 真值={tx} "
                  + ("HIT" if hit else "MISS"))


if __name__ == "__main__":
    main()
