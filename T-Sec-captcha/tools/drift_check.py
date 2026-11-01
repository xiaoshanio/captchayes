# -*- coding: utf-8 -*-
"""drift_check: 量化「数据/模型老化」—— 把现有 ONNX 模型跑在新域样本上，逐项体检。

回答的问题：
    1. 现有 YOLO 在新域底图上的类别分布是否还正常？（旧域: 4 个 class0 + 4 个 class2）
    2. 置信度分布是否整体下滑（0.9 -> 0.3 这类漂移）？
    3. 现有 Siamese 对新域图标的相似度矩阵还有没有区分度（对角线是否显著）？
    4. 运行时会不会崩（框数对不上、坐标越界）？

用法：
    python tools/drift_check.py --samples data/samples --limit 40
    python tools/drift_check.py --samples data/samples --compose     # 合成成旧域版式后再测
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.captcha import TextSelectCaptcha          # noqa: E402
from src.utils import matchingMode                 # noqa: E402

STRIP_H = 56


def compose(bg, strip):
    h, w = bg.shape[:2]
    s = cv2.resize(strip, (w, STRIP_H), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h + STRIP_H, w, 3), 255, dtype=np.uint8)
    canvas[:h] = bg
    canvas[h:] = s
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--compose", action="store_true", help="拼成旧域版式（底图+提示条）再送入模型")
    ap.add_argument("--conf", type=float, default=0.3, help="YOLO 置信度阈值（默认同 yolo_onnx.py）")
    ap.add_argument("--save-dir", default="out/drift")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    metas = sorted(glob.glob(os.path.join(args.samples, "*", "meta.json")))[:args.limit]
    if not metas:
        raise SystemExit("没有样本")

    print(f"[load] 现有模型 ...")
    cap = TextSelectCaptcha()
    cap.yolo.conf_threshold = args.conf

    rows = []
    for mp in metas:
        d = os.path.dirname(mp)
        sid = os.path.basename(d)
        meta = json.load(open(mp, encoding="utf-8"))
        bg = matchingMode.open_image(os.path.join(d, "bg.jpg"))
        prompt_file = meta["prompts"][0]["file"]
        strip = matchingMode.open_image(os.path.join(d, prompt_file))
        if bg is None or strip is None:
            continue

        img = compose(bg, strip) if args.compose else bg
        # 提示条图标数（真值上界）
        from tools.autolabel import split_prompt_icons  # noqa
        n_prompt = len(split_prompt_icons(strip))

        t0 = time.time()
        det = cap.yolo.inference(img)
        t_yolo = (time.time() - t0) * 1000
        cls_counts = {}
        confs = []
        for it in det:
            cls_counts[int(it[5])] = cls_counts.get(int(it[5]), 0) + 1
            confs.append(float(it[4]))

        # 运行时完整调用
        t0 = time.time()
        try:
            res = cap.run(img)
            err = None
        except Exception as e:
            res, err = None, f"{type(e).__name__}: {e}"
        t_all = (time.time() - t0) * 1000

        rows.append({
            "sid": sid, "n_prompt": n_prompt,
            "det_n": len(det), "cls": cls_counts,
            "conf_mean": round(st.mean(confs), 3) if confs else None,
            "conf_max": round(max(confs), 3) if confs else None,
            "conf_min": round(min(confs), 3) if confs else None,
            "n_target(class0)": cls_counts.get(0, 0),
            "n_char(class2)": cls_counts.get(2, 0),
            "boxes_out": len(res) if res is not None else None,
            "err": err, "t_yolo_ms": round(t_yolo, 1), "t_all_ms": round(t_all, 1),
            "img_wh": [img.shape[1], img.shape[0]],
        })

        # 可视化
        vis = img.copy()
        for it in det:
            x1, y1, x2, y2, cf, cid = it
            col = (0, 0, 255) if cid == 0 else (0, 180, 0) if cid == 2 else (255, 0, 0)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(vis, f"{cid}:{cf:.2f}", (int(x1), max(12, int(y1) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if res:
            for i, b in enumerate(res):
                cv2.putText(vis, f"#{i+1}", (int(b[0]), int(b[3]) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.save_dir, f"{sid}.jpg"), vis)

    # ---------------- 汇总报告
    print("\n" + "=" * 78)
    print("漂移体检报告" + ("（已拼成旧域版式）" if args.compose else "（原始新域底图）"))
    print("=" * 78)
    print(f"样本数: {len(rows)}   置信度阈值: {args.conf}")
    errs = [r for r in rows if r["err"]]
    print(f"运行时异常: {len(errs)}/{len(rows)}")
    for e in errs[:5]:
        print(f"   - {e['sid']}: {e['err']}")

    n0 = [r["n_target(class0)"] for r in rows]
    n2 = [r["n_char(class2)"] for r in rows]
    npr = [r["n_prompt"] for r in rows]
    cm = [r["conf_mean"] for r in rows if r["conf_mean"] is not None]
    cmx = [r["conf_max"] for r in rows if r["conf_max"] is not None]
    bo = [r["boxes_out"] for r in rows if r["boxes_out"] is not None]
    ty = [r["t_yolo_ms"] for r in rows]
    ta = [r["t_all_ms"] for r in rows]

    def stat(name, xs, fmt="{:.2f}"):
        if not xs:
            print(f"  {name:24s}: (无)")
            return
        print(f"  {name:24s}: min={fmt.format(min(xs))} mean={fmt.format(st.mean(xs))} "
              f"max={fmt.format(max(xs))} 中位={fmt.format(st.median(xs))}")

    print("\n-- 检测输出 --")
    stat("class0(label) 框数", n0)
    stat("class2(char) 框数", n2)
    stat("提示图标真值数", npr)
    stat("平均置信度", cm, "{:.3f}")
    stat("最高置信度", cmx, "{:.3f}")
    stat("最终输出点数", bo)
    print("\n-- 耗时 --")
    stat("YOLO ms", ty, "{:.0f}")
    stat("端到端 ms", ta, "{:.0f}")

    def frac(xs, f):
        return f"{sum(1 for x in xs if f(x))}/{len(xs)} ({sum(1 for x in xs if f(x))/max(len(xs),1)*100:.0f}%)"

    print("\n-- 关键失效判据 --")
    print(f"  一个 label 也没检出 (n0==0)      : {frac(n0, lambda x: x == 0)}")
    print(f"  一个 char 也没检出  (n2==0)      : {frac(n2, lambda x: x == 0)}")
    print(f"  label 数 != 提示图标数           : {frac([(a,b) for a,b in zip(n0,npr)], lambda t: t[0] != t[1])}")
    print(f"  平均置信度 < 0.6                 : {frac(cm, lambda x: x < 0.6)}")
    print(f"  最高置信度 < 0.5                 : {frac(cmx, lambda x: x < 0.5)}")
    print(f"  端到端输出点数与提示数不符        : "
          f"{frac(list(zip(bo, npr)), lambda t: t[0] != t[1])}")

    crit = {
        "class0_zero_rate": sum(1 for x in n0 if x == 0) / max(len(n0), 1),
        "class2_zero_rate": sum(1 for x in n2 if x == 0) / max(len(n2), 1),
        "count_mismatch_rate": sum(1 for a, b in zip(n0, npr) if a != b) / max(len(n0), 1),
        "mean_conf_below_0.6": sum(1 for x in cm if x < 0.6) / max(len(cm), 1),
        "runtime_error_rate": len(errs) / max(len(rows), 1),
    }
    print("\n-- 结论指标 --")
    for k, v in crit.items():
        print(f"  {k:24s}: {v*100:.1f}%")

    out = {"samples": len(rows), "compose": args.compose, "conf": args.conf,
           "criteria": crit, "rows": rows}
    rp = os.path.join(args.save_dir, "drift_report.json")
    json.dump(out, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n明细: {rp}")
    print(f"可视化: {args.save_dir}/")


if __name__ == "__main__":
    main()
