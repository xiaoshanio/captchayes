# -*- coding: utf-8 -*-
"""drift_fix_check: 验证「重复框」是不是漂移域端到端失败的唯一原因。

诊断结论（eval/drift_diag.py 逐题打印得出）：
    漂移域上检测器的分项指标都很好
        class0 P=0.97 R=0.99
        class2 P=0.96 R=0.96
    匹配器（喂真值框）也有 87.5%
    但端到端只有 3.3%。
    逐题看发现问题：**class2 出现重复框**
        val_000001: gt=3，检出 class2=5（其中 D2[1]/D2[2] 几乎重合，D2[3]/D2[4] 几乎重合）
        val_000002: gt=3，检出 class2=4（D2[1]/D2[2] 重合）
    重复框让贪心指派的矩阵多出列，把正确候选挤掉 -> 顺序错。

本脚本做 A/B：加一次后处理去重（按置信度保留，抑制 IoU>阈值者），看端到端变化。
若去重后大幅回升，就证明"重复框"是主因，且修复点在**推理后处理**（不改模型）。

用法：
    python eval/drift_fix_check.py --limit 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def greedy(slys):
    m = np.array(slys, dtype=float)
    res = []
    for _ in range(min(m.shape[0], m.shape[1])):
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        res.append((int(i), int(j)))
        m[i, :] = -1e9
        m[:, j] = -1e9
    res.sort(key=lambda t: t[0])
    return res


def dedup(items, thr=0.45):
    """items: [(box, conf)]；按 conf 降序保留，抑制 IoU>thr 的重复框。"""
    out = []
    for b, c in sorted(items, key=lambda t: -t[1]):
        if all(iou(b, o[0]) <= thr for o in out):
            out.append((b, c))
    return out


def run(det_path, mat_path, synth, limit, conf, use_dedup, dd_thr=0.45):
    from src.utils import yolo_onnx, ver_onnx
    det = yolo_onnx.YOLO(os.path.join(REPO, det_path))
    det.conf_threshold = conf
    mat = ver_onnx.PreONNX(os.path.join(REPO, mat_path))

    d = os.path.join(REPO, synth)
    pdir = os.path.join(d, "pairs", "val")
    bases = [f[:-5] for f in sorted(os.listdir(pdir))][:limit]

    n = ok = err = n2bad = dup_pairs = 0
    for b in bases:
        ip = os.path.join(d, "images", "val", b + ".jpg")
        img = cv2.imdecode(np.fromfile(ip, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        truth = json.load(open(os.path.join(pdir, b + ".json"), encoding="utf-8"))
        gt = sorted(truth["pairs"], key=lambda q: q["click_order"])
        n += 1
        try:
            dets = det.inference(img)
            r0 = [(x[:4], float(x[4])) for x in dets if int(x[5]) == 0]
            r2 = [(x[:4], float(x[4])) for x in dets if int(x[5]) == 2]
            # 统计原始重复框对数
            raw2 = [z[0] for z in r2]
            for i in range(len(raw2)):
                for j in range(i + 1, len(raw2)):
                    if iou(raw2[i], raw2[j]) > 0.45:
                        dup_pairs += 1
            if use_dedup:
                r0 = dedup(r0, dd_thr)
                r2 = dedup(r2, dd_thr)
            tb = [z[0] for z in r0]
            cb = [z[0] for z in r2]
            cb.sort(key=lambda z: z[0])
            if len(cb) != len(gt):
                n2bad += 1
            if not cb or not tb:
                raise ValueError("nobox")
            ch = [img[int(z[1]):int(z[3]), int(z[0]):int(z[2])] for z in cb]
            tg = [img[int(z[1]):int(z[3]), int(z[0]):int(z[2])] for z in tb]
            if any(x.size == 0 for x in ch + tg):
                raise ValueError("crop")
            res = greedy(mat.reason_all_batch(ch, tg))
            out = [tb[j] for i, j in res]
            if len(out) == len(gt) and all(iou(out[k], gt[k]["target_box"]) > 0.5
                                           for k in range(len(gt))):
                ok += 1
        except Exception:
            err += 1
    return {"n": n, "exact": ok, "exact_rate": round(ok / max(n, 1), 4),
            "err_rate": round(err / max(n, 1), 4),
            "class2_mismatch": round(n2bad / max(n, 1), 4),
            "dup_pairs": dup_pairs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--dd-thr", type=float, default=0.45)
    ap.add_argument("--out", default="out/drift_fix")
    args = ap.parse_args()
    os.makedirs(os.path.join(REPO, args.out), exist_ok=True)

    combos = [
        ("旧模型", "runs/export_nms/best_v3_nms.onnx",
         "runs/export_nms/pre_model_v7_nms.onnx"),
        ("新模型", "runs/drift_rec/export/detB.onnx",
         "runs/drift_rec/export/matB.onnx"),
    ]
    domains = [("旧域A", "data/driftA"), ("漂移域B", "data/driftB")]

    rows = []
    for name, det, mat in combos:
        for dname, dpath in domains:
            for use_dd in (False, True):
                if not (os.path.exists(os.path.join(REPO, det)) and
                        os.path.exists(os.path.join(REPO, mat))):
                    print(f"  [skip] {name} 缺模型")
                    break
                r = run(det, mat, dpath, args.limit, args.conf, use_dd, args.dd_thr)
                r.update({"model": name, "domain": dname,
                          "dedup": use_dd})
                rows.append(r)
                tag = "去重后" if use_dd else "原始  "
                print(f"  {name:6s} × {dname:6s} {tag}  "
                      f"exact={r['exact']:2d}/{r['n']:<2d} ({r['exact_rate']*100:5.1f}%)  "
                      f"class2不符={r['class2_mismatch']*100:3.0f}%  "
                      f"原始重复框对={r['dup_pairs']}")

    print("\n" + "=" * 88)
    print("结论")
    print("=" * 88)
    by = {(r["model"], r["domain"], r["dedup"]): r for r in rows}

    def g(m, d, dd):
        return (by.get((m, d, dd)) or {}).get("exact_rate")

    oldA_raw = g("旧模型", "旧域A", False)
    oldB_raw = g("旧模型", "漂移域B", False)
    newB_raw = g("新模型", "漂移域B", False)
    newB_dd = g("新模型", "漂移域B", True)
    oldB_dd = g("旧模型", "漂移域B", True)

    if None not in (oldA_raw, oldB_raw):
        print(f"  数据老化代价（旧模型 旧域->漂移域）: {(oldB_raw-oldA_raw)*100:+.1f} pt")
    if None not in (oldB_raw, newB_raw):
        print(f"  重训收益（漂移域 旧->新）          : {(newB_raw-oldB_raw)*100:+.1f} pt")
    if None not in (newB_raw, newB_dd):
        print(f"  去重带来的收益（新模型@漂移域）    : {(newB_dd-newB_raw)*100:+.1f} pt")
    if None not in (oldB_raw, oldB_dd):
        print(f"  去重带来的收益（旧模型@漂移域）    : {(oldB_dd-oldB_raw)*100:+.1f} pt")

    print("\n  判读：")
    if None not in (newB_raw, newB_dd) and (newB_dd - newB_raw) > 0.15:
        print("    ✓ 重复框是漂移域端到端失败的主因；修复点在推理后处理，不需要改模型")
    elif None not in (newB_raw, newB_dd):
        print("    · 去重收益有限，瓶颈在别处（检测精度/匹配器）")
    if None not in (oldA_raw, oldB_raw) and (oldB_raw - oldA_raw) < -0.15:
        print("    ✓ 数据老化确实导致显著失效 —— 必须做数据更新")

    json.dump(rows, open(os.path.join(REPO, args.out, "fix_check.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n报告 -> {os.path.join(args.out, 'fix_check.json')}")


if __name__ == "__main__":
    main()
