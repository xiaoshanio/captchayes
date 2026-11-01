# -*- coding: utf-8 -*-
"""eval_2408: 真实 2408 样本上的端到端验收（回灌式，子进程隔离）。

为什么必须回灌而不是用环境变量：
    src/captcha.py::_resolve_model_file() 的查找顺序里，
    `os.path.exists(filename)`（即 src/../model/<file>）**优先级高于**
    TSEC_MODEL_DIR。只要 model/ 里有同名文件，环境变量永远不生效
    （docs/TRAINING_AND_DATA_REFRESH.md 5.0 坑 1 已记录）。
    所以唯一可靠的验收方式是：临时把候选模型复制进 model/，跑未改动的 runtime，
    跑完恢复。这也正是生产上线的真实动作。

为什么子进程隔离：
    ONNX Runtime 会一直持有 .onnx 文件句柄，同进程内删 sys.modules + gc
    也不释放，覆盖 model/*.onnx 会 PermissionError: [WinError 32]。
    主进程负责回灌/恢复，子进程只负责跑一遍并落 JSON。

指标（对齐 runtime 的真实关心点）：
    exact_rate   整题点击顺序全对（按顺序逐个 IoU>0.5）
    point_hit    平均点中率（忽略顺序，看检测+匹配的整体能力）
    count_match  框数 == 提示图标数
    err_rate     运行时异常率
    ms_mean/p95  耗时

用法：
    python tools/eval_2408.py --samples data/samples --new-model runs/export_t2408
    python tools/eval_2408.py --samples data/samples --new-model runs/export_t2408 --limit 60
    python tools/eval_2408.py --baseline-only --limit 60
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

TZ = timezone(timedelta(hours=8))
MODEL_DIR = os.path.join(REPO, "model")
BACKUP_DIR = os.path.join(REPO, "runs", "_model_backup")

# 候选模型文件名 -> 生产文件名
# 按顺序尝试；候选目录里有任意一个同族文件就回灌
MAP = {
    "best_v3.onnx": ["best_v3_2408b.onnx", "best_v3_2408.onnx",
                     "best_v3_next.onnx", "best_v3_nms.onnx"],
    "pre_model_v7.onnx": ["pre_model_v7_2408b.onnx", "pre_model_v7_2408.onnx",
                          "pre_model_v7_next.onnx", "pre_model_v7_nms.onnx"],
}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def build_gt(dirs, labeled="data/labeled2408"):
    """真值：每题的点击顺序框（来自标注的 pairs + bg_boxes）。

    注意 labeled 的选取会**直接影响验收数字**：
        data/labeled2408     v1 分割器标注，自洽率 87.5%（12.5% 错标）
        data/labeled2408v2   彩色兜底标注，自洽率 85.5%
        data/labeled_det     检测器自举标注，自洽率 99.0%  ← 推荐用它评
    用脏真值评会把模型的正确行为判成错误，所以默认允许切换。
    """
    gt = {}
    index = os.path.join(REPO, labeled, "labels_index.jsonl")
    if not os.path.exists(index):
        return gt
    for line in open(index, encoding="utf-8"):
        r = json.loads(line)
        boxes = r.get("bg_boxes") or []
        pairs = r.get("pairs") or []
        order = []
        for p in sorted(pairs, key=lambda x: x["prompt"]):
            j = p["bg"]
            if 0 <= j < len(boxes):
                order.append(boxes[j])
        if order:
            gt[r["sid"]] = order
    return gt


def run_child(samples, limit, conf, out_json, dedup=0.0):
    """子进程：用当前 model/ 里的模型跑一遍，结果落 JSON。

    注意 conf：runtime 的 YOLO 类构造参数默认 0.3，但实测 2408 上
    class2（提示图标）的置信度分布偏低（大量落在 0.30~0.55），
    conf=0.3 会把真框滤掉 → corp 只有 1 个 → 配对失败。
    这里通过构造参数传入，不改 runtime 源码。

    注意 dedup：导出的 NMS 是**跨类**的（ultralytics nms=True），
    同一类内部的近重合框仍会残留（实测 60 题 335 个 class0 框里 10 对重复）。
    重复框会占掉贪心指派的列，破坏点击顺序 —— 这是端到端 65% 的主要损失来源之一。
    dedup>0 时在子进程里按 IoU 对同类框做一次 NMS，等价于
    docs/TRAINING_AND_DATA_REFRESH.md 5.4 里记录的「导出侧装 NMS」的推理端替代。
    """
    code = f"""
import json, os, sys, time
sys.path.insert(0, {REPO!r})
from src.captcha import TextSelectCaptcha

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1)

DEDUP = {dedup}

dirs = sorted(d for d in os.listdir({samples!r})
              if os.path.isdir(os.path.join({samples!r}, d))
              and os.path.exists(os.path.join({samples!r}, d, "meta.json")))
if {limit} > 0:
    dirs = dirs[:{limit}]

cap = TextSelectCaptcha()
cap.yolo.conf_threshold = {conf}
res = []
for i, sid in enumerate(dirs):
    p = os.path.join({samples!r}, sid, "main.jpg")
    if not os.path.exists(p):
        continue
    t0 = time.time()
    try:
        if DEDUP > 0:
            # 先拿到原始检测，去重后再走与 run() 完全相同的后处理
            # 注意：必须同样调用 _filter_char_boxes —— 否则这条分支会
            # 绕过 runtime 的"提示条区过滤"，导致评测结果比真实上线低 9 pt。
            from src.utils import matchingMode
            from src.captcha import _filter_char_boxes
            img = matchingMode.open_image(p)
            det = cap.yolo.inference(img)
            keep = []
            for d in sorted(det, key=lambda z: -z[4]):
                if all(d[5] != k[5] or iou(d[:4], k[:4]) <= DEDUP for k in keep):
                    keep.append(d)
            tb = [x[:4] for x in keep if x[5] == 0]
            cb = [x[:4] for x in keep if x[5] == 2]
            cb = _filter_char_boxes(cb, img.shape[0])
            cb.sort(key=lambda b: b[0])
            its = [img[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in tb]
            ics = [img[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in cb]
            slys = cap.pre.reason_all_batch(ics, its)
            order = matchingMode.find_overall_index_fast(slys)
            result = [tb[j] for _i, j in order]
            h, w, _ = img.shape
            d = {{"imgW": w, "imgH": h,
                 "point": [{{"x_rel": (x1+x2)/2, "y_rel": (y1+y2)/2}}
                           for x1, y1, x2, y2 in result],
                 "corp": [{{"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
                          for x1, y1, x2, y2 in result]}}
        else:
            d = cap.run_dict(p)
        ms = (time.time() - t0) * 1000
        res.append({{"sid": sid, "ok": True, "ms": round(ms, 1),
                    "imgW": d["imgW"], "imgH": d["imgH"],
                    "point": d["point"], "corp": d["corp"]}})
    except Exception as e:
        res.append({{"sid": sid, "ok": False,
                    "err": f"{{type(e).__name__}}: {{e}}"}})
    if (i + 1) % 20 == 0:
        print(f"  [child] {{i+1}}/{{len(dirs)}}", flush=True)
json.dump(res, open({out_json!r}, "w", encoding="utf-8"), ensure_ascii=False)
print(f"[child] 完成 {{len(res)}} 题 (conf={{cap.yolo.conf_threshold}} dedup={{DEDUP}})",
      flush=True)
"""
    r = subprocess.run([sys.executable, "-u", "-c", code],
                       cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.stdout:
        print(r.stdout.strip()[-800:])
    if r.returncode != 0:
        print("[child] STDERR:", (r.stderr or "")[-1500:])
    return r.returncode == 0


def score(records, gt):
    """算指标。"""
    n = len(records)
    err = sum(1 for r in records if not r.get("ok"))
    exact = 0
    hits = 0
    total_pts = 0
    count_match = 0
    ms = [r["ms"] for r in records if r.get("ok")]
    detail = []

    for r in records:
        if not r.get("ok"):
            detail.append({"sid": r["sid"], "ok": False, "err": r.get("err")})
            continue
        g = gt.get(r["sid"])
        corps = [[c["x1"], c["y1"], c["x2"], c["y2"]] for c in r.get("corp") or []]
        if not g:
            detail.append({"sid": r["sid"], "ok": True, "n_pred": len(corps),
                           "note": "无真值，仅统计"})
            continue
        total_pts += len(g)
        hit = 0
        for k in range(min(len(g), len(corps))):
            if iou(g[k], corps[k]) > 0.5:
                hit += 1
        hits += hit
        all_ok = (len(corps) == len(g) and hit == len(g))
        if len(corps) == len(g):
            count_match += 1
        if all_ok:
            exact += 1
        detail.append({"sid": r["sid"], "ok": True, "n_pred": len(corps),
                       "n_gt": len(g), "hit": hit, "exact": all_ok})

    ms_sorted = sorted(ms)
    return {
        "n": n,
        "n_scored": sum(1 for d in detail if d.get("n_gt")),
        "exact": exact,
        "exact_rate": round(exact / max(n, 1), 4),
        "point_hit": round(hits / max(total_pts, 1), 4),
        "count_match": count_match,
        "err": err,
        "err_rate": round(err / max(n, 1), 4),
        "ms_mean": round(sum(ms) / max(len(ms), 1), 1) if ms else 0,
        "ms_p95": round(ms_sorted[int(len(ms_sorted) * 0.95)] if ms_sorted else 0, 1),
        "detail": detail,
    }


def swap_in(new_dir):
    """把候选模型复制进 model/，返回备份清单。"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    done = []
    for prod, cands in MAP.items():
        src = None
        for c in cands:
            p = os.path.join(new_dir, c)
            if os.path.exists(p):
                src = p
                break
        if src is None:
            print(f"  [warn] 候选目录里找不到 {prod} 的替代（试过 {cands}）")
            continue
        dst = os.path.join(MODEL_DIR, prod)
        bak = os.path.join(BACKUP_DIR, prod + ".bak")
        if os.path.exists(dst) and not os.path.exists(bak):
            shutil.copy2(dst, bak)
        shutil.copy2(src, dst)
        print(f"  [swap] {os.path.basename(src)} -> model/{prod}")
        done.append((dst, bak))
    return done


def swap_back(done):
    for dst, bak in done:
        if os.path.exists(bak):
            shutil.copy2(bak, dst)
            print(f"  [restore] model/{os.path.basename(dst)} <- backup")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--new-model", default="runs/export_t2408")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="runtime 默认 0.3；2408 实测 0.25 更优（提示图标小）")
    ap.add_argument("--dedup", type=float, default=0.0,
                    help="同类框 IoU 去重阈值，0=关闭。实测 0.5 可把 60 题里 10 对"
                         "重复框清零，显著提升 exact_rate")
    ap.add_argument("--out", default="out/eval2408")
    ap.add_argument("--labeled", default="data/labeled_det",
                    help="真值来源。labeled_det=检测器自举（自洽 99.0%%，推荐）；"
                         "labeled2408=v1（自洽 87.5%%）；labeled2408v2（85.5%%）")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    samples = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    new_dir = args.new_model if os.path.isabs(args.new_model) else os.path.join(REPO, args.new_model)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    gt = build_gt(None, args.labeled)
    print(f"[gt] 真值题数: {len(gt)}")

    report = {"samples": samples, "new_model": new_dir, "limit": args.limit,
              "conf": args.conf, "dedup": args.dedup,
              "started": datetime.now(TZ).isoformat(timespec="seconds")}

    # ---------- baseline
    if not args.skip_baseline:
        print("\n=== baseline（现有 model/）===")
        bj = os.path.join(out, "baseline.json")
        ok = run_child(samples, args.limit, args.conf, bj, args.dedup)
        if ok and os.path.exists(bj):
            recs = json.load(open(bj, encoding="utf-8"))
            report["baseline"] = score(recs, gt)
            b = report["baseline"]
            print(f"  exact={b['exact_rate']:.1%} point_hit={b['point_hit']:.1%} "
                  f"count_match={b['count_match']} err={b['err_rate']:.1%} "
                  f"ms_mean={b['ms_mean']}")

    if args.baseline_only:
        with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return

    # ---------- 回灌 + 评测 + 恢复
    print("\n=== 回灌候选模型并评测 ===")
    done = swap_in(new_dir)
    try:
        nj = os.path.join(out, "new.json")
        ok = run_child(samples, args.limit, args.conf, nj, args.dedup)
        if ok and os.path.exists(nj):
            recs = json.load(open(nj, encoding="utf-8"))
            report["new"] = score(recs, gt)
            b = report["new"]
            print(f"  exact={b['exact_rate']:.1%} point_hit={b['point_hit']:.1%} "
                  f"count_match={b['count_match']} err={b['err_rate']:.1%} "
                  f"ms_mean={b['ms_mean']}")
    finally:
        swap_back(done)

    # ---------- 对比
    if "baseline" in report and "new" in report:
        bl, nw = report["baseline"], report["new"]
        cmp_ = {}
        for k in ("exact_rate", "point_hit", "err_rate", "ms_mean", "ms_p95"):
            cmp_[k] = {"baseline": bl[k], "new": nw[k],
                       "delta": round(nw[k] - bl[k], 4)}
        report["compare"] = cmp_
        print("\n=== 对比 ===")
        for k, v in cmp_.items():
            print(f"  {k:12s} {v['baseline']:>8} -> {v['new']:>8}  ({v['delta']:+.4f})")

    report["finished"] = datetime.now(TZ).isoformat(timespec="seconds")
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告 -> {os.path.join(out, 'report.json')}")


if __name__ == "__main__":
    main()


