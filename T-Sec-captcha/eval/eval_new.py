# -*- coding: utf-8 -*-
"""eval_new: 端到端新旧模型验收 —— 用**未改动的线上 runtime**对比评测。

=============================================================================
设计要点（两个坑，都踩过）
=============================================================================

坑 1：不能用 TSEC_MODEL_DIR 切换模型
    src/captcha.py::_resolve_model_file() 的查找顺序是：
        1) 绝对路径且存在
        2) os.path.exists(filename)     <- src/../model/<file>，仓库里有就命中
        3) TSEC_MODEL_DIR
        4) repo/model, repo/models
    第 2 条优先于第 3 条，所以只要仓库 model/ 里有同名文件，TSEC_MODEL_DIR 永远不生效。
    （实测：设了 TSEC_MODEL_DIR 后 cap.yolo.session._model_path 仍指向 src/../model/）
    → 正确做法是**回灌替换**：把候选模型临时复制进 model/，跑未改动的 runtime，
      跑完恢复。这也正是生产上线的真实动作。

坑 2：ONNX Runtime 会一直持有 .onnx 文件句柄
    同一进程内切换模型时，删模块 + gc.collect() 也不释放，
    覆盖 model/*.onnx 会报 WinError 32。
    → 用**子进程隔离**每个阶段：主进程负责回灌/恢复，子进程只负责跑一遍并落 JSON。

用法：
    python eval/eval_new.py --synth data/selfgen --new-model runs/export_next --limit 60
    python eval/eval_new.py --synth data/selfgen --new-model runs/export_next \
        --baseline model --out out/eval --conf 0.3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "model")
BACKUP_DIR = os.path.join(REPO, "runs", "_model_backup")
KEEP_NAMES = ("best_v3.onnx", "pre_model_v7.onnx")


# ==================================================================== 模型回灌
def snapshot_model_dir():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    files = {}
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".onnx"):
            shutil.copy2(os.path.join(MODEL_DIR, f), os.path.join(BACKUP_DIR, f))
            files[f] = os.path.join(BACKUP_DIR, f)
    # 额外记录基线指纹，便于确认恢复成功
    return files


def restore_model_dir(files):
    for f, b in files.items():
        dst = os.path.join(MODEL_DIR, f)
        for attempt in range(5):
            try:
                shutil.copy2(b, dst)
                break
            except PermissionError:
                time.sleep(0.8)
        else:
            raise RuntimeError(f"无法恢复 {dst}")


def list_models(d):
    out = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".onnx"):
                out[f] = os.path.getsize(os.path.join(d, f))
    return out


def _classify_onnx(path):
    """判断是检测器还是匹配器（靠输入个数/名字）。"""
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        names = [i.name for i in m.graph.input]
        if len(names) >= 2 and "input1" in names:
            return "matcher"
        return "detector"
    except Exception:
        return "unknown"


def apply_model_dir(src_dir):
    """把 src_dir 下的 onnx 按契约名回灌到 model/。返回替换记录。"""
    if os.path.abspath(src_dir) == os.path.abspath(MODEL_DIR):
        return ["(src == model/, 无需回灌)"]
    onnx_files = [f for f in os.listdir(src_dir) if f.endswith(".onnx")]
    if not onnx_files:
        return []
    applied = []
    for target in KEEP_NAMES:
        want = "detector" if target.startswith("best_v3") else "matcher"
        cand = None
        if target in onnx_files:
            cand = target
        else:
            for f in onnx_files:
                if _classify_onnx(os.path.join(src_dir, f)) == want:
                    cand = f
                    break
        if cand:
            shutil.copy2(os.path.join(src_dir, cand), os.path.join(MODEL_DIR, target))
            applied.append(f"{cand} -> {target}")
    return applied


# ==================================================================== 子进程评测
WORKER = r'''
import argparse, json, os, sys, time, statistics as st
import cv2, numpy as np
REPO = sys.argv[1]; out_json = sys.argv[2]; cases_json = sys.argv[3]
conf = None if sys.argv[4] in ("", "None") else float(sys.argv[4])
sys.path.insert(0, REPO)
os.chdir(REPO)
from src.captcha import TextSelectCaptcha

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter == 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/max(ua, 1)

cap = TextSelectCaptcha()
if conf is not None:
    cap.yolo.conf_threshold = conf
loaded = {"yolo": os.path.basename(cap.yolo.session._model_path),
          "pre": os.path.basename(cap.pre.session._model_path)}

cases = json.load(open(cases_json, encoding="utf-8"))
rows = []; n_ok = n_err = n_exact = n_det = n_cnt = 0; times = []
vis_dir = os.path.join(REPO, "out", "eval_vis"); os.makedirs(vis_dir, exist_ok=True)
tag = os.path.basename(out_json).replace("_report.json", "")
for c in cases:
    img = cv2.imdecode(np.fromfile(c["img"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: continue
    t0 = time.time()
    try:
        boxes = cap.run(img); err = None
    except Exception as e:
        boxes, err = None, "%s: %s" % (type(e).__name__, str(e)[:110])
    ms = (time.time()-t0)*1000
    times.append(ms)
    rec = {"id": c["id"], "err": err, "ms": round(ms,1),
           "n_boxes": len(boxes) if boxes else 0}
    if err:
        n_err += 1
    else:
        n_ok += 1
        gt = c.get("truth", {}).get("pairs", [])
        if gt:
            rec["n_gt"] = len(gt)
            if len(boxes) == len(gt): n_cnt += 1
            hits = 0
            for k, q in enumerate(sorted(gt, key=lambda d: d["click_order"])):
                if k < len(boxes) and iou(boxes[k], q["target_box"]) > 0.5:
                    hits += 1
            rec["order_hits"] = hits
            if hits == len(gt): n_exact += 1
            found = sum(1 for q in gt if any(iou(b, q["target_box"]) > 0.5 for b in boxes))
            rec["det_hits"] = found
            n_det += found
    rows.append(rec)
    vis = img.copy()
    if boxes:
        for k, b in enumerate(boxes):
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,0,255), 2)
            cv2.putText(vis, "#%d" % (k+1), (int(b[0]), max(12,int(b[1])-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2, cv2.LINE_AA)
    for q in c.get("truth", {}).get("pairs", []):
        x1,y1,x2,y2 = [int(v) for v in q["target_box"]]
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 1)
    cv2.imwrite(os.path.join(vis_dir, "%s_%s.jpg" % (tag, c["id"])), vis)

n = len(rows)
summ = {"n": n, "ok": n_ok, "err": n_err, "exact": n_exact,
        "count_match": n_cnt, "det_hits": n_det, "loaded": loaded,
        "ms_mean": round(st.mean(times),1) if times else None,
        "ms_p95": round(sorted(times)[int(len(times)*0.95)],1) if times else None}
summ["exact_rate"] = round(n_exact/n, 4) if n else 0
summ["err_rate"] = round(n_err/n, 4) if n else 0
json.dump({"summary": summ, "rows": rows}, open(out_json, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(json.dumps(summ, ensure_ascii=False))
'''


def run_stage(tag, cases_json, out_json, conf, label):
    print(f"\n===== 评测 {tag} =====")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([sys.executable, "-c", WORKER, REPO, out_json, cases_json,
                        "" if conf is None else str(conf)],
                       capture_output=True, text=True, env=env, cwd=REPO, timeout=3600)
    out = (p.stdout or "").strip()
    errout = (p.stderr or "").strip()
    if p.returncode != 0:
        print(f"  [FAIL] rc={p.returncode}")
        print("  " + errout[-800:].replace("\n", "\n  "))
        return None
    try:
        summ = json.loads(out.splitlines()[-1])
    except Exception:
        print("  [FAIL] 无法解析输出:", out[-400:])
        return None
    print(f"  实际加载 yolo={summ['loaded']['yolo']}  pre={summ['loaded']['pre']}")
    print(f"  --> exact={summ['exact']}/{summ['n']} ({summ['exact_rate']*100:.1f}%) "
          f"err={summ['err']} ({summ['err_rate']*100:.1f}%) "
          f"ms_mean={summ['ms_mean']}")
    return summ


# ==================================================================== 主流程
def load_cases(synth_dir, samples_dir, limit):
    cases = []
    if synth_dir and os.path.isdir(os.path.join(synth_dir, "pairs", "val")):
        d = os.path.join(synth_dir, "pairs", "val")
        for fn in sorted(os.listdir(d))[:limit]:
            base = fn[:-5]
            img_p = os.path.join(synth_dir, "images", "val", base + ".jpg")
            if os.path.exists(img_p):
                cases.append({"id": base, "img": img_p,
                              "truth": json.load(open(os.path.join(d, fn), encoding="utf-8"))})
    if not cases and samples_dir:
        import cv2
        import numpy as np
        tmpdir = os.path.join(REPO, "data", "_eval_collected")
        os.makedirs(tmpdir, exist_ok=True)
        for mp in sorted(glob.glob(os.path.join(samples_dir, "*", "meta.json")))[:limit]:
            sd = os.path.dirname(mp)
            meta = json.load(open(mp, encoding="utf-8"))
            bg = cv2.imdecode(np.fromfile(os.path.join(sd, "bg.jpg"), dtype=np.uint8),
                              cv2.IMREAD_COLOR)
            st_img = cv2.imdecode(np.fromfile(
                os.path.join(sd, meta["prompts"][0]["file"]), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bg is None or st_img is None:
                continue
            h, w = bg.shape[:2]
            s = cv2.resize(st_img, (w, 56), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((h + 56, w, 3), 255, np.uint8)
            canvas[:h] = bg
            canvas[h:] = s
            sid = os.path.basename(sd)
            p = os.path.join(tmpdir, sid + ".jpg")
            ok, buf = cv2.imencode(".jpg", canvas)
            buf.tofile(p)
            cases.append({"id": sid, "img": p, "truth": {"pairs": []}})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--synth", default="data/selfgen")
    ap.add_argument("--baseline", default="model")
    ap.add_argument("--new-model", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--out", default="out/eval")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    outdir = os.path.abspath(args.out)

    cases = load_cases(os.path.abspath(args.synth), os.path.abspath(args.samples), args.limit)
    if not cases:
        raise SystemExit("没有测试样例")
    cases_json = os.path.join(outdir, "_cases.json")
    json.dump(cases, open(cases_json, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[cases] {len(cases)} 个")
    print(f"[models] baseline = {list_models(args.baseline)}")
    print(f"[models] new      = {list_models(args.new_model)}")

    backup = snapshot_model_dir()
    print(f"[backup] model/ -> {BACKUP_DIR} ({len(backup)} 个文件)")
    results = {}
    try:
        restore_model_dir(backup)
        results["baseline"] = run_stage("baseline", cases_json,
                                        os.path.join(outdir, "baseline_report.json"),
                                        args.conf, args.baseline)
        restore_model_dir(backup)
        applied = apply_model_dir(os.path.abspath(args.new_model))
        print(f"  [apply] 回灌: {applied}")
        results["new"] = run_stage("new", cases_json,
                                   os.path.join(outdir, "new_report.json"),
                                   args.conf, args.new_model)
    finally:
        restore_model_dir(backup)
        print(f"\n[restore] model/ 已恢复为基线模型")

    # ---------------- 结论
    print("\n" + "=" * 78)
    print("对比结论")
    print("=" * 78)
    b, nn = results.get("baseline"), results.get("new")
    if b and nn:
        print(f"{'指标':<16}{'baseline':>12}{'new':>12}{'变化':>12}")
        for k in ("exact_rate", "err_rate", "count_match", "det_hits", "ms_mean"):
            vb, vn = b.get(k), nn.get(k)
            if vb is None or vn is None:
                continue
            print(f"{k:<16}{vb:>12}{vn:>12}{vn-vb:>+12.4f}")
        if nn["exact_rate"] > b["exact_rate"]:
            print(f"\n>>> 新模型更好：端到端准确率 "
                  f"{(nn['exact_rate']-b['exact_rate'])*100:+.1f} 个百分点")
        elif nn["exact_rate"] == b["exact_rate"]:
            print("\n>>> 端到端准确率持平（对照其余指标与耗时）")
        else:
            print("\n>>> 新模型更低，检查训练数据/超参/类别契约")
        json.dump({"baseline": b, "new": nn}, open(os.path.join(outdir, "compare.json"),
                  "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    elif nn:
        print("new:", json.dumps(nn, ensure_ascii=False, indent=2))
    print(f"\n可视化 -> {outdir}/ 与 out/eval_vis/（baseline_* / new_* 可逐题对照）")


if __name__ == "__main__":
    main()
