# -*- coding: utf-8 -*-
"""drift_recover: 「数据老化 → 重训恢复」的**低成本因果实验**。

为什么不用 eval/drift_experiment.py：
    那个脚本要训 6 个模型（旧/新/多训/混合 × 检测+匹配），本机 CPU 上要 5+ 小时，
    而且如果训练轮数不够，模型根本没收敛（实测 6 epoch/300 张 -> 全部指标为 0），
    实验就白跑了。

本脚本的思路：**复用已经训好的模型当"旧模型"，只训一个"新模型"**。
    旧模型 = runs/export_nms（上一轮在 v1 域上训了 19 epoch / 800 张，是可信的基线）
    新模型 = 在 v2 漂移域上现训一个
然后跑 2×2 矩阵：

              域A(旧域)        域B(漂移域)
    旧模型     ① 基线           ② 老化代价      <- 核心：同模型换域掉多少
    新模型     ④ 遗忘检查        ③ 恢复效果      <- 核心：换数据能不能救回来

四个格子就能回答全部决策问题，而且只需训 1 个模型（约 40 分钟）。

用法：
    python eval/drift_recover.py --train-new          # 训新模型 + 跑 2x2 矩阵
    python eval/drift_recover.py --matrix-only        # 只跑矩阵（新模型已存在）
    python eval/drift_recover.py --epochs 8 --n-train 300
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
PY = sys.executable

OLD_DET = "runs/export_nms/best_v3_nms.onnx"
OLD_MAT = "runs/export_nms/pre_model_v7_nms.onnx"
NEW_DET = "runs/drift_rec/export/detB.onnx"
NEW_MAT = "runs/drift_rec/export/matB.onnx"


def sh(cmd, log=None, timeout=7200):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["POLARS_SKIP_CPU_CHECK"] = "1"
    p = subprocess.run(cmd, capture_output=True, env=env, cwd=REPO, timeout=timeout)
    out = (p.stdout or b"").decode("utf-8", "replace") + \
          (p.stderr or b"").decode("utf-8", "replace")
    if log:
        os.makedirs(os.path.join(REPO, "out"), exist_ok=True)
        with open(os.path.join(REPO, log), "w", encoding="utf-8") as f:
            f.write(out)
    return p.returncode, out


def step(msg):
    print(f"\n{'='*90}\n{msg}\n{'='*90}", flush=True)


# ------------------------------------------------------------------ 评测
EVAL_CODE = r'''
import json, os, sys, time, statistics as st
import cv2, numpy as np
REPO, det_p, mat_p, synth, limit, conf, tag, outj = sys.argv[1:9]
limit = int(limit); conf = float(conf)
sys.path.insert(0, REPO); os.chdir(REPO)
from src.utils import yolo_onnx, ver_onnx

def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    if inter==0: return 0.0
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/max(ua,1)

def greedy(slys):
    m=np.array(slys,float); res=[]
    for _ in range(min(m.shape[0],m.shape[1])):
        i,j=np.unravel_index(int(np.argmax(m)),m.shape); res.append((int(i),int(j)))
        m[i,:]=-1e9; m[:,j]=-1e9
    res.sort(key=lambda t:t[0]); return res

det=yolo_onnx.YOLO(det_p); det.conf_threshold=conf
mat=ver_onnx.PreONNX(mat_p)
d=os.path.join(REPO,synth); pdir=os.path.join(d,'pairs','val')
if not os.path.isdir(pdir):
    json.dump({'tag':tag,'n':0,'exact':0,'exact_rate':0,'err_rate':1,'note':'no val set'},
              open(outj,'w',encoding='utf-8'),ensure_ascii=False)
    print(json.dumps({'tag':tag,'n':0,'exact':0,'exact_rate':0,'err_rate':1},ensure_ascii=False)); raise SystemExit
bases=[f[:-5] for f in sorted(os.listdir(pdir))][:limit]
n=ok=err=n2bad=0; times=[]
for b in bases:
    ip=os.path.join(d,'images','val',b+'.jpg'); tp=os.path.join(pdir,b+'.json')
    if not (os.path.exists(ip) and os.path.exists(tp)): continue
    img=cv2.imdecode(np.fromfile(ip,dtype=np.uint8),cv2.IMREAD_COLOR)
    if img is None: continue
    truth=json.load(open(tp,encoding='utf-8'))
    gt=sorted(truth['pairs'],key=lambda q:q['click_order']); n+=1
    t0=time.time()
    try:
        dets=det.inference(img)
        tb=[x[:4] for x in dets if int(x[5])==0]; cb=[x[:4] for x in dets if int(x[5])==2]
        cb.sort(key=lambda z:z[0])
        if len(cb)!=len(gt): n2bad+=1
        if not cb or not tb: raise ValueError('nobox')
        chars=[img[int(z[1]):int(z[3]),int(z[0]):int(z[2])] for z in cb]
        tg=[img[int(z[1]):int(z[3]),int(z[0]):int(z[2])] for z in tb]
        if any(x.size==0 for x in chars+tg): raise ValueError('crop')
        res=greedy(mat.reason_all_batch(chars,tg))
        out=[tb[j] for i,j in res]
        if len(out)==len(gt) and all(iou(out[k],gt[k]['target_box'])>0.5 for k in range(len(gt))): ok+=1
    except Exception:
        err+=1
    times.append((time.time()-t0)*1000)
r={'tag':tag,'n':n,'exact':ok,'exact_rate':round(ok/max(n,1),4),
   'err_rate':round(err/max(n,1),4),'class2_mismatch':round(n2bad/max(n,1),4),
   'ms_mean':round(st.mean(times),1) if times else None}
json.dump(r, open(outj,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps(r,ensure_ascii=False))
'''


def eval_cell(tag, det, mat, synth, limit, conf, outdir):
    os.makedirs(os.path.join(REPO, outdir), exist_ok=True)
    outj = os.path.join(REPO, outdir, f"{tag}.json")
    rc, out = sh([PY, "-c", EVAL_CODE, REPO, det, mat, synth,
                  str(limit), str(conf), tag, outj])
    if rc != 0:
        print(f"  [{tag}] FAIL\n{out[-800:]}")
        return None
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        print(f"  [{tag}] 解析失败: {out[-300:]}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-new", action="store_true", help="训练漂移域新模型")
    ap.add_argument("--matrix-only", action="store_true", help="只跑 2x2 矩阵")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--mat-epochs", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=80)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--out", default="out/drift_rec")
    args = ap.parse_args()
    os.makedirs(os.path.join(REPO, args.out), exist_ok=True)
    t0 = time.time()

    A, B = "data/driftA", "data/driftB"

    # ---------------- 前置：两个域的数据 + 域差异
    if not (os.path.isdir(os.path.join(REPO, A, "pairs", "val")) and
            os.path.isdir(os.path.join(REPO, B, "pairs", "val"))):
        step("准备两个域的数据")
        for out, style, seed in ((A, "v1", 101), (B, "v2", 202)):
            rc, o = sh([PY, "tools/synth_selfcontained.py", "--out", out,
                        "--n-train", str(args.n_train), "--n-val", str(args.n_val),
                        "--seed", str(seed), "--style", style],
                       log=f"out/drift_rec_gen_{style}.log")
            if rc != 0:
                print(o[-1200:]); raise SystemExit("生成失败")
            print(f"  [gen] {out} style={style}")

    step("域差异量化（证明漂移成立）")
    rc, gap_out = sh([PY, "eval/domain_gap.py", "--a", A, "--b", B,
                      "--split", "val", "--out", os.path.join(args.out, "gap")])
    print(gap_out[-1400:] if rc == 0 else gap_out[-600:])
    gap = {}
    gp = os.path.join(REPO, args.out, "gap", "gap.json")
    if os.path.exists(gp):
        gap = json.load(open(gp, encoding="utf-8"))

    # ---------------- 训练新模型（漂移域）
    if args.train_new and not args.matrix_only:
        step(f"训练漂移域新模型（{args.epochs} epoch / {args.n_train} 张）")
        rc, o = sh([PY, "train/train_detector.py", "--data", f"{B}/data.yaml",
                    "--epochs", str(args.epochs), "--imgsz", "640", "--batch", "8",
                    "--workers", "4", "--device", "cpu", "--cache", "ram",
                    "--project", "runs/drift_rec/detect", "--name", "detB"],
                   log="out/drift_rec_det.log")
        if rc != 0:
            print(o[-1500:]); raise SystemExit("检测器训练失败")
        pt = os.path.join(REPO, "runs/drift_rec/detect/detB/weights/best.pt")
        code = (
            "import os,sys,shutil\n"
            "os.environ['POLARS_SKIP_CPU_CHECK']='1'\n"
            "sys.path.insert(0,r'%s')\n"
            "from ultralytics import YOLO\n"
            "m=YOLO(r'%s')\n"
            "o=m.export(format='onnx',imgsz=640,batch=1,dynamic=False,simplify=True,opset=14,nms=True)\n"
            "os.makedirs(os.path.dirname(r'%s'),exist_ok=True)\n"
            "shutil.copyfile(o, r'%s')\n"
            "print('OK')\n" % (REPO, pt, os.path.join(REPO, NEW_DET), os.path.join(REPO, NEW_DET))
        )
        rc, o = sh([PY, "-c", code], log="out/drift_rec_export.log")
        if rc != 0:
            print(o[-1200:]); raise SystemExit("导出失败")
        print(f"  [export] {NEW_DET}")

        rc, o = sh([PY, "train/prepare_pairs.py", "--synth", B,
                    "--out", "data/driftB_pairs"], log="out/drift_rec_pairs.log")
        if rc != 0:
            print(o[-1000:]); raise SystemExit("配对生成失败")
        rc, o = sh([PY, "train/train_matcher.py", "--pairs", "data/driftB_pairs",
                    "--epochs", str(args.mat_epochs), "--batch", "64", "--workers", "0",
                    "--threads", "10", "--project", "runs/drift_rec/matcher", "--name", "matB",
                    "--out-onnx", NEW_MAT], log="out/drift_rec_mat.log")
        if rc != 0:
            print(o[-1500:]); raise SystemExit("匹配器训练失败")
        print(f"  [export] {NEW_MAT}")

    # ---------------- 2x2 矩阵
    step("2×2 矩阵评测")
    for p in (OLD_DET, OLD_MAT, NEW_DET, NEW_MAT):
        if not os.path.exists(os.path.join(REPO, p)):
            raise SystemExit(f"缺模型: {p}（先跑 --train-new）")

    cells = [
        ("old_on_A", "旧模型 @ 旧域",   OLD_DET, OLD_MAT, A),
        ("old_on_B", "旧模型 @ 漂移域", OLD_DET, OLD_MAT, B),
        ("new_on_B", "新模型 @ 漂移域", NEW_DET, NEW_MAT, B),
        ("new_on_A", "新模型 @ 旧域",   NEW_DET, NEW_MAT, A),
    ]
    rows = []
    for tag, label, dp, mp, sdir in cells:
        r = eval_cell(tag, os.path.join(REPO, dp), os.path.join(REPO, mp),
                      sdir, args.limit, args.conf, args.out)
        if r:
            r["label"] = label
            rows.append(r)
            print(f"  {label:18s} exact={r['exact_rate']*100:5.1f}%  "
                  f"err={r['err_rate']*100:4.1f}%  class2不符={r['class2_mismatch']*100:4.0f}%  "
                  f"ms={r['ms_mean']}")

    # ---------------- 结论
    step("结论")
    by = {r["tag"]: r for r in rows}
    g = lambda t: (by.get(t) or {}).get("exact_rate")
    oa, ob, nb, na = g("old_on_A"), g("old_on_B"), g("new_on_B"), g("new_on_A")

    if None in (oa, ob, nb):
        print("  数据不足，无法给出结论")
        return

    drift_cost = (ob - oa) * 100
    recover = (nb - ob) * 100
    forgetting = (na - oa) * 100

    print(f"  ① 旧模型 @ 旧域     : {oa*100:5.1f}%")
    print(f"  ② 旧模型 @ 漂移域   : {ob*100:5.1f}%")
    print(f"  ③ 新模型 @ 漂移域   : {nb*100:5.1f}%")
    print(f"  ④ 新模型 @ 旧域     : {na*100:5.1f}%")
    print()
    print(f"  ▶ 数据老化代价（①→②）      : {drift_cost:+.1f} pt")
    print(f"  ▶ 数据更新收益（②→③）      : {recover:+.1f} pt")
    print(f"  ▶ 灾难性遗忘（①→④）        : {forgetting:+.1f} pt")
    print()
    print("  判读：")
    if drift_cost < -15:
        print(f"    ✓ 漂移确实导致显著失效（{drift_cost:+.1f} pt）—— '数据老了' 是真的")
    else:
        print(f"    · 漂移影响有限（{drift_cost:+.1f} pt）—— 本例两域不够远")
    if recover > 15:
        print(f"    ✓ 用新域数据重训可恢复（{recover:+.1f} pt）—— '数据更新' 是对的解法")
    else:
        print(f"    · 重训收益有限（{recover:+.1f} pt）—— 检查训练轮数是否足够")
    if forgetting < -8:
        print(f"    ! 存在明显遗忘（{forgetting:+.1f} pt）—— 上线前应混合旧数据训练")
    else:
        print(f"    ✓ 遗忘可控（{forgetting:+.1f} pt）")

    rep = {"gap": gap, "rows": rows,
           "verdict": {"drift_cost_pt": round(drift_cost, 1),
                       "recover_pt": round(recover, 1),
                       "forgetting_pt": round(forgetting, 1)},
           "config": {"epochs": args.epochs, "n_train": args.n_train,
                      "limit": args.limit, "conf": args.conf},
           "elapsed_min": round((time.time() - t0) / 60, 1)}
    outp = os.path.join(REPO, args.out, "drift_recover.json")
    json.dump(rep, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n报告 -> {outp}")
    print(f"耗时 {rep['elapsed_min']} 分钟")


if __name__ == "__main__":
    main()
