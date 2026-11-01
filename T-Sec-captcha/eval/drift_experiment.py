# -*- coding: utf-8 -*-
"""drift_experiment: 「数据过于老了」的可复现实验 —— 漂移 → 失效 → 重训 → 恢复。

这是回答用户原始问题的核心实验。它把「数据老化」从一句模糊的描述
变成一组可复现、可量化的数字。

实验设计（四个阶段，全部用自造数据，零第三方依赖）：

    阶段 0  数据准备
       域 A（旧域，style=v1）：模拟模型最初训练时的出题风格
       域 B（漂移域，style=v2）：模拟出题方改版后的新风格
       两者用 eval/domain_gap.py 量化差异

    阶段 1  训练"旧模型"：只用域 A 训练
    阶段 2  在域 B 上评测旧模型  ->  得到「老化后的准确率」
    阶段 3  用域 B 数据重训（继续训练 / 数据更新）
    阶段 4  在域 B 上评测新模型  ->  得到「更新后的准确率」

    对照组（关键，用来证明"光加数据不够 / 光加训练不够"）：
       a) 旧模型 + 域 B 评测            = 老化基线
       b) 只用域 A 数据多训 N 轮 + 域 B 评测 = 证明"加训不解决漂移"
       c) 域 A + 域 B 混合训练 + 域 B 评测   = 证明"混合数据能兼顾"
       d) 只用域 B 重训 + 域 A 评测     = 证明"新数据会遗忘旧域"

用法：
    # 完整跑（约 2~3 小时，CPU）
    python eval/drift_experiment.py --full

    # 只准备数据 + 域差异（快，用于先看漂移是否成立）
    python eval/drift_experiment.py --prepare-only

    # 用小规模快速验证流程（每阶段 2 epoch）
    python eval/drift_experiment.py --full --smoke
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


def sh(cmd, log=None, timeout=7200):
    """跑一条命令，返回 (rc, 输出)。

    注意：Windows 上 subprocess 默认按系统代码页(GBK)解码，而我们的脚本输出 UTF-8，
    直接 text=True 会在读取线程里抛 UnicodeDecodeError（而且是在后台线程里，
    主流程看起来"没报错但没输出"）。所以这里显式指定 utf-8 + errors='replace'。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["POLARS_SKIP_CPU_CHECK"] = "1"
    p = subprocess.run(cmd, capture_output=True, env=env, cwd=REPO, timeout=timeout)
    out = (p.stdout or b"").decode("utf-8", "replace") + \
          (p.stderr or b"").decode("utf-8", "replace")
    if log:
        with open(os.path.join(REPO, log), "w", encoding="utf-8") as f:
            f.write(out)
    return p.returncode, out


def step(msg):
    print(f"\n{'='*92}\n{msg}\n{'='*92}", flush=True)


# ------------------------------------------------------------------ 数据准备
def gen_domain(out, style, n_train, n_val, seed, thick=None):
    cmd = [PY, "tools/synth_selfcontained.py", "--out", out,
           "--n-train", str(n_train), "--n-val", str(n_val),
           "--seed", str(seed), "--style", style]
    rc, out_ = sh(cmd, log=f"out/drift_{style}_gen.log")
    if rc != 0:
        print(out_[-1500:])
        raise SystemExit(f"生成 {style} 失败")
    print(f"  [gen] {out}  style={style}")
    return out


def mix_domains(a, b, out):
    """把域 A、域 B 的训练集按 1:1 混合，用于"混合训练"对照。"""
    import random
    for sub in ("images/train", "labels/train", "pairs/train"):
        os.makedirs(os.path.join(REPO, out, sub), exist_ok=True)
    n = 0
    for tag, src in (("A", a), ("B", b)):
        d = os.path.join(REPO, src, "images", "train")
        for f in sorted(os.listdir(d)):
            base = f[:-4]
            for kind, ext in (("images", ".jpg"), ("labels", ".txt"), ("pairs", ".json")):
                s = os.path.join(REPO, src, kind, "train", base + ext)
                if os.path.exists(s):
                    shutil.copy2(s, os.path.join(REPO, out, kind, "train",
                                                 f"{tag}_{base}{ext}"))
            n += 1
    # val 用域 B（评测目标域）
    for kind in ("images", "labels", "pairs"):
        sd = os.path.join(REPO, b, kind, "val")
        dd = os.path.join(REPO, out, kind, "val")
        os.makedirs(dd, exist_ok=True)
        for f in sorted(os.listdir(sd)):
            shutil.copy2(os.path.join(sd, f), os.path.join(dd, f))
    with open(os.path.join(REPO, out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.join(REPO, out)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write("names:\n  0: label\n  1: title\n  2: char\n")
    print(f"  [mix] {out}  训练图 {n} 张（A/B 各半），val 用域 B")
    return out


def gen_pairs(synth_dir, out_dir):
    rc, out_ = sh([PY, "train/prepare_pairs.py", "--synth", synth_dir, "--out", out_dir],
                  log="out/drift_pairs.log")
    if rc != 0:
        print(out_[-1200:])
        raise SystemExit(f"生成配对失败: {synth_dir}")
    print(f"  [pairs] {out_dir}")
    return out_dir


# ------------------------------------------------------------------ 训练
def train_det(data_yaml, name, epochs, batch=8, imgsz=640, out_onnx=None):
    """训练检测器。out_onnx=None 时只训练（用 --train-only），由调用方另行导出。"""
    cmd = [PY, "train/train_detector.py", "--data", data_yaml,
           "--epochs", str(epochs), "--imgsz", str(imgsz), "--batch", str(batch),
           "--workers", "4", "--device", "cpu", "--cache", "ram",
           "--project", "runs/drift/detect", "--name", name]
    if out_onnx:
        cmd += ["--out-onnx", out_onnx]
    else:
        cmd += ["--train-only"]
    rc, out_ = sh(cmd, log=f"out/drift_det_{name}.log")
    if rc != 0:
        print(out_[-2000:])
        raise SystemExit(f"训练检测器失败: {name}")
    pt = os.path.join(REPO, "runs/drift/detect", name, "weights", "best.pt")
    if not os.path.exists(pt):
        raise SystemExit(f"训练完成但没有权重: {pt}")
    return pt


def export_det(pt, out_onnx):
    """把 best.pt 导出成 nms=True 的 ONNX（实测这是端到端的关键）。"""
    code = (
        "import os,sys,shutil\n"
        "os.environ['POLARS_SKIP_CPU_CHECK']='1'\n"
        "sys.path.insert(0,r'%s')\n"
        "from ultralytics import YOLO\n"
        "m=YOLO(r'%s')\n"
        "o=m.export(format='onnx',imgsz=640,batch=1,dynamic=False,simplify=True,opset=14,nms=True)\n"
        "os.makedirs(os.path.dirname(r'%s'),exist_ok=True)\n"
        "shutil.copyfile(o, r'%s')\n"
        "print('OK',os.path.getsize(r'%s'))\n"
    ) % (REPO, pt, out_onnx, out_onnx, out_onnx)
    rc, out_ = sh([PY, "-c", code], log="out/drift_export.log")
    if rc != 0:
        print(out_[-1500:])
        raise SystemExit(f"导出失败: {pt}")
    return out_onnx


def train_mat(pairs_dir, name, epochs, batch=64, out_onnx=None):
    cmd = [PY, "train/train_matcher.py", "--pairs", pairs_dir,
           "--epochs", str(epochs), "--batch", str(batch), "--workers", "0",
           "--threads", "10", "--project", "runs/drift/matcher", "--name", name]
    if out_onnx:
        cmd += ["--out-onnx", out_onnx]
    rc, out_ = sh(cmd, log=f"out/drift_mat_{name}.log")
    if rc != 0:
        print(out_[-2000:])
        raise SystemExit(f"训练匹配器失败: {name}")
    return out_onnx


# ------------------------------------------------------------------ 评测
def eval_combo(tag, det_onnx, mat_onnx, synth_dir, limit=60, conf=0.2):
    """用未改动的 runtime 在同一评测集上跑端到端，返回指标 dict。"""
    code = r'''
import json, os, sys, statistics as st
import cv2, numpy as np
REPO = sys.argv[1]; det_p = sys.argv[2]; mat_p = sys.argv[3]
synth = sys.argv[4]; limit = int(sys.argv[5]); conf = float(sys.argv[6])
tag = sys.argv[7]; outj = sys.argv[8]
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
bases=[f[:-5] for f in sorted(os.listdir(pdir))][:limit]
n=ok=err=n2bad=0; times=[]
for b in bases:
    ip=os.path.join(d,'images','val',b+'.jpg')
    tp=os.path.join(pdir,b+'.json')
    if not (os.path.exists(ip) and os.path.exists(tp)): continue
    img=cv2.imdecode(np.fromfile(ip,dtype=np.uint8),cv2.IMREAD_COLOR)
    if img is None: continue
    truth=json.load(open(tp,encoding='utf-8'))
    gt=sorted(truth['pairs'],key=lambda q:q['click_order'])
    n+=1
    import time; t0=time.time()
    try:
        dets=det.inference(img)
        tb=[x[:4] for x in dets if int(x[5])==0]
        cb=[x[:4] for x in dets if int(x[5])==2]
        cb.sort(key=lambda z:z[0])
        if len(cb)!=len(gt): n2bad+=1
        if not cb or not tb: raise ValueError('no boxes')
        chars=[img[int(z[1]):int(z[3]),int(z[0]):int(z[2])] for z in cb]
        tg=[img[int(z[1]):int(z[3]),int(z[0]):int(z[2])] for z in tb]
        if any(x.size==0 for x in chars+tg): raise ValueError('empty crop')
        res=greedy(mat.reason_all_batch(chars,tg))
        out=[tb[j] for i,j in res]
        if len(out)==len(gt) and all(iou(out[k],gt[k]['target_box'])>0.5 for k in range(len(gt))): ok+=1
    except Exception as e:
        err+=1
    times.append((time.time()-t0)*1000)
r={'tag':tag,'n':n,'exact':ok,'exact_rate':round(ok/max(n,1),4),
   'err_rate':round(err/max(n,1),4),'class2_mismatch':round(n2bad/max(n,1),4),
   'ms_mean':round(st.mean(times),1) if times else None}
json.dump(r, open(outj,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps(r,ensure_ascii=False))
'''
    outj = os.path.join(REPO, "out", f"drift_eval_{tag.replace(' ', '_').replace('+','_')}.json")
    os.makedirs(os.path.join(REPO, "out"), exist_ok=True)
    rc, out_ = sh([PY, "-c", code, REPO, det_onnx, mat_onnx, synth_dir,
                   str(limit), str(conf), tag, outj], log=None)
    if rc != 0:
        print(out_[-1200:])
        return None
    try:
        return json.loads(out_.strip().splitlines()[-1])
    except Exception:
        print("  解析失败:", out_[-400:])
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="跑完整实验（含所有对照组）")
    ap.add_argument("--prepare-only", action="store_true", help="只准备数据 + 域差异")
    ap.add_argument("--smoke", action="store_true", help="小规模快跑（2 epoch / 小数据）")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--mat-epochs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--out", default="out/drift_exp")
    args = ap.parse_args()

    if args.smoke:
        args.n_train, args.n_val = 160, 40
        args.epochs, args.mat_epochs = 2, 2
        args.limit = 30

    os.makedirs(os.path.join(REPO, args.out), exist_ok=True)
    t_all = time.time()

    # ---------------- 阶段 0：数据
    step("阶段 0/4  准备两个域的数据")
    A = gen_domain("data/driftA", "v1", args.n_train, args.n_val, seed=101)
    B = gen_domain("data/driftB", "v2", args.n_train, args.n_val, seed=202)
    MIX = mix_domains("data/driftA", "data/driftB", "data/driftAB")

    rc, gap_out = sh([PY, "eval/domain_gap.py", "--a", A, "--b", B, "--split", "val",
                      "--out", os.path.join(args.out, "gap")])
    print(gap_out[-1800:] if rc == 0 else gap_out[-800:])
    gap = {}
    gp = os.path.join(REPO, args.out, "gap", "gap.json")
    if os.path.exists(gp):
        gap = json.load(open(gp, encoding="utf-8"))

    if args.prepare_only:
        print("\n[prepare-only] 数据与域差异已就绪，未训练。")
        return

    # ---------------- 阶段 1：训练"旧模型"（只用域 A）
    step("阶段 1/4  训练旧模型（只用域 A 的数据）")
    ptA = train_det(os.path.join(A, "data.yaml"), "detA", args.epochs)
    detA = export_det(ptA, "runs/drift/export/detA.onnx")
    pairsA = gen_pairs(A, "data/driftA_pairs")
    matA = train_mat(pairsA, "matA", args.mat_epochs,
                     out_onnx="runs/drift/export/matA.onnx")
    print(f"  旧模型: {detA} + {matA}")

    # ---------------- 阶段 2：旧模型在漂移域上评测（老化基线）
    step("阶段 2/4  评测：旧模型 × 漂移域 B  ->  这就是「数据老化」的代价")
    r_old_on_B = eval_combo("A-on-B 旧模型@漂移域", detA, matA, B, args.limit, args.conf)
    r_old_on_A = eval_combo("A-on-A 旧模型@旧域", detA, matA, A, args.limit, args.conf)
    print(f"  旧模型 @ 旧域 A : {r_old_on_A}")
    print(f"  旧模型 @ 漂移域 B: {r_old_on_B}")

    if not args.full:
        print("\n[提示] 未指定 --full，跳过阶段 3/4（重训与对照）。")
        print("       加上 --full 可跑完整实验（含重训恢复 + 四个对照组）。")
        return

    # ---------------- 阶段 3：用漂移域数据重训
    step("阶段 3/4  重训：用漂移域 B 的数据更新模型（+ 混合对照）")
    ptB = train_det(os.path.join(B, "data.yaml"), "detB", args.epochs)
    detB = export_det(ptB, "runs/drift/export/detB.onnx")
    pairsB = gen_pairs(B, "data/driftB_pairs")
    matB = train_mat(pairsB, "matB", args.mat_epochs,
                     out_onnx="runs/drift/export/matB.onnx")

    # 对照组 b：只用域 A 数据多训（证明"加训不解决漂移"）
    ptA2 = train_det(os.path.join(A, "data.yaml"), "detA_more", args.epochs * 2)
    detA2 = export_det(ptA2, "runs/drift/export/detA_more.onnx")

    # 对照组 c：A+B 混合训练
    ptAB = train_det(os.path.join(MIX, "data.yaml"), "detAB", args.epochs)
    detAB = export_det(ptAB, "runs/drift/export/detAB.onnx")

    # ---------------- 阶段 4：全矩阵评测
    step("阶段 4/4  全矩阵评测")
    rows = []
    combos = [
        ("A-on-A", "旧模型@旧域", detA, matA, A),
        ("A-on-B", "旧模型@漂移域", detA, matA, B),
        ("A2-on-B", "旧数据多训@漂移域", detA2, matA, B),
        ("AB-on-B", "混合训练@漂移域", detAB, matB, B),
        ("B-on-B", "重训@漂移域", detB, matB, B),
        ("B-on-A", "重训@旧域（遗忘检查）", detB, matB, A),
    ]
    for tag, label, dp, mp, sdir in combos:
        r = eval_combo(tag, dp, mp, sdir, args.limit, args.conf)
        if r:
            r["label"] = label
            rows.append(r)
            print(f"  {label:26s} exact={r['exact_rate']*100:5.1f}%  "
                  f"err={r['err_rate']*100:4.1f}%  class2不符={r['class2_mismatch']*100:4.0f}%")

    # ---------------- 结论
    step("结论")
    by = {r["tag"]: r for r in rows}

    def g(t):
        return by.get(t, {}).get("exact_rate")

    a_on_a, a_on_b = g("A-on-A"), g("A-on-B")
    a2_on_b = g("A2-on-B")
    ab_on_b = g("AB-on-B")
    b_on_b, b_on_a = g("B-on-B"), g("B-on-A")

    print(f"  ① 旧模型在旧域          : {a_on_a*100:.1f}%   (模型本来能用)")
    print(f"  ② 旧模型在漂移域        : {a_on_b*100:.1f}%   <- 数据老化的代价")
    if a_on_a is not None and a_on_b is not None:
        print(f"     => 漂移导致掉 {abs(a_on_a-a_on_b)*100:.1f} 个百分点")
    if a2_on_b is not None:
        print(f"  ③ 旧数据多训 2x 轮@漂移域: {a2_on_b*100:.1f}%   <- 证明加训救不了漂移")
    if ab_on_b is not None:
        print(f"  ④ 混合数据训练@漂移域    : {ab_on_b*100:.1f}%   <- 兼顾新旧")
    if b_on_b is not None:
        print(f"  ⑤ 重训后@漂移域          : {b_on_b*100:.1f}%   <- 数据更新生效")
    if b_on_a is not None:
        print(f"  ⑥ 重训后@旧域（遗忘）    : {b_on_a*100:.1f}%   <- 检查灾难性遗忘")

    verdict = {
        "drift_cost_pt": round((a_on_a - a_on_b) * 100, 1) if None not in (a_on_a, a_on_b) else None,
        "retrain_gain_pt": round((b_on_b - a_on_b) * 100, 1) if None not in (b_on_b, a_on_b) else None,
        "more_epochs_gain_pt": round((a2_on_b - a_on_b) * 100, 1) if None not in (a2_on_b, a_on_b) else None,
        "mix_vs_retrain_pt": round((ab_on_b - b_on_b) * 100, 1) if None not in (ab_on_b, b_on_b) else None,
        "forgetting_pt": round((a_on_a - b_on_a) * 100, 1) if None not in (a_on_a, b_on_a) else None,
    }
    print("\n  关键结论：")
    v = verdict
    if v["drift_cost_pt"] is not None:
        print(f"    - 数据老化代价      : {v['drift_cost_pt']:+.1f} pt")
    if v["retrain_gain_pt"] is not None:
        print(f"    - 数据更新收益      : {v['retrain_gain_pt']:+.1f} pt")
    if v["more_epochs_gain_pt"] is not None:
        print(f"    - 只加训练轮数收益  : {v['more_epochs_gain_pt']:+.1f} pt  "
              f"({'加训无效，必须换数据' if abs(v['more_epochs_gain_pt']) < 5 else '加训也有帮助'})")
    if v["forgetting_pt"] is not None:
        print(f"    - 灾难性遗忘（旧域）: {v['forgetting_pt']:+.1f} pt  "
              f"({'需混合旧数据' if v['forgetting_pt'] > 8 else '遗忘可控'})")

    rep = {"gap": gap, "rows": rows, "verdict": verdict,
           "config": {"n_train": args.n_train, "n_val": args.n_val,
                      "epochs": args.epochs, "mat_epochs": args.mat_epochs,
                      "limit": args.limit, "conf": args.conf, "smoke": args.smoke},
           "elapsed_min": round((time.time() - t_all) / 60, 1)}
    outp = os.path.join(REPO, args.out, "drift_experiment.json")
    json.dump(rep, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n报告 -> {outp}")
    print(f"总耗时 {rep['elapsed_min']} 分钟")


if __name__ == "__main__":
    main()
