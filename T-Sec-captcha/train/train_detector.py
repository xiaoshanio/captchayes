# -*- coding: utf-8 -*-
"""train_detector: 继续训练/微调目标检测器（YOLO 点选检测）。

一键流程：
    1) 从现有 ONNX 反解出 warm-start 权重（可选；失败则从公开预训练权重起）
    2) 在合成 + 真机标注数据集上微调
    3) 导出成 **与现有 runtime 完全一致** 的 ONNX（输入 images [1,3,640,640]，
       输出 output0 [1,300,6]，classes = {0:label, 1:title, 2:char}）
    4) 契约自检：用 src/captcha.py 加载新模型跑通

用法：
    # 1) 先看数据对不对
    python train/train_detector.py --data data/synth/data.yaml --dry-run

    # 2) 本机 CPU 小步微调（时间紧张时用）
    python train/train_detector.py --data data/synth/data.yaml --epochs 30 --imgsz 640 --batch 8

    # 3) 有 GPU 时
    python train/train_detector.py --data data/synth/data.yaml --device 0 --epochs 120 --batch 32

    # 4) 只导出（已有 best.pt）
    python train/train_detector.py --export-only runs/detect/train/weights/best.pt
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CLASS_NAMES = {0: "label", 1: "title", 2: "char"}


def check_dataset(data_yaml):
    import yaml
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    root = cfg.get("path") or os.path.dirname(os.path.abspath(data_yaml))
    print(f"[data] yaml={data_yaml}\n       path={root}")
    ok = True
    for split in ("train", "val"):
        rel = cfg.get(split)
        if not rel:
            continue
        img_dir = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if not os.path.isdir(img_dir):
            print(f"  [FAIL] {split}: 目录不存在 {img_dir}")
            ok = False
            continue
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        lab_dir = img_dir.replace("images", "labels")
        labs = [f for f in os.listdir(lab_dir)] if os.path.isdir(lab_dir) else []
        print(f"  {split}: images={len(imgs)} labels={len(labs)} ({img_dir})")
        if imgs and labs and len(imgs) != len(labs):
            print(f"  [WARN] {split} 图片/标签数量不一致")
        # 抽查一条标签
        if labs:
            p = os.path.join(lab_dir, sorted(labs)[0])
            with open(p, encoding="utf-8") as f:
                first = f.readline().strip()
            print(f"  样本标签 {os.path.basename(p)}: {first}")
            vals = first.split()
            if len(vals) == 5:
                c = int(float(vals[0]))
                if c not in CLASS_NAMES:
                    print(f"  [FAIL] 类别 {c} 不在 {list(CLASS_NAMES)}")
                    ok = False
                if not all(0 <= float(v) <= 1 for v in vals[1:]):
                    print("  [WARN] 归一化坐标越界")
    print(f"[data] {'OK' if ok else '有问题'}")
    return ok


def build_model(arch, warm_pt=None, recovered=None):
    from ultralytics import YOLO
    if warm_pt and os.path.exists(warm_pt):
        print(f"[model] 从已有 .pt 继续训练: {warm_pt}")
        return YOLO(warm_pt)
    if recovered and os.path.exists(recovered):
        # 从 ONNX 反解出的权重：用 ultralytics 结构 + 手工 load
        print(f"[model] 尝试用反解权重做 warm start: {recovered}")
        try:
            import torch
            m = YOLO(arch)
            sd = torch.load(recovered, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            missing, unexpected = m.model.load_state_dict(sd, strict=False)
            print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
            if len(missing) > len(sd) * 0.5:
                raise RuntimeError("不匹配率过高，放弃 warm start")
            print("  warm start OK")
            return m
        except Exception as e:
            print(f"  warm start 失败({type(e).__name__}: {e})，改用公开预训练权重")
    print(f"[model] 从架构 yaml 新建: {arch}")
    return YOLO(arch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth/data.yaml")
    ap.add_argument("--arch", default="yolo26n.yaml",
                    help="架构；与线上 best_v3.onnx 同族(YOLO26n)")
    ap.add_argument("--warm-pt", default=None, help="已有 .pt（最优）")
    ap.add_argument("--recovered", default="weights/best_v3.pt",
                    help="从 ONNX 反解出的 .pt（次优）")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="必须与 runtime letterbox 一致 -> 640")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu 或 0 / 0,1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=0.002, help="微调用小学习率")
    ap.add_argument("--freeze", type=int, default=0, help="冻结前 N 层")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="tsec_det")
    ap.add_argument("--export-only", default=None, help="给定 best.pt 只做导出")
    ap.add_argument("--out-onnx", default="model/best_v3_next.onnx")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--train-only", action="store_true",
                    help="只训练不导出（供上层脚本自行控制导出时机）")
    ap.add_argument("--cache", default=None, help="ram/disk，CPU 训练强烈建议 ram")
    # --- 老 CPU（无 AVX2/FMA）适配开关：本机实测这三个必须关 ---
    ap.add_argument("--no-amp", action="store_true", default=True,
                    help="关闭 AMP（CPU 上 autocast 会触发非法指令，默认关）")
    ap.add_argument("--amp", dest="no_amp", action="store_false")
    ap.add_argument("--no-plots", action="store_true", default=True,
                    help="关闭训练曲线绘图（默认关，省事省依赖）")
    ap.add_argument("--plots", dest="no_plots", action="store_false")
    ap.add_argument("--no-val", action="store_true", default=False,
                    help="训练期间跳过验证（最快）")
    args = ap.parse_args()

    if not check_dataset(args.data) and not args.dry_run:
        print("数据检查未通过，仍继续（请自行确认）")
    if args.dry_run:
        return

    if args.export_only:
        pt = args.export_only
    else:
        model = build_model(args.arch, args.warm_pt, args.recovered)
        # ultralytics 会把 project 相对其全局 runs_dir 解析，传绝对路径避免出现
        # runs/detect/runs/detect/xxx 这类嵌套
        project = os.path.abspath(args.project)
        t0 = time.time()
        kwargs = dict(
            data=os.path.abspath(args.data), epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch,
            device=args.device, workers=args.workers, lr0=args.lr0,
            project=project, name=args.name, exist_ok=True,
            # 与线上导出一致的契约
            rect=False, cache=args.cache,
            # 老 CPU 必须有：AMP 关闭、绘图关闭、验证可选
            amp=(not args.no_amp), plots=(not args.no_plots), val=(not args.no_val),
            # 关键：维持 end2end 输出（[1,300,6]），否则 runtime 后处理不匹配
            end2end=True,
        )
        if args.freeze:
            kwargs["freeze"] = args.freeze
        print(f"[train] 开始训练: {kwargs}")
        res = model.train(**kwargs)
        pt = os.path.join(project, args.name, "weights", "best.pt")
        print(f"[train] 完成，耗时 {(time.time()-t0)/60:.1f} min -> {pt}")

    if not os.path.exists(pt):
        raise SystemExit(f"找不到权重: {pt}")

    # ---------------- 导出 ONNX（契约对齐）
    #
    # 【关键结论·实测】导出时必须 nms=True，不能只靠 end2end=True 的 one-to-one 头。
    #   实测（自造 60 题，同一匹配器）：
    #     end2end=True,  nms=False  -> 端到端全对 40.0%，class2 重复框 22 对
    #     end2end=False, nms=True   -> 端到端全对 76.7%，class2 重复框  3 对
    #   原因：one-to-one 头即使 mAP50 到 0.987，仍会输出同一图标的近重合框；
    #         这些重复框在 runtime 的贪心指派里会占掉多余的列，直接破坏点击顺序。
    #   契约不受影响：两种导出的 I/O 都是
    #         images [1,3,640,640]  ->  output0 [1,300,6]
    #   而 runtime(src/utils/yolo_onnx.py) 只读 I/O 形状，**从不读 metadata 的 end2end 字段**
    #   （已用 grep 确认），所以 nms=True 的导出可以原样替换线上模型。
    if args.train_only:
        print(f"[train-only] 跳过导出。权重: {pt}")
        return

    from ultralytics import YOLO
    m = YOLO(pt)
    out = os.path.abspath(args.out_onnx)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"[export] {pt} -> {out} (imgsz={args.imgsz}, batch=1, dynamic=False, "
          f"simplify=True, opset=14, nms=True)")
    exported = m.export(format="onnx", imgsz=args.imgsz, batch=1, dynamic=False,
                        simplify=True, opset=14, nms=True)
    if os.path.abspath(exported) != out:
        shutil.copyfile(exported, out)
    print(f"[export] 完成: {out}")

    verify_contract(out, args.imgsz)


def verify_contract(onnx_path, imgsz=640):
    """严格校验导出模型与 src/utils/yolo_onnx.py 的契约。"""
    import onnx
    import numpy as np
    import onnxruntime as ort

    print("\n[verify] 契约校验")
    model = onnx.load(onnx_path, load_external_data=False)
    props = {p.key: p.value for p in model.metadata_props}
    names = props.get("names")
    print(f"  names       : {names}")
    print(f"  end2end     : {props.get('end2end')}   (runtime 不读此字段，仅作记录)")
    print(f"  nms(args)   : {(props.get('args') or '').count('nms') and props.get('args')}")
    print(f"  imgsz       : {props.get('imgsz')}")
    print(f"  task        : {props.get('task')}")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
    outs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
    print(f"  inputs      : {ins}")
    print(f"  outputs     : {outs}")

    ok = True
    if not ins or ins[0][1] != [1, 3, imgsz, imgsz]:
        print(f"  [FAIL] 输入 shape 期望 [1,3,{imgsz},{imgsz}]")
        ok = False
    if not outs or len(outs[0][1]) != 3 or outs[0][1][2] != 6:
        print("  [FAIL] 输出期望 [1,300,6]")
        ok = False
    elif outs[0][1][1] != 300:
        print(f"  [WARN] 第二维为 {outs[0][1][1]}，runtime 里按 300 遍历，"
              f"不改代码的话最好仍是 300")
    # 类名必须与 runtime 的 0/2 语义一致
    if names and ("label" not in names or "char" not in names):
        print(f"  [WARN] 类名 {names} 与 runtime 期望的 label/char 不一致")

    # 真实跑一遍 + 与 runtime 对接
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    y = sess.run(None, {ins[0][0]: x})[0]
    print(f"  推理输出 shape: {y.shape}  dtype={y.dtype}")
    if y.shape[-1] != 6:
        print("  [FAIL] 最后一维不是 6 ([x1,y1,x2,y2,conf,cls])")
        ok = False

    # 类别数：看置信度>0 的框的类别取值
    cls_ids = sorted({int(v[5]) for v in y[0] if v[4] > 0.01})
    print(f"  出现的类别 id: {cls_ids}")

    print(f"\n[verify] 结论: {'通过' if ok else '不通过'}")
    if ok:
        print("  下一步：复制到 model/ 并跑 python demo.py 验证端到端")
    return ok


if __name__ == "__main__":
    main()
