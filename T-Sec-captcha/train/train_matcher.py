# -*- coding: utf-8 -*-
"""train_matcher: 继续训练/微调「提示图标 ↔ 底图图标」匹配器。

线上 runtime 的事实（已从 ONNX 反解确认）：
    输入 : input1, input2 各 [batch, 3, 112, 112] float32，值域 0~1（letterbox pad=128）
    输出 : logits [batch, 1] float32（runtime 里做 sigmoid 当相似度）
    结构 : 共享 backbone + fusion_head(3 层 Gemm/BN/ReLU)，Sub/Abs/Mul/Concat 做差特征融合
    —— 本质是一个 siamese 二分类器：这对图标是同一个吗？

训练数据从哪来：
    合成器 data/synth/pairs/*.json 里已经写好 ground truth：
    第 i 个提示图标 ↔ 底图哪个框 是正样本；其余组合是负样本。
    这让我们可以无限采样正负对，而且**标签 0 成本**。

关键设计（为什么这么做）：
    1. **不改 I/O 契约**：导出时严格保持 input1/input2 双输入、logits 单输出、112x112、动态 batch。
       runtime(src/utils/ver_onnx.py) 一行都不用改。
    2. **支持从 ONNX 反解权重继续训练**（真·继续训练），也支持从零训练新 backbone。
    3. **难负样本挖掘**：把「形近图标」当重点负样本（如 0/O、建筑/宫殿）。

用法：
    # 1) 准备配对数据（从合成集）
    python train/prepare_pairs.py --synth data/synth --out data/pairs

    # 2) 训练（CPU）
    python train/train_matcher.py --pairs data/pairs --epochs 20 --batch 64 --device cpu

    # 3) 导出并校验契约
    python train/train_matcher.py --export-only runs/matcher/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

IN_SIZE = 112


# ------------------------------------------------------------------ 预处理
# 必须与 src/utils/ver_onnx.py 完全一致，否则线上线下不一致
def letterbox_image(image_np, target_size=(112, 112)):
    h, w = target_size
    ih, iw = image_np.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.full((h, w, 3), 128, dtype=np.uint8)
    dx, dy = (w - nw) // 2, (h - nh) // 2
    new_image[dy:dy + nh, dx:dx + nw] = resized
    return new_image


def preprocess_bgr(img_bgr):
    """BGR -> RGB -> letterbox(128) -> /255 -> CHW。复刻 ver_onnx.preprocess_image。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lb = letterbox_image(rgb, (IN_SIZE, IN_SIZE))
    x = lb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return x


# ------------------------------------------------------------------ 数据集
class PairDataset:
    """读取 prepare_pairs.py 产出的 .npz 或 on-the-fly 从图片裁剪。"""

    def __init__(self, index_file, mode="ez"):
        self.items = []
        for line in open(index_file, encoding="utf-8"):
            line = line.strip()
            if line:
                self.items.append(json.loads(line))
        self.mode = mode

    def __len__(self):
        return len(self.items)

    def load_pair(self, rec):
        """返回 (img1, img2, label)：BGR ndarray。"""
        a = cv2.imdecode(np.fromfile(rec["a"], dtype=np.uint8), cv2.IMREAD_COLOR)
        b = cv2.imdecode(np.fromfile(rec["b"], dtype=np.uint8), cv2.IMREAD_COLOR)
        return a, b, int(rec["label"])


def collate(batch):
    import torch
    x1 = np.stack([preprocess_bgr(a) for a, b, l in batch], axis=0)
    x2 = np.stack([preprocess_bgr(b) for a, b, l in batch], axis=0)
    y = np.array([l for a, b, l in batch], dtype=np.float32).reshape(-1, 1)
    return (torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(y))


# ------------------------------------------------------------------ 模型
def build_matcher(from_onnx=None, from_pt=None, backbone="mobilenetv3",
                  pretrained=True):
    """构造与线上同契约的双输入匹配器。

    优先：从 ONNX 反解（真·继续训练）
    其次：从已存 .pt 恢复
    最后：新建 backbone（默认 ImageNet 预训练，见 SiameseMatcher 文档）
    """
    import torch
    import torch.nn as nn

    if from_onnx and os.path.exists(from_onnx):
        try:
            from onnx2torch import convert
            m = convert(from_onnx)
            print(f"[model] 从 ONNX 反解: {from_onnx}")
            return m, "onnx2torch"
        except Exception as e:
            print(f"[model] ONNX 反解失败: {type(e).__name__}: {e}")

    if from_pt and os.path.exists(from_pt):
        print(f"[model] 从 .pt 恢复: {from_pt}")
        m = SiameseMatcher(backbone, pretrained=False)
        sd = torch.load(from_pt, map_location="cpu", weights_only=False)
        m.load_state_dict(sd.get("state_dict", sd), strict=False)
        return m, "pt"

    print(f"[model] 新建: {backbone} (pretrained={pretrained})")
    return SiameseMatcher(backbone, pretrained), "new"


class SiameseMatcher:
    """工厂：返回与线上契约一致的结构。

    线上是「共享 backbone + 差特征融合」。
    为便于训练，本实现用 torchvision 的 mobilenet_v3_small 当 backbone（轻量、112 输入快），
    融合头复刻 |f1-f2| ⊕ f1⊙f2 -> MLP -> 1 logit。
    若要用 ONNX 反解的原结构，直接走 onnx2torch 分支即可。

    【关键修正·pretrained 默认开启】
        实测在 3286 对真实 2408 配对（正:负 ≈ 1:3）上，
        从随机初始化训练会**立刻塌缩**：val_acc 精确停在 0.25（= 正样本比例），
        loss 只从 1.02 降到 0.86 就不再动 —— 因为随机 backbone 的 576 维特征
        对「同一个图标 vs 不同图标」几乎无区分度，融合头只能学出常数。
        换成 ImageNet 预训练的 mobilenet_v3_small（9.8 MB，首次自动下载）后，
        特征一开始就带边缘/形状语义，融合头能正常收敛。
    """

    def __new__(cls, backbone="mobilenetv3", pretrained=True):
        import torch
        import torch.nn as nn
        import torchvision.models as tvm

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                w = None
                if pretrained:
                    try:
                        w = "DEFAULT"
                    except Exception:
                        w = None
                if backbone == "mobilenetv3":
                    try:
                        bb = tvm.mobilenet_v3_small(weights=w)
                    except Exception as e:
                        print(f"[model] 预训练权重不可用({type(e).__name__})，退回随机初始化")
                        bb = tvm.mobilenet_v3_small(weights=None)
                    self.feat_dim = bb.classifier[0].in_features
                    bb.classifier = nn.Identity()
                    self.backbone = bb
                else:
                    try:
                        bb = tvm.resnet18(weights=w)
                    except Exception:
                        bb = tvm.resnet18(weights=None)
                    self.feat_dim = bb.fc.in_features
                    bb.fc = nn.Identity()
                    self.backbone = bb
                d = self.feat_dim
                self.fusion_head = nn.Sequential(
                    nn.Linear(d * 2, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                    nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                    nn.Linear(64, 1),
                )

            def forward(self, x1, x2):
                f1 = self.backbone(x1)
                f2 = self.backbone(x2)
                z = torch.cat([torch.abs(f1 - f2), f1 * f2], dim=1)
                return self.fusion_head(z)

        return Net()


# ------------------------------------------------------------------ 训练循环
def train(args):
    import torch
    from torch.utils.data import DataLoader

    torch.set_num_threads(args.threads)
    device = torch.device(args.device if args.device != "cpu" else "cpu")

    model, src = build_matcher(from_onnx=args.from_onnx, from_pt=args.from_pt,
                               backbone=args.backbone,
                               pretrained=not args.no_pretrained)
    model = model.to(device)

    ds_tr = PairDataset(os.path.join(args.pairs, "train.jsonl"))
    ds_va = PairDataset(os.path.join(args.pairs, "val.jsonl"))
    print(f"[data] train={len(ds_tr)} val={len(ds_va)}  (模型来源: {src})")

    def make_loader(ds, shuffle):
        return DataLoader(_TorchPairSet(ds), batch_size=args.batch, shuffle=shuffle,
                          num_workers=args.workers, collate_fn=collate)

    tr = make_loader(ds_tr, True)
    va = make_loader(ds_va, False) if len(ds_va) else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    # 正样本加权。
    #
    # 为什么需要：难负样本配比是 1:3（neg_per_pos=3 且全部 hard），
    # 未加权时 BCE 的最优解就是"全判负"，val_acc 会正好停在负样本比例上
    # （实测 driftB_pairs_det: 0.7177 == 600/836，模型塌缩成常数输出）。
    # pos_weight = 负/正 把这个先验抵消掉。
    n_pos = sum(1 for r in ds_tr.items if r["label"] == 1)
    n_neg = len(ds_tr.items) - n_pos
    pw = None
    if args.pos_weight is not None:
        pw = args.pos_weight
    elif args.auto_pos_weight and n_pos > 0:
        pw = n_neg / n_pos
    if pw:
        print(f"[loss] pos_weight={pw:.3f}  (pos={n_pos} neg={n_neg})")
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    else:
        crit = torch.nn.BCEWithLogitsLoss()

    outdir = os.path.join(args.project, args.name)
    os.makedirs(outdir, exist_ok=True)
    best = -1.0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x1, x2, y in tr:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x1, x2)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * x1.size(0)
            n += x1.size(0)
        sched.step()
        msg = f"[ep {ep+1}/{args.epochs}] loss={tot/max(n,1):.4f} {time.time()-t0:.1f}s"
        if va:
            acc, auc = evaluate(model, va, device)
            msg += f" val_acc={acc:.4f} val_auc={auc:.4f}"
            # 塌缩检测：模型若退化成常数输出，val_acc 会精确等于"多数类比例"。
            # 这在难负样本 3:1 的配对集上很常见（实测见 docs/DRIFT_EXPERIMENT.md 6.5）。
            prior = max(n_pos, n_neg) / max(n_pos + n_neg, 1)
            if abs(acc - prior) < 0.005:
                msg += f" [塌缩! acc≈多数类比例 {prior:.4f}]"
            if acc > best:
                best = acc
                torch.save({"state_dict": model.state_dict(), "val_acc": acc},
                           os.path.join(outdir, "best.pt"))
                msg += " *"
        else:
            torch.save({"state_dict": model.state_dict()},
                       os.path.join(outdir, "best.pt"))
        print(msg, flush=True)

    print(f"[train] 完成 -> {outdir}/best.pt")
    return os.path.join(outdir, "best.pt")


class _TorchPairSet:
    def __init__(self, ds):
        self.ds = ds
        self.items = ds.items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        a = cv2.imdecode(np.fromfile(r["a"], dtype=np.uint8), cv2.IMREAD_COLOR)
        b = cv2.imdecode(np.fromfile(r["b"], dtype=np.uint8), cv2.IMREAD_COLOR)
        return (a, b, int(r["label"]))


def evaluate(model, loader, device):
    import torch
    model.eval()
    correct = n = 0
    probs_all, ys_all = [], []
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            logits = model(x1, x2)
            p = torch.sigmoid(logits)
            pred = (p > 0.5).float()
            correct += int((pred == y).sum())
            n += y.numel()
            probs_all.append(p.cpu().numpy().ravel())
            ys_all.append(y.cpu().numpy().ravel())
    acc = correct / max(n, 1)
    auc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        y_true = np.concatenate(ys_all)
        if len(set(y_true.tolist())) > 1:      # AUC 需要两类都出现
            auc = roc_auc_score(y_true, np.concatenate(probs_all))
    except Exception:
        pass
    return acc, auc


# ------------------------------------------------------------------ 导出
def export_onnx(pt_path, out_path, backbone="mobilenetv3", opset=14):
    import torch
    m, _ = build_matcher(from_pt=pt_path, backbone=backbone, pretrained=False)
    m.eval()
    dummy1 = torch.rand(2, 3, IN_SIZE, IN_SIZE)
    dummy2 = torch.rand(2, 3, IN_SIZE, IN_SIZE)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    kw = dict(
        input_names=["input1", "input2"], output_names=["logits"],
        dynamic_axes={"input1": {0: "batch"}, "input2": {0: "batch"},
                      "logits": {0: "batch"}},
        opset_version=opset,
    )
    try:
        # torch>=2.5 默认用 dynamo 导出器（需要 onnxscript）；失败则回退传统导出器
        torch.onnx.export(m, (dummy1, dummy2), out_path,
                          do_constant_folding=True, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(m, (dummy1, dummy2), out_path, **kw)
    except Exception as e:
        print(f"[export] 默认导出器失败({type(e).__name__}: {e})，回退 torchscript 导出器")
        torch.onnx.export(m, (dummy1, dummy2), out_path, dynamo=False, **kw)
    print(f"[export] -> {out_path}")
    ok = verify_matcher_contract(out_path)
    return out_path if ok else out_path


def verify_matcher_contract(onnx_path):
    import onnxruntime as ort
    import numpy as np
    print("\n[verify] 匹配器契约校验")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape) for i in sess.get_inputs()]
    outs = [(o.name, o.shape) for o in sess.get_outputs()]
    print(f"  inputs : {ins}")
    print(f"  outputs: {outs}")
    ok = True
    if [n for n, s in ins] != ["input1", "input2"]:
        print("  [FAIL] 输入名必须是 input1/input2")
        ok = False
    if [n for n, s in outs] != ["logits"]:
        print("  [FAIL] 输出名必须是 logits")
        ok = False
    for n, s in ins:
        if len(s) != 4 or s[1] != 3 or s[2] != IN_SIZE or s[3] != IN_SIZE:
            print(f"  [FAIL] {n} shape 应为 [batch,3,{IN_SIZE},{IN_SIZE}]，实际 {s}")
            ok = False
    if len(outs) == 1 and len(outs[0][1]) != 2:
        print("  [FAIL] logits 应为 [batch,1]")
        ok = False
    # 动态 batch 跑通
    x = np.random.rand(7, 3, IN_SIZE, IN_SIZE).astype(np.float32)
    y = sess.run(None, {"input1": x, "input2": x})[0]
    print(f"  动态 batch=7 推理: {y.shape}")
    if y.shape != (7, 1):
        print("  [FAIL] 动态 batch 输出不对")
        ok = False
    print(f"[verify] 结论: {'通过' if ok else '不通过'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--auto-pos-weight", action="store_true",
                    help="按 负/正 比例自动给正样本加权。默认关闭以保持既有行为；"
                         "当配对集**又小又难**时（如用检测框对齐生成的），"
                         "不加权会让模型塌缩成「全判负」（val_acc 恰好等于负样本比例）。")
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="手动指定 pos_weight（覆盖自动计算）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--backbone", default="mobilenetv3")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="关闭 ImageNet 预训练初始化。默认开启 —— 实测随机初始化在 "
                         "1:3 难负配对集上会立刻塌缩（val_acc 恰好等于正样本比例）")
    ap.add_argument("--from-onnx", default=None, help="线上 ONNX，反解后继续训练")
    ap.add_argument("--from-pt", default=None)
    ap.add_argument("--project", default="runs/matcher")
    ap.add_argument("--name", default="tsec_matcher")
    ap.add_argument("--export-only", default=None)
    ap.add_argument("--out-onnx", default="model/pre_model_v7_next.onnx")
    args = ap.parse_args()

    if args.export_only:
        export_onnx(args.export_only, args.out_onnx, args.backbone)
        return
    pt = train(args)
    if args.epochs > 0:
        try:
            export_onnx(pt, args.out_onnx, args.backbone)
        except Exception as e:
            print(f"[export] 失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
