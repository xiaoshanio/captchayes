# -*- coding: utf-8 -*-
"""verify_recovered: 对 weights/*.pt 做端到端 warm-start 冒烟测试。

证明的不是「能前向」，而是「能当训练起点用」：
    pre_model_v7.pt      : 真实 BCE 损失 + SGD 若干步 -> 损失必须下降
    best_v3_native.pt    : ultralytics v8DetectionLoss + SGD 若干步 -> 损失必须下降
    best_v3.pt           : 真实图 docs/res.jpg 推理，打印有效检测框

用法：
    python tools/verify_recovered.py
    python tools/verify_recovered.py --steps 6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "tools"))

from recover_weights import (  # noqa: E402
    build_native_yolo26n,
    map_onnx_into_native,
    ort_session,
    yolo_preprocess,
)


def train_siamese(pt_path: str, onnx_path: str, steps: int = 60, seed: int = 7) -> dict:
    """过拟合单 batch 冒烟测试：固定随机输入对 + 固定目标，看 BCE 损失能否收敛到接近 0。

    这是最干净的「能否训练」判据：
      * 损失下降 => 计算图可导、梯度有效、优化器能更新参数；
      * 若梯度在某处断掉，损失会卡住不动。
    """
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    from recover_weights import convert_onnx

    stage = os.path.join(_REPO, "weights", "_stage")
    module, _ = convert_onnx(onnx_path, stage)
    missing, unexpected = module.load_state_dict(payload["state_dict"], strict=False)
    module.train()

    g = torch.Generator().manual_seed(seed)
    x1 = torch.rand(8, 3, 112, 112, generator=g)
    x2 = torch.rand(8, 3, 112, 112, generator=g)
    # 用一个固定的（含 Siamese 语义的）目标：相同索引 -> 1，逆序 -> 0
    with torch.no_grad():
        module.eval()
        base = module(x1, x2)
        if isinstance(base, (list, tuple)):
            base = base[0]
        module.train()
    y = torch.where(base.detach().reshape(-1) > base.detach().mean(),
                    torch.tensor(1.0), torch.tensor(0.0))  # 自洽目标，保证初始必然犯错

    opt = torch.optim.Adam(module.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    lossf = torch.nn.BCEWithLogitsLoss()
    history = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = module(x1, x2)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        loss = lossf(logits.reshape(-1), y)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(module.parameters(), 1e9))
        opt.step()
        sched.step()
        if i % max(1, steps // 6) == 0 or i == steps - 1:
            history.append({"step": i, "loss": float(loss), "grad_norm": gn,
                            "lr": float(opt.param_groups[0]["lr"])})
    first, last = history[0]["loss"], history[-1]["loss"]
    best = min(h["loss"] for h in history)
    return {
        "pt": pt_path,
        "load_missing": len(missing),
        "load_unexpected": len(unexpected),
        "steps": steps,
        "optimizer": "Adam lr=1e-3 -> 1e-5 (CosineAnnealingLR)",
        "history": history,
        "loss_first": first,
        "loss_last": last,
        "loss_best": best,
        "loss_dropped": bool(last < first * 0.9),
        "loss_ratio": round(last / max(1e-12, first), 4),
        "loss_ratio_best": round(best / max(1e-12, first), 4),
    }


def train_yolo_native(pt_path: str, onnx_path: str, steps: int = 30, seed: int = 11) -> dict:
    """用 ultralytics 官方 E2ELoss（end2end 训练用的真实损失）在合成 batch 上跑 SGD。

    验证点：真实检测损失能前向 + 反传 + 更新参数，且 loss 下降。
    """
    from types import SimpleNamespace

    from ultralytics.utils.loss import E2ELoss

    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = build_native_yolo26n(nc=3, end2end=True)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    model.train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=100)
    model.class_weights = None

    criterion = E2ELoss(model)
    # 3 个类别各 2 个目标，坐标是 xywh 归一化（ultralytics 约定）
    targets = torch.tensor(
        [
            [0, 2, 0.30, 0.35, 0.12, 0.10],
            [0, 0, 0.70, 0.40, 0.16, 0.12],
            [0, 1, 0.50, 0.75, 0.20, 0.14],
            [0, 2, 0.22, 0.62, 0.10, 0.09],
            [0, 0, 0.80, 0.70, 0.14, 0.11],
            [0, 1, 0.40, 0.25, 0.18, 0.15],
        ],
        dtype=torch.float32,
    )
    batch = {
        "batch_idx": targets[:, 0],
        "cls": targets[:, 1],
        "bboxes": targets[:, 2:],
        "img": torch.rand(1, 3, 640, 640),
    }

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        preds = model(batch["img"])
        loss, items = criterion(preds, batch)
        # ultralytics 的 E2ELoss/v8DetectionLoss 返回 (box, cls, dfl) 三元组向量
        # （trainer 里是 self.loss += loss.detach() 然后 loss.sum().backward()），这里同样求和。
        loss_sum = loss.sum()
        loss_sum.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
        opt.step()
        if i % max(1, steps // 6) == 0 or i == steps - 1:
            history.append(
                {
                    "step": i,
                    "loss": float(loss_sum),
                    "loss_vec": [float(v) for v in loss.detach()],
                    "grad_norm": gn,
                    "items": [float(v) for v in items.values()] if isinstance(items, dict) else [float(v) for v in items],
                }
            )
    first, last = history[0]["loss"], history[-1]["loss"]
    return {
        "pt": pt_path,
        "load_missing": len(missing),
        "load_unexpected": len(unexpected),
        "steps": steps,
        "loss_names": list(criterion.one2one.loss_names),
        "history": history,
        "loss_first": first,
        "loss_last": last,
        "loss_dropped": bool(last < first * 0.9),
        "loss_ratio": round(last / max(1e-12, first), 4),
    }


def infer_best_v3(pt_path: str, onnx_path: str, img_path: str, conf: float = 0.25) -> dict:
    """weights/best_v3.pt 走 onnx2torch 图，直接推理真实图并与 ORT 对比。"""
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    module = _rebuild_graph_module(payload)
    x = yolo_preprocess(img_path)
    with torch.no_grad():
        out = module(torch.from_numpy(x))
    if isinstance(out, (list, tuple)):
        out = out[0]
    t = out.numpy()[0]

    sess = ort_session(onnx_path)
    o = sess.run(None, {"images": x})[0][0]

    def boxes(a):
        m = a[:, 4] >= conf
        b = a[m]
        return b[np.argsort(-b[:, 4])]

    bt, bo = boxes(t), boxes(o)
    rows = []
    for i in range(min(len(bt), len(bo), 10)):
        rows.append(
            {
                "rank": i,
                "torch_xyxy": [round(float(v), 2) for v in bt[i, :4]],
                "ort_xyxy": [round(float(v), 2) for v in bo[i, :4]],
                "torch_conf": round(float(bt[i, 4]), 5),
                "ort_conf": round(float(bo[i, 4]), 5),
                "torch_cls": int(bt[i, 5]),
                "ort_cls": int(bo[i, 5]),
            }
        )
    return {
        "pt": pt_path,
        "image": img_path,
        "conf_thr": conf,
        "n_torch": int(len(bt)),
        "n_ort": int(len(bo)),
        "boxes": rows,
    }


def _rebuild_graph_module(payload):
    """从 payload 的 onnx_source 重建 onnx2torch 图并载入 state_dict。"""
    import tempfile
    import shutil

    from recover_weights import convert_onnx

    stage = os.path.join(_REPO, "weights", "_stage")
    os.makedirs(stage, exist_ok=True)
    module, _ = convert_onnx(payload["onnx_source"], stage)
    module.load_state_dict(payload["state_dict"])
    module.eval()
    return module


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=_REPO)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--steps-siamese", type=int, default=None)
    args = ap.parse_args()

    repo = args.repo
    out = {}
    si_steps = args.steps_siamese or args.steps

    print("=" * 70)
    print(f"[1/3] pre_model_v7.pt  warm-start 冒烟测试（过拟合单 batch，{si_steps} 步）")
    r = train_siamese(
        os.path.join(repo, "weights", "pre_model_v7.pt"),
        os.path.join(repo, "model", "pre_model_v7.onnx"),
        si_steps,
    )
    print(f"  load_state_dict missing={r['load_missing']} unexpected={r['load_unexpected']}")
    for h in r["history"]:
        print(f"  step {h['step']:>3}  loss={h['loss']:.6f}  lr={h['lr']:.2e}  grad_norm={h['grad_norm']:.4g}")
    print(f"  loss {r['loss_first']:.6f} -> {r['loss_last']:.6f} (best {r['loss_best']:.6f})  "
          f"ratio={r['loss_ratio']} best_ratio={r['loss_ratio_best']}  dropped={r['loss_dropped']}")
    out["siamese"] = r

    print("=" * 70)
    print(f"[2/3] best_v3_native.pt  warm-start 冒烟测试（ultralytics E2ELoss，{args.steps} 步）")
    r = train_yolo_native(
        os.path.join(repo, "weights", "best_v3_native.pt"),
        os.path.join(repo, "model", "best_v3.onnx"),
        args.steps,
    )
    print(f"  load_state_dict missing={r['load_missing']} unexpected={r['load_unexpected']}")
    print(f"  loss_names={r['loss_names']}")
    for h in r["history"]:
        print(f"  step {h['step']:>3}  loss={h['loss']:.6f}  vec={[round(v, 5) for v in h['loss_vec']]}  "
              f"grad_norm={h['grad_norm']:.4g}")
    print(f"  loss {r['loss_first']:.6f} -> {r['loss_last']:.6f}  "
          f"ratio={r['loss_ratio']}  dropped={r['loss_dropped']}")
    out["yolo_native"] = r

    print("=" * 70)
    print("[3/3] best_v3.pt  真实图推理（docs/res.jpg）")
    r = infer_best_v3(
        os.path.join(repo, "weights", "best_v3.pt"),
        os.path.join(repo, "model", "best_v3.onnx"),
        os.path.join(repo, "docs", "res.jpg"),
    )
    print(f"  有效框数 torch={r['n_torch']}  ort={r['n_ort']}")
    print(f"  {'rank':>4} {'torch xyxy':>34} {'conf':>7} {'cls':>3} | {'ort xyxy':>34} {'conf':>7} {'cls':>3}")
    for b in r["boxes"]:
        print(f"  {b['rank']:>4} {str(b['torch_xyxy']):>34} {b['torch_conf']:>7} {b['torch_cls']:>3} | "
              f"{str(b['ort_xyxy']):>34} {b['ort_conf']:>7} {b['ort_cls']:>3}")
    out["best_v3_infer"] = r

    print("=" * 70)
    ok1 = out["siamese"]["loss_dropped"]
    ok2 = out["yolo_native"]["loss_dropped"]
    ok3 = out["best_v3_infer"]["n_torch"] == out["best_v3_infer"]["n_ort"]
    print(f"结论：pre_model_v7 可训练={ok1}  best_v3_native 可训练={ok2}  best_v3 推理框数一致={ok3}")

    import json

    jp = os.path.join(repo, "out", "smoke_test.json")
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    print(f"[json] {jp}")
    return out


if __name__ == "__main__":
    main()
