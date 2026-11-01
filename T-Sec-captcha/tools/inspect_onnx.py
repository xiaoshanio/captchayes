# -*- coding: utf-8 -*-
"""inspect_onnx: 打印仓库内两个 ONNX 模型的完整签名，用于“继续训练”时对齐 I/O。

用法：
    python tools/inspect_onnx.py
    python tools/inspect_onnx.py model/best_v3.onnx model/pre_model_v7.onnx
"""
import os
import sys
import json
import numpy as np

import onnx
import onnxruntime as ort


def dump(path):
    print("=" * 78)
    print(f"MODEL: {path}  ({os.path.getsize(path)/1024/1024:.2f} MB)")
    print("=" * 78)

    m = onnx.load(path, load_external_data=False)
    g = m.graph
    print(f"ir_version={m.ir_version} opset={[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}")
    print(f"producer={m.producer_name} {m.producer_version}")
    if m.metadata_props:
        print("-- metadata_props --")
        for p in m.metadata_props:
            v = p.value
            print(f"  {p.key} = {v[:2000]}{'...' if len(v) > 2000 else ''}")

    print("-- graph inputs --")
    for i in g.input:
        t = i.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.shape.dim]
        print(f"  {i.name}: elem_type={t.elem_type} shape={dims}")
    print("-- graph outputs --")
    for o in g.output:
        t = o.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.shape.dim]
        print(f"  {o.name}: elem_type={t.elem_type} shape={dims}")

    from collections import Counter
    ops = Counter(n.op_type for n in g.node)
    print(f"-- nodes={len(g.node)} --")
    print("  " + ", ".join(f"{k}x{v}" for k, v in ops.most_common()))
    interesting = {"NonMaxSuppression", "TopK", "Sigmoid", "Split", "Concat", "Resize", "Cast",
                   "ReduceMax", "Gather", "Softmax", "Sub", "Mul", "Transpose", "ScatterND"}
    print("  nms-in-graph:", "NonMaxSuppression" in ops)
    print("  end2end-like (NonMaxSuppression+TopK):", {"NonMaxSuppression", "TopK"} <= set(ops))
    print("  preset-ops:", sorted(set(ops) & interesting))

    # 动态维度检测
    dyn = []
    for i in g.input:
        for d in i.type.tensor_type.shape.dim:
            if d.dim_param:
                dyn.append(f"{i.name}:{d.dim_param}")
    print("  dynamic axes:", dyn or "none (静态 shape)")

    print("-- ort session --")
    try:
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        for i in s.get_inputs():
            print(f"  IN  {i.name}: {i.type} {i.shape}")
        for o in s.get_outputs():
            print(f"  OUT {o.name}: {o.type} {o.shape}")
    except Exception as e:
        print("  FAIL:", repr(e))

    # 模型结构摘要（前 20 / 后 20 层名称）
    names = [f"{n.op_type}:{n.name}" for n in g.node]
    print("  head:", names[:8])
    print("  tail:", names[-8:])
    print()


if __name__ == "__main__":
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = sys.argv[1:] or [os.path.join(repo, "model", "best_v3.onnx"),
                             os.path.join(repo, "model", "pre_model_v7.onnx")]
    for p in paths:
        if os.path.exists(p):
            dump(p)
        else:
            print("MISSING:", p)
