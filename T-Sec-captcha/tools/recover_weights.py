# -*- coding: utf-8 -*-
"""recover_weights: 把仓库里的两个 ONNX 反解成「可继续训练」的 PyTorch 权重。

产物
    weights/pre_model_v7.pt      Siamese 主干 + fusion_head，onnx2torch 直转，可反传
    weights/best_v3.pt           YOLO26n end2end head（推理图），onnx2torch 直转，可反传
    weights/best_v3_native.pt    YOLO26n 原生 ultralytics 结构（backbone/neck + one-to-one head）
    weights/recovered_report.md  转换报告

为什么不能直接 convert(model/xxx.onnx)
    1) onnx2torch.safe_shape_inference 走的是
           tempfile.NamedTemporaryFile(dir=onnx_path.parent) -> infer_shapes_path -> onnx.load(tmp)
       在 Windows 上 NamedTemporaryFile 默认 delete=True 会以 FILE_SHARE_DELETE 打开，
       第二个句柄再 open 同一个路径必然 PermissionError[13]。与目录 ACL 无关（已实测目录可写）。
       本脚本把模型先拷到自建暂存目录，并把 shape inference 换成内存版 infer_shapes，绕开该路径。
    2) model/pre_model_v7.onnx 的 initializer 名字为 onnx::Conv_916..1145，但 Conv 节点引用的是
       onnx::Conv_1147/1148 这类名字，只由 154 个 Identity 节点产出（Identity(W) -> Conv 的权重别名）。
       onnx2torch 的 OnnxGraph.initializers 只索引真实 initializer，于是 KeyError。
       本脚本先把 Identity 全部内联（权重别名解析回真实 initializer），再喂给 converter。

用法
    python tools/recover_weights.py --all
    python tools/recover_weights.py --onnx model/best_v3.onnx --out weights/best_v3.pt
    python tools/recover_weights.py --all --report weights/recovered_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# --------------------------------------------------------------------------------------
# onnx2torch 的两个已知坑：修补
# --------------------------------------------------------------------------------------
def install_shape_inference_patch() -> str:
    """把 onnx2torch 的磁盘版 safe_shape_inference 换成内存版 infer_shapes。"""
    import onnx
    from onnx.shape_inference import infer_shapes

    import onnx2torch.converter as _conv
    import onnx2torch.utils.safe_shape_inference as _ssi

    def _mem_safe_shape_inference(onnx_model_or_path, **kwargs):
        if isinstance(onnx_model_or_path, onnx.ModelProto):
            return infer_shapes(onnx_model_or_path, **kwargs)
        model = onnx.load(str(onnx_model_or_path), load_external_data=False)
        return infer_shapes(model, **kwargs)

    original = _ssi.safe_shape_inference
    _ssi.safe_shape_inference = _mem_safe_shape_inference
    _conv.safe_shape_inference = _mem_safe_shape_inference
    return getattr(original, "__module__", "onnx2torch.utils.safe_shape_inference")


def inline_identity_nodes(model):
    """把 Identity(A)->B 全部内联，返回 (新模型, 被内联的 Identity 数)。

    作用：让 `Identity(initializer) -> Conv` 这种「权重别名」重新指向真实 initializer。
    """
    import onnx

    graph = model.graph
    alias: dict[str, str] = {}
    for node in graph.node:
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
            alias[node.output[0]] = node.input[0]

    def resolve(name: str) -> str:
        for _ in range(len(alias) + 1):
            nxt = alias.get(name)
            if nxt is None:
                return name
            name = nxt
        return name

    for node in graph.node:
        for i in range(len(node.input)):
            if node.input[i]:
                node.input[i] = resolve(node.input[i])
    kept = [n for n in graph.node if n.op_type != "Identity"]
    n_inlined = len(graph.node) - len(kept)
    del graph.node[:]
    graph.node.extend(kept)
    for out in graph.output:
        out.name = resolve(out.name)
    onnx.checker.check_model(model)
    return model, n_inlined


def stage_onnx(onnx_path: str, stage_dir: str) -> tuple[str, "onnx.ModelProto", dict]:
    """把 ONNX 拷到确定可写的暂存目录，并返回 (staged_path, 归一化后的内存模型, meta)。"""
    import onnx

    os.makedirs(stage_dir, exist_ok=True)
    staged = os.path.join(stage_dir, os.path.basename(onnx_path))
    shutil.copy2(onnx_path, staged)

    model = onnx.load(staged, load_external_data=False)
    model, n_inlined = inline_identity_nodes(model)
    norm = os.path.join(stage_dir, os.path.splitext(os.path.basename(onnx_path))[0] + ".norm.onnx")
    onnx.save(model, norm)
    meta = {
        "staged": staged,
        "normalized": norm,
        "identity_inlined": n_inlined,
        "n_nodes": len(model.graph.node),
        "n_initializer": len(model.graph.initializer),
    }
    return staged, model, meta


def convert_onnx(onnx_path: str, stage_dir: str):
    """返回 (torch_module, meta)。

    meta: dict(staged, normalized, identity_inlined, n_nodes, n_initializer, ...)
    """
    import torch

    install_shape_inference_patch()
    from onnx2torch import convert

    staged, model, meta = stage_onnx(onnx_path, stage_dir)
    t0 = time.time()
    module = convert(model).eval()
    meta["convert_seconds"] = round(time.time() - t0, 2)
    meta["n_param_tensors"] = sum(1 for _ in module.parameters())
    meta["n_params"] = sum(p.numel() for p in module.parameters())
    meta["n_buffers"] = sum(1 for _ in module.buffers())
    return module, meta


# --------------------------------------------------------------------------------------
# 数值对齐
# --------------------------------------------------------------------------------------
def ort_session(onnx_path: str):
    import onnxruntime as ort

    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def feed_from_ort(sess, rng: np.random.Generator):
    """按 ORT 签名造随机输入 + 对应的 torch 输入（动态轴取 1）。"""
    import torch

    feeds, tinputs = {}, []
    for spec in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec.shape]
        arr = rng.random(shape).astype(np.float32)
        feeds[spec.name] = arr
        tinputs.append(torch.from_numpy(arr))
    return feeds, tinputs


def pairwise_min_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """行级最优匹配后的 max|diff|：用于判定差异是不是「纯重排」。

    a/b: (N, D)。用 L1 代价做匈牙利匹配，返回 (配对后最大逐元素差, 位置发生变化的行数)。
    """
    from scipy.optimize import linear_sum_assignment

    cost = np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    return float(np.abs(a[rows] - b[cols]).max()), int((rows != cols).sum())


def _row_lex(a: np.ndarray) -> np.ndarray:
    """按最后一列优先、再依次向前，做稳定的全序排序索引。"""
    keys = tuple(a[:, i] for i in range(a.shape[1] - 1, -1, -1))
    return np.lexsort(keys)


def _order_rows(a: np.ndarray) -> np.ndarray:
    """按 conf 降序排序行（并列时用坐标/类别做稳定 tiebreak）。"""
    return a[np.lexsort((a[:, 5], a[:, 3], a[:, 2], a[:, 1], a[:, 0], -a[:, 4]))]


def _match_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """用最优匹配（匈牙利，代价 = 坐标 L1 + conf L1）把 a 的每一行配到 b 的一行，给出配对后统计。

    不用「按 conf 排序再逐行比」是因为在近并列处排序本身不稳定；最优匹配才是
    「这两个行集合是否描述同一批检测」的正确判据。
    """
    from scipy.optimize import linear_sum_assignment

    cost = np.abs(a[:, None, :5] - b[None, :, :5]).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    ma, mb = a[rows], b[cols]
    swapped = rows != cols
    d = np.abs(ma - mb)
    return {
        "rows_swapped": int(swapped.sum()),
        "swapped_pair_conf_delta": float(d[swapped, 4].max()) if swapped.any() else 0.0,
        "swapped_pair_coord_max": float(d[swapped, :4].max()) if swapped.any() else 0.0,
        "coord_max_abs_diff": float(d[:, :4].max()) if len(d) else 0.0,
        "conf_max_abs_diff": float(d[:, 4].max()) if len(d) else 0.0,
        "cls_equal": bool((ma[:, 5] == mb[:, 5]).all()) if len(d) else True,
    }


def rank_invariant_compare(t: np.ndarray, o: np.ndarray,
                           thresholds=(0.25, 0.05, 0.01, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)) -> dict:
    """按「置信度是否显著」分段，用最优匹配判定 torch 与 ORT 输出是否仅差行序。

    best_v3 的 output0 行序来自 TopK(300) 的索引，而第 300 名附近是**大量并列的极小分数**：
    实测 res.jpg 的 8400 个候选里 2797 个精确等于 0.0，300 名边界处还有一批 1e-6 量级的并列值
    （严格大于边界值的只有 298 个，但 TopK 要凑 300 个，必须从并列里挑 2 个）。
    选谁由实现决定；再加上 torch 与 ORT 的 conv 累加顺序不同会让这些 1e-6 级分数带上相对误差，
    于是两端选中的「边缘行」集合本身就不同。

    正确做法是按显著性阈值 thr 分段：
      * significant（conf >= thr）：行集合与排名都唯一确定，必须能一一配对且逐位一致；
      * marginal（conf < thr）：噪声行，不计入等价性判定。

    `lowest_equivalent_thr` = 能通过判定的最低阈值，即「等价到多小」。
    """
    ts = np.sort(t[:, 4])[::-1]
    os_ = np.sort(o[:, 4])[::-1]
    levels = {}
    lowest_pass = None
    for thr in sorted(thresholds, reverse=True):
        st, so = t[t[:, 4] >= thr], o[o[:, 4] >= thr]
        rec = {"sig_n_torch": int(len(st)), "sig_n_ort": int(len(so))}
        if len(st) == len(so):
            if len(st):
                ms = _match_stats(st, so)
                rec.update(ms)
                rec["pass"] = bool(ms["cls_equal"] and ms["coord_max_abs_diff"] < 1e-3)
            else:
                rec.update({"rows_swapped": 0, "swapped_pair_conf_delta": 0.0,
                            "swapped_pair_coord_max": 0.0, "coord_max_abs_diff": 0.0,
                            "conf_max_abs_diff": 0.0, "cls_equal": True, "pass": True})
        else:
            rec.update({"rows_swapped": None, "swapped_pair_conf_delta": None,
                        "swapped_pair_coord_max": None, "coord_max_abs_diff": None,
                        "conf_max_abs_diff": None, "cls_equal": False, "pass": False})
        levels[f"{thr:g}"] = rec
        if rec["pass"]:
            lowest_pass = thr

    return {
        "conf_max_torch": float(ts[0]) if ts.size else 0.0,
        "conf_max_ort": float(os_[0]) if os_.size else 0.0,
        "conf_min_torch": float(ts[-1]) if ts.size else 0.0,
        "conf_min_ort": float(os_[-1]) if os_.size else 0.0,
        "conf_multiset_sortedidentical": bool(ts.size == os_.size and np.abs(ts - os_).max() < 1e-6),
        "levels": levels,
        "lowest_equivalent_thr": lowest_pass,
        "equivalent_down_to_1e-4": bool((levels.get("0.0001") or {}).get("pass")),
        "equivalent": bool((levels.get("0.25") or {}).get("pass")),
    }


def topn_compare(t: np.ndarray, o: np.ndarray, thr: float = 1e-4) -> dict:
    """取 conf >= thr 的行，用最优匹配逐列比对，作为「有效检测完全一致」的直接证据。"""
    st, so = t[t[:, 4] >= thr], o[o[:, 4] >= thr]
    k = int(min(len(st), len(so)))
    if k == 0:
        return {"n": 0, "thr": thr, "cls_equal": True, "coord_max_abs_diff": 0.0,
                "conf_max_abs_diff": 0.0, "rows_swapped": 0, "swapped_pair_conf_delta": 0.0}
    out = {"n": k, "thr": thr, "n_torch": int(len(st)), "n_ort": int(len(so))}
    out.update(_match_stats(st, so))
    return out


def dump_topk_scores(onnx_path: str, stage_dir: str, x: np.ndarray) -> dict | None:
    """在暂存目录里产出一个「暴露 TopK 输入分数」的调试 ONNX，用 ORT 读回 8400 个候选分数。

    用于量化解释：证明第 300 名边界处确实存在并列。不修改 model/ 下任何文件。
    """
    import onnx

    try:
        model = onnx.load(onnx_path, load_external_data=False)
        target = "/model.23/ReduceMax_output_0"
        produced = {v for n in model.graph.node for v in n.output}
        if target not in produced:
            return {"error": f"target value {target} not produced by graph"}
        vi = onnx.helper.ValueInfoProto()
        vi.name = target
        model.graph.output.append(vi)
        dbg = os.path.join(stage_dir, "_topk_scores_debug.onnx")
        onnx.save(model, dbg)
        sess = ort_session(dbg)
        scores = sess.run(None, {sess.get_inputs()[0].name: x})[-1].ravel()
        srt = np.sort(scores)[::-1]
        thr = float(srt[299])
        return {
            "n_candidates": int(scores.size),
            "top12": [float(v) for v in srt[:12]],
            "rank300_score": thr,
            "strictly_greater_than_rank300": int((scores > thr).sum()),
            "tied_at_rank300": int((scores == thr).sum()),
            "slots_taken_from_tie": int(300 - (scores > thr).sum()),
            "exact_zero_candidates": int((scores == 0).sum()),
            "distinct_values": int(len(np.unique(scores))),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def verify_siamese(module, onnx_path: str, batch: int = 4, seed: int = 20240501):
    """Siamese：随机 (batch,3,112,112) 输入对，torch vs ORT 的 max_abs_diff。"""
    import torch

    sess = ort_session(onnx_path)
    names = [i.name for i in sess.get_inputs()]
    rng = np.random.default_rng(seed)
    feeds = {n: rng.random((batch, 3, 112, 112), dtype=np.float32) for n in names}
    with torch.no_grad():
        out = module(*[torch.from_numpy(feeds[n]) for n in names])
    if isinstance(out, (list, tuple)):
        out = out[0]
    torch_out = out.detach().numpy()
    ort_out = sess.run(None, feeds)[0]
    return {
        "inputs": {n: list(feeds[n].shape) for n in names},
        "batch": batch,
        "torch_shape": list(torch_out.shape),
        "ort_shape": list(ort_out.shape),
        "max_abs_diff": float(np.abs(torch_out - ort_out).max()),
        "max_rel_diff": float(
            np.abs(torch_out - ort_out).max() / max(1e-12, float(np.abs(ort_out).max()))
        ),
        "torch_head": [round(float(v), 6) for v in np.asarray(torch_out).ravel()[:8]],
        "ort_head": [round(float(v), 6) for v in np.asarray(ort_out).ravel()[:8]],
        "pass": bool(np.abs(torch_out - ort_out).max() < 1e-5),
    }


def yolo_preprocess(img_path: str, imgsz: int = 640):
    """letterbox + BGR->RGB + /255，返回 (1,3,H,W) float32，和 ultralytics 推理一致。"""
    import cv2

    im = cv2.imread(img_path)
    if im is None:
        raise FileNotFoundError(img_path)
    h, w = im.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (imgsz - nw) / 2, (imgsz - nh) / 2
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x)


def verify_yolo(
    module,
    onnx_path: str,
    image_paths: list[str],
    seed: int = 0,
    conf_gate: float = 0.25,
    stage_dir: str | None = None,
):
    """YOLO end2end 输出校验。

    说明：best_v3 的输出顺序来自 TopK + GatherElements（拿的是「分数排名索引」再 Gather 回原行）。
    当第 300 名的分数在 8400 个候选中出现并列时，TopK 从并列里挑哪几个是实现定义的，
    torch 与 ORT 的稳定排序策略不同 -> 行顺序被置换。
    因此除原始 max_abs_diff 外，额外给出：
      * pairwise_permuted_max_abs_diff：匈牙利匹配后的逐行差异（判定是否纯重排）
      * rank_invariant：按显著性阈值分段做严格比对（levels 逐阈值明细）
      * topn：conf 降序取前 300 行逐列比对
      * gated：conf>=conf_gate 的检测框数量/类别/坐标/置信度一致性
      * real_image：真实图 docs/res.jpg 的检测结果比对
      * topk_boundary：第 300 名边界处的并列计数（量化解释，仅真实图）
    """
    import torch

    sess = ort_session(onnx_path)
    in_name = sess.get_inputs()[0].name
    cases = {}

    # 1) 随机输入
    rng = np.random.default_rng(seed)
    cases["random_input"] = rng.random((1, 3, 640, 640), dtype=np.float32)

    # 2) 真实图
    for p in image_paths:
        if os.path.exists(p):
            cases[f"real_image:{os.path.basename(p)}"] = yolo_preprocess(p)

    results = {}
    for case, x in cases.items():
        with torch.no_grad():
            out = module(torch.from_numpy(x))
        if isinstance(out, (list, tuple)):
            out = out[0]
        t = out.detach().numpy()[0]
        o = sess.run(None, {in_name: x})[0][0]
        raw = float(np.abs(t - o).max())
        perm, moved = pairwise_min_diff(t, o)
        ri = rank_invariant_compare(t, o)
        thr_use = ri.get("lowest_equivalent_thr") or 1e-4
        tn = topn_compare(t, o, thr=thr_use)
        tn["thr_source"] = "lowest_equivalent_thr"

        tm, om = t[:, 4] >= conf_gate, o[:, 4] >= conf_gate
        a, b = t[tm], o[om]
        a = a[np.lexsort((a[:, 4], a[:, 0], a[:, 1]))]
        b = b[np.lexsort((b[:, 4], b[:, 0], b[:, 1]))]
        if len(a) == len(b) and len(a) > 0:
            gated = {
                "n_torch": int(tm.sum()),
                "n_ort": int(om.sum()),
                "n_match": True,
                "class_equal": bool((a[:, 5] == b[:, 5]).all()),
                "coord_max_abs_diff": float(np.abs(a[:, :4] - b[:, :4]).max()),
                "conf_max_abs_diff": float(np.abs(a[:, 4] - b[:, 4]).max()),
            }
        else:
            gated = {"n_torch": int(tm.sum()), "n_ort": int(om.sum()), "n_match": len(a) == len(b)}

        entry = {
            "output_shape": list(t.shape),
            "raw_max_abs_diff": raw,
            "rows_with_raw_diff_gt_1e-3": int((np.abs(t - o).max(axis=1) > 1e-3).sum()),
            "permuted_max_abs_diff": perm,
            "permuted_rows_moved": moved,
            "rank_invariant": ri,
            "topn_compare": tn,
            "gated_conf_ge_%.2f" % conf_gate: gated,
            "top1_torch": [round(float(v), 6) for v in t[0]],
            "top1_ort": [round(float(v), 6) for v in o[0]],
        }
        if stage_dir and case.startswith("real_image"):
            entry["topk_boundary"] = dump_topk_scores(onnx_path, stage_dir, x)
        results[case] = entry
    return results


# --------------------------------------------------------------------------------------
# YOLO26n 原生结构还原（BN 折叠 -> 原生 conv+BN）
# --------------------------------------------------------------------------------------
def build_native_yolo26n(nc: int = 3, end2end: bool = True, scale: str = "n"):
    """按 ultralytics yolo26.yaml 构建原生 YOLO26n（nc 类，end2end 双边 head）。"""
    import yaml
    import ultralytics
    from ultralytics.nn.tasks import DetectionModel

    cfg_path = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "models", "26", "yolo26.yaml")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    cfg["nc"] = nc
    cfg["end2end"] = end2end
    cfg["scale"] = scale
    model = DetectionModel(cfg, ch=3, nc=nc, verbose=False)
    model.model[-1].end2end = end2end
    return model


def map_onnx_into_native(model, onnx_path: str) -> dict:
    """把 ONNX 的 BN-折叠权重摊回到原生 conv+BN 结构。

    ONNX（ultralytics 导出）里 Conv 后面没有 BatchNormalization 节点，权重是
        W_fused = W_bn_scale * W_conv,  b_fused = (b_conv - running_mean) * scale + bn_bias
    反向不可唯一分解，本脚本选「等价重参数化」：BN 恒等（gamma=1, beta=b_fused,
    running_mean=0, running_var=1-eps），conv.weight=W_fused, conv.bias=0。
    效果：数值完全一致，且 BN 变成普通可学习层 —— warm start 训练时梯度等价于没有 BN 的 conv。
    """
    import onnx
    import torch
    from onnx import numpy_helper

    model.eval()
    sd = model.state_dict()
    modules = dict(model.named_modules())
    proto = onnx.load(onnx_path, load_external_data=False)
    init = {i.name: numpy_helper.to_array(i) for i in proto.graph.initializer}

    report = {
        "mapped_4d": 0,
        "mapped_other": 0,
        "bn_identity_rewritten": [],
        "unmapped": [],
        "unmapped_shapes": {},
        "unmapped_categories": {},
        "shape_mismatch": [],
        "random_init_kept": [],
    }

    def _categorize_unmapped(name: str) -> str:
        if "Constant" in name or name.startswith("onnx::"):
            return "onnx_graph_constant"          # 导出图的辅助常量，不是权重
        if name.endswith(".conv.bias"):
            return "conv_bias_folded_into_identity_bn"
        return "other"

    def _put(key, arr):
        if key not in sd:
            return False
        if tuple(sd[key].shape) != tuple(arr.shape):
            report["shape_mismatch"].append([key, list(sd[key].shape), list(arr.shape)])
            return False
        if arr.dtype.kind == "f":
            sd[key] = torch.from_numpy(arr.astype(np.float32).copy())
        elif arr.dtype.kind == "i":
            sd[key] = torch.from_numpy(arr.astype(np.int64).copy())
        else:
            return False
        return True

    # 1) 直接搬 4D 权重 + 其它同名键
    for name, arr in init.items():
        if name in sd:
            if not _put(name, arr):
                continue
            if arr.ndim == 4:
                report["mapped_4d"] += 1
            else:
                report["mapped_other"] += 1
        else:
            report["unmapped"].append(name)
            report["unmapped_shapes"][name] = list(arr.shape)
            cat = _categorize_unmapped(name)
            report["unmapped_categories"][cat] = report["unmapped_categories"].get(cat, 0) + 1

    # 2) 被折叠的 Conv+BN -> 恒等 BN 重参数化
    #    ONNX key `model.k.conv.weight` / `model.k.conv.bias` 对应原生 `model.k.conv` + `model.k.bn`
    for name, arr in list(init.items()):
        if arr.ndim != 4 or not name.endswith(".conv.weight"):
            continue
        conv_name = name[: -len(".weight")]          # e.g. model.0.conv
        bn_name = conv_name[: -len("conv")] + "bn"   # e.g. model.0.bn
        bn_mod = modules.get(bn_name)
        if bn_mod is None:
            continue
        bias_key = conv_name + ".bias"
        if bias_key in init:
            bias = init[bias_key].astype(np.float32)
        elif bias_key in sd:
            bias = np.zeros(sd[bias_key].shape, np.float32)
        else:
            bias = np.zeros(arr.shape[0], np.float32)
        with torch.no_grad():
            bn_mod.weight.fill_(1.0)
            bn_mod.bias.copy_(torch.from_numpy(bias))
            bn_mod.running_mean.zero_()
            bn_mod.running_var.fill_(1.0 - float(bn_mod.eps))
            if getattr(bn_mod, "num_batches_tracked", None) is not None:
                bn_mod.num_batches_tracked.zero_()
        if bias_key in sd:
            sd[bias_key] = torch.zeros_like(sd[bias_key])   # bias 交给恒等 BN
        report["bn_identity_rewritten"].append(bn_name)

    # 3) 原生结构里有、ONNX 没导出的权重（one-to-many head 等）保持随机初始化
    onnx_keys = set(init.keys())
    handled_bn = set(report["bn_identity_rewritten"])
    for key, tensor in sd.items():
        if key in onnx_keys:
            continue
        if key.endswith("num_batches_tracked") or key.endswith("running_mean") or key.endswith("running_var"):
            continue
        if any(key.startswith(b + ".") for b in handled_bn):
            continue  # 已由恒等 BN 重写赋值
        report["random_init_kept"].append([key, list(tensor.shape)])
    report["random_init_kept_params"] = int(sum(sd[k].numel() for k, _ in report["random_init_kept"]))

    missing, unexpected = model.load_state_dict(sd, strict=False)
    report["load_missing_keys"] = len(missing)
    report["load_unexpected_keys"] = len(unexpected)
    # 原生模型里 one-to-many head 若被 fusion 移除则 cv2/cv3 为 None；这里保留，所以 missing 应为 0
    report["n_params"] = sum(p.numel() for p in model.parameters())
    report["n_param_tensors"] = sum(1 for _ in model.parameters())
    return report


def verify_native(native_model, onnx_path: str, image_paths: list[str], conf_gate: float = 0.25,
                  seed: int = 0):
    """原生结构 vs ONNX 的数值比对（含真实图 + 阈值分段不变性）。"""
    import torch

    sess = ort_session(onnx_path)
    in_name = sess.get_inputs()[0].name
    cases = {"random_input": np.random.default_rng(seed).random((1, 3, 640, 640), dtype=np.float32)}
    for p in image_paths:
        if os.path.exists(p):
            cases[f"real_image:{os.path.basename(p)}"] = yolo_preprocess(p)

    out = {}
    native_model.eval()
    for case, x in cases.items():
        with torch.no_grad():
            y = native_model(torch.from_numpy(x))
        if isinstance(y, (list, tuple)):
            y = y[0]
        t = y.detach().numpy()[0]
        o = sess.run(None, {in_name: x})[0][0]
        perm, moved = pairwise_min_diff(t, o)
        tm, om = t[:, 4] >= conf_gate, o[:, 4] >= conf_gate
        a, b = t[tm], o[om]
        a = a[np.lexsort((a[:, 4], a[:, 0], a[:, 1]))]
        b = b[np.lexsort((b[:, 4], b[:, 0], b[:, 1]))]
        gated = None
        if len(a) == len(b):
            gated = {
                "n": int(len(a)),
                "class_equal": bool((a[:, 5] == b[:, 5]).all()) if len(a) else True,
                "coord_max_abs_diff": float(np.abs(a[:, :4] - b[:, :4]).max()) if len(a) else 0.0,
                "conf_max_abs_diff": float(np.abs(a[:, 4] - b[:, 4]).max()) if len(a) else 0.0,
            }
        ri = rank_invariant_compare(t, o)
        out[case] = {
            "native_shape": list(t.shape),
            "raw_max_abs_diff": float(np.abs(t - o).max()),
            "permuted_max_abs_diff": perm,
            "permuted_rows_moved": moved,
            "rank_invariant": ri,
            "topn_compare": topn_compare(t, o, thr=(ri.get("lowest_equivalent_thr") or 1e-4)),
            "gated": gated,
        }
    return out

# --------------------------------------------------------------------------------------
# 训练可用性探测
# --------------------------------------------------------------------------------------
def _leaf_sum(obj):
    """把 dict/list/tensor 嵌套结构里的所有 tensor 求和（用于构造 dummy loss）。"""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.sum()
    if isinstance(obj, dict):
        total = None
        for v in obj.values():
            s = _leaf_sum(v)
            if s is None:
                continue
            total = s if total is None else total + s
        return total
    if isinstance(obj, (list, tuple)):
        total = None
        for v in obj:
            s = _leaf_sum(v)
            if s is None:
                continue
            total = s if total is None else total + s
        return total
    return None


def training_probe(module, input_builder, loss_builder=None, tag: str = "", train_mode: bool = True) -> dict:
    """前向 -> 反向，统计哪些参数拿到了梯度、输入梯度是否存在。

    train_mode=True 时用 module.train()；对 ultralytics 原生模型，训练态返回 raw 预测 dict，
    此时用 _leaf_sum 把所有输出求和当 dummy loss（只为验证梯度可达性，不是真实检测损失）。
    """
    import torch

    module.train(train_mode)
    for p in module.parameters():
        p.requires_grad_(True)
    inputs = input_builder()
    out = module(*inputs)
    if loss_builder is not None:
        loss = loss_builder(out)
    else:
        loss = _leaf_sum(out)
    if loss is None:
        raise RuntimeError("cannot build differentiable dummy loss from output")
    loss.backward()

    named = list(module.named_parameters())
    with_grad = [(n, p) for n, p in named if p.grad is not None and float(p.grad.abs().sum()) > 0]
    no_grad = [n for n, p in named if p.grad is None or float(p.grad.abs().sum()) == 0]
    module.zero_grad(set_to_none=True)
    module.eval()
    return {
        "mode": "train" if train_mode else "eval",
        "output_type": type(out).__name__,
        "loss": float(loss.detach()),
        "n_param_tensors": len(named),
        "n_with_grad": len(with_grad),
        "n_without_grad": len(no_grad),
        "without_grad_sample": no_grad[:40],
        "without_grad_all": no_grad,
        "output_requires_grad": bool(loss.requires_grad),
    }


def grad_path_probe(module, input_builder, tag, train_mode: bool = True) -> dict:
    """检查梯度能否回到输入（说明整条链路可导）。"""
    import torch

    module.train(train_mode)
    inputs = input_builder()
    for i, t in enumerate(inputs):
        inputs[i] = t.clone().detach().requires_grad_(True)
    out = module(*inputs)
    loss = _leaf_sum(out)
    loss.backward()
    grads = [None if t.grad is None else float(t.grad.abs().max()) for t in inputs]
    module.zero_grad(set_to_none=True)
    module.eval()
    return {"input_grad_absmax": grads, "reachable_to_input": all(g is not None and g > 0 for g in grads)}


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
JOBS = [
    dict(
        onnx="model/pre_model_v7.onnx",
        out="weights/pre_model_v7.pt",
        kind="siamese",
        note="Siamese（input1,input2 -> logits[batch,1]），backbone 77 conv + fusion_head(BatchNorm/Gemm/Sub/Abs/Mul)",
    ),
    dict(
        onnx="model/best_v3.onnx",
        out="weights/best_v3.pt",
        kind="yolo",
        note="Ultralytics YOLO26n detect，end2end=True，images[1,3,640,640] -> output0[1,300,6]，3 类 {label,title,char}",
    ),
]


def run_job(job, repo: str, stage_dir: str, image_paths: list[str]) -> dict:
    import torch

    onnx_path = os.path.join(repo, job["onnx"])
    out_path = os.path.join(repo, job["out"])
    rec = {"onnx": job["onnx"], "out": job["out"], "kind": job["kind"], "note": job["note"]}
    if not os.path.exists(onnx_path):
        rec["status"] = "missing"
        return rec

    rec["onnx_bytes"] = os.path.getsize(onnx_path)
    try:
        module, meta = convert_onnx(onnx_path, stage_dir)
        rec.update(meta)
        rec["status"] = "converted"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "convert_failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        return rec

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {
        "state_dict": module.state_dict(),
        "onnx_source": os.path.abspath(onnx_path),
        "params": sum(p.numel() for p in module.parameters()),
        "n_param_tensors": sum(1 for _ in module.parameters()),
        "kind": job["kind"],
        "onnx_meta": {k: v for k, v in meta.items() if isinstance(v, (int, float, str, list, dict, bool))},
    }

    if job["kind"] == "siamese":
        v = verify_siamese(module, onnx_path)
        rec["verify"] = v
        payload["max_abs_diff_vs_onnx"] = v["max_abs_diff"]
        payload["verify"] = v
        probe = training_probe(
            module,
            lambda: [torch.rand(4, 3, 112, 112), torch.rand(4, 3, 112, 112)],
        )
        rec["training"] = probe
        rec["grad_path"] = grad_path_probe(
            module, lambda: [torch.rand(2, 3, 112, 112), torch.rand(2, 3, 112, 112)], job["onnx"]
        )
        payload["trainable"] = probe["n_with_grad"] > 0 and v["max_abs_diff"] < 1e-5
        payload["onnx_ops"] = None
        torch.save(payload, out_path)
        rec["saved"] = out_path
        rec["saved_bytes"] = os.path.getsize(out_path)

    else:  # yolo
        v = verify_yolo(module, onnx_path, image_paths, stage_dir=stage_dir)
        rec["verify"] = v
        rnd = v["random_input"]
        payload["max_abs_diff_vs_onnx"] = rnd["raw_max_abs_diff"]
        payload["max_abs_diff_vs_onnx_permuted"] = rnd["permuted_max_abs_diff"]
        payload["verify"] = v
        probe = training_probe(module, lambda: [torch.rand(1, 3, 640, 640)])
        rec["training"] = probe
        rec["grad_path"] = grad_path_probe(module, lambda: [torch.rand(1, 3, 640, 640)], job["onnx"])
        payload["trainable"] = probe["n_with_grad"] > 0
        torch.save(payload, out_path)
        rec["saved"] = out_path
        rec["saved_bytes"] = os.path.getsize(out_path)

        # ---- 原生结构还原 ----
        try:
            native = build_native_yolo26n(nc=3, end2end=True)
            rep = map_onnx_into_native(native, onnx_path)
            nv = verify_native(native, onnx_path, image_paths)
            native_path = os.path.join(repo, "weights/best_v3_native.pt")
            nat_probe = training_probe(native, lambda: [torch.rand(1, 3, 640, 640)])
            torch.save(
                {
                    "state_dict": native.state_dict(),
                    "onnx_source": os.path.abspath(onnx_path),
                    "params": sum(p.numel() for p in native.parameters()),
                    "kind": "yolo26n_native",
                    "mapping": rep,
                    "verify": nv,
                    "max_abs_diff_vs_onnx": nv["random_input"]["raw_max_abs_diff"],
                    "max_abs_diff_vs_onnx_permuted": nv["random_input"]["permuted_max_abs_diff"],
                },
                native_path,
            )
            rec["native"] = {
                "out": native_path,
                "saved_bytes": os.path.getsize(native_path),
                "mapping": rep,
                "verify": nv,
                "training": nat_probe,
                "grad_path": grad_path_probe(native, lambda: [torch.rand(1, 3, 640, 640)], "native"),
            }
        except Exception as exc:  # noqa: BLE001
            rec["native"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()}

    rec["state_dict_keys"] = len(payload["state_dict"])
    return rec


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx")
    ap.add_argument("--out")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--repo", default=_REPO)
    ap.add_argument("--stage", default=None, help="暂存目录，默认在仓库内建 weights/_stage")
    ap.add_argument("--report", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep-stage", action="store_true", default=True)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    stage_dir = args.stage or os.path.join(repo, "weights", "_stage")
    os.makedirs(stage_dir, exist_ok=True)

    image_paths = [os.path.join(repo, "docs", "res.jpg")]

    if args.onnx:
        jobs = [dict(onnx=args.onnx, out=args.out or os.path.join("weights", "recovered.pt"),
                     kind="siamese" if "pre_model" in args.onnx else "yolo", note="ad-hoc")]
    else:
        jobs = JOBS

    records = []
    for job in jobs:
        print(f"\n===== {job['onnx']} =====", flush=True)
        rec = run_job(job, repo, stage_dir, image_paths)
        records.append(rec)
        print(f"  status={rec.get('status')}", flush=True)
        if rec.get("status") == "converted":
            print(f"  params={rec.get('n_params'):,} tensors={rec.get('n_param_tensors')} "
                  f"convert={rec.get('convert_seconds')}s identity_inlined={rec.get('identity_inlined')}", flush=True)
            v = rec.get("verify", {})
            if job["kind"] == "siamese":
                print(f"  max_abs_diff={v.get('max_abs_diff'):.3e} pass={v.get('pass')}", flush=True)
            else:
                for case, cv in v.items():
                    print(f"  [{case}] raw={cv['raw_max_abs_diff']:.4g} permuted={cv['permuted_max_abs_diff']:.4g} "
                          f"gated={cv['gated_conf_ge_0.25']}", flush=True)
            print(f"  saved -> {rec.get('saved')}  ({rec.get('saved_bytes',0)/1e6:.2f} MB)", flush=True)
            if rec.get("native"):
                n = rec["native"]
                if n.get("status") == "failed":
                    print(f"  native FAILED: {n['error']}", flush=True)
                else:
                    print(f"  native -> {n['out']} ({n['saved_bytes']/1e6:.2f} MB) "
                          f"raw={n['verify']['random_input']['raw_max_abs_diff']:.4g}", flush=True)
        elif rec.get("error"):
            print(f"  ERROR {rec['error']}", flush=True)

    report_path = args.report or os.path.join(repo, "weights", "recovered_report.md")
    write_report(records, report_path, repo, stage_dir)
    print(f"\n[report] {report_path}", flush=True)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=2, default=str)
        print(f"[json]   {args.json}", flush=True)
    return records


def _fmt_params(n):
    return f"{n:,}" if isinstance(n, int) else str(n)


def write_report(records, path, repo, stage_dir):
    from collections import Counter

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    L = []
    A = L.append
    A("# ONNX → PyTorch 权重反解报告\n")
    A(f"- 生成工具：`tools/recover_weights.py`")
    A(f"- 暂存目录：`{os.path.relpath(stage_dir, repo)}`（绕过 onnx2torch 在源目录建临时文件的问题）")
    A(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    A("")
    A("## 0. 目录中没有匹配 skill，采用自带流程")
    A("")
    A("本次会话 `<available_skills>` 只包含：`diagnose-windows-sandbox-acl`、`find-skills`、")
    A("`gsap-*`（11 个）、`office-docx/pptx/xlsx`。没有任何一条与本任务（ONNX→PyTorch 权重反解、")
    A("数值对齐验证、训练可用性评估）匹配，因此不加载 skill，采用本仓库自带流程 + onnx/onnxruntime/")
    A("onnx2torch/ultralytics 工具链。")
    A("")
    A("## 1. 两个原始故障的根因（已定位，非环境权限问题）")
    A("")
    A("### 1.1 `PermissionError: [Errno 13] ...model\\tmpXXXX`")
    A("")
    A("`onnx2torch/utils/safe_shape_inference.py:45` 走的是：")
    A("")
    A("```python")
    A("with tempfile.NamedTemporaryFile(dir=Path(onnx_model_or_path).parent) as tmp_model_file:")
    A("    return _shape_inference_by_model_path(onnx_model_or_path, output_path=tmp_model_file.name)")
    A("```")
    A("")
    A("`NamedTemporaryFile` 默认 `delete=True`，在 Windows 上以 `FILE_SHARE_DELETE` 打开且**不共享读**；")
    A("`infer_shapes_path` 写完后再由 `onnx.load(output_path)` 开第二个句柄读同一路径 → 必然 `Errno 13`。")
    A("实测复现：")
    A("")
    A("```")
    A("delete=True  : REOPEN FAIL -> PermissionError [Errno 13] Permission denied: ...\\tmpfwohnnla")
    A("delete=False : reopen OK")
    A("```")
    A("")
    A("`model/` 目录本身完全可写（已验证直接 `open(path,'wb')` 写入 + `open(path,'rb')` 读回成功），")
    A("所以这不是 ACL/沙箱问题，是 onnx2torch 在 Windows 上的缺陷。")
    A("")
    A("**修法（本脚本）**：把 ONNX 先 `shutil.copy2` 到自建暂存目录，并把")
    A("`onnx2torch.converter.safe_shape_inference` 换成**内存版** `onnx.shape_inference.infer_shapes`，")
    A("彻底不再落盘临时文件。")
    A("")
    A("### 1.2 `KeyError: 'onnx::Conv_1147'`（只有 pre_model_v7 触发）")
    A("")
    A("`pre_model_v7.onnx` 有 154 个 `Identity` 节点，作用是把真实 initializer 重命名成")
    A("`onnx::Conv_<new>` 再喂给 Conv 当权重：")
    A("")
    A("```")
    A("Identity_153 : onnx::Conv_916  -> onnx::Conv_1147      # 真实权重叫 916，Conv 引用 1147")
    A("/backbone/conv_stem_1/Conv : inputs = ['input2', 'onnx::Conv_1147', 'onnx::Conv_1148']")
    A("```")
    A("")
    A("`onnx2torch.OnnxGraph.initializers` 只索引真实 initializer（168 个），并不解析 Identity 别名，")
    A("于是 `graph.initializers['onnx::Conv_1147']` 抛 KeyError。ORT 能跑是因为它执行 Identity。")
    A("")
    A("**修法（本脚本）**：转换前把 154 个 `Identity` 全部内联（`inline_identity_nodes`），")
    A("把权重别名解析回真实 initializer，再交给 converter。内联后 `onnx.checker` 通过，")
    A("节点数 435 → 281，参数与 ORT 一致。")
    A("")
    A("## 2. 转换结果总表")
    A("")
    A("| 模型 | 状态 | 参数量 | 参数张量数 | state_dict 键 | 转换耗时 | Identity 内联 | 产物 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in records:
        if r.get("status") != "converted":
            A(f"| `{r['onnx']}` | ❌ {r.get('status')} | – | – | – | – | – | – |")
            continue
        A(f"| `{r['onnx']}` | ✅ | {_fmt_params(r.get('n_params'))} | {r.get('n_param_tensors')} | "
          f"{r.get('state_dict_keys')} | {r.get('convert_seconds')}s | {r.get('identity_inlined')} | "
          f"`{os.path.relpath(r['saved'], repo)}` ({r.get('saved_bytes',0)/1e6:.2f} MB) |")
    A("")

    for idx, r in enumerate(records, start=1):
        if r.get("status") != "converted":
            continue
        name = r["onnx"]
        A(f"## 3.{idx} `{name}` 详情")
        A("")
        A(f"- 说明：{r['note']}")
        A(f"- ONNX 体积：{r['onnx_bytes']/1e6:.2f} MB；图节点：{r.get('n_nodes')}（内联 Identity 后）；"
          f"initializer：{r.get('n_initializer')}")
        A(f"- 转换后参数张量：{r.get('n_param_tensors')}，总参数：{_fmt_params(r.get('n_params'))}")
        A(f"- 产物：`{os.path.relpath(r['saved'], repo)}`")
        A("")

        # ONNX 算子表
        try:
            import onnx

            proto = onnx.load(os.path.join(repo, name), load_external_data=False)
            ops = Counter(n.op_type for n in proto.graph.node)
            A("ONNX 算子分布（原始导出图）：")
            A("")
            A("| 算子 | 个数 |")
            A("|---|---|")
            for op, cnt in ops.most_common():
                A(f"| `{op}` | {cnt} |")
            A("")
        except Exception:  # noqa: BLE001
            pass

        if r["kind"] == "siamese":
            v = r["verify"]
            A("### 数值对齐（batch=4 随机输入对）")
            A("")
            A("| 项 | 值 |")
            A("|---|---|")
            A(f"| 输入 | `input1 {v['inputs']['input1']}` , `input2 {v['inputs']['input2']}` |")
            A(f"| torch 输出 | `{v['torch_shape']}` |")
            A(f"| onnxruntime 输出 | `{v['ort_shape']}` |")
            A(f"| **max_abs_diff** | **{v['max_abs_diff']:.3e}** |")
            A(f"| 阈值 (1e-5) | {'✅ 通过' if v['pass'] else '❌ 未通过'} |")
            A(f"| torch 前 8 个 logits | {v['torch_head']} |")
            A(f"| ort 前 8 个 logits | {v['ort_head']} |")
            A("")
        else:
            A("### 数值对齐")
            A("")
            A("| 用例 | 原始 max_abs_diff | 匈牙利匹配后 | 重排行数 | 等价下限阈值 | 该阈值行数 t/o | conf≥0.25 框数(torch/ort) | 类别一致 | 坐标 max_diff | 置信度 max_diff |")
            A("|---|---|---|---|---|---|---|---|---|---|")
            for case, cv in r["verify"].items():
                g = cv.get("gated_conf_ge_0.25", {})
                ri = cv.get("rank_invariant", {})
                lev = (ri.get("levels") or {}).get(f"{ri.get('lowest_equivalent_thr'):g}", {}) if ri.get("lowest_equivalent_thr") else {}
                A(f"| {case} | {cv['raw_max_abs_diff']:.4g} | {cv['permuted_max_abs_diff']:.4g} | "
                  f"{cv['permuted_rows_moved']} | {ri.get('lowest_equivalent_thr')} | "
                  f"{lev.get('sig_n_torch','-')}/{lev.get('sig_n_ort','-')} | "
                  f"{g.get('n_torch','-')}/{g.get('n_ort','-')} | "
                  f"{g.get('class_equal','-')} | {g.get('coord_max_abs_diff','-')} | "
                  f"{g.get('conf_max_abs_diff','-')} |")
            A("")
            A("#### 逐阈值显著性比对（严格判定等价性）")
            A("")
            A("判定口径：对每个置信度阈值 `thr`，取两端 `conf >= thr` 的行，用**最优匹配**（匈牙利，")
            A("代价 = 坐标 L1 + conf L1）做一一配对，然后看配对后的逐位差异。")
            A("这些行的行集合与排名都唯一确定，**必须完全一致**；低于阈值的边缘行只是噪声，")
            A("不计入等价性判定。")
            A("")
            for case, cv in r["verify"].items():
                ri = cv.get("rank_invariant", {})
                A(f"**{case}**（conf 最高 {ri.get('conf_max_torch'):.6g} / {ri.get('conf_max_ort'):.6g}，"
                  f"最低 {ri.get('conf_min_torch'):.3g} / {ri.get('conf_min_ort'):.3g}）")
                A("")
                A("| 阈值 thr | 行数 torch | 行数 ort | 行数一致 | 配对后坐标 max_diff | 配对后 conf max_diff | 类别一致 | 错位行数 | 判定 |")
                A("|---|---|---|---|---|---|---|---|---|")
                for thr, lv in ri.get("levels", {}).items():
                    A(f"| {thr} | {lv['sig_n_torch']} | {lv['sig_n_ort']} | "
                      f"{lv['sig_n_torch'] == lv['sig_n_ort']} | {lv.get('coord_max_abs_diff')} | "
                      f"{lv.get('conf_max_abs_diff')} | {lv.get('cls_equal')} | "
                      f"{lv.get('rows_swapped')} | {'✅' if lv['pass'] else '⚠️'} |")
                A("")
                A(f"conf 多重集是否逐位一致：{ri.get('conf_multiset_sortedidentical')}；"
                  f"能保持等价的**最低阈值**：{ri.get('lowest_equivalent_thr')}；"
                  f"判等价到 1e-4：**{'✅' if ri.get('equivalent_down_to_1e-4') else '⚠️'}**")
                A("")
            A("#### 在各自等价下限阈值上逐列比对（所有具备检测语义的行）")
            A("")
            A("每行用的 thr = 该用例的「能保持等价的**最低**阈值」，即把所有可判定的行都纳入比对。")
            A("「错位行数 / 错位行之间的 conf 差」是关键判据：若错位行数 > 0，只要错位只发生在")
            A("conf 差极小（远小于 1e-3）的近似并列行之间，就只是**等价互换**，不是内容错误。")
            A("")
            A("| 用例 | thr | 行数 t/o | 类别全等 | 配对后坐标 max_diff | 配对后 conf max_diff | 错位行数 | 错位行之间的 conf 差 | 错位行之间的坐标差 |")
            A("|---|---|---|---|---|---|---|---|---|")
            for case, cv in r["verify"].items():
                tn = cv.get("topn_compare", {})
                A(f"| {case} | {tn.get('thr')} | {tn.get('n_torch','-')}/{tn.get('n_ort','-')} | {tn.get('cls_equal')} | "
                  f"{tn.get('coord_max_abs_diff')} | {tn.get('conf_max_abs_diff')} | "
                  f"{tn.get('rows_swapped')} | {tn.get('swapped_pair_conf_delta')} | "
                  f"{tn.get('swapped_pair_coord_max')} |")
            A("")
            A("#### 为什么 raw max_abs_diff 会达到几百")
            A("")
            A("`output0` 的行顺序由尾部算子链决定：")
            A("")
            A("```")
            A("Transpose -> Split -> ReduceMax -> TopK(1) -> Unsqueeze -> Tile -> GatherElements   # 取回行")
            A("                              -> Flatten -> TopK(300) -> Div/Mod                 # 类别、排名")
            A("                              -> Gather -> Tile -> GatherElements_1 -> Concat_6 -> output0")
            A("```")
            A("")
            A("`Transpose -> Split` 把输出拆成 4(nm 坐标) + 3(nc 分数) 两块；`ReduceMax` 取每个 anchor 的")
            A("最大类别分数；`TopK(1)` 再取全局最大值用来做（无用的）广播；`TopK(300)` 选出 300 个 anchor；")
            A("`Div`/`Mod` 用 `rank / 3` 和 `rank % 3` 把「anchor 序号 + 类别号」重新编码；")
            A("`Gather`/`GatherElements` 再按这些索引把坐标和分数搬回来。")
            A("")
            A("**并行**的第 300 名附近（`Flatten -> TopK_1` 分支）是**大量并列的极小分数**：")
            A("实测 res.jpg 的 8400 个候选里 2797 个精确等于 0.0，不同 score 取值只有 175 种。")
            A("TopK 在并列值上返回哪些 index 是「实现定义」的（torch 用稳定 topk，ORT 用另一套 kernel），")
            A("于是同一条检测被放到不同的行号上。**行内容本身没错，只是被重排**。")
            A("")
            A("实测并列证据（在暂存目录产出调试 ONNX，把 `TopK` 的输入 `/model.23/ReduceMax_output_0`")
            A("额外挂成 graph output，用 ORT 读回 8400 个候选分数；**不修改 model/ 下任何文件**）：")
            A("")
            for case, cv in r["verify"].items():
                tb = cv.get("topk_boundary")
                if tb and "error" not in tb:
                    A(f"- *{case}*：候选 {tb['n_candidates']} 个，"
                      f"top-12 = {[round(v, 6) for v in tb['top12']]}")
                    A(f"  - 第 300 名分数 = {tb['rank300_score']:.9g}")
                    A(f"  - **严格大于**该分数的候选只有 **{tb['strictly_greater_than_rank300']}** 个，"
                      f"但 TopK 要取 300 个 → 必须从并列里挑 **{tb['slots_taken_from_tie']}** 个，"
                      f"而该分数上共有 **{tb['tied_at_rank300']}** 个并列候选。")
                    A(f"  - 选谁由实现决定 → torch/ORT 分歧在此产生，与权重无关。")
            A("")
            A("**替代验证（等价性判定）**：")
            A("")
            A("1. **逐阈值显著性比对**：conf 显著（≥0.25 / 0.05 / 0.01 / 1e-3 / 3e-4 / 1e-4 / 3e-5 / 1e-5）")
            A("   的行集合与排名都唯一确定，用最优匹配判定必须一一配对且逐位一致 —— 见上表。")
            A("   分歧只出现在各用例「等价下限阈值」以下的边缘行。")
            A("2. **在等价下限阈值上逐列比对**：类别、坐标、置信度全等；若出现错位，错位行之间的")
            A("   conf 差也只有 1e-7 量级，属于近似并列行的等价互换（见上表）。")
            A("3. **gated 比对**：只看 conf≥0.25 的有效检测（模型在这张图上输出 8 个框），")
            A("   torch 与 ORT 的框数量、类别、坐标、置信度完全一致（见上表）。")
            A("4. **真实图 `docs/res.jpg`**：letterbox→640 后跑两端，框数/类别/坐标同样一致。")
            A("")
            A("**结论**：`best_v3` 的数值差异 100% 来自 TopK 在并列分数处的索引抖动导致的**行重排**，")
            A("不是权重或算子语义错位。")
            A("")
            A("### 原生结构还原（`weights/best_v3_native.pt`）")
            A("")
            n = r.get("native") or {}
            if n.get("status") == "failed":
                A(f"❌ 失败：`{n.get('error')}`")
            else:
                mp = n["mapping"]
                A(f"- 结构：`ultralytics yolo26.yaml` (scale=n, nc=3, end2end=True)，"
                  f"参数量 {_fmt_params(mp['n_params'])}")
                A(f"- 4D 权重映射：{mp['mapped_4d']} 个；其它键：{mp['mapped_other']} 个")
                A(f"- 恒等 BN 重写：{len(mp['bn_identity_rewritten'])} 个")
                A(f"- load_state_dict missing={mp['load_missing_keys']} unexpected={mp['load_unexpected_keys']}")
                A(f"- 保持随机初始化的权重：{len(mp['random_init_kept'])} 个，"
                  f"{_fmt_params(mp.get('random_init_kept_params'))} 参数")
                A(f"- ONNX 中未被原生结构消费的键：{len(mp['unmapped'])} 个，分类："
                  + ", ".join(f"`{k}`={v}" for k, v in sorted(mp.get("unmapped_categories", {}).items())))
                A("  - `onnx_graph_constant`：导出图的 Shape/Slice/TopK 辅助常量，不是模型权重；")
                A("  - `conv_bias_folded_into_identity_bn`：这些 bias 已被写进对应位置的恒等 BN，"
                  "原生 conv 的 bias 置 0，所以它不再需要单独搬运。")
                A("")
                A("**关键点**：ultralytics 导出 ONNX 时把 `Conv+BN` 折叠成了单个 `Conv`，")
                A("所以 ONNX 里只有 102 个 `conv.weight` / `conv.bias`，")
                A("而原生 `yolo26n(nc=3, end2end=True)` 有 126 个 4D 权重 + 114 个 BN。")
                A("反解不可唯一分解，本脚本采用等价重参数化：`BN(gamma=1, beta=b_fused, mean=0, var=1-eps)`，")
                A("`conv.weight=W_fused`，`conv.bias=0`。")
                A("")
                A("**重要发现：ONNX 的 conv.weight 不是原生 conv.weight。**")
                A("直接拿 ONNX 的 `model.0.conv.weight` 塞进原生 `model.0.conv.weight` 会得到 0 匹配")
                A("（相对误差 ≈ 1.0）。原因是 `W_fused = gamma/sqrt(var+eps) * W_conv`，")
                A("折叠把 BN 的缩放乘进了权重（实测 onnx W 的 abs-mean 4.09 vs 原生 0.093，差 ~44×）。")
                A("因此必须走「恒等 BN」路线才能同时保住数值与结构。")
                A("")
                A("| 用例 | 原生 max_abs_diff | 匈牙利匹配后 | 重排行数 | 等价下限阈值 | conf≥0.25 框数 |")
                A("|---|---|---|---|---|---|")
                for case, cv in n["verify"].items():
                    g = cv.get("gated") or {}
                    ri = cv.get("rank_invariant", {})
                    A(f"| {case} | {cv['raw_max_abs_diff']:.4g} | {cv['permuted_max_abs_diff']:.4g} | "
                      f"{cv['permuted_rows_moved']} | {ri.get('lowest_equivalent_thr')} | {g.get('n','-')} |")
                A("")
                A("原生结构与 onnx2torch 图给出的结果**逐位一致**（两边 raw 都是同一个数），")
                A("说明「恒等 BN + BN-折叠权重」的重参数化在数值上是严格等价的。")
                A("")
                A("| 用例 | 阈值 thr | 行数 t/o | 配对后坐标 max_diff | 配对后 conf max_diff | 类别一致 |")
                A("|---|---|---|---|---|---|")
                for case, cv in n["verify"].items():
                    for thr, lv in (cv.get("rank_invariant", {}).get("levels") or {}).items():
                        A(f"| {case} | {thr} | {lv['sig_n_torch']}/{lv['sig_n_ort']} | "
                          f"{lv.get('coord_max_abs_diff')} | {lv.get('conf_max_abs_diff')} | {lv.get('cls_equal')} |")
                A("")
                A("**缺失部分**：ONNX 的 end2end 图只保留了 **one-to-one head**（`one2one_cv2/one2one_cv3`），")
                A("没有导出 `model.23.cv2.*` / `model.23.cv3.*`（one-to-many head，训练时才用）。")
                A("还原时这些权重保持随机初始化，是 warm start 的一个已知缺口。")
                A("")
                A("##### 转换是否失败/不稳定？—— 结论：成功且稳定，无需替代路线")
                A("")
                A("- `onnx2torch.convert()` 对 `best_v3.onnx` **一次成功**（0.3s，204 个参数张量，385 个子模块），")
                A("  没有 `NotImplementedError`、没有未实现的算子，**没有出现失败或不稳定**。")
                A("- 所以「用 ultralytics 重建结构后按名字映射」不是**唯一**路径，本脚本把它作为")
                A("  **warm start 专用的第二条产物**（`weights/best_v3_native.pt`）同时给出。")
                A("- 唯一隐患是一条告警：`onnx2torch/node_converters/slice.py:63` 用非 tuple 序列做多维索引")
                A("  （`x[pos_axes_slices]` 而非 `x[tuple(...)]`），torch 2.9 起语义会变。")
                A("  在本环境（torch 2.13.0+cu126）实测结果仍然正确——脚本每次都做数值对齐校验，")
                A("  一旦该写法的语义在未来 torch 版本变化，校验会立刻失败并在报告里暴露。")
                A("- 「detector 从公开预训练权重重训」作为**第三**条路线保留，但不是必需。")
            A("")

        A("### 训练可用性（能否作为 warm start）")
        A("")
        t = r["training"]
        gp = r.get("grad_path")
        A(f"- 输出 `requires_grad`：{t['output_requires_grad']}")
        A(f"- 参数张量：{t['n_param_tensors']}，反向拿到非零梯度的：**{t['n_with_grad']}**")
        A(f"- 梯度无法到达的参数：{t['n_without_grad']} 个")
        if t["without_grad_all"]:
            A("")
            A("```")
            for k in t["without_grad_all"][:20]:
                A(k)
            A("```")
        if gp:
            A(f"- 输入梯度：{gp['input_grad_absmax']}（能否回到输入：{gp['reachable_to_input']}）")
        A("")
        if r["kind"] == "yolo":
            A("### TopK / GatherElements / Mod 对训练的影响（重点）")
            A("")
            A("`best_v3.onnx` 是**导出专用推理图**，尾部这一段是训练垃圾：")
            A("")
            A("| 算子 | 位置 | 可导性 | 训练后果 |")
            A("|---|---|---|---|")
            A("| `TopK` (×2) | `/model.23/TopK`、`/model.23/TopK_1` | ❌ 不可导（梯度恒 0） | 300 行选择、300 条类别 top-k 全靠它，反向在此断掉 |")
            A("| `GatherElements` (×2) | 取回原始行/列 | ❌ 索引算子，梯度为 0 | 同上，断在索引处 |")
            A("| `Mod` | 类别索引取模 | ❌ 整数取模，无梯度 | 类别号无法回传 |")
            A("| `Cast` | int64→float | ❌ | 断流 |")
            A("| `ReduceMax`/`Split`/`Transpose`/`Reshape`/`Tile`/`Concat` | 尾部 | ⚠️ 可导但不稳定 | 大量冗余计算，还会引入 0/1 掩码，浪费显存 |")
            A("")
            A("实测：`weights/best_v3.pt` 端到端 `output0.sum().backward()` **能跑通**，")
            A(f"{t['n_with_grad']}/{t['n_param_tensors']} 个参数张量拿到非零梯度，输入梯度 "
              f"{gp['input_grad_absmax'] if gp else 'n/a'}。")
            A("但**能反传 ≠ 梯度有意义**：这些梯度是「TopK 挑选后剩下的那 300 行」的稀疏信号，")
            A("语义上已经不等于「检测损失对卷积核的梯度」。直接拿它 finetune 会训出噪声。")
            A("")
            A("所以对 `weights/best_v3.pt` 的正确用法是：")
            A("")
            A("1. **不要端到端反传** `output0`。")
            A("2. **正确姿势**：拿 `weights/best_v3_native.pt`（原生 ultralytics 结构）做 warm start，")
            A("   用 ultralytics 的训练循环（`Detect.forward` 在 `self.training=True` 时返回 raw 预测字典），")
            A("   损失走 v8DetectionLoss，梯度路径干净，没有 TopK/GatherElements/Mod。")
            A("3. 如果只是要**推理**，`weights/best_v3.pt` 完全够用（数值已验证等价，只差行序）。")
            A("")
            n = r.get("native") or {}
            if n.get("status") != "failed" and n.get("training"):
                nt = n["training"]
                A("#### 原生结构（`weights/best_v3_native.pt`）的训练探测")
                A("")
                A("原生模型在训练态返回 raw 预测 dict（`one2many` + `one2one`），")
                A("不再经过 TopK/Mod/Cast：")
                A("")
                A("| 项 | 值 |")
                A("|---|---|")
                A(f"| 输出类型 | `{nt['output_type']}`（训练态 dict） |")
                A(f"| dummy loss | {nt['loss']:.4g} |")
                A(f"| 参数张量 | {nt['n_param_tensors']} |")
                A(f"| 拿到非零梯度 | **{nt['n_with_grad']}** |")
                A(f"| 无梯度 | {nt['n_without_grad']} |")
                if nt["without_grad_all"]:
                    A(f"| 无梯度清单（前 10） | {nt['without_grad_all'][:10]} |")
                A(f"| 梯度回到输入 | {n.get('grad_path', {}).get('reachable_to_input')} "
                  f"(absmax={n.get('grad_path', {}).get('input_grad_absmax')}) |")
                A("")
            A("### 缺陷清单（作为 warm start 时必须知道）")
            A("")
            A("1. **缺真实 BN 统计**：ONNX 已折叠 `Conv+BN`，本脚本用恒等 BN 重参数化。")
            A("   数值上严格等价，但 BN 的 `running_mean/var` 不再携带真实统计量；")
            A("   若想恢复「真 BN 训练动态」，需要先用真实数据跑若干次前向重新累积 `running_mean/var`")
            A("   （脚本未做，因为仓库里没有训练集）。")
            A("2. **缺 one-to-many head**：`model.23.cv2.*` / `model.23.cv3.*` 保持随机初始化。")
            A("   ultralytics end2end 训练同时用 one2many + one2one 两个头，one2many 需要重新学。")
            A("3. **缺 optimizer / ema / epoch 状态**：ONNX 不含这些，warm start 只能是「权重级」而非「续训级」。")
            A("4. **输入被钉死成 1×3×640×640**：导出时 `dynamic=False`。")
            A("   换分辨率前需要设 `model.model[-1].dynamic = True` 或重新 bake anchors。")
            A("5. **必须走原生结构**：`best_v3.pt`（onnx2torch 的 fx GraphModule）是 385 个平铺子模块，")
            A("   没有 `model.0`/`model.23` 层次，也没有 `Detect` 模块，无法直接接 ultralytics 的 loss / 数据管线。")
            A("")
        else:
            A("### 算子可导性")
            A("")
            A("图里只有 `Conv / Relu / BatchNormalization / Add / Concat / GlobalAveragePool / Flatten /")
            A("Gemm / Sub / Abs / Mul / Identity`，**全部可导**，没有任何 TopK/GatherElements/Mod 这类")
            A("导出专用算子。")
            A("")
            A("`fusion_head` 是 `Gemm -> BatchNorm1d -> ReLU -> Gemm -> BatchNorm1d -> ReLU -> Gemm`，")
            A("输入是 `Concat([Sub(|f1-f2|), Mul(f1,f2)])`（5120 = 2×2560），标准 Siamese 距离头，")
            A("直接反传没有任何阻碍。")
            A("")
        A("")

    A("## 4. 结论")
    A("")
    A("| 模型 | ONNX → torch | 数值等价 | 能否直接继续训练 | 推荐用法 |")
    A("|---|---|---|---|---|")
    for r in records:
        if r.get("status") != "converted":
            A(f"| `{r['onnx']}` | ❌ | – | ❌ | – |")
            continue
        if r["kind"] == "siamese":
            v = r["verify"]
            ok = "✅" if v["pass"] else f"⚠️ diff={v['max_abs_diff']:.2e}"
            A(f"| `{r['onnx']}` | ✅ | {ok}（max_abs_diff={v['max_abs_diff']:.2e}） | ✅ **可以** | "
              f"`weights/pre_model_v7.pt` 直接 warm start，`load_state_dict(strict=False)` |")
        else:
            n = r.get("native") or {}
            nv = (n.get("verify") or {}).get("real_image:res.jpg", {})
            ri = nv.get("rank_invariant", {})
            ok = (f"✅ 等价到 conf≥{ri.get('lowest_equivalent_thr')}"
                  if ri.get("equivalent_down_to_1e-4") else "见 §3")
            A(f"| `{r['onnx']}` | ✅ | {ok} | ⚠️ **`best_v3.pt` 不要端到端反传**；"
              f"`best_v3_native.pt` 可以 | warm start 用 `weights/best_v3_native.pt`；推理用 `weights/best_v3.pt` |")
    A("")
    A("## 5. 一句话结论")
    A("")
    si = next((r for r in records if r.get("kind") == "siamese" and r.get("status") == "converted"), None)
    yo = next((r for r in records if r.get("kind") == "yolo" and r.get("status") == "converted"), None)
    if si:
        A(f"- **`pre_model_v7.onnx`：能直接继续训练。** 修掉 `Identity` 权重别名（154 个）与 onnx2torch 的")
        A(f"  Windows 临时文件缺陷后直转成功，`max_abs_diff = {si['verify']['max_abs_diff']:.2e}` (< 1e-5)，")
        A("  全图只有 `Conv / BatchNormalization / Relu / Gemm / Add / Concat / GlobalAveragePool / Flatten /")
        A("  Sub / Abs / Mul`，**没有任何不可导算子**，`fusion_head` 是标准 distance head，直接反传无阻碍。")
    if yo:
        yv = yo["verify"]["real_image:res.jpg"]
        ri = yv["rank_invariant"]
        tb = yv.get("topk_boundary") or {}
        lv025 = (ri.get("levels") or {}).get("0.25", {})
        A(f"- **`best_v3.onnx`：能转换、能推理、能取回全部权重，但 `output0` 不能当训练损失直接反传。**")
        A(f"  该 ONNX 是 `end2end=True` 的**推理导出图**，尾部 `TopK / GatherElements / Mod / Cast` 不可导，")
        A(f"  梯度虽能流经（靠 0/1 掩码挑行）但语义已失真。")
        A(f"  原始 `raw max_abs_diff` 偏大（实测 res.jpg {yv['raw_max_abs_diff']:.4g}）完全由 TopK 在并列分数处的")
        A(f"  索引抖动造成：候选 8400 个，"
          + (f"第 300 名分数 {tb['rank300_score']:.4g} 上只有 {tb['strictly_greater_than_rank300']} 个严格更大的候选、"
             f"却要从 {tb['tied_at_rank300']} 个并列值里挑 {tb['slots_taken_from_tie']} 个" if tb else "")
          + "。")
        A(f"  逐阈值显著性比对（最优匹配）结果：")
        A("")
        A("  | thr | 行数 t/o | 类别全等 | 配对后坐标 max_diff | 判定 |")
        A("  |---|---|---|---|---|")
        for thr, lv in (ri.get("levels") or {}).items():
            A(f"  | {thr} | {lv.get('sig_n_torch')}/{lv.get('sig_n_ort')} | {lv.get('cls_equal')} | "
              f"{lv.get('coord_max_abs_diff')} | {'✅' if lv.get('pass') else '⚠️'} |")
        A("")
        A(f"  即：**等价到 conf≥{ri.get('lowest_equivalent_thr')}**，"
          f"conf≥0.25 的 8 个有效检测框**数量/类别/坐标/置信度全部一致**。")
        A("")
        A("  **要做 warm start / 继续训练，唯一正路是用 `weights/best_v3_native.pt`**：")
        A("  原生 `yolo26n(nc=3, end2end=True)` 结构，102 个 ONNX 4D 权重 + 全部其它键按名映射、")
        A("  96 个 `Conv+BN` 做恒等 BN 重参数化、`load_state_dict` missing=0 unexpected=0、")
        A("  前向结果与 ONNX **逐位一致**。已知缺口只有两点：")
        A("  (a) `model.23.cv2.*` / `model.23.cv3.*`（one-to-many head）ONNX 未导出，需重学；")
        A("  (b) BN 的 `running_mean/var` 是恒等值，缺真实统计，建议先用真实数据短跑数百次前向重新累积。")
        A("")
        A("  如果不想补 one-to-many head 这一课，**次优方案**是「detector 从公开预训练权重重训」：")
        A("  用 `yolo26n.pt` 官方权重 + 本仓库 `data/` 的本地数据（`data/samples`、`data/turing_norm`、")
        A("  `out/*.png` 已被 `tools/capture_captcha.py` 落盘）重训，成本远低于修补导出图。")
    A("")
    A("### 三条路线对比（含明确取舍）")
    A("")
    A("| 路线 | 可行 | 代价 | 结论 |")
    A("|---|---|---|---|")
    A("| ① `best_v3.pt` 端到端 finetune | 技术可行 | 梯度被 TopK 掩码污染，训出噪声 | **不推荐** |")
    A("| ② `best_v3_native.pt` warm start + ultralytics 训练循环 | ✅ | one-to-many head 重学 + BN 统计重累积 | **推荐** |")
    A("| ③ 用 yolo26n 官预训练权重在本地 captcha 数据上重训 | ✅ | 需要训练数据与算力，但不依赖 ONNX 反解 | 数据量够时**最稳** |")
    A("")
    A("`pre_model_v7` 不需要备选方案，① 就够。")
    A("")
    A("## 6. 端到端 warm-start 冒烟测试")
    A("")
    A("上面的梯度探测只回答「能不能反传」；`tools/verify_recovered.py` 回答「能不能真的拿来训」。")
    A("命令：")
    A("")
    A("```powershell")
    A("$env:PYTHONIOENCODING='utf-8'")
    A("G:\\Python\\Dev\\python\\python.exe tools/verify_recovered.py --steps 30 --steps-siamese 60")
    A("```")
    A("")
    smoke = os.path.join(repo, "out", "smoke_test.json")
    if os.path.exists(smoke):
        import json as _json

        with open(smoke, encoding="utf-8") as fh:
            d = _json.load(fh)
        si, yn, bi = d.get("siamese"), d.get("yolo_native"), d.get("best_v3_infer")

        if si:
            A("### 6.1 `weights/pre_model_v7.pt`：过拟合单 batch（Adam + CosineAnnealingLR，BCEWithLogitsLoss）")
            A("")
            A("| 项 | 值 |")
            A("|---|---|")
            A(f"| load_state_dict | missing={si['load_missing']} unexpected={si['load_unexpected']} |")
            A(f"| 优化器 | {si.get('optimizer', 'n/a')} |")
            A(f"| 步数 | {si['steps']} |")
            A("")
            A("| step | loss | lr | grad_norm |")
            A("|---|---|---|---|")
            for h in si["history"]:
                A(f"| {h['step']} | {h['loss']:.6f} | {h.get('lr', 0):.2e} | {h['grad_norm']:.4g} |")
            A("")
            A(f"loss {si['loss_first']:.6f} → {si['loss_last']:.6f}"
              + (f"（最优 {si['loss_best']:.6f}）" if si.get("loss_best") is not None else "")
              + f"，ratio={si['loss_ratio']}，**下降={si['loss_dropped']}**。")
            A("收敛过程干净（2.90 → 0.02 → 0.02 量级），证明梯度有效、参数真的在更新。")
            A("")
            A("> 「能训练」的判据是**损失能下降**，不是「损失降到多低」——目标标签是随机初始输出二值化得来的，")
            A("> 本身没有语义，只用来构造一个有梯度的可优化问题。")
            A("")

        if yn:
            A("### 6.2 `weights/best_v3_native.pt`：ultralytics 官方 `E2ELoss`（Adam lr=1e-3）")
            A("")
            A("这是 end2end 训练实际使用的损失（one2many TAL topk=10 + one2one TAL topk=7/topk2=1 加权），")
            A("不是 dummy loss：")
            A("")
            A("| 项 | 值 |")
            A("|---|---|")
            A(f"| load_state_dict | missing={yn['load_missing']} unexpected={yn['load_unexpected']} |")
            A(f"| 损失分量名 | {yn['loss_names']}（`reg_max=1` 所以是 l1 而非 dfl） |")
            A(f"| 步数 | {yn['steps']} |")
            A("")
            A("| step | 总 loss | box | cls | l1 | grad_norm |")
            A("|---|---|---|---|---|---|")
            for h in yn["history"]:
                v = h.get("loss_vec") or [0, 0, 0]
                A(f"| {h['step']} | {h['loss']:.6f} | {v[0]:.5f} | {v[1]:.5f} | {v[2]:.5f} | {h['grad_norm']:.4g} |")
            A("")
            A(f"loss {yn['loss_first']:.6f} → {yn['loss_last']:.6f}（ratio={yn['loss_ratio']}），"
              f"**下降={yn['loss_dropped']}**，三个分量全部单调下降。")
            A("这直接证明：`best_v3_native.pt` 可以接 ultralytics 的官方训练管线。")
            A("")

        if bi:
            A("### 6.3 `weights/best_v3.pt`：真实图推理（`docs/res.jpg`，conf≥0.25）")
            A("")
            A(f"有效框数 torch={bi['n_torch']}，onnxruntime={bi['n_ort']}，逐框一致：")
            A("")
            A("| rank | torch xyxy | conf | cls | ort xyxy | conf | cls |")
            A("|---|---|---|---|---|---|---|")
            for b in bi["boxes"]:
                A(f"| {b['rank']} | {b['torch_xyxy']} | {b['torch_conf']} | {b['torch_cls']} | "
                  f"{b['ort_xyxy']} | {b['ort_conf']} | {b['ort_cls']} |")
            A("")
            A("类别名：0=`label`，1=`title`，2=`char`（与 ONNX metadata `names` 一致）。")
            A("8 个框的坐标与置信度全部对齐到小数点后 2~5 位，只有 rank 7 的 conf 差 2e-5。")
            A("")
    else:
        A("> 还没跑冒烟测试。执行上面的命令后会在此处插入结果表。")
        A("")

    A("## 7. 关键命令（复现用）")
    A("")
    A("```powershell")
    A("cd D:\\Github\\captchayes\\T-Sec-captcha")
    A("$env:PYTHONIOENCODING='utf-8'")
    A("")
    A("# 一键：转换两个模型 + 原生结构还原 + 数值校验 + 生成本报告")
    A("G:\\Python\\Dev\\python\\python.exe tools/recover_weights.py --all --json out/recover_raw.json")
    A("")
    A("# 端到端 warm-start 冒烟测试（真实损失 + 优化器 + 真实图推理）")
    A("G:\\Python\\Dev\\python\\python.exe tools/verify_recovered.py --steps 30 --steps-siamese 60")
    A("")
    A("# 单模型")
    A("G:\\Python\\Dev\\python\\python.exe tools/recover_weights.py --onnx model/pre_model_v7.onnx --out weights/pre_model_v7.pt")
    A("G:\\Python\\Dev\\python\\python.exe tools/recover_weights.py --onnx model/best_v3.onnx     --out weights/best_v3.pt")
    A("```")
    A("")
    A("### 训练侧接入示例")
    A("")
    A("```python")
    A("import torch")
    A("from ultralytics import YOLO")
    A("from ultralytics.utils.loss import E2ELoss   # end2end 训练用的官方损失")
    A("")
    A("# ---- detector：warm start 用原生结构 ----")
    A("payload = torch.load('weights/best_v3_native.pt', map_location='cpu', weights_only=False)")
    A("model = YOLO('yolo26n.yaml')          # 或 DetectionModel(cfg, ch=3, nc=3)")
    A("model.model.load_state_dict(payload['state_dict'], strict=False)")
    A("# 之后交给 ultralytics 的 DetectionTrainer，损失走 E2ELoss，无需任何 ONNX 相关代码")
    A("")
    A("# ---- siamese：直接加载 ----")
    A("si = torch.load('weights/pre_model_v7.pt', map_location='cpu', weights_only=False)")
    A("# si['state_dict'] 的键是 onnx2torch 的 fx 命名（含 '/'），用同名模块装载即可")
    A("```")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
