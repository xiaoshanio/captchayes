# -*- coding: utf-8 -*-
"""check_refs: 校验文档里引用的脚本/产物是否真实存在（避免文档漂移）。

用法：
    python tools/check_refs.py
"""
from __future__ import annotations

import os
import re
import sys

DOCS = ["docs/TRAINING_AND_DATA_REFRESH.md", "docs/TRAINING_BENCHMARK.md"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAT = re.compile(r"`([A-Za-z0-9_\-/]+\.(?:py|ps1|md|onnx|pt|json|yaml|txt))`")

# 这些是"说明性引用"，不要求仓库里存在
ALLOW = {
    "onnx2torch/utils/safe_shape_inference.py",   # 第三方库内部文件
    "best.pt", "best_v3.pt", "best_v3_native.pt", "pre_model_v7.pt",
    "best_v3.onnx", "pre_model_v7.onnx",
    "best_v3_next.onnx", "pre_model_v7_next.onnx",
    "best_v3_nms.onnx", "pre_model_v7_nms.onnx",
    "recovered_report.md", "data.yaml",
    "drift_report.json",   # tools/drift_check.py 的运行时产物
    "meta.json",           # 采集样本的元数据文件名（非仓库文件）
}


def _exists_anywhere(name):
    """在仓库里递归找同名文件（排除 .git / data / runs 等大目录）。"""
    skip = {".git", "data", "runs", "weights", "__pycache__", "node_modules"}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in skip]
        if name in files:
            return True
    return False


def main():
    missing = []
    for doc in DOCS:
        p = os.path.join(REPO, doc)
        if not os.path.exists(p):
            print(f"[skip] 缺文档 {doc}")
            continue
        txt = open(p, encoding="utf-8").read()
        for ref in sorted(set(PAT.findall(txt))):
            if ref in ALLOW or ref.startswith(("http", "www")):
                continue
            cand = os.path.join(REPO, ref)
            if os.path.exists(cand):
                continue
            if any(os.path.exists(os.path.join(REPO, sub, ref))
                   for sub in ("tools", "train", "eval", "docs", "model")):
                continue
            # 递归兜底（例如 src/utils/yolo_onnx.py 这种深层路径）
            if _exists_anywhere(os.path.basename(ref)):
                continue
            missing.append((doc, ref))

    if not missing:
        print("OK 文档引用全部存在")
        return 0
    print(f"发现 {len(missing)} 处失效引用：")
    for doc, ref in missing:
        print(f"  {doc}: {ref}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
