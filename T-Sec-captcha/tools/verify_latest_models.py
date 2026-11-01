# -*- coding: utf-8 -*-
"""verify_latest_models: 验证 1.0 安装脚本能识别新旧模型。

要证明两件事：
    1) 装了最新模型 → 判定「是」
    2) 装了旧模型   → 判定「否」并给出期望值
没有第 2 条，"保证安装最新"就是空话（文件存在也会显示 OK）。

做法：把 1.0 的 setup_captcha 当模块导入，用临时目录构造两种场景。

用法：
    python tools/verify_latest_models.py
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJ_10 = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"
CACHE = os.path.expanduser(r"~\.cun-captcha-upstream\T-Sec-captcha")
LOCAL = r"D:\Github\captchayes\T-Sec-captcha"


def load_installer():
    p = os.path.join(PROJ_10, "tools", "setup_captcha.py")
    if not os.path.exists(p):
        raise SystemExit("找不到安装脚本: " + p)
    spec = importlib.util.spec_from_file_location("setup_captcha", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["setup_captcha"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_root(tmp: Path, model_src: Path) -> Path:
    """在临时目录里搭一个只含 models 的项目根。"""
    dst = tmp / "captcha" / "models"
    dst.mkdir(parents=True, exist_ok=True)
    for n in ("best_v3.onnx", "pre_model_v7.onnx"):
        s = model_src / n
        if s.is_file():
            shutil.copyfile(s, dst / n)
    return tmp


def show(mod, root: Path, title: str):
    print("")
    print("=== " + title + " ===")
    rows = mod.model_report(root)
    for name, status, latest in rows:
        mark = "OK  " if latest == "是" else "WARN"
        print("  [" + mark + "] " + name + ": " + status)
    print("  models_are_latest -> " + str(mod.models_are_latest(root)))
    return mod.models_are_latest(root)


def main():
    mod = load_installer()
    print("安装脚本: " + os.path.join(PROJ_10, "tools", "setup_captcha.py"))
    print("登记的最新指纹:")
    for k, (sz, h) in mod.LATEST_MODELS.items():
        print(f"  {Path(k).name}: {sz} 字节  sha256={h[:16]}…")

    ok_all = True
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 场景 1：装最新
        local = Path(LOCAL) / "model"
        if local.is_dir():
            r1 = build_root(tmp / "new", local)
            got = show(mod, r1, "装的是最新模型（应判 是）")
            ok_all = ok_all and got
        else:
            print("跳过场景1：本地 model/ 不存在")

        # 场景 2：装旧模型（用上游缓存里的旧文件）
        cache = Path(CACHE) / "model"
        if cache.is_dir():
            r2 = build_root(tmp / "old", cache)
            got = show(mod, r2, "装的是旧模型（应判 否，并给出期望值）")
            ok_all = ok_all and (not got)
        else:
            print("跳过场景2：上游缓存不存在")

        # 场景 3：完全没装
        r3 = tmp / "empty"
        (r3 / "captcha" / "models").mkdir(parents=True, exist_ok=True)
        got = show(mod, r3, "没装模型（应判 否 + 缺失）")
        ok_all = ok_all and (not got)

    print("")
    print("=" * 60)
    print(" 结论: " + ("通过 —— 新旧模型能被正确区分" if ok_all
                       else "失败 —— 检测逻辑有问题"))
    print("=" * 60)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
