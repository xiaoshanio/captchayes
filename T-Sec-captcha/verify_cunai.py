"""verify_cunai: 验证本仓库 git clone 后可直接应用于 cun.ai。

原则：只读 F:\\Kaifa\\cun.ai，绝不写入/修改它。
用法：
    python verify_cunai.py
    python verify_cunai.py --cunai F:\\Kaifa\\cun.ai
    python verify_cunai.py --image data/samples/<sid>/main.jpg
"""
import argparse
import hashlib
import glob
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PAIRS = [
    ("model/best_v3.onnx", "captcha/models/best_v3.onnx"),
    ("model/pre_model_v7.onnx", "captcha/models/pre_model_v7.onnx"),
    ("cv/matchingMode.py", "captcha/cv/matchingMode.py"),
    ("cv/ver_onnx.py", "captcha/cv/ver_onnx.py"),
    ("cv/yolo_onnx.py", "captcha/cv/yolo_onnx.py"),
    ("src/utils/matchingMode.py", "captcha/cv/matchingMode.py"),
    ("src/utils/ver_onnx.py", "captcha/cv/ver_onnx.py"),
    ("src/utils/yolo_onnx.py", "captcha/cv/yolo_onnx.py"),
]


def sha1(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def _read_norm(p):
    """文本文件按换行归一化读取（兼容 git autocrlf），二进制保持原样。"""
    with open(p, "rb") as f:
        data = f.read()
    if os.path.splitext(p)[1].lower() in (".py", ".md", ".txt", ".json"):
        data = data.replace(b"\r\n", b"\n")
    return data


def pick_image(explicit=None):
    """挑一张与当前模型同域的图。

    为什么需要：
        本仓库当前 `model/` 里是**2408 图标点选**域的重训模型。
        `docs/res.jpg` 是**旧域**（中文文字点选 344x384）样本，
        换域后模型不认识它，会抛 ValueError（这是预期行为）。
        所以优先用 data/samples 里的 2408 样本；没有才回退到 docs/res.jpg。
    """
    if explicit:
        return explicit
    cands = sorted(glob.glob(os.path.join(REPO, "data", "samples", "*", "main.jpg")))
    if cands:
        return cands[0]
    return os.path.join(REPO, "docs", "res.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cunai", default=r"F:\Kaifa\cun.ai")
    ap.add_argument("--image", default=None,
                    help="显式指定验证图；默认自动挑 2408 样本")
    args = ap.parse_args()
    ok = True

    img = pick_image(args.image)
    print("== 1. 本仓库独立推理 ==")
    print(f"   验证图: {os.path.relpath(img, REPO)}")
    try:
        from src.captcha import TextSelectCaptcha
        cap = TextSelectCaptcha()
        d = cap.run_dict(img)
        assert len(d.get("point", [])) >= 1, d
        print(f"PASS src.captcha 推理正常: {len(d['point'])} 点")
    except Exception as e:
        ok = False
        print("FAIL src.captcha:", repr(e))
        print("     提示：若 model/ 是 2408 域模型，请用 --image 指定 2408 样本；"
              "docs/res.jpg 属旧域，换域后必然失败（预期）")

    print("== 2. cun.ai 风格 from cv import ==")
    try:
        from cv import matchingMode, ver_onnx, yolo_onnx  # noqa
        print("PASS from cv import matchingMode/ver_onnx/yolo_onnx")
    except Exception as e:
        ok = False
        print("FAIL from cv import:", repr(e))

    print("== 3. 与 cun.ai 只读比对（不修改 cun.ai） ==")
    if not os.path.isdir(args.cunai):
        print(f"SKIP cun.ai 不存在: {args.cunai}")
    else:
        for a, b in PAIRS:
            pa, pb = os.path.join(REPO, a), os.path.join(args.cunai, b)
            if not os.path.exists(pa):
                ok = False
                print(f"FAIL 缺失本仓库文件: {a}")
            elif not os.path.exists(pb):
                print(f"WARN cun.ai 缺失(仅提示): {b}")
            elif _read_norm(pa) != _read_norm(pb):
                ok = False
                print(f"FAIL 内容不一致: {a} <-> {b}")
            else:
                print(f"PASS {a} == {b}")

    print("RESULT:", "ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
