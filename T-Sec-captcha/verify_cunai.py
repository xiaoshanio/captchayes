"""verify_cunai: 验证本仓库 git clone 后可直接应用于 cun.ai。

原则：只读 F:\\Kaifa\\cun.ai，绝不写入/修改它。
用法：
    python verify_cunai.py
    python verify_cunai.py --cunai F:\\Kaifa\\cun.ai
"""
import argparse
import hashlib
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cunai", default=r"F:\Kaifa\cun.ai")
    args = ap.parse_args()
    ok = True

    print("== 1. 本仓库独立推理 ==")
    try:
        from src.captcha import TextSelectCaptcha
        cap = TextSelectCaptcha()
        img = os.path.join(REPO, "docs", "res.jpg")
        d = cap.run_dict(img)
        assert len(d.get("point", [])) == 4, d
        print("PASS src.captcha 推理正常:", d["point"])
    except Exception as e:
        ok = False
        print("FAIL src.captcha:", repr(e))

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
            elif sha1(pa) != sha1(pb):
                ok = False
                print(f"FAIL 哈希不一致: {a} <-> {b}")
            else:
                print(f"PASS {a} == {b}")

    print("RESULT:", "ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
