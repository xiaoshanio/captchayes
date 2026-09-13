"""以 src/utils 为准同步到 cv/（保持双份同哈希，方便直接复制到 cun.ai）。

    python tools/sync_cv.py          # 同步
    python tools/sync_cv.py --check  # 仅检查
"""
import hashlib
import os
import shutil
import sys

FILES = ["matchingMode.py", "ver_onnx.py", "yolo_onnx.py"]


def sha1(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    check = "--check" in sys.argv
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_d = os.path.join(repo, "src", "utils")
    cv_d = os.path.join(repo, "cv")
    os.makedirs(cv_d, exist_ok=True)
    bad = 0
    for name in FILES:
        s, d = os.path.join(src_d, name), os.path.join(cv_d, name)
        if not os.path.exists(s):
            print(f"FAIL 缺失: {s}")
            bad += 1
            continue
        if check:
            if not os.path.exists(d) or sha1(s) != sha1(d):
                print(f"FAIL 不一致: src/utils/{name} != cv/{name}")
                bad += 1
            else:
                print(f"PASS cv/{name}")
        else:
            shutil.copyfile(s, d)
            print(f"SYNC cv/{name}")
    if not os.path.exists(os.path.join(cv_d, "__init__.py")) and not check:
        open(os.path.join(cv_d, "__init__.py"), "w").close()
    print("CHECK FAILED" if check and bad else ("DONE" if not check else "ALL SAME"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
