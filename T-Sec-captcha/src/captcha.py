# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : jy_click.py
# Time       ：2023/11/13 16:49
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import os
from typing import List, Tuple, Dict, Any

try:
    from src.utils import ver_onnx
    from src.utils import yolo_onnx
    from src.utils import matchingMode
except ImportError:  # 兼容 git clone 后被当作 captcha/cv 子目录使用的情况
    try:
        from .utils import ver_onnx
        from .utils import yolo_onnx
        from .utils import matchingMode
    except ImportError:
        from cv import ver_onnx  # type: ignore
        from cv import yolo_onnx  # type: ignore
        from cv import matchingMode  # type: ignore


def _resolve_model_file(filename: str) -> str:
    """解析模型文件，git clone 后开箱即用。

    查找顺序（都不改 cun.ai 原项目）：
    1. 绝对路径 / 已存在路径直接使用
    2. 环境变量 TSEC_MODEL_DIR 指定的目录
    3. 本仓库 <repo>/model/<file>（本项目默认）
    4. 本仓库 <repo>/models/<file>（兼容 cun.ai 的 captcha/models 布局）
    """
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists(filename):
        return filename
    env_dir = os.environ.get("TSEC_MODEL_DIR", "").strip()
    if env_dir:
        cand = os.path.join(env_dir, os.path.basename(filename))
        if os.path.exists(cand):
            return cand
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)  # src/ 的上一级
    for d in (os.path.join(repo_root, "model"),
              os.path.join(repo_root, "models")):
        cand = os.path.join(d, os.path.basename(filename))
        if os.path.exists(cand):
            return cand
    # 默认返回 model/ 路径，让 onnxruntime 报错信息保持明确
    return os.path.join(repo_root, "model", os.path.basename(filename))


class TextSelectCaptcha(object):
    def __init__(self, per_path: str = 'pre_model_v7.onnx', yolo_path: str = 'best_v3.onnx') -> None:
        """
        初始化识别器，加载 ONNX 模型。

        :param per_path: 字符匹配模型文件名（PreONNX），相对路径，默认 'pre_model_v6.onnx'
        :param yolo_path: YOLO 目标检测模型文件名，默认 'best_v3.onnx'
        """
        save_path = os.path.join(os.path.dirname(__file__), '../model')
        path = lambda a, b: os.path.join(a, b)
        per_path = path(save_path, per_path)
        yolo_path = path(save_path, yolo_path)
        per_path = _resolve_model_file(per_path)
        yolo_path = _resolve_model_file(yolo_path)
        if not os.path.exists(per_path) or not os.path.exists(yolo_path):
            raise FileNotFoundError(
                f"验证模型缺失: {per_path}, {yolo_path} "
                f"(可设置环境变量 TSEC_MODEL_DIR 指向模型目录)")
        self.yolo = yolo_onnx.YOLO(yolo_path)
        self.pre = ver_onnx.PreONNX(per_path)

    def detection(self, image_path: str) -> List[List[float]]:
        img = matchingMode.open_image(image_path)
        data = self.yolo.inference(img)
        return data

    def run(self, image_path: str) -> List[List[float]]:
        img = matchingMode.open_image(image_path)
        data = self.yolo.inference(img)
        target_boxes = [item[:4] for item in data if len(item) >= 6 and item[5] == 0]
        char_boxes = [item[:4] for item in data if len(item) >= 6 and item[5] == 2]
        char_boxes.sort(key=lambda box: box[0])
        img_targets = [img[int(box[1]):int(box[3]), int(box[0]):int(box[2])] for box in target_boxes]
        chars = [img[int(box[1]):int(box[3]), int(box[0]):int(box[2])] for box in char_boxes]
        slys = self.pre.reason_all_batch(chars, img_targets)
        sorted_result = matchingMode.find_overall_index_fast(slys)
        result = [target_boxes[j] for i, j in sorted_result]
        return result

    def run_dict(self, image_path: str) -> Dict[str, Any]:
        img = matchingMode.open_image(image_path)
        h, w, _ = img.shape
        result = self.run(image_path)
        return {
            "imgW": w,
            "imgH": h,
            "point": [{"x_rel": (x1 + x2) / 2, "y_rel": (y1 + y2) / 2} for x1, y1, x2, y2 in result],
            "corp": [{"x1": x1, "y1": y1 , "x2": x2, "y2":  y2 } for x1, y1, x2, y2 in result],
        }


if __name__ == '__main__':
    from src.drawing import drow_img
    cap = TextSelectCaptcha()
    image_path = r"../docs/res.jpg"
    result = cap.run(image_path)
    print(result)
    drow_img(image_path, result)
    print(cap.run_dict(image_path))