# -*- coding: utf-8 -*-
"""patch_install_bat: 让 1.0 的 install.bat 安装**最新模型**。

背景（实测）：
    install.bat 第 3 步写死 `--skip`，注释理由是「当前站点注册/登录无验证」。
    但这个前提已过时 —— 站点现在明确下发 2404/2408/2401 三种验证码。

    更要命的是：新模型只在本地提交 e23266c 里，origin/main 还是 812deeb
    （旧模型 best_v3=9806720 / pre_model_v7=44452568）。
    所以即使不 skip，走 GitHub 也只能装到旧模型。

改法：
    第 3 步改为「真的安装」，并按优先级选源：
        1) 环境变量 CUN_CAPTCHA_SOURCE 指定的本地目录（必然最新）
        2) 自动探测本机 captchayes 源码目录（若存在）
        3) GitHub 上游（--non-interactive，会打印过期告警）
    安装完再跑一次 --check，把模型指纹打印出来。

注意：install.bat 是 GBK 编码，必须按 GBK 读回写，否则中文会乱码。

用法：
    python tools/patch_install_bat.py
    python tools/patch_install_bat.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

PROJ_10 = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"

OLD_BLOCK = """echo.
echo [3/4] 验证码模型：当前站点注册/登录无验证，本步灰色跳过（不下载）。
set "CUN_SKIP_CAPTCHA=1"
"%PYTHON_EXE%" tools\\setup_captcha.py --skip
set "CUN_SKIP_CAPTCHA="
if errorlevel 1 (
  echo.
  echo 验证码步骤未完成，可重新双击 install.bat 继续。
  pause
  exit /b %ERRORLEVEL%
)
"""

NEW_BLOCK = """echo.
echo [3/4] 验证码模型：安装最新模型（站点注册/登录需要人机验证）。
rem 源优先级：本地源码 > 自动探测的本机源码 > GitHub 上游。
rem 说明：上游 origin 可能落后于本地重训结果，只有本地源才能保证是最新模型。
if defined CUN_CAPTCHA_SOURCE (
  echo 使用本地源：%CUN_CAPTCHA_SOURCE%
  "%PYTHON_EXE%" tools\\setup_captcha.py --source "%CUN_CAPTCHA_SOURCE%"
) else if exist "%~dp0..\\..\\..\\captchayes\\T-Sec-captcha\\model\\best_v3.onnx" (
  echo 检测到本机源码目录，用它安装最新模型。
  "%PYTHON_EXE%" tools\\setup_captcha.py --source "%~dp0..\\..\\..\\captchayes\\T-Sec-captcha"
) else (
  echo 未找到本地源码，改从 GitHub 上游安装（可能不是最新模型）。
  "%PYTHON_EXE%" tools\\setup_captcha.py --non-interactive
)
if errorlevel 1 (
  echo.
  echo 验证码步骤未完成，可重新双击 install.bat 继续。
  echo 也可先设置 CUN_CAPTCHA_SOURCE 指向本地 T-Sec-captcha 目录再重跑。
  pause
  exit /b %ERRORLEVEL%
)
echo.
echo 模型指纹核对：
"%PYTHON_EXE%" tools\\setup_captcha.py --check
"""


def find_block(lines: list[str]) -> tuple[int, int] | None:
    """按行定位第 3 步代码块，返回 (start, end) 闭区间下标。

    不能直接字符串匹配旧块：install.bat 是 CRLF，而脚本里的常量是 LF，
    直接 `in` 判断永远失败（实测）。
    """
    marker = None
    for i, line in enumerate(lines):
        if "[3/4] 验证码模型" in line:
            marker = i
            break
    if marker is None:
        return None
    start = marker - 1 if marker > 0 and lines[marker - 1].strip() == "echo." \
        else marker
    end = None
    for j in range(marker, min(len(lines), marker + 20)):
        if "exit /b %ERRORLEVEL%" in lines[j]:
            # 下一行应是右括号
            for k in range(j, min(len(lines), j + 4)):
                if lines[k].strip() == ")":
                    end = k
                    break
            break
    if end is None:
        return None
    return start, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=PROJ_10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    bat = os.path.join(args.root, "install.bat")
    if not os.path.exists(bat):
        raise SystemExit("找不到 install.bat: " + bat)

    with open(bat, "rb") as fh:
        raw = fh.read()
    text = raw.decode("gbk")

    if "CUN_CAPTCHA_SOURCE" in text and not args.force:
        print("install.bat 已经改过（含 CUN_CAPTCHA_SOURCE），跳过。")
        return 0

    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    span = find_block(lines)
    if span is None:
        print("未找到第 3 步代码块，请手动确认。附近内容：")
        for i, line in enumerate(lines, 1):
            if 85 <= i <= 100:
                print(f"  {i}: {line}")
        return 1
    start, end = span

    new_lines = NEW_BLOCK.replace("\r\n", "\n").split("\n")
    new_text = nl.join(lines[:start] + new_lines + lines[end + 1:])

    if args.dry_run:
        print(f"将替换第 {start + 1}~{end + 1} 行，改成：")
        print("-" * 60)
        print(NEW_BLOCK)
        return 0

    bak = bat + ".bak"
    if not os.path.exists(bak):
        shutil.copyfile(bat, bak)
        print("已备份: " + os.path.basename(bak))

    with open(bat, "wb") as fh:
        fh.write(new_text.encode("gbk"))
    print(f"已更新 install.bat（替换第 {start + 1}~{end + 1} 行，GBK 保持）")
    print("第 3 步现在会：")
    print("  1) 优先用 CUN_CAPTCHA_SOURCE 本地源（保证最新模型）")
    print("  2) 其次自动探测本机 captchayes 源码目录")
    print("  3) 最后才回退 GitHub 上游，并打印过期告警")
    print("  4) 安装后跑 --check 打印模型指纹")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
