# -*- coding: utf-8 -*-
"""cunai2_ticket_provider: 给 cun.ai 2.0 自动产出 tencent ticket/randstr。

════════════════════════════════════════════════════════════════
2.0 的现状（读源码确认）
════════════════════════════════════════════════════════════════
`cunai/workflow.py:307-316`：

    if probe.captcha_enabled:
        if "tencent" in methods and not (
            self.tencent_captcha_ticket and self.tencent_captcha_randstr
        ):
            raise ApiError("目标站点需要腾讯验证码结果，请在站点正常流程中
                            取得 ticket 和 randstr 后传入")

也就是说 **2.0 已经把接入点留好了**，缺的只是「谁来产出这两个值」。

`cunai/sites.py:298` 的 `_sub2_captcha_fields()` 会把它们注入注册请求：

    fields["tencent_captcha_ticket"] = self.tencent_captcha_ticket
    fields["tencent_captcha_randstr"] = self.tencent_captcha_randstr

════════════════════════════════════════════════════════════════
本模块提供什么
════════════════════════════════════════════════════════════════
不给 2.0 打补丁，而是提供一个**即插即用的 provider**：

    from cunai2_ticket_provider import TicketProvider
    t = TicketProvider(entry_url="https://www.cun.ai/sign-up").get()
    RegistrationWorkflow(
        site_url=...,
        tencent_captcha_ticket=t["ticket"],
        tencent_captcha_randstr=t["randstr"],
        ...
    )

内部复用 T-Sec-captcha 的 `tools/tencent_ticket.py`。

════════════════════════════════════════════════════════════════
为什么必须用浏览器（不是设计选择，是硬约束）
════════════════════════════════════════════════════════════════
实测（tools/probe_http_captcha.py）：取题/取图/求解都能纯 HTTP，
但提交 `cap_union_new_verify` 时服务端要一个 `collect` 参数——
它是**服务端密钥的分组加密串**（base64 后 427 字节高熵二进制，
zlib 无法解压，长度按 16 字节分组浮动），封装了 sess/answer/
鼠标轨迹/时间戳。纯 HTTP 提交返回 errorCode=202。

所以：**取题解题可以 HTTP，提交必须浏览器。** 本 provider 因此
只在提交环节用浏览器，2.0 的其他流程仍走 HTTP，架构不受影响。

用法（命令行自测）：
    python tools/cunai2_ticket_provider.py --entry https://www.cun.ai/sign-up
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_ENTRY = "https://www.cun.ai/sign-up"


class TicketProvider:
    """2.0 用的天御票据提供器。

    参数：
        entry_url : 站点的注册/登录页（弹层在这个页面触发）
        max_retry : 最多重试次数（含风控退避）
        cooldown  : 首次风控冷却秒数（之后 3x、10x 递增）

    返回：
        {"ticket": str, "randstr": str, "errorCode": str, ...}
        成功时 errorCode == "0" 且 ticket/randstr 非空。
    """

    def __init__(self, entry_url: str = DEFAULT_ENTRY, max_retry: int = 3,
                 cooldown: int = 60, username: str = "", password: str = "",
                 email: str = ""):
        self.entry_url = entry_url
        self.max_retry = max_retry
        self.cooldown = cooldown
        self.username = username
        self.password = password
        self.email = email
        self._cache: dict | None = None

    def get(self, force: bool = False) -> dict:
        """取一张票。同一实例会缓存结果（票据有时效，按需 force 刷新）。"""
        if self._cache and not force:
            return self._cache
        from tencent_ticket import fetch_ticket
        t = fetch_ticket(
            self.entry_url,
            max_attempts=self.max_retry,
            cooldown_on_12=self.cooldown,
            username=self.username,
            password=self.password,
            email=self.email,
        )
        if t.get("errorCode") == "0" and t.get("ticket"):
            self._cache = t
        return t

    def as_workflow_kwargs(self, force: bool = False) -> dict:
        """直接喂给 RegistrationWorkflow 的 **kwargs。"""
        t = self.get(force=force)
        return {
            "tencent_captcha_ticket": t.get("ticket", ""),
            "tencent_captcha_randstr": t.get("randstr", ""),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default=DEFAULT_ENTRY)
    ap.add_argument("--retry", type=int, default=2)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--username", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--email", default="")
    args = ap.parse_args()

    p = TicketProvider(args.entry, max_retry=args.retry,
                       cooldown=args.cooldown, username=args.username,
                       password=args.password, email=args.email)
    t = p.get()
    print("\n" + "=" * 64)
    print("  2.0 可用参数：")
    print(f"    tencent_captcha_ticket  = {t.get('ticket', '')[:48]}")
    print(f"    tencent_captcha_randstr = {t.get('randstr', '')[:48]}")
    print(f"    errorCode               = {t.get('errorCode')!r}")
    if t.get("error"):
        print(f"    error                   = {t['error']}")
    print("=" * 64)

    out = os.path.join(REPO, "out", "ticket", "for_cunai2.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(p.as_workflow_kwargs(), open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"-> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
