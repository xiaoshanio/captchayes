# -*- coding: utf-8 -*-
"""drission_verify_ext: 给 cun.ai 1.0 的验证器补「应答码判定 + 风控退避」。

════════════════════════════════════════════════════════════════
为什么需要这个补丁
════════════════════════════════════════════════════════════════
1.0 的 `_read_outcome()` 只用页面文案判限频：

    if status and ("频繁" in status or "过于频繁" in status):
        return "rate-limited"

问题：实测风控**不一定在页面上写"频繁"**。
我这轮实测 19 次登录，应答码与成败零例外对应：

    errorCode = "0"   → 成功 8/8
    errorCode = "12"  → 失败 11/11

而 1.0 完全没有读 `cap_union_new_verify` 的响应
（`start_capture` 只订阅 prehandle / getcapbysig）。
所以 code 12 时它会误判成 "refresh"，继续换题硬试 ——
这正是实测「连续猛跑 8 次全部 12」时浪费时间的根源。

════════════════════════════════════════════════════════════════
实测踩到的三个坑（都已在这里规避）
════════════════════════════════════════════════════════════════
【坑 1】`listen.steps()` 是无限阻塞生成器
    DrissionPage 4.x:
        def steps(self, count=None, timeout=None, gap=1):
            if timeout is None:
                while self._driver.is_running and self.listening:
                    if self._caught.qsize() >= gap:
                        yield ...
                    sleep(.03)
    它内部是 while True。`list(steps())` 会**永远挂住**（实测卡死 10 分钟）。
    所以只能窥视内部队列 `_caught.queue`，或给 steps 传 timeout。

【坑 2】`Listener.start()` 会 clear() 队列，且已监听时只清空不换 target
        def start(self, targets=None, ...):
            if targets is not None: is_regex = False
            if targets or ...: self.set_targets(...)
            self.clear()             # ← 清空
            if self.listening: return # ← 已监听就直接返回
    1.0 的 start_capture() 用旧 targets 调 start，会清掉外部预订阅，
    且不会把 verify 加进去。必须在监听开启后单独 set_targets()。

【坑 3】窥视队列不能消费
    1.0 自己的 captured_image() 要从同一队列取图片包。
    如果我们 get_nowait() 消费掉，它就拿不到图了。
    所以只读 `q.queue`（浅拷贝窥视），不真正出队。

════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════
    import sys
    sys.path.insert(0, "D:/GitHub/Auto-Register/cun.ai/cun.ai 1.0")
    sys.path.insert(0, "D:/GitHub/Auto-Register/cun.ai/cun.ai 1.0/captcha")

    from drission_verify_ext import install, verify_with_retry
    install()
    ok = verify_with_retry(verifier)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

TZ = timezone(timedelta(hours=8))

CODE_OK = "0"       # 通过
CODE_RISK = "12"    # 风控拒绝：答案无法生效，必须退避
BACKOFF = (60, 180, 600)

INSTALLED_MARK = "_tsec_ext_installed"


def log(m):
    ts = datetime.now(TZ).strftime("%H:%M:%S")
    print("[" + ts + "] " + str(m), flush=True)


def _parse_body(txt):
    """天御响应可能是 JSON / JSONP(fn({...})) / ({...})。"""
    if txt is None:
        return None
    if isinstance(txt, (bytes, bytearray)):
        try:
            txt = bytes(txt).decode("utf-8", "replace")
        except Exception:
            return None
    t = str(txt).strip()
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.match(r"^\s*[\w$.]+\((.*)\)\s*;?\s*$", t, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    if t.startswith("(") and t.endswith(")"):
        try:
            return json.loads(t[1:-1])
        except Exception:
            return None
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return None
    return None


def _iter_packets(page):
    """窥视监听队列里**当前已到达**的包，绝不阻塞、绝不消费。"""
    try:
        q = page.listen._caught
    except Exception:
        return []
    items = []
    try:
        items = list(getattr(q, "queue", []))
    except Exception:
        return []
    flat = []
    for s in items:
        if isinstance(s, (list, tuple)):
            flat.extend(s)
        else:
            flat.append(s)
    return flat


def _packet_url(packet):
    try:
        u = getattr(packet, "url", None)
        if u:
            return str(u)
    except Exception:
        pass
    try:
        u = packet.request.url
        if u:
            return str(u)
    except Exception:
        pass
    return ""


def _packet_body_text(packet):
    """取响应体文本（JSON 也能取；1.0 的 _packet_body_bytes 只认图片）。"""
    try:
        body = packet.response.body
    except Exception:
        return None
    if body is None:
        return None
    if isinstance(body, (bytes, bytearray)):
        try:
            return bytes(body).decode("utf-8", "replace")
        except Exception:
            return None
    return str(body)


def harden_fingerprint(page, headless=True, width=1920, height=1080,
                       verbose=True):
    """修正会让反爬一眼识别的指纹破绽。

    实测（tools/fingerprint_audit.py + tools/probe_screen.py）：

        无头模式启动后 screen = 800x600       ← 无头默认虚拟显示器
        加 --window-size 只改 outer，screen 仍是 800x600
        CDP Emulation.setDeviceMetricsOverride 可改成 1920x1080   ✅
        有头模式 screen = 2560x1440（真实显示器）——此时**不要**覆盖，
        覆盖反而变成假值。

    结论：只在国内 headless 时用 CDP 修正 screen；有头保持真实值。

    为什么不靠改拖动速度：
        实测拖动已降到 136px/s（人类 300~800px/s），站点仍报
        「速度已超过 99% 的用户」——说明该提示并非真按速度判定，
        而是被识别为非人后的固定话术。真正该修的是指纹与出口 IP。
    """
    if not headless:
        if verbose:
            log("有头模式：保留真实 screen，不做覆盖")
        return False
    try:
        page.run_cdp("Emulation.setDeviceMetricsOverride",
                     width=width, height=height, deviceScaleFactor=1,
                     mobile=False, screenWidth=width, screenHeight=height)
        if verbose:
            try:
                got = page.run_js(
                    "return screen.width + 'x' + screen.height")
            except Exception:
                got = "?"
            log(f"已覆盖 screen -> {got}（原始 800x600 是反爬特征）")
        return True
    except Exception as e:
        if verbose:
            log(f"screen 覆盖失败: {type(e).__name__}: {e}")
        return False


def install(verbose: bool = True):
    """给 cun.ai 1.0 的 DrissionVerifier 打补丁（幂等）。"""
    try:
        from captcha.drission_verify import DrissionVerifier
    except Exception as e:
        raise SystemExit("导入 1.0 验证器失败: "
                         + type(e).__name__ + ": " + str(e))

    if getattr(DrissionVerifier, INSTALLED_MARK, False):
        if verbose:
            log("补丁已安装过，跳过")
        return DrissionVerifier

    # ---------- 0) start_capture：强制（重新）订阅含 verify 的 targets ----------
    orig_start = DrissionVerifier.start_capture

    def start_capture(self):
        """强制重新订阅。

        【坑 4，实测队列恒为空的原因】
            1.0 的 `fetch_image()` 是「listen 优先 + requests 兜底」，
            而 listen 经常抓不到（队列空），它就走 requests 兜底 —— 所以
            整个流程看起来正常，但队列一直是空的，我的补丁读不到任何包。

            另外 `Listener.start()`：
                - 第一步 `self.clear()` 清空队列
                - `if self.listening: return` —— 已监听时只清空，不换 target

            所以必须：先 stop，再 start（带新 targets），确保真正生效。
        """
        try:
            self.page.listen.stop()
        except Exception:
            pass
        try:
            self.page.listen.start(
                targets=["prehandle", "getcapbysig", "cap_union_new_verify"],
                method=("GET", "POST"), res_type=True)
        except Exception:
            try:
                orig_start(self)
            except Exception:
                pass

    DrissionVerifier.start_capture = start_capture

    # ---------- 1) 追加 target（不清队列） ----------
    def ensure_verify_targets(self):
        try:
            lst = self.page.listen
            cur = getattr(lst, "_targets", None)
            if cur is True:
                return True
            want = set(cur or []) | {"cap_union_new_verify"}
            lst.set_targets(targets=list(want),
                            method=("GET", "POST"), res_type=True)
            return True
        except Exception as e:
            log("ensure_verify_targets 失败: " + type(e).__name__ + ": " + str(e))
            return False

    DrissionVerifier.ensure_verify_targets = ensure_verify_targets

    # ---------- 2) 读应答码 ----------
    def read_verify_code(self):
        last = None
        for p in _iter_packets(self.page):
            url = _packet_url(p)
            if "cap_union_new_verify" not in url:
                continue
            j = _parse_body(_packet_body_text(p))
            if isinstance(j, dict):
                last = j
        return last

    DrissionVerifier.read_verify_code = read_verify_code

    # ---------- 3) _read_outcome 优先信应答码 ----------
    orig_outcome = DrissionVerifier._read_outcome

    def _read_outcome(self, kind):
        try:
            ensure_verify_targets(self)
        except Exception:
            pass
        vj = None
        try:
            vj = self.read_verify_code()
        except Exception:
            vj = None

        if isinstance(vj, dict):
            code = str(vj.get("errorCode", ""))
            has_ticket = bool(vj.get("ticket"))
            self._say("human_verify",
                      "应答码 errorCode=" + repr(code)
                      + " ticket=" + ("有" if has_ticket else "无"))
            if code == CODE_OK or has_ticket:
                try:
                    if not self.popup_open():
                        self._say("human_verify", kind + "验证通过（应答码 0）")
                        return "pass"
                except Exception:
                    pass
                time.sleep(1.2)
                try:
                    if not self.popup_open():
                        self._say("human_verify", kind + "验证通过（应答码 0）")
                        return "pass"
                except Exception:
                    pass
                return "pass"
            if code == CODE_RISK:
                self._say("human_verify", "应答码 12 = 风控拒绝，需退避")
                return "rate-limited"

        return orig_outcome(self, kind)

    DrissionVerifier._read_outcome = _read_outcome

    # ---------- 4) 修拖动：补 buttons=1 + 重写节奏 ----------
    #
    # 【坑 5，实测站点原话】
    #   桥接滑块定位后，1.0 真的开始拖了，但连续 3 轮都被判：
    #       校验状态: 您的速度已超过 99% 的用户    验证错误，请重试
    #
    #   原因：1.0 的 mouse_move 只发 type=mouseMoved，**没有 buttons=1**。
    #   CDP 里按钮状态由 buttons 位掩码表示；拖动过程中若 buttons=0，
    #   浏览器/页面可能不认这是一次"按住移动"，于是站点只看到
    #   mousedown → 终点 → mouseup，等价于瞬移 → 判定速度异常。
    #
    #   修法：拖动阶段所有 mouseMoved 都带 buttons=1，
    #   并把节奏改成「先快后慢 + 中段微停顿 + 终点回正」，总时长 ~1.8~2.8s。
    def mouse_move_drag(self, x, y):
        try:
            self.page.run_cdp("Input.dispatchMouseEvent", type="mouseMoved",
                              x=float(x), y=float(y), buttons=1)
        except Exception:
            try:
                self.mouse_move(x, y)
            except Exception:
                pass

    DrissionVerifier.mouse_move_drag = mouse_move_drag

    def human_drag(self, sx, sy, dx):
        import random as _r
        total = int(dx)
        # 先把指针移到起点（不按按钮，走原来的轨迹函数）
        try:
            self.human_move(sx, sy, dur=_r.uniform(0.5, 0.9))
        except Exception:
            pass
        time.sleep(_r.uniform(0.45, 0.9))
        try:
            self.mouse_down(sx, sy)
        except Exception:
            pass
        time.sleep(_r.uniform(0.18, 0.35))

        if total <= 0:
            try:
                self.mouse_up(sx, sy)
            except Exception:
                pass
            return

        # 总时长：按距离给，但限制在人类常见区间
        total_sec = min(2.8, max(1.1, total / 120.0))
        n = max(28, min(64, int(total / 4) + 24))
        pause_at = _r.uniform(0.35, 0.65)
        paused = False
        prev = 0
        t_used = 0.0
        for i in range(1, n + 1):
            frac = i / n
            # 先快后慢（ease-out），符合人类甩动后减速
            eased = 1 - (1 - frac) ** 2.4
            cur = int(round(total * eased))
            if cur == prev:
                continue
            step_sec = total_sec / n
            step_sec *= _r.uniform(0.7, 1.35)
            t_used += step_sec
            try:
                self.mouse_move_drag(sx + cur,
                                     sy + _r.uniform(-1.2, 1.2))
            except Exception:
                pass
            time.sleep(step_sec)
            prev = cur
            if not paused and frac >= pause_at:
                paused = True
                time.sleep(_r.uniform(0.18, 0.42))

        # 轻微过冲再回正（真人常见）
        over = _r.randint(2, 5)
        try:
            self.mouse_move_drag(sx + total + over, sy + _r.uniform(-0.8, 0.8))
        except Exception:
            pass
        time.sleep(_r.uniform(0.12, 0.26))
        try:
            self.mouse_move_drag(sx + total, sy)
        except Exception:
            pass
        time.sleep(_r.uniform(0.30, 0.55))
        try:
            self.mouse_up(sx + total, sy)
        except Exception:
            pass
        self._mx, self._my = float(sx + total), float(sy)
        log(f"拖动完成: dx={total} 用时≈{t_used:.2f}s "
            f"速度≈{total / max(t_used, 0.01):.0f}px/s")

    DrissionVerifier.human_drag = human_drag

    # ---------- 5) 拟人停顿：站点会判「速度已超过 99% 的用户」 ----------
    #
    # 【坑 6，实测站点原话】
    #   拖动速度已降到 136px/s（人类通常 300~800px/s，我们比人慢一倍多），
    #   但连续 3 轮仍被判：
    #       校验状态: 您的速度已超过 99% 的用户
    #
    #   结论：这条提示**不是**指拖动速度，而是指
    #   「从弹层出现到提交」的整体耗时。我们模型 200ms 就算完了，
    #   加上拖动 1.5s，全程 3~5s；而真人要看清图片再操作，通常 6~15s。
    #
    #   修法：在开始解题前插入「看图停顿」，并在停顿期间做少量
    #   无目的鼠标移动（真人不会一动不动地盯着）。
    def _solve_slider_once_slow(self, rnd):
        import random as _r
        think = _r.uniform(2.2, 4.6)
        log(f"拟人看图停顿 {think:.1f}s（规避『速度超过99%用户』判定）")
        t0 = time.time()
        while time.time() - t0 < think:
            try:
                self.mouse_move(self._mx + _r.uniform(-6, 6),
                                self._my + _r.uniform(-5, 5))
            except Exception:
                pass
            time.sleep(_r.uniform(0.25, 0.7))
        return orig_slider(self, rnd)

    orig_slider = DrissionVerifier._solve_slider_once
    DrissionVerifier._solve_slider_once = _solve_slider_once_slow

    setattr(DrissionVerifier, INSTALLED_MARK, True)
    if verbose:
        log("已安装补丁：应答码判定 + 拟人拖动 + 看图停顿")
    return DrissionVerifier


def verify_with_retry(verifier, max_rounds: int = 3, backoff=BACKOFF,
                      on_wait=None):
    """带风控退避的 ensure_verified。

    与直接调用的差别：命中 errorCode=12 时不立刻放弃，
    按 60 → 180 → 600 秒退避后重试（实测 600s 可完全恢复）。
    """
    try:
        from captcha.drission_verify import CaptchaRateLimitedError
    except Exception:
        CaptchaRateLimitedError = RuntimeError  # type: ignore

    for i in range(1, max_rounds + 1):
        try:
            log("验证尝试 " + str(i) + "/" + str(max_rounds))
            ok = verifier.ensure_verified()
            if ok:
                log("验证通过")
                return True
            log("验证未通过（非风控），准备重试")
        except CaptchaRateLimitedError as e:
            wait = backoff[min(i - 1, len(backoff) - 1)]
            log("命中风控（" + str(e) + "），冷却 " + str(wait) + "s 后重试")
            if on_wait:
                try:
                    on_wait(wait)
                except Exception:
                    pass
            time.sleep(wait)
            continue
        except Exception as e:
            log("验证异常 " + type(e).__name__ + ": " + str(e))
        if i < max_rounds:
            time.sleep(3)
    log("重试用尽，未通过")
    return False


if __name__ == "__main__":
    PROJ = r"D:\GitHub\Auto-Register\cun.ai\cun.ai 1.0"
    for _p in (PROJ, os.path.join(PROJ, "captcha")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    cls = install()
    print("DrissionVerifier 已增强: " + str(cls))
    print("  read_verify_code     : " + str(hasattr(cls, "read_verify_code")))
    print("  ensure_verify_targets: " + str(hasattr(cls, "ensure_verify_targets")))
    print("  _read_outcome        : " + str(hasattr(cls, "_read_outcome")))
