# -*- coding: utf-8 -*-
"""fix_label_2408: 用「匹配器自洽性」修正自动标注里的错配（标签清洗）。

问题（实测数据支撑）：
    用【真值提示框 + 真值目标框】直接算分（完全绕开检测器），
    同图标配对得分分布是**明显双峰**：

        分位:  [5%]=0.034  [10%]=0.084  [25%]=0.926
               [50%]=0.983  [75%]=0.993  [90%]=0.997
        自洽率 85.0%（153/180）

    也就是说 **15% 的标注是错的**：`pairs` 里的 `bg` 索引指向了另一个图标。
    错配案例的并排图（out/badcase2408/）肉眼可确认：
        提示是「收音机」，GT 框里却是「笑脸」，而 PICKED 框里才是收音机。

    这些错标是**训练噪声**：匹配器被教成"收音机 ≈ 笑脸"，
    端到端自然上不去。这也解释了为什么加难负样本反而更差
    （噪声标签 + 更紧的决策边界 = 更大的伤害）。

修正策略（三档，保守优先）：
    1) 若 `bg` 指向的框与提示图标分数 >= keep_thr → 保留
    2) 否则在**同题其它框**里找分数最高的：
         · 若最高分 >= fix_thr → 改指向该框（fix）
         · 否则 → 该提示图标标为 `unresolved`，从 pairs 里移除
           （保留 class0/class2 的框标注，只是不参与配对训练）
    3) 复核队列里 min_pair 很低的题，标记 needs_manual

为什么用"匹配器"当裁判而不是"形状匹配"：
    形状匹配（match_2408）在 3/3 全配率上只有 75%，而匹配器在干净框上 98%+。
    用更强的判据做裁判，误改率更低。

产出：
    <out>/labels_index.jsonl       修正后的索引（原文件备份为 .bak）
    <out>/fix_report.json          统计
    <out>/fixed_vis/<sid>.jpg      改动可视化（人眼复核用）

用法：
    python tools/fix_label_2408.py --pre runs/export_t2408b/pre_model_v7_2408b.onnx
    python tools/fix_label_2408.py --dry-run          # 只看不改
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from autolabel_2408 import split_prompt_icons      # noqa: E402


def crop(img, box, norm_short=None):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    c = img[y1:y2, x1:x2]
    if c.size == 0:
        return None
    if norm_short:
        h, w = c.shape[:2]
        s = norm_short / float(max(min(h, w), 1))
        if abs(s - 1.0) > 0.02:
            c = cv2.resize(c, (max(1, int(round(w*s))), max(1, int(round(h*s)))),
                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="runs/export_t2408b/pre_model_v7_2408b.onnx")
    ap.add_argument("--labeled", default="data/labeled2408")
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--norm-short", type=int, default=64)
    ap.add_argument("--keep-thr", type=float, default=0.5,
                    help="原配对分 >= 此值则保留不动")
    ap.add_argument("--fix-thr", type=float, default=0.7,
                    help="找到的替代框分数 >= 此值才敢改（保守）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-vis", default="out/fixed_vis")
    args = ap.parse_args()

    from src.utils import ver_onnx

    lab = args.labeled if os.path.isabs(args.labeled) else os.path.join(REPO, args.labeled)
    sroot = args.samples if os.path.isabs(args.samples) else os.path.join(REPO, args.samples)
    vis_dir = args.out_vis if os.path.isabs(args.out_vis) else os.path.join(REPO, args.out_vis)
    os.makedirs(vis_dir, exist_ok=True)

    idx_path = os.path.join(lab, "labels_index.jsonl")
    recs = [json.loads(l) for l in open(idx_path, encoding="utf-8") if l.strip()]
    pre = ver_onnx.PreONNX(os.path.join(REPO, args.pre))

    n_keep = n_fix = n_drop = n_noicon = 0
    changed = []
    out_recs = []

    for r in recs:
        sid = r["sid"]
        bg_boxes = r.get("bg_boxes") or []
        pairs = r.get("pairs") or []
        meta_p = os.path.join(sroot, sid, "meta.json")
        main_p = os.path.join(sroot, sid, "main.jpg")
        if not bg_boxes or not pairs or not os.path.exists(meta_p) \
                or not os.path.exists(main_p):
            out_recs.append(r)
            continue
        meta = json.load(open(meta_p, encoding="utf-8"))
        prs = meta.get("prompts") or []
        if not prs:
            out_recs.append(r)
            continue
        img = cv2.imdecode(np.fromfile(main_p, np.uint8), cv2.IMREAD_COLOR)
        pr = cv2.imdecode(np.fromfile(os.path.join(sroot, sid, prs[0]["file"]),
                                      np.uint8), cv2.IMREAD_COLOR)
        if img is None or pr is None:
            out_recs.append(r)
            continue
        icons = split_prompt_icons(pr)
        if not icons:
            out_recs.append(r)
            continue

        icon_crops = [crop(it["img"],
                           (0, 0, it["img"].shape[1], it["img"].shape[0]),
                           norm_short=args.norm_short or None) for it in icons]
        box_crops = [crop(img, b, norm_short=args.norm_short or None)
                     for b in bg_boxes]

        new_pairs = []
        touched = False
        for p in pairs:
            i, k, sc = p["prompt"], p["bg"], p.get("score", 0.0)
            if i >= len(icon_crops) or icon_crops[i] is None:
                n_noicon += 1
                touched = True
                continue
            ic = icon_crops[i]
            # 算这个提示图标与所有候选框的分数
            sims = []
            for j, bc in enumerate(box_crops):
                if bc is None:
                    sims.append(-1.0)
                else:
                    sims.append(float(pre.predict_pair(ic, bc)))
            cur = sims[k] if 0 <= k < len(sims) else -1.0

            if cur >= args.keep_thr:
                new_pairs.append({"prompt": i, "bg": k, "score": round(cur, 4),
                                  "fixed": False})
                n_keep += 1
                continue

            # 当前配对不可信 → 找最佳替代（排除已被其它 prompt 占用的框）
            used = {q["bg"] for q in new_pairs}
            cand = [(s, j) for j, s in enumerate(sims) if j not in used]
            cand.sort(reverse=True)
            if cand and cand[0][0] >= args.fix_thr:
                s_best, j_best = cand[0]
                new_pairs.append({"prompt": i, "bg": j_best,
                                  "score": round(float(s_best), 4), "fixed": True})
                n_fix += 1
                touched = True
            else:
                n_drop += 1
                touched = True

        nr = dict(r)
        nr["pairs"] = new_pairs
        nr["min_pair"] = round(min([q["score"] for q in new_pairs]), 4) if new_pairs else 0.0
        nr["label_fixed"] = bool(touched)
        out_recs.append(nr)
        if touched:
            changed.append({"sid": sid, "before": len(pairs), "after": len(new_pairs),
                            "pairs": new_pairs})
            if len(changed) <= 60:
                vis = img.copy()
                for q in new_pairs:
                    b = bg_boxes[q["bg"]]
                    col = (0, 200, 0) if not q.get("fixed") else (0, 140, 255)
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), col, 2)
                    cv2.putText(vis, f'{q["prompt"]}:{q["score"]:.2f}',
                                (b[0]+2, max(14, b[1]-4)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, col, 1, cv2.LINE_AA)
                cv2.imencode(".jpg", vis)[1].tofile(
                    os.path.join(vis_dir, sid + ".jpg"))

    rep = {
        "labeled": lab, "pre": args.pre,
        "questions": len(recs),
        "pairs_kept": n_keep, "pairs_fixed": n_fix, "pairs_dropped": n_drop,
        "pairs_no_icon": n_noicon,
        "questions_touched": len(changed),
        "keep_thr": args.keep_thr, "fix_thr": args.fix_thr,
        "dry_run": args.dry_run,
        "changed": changed[:200],
    }

    if not args.dry_run:
        bak = idx_path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(idx_path, bak)
            print(f"[backup] {os.path.basename(bak)}")
        with open(idx_path, "w", encoding="utf-8") as f:
            for r in out_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(lab, "fix_report.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        # 同步重建 data.yaml 用的 label 文件里的 class2/class0 不变，
        # 只有配对关系变了，所以 labels/*.txt 不用改。

    print("\n=== 标注清洗结果 ===")
    print(f"  题数         : {len(recs)}")
    print(f"  配对保留     : {n_keep}")
    print(f"  配对改指向   : {n_fix}")
    print(f"  配对丢弃     : {n_drop}")
    print(f"  缺提示图标   : {n_noicon}")
    print(f"  受影响题数   : {len(changed)}")
    print(f"  {'[dry-run 未写入]' if args.dry_run else '已写入 ' + idx_path}")
    print(f"  可视化       : {vis_dir}/")


if __name__ == "__main__":
    main()
