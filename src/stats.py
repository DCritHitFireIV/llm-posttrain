#!/usr/bin/env python
"""统计：配对 McNemar 精确检验 + bootstrap CI + MDE + 报告。"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as R  # noqa: E402

C = R.C


def main():
    sft_p = os.path.join(R.LOGS, "eval_sft_n2000.jsonl")
    rl_p = os.path.join(R.LOGS, "eval_rl_n2000.jsonl")
    sft = [json.loads(l) for l in open(sft_p, encoding="utf-8") if l.strip()]
    rl = [json.loads(l) for l in open(rl_p, encoding="utf-8") if l.strip()]
    assert len(sft) == len(rl), "两臂题数不一致"
    for a, b in zip(sft, rl):
        assert a["id"] == b["id"], f"题目顺序不一致 {a['id']} vs {b['id']}"
    n = len(sft)
    a = [1 if r["correct"] else 0 for r in sft]
    b = [1 if r["correct"] else 0 for r in rl]
    acc_a = sum(a) / n
    acc_b = sum(b) / n
    b_only = sum(1 for x, y in zip(a, b) if y == 1 and x == 0)
    c_only = sum(1 for x, y in zip(a, b) if y == 0 and x == 1)
    both_ok = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    both_no = n - b_only - c_only - both_ok
    d = b_only + c_only
    k = min(b_only, c_only)
    p_exact = 2.0 * sum(comb(d, i) for i in range(k + 1)) * (0.5 ** d) if d > 0 else 1.0
    p_exact = min(p_exact, 1.0)

    rng = random.Random(20260914)
    deltas = []
    for _ in range(10000):
        delta = 0
        for _i in range(n):
            j = rng.randrange(n)
            delta += b[j] - a[j]
        deltas.append(delta / n)
    deltas.sort()
    ci_lo = deltas[int(0.025 * 10000)]
    ci_hi = deltas[int(0.975 * 10000) - 1]
    mde_pp = (100.0 * 2.8 * math.sqrt(d) / n) if d > 0 else None

    stats = {
        "n": n, "test_sha256": R.C.sha256_file(os.path.join(R.DATA, "test.jsonl")),
        "acc_sft": round(acc_a, 4), "acc_rl": round(acc_b, 4),
        "delta_rl_minus_sft_pp": round(100 * (acc_b - acc_a), 3),
        "both_correct": both_ok, "rl_only": b_only, "sft_only": c_only, "both_wrong": both_no,
        "discordant": d, "mcnemar_exact_p": p_exact,
        "bootstrap": {"n_boot": 10000, "seed": 20260914, "ci95_pp": [round(100 * ci_lo, 3), round(100 * ci_hi, 3)]},
        "mde_pp_approx_80pct_power": round(mde_pp, 2) if mde_pp is not None else None,
        "success_line": {"definition": "McNemar exact p<0.05 且 bootstrap 95% CI 下界>0",
                         "pass": bool(p_exact < 0.05 and ci_lo > 0)},
        "backend": "HF 4bit (NF4+bf16) 贪心 pass@1；两臂同 session 交替生成",
    }
    C.write_json_atomic(os.path.join(R.STATE, "STATS.json"), stats)

    sel = json.load(open(os.path.join(R.STATE, "SELECT.json"), encoding="utf-8"))
    lines = [
        "# 报告：S-GRPO on multistep", "",
        f"- 结论：{'**双门通过（正结果）**' if stats['success_line']['pass'] else '未通过成功线（负结果）'}",
        f"- test n={n}；SFT={acc_a:.4f}（{sum(a)}/{n}）；RL={acc_b:.4f}（{sum(b)}/{n}）；Δ={stats['delta_rl_minus_sft_pp']:+.3f}pp",
        f"- McNemar 精确 p={p_exact:.6f}；bootstrap 95% CI={stats['bootstrap']['ci95_pp']}pp；MDE≈{mde_pp:.2f}pp" if mde_pp else "",
        f"- discordant={d}（RL-only={b_only} / SFT-only={c_only}）；both_correct={both_ok}；both_wrong={both_no}", "",
        "## 协议要点",
        "- 任务：multistep 程序化应用题（5–6 步），split seeds train=3001 / val=3002 / test=3003（test 一次性）",
        "- 起点：4bit Qwen3-1.7B-Base + GSM8K SFT adapter（policy 副本可训 / ref 冻结）",
        "- 方法：S-GRPO，G=8，α=50 / k=100 / P=0.5，μ=1 无 clip，β_KL=0.01，LoRA q,v 最后 1/3 层 r16，lr=1e-4，450 步",
        f"- ckpt 选择：val 贪心最优（{sel['best_ckpt']}，val={sel['best_val_acc']}；SFT val={sel['sft_val_acc']}）",
        f"- 采样温度：{json.load(open(os.path.join(R.STATE, 'TEMP_PROBE.json'), encoding='utf-8'))['chosen'] if os.path.exists(os.path.join(R.STATE, 'TEMP_PROBE.json')) else 'n/a'}", "",
        "## 解读护栏",
        "- 绝对值是 HF 4bit 贪心口径；两臂同后端同 session，配对 Δ 有效。",
        "- 未做 test 后调参；任何偏离见 DEVIATIONS.md。",
    ]
    rep = os.path.join(R.ROOT, "reports")
    os.makedirs(rep, exist_ok=True)
    open(os.path.join(rep, "REPORT.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    R.marker("FINAL", f"pass={stats['success_line']['pass']} delta_pp={stats['delta_rl_minus_sft_pp']}")
    print("ROOT STATS DONE " + json.dumps(stats["success_line"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
