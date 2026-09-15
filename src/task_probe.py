#!/usr/bin/env python
"""预注册前置：采样温度标定（仅用 train-only val 子集）。

规则（事先冻结，写进实验协议）：
  在 temp ∈ {0.3, 1.0} 中，取「mixed 组比例」更高者；平局取「all_wrong 更低」者；
  再平局取「adv_scale 更高」者；仍平局取 1.0。
产物：results/TEMP_PROBE.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("grpo_src", "./src/grpo.py")
G = importlib.util.module_from_spec(_spec)
sys.modules["grpo_src"] = G
_spec.loader.exec_module(G)

ROOT = "."
STATE = os.path.join(ROOT, "state")
os.makedirs(STATE, exist_ok=True)

TEMPS = [0.3, 1.0]
N_ITEMS = 24
G_SIZE = 8
MAX_NEW = 256


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=os.path.join(ROOT, "data", "val.jsonl"))
    ap.add_argument("--out", default=os.path.join(STATE, "TEMP_PROBE.json"))
    ap.add_argument("--n-items", type=int, default=N_ITEMS)
    args = ap.parse_args()

    torch.manual_seed(20260914)
    ga = argparse.Namespace(model=os.path.join(ROOT, "models", "Qwen3-1.7B-Base"),
                            adapter=os.path.join(ROOT, "outputs", "sft_lora"),
                            load_4bit=1, lora_r=16, lora_alpha=32,
                            max_prompt_len=256, max_new_tokens=MAX_NEW,
                            temperature=1.0, top_p=0.95, group_size=G_SIZE, micro_batch=1)
    model, tok, _ = G.load_model(ga, lambda m: print(m, flush=True))
    items = load_jsonl(args.val)[: args.n_items]
    rep = {"rule": "max mixed; tie->min all_wrong; tie->max adv_scale; tie->1.0",
           "n_items": len(items), "G": G_SIZE, "results": {}, "chosen": None}
    for temp in TEMPS:
        ga.temperature = temp
        model.eval()
        model.gradient_checkpointing_disable()
        G.set_adapter(model, "policy")
        t0 = time.time()
        rates = []
        for it in items:
            prompt = G.build_prompt(it["question"])
            comps = G.sample_group(model, tok, prompt, ga)
            cs = [G.reward_of(t, it["gold"]) for _, t in comps]
            rates.append(sum(cs) / len(cs))
        acc = sum(rates) / len(rates)
        mixed = sum(1 for p in rates if 0 < p < 1) / len(rates)
        all_w = sum(1 for p in rates if p == 0) / len(rates)
        all_c = sum(1 for p in rates if p == 1) / len(rates)
        rep["results"][str(temp)] = {
            "acc": round(acc, 4), "mixed": round(mixed, 4), "all_wrong": round(all_w, 4),
            "all_correct": round(all_c, 4), "zero_adv": round(all_w + all_c, 4),
            "adv_scale": round(sum(math.sqrt(p * (1 - p)) for p in rates) / len(rates), 4),
            "sec": round(time.time() - t0, 1)}
        print(f"[probe temp={temp}] {rep['results'][str(temp)]}", flush=True)

    r = rep["results"]
    best = None
    for temp in TEMPS:
        s = r[str(temp)]
        key = (s["mixed"], -s["all_wrong"], s["adv_scale"], 1.0 if temp == 1.0 else 0.0)
        if best is None or key > best[0]:
            best = (key, temp)
    rep["chosen"] = best[1]
    C.write_json_atomic(args.out, rep)
    print(f"TEMP PROBE DONE chosen={rep['chosen']}", flush=True)


if __name__ == "__main__":
    main()
