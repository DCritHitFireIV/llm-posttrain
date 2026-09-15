#!/usr/bin/env python
"""终评（一次性）：test n=2000，SFT 与 RL 两臂配对贪心评测。

守卫（A′ 同款纪律）：
  · state/TEST_TOUCHED + state/eval_attempts.json
  · 若已 touch 且已有任一臂产物 → 拒绝重跑（rc=3）
  · 本脚本只读 test.jsonl；写 logs/eval_{sft,rl}_n2000.jsonl + state/EVAL.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as R  # noqa: E402

from peft import PeftModel, prepare_model_for_kbit_training  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

_spec = importlib.util.spec_from_file_location("grpo_src", os.path.join(R.ROOT, "src", "grpo.py"))
G = importlib.util.module_from_spec(_spec)
sys.modules["grpo_src"] = G
_spec.loader.exec_module(G)
C = R.C

TAG = "evaluate"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_two(base_path, sft_path, rl_path, log):
    log(f"加载 4bit 基座 {base_path}")
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb,
                                                 device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = True   # 纯推理：开 KV cache 加速贪心生成
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, sft_path, is_trainable=False, adapter_name="sft")
    model.load_adapter(rl_path, adapter_name="rl", is_trainable=False)
    log("双 adapter 就绪: sft / rl")
    return model, tok


def eval_batch(model, tok, chunk, adapter, max_new):
    prompts = [R.PROMPT_T.format(q=it["question"]) for it in chunk]
    tok.padding_side = "left"
    G.set_adapter(model, adapter)
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(G.DEV)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    tok.padding_side = "right"
    in_len = enc["input_ids"].shape[1]
    out = []
    for j, it in enumerate(chunk):
        text = tok.decode(gen[j, in_len:], skip_special_tokens=True)
        out.append({"id": it["id"], "text": text, "correct": R.correct_of(text, it["gold"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=os.path.join(R.DATA, "test.jsonl"))
    ap.add_argument("--sft-adapter", default=os.path.join(R.ROOT, "outputs", "sft_lora"))
    ap.add_argument("--rl-adapter", default=None, help="缺省读 state/SELECT.json 的 best_ckpt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    log_path = os.path.join(R.LOGS, "evaluate.log")

    def log(m):
        C.log(m, log_path)

    sel = json.load(open(os.path.join(R.STATE, "SELECT.json"), encoding="utf-8"))
    rl_path = args.rl_adapter or os.path.join(R.OUT, sel["best_ckpt"])
    assert os.path.isdir(rl_path), f"RL adapter 不存在：{rl_path}"

    sft_out = os.path.join(R.LOGS, "eval_sft_n2000.jsonl")
    rl_out = os.path.join(R.LOGS, "eval_rl_n2000.jsonl")
    touched = os.path.join(R.STATE, "TEST_TOUCHED")
    attempts = os.path.join(R.STATE, "eval_attempts.json")
    if os.path.exists(touched) and (os.path.exists(sft_out) or os.path.exists(rl_out)):
        log("检测到 TEST 已有产物且已 touch → 拒绝重跑（一次性纪律）")
        sys.exit(3)
    R.marker("TEST_TOUCHED", f"S4 {C.now()} rl={rl_path}")
    n_att = 0
    if os.path.exists(attempts):
        try:
            n_att = json.load(open(attempts, encoding="utf-8")).get("attempts", 0)
        except Exception:
            pass
    C.write_json_atomic(attempts, {"attempts": n_att + 1, "last": C.now(), "rl": rl_path})

    items = R.load_jsonl(args.test)
    # [BUDGET] 预算截断（不改动其余评测逻辑）：state/EVAL_N.txt 存在且为正整数 n 时，
    # 仅评测 test.jsonl 前 n 题；文件不存在、为空、非法或 n<=0 时用全量。
    n_available = len(items)
    n_budget = 0
    evn_path = os.path.join(R.STATE, "EVAL_N.txt")
    if os.path.exists(evn_path):
        try:
            n_budget = int(open(evn_path, encoding="utf-8").read().strip() or "0")
        except (OSError, ValueError):
            n_budget = 0
            log(f"[BUDGET] EVAL_N.txt 非法（{evn_path}）→ 按全量评测")
    n_used = min(n_budget, n_available) if n_budget > 0 else n_available
    if n_used < n_available:
        items = items[:n_used]
        log(f"[BUDGET] 预算截断：仅评测 test 前 {n_used}/{n_available} 题（EVAL_N.txt={n_budget}）")
    else:
        log(f"[BUDGET] 未截断：全量 {n_used}/{n_available} 题（EVAL_N.txt={n_budget}）")
    log(f"test 题数 {len(items)}；两臂同 session 交替评测（batch={args.batch}）")
    model, tok = load_two(os.path.join(R.ROOT, "models", "Qwen3-1.7B-Base"), args.sft_adapter, rl_path, log)

    n_sft = n_rl = 0
    for i0 in range(0, len(items), args.batch):
        chunk = items[i0:i0 + args.batch]
        rs = eval_batch(model, tok, chunk, "sft", args.max_new)
        rr = eval_batch(model, tok, chunk, "rl", args.max_new)
        for r in rs:
            C.append_jsonl(sft_out, r)
        for r in rr:
            C.append_jsonl(rl_out, r)
        n_sft += len(rs)
        n_rl += len(rr)
        if (i0 // args.batch) % 10 == 0:
            log(f"[eval] {n_sft}/{len(items)}")
            R.hb("S4_eval", done=n_sft, total=len(items))
    acc_sft = sum(1 for l in open(sft_out, encoding="utf-8") if json.loads(l)["correct"]) / len(items)
    acc_rl = sum(1 for l in open(rl_out, encoding="utf-8") if json.loads(l)["correct"]) / len(items)
    ev = {"test": sha(args.test), "sft_adapter": args.sft_adapter, "rl_adapter": rl_path,
          "n": len(items), "batch": args.batch, "model": os.path.join(R.ROOT, "models", "Qwen3-1.7B-Base"),
          "acc_sft": round(acc_sft, 4), "acc_rl": round(acc_rl, 4),
          "sft_out": sft_out, "rl_out": rl_out, "attempt": n_att + 1}
    ev["n_used"], ev["n_available"] = n_used, n_available  # [BUDGET] 如实登记实际评测题数
    C.write_json_atomic(os.path.join(R.STATE, "EVAL.json"), ev)
    R.marker("S4.done", f"acc_sft={acc_sft:.4f} acc_rl={acc_rl:.4f}")
    log(f"终评完成 sft={acc_sft:.4f} rl={acc_rl:.4f}")
    print("EVAL DONE " + json.dumps(ev, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
