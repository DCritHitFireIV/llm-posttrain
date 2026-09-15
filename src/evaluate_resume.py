#!/usr/bin/env python
"""S4 终评续跑（恢复通道，不重跑已完成题）。

背景：首次 S4 在 1552/2000 对处因 `torch.AcceleratorError: CUDA error: unknown error`
中断（共享 GPU 环境）。本脚本从已完成对数处继续追加写同一对 JSONL（确定性贪心，
每臂每题仍只生成一次；不涉及任何重新选择/重跑已完成项）。

保护：每 batch 失败 → empty_cache + 重试 ≤3；仍失败则 batch 减半（8→4→2）继续。
完成：写 state/EVAL.json（n_used/n_available/acc_sft/acc_rl）+ state/S4.done。
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
    model.config.use_cache = True
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


def count_lines(p):
    if not os.path.exists(p):
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for l in f if l.strip())


def truncate_to(p, n):
    """只保留前 n 行（两臂计数不一致时丢弃未配对的尾部 batch）"""
    lines = [l for l in open(p, encoding="utf-8") if l.strip()]
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(lines[:n])


def eval_retry(model, tok, chunk, adapter, max_new, log, batch):
    for attempt in range(4):
        try:
            return eval_batch(model, tok, chunk, adapter, max_new), batch
        except torch.AcceleratorError as e:
            log(f"[retry] {adapter} batch={len(chunk)} attempt={attempt+1} CUDA error: {str(e)[:120]}")
            torch.cuda.empty_cache()
            import time as _t
            _t.sleep(20)
    raise RuntimeError("batch 连续失败 4 次")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=os.path.join(R.DATA, "test.jsonl"))
    ap.add_argument("--sft-adapter", default=os.path.join(R.ROOT, "outputs", "sft_lora"))
    ap.add_argument("--rl-adapter", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    log_path = os.path.join(R.LOGS, "evaluate_resume.log")

    def log(m):
        C.log(m, log_path)

    sel = json.load(open(os.path.join(R.STATE, "SELECT.json"), encoding="utf-8"))
    rl_path = args.rl_adapter or os.path.join(R.OUT, sel["best_ckpt"])
    assert os.path.isdir(rl_path), f"RL adapter 不存在: {rl_path}"

    sft_out = os.path.join(R.LOGS, "eval_sft_n2000.jsonl")
    rl_out = os.path.join(R.LOGS, "eval_rl_n2000.jsonl")
    n_sft, n_rl = count_lines(sft_out), count_lines(rl_out)
    n_done = min(n_sft, n_rl)
    if n_sft != n_rl:
        log(f"计数不一致 sft={n_sft} rl={n_rl} → 截断到 {n_done}（丢弃未配对尾部）")
        truncate_to(sft_out, n_done)
        truncate_to(rl_out, n_done)
    items_all = R.load_jsonl(args.test)
    n_avail = len(items_all)
    assert n_done <= n_avail
    items = items_all[n_done:]
    log(f"续跑：已完成 {n_done}/{n_avail}，剩余 {len(items)} 题（batch={args.batch}）")

    model, tok = load_two(os.path.join(R.ROOT, "models", "Qwen3-1.7B-Base"), args.sft_adapter, rl_path, log)
    batch = args.batch
    i = 0
    while i < len(items):
        chunk = items[i:i + batch]
        rs, batch_used = eval_retry(model, tok, chunk, "sft", args.max_new, log, batch)
        rr, _ = eval_retry(model, tok, chunk, "rl", args.max_new, log, batch)
        for r in rs:
            C.append_jsonl(sft_out, r)
        for r in rr:
            C.append_jsonl(rl_out, r)
        i += len(chunk)
        if (n_done + i) % 100 < batch:
            log(f"[eval] {n_done + i}/{n_avail} (batch={batch})")
            R.hb("S4_eval_resume", done=n_done + i, total=n_avail)
            torch.cuda.empty_cache()
    acc_sft = sum(1 for l in open(sft_out, encoding="utf-8") if json.loads(l)["correct"]) / n_avail
    acc_rl = sum(1 for l in open(rl_out, encoding="utf-8") if json.loads(l)["correct"]) / n_avail
    ev = {"test": sha(args.test), "sft_adapter": args.sft_adapter, "rl_adapter": rl_path,
          "n": n_avail, "n_used": n_avail, "n_available": n_avail, "batch": args.batch,
          "model": os.path.join(R.ROOT, "models", "Qwen3-1.7B-Base"),
          "acc_sft": round(acc_sft, 4), "acc_rl": round(acc_rl, 4),
          "sft_out": sft_out, "rl_out": rl_out, "resumed_from": n_done}
    C.write_json_atomic(os.path.join(R.STATE, "EVAL.json"), ev)
    R.marker("S4.done", f"acc_sft={acc_sft:.4f} acc_rl={acc_rl:.4f} (resumed@{n_done})")
    log(f"续跑完成 sft={acc_sft:.4f} rl={acc_rl:.4f}")
    print("EVAL RESUME DONE " + json.dumps(ev, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
