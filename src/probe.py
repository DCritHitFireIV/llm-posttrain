#!/usr/bin/env python
"""Qwen3-1.7B-Base 环境探针 + GSM8K 能力基线

三个功能：
  1. 4bit 加载模型，报告显存占用
  2. 一次生成冒烟测试
  3. GSM8K few-shot 评测（默认 50 题，可调）

用法（容器内）:
  cd /workspace/llm-posttrain
  python src/probe.py --model models/Qwen3-1.7B-Base --n 50 --shots 4
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import pyarrow.parquet as pq
except ImportError:
    print("need pyarrow: pip install pyarrow")
    sys.exit(1)


def load_parquet(path):
    return pq.read_table(path).to_pylist()


def extract_answer(text: str):
    """从生成文本抽取答案：优先 #### 后的数字，否则最后一个数字"""
    m = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[-+]?[\d,]+(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return None


def build_prompt(shots, question):
    """GSM8K few-shot completion 格式"""
    parts = []
    for s in shots:
        parts.append(f"Question: {s['question']}\nAnswer: {s['answer']}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-1.7B-Base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=50, help="评测题数")
    ap.add_argument("--shots", type=int, default=4, help="few-shot 数量")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default="logs/probe_result.json")
    ap.add_argument("--skip-eval", action="store_true", help="只测显存不评测")
    args = ap.parse_args()

    result = {"model": args.model, "n": args.n, "shots": args.shots}

    # ---------- 1. 加载 ----------
    print(f"[1/3] 加载 tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("[1/3] 加载模型 (4bit NF4 + bf16 compute)...")
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    load_s = time.time() - t0
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"    加载耗时 {load_s:.1f}s | 权重显存 {alloc:.2f} GB (reserved {reserved:.2f} GB)")
    result["load_seconds"] = round(load_s, 1)
    result["vram_after_load_gb"] = round(alloc, 2)

    # ---------- 2. 冒烟测试 ----------
    print("[2/3] 生成冒烟测试...")
    torch.cuda.reset_peak_memory_stats()
    smoke = tok("The capital of France is", return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**smoke, max_new_tokens=16, do_sample=False)
    smoke_text = tok.decode(out[0][smoke["input_ids"].shape[1]:], skip_special_tokens=True)
    gen_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"    输出: {smoke_text!r} | {gen_s:.1f}s | 峰值显存 {peak:.2f} GB")
    result["smoke_output"] = smoke_text
    result["vram_peak_gen_gb"] = round(peak, 2)

    if args.skip_eval:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"结果写入 {args.out}")
        return

    # ---------- 3. GSM8K 评测 ----------
    print(f"[3/3] GSM8K 评测 ({args.n} 题, {args.shots}-shot)...")
    train = load_parquet(os.path.join(args.data, "gsm8k_train.parquet"))
    test = load_parquet(os.path.join(args.data, "gsm8k_test.parquet"))
    shots = train[: args.shots]

    correct, total = 0, 0
    details = []
    t_start = time.time()
    for i, ex in enumerate(test[: args.n]):
        prompt = build_prompt(shots, ex["question"])
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred, gold = extract_answer(gen), extract_answer(ex["answer"])
        ok = pred == gold
        correct += ok
        total += 1
        details.append({"i": i, "pred": pred, "gold": gold, "ok": ok, "gen_len": len(gen)})
        if (i + 1) % 10 == 0:
            acc = correct / total
            el = time.time() - t_start
            print(f"    [{i+1}/{args.n}] acc={acc:.3f} 用时{el:.0f}s 预计剩余{el/(i+1)*(args.n-i-1):.0f}s")

    acc = correct / total
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n=== 基座 GSM8K 准确率: {acc:.4f} ({correct}/{total}) | 峰值显存 {peak:.2f} GB ===")
    result.update(
        accuracy=round(acc, 4),
        correct=correct,
        total=total,
        eval_seconds=round(time.time() - t_start, 1),
        vram_peak_eval_gb=round(peak, 2),
        details=details,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"结果写入 {args.out}")


if __name__ == "__main__":
    main()
