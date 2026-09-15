#!/usr/bin/env python
"""手写 LoRA SFT 训练循环 —— Qwen3-1.7B-Base on GSM8K

自研边界：
  - 手写：数据集构造、prompt/completion label mask、训练循环、梯度累积、
          余弦调度、grad clipping、保存
  - 用库：transformers（加载/前向）、peft（LoRA 注入）、bitsandbytes（4bit）

用法（容器内）:
  cd llm-posttrain
  python src/sft.py --data data/sft_train.jsonl --out outputs/sft_lora \
      --epochs 2 --batch-size 4 --grad-accum 4 --lr 2e-4 --dry-run   # 先干跑验证
"""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class GSM8KDataset(Dataset):
    """prompt/completion 格式：只对 completion 计算 loss（prompt 部分 mask 为 -100）"""

    def __init__(self, path, tok, max_len=512):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = f"Question: {r['question']}\nAnswer:"
        completion = f" {r['answer']}{self.tok.eos_token}"
        p_ids = self.tok(prompt, add_special_tokens=False).input_ids
        c_ids = self.tok(completion, add_special_tokens=False).input_ids
        input_ids = (p_ids + c_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + c_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def cosine_schedule(step, total, warmup, base_lr, min_ratio=0.1):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-1.7B-Base")
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--out", default="outputs/sft_lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500, help="每多少 optimizer step 存一次")
    ap.add_argument("--max-steps", type=int, default=0, help=">0 时截断训练（干跑用）")
    ap.add_argument("--dry-run", action="store_true", help="只跑 20 step 验证显存")
    args = ap.parse_args()

    # ---------- 模型 ----------
    print(f"加载 tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("加载 4bit 基座...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---------- 数据 ----------
    ds = GSM8KDataset(args.data, tok, max_len=args.max_len)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        num_workers=0,
    )
    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.dry_run:
        total_steps = min(total_steps, 20)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * args.warmup_frac))
    print(f"数据 {len(ds)} 条 | batch {args.batch_size}×accum {args.grad_accum} | "
          f"总 optimizer step ≈ {total_steps} (warmup {warmup})")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )

    # ---------- 训练 ----------
    model.train()
    torch.cuda.reset_peak_memory_stats()
    done, running, t0 = 0, 0.0, time.time()
    running_tok = 0
    stop = False

    for epoch in range(math.ceil(args.epochs) * 2):  # epoch 上限保护（epochs 可为小数）
        for step, batch in enumerate(dl):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += loss.item()
            running_tok += int(batch["attention_mask"].sum())

            if (step + 1) % args.grad_accum == 0:
                # 手动设置本 step 的 LR（余弦 + warmup）
                lr_now = cosine_schedule(done, total_steps, warmup, args.lr)
                for g in opt.param_groups:
                    g["lr"] = lr_now
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                done += 1

                if done % args.log_every == 0 or done == 1:
                    el = time.time() - t0
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    print(
                        f"[step {done}/{total_steps}] loss={running/args.log_every:.4f} "
                        f"lr={lr_now:.2e} {running_tok/el:.0f} tok/s peak_vram={peak:.2f}GB el={el:.0f}s"
                    )
                    running, running_tok, t0 = 0.0, 0, time.time()

                if done % args.save_every == 0:
                    os.makedirs(args.out, exist_ok=True)
                    model.save_pretrained(args.out)
                    print(f"  ↳ 已保存 LoRA adapter → {args.out}")

                if done >= total_steps:
                    stop = True
                    break
        if stop:
            break

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n完成 {done} step | 峰值显存 {peak:.2f} GB | LoRA 保存至 {args.out}")


if __name__ == "__main__":
    main()
