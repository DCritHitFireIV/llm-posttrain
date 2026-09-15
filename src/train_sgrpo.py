#!/usr/bin/env python
"""主训：S-GRPO（arXiv 2504.20834）在 multistep 甜区任务上的 8GB 适配实现。

忠实实现 S-GRPO：
  · 每题采样 G=8 完成；组内标准化 advantage
  · token 入选：t<α 恒选；t≥α 且 |T_i|<k 时 Bernoulli(P)；其余不选
  · 每题仅 1 次更新（μ=1，无 PPO clip），ref 不更新
  · 损失 = -(1/G)Σ_i (1/|T_i|)Σ_{t∈T_i}[ (π_θ/π_θ^nograd)·Â − β·D_KL(π_θ‖π_ref) ]
  · LoRA：q,v（只挂最后 1/3 层）；AdamW lr=1e-4；β_KL=0.01；weight_decay=0.01
起点：4bit Qwen3-1.7B-Base + GSM8K SFT adapter（policy 可训练副本 / ref 冻结）

用法：
  train_sgrpo.py --steps 450 --temp 1.0 --out-dir outputs
产出：outputs/ckpt_XXXX/、state/metrics.jsonl、state/SELECT.json、state/heartbeat.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as R  # noqa: E402

_spec = importlib.util.spec_from_file_location("grpo_src", os.path.join(R.ROOT, "src", "grpo.py"))
G = importlib.util.module_from_spec(_spec)
sys.modules["grpo_src"] = G
_spec.loader.exec_module(G)

C = R.C


def make_ga(args):
    return argparse.Namespace(
        model=os.path.join(R.ROOT, "models", "Qwen3-1.7B-Base"),
        adapter=os.path.join(R.ROOT, "outputs", "sft_lora"),
        load_4bit=1, lora_r=16, lora_alpha=32,
        max_prompt_len=256, max_new_tokens=args.max_new,
        temperature=args.temp, top_p=0.95, group_size=args.group, micro_batch=1,
    )


def inclusion_mask(comp_len: int, alpha: int, k: int, p: float, rng: random.Random) -> list[int]:
    """返回入选的 completion 内下标（0-based）。算法规定。"""
    out, cnt = [], 0
    for t in range(comp_len):
        inc = False
        if t < alpha:
            inc = True
        elif cnt < k and rng.random() < p:
            inc = True
        if inc:
            out.append(t)
            cnt += 1
    return out


def sgrpo_step(model, tok, opt, ga, item, args, step):
    prompt = G.build_prompt(item["question"])
    p_ids = tok(prompt, add_special_tokens=False).input_ids[: ga.max_prompt_len]
    torch.manual_seed(args.seed + step)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    G.set_adapter(model, "policy")
    t_s = time.time()
    with torch.no_grad():
        comps = G.sample_group(model, tok, prompt, ga)
    sample_s = time.time() - t_s

    seqs, rewards, texts, lens = [], [], [], []
    for c_ids, c_text in comps:
        if not c_ids:
            continue
        seqs.append((p_ids, c_ids))
        texts.append(c_text)
        rewards.append(G.reward_of(c_text, item["gold"]))
        lens.append(len(c_ids))
    if not seqs or sum(lens) == 0:
        return None
    adv = G.group_advantages(rewards)

    rng = random.Random(args.seed * 1000 + step)
    incl = [inclusion_mask(L, args.alpha, args.k, args.p, rng) for L in lens]

    # ref logprobs（no_grad，冻结 SFT adapter）
    ref_lps = G.forward_logprobs(model, seqs, "ref", 1, tok.pad_token_id)
    assert len(ref_lps) == len(seqs)

    # 训练前向（逐序列 microbatch=1，梯度累积后一次性 step）
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    G.set_adapter(model, "policy")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n = len(seqs)
    bg = [1.0 if r >= 0.5 else 0.0 for r in rewards]
    mixed = 1.0 if 0 < sum(bg) < n else 0.0
    tot_incl = sum(len(x) for x in incl)
    kl_sum = adv_abs = 0.0
    loss_val = 0.0
    t_t = time.time()
    for i in range(n):
        p, c = seqs[i]
        input_ids, attn, _ = G.build_batch([seqs[i]], tok.pad_token_id)
        input_ids, attn = input_ids.to(G.DEV), attn.to(G.DEV)
        logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        lp = G.token_logprobs(logits[:, :-1, :], input_ids[:, 1:])          # (1,T-1)
        r = ref_lps[i].to(lp.device)                                        # (1,T-1) detached
        m = torch.zeros_like(lp)
        base = len(p) - 1
        for t in incl[i]:
            m[0, base + t] = 1.0
        if m.sum() < 0.5:
            del logits, lp
            continue
        ratio = torch.exp(lp - lp.detach())          # ≡1（μ=1，无 clip），梯度走 lp
        kl = torch.exp(r - lp) - (r - lp) - 1.0      # k3 无偏估计
        per_tok = ratio * float(adv[i]) - args.beta_kl * kl
        seq_loss = (per_tok * m).sum() / m.sum()
        (-seq_loss / n).backward()
        with torch.no_grad():
            kl_sum += (kl * m).sum().item()
            adv_abs += abs(float(adv[i])) * m.sum().item()
            loss_val += seq_loss.item() / n
        del logits, lp, r, m, ratio, kl, per_tok, seq_loss
    gn = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
    opt.step()
    opt.zero_grad(set_to_none=True)
    train_s = time.time() - t_t

    return {
        "step": step, "reward_mean": sum(rewards) / n, "reward_std": float(
            torch.as_tensor(rewards).std().item()) if n > 1 else 0.0,
        "mixed": mixed, "n": n, "all_zero_adv": bool(torch.all(adv == 0)),
        "loss": round(loss_val, 5), "kl_mean_incl": round(kl_sum / max(tot_incl, 1), 5),
        "adv_abs": round(adv_abs / max(tot_incl, 1), 4),
        "included_frac": round(tot_incl / max(sum(lens), 1), 4),
        "avg_len": round(sum(lens) / n, 1), "gnorm": float(gn),
        "peak_gb": round(G.vram_peak(), 2), "sample_s": round(sample_s, 2),
        "train_s": round(train_s, 2), "sec": round(sample_s + train_s, 2),
        "text0": texts[0][:120].replace("\n", " "),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=50)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--beta-kl", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--val-batch", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(R.OUT))
    ap.add_argument("--val", default=os.path.join(R.DATA, "val.jsonl"))
    ap.add_argument("--train", default=os.path.join(R.DATA, "train.jsonl"))
    args = ap.parse_args()

    log_path = os.path.join(R.LOGS, "train_sgrpo.log")

    def log(msg):
        C.log(msg, log_path)
    metrics_path = os.path.join(R.STATE, "metrics.jsonl")

    train_items = R.load_jsonl(args.train)
    val_items = R.load_jsonl(args.val)
    ga = make_ga(args)

    log(f"=== S-GRPO 训练开始 steps={args.steps} temp={args.temp} G={args.group} α={args.alpha} k={args.k} P={args.p} ===")
    model, tok, _ = G.load_model(ga, lambda m: log(m))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    done = 0
    if os.path.exists(metrics_path):
        for line in open(metrics_path, encoding="utf-8"):
            try:
                done = max(done, json.loads(line)["step"] + 1)
            except Exception:
                pass
    log(f"resume: 从 step {done} 继续")

    t0 = time.time()
    for step in range(done, args.steps):
        item = train_items[step % len(train_items)]
        st = sgrpo_step(model, tok, opt, ga, item, args, step)
        if st is None:
            log(f"[step {step}] 空 completion，跳过")
            continue
        C.append_jsonl(metrics_path, st)
        R.hb("S3_train", step=step, steps=args.steps, reward=st["reward_mean"], sec=st["sec"])
        if step % 5 == 0 or step == args.steps - 1:
            log(f"[step {step}] r={st['reward_mean']:.2f} mixed={st['mixed']:.0f} loss={st['loss']} "
                  f"kl={st['kl_mean_incl']} incl={st['included_frac']} sec={st['sec']} peak={st['peak_gb']}GB")
        if (step + 1) % args.save_every == 0:
            ck = os.path.join(args.out_dir, f"ckpt_{step+1:04d}")
            G.save_policy(model, ck, tok)
            log(f"[ckpt] {ck}")
    log(f"训练结束，用时 {(time.time()-t0)/60:.1f} min")

    # ---- val 选择（train-only；贪心批量） ----
    ckpts = sorted([d for d in os.listdir(args.out_dir) if d.startswith("ckpt_")])
    log(f"val 选择：{len(ckpts)} 个 ckpt × {len(val_items)} 题（batch={args.val_batch}）")
    val = {"sft": {}, "ckpts": {}, "n_val": len(val_items)}
    sft_res = R.batched_greedy_eval(model, tok, val_items, "ref", max_new=args.max_new,
                                    batch=args.val_batch, log=lambda m: log(m), tag=":sft")
    val["sft"] = round(sum(r["correct"] for r in sft_res) / len(sft_res), 4)
    for ck in ckpts:
        model.load_adapter(os.path.join(args.out_dir, ck), adapter_name=ck, is_trainable=False)
        res = R.batched_greedy_eval(model, tok, val_items, ck, max_new=args.max_new,
                                    batch=args.val_batch, log=lambda m: log(m), tag=f":{ck}")
        val["ckpts"][ck] = round(sum(r["correct"] for r in res) / len(res), 4)
        log(f"[val {ck}] acc={val['ckpts'][ck]}")
        R.hb("S4_val", ck=ck, acc=val["ckpts"][ck])
    best = max(sorted(val["ckpts"]), key=lambda c: (val["ckpts"][c], -int(c.split("_")[1])))
    sel = {"val": val, "best_ckpt": best, "best_val_acc": val["ckpts"][best],
           "sft_val_acc": val["sft"], "protocol": "max val acc; tie -> earliest ckpt"}
    C.write_json_atomic(os.path.join(R.STATE, "SELECT.json"), sel)
    log(f"SELECT best={best} val={val['ckpts'][best]}（SFT val={val['sft']}）")
    R.marker("S3.done", f"best={best}")
    print("ROOT SGRPO DONE " + json.dumps(sel["val"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
