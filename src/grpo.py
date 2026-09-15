#!/usr/bin/env python
"""手写 GRPO 训练脚本 —— Qwen3-1.7B-Base + GSM8K（8GB 显存硬约束）

自研边界（遵循用户要求：核心算法手写，不调用 TRL / trainer 库）
  - 手写：K 样本采样循环、可验证奖励（GSM8K 抽取答案判对错）、组内标准化 advantage、
          策略梯度损失（REINFORCE + 可选 PPO clip）、KL 惩罚项、完整训练循环
  - 用库：transformers（加载/前向/generate）、peft（LoRA 注入 / adapter 加载）、
          bitsandbytes（4bit 量化）

显存设计（关键，8GB 卡上必须）
  1. 只加载 **一份** 4bit 基座；策略 adapter 与参考 adapter 共用同一份基座权重，
     参考模型 = 冻结的 SFT adapter（用 set_adapter 切换），省掉第二份 1.7B 权重。
  2. 采样阶段 model.eval() + torch.no_grad()。
  3. 训练阶段只更新 LoRA 参数（基座 4bit 冻结），optimizer 只接 requires_grad 的参数。
  4. 前向按 --micro-batch 切分 + 梯度累积，把 logits/log_softmax 的 (B,T,V) 峰值压住
     （V=151936，是显存主要来源）。
  5. 参考 logprob / old logprob 全部在 no_grad 下预先算好，只缓存 (B,T) 的小张量。

用法:
  cd /workspace/llm-posttrain
  /opt/train-venv/bin/python -u src/grpo.py --dry-run          # 干跑 1 步 + 显存报告
  /opt/train-venv/bin/python -u src/grpo.py --smoke            # 1 步快速验证
  /opt/train-venv/bin/python -u src/grpo.py --steps 200        # 正式训练
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# 【坑】必须在这里做模块级 import。4bit 分支只在 CUDA 上走到，若把
# prepare_model_for_kbit_training 写成函数内 import，CPU 干跑永远发现不了它缺失，
# 一到 GPU 就直接 NameError（本实验实测）。模块级 import 让 CPU 也能挡住这类错。
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

try:
    import pyarrow.parquet as pq
except ImportError:
    print("need pyarrow: pip install pyarrow")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import extract_answer  # noqa: E402  复用 GSM8K 答案抽取


# --------------------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------------------
class Logger:
    """同时写 stdout 和 logs/grpo_train.log"""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


# --------------------------------------------------------------------------------------
# 设备 / 显存（CPU 上运行时全部退化为 0，便于无 GPU 的逻辑干跑）
# --------------------------------------------------------------------------------------
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEV.type == "cuda"


def vram_alloc():
    return torch.cuda.memory_allocated() / 1024**3 if IS_CUDA else 0.0


def vram_peak():
    return torch.cuda.max_memory_allocated() / 1024**3 if IS_CUDA else 0.0


def vram_reserved_peak():
    return torch.cuda.max_memory_reserved() / 1024**3 if IS_CUDA else 0.0


def vram_reset_peak():
    if IS_CUDA:
        torch.cuda.reset_peak_memory_stats()


def vram_total():
    return torch.cuda.get_device_properties(0).total_memory / 1024**3 if IS_CUDA else 0.0


# --------------------------------------------------------------------------------------
# 数据 / 奖励
# --------------------------------------------------------------------------------------
def build_prompt(question):
    """与 src/sft.py 的 SFT 格式保持一致（base 模型 completion 格式，非 chat）"""
    return f"Question: {question}\nAnswer:"


def reward_of(completion_text, gold_answer):
    """可验证奖励：抽取答案与 gold 一致 +1，否则 0"""
    pred = extract_answer(completion_text)
    if pred is None:
        return 0.0
    try:  # 数值等价（72 == 72.0）优先，否则字符串比较
        return 1.0 if float(pred) == float(gold_answer) else 0.0
    except (TypeError, ValueError):
        return 1.0 if str(pred).strip() == str(gold_answer).strip() else 0.0


def group_advantages(rewards, eps=1e-4):
    """组内标准化 advantage: (r - mean(r)) / (std(r) + eps)

    组内奖励全相同 → std≈0 → advantage 全 0（该组无学习信号，属预期行为）。
    """
    r = torch.as_tensor(rewards, dtype=torch.float32)
    if r.numel() <= 1:
        return torch.zeros_like(r)
    adv = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return adv


# --------------------------------------------------------------------------------------
# 序列构造 / logprob
# --------------------------------------------------------------------------------------
def build_batch(seqs, pad_id):
    """seqs: [(prompt_ids, comp_ids), ...] → 右 padding 的 input_ids / attn / loss_mask

    loss_mask[b, t] = 1 表示第 t 个 **预测位置**（logits[:, t] 预测 input_ids[:, t+1]）
    对应 completion token。prompt 段与 padding 段均为 0。
    """
    maxlen = max(len(p) + len(c) for p, c in seqs)
    B = len(seqs)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    loss_mask = torch.zeros((B, maxlen - 1), dtype=torch.float32)
    for b, (p, c) in enumerate(seqs):
        ids = list(p) + list(c)
        input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[b, : len(ids)] = 1
        if c:
            loss_mask[b, len(p) - 1 : len(p) + len(c) - 1] = 1.0
    return input_ids, attn, loss_mask


def token_logprobs(logits, targets):
    """logits (B,T,V), targets (B,T) → 每个 target 的 log π(target)  (B,T)

    log_softmax 在 fp32 下做（bf16 对 151936 类的归一化精度不够）。
    """
    logits = logits.float()
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def masked_mean(x, mask, dim=None):
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1.0)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1.0)


# --------------------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------------------
def load_model(args, log):
    log(f"加载 tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # 训练前向用右 padding

    t0 = time.time()
    if args.load_4bit:
        log("加载 4bit 基座 (NF4 + bf16 compute + double quant)...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True
        )
    else:
        log(f"加载基座（未量化，device={DEV}）...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, trust_remote_code=True
        ).to(DEV)
    model.config.use_cache = False
    if args.load_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    log(f"    基座加载 {time.time()-t0:.1f}s | 权重显存 {vram_alloc():.2f} GB")

    adapter_path = args.adapter if (args.adapter and os.path.isdir(args.adapter)) else None
    if adapter_path:

        log(f"[起点] 4bit 基座 + SFT LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True, adapter_name="policy")
        # 参考策略 = 冻结的 SFT adapter（与策略共用同一份 4bit 基座，显存零额外权重开销）
        model.load_adapter(adapter_path, adapter_name="ref", is_trainable=False)
    else:

        log(f"[起点] 未找到 adapter ({args.adapter})，从基座起步（新建 LoRA；ref=零初始化 LoRA≡基座）")
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg, adapter_name="policy")
        model.add_adapter("ref", lora_cfg)

    set_adapter(model, "policy")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"    可训练参数 {n_tr/1e6:.1f}M (仅 policy LoRA) | 显存 {vram_alloc():.2f} GB")
    return model, tok, adapter_path


def set_adapter(model, name):
    """切换 adapter，并强制不变量：只允许 policy LoRA 可训练，其余参数全部冻结。

    【坑】peft 的 set_adapter() 会把 *当前激活* 的 adapter 标为可训练、其它 adapter 冻结。
    因此每次 set_adapter("ref") 之后，policy 的 requires_grad 会被置 False，若不在切回时
    重新断言，反向传播将拿不到任何 LoRA 梯度（loss 正常但参数永不更新）。这里每次切换后
    统一重新断言，保证语义与激活的 adapter 无关。
    """
    model.set_adapter(name)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(".policy." in n)


# --------------------------------------------------------------------------------------
# 采样
# --------------------------------------------------------------------------------------
def sample_group(model, tok, prompt, args):
    """K 样本采样循环（手写）：对同一 prompt 生成 group_size 个 completion

    返回 [(comp_ids, comp_text), ...]
    """
    ids = tok(prompt, return_tensors="pt", truncation=True,
              max_length=args.max_prompt_len).to(DEV)
    prompt_len = ids["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.group_size,
            pad_token_id=tok.pad_token_id,
        )
    results = []
    for i in range(out.shape[0]):
        new_ids = out[i, prompt_len:]
        # 截断到 eos（含）
        eos_pos = (new_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            new_ids = new_ids[: eos_pos[0] + 1]
        text = tok.decode(new_ids, skip_special_tokens=True)
        results.append((new_ids.tolist(), text))
    return results


# --------------------------------------------------------------------------------------
# 前向：参考 / old logprob（no_grad，预先算好，只缓存 (B,T)）
# --------------------------------------------------------------------------------------
def forward_logprobs(model, seqs, adapter, micro_batch, pad_id, log=None, tag=""):
    """在 no_grad 下算一批序列的逐 token logprob，返回与 micro-batch 对齐的列表"""
    chunks, out = chunk_seqs(seqs, micro_batch), []
    set_adapter(model, adapter)
    with torch.no_grad():
        for mb in chunks:
            input_ids, attn, _ = build_batch(mb, pad_id)
            input_ids, attn = input_ids.to(DEV), attn.to(DEV)
            logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
            lp = token_logprobs(logits[:, :-1, :], input_ids[:, 1:])
            out.append(lp.detach())
            del logits, lp
    return out


def chunk_seqs(seqs, n):
    return [seqs[i : i + n] for i in range(0, len(seqs), n)]


# --------------------------------------------------------------------------------------
# 单步训练
# --------------------------------------------------------------------------------------
def train_step(model, opt, tok, examples, args, log, step):
    """一个 GRPO step：采样 → 奖励 → advantage → 策略梯度 + KL → 反传 → 更新"""
    pad_id = tok.pad_token_id
    t0 = time.time()

    # ---------- 1. K 样本采样 ----------
    model.eval()
    model.gradient_checkpointing_disable()  # generate 期间关闭，否则 use_cache 失效
    set_adapter(model, "policy")
    t_samp0 = time.time()          # 【仅计时】不改变任何数值
    seqs, rewards, groups, texts, golds = [], [], [], [], []
    for gi, ex in enumerate(examples):
        prompt = build_prompt(ex["question"])
        p_ids = tok(prompt, add_special_tokens=False).input_ids[: args.max_prompt_len]
        gold = extract_answer(ex["answer"])
        comps = sample_group(model, tok, prompt, args)
        gr = []
        for c_ids, c_text in comps:
            if not c_ids:            # 空 completion 直接跳过（无 token 可算 logprob）
                continue
            seqs.append((p_ids, c_ids))
            texts.append(c_text)
            golds.append(gold)
            gr.append(reward_of(c_text, gold))
        rewards.extend(gr)
        groups.append(gr)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    t_sample = time.time() - t_samp0      # 【仅计时】纯采样耗时

    if not seqs:
        log(f"[step {step}] 全部 completion 为空，跳过")
        return None

    # ---------- 2. 组内标准化 advantage ----------
    adv = []
    for gr in groups:
        adv.extend(group_advantages(gr).tolist())
    adv_t = torch.tensor(adv, dtype=torch.float32, device=DEV)

    n_seq = len(seqs)
    acc = sum(rewards) / len(rewards)
    std_r = float(torch.tensor(rewards, dtype=torch.float32).std(unbiased=False))
    # 【GRPO 健康度关键指标】组内奖励全同 → advantage 全 0 → 该组完全无学习信号。
    # 组大小 K 越小、模型越"自信"（全对或全错），这种退化组越多。必须显式监控，
    # 否则会看到 reward 正常、loss/gnorm 恒为 0 而误以为训练在正常进行。
    uni_groups = sum(1 for gr in groups if len(set(gr)) <= 1)
    uni_frac = uni_groups / max(1, len(groups))

    # ---------- 3. 参考策略 logprob（冻结 SFT adapter，与策略共用 4bit 基座）----------
    #   k3 模式：no_grad 预算好 ref logprob，只缓存 (B,T) 小张量，训练前向无需再存 ref logits
    t_lp0 = time.time()            # 【仅计时】ref(+old) logprob 前向
    ref_lp = forward_logprobs(model, seqs, "ref", args.micro_batch, pad_id) \
        if args.kl_mode == "k3" else None

    # ---------- 4. old logprob（仅 inner-epochs>1 时需要，用于 PPO ratio）----------
    old_lp = None
    if args.inner_epochs > 1:
        old_lp = forward_logprobs(model, seqs, "policy", args.micro_batch, pad_id)
    t_logprob = time.time() - t_lp0

    # ---------- 5. 策略梯度 + KL，梯度累积 ----------
    t_train0 = time.time()         # 【仅计时】前向+反传+更新
    model.train()
    set_adapter(model, "policy")
    opt.zero_grad(set_to_none=True)
    chunks = chunk_seqs(seqs, args.micro_batch)
    n_mb = len(chunks)
    # 预先构造所有 micro-batch 的掩码，得到全局 token 数，用于 token-level 归一。
    # 【坑】若按"每条序列先平均、再对序列取平均"归一，K=2 的对称组（reward=[1,0] →
    # advantage=[+a,-a]）损失会**恰好抵消为 0**，看起来像没学到东西，其实梯度非零。
    # 采用 GRPO 标准的 token-level 归一：Σ_t(A_t·logπ) / Σ_t 1，既避免抵消，
    # 又保证"梯度累积 == 一次性大 batch 前向"在数学上严格等价。
    built = [build_batch(mb, pad_id) for mb in chunks]
    total_tokens = sum(float(m.sum()) for _, _, m in built) or 1.0
    stats = {"pg": 0.0, "kl": 0.0, "clip": 0.0, "absadv": 0.0}
    params = [p for p in model.parameters() if p.requires_grad]
    n_ep = max(1, args.inner_epochs)
    gn = torch.tensor(0.0)

    # inner-epochs > 1 时才真正进入多轮循环：第 1 轮用采样策略的 logprob 作 old，
    # 后续轮次复用同一份 old，于是 ratio 偏离 1，PPO 的 clip 才真正起作用。
    for ep in range(n_ep):
        set_adapter(model, "policy")
        opt.zero_grad(set_to_none=True)
        idx0 = 0
        for ci, mb in enumerate(chunks):
            input_ids, attn, loss_mask = built[ci]
            input_ids, attn, loss_mask = input_ids.to(DEV), attn.to(DEV), loss_mask.to(DEV)
            logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
            pred = logits[:, :-1, :]                      # (B, L-1, V)
            tgt = input_ids[:, 1:]
            m = loss_mask
            A = adv_t[idx0 : idx0 + len(mb)].unsqueeze(1)

            # ---- 采样 token 的 logprob（策略梯度用，两种 KL 口径都需要）----
            lp = token_logprobs(pred, tgt)                # (B,T) 有梯度

            # ---- KL 惩罚项（两种口径）----
            if args.kl_mode == "exact":
                # 精确全词表 KL(π‖π_ref) = Σ_v π(v)·(logπ(v) − logπ_ref(v))
                # 参考 logits 在同 micro-batch 内重新前向（no_grad），显存随 micro-batch 有界。
                # 注意：必须在 backward 前切回 policy —— 梯度检查点的重算依赖当前激活的 adapter。
                p_lp = F.log_softmax(pred.float(), dim=-1)
                set_adapter(model, "ref")
                with torch.no_grad():
                    r_logits = model(input_ids=input_ids, attention_mask=attn,
                                     use_cache=False).logits
                    q_lp = F.log_softmax(r_logits[:, :-1, :].float(), dim=-1)
                    del r_logits
                set_adapter(model, "policy")
                kl_tok = (p_lp.exp() * (p_lp - q_lp)).sum(-1)   # (B, L-1) 有梯度
                del p_lp, q_lp
            else:
                # k3 估计量（Schulman 近似，无偏且非负），只用到采样 token：O(B·T) 显存
                log_ratio = ref_lp[ci] - lp
                kl_tok = torch.exp(log_ratio) - log_ratio - 1.0
            del pred, input_ids, attn

            # ---- 策略梯度损失（PPO/GRPO 裁剪代理目标，token-level 归一）----
            # 【坑】inner_epochs=1 时不能把 surr 直接写成 A（丢掉 lp），否则损失对策略**完全
            # 没有依赖**：策略梯度项梯度恒为 0，只剩 KL 项在提供梯度（而 KL 在起点梯度恰为 0），
            # 表现为"loss 看着正常、参数几乎不动"。正确做法是始终走 ratio 形式：
            # old = lp.detach() 时 ratio 数值上恒等于 1，但 d(ratio)/dθ = ratio·d(lp)/dθ ≠ 0，
            # 裁剪代理目标恰好退化为 REINFORCE 的梯度。
            old = old_lp[ci] if old_lp is not None else lp.detach()
            ratio = torch.exp(lp - old)
            unclipped = ratio * A
            if old_lp is not None:
                clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * A
                surr = torch.min(unclipped, clipped)
                with torch.no_grad():
                    stats["clip"] += ((unclipped > clipped).float() * m).sum().item() / total_tokens
            else:
                surr = unclipped
            pg_sum = -(surr * m).sum()
            kl_sum = (kl_tok * m).sum()
            loss = (pg_sum + args.kl_coef * kl_sum) / total_tokens
            loss.backward()

            with torch.no_grad():
                stats["pg"] += pg_sum.item() / total_tokens
                stats["kl"] += kl_sum.item() / total_tokens
                stats["absadv"] += (A.abs() * m).sum().item() / total_tokens
            idx0 += len(mb)
            del lp, kl_tok, pg_sum, kl_sum, loss, m, surr, A, ratio, unclipped, old

        # ---------- 6. 梯度裁剪 + 更新（只更新 LoRA 参数）----------
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        # 注意：这里**不要** reset_peak_memory_stats()。reset 会把峰值基线抬到当前占用，
        # 从而丢掉前面 epoch / 采样阶段的真实峰值，让干跑的显存报告偏小。

    peak = vram_peak()
    t_train = time.time() - t_train0
    n_tok = sum(len(c) for _, c in seqs)
    n_capped = sum(1 for _, c in seqs if len(c) >= args.max_new_tokens)
    return {
        "step": step, "loss": (stats["pg"] + args.kl_coef * stats["kl"]) / n_ep,
        "pg_loss": stats["pg"] / n_ep, "kl": stats["kl"] / n_ep,
        "reward": acc, "std_r": std_r,
        "acc": acc, "gnorm": float(gn), "peak_gb": peak,
        "mean_abs_adv": stats["absadv"] / n_ep, "inner_epochs": n_ep,
        "uni_frac": uni_frac, "group_rewards": [list(g) for g in groups],
        "n_seq": n_seq, "avg_len": sum(len(c) for _, c in seqs) / n_seq,
        "clip_frac": stats["clip"] / n_ep, "sec": time.time() - t0,
        # 【仅计时/吞吐，不参与任何数值】阶段耗时与采样吞吐
        "sample_s": round(t_sample, 2), "logprob_s": round(t_logprob, 2),
        "train_s": round(t_train, 2), "n_tok": n_tok,
        "tok_per_s": round(n_tok / max(t_sample, 1e-6), 1),
        "n_capped": n_capped,
        "sample": texts[0][:160].replace("\n", " "),
    }


def save_policy(model, out_dir, tok=None):
    """只保存 policy adapter 到 out_dir 顶层（可直接 PeftModel.from_pretrained 加载）。

    【坑】peft 0.20 的 save_pretrained() 在多 adapter 场景下会把每个 adapter 写进
    out/<adapter_name>/ 子目录（`selected_adapters` 也一样），这样 outputs/grpo_lora/
    顶层就没有 adapter_config.json，加载端得多套一层路径。这里保存后把 policy 子目录
    的内容搬到顶层，得到与 outputs/sft_lora/ 一致、可直接加载的布局。
    """
    os.makedirs(out_dir, exist_ok=True)
    set_adapter(model, "policy")
    tmp = out_dir.rstrip("/") + ".tmp_adapter"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, selected_adapters=["policy"])
    src = os.path.join(tmp, "policy")
    if not os.path.isdir(src):          # 单 adapter 时可能直接落在 tmp 顶层
        src = tmp
    for f in os.listdir(src):
        dst = os.path.join(out_dir, f)
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        elif os.path.exists(dst):
            os.remove(dst)
        shutil.move(os.path.join(src, f), dst)
    shutil.rmtree(tmp, ignore_errors=True)
    if tok is not None:
        tok.save_pretrained(out_dir)
    return out_dir


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-1.7B-Base")
    ap.add_argument("--adapter", default="outputs/sft_lora", help="SFT LoRA 起点（不存在则从基座起步）")
    ap.add_argument("--data", default="data/gsm8k_train.parquet")
    ap.add_argument("--out", default="outputs/grpo_lora")
    ap.add_argument("--log", default="logs/grpo_train.log")
    # GRPO 超参
    ap.add_argument("--group-size", type=int, default=4, help="K：每个 prompt 采样数")
    ap.add_argument("--kl-coef", type=float, default=0.04)
    ap.add_argument("--kl-mode", choices=["k3", "exact"], default="k3")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2, help="PPO ratio 裁剪（inner-epochs>1 时生效）")
    ap.add_argument("--inner-epochs", type=int, default=1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-prompt-len", type=int, default=256)
    ap.add_argument("--micro-batch", type=int, default=2, help="前向切分大小（显存开关）")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda / cpu；默认自动（CPU 用于无 GPU 逻辑干跑）")
    ap.add_argument("--load-4bit", type=int, default=1, help="1=4bit 量化（仅 CUDA 有效）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true", help="只跑 1 步 + 显存峰值报告，不保存")
    ap.add_argument("--smoke", action="store_true", help="1 步快速验证")
    args = ap.parse_args()

    if args.smoke:
        args.steps = 1
    if args.dry_run:
        args.steps = 1

    global DEV, IS_CUDA
    if args.device:
        DEV = torch.device(args.device)
        IS_CUDA = DEV.type == "cuda"
    if DEV.type == "cpu":
        args.load_4bit = 0
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    log = Logger(args.log)
    log("=" * 78)
    log(f"GRPO run @ {time.strftime('%F %T')} | dry_run={args.dry_run} smoke={args.smoke}")
    log(f"  group_size={args.group_size} kl_coef={args.kl_coef} kl_mode={args.kl_mode} "
        f"lr={args.lr} steps={args.steps} prompts/step={args.prompts_per_step} "
        f"max_new_tokens={args.max_new_tokens} micro_batch={args.micro_batch}")
    log(f"  inner_epochs={args.inner_epochs} clip_eps={args.clip_eps} temp={args.temperature}")

    if IS_CUDA:
        torch.cuda.empty_cache()
    vram_reset_peak()
    base_used = vram_alloc()

    # ---------- 模型 ----------
    model, tok, adapter_path = load_model(args, log)
    vram_after_load = vram_alloc()
    vram_after_load_peak = vram_peak()

    # ---------- 数据 ----------
    rows = pq.read_table(args.data).to_pylist()
    log(f"GSM8K prompts: {len(rows)} 条 ({args.data})")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.0,
    )

    # ---------- 训练循环 ----------
    vram_reset_peak()
    history, t_start = [], time.time()
    for step in range(1, args.steps + 1):
        examples = [rows[random.randrange(len(rows))] for _ in range(args.prompts_per_step)]
        rec = train_step(model, opt, tok, examples, args, log, step)
        if rec is None:
            continue
        history.append(rec)
        if step % args.log_every == 0 or step == 1:
            # 注意：组内 advantage 恒有 ΣA=0，ratio≡1 时裁剪代理目标的**数值**自然接近 0，
            # 因此 loss 不能当作"是否在学"的指标；gnorm / mean|A| / reward 才是。
            log(f"[step {step}/{args.steps}] loss={rec['loss']:+.4f} "
                f"(pg={rec['pg_loss']:+.4f} kl={rec['kl']:.4f}) "
                f"reward={rec['reward']:.3f} acc={rec['acc']:.3f} std_r={rec['std_r']:.3f} "
                f"|A|={rec['mean_abs_adv']:.3f} gnorm={rec['gnorm']:.3f} "
                f"uni_grp={rec['uni_frac']:.2f} "
                f"peak_vram={rec['peak_gb']:.2f}GB "
                f"n={rec['n_seq']} len={rec['avg_len']:.0f} {rec['sec']:.0f}s "
                f"[sample={rec['sample_s']:.0f}s ref={rec['logprob_s']:.0f}s "
                f"train={rec['train_s']:.0f}s {rec['tok_per_s']:.1f}tok/s "
                f"capped={rec['n_capped']}/{rec['n_seq']}]")
        if args.dry_run:
            log(f"    各组奖励: {rec['group_rewards']}  "
                f"(全同组 {rec['uni_frac']*100:.0f}% → 这些组 advantage 恒为 0)")
        if not args.dry_run and step % args.save_every == 0:
            save_policy(model, args.out, tok)
            log(f"  ↳ 已保存 policy LoRA → {args.out}")

    # ---------- 保存 ----------
    if not args.dry_run:
        save_policy(model, args.out, tok)
        log(f"  ↳ 已保存 policy LoRA + tokenizer → {args.out}")

    # ---------- 汇总 ----------
    peak_alloc = vram_peak()
    peak_res = vram_reserved_peak()
    total = vram_total()
    summary = {
        "adapter_start": adapter_path or "BASE (no adapter found)",
        "steps": len(history),
        "vram_base_gb": round(base_used, 3),
        "vram_after_load_gb": round(vram_after_load, 3),
        "vram_after_load_peak_gb": round(vram_after_load_peak, 3),
        "vram_peak_alloc_gb": round(peak_alloc, 3),
        "vram_peak_reserved_gb": round(peak_res, 3),
        "gpu_total_gb": round(total, 2),
        "elapsed_s": round(time.time() - t_start, 1),
        "config": vars(args),
        "history": history,
    }
    log("-" * 78)
    log("显存报告:")
    log(f"  加载前基线 allocated      : {base_used:.3f} GB (外部占用，非本进程)")
    log(f"  加载后 allocated          : {vram_after_load:.3f} GB")
    log(f"  训练峰值 allocated        : {peak_alloc:.3f} GB")
    log(f"  训练峰值 reserved         : {peak_res:.3f} GB")
    log(f"  GPU 总显存                : {total:.2f} GB  → 余量 {total - peak_res:.2f} GB")
    if history:
        h = history[-1]
        log(f"  最后一步: loss={h['loss']:+.4f} reward={h['reward']:.3f} acc={h['acc']:.3f} "
            f"kl={h['kl']:.4f} peak={h['peak_gb']:.2f}GB")
    log(f"总耗时 {time.time()-t_start:.1f}s")

    with open("logs/grpo_result.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("结果写入 logs/grpo_result.json")


if __name__ == "__main__":
    main()
