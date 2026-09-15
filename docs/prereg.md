# 预注册（PREREG）— S-GRPO on multistep

> 状态：冻结（写于训练与终评之前）｜本文件是验收合同；任何执行偏离如实记入最终报告。
> 评测口径、成功线与数据划分在训练开始前确定，训练后不得更改。

## 1. 目标与任务选择依据
- 目标：在可自动判定的多步推理任务上，检验 **S-GRPO**（token 子采样 GRPO 变体）相对 SFT 起点能否带来统计显著的提升。
- 任务选择依据：RL 的有效信号取决于题目难度——组内 8 个答案全对或全错时优势恒为 0，该步不产生梯度。
  因此任务按"SFT 起点准确率落在 20–60%"构造，并在训练前用 val 前 24 题探针确认（§4.1）。

## 2. 任务与数据（冻结）
- 任务族：**multistep**——程序化生成的 5–6 步文字应用题（四类模板轮换），答案整数、程序化 verifier 精确匹配。
- 划分（互斥，题面 sha256 交集=0，由 `tasks.py` 校验）：
  - train = 2000 题（seed 3001）
  - val = 240 题（seed 3002，仅用于 checkpoint 选择）
  - **test = 2000 题（seed 3003，终评一次性使用，训练/选择期间不得读取或推理）**
- 数据生成器与种子即本文件冻结内容；生成后的 sha256 记入 `results/manifest.json`。

## 3. 模型（冻结）
- 基座：`models/Qwen3-1.7B-Base`（4bit NF4 + bf16 compute，double quant）。
- 起点/参考策略：`outputs/sft_lora`（GSM8K SFT adapter，冻结 ref）。
- 策略：在同一 4bit 基座上新建 policy LoRA（初始化=SFT 权重副本），只训 policy。

## 4. 方法（冻结）：S-GRPO
- 每题采样 G=8；组内标准化 advantage Â=(r−μ)/σ（全同则 Â=0）。
- token 入选规则：completion 内下标 t<α=50 恒入选；t≥α 且当前入选数<k=100 时以概率 P=0.5 入选；其余不入选。
- μ=1（每题仅一次更新）、无 PPO clip、ref 不更新。
- 损失：`-(1/G) Σ_i (1/|T_i|) Σ_{t∈T_i} [ exp(lp−lp.detach())·Â_i − β·(k3 KL(π_θ‖π_ref)) ]`。
- LoRA：`q_proj, v_proj`，仅最后 1/3 层；r=16，α=32，dropout 0。
- 优化器：AdamW，lr=1e-4，weight_decay=0.01（λ），grad clip=1.0，β_KL=0.01。
- 采样：max_new=256，top_p=0.95，**temperature 由下节的冻结规则在训练前确定**。

### 4.1 温度选择规则（冻结，train-only）
- 在 val 的前 24 题上，用 SFT 策略分别以 temp∈{0.3, 1.0} 各采 G=8，计算组统计。
- 规则：取 mixed 比例更高者；平局取 all_wrong 更低者；再平局取 adv_scale 更高者；仍平局取 1.0。
- 结果写入 `results/TEMP_PROBE.json` 与 `results/TEMP_FROZEN.txt`；该值即训练超参，不再更改。

## 5. 训练协议（冻结）
- 450 update steps；每题一次更新；train 题按文件顺序取 `item[step % 2000]`。
- 随机性：`torch.manual_seed(20260914 + step)`（采样），mask Bernoulli 用 `Random(20260914*1000+step)`。
- 每 50 步保存 checkpoint。
- 训练日志逐 step 记 `results/metrics.jsonl`（reward/mixed/loss/KL/included 比例/显存/耗时）。

## 6. 模型选择（冻结，train-only）
- 训练结束后，对 **全部 10 个 checkpoint + SFT ref** 在 val（240 题）上做贪心评测（batch=8）。
- 选择规则：val acc 最高者；平局取 **最早** checkpoint。结果写入 `results/SELECT.json`。

## 7. 终评协议（冻结，test 一次）
- 仅当训练与选择完成后，对 test n=2000 以 **贪心** 生成做**配对**评测：SFT ref 与选中 checkpoint 在同一 session 内按 batch 交替生成（每 batch 先 sft 后 rl），每臂每题 1 次生成。
- 守卫：`results/TEST_TOUCHED` + `results/eval_attempts.json`；已有产物时拒绝重跑。
- 两臂同一 4bit 后端，绝对值为 HF 4bit 贪心口径；配对差有效。

## 8. 成功线（冻结）
- Δ = acc(RL) − acc(SFT)（2000 题配对）；**McNemar 精确双侧 p < 0.05 且 bootstrap（10k, seed 20260914）95% CI 下界 > 0**。
- 报告必须给出：点估计、discordant 构成（RL-only/SFT-only）、CI、MDE≈2.8·√d/n（80% 功效近似）。
- 未过线即如实报告负结果；禁止换 seed、换题、后调参补显著。

## 9. 预算与纪律
- 预算 ≤12h（预计 3–5h）；一次预热探针 + 一次训练 + 一次终评。
- test 在终评前不被读取/推理；val 仅用于温度探针与 checkpoint 选择。
- 实现/执行上的偏离与中断处理在最终报告中如实说明。
- 产物目录：`data/`（任务数据）、`results/`（统计与凭据）、`outputs/`（adapter 与 checkpoint）。
