# Qwen3-1.7B Post-Training on Multi-Step Word Problems

S-GRPO (a token-subsampled GRPO variant) post-training of Qwen3-1.7B-Base on programmatically generated multi-step word problems, on a single 8 GB consumer GPU: pass@1 accuracy 53.1% -> 88.3%.

- Task — 5-6 step word problems from four template families (warehouse, pool, savings, machine); integer answers, programmatic verifier, no human or model judging.
- Method — S-GRPO: 8 samples per question, group-normalized advantage, token subsampling (first 50 tokens always backpropagated, later tokens Bernoulli(0.5) up to a cap) to fit long reasoning chains into 8 GB.
- Protocol — disjoint train/val/test splits (question-text SHA-256 intersection = 0); test evaluated once; success criterion frozen before training; paired McNemar + bootstrap.
- Compute — 450 steps, 233 minutes, 2.98 GB peak memory (4-bit QLoRA, single GPU).

## Results

| Metric | SFT baseline | After S-GRPO |
|---|---|---|
| pass@1 accuracy (n = 2000, evaluated once) | 53.05% | 88.25% |
| Paired statistics | +35.2 pp; McNemar exact p = 2.2e-139; bootstrap 95% CI [+32.7, +37.8] pp | |
| Training cost | 450 steps / 233 min / 2.98 GB peak (single 8 GB GPU) | |

By template — the gain comes from the families the SFT model failed almost completely:

| Template | SFT | S-GRPO |
|---|---|---|
| Pool (fill / drain) | 3.6% | 88.2% |
| Savings (earn / spend) | 27.8% | 90.4% |
| Machine (parts) | 87.8% | 87.8% |
| Warehouse (boxes, easiest) | 93.0% | 86.6% |

## Repository layout

```
src/          tasks.py (task generation) · train_sgrpo.py (training) · evaluate.py (one-shot paired evaluation)
              stats.py (McNemar / bootstrap) · task_probe.py (sampling-temperature probe) · common.py (utilities)
              starting point: prepare_data.py (data) · sft.py (QLoRA SFT) · grpo.py (policy-optimization core)
scripts/      run_all.sh — one-command reproduction chain
docs/         prereg.md (experiment protocol) · report.md (technical report)
results/      stats / eval / select / manifest + per-question evaluation outputs (2000 rows each)
```

## Reproduce

```bash
# requires PyTorch (CUDA) + transformers / peft / bitsandbytes / datasets, single GPU >= 8 GB
python src/prepare_data.py      # GSM8K -> SFT training data
python src/sft.py               # QLoRA SFT (starting point)
bash scripts/run_all.sh         # task generation -> temperature probe -> S-GRPO -> checkpoint selection -> final eval -> statistics
```

Base-model weights and datasets are not included; every number can be traced in `results/` and `docs/report.md`.

## Scope

Results hold within the same generator distribution (same templates, new random seeds); they do not extend to GSM8K or real-data distributions. Accuracy is HF 4-bit greedy (pass@1); no full-token GRPO control on the same task.

## License

[MIT](LICENSE)
