#!/usr/bin/env bash
# S-GRPO 多步应用题后训练：一键复现链
#   S0 预检 → S1 任务数据 → S2 采样温度探针 → S3 S-GRPO 训练 + val 选点 → S4 一次性终评 → S5 配对统计
# 幂等：每步在 results/ 下写 <S>.done 标记；训练支持按 metrics.jsonl 续跑。
set -uo pipefail

ROOT=.
PY=/opt/train-venv/bin/python
STATE=$ROOT/results
LOGS=$ROOT/results
cd "$ROOT" || exit 1
mkdir -p "$STATE" "$LOGS"

log() { echo "[$(date '+%F %T')] $*"; }
fail() { echo "[$(date '+%F %T')] FAILED at $1" >> "$STATE/FAILED.log"; echo "$1" > "$STATE/FAILED"; exit 1; }
step() { local name=$1; shift; if [ -f "$STATE/$name.done" ]; then log "SKIP $name"; return 0; fi; log ">>> $name"; "$@" || fail "$name"; touch "$STATE/$name.done"; }

s0() {
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -ge 3500 ] || { echo "GPU free ${free}MiB < 3500"; return 1; }
  df -BG /workspace | tail -1
  for f in models/Qwen3-1.7B-Base outputs/sft_lora/adapter_model.safetensors; do
    [ -e "$f" ] || { echo "missing $f"; return 1; }
  done
  "$PY" - <<'EOF'
import hashlib, json, os, time

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

lock = {"ts": time.strftime("%F %T"),
        "model": "models/Qwen3-1.7B-Base",
        "sft_adapter": "outputs/sft_lora/adapter_model.safetensors",
        "sft_sha256": sha("./outputs/sft_lora/adapter_model.safetensors"),
        "scripts": {}}
for f in ["tasks.py", "task_probe.py", "train_sgrpo.py", "evaluate.py", "stats.py", "common.py"]:
    lock["scripts"][f] = sha(os.path.join("src", f))
json.dump(lock, open(os.path.join("results", "manifest.json"), "w"), ensure_ascii=False, indent=1)
print("results/manifest.json written")
EOF
}

s1() { "$PY" "$ROOT/src/tasks.py"; }

s2() {
  "$PY" "$ROOT/src/task_probe.py"
  "$PY" - <<'EOF'
import json
d = json.load(open("./results/TEMP_PROBE.json"))
open("./results/TEMP_FROZEN.txt", "w").write(str(d["chosen"]))
print("frozen temp =", d["chosen"])
EOF
}

s3() {
  TEMP=$(cat "$STATE/TEMP_FROZEN.txt")
  "$PY" "$ROOT/src/train_sgrpo.py" --steps 450 --temp "$TEMP"
}

s4() { "$PY" "$ROOT/src/evaluate.py"; }
s5() { "$PY" "$ROOT/src/stats.py"; }

step S0 s0
step S1 s1
step S2 s2
step S3 s3
step S4 s4
step S5 s5
log "PIPELINE FINISHED"
