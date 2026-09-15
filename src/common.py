#!/usr/bin/env python
"""共用工具：路径常量、JSONL 读写、答案抽取、JSON 原子写、HF 4bit 批量贪心评测。"""
from __future__ import annotations

import json
import os
import re
import sys
import time

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

ROOT = "."
DATA = os.path.join(ROOT, "data")          # 程序化生成的任务数据（不入库，可复现）
STATE = os.path.join(ROOT, "results")      # 统计与凭据（入库）
LOGS = os.path.join(ROOT, "results")       # 逐题结果（入库）
OUT = os.path.join(ROOT, "outputs")        # adapter / checkpoint / 临时产物（不入库）
for _d in (DATA, STATE, LOGS, OUT):
    os.makedirs(_d, exist_ok=True)

PROMPT_T = "Question: {q}\nAnswer:"


# ---------- 基础 IO ----------
def load_jsonl(path: str) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def append_jsonl(path: str, obj) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def marker(name: str, content: str = "") -> None:
    p = os.path.join(STATE, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content if content else time.strftime("%F %T") + "\n")
        f.flush()
        os.fsync(f.fileno())


def heartbeat(payload: dict) -> None:
    write_json_atomic(os.path.join(STATE, "heartbeat.json"),
                      dict(payload, ts=time.strftime("%F %T")))


# ---------- 答案抽取 ----------
def extract_answer(text: str):
    """从生成文本抽取答案：优先 #### 后的数字，否则取最后一个数字。"""
    m = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[-+]?[\d,]+(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return None


# ---------- HF 4bit 批量贪心评测 ----------
def batch_greedy(model, tok, G, prompts, adapter, max_new: int = 256):
    """对 prompts 逐批贪心生成；返回 list[str]（与 prompts 等长）。"""
    import torch
    G.set_adapter(model, adapter)
    outs = []
    for i in range(0, len(prompts), 8):
        chunk = prompts[i:i + 8]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(G.DEV)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new,
                                 pad_token_id=tok.pad_token_id)
        in_len = enc["input_ids"].shape[1]
        for j in range(gen.shape[0]):
            text = tok.decode(gen[j, in_len:], skip_special_tokens=True)
            outs.append(text)
    return outs
