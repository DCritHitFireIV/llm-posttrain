#!/usr/bin/env python
"""GSM8K -> SFT 训练数据（completion 格式，供 base 模型训练）

输出: data/sft_train.jsonl
  {"question": ..., "answer": ..., "text": "Question: ...\nAnswer: ..."}

包含 test 泄漏检查（GSM8K 官方划分本不重叠，此处做 sanity check + 归一化去重）。
"""
import json
import os
import re
import sys

import pyarrow.parquet as pq

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def normalize(q: str) -> str:
    """归一化问句用于重叠检测"""
    q = q.lower()
    q = re.sub(r"[^a-z0-9]+", " ", q)
    return " ".join(q.split())


def main():
    train = pq.read_table(os.path.join(DATA, "gsm8k_train.parquet")).to_pylist()
    test = pq.read_table(os.path.join(DATA, "gsm8k_test.parquet")).to_pylist()

    train_norm = {normalize(x["question"]) for x in train}
    overlap = [x for x in test if normalize(x["question"]) in train_norm]
    print(f"train={len(train)}  test={len(test)}  泄漏重合={len(overlap)}")
    if overlap:
        print("⚠️  发现 train/test 重叠，需剔除后再训练")
        for x in overlap[:3]:
            print("   -", x["question"][:60])

    # 内部去重（问句归一化后）
    seen, dedup = set(), []
    for x in train:
        k = normalize(x["question"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(x)
    print(f"train 内部去重后: {len(dedup)} 条")

    out = os.path.join(DATA, "sft_train.jsonl")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for x in dedup:
            rec = {
                "question": x["question"],
                "answer": x["answer"],
                "text": f"Question: {x['question']}\nAnswer: {x['answer']}",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"写入 {out}")

    # 顺带输出一份 test.jsonl 供评测统一样式
    out_t = os.path.join(DATA, "test.jsonl")
    with open(out_t, "w", encoding="utf-8", newline="\n") as f:
        for x in test:
            f.write(json.dumps({"question": x["question"], "answer": x["answer"]}, ensure_ascii=False) + "\n")
    print(f"写入 {out_t}")


if __name__ == "__main__":
    main()
