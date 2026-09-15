#!/usr/bin/env python
"""数据划分：multistep（5-6 步应用题）确定性生成。

split seeds（互斥，且与 pilot seed 20260914 不同）：
  train=3001 (2000 题) / val=3002 (240 题) / test=3003 (2000 题)

产物：data/{train,val,test}.jsonl
验证：gold 全为非负整数；题面 sha256 三集两两交集=0。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re

ROOT = "."
DATA = os.path.join(ROOT, ".", "data")

NAMES = ["Nuwan", "Maya", "Ali", "Ravi", "Chen", "Sara", "Tom", "Lena", "Diego", "Priya"]


def gen_multistep(seed: int, n: int, split: str) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        t = i % 4
        name = NAMES[i % len(NAMES)]
        if t == 0:
            a = rng.randint(800, 2500); b = rng.randint(150, 500); c = rng.randint(100, 400)
            d = rng.randint(50, 300); e = rng.randint(100, 500)
            while a - b < 0:
                b = rng.randint(150, 500)
            q = (f"A warehouse had {a} boxes. A truck took {b} boxes away, then another truck brought {c} boxes. "
                 f"Later {d} boxes were damaged and thrown out, and finally {e} new boxes arrived. "
                 f"How many boxes are in the warehouse now?")
            gold = a - b + c - d + e
        elif t == 1:
            a, b, m, c, m2, d = (rng.randint(3000, 8000), rng.randint(20, 70), rng.randint(4, 10),
                                 rng.randint(30, 90), rng.randint(3, 8), rng.randint(50, 400))
            while a - b * m < 0:
                a = rng.randint(3000, 8000)
            gold = a - b * m + c * m2 - d
            while gold < 0:
                a = rng.randint(3000, 8000)
                gold = a - b * m + c * m2 - d
            q = (f"A pool contains {a} liters of water. A drain removes {b} liters per minute for {m} minutes. "
                 f"Then a hose adds {c} liters per minute for {m2} minutes, and finally {d} liters evaporate. "
                 f"How much water is in the pool now?")
        elif t == 2:
            a, w, b, c, d, e = (rng.randint(60, 140), rng.randint(4, 12), rng.randint(40, 160),
                                rng.randint(30, 120), rng.randint(20, 100), rng.randint(30, 90))
            while a * w - b - c + d - e < 0:
                b = rng.randint(40, 160)
            q = (f"{name} saves {a} dollars per week for {w} weeks. They spend {b} dollars on a gift, "
                 f"{c} dollars on books, then earn {d} dollars walking dogs, and finally spend {e} dollars on food. "
                 f"How much money do they have now?")
            gold = a * w - b - c + d - e
        else:
            a, h, b, k, extra = (rng.randint(60, 200), rng.randint(6, 12), rng.randint(50, 300),
                                 rng.randint(8, 20), rng.randint(5, 40))
            while a * h - b < 0:
                b = rng.randint(50, 300)
            q = (f"A machine produces {a} parts per hour and runs for {h} hours. {b} parts are defective and "
                 f"thrown away. A customer then returns {extra} parts, which are added back before packing. "
                 f"The remaining parts are packed into boxes of {k} parts each. "
                 f"How many full boxes are filled?")
            gold = (a * h - b + extra) // k
        assert gold >= 0, (split, i, gold)
        items.append({"id": f"m2-{split}-{i:04d}", "split": split, "question": q, "gold": str(gold),
                      "family": "multistep"})
    return items


def h(s: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", s).strip().encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DATA)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    splits = {"train": (3001, 2000), "val": (3002, 240), "test": (3003, 2000)}
    all_h, meta = {}, {}
    for name, (seed, n) in splits.items():
        items = gen_multistep(seed, n, name)
        hs = {h(it["question"]) for it in items}
        assert len(hs) == n, f"{name} 内部重复：{n - len(hs)}"
        all_h[name] = hs
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        meta[name] = {"n": n, "seed": seed, "path": path, "sha256": hashlib.sha256(
            open(path, "rb").read()).hexdigest(), "q_sha256": hashlib.sha256(
            "\n".join(sorted(hs)).encode()).hexdigest()}
        print(f"[{name}] n={n} -> {path}")
    for a in splits:
        for b in splits:
            if a < b:
                inter = all_h[a] & all_h[b]
                assert not inter, f"{a}/{b} 题面重叠 {len(inter)}"
                print(f"[隔离] {a} ∩ {b} = 0")
    meta["_generated"] = True
    with open(os.path.join(args.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("ROOT TASKS DONE")


if __name__ == "__main__":
    main()
