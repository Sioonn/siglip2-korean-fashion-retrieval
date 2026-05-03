"""Dedupe train_pairs.jsonl: keep at most 3 queries per product (one per query_type).

8 products had duplicate parse hits (6 pairs instead of 3). Keep first occurrence
per (product_id, query_type) and write train_pairs_final.jsonl.
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
SRC = Path(CONFIG["paths"]["train_pairs"])
DST = Path(CONFIG["paths"]["train_pairs_final"])

seen = set()
kept = []
with SRC.open() as f:
    for line in f:
        rec = json.loads(line)
        key = (rec["product_id"], rec["query_type"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(rec)

with DST.open("w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"kept {len(kept)} pairs (dedupe complete) → {DST}")
