"""Build the eval query template — 30 random test products, blank query field.

User opens output and fills in 'query' and 'difficulty' for each row.
- difficulty must be one of: short_ambiguous / medium / long_specific
- aim for 10 of each difficulty (will be enforced at eval time)
"""
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
EVAL_OUT = Path(CONFIG["paths"]["eval_queries"])
SEED = CONFIG["project"]["seed"]
N = CONFIG["eval"]["num_queries"]

DIFFICULTIES = ["short_ambiguous", "medium", "long_specific"]


def main():
    test = json.loads(SPLIT_TEST.read_text())
    rng = random.Random(SEED)
    sample = rng.sample(test, N)

    # assign target difficulty in round-robin so user has to write 10 of each
    targets = []
    for i in range(N):
        targets.append(DIFFICULTIES[i % 3])
    rng.shuffle(targets)

    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_OUT.open("w", encoding="utf-8") as f:
        for it, diff in zip(sample, targets):
            rec = {
                "product_id": it["product_id"],
                "brand_id": it.get("brand_id", ""),
                "product_name": it.get("product_name", ""),
                "img_url": it.get("img_url", ""),
                "description_snippet": (it.get("text_description", "") or "")[:200],
                "difficulty": diff,  # target difficulty — user should write a query of this style
                "query": "",  # FILL IN
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote template with {N} blank queries to {EVAL_OUT}")
    print(f"difficulty distribution: short_ambiguous={targets.count('short_ambiguous')}, medium={targets.count('medium')}, long_specific={targets.count('long_specific')}")
    print("\nGuide:")
    print("  short_ambiguous (15~30자): '흰 셔츠', '검정 후드 흰 글씨'")
    print("  medium (30~60자): '그레이 후드에 가슴쪽 영어 로고 적힘'")
    print("  long_specific (80~150자): 두 문장 또는 디테일 포함")


if __name__ == "__main__":
    main()
