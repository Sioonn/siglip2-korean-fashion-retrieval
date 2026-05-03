"""Apply approved eval-set swaps:
- Remove 9 noisy products (5 multi-pack + 4 low-detail).
- Add 9 new approved products (3 per difficulty).
- Move 4129040 short_ambiguous → long_specific.

Updates eval_queries.jsonl (preserves any filled queries) and reorganizes
eval_images folders.
"""
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

EVAL_QUERIES = Path(CONFIG["paths"]["eval_queries"])
SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
IMAGES_DIR = Path(CONFIG["paths"]["images_dir"])
EVAL_IMAGES_ROOT = Path("/root/code/jolnon/data/eval_images")

REMOVE_IDS = {
    "3298988", "2567367",  # short multi-pack
    "2711584", "4437566", "2396645",  # medium multi-pack
    "1523634", "2052787", "3025663", "2866656",  # long low-detail
}

ADD = {
    "short_ambiguous": ["1383822", "2775711", "4036277"],
    "medium": ["2348247", "3172527", "4227436"],
    "long_specific": ["2062553", "3387448", "4318523"],
}

MOVE = {"4129040": "long_specific"}  # was short_ambiguous


def main():
    test = json.loads(SPLIT_TEST.read_text())
    test_by_pid = {it["product_id"]: it for it in test}

    cur = [json.loads(l) for l in EVAL_QUERIES.open()]
    print(f"current eval queries: {len(cur)}")

    new = []
    for q in cur:
        pid = q["product_id"]
        if pid in REMOVE_IDS:
            continue
        if pid in MOVE:
            q["difficulty"] = MOVE[pid]
        new.append(q)
    print(f"after removals + moves: {len(new)}")

    for diff, pids in ADD.items():
        for pid in pids:
            it = test_by_pid[pid]
            new.append({
                "product_id": pid,
                "brand_id": it.get("brand_id", ""),
                "product_name": it.get("product_name", ""),
                "img_url": it.get("img_url", ""),
                "description_snippet": (it.get("text_description", "") or "")[:200],
                "difficulty": diff,
                "query": "",
            })
    print(f"after additions: {len(new)}")

    counts = {}
    for q in new:
        counts[q["difficulty"]] = counts.get(q["difficulty"], 0) + 1
    print(f"difficulty distribution: {counts}")
    assert counts == {"short_ambiguous": 10, "medium": 10, "long_specific": 10}, "balance broken!"

    EVAL_QUERIES.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in new) + "\n")
    print(f"wrote {EVAL_QUERIES}")

    # reorganize images: rebuild eval_images/{difficulty}/ folders from scratch
    if EVAL_IMAGES_ROOT.exists():
        shutil.rmtree(EVAL_IMAGES_ROOT)
    EVAL_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
    for q in new:
        sub = EVAL_IMAGES_ROOT / q["difficulty"]
        sub.mkdir(parents=True, exist_ok=True)
        src = IMAGES_DIR / f"{q['product_id']}.jpg"
        dst = sub / f"{q['product_id']}.jpg"
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"WARN: missing image for {q['product_id']}")
    print(f"rebuilt {EVAL_IMAGES_ROOT}/")


if __name__ == "__main__":
    main()
