"""Product-level 80/20 split of top.json.

Outputs split_train.json and split_test.json. Each file contains a list of
the same dicts as top.json, restricted to the assigned products.
"""
import json
import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

SEED = CONFIG["project"]["seed"]
TRAIN_RATIO = CONFIG["split"]["train_ratio"]
TOP_JSON = Path(CONFIG["paths"]["top_json"])
SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
SPLIT_TEST = Path(CONFIG["paths"]["split_test"])


def product_id_from_link(link: str) -> str:
    m = re.search(r"/(?:goods|products)/(\d+)", link)
    return m.group(1) if m else None


def main():
    items = json.loads(TOP_JSON.read_text())
    images_dir = Path(CONFIG["paths"]["images_dir"])
    # attach product_id and filter to products with downloaded images
    for it in items:
        it["product_id"] = product_id_from_link(it["product_link"])
    items = [
        it
        for it in items
        if it["product_id"] is not None
        and (images_dir / f"{it['product_id']}.jpg").exists()
    ]
    print(f"products with images: {len(items)}")

    rng = random.Random(SEED)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    split = int(len(items) * TRAIN_RATIO)
    train_idx = set(indices[:split])

    train_items = [items[i] for i in range(len(items)) if i in train_idx]
    test_items = [items[i] for i in range(len(items)) if i not in train_idx]

    SPLIT_TRAIN.write_text(json.dumps(train_items, ensure_ascii=False, indent=2))
    SPLIT_TEST.write_text(json.dumps(test_items, ensure_ascii=False, indent=2))
    print(f"train: {len(train_items)} / test: {len(test_items)}")


if __name__ == "__main__":
    main()
