"""Find candidate replacements for the 9 noisy eval products.

Filters test_split aggressively to exclude multi-pack and color-set products,
then picks ~25 candidates with varying description richness. User will visually
inspect via copied images and approve final 9.
"""
import json
import random
import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
EVAL_QUERIES = Path(CONFIG["paths"]["eval_queries"])
IMAGES_DIR = Path(CONFIG["paths"]["images_dir"])
OUT_DIR = Path("/root/code/jolnon/data/eval_candidates")
SEED = 4242  # different seed from main eval to get fresh picks

# Aggressive keyword filter for multi-pack / color-set products.
# Matches product_name OR description.
MULTI_KW = re.compile(
    r"\b(2pack|3pack|4pack|2-pack|3-pack|2 pack|3 pack)\b"
    r"|\[?\s*pack\s*\]?"
    r"|\[?\s*set\s*\]?"
    r"|2팩|3팩|4팩|2pcs|3pcs"
    r"|택[0-9]"
    r"|2종|3종|4종|2벌|3벌"
    r"|[2-9]+\s*color(s)?"
    r"|[2-9]+\s*컬러"
    r"|[0-9]+\s*COL\b"
    r"|두\s*가지\s*색|세\s*가지\s*색|네\s*가지\s*색|다섯\s*가지\s*색"
    r"|두\s*가지\s*컬러|세\s*가지\s*컬러"
    r"|두\s*가지로\s*구성|세\s*가지로\s*구성|네\s*가지로\s*구성"
    r"|두\s*벌|세\s*벌"
    r"|밀리\s*컬러|melange|MELANGE"
    r"|각각의\s*색상|각각의\s*컬러"
    r"|복합|컬렉션",
    re.IGNORECASE,
)


def is_multipack(item):
    blob = (item.get("product_name", "") + "\n" + item.get("text_description", "")).lower()
    return bool(MULTI_KW.search(blob))


def feature_score(item):
    """Heuristic: long description with graphic/text mentions → richer features."""
    desc = item.get("text_description", "") or ""
    score = len(desc)  # length component
    keywords = ["그래픽", "프린팅", "프린트", "프린", "글씨", "문구", "로고", "패턴", "그림", "자수", "체크", "스트라이프"]
    score += 100 * sum(1 for k in keywords if k in desc)
    return score


def main():
    test = json.loads(SPLIT_TEST.read_text())
    eval_qs = [json.loads(l) for l in EVAL_QUERIES.open()]
    used_pids = {q["product_id"] for q in eval_qs}

    pool = [it for it in test if it["product_id"] not in used_pids and not is_multipack(it)]
    print(f"test split: {len(test)}, after exclude+multipack filter: {len(pool)}")

    # check that the image actually exists
    pool = [it for it in pool if (IMAGES_DIR / f"{it['product_id']}.jpg").exists()]
    print(f"with image on disk: {len(pool)}")

    # split by feature score tertiles for difficulty matching
    pool_scored = [(feature_score(it), it) for it in pool]
    pool_scored.sort(key=lambda x: x[0])
    n = len(pool_scored)
    low = pool_scored[: n // 3]  # → short_ambiguous candidates
    mid = pool_scored[n // 3 : 2 * n // 3]  # → medium candidates
    high = pool_scored[2 * n // 3 :]  # → long_specific candidates

    rng = random.Random(SEED)
    picks = {
        "short_ambiguous": rng.sample(low, 8),
        "medium": rng.sample(mid, 8),
        "long_specific": rng.sample(high, 8),
    }

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for diff, items in picks.items():
        sub = OUT_DIR / diff
        sub.mkdir(parents=True, exist_ok=True)
        manifest[diff] = []
        for score, it in items:
            src = IMAGES_DIR / f"{it['product_id']}.jpg"
            dst = sub / f"{it['product_id']}.jpg"
            shutil.copy2(src, dst)
            manifest[diff].append({
                "product_id": it["product_id"],
                "brand_id": it.get("brand_id", ""),
                "product_name": it.get("product_name", ""),
                "feature_score": score,
                "img_url": it.get("img_url", ""),
                "description_snippet": (it.get("text_description", "") or "")[:200],
            })
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nwrote {sum(len(v) for v in picks.values())} candidates to {OUT_DIR}/")
    for d, items in picks.items():
        print(f"  {d}: {len(items)} candidates")


if __name__ == "__main__":
    main()
