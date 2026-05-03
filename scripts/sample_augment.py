"""Generate sample augmentations for prompt validation.

Reads N random products from train split, runs EXAONE via vLLM, dumps the raw
output to logs/sample_augment.txt for manual review.
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
PROMPT_FILE = ROOT / "prompts/augment_v1.txt"
LOG_DIR = Path(CONFIG["paths"]["logs_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=CONFIG["project"]["seed"])
    parser.add_argument("--prompt", default=str(PROMPT_FILE))
    parser.add_argument("--out", default=str(LOG_DIR / "sample_augment.txt"))
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    train = json.loads(SPLIT_TRAIN.read_text())
    rng = random.Random(args.seed)
    samples = rng.sample(train, args.n)

    template = Path(args.prompt).read_text()
    prompts = []
    for it in samples:
        p = template.replace("{product_name}", it["product_name"]).replace(
            "{description}", it["text_description"]
        )
        prompts.append(p)

    llm = LLM(
        model=CONFIG["augment"]["model_id"],
        dtype=CONFIG["augment"]["vllm"]["dtype"],
        gpu_memory_utilization=CONFIG["augment"]["vllm"]["gpu_memory_utilization"],
        max_model_len=CONFIG["augment"]["vllm"]["max_model_len"],
        trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=CONFIG["augment"]["temperature"],
        top_p=CONFIG["augment"]["top_p"],
        max_tokens=CONFIG["augment"]["max_new_tokens"],
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sp)

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for it, o in zip(samples, outputs):
            f.write("=" * 80 + "\n")
            f.write(f"PRODUCT_ID: {it.get('product_id','?')}\n")
            f.write(f"BRAND: {it.get('brand_id','?')}\n")
            f.write(f"NAME: {it.get('product_name','?')}\n")
            f.write("DESC: " + it.get("text_description", "")[:300] + "...\n")
            f.write("---\n")
            f.write(o.outputs[0].text.strip() + "\n")
            f.write("\n")
    print(f"wrote {len(outputs)} samples to {out_path}")


if __name__ == "__main__":
    main()
