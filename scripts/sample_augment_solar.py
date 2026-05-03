"""Generate sample augmentations using Upstage solar-pro3 API.

Concurrent calls via asyncio. Validates the augmentation prompt.
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import yaml

CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
PROMPT_FILE = ROOT / "prompts/augment_v3.txt"
LOG_DIR = Path(CONFIG["paths"]["logs_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

CONCURRENCY = 8
REASONING_EFFORT = "high"
MODEL = "solar-pro3"


async def gen_one(client, sem, prompt_text, retries=2):
    async with sem:
        for attempt in range(retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt_text}],
                    reasoning_effort=REASONING_EFFORT,
                    stream=False,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == retries:
                    return f"[ERROR] {type(e).__name__}: {e}"
                await asyncio.sleep(2.0 * (attempt + 1))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=CONFIG["project"]["seed"])
    parser.add_argument("--prompt", default=str(PROMPT_FILE))
    parser.add_argument("--out", default=str(LOG_DIR / "sample_augment_solar.txt"))
    args = parser.parse_args()

    train = json.loads(SPLIT_TRAIN.read_text())
    rng = random.Random(args.seed)
    samples = rng.sample(train, args.n)

    template = Path(args.prompt).read_text()
    prompts = [
        template.replace("{product_name}", it["product_name"]).replace(
            "{description}", it["text_description"]
        )
        for it in samples
    ]

    client = AsyncOpenAI(
        api_key=os.environ["UPSTAGE_API_KEY"],
        base_url="https://api.upstage.ai/v1",
    )
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [gen_one(client, sem, p) for p in prompts]
    outputs = await tqdm_asyncio.gather(*tasks, desc=f"solar-pro3 (conc={CONCURRENCY})", ncols=80)

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for it, o in zip(samples, outputs):
            f.write("=" * 80 + "\n")
            f.write(f"PRODUCT_ID: {it.get('product_id','?')}\n")
            f.write(f"BRAND: {it.get('brand_id','?')}\n")
            f.write(f"NAME: {it.get('product_name','?')}\n")
            f.write("DESC: " + it.get("text_description", "")[:300] + "...\n")
            f.write("---\n")
            f.write(o.strip() + "\n")
            f.write("\n")
    print(f"wrote {len(outputs)} samples to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
