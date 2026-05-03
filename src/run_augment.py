"""Full-scale augmentation generation via Upstage solar-pro3 API.

Reads split_train.json, sends each product to solar-pro3 with the locked prompt,
parses 3 queries per product, writes jsonl: {product_id, query, query_type, raw}.
Resumable — skips products already in output file.
"""
import argparse
import asyncio
import json
import os
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
OUT_PATH = Path(CONFIG["paths"]["train_pairs"])
RAW_PATH = OUT_PATH.with_suffix(".raw.jsonl")
PROMPT_PATH = Path(CONFIG["paths"]["augment_prompt"])

MODEL = CONFIG["augment"]["model_id"]
BASE_URL = CONFIG["augment"]["base_url"]
REASONING_EFFORT = CONFIG["augment"]["reasoning_effort"]
CONCURRENCY = CONFIG["augment"]["concurrency"]
RETRIES = CONFIG["augment"]["retries"]

QUERY_RE = re.compile(r"쿼리\s*([123])\s*[:：]\s*(.+?)(?=\n\s*쿼리\s*[123]\s*[:：]|\Z)", re.DOTALL)


def parse_queries(raw: str):
    out = {}
    for m in QUERY_RE.finditer(raw):
        idx = m.group(1)
        text = m.group(2).strip()
        # strip leading bullets/dashes/quotes
        text = re.sub(r"^[-*•\s\"'<]+", "", text).rstrip("\"'>").strip()
        # collapse whitespace
        text = re.sub(r"\s+", " ", text)
        out[idx] = text
    return out


async def gen_one(client, sem, prompt_text, retries=RETRIES):
    async with sem:
        for attempt in range(retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt_text}],
                    reasoning_effort=REASONING_EFFORT,
                    stream=False,
                )
                return resp.choices[0].message.content, None
            except Exception as e:
                if attempt == retries:
                    return None, f"{type(e).__name__}: {e}"
                await asyncio.sleep(2.0 * (attempt + 1))


async def process(client, sem, item, template):
    prompt = template.replace("{product_name}", item["product_name"]).replace(
        "{description}", item["text_description"]
    )
    raw, err = await gen_one(client, sem, prompt)
    if err is not None:
        return item, None, err
    queries = parse_queries(raw or "")
    return item, queries, raw


def load_done(path):
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["product_id"])
            except Exception:
                continue
    return done


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap number of products (debug)")
    args = parser.parse_args()

    train = json.loads(SPLIT_TRAIN.read_text())
    if args.limit:
        train = train[: args.limit]

    done = load_done(OUT_PATH)
    remaining = [it for it in train if it["product_id"] not in done]
    print(f"total train: {len(train)}, already done: {len(done)}, remaining: {len(remaining)}")
    if not remaining:
        print("nothing to do.")
        return

    template = PROMPT_PATH.read_text()
    client = AsyncOpenAI(api_key=os.environ["UPSTAGE_API_KEY"], base_url=BASE_URL)
    sem = asyncio.Semaphore(CONCURRENCY)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_f = OUT_PATH.open("a", encoding="utf-8")
    raw_f = RAW_PATH.open("a", encoding="utf-8")

    async def worker(item):
        it, queries, raw_or_err = await process(client, sem, item, template)
        return it, queries, raw_or_err

    tasks = [worker(it) for it in remaining]
    n_ok = 0
    n_err = 0
    n_partial = 0
    pbar = tqdm_asyncio(asyncio.as_completed(tasks), total=len(tasks), ncols=80, desc="augment")
    async for fut in pbar:
        it, queries, raw_or_err = await fut
        pid = it["product_id"]
        if queries is None:
            raw_f.write(json.dumps({"product_id": pid, "error": raw_or_err}, ensure_ascii=False) + "\n")
            raw_f.flush()
            n_err += 1
            continue
        raw_f.write(json.dumps({"product_id": pid, "raw": raw_or_err}, ensure_ascii=False) + "\n")
        raw_f.flush()
        if len(queries) < 3:
            n_partial += 1
        for k in ("1", "2", "3"):
            if k in queries and queries[k]:
                rec = {
                    "product_id": pid,
                    "brand_id": it.get("brand_id", ""),
                    "product_name": it.get("product_name", ""),
                    "query": queries[k],
                    "query_type": int(k),
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        n_ok += 1

    out_f.close()
    raw_f.close()
    print(f"ok: {n_ok}, err: {n_err}, partial(<3 queries): {n_partial}")


if __name__ == "__main__":
    asyncio.run(main())
