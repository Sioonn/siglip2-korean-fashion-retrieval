"""Download Musinsa product images in parallel.

Reads img_url from data/top.json, saves images named by product_id (extracted
from product_link) into data/images/. Failures are logged to logs/download_failures.json.
"""
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import aiohttp
import yaml
from tqdm.asyncio import tqdm_asyncio

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

TOP_JSON = Path(CONFIG["paths"]["top_json"])
IMAGES_DIR = Path(CONFIG["paths"]["images_dir"])
LOG_DIR = Path(CONFIG["paths"]["logs_dir"])
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CONCURRENCY = 100
RETRIES = 3
TIMEOUT = aiohttp.ClientTimeout(total=20, connect=10)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}


def product_id_from_link(link: str) -> str:
    m = re.search(r"/(?:goods|products)/(\d+)", link)
    return m.group(1) if m else None


async def download_one(session, sem, item, progress):
    pid = product_id_from_link(item["product_link"])
    if pid is None:
        progress.update(1)
        return {"product_link": item["product_link"], "error": "no product id"}
    out_path = IMAGES_DIR / f"{pid}.jpg"
    if out_path.exists() and out_path.stat().st_size > 0:
        progress.update(1)
        return None

    url = item["img_url"]
    last_err = None
    async with sem:
        for attempt in range(RETRIES):
            try:
                async with session.get(url, headers=HEADERS) as r:
                    if r.status != 200:
                        last_err = f"HTTP {r.status}"
                        continue
                    data = await r.read()
                    if len(data) < 1024:
                        last_err = f"too small: {len(data)} bytes"
                        continue
                    out_path.write_bytes(data)
                    progress.update(1)
                    return None
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(1.0 + attempt)
    progress.update(1)
    return {"product_link": item["product_link"], "url": url, "error": last_err}


async def main():
    items = json.loads(TOP_JSON.read_text())
    print(f"total products: {len(items)}")

    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    progress = tqdm_asyncio(total=len(items), desc="download", ncols=80)
    failures = []
    async with aiohttp.ClientSession(connector=connector, timeout=TIMEOUT) as session:
        tasks = [download_one(session, sem, it, progress) for it in items]
        results = await asyncio.gather(*tasks)
    progress.close()

    failures = [r for r in results if r]
    print(f"failures: {len(failures)} / {len(items)}")
    (LOG_DIR / "download_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2)
    )

    succeeded = sum(1 for f in IMAGES_DIR.glob("*.jpg") if f.stat().st_size > 0)
    print(f"images on disk: {succeeded}")


if __name__ == "__main__":
    asyncio.run(main())
