"""Evaluate a dual-tower LoRA checkpoint by encoding test images live."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

from src.hybrid_retrieval import metrics_at_k, product_id_from_link

SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
EVAL_QUERIES = Path(CONFIG["paths"]["eval_queries"])
IMAGES_DIR = Path(CONFIG["paths"]["images_dir"])
OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"])
MODEL_ID = CONFIG["backbone"]["model_id"]
MAX_LEN = CONFIG["backbone"]["text_max_length"]


def with_pid(items):
    out = []
    for item in items:
        row = dict(item)
        row["product_id"] = row.get("product_id") or product_id_from_link(row.get("product_link", ""))
        if row["product_id"]:
            out.append(row)
    return out


def encode_text(model, tokenizer, texts, device, batch_size=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        toks = tokenizer(texts[i : i + batch_size], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.get_text_features(input_ids=toks["input_ids"])
        out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        embs.append(out.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def encode_images(model, processor, products, device, batch_size=48):
    embs = []
    pids = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(products), batch_size), ncols=80, desc="images"):
            chunk = products[i : i + batch_size]
            imgs = [Image.open(IMAGES_DIR / f"{p['product_id']}.jpg").convert("RGB") for p in chunk]
            inputs = processor(images=imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
            out = model.get_image_features(pixel_values=pixel_values)
            out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            embs.append(out.float().cpu().numpy())
            pids.extend([p["product_id"] for p in chunk])
    return pids, np.concatenate(embs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=True)
    parser.add_argument("--tag", default="dual_lora")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(model, args.lora).eval().to(device)

    products = [p for p in with_pid(json.loads(SPLIT_TEST.read_text())) if (IMAGES_DIR / f"{p['product_id']}.jpg").exists()]
    queries = [json.loads(line) for line in EVAL_QUERIES.read_text().splitlines() if line.strip()]
    t0 = time.perf_counter()
    gallery_pids, gallery = encode_images(model, processor, products, device)
    text = encode_text(model, processor.tokenizer, [q["query"] for q in queries], device)
    scores = text @ gallery.T
    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(len(queries), 1)
    pid_to_idx = {pid: i for i, pid in enumerate(gallery_pids)}
    gt = np.array([pid_to_idx[q["product_id"]] for q in queries], dtype=np.int64)
    metrics, ranks = metrics_at_k(scores, gt)
    summary = {
        "tag": args.tag,
        "method": "dual-tower text+vision LoRA evaluated with live test image encoding",
        "lora": args.lora,
        "eval_queries": len(queries),
        "eval_gallery_size": len(gallery_pids),
        "latency_ms_per_query_including_image_encoding": round(latency_ms, 4),
        "metrics": metrics,
    }
    out_path = OUTPUT_DIR / f"eval_{args.tag}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    order = np.argsort(-scores, axis=1)
    detail = [
        {**q, "rank": ranks[qi], "top10_pids": [gallery_pids[i] for i in order[qi, :10].tolist()]}
        for qi, q in enumerate(queries)
    ]
    (OUTPUT_DIR / f"eval_{args.tag}_detail.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
