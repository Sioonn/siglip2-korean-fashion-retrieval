"""Dual-tower LoRA first stage plus candidate-preserving BM25 rerank."""

import argparse
import json
import os
import random
import sys
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

from src.hybrid_retrieval import BM25, dedupe_products, make_docs, metrics_at_k, normalize_rows, product_id_from_link

SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
TRAIN_PAIRS = Path(CONFIG["paths"]["train_pairs_final"])
EVAL_QUERIES = Path(CONFIG["paths"]["eval_queries"])
IMAGES_DIR = Path(CONFIG["paths"]["images_dir"])
OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"])
MODEL_ID = CONFIG["backbone"]["model_id"]
MAX_LEN = CONFIG["backbone"]["text_max_length"]
SEED = CONFIG["project"]["seed"]


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
    with torch.inference_mode():
        for i in tqdm(range(0, len(products), batch_size), ncols=80, desc="images"):
            chunk = products[i : i + batch_size]
            imgs = [Image.open(IMAGES_DIR / f"{p['product_id']}.jpg").convert("RGB") for p in chunk]
            pixel_values = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device, dtype=torch.bfloat16)
            out = model.get_image_features(pixel_values=pixel_values)
            out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            embs.append(out.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def rerank_topn(base_scores, rerank_scores, n, weight):
    out = np.full_like(base_scores, -1e9, dtype=np.float32)
    top = np.argsort(-base_scores, axis=1)[:, :n]
    mixed = normalize_rows(base_scores) + weight * normalize_rows(rerank_scores)
    for qi, cols in enumerate(top):
        out[qi, cols] = mixed[qi, cols]
    return out


def eval_method(name, scores, gt):
    metrics, ranks = metrics_at_k(scores, gt)
    return {"name": name, "metrics": metrics, "ranks": ranks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=True)
    parser.add_argument("--tag", default="dual_hybrid_rerank")
    parser.add_argument("--val-products", type=int, default=240)
    parser.add_argument("--doc-mode", default="all", choices=["name", "caption", "all"])
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(model, args.lora).eval().to(device)

    train_products = dedupe_products(json.loads(SPLIT_TRAIN.read_text()))
    test_products = with_pid(json.loads(SPLIT_TEST.read_text()))
    test_pids = {p["product_id"] for p in test_products}
    eligible_train = [p for p in train_products if p["product_id"] not in test_pids and (IMAGES_DIR / f"{p['product_id']}.jpg").exists()]
    rng = random.Random(SEED)
    val_pids = set(rng.sample([p["product_id"] for p in eligible_train], min(args.val_products, len(eligible_train))))
    val_products = [p for p in eligible_train if p["product_id"] in val_pids]
    eval_products = [p for p in test_products if (IMAGES_DIR / f"{p['product_id']}.jpg").exists()]

    pairs = [json.loads(line) for line in TRAIN_PAIRS.read_text().splitlines() if line.strip()]
    val_pairs = [p for p in pairs if p.get("product_id") in val_pids]
    eval_queries = [json.loads(line) for line in EVAL_QUERIES.read_text().splitlines() if line.strip()]
    val_texts = [p["query"] for p in val_pairs]
    eval_texts = [q["query"] for q in eval_queries]

    val_img = encode_images(model, processor, val_products, device)
    eval_img = encode_images(model, processor, eval_products, device)
    val_q = encode_text(model, processor.tokenizer, val_texts, device)
    eval_q = encode_text(model, processor.tokenizer, eval_texts, device)
    val_base = val_q @ val_img.T
    eval_base = eval_q @ eval_img.T
    val_bm25 = BM25(make_docs(val_products, args.doc_mode)).score_many(val_texts)
    eval_bm25 = BM25(make_docs(eval_products, args.doc_mode)).score_many(eval_texts)

    val_pid_to_idx = {p["product_id"]: i for i, p in enumerate(val_products)}
    eval_pid_to_idx = {p["product_id"]: i for i, p in enumerate(eval_products)}
    val_gt = np.array([val_pid_to_idx[p["product_id"]] for p in val_pairs], dtype=np.int64)
    eval_gt = np.array([eval_pid_to_idx[q["product_id"]] for q in eval_queries], dtype=np.int64)

    candidates = [("dual_image", val_base, eval_base)]
    for n in [10, 20]:
        for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]:
            candidates.append((f"rerank_top{n}_{w:g}bm25", rerank_topn(val_base, val_bm25, n, w), rerank_topn(eval_base, eval_bm25, n, w)))
    val_results = [eval_method(name, val_scores, val_gt) for name, val_scores, _ in candidates]
    best_idx = max(range(len(val_results)), key=lambda i: (val_results[i]["metrics"]["MRR"], val_results[i]["metrics"]["R@1"], val_results[i]["metrics"]["R@5"], val_results[i]["metrics"]["R@10"]))
    best_name, _, best_eval_scores = candidates[best_idx]
    eval_results = [eval_method(name, eval_scores, eval_gt) for name, _, eval_scores in candidates]
    selected_eval = eval_method(best_name, best_eval_scores, eval_gt)
    summary = {
        "tag": args.tag,
        "method": "dual-tower LoRA first stage plus candidate-preserving BM25 rerank",
        "tuning_policy": "weight/topn selected only on train-split validation products; locked eval not used for selection",
        "selected": {"name": best_name, "val_metrics": val_results[best_idx]["metrics"], "eval_metrics": selected_eval["metrics"]},
        "all_val_results": val_results,
        "all_eval_results": eval_results,
    }
    out_path = OUTPUT_DIR / f"eval_{args.tag}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    order = np.argsort(-best_eval_scores, axis=1)
    detail = [
        {**q, "rank": selected_eval["ranks"][qi], "top10_pids": [eval_products[i]["product_id"] for i in order[qi, :10].tolist()]}
        for qi, q in enumerate(eval_queries)
    ]
    (OUTPUT_DIR / f"eval_{args.tag}_detail.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2))
    print(json.dumps(summary["selected"], ensure_ascii=False, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
