"""Validation-selected ensemble of LoRA+BM25 rerank and dual-tower LoRA."""

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

from src.hybrid_retrieval import BM25, dedupe_products, encode_text_cls, load_lora_model, make_docs, metrics_at_k, normalize_rows, product_id_from_link

SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
TRAIN_PAIRS = Path(CONFIG["paths"]["train_pairs_final"])
EVAL_QUERIES = Path(CONFIG["paths"]["eval_queries"])
VISION_EMB = Path(CONFIG["paths"]["vision_emb"])
VISION_IDX = Path(CONFIG["paths"]["vision_idx"])
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


def encode_dual_text(model, tokenizer, texts, device, batch_size=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        toks = tokenizer(texts[i : i + batch_size], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.get_text_features(input_ids=toks["input_ids"])
        out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        embs.append(out.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def encode_dual_images(model, processor, products, device, batch_size=48):
    embs = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(products), batch_size), ncols=80, desc="dual images"):
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


def rrf(scores_a, scores_b, k):
    ranks_a = np.argsort(np.argsort(-scores_a, axis=1), axis=1) + 1
    ranks_b = np.argsort(np.argsort(-scores_b, axis=1), axis=1) + 1
    return 1.0 / (k + ranks_a) + 1.0 / (k + ranks_b)


def gated_scores(texts, scores_short, scores_long, policy):
    out = scores_short.copy()
    for i, text in enumerate(texts):
        compact_len = len(text.replace(" ", ""))
        if compact_len <= 16:
            bucket = "short"
        elif compact_len <= 34:
            bucket = "medium"
        else:
            bucket = "long"
        if policy[bucket] == "dual":
            out[i] = scores_long[i]
    return out


def eval_method(name, scores, gt):
    metrics, ranks = metrics_at_k(scores, gt)
    return {"name": name, "metrics": metrics, "ranks": ranks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-lora", required=True)
    parser.add_argument("--text-lora", default="checkpoints/lora_cls_384")
    parser.add_argument("--tag", default="ensemble_dual_lora")
    parser.add_argument("--val-products", type=int, default=240)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_products = dedupe_products(json.loads(SPLIT_TRAIN.read_text()))
    test_products = with_pid(json.loads(SPLIT_TEST.read_text()))
    test_pids = {p["product_id"] for p in test_products}
    vision_emb_all = np.load(VISION_EMB)
    vision_idx = json.loads(VISION_IDX.read_text())
    pid_to_row = {pid: i for i, pid in enumerate(vision_idx)}
    eligible_train = [p for p in train_products if p["product_id"] in pid_to_row and p["product_id"] not in test_pids and (IMAGES_DIR / f"{p['product_id']}.jpg").exists()]
    rng = random.Random(SEED)
    val_pids = set(rng.sample([p["product_id"] for p in eligible_train], min(args.val_products, len(eligible_train))))
    val_products = [p for p in eligible_train if p["product_id"] in val_pids]
    eval_products = [p for p in test_products if p["product_id"] in pid_to_row and (IMAGES_DIR / f"{p['product_id']}.jpg").exists()]

    pairs = [json.loads(line) for line in TRAIN_PAIRS.read_text().splitlines() if line.strip()]
    val_pairs = [p for p in pairs if p.get("product_id") in val_pids]
    eval_queries = [json.loads(line) for line in EVAL_QUERIES.read_text().splitlines() if line.strip()]
    val_texts = [p["query"] for p in val_pairs]
    eval_texts = [q["query"] for q in eval_queries]

    val_pids_gallery = [p["product_id"] for p in val_products]
    eval_pids_gallery = [p["product_id"] for p in eval_products]
    val_gt = np.array([{pid: i for i, pid in enumerate(val_pids_gallery)}[p["product_id"]] for p in val_pairs], dtype=np.int64)
    eval_gt = np.array([{pid: i for i, pid in enumerate(eval_pids_gallery)}[q["product_id"]] for q in eval_queries], dtype=np.int64)

    text_model, text_tokenizer = load_lora_model(args.text_lora, device)
    val_text_q = encode_text_cls(text_model, text_tokenizer, val_texts, device)
    eval_text_q = encode_text_cls(text_model, text_tokenizer, eval_texts, device)
    val_frozen_img = vision_emb_all[[pid_to_row[pid] for pid in val_pids_gallery]]
    eval_frozen_img = vision_emb_all[[pid_to_row[pid] for pid in eval_pids_gallery]]
    val_lora = val_text_q @ val_frozen_img.T
    eval_lora = eval_text_q @ eval_frozen_img.T
    val_bm25 = BM25(make_docs(val_products, "all")).score_many(val_texts)
    eval_bm25 = BM25(make_docs(eval_products, "all")).score_many(eval_texts)
    val_lora_bm25 = rerank_topn(val_lora, val_bm25, 10, 0.15)
    eval_lora_bm25 = rerank_topn(eval_lora, eval_bm25, 10, 0.15)
    del text_model
    torch.cuda.empty_cache() if device == "cuda" else None

    dual_model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    dual_model = PeftModel.from_pretrained(dual_model, args.dual_lora).eval().to(device)
    val_dual_img = encode_dual_images(dual_model, processor, val_products, device)
    eval_dual_img = encode_dual_images(dual_model, processor, eval_products, device)
    val_dual_q = encode_dual_text(dual_model, processor.tokenizer, val_texts, device)
    eval_dual_q = encode_dual_text(dual_model, processor.tokenizer, eval_texts, device)
    val_dual = val_dual_q @ val_dual_img.T
    eval_dual = eval_dual_q @ eval_dual_img.T

    candidates = [("lora384_top10_015bm25", val_lora_bm25, eval_lora_bm25), ("dual_infonce", val_dual, eval_dual)]
    for w in [0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0]:
        candidates.append((f"sum_lora_bm25_plus_{w:g}dual", normalize_rows(val_lora_bm25) + w * normalize_rows(val_dual), normalize_rows(eval_lora_bm25) + w * normalize_rows(eval_dual)))
    for k in [5, 10, 20, 60]:
        candidates.append((f"rrf_lora_bm25_dual_k{k}", rrf(val_lora_bm25, val_dual, k), rrf(eval_lora_bm25, eval_dual, k)))
    for short in ["lora", "dual"]:
        for medium in ["lora", "dual"]:
            for long in ["lora", "dual"]:
                policy = {"short": short, "medium": medium, "long": long}
                name = f"gate_len_s{short}_m{medium}_l{long}"
                candidates.append(
                    (
                        name,
                        gated_scores(val_texts, val_lora_bm25, val_dual, policy),
                        gated_scores(eval_texts, eval_lora_bm25, eval_dual, policy),
                    )
                )

    val_results = [eval_method(name, val_scores, val_gt) for name, val_scores, _ in candidates]
    best_idx = max(range(len(val_results)), key=lambda i: (val_results[i]["metrics"]["MRR"], val_results[i]["metrics"]["R@1"], val_results[i]["metrics"]["R@5"], val_results[i]["metrics"]["R@10"]))
    best_name, _, best_eval_scores = candidates[best_idx]
    eval_results = [eval_method(name, eval_scores, eval_gt) for name, _, eval_scores in candidates]
    selected_eval = eval_method(best_name, best_eval_scores, eval_gt)
    summary = {
        "tag": args.tag,
        "method": "validation-selected ensemble of LoRA-384+BM25 rerank and dual-tower InfoNCE LoRA",
        "tuning_policy": "ensemble candidate selected only on train-split validation products; locked eval not used for selection",
        "selected": {"name": best_name, "val_metrics": val_results[best_idx]["metrics"], "eval_metrics": selected_eval["metrics"]},
        "all_val_results": val_results,
        "all_eval_results": eval_results,
    }
    out_path = OUTPUT_DIR / f"eval_{args.tag}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    order = np.argsort(-best_eval_scores, axis=1)
    detail = [
        {**q, "rank": selected_eval["ranks"][qi], "top10_pids": [eval_pids_gallery[i] for i in order[qi, :10].tolist()]}
        for qi, q in enumerate(eval_queries)
    ]
    (OUTPUT_DIR / f"eval_{args.tag}_detail.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2))
    print(json.dumps(summary["selected"], ensure_ascii=False, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
