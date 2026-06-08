"""Candidate-preserving token-patch reranking.

This evaluates a high-resolution second stage: LoRA-384 CLS retrieval produces
the candidate set, then query token vectors interact with cached image patch
vectors using ColBERT-style MaxSim. Only the first-stage top-N items are
reordered, so first-stage recall at N is preserved.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

from src.evaluate import encode_text_cls, encode_text_tokens
from src.hybrid_retrieval import dedupe_products, metrics_at_k, normalize_rows, product_id_from_link
from src.maxsim import maxsim_score

SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
TRAIN_PAIRS = Path(CONFIG["paths"]["train_pairs_final"])
EVAL_QUERIES = Path(CONFIG["paths"]["eval_queries"])
VISION_EMB = Path(CONFIG["paths"]["vision_emb"])
VISION_IDX = Path(CONFIG["paths"]["vision_idx"])
VISION_PATCHES = Path(CONFIG["paths"]["vision_patches"])
VISION_PATCHES_IDX = Path(CONFIG["paths"]["vision_patches_idx"])
OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"])

MODEL_ID = CONFIG["backbone"]["model_id"]
SEED = CONFIG["project"]["seed"]


def load_lora_model(lora_path: str, device: str):
    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(model, lora_path).eval().to(device)
    return model, processor.tokenizer


def with_pid(items):
    out = []
    for item in items:
        row = dict(item)
        row["product_id"] = row.get("product_id") or product_id_from_link(row.get("product_link", ""))
        if row["product_id"]:
            out.append(row)
    return out


def gallery_for(products, pid_to_cls, cls_all, pid_to_patch, patches_all):
    pids = [p["product_id"] for p in products if p["product_id"] in pid_to_cls and p["product_id"] in pid_to_patch]
    cls = cls_all[[pid_to_cls[pid] for pid in pids]].astype("float32")
    patches = np.array(patches_all[[pid_to_patch[pid] for pid in pids]], dtype=np.float32)
    return pids, cls, patches


def maxsim_topn(text_vecs, text_mask, patches, base_scores, topn, device, chunk_q=32):
    top = np.argsort(-base_scores, axis=1)[:, :topn]
    out = np.full_like(base_scores, -1e9, dtype=np.float32)
    patches_t = torch.from_numpy(patches).to(device, dtype=torch.float32)
    for start in range(0, text_vecs.shape[0], chunk_q):
        end = min(start + chunk_q, text_vecs.shape[0])
        tv = text_vecs[start:end].to(device, dtype=torch.float32)
        tm = text_mask[start:end].to(device)
        for local_q, global_q in enumerate(range(start, end)):
            cols = top[global_q]
            scores = maxsim_score(tv[local_q : local_q + 1], tm[local_q : local_q + 1], patches_t[cols])
            out[global_q, cols] = scores.squeeze(0).detach().cpu().numpy()
    return out


def rerank_topn(base_scores, rerank_scores, topn, weight):
    out = np.full_like(base_scores, -1e9, dtype=np.float32)
    top = np.argsort(-base_scores, axis=1)[:, :topn]
    mixed = normalize_rows(base_scores) + weight * normalize_rows(rerank_scores)
    for qi, cols in enumerate(top):
        out[qi, cols] = mixed[qi, cols]
    return out


def eval_method(name, scores, gt):
    metrics, ranks = metrics_at_k(scores, gt)
    return {"name": name, "metrics": metrics, "ranks": ranks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default="checkpoints/lora_cls_384")
    parser.add_argument("--tag", default="patch_rerank_lora384")
    parser.add_argument("--val-products", type=int, default=240)
    parser.add_argument("--topn", type=int, default=10)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_products = dedupe_products(json.loads(SPLIT_TRAIN.read_text()))
    test_products = with_pid(json.loads(SPLIT_TEST.read_text()))
    test_pids = {p["product_id"] for p in test_products}

    cls_all = np.load(VISION_EMB)
    cls_idx = json.loads(VISION_IDX.read_text())
    pid_to_cls = {pid: i for i, pid in enumerate(cls_idx)}
    patches_all = np.load(VISION_PATCHES, mmap_mode="r")
    patch_idx = json.loads(VISION_PATCHES_IDX.read_text())
    pid_to_patch = {pid: i for i, pid in enumerate(patch_idx)}

    eligible_train = [
        p for p in train_products
        if p["product_id"] in pid_to_cls and p["product_id"] in pid_to_patch and p["product_id"] not in test_pids
    ]
    rng = random.Random(SEED)
    val_pids = set(rng.sample([p["product_id"] for p in eligible_train], min(args.val_products, len(eligible_train))))
    val_products = [p for p in eligible_train if p["product_id"] in val_pids]
    eval_products = [p for p in test_products if p["product_id"] in pid_to_cls and p["product_id"] in pid_to_patch]

    pairs = [json.loads(line) for line in TRAIN_PAIRS.read_text().splitlines() if line.strip()]
    val_pairs = [p for p in pairs if p.get("product_id") in val_pids]
    eval_queries = [json.loads(line) for line in EVAL_QUERIES.read_text().splitlines() if line.strip()]

    val_texts = [p["query"] for p in val_pairs]
    eval_texts = [q["query"] for q in eval_queries]
    val_pids_gallery, val_cls, val_patches = gallery_for(val_products, pid_to_cls, cls_all, pid_to_patch, patches_all)
    eval_pids_gallery, eval_cls, eval_patches = gallery_for(eval_products, pid_to_cls, cls_all, pid_to_patch, patches_all)
    val_pid_to_idx = {pid: i for i, pid in enumerate(val_pids_gallery)}
    eval_pid_to_idx = {pid: i for i, pid in enumerate(eval_pids_gallery)}
    val_gt = np.array([val_pid_to_idx[p["product_id"]] for p in val_pairs], dtype=np.int64)
    eval_gt = np.array([eval_pid_to_idx[q["product_id"]] for q in eval_queries], dtype=np.int64)

    t0 = time.perf_counter()
    model, tokenizer = load_lora_model(args.lora, device)
    val_cls_q = encode_text_cls(model, tokenizer, val_texts, device)
    eval_cls_q = encode_text_cls(model, tokenizer, eval_texts, device)
    val_base = val_cls_q @ val_cls.T
    eval_base = eval_cls_q @ eval_cls.T
    val_tok, val_mask = encode_text_tokens(model, tokenizer, val_texts, device)
    eval_tok, eval_mask = encode_text_tokens(model, tokenizer, eval_texts, device)
    val_patch_scores = maxsim_topn(val_tok, val_mask, val_patches, val_base, args.topn, device)
    eval_patch_scores = maxsim_topn(eval_tok, eval_mask, eval_patches, eval_base, args.topn, device)

    candidates = [("image_lora384", val_base, eval_base)]
    for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0]:
        candidates.append(
            (
                f"rerank_top{args.topn}_{w:g}patch_maxsim",
                rerank_topn(val_base, val_patch_scores, args.topn, w),
                rerank_topn(eval_base, eval_patch_scores, args.topn, w),
            )
        )

    val_results = [eval_method(name, val_scores, val_gt) for name, val_scores, _ in candidates]
    best_idx = max(
        range(len(val_results)),
        key=lambda i: (
            val_results[i]["metrics"]["MRR"],
            val_results[i]["metrics"]["R@1"],
            val_results[i]["metrics"]["R@5"],
            val_results[i]["metrics"]["R@10"],
        ),
    )
    best_name, _, best_eval_scores = candidates[best_idx]
    eval_results = [eval_method(name, eval_scores, eval_gt) for name, _, eval_scores in candidates]
    selected_eval = eval_method(best_name, best_eval_scores, eval_gt)
    runtime_sec = time.perf_counter() - t0

    summary = {
        "tag": args.tag,
        "method": "LoRA-384 CLS first-stage + candidate-preserving query-token/image-patch MaxSim rerank",
        "tuning_policy": "weight selected only on train-split validation products; locked eval not used for selection",
        "lora": args.lora,
        "topn": args.topn,
        "val_products": len(val_products),
        "val_queries": len(val_pairs),
        "eval_queries": len(eval_queries),
        "eval_gallery_size": len(eval_pids_gallery),
        "selected": {
            "name": best_name,
            "val_metrics": val_results[best_idx]["metrics"],
            "eval_metrics": selected_eval["metrics"],
        },
        "all_val_results": val_results,
        "all_eval_results": eval_results,
        "runtime_sec": round(runtime_sec, 3),
    }
    out_path = OUTPUT_DIR / f"eval_{args.tag}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    order = np.argsort(-best_eval_scores, axis=1)
    details = []
    for qi, q in enumerate(eval_queries):
        details.append(
            {
                **q,
                "rank": selected_eval["ranks"][qi],
                "top10_pids": [eval_pids_gallery[i] for i in order[qi, :10].tolist()],
            }
        )
    (OUTPUT_DIR / f"eval_{args.tag}_detail.json").write_text(json.dumps(details, ensure_ascii=False, indent=2))
    print(json.dumps(summary["selected"], ensure_ascii=False, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
