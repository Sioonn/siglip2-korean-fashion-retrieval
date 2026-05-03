"""Evaluate retrieval — supports CLS pooling or ColBERT-style MaxSim.

Reads eval_queries.jsonl, encodes 30 query texts, computes similarity vs
cached vision representations restricted to test split, reports R@K, MRR
overall and per difficulty band, plus avg per-query latency and storage.
"""
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

from src.maxsim import maxsim_score_chunked  # noqa: E402

EVAL_QUERIES = Path(CONFIG["paths"]["eval_queries"])
SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
VISION_EMB = Path(CONFIG["paths"]["vision_emb"])
VISION_IDX = Path(CONFIG["paths"]["vision_idx"])
VISION_PATCHES = Path(CONFIG["paths"]["vision_patches"])
VISION_PATCHES_IDX = Path(CONFIG["paths"]["vision_patches_idx"])
OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"])

MODEL_ID = CONFIG["backbone"]["model_id"]
MAX_LEN = CONFIG["backbone"]["text_max_length"]


def encode_text_cls(model, tokenizer, queries, device, batch_size=32):
    embs = []
    for i in range(0, len(queries), batch_size):
        chunk = queries[i : i + batch_size]
        toks = tokenizer(chunk, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.get_text_features(input_ids=toks["input_ids"])
        out = out / out.norm(dim=-1, keepdim=True)
        embs.append(out.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def encode_text_tokens(model, tokenizer, queries, device, batch_size=32):
    """Returns L2-normalized per-token vectors (N, T, D) and padding mask (N, T)."""
    all_vecs = []
    all_masks = []
    pad_id = tokenizer.pad_token_id
    # peft wraps as model.base_model.model; raw HF model has text_model directly
    base = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    for i in range(0, len(queries), batch_size):
        chunk = queries[i : i + batch_size]
        toks = tokenizer(chunk, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = base.text_model(input_ids=toks["input_ids"])
        h = out.last_hidden_state  # (B, T, D), post final_layer_norm
        h = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        mask = (toks["input_ids"] != pad_id)
        all_vecs.append(h.float().cpu())
        all_masks.append(mask.cpu())
    return torch.cat(all_vecs, dim=0), torch.cat(all_masks, dim=0)


def metrics_at_k(scores, gt_indices, k_list=(1, 5, 10)):
    ranks = []
    for s, gt in zip(scores, gt_indices):
        order = np.argsort(-s)
        rank = int(np.where(order == gt)[0][0]) + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    recall = {f"R@{k}": float((ranks <= k).mean()) for k in k_list}
    mrr = float((1.0 / ranks).mean())
    return recall, mrr, ranks.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default=None, help="path to LoRA checkpoint dir; omit for baseline")
    parser.add_argument("--tag", default=None, help="result tag (e.g. baseline / ours)")
    parser.add_argument("--pooling", default="cls", choices=["cls", "maxsim"])
    args = parser.parse_args()

    from transformers import AutoModel, AutoProcessor
    from peft import PeftModel

    tag = args.tag or ("ours" if args.lora else "baseline")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_split = json.loads(SPLIT_TEST.read_text())
    test_pids = {it["product_id"] for it in test_split}

    if args.pooling == "cls":
        vision_emb_all = np.load(VISION_EMB)
        vision_idx_all = json.loads(VISION_IDX.read_text())
        gallery_rows = [i for i, pid in enumerate(vision_idx_all) if pid in test_pids]
        gallery_pids = [vision_idx_all[r] for r in gallery_rows]
        gallery_vecs = vision_emb_all[gallery_rows]  # (G, D)
        storage_per_item = gallery_vecs.shape[1]
    else:
        patches_all = np.load(VISION_PATCHES, mmap_mode="r")
        patches_idx_all = json.loads(VISION_PATCHES_IDX.read_text())
        gallery_rows = [i for i, pid in enumerate(patches_idx_all) if pid in test_pids]
        gallery_pids = [patches_idx_all[r] for r in gallery_rows]
        gallery_vecs = np.array(patches_all[gallery_rows])  # (G, P, D)
        storage_per_item = gallery_vecs.shape[1] * gallery_vecs.shape[2]
    print(f"gallery size: {len(gallery_pids)}, pooling: {args.pooling}")

    queries = []
    with EVAL_QUERIES.open() as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"queries: {len(queries)}")
    pid_to_gallery = {pid: i for i, pid in enumerate(gallery_pids)}
    for q in queries:
        if q["product_id"] not in pid_to_gallery:
            raise ValueError(f"eval query references {q['product_id']} not in gallery")

    print("loading model...")
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    if args.lora:
        print(f"loading LoRA from {args.lora}")
        model = PeftModel.from_pretrained(model, args.lora).eval().to(device)

    query_texts = [q["query"] for q in queries]

    if args.pooling == "cls":
        text_emb = encode_text_cls(model, tokenizer, query_texts, device)
        gt_idx = np.array([pid_to_gallery[q["product_id"]] for q in queries])
        # latency: time the score computation portion only
        t0 = time.perf_counter()
        scores = text_emb @ gallery_vecs.T
        latency_ms = (time.perf_counter() - t0) * 1000.0 / len(queries)
    else:
        text_vecs, text_mask = encode_text_tokens(model, tokenizer, query_texts, device)
        # gallery patches as torch tensor on device
        patches_t = torch.from_numpy(gallery_vecs).to(device, dtype=torch.float32)
        text_vecs_t = text_vecs.to(device, dtype=torch.float32)
        text_mask_t = text_mask.to(device)
        gt_idx = np.array([pid_to_gallery[q["product_id"]] for q in queries])

        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        scores_t = maxsim_score_chunked(text_vecs_t, text_mask_t, patches_t, chunk_g=64)
        torch.cuda.synchronize() if device == "cuda" else None
        latency_ms = (time.perf_counter() - t0) * 1000.0 / len(queries)
        scores = scores_t.cpu().numpy()

    recall, mrr, ranks = metrics_at_k(scores, gt_idx)

    by_diff = {}
    for q, r in zip(queries, ranks):
        d = q.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append(r)
    diff_metrics = {
        d: {
            "n": len(rs),
            "R@10": float((np.array(rs) <= 10).mean()),
            "R@5": float((np.array(rs) <= 5).mean()),
            "MRR": float((1.0 / np.array(rs)).mean()),
        }
        for d, rs in by_diff.items()
    }

    summary = {
        "tag": tag,
        "lora": str(args.lora) if args.lora else None,
        "pooling": args.pooling,
        "backbone": MODEL_ID,
        "n_queries": len(queries),
        "gallery_size": len(gallery_pids),
        "storage_floats_per_item": int(storage_per_item),
        "latency_ms_per_query": round(latency_ms, 4),
        "metrics": {**recall, "MRR": mrr},
        "by_difficulty": diff_metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_path = OUTPUT_DIR / f"eval_{tag}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    detail = [
        {**q, "rank": r, "top10_pids": [gallery_pids[i] for i in np.argsort(-scores[qi])[:10].tolist()]}
        for qi, (q, r) in enumerate(zip(queries, ranks))
    ]
    (OUTPUT_DIR / f"eval_{tag}_detail.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
