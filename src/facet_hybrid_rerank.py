"""Combine lexical BM25 and interpretable attribute facets for top-N rerank."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

from src.attribute_rerank import facet_matrix
from src.hybrid_retrieval import (
    BM25,
    dedupe_products,
    encode_text_cls,
    load_lora_model,
    make_docs,
    metrics_at_k,
    normalize_rows,
    product_id_from_link,
)

SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
TRAIN_PAIRS = Path(CONFIG["paths"]["train_pairs_final"])
EVAL_QUERIES = Path(CONFIG["paths"]["eval_queries"])
VISION_EMB = Path(CONFIG["paths"]["vision_emb"])
VISION_IDX = Path(CONFIG["paths"]["vision_idx"])
OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"])
SEED = CONFIG["project"]["seed"]


def with_pid(items):
    out = []
    for item in items:
        row = dict(item)
        row["product_id"] = row.get("product_id") or product_id_from_link(row.get("product_link", ""))
        if row["product_id"]:
            out.append(row)
    return out


def rerank_topn(base_scores, extra_scores, topn):
    out = np.full_like(base_scores, -1e9, dtype=np.float32)
    top = np.argsort(-base_scores, axis=1)[:, :topn]
    for qi, cols in enumerate(top):
        out[qi, cols] = extra_scores[qi, cols]
    return out


def eval_method(name, scores, gt):
    metrics, ranks = metrics_at_k(scores, gt)
    return {"name": name, "metrics": metrics, "ranks": ranks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default="checkpoints/lora_cls_384")
    parser.add_argument("--tag", default="facet_hybrid_rerank_lora384")
    parser.add_argument("--val-products", type=int, default=240)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--doc-mode", default="all", choices=["name", "caption", "all"])
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    train_products = dedupe_products(json.loads(SPLIT_TRAIN.read_text()))
    test_products = with_pid(json.loads(SPLIT_TEST.read_text()))
    test_pids = {p["product_id"] for p in test_products}
    vision_emb_all = np.load(VISION_EMB)
    vision_idx = json.loads(VISION_IDX.read_text())
    pid_to_row = {pid: i for i, pid in enumerate(vision_idx)}

    eligible_train = [p for p in train_products if p["product_id"] in pid_to_row and p["product_id"] not in test_pids]
    rng = random.Random(SEED)
    val_pids = set(rng.sample([p["product_id"] for p in eligible_train], min(args.val_products, len(eligible_train))))
    val_products = [p for p in eligible_train if p["product_id"] in val_pids]
    eval_products = [p for p in test_products if p["product_id"] in pid_to_row]

    pairs = [json.loads(line) for line in TRAIN_PAIRS.read_text().splitlines() if line.strip()]
    val_pairs = [p for p in pairs if p.get("product_id") in val_pids]
    eval_queries = [json.loads(line) for line in EVAL_QUERIES.read_text().splitlines() if line.strip()]
    val_texts = [p["query"] for p in val_pairs]
    eval_texts = [q["query"] for q in eval_queries]

    def gallery(products):
        pids = [p["product_id"] for p in products]
        return pids, vision_emb_all[[pid_to_row[pid] for pid in pids]]

    val_pids_gallery, val_img = gallery(val_products)
    eval_pids_gallery, eval_img = gallery(eval_products)
    val_pid_to_idx = {pid: i for i, pid in enumerate(val_pids_gallery)}
    eval_pid_to_idx = {pid: i for i, pid in enumerate(eval_pids_gallery)}
    val_gt = np.array([val_pid_to_idx[p["product_id"]] for p in val_pairs], dtype=np.int64)
    eval_gt = np.array([eval_pid_to_idx[q["product_id"]] for q in eval_queries], dtype=np.int64)

    model, tokenizer = load_lora_model(args.lora, device)
    val_q = encode_text_cls(model, tokenizer, val_texts, device)
    eval_q = encode_text_cls(model, tokenizer, eval_texts, device)
    val_base = val_q @ val_img.T
    eval_base = eval_q @ eval_img.T

    val_docs = make_docs(val_products, args.doc_mode)
    eval_docs = make_docs(eval_products, args.doc_mode)
    val_bm25 = BM25(val_docs).score_many(val_texts)
    eval_bm25 = BM25(eval_docs).score_many(eval_texts)
    val_attr = facet_matrix(val_texts, val_docs)
    eval_attr = facet_matrix(eval_texts, eval_docs)

    val_base_z = normalize_rows(val_base)
    eval_base_z = normalize_rows(eval_base)
    val_bm25_z = normalize_rows(val_bm25)
    eval_bm25_z = normalize_rows(eval_bm25)
    val_attr_z = normalize_rows(val_attr)
    eval_attr_z = normalize_rows(eval_attr)

    candidates = [("image_lora384", val_base, eval_base)]
    weights = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75]
    for wb in weights:
        for wa in weights:
            val_mix = val_base_z + wb * val_bm25_z + wa * val_attr_z
            eval_mix = eval_base_z + wb * eval_bm25_z + wa * eval_attr_z
            candidates.append((f"rerank_top{args.topn}_{wb:g}bm25_{wa:g}attr", rerank_topn(val_base, val_mix, args.topn), rerank_topn(eval_base, eval_mix, args.topn)))

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
    summary = {
        "tag": args.tag,
        "method": "LoRA-384 top-N rerank with Korean char-ngram BM25 plus attribute-facet score",
        "tuning_policy": "weights selected only on train-split validation products; locked eval not used for selection",
        "doc_mode": args.doc_mode,
        "topn": args.topn,
        "selected": {
            "name": best_name,
            "val_metrics": val_results[best_idx]["metrics"],
            "eval_metrics": selected_eval["metrics"],
        },
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
