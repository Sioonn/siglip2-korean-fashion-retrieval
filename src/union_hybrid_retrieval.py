"""Union-candidate hybrid retrieval.

Unlike candidate-preserving reranking, this experiment allows metadata BM25 to
add candidates that were missed by the first-stage vision-language retriever.
The union policy is selected on train-split validation products only.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

from src.hybrid_retrieval import (  # noqa: E402
    BM25,
    dedupe_products,
    encode_text_cls,
    eval_method,
    load_lora_model,
    make_docs,
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


def test_products_with_pid():
    out = []
    for item in json.loads(SPLIT_TEST.read_text()):
        row = dict(item)
        row["product_id"] = row.get("product_id") or product_id_from_link(row.get("product_link", ""))
        if row["product_id"]:
            out.append(row)
    return out


def union_scores(base_scores, bm25_scores, image_k, bm25_k, weight):
    out = np.full_like(base_scores, -1e9, dtype=np.float32)
    base_z = normalize_rows(base_scores)
    bm25_z = normalize_rows(bm25_scores)
    mixed = base_z + weight * bm25_z
    top_image = np.argsort(-base_scores, axis=1)[:, :image_k]
    top_bm25 = np.argsort(-bm25_scores, axis=1)[:, :bm25_k]
    for qi in range(base_scores.shape[0]):
        cols = np.unique(np.concatenate([top_image[qi], top_bm25[qi]]))
        out[qi, cols] = mixed[qi, cols]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default="checkpoints/lora_cls_384")
    parser.add_argument("--tag", default="union_hybrid_lora384_bm25")
    parser.add_argument("--val-products", type=int, default=240)
    parser.add_argument("--doc-mode", default="all", choices=["name", "caption", "all"])
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    train_products = dedupe_products(json.loads(SPLIT_TRAIN.read_text()))
    test_products = test_products_with_pid()
    test_pids = {p["product_id"] for p in test_products}
    vision_emb = np.load(VISION_EMB)
    vision_idx = json.loads(VISION_IDX.read_text())
    pid_to_row = {pid: i for i, pid in enumerate(vision_idx)}
    eligible_train = [p for p in train_products if p["product_id"] in pid_to_row and p["product_id"] not in test_pids]
    val_pids = set(random.Random(SEED).sample([p["product_id"] for p in eligible_train], min(args.val_products, len(eligible_train))))
    val_products = [p for p in eligible_train if p["product_id"] in val_pids]
    eval_products = [p for p in test_products if p["product_id"] in pid_to_row]

    pairs = [json.loads(line) for line in TRAIN_PAIRS.read_text().splitlines() if line.strip()]
    val_pairs = [p for p in pairs if p.get("product_id") in val_pids]
    eval_queries = [json.loads(line) for line in EVAL_QUERIES.read_text().splitlines() if line.strip()]
    val_texts = [p["query"] for p in val_pairs]
    eval_texts = [q["query"] for q in eval_queries]
    val_gallery_pids = [p["product_id"] for p in val_products]
    eval_gallery_pids = [p["product_id"] for p in eval_products]
    val_gt = np.array([{pid: i for i, pid in enumerate(val_gallery_pids)}[p["product_id"]] for p in val_pairs], dtype=np.int64)
    eval_gt = np.array([{pid: i for i, pid in enumerate(eval_gallery_pids)}[q["product_id"]] for q in eval_queries], dtype=np.int64)

    model, tokenizer = load_lora_model(args.lora, device)
    val_q = encode_text_cls(model, tokenizer, val_texts, device)
    eval_q = encode_text_cls(model, tokenizer, eval_texts, device)
    val_img = vision_emb[[pid_to_row[pid] for pid in val_gallery_pids]]
    eval_img = vision_emb[[pid_to_row[pid] for pid in eval_gallery_pids]]
    val_base = val_q @ val_img.T
    eval_base = eval_q @ eval_img.T
    val_bm25 = BM25(make_docs(val_products, args.doc_mode)).score_many(val_texts)
    eval_bm25 = BM25(make_docs(eval_products, args.doc_mode)).score_many(eval_texts)

    candidates = []
    for image_k in [10, 20, 50]:
        for bm25_k in [5, 10, 20, 50, 100]:
            for weight in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]:
                name = f"union_i{image_k}_b{bm25_k}_w{weight:g}"
                candidates.append(
                    (
                        name,
                        image_k,
                        bm25_k,
                        weight,
                        union_scores(val_base, val_bm25, image_k, bm25_k, weight),
                        union_scores(eval_base, eval_bm25, image_k, bm25_k, weight),
                    )
                )

    val_results = [eval_method(name, val_scores, val_gt) for name, _, _, _, val_scores, _ in candidates]
    best_idx = max(range(len(val_results)), key=lambda i: (val_results[i]["metrics"]["MRR"], val_results[i]["metrics"]["R@1"], val_results[i]["metrics"]["R@5"], val_results[i]["metrics"]["R@10"]))
    best_name, image_k, bm25_k, weight, _, best_eval_scores = candidates[best_idx]
    eval_results = [eval_method(name, eval_scores, eval_gt) for name, _, _, _, _, eval_scores in candidates]
    selected_eval = eval_method(best_name, best_eval_scores, eval_gt)
    order = np.argsort(-best_eval_scores, axis=1)

    summary = {
        "tag": args.tag,
        "method": "union of LoRA-384 image candidates and Korean char-ngram BM25 metadata candidates",
        "tuning_policy": "union sizes and weight selected only on train-split validation products",
        "selected": {
            "name": best_name,
            "image_k": image_k,
            "bm25_k": bm25_k,
            "bm25_weight": weight,
            "val_metrics": val_results[best_idx]["metrics"],
            "eval_metrics": selected_eval["metrics"],
        },
        "all_val_results": val_results,
        "all_eval_results": eval_results,
    }
    out_path = OUTPUT_DIR / f"eval_{args.tag}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    detail = [
        {**q, "rank": selected_eval["ranks"][qi], "top10_pids": [eval_gallery_pids[i] for i in order[qi, :10].tolist()]}
        for qi, q in enumerate(eval_queries)
    ]
    (OUTPUT_DIR / f"eval_{args.tag}_detail.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2))
    print(json.dumps(summary["selected"], ensure_ascii=False, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
