"""Evaluate fair hybrid retrieval variants.

This script keeps the held-out hand-written eval set untouched for tuning.
It tunes fusion weights on a product-level validation slice drawn only from
training products, then reports the chosen configuration on the locked eval
queries.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
TRAIN_PAIRS = Path(CONFIG["paths"]["train_pairs_final"])
EVAL_QUERIES = Path(CONFIG["paths"]["eval_queries"])
VISION_EMB = Path(CONFIG["paths"]["vision_emb"])
VISION_IDX = Path(CONFIG["paths"]["vision_idx"])
OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"])

MODEL_ID = CONFIG["backbone"]["model_id"]
MAX_LEN = CONFIG["backbone"]["text_max_length"]
SEED = CONFIG["project"]["seed"]


def product_id_from_link(link: str) -> str | None:
    m = re.search(r"/(?:goods|products)/(\d+)", link)
    return m.group(1) if link else None


def dedupe_products(items: list[dict]) -> list[dict]:
    deduped = {}
    for item in items:
        pid = item.get("product_id") or product_id_from_link(item.get("product_link", ""))
        if pid is None:
            continue
        row = dict(item)
        row["product_id"] = pid
        deduped.setdefault(pid, row)
    return list(deduped.values())


def normalize_rows(scores: np.ndarray) -> np.ndarray:
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return (scores - mean) / np.maximum(std, 1e-6)


def metrics_at_k(scores: np.ndarray, gt_indices: np.ndarray, k_list=(1, 5, 10)):
    ranks = []
    for s, gt in zip(scores, gt_indices):
        order = np.argsort(-s)
        rank = int(np.where(order == gt)[0][0]) + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    recall = {f"R@{k}": float((ranks <= k).mean()) for k in k_list}
    return {**recall, "MRR": float((1.0 / ranks).mean())}, ranks.tolist()


def char_ngrams(text: str, min_n=1, max_n=3) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    compact = cleaned.replace(" ", "")
    toks = []
    toks.extend(re.findall(r"[가-힣a-z0-9]+", cleaned))
    for n in range(min_n, max_n + 1):
        toks.extend(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
    return [t for t in toks if t]


class BM25:
    def __init__(self, docs: list[str], k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [char_ngrams(d) for d in docs]
        self.doc_len = np.array([len(toks) for toks in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 1.0
        self.term_freqs = [Counter(toks) for toks in self.doc_tokens]
        df = defaultdict(int)
        for tf in self.term_freqs:
            for term in tf:
                df[term] += 1
        n_docs = len(docs)
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.term_freqs), dtype=np.float32)
        q_terms = Counter(char_ngrams(query))
        for term, qf in q_terms.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1.0) / denom) * min(qf, 3)
        return scores

    def score_many(self, queries: list[str]) -> np.ndarray:
        return np.stack([self.score(q) for q in queries], axis=0)


def encode_text_cls(model, tokenizer, queries, device, batch_size=64):
    embs = []
    for i in range(0, len(queries), batch_size):
        chunk = queries[i : i + batch_size]
        toks = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            out = model.get_text_features(input_ids=toks["input_ids"])
        out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        embs.append(out.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def load_lora_model(lora_path: str, device: str):
    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(model, lora_path).eval().to(device)
    return model, processor.tokenizer


def make_docs(products: list[dict], mode: str) -> list[str]:
    docs = []
    for item in products:
        name = item.get("product_name") or ""
        brand = item.get("brand_id") or ""
        desc = item.get("text_description") or ""
        if mode == "name":
            docs.append(name)
        elif mode == "caption":
            docs.append(desc)
        else:
            docs.append(f"{name} {brand} {desc}")
    return docs


def eval_method(name: str, scores: np.ndarray, gt: np.ndarray) -> dict:
    metrics, ranks = metrics_at_k(scores, gt)
    return {"name": name, "metrics": metrics, "ranks": ranks}


def rerank_topn(base_scores: np.ndarray, rerank_scores: np.ndarray, n: int, weight: float) -> np.ndarray:
    out = np.full_like(base_scores, -1e9, dtype=np.float32)
    topn = np.argsort(-base_scores, axis=1)[:, :n]
    mixed = normalize_rows(base_scores) + weight * normalize_rows(rerank_scores)
    for qi, cols in enumerate(topn):
        out[qi, cols] = mixed[qi, cols]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default="checkpoints/lora_cls_384")
    parser.add_argument("--tag", default="hybrid_lora384_bm25")
    parser.add_argument("--val-products", type=int, default=240)
    parser.add_argument("--doc-mode", default="all", choices=["name", "caption", "all"])
    parser.add_argument("--select-name", default=None, help="force a named candidate after the sweep")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_products = dedupe_products(json.loads(SPLIT_TRAIN.read_text()))
    test_products = json.loads(SPLIT_TEST.read_text())
    for item in test_products:
        item["product_id"] = item.get("product_id") or product_id_from_link(item.get("product_link", ""))
    test_pids = {p["product_id"] for p in test_products}

    vision_emb_all = np.load(VISION_EMB)
    vision_idx_all = json.loads(VISION_IDX.read_text())
    pid_to_vision_row = {pid: i for i, pid in enumerate(vision_idx_all)}

    eligible_train = [p for p in train_products if p["product_id"] in pid_to_vision_row and p["product_id"] not in test_pids]
    rng = random.Random(SEED)
    val_pids = set(rng.sample([p["product_id"] for p in eligible_train], min(args.val_products, len(eligible_train))))
    val_products = [p for p in eligible_train if p["product_id"] in val_pids]

    pairs = []
    with TRAIN_PAIRS.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("query") and rec["product_id"] in val_pids:
                pairs.append(rec)
    val_queries = [p["query"] for p in pairs]
    val_gt_pids = [p["product_id"] for p in pairs]

    eval_queries = [json.loads(line) for line in EVAL_QUERIES.read_text().splitlines() if line.strip()]
    eval_texts = [q["query"] for q in eval_queries]
    eval_gt_pids = [q["product_id"] for q in eval_queries]

    def gallery_matrix(products: list[dict]):
        rows = [pid_to_vision_row[p["product_id"]] for p in products]
        pids = [p["product_id"] for p in products]
        embs = vision_emb_all[rows]
        return pids, embs

    val_gallery_pids, val_img = gallery_matrix(val_products)
    eval_gallery_pids, eval_img = gallery_matrix([p for p in test_products if p["product_id"] in pid_to_vision_row])
    val_pid_to_idx = {pid: i for i, pid in enumerate(val_gallery_pids)}
    eval_pid_to_idx = {pid: i for i, pid in enumerate(eval_gallery_pids)}
    val_gt = np.array([val_pid_to_idx[pid] for pid in val_gt_pids], dtype=np.int64)
    eval_gt = np.array([eval_pid_to_idx[pid] for pid in eval_gt_pids], dtype=np.int64)

    t0 = time.perf_counter()
    model, tokenizer = load_lora_model(args.lora, device)
    val_q_emb = encode_text_cls(model, tokenizer, val_queries, device)
    eval_q_emb = encode_text_cls(model, tokenizer, eval_texts, device)
    val_doc_emb = encode_text_cls(model, tokenizer, make_docs(val_products, args.doc_mode), device)
    eval_products = [p for p in test_products if p["product_id"] in pid_to_vision_row]
    eval_doc_emb = encode_text_cls(model, tokenizer, make_docs(eval_products, args.doc_mode), device)
    val_img_scores = val_q_emb @ val_img.T
    eval_img_scores = eval_q_emb @ eval_img.T
    val_dense_text_scores = val_q_emb @ val_doc_emb.T
    eval_dense_text_scores = eval_q_emb @ eval_doc_emb.T
    model_load_encode_sec = time.perf_counter() - t0

    val_bm25 = BM25(make_docs(val_products, args.doc_mode)).score_many(val_queries)
    eval_bm25 = BM25(make_docs(eval_products, args.doc_mode)).score_many(eval_texts)

    val_img_z = normalize_rows(val_img_scores)
    eval_img_z = normalize_rows(eval_img_scores)
    val_bm25_z = normalize_rows(val_bm25)
    eval_bm25_z = normalize_rows(eval_bm25)
    val_dense_z = normalize_rows(val_dense_text_scores)
    eval_dense_z = normalize_rows(eval_dense_text_scores)

    candidates = []
    candidates.append(("image_lora384", 1.0, 0.0, 0.0, val_img_z, eval_img_z))
    candidates.append(("bm25", 0.0, 1.0, 0.0, val_bm25_z, eval_bm25_z))
    candidates.append(("dense_text", 0.0, 0.0, 1.0, val_dense_z, eval_dense_z))
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0]:
        candidates.append((f"image_plus_{w:g}bm25", 1.0, w, 0.0, val_img_z + w * val_bm25_z, eval_img_z + w * eval_bm25_z))
        candidates.append((f"image_plus_{w:g}dense", 1.0, 0.0, w, val_img_z + w * val_dense_z, eval_img_z + w * eval_dense_z))
    for wb in [0.05, 0.10, 0.15, 0.20]:
        for wd in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            candidates.append(
                (
                    f"image_plus_{wb:g}bm25_{wd:g}dense",
                    1.0,
                    wb,
                    wd,
                    val_img_z + wb * val_bm25_z + wd * val_dense_z,
                    eval_img_z + wb * eval_bm25_z + wd * eval_dense_z,
                )
            )
    for n in [10, 20, 50]:
        for w in [0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.25, 0.30, 0.50, 0.75, 1.0]:
            candidates.append(
                (
                    f"rerank_top{n}_{w:g}bm25",
                    1.0,
                    w,
                    0.0,
                    rerank_topn(val_img_scores, val_bm25, n, w),
                    rerank_topn(eval_img_scores, eval_bm25, n, w),
                )
            )
            candidates.append(
                (
                    f"rerank_top{n}_{w:g}dense",
                    1.0,
                    0.0,
                    w,
                    rerank_topn(val_img_scores, val_dense_text_scores, n, w),
                    rerank_topn(eval_img_scores, eval_dense_text_scores, n, w),
                )
            )

    val_results = [eval_method(name, val_scores, val_gt) for name, _, _, _, val_scores, _ in candidates]
    if args.select_name:
        names = [r["name"] for r in val_results]
        if args.select_name not in names:
            raise ValueError(f"unknown candidate {args.select_name}; available examples: {names[:10]}")
        best_idx = names.index(args.select_name)
    else:
        best_idx = max(
            range(len(val_results)),
            key=lambda i: (
                val_results[i]["metrics"]["MRR"],
                val_results[i]["metrics"]["R@1"],
                val_results[i]["metrics"]["R@5"],
                val_results[i]["metrics"]["R@10"],
            ),
        )
    best_name, image_weight, bm25_weight, dense_text_weight, _, best_eval_scores = candidates[best_idx]

    eval_results = [eval_method(name, eval_scores, eval_gt) for name, _, _, _, _, eval_scores in candidates]
    selected_eval = eval_method(best_name, best_eval_scores, eval_gt)

    summary = {
        "tag": args.tag,
        "method": "CLS LoRA image score + Korean char-ngram BM25 metadata score",
        "tuning_policy": "fusion weight selected only on train-split validation products; locked eval not used for selection",
        "lora": args.lora,
        "backbone": MODEL_ID,
        "doc_mode": args.doc_mode,
        "val_products": len(val_products),
        "val_queries": len(val_queries),
        "eval_queries": len(eval_queries),
        "eval_gallery_size": len(eval_gallery_pids),
        "selected": {
            "name": best_name,
            "image_weight": image_weight,
            "bm25_weight": bm25_weight,
            "dense_text_weight": dense_text_weight,
            "val_metrics": val_results[best_idx]["metrics"],
            "eval_metrics": selected_eval["metrics"],
        },
        "all_val_results": val_results,
        "all_eval_results": eval_results,
        "runtime_sec": round(model_load_encode_sec, 3),
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
                "top10_pids": [eval_gallery_pids[i] for i in order[qi, :10].tolist()],
            }
        )
    (OUTPUT_DIR / f"eval_{args.tag}_detail.json").write_text(json.dumps(details, ensure_ascii=False, indent=2))
    print(json.dumps(summary["selected"], ensure_ascii=False, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
