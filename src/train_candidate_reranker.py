"""Train a lightweight candidate-preserving reranker.

The first-stage candidate set is the top-10 list from LoRA-384 image retrieval.
The reranker only reorders those candidates, so first-stage R@10 is preserved.
Features are deliberately simple and report-friendly:

  [image_score_z, bm25_z, dense_text_z, reciprocal_image_rank]

The model is trained on train-split augmented queries and selected on a
train-product validation slice. Locked eval is used only once at the end.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

from src.hybrid_retrieval import BM25, char_ngrams, encode_text_cls, make_docs, metrics_at_k, product_id_from_link

SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
TRAIN_PAIRS = Path(CONFIG["paths"]["train_pairs_final"])
EVAL_QUERIES = Path(CONFIG["paths"]["eval_queries"])
VISION_EMB = Path(CONFIG["paths"]["vision_emb"])
VISION_IDX = Path(CONFIG["paths"]["vision_idx"])
OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"])
CKPT_DIR = Path(CONFIG["paths"]["checkpoints_dir"])

MODEL_ID = CONFIG["backbone"]["model_id"]
SEED = CONFIG["project"]["seed"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_lora_model(lora_path: str, device: str):
    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(model, lora_path).eval().to(device)
    return model, processor.tokenizer


def with_pid(items):
    out = []
    for it in items:
        row = dict(it)
        row["product_id"] = row.get("product_id") or product_id_from_link(row.get("product_link", ""))
        if row["product_id"]:
            out.append(row)
    return out


def dedupe(items):
    seen = {}
    for it in with_pid(items):
        seen.setdefault(it["product_id"], it)
    return list(seen.values())


def row_normalize(x):
    return (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)


def bm25_subset_score(bm25, query: str, cols: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(cols), dtype=np.float32)
    q_terms = Counter(char_ngrams(query))
    for term, qf in q_terms.items():
        idf = bm25.idf.get(term)
        if idf is None:
            continue
        for j, i in enumerate(cols):
            tf = bm25.term_freqs[int(i)]
            f = tf.get(term, 0)
            if f == 0:
                continue
            denom = f + bm25.k1 * (1.0 - bm25.b + bm25.b * bm25.doc_len[int(i)] / bm25.avgdl)
            scores[j] += idf * (f * (bm25.k1 + 1.0) / denom) * min(qf, 3)
    return scores


def normalize_vector(x):
    return (x - x.mean()) / max(float(x.std()), 1e-6)


def build_top_features(image_scores, dense_scores, bm25, query_texts, topn=10):
    top = np.argsort(-image_scores, axis=1)[:, :topn]
    feats = []
    for qi, cols in enumerate(top):
        image_z = normalize_vector(image_scores[qi, cols])
        dense_z = normalize_vector(dense_scores[qi, cols])
        bm25_z = normalize_vector(bm25_subset_score(bm25, query_texts[qi], cols))
        rows = []
        for rank0 in range(len(cols)):
            rows.append([image_z[rank0], bm25_z[rank0], dense_z[rank0], 1.0 / (rank0 + 1)])
        feats.append(rows)
    return np.array(feats, dtype=np.float32), top


class LogisticReranker(nn.Module):
    def __init__(self, n_features=4):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPReranker(nn.Module):
    def __init__(self, n_features=4, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_epoch(model, optim, feats, top_idx, gt_idx, device):
    model.train()
    labels = (top_idx == gt_idx[:, None]).astype(np.float32)
    keep = labels.sum(axis=1) > 0
    if not keep.any():
        return 0.0
    x = torch.from_numpy(feats[keep]).to(device)
    y = torch.from_numpy(labels[keep]).to(device)
    logits = model(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], x.shape[1])
    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(9.0, device=device))
    optim.zero_grad()
    loss.backward()
    optim.step()
    return float(loss.detach().cpu())


def score_with_model(model, feats, top_idx, gallery_size, device):
    model.eval()
    scores = np.full((feats.shape[0], gallery_size), -1e9, dtype=np.float32)
    with torch.inference_mode():
        for qi in range(feats.shape[0]):
            x = torch.from_numpy(feats[qi]).to(device)
            pred = model(x).detach().cpu().numpy()
            scores[qi, top_idx[qi]] = pred
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default="checkpoints/lora_cls_384")
    parser.add_argument("--tag", default="candidate_reranker_lora384")
    parser.add_argument("--model", default="logistic", choices=["logistic", "mlp"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--val-products", type=int, default=240)
    args = parser.parse_args()

    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_DIR / args.tag).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_products = dedupe(json.loads(SPLIT_TRAIN.read_text()))
    test_products = with_pid(json.loads(SPLIT_TEST.read_text()))
    test_pids = {p["product_id"] for p in test_products}

    vision_emb_all = np.load(VISION_EMB).astype("float32")
    vision_idx = json.loads(VISION_IDX.read_text())
    pid_to_row = {pid: i for i, pid in enumerate(vision_idx)}

    train_products = [p for p in train_products if p["product_id"] in pid_to_row and p["product_id"] not in test_pids]
    rng = random.Random(SEED)
    val_pids = set(rng.sample([p["product_id"] for p in train_products], min(args.val_products, len(train_products))))
    fit_products = [p for p in train_products if p["product_id"] not in val_pids]
    val_products = [p for p in train_products if p["product_id"] in val_pids]
    eval_products = [p for p in test_products if p["product_id"] in pid_to_row]

    pairs = [json.loads(l) for l in TRAIN_PAIRS.read_text().splitlines() if l.strip()]
    fit_pairs = [p for p in pairs if p["product_id"] in {x["product_id"] for x in fit_products}]
    val_pairs = [p for p in pairs if p["product_id"] in val_pids]
    eval_queries = [json.loads(l) for l in EVAL_QUERIES.read_text().splitlines() if l.strip()]

    model, tokenizer = load_lora_model(args.lora, device)

    def prepare(query_texts, products, gt_pids):
        pids = [p["product_id"] for p in products]
        gallery = vision_emb_all[[pid_to_row[pid] for pid in pids]]
        q_emb = encode_text_cls(model, tokenizer, query_texts, device)
        docs = make_docs(products, "all")
        doc_emb = encode_text_cls(model, tokenizer, docs, device)
        img_scores = q_emb @ gallery.T
        dense_scores = q_emb @ doc_emb.T
        feats, top_idx = build_top_features(img_scores, dense_scores, BM25(docs), query_texts, topn=10)
        pid_to_idx = {pid: i for i, pid in enumerate(pids)}
        gt_idx = np.array([pid_to_idx[pid] for pid in gt_pids], dtype=np.int64)
        return feats, top_idx, gt_idx, len(pids), pids

    fit_feats, fit_top, fit_gt, _, _ = prepare([p["query"] for p in fit_pairs], fit_products, [p["product_id"] for p in fit_pairs])
    val_feats, val_top, val_gt, val_gallery_size, _ = prepare([p["query"] for p in val_pairs], val_products, [p["product_id"] for p in val_pairs])
    eval_feats, eval_top, eval_gt, eval_gallery_size, eval_pids = prepare([q["query"] for q in eval_queries], eval_products, [q["product_id"] for q in eval_queries])

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    reranker = LogisticReranker() if args.model == "logistic" else MLPReranker()
    reranker.to(device)
    optim = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=1e-3)

    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(reranker, optim, fit_feats, fit_top, fit_gt, device)
        val_scores = score_with_model(reranker, val_feats, val_top, val_gallery_size, device)
        val_metrics, _ = metrics_at_k(val_scores, val_gt)
        history.append({"epoch": epoch, "loss": loss, "val_metrics": val_metrics})
        key = (val_metrics["MRR"], val_metrics["R@1"], val_metrics["R@5"], val_metrics["R@10"])
        if best is None or key > best["key"]:
            best = {"key": key, "epoch": epoch, "state": {k: v.detach().cpu() for k, v in reranker.state_dict().items()}, "val_metrics": val_metrics}

    reranker.load_state_dict(best["state"])
    eval_scores = score_with_model(reranker, eval_feats, eval_top, eval_gallery_size, device)
    eval_metrics, eval_ranks = metrics_at_k(eval_scores, eval_gt)

    torch.save({"state_dict": best["state"], "args": vars(args), "best_epoch": best["epoch"]}, CKPT_DIR / args.tag / "reranker.pt")
    summary = {
        "tag": args.tag,
        "method": "candidate-preserving learned reranker over LoRA-384 top-10",
        "selection_policy": "best epoch selected on train-split validation products only",
        "model": args.model,
        "features": ["image_score_z", "bm25_z", "dense_text_z", "reciprocal_image_rank"],
        "fit_pairs": len(fit_pairs),
        "val_pairs": len(val_pairs),
        "eval_queries": len(eval_queries),
        "best_epoch": best["epoch"],
        "best_val_metrics": best["val_metrics"],
        "eval_metrics": eval_metrics,
        "history": history,
    }
    (OUTPUT_DIR / f"eval_{args.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    order = np.argsort(-eval_scores, axis=1)
    detail = []
    for qi, q in enumerate(eval_queries):
        detail.append({**q, "rank": eval_ranks[qi], "top10_pids": [eval_pids[i] for i in order[qi, :10].tolist()]})
    (OUTPUT_DIR / f"eval_{args.tag}_detail.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ["tag", "best_epoch", "best_val_metrics", "eval_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
