"""Shared experiment library for novel test-time retrieval methods.

Goal: load the SAME data / gallery / eval protocol used by the existing
scripts (evaluate.py, hybrid_retrieval.py) so any new method is directly
comparable, plus add statistics the original repo lacks (bootstrap CIs and
paired tests over the 30-query locked eval).

Design choices that match the original repo exactly:
- text encoding: get_text_features(input_ids only), padding="max_length",
  max_length=64, truncation=True, then L2-normalize  (see evaluate.py).
- gallery: cached FROZEN vision_emb subset to test-split products.
- metrics: single-positive R@1/R@5/R@10 and MRR.

Everything downstream of "queries encoded" is pure numpy, so dozens of
test-time methods can be swept in seconds.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

SPLIT_TRAIN = ROOT / "data" / "split_train.json"
SPLIT_TEST = ROOT / "data" / "split_test.json"
TRAIN_PAIRS = ROOT / "data" / "train_pairs_final.jsonl"
EVAL_QUERIES = ROOT / "data" / "eval_queries.jsonl"
VISION_EMB = ROOT / "data" / "vision_emb.npy"
VISION_IDX = ROOT / "data" / "vision_idx.json"
CACHE_DIR = ROOT / "data" / "novel_cache"
OUTPUT_DIR = ROOT / "output" / "novel"

MODEL_ID = CONFIG["backbone"]["model_id"]
MAX_LEN = CONFIG["backbone"]["text_max_length"]
SEED = CONFIG["project"]["seed"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def product_id_from_link(link: str) -> str | None:
    m = re.search(r"/(?:goods|products)/(\d+)", link or "")
    return m.group(1) if m else None


def with_pid(items: list[dict]) -> list[dict]:
    out = []
    for item in items:
        row = dict(item)
        row["product_id"] = row.get("product_id") or product_id_from_link(row.get("product_link", ""))
        if row["product_id"]:
            out.append(row)
    return out


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


@dataclass
class DataBundle:
    # gallery (test split, with cached vision embedding)
    gallery_pids: list[str]
    gallery_emb: np.ndarray            # (G, D) L2-normalized frozen vision
    gallery_products: list[dict]
    # locked eval
    eval_queries: list[dict]
    eval_texts: list[str]
    eval_gt: np.ndarray                # (n_eval,) index into gallery
    eval_diff: list[str]
    # validation (train-split synthetic queries) — for HP selection ONLY
    val_products: list[dict]
    val_gallery_pids: list[str]
    val_gallery_emb: np.ndarray        # (Gv, D)
    val_texts: list[str]
    val_gt: np.ndarray                 # (n_val,) index into val gallery
    # query bank (all train synthetic queries) — for hubness normalization
    bank_texts: list[str]


def load_data(val_products: int = 240) -> DataBundle:
    vision_emb_all = np.load(VISION_EMB)
    vision_idx_all = json.loads(VISION_IDX.read_text())
    pid_to_row = {pid: i for i, pid in enumerate(vision_idx_all)}

    test_products = with_pid(json.loads(SPLIT_TEST.read_text()))
    test_pids = {p["product_id"] for p in test_products}
    gallery_products = [p for p in test_products if p["product_id"] in pid_to_row]
    gallery_pids = [p["product_id"] for p in gallery_products]
    gallery_emb = vision_emb_all[[pid_to_row[pid] for pid in gallery_pids]].astype(np.float32)
    gallery_pid_to_idx = {pid: i for i, pid in enumerate(gallery_pids)}

    eval_queries = [json.loads(l) for l in EVAL_QUERIES.read_text().splitlines() if l.strip()]
    for q in eval_queries:
        if q["product_id"] not in gallery_pid_to_idx:
            raise ValueError(f"eval query references {q['product_id']} not in gallery")
    eval_texts = [q["query"] for q in eval_queries]
    eval_gt = np.array([gallery_pid_to_idx[q["product_id"]] for q in eval_queries], dtype=np.int64)
    eval_diff = [q.get("difficulty", "unknown") for q in eval_queries]

    # validation split (mirror hybrid_retrieval.py exactly)
    train_products = dedupe_products(json.loads(SPLIT_TRAIN.read_text()))
    eligible_train = [p for p in train_products if p["product_id"] in pid_to_row and p["product_id"] not in test_pids]
    rng = random.Random(SEED)
    val_pids = set(rng.sample([p["product_id"] for p in eligible_train], min(val_products, len(eligible_train))))
    val_products_l = [p for p in eligible_train if p["product_id"] in val_pids]
    val_gallery_pids = [p["product_id"] for p in val_products_l]
    val_gallery_emb = vision_emb_all[[pid_to_row[pid] for pid in val_gallery_pids]].astype(np.float32)
    val_pid_to_idx = {pid: i for i, pid in enumerate(val_gallery_pids)}

    pairs = [json.loads(l) for l in TRAIN_PAIRS.read_text().splitlines() if l.strip()]
    val_pairs = [p for p in pairs if p.get("query") and p.get("product_id") in val_pids]
    val_texts = [p["query"] for p in val_pairs]
    val_gt = np.array([val_pid_to_idx[p["product_id"]] for p in val_pairs], dtype=np.int64)

    # query bank = ALL train synthetic queries (for hubness statistics)
    bank_texts = [p["query"] for p in pairs if p.get("query")]

    return DataBundle(
        gallery_pids=gallery_pids,
        gallery_emb=gallery_emb,
        gallery_products=gallery_products,
        eval_queries=eval_queries,
        eval_texts=eval_texts,
        eval_gt=eval_gt,
        eval_diff=eval_diff,
        val_products=val_products_l,
        val_gallery_pids=val_gallery_pids,
        val_gallery_emb=val_gallery_emb,
        val_texts=val_texts,
        val_gt=val_gt,
        bank_texts=bank_texts,
    )


def make_docs(products: list[dict], mode: str = "all") -> list[str]:
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


# --------------------------------------------------------------------------
# encoder (cached)
# --------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


def get_model(lora_path: str | None, device: str = "cuda"):
    key = lora_path or "__base__"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import torch
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(lora_path)).eval().to(device)
    _MODEL_CACHE[key] = (model, processor.tokenizer)
    return _MODEL_CACHE[key]


def encode_text(texts: list[str], lora_path: str | None, device: str = "cuda", batch_size: int = 128) -> np.ndarray:
    """L2-normalized CLS text features, matching evaluate.py exactly."""
    import torch

    model, tokenizer = get_model(lora_path, device)
    embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        toks = tokenizer(
            chunk, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            out = model.get_text_features(input_ids=toks["input_ids"])
        out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        embs.append(out.float().cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def cache_path(tag: str, name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{tag}__{name}.npy"


def encode_cached(texts: list[str], lora_path: str | None, tag: str, name: str,
                  device: str = "cuda") -> np.ndarray:
    p = cache_path(tag, name)
    if p.exists():
        arr = np.load(p)
        if arr.shape[0] == len(texts):
            return arr
    arr = encode_text(texts, lora_path, device)
    np.save(p, arr)
    return arr


# --------------------------------------------------------------------------
# metrics + statistics
# --------------------------------------------------------------------------
def ranks_of(scores: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """rank (1-based) of the ground-truth gallery item for each query."""
    ranks = np.empty(len(gt), dtype=np.int64)
    for i in range(len(gt)):
        order = np.argsort(-scores[i])
        ranks[i] = int(np.where(order == gt[i])[0][0]) + 1
    return ranks


def metrics_from_ranks(ranks: np.ndarray, k_list=(1, 5, 10)) -> dict:
    out = {f"R@{k}": float((ranks <= k).mean()) for k in k_list}
    out["MRR"] = float((1.0 / ranks).mean())
    return out


def metrics_at_k(scores: np.ndarray, gt: np.ndarray, k_list=(1, 5, 10)) -> dict:
    return metrics_from_ranks(ranks_of(scores, gt), k_list)


def bootstrap_ci(ranks: np.ndarray, metric: str = "MRR", B: int = 10000, seed: int = 0):
    """Percentile bootstrap CI for a metric over the query set."""
    rng = np.random.default_rng(seed)
    n = len(ranks)
    idx = rng.integers(0, n, size=(B, n))
    rr = ranks[idx]
    if metric == "MRR":
        stat = (1.0 / rr).mean(axis=1)
    elif metric.startswith("R@"):
        k = int(metric[2:])
        stat = (rr <= k).mean(axis=1)
    else:
        raise ValueError(metric)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


def paired_compare(ranks_a: np.ndarray, ranks_b: np.ndarray, metric: str = "MRR",
                   B: int = 10000, seed: int = 0):
    """Paired bootstrap of (A - B) per-query metric difference + sign test.

    Positive delta => A better than B.  Returns dict with mean delta, 95% CI,
    P(delta>0), and an exact two-sided sign-test p-value on per-query wins.
    """
    from scipy import stats

    def per_query(ranks):
        if metric == "MRR":
            return 1.0 / ranks
        if metric.startswith("R@"):
            k = int(metric[2:])
            return (ranks <= k).astype(np.float64)
        raise ValueError(metric)

    a = per_query(ranks_a)
    b = per_query(ranks_b)
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    nz = wins + losses
    sign_p = float(stats.binomtest(wins, nz, 0.5).pvalue) if nz > 0 else 1.0
    return {
        "metric": metric,
        "mean_delta": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "p_delta_gt0": float((boot > 0).mean()),
        "wins": wins,
        "losses": losses,
        "ties": int((d == 0).sum()),
        "sign_test_p": sign_p,
    }


def summarize(ranks: np.ndarray, diff: list[str] | None = None) -> dict:
    m = metrics_from_ranks(ranks)
    ci = {k: bootstrap_ci(ranks, k) for k in ["R@1", "R@5", "R@10", "MRR"]}
    out = {"metrics": m, "ci95": ci, "ranks": ranks.tolist()}
    if diff is not None:
        by = {}
        for d, r in zip(diff, ranks):
            by.setdefault(d, []).append(r)
        out["by_difficulty"] = {d: metrics_from_ranks(np.array(rs)) for d, rs in by.items()}
    return out


# --------------------------------------------------------------------------
# lexical (BM25) + fusion helpers — identical math to hybrid_retrieval.py
# --------------------------------------------------------------------------
def normalize_rows(scores: np.ndarray) -> np.ndarray:
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return (scores - mean) / np.maximum(std, 1e-6)


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
        self.k1, self.b = k1, b
        self.doc_tokens = [char_ngrams(d) for d in docs]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 1.0
        self.term_freqs = [Counter(t) for t in self.doc_tokens]
        df = defaultdict(int)
        for tf in self.term_freqs:
            for term in tf:
                df[term] += 1
        n = len(docs)
        self.idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.term_freqs), dtype=np.float32)
        for term, qf in Counter(char_ngrams(query)).items():
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


def rerank_topn(base_scores: np.ndarray, rerank_scores: np.ndarray, n: int, weight: float) -> np.ndarray:
    """Candidate-preserving rerank: mix normalized scores only inside top-n of base."""
    out = np.full_like(base_scores, -1e9, dtype=np.float32)
    topn = np.argsort(-base_scores, axis=1)[:, :n]
    mixed = normalize_rows(base_scores) + weight * normalize_rows(rerank_scores)
    for qi, cols in enumerate(topn):
        out[qi, cols] = mixed[qi, cols]
    return out


def rrf(score_matrices: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of several score matrices (higher = better)."""
    fused = np.zeros_like(score_matrices[0], dtype=np.float64)
    for s in score_matrices:
        ranks = (-s).argsort(axis=1).argsort(axis=1) + 1  # 1-based rank per item
        fused += 1.0 / (k + ranks)
    return fused


# --------------------------------------------------------------------------
# cache loading + turnkey method evaluation (for WF2 agents)
# --------------------------------------------------------------------------
@dataclass
class Caches:
    gallery_emb: np.ndarray
    val_gallery_emb: np.ndarray
    eval_gt: np.ndarray
    val_gt: np.ndarray
    eval_diff: list
    gallery_pids: list
    val_gallery_pids: list
    enc: dict = field(default_factory=dict)  # tag -> {eval_q,val_q,bank_q,gallery_doc,val_doc}

    def q(self, tag):  # eval query emb under encoder `tag`
        return self.enc[tag]["eval_q"]

    def vq(self, tag):
        return self.enc[tag]["val_q"]

    def bank(self, tag):
        return self.enc[tag]["bank_q"]


def load_refs() -> dict:
    """{name: eval_ranks(np.int64)} for paired comparisons (built by refs.py / final suite)."""
    d = json.loads((CACHE_DIR / "refs.json").read_text())
    return {k: np.array(v, dtype=np.int64) for k, v in d.items()}


def load_synth_eval(encoders=("base", "lora384")) -> dict | None:
    """High-power secondary eval: synthetic user-style queries on UNSEEN test
    products (generated locally, never used in training). Contract produced by
    the generator: data/novel_cache/synth_eval.jsonl (lines {product_id,query,qtype})
    and cached embeddings <tag>__synth_eval_q.npy aligned row-for-row to the jsonl.
    Returns None if not generated yet. gt indexes the SAME 588 gallery as eval.
    """
    f = CACHE_DIR / "synth_eval.jsonl"
    if not f.exists():
        return None
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    gpid_to_idx = {pid: i for i, pid in enumerate(meta["gallery_pids"])}
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    keep = [r for r in rows if r["product_id"] in gpid_to_idx]
    if len(keep) != len(rows):
        raise ValueError(f"synth_eval has {len(rows)-len(keep)} queries on products not in gallery")
    texts = [r["query"] for r in keep]
    gt = np.array([gpid_to_idx[r["product_id"]] for r in keep], dtype=np.int64)
    diff = [r.get("qtype", "synth") for r in keep]
    enc = {}
    for tag in encoders:
        arr = np.load(cache_path(tag, "synth_eval_q"))
        if arr.shape[0] != len(texts):
            raise ValueError(f"{tag} synth emb rows {arr.shape[0]} != {len(texts)} queries")
        enc[tag] = arr
    return {"texts": texts, "gt": gt, "diff": diff, "enc": enc, "n": len(texts)}


def load_caches(encoders=("base", "lora384")) -> Caches:
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    c = Caches(
        gallery_emb=np.load(cache_path("shared", "gallery_emb")),
        val_gallery_emb=np.load(cache_path("shared", "val_gallery_emb")),
        eval_gt=np.array(meta["eval_gt"], dtype=np.int64),
        val_gt=np.array(meta["val_gt"], dtype=np.int64),
        eval_diff=meta["eval_diff"],
        gallery_pids=meta["gallery_pids"],
        val_gallery_pids=meta["val_gallery_pids"],
    )
    for tag in encoders:
        c.enc[tag] = {
            nm: np.load(cache_path(tag, nm))
            for nm in ["eval_q", "val_q", "bank_q", "gallery_doc", "val_doc"]
        }
    return c


def select_eval(candidates, val_gt, eval_gt, eval_diff=None, refs=None,
                select_key=("MRR", "R@1", "R@5", "R@10")):
    """candidates: list of (label, val_scores, eval_scores).

    Selects the candidate maximizing the lexicographic select_key on VALIDATION,
    then reports the chosen candidate's locked-eval metrics + bootstrap CIs +
    per-difficulty + paired comparisons vs each reference (refs: name->eval_ranks).
    Also returns the full per-candidate eval table for transparency.
    """
    val_m = [metrics_at_k(vs, val_gt) for _, vs, _ in candidates]
    best = max(range(len(candidates)), key=lambda i: tuple(val_m[i][k] for k in select_key))
    best_label, _, best_eval_scores = candidates[best]
    eval_ranks = ranks_of(best_eval_scores, eval_gt)
    res = {
        "selected": best_label,
        "val_metrics": val_m[best],
        "eval": summarize(eval_ranks, eval_diff),
        "all_eval": [
            {"label": lab, "val": val_m[i],
             "eval": metrics_at_k(es, eval_gt)}
            for i, (lab, _, es) in enumerate(candidates)
        ],
    }
    if refs:
        res["paired_vs"] = {
            name: {m: paired_compare(eval_ranks, rr, m) for m in ["MRR", "R@1", "R@10"]}
            for name, rr in refs.items()
        }
    res["eval_ranks"] = eval_ranks.tolist()
    return res
