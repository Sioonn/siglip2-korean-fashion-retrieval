"""Attribute-facet reranking for Korean fashion retrieval.

The first stage remains LoRA-384 image retrieval. This second stage extracts
coarse fashion facets from the query and product text, then reorders only the
first-stage top-N candidates. The module is intentionally interpretable:
colors, garment type, sleeve length, pattern/graphic cues, fit, and neck/collar
terms become a sparse matching score.
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

from src.hybrid_retrieval import (
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

FACETS = {
    "color": {
        "white": ["흰", "화이트", "아이보리", "크림", "오트밀"],
        "black": ["검정", "검은", "블랙", "차콜"],
        "gray": ["회색", "그레이", "멜란지"],
        "navy": ["네이비", "남색"],
        "blue": ["파란", "파랑", "블루", "하늘색", "스카이"],
        "green": ["초록", "그린", "카키", "민트"],
        "red": ["빨간", "빨강", "레드", "버건디", "와인"],
        "brown": ["갈색", "브라운", "베이지", "카멜"],
        "yellow": ["노란", "노랑", "옐로"],
        "pink": ["분홍", "핑크"],
        "purple": ["보라", "퍼플", "라벤더"],
    },
    "garment": {
        "short_t": ["반팔", "티셔츠", "티"],
        "long_t": ["긴팔", "롱슬리브"],
        "sweatshirt": ["맨투맨", "스웨트셔츠", "스웨트"],
        "shirt": ["셔츠", "남방"],
        "knit": ["니트", "스웨터", "가디건"],
        "hoodie": ["후드", "후디"],
        "jacket": ["자켓", "재킷", "점퍼", "블루종"],
        "pants": ["팬츠", "바지", "데님", "진"],
    },
    "sleeve": {
        "short": ["반팔", "짧은 소매"],
        "long": ["긴팔", "긴 소매", "롱슬리브"],
        "sleeveless": ["민소매", "슬리브리스"],
    },
    "pattern": {
        "stripe": ["스트라이프", "줄무늬", "세로선", "가로선"],
        "check": ["체크", "격자"],
        "graphic": ["그림", "그래픽", "프린팅", "프린트", "캐릭터", "일러스트", "로고"],
        "letter": ["글씨", "문구", "레터링", "영어", "알파벳"],
        "animal": ["강아지", "고양이", "고래", "동물", "말"],
        "solid": ["무지", "기본", "심플"],
    },
    "fit": {
        "over": ["오버핏", "루즈", "넉넉", "여유"],
        "regular": ["레귤러", "기본핏"],
        "slim": ["슬림", "타이트"],
    },
    "neck": {
        "crew": ["크루넥", "라운드", "원형"],
        "v": ["브이넥", "v넥"],
        "collar": ["카라", "칼라"],
        "hood": ["후드"],
    },
}

FACET_WEIGHT = {
    "color": 1.4,
    "garment": 1.2,
    "sleeve": 1.1,
    "pattern": 1.5,
    "fit": 0.7,
    "neck": 0.7,
}


def with_pid(items):
    out = []
    for item in items:
        row = dict(item)
        row["product_id"] = row.get("product_id") or product_id_from_link(row.get("product_link", ""))
        if row["product_id"]:
            out.append(row)
    return out


def extract_facets(text):
    text = text.lower()
    found = {}
    for facet, groups in FACETS.items():
        vals = set()
        for key, terms in groups.items():
            if any(term.lower() in text for term in terms):
                vals.add(key)
        if vals:
            found[facet] = vals
    return found


def facet_score(query_facets, doc_facets):
    score = 0.0
    for facet, q_vals in query_facets.items():
        if facet not in doc_facets:
            continue
        overlap = len(q_vals & doc_facets[facet])
        if overlap:
            score += FACET_WEIGHT[facet] * overlap / max(len(q_vals), 1)
    return score


def facet_matrix(queries, docs):
    q_facets = [extract_facets(q) for q in queries]
    d_facets = [extract_facets(d) for d in docs]
    scores = np.zeros((len(queries), len(docs)), dtype=np.float32)
    for qi, qf in enumerate(q_facets):
        for di, df in enumerate(d_facets):
            scores[qi, di] = facet_score(qf, df)
    return scores


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
    parser.add_argument("--tag", default="attribute_rerank_lora384")
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
    val_attr = facet_matrix(val_texts, make_docs(val_products, args.doc_mode))
    eval_attr = facet_matrix(eval_texts, make_docs(eval_products, args.doc_mode))

    candidates = [("image_lora384", val_base, eval_base)]
    for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0]:
        candidates.append(
            (
                f"rerank_top{args.topn}_{w:g}attribute",
                rerank_topn(val_base, val_attr, args.topn, w),
                rerank_topn(eval_base, eval_attr, args.topn, w),
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
    summary = {
        "tag": args.tag,
        "method": "LoRA-384 CLS first-stage + interpretable Korean fashion attribute-facet rerank",
        "tuning_policy": "weight selected only on train-split validation products; locked eval not used for selection",
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
    details = [
        {**q, "rank": selected_eval["ranks"][qi], "top10_pids": [eval_pids_gallery[i] for i in order[qi, :10].tolist()]}
        for qi, q in enumerate(eval_queries)
    ]
    (OUTPUT_DIR / f"eval_{args.tag}_detail.json").write_text(json.dumps(details, ensure_ascii=False, indent=2))
    print(json.dumps(summary["selected"], ensure_ascii=False, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
