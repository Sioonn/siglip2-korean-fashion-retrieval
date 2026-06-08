"""Novel method: Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).

A principled, score-scale-free alternative to the repo's z-score LINEAR
fusion of dense + BM25. RRF combines component rankings via sum of
1/(K + rank) and is robust to the very different score distributions of
cosine-similarity dense retrieval and BM25 lexical scoring.

Components (built per encoder, leakage-safe):
  - dense       : q @ gallery.T (cosine, L2-normalized embeddings)
  - bm25        : lexical BM25 over gallery product text
  - mgap_dense  : modality-gap-corrected dense (subtract lam*(mu_g - mu_q),
                  renormalize). Means estimated from the TRAIN bank + gallery.
  - csls_dense  : CSLS-corrected dense (k=10). Gallery-side neighbour radius
                  r_g estimated from the TRAIN bank vs gallery (leakage-safe);
                  query-side radius r_q is a per-query transform.

Leakage rules: every statistic (bank means, gallery means, CSLS gallery
radius) comes from the synthetic-query BANK (train) and the gallery only.
The 30 eval queries are NEVER used to estimate anything. HP/subset/encoder
selection is done by select_eval on VALIDATION only.
"""
import json
import os
from itertools import combinations

import numpy as np

from src.novel import lib

NAME = "rrf_fusion"

MGAP_LAM = 0.4
CSLS_K = 10


def mgap_correct(q, mu_q, mu_g, lam):
    """Modality-gap correction: shift queries toward gallery cone, renorm."""
    gap = mu_g - mu_q
    return lib.normalize_rows(q + lam * gap)


def csls_scores(q, gal, r_g):
    """CSLS-corrected similarity.

    csls(x, y) = 2*cos(x, y) - r_q(x) - r_g(y)
    r_q(x): mean cosine of query x to its CSLS_K nearest gallery items
            (per-query transform, computed from the query itself).
    r_g(y): mean cosine of gallery item y to its CSLS_K nearest *bank*
            queries (estimated leakage-safe from TRAIN bank + gallery).
    """
    sim = q @ gal.T
    # query-side radius: top-k over gallery, per query row
    k = min(CSLS_K, sim.shape[1])
    part = np.partition(sim, -k, axis=1)[:, -k:]
    r_q = part.mean(axis=1, keepdims=True)  # (nq, 1)
    return 2.0 * sim - r_q - r_g[None, :]


def gallery_radius_from_bank(bank, gal):
    """r_g(y) = mean cosine of gallery item y to its CSLS_K nearest bank
    (train synthetic) queries. Leakage-safe: uses bank + gallery only."""
    sim = gal @ bank.T  # (ngal, nbank)
    k = min(CSLS_K, sim.shape[1])
    part = np.partition(sim, -k, axis=1)[:, -k:]
    return part.mean(axis=1)  # (ngal,)


def main():
    c = lib.load_caches()
    refs = lib.load_refs()
    data = lib.load_data()

    # BM25 lexical scores (shared across encoders; depends only on text).
    # eval uses the 588-item gallery; val uses the 240-item val gallery.
    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(
        data.eval_texts
    )
    val_bm25 = lib.BM25(lib.make_docs(data.val_products, "all")).score_many(
        data.val_texts
    )

    gal = c.gallery_emb
    val_gal = c.val_gallery_emb
    mu_g_eval = gal.mean(axis=0, keepdims=True)
    mu_g_val = val_gal.mean(axis=0, keepdims=True)

    cands = []
    K_GRID = (10, 20, 40, 60, 80)

    for enc in ("base", "lora384"):
        q = c.q(enc)
        vq = c.vq(enc)
        bank = c.bank(enc)
        mu_q = bank.mean(axis=0, keepdims=True)  # train-query mean (leakage-safe)

        # ---- component score matrices (eval + val) ----
        comp_eval = {}
        comp_val = {}

        comp_eval["dense"] = q @ gal.T
        comp_val["dense"] = vq @ val_gal.T

        comp_eval["bm25"] = eval_bm25
        comp_val["bm25"] = val_bm25

        comp_eval["mgap_dense"] = mgap_correct(q, mu_q, mu_g_eval, MGAP_LAM) @ gal.T
        comp_val["mgap_dense"] = (
            mgap_correct(vq, mu_q, mu_g_val, MGAP_LAM) @ val_gal.T
        )

        # CSLS gallery radius estimated from TRAIN bank + gallery (leakage-safe).
        r_g_eval = gallery_radius_from_bank(bank, gal)
        r_g_val = gallery_radius_from_bank(bank, val_gal)
        comp_eval["csls_dense"] = csls_scores(q, gal, r_g_eval)
        comp_val["csls_dense"] = csls_scores(vq, val_gal, r_g_val)

        # ---- single-component references (ranking == raw score) ----
        cands.append((f"dense_{enc}", comp_val["dense"], comp_eval["dense"]))
        cands.append((f"bm25_{enc}", comp_val["bm25"], comp_eval["bm25"]))

        # ---- RRF over subsets of components ----
        comp_names = ["dense", "bm25", "mgap_dense", "csls_dense"]
        subsets = []
        for r in range(2, len(comp_names) + 1):
            for combo in combinations(comp_names, r):
                subsets.append(combo)
        # ensure the canonical 2-way dense+bm25 is present (it is, via r=2)

        for K in K_GRID:
            for combo in subsets:
                tag = "+".join(combo)
                fused_eval = lib.rrf([comp_eval[n] for n in combo], K)
                fused_val = lib.rrf([comp_val[n] for n in combo], K)
                cands.append((f"rrf_K{K}_{tag}_{enc}", fused_val, fused_eval))

    sel = lib.select_eval(cands, c.val_gt, c.eval_gt, c.eval_diff, refs)

    os.makedirs("output/novel", exist_ok=True)
    out = f"output/novel/{NAME}.json"
    with open(out, "w") as f:
        json.dump(sel, f, indent=2, default=lib.to_jsonable)

    m = sel["eval"]["metrics"]
    bestp = sel["selected"]
    print(
        f"EXIT=0 {NAME} selected={bestp} evalMRR={m['MRR']:.3f} "
        f"R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} R@10={m['R@10']:.3f}"
    )


if __name__ == "__main__":
    main()
