"""QB-Norm / Dynamic Inverted Softmax (Bogolin et al., CVPR 2022).

Test-time hubness normalization for text->image retrieval.

Some gallery images are "hubs": they are the nearest neighbour of many queries
regardless of relevance, inflating their similarity. QB-Norm estimates each
gallery item's *popularity* (probability mass it attracts) from a held-out
probe set of TRAIN synthetic queries (the hubness bank) and down-weights popular
hubs at test time, either subtractively in log space (Dynamic Inverted Softmax,
DIS) or multiplicatively.

    P   = softmax(beta * Sb, axis=1)   per bank-query distribution over gallery
    pop = P.sum(0)                     gallery popularity (n_gallery,)
    DIS : Sn = S - log(pop + eps)
    MUL : Sn = S / (pop**alpha + eps)

Leakage: popularity is estimated ONLY from c.bank(enc) (train synthetic queries)
and the gallery. The 30 eval queries never enter any statistic. HP selection is
on validation only via lib.select_eval.

NAME=qbnorm_dis.
"""
import json
import os
from pathlib import Path

import numpy as np

from src.novel import lib

NAME = "qbnorm_dis"


def _softmax_rows(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _popularity(bank_vs_gallery, beta):
    """Gallery popularity from the hubness bank (train synthetic queries only)."""
    P = _softmax_rows(beta * bank_vs_gallery)
    return P.sum(0)  # (n_gallery,)


def _ranks_from_detail(path):
    rows = json.loads(Path(path).read_text())
    return np.array([r["rank"] for r in rows], dtype=np.int64)


def build_refs(c, data, eval_bm25, val_bm25):
    """Reference per-query eval ranks for paired tests (mirrors src/novel/refs.py).

    Used instead of lib.load_refs() because the cached refs.json is not present.
      baseline      : frozen SigLIP2-384 cosine
      lora384       : text-only LoRA cosine
      lora384_bm25  : LoRA384 + candidate-preserving BM25, val-selected (CURRENT BEST)
      lora384_bm25_pub / dual_bm25_pub : published detail jsons (if available)
    """
    refs = {}
    refs["baseline"] = lib.ranks_of(c.q("base") @ c.gallery_emb.T, c.eval_gt)
    refs["lora384"] = lib.ranks_of(c.q("lora384") @ c.gallery_emb.T, c.eval_gt)

    # reproduce LoRA384 + BM25 (candidate-preserving), select on validation
    val_img = c.vq("lora384") @ c.val_gallery_emb.T
    eval_img = c.q("lora384") @ c.gallery_emb.T
    cands = [("image_only", val_img, eval_img)]
    for n in [10, 20, 50]:
        for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.50, 0.75, 1.0]:
            cands.append((f"rerank_top{n}_{w:g}bm25",
                          lib.rerank_topn(val_img, val_bm25, n, w),
                          lib.rerank_topn(eval_img, eval_bm25, n, w)))
    res = lib.select_eval(cands, c.val_gt, c.eval_gt, c.eval_diff)
    refs["lora384_bm25"] = np.array(res["eval_ranks"], dtype=np.int64)

    for name, fname in [
        ("lora384_bm25_pub", "eval_hybrid_rerank_top10_015bm25_detail.json"),
        ("dual_bm25_pub", "eval_dual_hybrid_rerank_r4_infonce10_detail.json"),
    ]:
        p = lib.ROOT / "output" / fname
        if p.exists():
            try:
                refs[name] = _ranks_from_detail(p)
            except Exception:
                pass
    return refs


def main():
    c = lib.load_caches()
    data = lib.load_data()

    # BM25 lexical scores (depend only on text); for the on-BM25 fusion set.
    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(
        data.eval_texts
    )
    val_bm25 = lib.BM25(lib.make_docs(data.val_products, "all")).score_many(
        data.val_texts
    )

    refs = build_refs(c, data, eval_bm25, val_bm25)

    BETAS = (1, 5, 10, 15, 20, 30)
    ALPHAS = (0.3, 0.5, 0.7, 1.0)
    RERANK_N = (20, 50)
    RERANK_W = (0.2, 0.5)

    candidates = []

    # ---- plain identity candidates (dense cosine, no correction) ----
    for enc in ("base", "lora384"):
        S = c.q(enc) @ c.gallery_emb.T
        Sv = c.vq(enc) @ c.val_gallery_emb.T
        candidates.append((f"identity_{enc}", Sv, S))

    # ---- hubness correction per encoder ----
    for enc in ("base", "lora384"):
        S = c.q(enc) @ c.gallery_emb.T            # (30, 588)
        Sb = c.bank(enc) @ c.gallery_emb.T        # (7023, 588) train synth bank
        Sv = c.vq(enc) @ c.val_gallery_emb.T      # (717, 240)
        Svb = c.bank(enc) @ c.val_gallery_emb.T   # (7023, 240) train synth bank

        for beta in BETAS:
            pop_e = _popularity(Sb, beta)         # (588,)
            pop_v = _popularity(Svb, beta)        # (240,)

            # ---- DIS: subtractive in log space ----
            Sn_e_dis = S - np.log(pop_e[None, :] + 1e-8)
            Sn_v_dis = Sv - np.log(pop_v[None, :] + 1e-8)
            candidates.append((f"{enc}_DISlog_b{beta}", Sn_v_dis, Sn_e_dis))

            # ---- MUL: multiplicative down-weight ----
            for alpha in ALPHAS:
                Sn_e_mul = S / (pop_e[None, :] ** alpha + 1e-8)
                Sn_v_mul = Sv / (pop_v[None, :] ** alpha + 1e-8)
                candidates.append(
                    (f"{enc}_MULa{alpha}_b{beta}", Sn_v_mul, Sn_e_mul)
                )

            # ---- on-BM25: hubness-corrected dense + candidate-preserving BM25
            #      rerank. Base = corrected dense, rerank = BM25 (per spec). ----
            Sn_e_mul05 = S / (pop_e[None, :] ** 0.5 + 1e-8)
            Sn_v_mul05 = Sv / (pop_v[None, :] ** 0.5 + 1e-8)
            for n in RERANK_N:
                for w in RERANK_W:
                    candidates.append((
                        f"{enc}_DISlog_b{beta}_bm25n{n}w{w}",
                        lib.rerank_topn(Sn_v_dis, val_bm25, n, w),
                        lib.rerank_topn(Sn_e_dis, eval_bm25, n, w),
                    ))
                    candidates.append((
                        f"{enc}_MULa0.5_b{beta}_bm25n{n}w{w}",
                        lib.rerank_topn(Sn_v_mul05, val_bm25, n, w),
                        lib.rerank_topn(Sn_e_mul05, eval_bm25, n, w),
                    ))

    sel = lib.select_eval(candidates, c.val_gt, c.eval_gt, c.eval_diff, refs)

    os.makedirs(str(lib.OUTPUT_DIR), exist_ok=True)
    out_path = lib.OUTPUT_DIR / f"{NAME}.json"
    with open(out_path, "w") as f:
        json.dump(sel, f, indent=2, default=float)

    m = sel["eval"]["metrics"]
    print(
        f"EXIT=0 NAME={NAME} selected={sel['selected']} ncand={len(candidates)} "
        f"evalMRR={m['MRR']:.3f} R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} "
        f"R@10={m['R@10']:.3f}"
    )
    for ref, st in sel.get("paired_vs", {}).items():
        d = st["MRR"]["mean_delta"]
        p = st["MRR"]["sign_test_p"]
        print(f"  vs {ref}: dMRR={d:+.3f} sign_p={p:.3f}")


if __name__ == "__main__":
    main()
