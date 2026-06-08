"""CSLS (Cross-domain Similarity Local Scaling, Conneau ICLR2018) cross-modal hubness fix.

Idea: hub galleries inflate cosine similarity for many queries. CSLS penalizes each
gallery by its local density (mean top-k similarity to BANK queries) and each query by
its own top-k gallery radius:
    CSLS(q,g) = 2*S(q,g) - r_Q(q) - r_G(g)
r_G is estimated from the synthetic query BANK (leak-free); r_Q uses only the query's
own top-k neighbours (allowed). Pure-dense and on-BM25 variants.
NAME=csls.
"""
import json, os
import numpy as np
from src.novel import lib

NAME = "csls"


def load_refs():
    """Load paired-test reference per-query eval ranks from refs.json.

    lib has no load_refs(); refs are persisted by src/novel/refs.py to
    data/novel_cache/refs.json as {name: list[rank]}. Return np.int64 arrays.
    """
    raw = json.loads((lib.CACHE_DIR / "refs.json").read_text())
    return {name: np.array(ranks, dtype=np.int64) for name, ranks in raw.items()}


def topk_mean(scores, k, axis):
    """Mean of the top-k values along `axis`."""
    k = min(k, scores.shape[axis])
    # partition for the k largest along axis
    part = np.partition(scores, -k, axis=axis)
    if axis == 0:
        topk = part[-k:, :]
    else:
        topk = part[:, -k:]
    return topk.mean(axis=axis)


def main():
    c = lib.load_caches()
    refs = load_refs()
    data = lib.load_data()

    # BM25 (lexical) scores for fusion variants
    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(data.eval_texts)
    val_bm25 = lib.BM25(lib.make_docs(data.val_products, "all")).score_many(data.val_texts)

    candidates = []

    for enc in ["base", "lora384"]:
        q = c.q(enc)          # (30,768)
        vq = c.vq(enc)        # (717,768)
        bank = c.bank(enc)    # (7023,768)

        # Raw cosine similarity matrices (all embeddings L2-normalized)
        S_eval = q @ c.gallery_emb.T               # (30,588)
        S_val = vq @ c.val_gallery_emb.T           # (717,240)

        # BANK vs gallery: leak-free gallery hubness source
        Sb_eval = bank @ c.gallery_emb.T           # (7023,588)
        Sb_val = bank @ c.val_gallery_emb.T        # (7023,240)

        # identity (raw dense) candidate
        candidates.append((f"{enc}_dense", S_val, S_eval))

        for k in [3, 5, 10, 15, 20]:
            # r_G: gallery hubness = mean top-k over BANK queries (axis=0) -> (n_gallery,)
            r_G_eval = topk_mean(Sb_eval, k, axis=0)   # (588,)
            r_G_val = topk_mean(Sb_val, k, axis=0)     # (240,)

            # r_Q: query radius = mean top-k galleries for each query (axis=1) -> (n_query,)
            r_Q_eval = topk_mean(S_eval, k, axis=1)    # (30,)
            r_Q_val = topk_mean(S_val, k, axis=1)      # (717,)

            CSLS_eval = 2 * S_eval - r_Q_eval[:, None] - r_G_eval[None, :]
            CSLS_val = 2 * S_val - r_Q_val[:, None] - r_G_val[None, :]

            candidates.append((f"{enc}_csls_k{k}", CSLS_val, CSLS_eval))

            # On-BM25: candidate-preserving rerank with CSLS as the dense reranker
            for n in [20, 50]:
                for w in [0.2, 0.5]:
                    ev = lib.rerank_topn(eval_bm25, CSLS_eval, n, w)
                    vv = lib.rerank_topn(val_bm25, CSLS_val, n, w)
                    candidates.append((f"{enc}_csls_k{k}_bm25_n{n}_w{w}", vv, ev))

    sel = lib.select_eval(candidates, c.val_gt, c.eval_gt, c.eval_diff, refs)

    os.makedirs("output/novel", exist_ok=True)
    out_path = f"output/novel/{NAME}.json"
    with open(out_path, "w") as f:
        json.dump(sel, f, indent=2, default=str)

    m = sel["eval"]["metrics"]
    ci = sel["eval"]["ci95"]["MRR"]
    sb = sel.get("paired_vs", {})
    print(f"EXIT=0 NAME={NAME} selected={sel['selected']} "
          f"evalMRR={m['MRR']:.3f} R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} R@10={m['R@10']:.3f} "
          f"ci=[{ci[0]:.3f},{ci[1]:.3f}]")
    for ref, st in sb.items():
        d = st["MRR"]["mean_delta"]; p = st["MRR"]["sign_test_p"]
        print(f"  vs {ref}: dMRR={d:+.3f} sign_p={p:.3f}")


if __name__ == "__main__":
    main()
