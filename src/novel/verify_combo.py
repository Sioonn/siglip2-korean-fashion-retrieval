"""INDEPENDENT re-derivation / adversarial verification of combos.json claims.

Does NOT read combos.json. Recomputes from scratch using only lib.py primitives:
 (a) published current-best ranks/MRR from the detail JSON (rank field).
 (b) CSLS(k=20)+BM25(n50,w0.2) on lora384 for human-30 AND synth, its MRR,
     and paired sign tests vs published and vs LoRA cosine.
 (c) modality_gap(0.3)+CSLS(k10)+BM25(n50,w0.25) likewise.

CSLS and modality-gap math are re-implemented here independently (not imported
from run_combos.py).
"""
import json
from pathlib import Path

import numpy as np

from src.novel import lib

ROOT = Path(__file__).resolve().parent.parent.parent
ENC = "lora384"
PUB = ROOT / "output" / "eval_hybrid_rerank_top10_015bm25_detail.json"


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


def csls(Q, gal, bank, k):
    """CSLS: 2*cos - mean_topk(query-side over gallery) - mean_topk(gallery-side over bank)."""
    S = Q @ gal.T                       # (nq, G) query-to-gallery cosine
    rQ = np.sort(S, axis=1)[:, -k:].mean(1)              # per-query local mean
    rG = np.sort(bank @ gal.T, axis=0)[-k:, :].mean(0)   # per-gallery local mean over bank
    return 2.0 * S - rQ[:, None] - rG[None, :]


def modality_gap(lam, Q, gal, bank):
    """Subtract lam * mean text/image direction, then renormalize."""
    tm = bank.mean(0, keepdims=True)
    Qc = l2(Q - lam * tm)
    galc = l2(gal - lam * gal.mean(0, keepdims=True))
    Bc = l2(bank - lam * tm)
    return Qc, galc, Bc


def main():
    out = {}
    c = lib.load_caches()
    data = lib.load_data()
    synth = lib.load_synth_eval()
    G = c.gallery_emb
    egt, sgt = c.eval_gt, synth["gt"]
    bank = c.bank(ENC)

    # ---- BM25 score matrices (independent build) ----
    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(data.eval_texts)
    synth_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(synth["texts"])

    # ============ (a) published current-best ============
    pub_ranks = np.array([r["rank"] for r in json.loads(PUB.read_text())], dtype=np.int64)
    pub_m = lib.metrics_from_ranks(pub_ranks)
    out["a_published"] = {"n": int(len(pub_ranks)), "MRR": pub_m["MRR"],
                          "R@1": pub_m["R@1"], "R@5": pub_m["R@5"], "R@10": pub_m["R@10"],
                          "ranks": pub_ranks.tolist()}

    # ---- LoRA cosine references (no BM25, no hubness) ----
    lora_eval_ranks = lib.ranks_of(c.q(ENC) @ G.T, egt)
    lora_synth_ranks = lib.ranks_of(synth["enc"][ENC] @ G.T, sgt)
    out["lora_cosine"] = {
        "human30_MRR": lib.metrics_from_ranks(lora_eval_ranks)["MRR"],
        "synth_MRR": lib.metrics_from_ranks(lora_synth_ranks)["MRR"],
    }

    # ---- dense+BM25 repro reference on synth (val-selected n,w like the report) ----
    val_bm25 = lib.BM25(lib.make_docs(data.val_products, "all")).score_many(data.val_texts)
    Gv = c.val_gallery_emb
    vb = c.vq(ENC) @ Gv.T
    BM_N = [20, 50]
    BM_W = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    best, bv = None, -1.0
    for n in BM_N:
        for w in BM_W:
            m = lib.metrics_at_k(lib.rerank_topn(vb, val_bm25, n, w), c.val_gt)["MRR"]
            if m > bv:
                bv, best = m, (n, w)
    n0, w0 = best
    dense_bm25_synth_ranks = lib.ranks_of(lib.rerank_topn(synth["enc"][ENC] @ G.T, synth_bm25, n0, w0), sgt)
    dense_bm25_eval_ranks = lib.ranks_of(lib.rerank_topn(c.q(ENC) @ G.T, eval_bm25, n0, w0), egt)
    out["dense_bm25_repro"] = {
        "val_selected": f"n{n0}_w{w0}",
        "human30_MRR": lib.metrics_from_ranks(dense_bm25_eval_ranks)["MRR"],
        "synth_MRR": lib.metrics_from_ranks(dense_bm25_synth_ranks)["MRR"],
    }

    # ============ (b) CSLS(k=20)+BM25(n50,w0.2) ============
    Se = csls(c.q(ENC), G, bank, 20)
    Se = lib.rerank_topn(Se, eval_bm25, 50, 0.20)
    b_eval_ranks = lib.ranks_of(Se, egt)
    Ss = csls(synth["enc"][ENC], G, bank, 20)
    Ss = lib.rerank_topn(Ss, synth_bm25, 50, 0.20)
    b_synth_ranks = lib.ranks_of(Ss, sgt)
    out["b_csls20_bm25_n50_w0.2"] = {
        "human30_MRR": lib.metrics_from_ranks(b_eval_ranks)["MRR"],
        "human30_R@1": lib.metrics_from_ranks(b_eval_ranks)["R@1"],
        "human30_R@10": lib.metrics_from_ranks(b_eval_ranks)["R@10"],
        "human30_ci": list(lib.bootstrap_ci(b_eval_ranks, "MRR")),
        "synth_MRR": lib.metrics_from_ranks(b_synth_ranks)["MRR"],
        "synth_R@1": lib.metrics_from_ranks(b_synth_ranks)["R@1"],
        "synth_ci": list(lib.bootstrap_ci(b_synth_ranks, "MRR")),
        "vs_published_human30": lib.paired_compare(b_eval_ranks, pub_ranks, "MRR"),
        "vs_loracos_human30": lib.paired_compare(b_eval_ranks, lora_eval_ranks, "MRR"),
        "vs_loracos_synth": lib.paired_compare(b_synth_ranks, lora_synth_ranks, "MRR"),
        "vs_densebm25_synth": lib.paired_compare(b_synth_ranks, dense_bm25_synth_ranks, "MRR"),
        "human30_ranks": b_eval_ranks.tolist(),
    }

    # ============ (c) modality_gap(0.3)+CSLS(k10)+BM25(n50,w0.25) ============
    Qe, gale, Bce = modality_gap(0.3, c.q(ENC), G, bank)
    Se2 = csls(Qe, gale, Bce, 10)
    Se2 = lib.rerank_topn(Se2, eval_bm25, 50, 0.25)
    c_eval_ranks = lib.ranks_of(Se2, egt)
    Qs, gals, Bcs = modality_gap(0.3, synth["enc"][ENC], G, bank)
    Ss2 = csls(Qs, gals, Bcs, 10)
    Ss2 = lib.rerank_topn(Ss2, synth_bm25, 50, 0.25)
    c_synth_ranks = lib.ranks_of(Ss2, sgt)
    out["c_mgap0.3_csls10_bm25_n50_w0.25"] = {
        "human30_MRR": lib.metrics_from_ranks(c_eval_ranks)["MRR"],
        "human30_R@1": lib.metrics_from_ranks(c_eval_ranks)["R@1"],
        "human30_R@10": lib.metrics_from_ranks(c_eval_ranks)["R@10"],
        "human30_ci": list(lib.bootstrap_ci(c_eval_ranks, "MRR")),
        "synth_MRR": lib.metrics_from_ranks(c_synth_ranks)["MRR"],
        "synth_R@1": lib.metrics_from_ranks(c_synth_ranks)["R@1"],
        "synth_ci": list(lib.bootstrap_ci(c_synth_ranks, "MRR")),
        "vs_published_human30": lib.paired_compare(c_eval_ranks, pub_ranks, "MRR"),
        "vs_loracos_human30": lib.paired_compare(c_eval_ranks, lora_eval_ranks, "MRR"),
        "vs_densebm25_synth": lib.paired_compare(c_synth_ranks, dense_bm25_synth_ranks, "MRR"),
        "human30_ranks": c_eval_ranks.tolist(),
    }

    (ROOT / "output" / "novel" / "verify_combo.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
