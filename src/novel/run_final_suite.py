"""Authoritative, self-contained re-benchmark of the novel test-time methods.

Refactored so each correction is `apply(method, enc, hp, Q, gallery)` and is
applied identically to (1) validation [HP selection], (2) the 30 human queries,
and (3) a high-power synthetic eval on UNSEEN test products (if generated).

Fixes the WF2 integrity issues: anchors current-best to the PUBLISHED pipeline
(rerank_top10_0.15bm25, MRR 0.756) loaded from its detail json; uses the proper
small-weight BM25 grid (w from 0.05); separates standalone (no-metadata) effect
from the +BM25 lever; bootstrap CIs + paired sign tests on BOTH eval sets. All
HP selection on validation only.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from scipy.special import logsumexp, softmax  # noqa: E402
from src.novel import lib  # noqa: E402

ENCS = ["base", "lora384"]
BM_N = [10, 20, 50]
BM_W = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
ROWS = []


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


def ranks_from_detail(path):
    return np.array([r["rank"] for r in json.loads(Path(path).read_text())], dtype=np.int64)


def main():
    c = lib.load_caches()
    data = lib.load_data()
    synth = lib.load_synth_eval()
    G, Gv = c.gallery_emb, c.val_gallery_emb
    egt, vgt, ediff = c.eval_gt, c.val_gt, c.eval_diff
    print(f"synth eval: {'N='+str(synth['n']) if synth else 'NOT PRESENT (human-30 only)'}")

    # BM25 score matrices
    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(data.eval_texts)
    val_bm25 = lib.BM25(lib.make_docs(data.val_products, "all")).score_many(data.val_texts)
    synth_bm25 = (lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(synth["texts"])
                  if synth else None)

    METHODS = {
        "modality_gap": [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)],
        "qbnorm_dis": [1, 2, 5, 10, 20, 50],
        "qbnorm_mul": [(b, a) for b in [1, 5, 10, 20] for a in [0.3, 0.5, 0.7, 1.0]],
        "csls": [3, 5, 10, 15, 20],
        "aqe": [(k, lam, al) for k in [3, 5, 10] for lam in [0.2, 0.3, 0.5] for al in [0.0, 1.0]],
    }

    def apply(method, enc, hp, Q, gal):
        B = c.bank(enc)
        if method == "identity":
            return Q @ gal.T
        if method == "modality_gap":
            lam = hp
            tm = B.mean(0, keepdims=True)
            return l2(Q - lam * tm) @ l2(gal - lam * gal.mean(0, keepdims=True)).T
        if method == "qbnorm_dis":
            beta = hp
            A = B @ gal.T
            return beta * (Q @ gal.T) - logsumexp(beta * A, axis=0)[None, :]
        if method == "qbnorm_mul":
            beta, alpha = hp
            pop = softmax(beta * (B @ gal.T), axis=1).sum(0)
            return (Q @ gal.T) / (pop[None, :] ** alpha + 1e-8)
        if method == "csls":
            k = hp
            rG = np.sort(B @ gal.T, axis=0)[-k:].mean(0)
            S = Q @ gal.T
            rQ = np.sort(S, axis=1)[:, -k:].mean(1)
            return 2 * S - rQ[:, None] - rG[None, :]
        if method == "aqe":
            k, lam, al = hp
            sims = Q @ B.T
            out = np.empty_like(Q)
            for i in range(Q.shape[0]):
                s = sims[i].copy()
                s[s > 0.999] = -1.0
                nn = np.argsort(-s)[:k]
                w = np.clip(s[nn], 0, None) ** al
                out[i] = Q[i] if w.sum() < 1e-8 else l2(Q[i] + lam * (w[:, None] * B[nn]).sum(0) / w.sum())
            return out @ gal.T
        raise ValueError(method)

    # ---- references: baseline / lora384 / published bm25 / proper-grid bm25 repro ----
    def ranks_eval(Q):
        return lib.ranks_of(Q, egt)

    refs_eval = {"baseline": ranks_eval(apply("identity", "base", None, c.q("base"), G)),
                 "lora384": ranks_eval(apply("identity", "lora384", None, c.q("lora384"), G))}
    refs_synth = None
    if synth:
        refs_synth = {"baseline": lib.ranks_of(apply("identity", "base", None, synth["enc"]["base"], G), synth["gt"]),
                      "lora384": lib.ranks_of(apply("identity", "lora384", None, synth["enc"]["lora384"], G), synth["gt"])}
    try:
        refs_eval["lora384_bm25_pub"] = ranks_from_detail(ROOT / "output" / "eval_hybrid_rerank_top10_015bm25_detail.json")
    except Exception as e:
        print("pub load err", e)

    # proper-grid dense+BM25 repro on lora384 (current-best, in-harness)
    vd, ed = c.q("lora384") @ G.T, None
    vbase = c.vq("lora384") @ Gv.T
    ebase = c.q("lora384") @ G.T
    best, bestval = None, -1
    for n in BM_N:
        for w in BM_W:
            vm = lib.metrics_at_k(lib.rerank_topn(vbase, val_bm25, n, w), vgt)["MRR"]
            if vm > bestval:
                bestval, best = vm, (n, w)
    n0, w0 = best
    refs_eval["lora384_bm25_repro"] = ranks_eval(lib.rerank_topn(ebase, eval_bm25, n0, w0))
    if synth:
        sbase = synth["enc"]["lora384"] @ G.T
        refs_synth["lora384_bm25_repro"] = lib.ranks_of(lib.rerank_topn(sbase, synth_bm25, n0, w0), synth["gt"])
    print(f"dense+BM25 repro selected n{n0} w{w0}: eval MRR={lib.metrics_from_ranks(refs_eval['lora384_bm25_repro'])['MRR']:.4f}")

    REF_KEYS = ["lora384", "lora384_bm25_repro"]

    def emit(method, mode, enc, sel, eval_ranks, synth_ranks):
        em = lib.metrics_from_ranks(eval_ranks)
        eci = lib.bootstrap_ci(eval_ranks, "MRR")
        row = {"method": method, "mode": mode, "enc": enc, "selected": sel,
               "eval_MRR": round(em["MRR"], 4), "eval_R1": em["R@1"], "eval_R5": em["R@5"], "eval_R10": em["R@10"],
               "eval_MRR_ci": [round(eci[0], 3), round(eci[1], 3)], "eval_ranks": eval_ranks.tolist()}
        for rk in REF_KEYS + (["lora384_bm25_pub"] if "lora384_bm25_pub" in refs_eval else []):
            p = lib.paired_compare(eval_ranks, refs_eval[rk], "MRR")
            row[f"eval_vs_{rk}"] = [round(p["mean_delta"], 4), round(p["sign_test_p"], 3), p["wins"], p["losses"]]
        if synth_ranks is not None:
            sm = lib.metrics_from_ranks(synth_ranks)
            sci = lib.bootstrap_ci(synth_ranks, "MRR")
            row["synth_MRR"] = round(sm["MRR"], 4)
            row["synth_R1"] = round(sm["R@1"], 3)
            row["synth_R10"] = round(sm["R@10"], 3)
            row["synth_MRR_ci"] = [round(sci[0], 3), round(sci[1], 3)]
            for rk in REF_KEYS:
                p = lib.paired_compare(synth_ranks, refs_synth[rk], "MRR")
                row[f"synth_vs_{rk}"] = [round(p["mean_delta"], 4), round(p["sign_test_p"], 4), p["wins"], p["losses"]]
        ROWS.append(row)
        sx = (f" || synthMRR={row.get('synth_MRR')} ci{row.get('synth_MRR_ci')} "
              f"vsLoRA={row.get('synth_vs_lora384')} vsBM25={row.get('synth_vs_lora384_bm25_repro')}") if synth_ranks is not None else ""
        print(f"ROW {method:13s} {mode:10s} {enc:8s} sel={str(sel)[:26]:26s} "
              f"evalMRR={row['eval_MRR']:.3f} ci{row['eval_MRR_ci']} vsLoRA={row['eval_vs_lora384']} "
              f"vsBM25repro={row['eval_vs_lora384_bm25_repro']}{sx}")

    def best_hp_on_val(method, enc, transform=None):
        bh, bv = None, -1
        for hp in METHODS[method]:
            sc = apply(method, enc, hp, c.vq(enc), Gv)
            if transform:
                sc = transform(sc, "val")
            vm = lib.metrics_at_k(sc, vgt)["MRR"]
            if vm > bv:
                bv, bh = vm, hp
        return bh

    # identity rows (per encoder)
    for enc in ENCS:
        er = ranks_eval(apply("identity", enc, None, c.q(enc), G))
        sr = lib.ranks_of(apply("identity", enc, None, synth["enc"][enc], G), synth["gt"]) if synth else None
        emit("identity", "standalone", enc, "cosine", er, sr)

    # standalone corrections
    for enc in ENCS:
        for method in METHODS:
            hp = best_hp_on_val(method, enc)
            er = ranks_eval(apply(method, enc, hp, c.q(enc), G))
            sr = lib.ranks_of(apply(method, enc, hp, synth["enc"][enc], G), synth["gt"]) if synth else None
            emit(method, "standalone", enc, hp, er, sr)

    # dense + BM25 (current-best repro) per encoder
    for enc in ENCS:
        vb, eb = c.vq(enc) @ Gv.T, c.q(enc) @ G.T
        bh, bvv = None, -1
        for n in BM_N:
            for w in BM_W:
                vm = lib.metrics_at_k(lib.rerank_topn(vb, val_bm25, n, w), vgt)["MRR"]
                if vm > bvv:
                    bvv, bh = vm, (n, w)
        n1, w1 = bh
        er = ranks_eval(lib.rerank_topn(eb, eval_bm25, n1, w1))
        sr = lib.ranks_of(lib.rerank_topn(synth["enc"][enc] @ G.T, synth_bm25, n1, w1), synth["gt"]) if synth else None
        emit("dense_bm25", "+bm25", enc, f"n{n1}_w{w1:g}", er, sr)

    # correction + BM25 (fix method HP to standalone val-best, then grid BM25 on val)
    for enc in ENCS:
        for method in METHODS:
            hp = best_hp_on_val(method, enc)
            vdn = apply(method, enc, hp, c.vq(enc), Gv)
            edn = apply(method, enc, hp, c.q(enc), G)
            sdn = apply(method, enc, hp, synth["enc"][enc], G) if synth else None
            bh, bvv = None, -1
            for n in BM_N:
                for w in BM_W:
                    vm = lib.metrics_at_k(lib.rerank_topn(vdn, val_bm25, n, w), vgt)["MRR"]
                    if vm > bvv:
                        bvv, bh = vm, (n, w)
            n1, w1 = bh
            er = ranks_eval(lib.rerank_topn(edn, eval_bm25, n1, w1))
            sr = lib.ranks_of(lib.rerank_topn(sdn, synth_bm25, n1, w1), synth["gt"]) if synth else None
            emit(method, "+bm25", enc, f"{hp}+n{n1}_w{w1:g}", er, sr)

    out = {"refs_eval": {k: round(lib.metrics_from_ranks(v)["MRR"], 4) for k, v in refs_eval.items()},
           "refs_synth": ({k: round(lib.metrics_from_ranks(v)["MRR"], 4) for k, v in refs_synth.items()} if refs_synth else None),
           "synth_n": synth["n"] if synth else 0,
           "rows": ROWS}
    (ROOT / "output" / "novel" / "final_suite.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("REFS_EVAL", out["refs_eval"], "REFS_SYNTH", out["refs_synth"])
    print("saved output/novel/final_suite.json")


if __name__ == "__main__":
    main()
