"""Combination search: can STACKED test-time corrections + BM25 beat the
current best (dense+BM25)?

Pipeline = [embedding correction: identity | modality-gap(lam)]
         -> dense score
         -> [score hubness: none | DIS(beta) | CSLS(k) | MUL(beta,alpha)]
         -> [candidate-preserving BM25 fusion: none | rerank_topn(n,w)]

All embedding corrections also applied to the BANK so hubness stats live in the
same corrected space. EVERY HP selected on VALIDATION only (717 synth q / 240
gallery). The single val-best pipeline is reported on human-30 AND high-power
synth-1706 with bootstrap CIs + paired sign tests vs (a) LoRA cosine and (b) the
current best. Top-K val pipelines also reported for multiple-comparison honesty.
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

ENC = "lora384"
BM_N = [20, 50]
BM_W = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
EMB = [("id", 0.0), ("mgap", 0.2), ("mgap", 0.3), ("mgap", 0.4), ("mgap", 0.5)]
HUB = [("none", None), ("dis", 2), ("dis", 5), ("dis", 10),
       ("csls", 5), ("csls", 10), ("csls", 20),
       ("mul", (5, 0.5)), ("mul", (10, 0.5)), ("mul", (10, 1.0))]


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


def ranks_from_detail(path):
    return np.array([r["rank"] for r in json.loads(Path(path).read_text())], dtype=np.int64)


def main():
    c = lib.load_caches()
    data = lib.load_data()
    synth = lib.load_synth_eval()
    G, Gv = c.gallery_emb, c.val_gallery_emb
    egt, vgt, sgt = c.eval_gt, c.val_gt, synth["gt"]
    B = c.bank(ENC)

    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(data.eval_texts)
    val_bm25 = lib.BM25(lib.make_docs(data.val_products, "all")).score_many(data.val_texts)
    synth_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(synth["texts"])

    def correct(kind, lam, Q, gal):
        if kind == "id":
            return Q, gal, B
        tm = B.mean(0, keepdims=True)
        return l2(Q - lam * tm), l2(gal - lam * gal.mean(0, keepdims=True)), l2(B - lam * tm)

    def hub(kind, hp, Q, gal, Bc):
        S = Q @ gal.T
        if kind == "none":
            return S
        if kind == "dis":
            return hp * S - logsumexp(hp * (Bc @ gal.T), axis=0)[None, :]
        if kind == "csls":
            k = hp
            rG = np.sort(Bc @ gal.T, axis=0)[-k:].mean(0)
            rQ = np.sort(S, axis=1)[:, -k:].mean(1)
            return 2 * S - rQ[:, None] - rG[None, :]
        if kind == "mul":
            beta, al = hp
            pop = softmax(beta * (Bc @ gal.T), axis=1).sum(0)
            return S / (pop[None, :] ** al + 1e-8)
        raise ValueError(kind)

    # ---- references ----
    refs_eval = {
        "lora384": lib.ranks_of(c.q(ENC) @ G.T, egt),
        "lora384_bm25_pub": ranks_from_detail(ROOT / "output" / "eval_hybrid_rerank_top10_015bm25_detail.json"),
    }
    refs_synth = {"lora384": lib.ranks_of(synth["enc"][ENC] @ G.T, sgt)}
    # dense+BM25 repro (val-selected), used as current-best ref on synth
    vb = c.vq(ENC) @ Gv.T
    best, bv = None, -1
    for n in BM_N:
        for w in BM_W:
            m = lib.metrics_at_k(lib.rerank_topn(vb, val_bm25, n, w), vgt)["MRR"]
            if m > bv:
                bv, best = m, (n, w)
    n0, w0 = best
    refs_eval["dense_bm25_repro"] = lib.ranks_of(lib.rerank_topn(c.q(ENC) @ G.T, eval_bm25, n0, w0), egt)
    refs_synth["dense_bm25_repro"] = lib.ranks_of(lib.rerank_topn(synth["enc"][ENC] @ G.T, synth_bm25, n0, w0), sgt)

    # ---- sweep: score val MRR for every pipeline; remember configs ----
    cands = []  # (cfg tuple, valMRR)
    for ek, lam in EMB:
        Qv, galv, Bv = correct(ek, lam, c.vq(ENC), Gv)
        for hk, hp in HUB:
            Sval = hub(hk, hp, Qv, galv, Bv)
            cands.append(((ek, lam, hk, hp, None, None), lib.metrics_at_k(Sval, vgt)["MRR"]))
            for n in BM_N:
                for w in BM_W:
                    rv = lib.rerank_topn(Sval, val_bm25, n, w)
                    cands.append(((ek, lam, hk, hp, n, w), lib.metrics_at_k(rv, vgt)["MRR"]))
    cands.sort(key=lambda x: -x[1])
    print(f"swept {len(cands)} pipelines; current-best dense+BM25 repro selected n{n0}_w{w0}")

    # ---- recompute eval+synth for the top-K val pipelines ----
    def eval_cfg(cfg, Q, gal, bm25, gt):
        ek, lam, hk, hp, n, w = cfg
        Qc, galc, Bc = correct(ek, lam, Q, gal)
        S = hub(hk, hp, Qc, galc, Bc)
        if n is not None:
            S = lib.rerank_topn(S, bm25, n, w)
        return lib.ranks_of(S, gt)

    topk = cands[:8]
    rows = []
    for cfg, vm in topk:
        er = eval_cfg(cfg, c.q(ENC), G, eval_bm25, egt)
        sr = eval_cfg(cfg, synth["enc"][ENC], G, synth_bm25, sgt)
        em, sm = lib.metrics_from_ranks(er), lib.metrics_from_ranks(sr)
        eci, sci = lib.bootstrap_ci(er, "MRR"), lib.bootstrap_ci(sr, "MRR")
        row = {
            "cfg": f"emb={cfg[0]}{cfg[1]}|hub={cfg[2]}{cfg[3]}|bm25={('n%sw%g'%(cfg[4],cfg[5])) if cfg[4] else 'none'}",
            "val_MRR": round(vm, 4),
            "eval_MRR": round(em["MRR"], 4), "eval_R1": em["R@1"], "eval_R10": em["R@10"], "eval_ci": [round(eci[0], 3), round(eci[1], 3)],
            "synth_MRR": round(sm["MRR"], 4), "synth_R1": round(sm["R@1"], 3), "synth_ci": [round(sci[0], 3), round(sci[1], 3)],
        }
        for nm, rk in refs_eval.items():
            p = lib.paired_compare(er, rk, "MRR")
            row[f"eval_vs_{nm}"] = [round(p["mean_delta"], 4), round(p["sign_test_p"], 3), p["wins"], p["losses"]]
        for nm, rk in refs_synth.items():
            p = lib.paired_compare(sr, rk, "MRR")
            row[f"synth_vs_{nm}"] = [round(p["mean_delta"], 4), round(p["sign_test_p"], 4), p["wins"], p["losses"]]
        rows.append(row)

    out = {
        "refs_eval": {k: round(lib.metrics_from_ranks(v)["MRR"], 4) for k, v in refs_eval.items()},
        "refs_synth": {k: round(lib.metrics_from_ranks(v)["MRR"], 4) for k, v in refs_synth.items()},
        "dense_bm25_selected": f"n{n0}_w{w0}",
        "val_best": rows[0],
        "topk": rows,
    }
    (ROOT / "output" / "novel" / "combos.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("REFS_EVAL", out["refs_eval"])
    print("REFS_SYNTH", out["refs_synth"])
    for r in rows:
        print(f"VAL{r['val_MRR']:.3f} {r['cfg'][:54]:54s} | eMRR={r['eval_MRR']:.3f}{r['eval_ci']} "
              f"vsPub={r.get('eval_vs_lora384_bm25_pub')} vsRepro={r.get('eval_vs_dense_bm25_repro')} "
              f"| sMRR={r['synth_MRR']:.3f} vsBM25={r.get('synth_vs_dense_bm25_repro')}")
    print("saved output/novel/combos.json")


if __name__ == "__main__":
    main()
