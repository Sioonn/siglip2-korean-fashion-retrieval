"""Generate the two figures the paper needs (no sklearn; PCA via numpy SVD).
Fig 1: distribution/modality gap (PCA of images vs formal captions vs user-style queries).
Fig 2: forest plot of the 4-step ladder with bootstrap 95% CIs across 3 eval protocols.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.novel import lib  # noqa

FIG = ROOT / "docs" / "novel_research" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def pca2(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def fig_gap():
    c = lib.load_caches()
    imgs = c.gallery_emb                                  # base frozen vision (588)
    caps = np.load(lib.cache_path("base", "gallery_doc")) # formal captions (588)
    bank = np.load(lib.cache_path("base", "bank_q"))      # synthetic user-style queries
    evq = np.load(lib.cache_path("base", "eval_q"))       # human queries (30)
    rng = np.random.default_rng(0)
    bank_s = bank[rng.choice(len(bank), size=min(600, len(bank)), replace=False)]
    groups = [("Product images", imgs, "#1f77b4", 8, 0.35),
              ("Formal captions (index)", caps, "#ff7f0e", 8, 0.35),
              ("User-style queries (synthetic)", bank_s, "#2ca02c", 8, 0.35),
              ("Human eval queries", evq, "#d62728", 36, 0.95)]
    allX = np.concatenate([g[1] for g in groups], 0)
    P = pca2(allX)
    i = 0
    plt.figure(figsize=(6.2, 5.2))
    for name, arr, col, sz, al in groups:
        n = len(arr)
        plt.scatter(P[i:i+n, 0], P[i:i+n, 1], s=sz, c=col, alpha=al, label=name, edgecolors="none")
        i += n
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Figure 1. Caption-query distribution gap & modality gap\n(zero-shot SigLIP 2 embedding space, PCA)")
    plt.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(FIG / "fig1_distribution_gap.png", dpi=160)
    plt.close()
    print("saved fig1_distribution_gap.png")


def _mrr_ci(rows, enc, method, mode, which):
    for r in rows:
        if r["enc"] == enc and r["method"] == method and r["mode"] == mode:
            if which == "eval":
                return r["eval_MRR"], r["eval_MRR_ci"]
            return r.get("synth_MRR"), r.get("synth_MRR_ci")
    return None, None


def fig_forest():
    fs = json.load(open(ROOT / "output" / "novel" / "final_suite.json"))
    mp = json.load(open(ROOT / "output" / "novel" / "multipos.json"))
    rows = fs["rows"]
    steps = [
        ("Zero-shot", ("base", "identity", "standalone"), "baseline"),
        ("+Text-LoRA", ("lora384", "identity", "standalone"), "lora"),
        ("+BM25 (hybrid)", ("lora384", "dense_bm25", "+bm25"), "dense_bm25"),
        ("+Bank-prior (mgap, ours)", ("lora384", "modality_gap", "+bm25"), "modality_gap_bm25"),
    ]
    protocols = [("Gold (N=30)", "eval", "#d62728"),
                 ("Multi-positive (N=30)", "multipos", "#2ca02c"),
                 ("High-power (N=1706)", "synth", "#1f77b4")]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ylabels, ypos = [], []
    y = 0
    gap = 0.22
    for si, (sname, key, mpkey) in enumerate(steps):
        for pi, (pname, proto, col) in enumerate(protocols):
            if proto == "multipos":
                m = mp["methods"][mpkey]["MRR"]; lo, hi = mp["methods"][mpkey]["MRR_ci"]
            else:
                m, ci = _mrr_ci(rows, key[0], key[1], key[2], proto)
                lo, hi = ci
            yy = y + (pi - 1) * gap
            ax.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt="o", color=col, capsize=3,
                        ms=5, label=pname if si == 0 else None)
        ylabels.append(sname); ypos.append(y)
        y += 1.2
    ax.set_yticks(ypos); ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("MRR (point = estimate, bar = bootstrap 95% CI)")
    ax.set_title("Figure 2. 4-step ladder x 3 eval protocols (MRR +/- 95% CI)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.grid(axis="x", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG / "fig2_forest.png", dpi=160)
    plt.close()
    print("saved fig2_forest.png")


if __name__ == "__main__":
    fig_gap()
    fig_forest()
