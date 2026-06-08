"""Build reference per-query ranks for paired comparisons, and verify the
harness reproduces the repo's strong text-only reference (LoRA384+BM25).

References saved to data/novel_cache/refs.json:
  - baseline           : frozen SigLIP2-384 cosine
  - lora384            : text-only LoRA cosine
  - lora384_bm25       : LoRA384 + candidate-preserving BM25 (reproduced here, val-selected)
  - lora384_bm25_pub   : same, loaded from repo's published detail json (cross-check)
  - dual_bm25_pub      : dual-tower r4 10ep + BM25 (repo's headline 'best'), from detail json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.novel import lib  # noqa: E402

OUT = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)


def ranks_from_detail(path):
    rows = json.loads(Path(path).read_text())
    return np.array([r["rank"] for r in rows], dtype=np.int64)


def main():
    data = lib.load_data()
    c = lib.load_caches()

    refs = {}

    # 1. baseline + lora384 cosine
    base_scores = c.q("base") @ c.gallery_emb.T
    base_ranks = lib.ranks_of(base_scores, c.eval_gt)
    refs["baseline"] = base_ranks
    lora_scores = c.q("lora384") @ c.gallery_emb.T
    lora_ranks = lib.ranks_of(lora_scores, c.eval_gt)
    refs["lora384"] = lora_ranks

    # 2. reproduce LoRA384 + BM25 (candidate-preserving), select on val
    eval_docs = lib.make_docs(data.gallery_products, "all")
    val_docs = lib.make_docs(data.val_products, "all")
    eval_bm25 = lib.BM25(eval_docs).score_many(data.eval_texts)
    val_bm25 = lib.BM25(val_docs).score_many(data.val_texts)
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
    log("LoRA384+BM25 reproduced:", res["selected"], json.dumps(res["eval"]["metrics"]))

    # 3. published references from detail jsons (cross-check / headline best)
    try:
        refs["lora384_bm25_pub"] = ranks_from_detail(ROOT / "output" / "eval_hybrid_rerank_top10_015bm25_detail.json")
    except Exception as e:
        log("pub lora bm25 load err", e)
    try:
        refs["dual_bm25_pub"] = ranks_from_detail(ROOT / "output" / "eval_dual_hybrid_rerank_r4_infonce10_detail.json")
    except Exception as e:
        log("pub dual load err", e)

    log("\n=== reference metrics (locked eval, N=30) ===")
    for name, r in refs.items():
        m = lib.metrics_from_ranks(r)
        ci = lib.bootstrap_ci(r, "MRR")
        log(f"  {name:20s} R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} R@10={m['R@10']:.3f} "
            f"MRR={m['MRR']:.3f}  (MRR 95%CI {ci[0]:.3f}-{ci[1]:.3f})")

    # paired: does LoRA384+BM25 actually beat frozen baseline? and dual vs lora384_bm25?
    log("\n=== paired tests ===")
    for a, b in [("lora384_bm25", "baseline"), ("lora384", "baseline"),
                 ("dual_bm25_pub", "lora384_bm25_pub")]:
        if a in refs and b in refs:
            for metric in ["MRR", "R@1"]:
                pc = lib.paired_compare(refs[a], refs[b], metric)
                log(f"  {a} vs {b} [{metric}]: Δ={pc['mean_delta']:+.4f} "
                    f"CI[{pc['ci95'][0]:+.4f},{pc['ci95'][1]:+.4f}] "
                    f"P(Δ>0)={pc['p_delta_gt0']:.2f} sign_p={pc['sign_test_p']:.3f} "
                    f"(W{pc['wins']}/L{pc['losses']}/T{pc['ties']})")

    (lib.CACHE_DIR / "refs.json").write_text(json.dumps(
        {k: v.tolist() for k, v in refs.items()}, ensure_ascii=False))
    log("\nsaved refs.json")


if __name__ == "__main__":
    main()
    Path("/root/.claude/jobs/d8ed9167/tmp/refs_out.txt").write_text("\n".join(OUT))
