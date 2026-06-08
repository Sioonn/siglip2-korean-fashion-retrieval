"""Precompute & cache all text embeddings under each encoder, and validate
that the harness reproduces the repo's published baseline numbers.

Encoders:
  - base : frozen google/siglip2-base-patch16-384
  - lora384 : checkpoints/lora_cls_384 (text-only LoRA, the repo's hybrid base)

Caches: eval queries, validation queries, gallery docs (all 3 doc modes are
not needed here; "all" is enough), and the synthetic query bank.
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

ENCODERS = {
    "base": None,
    "lora384": str(ROOT / "checkpoints" / "lora_cls_384"),
}

OUT = []


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    OUT.append(msg)


def main():
    data = lib.load_data()
    log(f"gallery_size={len(data.gallery_pids)}  n_eval={len(data.eval_texts)}  "
        f"n_val_queries={len(data.val_texts)}  val_gallery={len(data.val_gallery_pids)}  "
        f"bank={len(data.bank_texts)}")

    docs = lib.make_docs(data.gallery_products, "all")
    val_docs = lib.make_docs(data.val_products, "all")

    for tag, lora in ENCODERS.items():
        log(f"\n=== encoder: {tag} ===")
        eval_q = lib.encode_cached(data.eval_texts, lora, tag, "eval_q")
        val_q = lib.encode_cached(data.val_texts, lora, tag, "val_q")
        bank = lib.encode_cached(data.bank_texts, lora, tag, "bank_q")
        doc = lib.encode_cached(docs, lora, tag, "gallery_doc")
        vdoc = lib.encode_cached(val_docs, lora, tag, "val_doc")
        log(f"shapes eval_q={eval_q.shape} val_q={val_q.shape} bank={bank.shape} "
            f"doc={doc.shape} vdoc={vdoc.shape}")

        # validate: plain cosine retrieval on eval
        scores = eval_q @ data.gallery_emb.T
        ranks = lib.ranks_of(scores, data.eval_gt)
        m = lib.metrics_from_ranks(ranks)
        ci = {k: lib.bootstrap_ci(ranks, k) for k in ["R@1", "R@10", "MRR"]}
        log(f"  COSINE eval metrics: {json.dumps(m)}")
        log(f"  CI95: R@1={ci['R@1']} R@10={ci['R@10']} MRR={ci['MRR']}")

    # also cache the frozen gallery image emb under a fixed name for methods
    np.save(lib.cache_path("shared", "gallery_emb"), data.gallery_emb)
    np.save(lib.cache_path("shared", "val_gallery_emb"), data.val_gallery_emb)
    log("\nsaved shared gallery emb caches")

    (lib.CACHE_DIR / "meta.json").write_text(json.dumps({
        "gallery_pids": data.gallery_pids,
        "eval_gt": data.eval_gt.tolist(),
        "eval_diff": data.eval_diff,
        "val_gt": data.val_gt.tolist(),
        "val_gallery_pids": data.val_gallery_pids,
    }, ensure_ascii=False))
    log("saved meta.json")


if __name__ == "__main__":
    main()
    Path("/root/.claude/jobs/d8ed9167/tmp/precompute_out.txt").write_text("\n".join(OUT))
