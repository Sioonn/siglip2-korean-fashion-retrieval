"""GME (General Multimodal Embedding) contemporary VLM-embedder baseline.

Model: Alibaba-NLP/gme-Qwen2-VL-2B-Instruct (2B, ~4.5GB). Influential 2024/25
multimodal embedder; we report it as a modern strong baseline against our cheap
SigLIP2 + text-LoRA + BM25 pipeline.

Encoder path (documented): we use the model's own custom inference class
`GmeQwen2VL` (vendored from the HF repo's gme_inference.py) which exposes
`get_text_embeddings(texts=[...])` and `get_image_embeddings(images=[...])`,
both L2-normalized last-token hidden states (Qwen2-VL backbone, fp16). This is
the primary path on the model card. No trust_remote_code shim needed because the
backbone is the standard Qwen2-VL (AutoModelForVision2Seq); the custom class only
wraps pooling/prompt formatting, which we vendor.

We encode:
  - 588 gallery IMAGES  (data/images/<pid>.jpg, c.gallery_pids order)
  - 240 val gallery IMAGES
  - query TEXTS: eval(30), val(717), synth(1706)
all L2-normalized, cached to data/novel_cache/gme__*.npy.

Scores: GME-dense (query @ gallery.T) and GME + Korean char-ngram BM25 (lib.BM25,
fusion weight/topn tuned on VAL ONLY via lib.rerank_topn, then frozen for eval &
synth). Reports human-30 (single-pos) and synth-1706 with MRR bootstrap CIs and a
paired sign test vs current-best LoRA+BM25 (refs 'lora384_bm25_pub' on human-30;
dense+BM25 recomputed on synth as the synth ref).

Run:  python -m src.novel.run_gme_baseline encode   # GPU, caches embeddings
      python -m src.novel.run_gme_baseline report   # CPU/numpy, writes json
      python -m src.novel.run_gme_baseline all       # both
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from src.novel import lib  # noqa: E402

GME_MODEL = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
TAG = "gme"
IMG_DIR = ROOT / "data" / "images"
OUT = ROOT / "output" / "novel" / "gme_baseline.json"

# BM25 fusion grid — same family as run_final_suite.py (small-weight grid)
BM_N = [10, 20, 50]
BM_W = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


# --------------------------------------------------------------------------
# encode (GPU)
# --------------------------------------------------------------------------
def _image_paths(pids: list[str]) -> list[str]:
    paths = []
    for pid in pids:
        p = IMG_DIR / f"{pid}.jpg"
        if not p.exists():
            raise FileNotFoundError(f"missing gallery image {p}")
        paths.append(str(p))
    return paths


def encode_all():
    import torch
    from PIL import Image
    from src.novel.gme_inference import GmeQwen2VL

    data = lib.load_data()
    synth = lib.load_synth_eval()
    assert synth is not None, "synth_eval not present"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading GME on {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    gme = GmeQwen2VL(GME_MODEL, device=device)

    def enc_text(texts, name):
        p = lib.cache_path(TAG, name)
        if p.exists() and np.load(p).shape[0] == len(texts):
            print(f"  cached {name}: {p}")
            return
        emb = gme.get_text_embeddings(texts=texts, batch_size=32, show_progress_bar=True)
        arr = l2(emb.float().cpu().numpy().astype(np.float32))
        np.save(p, arr)
        print(f"  saved {name}: {arr.shape} -> {p}")

    def enc_img(pids, name):
        p = lib.cache_path(TAG, name)
        if p.exists() and np.load(p).shape[0] == len(pids):
            print(f"  cached {name}: {p}")
            return
        imgs = [Image.open(pp).convert("RGB") for pp in _image_paths(pids)]
        emb = gme.get_image_embeddings(images=imgs, batch_size=8, show_progress_bar=True)
        arr = l2(emb.float().cpu().numpy().astype(np.float32))
        np.save(p, arr)
        print(f"  saved {name}: {arr.shape} -> {p}")

    # gallery / val images
    enc_img(data.gallery_pids, "gallery_img")          # (588, D)
    enc_img(data.val_gallery_pids, "val_gallery_img")  # (240, D)
    # query texts
    enc_text(data.eval_texts, "eval_q")                # (30, D)
    enc_text(data.val_texts, "val_q")                  # (717, D)
    enc_text(synth["texts"], "synth_eval_q")           # (1706, D)
    print("encode done.")


# --------------------------------------------------------------------------
# report (numpy)
# --------------------------------------------------------------------------
def _load(name):
    return np.load(lib.cache_path(TAG, name))


def _select_bm25(dense_val, val_bm25, vgt):
    """Tune (n, w) on validation MRR only."""
    best, bv = None, -1.0
    for n in BM_N:
        for w in BM_W:
            m = lib.metrics_at_k(lib.rerank_topn(dense_val, val_bm25, n, w), vgt)["MRR"]
            if m > bv:
                bv, best = m, (n, w)
    return best, bv


def report():
    data = lib.load_data()
    synth = lib.load_synth_eval()
    refs = lib.load_refs()  # human-30 ranks for baseline/lora384/lora384_bm25_pub/...

    # embeddings
    Gi = _load("gallery_img")          # (588, D) gallery images
    Gvi = _load("val_gallery_img")     # (240, D) val gallery images
    Qe = _load("eval_q")               # (30, D)
    Qv = _load("val_q")                # (717, D)
    Qs = _load("synth_eval_q")         # (1706, D)
    print(f"dims: gallery_img {Gi.shape} val_img {Gvi.shape} eval_q {Qe.shape} "
          f"val_q {Qv.shape} synth_q {Qs.shape}")

    egt = data.eval_gt
    vgt = data.val_gt
    sgt = synth["gt"]

    # ---- dense scores ----
    dense_eval = Qe @ Gi.T
    dense_val = Qv @ Gvi.T
    dense_synth = Qs @ Gi.T

    # ---- BM25 (Korean char-ngram), same docs as the rest of the suite ----
    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(data.eval_texts)
    val_bm25 = lib.BM25(lib.make_docs(data.val_products, "all")).score_many(data.val_texts)
    synth_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(synth["texts"])

    (n0, w0), val_mrr = _select_bm25(dense_val, val_bm25, vgt)
    print(f"BM25 fusion selected on VAL: n={n0} w={w0} (val MRR {val_mrr:.4f})")

    hyb_eval = lib.rerank_topn(dense_eval, eval_bm25, n0, w0)
    hyb_synth = lib.rerank_topn(dense_synth, synth_bm25, n0, w0)

    # ---- ranks ----
    r_dense_eval = lib.ranks_of(dense_eval, egt)
    r_hyb_eval = lib.ranks_of(hyb_eval, egt)
    r_dense_synth = lib.ranks_of(dense_synth, sgt)
    r_hyb_synth = lib.ranks_of(hyb_synth, sgt)

    # synth reference: recompute LoRA384 dense+BM25 (current-best on synth) ourselves
    c = lib.load_caches()
    G_siglip = c.gallery_emb
    lora_dense_val = c.vq("lora384") @ c.val_gallery_emb.T
    (ln, lw), _ = _select_bm25(lora_dense_val, val_bm25, vgt)
    lora_synth_hyb = lib.rerank_topn(synth["enc"]["lora384"] @ G_siglip.T, synth_bm25, ln, lw)
    r_lora_bm25_synth = lib.ranks_of(lora_synth_hyb, sgt)
    print(f"synth ref LoRA+BM25 selected on VAL: n={ln} w={lw} -> synth MRR "
          f"{lib.metrics_from_ranks(r_lora_bm25_synth)['MRR']:.4f}")

    def block(ranks):
        m = lib.metrics_from_ranks(ranks)
        ci = lib.bootstrap_ci(ranks, "MRR")
        return {"R@1": round(m["R@1"], 4), "R@5": round(m["R@5"], 4),
                "R@10": round(m["R@10"], 4), "MRR": round(m["MRR"], 4),
                "MRR_ci95": [round(ci[0], 4), round(ci[1], 4)],
                "ranks": ranks.tolist()}

    out = {
        "model": GME_MODEL,
        "encoder_path": ("vendored GmeQwen2VL.get_text_embeddings / get_image_embeddings "
                         "(Qwen2-VL backbone, fp16, last-token L2-normalized pooling)"),
        "bm25_fusion_selected_on_val": {"n": n0, "w": w0, "val_MRR": round(val_mrr, 4)},
        "human30": {
            "gme_dense": block(r_dense_eval),
            "gme_bm25": block(r_hyb_eval),
        },
        "synth1706": {
            "gme_dense": block(r_dense_synth),
            "gme_bm25": block(r_hyb_synth),
        },
        "reference_MRR": {
            "human30_baseline": round(lib.metrics_from_ranks(refs["baseline"])["MRR"], 4),
            "human30_lora384": round(lib.metrics_from_ranks(refs["lora384"])["MRR"], 4),
            "human30_lora384_bm25_pub": round(lib.metrics_from_ranks(refs["lora384_bm25_pub"])["MRR"], 4),
            "synth_lora384_bm25_recomputed": round(lib.metrics_from_ranks(r_lora_bm25_synth)["MRR"], 4),
        },
        "paired_vs_current_best": {},
    }

    # paired sign tests: GME-dense and GME+BM25 vs current best LoRA+BM25
    pub = refs["lora384_bm25_pub"]
    for metric in ["MRR", "R@1", "R@10"]:
        out["paired_vs_current_best"][f"human30_gme_dense_vs_lora_bm25_pub[{metric}]"] = \
            lib.paired_compare(r_dense_eval, pub, metric)
        out["paired_vs_current_best"][f"human30_gme_bm25_vs_lora_bm25_pub[{metric}]"] = \
            lib.paired_compare(r_hyb_eval, pub, metric)
        out["paired_vs_current_best"][f"synth_gme_dense_vs_lora_bm25[{metric}]"] = \
            lib.paired_compare(r_dense_synth, r_lora_bm25_synth, metric)
        out["paired_vs_current_best"][f"synth_gme_bm25_vs_lora_bm25[{metric}]"] = \
            lib.paired_compare(r_hyb_synth, r_lora_bm25_synth, metric)

    out["multipos_human30"] = "n/a"
    out["multipos_note"] = (
        "Cached multi-positive labels (data/novel_cache/multipos_judge.json) were "
        "built from a candidate pool = union of top-15 of SigLIP/LoRA-based methods. "
        "GME's top-ranked items can fall outside that pool, so reusing the cached "
        "positives would UNDERCOUNT GME's valid hits (penalising it unfairly). A fair "
        "multi-pos would require expanding the pool with GME top-15 and re-judging the "
        "new candidates with the EXAONE prompt. Out of scope for this baseline; "
        "single-positive human-30 + synth-1706 are the reported metrics.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== GME BASELINE ===")
    for split in ["human30", "synth1706"]:
        for k, v in out[split].items():
            print(f"{split:10s} {k:10s} R@1={v['R@1']:.3f} R@5={v['R@5']:.3f} "
                  f"R@10={v['R@10']:.3f} MRR={v['MRR']:.3f} ci{v['MRR_ci95']}")
    print("refs:", out["reference_MRR"])
    print("\npaired vs current best:")
    for k, p in out["paired_vs_current_best"].items():
        print(f"  {k}: dMRR={p['mean_delta']:+.4f} sign_p={p['sign_test_p']:.4f} "
              f"W{p['wins']}/L{p['losses']}/T{p['ties']}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("encode", "all"):
        encode_all()
    if mode in ("report", "all"):
        report()
