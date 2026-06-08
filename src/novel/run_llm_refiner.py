"""Novel method: inference-time Local-LLM Query Refiner (HyDE-style).

Closes the user->caption distribution gap at INFERENCE (Gao et al., ACL 2023,
"Precise Zero-Shot Dense Retrieval without Relevance Labels").  A local LLM
(vLLM, no external API) rewrites each casual user query into a hypothetical
FORMAL product caption that looks like the catalogue distribution.  We encode
the rewrites with the SAME SigLIP text encoder we compare on, blend with the
original query embedding, and score against the frozen gallery.

No eval leakage: rewrites are produced per-query from the query text only; HP
selection (model, encoder, alpha, K, +bm25) is done on VALIDATION only via
lib.select_eval.

Run:
  unset LS_COLORS; cd /root/code/jolnon && CUDA_VISIBLE_DEVICES=1 \
    /opt/miniconda3/envs/jolnon/bin/python -m src.novel.run_llm_refiner
"""
from __future__ import annotations

import json
import os

# Force the XFORMERS attention backend BEFORE vLLM is imported.  flash_attn /
# flashinfer are not installed in this env and vLLM's default backend can
# trigger a CUDA "illegal memory access" during warmup profiling on this
# RTX A6000 setup.  XFORMERS is the safe, always-available fallback.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")

import numpy as np

from src.novel import lib

NAME = "llm_refiner"
CACHE_DIR = str(lib.CACHE_DIR)
REWRITE_CACHE = os.path.join(CACHE_DIR, "refiner_rewrites.json")
EMB_DIR = os.path.join(CACHE_DIR, "refiner_emb")
OUT_PATH = str(lib.OUTPUT_DIR / f"{NAME}.json")
SUMMARY_PATH = os.path.join(CACHE_DIR, "refiner_summary.json")
KMAX = 3

# Models to try.  Qwen first (robust in vLLM); EXAONE as Korean alternative.
MODELS = {
    "qwen": {"id": "Qwen/Qwen2.5-7B-Instruct", "trust_remote_code": False},
    "exaone": {"id": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "trust_remote_code": True},
}

ENC_PATHS = {"base": None, "lora384": "checkpoints/lora_cls_384"}

# Few-shot: casual user query -> formal catalogue-style caption (Korean).
SYS_PROMPT = (
    "당신은 한국 패션 이커머스의 상품 설명 작성자입니다. "
    "사용자가 입력한 짧고 캐주얼한 검색어를, 해당 상품을 묘사하는 "
    "격식 있는 상품 카탈로그 설명문 한 문장으로 바꿔 쓰세요. "
    "소재, 핏, 색상, 디테일, 분위기 등 상품 카탈로그에 어울리는 표현을 사용하고, "
    "원래 검색어의 의미를 유지하세요. 설명문만 출력하세요."
)
FEWSHOT = [
    ("가벼운 셔츠",
     "가볍고 통기성이 좋은 소재로 제작된 베이직 셔츠로, 깔끔한 핏과 단정한 실루엣이 돋보이는 데일리 아이템입니다."),
    ("따뜻한 니트",
     "부드럽고 보온성이 뛰어난 니트 소재로 제작된 스웨터로, 편안한 착용감과 포근한 분위기를 자아내는 가을·겨울 아이템입니다."),
]


def _load_refs(c, data):
    """Reference per-query eval-ranks for paired tests.

    Prefer lib.load_refs() if present; else read the refs.json written by
    src/novel/refs.py; else recompute baseline/lora384/lora384_bm25 inline.
    All leakage-safe: lora384_bm25 weight is selected on VALIDATION only.
    """
    if hasattr(lib, "load_refs"):
        return lib.load_refs()

    refs_json = os.path.join(CACHE_DIR, "refs.json")
    if os.path.exists(refs_json):
        with open(refs_json) as f:
            d = json.load(f)
        return {k: np.array(v, dtype=np.int64) for k, v in d.items()}

    refs = {}
    refs["baseline"] = lib.ranks_of(c.q("base") @ c.gallery_emb.T, c.eval_gt)
    lora_eval = c.q("lora384") @ c.gallery_emb.T
    refs["lora384"] = lib.ranks_of(lora_eval, c.eval_gt)
    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(
        data.eval_texts
    )
    val_bm25 = lib.BM25(lib.make_docs(data.val_products, "all")).score_many(
        data.val_texts
    )
    lora_val = c.vq("lora384") @ c.val_gallery_emb.T
    cands = [("image_only", lora_val, lora_eval)]
    for n in (10, 20, 50):
        for w in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.50):
            cands.append((
                f"rerank_top{n}_{w:g}bm25",
                lib.rerank_topn(lora_val, val_bm25, n, w),
                lib.rerank_topn(lora_eval, eval_bm25, n, w),
            ))
    r = lib.select_eval(cands, c.val_gt, c.eval_gt)
    refs["lora384_bm25"] = np.array(r["eval_ranks"], dtype=np.int64)
    return refs


def build_messages(query: str) -> list:
    msgs = [{"role": "system", "content": SYS_PROMPT}]
    for u, a in FEWSHOT:
        msgs.append({"role": "user", "content": f"검색어: {u}"})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"검색어: {query}"})
    return msgs


# --------------------------------------------------------------------------- #
#  Step 1: generate rewrites with vLLM (cached)
# --------------------------------------------------------------------------- #
def generate_rewrites(model_keys, eval_texts, val_texts):
    """Generate KMAX rewrites per query per model; cache to disk.

    Cache schema: {model_key: {"eval": [[r1..rK], ...], "val": [[...], ...]}}
    """
    cache = {}
    if os.path.exists(REWRITE_CACHE):
        with open(REWRITE_CACHE) as f:
            cache = json.load(f)

    need = [m for m in model_keys if m not in cache]
    if not need:
        print(f"[{NAME}] rewrites cache hit for {model_keys}", flush=True)
        return cache

    all_texts = list(eval_texts) + list(val_texts)
    n_eval = len(eval_texts)

    from vllm import LLM, SamplingParams

    for mk in need:
        spec = MODELS[mk]
        print(f"[{NAME}] loading vLLM model {spec['id']} ...", flush=True)
        llm = LLM(
            model=spec["id"],
            trust_remote_code=spec["trust_remote_code"],
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=2048,
            enforce_eager=True,
        )
        tok = llm.get_tokenizer()

        prompts = []
        for q in all_texts:
            prompts.append(tok.apply_chat_template(
                build_messages(q), tokenize=False, add_generation_prompt=True
            ))

        sp = SamplingParams(
            n=KMAX, temperature=0.7, top_p=0.9, max_tokens=96, seed=0
        )
        outs = llm.generate(prompts, sp)

        def collect(outputs):
            res = []
            for o in outputs:
                cands = [c.text.strip().replace("\n", " ") for c in o.outputs]
                while len(cands) < KMAX:
                    cands.append(cands[-1] if cands else "")
                res.append(cands[:KMAX])
            return res

        rew = collect(outs)
        cache[mk] = {"eval": rew[:n_eval], "val": rew[n_eval:]}
        print(f"[{NAME}] generated rewrites for {mk}: "
              f"eval={len(cache[mk]['eval'])} val={len(cache[mk]['val'])}",
              flush=True)

        del llm
        import gc
        import torch
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(REWRITE_CACHE, "w") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    return cache


# --------------------------------------------------------------------------- #
#  Step 2: encode rewrites with SigLIP text encoder (cached np arrays)
# --------------------------------------------------------------------------- #
def encode_rewrites(cache, model_keys, encs):
    os.makedirs(EMB_DIR, exist_ok=True)
    result = {}
    for mk in model_keys:
        for split in ("eval", "val"):
            rew = cache[mk][split]            # list[Nq] of list[KMAX]
            nq = len(rew)
            flat = [rew[i][k] for i in range(nq) for k in range(KMAX)]
            for enc in encs:
                path = os.path.join(EMB_DIR, f"{mk}_{split}_{enc}.npy")
                if os.path.exists(path):
                    arr = np.load(path)
                else:
                    flat_emb = lib.encode_text(flat, lora_path=ENC_PATHS[enc])
                    arr = flat_emb.reshape(nq, KMAX, -1).astype(np.float32)
                    np.save(path, arr)
                    print(f"[{NAME}] encoded {path} shape={arr.shape}",
                          flush=True)
                result[(mk, split, enc)] = arr
    return result


# --------------------------------------------------------------------------- #
#  Step 3: blend + score
# --------------------------------------------------------------------------- #
def blend_score(q_orig, rew_emb, alpha, k, gallery):
    """emb = L2norm((1-a)*q_orig + a*mean(rewrite_embs[:k])); dot gallery."""
    if k == 0 or alpha == 0.0:
        emb = q_orig
    else:
        rmean = lib.normalize_rows(rew_emb[:, :k, :].mean(axis=1))
        emb = (1.0 - alpha) * q_orig + alpha * rmean
    emb = lib.normalize_rows(emb)
    return emb @ gallery.T


def main() -> None:
    c = lib.load_caches()
    data = lib.load_data()
    refs = _load_refs(c, data)

    model_keys = ["qwen", "exaone"]

    # ---- step 1: rewrites (vLLM) ----
    try:
        cache = generate_rewrites(model_keys, data.eval_texts, data.val_texts)
    except Exception as e:  # noqa: BLE001
        print(f"[{NAME}] WARN full-set generation failed ({e!r}); "
              f"falling back to qwen-only", flush=True)
        model_keys = ["qwen"]
        cache = generate_rewrites(model_keys, data.eval_texts, data.val_texts)
    model_keys = [m for m in model_keys if m in cache]

    # ---- step 2: encode rewrites with SigLIP ----
    encs = ["lora384", "base"]
    rew_emb = encode_rewrites(cache, model_keys, encs)

    # ---- BM25 for the fusion extra set (mirror refs.py) ----
    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(
        data.eval_texts
    )
    val_bm25 = lib.BM25(lib.make_docs(data.val_products, "all")).score_many(
        data.val_texts
    )

    # ---- step 3: candidates across {model, enc, alpha, K, +bm25} ----
    candidates = []
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    Ks = (1, 2, 3)

    for mk in model_keys:
        for enc in encs:
            q_e = c.q(enc)
            q_v = c.vq(enc)
            re_e = rew_emb[(mk, "eval", enc)]
            re_v = rew_emb[(mk, "val", enc)]
            for alpha in alphas:
                k_list = Ks if alpha > 0.0 else (1,)  # K irrelevant at a=0
                for k in k_list:
                    es = blend_score(q_e, re_e, alpha, k, c.gallery_emb)
                    vs = blend_score(q_v, re_v, alpha, k, c.val_gallery_emb)
                    label = f"{mk}|{enc}|a={alpha}|K={k}"
                    candidates.append((label, vs, es))
                    for w in (0.3, 0.5):
                        es_r = lib.rerank_topn(es, eval_bm25, 50, w)
                        vs_r = lib.rerank_topn(vs, val_bm25, 50, w)
                        candidates.append((f"{label}|bm25w={w}", vs_r, es_r))

    sel = lib.select_eval(candidates, c.val_gt, c.eval_gt, c.eval_diff, refs)

    out = {
        "method": NAME,
        "selected": sel["selected"],
        "val_metrics": sel["val_metrics"],
        "eval": sel["eval"],
        "paired_vs": sel.get("paired_vs", {}),
        "all_eval": sel["all_eval"],
        "models_used": model_keys,
        "n_candidates": len(candidates),
    }
    lib.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- compact machine-readable summary (single source of truth) ----
    s = sel["selected"]
    m = sel["eval"]["metrics"]
    ci = sel["eval"]["ci95"]
    bd = sel["eval"].get("by_difficulty", {})
    pv = sel.get("paired_vs", {})

    def pick(ref):
        r = pv.get(ref, {}).get("MRR", {})
        return {
            "mean_delta": r.get("mean_delta"),
            "ci95": r.get("ci95"),
            "p_delta_gt0": r.get("p_delta_gt0"),
            "wins": r.get("wins"),
            "losses": r.get("losses"),
            "sign_test_p": r.get("sign_test_p"),
        }

    by_group = {}
    a0 = {}
    for row in sel["all_eval"]:
        lab = row["label"]
        ev = row["eval"]["MRR"]
        vv = row["val"]["MRR"]
        parts = lab.split("|")
        mk, enc = parts[0], parts[1]
        grp = f"{mk}|{enc}|{'bm25' if 'bm25' in lab else 'dense'}"
        if grp not in by_group or ev > by_group[grp]["eval_MRR"]:
            by_group[grp] = {"label": lab, "eval_MRR": ev, "val_MRR": vv}
        if "a=0.0" in lab:
            a0[lab] = {"eval_MRR": ev, "val_MRR": vv}

    summary = {
        "selected": s,
        "val_MRR": sel["val_metrics"]["MRR"],
        "eval": {k: m[k] for k in ("R@1", "R@5", "R@10", "MRR")},
        "eval_ci95_MRR": ci.get("MRR"),
        "by_difficulty": bd,
        "paired_vs": {r: pick(r) for r in ("baseline", "lora384", "lora384_bm25")},
        "best_by_group": by_group,
        "alpha0_refs": a0,
        "models_used": model_keys,
        "n_candidates": len(candidates),
        "beats_current_best": bool(m["MRR"] >= 0.768),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    sa = bd.get("short_ambiguous", {}).get("MRR", float("nan"))
    print(f"[{NAME}] EXIT_OK selected={s} MRR={m['MRR']:.4f} R@1={m['R@1']:.3f} "
          f"R@10={m['R@10']:.3f} short_amb_MRR={sa:.4f} "
          f"n_cand={len(candidates)}", flush=True)


if __name__ == "__main__":
    main()
