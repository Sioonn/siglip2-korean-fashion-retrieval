"""Novel addition: TRAINING-FREE local-LLM reranker (MM-Embed / LamRA idea).

We take the current-best retrieval pipeline (LoRA-384 dense + BM25, the published
operating point n=10 / w=0.15, MRR 0.756) and apply a *training-free* re-ranking
pass with an off-the-shelf local LLM (EXAONE-3.5-7.8B via vLLM, no external API).

Method (pointwise, robust):
  - For each query, take the top-k=10 gallery candidates from the base ranking.
  - Prompt the LLM (temp 0) with the query + each candidate's product_name and
    text_description[:220] and ask for a graded relevance score 0-3.
  - Reorder the top-k by (LLM_score desc, base_score desc); items beyond top-k
    keep their original base order below.  All re-ranking stays inside the
    SigLIP candidate space (no new candidates), so the multi-positive labels
    from output/novel/multipos.json apply unchanged.

All LLM judgements are cached to data/novel_cache/reranker_judge.json so the run
is reproducible without re-loading the LLM.

Hyperparameter / leakage:
  - Primary variant ("pure"): pure LLM reorder of the top-10.  ZERO hyperparameter,
    therefore ZERO eval leakage.
  - Secondary variant ("tuned"): a single trust-weight `w_llm` blending the LLM
    grade with the base score inside the top-k, selected on VALIDATION only.

Evaluation: human-30 single-positive AND multi-positive (reusing the multipos
candidate pool + cached judge), plus high-power synth-1706.  Reports metrics +
MRR bootstrap CIs + paired sign tests vs the un-reranked current-best base.

Run:
  unset LS_COLORS; cd /root/code/jolnon && CUDA_VISIBLE_DEVICES=<free> \
    /opt/miniconda3/envs/jolnon/bin/python -m src.novel.run_llm_reranker
"""
from __future__ import annotations

import json
import os

# XFORMERS backend: flash_attn/flashinfer not installed; default backend can
# trigger illegal-memory-access on this A6000 setup (see run_llm_refiner.py).
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")

import numpy as np

from src.novel import lib

NAME = "llm_reranker"
JUDGE_CACHE = lib.CACHE_DIR / "reranker_judge.json"
MULTIPOS_JUDGE = lib.CACHE_DIR / "multipos_judge.json"
OUT_PATH = lib.OUTPUT_DIR / f"{NAME}.json"

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
TOPK = 10                 # candidates re-ranked per query
DESC_CHARS = 220
# current-best base = published LoRA384 dense + BM25 operating point
BASE_N, BASE_W = 10, 0.15
# validation grid for the trust-weight (tuned variant only)
WLLM_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e6]  # 1e6 == hard LLM-primary sort


# --------------------------------------------------------------------------- #
#  prompt + parsing
# --------------------------------------------------------------------------- #
def build_messages(query: str, name: str, desc: str) -> list:
    sys = (
        "당신은 한국 패션 이커머스 검색 결과의 적합성을 채점하는 평가자입니다. "
        "검색어와 후보 상품을 보고 0~3점의 관련도 점수를 매기세요.\n"
        "3 = 검색어가 찾는 상품과 매우 잘 맞음 (색상·아이템 종류·핵심 디테일 일치)\n"
        "2 = 대체로 맞음 (아이템 종류는 같고 일부 속성이 다름)\n"
        "1 = 약간 관련 있음 (카테고리는 비슷하나 핵심 속성이 어긋남)\n"
        "0 = 관련 없음\n"
        "반드시 0,1,2,3 중 숫자 하나만 출력하세요."
    )
    usr = (
        f'검색어: "{query}"\n'
        f'후보 상품: 이름="{name}", 설명="{desc}"\n'
        "관련도 점수(0-3):"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def parse_grade(text: str) -> int:
    for ch in text.strip():
        if ch in "0123":
            return int(ch)
    return 0


# --------------------------------------------------------------------------- #
#  base rankings (current-best LoRA+BM25) for eval / val / synth
# --------------------------------------------------------------------------- #
def build_bases(c, data, synth):
    G, Gv = c.gallery_emb, c.val_gallery_emb
    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(data.eval_texts)
    val_bm25 = lib.BM25(lib.make_docs(data.val_products, "all")).score_many(data.val_texts)
    base_eval = lib.rerank_topn(c.q("lora384") @ G.T, eval_bm25, BASE_N, BASE_W)
    base_val = lib.rerank_topn(c.vq("lora384") @ Gv.T, val_bm25, BASE_N, BASE_W)
    base_synth = None
    if synth is not None:
        synth_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(synth["texts"])
        base_synth = lib.rerank_topn(synth["enc"]["lora384"] @ G.T, synth_bm25, BASE_N, BASE_W)
    return base_eval, base_val, base_synth


# --------------------------------------------------------------------------- #
#  LLM judging (cached); judges top-k of EACH base split
# --------------------------------------------------------------------------- #
def collect_judge_jobs(base_scores, queries, products, gallery_pids, split):
    """Return list of (key, query, name, desc) for the top-k candidates."""
    pid2p = {p["product_id"]: p for p in products}
    topk = np.argsort(-base_scores, axis=1)[:, :TOPK]
    jobs = []
    for i, q in enumerate(queries):
        for gi in topk[i]:
            gi = int(gi)
            p = pid2p[gallery_pids[gi]]
            desc = (p.get("text_description") or "")[:DESC_CHARS]
            jobs.append((f"{split}_{i}_{gi}", q, p.get("product_name", ""), desc))
    return jobs


def run_judge(jobs, cache):
    need = [j for j in jobs if j[0] not in cache]
    if not need:
        print(f"[{NAME}] judge cache hit ({len(jobs)} jobs)", flush=True)
        return cache
    print(f"[{NAME}] judging {len(need)} (query,candidate) pairs with {MODEL_ID}", flush=True)
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL_ID, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=4)
    prompts = [build_messages(q, n, d) for _, q, n, d in need]
    outs = llm.chat(prompts, sp)
    for (key, _, _, _), o in zip(need, outs):
        cache[key] = parse_grade(o.outputs[0].text)
    JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"[{NAME}] cached {len(cache)} judgements -> {JUDGE_CACHE}", flush=True)
    return cache


# --------------------------------------------------------------------------- #
#  reorder top-k by LLM grade
# --------------------------------------------------------------------------- #
def rerank(base_scores, queries, judge, split, w_llm):
    """Return new score matrix that re-orders only the top-k of base_scores.

    w_llm == inf  => hard sort by (grade desc, base desc).
    finite w_llm  => new_score = base_score + w_llm * grade  inside top-k.
    Items below top-k keep original base scores (and stay below the top-k).
    """
    out = base_scores.copy().astype(np.float64)
    topk = np.argsort(-base_scores, axis=1)[:, :TOPK]
    for i in range(len(queries)):
        cols = topk[i]
        base_vals = base_scores[i, cols].astype(np.float64)
        grades = np.array([judge.get(f"{split}_{i}_{int(gi)}", 0) for gi in cols], dtype=np.float64)
        if np.isinf(w_llm):
            # rank within top-k by (grade desc, base desc); reassign descending
            # scores strictly above the max base score of the block so order is
            # exactly (grade, base) and items stay above the non-top-k tail.
            order = sorted(range(len(cols)), key=lambda j: (grades[j], base_vals[j]), reverse=True)
            top = float(base_vals.max())
            span = float(base_vals.max() - base_vals.min()) + 1.0
            for r, j in enumerate(order):
                out[i, cols[j]] = top + (len(cols) - r) * span
        else:
            new = base_vals + w_llm * grades
            # keep the reranked block strictly above the tail: shift up to the
            # original top-k floor so no non-top-k item can overtake them.
            floor = float(base_vals.min())
            new = new - new.min() + floor
            out[i, cols] = new
    return out


# --------------------------------------------------------------------------- #
#  multi-positive positives (replicate run_multipos.py exactly)
# --------------------------------------------------------------------------- #
def build_multipos_positives(c, data, G, egt):
    """Reconstruct the EXACT positives used by run_multipos.py: candidate pool =
    union of top-15 over the 6 compared methods (+ GT), labelled by the cached
    multipos LLM judge.  Our reranker lives in the same SigLIP candidate space,
    so these positives apply to it unchanged."""
    from scipy.special import softmax

    def l2(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)

    B = c.bank("lora384")
    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(data.eval_texts)
    Q = c.q("lora384")
    Qb = c.q("base")
    tm = B.mean(0, keepdims=True)
    gm = G.mean(0, keepdims=True)
    dense = Q @ G.T
    mgap = l2(Q - 0.4 * tm) @ l2(G - 0.4 * gm).T
    pop = softmax(10 * (B @ G.T), axis=1).sum(0)
    qmul = (Q @ G.T) / (pop[None, :] ** 1.0 + 1e-8)
    rG = np.sort(B @ G.T, axis=0)[-20:].mean(0)
    rQ = np.sort(dense, axis=1)[:, -20:].mean(1)
    csls = 2 * dense - rQ[:, None] - rG[None, :]
    rr = lambda S: lib.rerank_topn(S, eval_bm25, 50, 0.2)
    methods = {
        "baseline": Qb @ G.T, "lora": dense, "dense_bm25": rr(dense),
        "modality_gap_bm25": rr(mgap), "qbnorm_mul_bm25": rr(qmul), "csls_bm25": rr(csls),
    }
    pool = [set() for _ in range(len(egt))]
    for S in methods.values():
        top = np.argsort(-S, axis=1)[:, :15]
        for i in range(len(egt)):
            pool[i].update(int(x) for x in top[i])
    for i in range(len(egt)):
        pool[i].add(int(egt[i]))
    judged = json.loads(MULTIPOS_JUDGE.read_text())
    positives = []
    for i in range(len(egt)):
        pos = {int(egt[i])}
        for gi in pool[i]:
            if judged.get(f"{i}_{gi}", 0) == 1:
                pos.add(int(gi))
        positives.append(pos)
    return positives


def multipos_first_rank(scores, positives):
    ranks = np.empty(len(scores), dtype=np.int64)
    for i in range(len(scores)):
        order = np.argsort(-scores[i])
        rank_of = {int(p): k for k, p in enumerate(order)}
        ranks[i] = min(rank_of[p] for p in positives[i]) + 1
    return ranks


# --------------------------------------------------------------------------- #
def main():
    c = lib.load_caches()
    data = lib.load_data()
    synth = lib.load_synth_eval()
    G = c.gallery_emb
    egt, vgt = c.eval_gt, c.val_gt
    refs = lib.load_refs()
    print(f"synth eval: {'N=' + str(synth['n']) if synth else 'ABSENT'}", flush=True)

    base_eval, base_val, base_synth = build_bases(c, data, synth)

    # sanity: our base == published current-best lora384_bm25_pub
    base_eval_ranks = lib.ranks_of(base_eval, egt)
    pub = refs["lora384_bm25_pub"]
    print(f"base eval MRR={lib.metrics_from_ranks(base_eval_ranks)['MRR']:.4f} "
          f"(pub {lib.metrics_from_ranks(pub)['MRR']:.4f}, ranks_match={int((base_eval_ranks == pub).all())})",
          flush=True)

    # ---- LLM judging (cached) over top-k of every split ----
    cache = json.loads(JUDGE_CACHE.read_text()) if JUDGE_CACHE.exists() else {}
    jobs = collect_judge_jobs(base_eval, data.eval_texts, data.gallery_products, c.gallery_pids, "eval")
    jobs += collect_judge_jobs(base_val, data.val_texts, data.val_products, c.val_gallery_pids, "val")
    if synth is not None:
        jobs += collect_judge_jobs(base_synth, synth["texts"], data.gallery_products, c.gallery_pids, "synth")
    cache = run_judge(jobs, cache)

    # grade distribution diagnostic
    eval_grades = [cache[k] for k in cache if k.startswith("eval_")]
    print(f"eval grade dist {np.bincount(eval_grades, minlength=4).tolist()}", flush=True)

    # ---- select trust-weight on VALIDATION only ----
    best_w, best_vm = None, -1.0
    val_table = {}
    for w in WLLM_GRID:
        vs = rerank(base_val, data.val_texts, cache, "val", w)
        vm = lib.metrics_at_k(vs, vgt)["MRR"]
        val_table[("inf" if np.isinf(w) else w)] = round(vm, 4)
        if vm > best_vm:
            best_vm, best_w = vm, w
    print(f"val MRR by w_llm: {val_table}  -> selected w_llm={best_w}", flush=True)

    # ---- build multi-positive labels (same pool as run_multipos.py) ----
    positives = build_multipos_positives(c, data, G, egt)
    npos = [len(p) for p in positives]
    print(f"multipos positives/query mean {np.mean(npos):.2f}", flush=True)

    # ---- evaluate the two variants ----
    variants = {"pure": float("inf"), "tuned": best_w}
    base_synth_ranks = lib.ranks_of(base_synth, synth["gt"]) if synth is not None else None
    base_multipos_ranks = multipos_first_rank(base_eval, positives)

    results = {}
    for vname, w in variants.items():
        re_eval = rerank(base_eval, data.eval_texts, cache, "eval", w)
        eval_ranks = lib.ranks_of(re_eval, egt)
        re_mp = rerank(base_eval, data.eval_texts, cache, "eval", w)  # same matrix
        mp_ranks = multipos_first_rank(re_mp, positives)
        entry = {
            "w_llm": ("inf" if np.isinf(w) else w),
            "singlepos_human30": {
                **lib.metrics_from_ranks(eval_ranks),
                "MRR_ci": [round(x, 3) for x in lib.bootstrap_ci(eval_ranks, "MRR")],
                "ranks": eval_ranks.tolist(),
            },
            "multipos_human30": {
                **lib.metrics_from_ranks(mp_ranks),
                "MRR_ci": [round(x, 3) for x in lib.bootstrap_ci(mp_ranks, "MRR")],
            },
            "paired_vs_base": {
                "singlepos_MRR": lib.paired_compare(eval_ranks, base_eval_ranks, "MRR"),
                "singlepos_R@1": lib.paired_compare(eval_ranks, base_eval_ranks, "R@1"),
                "multipos_MRR": lib.paired_compare(mp_ranks, base_multipos_ranks, "MRR"),
            },
        }
        if synth is not None:
            re_synth = rerank(base_synth, synth["texts"], cache, "synth", w)
            synth_ranks = lib.ranks_of(re_synth, synth["gt"])
            entry["synth1706"] = {
                **lib.metrics_from_ranks(synth_ranks),
                "MRR_ci": [round(x, 3) for x in lib.bootstrap_ci(synth_ranks, "MRR")],
            }
            entry["paired_vs_base"]["synth_MRR"] = lib.paired_compare(synth_ranks, base_synth_ranks, "MRR")
            entry["paired_vs_base"]["synth_R@1"] = lib.paired_compare(synth_ranks, base_synth_ranks, "R@1")
        results[vname] = entry

    out = {
        "name": NAME,
        "model": MODEL_ID,
        "base": {"pipeline": "lora384 dense + BM25", "n": BASE_N, "w": BASE_W, "topk": TOPK,
                 "desc_chars": DESC_CHARS,
                 "singlepos_MRR": round(lib.metrics_from_ranks(base_eval_ranks)["MRR"], 4),
                 "matches_pub_current_best": int((base_eval_ranks == pub).all())},
        "base_metrics": {
            "singlepos_human30": lib.metrics_from_ranks(base_eval_ranks),
            "multipos_human30": lib.metrics_from_ranks(base_multipos_ranks),
            "synth1706": (lib.metrics_from_ranks(base_synth_ranks) if synth is not None else None),
        },
        "val_w_table": {str(k): v for k, v in val_table.items()},
        "selected_w_llm": ("inf" if np.isinf(best_w) else best_w),
        "mean_positives": round(float(np.mean(npos)), 2),
        "synth_n": (synth["n"] if synth else 0),
        "variants": results,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved", OUT_PATH, flush=True)

    # console summary
    for vname, e in results.items():
        sp = e["singlepos_human30"]; mp = e["multipos_human30"]
        pv = e["paired_vs_base"]
        line = (f"[{vname} w={e['w_llm']}] singlepos MRR={sp['MRR']:.4f} ci{sp['MRR_ci']} "
                f"(base {out['base_metrics']['singlepos_human30']['MRR']:.4f}) "
                f"vsBase dMRR={pv['singlepos_MRR']['mean_delta']:+.4f} "
                f"signp={pv['singlepos_MRR']['sign_test_p']:.3f} "
                f"W{pv['singlepos_MRR']['wins']}/L{pv['singlepos_MRR']['losses']} || "
                f"multipos MRR={mp['MRR']:.4f} vsBase dMRR={pv['multipos_MRR']['mean_delta']:+.4f} "
                f"signp={pv['multipos_MRR']['sign_test_p']:.3f}")
        if "synth1706" in e:
            sy = e["synth1706"]
            line += (f" || synth MRR={sy['MRR']:.4f} ci{sy['MRR_ci']} "
                     f"(base {out['base_metrics']['synth1706']['MRR']:.4f}) "
                     f"vsBase dMRR={pv['synth_MRR']['mean_delta']:+.4f} "
                     f"signp={pv['synth_MRR']['sign_test_p']:.4f} "
                     f"W{pv['synth_MRR']['wins']}/L{pv['synth_MRR']['losses']}")
        print(line, flush=True)


if __name__ == "__main__":
    main()
