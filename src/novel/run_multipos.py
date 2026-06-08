"""Multi-positive re-evaluation of the existing 30 human queries (NO new queries).

Motivation (diagnose_failures.py): the catastrophic misses are single-positive
artifacts — e.g. "흰색 긴팔 셔츠" has dozens of valid white long-sleeve tops in the
gallery but only one is labelled correct. Standard fix (recommended in explan.md,
never implemented): mark ALL valid products per query as positives, then R@k/MRR
give credit for any valid hit. Applied identically to every method => fair.

Protocol (defensible):
  1. method-AGNOSTIC candidate pool = union of top-15 retrieved by ALL compared
     methods (so the pool is not biased toward any method).
  2. a LOCAL LLM judge (EXAONE via vLLM, no API) labels each (query, product)
     yes/no, BLIND to which method retrieved it. The original GT is always a positive.
  3. recompute multi-positive R@1/5/10 and MRR (rank of the FIRST positive) for
     every method + paired sign tests.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from scipy.special import logsumexp, softmax  # noqa: E402
from src.novel import lib  # noqa: E402

JUDGE_CACHE = ROOT / "data" / "novel_cache" / "multipos_judge.json"


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


def build_methods(c, data, G):
    """eval score matrices for the methods we compare (lora384 ops point)."""
    B = c.bank("lora384")
    eval_bm25 = lib.BM25(lib.make_docs(data.gallery_products, "all")).score_many(data.eval_texts)
    Q = c.q("lora384")
    Qb = c.q("base")
    tm = B.mean(0, keepdims=True)
    gm = G.mean(0, keepdims=True)
    dense = Q @ G.T
    # corrections
    mgap = l2(Q - 0.4 * tm) @ l2(G - 0.4 * gm).T
    pop = softmax(10 * (B @ G.T), axis=1).sum(0)
    qmul = (Q @ G.T) / (pop[None, :] ** 1.0 + 1e-8)
    rG = np.sort(B @ G.T, axis=0)[-20:].mean(0)
    rQ = np.sort(dense, axis=1)[:, -20:].mean(1)
    csls = 2 * dense - rQ[:, None] - rG[None, :]
    rr = lambda S, n=50, w=0.2: lib.rerank_topn(S, eval_bm25, n, w)
    return {
        "baseline": Qb @ G.T,
        "lora": dense,
        "dense_bm25": rr(dense),
        "modality_gap_bm25": rr(mgap),
        "qbnorm_mul_bm25": rr(qmul),
        "csls_bm25": rr(csls),
    }


def multipos_first_rank(scores, positives):
    ranks = np.empty(len(scores), dtype=np.int64)
    for i in range(len(scores)):
        order = np.argsort(-scores[i])
        rank_of = {int(p): k for k, p in enumerate(order)}
        ranks[i] = min(rank_of[p] for p in positives[i]) + 1
    return ranks


def main():
    c = lib.load_caches()
    data = lib.load_data()
    G = c.gallery_emb
    egt = c.eval_gt
    methods = build_methods(c, data, G)

    # candidate pool: union of top-15 across all methods
    pool = [set() for _ in range(len(egt))]
    for S in methods.values():
        top = np.argsort(-S, axis=1)[:, :15]
        for i in range(len(egt)):
            pool[i].update(int(x) for x in top[i])
    for i in range(len(egt)):
        pool[i].add(int(egt[i]))  # GT always in pool
    print("pool sizes:", [len(p) for p in pool][:8], "... mean", round(np.mean([len(p) for p in pool]), 1))

    # ---- LLM judge (cached) ----
    if JUDGE_CACHE.exists():
        judged = json.loads(JUDGE_CACHE.read_text())
        print("loaded cached judgements")
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", trust_remote_code=True,
                  dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048)
        sp = SamplingParams(temperature=0.0, max_tokens=8)
        pid2p = {p["product_id"]: p for p in data.gallery_products}
        prompts, keys = [], []
        for i, q in enumerate(data.eval_texts):
            for gi in sorted(pool[i]):
                p = pid2p[c.gallery_pids[gi]]
                desc = (p.get("text_description") or "")[:220]
                msg = (f"패션 상품 검색 적합성 판정.\n검색어: \"{q}\"\n"
                       f"후보: 이름=\"{p.get('product_name','')}\", 설명=\"{desc}\"\n"
                       f"이 후보가 검색어가 찾는 상품으로 적절합니까? 색상·아이템 종류·그래픽이 대체로 부합하면 예, 명백히 다르면 아니오.\n"
                       f"반드시 '예' 또는 '아니오' 한 단어로만 답하세요.")
                prompts.append([{"role": "user", "content": msg}])
                keys.append((i, gi))
        outs = llm.chat(prompts, sp)
        judged = {}
        for (i, gi), o in zip(keys, outs):
            t = o.outputs[0].text.strip()
            judged[f"{i}_{gi}"] = 1 if ("예" in t and "아니오" not in t[:3]) else 0
        JUDGE_CACHE.write_text(json.dumps(judged, ensure_ascii=False))
        print("judged", len(judged))

    positives = []
    for i in range(len(egt)):
        pos = {int(egt[i])}
        for gi in pool[i]:
            if judged.get(f"{i}_{gi}", 0) == 1:
                pos.add(int(gi))
        positives.append(pos)
    npos = [len(p) for p in positives]
    print("positives per query: mean", round(np.mean(npos), 2), "min", min(npos), "max", max(npos))

    # ---- multi-positive metrics + paired tests ----
    res = {}
    fr = {name: multipos_first_rank(S, positives) for name, S in methods.items()}
    for name, ranks in fr.items():
        m = lib.metrics_from_ranks(ranks)
        ci = lib.bootstrap_ci(ranks, "MRR")
        res[name] = {"R@1": m["R@1"], "R@5": m["R@5"], "R@10": m["R@10"], "MRR": round(m["MRR"], 4),
                     "MRR_ci": [round(ci[0], 3), round(ci[1], 3)]}
    out = {"mean_positives": round(float(np.mean(npos)), 2), "methods": res, "paired": {}}
    for name in methods:
        if name in ("baseline", "dense_bm25"):
            continue
        out["paired"][f"{name}_vs_dense_bm25"] = lib.paired_compare(fr[name], fr["dense_bm25"], "MRR")
    out["paired"]["dense_bm25_vs_baseline"] = lib.paired_compare(fr["dense_bm25"], fr["baseline"], "MRR")
    (ROOT / "output" / "novel" / "multipos.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print("\n=== MULTI-POSITIVE (mean %.2f positives/query) ===" % out["mean_positives"])
    for name, m in res.items():
        print(f"{name:20s} R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} R@10={m['R@10']:.3f} MRR={m['MRR']:.3f} ci{m['MRR_ci']}")
    for k, p in out["paired"].items():
        print(f"  {k}: dMRR={p['mean_delta']:+.4f} sign_p={p['sign_test_p']:.3f} W{p['wins']}/L{p['losses']}")
    print("saved output/novel/multipos.json")


if __name__ == "__main__":
    main()
