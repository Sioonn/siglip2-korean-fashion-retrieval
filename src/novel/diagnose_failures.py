"""Look at WHERE the current best (LoRA+BM25) fails. At N=30 the only way to
beat it is to rescue the few catastrophic-miss queries, not improve the average.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.novel import lib  # noqa

def main():
    data = lib.load_data()
    pid2p = {p["product_id"]: p for p in data.gallery_products}
    detail = json.loads((ROOT / "output" / "eval_hybrid_rerank_top10_015bm25_detail.json").read_text())
    # also dense-only (lora384) and baseline ranks for context
    c = lib.load_caches()
    dense_ranks = lib.ranks_of(c.q("lora384") @ c.gallery_emb.T, c.eval_gt)

    rows = sorted(enumerate(detail), key=lambda x: -x[1]["rank"])
    out = []
    out.append(f"{'rk':>4} {'dense_rk':>8} {'diff':<15} gt_name | query")
    for qi, d in rows:
        gt = pid2p.get(d["product_id"], {})
        top3 = [pid2p.get(pp, {}).get("product_name", pp)[:22] for pp in d.get("top10_pids", [])[:3]]
        out.append(
            f"{d['rank']:>4} {int(dense_ranks[qi]):>8} {d.get('difficulty',''):<15} "
            f"GT[{gt.get('product_name','?')[:26]}] Q[{d['query']}]")
        if d["rank"] > 5:
            out.append(f"        retrieved_top3: {top3}")
            out.append(f"        GT_desc: {(gt.get('text_description','') or '')[:240]}")
    (Path("/root/.claude/jobs/d8ed9167/tmp/failures.txt")).write_text("\n".join(out))
    miss = [d['rank'] for _, d in rows if d['rank'] > 10]
    print(f"current best: R@10 misses={len(miss)} ranks>{10}: {sorted(miss)} ; ranks>5: {sorted([d['rank'] for _,d in rows if d['rank']>5])}")

if __name__ == "__main__":
    main()
