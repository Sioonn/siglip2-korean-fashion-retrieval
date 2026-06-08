"""Read on-disk output/novel/*.json (ground truth, ignore agent self-reports) and
extract: selected val/eval gap, best STANDALONE (no-BM25) candidate, best +BM25
candidate, and the eval-oracle best — to separate true method effect from
validation-overfit and from the BM25 lever."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
NOVEL = ROOT / "output" / "novel"

def mrr(d):
    return d["MRR"]

def main():
    out = []
    for jf in sorted(NOVEL.glob("*.json")):
        r = json.loads(jf.read_text())
        name = jf.stem
        sel = r.get("selected", "?")
        sel_eval = r.get("eval", {}).get("metrics", {})
        # find selected's val metrics
        allc = r.get("all_eval", [])
        sel_val = next((c["val"] for c in allc if c["label"] == sel), {})
        def split(c):
            return "bm25" in c["label"].lower()
        standalone = [c for c in allc if not split(c)]
        withbm = [c for c in allc if split(c)]
        def best_by(cands, key):
            return max(cands, key=lambda c: c[key]["MRR"]) if cands else None
        # best standalone by EVAL (oracle) and by VAL (honest selection within standalone)
        sa_eval = best_by(standalone, "eval")
        sa_val = best_by(standalone, "val")
        bm_eval = best_by(withbm, "eval")
        oracle = best_by(allc, "eval")
        line = {
            "method": name,
            "n_cands": len(allc),
            "selected": sel,
            "sel_val_MRR": round(sel_val.get("MRR", -1), 4),
            "sel_eval_MRR": round(sel_eval.get("MRR", -1), 4),
            "sel_eval_R1": round(sel_eval.get("R@1", -1), 3),
            "sel_eval_R10": round(sel_eval.get("R@10", -1), 3),
            "best_standalone_by_val": (sa_val["label"], round(sa_val["val"]["MRR"], 4), round(sa_val["eval"]["MRR"], 4)) if sa_val else None,
            "best_standalone_by_eval": (sa_eval["label"], round(sa_eval["eval"]["MRR"], 4)) if sa_eval else None,
            "best_withbm25_by_eval": (bm_eval["label"], round(bm_eval["eval"]["MRR"], 4)) if bm_eval else None,
            "oracle_eval_best": (oracle["label"], round(oracle["eval"]["MRR"], 4)) if oracle else None,
        }
        # paired vs proper refs
        pv = r.get("paired_vs", {})
        for ref in ["lora384", "lora384_bm25", "lora384_bm25_pub"]:
            if ref in pv and "MRR" in pv[ref]:
                p = pv[ref]["MRR"]
                line[f"vs_{ref}"] = f"d{p['mean_delta']:+.3f},signp={p.get('sign_test_p',-1):.2f},W{p['wins']}/L{p['losses']}"
        out.append(line)
    (Path("/root/.claude/jobs/d8ed9167/tmp/inspect.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # single-line compact per method
    for l in out:
        print(f"### {l['method']} | nC={l['n_cands']} sel={l['selected']} valMRR={l['sel_val_MRR']} -> evalMRR={l['sel_eval_MRR']} (R1={l['sel_eval_R1']} R10={l['sel_eval_R10']})")
        print(f"    standalone_by_val={l['best_standalone_by_val']} | standalone_oracle={l['best_standalone_by_eval']} | withBM25_oracle={l['best_withbm25_by_eval']} | oracle_all={l['oracle_eval_best']}")
        print(f"    vs_lora384={l.get('vs_lora384','?')} | vs_lora384_bm25(0.768)={l.get('vs_lora384_bm25','?')} | vs_pub(0.756)={l.get('vs_lora384_bm25_pub','?')}")

if __name__ == "__main__":
    main()
