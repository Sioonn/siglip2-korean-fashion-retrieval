"""Summarize retrieval experiment JSON files into markdown tables."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
EVAL_QUERIES = ROOT / "data" / "eval_queries.jsonl"

METHODS = [
    ("Baseline-256", "eval_baseline.json"),
    ("LoRA-256", "eval_ours.json"),
    ("LoRA-384 sigmoid10", "eval_ours_cls_384.json"),
    ("LoRA-384 + BM25 top-10 rerank", "eval_hybrid_rerank_top10_015bm25.json"),
    ("Dual r4 InfoNCE", "eval_dual_lora_r4_infonce5.json"),
    ("Dual r4 + BM25 rerank", "eval_dual_hybrid_rerank_r4_infonce5.json"),
    ("Dual r4 InfoNCE 10ep", "eval_dual_lora_r4_infonce10.json"),
    ("Dual r4 10ep + BM25 rerank", "eval_dual_hybrid_rerank_r4_infonce10.json"),
    ("Dual r8 InfoNCE", "eval_dual_lora_r8_infonce5.json"),
    ("Dual r16 InfoNCE", "eval_dual_lora_r16_infonce5.json"),
    ("Dual r8 InfoNCE 15ep", "eval_dual_lora_r8_infonce15.json"),
    ("Adaptive BM25 rerank", "eval_adaptive_hybrid_rerank_lora384.json"),
    ("Union LoRA/BM25 candidates", "eval_union_hybrid_lora384_bm25.json"),
    ("Attribute facet rerank", "eval_attribute_rerank_top10_lora384.json"),
    ("Patch MaxSim rerank", "eval_patch_rerank_top10_lora384.json"),
    ("Learned top-10 reranker", "eval_candidate_reranker_logistic_fast.json"),
]


def selected_metrics(payload):
    if "metrics" in payload:
        return payload["metrics"]
    if "eval_metrics" in payload:
        return payload["eval_metrics"]
    if "selected" in payload:
        return payload["selected"]["eval_metrics"]
    raise KeyError("no metrics found")


def detail_path(result_file):
    stem = Path(result_file).stem
    return OUTPUT / f"{stem}_detail.json"


def load_eval_queries():
    return [json.loads(line) for line in EVAL_QUERIES.read_text().splitlines() if line.strip()]


def fmt(x):
    return f"{x:.3f}"


def metrics_table(rows):
    lines = [
        "| Method | R@1 | R@5 | R@10 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in rows:
        lines.append(f"| {name} | {fmt(metrics['R@1'])} | {fmt(metrics['R@5'])} | {fmt(metrics['R@10'])} | {fmt(metrics['MRR'])} |")
    return "\n".join(lines)


def difficulty_breakdown(result_file):
    path = detail_path(result_file)
    if not path.exists():
        return []
    details = json.loads(path.read_text())
    by_level = {}
    for row in details:
        level = row.get("difficulty") or row.get("bucket") or "unknown"
        by_level.setdefault(level, []).append(int(row["rank"]))
    rows = []
    for level, ranks in sorted(by_level.items()):
        n = len(ranks)
        rows.append(
            (
                level,
                {
                    "R@1": sum(r <= 1 for r in ranks) / n,
                    "R@5": sum(r <= 5 for r in ranks) / n,
                    "R@10": sum(r <= 10 for r in ranks) / n,
                    "MRR": sum(1 / r for r in ranks) / n,
                },
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/retrieval_results_summary.md")
    args = parser.parse_args()

    rows = []
    for name, file_name in METHODS:
        path = OUTPUT / file_name
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rows.append((name, selected_metrics(payload)))

    best_file = "eval_dual_hybrid_rerank_r4_infonce10.json"
    breakdown = difficulty_breakdown(best_file)
    text = [
        "# Retrieval Experiment Summary",
        "",
        "All metrics are measured on the locked 30-query hand-written eval set.",
        "",
        "## Overall Metrics",
        "",
        metrics_table(rows),
        "",
        "## Best Method Difficulty Breakdown",
        "",
        metrics_table([(level, metrics) for level, metrics in breakdown]),
        "",
    ]
    out_path = ROOT / args.out
    out_path.write_text("\n".join(text))
    print(out_path)


if __name__ == "__main__":
    main()
