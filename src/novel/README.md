# src/novel — novelty experiments (training-free test-time retrieval methods)

Goal: go beyond the repo's "obvious tricks" (LoRA rank sweeps, BM25 rerank) with
literature-grounded, mostly training-free methods that attack the structural
pathologies of contrastive cross-modal retrieval (hubness, modality gap), plus an
inference-time local-LLM query refiner. Everything is evaluated on the SAME locked
30-query eval / 588 gallery as the rest of the repo, with bootstrap CIs + paired
tests the original repo lacked.

## Environment gotcha
Prepend `unset LS_COLORS; ` to every shell command (see project memory). Interpreter:
`/opt/miniconda3/envs/jolnon/bin/python`. Run from repo root with `-m src.novel.<mod>`.

## Files
- `lib.py` — shared harness: data/cache loading (matches evaluate.py/hybrid_retrieval.py
  exactly), metrics, bootstrap CI, paired bootstrap + sign test, BM25, rerank_topn, RRF,
  `select_eval` (HP selection on validation only, then locked-eval report + paired_vs refs).
- `precompute.py` — encode & cache eval/val/bank/doc embeddings for encoders {base, lora384}.
- `refs.py` — build reference per-query ranks (baseline, lora384, lora384_bm25, dual_bm25_pub).
- `run_modality_gap.py` — TEMPLATE method + real result (Mind-the-Gap test-time correction).
- `run_<method>.py` — one self-contained script per method (written by WF2 agents).

## Reference numbers (locked eval, N=30)
| ref | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| baseline (frozen SigLIP2-384) | 0.467 | 0.633 | 0.800 | 0.571 |
| lora384 (text-only LoRA) | 0.600 | 0.800 | 0.900 | 0.684 |
| **lora384 + BM25 (current best)** | 0.700 | 0.833 | 0.900 | **0.768** |
| dual r4 + BM25 (repo headline) | 0.700 | 0.833 | 0.900 | 0.757 |

Note: the repo's dual-tower "best" (0.757) is actually below plain LoRA+BM25 (0.768).
