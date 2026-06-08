# Final Retrieval Report

이 폴더는 최종 결과만 모은 보고서 패키지다. 모든 수치는 locked hand-written eval 30 queries 기준이며, 방법 선택은 train split validation에서만 수행했다.

## 결론

최종 전략은 **Dual-tower r4 InfoNCE 10epoch + top-20 후보 보존 BM25 rerank**이다.

| Method | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Dual r4 10ep + BM25 | 0.700 | 0.833 | 0.900 | 0.757 |

이 전략은 baseline 대비 모든 지표를 올리고, 기존 LoRA-256 대비 R@1/R@10/MRR을 올린다. LoRA-384+BM25와 R@1/R@5/R@10은 같지만 MRR이 더 높아서 최종 rank quality 기준으로 가장 좋다.

## 핵심 표

| Method | Family | R@1 | R@5 | R@10 | MRR | Role |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Dual r4 10ep + BM25 | dual_hybrid | 0.700 | 0.833 | 0.900 | 0.757 | 최종 전략 |
| Dual r4 5ep + BM25 | dual_hybrid | 0.700 | 0.833 | 0.867 | 0.756 | dual + lexical ablation |
| LoRA-384 + BM25 top-10 | hybrid_rerank | 0.700 | 0.833 | 0.900 | 0.756 | 강한 lexical rerank baseline |
| Ensemble dual r4 + LoRA/BM25 | ensemble | 0.700 | 0.833 | 0.900 | 0.756 | ensemble ablation |
| Dual r16 InfoNCE 5ep | dual_lora | 0.667 | 0.833 | 0.867 | 0.749 | capacity ablation |
| Dual r16 5ep + BM25 | dual_hybrid | 0.667 | 0.833 | 0.900 | 0.746 | capacity/rerank ablation |
| Dual r4 InfoNCE 5ep | dual_lora | 0.667 | 0.833 | 0.833 | 0.744 | low-rank dual adaptation |
| Dual r8 5ep + BM25 | dual_hybrid | 0.667 | 0.833 | 0.900 | 0.740 | rank/rerank ablation |

## 랜덤 선택 대비

| Method | R@1 | R@5 | R@10 | MRR | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| Random ranking expected | 0.002 | 0.009 | 0.017 | 0.012 | 무작위 선택 기준선 |
| Baseline-256 | 0.533 | 0.733 | 0.867 | 0.624 | 기준선 |
| Dual r4 10ep + BM25 | 0.700 | 0.833 | 0.900 | 0.757 | 최종 전략 |

## 파일 구성

- [01_experiment_catalog.md](01_experiment_catalog.md): 시도한 방법 하나하나의 목적, 구조, 결과, 해석
- [02_best_strategy.md](02_best_strategy.md): 최종 전략의 구체적 pipeline과 hyperparameter
- [03_ablation_and_necessity.md](03_ablation_and_necessity.md): 어떤 구성요소가 왜 필요한지에 대한 ablation 근거
- [04_visualizations.md](04_visualizations.md): 그래프 모음과 해석

## 대표 그래프

![Overall metrics](figures/overall_metrics_grouped_bar.png)

![MRR vs R@10](figures/mrr_vs_r10_scatter.png)
