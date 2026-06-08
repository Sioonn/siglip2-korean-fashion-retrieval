# Experiment Catalog

아래 표와 섹션은 이번 라운드에서 시도한 방법들을 하나씩 정리한 것이다.

## 전체 결과 테이블

| Method | Family | Role | Novelty | R@1 | R@5 | R@10 | MRR | Finding |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Random ranking expected | random | 무작위 선택 기준선 | 없음 | 0.002 | 0.009 | 0.017 | 0.012 | 모든 retrieval 방법이 랜덤 선택보다 큰 폭으로 높아야 함을 확인하는 하한선이다. |
| Baseline-256 | baseline | 기준선 | 낮음 | 0.533 | 0.733 | 0.867 | 0.624 | R@10은 높지만 첫 순위 정확도와 MRR이 부족하다. |
| LoRA-256 | text_lora | 기존 LoRA 개선선 | 낮음 | 0.633 | 0.833 | 0.833 | 0.712 | R@1/MRR은 오르지만 R@10이 baseline보다 낮아진다. |
| LoRA-384 sigmoid10 | text_lora | 해상도/체크포인트 ablation | 낮음 | 0.600 | 0.800 | 0.900 | 0.684 | R@10은 회복하지만 R@1/MRR은 LoRA-256보다 낮다. |
| LoRA-384 InfoNCE20 | text_lora | loss ablation | 낮음 | 0.567 | 0.800 | 0.867 | 0.675 | loss 변경만으로는 개선되지 않는다. |
| LoRA-384 Sigmoid20 | text_lora | epoch ablation | 낮음 | 0.600 | 0.767 | 0.867 | 0.678 | epoch 증가가 R@5/R@10을 손상한다. |
| LoRA-384 + BM25 top-10 | hybrid_rerank | 강한 lexical rerank baseline | 중간 | 0.700 | 0.833 | 0.900 | 0.756 | R@10을 유지하면서 R@1/MRR을 크게 올린다. |
| Dual r4 InfoNCE 5ep | dual_lora | low-rank dual adaptation | 높음 | 0.667 | 0.833 | 0.833 | 0.744 | R@1/MRR은 좋지만 R@10이 낮아 후보 누락 위험이 있다. |
| Dual r4 5ep + BM25 | dual_hybrid | dual + lexical ablation | 높음 | 0.700 | 0.833 | 0.867 | 0.756 | MRR은 best급이지만 R@10이 완전히 회복되지 않는다. |
| Dual r4 InfoNCE 10ep | dual_lora | epoch ablation | 높음 | 0.633 | 0.867 | 0.900 | 0.726 | R@5/R@10은 회복하지만 R@1/MRR은 BM25 결합 전보다 부족하다. |
| Dual r4 10ep + BM25 | dual_hybrid | 최종 전략 | 높음 | 0.700 | 0.833 | 0.900 | 0.757 | R@1/R@10을 보존하고 전체 MRR이 가장 높다. |
| Dual r8 sigmoid5 | dual_lora | loss/rank ablation | 높음 | 0.600 | 0.833 | 0.900 | 0.714 | vision LoRA 가능성은 보이나 InfoNCE보다 약하다. |
| Dual r8 InfoNCE 5ep | dual_lora | R@5 strong variant | 높음 | 0.633 | 0.867 | 0.900 | 0.723 | R@5는 최고지만 R@1/MRR은 최종 전략보다 낮다. |
| Dual r8 5ep + BM25 | dual_hybrid | rank/rerank ablation | 높음 | 0.667 | 0.833 | 0.900 | 0.740 | R@10은 유지하지만 MRR이 최종 전략보다 낮다. |
| Dual r16 InfoNCE 5ep | dual_lora | capacity ablation | 높음 | 0.667 | 0.833 | 0.867 | 0.749 | capacity 증가가 R@10 손상을 만든다. |
| Dual r16 5ep + BM25 | dual_hybrid | capacity/rerank ablation | 높음 | 0.667 | 0.833 | 0.900 | 0.746 | R@10은 회복되지만 MRR은 최종 전략보다 낮다. |
| Dual r8 InfoNCE 15ep | dual_lora | long training ablation | 높음 | 0.567 | 0.833 | 0.867 | 0.674 | 장기 학습은 eval rank quality를 악화한다. |
| Dual r8 15ep + BM25 | dual_hybrid | long training + rerank | 높음 | 0.633 | 0.767 | 0.867 | 0.708 | 장기 학습의 손실을 rerank가 충분히 복구하지 못한다. |
| Query adapter residual256 | adapter | lightweight adapter ablation | 중간 | 0.467 | 0.533 | 0.633 | 0.520 | query-only 보정은 심하게 부족하다. |
| Query adapter linear | adapter | lightweight adapter ablation | 중간 | 0.633 | 0.767 | 0.867 | 0.701 | 일부 MRR은 회복하지만 R@5/R@10이 낮다. |
| Patch MaxSim top10 | late_interaction | ColBERT-style ablation | 높음 | 0.600 | 0.800 | 0.900 | 0.684 | 현재 SigLIP hidden/patch 신호만으로는 개선이 없다. |
| Attribute facet top10 | facet | 해석 가능 rerank | 중간 | 0.633 | 0.800 | 0.900 | 0.714 | 해석 가능하지만 BM25보다 약하다. |
| Facet+BM25 top10 | facet | facet/lexical fusion | 중간 | 0.600 | 0.833 | 0.900 | 0.706 | validation 과적합 성향이 있고 R@1/MRR이 낮다. |
| Learned top10 reranker | learned_rerank | 학습형 reranker stress test | 중간 | 0.333 | 0.833 | 0.900 | 0.504 | validation은 매우 높지만 locked eval에서 붕괴한다. |
| Adaptive BM25 rerank | hybrid_rerank | query-adaptive rerank ablation | 중간 | 0.600 | 0.833 | 0.867 | 0.705 | validation 선택이 eval로 일반화되지 않는다. |
| Union LoRA/BM25 candidates | hybrid_retrieval | candidate expansion ablation | 중간 | 0.633 | 0.833 | 0.867 | 0.720 | 후보 확장은 R@10을 손상한다. 후보 보존이 더 안정적이다. |
| Query expansion LoRA384 | query_expansion | preprocess ablation | 낮음 | 0.600 | 0.800 | 0.900 | 0.684 | validation에서 expansion이 선택되지 않아 실질 개선이 없다. |
| Ensemble dual r4 + LoRA/BM25 | ensemble | ensemble ablation | 중간 | 0.700 | 0.833 | 0.900 | 0.756 | validation이 LoRA/BM25 단독을 선택해 ensemble 이득은 없다. |

## 방법별 상세 정리

### Random ranking expected

- 목적: 무작위 선택 기준선
- 적용 전략: gallery size N에서 정답 1개가 균등 랜덤 rank를 갖는 analytic expectation
- 결과: R@1 0.002, R@5 0.009, R@10 0.017, MRR 0.012
- 해석: 모든 retrieval 방법이 랜덤 선택보다 큰 폭으로 높아야 함을 확인하는 하한선이다.
- 근거: `analytic_random_expectation`

### Baseline-256

- 목적: 기준선
- 적용 전략: SigLIP2 256px frozen image-text retrieval
- 결과: R@1 0.533, R@5 0.733, R@10 0.867, MRR 0.624
- 해석: R@10은 높지만 첫 순위 정확도와 MRR이 부족하다.
- 근거: `output/eval_baseline.json`

### LoRA-256

- 목적: 기존 LoRA 개선선
- 적용 전략: text-side LoRA on SigLIP2 256px
- 결과: R@1 0.633, R@5 0.833, R@10 0.833, MRR 0.712
- 해석: R@1/MRR은 오르지만 R@10이 baseline보다 낮아진다.
- 근거: `output/eval_ours.json`

### LoRA-384 sigmoid10

- 목적: 해상도/체크포인트 ablation
- 적용 전략: text-side LoRA on SigLIP2 384px, sigmoid loss, 10ep
- 결과: R@1 0.600, R@5 0.800, R@10 0.900, MRR 0.684
- 해석: R@10은 회복하지만 R@1/MRR은 LoRA-256보다 낮다.
- 근거: `output/eval_ours_cls_384.json`

### LoRA-384 InfoNCE20

- 목적: loss ablation
- 적용 전략: text-side LoRA 384px, InfoNCE, longer schedule
- 결과: R@1 0.567, R@5 0.800, R@10 0.867, MRR 0.675
- 해석: loss 변경만으로는 개선되지 않는다.
- 근거: `output/eval_lora_cls_384_infonce.json`

### LoRA-384 Sigmoid20

- 목적: epoch ablation
- 적용 전략: text-side LoRA 384px, sigmoid, 20ep
- 결과: R@1 0.600, R@5 0.767, R@10 0.867, MRR 0.678
- 해석: epoch 증가가 R@5/R@10을 손상한다.
- 근거: `output/eval_lora_cls_384_sigmoid20.json`

### LoRA-384 + BM25 top-10

- 목적: 강한 lexical rerank baseline
- 적용 전략: LoRA-384 first stage + candidate-preserving Korean char-ngram BM25 rerank
- 결과: R@1 0.700, R@5 0.833, R@10 0.900, MRR 0.756
- 해석: R@10을 유지하면서 R@1/MRR을 크게 올린다.
- 근거: `output/eval_hybrid_rerank_top10_015bm25.json`

### Dual r4 InfoNCE 5ep

- 목적: low-rank dual adaptation
- 적용 전략: text/vision dual-tower LoRA r=4, InfoNCE, 5ep
- 결과: R@1 0.667, R@5 0.833, R@10 0.833, MRR 0.744
- 해석: R@1/MRR은 좋지만 R@10이 낮아 후보 누락 위험이 있다.
- 근거: `output/eval_dual_lora_r4_infonce5.json`

### Dual r4 5ep + BM25

- 목적: dual + lexical ablation
- 적용 전략: Dual r4 5ep + top-20 BM25 rerank
- 결과: R@1 0.700, R@5 0.833, R@10 0.867, MRR 0.756
- 해석: MRR은 best급이지만 R@10이 완전히 회복되지 않는다.
- 근거: `output/eval_dual_hybrid_rerank_r4_infonce5.json`

### Dual r4 InfoNCE 10ep

- 목적: epoch ablation
- 적용 전략: text/vision dual-tower LoRA r=4, InfoNCE, 10ep
- 결과: R@1 0.633, R@5 0.867, R@10 0.900, MRR 0.726
- 해석: R@5/R@10은 회복하지만 R@1/MRR은 BM25 결합 전보다 부족하다.
- 근거: `output/eval_dual_lora_r4_infonce10.json`

### Dual r4 10ep + BM25

- 목적: 최종 전략
- 적용 전략: Dual r4 InfoNCE 10ep + top-20 candidate-preserving BM25 rerank
- 결과: R@1 0.700, R@5 0.833, R@10 0.900, MRR 0.757
- 해석: R@1/R@10을 보존하고 전체 MRR이 가장 높다.
- 근거: `output/eval_dual_hybrid_rerank_r4_infonce10.json`

### Dual r8 sigmoid5

- 목적: loss/rank ablation
- 적용 전략: dual-tower LoRA r=8, sigmoid, 5ep
- 결과: R@1 0.600, R@5 0.833, R@10 0.900, MRR 0.714
- 해석: vision LoRA 가능성은 보이나 InfoNCE보다 약하다.
- 근거: `output/eval_dual_lora_r8_sigmoid5.json`

### Dual r8 InfoNCE 5ep

- 목적: R@5 strong variant
- 적용 전략: dual-tower LoRA r=8, InfoNCE, 5ep
- 결과: R@1 0.633, R@5 0.867, R@10 0.900, MRR 0.723
- 해석: R@5는 최고지만 R@1/MRR은 최종 전략보다 낮다.
- 근거: `output/eval_dual_lora_r8_infonce5.json`

### Dual r8 5ep + BM25

- 목적: rank/rerank ablation
- 적용 전략: Dual r8 InfoNCE 5ep + BM25 rerank
- 결과: R@1 0.667, R@5 0.833, R@10 0.900, MRR 0.740
- 해석: R@10은 유지하지만 MRR이 최종 전략보다 낮다.
- 근거: `output/eval_dual_hybrid_rerank_r8_infonce5.json`

### Dual r16 InfoNCE 5ep

- 목적: capacity ablation
- 적용 전략: dual-tower LoRA r=16, InfoNCE, 5ep
- 결과: R@1 0.667, R@5 0.833, R@10 0.867, MRR 0.749
- 해석: capacity 증가가 R@10 손상을 만든다.
- 근거: `output/eval_dual_lora_r16_infonce5.json`

### Dual r16 5ep + BM25

- 목적: capacity/rerank ablation
- 적용 전략: Dual r16 InfoNCE 5ep + BM25 rerank
- 결과: R@1 0.667, R@5 0.833, R@10 0.900, MRR 0.746
- 해석: R@10은 회복되지만 MRR은 최종 전략보다 낮다.
- 근거: `output/eval_dual_hybrid_rerank_r16_infonce5.json`

### Dual r8 InfoNCE 15ep

- 목적: long training ablation
- 적용 전략: dual-tower LoRA r=8, InfoNCE, 15ep
- 결과: R@1 0.567, R@5 0.833, R@10 0.867, MRR 0.674
- 해석: 장기 학습은 eval rank quality를 악화한다.
- 근거: `output/eval_dual_lora_r8_infonce15.json`

### Dual r8 15ep + BM25

- 목적: long training + rerank
- 적용 전략: Dual r8 InfoNCE 15ep + BM25 rerank
- 결과: R@1 0.633, R@5 0.767, R@10 0.867, MRR 0.708
- 해석: 장기 학습의 손실을 rerank가 충분히 복구하지 못한다.
- 근거: `output/eval_dual_hybrid_rerank_r8_infonce15.json`

### Query adapter residual256

- 목적: lightweight adapter ablation
- 적용 전략: query-side residual adapter over frozen embeddings
- 결과: R@1 0.467, R@5 0.533, R@10 0.633, MRR 0.520
- 해석: query-only 보정은 심하게 부족하다.
- 근거: `output/eval_query_adapter_residual256_infonce.json`

### Query adapter linear

- 목적: lightweight adapter ablation
- 적용 전략: query-side linear adapter over frozen embeddings
- 결과: R@1 0.633, R@5 0.767, R@10 0.867, MRR 0.701
- 해석: 일부 MRR은 회복하지만 R@5/R@10이 낮다.
- 근거: `output/eval_query_adapter_linear_infonce.json`

### Patch MaxSim top10

- 목적: ColBERT-style ablation
- 적용 전략: token-patch late interaction rerank over top-10
- 결과: R@1 0.600, R@5 0.800, R@10 0.900, MRR 0.684
- 해석: 현재 SigLIP hidden/patch 신호만으로는 개선이 없다.
- 근거: `output/eval_patch_rerank_top10_lora384.json`

### Attribute facet top10

- 목적: 해석 가능 rerank
- 적용 전략: interpretable Korean fashion facet rerank
- 결과: R@1 0.633, R@5 0.800, R@10 0.900, MRR 0.714
- 해석: 해석 가능하지만 BM25보다 약하다.
- 근거: `output/eval_attribute_rerank_top10_lora384.json`

### Facet+BM25 top10

- 목적: facet/lexical fusion
- 적용 전략: facet score + BM25 hybrid rerank
- 결과: R@1 0.600, R@5 0.833, R@10 0.900, MRR 0.706
- 해석: validation 과적합 성향이 있고 R@1/MRR이 낮다.
- 근거: `output/eval_facet_hybrid_rerank_top10_lora384.json`

### Learned top10 reranker

- 목적: 학습형 reranker stress test
- 적용 전략: learned logistic reranker over top-10 features
- 결과: R@1 0.333, R@5 0.833, R@10 0.900, MRR 0.504
- 해석: validation은 매우 높지만 locked eval에서 붕괴한다.
- 근거: `output/eval_candidate_reranker_logistic_fast.json`

### Adaptive BM25 rerank

- 목적: query-adaptive rerank ablation
- 적용 전략: query-length bucket별 BM25 weight selection
- 결과: R@1 0.600, R@5 0.833, R@10 0.867, MRR 0.705
- 해석: validation 선택이 eval로 일반화되지 않는다.
- 근거: `output/eval_adaptive_hybrid_rerank_lora384.json`

### Union LoRA/BM25 candidates

- 목적: candidate expansion ablation
- 적용 전략: LoRA top-K와 BM25 top-M 후보 union
- 결과: R@1 0.633, R@5 0.833, R@10 0.867, MRR 0.720
- 해석: 후보 확장은 R@10을 손상한다. 후보 보존이 더 안정적이다.
- 근거: `output/eval_union_hybrid_lora384_bm25.json`

### Query expansion LoRA384

- 목적: preprocess ablation
- 적용 전략: rule-based fashion query expansion before retrieval
- 결과: R@1 0.600, R@5 0.800, R@10 0.900, MRR 0.684
- 해석: validation에서 expansion이 선택되지 않아 실질 개선이 없다.
- 근거: `output/eval_query_expansion_lora384.json`

### Ensemble dual r4 + LoRA/BM25

- 목적: ensemble ablation
- 적용 전략: validation-selected ensemble of dual r4 and LoRA/BM25
- 결과: R@1 0.700, R@5 0.833, R@10 0.900, MRR 0.756
- 해석: validation이 LoRA/BM25 단독을 선택해 ensemble 이득은 없다.
- 근거: `output/eval_ensemble_dual_r4_lora_bm25.json`

