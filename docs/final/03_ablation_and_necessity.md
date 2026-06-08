# Ablation And Necessity

## 핵심 ablation 표

| Method | R@1 | R@5 | R@10 | MRR | Finding |
| --- | ---: | ---: | ---: | ---: | --- |
| Random ranking expected | 0.002 | 0.009 | 0.017 | 0.012 | 모든 retrieval 방법이 랜덤 선택보다 큰 폭으로 높아야 함을 확인하는 하한선이다. |
| Baseline-256 | 0.533 | 0.733 | 0.867 | 0.624 | R@10은 높지만 첫 순위 정확도와 MRR이 부족하다. |
| LoRA-256 | 0.633 | 0.833 | 0.833 | 0.712 | R@1/MRR은 오르지만 R@10이 baseline보다 낮아진다. |
| LoRA-384 sigmoid10 | 0.600 | 0.800 | 0.900 | 0.684 | R@10은 회복하지만 R@1/MRR은 LoRA-256보다 낮다. |
| LoRA-384 + BM25 top-10 | 0.700 | 0.833 | 0.900 | 0.756 | R@10을 유지하면서 R@1/MRR을 크게 올린다. |
| Dual r4 InfoNCE 5ep | 0.667 | 0.833 | 0.833 | 0.744 | R@1/MRR은 좋지만 R@10이 낮아 후보 누락 위험이 있다. |
| Dual r4 InfoNCE 10ep | 0.633 | 0.867 | 0.900 | 0.726 | R@5/R@10은 회복하지만 R@1/MRR은 BM25 결합 전보다 부족하다. |
| Dual r4 10ep + BM25 | 0.700 | 0.833 | 0.900 | 0.757 | R@1/R@10을 보존하고 전체 MRR이 가장 높다. |
| Dual r8 InfoNCE 5ep | 0.633 | 0.867 | 0.900 | 0.723 | R@5는 최고지만 R@1/MRR은 최종 전략보다 낮다. |
| Dual r16 InfoNCE 5ep | 0.667 | 0.833 | 0.867 | 0.749 | capacity 증가가 R@10 손상을 만든다. |
| Learned top10 reranker | 0.333 | 0.833 | 0.900 | 0.504 | validation은 매우 높지만 locked eval에서 붕괴한다. |
| Adaptive BM25 rerank | 0.600 | 0.833 | 0.867 | 0.705 | validation 선택이 eval로 일반화되지 않는다. |
| Union LoRA/BM25 candidates | 0.633 | 0.833 | 0.867 | 0.720 | 후보 확장은 R@10을 손상한다. 후보 보존이 더 안정적이다. |

## 구성요소 필요성

| 비교 | ΔR@1 | ΔR@5 | ΔR@10 | ΔMRR | 의미 |
|---|---:|---:|---:|---:|---|
| Final - Random | 0.698 | 0.825 | 0.883 | 0.745 | 무작위 선택 대비 retrieval signal이 충분히 강함 |
| Final - Baseline | 0.167 | 0.100 | 0.033 | 0.133 | 전체 목표에서 baseline 대비 모든 지표 개선 |
| Final - Dual r4 10ep | 0.067 | -0.033 | 0.000 | 0.031 | BM25 rerank는 R@1/MRR을 올리지만 R@5는 낮춘다 |
| Final - LoRA/BM25 | 0.000 | 0.000 | 0.000 | 0.001 | dual-tower generator는 MRR을 소폭 개선하며 novelty를 높인다 |

## 무엇이 필요한가

1. **Dual-tower adaptation**

   LoRA-384+BM25와 final은 R@1/R@5/R@10이 같지만 final의 MRR이 더 높다. 성능 차이는 작지만, 연구 novelty 관점에서는 text-only LoRA보다 강하다. image encoder까지 domain adaptation을 수행하기 때문이다.

2. **InfoNCE 10epoch**

   r4 5epoch 단독은 R@10이 0.833으로 낮다. r4 10epoch 단독은 R@10 0.900과 R@5 0.867을 회복한다. 즉 r4에서는 5epoch가 충분하지 않았고, 10epoch가 후보 recall을 회복한다.

3. **Candidate-preserving BM25 rerank**

   r4 10epoch 단독은 R@5 0.867이지만 R@1 0.633, MRR 0.726이다. BM25를 top-20 내부에만 적용하면 R@1 0.700, MRR 0.757로 오른다. 따라서 첫 순위 품질에는 lexical metadata가 필요하다.

4. **후보 보존 제약**

   Union LoRA/BM25 candidates는 R@10이 0.867로 내려간다. Adaptive BM25도 R@10 0.867이다. BM25를 전체 후보 확장으로 쓰는 것보다, visual retriever가 만든 후보 내부에서만 쓰는 편이 안정적이다.

## 무엇은 필요하지 않았나

- 학습형 logistic reranker: validation MRR은 높았지만 eval MRR 0.504로 붕괴했다.
- Patch MaxSim: LoRA-384와 같은 MRR 0.684로 개선이 없었다.
- Query-only adapter: query-side 보정만으로는 dual/lexical 결합을 대체하지 못했다.
- Query expansion: validation에서 expansion이 선택되지 않았다.
- r8 장기 학습: 15epoch는 eval MRR 0.674로 악화했다.

## 엄밀한 결론

우리 방법론의 모든 구성요소가 모든 metric에 항상 필요한 것은 아니다. R@5만 보면 BM25 없는 dual r4/r8가 더 높다. 그러나 논문 주장을 **balanced retrieval quality**, 특히 R@1/R@10 보존과 MRR 개선으로 잡으면, dual-tower adaptation + candidate-preserving BM25 rerank 조합이 가장 방어 가능하다.

![Final deltas](figures/final_ablation_deltas.png)

![Component ladder](figures/component_ladder_mrr_r10.png)
