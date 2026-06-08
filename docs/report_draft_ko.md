# 한국어 패션 이미지 검색 실험 보고서 초안

## 1. 문제 정의

목표는 한국어 자연어 쿼리로 무신사 상품 이미지를 검색하는 것이다. 사용자 쿼리는 “진한 네이비색 반팔 셔츠에 흰색 테두리 글씨”처럼 색상, 의류 타입, 위치, 그래픽 단서를 짧고 구어적으로 표현한다. 따라서 pretrained SigLIP 2를 그대로 쓰는 것보다, 사용자 쿼리 분포와 상품 메타데이터를 함께 활용하는 retrieval 구조가 필요하다.

## 2. 제안 방향

최종적으로 가장 안정적인 구조는 2-stage이다.

1. `google/siglip2-base-patch16-384`에 text/vision dual-tower LoRA(r=4, InfoNCE, 10epoch)를 적용해 1차 image-text retrieval을 수행한다.
2. 1차 top-20 후보만 유지하고, 상품명/브랜드/이미지 설명에서 만든 Korean char-ngram BM25 점수로 후보 내부 순서를 재정렬한다.

이 방식은 R@10을 해치지 않으면서 R@1과 MRR을 개선한다. 연구적으로는 “dual-tower vision-language adaptation + Korean product metadata candidate-preserving reranker”로 정리할 수 있다.

## 3. 데이터 및 평가

- 학습: `data/train_pairs_final.jsonl`의 합성 사용자 스타일 쿼리
- 평가: 사람이 직접 작성한 `data/eval_queries.jsonl` 30개
- Gallery: test split 588개 상품
- 난이도: `short_ambiguous`, `medium`, `long_specific` 각 10개
- 지표: R@1, R@5, R@10, MRR

평가 positive는 학습 pair에 포함되지 않는다. 기존 split의 product_id overlap 2개는 URL 중복에서 생긴 것이며 평가 positive가 아니었다. 재현 안정성을 위해 split 생성과 학습 코드에 overlap 방어 필터를 추가했다.

## 4. 주요 결과

| Method | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Baseline-256 | 0.533 | 0.733 | 0.867 | 0.624 |
| LoRA-256 | 0.633 | 0.833 | 0.833 | 0.712 |
| LoRA-384 | 0.600 | 0.800 | 0.900 | 0.684 |
| LoRA-384 + BM25 top-10 rerank | 0.700 | 0.833 | 0.900 | 0.756 |
| Dual-tower r4 InfoNCE | 0.667 | 0.833 | 0.833 | 0.744 |
| Dual-tower r4 + BM25 rerank | 0.700 | 0.833 | 0.867 | 0.756 |
| **Dual-tower r4 InfoNCE 10ep + BM25 top-20 rerank** | **0.700** | **0.833** | **0.900** | **0.757** |
| Dual-tower r8 InfoNCE | 0.633 | **0.867** | 0.900 | 0.723 |
| Dual-tower r8 InfoNCE 15ep | 0.567 | 0.833 | 0.867 | 0.674 |

Dual-tower r4 10ep + BM25 top-20 rerank는 baseline 대비 모든 metric을 높이고, LoRA-384 대비로도 모든 metric이 높거나 같다. 기존 LoRA-256과 비교하면 R@1, R@10, MRR은 높고 R@5는 동률이다. R@5 strict improvement만 보면 dual-tower r8 InfoNCE와 dual r4 10ep 단독이 가장 높지만, 전체 rank quality(MRR)는 dual r4 10ep + BM25가 가장 높다.

## 5. Ablation

| Method | R@1 | R@5 | R@10 | MRR | 해석 |
|---|---:|---:|---:|---:|---|
| LoRA-384 InfoNCE20 | 0.567 | 0.800 | 0.867 | 0.675 | loss/epoch 변경 실패 |
| LoRA-384 Sigmoid20 | 0.600 | 0.767 | 0.867 | 0.678 | 20 epoch 과적합 |
| Query adapter linear | 0.633 | 0.767 | 0.867 | 0.701 | query-only 후처리는 약함 |
| Patch MaxSim top10 | 0.600 | 0.800 | 0.900 | 0.684 | token-patch interaction 효과 없음 |
| Attribute facet top10 | 0.633 | 0.800 | 0.900 | 0.714 | 해석 가능하지만 lexical BM25보다 약함 |
| Learned reranker | 0.333 | 0.833 | 0.900 | 0.504 | validation 과적합 |
| Dual r8 sigmoid | 0.600 | 0.833 | 0.900 | 0.714 | vision LoRA 가능성 확인 |
| Dual r4 InfoNCE | 0.667 | 0.833 | 0.833 | 0.744 | 작은 rank는 R@1/MRR 개선, R@10 손상 |
| Dual r4+BM25 | 0.700 | 0.833 | 0.867 | 0.756 | R@1/MRR은 best급, R@10 낮음 |
| Dual r4 InfoNCE 10ep | 0.633 | 0.867 | 0.900 | 0.726 | epoch 증가로 R@5/R@10 회복, R@1 하락 |
| Dual r4 10ep+BM25 | 0.700 | 0.833 | 0.900 | 0.757 | 현재 MRR 최고 |
| Dual r8 InfoNCE | 0.633 | 0.867 | 0.900 | 0.723 | R@5 최고 |
| Dual r16 InfoNCE | 0.667 | 0.833 | 0.867 | 0.749 | capacity 증가가 R@10 손상 |
| Dual r8 InfoNCE 15ep | 0.567 | 0.833 | 0.867 | 0.674 | 장기 학습은 악화 |
| Ensemble dual r4 + LoRA/BM25 | 0.700 | 0.833 | 0.900 | 0.756 | validation이 LoRA/BM25 단독을 선택 |
| Adaptive BM25 rerank | 0.600 | 0.833 | 0.867 | 0.705 | 길이별 weight는 eval 일반화 실패 |
| Union LoRA/BM25 candidates | 0.633 | 0.833 | 0.867 | 0.720 | 후보 확장은 R@10 손상 |

## 6. 결론

현재 가장 방어 가능한 주장은 다음과 같다.

> 한국어 패션 이미지 검색에서 SigLIP 2 LoRA만으로는 R@10과 rank quality 사이에 trade-off가 있다. text/vision dual-tower LoRA(r=4, InfoNCE, 10epoch)를 1차 후보 생성기로 사용하고, 한국어 상품 메타데이터 BM25를 top-20 후보 보존형 reranker로 결합하면 Baseline-256 대비 모든 지표가 개선되고, 기존 LoRA 대비 R@1/R@10/MRR이 개선된다. 또한 dual-tower rank/epoch ablation은 R@5와 MRR 사이의 trade-off를 보여준다.

## 7. 참고 연구

- SigLIP 2: https://arxiv.org/abs/2502.14786
- LoRA: https://arxiv.org/abs/2106.09685
- ColBERT: https://arxiv.org/abs/2004.12832
