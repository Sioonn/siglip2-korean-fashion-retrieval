# Paper Readiness Audit

이 문서는 현재 실험 묶음이 **간이 논문**으로 작성될 수 있을 정도로 논리적으로 이어지는지 점검하고, 기존 문서에 흩어져 있던 연구 서사와 주장 강도를 한 곳에 모은 것이다. 결과 수치의 원천 표와 재현 정보는 `docs/final/`를 기준으로 한다.

## 1. 최종 판단

현재 자료는 **간이 논문 초안 작성에는 충분하다.** 특히 문제 정의, baseline, 제안 방법, ablation, 실패 실험의 해석, random baseline 비교가 모두 존재한다. 다만 강한 학회 논문 수준의 claim으로 밀기에는 아직 약한 지점이 있다.

| 항목 | 판단 | 이유 |
|---|---|---|
| 문제 정의 | 충분 | 한국어 자연어 패션 쿼리와 상품 이미지 retrieval이라는 task가 명확하다. |
| 방법 novelty | 충분 | text-only LoRA가 아니라 text/vision dual-tower LoRA와 Korean product metadata rerank를 결합한다. |
| baseline 비교 | 충분 | random, frozen SigLIP baseline, text-only LoRA, LoRA+BM25, dual-only, reranker 변형이 있다. |
| ablation | 충분 | rank, epoch, BM25, 후보 보존, learned reranker, MaxSim, query adapter, union 후보 확장을 비교했다. |
| metric 다양성 | 충분 | R@1, R@5, R@10, MRR을 모두 비교했고 random baseline도 포함했다. |
| 통계적 강도 | 제한적 | locked eval이 30 queries라 metric 차이에 대한 신뢰구간이 넓다. |
| 최종 우월성 주장 | 제한적으로 가능 | final은 MRR 기준 최고이며 R@1/R@10을 보존하지만, 모든 metric에서 strict best는 아니다. |
| 데이터 청결성 | 보완 필요 | eval positive leakage는 확인되지 않았지만 현재 split 파일에는 product_id overlap 2개가 남아 있다. |

따라서 논문 목표는 “새로운 방법이 모든 면에서 압도적으로 최고”가 아니라, **한국어 패션 이미지 검색에서 dual-tower domain adaptation과 후보 보존형 lexical metadata reranking이 balanced retrieval quality를 만든다는 실험적 근거**로 잡는 것이 가장 방어 가능하다.

## 2. 권장 논문 thesis

권장 중심 주장:

> 한국어 패션 이미지 검색에서는 색상, 의류 타입, 위치, 그래픽 문구처럼 시각 단서와 상품 메타데이터 단서가 함께 등장한다. SigLIP 기반 dual-tower LoRA로 image/text encoder를 함께 domain-adapt해 후보 recall을 확보하고, top-20 후보 내부에서 Korean char-ngram BM25 metadata score를 결합하면 R@1과 MRR을 개선하면서 R@10을 유지하는 균형 잡힌 ranking을 얻을 수 있다.

영문으로 쓰면 다음 정도가 안전하다.

> For Korean fashion image retrieval, a dual-tower LoRA adaptation of a vision-language model provides a stronger visual-textual candidate generator, while candidate-preserving lexical metadata reranking improves top-rank precision without sacrificing top-10 recall.

이 thesis의 장점은 세 가지다.

1. `왜 dual-tower인가`: text encoder만 맞추는 것이 아니라 상품 이미지 분포까지 adaptation한다.
2. `왜 BM25인가`: 한국어 상품명/브랜드/설명에는 색상, 옷 종류, 그래픽 문구 같은 lexical clue가 남아 있다.
3. `왜 candidate-preserving인가`: BM25를 후보 확장으로 쓰면 lexical false positive가 들어와 R@10을 손상할 수 있으므로, visual retriever가 만든 후보 내부에서만 순서를 조정한다.

## 3. 논리 흐름

논문 본문은 아래 순서로 쓰면 실험이 자연스럽게 이어진다.

| 단계 | 질문 | 실험 근거 | 결론 |
|---|---|---|---|
| 1 | pretrained/frozen VLM만으로 충분한가? | `Baseline-256`: R@1 0.533, R@10 0.867, MRR 0.624 | R@10은 나쁘지 않지만 first-rank quality가 부족하다. |
| 2 | text-side LoRA만으로 해결되는가? | `LoRA-256`: R@1/MRR 상승, R@10 하락 | query distribution 적응은 도움되지만 후보 recall trade-off가 생긴다. |
| 3 | 해상도/loss/epoch 조정만으로 충분한가? | LoRA-384, InfoNCE20, Sigmoid20 | 단순 학습 설정 변경은 안정적인 개선을 만들지 못한다. |
| 4 | metadata rerank가 필요한가? | `LoRA-384 + BM25 top-10`: R@1 0.700, R@10 0.900, MRR 0.756 | lexical metadata는 top-rank precision에 강하다. |
| 5 | 더 높은 novelty의 generator가 가능한가? | dual-tower LoRA r4/r8/r16 | image/text encoder를 함께 적응시키면 strong candidate generator를 만들 수 있다. |
| 6 | low-rank dual adaptation의 안정점은 어디인가? | r4 5ep, r4 10ep, r8, r16, r8 15ep | r4 10epoch가 R@5/R@10을 회복하며 가장 안정적인 후보 생성기다. |
| 7 | final rerank는 어떻게 결합해야 하는가? | `Dual r4 10ep + BM25`: R@1 0.700, R@10 0.900, MRR 0.757 | BM25를 top-20 후보 내부에만 적용하면 R@1/MRR이 오르고 R@10이 유지된다. |
| 8 | 더 복잡한 대안은 필요한가? | learned reranker, adaptive BM25, union, MaxSim, query adapter | 현재 데이터 규모에서는 복잡한 변형이 일반화되지 않거나 후보 recall을 해친다. |

이 흐름은 “성공한 최종 방법”만 보여주는 것이 아니라, 왜 단순 LoRA, 왜 단순 BM25, 왜 learned reranker가 아닌지까지 설명한다. 그래서 간이 논문에서 납득 가능한 ablation story로 쓸 수 있다.

## 4. 최종 방법을 선택한 이유

최종 방법은 `Dual r4 10ep + BM25`다.

| Method | R@1 | R@5 | R@10 | MRR | 해석 |
|---|---:|---:|---:|---:|---|
| Random ranking expected | 0.002 | 0.009 | 0.017 | 0.012 | 무작위 하한선 |
| Baseline-256 | 0.533 | 0.733 | 0.867 | 0.624 | frozen VLM 기준선 |
| LoRA-256 | 0.633 | 0.833 | 0.833 | 0.712 | R@1/MRR 상승, R@10 하락 |
| LoRA-384 + BM25 top-10 | 0.700 | 0.833 | 0.900 | 0.756 | 강한 lexical rerank baseline |
| Dual r4 InfoNCE 10ep | 0.633 | 0.867 | 0.900 | 0.726 | 후보 recall은 좋지만 top-rank precision 부족 |
| Dual r8 InfoNCE 5ep | 0.633 | 0.867 | 0.900 | 0.723 | R@5는 높지만 MRR은 final보다 낮음 |
| **Dual r4 10ep + BM25** | **0.700** | **0.833** | **0.900** | **0.757** | 최종 balanced strategy |

선택 논리는 다음과 같다.

- final은 random과 Baseline-256보다 모든 metric이 높다.
- final은 LoRA-256보다 R@1, R@10, MRR이 높고 R@5는 동률이다.
- final은 LoRA-384+BM25와 R@1/R@5/R@10이 같고 MRR이 아주 소폭 높다.
- final은 dual-only r4 10epoch보다 R@1과 MRR이 높고 R@10을 유지한다.
- R@5만 보면 dual-only r4/r8이 더 높지만, final은 R@1/R@10/MRR의 균형이 더 좋다.

따라서 final을 “모든 metric에서 압도적 best”라고 쓰면 안 된다. 대신 **R@1/R@10을 동시에 보존하고 MRR이 가장 높은 balanced ranker**라고 쓰는 것이 정확하다.

## 5. Claim Strength Audit

| 주장 | 강도 | 논문에서의 권장 표현 |
|---|---|---|
| final은 random baseline보다 압도적으로 좋다 | 강함 | random ranking expected 대비 모든 retrieval metric에서 큰 차이를 보인다. |
| final은 frozen SigLIP baseline보다 좋다 | 강함 | Baseline-256 대비 R@1/R@5/R@10/MRR을 모두 개선한다. |
| candidate-preserving BM25는 top-rank precision에 도움이 된다 | 강함 | dual r4 10epoch에 BM25를 결합하면 R@1과 MRR이 상승하고 R@10은 유지된다. |
| 후보 보존 제약이 중요하다 | 중간-강함 | union/adaptive 계열은 R@10을 손상했으므로, 현재 task에서는 후보 내부 rerank가 더 안정적이다. |
| dual-tower adaptation은 text-only보다 연구적으로 더 새롭다 | 강함 | image encoder까지 domain-adapt하므로 구조적 novelty가 높다. |
| dual-tower final이 LoRA-384+BM25보다 실질적으로 크게 좋다 | 약함 | MRR 차이가 약 0.001이라 큰 성능 향상으로 주장하면 안 된다. |
| final이 모든 metric에서 strict best다 | 불가능 | R@5는 dual-only 계열이 더 높다. |
| 결과가 일반 한국어 패션 검색 전체로 일반화된다 | 약함 | eval query가 30개라 외부 일반화 claim은 제한해야 한다. |
| learned reranker는 항상 나쁘다 | 불가능 | 현재 데이터 규모와 validation protocol에서는 일반화 실패했다고만 말해야 한다. |

## 6. 채워진 실험 맥락

기존 문서에는 수치가 많지만, 왜 그 실험들이 필요했는지는 흩어져 있었다. 논문에서는 아래처럼 연결하면 된다.

### 6.1 Baseline과 text-only LoRA

Baseline-256은 pretrained SigLIP 계열의 zero/frozen retrieval capacity를 보여준다. R@10이 0.867이므로 완전히 실패한 baseline은 아니다. 그러나 R@1 0.533과 MRR 0.624는 사용자가 원하는 상품을 첫 화면 상단에서 찾는 품질로는 부족하다.

LoRA-256은 text-side adaptation의 이득을 보여준다. R@1과 MRR은 크게 오르지만 R@10이 0.833으로 내려간다. 이것은 query distribution에 맞추는 과정이 top-rank precision을 개선하는 대신 후보 다양성 또는 recall을 희생할 수 있다는 신호다.

### 6.2 LoRA-384와 BM25

LoRA-384는 R@10을 0.900까지 회복하지만 R@1/MRR이 LoRA-256보다 낮다. 따라서 단순 해상도 증가나 checkpoint 변경만으로는 해결되지 않는다.

BM25 rerank는 이 약점을 보완한다. 패션 상품 metadata에는 색상, 아이템 종류, 로고/프린트, 브랜드, 상품명 표현이 직접 남아 있고, 한국어 쿼리도 이런 단어 또는 부분 문자열을 많이 포함한다. 그래서 char n-gram 기반 BM25는 semantic image embedding이 놓친 lexical clue를 top 후보 내부에서 보정한다.

### 6.3 Dual-tower LoRA

text-only LoRA는 image embedding을 frozen cache로 둔다. 반면 dual-tower LoRA는 text encoder와 vision encoder를 모두 task distribution에 맞춘다. 이것이 이번 실험 묶음에서 가장 중요한 novelty 축이다.

r4 5epoch는 R@1/MRR이 좋지만 R@10이 낮다. r4 10epoch는 R@5/R@10을 회복하므로 후보 생성기로 더 안전하다. r8은 R@5가 높지만 MRR이 낮고, r16은 capacity 증가가 R@10 손상을 만든다. r8 15epoch는 장기 학습이 오히려 악화한다. 이 흐름은 작은 데이터 규모에서 low-rank와 epoch를 과하게 키우면 일반화가 흔들릴 수 있음을 보여준다.

### 6.4 Candidate-preserving rerank

BM25를 전체 gallery에서 후보 확장으로 쓰는 것이 아니라, dual retriever의 top-20 안에서만 적용한 이유가 중요하다. query와 metadata의 문자 일치는 강한 신호지만, image relevance를 보장하지는 않는다. Union 후보 확장은 lexical false positive를 끌어들일 수 있고 실제로 R@10이 내려갔다. 따라서 final은 “lexical retrieval”이 아니라 **visual-semantic candidate generation 후 lexical reranking**이다.

### 6.5 실패 실험의 논문적 가치

실패 실험도 논문 서사에 가치가 있다.

- learned reranker는 validation에서는 좋아 보였지만 locked eval에서 붕괴했다. 작은 train/eval 규모에서 feature-based reranker가 overfit될 수 있음을 보여준다.
- patch MaxSim은 ColBERT-style late interaction 아이디어를 시험했지만 개선이 없었다. patch-token alignment를 직접 학습하지 않은 hidden feature 재활용만으로는 부족할 수 있다.
- query adapter는 query-side만 보정해서는 image/text 양쪽의 domain mismatch를 해결하기 어렵다는 근거다.
- adaptive BM25는 query 길이별 weight selection이 validation에는 맞지만 locked eval에는 일반화되지 않았다는 사례다.
- query expansion은 한국어 패션 단어를 규칙으로 보강하는 단순 전처리가 현재 retrieval model보다 나은 signal을 만들지 못했음을 보여준다.

이 실패들은 “무작정 복잡한 모델을 올렸다”가 아니라, 가능한 novelty 방향을 시도하고 데이터 규모에서 방어 가능한 단순 결합으로 돌아왔다는 연구 흐름을 만든다.

## 7. 권장 논문 구조

간이 논문은 아래 구조가 가장 자연스럽다.

| Section | 핵심 내용 |
|---|---|
| Abstract | 한국어 패션 이미지 검색 문제, dual-tower LoRA + candidate-preserving BM25, 핵심 결과와 한계 |
| Introduction | 한국어 쿼리의 색상/위치/그래픽 단서, pretrained VLM의 한계, metadata의 필요성 |
| Task and Data | Musinsa 상품 이미지, train pairs, locked 30-query eval, gallery 588, difficulty split |
| Method | SigLIP2 backbone, dual-tower LoRA, InfoNCE, BM25 rerank, top-20 candidate preservation |
| Experiments | random, Baseline-256, text-only LoRA, LoRA+BM25, dual-only, final comparison |
| Ablation | rank/epoch, BM25, 후보 보존, learned/adaptive/MaxSim/query adapter 실패 |
| Discussion | R@5와 MRR trade-off, metadata 의존성, small eval, split overlap caveat |
| Conclusion | balanced retrieval quality 개선과 후속 검증 필요 |

## 8. 초록 초안

한국어 패션 이미지 검색에서는 사용자가 색상, 의류 종류, 그래픽 위치, 브랜드 또는 상품명 단서를 구어적으로 조합해 질의하는 경우가 많다. 본 연구는 SigLIP2 기반 vision-language retrieval을 한국어 패션 상품 검색에 맞추기 위해 text encoder와 vision encoder 양쪽에 low-rank adaptation을 적용하고, 검색된 top 후보 내부에서 상품 메타데이터 기반 Korean char n-gram BM25 점수를 결합하는 2-stage retrieval 방법을 제안한다. 588개 test gallery와 30개 hand-written locked query로 평가한 결과, 제안 방법은 random ranking 및 frozen SigLIP baseline 대비 R@1, R@5, R@10, MRR을 모두 개선했으며, text-only LoRA 및 여러 reranking ablation과 비교해 R@1/R@10을 보존하면서 가장 높은 MRR을 보였다. 추가 실험에서는 learned reranker, query adapter, patch-level MaxSim, 후보 확장형 BM25가 현재 데이터 규모에서 안정적으로 일반화되지 않음을 확인했다. 다만 평가 query 수가 작고 strong lexical rerank baseline과의 MRR 차이는 작기 때문에, 본 결과는 대규모 일반화 claim보다는 한국어 패션 검색에서 dual-tower adaptation과 후보 보존형 lexical reranking의 유효성을 보이는 간이 실험으로 해석되어야 한다.

## 9. Reviewer Risk Checklist

| 예상 질문 | 현재 답변 | 추가 보완하면 좋은 것 |
|---|---|---|
| eval query가 30개뿐인데 충분한가? | 간이 논문으로는 가능하지만 strong claim은 제한한다. | 100~300개 unseen hand-written query 추가, bootstrap confidence interval |
| final이 LoRA+BM25보다 정말 좋은가? | R@1/R@5/R@10은 같고 MRR만 소폭 높다. novelty와 balanced quality 중심으로 주장한다. | 새 eval set에서 재검증 |
| BM25가 상품 설명을 너무 많이 이용하는 것 아닌가? | 이 task는 image-only가 아니라 product image retrieval이며 metadata rerank를 명시한다. | name-only, brand 제거, generated description 제거 ablation |
| split overlap은 문제 아닌가? | eval positive는 train pair에 없고 방어 코드가 추가되어 있다. | deduped split으로 full rerun |
| learned reranker가 약한 것은 implementation 문제 아닌가? | 가능하다. 현재 결론은 데이터 규모/validation setup에서 일반화 실패했다는 제한된 주장이다. | 더 큰 validation, cross-validation, stronger negative mining |
| dual-tower live encoding은 실서비스 비용이 큰가? | 실험 목적상 품질 검증을 우선했다. | test image embedding cache 재생성 또는 adapter-merged image index 구축 |

## 10. 지금 당장 논문에 넣어도 되는 문장

아래 문장은 현재 결과로 방어 가능하다.

> The proposed dual-tower LoRA + candidate-preserving BM25 reranker improves all metrics over the frozen SigLIP baseline and achieves the highest MRR among the tested methods, while maintaining R@10 at 0.900.

> BM25 is most effective when used as a reranker over visually retrieved candidates rather than as an independent candidate expansion mechanism.

> The results suggest that Korean fashion retrieval benefits from both visual-language domain adaptation and lexical product metadata, but the small locked evaluation set prevents strong statistical claims.

아래 문장은 쓰지 않는 편이 좋다.

> Our method significantly outperforms all baselines.

> Our method is best on every metric.

> Learned rerankers are unsuitable for Korean fashion retrieval.

> The method generalizes to all Korean fashion search scenarios.

## 11. 남은 공백

논문 작성 자체를 막는 공백은 거의 없다. 다만 논문을 더 강하게 만들려면 아래를 추가하면 좋다.

1. `deduped split`을 새로 만들고 final 계열을 full rerun한다.
2. locked eval을 최소 100개 이상으로 늘린다.
3. bootstrap confidence interval 또는 query-level win/loss table을 넣는다.
4. metadata ablation을 추가한다: brand 제거, product name only, description only, no metadata.
5. qualitative case study를 넣는다: final이 맞춘 query, BM25가 올린 query, BM25가 망친 query.

하지만 현재 사용자의 목표가 “간이 논문”이라면 위 항목은 필수라기보다 강화 옵션이다. 지금 문서 세트만으로도 문제-방법-실험-해석-한계가 논리적으로 이어지는 초안 작성은 가능하다.
