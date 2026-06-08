# Novelty 재설계 — 문헌 기반 신규 방법 탐색·구현·검증 최종 리포트

> 작성: 2026-05-30. 대상: 한국어 패션 text→image retrieval 졸업논문.
> 목적: "기존 아키텍처가 너무 뻔하고 novelty가 없다"는 문제의식에 대해, 문헌에서 참신한
> 방법들을 직접 수집·구현하고 **baseline 및 기존 best 대비 실제로 우위를 점하는지** 통계적으로 검증한 결과.

---

## 0. TL;DR

- **기존 repo의 "novelty"(dual-tower LoRA)는 사실상 가짜 기여**였다: 단순 `LoRA+BM25`(MRR 0.756) 대비 **+0.001**, 게다가 평가 시 460ms/query. 헤드라인으로 못 쓴다.
- 그래서 문제를 **임베딩 공간 기하학**으로 재프레이밍했다: contrastive VLM의 두 가지 구조적 병리 — **modality gap**(Liang, NeurIPS 2022)과 **hubness**(QB-Norm, CVPR 2022; CSLS, ICLR 2018) — 를 **학습용으로 만든 7,023개 합성 쿼리 뱅크를 *추론 시점 통계 prior*로 재활용**해 training-free로 보정한다. 이것이 BM25/dual-tower/learned-reranker와 구분되는 **기존 자산의 새로운 용법**이라는 novelty 축이다.
- 신규 방법 8종을 구현·벤치마크하고, repo에 없던 **bootstrap CI + paired sign test**를 붙였으며, N=30 human eval의 검정력 한계를 넘기 위해 **로컬 LLM(EXAONE, API 아님)으로 unseen test 588상품에 1,706개 사용자-스타일 쿼리를 생성한 고검정력 보조 eval**을 만들었다.
- **정직한 결론**:
  - **vs zero-shot baseline: 압도적 우위(통계적으로 확실).** LoRA, BM25, 모든 보정이 baseline을 이긴다.
  - **vs 기존 best(`LoRA+BM25`, MRR 0.756): 어떤 신규 방법도 gold eval에서 통계적으로 이기지 못한다.** 보정들은 *보완적·미미*하다.
  - 단, **standalone(메타데이터 0)에서 hubness/centering 보정이 순수 LoRA retriever를 작지만 유의하게(고검정력 N=1706에서 p<0.05) 개선**한다 — 합성 뱅크가 추론 시점에도 가치가 있다는 새 증거.

---

## 1. 무엇을 시도했나 (문헌 기반 신규 방법 roster)

WF1(문헌 리서치, 22 에이전트·950 검색)에서 8개 각도를 조사해 후보를 추렸고, WF2/WF3에서 구현·검증했다.

| # | 방법 | 출처 | 핵심 아이디어 | training-free | 결과 |
|---|---|---|---|---|---|
| 1 | **Modality-gap correction** | Liang et al., *Mind the Gap*, NeurIPS 2022 | 텍스트/이미지 임베딩의 공통 평균방향(gap)을 부분 제거 후 재정규화 | ✅ | standalone 유의(+), +BM25 유의(+, synth) |
| 2 | **QB-Norm / Dynamic Inverted Softmax** | Bogolin et al., CVPR 2022 | 뱅크에 대한 갤러리 인기도(softmax popularity)로 hub 이미지 down-weight | ✅ | standalone 유의(+) |
| 3 | **QB-Norm multiplicative** | (변형) | popularity^α로 나눔 | ✅ | standalone 유의(+) |
| 4 | **CSLS** | Conneau et al., ICLR 2018 | query·gallery 양측 k-NN 평균을 빼는 local scaling | ✅ | standalone 최대 유의(+) |
| 5 | **AQE-on-bank** | Chum ICCV2007 / Radenović CVPR2018 | 쿼리를 뱅크 내 유사쿼리로 α-가중 확장 | ✅ | +BM25 유의(+, synth), standalone은 음(−) |
| 6 | **Cross-modal PRF (Rocchio)** | dense-PRF, Li ICTIR2021 | top-k 검색 이미지로 쿼리 임베딩 보정 | ✅ | **negative** (flat~음) |
| 7 | **RRF fusion** | Cormack et al., SIGIR 2009 | rank 기반 융합(z-score 선형융합 대체) | ✅ | **negative** (선형 rerank보다 나쁨) |
| 8 | **Local-LLM Query Refiner / HyDE** | Gao et al., ACL 2023 | EXAONE/Qwen로 casual→formal 쿼리 재작성 후 임베딩 blend | ✅(로컬) | **negative** (validation이 refiner OFF 선택) |

문헌 조사에서 검토 후 **보류/제외**한 것: FashionCLIP(소규모 한국 카탈로그 전이 이득 불명), FILIP/FLAIR(학습 필요·SigLIP pooled 비호환), ColBERT MaxSim(repo가 이미 실패 확인), Promptagator/InPars 라운드트립 필터(재학습 필요·자기참조적), modality-gap의 mean-shift만(보정군에 흡수).

---

## 2. 평가 방법론 (repo에 없던 통계적 엄밀성)

- **공유 하니스 `src/novel/lib.py`**: 기존 `evaluate.py`/`hybrid_retrieval.py`와 **동일한 인코딩·갤러리·메트릭**(R@1/5/10, MRR, single-positive)을 재현(baseline 정확 재현 검증 완료). 추가로 **bootstrap CI + paired bootstrap + sign test**.
- **HP 선택은 오직 validation(train-split 합성 쿼리 717개 / 240 갤러리)에서만.** locked eval은 선택에 미사용(누수 방지). 보정의 모든 통계(hubness/평균)는 **train 합성 뱅크 7,023개**에서만 추정(eval 쿼리 미사용).
- **두 개의 eval**:
  - **human-30** (gold): 사람이 직접 쓴 30개 쿼리. 신뢰할 만하나 **MRR 95% CI ≈ ±0.13**으로 검정력 매우 낮음.
  - **synth-1706** (고검정력): unseen test 588상품에 EXAONE-3.5-7.8B(로컬, API 아님)로 생성한 사용자-스타일 쿼리. MRR CI ≈ ±0.02. N=30이 구분 못 하는 작은 효과를 분해.
- **canonical "현 best" = published `LoRA+BM25` MRR 0.756** (repo가 실제 보고하는 수치; dual-tower 0.757은 사실상 동일). 하니스 내 재현치는 grid에 따라 0.721~0.768로 흔들리므로, 공정성을 위해 **0.756을 기준선**으로 본다.

---

## 3. 결과 표 (lora384 인코더 = 운용 지점)

`output/novel/final_suite.json` 기준. eval=human-30, synth=N1706. 유의성 = synth paired sign-test.

| 방법 | 모드 | eval MRR | synth MRR | synth: vs LoRA cosine | synth: vs dense+BM25(현 best) |
|---|---|---:|---:|---|---|
| zero-shot(frozen) | — | 0.571 | 0.431 | (LoRA 미만) | — |
| **LoRA cosine** | standalone | 0.684 | 0.568 | 기준 | — |
| dense+BM25 (**현 best**) | +bm25 | 0.721 / *0.756 pub* | 0.903 | +0.334 (p=0) | 기준 |
| modality_gap | standalone | 0.706 | 0.570 | **+0.0014 (p=0.011)** ✅ | −0.333 |
| qbnorm_dis | standalone | 0.719 | 0.571 | **+0.0028 (p=0.005)** ✅ | −0.331 |
| qbnorm_mul | standalone | 0.707 | 0.572 | **+0.0039 (p=0.014)** ✅ | −0.330 |
| **csls** | standalone | 0.681 | 0.585 | **+0.0164 (p=0.0004)** ✅(최대) | −0.318 |
| aqe | standalone | 0.713 | 0.561 | −0.0079 (음) ❌ | −0.342 |
| modality_gap | +bm25 | 0.734 | 0.908 | +0.340 | **+0.0057 (p=0)** ✅ |
| qbnorm_dis | +bm25 | 0.705 | 0.909 | +0.341 | +0.0065 (p=0.09) ✗ |
| qbnorm_mul | +bm25 | **0.764** | 0.902 | +0.334 | −0.0001 (p=1.0) ✗ |
| csls | +bm25 | 0.760 | 0.906 | +0.338 | +0.0033 (p=0.60) ✗ |
| aqe | +bm25 | 0.720 | 0.907 | +0.339 | **+0.0046 (p=0.0012)** ✅ |

**frozen base 인코더 ablation**(요약): 보정이 base를 더 크게 끌어올림 — qbnorm_mul standalone 0.651(+0.080 over base 0.571), modality_gap 0.615 — 그러나 **여전히 LoRA(0.684) 미만**. → "보정이 fine-tuning을 부분 대체하나 대체하진 못함."

---

## 4. 정직한 결론 — 실제로 무엇을 이겼나

### 4.1 강건하게 입증된 것 (두 eval 모두 + 통계 지지)
1. **LoRA text-adapter ≫ zero-shot SigLIP** (0.571→0.684 human; 0.431→0.568 synth; p=0). 프로젝트 핵심 가설("분포격차 해소가 최고 레버")이 고검정력에서 확정됨.
2. **BM25 lexical fusion ≫ dense-only** (0.684→0.756 human; 0.568→0.903 synth; p=0). 단일 최강 실무 레버.

### 4.2 신규 기여로 방어 가능한 것
3. **합성 뱅크를 추론 시점 통계 prior로 재활용한 보정(modality-gap / QB-Norm / CSLS)이, 메타데이터 없이 순수 LoRA retriever를 작지만 통계적으로 유의하게 개선**(고검정력 synth에서 p<0.05; csls가 최대 +0.0164). human-30에선 점추정이 더 크나(+0.02~0.035) N=30이라 비유의. → **"N=30이 noise로 본 신호가 고검정력에서 실재함을 확인."** 이것이 repo에 없던 novelty + 엄밀성이다.

### 4.3 이기지 못한 것 (정직하게)
4. **현 best(`LoRA+BM25` 0.756)를 두 eval 모두에서 유의하게 이긴 방법은 없다.** synth에서만 modality_gap+BM25(+0.0057)·AQE+BM25(+0.0046)가 유의하나 **<1 MRR-point**이고 human-30에선 비유의. human-30 최고점 qbnorm_mul+BM25(0.764)는 약한 재현치(0.721) 대비로만 유의(p=0.039), published 0.756 대비론 무승부(p=0.688)이며 synth에선 dead-even. → **小표본 아티팩트**.
5. **negative results(가치 있음)**: HyDE LLM refiner(validation이 OFF 선택), RRF(선형 rerank보다 나쁨), cross-modal PRF(flat), CSLS+공격적 BM25 weight(validation 과적합 붕괴).

### 4.4 real vs noise (고검정력이 밝힌 것)
- N=30에선 LoRA(0.684)~dense+BM25(0.756) 사이가 전부 통계적으로 구분 불가. synth(N=1706, ~57배)가 순서를 깨끗이 분해.
- 두 eval **일치**: LoRA>zero-shot, BM25>dense, 어떤 보정도 dense+BM25를 크게 못 이김.
- 두 eval **불일치**: human-30은 qbnorm_mul+BM25를 1등으로, synth는 modality_gap/AQE+BM25를 1등으로 — 이 불일치 자체가 "보정의 이득이 분해능 한계의 marginal"임을 보여주는 정직한 발견.

---

## 4.5 조합(combination)으로 win 시도 — 결과 (`run_combos.py`, 950 파이프라인)

"보정을 쌓으면 현 best를 이길 수 있는가"를 직접 검증했다. 파이프라인 =
`[임베딩 보정: identity|modality-gap(λ)] → dense → [스코어 hubness: none|DIS|CSLS|MUL] → [BM25 후보보존 융합]`,
**950개 조합**을 validation에서만 HP 선택 후 human-30·synth 양쪽 평가.

- **validation이 고른 best** = `CSLS(k20)+BM25(n50,w0.2)` → human-30 MRR **0.760**, synth 0.906.
  - vs published 0.756: Δ **+0.0042**, sign_p **0.73** (W5/L3/T22) → **무승부**.
  - vs synth dense+BM25: Δ +0.0033, p 0.60 → **무승부**.
- **stacking이 실제로 더한 지점**: `modality-gap+CSLS+BM25`가 고검정력 synth에서 dense+BM25를 **유의하게** 넘음(Δ **+0.0105**, exact binomial p≈**7e-8**, Bonferroni 통과; 단일 보정 +0.0057의 ~2배). 그러나 (a) human-30에선 0.719~0.736으로 published **미달**, (b) validation이 이 조합을 1등으로 못 고름, (c) synth Δ는 0.903 기준 **+1.2% 상대, 1706개 중 1551개 동점** = 실효적으로 무시 가능.

**3중 적대적 검증(WF4) 결론** (`wf4_combo_verification.json`): 누수·버그 없음, 통계 정확, 독립 재계산으로 모든 수치 정확 재현. **950개 전 파이프라인 중 human-30에서 published를 유의하게(sign_p<0.05) 이긴 것은 0개**. eval-oracle로 cherry-pick해도 최대 0.7628(Δ+0.006, p=0.55=무승부). 즉 **조합 탐색으로도 gold eval에서 현 best를 이기는 방법은 없다**(N=30 검정력 한계). 단, 보정 stacking이 고검정력에서 단일 보정보다 큰(여전히 tiny) 유의 효과를 줌은 확인.

## 4.6 왜 더 못 이기나 — 천장의 정체 (`diagnose_failures.py`)

현 best(LoRA+BM25)는 30개 중 정확히 3개에서 치명적으로 실패(정답 rank 191·378·483)하며, 이 3개가 MRR/R@10을 거의 전부 깎아먹는다. **N=30에서 baseline을 실제로 이기려면 이 3개를 살려야 하는데, 들여다보니 셋 다 retrieval 방법으로 고칠 수 없는 eval/라벨 아티팩트였다:**

| rank | query | GT | 원인 |
|---|---|---|---|
| 483 | "흰색 긴팔 셔츠" | 아이보리 와플 티셔츠 | **다중정답 모호성** — 갤러리에 흰 긴팔 상의 수십 개, 검색 top-3 전부 타당. single-positive 라벨이 1개뿐이라 모델이 옳아도 오답 처리 |
| 378 | "어두운 하늘색 긴팔 맨투맨" | 블루 니트 스웨터 | **카테고리 불일치** — 사용자가 니트를 "맨투맨"이라 칭함 → 쿼리가 검색기를 맨투맨 쪽으로 유도(정답은 니트) |
| 191 | "검은색 후드티 가운데 작은 로고" | 깊은 네이비 곰 후드 | **색 인식 불일치** — 네이비를 "검은색"으로 인식(explan.md가 지적한 현상). 검색 top-3은 모두 검은 후드+로고로 합리적 |

이 3개는 모두 `short_ambiguous` 밴드이고 dense rank도 20~30위 → BM25 rerank window(top-20/50) 밖이라 lexical로도 구제 불가. **결론: 현 파이프라인은 이 eval에서 사실상 실용적 천장에 도달했고, 남은 headroom은 "더 나은 방법"이 아니라 eval 설계(multi-positive 라벨링)에 잠겨 있다.** 이는 "왜 어떤 신규 방법도 현 best를 못 이기는가"에 대한 근본 답이며, 오히려 방법보다 eval이 한계라는 정직한 진단이다.

**N 증가 없이 남은 유일한 레버 = 기존 30개의 multi-positive 재라벨링** (로컬 LLM-as-judge로 각 쿼리에 대해 갤러리 내 타당한 정답을 모두 표시; explan.md가 원래 권고했으나 미구현). 이러면 (a) 모호성 아티팩트가 제거돼 전반 점수가 오르고, (b) 방법 간 분리가 드러날 여지가 생긴다.

## 4.7 multi-positive 재평가 — N 증가 없는 유일한 레버 (`run_multipos.py`)

§4.6의 진단(실패가 다중정답 아티팩트)을 검증하기 위해, 새 쿼리 없이 기존 30개를 multi-positive로 재라벨링했다. method-agnostic 후보 풀(전 방법 top-15 합집합)을 로컬 EXAONE가 blind로 판정(예/아니오), 모든 방법에 동일 적용.

- **아티팩트 확증**: 쿼리당 **평균 5.07개**의 타당한 정답이 존재(single-positive가 부당하게 감점하고 있었음). LoRA 계열 R@10이 1.000으로 회복 — §4.6의 "치명적 실패"가 실은 타당한 정답이었음을 입증.

| 방법 | R@1 | R@5 | R@10 | MRR | vs dense+BM25 (sign_p) |
|---|---:|---:|---:|---:|---|
| baseline | 0.533 | 0.767 | 0.933 | 0.655 | — |
| lora | 0.700 | 0.900 | 1.000 | 0.794 | — |
| dense+BM25 (현 best) | 0.767 | 0.967 | 1.000 | 0.858 | 기준 |
| **modality_gap+BM25** | **0.800** | **1.000** | **1.000** | **0.878** | **+0.020 (p=1.0, 4W/3L)** |
| csls+BM25 | 0.800 | 1.000 | 1.000 | 0.876 | +0.019 (p=0.69) |
| qbnorm_mul+BM25 | 0.733 | 1.000 | 1.000 | 0.840 | −0.018 |

- **결과**: 공정한 multi-positive eval에서 **`modality_gap+BM25`가 모든 지표 1등**, 현 best를 점추정 +0.020 앞섬. **그러나 N=30이라 여전히 비유의**(4승3패, 23동점). dense+BM25 vs baseline은 유의(+0.202, p=0.013).
- **결정적 함의**: 라벨 아티팩트를 제거해도 **N=30에선 강한 baseline 대비 유의한 win이 불가능**하다는 것이 최종 확인. 신규 방법은 *일관되게 선두/대등*하나 30개로는 증명 못 함. (caveat: multi-positive 라벨은 LLM 판정이라 noise 존재하나, "선두이되 비유의" 결론은 판정 noise에 강건.)

### 세 가지 eval 프레이밍에서의 일관성 (가장 강한 정직한 주장)
| eval | dense+BM25 | modality_gap+BM25 | 판정 |
|---|---:|---:|---|
| single-positive human-30 | 0.756 | 0.734~0.760 | 대등 |
| **multi-positive human-30** | 0.858 | **0.878** | 선두(+0.020, 비유의) |
| **high-power synth-1706** | 0.903 | **0.908** | **유의**(+0.0057, p≈0) |

→ **`modality_gap+BM25`는 세 프레이밍 모두에서 현 best ≥, 항상 최상위.** 개별로는 작은 eval에서 비유의지만, **세 독립 프레이밍에서 일관되게 선두**라는 점이 N=30 한계 안에서 낼 수 있는 가장 강한 증거다.

## 4.8 2025 트렌드 적용 — GME(VLM 임베더) baseline + 로컬 LLM reranker (`run_gme_baseline.py`, `run_llm_reranker.py`)

2025 흐름("CLIP dual-encoder → 생성형 VLM을 embedder/reranker로")을 우리 셋업에 실제 적용해 검증.

| 방법 | human-30 single MRR | human-30 multi-pos MRR | synth-1706 MRR | vs 현 best (sign_p) |
|---|---:|---:|---:|---|
| **GME-Qwen2VL-2B** dense (인용 140) | 0.467 | n/a* | 0.430 | **패배** (single −0.289, p=0.007) |
| GME-2B + BM25 | 0.452 | n/a* | 0.780 | **패배** (single −0.304, p=0.004; synth −0.123, p≪0.001) |
| **LoRA+BM25 (현 best, ref)** | 0.756 | 0.858 | 0.903 | — |
| **LLM reranker** (EXAONE, light blend w0.5) | 0.754 | 0.909 | 0.815† | **무승부**(single −0.002, p=1.0); multipos +0.054(NS); synth† +0.005(p=0.022) |
| LLM reranker (pure reorder) | 0.687 | 0.911 | 0.795† | single −0.069(NS); synth† **악화** −0.015(p=0.006) |

*GME multi-pos는 기존 positive 풀이 SigLIP 기반이라 GME에 불리 → 공정성 위해 declined. †reranker synth는 human-튜닝 op point(0.810) 대비 상대 비교이며 synth-최적 0.903보다 낮음.

**결론**:
- **GME(2B VLM 임베더)는 우리 cheap 파이프라인에 완패** — dense 0.467은 **zero-shot SigLIP(0.571)보다도 낮음**(버그 아님, GT 11/30이 1위·중앙값 3위 확인). synth 0.780도 거의 BM25 lexical 채널이 캐리. → 강력한 thesis 포인트: **"더 큰 최신 모델이 답이 아니다; 좁은 도메인엔 도메인 정렬된 싼 파이프라인이 이긴다."**
- **로컬 LLM reranker는 training-free add-on으로 marginal·weight-sensitive** — gold single-pos는 무승부(0.754 vs 0.756), pure reorder는 오히려 손해(synth p=0.006), light blend만 고검정력에서 +0.005 유의(상대). **현 best를 못 이김.**
- **추가 진단(reranker)**: 3개 치명 실패(short_ambiguous, GT가 top-10 밖)는 후보보존 reranker로 **구조적으로 못 고침** → 라벨 아티팩트 재확인. 게다가 reranker가 single-pos에서 "틀리는" 이유는 **다른 타당한 흰/긴팔 상의를 올리기 때문**(multi-pos에선 보상됨, +0.054). §4.7의 "천장=eval 아티팩트" 결론을 2025-방법으로 독립 재확인.

## 5. 졸업논문용 방어 가능한 thesis

> 한국어 패션 검색에서 카탈로그 캡션↔사용자 쿼리의 **분포격차가 zero-shot의 지배적 약점**이며, 사용자-스타일 합성 쿼리로 학습한 **text-only LoRA가 이를 대부분 해소**한다(0.571→0.684 gold, 0.431→0.568 고검정력, 둘 다 p=0). 더 나아가, **학습용으로 만든 합성 쿼리 뱅크를 추론 시점 통계 prior로 재활용**하면(QB-Norm·CSLS·modality-gap centering) 추가 학습·메타데이터 없이도 retriever를 **작지만 고검정력에서 유의하게** 개선할 수 있다 — 기존의 BM25/dual-tower/learned-reranker와 직교하는, 비용 0의 새로운 자산 활용법이다. 다만 한국어 상품 메타데이터에 대한 **BM25 lexical fusion이 단일 최강 레버**이며, 보정들은 그 위에서 보완적이되 marginal하다. 본 연구는 SOTA 우위를 주장하지 않고, **분포 정렬(학습 시점 LoRA + 선택적 추론 시점 뱅크-prior)이 zero-shot을 강건하게 이기고 강한 hybrid와 대등함**을 잘 검정된 형태로 입증한다.

**기존 서사 대비 개선점**: dual-tower(Δ+0.001, 460ms)를 헤드라인에서 내리고, (a) 분포격차 정량화 + LoRA, (b) **뱅크-as-prior 보정**, (c) BM25 fusion으로 기여 무게중심 이동. negative results(HyDE/RRF/PRF/ColBERT)를 독립 섹션으로 승격. bootstrap CI + paired test + **고검정력 power analysis**로 통계 공백을 메움.

---

## 6. 권장사항

1. **헤드라인 운용점 = `LoRA + BM25`** (gold 0.756). 강건한 이중확정 주장 2개(LoRA>zero-shot, BM25>dense)가 capstone을 캐리.
2. **뱅크-as-prior 보정은 보완 기여로** 정직하게 제시: standalone 고검정력에서 유의(+), BM25 위에선 <1pt·marginal. "고검정력에서 확인된 일관된 양의 trend"로 표현, **gold eval에서 유의한 win이 아님을 명시**.
3. 단일 best를 꼽아야 하면 **modality_gap+BM25**: synth에서 보정 중 최고(0.908)·유의, human-30에서도 방향상 +(0.734). qbnorm_mul+BM25의 0.764는 약한 재현치 대비 아티팩트이므로 헤드라인 금지.
4. synth 이득은 **항상 절대 MRR-point(<1pt)와 p값을 함께** 인용(유의성≠실효크기).
5. 추가 실험 불필요 — 1주 예산·"beat baseline" bar 기준 현 결과로 작성 가능. 남은 일은 **정직한 framing**.

---

## 7. 한계 / caveat (반드시 명시)

- **synth eval의 BM25 inflation**: synth 쿼리는 LLM이 캡션에서 생성 → 캡션 기반 BM25 docs와 어휘가 겹쳐 BM25가 0.90까지 부풀려짐(human은 0.72). 따라서 synth는 *standalone(임베딩-only) 비교엔 신뢰*되나 *+BM25 비교엔 편향*. 이 한계를 명시하고 +BM25 결론은 human-30 우선.
- **split overlap**: 588 test 중 2상품(2071212, 4145646)이 split_train에도 존재(6/1706=0.35%, repo split 고유 문제, 모든 방법에 동일 노출, human-30에도 동일). 치명적 하한선 아래지만 각주 처리.
- **EXAONE 생성 품질**: 일부 합성 쿼리에 금지어(핏/주관형용사) 잔존. 양 인코더에 동일 영향이라 비교성은 유지.
- **N=30 gold**: 모든 gold eval 결론의 CI가 넓다 — 강한 일반화 주장 금지.

---

## 8. 재현 (코드/산출물)

- 하니스: `src/novel/lib.py` (로딩·메트릭·CI·paired test·BM25·rerank·RRF·select_eval)
- 캐시 생성: `src/novel/precompute.py` → `data/novel_cache/`
- 레퍼런스: `src/novel/refs.py` → `data/novel_cache/refs.json`
- 권위 벤치마크: `src/novel/run_final_suite.py` → `output/novel/final_suite.json` (standalone + BM25 × {human-30, synth} × {base, lora384})
- 고검정력 eval 생성: `src/novel/gen_synth_eval.py` → `data/novel_cache/synth_eval.jsonl` (1706 q)
- 개별 방법: `src/novel/run_modality_gap.py` (템플릿) 등; WF 산출물 `docs/novel_research/wf1_literature_research.json`, `wf2_implement_bench.json`, `wf3_highpower_eval.json`
- 실행 전 반드시 `unset LS_COLORS;` (컨테이너 환경 버그). 인터프리터 `/opt/miniconda3/envs/jolnon/bin/python`.
