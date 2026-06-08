# 연구 자문용 프로젝트 컨텍스트 (docs 보충 문서)

> **이 문서의 목적**: `docs/` 폴더의 모든 파일과 **함께** 다른 리서치 전문 LLM에게 전달하기 위한 보충 자료다.
> `docs/`에는 실험 내용·결과·ablation이 이미 상세히 들어 있다. 이 문서는 **docs에 없는 메타·맥락·코드 실체·문서 간 불일치·숨은 약점**만을 정리한다.
> docs를 읽을 때 반드시 이 문서를 같이 보고 **어떤 수치/주장이 canonical이고 어떤 것이 stale한지** 판단해야 한다.
>
> 작성 기준일: 2026-05-19. 저장소 경로: `/root/code/jolnon` (저장소명 `jolnon` = **졸업논문**).

---

## 0. 한 줄 요약 (가장 먼저 읽을 것)

이 프로젝트는 **학회 논문이 아니라 학부/석사 졸업논문**이며, 명시적 제약은 **"1주일 안에, 단일 A6000으로, 방어 가능한 수준에서 baseline을 이기는 것"**이다. 시간 예산이 신규성·엄밀성·공정성보다 우선한다. docs에는 결과가 풍부하게 정리돼 있으나, **(a) 코드/config/문서가 서로 불일치**하고, **(b) 헤드라인으로 내세운 dual-tower 방법의 실제 우위가 통계적으로 noise 수준**이며, **(c) 핵심 확장 실험 전체가 git에 커밋되지 않은 작업 트리 상태**다. 논문 자문 시 이 세 가지를 반드시 전제로 깔아야 한다.

---

## 1. 메타 컨텍스트 — 프로젝트를 지배하는 제약과 의사결정 철학 (docs에 거의 없음, 가장 중요)

저장소 루트 `CLAUDE.md`에 명시된 프로젝트 헌법이다. docs의 톤과 깊이를 결정한 근본 전제이므로 자문 LLM이 반드시 알아야 한다.

### 1.1 프로젝트 정체성
- **무엇인가**: 한국어 패션(무신사 상의 카테고리) **text→image retrieval** 단기 캡스톤/졸업논문 실험.
- **무엇이 아닌가**: top-tier 학회 제출 아님. 강한 의미의 연구 기여 아님. 신규성·포괄적 ablation·SOTA 대비 엄격한 공정성 **불필요**. 제품 아님(배포·UX·스케일 고려 없음).
- **합격 기준(bar)**: "우리 방법이 우리가 고른 baseline을, 우리가 고른 평가에서 이긴다"는 단순·방어 가능한 주장이면 충분.

### 1.2 하드 제약
- **시간**: 약 1주일, 가능하면 더 짧게. 제출까지의 시간 최소화가 다른 모든 것을 지배.
- **외부 API 예산: 없음. 모든 LLM 단계는 로컬 GPU에서 실행해야 함.** ← **그러나 실제로는 위반됨**(§3.2 참조: augmentation에 Upstage Solar-pro3 API 사용). 자문 시 이 제약 위반을 인지할 것.
- **단일 GPU**: A6000급(48GB) 한 장. 모든 설계가 이 안에서 편안해야 함.
- **소규모 데이터**: 상품 수천 개. 대규모 데이터 요구 방법은 선택지가 아님.

### 1.3 의사결정 철학 (방법론적 선택을 평가할 때 사용자의 우선순위)
1. **속도 > 엄밀성** (충돌 시, 명백히 터무니없거나 방법론적으로 깨지지만 않으면 속도 선택).
2. **단순 > 복잡** (ablation·조건·HP 스윕·움직이는 부품은 "baseline 이기기"에 직접 기여 안 하면 제거).
3. **방어 가능성 하한선**: 명백히 치명적 비판(학습 분포로 평가, 라벨 누수 등)만 피하면 됨. 그 위로는 빠른 길 선호.
4. **신규성은 목표가 아님**: 표준 레시피·known-good default 재사용을 영리한 아이디어보다 선호.
5. **상방 추구보다 리스크 감소**: 위험한 변형의 상방이 미미하면 안전한 변형 선택.

### 1.4 협업 방식
- 사용자는 **구체적 추천 + 짧은 근거**를 원함. 선택지 나열 메뉴를 원하지 않음.
- 방법론적 분기에서는 더 단순·빠른 변형을 추천하되, 방어 가능성 하한선을 넘으면 안 됨.
- 트레이드오프는 1~2줄로 제시하고 한쪽을 선택. hedge 금지.
- **프로젝트 논의의 작업 언어는 한국어.**

> **자문 LLM에게**: 이 프로젝트에 "더 많은 ablation을 하라", "여러 seed로 돌려라", "더 큰 데이터셋을 만들어라" 같은 일반적 학회-품질 조언을 그대로 주면 사용자 제약과 충돌한다. **이미 실행된 실험으로 쓸 수 있는 가장 방어 가능한 졸업논문을 어떻게 구성하느냐**가 실제 질문이다. 단, "거의 0비용으로 방어력을 크게 올리는 것"(예: 저장된 per-query rank로 bootstrap CI 계산 — 재학습 불필요)은 이 철학과 충돌하지 않으므로 적극 권장 가능.

---

## 2. 프로젝트의 3단계 진화사 (docs를 읽으면 혼란스러운 이유)

docs 안에 서로 다른 설계가 섞여 있는 이유는 프로젝트가 **세 번 크게 바뀌었기** 때문이다. 시간순으로 정리한다.

### 단계 A — 원본 전략 컨설팅 (`explan.md`, 저장소 루트, docs 아님, git 추적됨)
이전의 다른 리서치 LLM이 작성한 최초 전략 메모. **다음을 권고했고, 이 중 상당수는 최종 실행에서 폐기됨**:
- Backbone: SigLIP 2 base **patch16-256** (안전한 선택).
- 학습 시점 augmentation **+ 추론 시점 LLM query refiner** (양면 distribution alignment). → **refiner는 최종적으로 미구현/폐기**.
- 상품당 augmented 쿼리 **5~10개**. → 최종 **3개**.
- **30 epoch** 학습, **val split에서 best-epoch 선택**. → 최종 **10 epoch 고정, 마지막 epoch 사용, val-best-epoch 미구현**.
- 평가 셋 **50개**, **multi-positive 라벨링**. → 최종 **30개, single-positive**.
- Augmenter: 로컬 **EXAONE-3.5-7.8B / Qwen2.5-7B (vLLM)**. → 최종 **Upstage Solar-pro3 API**(로컬 아님, 제약 위반).
- 평가 조건 4개 단조 증가 서사(baseline1 < baseline2 < method1 < method2). → 최종 **2조건(zero-shot vs LoRA)**.
- 색 인식 오류(사용자가 베이지를 그레이로 부름)를 augmentation에 일부 반영 권고. → 미구현(향후 과제로만 남음).

### 단계 B — 확정 계획 (`docs/plan.md`, `docs/roadmap.md`)
- Backbone **patch16-256**, text-only LoRA, **sigmoid loss**, r=16, 10 epoch 고정.
- augmenter EXAONE via vLLM, 상품당 3 쿼리, **총 9,306쌍(3,102×3)**, split **train 2,481 / test 621**.
- 메인 메트릭 **R@10**, 평가 30개, single-positive.
- **이 계획의 수치(9,306 / 2,481 / 621)는 실제 실행과 다름**(§5 참조).

### 단계 C — 실제 메인 실행 (`docs/experiment_spec.md` §1–13)
- Backbone **patch16-256**, **text-only LoRA on cached frozen vision**, sigmoid, r=16, 10 epoch.
- augmenter **Solar-pro3 API**(EXAONE에서 교체), 상품당 3 쿼리, **train_pairs_final = 7,023쌍**, split **train 2,350 / test 588**.
- 평가 30 hand-written, single-positive, gallery 588.
- 결과: zero-shot 대비 **R@1 +0.10, R@5 +0.10, MRR +0.088, R@10 −0.033**. long_specific에서 큰 개선.

### 단계 D — 확장 캠페인 (`docs/experiment_spec.md` §14, `docs/final/*`, `docs/report_draft_ko.md`, `docs/successful_trials.md`, `docs/retrieval_results_summary.md`)
메인 서사 완료 후 추가로 진행된 대규모 캠페인:
- Backbone **patch16-384**로 업그레이드(보조 검증).
- **Dual-tower LoRA**(text+vision 둘 다 LoRA, cached vision 안 씀).
- **Candidate-preserving Korean char-ngram BM25 rerank**.
- **ColBERT-style late interaction (실패한 확장, negative result)**.
- 28개+ 방법 카탈로그, 7개 figure.
- **최종 best 주장**: `Dual r4 InfoNCE 10ep + top-20 BM25 rerank` → R@1 0.700 / R@5 0.833 / R@10 0.900 / MRR 0.757.

> **핵심**: docs의 §1–13은 단계 C(256, text-only), §14와 final/은 단계 D(384, dual-tower+BM25)다. **두 backbone(256 vs 384)의 수치가 한 문서 안에 섞여 있어** 표를 그대로 옮기면 심사자가 혼동한다. 어느 것을 canonical로 둘지는 §8 참조.

---

## 3. 문서 ↔ 코드 ↔ config 불일치 (자문 LLM이 반드시 알아야 할 함정)

docs만 보면 정합적으로 보이지만, 코드/config/제약과 대조하면 다음 불일치가 있다.

### 3.1 `config.yaml`은 stale하다 — docs/코드와 불일치
`config.yaml`은 현재 작업 트리 기준으로 **단계 D 일부만 반영하고 메인 서사와 충돌**한다:

| config.yaml 값 | 실제 메인 서사(단계 C) | 실제 최종 best(단계 D) |
|---|---|---|
| `backbone: patch16-384` | patch16-**256** | patch16-384 ✓(부분 일치) |
| `lora.r: 16` | r=16 ✓ | 최종 best는 **r=4** |
| `lora.apply_to: text` | text-only ✓ | 최종 best는 **dual(text+vision)** |
| `train.loss: sigmoid` | sigmoid ✓ | 최종 best는 **InfoNCE** |
| `train.epochs: 10` | 10 ✓ | 10 ✓ |
| 주석 `test gallery = 621 products` | 실제 **588** | 588 |
| `augment.provider: upstage_solar_pro3` | ✓ | ✓ |

→ **`config.yaml`은 단일 진실 원천(single source of truth)이라고 적혀 있지만 실제로는 아니다.** 최종 best 방법의 하이퍼파라미터(r=4, dual, InfoNCE, 10ep, BM25 weight 0.2, top-20)는 config가 아니라 **명령행 인자와 `dual_hybrid_rerank.py`/`train_dual_lora.py` 코드에 흩어져 있다.** 논문 재현 섹션은 config.yaml을 그대로 인용하면 안 된다.

### 3.2 "외부 API 금지" 제약 위반
- `CLAUDE.md`: "No external API budget. All LLM-based steps must run locally."
- 실제: `.env`에 `UPSTAGE_API_KEY` 존재, `config.yaml`의 `augment.base_url: https://api.upstage.ai/v1`, augmenter = **Solar-pro3 (Upstage 클라우드 API)**.
- 단계 B 계획은 로컬 EXAONE/Qwen via vLLM이었으나 사용자 요청으로 Solar-pro3 API로 교체됨(`experiment_spec §5.3`에 "사용자 요청"으로만 짧게 언급).
- **함의**: "로컬 전용"을 강점으로 쓰면 안 됨. augmentation 재현성은 외부 API·stochastic 디코딩에 의존(동일 seed로도 정확 재현 불가, `experiment_spec §12.4`도 인정).

### 3.3 plan.md 수치 ≠ 실제
- plan.md: 9,306 augmented pairs / split 2,481 train / 621 test.
- 실제: `data/train_pairs_final.jsonl` = **7,023줄** / split_train.json = **2,350** / split_test.json = **588**.
- 차이 원인: 이미지 다운로드 실패(2,938/3,102 성공) + URL 중복 dedupe(`/goods/`·`/products/` 동일 상품) + 파싱 실패 8개 제거. docs `experiment_spec §4`는 7,023을 맞게 적었으나 plan.md는 갱신 안 됨.

### 3.4 augmentation 규칙 개수 불일치
- `experiment_spec §4.5`: 프롬프트 규칙 **R1–R7**로 기술.
- 실제 `prompts/augment_final.txt`: **R1–R8** 존재(R8 = "상품 설명에 없는 정보 추측 금지"). 누락 보고. (사소하지만 논문에 프롬프트 전문을 부록으로 넣을 때 정정 필요.)

### 3.5 explan.md 권고 중 미구현 항목 (논문에서 "왜 안 했는지" 방어 필요)
multi-positive 라벨링, 추론 시점 query refiner, val-best-epoch 선택, 색 인식 오류 augmentation — 모두 **미구현**. `experiment_spec §10/§11`이 한계·향후과제로 일부 흡수했으나, 심사자가 explan.md를 본다면(전달 시) "초기 계획 대비 축소"로 보일 수 있음. 단계 A는 전달 대상 아님이면 무시 가능.

---

## 4. 코드베이스 실체 (docs에 없는 구현·방법론 디테일)

### 4.1 git 상태 — 확장 캠페인 전체가 미커밋
- **커밋은 단 1개**: `a37a979 initial commit: SigLIP 2 + text-only LoRA Korean fashion retrieval capstone`.
- 즉 **git에 박제된 "공식" 상태는 단계 C(256 text-only LoRA)뿐**이다.
- 단계 D 전체 — `docs/final/`, `docs/report_draft_ko.md`, `docs/successful_trials.md`, `docs/retrieval_results_summary.md`, `src/dual_hybrid_rerank.py`, `src/train_dual_lora.py`, `src/ensemble_dual_lora.py`, `src/hybrid_retrieval.py`, `src/patch_rerank.py`, `src/attribute_rerank.py`, `src/adaptive_hybrid_rerank.py`, `src/facet_hybrid_rerank.py`, `src/union_hybrid_retrieval.py`, `src/query_expansion_eval.py`, `src/train_candidate_reranker.py`, `src/train_query_adapter.py`, `src/evaluate_dual_lora.py` 등 — **전부 untracked**.
- 추적 파일 중 수정된 것 2개(미커밋): `scripts/split_products.py`(product_id dedupe 추가), `src/train_lora.py`(train/test overlap product 학습 제외 — `experiment_spec §3`이 말하는 "재현 안정성 방어 필터"의 실체).
- **함의**: 논문 재현성 섹션이 가리키는 코드 상태와 git HEAD가 불일치. 논문 제출 전 커밋·태깅 필요.

### 4.2 출력 파일은 90개, docs는 ~28개만 보고 (선택적 보고 리스크)
- `output/eval_*.json` = **90개**. docs `final/01_experiment_catalog.md`는 28개 방법만 카탈로그.
- 미보고 변형 예: `eval_hybrid_lora384_bm25_{all,caption,name}`, `eval_hybrid_lora384_dense_all_588[_rawvision]`, `eval_ensemble_dual_infonce_lora_bm25_gated`, `query_adapter_residual64_sigmoid` 등.
- **모든 평가가 동일 locked 30-query eval에서 측정됨**(긍정적: cherry-pick 아님). 단, "여러 변형을 돌린 뒤 best를 final로 명명"한 구조이므로, 논문은 **"방법 선택은 train-split validation에서만 했다"**는 정책을 반드시 명시해야 함(코드가 실제로 그렇게 함 — §4.3).

### 4.3 "validation"의 실체 — 사람 쿼리가 아니라 합성 쿼리
`src/dual_hybrid_rerank.py`(최종 best 생성 스크립트) 분석:
- BM25 가중치·top-n 선택용 **validation = train split에서 seed로 샘플한 240개 product, 그 product들의 augmented(LLM 합성) 쿼리 ~717개**.
- 즉 **방법 선택(weight grid 0.05–0.50 × top-n {10,20})의 검증 분포는 "합성 쿼리"이고, 최종 보고 30개는 "사람 쿼리"**다.
- 선택 기준: validation MRR → R@1 → R@5 → R@10 lexicographic. 최종 선택값 `rerank_top20_0.2bm25`.
- locked 30-query eval은 선택에 쓰이지 않음(정직). **그러나** "validation이 합성 분포라 사람 분포로의 일반화가 보장 안 됨"은 방어 포인트(실제로 `learned reranker`·`adaptive BM25`가 이 gap 때문에 붕괴 — docs도 인정).
- **자문 함의**: "method selection on validation only"라는 방어는 절반만 맞다. 정확히는 *합성 train-split validation*. 논문은 이걸 숨기지 말고 "선택 검증조차 사용자 분포가 아니라 합성 분포에서 했고, 그럼에도 사람 평가에서 유지됨"으로 정직하게 프레이밍하는 게 더 강함.

### 4.4 dual-tower의 숨은 비용 — 평가 시 460ms/query (docs가 거의 안 다룸)
- `output/eval_dual_lora_r4_infonce10.json`: `latency_ms_per_query_including_image_encoding: 460.9` ms.
- 이유: dual-tower는 vision encoder에도 LoRA가 붙어 **gallery 이미지를 평가 때마다 live forward**해야 함(`dual_hybrid_rerank.py`의 `encode_images`가 매 실행 전체 gallery 재인코딩). cached frozen vision을 못 씀.
- 대조: CLS text-only LoRA = `latency_ms_per_query: 0.04` ms (cached vision), ColBERT MaxSim = 11 ms.
- **즉 dual-tower는 CLS 대비 ~10,000배 느린 평가/추론 비용을 치르면서 MRR은 +0.001**. docs `experiment_spec §14.5`는 ColBERT 비용만 비판하고 **dual-tower 자신의 이 비용은 거의 보고하지 않음**. 이건 dual-tower를 헤드라인으로 내세울 때 치명적 약점이며 자문 LLM이 반드시 짚어야 함.

### 4.5 통계 검정 부재 (코드 전수 확인)
- 저장소 어디에도 bootstrap / CI / 유의성 검정 / 다중 seed 코드 **없음**. 모든 수치는 단일 seed·단일 run·N=30 점추정.
- `experiment_spec §10`이 한계로 인정하나, **per-query rank는 `output/eval_*_detail.json`에 전부 저장돼 있어 재학습 없이 bootstrap CI/paired sign test 계산 가능**(반나절). 이게 §1.3 철학과 충돌 없이 방어력을 가장 크게 올리는 단일 개입.

### 4.6 메서드 구현 핵심 디테일 (docs 수식보다 정확)
- **single-positive 가정 전역**: `metrics_at_k`는 쿼리당 정답 product_id 1개. gallery 588(test split 전체, 이미지 있는 것). 비슷한 상품이 gallery에 있으면 R@K가 과소추정(`experiment_spec §10.7`도 인정).
- **candidate-preserving rerank 메커니즘**(`rerank_topn`): base score를 row-normalize, top-n 후보에 한해 `norm(base) + weight·norm(bm25)`로 섞고, **top-n 밖은 −1e9로 강제** → R@10 보존의 실제 구현. union/adaptive 변형은 후보를 넓혀서 R@10 손상(docs 결론과 일치).
- **train_dual_lora.py**: text+vision attention(q/k/v/out)에 LoRA, `alpha = r*2`, dropout 0.05, AdamW(wd 1e-3), cosine+10%warmup, `logit_bias`는 sigmoid일 때만 학습. 기본값 epochs=5/batch=48/lr=5e-5이나 **최종 best는 r=4·InfoNCE·10ep**(명령행으로 지정, config 아님).
- **train_lora.py(메인, 256)**: cached vision_emb 사용, vision forward 0회 → A6000에서 10–15분. dual-tower는 이 가속을 포기.
- **BM25 문서**: `make_docs(..., "all")` = 상품명+브랜드+이미지설명 결합, Korean **char-ngram**(형태소 분석기 아님). 단순·재현 쉬움이 선택 이유.
- **CUDA device 하드코딩**: 다수 스크립트가 `CUDA_VISIBLE_DEVICES=4` setdefault. 다른 장비 재현 시 수정 필요.

---

## 5. 데이터 실체 (검증된 정확 수치)

| 항목 | 실측값 | 비고 |
|---|---|---|
| `data/top.json` | list, **3,102 entries** (≈3,092 unique product) | 무신사 상의 카탈로그. 필드: product_link, brand_id, product_name, img_url, price, text_description(GPT-4o 한국어 캡션 평균 509자) |
| 다운로드 성공 이미지 | **2,928** (`data/images/*.jpg`) | 3,102 시도, 164 HTTP 404, dedupe 후 |
| `data/real_user_query.txt` | **7개 쿼리 / 6개 상품** | few-shot seed. (SU HEART 상품에 쿼리 2개). 전부 `prompts/augment_final.txt`에 verbatim 삽입됨 |
| `data/train_pairs_final.jsonl` | **7,023줄** | 학습 페어. 필드: product_id, brand_id, product_name, query, query_type(1/2/3) |
| `data/train_pairs.jsonl` / `.raw.jsonl` | 7,047 / 2,350 | 중간 산출물(dedupe 전 / 원시) |
| `data/split_train.json` / `split_test.json` | **2,350 / 588** | product-level 80/20, seed 42. test 588이 gallery |
| `data/eval_queries.jsonl` | **30줄** | 사람 작성. 필드: product_id, brand_id, product_name, img_url, **description_snippet**, difficulty(short_ambiguous/medium/long_specific 각 10), query. single-positive |
| `data/eval_query.txt` | 34줄 | 평가 쿼리 작업 메모(비공식) |
| `prompts/augment_final.txt` | locked v4 | 규칙 **R1–R8**(docs는 R1–R7로 오기), 3-type 다양성 강제, seed 7개 verbatim, Solar-pro3용 |
| `prompts/augment_v1..v4` | 4회 iteration 기록 | v1→v4 프롬프트 진화 보존 |

augmentation 3-type 강제(프롬프트 실측):
- 쿼리1: 짧은 한 문장형 15~30자 (색+카테고리+핵심 그래픽1).
- 쿼리2: 중간 한 문장형 30~80자 (seed 7개 톤 그대로).
- 쿼리3: 한 측면 강조 15~40자 (색만/그래픽만/카테고리만 중 택1).

---

## 6. 논문 자문 시 알아야 할 "숨은 약점 / 방어 취약점" (docs가 약하게 다룸)

자문 LLM이 좋은 조언을 하려면 docs가 낙관적으로 서술한 부분의 실제 취약점을 알아야 한다.

1. **헤드라인 방법의 우위가 noise 수준**. `final/03`: 최종 best `Dual r4 10ep+BM25`(MRR 0.757) vs 단순 `LoRA-384+BM25`(MRR 0.756) — R@1/R@5/R@10 **완전 동일**, MRR **+0.001**, N=30. dual-tower(novelty가 높다고 docs가 주장)를 핵심 기여로 내세우면 *"당신의 핵심 방법 = 당신의 단순 baseline"* 한 줄 비판으로 무너짐. → 이전 자문 결론: dual-tower를 헤드라인에서 빼고 ablation/탐색으로 강등, 기여 무게중심을 **(분포격차 정량화 + 소수시드 LLM augmentation 레시피) + (candidate-preserving lexical rerank)**로 이동, ColBERT 실패·reranker 붕괴를 negative-results 섹션으로 승격.
2. **N=30, 단일 seed, 단일 annotator, 단일 gallery(588)**. R@10에서 1쿼리 = 0.033. 통계적 유의성 주장 불가. 가장 큰 구멍. → bootstrap CI/sign test(저장된 detail.json으로 재학습 없이) 선제 보강이 최우선.
3. **method selection validation = 합성 쿼리**(§4.3). "사람 분포 일반화 보장 없음"이 실제로 learned/adaptive 변형 붕괴로 나타남. 정직하게 프레이밍하면 오히려 강점(실패한 변형이 이 gap을 입증).
4. **dual-tower 추론 460ms/query**(§4.4). 산업 적용성을 주장(docs §13)하면서 이 비용을 숨기면 모순. text-only LoRA(0.04ms)+BM25가 실제로 더 배포 친화적.
5. **외부 API 의존·stochastic**(§3.2). augmentation 재현 불가, 단일 LLM(Solar-pro3) 의존, 다른 LLM 검증 없음. "로컬 전용/저비용"을 강점으로 쓰면 안 됨.
6. **R@10 회귀의 정직성**: 단계 C 메인 결과는 R@10 −0.033(zero-shot이 더 높음). docs는 failure mode 3개로 해석하나, 단계 D에서 BM25 rerank로 R@10 0.900 복구 → **단계 D를 본론으로 쓰면 "모든 지표 개선" 서사가 성립**(단계 C 단독은 R@10 후퇴 서사). 이게 단계 D를 canonical로 둘 실용적 이유.
7. **선택적 보고**(§4.2): 90개 결과 중 28개 보고. 모두 동일 locked eval이라 cherry-pick은 아니지만, 부록에 전체 표를 넣어 투명성 확보 권장.
8. **explan.md 대비 축소**: multi-positive·query refiner·val-best-epoch·색오류 augmentation 미구현. 한계/향후과제로 흡수돼 있으나 심사 질문 예상 지점.

---

## 7. 환경 / 재현성 실체

- conda env `jolnon` (실행 경로 예: `/opt/miniconda3/envs/jolnon/bin/python`). Python 3.11.15.
- torch 2.5.1, transformers 4.57.6, peft 0.13.2, accelerate 1.1.1, CUDA 12.4. backbone `google/siglip2-base-patch16-384`(단계 D) / `-256`(단계 C).
- GPU: RTX A6000 48GB. 다수 스크립트 `CUDA_VISIBLE_DEVICES=4` 하드코딩.
- seed=42 전역. 학습은 결정적(동일 seed 재학습 동일 결과). **단, augmentation은 외부 API·stochastic이라 비결정적**.
- `AGENTS.md`: 저장소 안전 규칙 — `/root/code/dust3r`, `/root/dataset` 접근 금지(이 프로젝트와 무관한 다른 디렉터리. 자문과 무관하나 환경 사실로 기록).
- 최종 산출물 경로: 체크포인트 `checkpoints/dual_lora_r4_infonce10/` 등 18개, 결과 `output/eval_*.json` 90개, figure `docs/final/figures/*.png` 7개, 재생성 스크립트 `scripts/build_final_docs.py`.

---

## 8. 권장 정합화 방향 (이전 자문 결론 — 자문 LLM이 이어받을 출발점)

이전 세션에서 도출한, 이 자료로 쓸 수 있는 가장 방어 가능한 논문 골격:

- **방어할 thesis(한 문장)**: "카탈로그 캡션 ↔ 사용자 쿼리의 분포 격차가 한국어 패션 zero-shot 검색의 주된 약점이며, 소수(N=7) 시드로부터의 LLM 쿼리 합성 + 텍스트측 LoRA 정렬, 그리고 후보 보존형 한국어 lexical rerank를 결합하면 zero-shot 대비 모든 지표에서 일관되게 우월하다."
- **canonical backbone을 patch16-384로 통일**(rerank 결과가 전부 384 기준이고, 단계 D라야 "모든 지표 개선" 서사 성립). 256은 backbone ablation으로 강등.
- **기여 무게중심**: ① 분포격차 정량화 + 소수시드 LLM augmentation 레시피(Promptagator/InPars 계열로 정직하게 positioning), ② candidate-preserving lexical rerank. **dual-tower와 ColBERT는 "탐색했고 한계가 있었다"는 negative-results/분석으로 강등**(0.001 함정·460ms 비용 회피).
- **글쓰기 전 최우선 보강 3가지**: (1) 저장된 detail.json으로 bootstrap CI + paired sign test(재학습 0), (2) 헤드라인 수치 384로 통일·256↔384 혼선 제거, (3) related work에 Promptagator/InPars/FashionCLIP 명시 인용으로 신규성 프레이밍 정리.
- **negative results를 독립 섹션으로**: ColBERT가 SigLIP에서 실패하는 아키텍처적 원인(`experiment_spec §14.4`: SigLIP은 pooled 1점에서만 contrastive supervision → per-token/patch는 부산물), learned reranker의 val↔eval 붕괴(0.877→0.504), 사전 식별한 3대 failure mode의 사후 재현(준-preregistration). 졸업논문에서 분석 깊이 가점 요소.

> 자문 LLM은 이 §8을 "확정"이 아니라 "이전 자문의 권고"로 받아들이고, docs+본 문서 전체를 근거로 동의/수정하면 된다.

---

## 9. docs 파일별 신뢰도 가이드 (어느 것을 믿을지)

| 파일 | 단계 | 신뢰도/용도 |
|---|---|---|
| `docs/experiment_spec.md` §1–13 | C | 메인 서사(256 text-only)의 가장 상세·정확한 기술. 단 R@10 후퇴 서사 |
| `docs/experiment_spec.md` §14 | D | 384 업그레이드 + ColBERT 실패. 정확하나 dual-tower 비용 누락 |
| `docs/final/*` | D | 최종 캠페인 결과의 canonical 정리. dual-tower 우위 과대평가 주의(§6-1) |
| `docs/report_draft_ko.md` | D | 초안. final/과 일관. 그대로 논문 골격으로 쓰기엔 framing 약함 |
| `docs/successful_trials.md`, `retrieval_results_summary.md` | D | 결과 표 보조. final/과 수치 일관 |
| `docs/plan.md`, `roadmap.md` | B | **계획 문서. 수치(9,306/2,481/621)는 실제와 불일치 — 사실 출처로 쓰지 말 것** |
| `explan.md`(루트, docs 아님) | A | 원본 컨설팅. 폐기된 권고 다수. 진화사 이해용으로만 |
| `config.yaml` | 혼합 | **stale. 최종 HP의 진실 원천 아님**(§3.1) |

---

*끝. 이 문서 + `docs/` 전체 + (선택)`explan.md`를 함께 전달하면, 자문 LLM이 docs의 낙관적 서술과 실제 코드/제약/통계 현실을 분리해 판단할 수 있다.*
