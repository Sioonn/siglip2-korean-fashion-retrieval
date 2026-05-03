# 실행 로드맵 — Task 단위

각 task는 다음 표기를 따른다:
- **[SEQ]** = 순차적, 이전 task가 끝나야 시작 가능
- **[PAR-X]** = 그룹 X 내에서 다른 task와 병렬 실행 가능
- **DEP**: 의존하는 task 명시
- **EST**: 예상 소요 시간

---

## D1 — 환경 / 셋업 / 데이터 준비 시작

### Phase 1.1 — 부트스트랩 (반드시 순차)

| Task | 설명 | DEP | EST |
|---|---|---|---|
| **T1.1 [SEQ]** | 프로젝트 구조 생성: `src/`, `output/`, `logs/`, `checkpoints/` 디렉토리 + `config.yaml` 골격 | — | 5분 |
| **T1.2 [SEQ]** | 의존성 설치: `transformers`, `peft`, `vllm`, `torch`, `aiohttp`, `pillow`, `pyyaml`, `tqdm` | T1.1 | 10분 |
| **T1.3 [SEQ]** | GPU/CUDA 동작 확인 + SigLIP 2 모델 다운로드 (`google/siglip2-base-patch16-256`) | T1.2 | 10분 |

### Phase 1.2 — 3-way 병렬 실행 (Phase 1.1 완료 후)

이 그룹의 3개 task는 서로 독립이며 동시에 진행한다.

| Task | 설명 | DEP | EST |
|---|---|---|---|
| **T1.4 [PAR-A]** | 이미지 다운로드 스크립트 작성 (`src/download_images.py`): aiohttp 병렬, 재시도 3회, 실패 로깅. **백그라운드 실행 시작** | T1.3 | 작성 30분 + 실행 30~60분 (백그라운드) |
| **T1.5 [PAR-A]** | vLLM + EXAONE-3.5-7.8B 로드 테스트 (`scripts/test_vllm.py`): 로드 → dummy prompt 1개 generation → 출력 확인. 실패 시 즉시 Qwen2.5-7B-Instruct로 교체 | T1.3 | 30분 (다운로드 포함) |
| **T1.6 [PAR-A]** | 평가 쿼리 30개 작성 시작 (수기, 코드 불필요): `data/eval_queries.jsonl` 빈 파일 만들고 작성. 난이도 3구간 × 10개. **이 시점에 test split 621개 상품 ID 결정 필요 → T1.6a로 분리** | T1.3 | 작성 시작 |
| **T1.6a [SEQ within T1.6]** | Product-level 80/20 split 실행 (`scripts/split_products.py`): seed=42로 무작위, `data/split_train.json` / `data/split_test.json` 출력 | T1.3 | 5분 |
| **T1.6b [SEQ within T1.6]** | test 621개 중 30개 무작위 선택 (난이도별 10개씩 할당), 사람이 직접 보고 쿼리 작성 시작 | T1.6a | 작성: D2까지 진행 |

### Phase 1.3 — Augmentation 프롬프트 확정 (T1.5 완료 후)

| Task | 설명 | DEP | EST |
|---|---|---|---|
| **T1.7 [SEQ]** | Augmentation 프롬프트 v1 작성 (`prompts/augment_v1.txt`): 7 few-shot, 3-style 강제, product_name + 캡션 입력 | T1.5 | 30분 |
| **T1.8 [SEQ]** | 시범 augmentation: train 상품 50개에 대해 v1으로 생성 → 직접 검수 (톤/hallucination/brand 끌림) | T1.7 | 1시간 |
| **T1.9 [SEQ, 조건부]** | 검수 결과에 따라 프롬프트 수정 (v2) → 다시 50개 → 검수. **최대 2회 반복**. 통과 기준: 명확한 hallucination/부자연 < 20% | T1.8 | 1~2시간 |
| **T1.10 [SEQ]** | 프롬프트 잠금 (`prompts/augment_final.txt`로 복사) | T1.9 | 즉시 |

**D1 종료 조건**: T1.10 완료, 이미지 다운로드 절반 이상 진행, 평가 쿼리 10~20개 작성됨.

---

## D2 — 풀스케일 데이터 + 갤러리 준비

### Phase 2.1 — 3-way 병렬 (D1 종료 직후)

| Task | 설명 | DEP | EST |
|---|---|---|---|
| **T2.1 [PAR-B]** | 풀스케일 augmentation 실행 (`src/run_augment.py`): train 2,481 상품 × 3 쿼리 = 7,443 쿼리 생성. vLLM 배칭. `data/train_pairs.jsonl` 출력 | T1.10 | 30~60분 |
| **T2.2 [PAR-B]** | 평가 쿼리 30개 완성 + 정답 라벨링 + **잠금** (`data/eval_queries.jsonl`을 read-only로 commit) | T1.6b | 2~3시간 (수작업) |
| **T2.3 [PAR-B]** | Vision 임베딩 사전 계산 스크립트 (`src/precompute_vision.py`): 다운로드된 모든 상품 이미지를 SigLIP 2 vision encoder로 한 번에 forward → `data/vision_emb.npy` (N × D), `data/vision_idx.json` (상품 ID 매핑). | T1.4 (다운로드 완료) | 작성 30분 + 실행 5분 |

### Phase 2.2 — Augmentation 검수 + 학습 데이터 정제 (T2.1 완료 후)

| Task | 설명 | DEP | EST |
|---|---|---|---|
| **T2.4 [SEQ]** | Augmentation 30개 무작위 샘플 검수 (수작업). 통과 기준 미충족 시 프롬프트 재수정 + T2.1 재실행 | T2.1 | 30분 |
| **T2.5 [SEQ]** | 학습 페어 jsonl 최종화 (`data/train_pairs_final.jsonl`): 다운로드 실패 상품 제외, train split 상품만 유지, (image_path, query) 튜플 형태로 정리 | T2.4 + T1.4 | 10분 |

### Phase 2.3 — 학습/평가 코드 작성 (T2.1과 병렬 가능)

T2.1이 백그라운드로 도는 동안 코드 작성을 진행한다.

| Task | 설명 | DEP | EST |
|---|---|---|---|
| **T2.6 [PAR-C]** | LoRA 학습 스크립트 작성 (`src/train_lora.py`): peft + transformers, text-only LoRA, vision frozen, 사전 캐싱된 vision_emb 활용, sigmoid loss, batch 256, BF16 | T1.10 | 1.5시간 |
| **T2.7 [PAR-C]** | 평가 스크립트 작성 (`src/evaluate.py`): config 입력 → text encoder로 30 query 임베딩 → 사전 캐싱된 vision_emb과 cosine similarity → R@10, R@5, MRR 출력 + 난이도 구간별 R@10 분리 출력 | T1.10 | 1시간 |

**D2 종료 조건**: 학습 데이터 jsonl 최종화 완료, 평가 셋 잠금, vision 임베딩 캐시 완료, 학습/평가 스크립트 작성 완료.

---

## D3 — 학습 + 측정

### Phase 3.1 — Baseline 먼저 (LoRA 없이도 가능)

| Task | 설명 | DEP | EST |
|---|---|---|---|
| **T3.1 [SEQ]** | Baseline 측정: `evaluate.py`에 LoRA weight 없이 (zero-shot SigLIP 2) 30 평가 쿼리 → R@10 / R@5 / MRR / 구간별 R@10 출력 | T2.3 + T2.2 + T2.7 | 5분 |

### Phase 3.2 — LoRA 학습 + Ours 측정

| Task | 설명 | DEP | EST |
|---|---|---|---|
| **T3.2 [SEQ]** | LoRA 학습 실행: `train_lora.py` config.yaml 읽고 10 epoch 학습. 로그 + 마지막 checkpoint 저장 | T2.5 + T2.6 | 10~15분 |
| **T3.3 [SEQ]** | Ours 측정: `evaluate.py`에 LoRA weight 로드 → 동일 30 쿼리 → R@10 / R@5 / MRR / 구간별 R@10 출력 | T3.2 + T2.3 + T2.2 | 5분 |
| **T3.4 [SEQ]** | Baseline vs Ours 비교 표 정리. R@10 차이가 +0.05 이상이면 통과. 그 이하면 재시도 카드 진입 | T3.1 + T3.3 | 10분 |

### Phase 3.3 — 조건부 재시도 (T3.4가 미통과 시만)

| Task | 설명 | DEP | EST |
|---|---|---|---|
| **T3.5 [SEQ, 조건부]** | epoch 20으로 재학습 + 재측정 | T3.4 미통과 | 25분 |
| **T3.6 [SEQ, 조건부]** | lr 5e-5 또는 2e-4로 재학습 + 재측정 (1~2회) | T3.5 미통과 | 25분 × N |
| **T3.7 [SEQ, 마지막 카드]** | LoRA r=8 또는 r=32로 재학습 + 재측정 | T3.6 미통과 | 25분 |

### Phase 3.4 — 정성 분석 준비 (T3.4 통과 즉시)

| Task | 설명 | DEP | EST |
|---|---|---|---|
| **T3.8 [SEQ]** | 정성 분석: Ours가 잘 맞춘 케이스 10개 + 못 맞춘 케이스 10개 선택 → top-10 retrieval 결과를 이미지 그리드로 렌더링 (`src/qual_viz.py`) | T3.3 | 1~2시간 |

**D3 종료 조건**: Baseline + Ours R@10 측정 완료, 차이 ≥ 0.05, 정성 분석 figure 생성됨.

---

## D4 — 보고서 작성

### 모두 [SEQ] (D3 결과 의존)

| Task | 설명 | DEP | EST |
|---|---|---|---|
| **T4.1 [SEQ]** | 결과 표 작성: 메인 R@10 비교 + 구간별 R@10 + 보조 메트릭 | T3.4 | 30분 |
| **T4.2 [SEQ]** | 정성 figure 정리: 성공/실패 케이스 그리드 | T3.8 | 30분 |
| **T4.3 [SEQ]** | 보고서 draft 작성: Introduction / Method / Experiments / Results / Conclusion. config.yaml 부록 첨부 | T4.1 + T4.2 | 4~6시간 |
| **T4.4 [SEQ]** | 검토 + 제출 | T4.3 | 1시간 |

---

## 병렬성 요약 — 시간 절약 효과

D1에서 [PAR-A] 그룹 (T1.4 / T1.5 / T1.6) 병렬 실행으로 약 **1.5~2시간 절약**.

D2에서 [PAR-B] 그룹 (T2.1 / T2.2 / T2.3) 병렬 실행으로 약 **2~3시간 절약** (T2.2 수작업이 가장 길어 임계 경로).

D2에서 [PAR-C] 그룹 (T2.6 / T2.7) + [PAR-B] T2.1 병렬 실행으로 augmentation 대기 시간을 코드 작성으로 흡수, 약 **1시간 절약**.

전체적으로 4일 일정을 **3일에 가깝게** 압축 가능. 단, 이는 임계 경로(T1.6 → T2.2 → T2.5 → T3.2 → T3.3 → T3.8 → T4.3)가 막힘없이 진행될 때 기준.

---

## 임계 경로 (Critical Path)

가장 길게 시간이 걸려 전체 일정을 결정하는 경로:

```
T1.1 → T1.2 → T1.3 → T1.6a → T1.6b (eval 작성, D1~D2 걸침) →
T2.2 (eval 잠금) → T2.5 (학습 데이터 최종화) →
T3.2 (학습 실행) → T3.3 (Ours 측정) → T3.4 (비교) →
T3.8 (정성 분석) → T4.3 (보고서)
```

**임계 task는 평가 쿼리 30개 직접 작성 (T1.6b → T2.2)**. 이게 막히면 전체가 막힘. 1일차 저녁부터 2일차 오전까지 우선 처리해야 함.

---

## 즉시 시작 가능한 액션 (지금)

1. T1.1 (프로젝트 구조 생성) — 5분
2. T1.2 (의존성 설치) — 10분
3. T1.3 (모델 다운로드) — 10분
4. **이후 T1.4 / T1.5 / T1.6 동시 시작**
