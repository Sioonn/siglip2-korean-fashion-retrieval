# 실험 설계 (확정)

## 문제 정의
- **태스크**: 한국어 사용자 자연어 쿼리 → 무신사 상의 카탈로그 이미지 검색
- **데이터**: `data/top.json` (3,102 상품: 이미지 URL, GPT-4o 캡션 평균 509자, product_name, brand_id), `data/real_user_query.txt` (6 상품 / 7 사용자 스타일 쿼리)
- **하드웨어**: A6000 48GB 단일 GPU
- **시간 예산**: 3~4일
- **외부 API 예산**: 없음 (모든 LLM은 로컬 실행)

## 핵심 가설
GPT-4o 캡션과 실제 사용자 쿼리 사이의 분포 격차(길이/톤/내용)가 zero-shot SigLIP 2 retrieval의 주된 약점이며, 학습 시점에 텍스트 측 분포를 사용자 쿼리에 맞추는 것만으로 R@10이 유의하게 향상된다.

## 비교 조건 (2개)
1. **Baseline**: SigLIP 2 base patch16-256 zero-shot (텍스트/이미지 인코더 그대로)
2. **Ours**: SigLIP 2 base patch16-256 + text encoder LoRA (LLM augmented 사용자 스타일 쿼리로 학습)

추론 시점 query refiner는 **포함하지 않음** (학습 시점 분포 정렬과 중복).

---

## 데이터 파이프라인

### Augmentation
- **모델**: EXAONE-3.5-7.8B-Instruct via vLLM (배칭 generation)
- **Fallback**: vLLM 호환 실패 시 Qwen2.5-7B-Instruct로 즉시 교체
- **Few-shot**: `data/real_user_query.txt`의 7개 사용자 쿼리 모두 in-context로 사용
- **상품당 생성**: 3개 쿼리, 다양성 강제 분산
  - 1번째: 짧은 키워드형 (5~15자)
  - 2번째: 한 문장 시각 묘사형 (예시 톤, 30~80자)
  - 3번째: 부분 정보형 (색만 또는 그래픽만)
- **프롬프트 컨텍스트**: GPT-4o 캡션 + product_name 함께 입력 (product_name은 학습 페어로는 쓰지 않음)
- **Sampling**: temperature=0.8, top-p=0.95
- **검수 루프**: 1일차에 50개 시범 → 직접 검수 → 프롬프트 수정 (최대 2회) → 풀스케일

### 학습 페어 구성
- 상품당 (이미지, augmented 쿼리) 3쌍
- 총 9,306쌍 (3,102 × 3)
- product_name은 학습 페어에서 제외 (catalogue 톤이라 분포 오염 우려)

### 분할
- **Product-level 80/20**: train 2,481 상품 / test 621 상품
- 같은 상품의 augmented 쿼리들은 train과 test에 동시 포함 금지
- val split 없음 (fixed 10 epoch + 마지막 checkpoint 사용)

### 이미지 획득
- 1일차에 `aiohttp` 병렬 다운로드 (동시 100 connection 수준)
- 재시도 3회, 그래도 실패 시 해당 상품 통째로 학습/평가 셋에서 제외
- 320×320 이상으로 저장 (SigLIP 2 patch16-256 입력에 충분)

---

## 학습 설정

| 항목 | 값 |
|---|---|
| Backbone | google/siglip2-base-patch16-256 |
| LoRA 적용 위치 | text encoder의 q_proj, k_proj, v_proj, out_proj |
| Vision tower | 완전 freeze, 임베딩 사전 캐싱 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Scheduler | cosine decay, 10% warmup |
| Batch size | 256 |
| Loss | SigLIP 표준 sigmoid loss |
| Precision | BF16 |
| Epoch | 10 (fixed) |
| Checkpoint | 마지막 epoch 사용 |
| Seed | 42 (단일 run) |

총 학습 step: train 7,440쌍 ≈ 30 step/epoch × 10 = 300 step. A6000에서 10~15분 예상.

### 학습 실패 시 재시도 카드 (순서대로)
1. epoch 20으로 확장
2. lr 5e-5 또는 2e-4로 흔들기 (1~2회)
3. LoRA r 8 또는 32로 변경 (마지막 카드)

---

## 평가

| 항목 | 값 |
|---|---|
| 메인 메트릭 | Recall@10 |
| 보조 메트릭 | R@5, MRR (보고서 부속 표) |
| 평가 셋 | 30개 사용자 스타일 쿼리 (직접 작성) |
| 난이도 분산 | 짧고 모호 10 / 중간 10 / 길고 구체 10 |
| 정답 라벨링 | 단일 positive |
| 갤러리 | test split 621 상품 (held-out) |
| 작성 시점 | **baseline 측정 전에 잠금** |
| 검색 구현 | 사전 계산된 임베딩 + numpy cosine similarity (FAISS 불필요) |

### 난이도 구간 정의
- **짧고 모호 (10개)**: 5~20자, 1~2단어 키워드형. 예: "흰 셔츠", "체크 패턴 셔츠"
- **중간 (10개)**: 30~60자, 한 문장. 사용자 예시 7개 톤. 예: "그레이 후드에 가슴쪽 영어 로고 적힘"
- **길고 구체 (10개)**: 80~150자, 두 문장 또는 디테일 포함

---

## 재현성
- 모든 hyperparameter, seed, 모델 path는 단일 `config.yaml`에 기록
- 학습/평가 스크립트는 config를 읽어 동작
- 보고서 부록에 config 그대로 첨부

---

## 위험 신호 + 대응

| 위험 | 대응 |
|---|---|
| Augmentation 품질이 20%+ 이상함 | 프롬프트에 "7개 예시 톤/길이 모방" 강조 → 재생성. 실패 시 Qwen2.5-7B로 모델 교체 |
| LoRA가 baseline 못 이김 | epoch 20 확장 → lr 흔들기 → LoRA r 변경 (이 순서) |
| Baseline R@10 < 0.05 | 평가 쿼리 너무 어려움. "길고 구체" 구간을 5~7개 추가하고 "짧고 모호"를 줄임 |
| Baseline R@10 > 0.9 | 평가 쿼리 너무 쉬움. "짧고 모호" 구간 추가하고 갤러리 노이즈 상품(비슷한 상품) 점검 |
| vLLM에서 EXAONE 안 뜸 | 즉시 Qwen2.5-7B-Instruct로 교체. 1시간 안에 결정 |
| 이미지 다운로드 실패율 > 5% | 실패 상품 제외 후 진행. 학습/평가 갤러리 크기만 약간 줄어듦 |

---

## 비고
- 본 프로젝트는 top-tier 학회 제출이 아닌 빠른 제출이 목표.
- novelty / fairness 엄격함 < 빠른 실험 종료 + baseline 대비 명확한 우위 증명.
- 위 결정은 모두 "방어 가능한 최소 수준"을 만족하면서 가장 빠른 path를 우선시함.
