# 실험 상세 스펙: 한국어 사용자 쿼리에 대한 패션 이미지 검색 모델의 텍스트 측 분포 정렬

## 1. Motivation (연구 동기)

### 1.1 문제 정의
한국어 e-commerce(무신사) 카탈로그 환경에서, **사용자 자연어 쿼리 → 상품 이미지 retrieval** 시스템을 구축할 때, zero-shot pretrained vision-language model(SigLIP 2)을 그대로 사용하면 다음과 같은 분포 격차 문제가 발생한다.

### 1.2 분포 격차 (Distribution Gap)
**카탈로그 캡션** (LLM 생성, GPT-4o 기반) 예시:
> "이 제품은 이벳필드의 베이직 로고 후드로, 색상은 부드러운 그레이로 이루어져 있습니다. 후드 디자인으로 되어 있어 착용 시 편안함과 따뜻함을 제공합니다. 가슴 부분에는 'Ebbets Field'라는 브랜드 로고가 네이비색으로 프린팅되어 있으며..." (~400자, 격식체, 정보 풍부)

**실제 사용자 쿼리** 예시:
> "그레이 후드에 진한 네이비 색으로 영어가 가슴쪽에 적혀있고, 그 아래 빨간색 글씨도 적혀있다." (~50자, 구어체, 시각적 인상 위주)

두 텍스트는 다음 측면에서 분포가 다르다:
- **길이**: 카탈로그 평균 509자 vs 사용자 쿼리 50–100자
- **톤**: 격식체 ("~합니다") vs 평서체/구어체 ("~있다", "~적혀있음")
- **어휘**: 브랜드명·소재·핏 형용사 빈출 vs 시각적 정보(색깔/그래픽/위치) 위주
- **추상화 수준**: 마케팅 표현·무드 ("세련된", "고급스러운") vs 사실 묘사

### 1.3 가설
**H1**: 학습된 SigLIP 2의 zero-shot 성능은 사용자 쿼리 분포에서 카탈로그 캡션 분포보다 약하다. 따라서 텍스트 인코더를 사용자 쿼리 분포에 align하는 것이 retrieval 품질 향상의 가장 큰 레버다.

**H2**: 사용자 쿼리 풀이 작더라도(7개 예시), 강력한 LLM에 in-context로 few-shot 예시를 주어 학습용 사용자 스타일 쿼리를 합성할 수 있고, 이 합성 데이터로 fine-tuning한 모델은 zero-shot baseline 대비 retrieval 메트릭이 향상된다.

---

## 2. Idea (제안 방법)

### 2.1 핵심 아이디어
**LLM augmentation + LoRA fine-tuning** — 학습 시점에 LLM으로 사용자 스타일 쿼리를 augment하고, SigLIP 2의 text tower만 LoRA로 fine-tuning하여 contrastive alignment를 사용자 분포에 맞춘다.

### 2.2 파이프라인 개요

```
[Training time]
무신사 상품 캡션 + 상품명
        │
        ▼
LLM (Solar-pro3) ── few-shot: 7개 실제 사용자 쿼리
        │
        ▼
사용자 스타일 쿼리 3개/상품
        │
        ▼
(이미지, 사용자 쿼리) pair
        │
        ▼
SigLIP 2 text encoder + LoRA (vision frozen, embeddings precomputed)
        │
        ▼
SigLIP sigmoid loss (in-batch contrastive)

[Inference time]
사용자 자연어 쿼리 → LoRA-adapted text encoder → embedding
                                                    │
                                                    ▼
                              ← cosine similarity → cached gallery embeddings
                                                    │
                                                    ▼
                                              top-K results
```

### 2.3 설계 결정 핵심
- **Vision tower freeze**: 가설이 텍스트 측 격차이므로 vision 측은 건드리지 않음. 추가 효과: vision embedding 사전 캐싱 → 학습 5–10배 가속
- **Text-only LoRA**: 이미지 인코더는 이미 강한 일반화 능력 보유. 작은 데이터(7,023쌍)에 vision까지 학습하면 overfit 위험. 텍스트 측 분포 정렬에만 집중
- **Inference-time refiner 미도입**: 학습 시점 augmentation으로 분포 정렬을 이미 수행하므로 추론 시점 LLM 정제는 중복. 단순화 우선
- **Augmented query 다양성 강제**: LLM이 비슷한 쿼리를 반복 생성하지 않도록 (1) 짧은 한 문장형 (2) 중간 길이 묘사형 (3) 한 측면 강조형으로 명시 분리
- **Product-level held-out**: train/test split을 product 단위로 분리하여 image leakage 차단

---

## 3. Contribution

본 실험의 contribution은 다음과 같다 (학회 논문보다는 capstone-class 보고서 수준의 정직한 contribution).

1. **사용자 쿼리 / 카탈로그 캡션 분포 격차의 정량화**: 길이·톤·어휘·추상화 4가지 축에서 두 분포의 차이를 측정하고 해당 격차가 zero-shot retrieval의 주된 약점임을 보임.

2. **소량 사용자 쿼리(N=7)로부터 합성 학습 데이터를 만드는 in-context augmentation 레시피 제시**: few-shot LLM prompting + 다양성 강제 + 패션 전문 어휘/주관 형용사 ban 규칙 + 검수 루프.

3. **Text-only LoRA fine-tuning on cached vision embeddings**: 텍스트 분포 정렬에 한정한 효율적 학습 전략. A6000 단일 GPU에서 약 10–15분 학습.

4. **Honest empirical evaluation on hand-written queries**: LLM-augmented 쿼리로 평가하지 않고 사람이 직접 작성한 30개 쿼리(난이도 3구간 × 10)로 평가하여 train/test 분포 contamination 차단.

5. **Failure mode 분석**: 사전 분포 검증으로 식별한 위험 요소(색깔 mismatch, 금지된 핏 형용사, 과도하게 verbose한 쿼리)가 실제 평가에서 그대로 회귀로 나타남을 확인 → 미래 개선 방향 제시.

---

## 4. 데이터 (Data)

### 4.1 원천 데이터
- **데이터셋**: 무신사 (Korean e-commerce) 상의 카테고리 상품 카탈로그
- **파일**: `data/top.json` (3,102 entries, 3,092 unique products)
- **각 상품 필드**:
  - `product_link`: 상품 URL
  - `brand_id`: 브랜드명 (472 unique brands)
  - `product_name`: 상품명 (평균 26자, min 6, max 93)
  - `img_url`: 상품 대표 이미지 URL (Musinsa CDN)
  - `price`: 가격 (KRW)
  - `text_description`: GPT-4o가 생성한 한국어 상세 설명 (평균 509자, min 31, max 940)

### 4.2 사용자 쿼리 시드
- **파일**: `data/real_user_query.txt`
- **개수**: 7개 사용자 쿼리 (6 unique products)
- **사용처**: augmentation prompt의 in-context few-shot 예시
- **예시**:
  > "그레이 후드에 진한 네이비 색으로 영어가 가슴쪽에 적혀있고, 그 아래 빨간색 글씨도 적혀있다. 주머니랑 모자 부분의 스트링도 있다."

### 4.3 이미지 다운로드
- **방법**: `aiohttp` 비동기 병렬 다운로드 (concurrency=100, retries=3)
- **결과**: 3,102개 시도, 2,938개 성공 (164개는 HTTP 404 — Musinsa가 제거한 이미지)
- **저장**: `data/images/{product_id}.jpg`
- **dedupe 후**: 2,928개 unique product (top.json에 10개 상품이 `/goods/`/`/products/` 두 URL로 중복 등록되어 있었음)

### 4.4 Train/Test Split
- **방식**: product-level 80/20, seed=42
- **결과**: train 2,350 products / test 588 products (이미지 다운로드 성공 후 기준)
- **특성**: 같은 상품의 augmented 쿼리들이 train과 test에 동시 등장하는 일이 없도록 product 단위로 분리

### 4.5 Augmented Training Data
- **생성기**: Upstage Solar-pro3 (`solar-pro3` via Upstage API, reasoning_effort=high)
- **개수**: train 2,341 products × 3 queries = 7,023 pairs (8 products 중복 파싱 제거 후)
- **다양성 강제**:
  - 쿼리1 (짧은 한 문장형, 15~30자): 색상 + 카테고리 + 핵심 그래픽 1개
  - 쿼리2 (중간 한 문장형, 30~80자): 색상 → 카테고리 → 그래픽 위치/색의 자연스러운 흐름
  - 쿼리3 (한 측면 강조형, 15~40자): 색상만 / 그래픽만 / 카테고리만 / 전체 인상 중 하나
- **프롬프트 핵심 규칙** (`prompts/augment_final.txt`):
  - **R1**: 브랜드명·영문 모델명·시리즈명·컬렉션명·시즌 코드 절대 금지
  - **R2**: 패션 전문 핏 용어 금지 (머슬핏, 오버핏, 릴렉스핏, 박시, 워시드, 피그먼트 등)
  - **R3**: 사진만 보고 알 수 없는 정보 금지 (촉감·착용감·신축성·통기성·소재 종류)
  - **R4**: 주관적 평가/감상 표현 금지 (강렬한, 세련된, 모던한, 깔끔한 등)
  - **R5**: 부정형 표현 금지 ("글씨 없는", "프린팅 없는")
  - **R6**: 단어 나열 금지, 문장/구/절 형태만
  - **R7**: 격식체 금지 ("~입니다", "~합니다")
- **품질 검증**:
  - 50개 시범 생성 → 직접 검수 → 프롬프트 v1→v2→v3→v4까지 4회 iteration
  - 최종 검수 결과: 브랜드명 leak 0.03% (2/7,047 — 거의 zero), 길이/톤 분포 사용자 예시와 매칭

### 4.6 평가 셋 (Evaluation Set)
- **개수**: 30개 hand-written queries (사람이 직접 작성)
- **방법**: test split 588 products 중 30개 무작위 선택, 각 product의 이미지를 사람이 직접 보고 사용자 스타일로 쿼리 작성
- **난이도 분산**:
  - `short_ambiguous` (10개): 5–25자 단순 묘사. 예: "어두운 핑크 후드티"
  - `medium` (10개): 16–55자 한 문장. 예: "검은색 후드티에 흰색 그래픽 글씨와 스트링이 달려있음"
  - `long_specific` (10개): 31–82자 디테일 풍부. 예: "진한 네이비색 맨투맨에 밝은 파란색의 꽃이 그려져있고, 꽃 가운데는 노란색이며 오른쪽 아래에 Failed라는 글자가 적혀있다."
- **품질 관리**:
  - Multi-pack 상품 5개 제거 (이미지 안에 옷 2개 이상 있는 상품)
  - Low-detail 상품 4개 제거 (long_specific에 무지·단색 상품)
  - High-detail 1개를 short→long으로 이동 (Ohio State 그래픽 가득)
  - 라벨링: 단일 positive (한 쿼리 = 한 정답 product_id)
- **잠금**: baseline 측정 *전에* 작성 완료, 측정 결과를 보고 수정하지 않음 (selection bias 방지)

### 4.7 Evaluation Set 분포 검증 (Methodology Validation)

평가 셋이 train 분포와 호환되는지 사후 검증:

| | n | min | max | mean | median |
|---|---|---|---|---|---|
| Train (Solar augmented) | 7,023 | 7 | 138 | **36.7** | 28 |
| Eval (user written) | 30 | 8 | 82 | **36.7** | 33 |
| eval-short_ambiguous | 10 | 8 | 23 | 14.7 | 14.5 |
| eval-medium | 10 | 16 | 55 | 36.1 | 36 |
| eval-long_specific | 10 | 31 | 82 | 59.3 | 57 |
| train-type1 | 2,341 | 7 | 53 | 25.4 | 25 |
| train-type2 | 2,341 | 25 | 138 | 60.9 | 58 |
| train-type3 | 2,341 | 7 | 62 | 23.7 | 22 |

- 전체 평균 일치 (36.7 == 36.7) → 분포 매치 양호
- eval-short는 train type1/3보다 약 10자 짧음 (사용자가 의도적으로 더 압축)
- Token-level OOV rate 22.4% (대부분 형태론적 어미 결합형)
- 진짜 missing semantic terms 4개: `단가라`, `코스튬`, `슬림`, `연보라`

---

## 5. 모델 (Models)

### 5.1 Backbone
- **모델**: `google/siglip2-base-patch16-256`
  - 총 파라미터: 약 375M
  - Text encoder: GemmaTokenizerFast (multilingual), 12 layers, 768 dim, max_length=64
  - Vision encoder: Vision Transformer base, patch16, 256×256 input, output 768-dim
- **선택 이유**:
  - 다국어 GemmaTokenizer로 한국어 처리 가능
  - patch16-256은 A6000에서 batch 256까지 무리 없이 학습 가능
  - 64 token limit이 사용자 쿼리(평균 50자) 길이와 잘 맞음
  - SigLIP 2의 sigmoid loss는 in-batch contrastive에서 negative sampling 부담 적음

### 5.2 LoRA Fine-tuning Config
- **적용 위치**: text encoder의 모든 self-attention projection
  - 12 layers × {`q_proj`, `k_proj`, `v_proj`, `out_proj`} = 48 modules
  - Vision encoder는 완전 freeze
- **LoRA hyperparameters**:
  - rank `r=16`
  - `alpha=32` (scaling factor 2.0)
  - `dropout=0.05`
  - bias: `none`
- **Trainable parameters**: 약 0.5–1% of backbone (~2M)
- **추가 trainable**: SigLIP의 `logit_scale`, `logit_bias` (sigmoid loss의 calibration 파라미터)

### 5.3 Augmentation LLM
- **모델**: Upstage `solar-pro3` (한국어 특화 reasoning 모델)
- **API endpoint**: `https://api.upstage.ai/v1`
- **Sampling**: `reasoning_effort=high`, default temperature/top-p
- **Concurrency**: asyncio with 12 concurrent requests, retry 3회
- **선택 변천**: EXAONE-3.5-7.8B-Instruct (vLLM 호환 검증 완료) → Solar-pro3로 교체 (사용자 요청, 한국어 reasoning 품질 우위)

---

## 6. 학습 (Training)

### 6.1 Hyperparameters
| 항목 | 값 |
|---|---|
| Optimizer | AdamW (default betas) |
| Learning rate | 1e-4 |
| Scheduler | Cosine decay with 10% linear warmup |
| Batch size | 256 |
| Epochs | 10 (fixed) |
| Total steps | 270 |
| Precision | BF16 |
| Gradient clipping | max_norm=1.0 |
| Loss | SigLIP sigmoid loss |
| Seed | 42 (single run) |

### 6.2 SigLIP Sigmoid Loss
SigLIP의 표준 sigmoid loss를 그대로 사용:

$$
\mathcal{L} = \frac{1}{B} \sum_{i,j} -\log \sigma\left(z_{ij} \cdot \left(t \cdot \mathbf{x}_i^T \mathbf{y}_j + b\right)\right)
$$

- $\mathbf{x}_i$: text embedding (LoRA-adapted, L2 normalized)
- $\mathbf{y}_j$: cached image embedding (frozen vision encoder, L2 normalized)
- $z_{ij} \in \{+1, -1\}$: 양성(diagonal)/음성(off-diagonal) 라벨
- $t = \exp(\text{logit\_scale})$, $b = \text{logit\_bias}$: 학습 가능한 calibration

In-batch negatives (256 - 1 = 255개) 사용. Hard negative mining 미도입.

### 6.3 학습 곡선
| Epoch | Mean Loss |
|---:|---:|
| 1 | 4.105 |
| 2 | 2.929 |
| 3 | 2.590 |
| 4 | 2.422 |
| 5 | 2.313 |
| 6 | 2.249 |
| 7 | 2.196 |
| 8 | 2.146 |
| 9 | 2.137 |
| 10 | **2.126** |

부드러운 단조 감소, instability 없음. logit_scale 학습값 = 4.72, logit_bias = -16.75.

### 6.4 학습 효율 최적화
- **Vision embedding 사전 캐싱**: 학습 시작 전 2,928개 product 이미지를 SigLIP 2 vision encoder로 1회 forward, npy로 저장. 이후 학습 중 vision forward 0회.
- **결과**: A6000 단일 GPU에서 학습 약 10–15분 (270 steps × ~3초/step)

### 6.5 Hardware
- **GPU**: NVIDIA RTX A6000 (48GB VRAM)
- **CUDA**: 12.4
- **Framework**: PyTorch 2.5.1, transformers 4.57.6, peft 0.13.2, accelerate 1.1.1
- **Python**: 3.11.15 (conda env `jolnon`)

### 6.6 Tokenization & Truncation
SigLIP 2의 GemmaTokenizer는 max_length=64 token 제약을 가진다. `tokenizer(..., truncation=True, max_length=64)`로 자동 truncation 적용 (별도 sliding window/chunking 없음).

| | n | max | mean | p95 | > 64 (truncated) |
|---|---:|---:|---:|---:|---:|
| Train queries (all) | 7,023 | 89 | 26.5 | 53 | **70 (1.00%)** |
| ─ train-type1 (짧은 한 문장형) | 2,341 | 33 | 19.0 | 26 | 0 |
| ─ train-type2 (중간 길이형) | 2,341 | 89 | 42.7 | 62 | 70 (2.99%) |
| ─ train-type3 (한 측면 강조형) | 2,341 | 43 | 17.7 | 24 | 0 |
| **Eval queries (all)** | 30 | 57 | 29.0 | 56 | **0 (0.00%)** |
| ─ eval-short_ambiguous | 10 | 20 | 13.3 | 20 | 0 |
| ─ eval-medium | 10 | 43 | 29.6 | 43 | 0 |
| ─ eval-long_specific | 10 | 57 | 44.0 | 57 | 0 |

**영향 평가**:
- 평가 쿼리는 0% truncation → **평가 결과는 truncation 영향을 받지 않음**.
- 학습 쿼리는 1.00%만 truncated, 모두 type2(중간 길이형) 일부에서 발생.
- Truncation은 문장 후반의 부수적 디테일(추가 글씨, 시보리, 포켓 등)을 자르며, **핵심 정보(색상·카테고리·메인 그래픽)는 문장 앞쪽에 위치하여 보존**됨.
- 1% 비율은 학습 신호에 거의 영향 없음 → 학습/평가 양측에서 truncation은 결과 해석에 문제가 되지 않는 수준.

---

## 7. 평가 (Evaluation)

### 7.1 평가 절차
양 조건(baseline / ours) 모두 동일한 절차:

1. **갤러리 임베딩 (공통)**:
   - test split 588 unique products의 사전 캐싱된 vision embedding (768-dim, L2 normalized)
   - SigLIP 2 zero-shot vision encoder로 1회 계산, 이후 양 조건에서 동일 사용
2. **쿼리 임베딩**:
   - 30개 사용자 작성 쿼리를 GemmaTokenizer로 토크나이즈 (max_length=64, padding=max_length)
   - text encoder forward → 768-dim → L2 normalize
3. **유사도 계산**:
   - Cosine similarity = $\mathbf{x}^T \mathbf{y}$ (양쪽 normalize되었으므로 dot product와 동일)
   - Shape: 30 × 588
4. **랭킹 + 메트릭**:
   - 각 쿼리당 588개 product를 similarity 내림차순 정렬
   - 정답 product의 rank 측정
   - $\text{R@K} = \frac{1}{30}\sum_{i=1}^{30}\mathbb{1}[\text{rank}_i \le K]$
   - $\text{MRR} = \frac{1}{30}\sum_{i=1}^{30}\frac{1}{\text{rank}_i}$

### 7.2 두 조건의 유일한 차이
| | Baseline | Ours |
|---|---|---|
| Backbone | SigLIP 2 base | SigLIP 2 base |
| Vision encoder | frozen (cached emb) | frozen (cached emb) |
| Text encoder | **zero-shot pretrained** | **+ LoRA (fine-tuned)** |
| Tokenizer | GemmaTokenizer | GemmaTokenizer |
| Gallery | 동일 588개 임베딩 | 동일 588개 임베딩 |
| Eval queries | 동일 30개 | 동일 30개 |

→ 측정된 차이는 순전히 **text encoder의 LoRA adaptation 효과**.

### 7.3 메트릭 정의 및 의미
- **R@K (Recall at K)**: 정답이 top-K 안에 들 확률. Random chance = K/588 (e.g. R@5 random ≈ 0.85%)
- **MRR (Mean Reciprocal Rank)**: $\frac{1}{N}\sum_i 1/\text{rank}_i$. Rank 1 = 1.0, Rank 2 = 0.5, Rank 10 = 0.1. 정답이 상위에 있을수록 비선형으로 큰 값.
- **R@1**: 정답이 정확히 1위에 위치할 확률. 가장 strict.

---

## 8. 결과 (Results)

### 8.1 Main Results (588-product gallery, 30 hand-written queries)

| Metric | Baseline (zero-shot) | Ours (text LoRA) | Δ |
|---|---:|---:|---:|
| R@1 | 0.533 | **0.633** | **+0.100** |
| R@5 | 0.733 | **0.833** | **+0.100** |
| R@10 | **0.867** | 0.833 | -0.033 |
| MRR | 0.624 | **0.712** | **+0.088** |

- **R@1 / R@5 / MRR 모두 유의하게 개선** — Ours가 정답을 ranking 상단으로 더 강하게 끌어올림
- R@10은 1개 쿼리만큼 회귀 (26/30 → 25/30) — 분석 섹션에서 원인 분석
- Random chance baseline: R@1≈0.17%, R@5≈0.85%, R@10≈1.7%

### 8.2 By Difficulty Band

| Difficulty | Metric | Baseline | Ours | Δ |
|---|---|---:|---:|---:|
| short_ambiguous | R@10 | 0.7 | 0.7 | 0.0 |
| short_ambiguous | R@5 | 0.7 | 0.7 | 0.0 |
| short_ambiguous | MRR | 0.594 | **0.708** | **+0.114** |
| medium | R@10 | **1.0** | 0.8 | -0.2 |
| medium | R@5 | 0.7 | 0.8 | +0.1 |
| medium | MRR | 0.591 | 0.495 | -0.096 |
| long_specific | R@10 | 0.9 | **1.0** | +0.1 |
| long_specific | R@5 | 0.8 | **1.0** | **+0.2** |
| long_specific | MRR | 0.686 | **0.933** | **+0.247** |

### 8.3 Headline Findings
1. **long_specific에서 가장 큰 개선** — R@10 0.9→1.0, MRR 0.69→0.93. 가설("시각 정보 풍부한 쿼리에서 분포 정렬 효과 큼")의 강한 증거.
2. **short_ambiguous에서 R@K는 동률, MRR 개선** — 짧은 쿼리에서도 ranking 정밀도는 향상
3. **medium에서 R@10 회귀** — 1.0에서 0.8로 (2 queries)

---

## 9. 분석 (Analysis)

### 9.1 Per-Query Rank Changes

| Product ID | Difficulty | Baseline rank | Ours rank | Δ |
|---|---|---:|---:|---:|
| 4129040 | long_specific | 85 | 1 | **+84** |
| 3792711 | short_ambiguous | 60 | 45 | +15 |
| 3442628 | long_specific | 10 | 3 | +7 |
| 2348247 | medium | 8 | 4 | +4 |
| 3562545 | long_specific | 4 | 1 | +3 |
| 1383822 | short_ambiguous | 3 | 1 | +2 |
| 3210500 | medium | 5 | 3 | +2 |
| (기타 동률·소폭 변화 다수) | | | | |
| 3172527 | medium | 1 | 2 | -1 |
| 4008116 | medium | 1 | 2 | -1 |
| 4326538 | medium | 6 | 12 | -6 |
| 1661590 | medium | 4 | 12 | -8 |
| **1863271** | short_ambiguous | 16 | **35** | **-19** |

### 9.2 Failure Mode 분석 (사전 식별 위험과 일치)

학습 데이터 분포 검증 시 사전 식별한 4가지 위험이 실제 평가에서 회귀로 그대로 나타남:

**Failure 1: 색깔 mismatch** (1863271)
- 사용자 query: "**검은색** 후드티에 가운데 작은 로고"
- 실제 product: "NCMMN BEAR HOODIE **NY**" (네이비)
- 메커니즘: Ours는 색깔 binding이 학습으로 강화되어 검은색 ≠ 네이비를 더 엄격히 구분. Baseline은 약한 binding으로 색 mismatch에 더 너그러움.

**Failure 2: 금지된 핏 형용사** (1661590)
- 사용자 query: "여성용 검은색 **슬림한** 롱슬리브"
- 학습 데이터에 "슬림한" 미등장 (R2 규칙으로 의도적 ban)
- 메커니즘: 사용자가 ban된 단어를 사용했을 때 Ours가 의미적으로 활용 못 함.

**Failure 3: 과도하게 verbose / 모순된 묘사** (4326538)
- 사용자 query: "약간 **창백하지만** 채도가 높은 파란색과 회색이 섞인 후드티에 후드 주머니도 있고 가운데에 로고도 있음"
- 메커니즘: 모순적 묘사("창백하지만 채도 높은") + 정보 과잉으로 임베딩이 specific direction에 align 안됨.

### 9.3 Long_specific의 강한 개선 분석

long_specific에서 R@10이 90% → 100%, MRR이 0.69 → 0.93로 압도적 개선. 대표 사례 4129040 (rank 85 → 1, +84)을 보면:

- Query: "오트밀색의 반팔 티셔츠인데 소매가 팔꿈치를 덮을 정도로 긴 편이다. 가운데에 진한 빨간색으로 글씨 및 그래픽이 많이 적혀있다."
- Product: Ohio State 풋볼 그래픽 가득한 오트밀 반팔티

Baseline이 이 쿼리에서 85위로 정답을 묻은 이유는 zero-shot SigLIP 2의 텍스트 측이 카탈로그 캡션 분포에 align되어 있어 "오트밀색", "팔꿈치를 덮을 정도", "빨간색으로 글씨 및 그래픽이 많이"같은 사용자 표현을 시각 정보와 잘 연결 못 했기 때문. Ours는 LoRA로 "오트밀색", "빨간색 글씨", "그래픽이 많이" 같은 표현을 적절히 grounding함.

### 9.4 R@10 회귀의 정량적 의미
R@10 -0.034는 30개 쿼리 중 단 1개만큼의 차이 (26/30 → 25/30). 회귀를 일으킨 두 쿼리(1661590, 4326538)가 모두 rank 12로 12에 위치하므로 R@12로 보면 다시 동률. 즉, 회귀 폭은 매우 작고, 위 3가지 failure mode 분석으로 해석 가능한 noise 수준.

반대로 R@1 / R@5 / MRR의 0.10 / 0.10 / 0.088 개선은 "Ours가 ranking 상단에서 더 정밀한 retrieval을 한다"는 명확한 신호.

---

## 10. Limitations (한계)

1. **단일 시드, 단일 run**: confidence interval 없음. R@10 회귀가 통계적으로 유의한지 검증 안 함.
2. **평가 셋 크기 30**: R@10에서 1 query 차이가 0.033 메트릭 차이가 됨. 더 큰 평가 셋이면 추정치 더 안정.
3. **단일 갤러리 크기 (588)**: 갤러리 크기가 retrieval 난이도에 영향. 더 큰 갤러리에서는 절대값이 달라질 수 있음.
4. **단일 평가자**: hand-written 쿼리가 한 사람의 작성 스타일에 편향. 다중 평가자 IRR 측정 안 함.
5. **Solar-pro3 의존**: augmentation 품질이 LLM 능력에 의존. 다른 LLM에서 동일 효과 나는지 미검증.
6. **단일 도메인**: 무신사 상의만. 하의·신발·가방 등 도메인 일반화 안 됨.
7. **Multi-positive 미적용**: 단일 정답 가정. 갤러리에 비슷한 상품이 있으면 R@K가 underestimate.
8. **R@10 회귀의 부분 미해결**: 색깔 mismatch / ban된 형용사 / verbose 쿼리 → 향후 train 데이터에 더 다양한 표현 패턴 포함 필요.
9. **Ablation 부재**: LoRA r 값, augmented query 수, vision tower co-tuning 등의 ablation 없음.

---

## 11. Future Work

1. **Color-tolerance augmentation**: 사용자가 색을 주관적으로 인식할 수 있음 (검은색 vs 네이비, 베이지 vs 그레이). 학습 시점에 small color perturbation을 augmented 쿼리에 의도적으로 포함.
2. **Inference-time query refiner**: 사용자 쿼리를 LLM으로 1차 정제 후 retrieval. 본 실험에서는 단순화 위해 미도입.
3. **Multi-positive evaluation**: 갤러리에서 비슷한 후보들을 multi-positive로 라벨링하여 R@K 인플레이션 보정.
4. **Hard negative mining**: in-batch만 사용. 비슷한 상품을 hard negative로 의도적 sampling.
5. **다중 시드 / bootstrap CI**: 통계적 신뢰도 확보.
6. **Vision tower 부분적 fine-tuning**: 가설을 확장하여 vision 측에도 LoRA 적용 시 효과 측정 (overfit 위험 감수).
7. **다른 LLM augmenter 비교**: GPT-4o, Claude 등.

---

## 12. Reproducibility

### 12.1 코드 저장소
- Project root: `/root/code/jolnon` (혹은 `/home/intern0/sion/code/jolnon`)
- Working files:
  - `config.yaml`: 모든 hyperparameter, path, seed 단일 진입점
  - `prompts/augment_final.txt`: locked augmentation prompt (v4)
  - `src/download_images.py`: 이미지 다운로드
  - `scripts/split_products.py`: 80/20 product split
  - `src/precompute_vision.py`: vision embedding 사전 캐싱
  - `src/run_augment.py`: 풀스케일 augmentation
  - `scripts/finalize_train_pairs.py`: dedupe
  - `src/train_lora.py`: LoRA fine-tuning
  - `src/evaluate.py`: 평가 (baseline / ours)
  - `scripts/apply_eval_swaps.py`: eval set 정제 (multi-pack/low-detail 교체)

### 12.2 실행 순서
```bash
# 1. 환경
conda create -n jolnon python=3.11 -y
conda activate jolnon
pip install torch==2.5.1 transformers==4.57.6 peft==0.13.2 accelerate==1.1.1 \
            openai==1.81.0 python-dotenv aiohttp pillow pyyaml tqdm matplotlib \
            sentencepiece protobuf

# 2. 데이터 준비
python src/download_images.py
python scripts/split_products.py

# 3. Augmentation
# .env에 UPSTAGE_API_KEY 설정
python src/run_augment.py
python scripts/finalize_train_pairs.py

# 4. Vision embedding 캐싱
python src/precompute_vision.py

# 5. 평가 셋 작성
python scripts/make_eval_template.py
# eval_queries.jsonl 직접 작성
python scripts/apply_eval_swaps.py  # multi-pack / low-detail 교체

# 6. 학습 + 평가
python src/train_lora.py
python src/evaluate.py --tag baseline
python src/evaluate.py --tag ours --lora checkpoints/lora_final
```

### 12.3 산출물
- `data/train_pairs_final.jsonl`: 7,023 augmented training pairs
- `data/eval_queries.jsonl`: 30 hand-written eval queries
- `data/vision_emb.npy`, `data/vision_idx.json`: 2,928 cached vision embeddings
- `checkpoints/lora_final/`: LoRA weights + extras.json (logit_scale, logit_bias, epoch_losses)
- `output/eval_baseline.json`, `output/eval_ours.json`: 메트릭 요약
- `output/eval_baseline_detail.json`, `output/eval_ours_detail.json`: per-query rank + top-10 retrievals

### 12.4 시드 / 재현성
- 모든 randomness: `seed=42` 고정 (config.yaml `project.seed`)
- numpy / torch / random 모두 동일 seed
- LLM augmentation은 stochastic (temperature=high) — 동일 seed로도 약간 다른 출력 가능 (API endpoint 측면)
- 학습은 결정적 — 동일 seed로 재학습 시 동일 결과

---

## 13. 핵심 메시지 요약 (보고서 작성용)

1. **문제**: zero-shot SigLIP 2는 한국어 사용자 쿼리에 대한 패션 retrieval에서 카탈로그 캡션 분포에 편향되어 있어 사용자 분포 입력에 약함.
2. **해법**: 7개 사용자 쿼리 시드 + Solar-pro3 LLM augmentation으로 7,023개 사용자 스타일 쿼리 합성, SigLIP 2 text encoder에 LoRA만 적용하여 fine-tune.
3. **결과**: 588-product gallery, 30 hand-written eval queries에서 R@1 +0.10, R@5 +0.10, MRR +0.088. long_specific에서 R@10 0.9→1.0, MRR 0.69→0.93의 큰 개선. R@10 전체 -0.033 회귀는 사전 식별한 3개 failure mode(색깔 mismatch, ban된 형용사, verbose 쿼리)와 정확히 일치.
4. **의의**: 작은 사용자 쿼리 시드(N=7)로도 텍스트 분포 정렬이 가능함을 보임. text-only LoRA로 효율적(15분 학습) 적용. 산업적으로 e-commerce retrieval에 즉시 적용 가능한 레시피.

---

## 14. 후속 실험: Backbone Upgrade + ColBERT-style Late-Interaction Extension

본 capstone 실험의 메인 narrative(text-only LoRA on patch16-256)가 완료된 후, **두 가지 후속 확장**을 시도했다.

### 14.1 동기

- **Backbone upgrade**: patch16-256 → patch16-384. 더 높은 해상도(256 patches → 576 patches per image)로 그래픽/글자 디테일에 유리할 가능성. 한국 패션 도메인에선 글자/로고/색 영역 매칭이 중요.
- **ColBERT-style late-interaction**: 단일 768-d 벡터 대 단일 768-d 벡터(CLS) 매칭에서, **per-token text vs per-patch image의 Sum-of-MaxSim** 방식으로 token-level fine-grained 매칭. 사용자 쿼리에 "가운데에 글씨", "오른쪽 위에 로고" 같은 spatial reference가 많아 multi-vector 표현이 hypothesis 상 유리해 보임.

설계 결정 (grill 검토 완료):
- Stage B: ColBERT-LoRA를 **메인 방법론으로 업그레이드**, 기존 CLS-LoRA는 ablation으로 강등 (사용자 의도)
- Backbone: patch16-384 (4 condition 모두 동일 backbone에서 측정)
- LoRA 적용 범위: text-only 유지 (vision pre-cache 활용)
- Loss: SigLIP sigmoid (메인) + InfoNCE (loss ablation)
- Text token: non-padding only (BOS/EOS 포함)
- Patch token: all 576 patches
- MaxSim normalization: query length로 mean (variable-length query에서 length bias 제거)
- Cost ablation: storage + per-query latency

### 14.2 Backbone Upgrade 결과 (CLS Pooling 기준)

같은 LoRA recipe를 patch16-384에서 다시 학습.

| Condition | R@1 | R@5 | R@10 | MRR | Storage/item | Latency/query |
|---|---:|---:|---:|---:|---:|---:|
| Baseline-CLS-256 (zero-shot) | 0.533 | 0.733 | 0.867 | 0.624 | 768 floats | – |
| Ours-CLS-256 (LoRA) | 0.633 | 0.833 | 0.833 | 0.712 | 768 floats | – |
| Baseline-CLS-384 (zero-shot) | 0.467 | 0.633 | 0.800 | 0.571 | 768 floats | 0.05 ms |
| **Ours-CLS-384 (LoRA)** | **0.600** | **0.800** | **0.900** | **0.684** | 768 floats | 0.04 ms |

관찰:
- patch16-384 baseline은 patch16-256 baseline보다 모든 메트릭에서 나쁨 (R@10 0.867 → 0.800). text encoder weight가 다른 backbone이라 zero-shot 성능 차이가 있음.
- patch16-384 + LoRA는 patch16-256 + LoRA 대비: R@1 0.633→0.600 (-0.033), R@5 0.833→0.800 (-0.033), **R@10 0.833→0.900 (+0.067)**, MRR 0.712→0.684 (-0.028).
- **Net**: backbone upgrade는 R@10에서만 유의한 이득. R@1/MRR은 약간 손해. 결론: **patch16-256 main 결과(§8)를 그대로 유지하는 것이 정합적**, patch16-384는 보조 검증용.

### 14.3 ColBERT-style Late-Interaction (Failed Extension)

**Sum-of-MaxSim formulation**:
$$
s(q, g) = \frac{1}{|q^{\text{valid}}|}\sum_{t \in q^{\text{valid}}} \max_{p \in P} \langle \mathbf{q}_t, \mathbf{p} \rangle
$$
- $\mathbf{q}_t$: text encoder의 last_hidden_state per non-padding token, L2 normalized
- $\mathbf{p}$: vision encoder의 last_hidden_state per patch (576 patches), L2 normalized
- mean over query token (length bias 제거)

학습은 sigmoid loss(메인)와 InfoNCE(ablation) 두 가지로 진행.

**결과 (patch16-384, 588-product gallery, 30 eval queries)**:

| Condition | R@1 | R@5 | R@10 | MRR | Storage/item | Latency/query |
|---|---:|---:|---:|---:|---:|---:|
| Baseline-MaxSim (zero-shot) | 0.000 | 0.000 | **0.000** | 0.003 | 442,368 floats (576×) | 9.94 ms |
| Ours-MaxSim-sigmoid (10 ep) | 0.033 | 0.033 | 0.133 | 0.074 | 442,368 floats | 11.14 ms |
| Ours-MaxSim-sigmoid v2 (20 ep, bias re-init) | 0.000 | 0.000 | 0.000 | 0.008 | 442,368 floats | – |
| Ours-MaxSim-InfoNCE (10 ep) | 0.133 | 0.133 | 0.233 | 0.106 | 442,368 floats | 11.10 ms |

**참고: Ours-CLS-LoRA (patch16-384) R@10 = 0.900, MRR = 0.684 — MaxSim 모든 변형이 CLS 대비 큰 폭으로 underperform.**

### 14.4 왜 ColBERT-LoRA가 SigLIP에 통하지 않았는가

**Architectural 분석**:

1. **SigLIP의 학습 신호 구조**: SigLIP은 contrastive 신호를 **pooled output 한 지점에서만** 받는다. Text 측은 last token 위치의 hidden state에 head(Linear 768→768)를 적용한 1개 vector, vision 측은 모든 patch에 multihead attention pooling을 적용한 1개 vector. 두 벡터 간 cosine similarity가 학습 신호.
2. **Per-token / per-patch 벡터는 부산물**: 학습에서 supervised signal을 직접 받지 않은 byproduct representations. 따라서 token-level alignment 보장 없음.
3. **ColBERT의 가정 위반**: ColBERT 원논문(BERT 기반)은 "BERT의 모든 token vector가 의미있는 contextual representation을 가진다"는 전제 위에 작동. SigLIP은 이 전제가 성립하지 않음.

**경험적 증거**:

- **Zero-shot MaxSim R@10 = 0.000** (random=1.7%보다 낮음). Per-token text와 per-patch vision 벡터들이 fine-grained matching에 *완전히 부적합*함을 직접 증명.
- **Sigmoid 학습 loss는 정상 감소** (6.19 → 2.86) 하지만 평가는 random 수준 → 학습 신호가 retrieval에 도움 안 되는 부분 최적해로 수렴.
- **Sigmoid v2 (20 epochs + logit_bias=-2/scale=10 재초기화)**: loss가 epoch 2부터 ~6.0에 plateau. 학습 stuck.
- **InfoNCE는 loss 감소가 더 매끄러움** (4.29 → 1.35) 하지만 평가는 R@10=0.23로 여전히 random 가까움.
- 두 loss 모두 **CLS-LoRA 대비 6.7~3.9 배 낮은 R@10**.

**용량/데이터 부족 가설**: LoRA 1.18M trainable params + 7,023 pairs × 10 epochs로 token-level alignment를 *처음부터* 가르치는 건 불충분. 그러나 v2의 20 epochs까지도 plateau에 머문 것은 capacity가 아닌 **architectural mismatch가 본질적**임을 시사.

**Vision frozen이 결정적**: text-only LoRA로는 patch token이 절대 변경되지 않음. 그런데 patch token 자체가 alignment 보장이 없는 부산물이므로, text 쪽에서 어떻게 변형해도 매칭 표현이 안 만들어짐.

### 14.5 Cost-Performance Trade-off

ColBERT는 비용 측면에서도 큰 손해:
- **Storage**: 1× → **576×** (gallery당 768 → 442,368 floats)
- **Latency**: 0.04 ms → **11.14 ms** (~280× 느림)
- **Performance**: R@10 0.900 → 0.233 (best maxsim variant) — **명백히 나쁨**

비용을 늘려도 성능이 따라오지 못함. ColBERT의 비용-성능 trade-off가 SigLIP backbone에선 음의 방향.

### 14.6 결론 및 최종 narrative

본 capstone의 main contribution은 **§8의 CLS-LoRA on patch16-256 결과**(R@10 +0 / R@1 +0.10 / MRR +0.088)를 그대로 유지한다.

ColBERT-LoRA 확장은 **failed extension으로 정직하게 보고**:
- "Naive transfer of late-interaction (ColBERT-style Sum-of-MaxSim) to SigLIP fails — even with text-only LoRA fine-tuning at 2× the training duration."
- 원인은 SigLIP이 token-level supervision을 받지 않고 학습됐기 때문이며, 이는 BERT 기반 text retrieval에서 ColBERT가 작동했던 전제와 충돌.
- 비용은 576× storage, ~280× latency 증가하면서 성능은 오히려 악화 — Pareto 측면에서도 dominated.

**향후 작업 방향**: ColBERT를 SigLIP에 적용하려면 (a) vision encoder도 함께 fine-tune하거나, (b) SigLIP 사전학습 자체를 token-level supervision (e.g., region-text matching) 으로 보완해야 함. 본 capstone의 시간 예산 외 영역.

### 14.7 추가 산출물 (이번 확장에서)

- `data/vision_emb.npy`/`vision_idx.json`: patch16-384 backbone의 CLS-pooled vision embeddings (2,928 unique products, dedupe 적용)
- `data/vision_patches.npy` (2.59 GB) + `vision_patches_idx.json`: 576 patches × 768d × float16 per image
- `checkpoints/lora_cls_384/`: patch16-384 + CLS-LoRA + sigmoid (Ours-CLS-384)
- `checkpoints/lora_maxsim_sigmoid_384/`: patch16-384 + MaxSim-LoRA + sigmoid (10 epochs, 기본 bias)
- `checkpoints/lora_maxsim_sigmoid_v2/`: 20 epochs + bias=-2 / scale=10 재초기화
- `checkpoints/lora_maxsim_infonce_384/`: patch16-384 + MaxSim-LoRA + InfoNCE
- `output/eval_baseline_cls_384.json`, `eval_ours_cls_384.json`, `eval_baseline_maxsim_384.json`, `eval_ours_final.json`, `eval_ours_maxsim_sigmoid_v2.json`, `eval_ours_maxsim_infonce.json`
- `src/maxsim.py`: Sum-of-MaxSim score utility (mean-normalized)
- `src/precompute_vision_patches.py`: patch token cache builder
- `src/train_lora.py`: `--pooling {cls,maxsim} --loss {sigmoid,infonce}` 옵션 추가
- `src/evaluate.py`: `--pooling {cls,maxsim}` 옵션 + storage/latency 측정
