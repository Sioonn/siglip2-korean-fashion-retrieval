# 한국어 패션 Text-to-Image 검색: 분포 정렬 기반 경량 파이프라인

SigLIP 2 백본 위에서 **텍스트-only LoRA 분포 정렬 + 후보보존 BM25 융합 + 뱅크-prior 보정**으로
한국어 자연어 쿼리 → 패션 상품 이미지 검색을 개선하는 캡스톤 연구 코드.

## 문제

무신사 같은 패션 플랫폼은 키워드·카테고리·브랜드 필터만 지원해, "회색 후드티에 가슴 네이비 로고"처럼
자연어로 묘사한 쿼리를 다루지 못한다. 사전학습 vision-language model(SigLIP 2)을 zero-shot으로 쓰면
원리상 가능하지만 검색 품질이 낮다(gold MRR 0.571). 기존 패션 image-text 데이터셋
(FashionPedia·FACAD·FashionGen)은 영어·장문·전문용어 중심이라 짧은 구어체 한국어 쿼리와 분포가
멀어 그대로 쓰기 어렵다. 핵심 약점은 **상품 텍스트와 사용자 쿼리 사이의 분포 격차**다.

## 방법 (파이프라인)

1. **캡션 생성** — 무신사 상의 상품 이미지를 GPT-4o로 캡셔닝(평균 ~512자). 사용자가 어떤 특징을
   검색할지 알 수 없으므로 옷의 가능한 모든 시각 특징을 망라하도록 길게 생성.
2. **합성 쿼리 뱅크** — 각 캡션에서 solar-pro3로 사용자-스타일 한국어 쿼리를 상품당 3개씩 생성(총 7,023개).
3. **텍스트-only LoRA 정렬** — SigLIP 2 텍스트 타워에만 LoRA(r=16) 부착, vision 타워 동결,
   in-batch contrastive로 합성 쿼리 ↔ 상품 이미지 임베딩을 정렬. 갤러리 임베딩은 캐시 재사용.
4. **후보보존 BM25 융합** — 추론 시 1차 dense top-_n_ 후보 *안에서만* 한국어 메타데이터의
   char n-gram BM25 점수를 dense 점수에 가중 합산(후보 밖 오검출로 recall이 깎이지 않게 제약).
5. **뱅크-prior 보정** — 학습용 합성 뱅크를 추론 시점 "사용자 쿼리 분포의 prior"로 재활용해
   modality gap·hubness를 추가 학습·라벨 없이(training-free) 보정.

## 핵심 결과 (gold human-30, 실제 사용자 쿼리)

| 단계 | MRR | R@1 | R@10 |
|---|---:|---:|---:|
| zero-shot SigLIP 2 | 0.571 | 0.47 | 0.80 |
| + 텍스트-only LoRA | 0.684 | 0.60 | 0.90 |
| + 후보보존 BM25 융합 | 0.721 | 0.63 | 0.87 |
| + 뱅크-prior (CSLS) | 0.760 | 0.70 | 0.87 |

- 텍스트-only LoRA의 zero-shot 대비 향상은 **gold(sign-test p=0.013)와 고검정력 합성 평가
  (N=1,706, paired bootstrap p≈0) 양쪽에서 유의**하다.
- 전체 파이프라인은 gold에서 zero-shot을 유의하게 능가하나(paired sign-test p<0.01),
  LoRA 이후 개별 증분은 N=30 gold에서 통계적으로 구분되지 않는다(검정력 한계).
- 훨씬 큰 현대 VLM 임베더 **GME-Qwen2VL-2B는 본 도메인에서 이 경량 파이프라인을 넘지 못한다**
  — 좁은 도메인에서는 임베더 규모보다 도메인 분포 정렬이 더 중요함을 시사.

## 데이터

- 무신사 상의 약 2,938개 상품(이미지 보유), product 단위 80/20 분할(train 2,350 / test 588, seed 42).
- 검색 갤러리 = test 588개 상품 이미지.
- 평가 = 사람이 작성한 30개 쿼리(gold) + EXAONE-3.5-7.8B로 생성한 고검정력 합성 1,706개.
- 이미지·임베딩 캐시 등 대용량·재생성 가능 데이터는 저장소에 포함하지 않는다(`.gitignore`).

## 저장소 구조

```
src/                 학습·검색·평가 스크립트 (train_lora.py 등) 및 추가 실험 변형
src/novel/           평가 하니스(lib.py), 단계별 누적 평가(run_final_suite.py),
                     GME baseline, CSLS/QB-Norm/modality-gap 보정, figures, precompute
scripts/             데이터 준비·집계 유틸리티
prompts/             캡션·합성 쿼리 생성 프롬프트
config.yaml          경로·하이퍼파라미터 단일 출처
data/                분할/쿼리 JSON (이미지·임베딩은 제외)
```

자세한 평가 하니스 설명은 [`src/novel/README.md`](src/novel/README.md) 참조.

## 사용 모델

| 역할 | 모델 | 위치 |
|---|---|---|
| 검색 backbone | `google/siglip2-base-patch16-384` | 로컬 |
| 상품 캡셔닝 | GPT-4o | 외부 API |
| 학습용 합성 쿼리 | solar-pro3 (Upstage) | 외부 API |
| 고검정력 합성 평가 쿼리 | EXAONE-3.5-7.8B-Instruct | 로컬 vLLM |
| 비교 baseline 임베더 | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` | 로컬 |

캡셔닝·학습 쿼리 생성만 외부 API를 쓰며, 검색 방법·LoRA 학습·모든 보정·모든 평가는 단일 A6000급
GPU에서 로컬로 수행된다.

## 실행 (개요)

레포 루트에서 실행하며, 인터프리터는 프로젝트 conda 환경을 사용한다. 상품 이미지와 vision 임베딩은
별도로 준비해야 한다(저장소에 미포함).

```bash
# 1) 텍스트-only LoRA 학습
python src/train_lora.py

# 2) base/lora 인코더로 평가·뱅크·갤러리 임베딩 사전계산·캐시
python -m src.novel.precompute

# 3) 단계별 누적 평가 (gold + 고검정력 합성), bootstrap CI·paired test 포함
python -m src.novel.run_final_suite
```

모든 경로·하이퍼파라미터는 `config.yaml`에 있다.
