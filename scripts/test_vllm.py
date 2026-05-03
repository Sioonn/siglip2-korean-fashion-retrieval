"""Smoke test: load EXAONE 7.8B in vLLM and run a Korean fashion query generation."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

from vllm import LLM, SamplingParams

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    trust_remote_code=True,
)

prompt = """당신은 한국어 패션 검색 시스템의 데이터 생성 도우미입니다.

다음 상품 설명을 보고, 사용자가 입력할 법한 짧은 한국어 검색 쿼리 1개만 생성하세요.

상품: 그레이 색상의 후드티, 가슴에 네이비 영어 로고와 빨간 글씨가 프린트되어 있음.

사용자 쿼리:"""

sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
out = llm.generate([prompt], sp)
print("=" * 60)
print("OUTPUT:")
print(out[0].outputs[0].text)
print("=" * 60)
