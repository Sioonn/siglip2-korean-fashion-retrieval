"""Generate a HIGH-POWER synthetic evaluation set of Korean user-style search
queries on the 588 UNSEEN test (gallery) products.

Goal: gain statistical power that the 30 human queries lack, while staying on
the SAME 588-product gallery and the SAME text-tower / metric protocol so the
synth eval is directly comparable to the locked human-30 eval.

Pipeline (mirrors the training augmentation in src/run_augment.py):
  1. For each of the 588 gallery products, fill prompts/augment_final.txt with
     {product_name} and {description}=text_description (brand is already inside
     text_description, exactly as the training augmentation did).
  2. Generate 3 Korean user-style queries per product with a LOCAL vLLM model
     (EXAONE-3.5-7.8B-Instruct, fall back to Qwen2.5-7B-Instruct), CUDA dev 1.
  3. Parse / clean: strip numbering+quotes, drop empty / >120 char / non-Korean /
     refusals / verbatim-caption echoes, dedup within a product.
  4. Cache raw -> data/novel_cache/synth_raw.json
  5. Write data/novel_cache/synth_eval.jsonl (one query per line:
     {product_id, query, qtype in {q1_short,q2_medium,q3_aspect}}).
  6. Encode FINAL queries (same jsonl order) under BOTH encoders:
       base    (lora_path=None)
       lora384 (checkpoints/lora_cls_384)
     -> data/novel_cache/{base,lora384}__synth_eval_q.npy
  7. Verify lib.load_synth_eval() loads and emb shapes match.

Run from repo root:
  CUDA_VISIBLE_DEVICES=1 python -m src.novel.gen_synth_eval
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.novel import lib  # noqa: E402

PROMPT_PATH = ROOT / "prompts" / "augment_final.txt"
RAW_PATH = lib.CACHE_DIR / "synth_raw.json"
JSONL_PATH = lib.CACHE_DIR / "synth_eval.jsonl"
LORA_PATH = ROOT / "checkpoints" / "lora_cls_384"

EXAONE = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
QWEN = "Qwen/Qwen2.5-7B-Instruct"

# qtype label per query slot (mirrors training query_type 1/2/3)
QTYPE = {"1": "q1_short", "2": "q2_medium", "3": "q3_aspect"}

QUERY_RE = re.compile(
    r"쿼리\s*([123])\s*[:：]\s*(.+?)(?=\n\s*쿼리\s*[123]\s*[:：]|\Z)", re.DOTALL
)

HANGUL_RE = re.compile(r"[가-힣]")
REFUSAL_TOKENS = (
    "죄송", "도와드릴 수 없", "도와 드릴 수 없", "as an ai", "i cannot",
    "할 수 없습니다", "제공할 수 없", "생성할 수 없",
)


def parse_queries(raw: str) -> dict:
    """Same parser as src/run_augment.py."""
    out = {}
    for m in QUERY_RE.finditer(raw or ""):
        idx = m.group(1)
        text = m.group(2).strip()
        text = re.sub(r"^[-*•\s\"'<]+", "", text).rstrip("\"'>").strip()
        text = re.sub(r"\s+", " ", text)
        out[idx] = text
    return out


def normalize_caption(s: str) -> str:
    return re.sub(r"[\s\.,;:!?~\-\(\)\[\]·]+", "", (s or "").lower())


def is_clean(query: str, caption_norm: str) -> tuple[bool, str]:
    """Return (keep, reason_if_dropped)."""
    q = query.strip()
    if not q:
        return False, "empty"
    if len(q) > 120:
        return False, "too_long"
    if not HANGUL_RE.search(q):
        return False, "no_hangul"
    low = q.lower()
    if any(tok in low for tok in REFUSAL_TOKENS):
        return False, "refusal"
    qn = normalize_caption(q)
    # echoes the formal caption verbatim (whole query is a substring of caption)
    if len(qn) >= 12 and qn in caption_norm:
        return False, "caption_echo"
    return True, ""


def build_prompts(products: list[dict], template: str) -> list[str]:
    prompts = []
    for it in products:
        name = it.get("product_name") or ""
        desc = it.get("text_description") or ""
        prompts.append(template.replace("{product_name}", name).replace("{description}", desc))
    return prompts


def load_llm():
    """Try EXAONE, fall back to Qwen. Returns (llm, sampling_params, model_name)."""
    from vllm import LLM, SamplingParams

    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=320, seed=lib.SEED)
    last_err = None
    for model_name, trust in [(EXAONE, True), (QWEN, True)]:
        try:
            print(f"[llm] loading {model_name} ...", flush=True)
            llm = LLM(
                model=model_name,
                trust_remote_code=trust,
                dtype="bfloat16",
                gpu_memory_utilization=0.85,
                max_model_len=4096,
                enforce_eager=False,
            )
            print(f"[llm] loaded {model_name}", flush=True)
            return llm, sp, model_name
        except Exception as e:  # noqa: BLE001
            print(f"[llm] FAILED to load {model_name}: {type(e).__name__}: {e}", flush=True)
            last_err = e
    raise RuntimeError(f"could not load any LLM: {last_err}")


def generate_raw(products: list[dict], template: str):
    """Returns list of {product_id, raw} aligned to products, plus model name."""
    prompts = build_prompts(products, template)
    llm, sp, model_name = load_llm()
    # chat-format each prompt so the model uses its instruct template
    messages = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(messages, sp)
    raws = []
    for it, out in zip(products, outputs):
        raws.append({"product_id": it["product_id"], "raw": out.outputs[0].text})
    return raws, model_name


def main():
    data = lib.load_data()
    products = data.gallery_products  # 588, gallery order == meta.json order
    assert len(products) == 588, f"expected 588 gallery products, got {len(products)}"
    template = PROMPT_PATH.read_text()

    # ---- raw generation (cache to avoid re-running the LLM) ----
    if RAW_PATH.exists():
        cached = json.loads(RAW_PATH.read_text())
        raws = cached["raws"]
        model_name = cached.get("model", "cached")
        cached_pids = [r["product_id"] for r in raws]
        if cached_pids == [p["product_id"] for p in products]:
            print(f"[raw] using cached generations from {RAW_PATH} (model={model_name})", flush=True)
        else:
            print("[raw] cache pid mismatch -> regenerating", flush=True)
            raws, model_name = generate_raw(products, template)
            RAW_PATH.write_text(json.dumps({"model": model_name, "raws": raws}, ensure_ascii=False))
    else:
        raws, model_name = generate_raw(products, template)
        RAW_PATH.write_text(json.dumps({"model": model_name, "raws": raws}, ensure_ascii=False))
        print(f"[raw] wrote {RAW_PATH} (model={model_name})", flush=True)

    raw_by_pid = {r["product_id"]: r["raw"] for r in raws}

    # ---- parse + clean -> jsonl ----
    pid_to_caption = {
        p["product_id"]: normalize_caption(
            (p.get("text_description") or "") + (p.get("product_name") or "")
        )
        for p in products
    }

    rows = []
    drop_counts = {}
    per_qtype = {"q1_short": 0, "q2_medium": 0, "q3_aspect": 0}
    n_partial = 0
    for p in products:  # iterate in gallery order for deterministic jsonl order
        pid = p["product_id"]
        parsed = parse_queries(raw_by_pid.get(pid, ""))
        if len(parsed) < 3:
            n_partial += 1
        caption_norm = pid_to_caption[pid]
        seen = set()  # dedup within product (normalized)
        for k in ("1", "2", "3"):
            q = parsed.get(k, "")
            keep, reason = is_clean(q, caption_norm)
            if not keep:
                drop_counts[reason] = drop_counts.get(reason, 0) + 1
                continue
            qn = normalize_caption(q)
            if qn in seen:
                drop_counts["dup_in_product"] = drop_counts.get("dup_in_product", 0) + 1
                continue
            seen.add(qn)
            qtype = QTYPE[k]
            rows.append({"product_id": pid, "query": q, "qtype": qtype})
            per_qtype[qtype] += 1

    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[jsonl] wrote {len(rows)} queries -> {JSONL_PATH}", flush=True)
    print(f"[jsonl] per_qtype={per_qtype} partial(<3 parsed)={n_partial}", flush=True)
    print(f"[jsonl] drops={drop_counts}", flush=True)

    # ---- encode under BOTH encoders, in jsonl order ----
    texts = [r["query"] for r in rows]
    print("[encode] base ...", flush=True)
    base_emb = lib.encode_cached(texts, None, "base", "synth_eval_q")
    print("[encode] lora384 ...", flush=True)
    lora_emb = lib.encode_cached(texts, str(LORA_PATH), "lora384", "synth_eval_q")
    print(f"[encode] base={base_emb.shape} lora384={lora_emb.shape}", flush=True)

    # ---- verify ----
    se = lib.load_synth_eval()
    ok = (
        se is not None
        and se["n"] == len(rows)
        and se["enc"]["base"].shape[0] == len(rows)
        and se["enc"]["lora384"].shape[0] == len(rows)
        and se["enc"]["base"].shape[1] == se["enc"]["lora384"].shape[1]
    )
    print(f"[verify] load_synth_eval n={se['n'] if se else None} loads_ok={ok}", flush=True)

    summary = {
        "model_used": model_name,
        "n_products": len(products),
        "n_queries": len(rows),
        "per_qtype": per_qtype,
        "loads_ok": bool(ok),
        "drops": drop_counts,
        "sample_queries": texts[:8],
    }
    (lib.CACHE_DIR / "synth_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print("[done] summary written to synth_eval_summary.json", flush=True)
    return summary


if __name__ == "__main__":
    main()
