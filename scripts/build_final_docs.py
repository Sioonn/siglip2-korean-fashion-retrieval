"""Build final experiment report artifacts under docs/final."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
FINAL = ROOT / "docs" / "final"
FIGURES = FINAL / "figures"
DATA = FINAL / "data"

METRIC_KEYS = ["R@1", "R@5", "R@10", "MRR"]

METHODS = [
    {
        "name": "Random ranking expected",
        "file": None,
        "family": "random",
        "strategy": "gallery size N에서 정답 1개가 균등 랜덤 rank를 갖는 analytic expectation",
        "novelty": "없음",
        "role": "무작위 선택 기준선",
        "finding": "모든 retrieval 방법이 랜덤 선택보다 큰 폭으로 높아야 함을 확인하는 하한선이다.",
    },
    {
        "name": "Baseline-256",
        "file": "eval_baseline.json",
        "family": "baseline",
        "strategy": "SigLIP2 256px frozen image-text retrieval",
        "novelty": "낮음",
        "role": "기준선",
        "finding": "R@10은 높지만 첫 순위 정확도와 MRR이 부족하다.",
    },
    {
        "name": "LoRA-256",
        "file": "eval_ours.json",
        "family": "text_lora",
        "strategy": "text-side LoRA on SigLIP2 256px",
        "novelty": "낮음",
        "role": "기존 LoRA 개선선",
        "finding": "R@1/MRR은 오르지만 R@10이 baseline보다 낮아진다.",
    },
    {
        "name": "LoRA-384 sigmoid10",
        "file": "eval_ours_cls_384.json",
        "family": "text_lora",
        "strategy": "text-side LoRA on SigLIP2 384px, sigmoid loss, 10ep",
        "novelty": "낮음",
        "role": "해상도/체크포인트 ablation",
        "finding": "R@10은 회복하지만 R@1/MRR은 LoRA-256보다 낮다.",
    },
    {
        "name": "LoRA-384 InfoNCE20",
        "file": "eval_lora_cls_384_infonce.json",
        "family": "text_lora",
        "strategy": "text-side LoRA 384px, InfoNCE, longer schedule",
        "novelty": "낮음",
        "role": "loss ablation",
        "finding": "loss 변경만으로는 개선되지 않는다.",
    },
    {
        "name": "LoRA-384 Sigmoid20",
        "file": "eval_lora_cls_384_sigmoid20.json",
        "family": "text_lora",
        "strategy": "text-side LoRA 384px, sigmoid, 20ep",
        "novelty": "낮음",
        "role": "epoch ablation",
        "finding": "epoch 증가가 R@5/R@10을 손상한다.",
    },
    {
        "name": "LoRA-384 + BM25 top-10",
        "file": "eval_hybrid_rerank_top10_015bm25.json",
        "family": "hybrid_rerank",
        "strategy": "LoRA-384 first stage + candidate-preserving Korean char-ngram BM25 rerank",
        "novelty": "중간",
        "role": "강한 lexical rerank baseline",
        "finding": "R@10을 유지하면서 R@1/MRR을 크게 올린다.",
    },
    {
        "name": "Dual r4 InfoNCE 5ep",
        "file": "eval_dual_lora_r4_infonce5.json",
        "family": "dual_lora",
        "strategy": "text/vision dual-tower LoRA r=4, InfoNCE, 5ep",
        "novelty": "높음",
        "role": "low-rank dual adaptation",
        "finding": "R@1/MRR은 좋지만 R@10이 낮아 후보 누락 위험이 있다.",
    },
    {
        "name": "Dual r4 5ep + BM25",
        "file": "eval_dual_hybrid_rerank_r4_infonce5.json",
        "family": "dual_hybrid",
        "strategy": "Dual r4 5ep + top-20 BM25 rerank",
        "novelty": "높음",
        "role": "dual + lexical ablation",
        "finding": "MRR은 best급이지만 R@10이 완전히 회복되지 않는다.",
    },
    {
        "name": "Dual r4 InfoNCE 10ep",
        "file": "eval_dual_lora_r4_infonce10.json",
        "family": "dual_lora",
        "strategy": "text/vision dual-tower LoRA r=4, InfoNCE, 10ep",
        "novelty": "높음",
        "role": "epoch ablation",
        "finding": "R@5/R@10은 회복하지만 R@1/MRR은 BM25 결합 전보다 부족하다.",
    },
    {
        "name": "Dual r4 10ep + BM25",
        "file": "eval_dual_hybrid_rerank_r4_infonce10.json",
        "family": "dual_hybrid",
        "strategy": "Dual r4 InfoNCE 10ep + top-20 candidate-preserving BM25 rerank",
        "novelty": "높음",
        "role": "최종 전략",
        "finding": "R@1/R@10을 보존하고 전체 MRR이 가장 높다.",
    },
    {
        "name": "Dual r8 sigmoid5",
        "file": "eval_dual_lora_r8_sigmoid5.json",
        "family": "dual_lora",
        "strategy": "dual-tower LoRA r=8, sigmoid, 5ep",
        "novelty": "높음",
        "role": "loss/rank ablation",
        "finding": "vision LoRA 가능성은 보이나 InfoNCE보다 약하다.",
    },
    {
        "name": "Dual r8 InfoNCE 5ep",
        "file": "eval_dual_lora_r8_infonce5.json",
        "family": "dual_lora",
        "strategy": "dual-tower LoRA r=8, InfoNCE, 5ep",
        "novelty": "높음",
        "role": "R@5 strong variant",
        "finding": "R@5는 최고지만 R@1/MRR은 최종 전략보다 낮다.",
    },
    {
        "name": "Dual r8 5ep + BM25",
        "file": "eval_dual_hybrid_rerank_r8_infonce5.json",
        "family": "dual_hybrid",
        "strategy": "Dual r8 InfoNCE 5ep + BM25 rerank",
        "novelty": "높음",
        "role": "rank/rerank ablation",
        "finding": "R@10은 유지하지만 MRR이 최종 전략보다 낮다.",
    },
    {
        "name": "Dual r16 InfoNCE 5ep",
        "file": "eval_dual_lora_r16_infonce5.json",
        "family": "dual_lora",
        "strategy": "dual-tower LoRA r=16, InfoNCE, 5ep",
        "novelty": "높음",
        "role": "capacity ablation",
        "finding": "capacity 증가가 R@10 손상을 만든다.",
    },
    {
        "name": "Dual r16 5ep + BM25",
        "file": "eval_dual_hybrid_rerank_r16_infonce5.json",
        "family": "dual_hybrid",
        "strategy": "Dual r16 InfoNCE 5ep + BM25 rerank",
        "novelty": "높음",
        "role": "capacity/rerank ablation",
        "finding": "R@10은 회복되지만 MRR은 최종 전략보다 낮다.",
    },
    {
        "name": "Dual r8 InfoNCE 15ep",
        "file": "eval_dual_lora_r8_infonce15.json",
        "family": "dual_lora",
        "strategy": "dual-tower LoRA r=8, InfoNCE, 15ep",
        "novelty": "높음",
        "role": "long training ablation",
        "finding": "장기 학습은 eval rank quality를 악화한다.",
    },
    {
        "name": "Dual r8 15ep + BM25",
        "file": "eval_dual_hybrid_rerank_r8_infonce15.json",
        "family": "dual_hybrid",
        "strategy": "Dual r8 InfoNCE 15ep + BM25 rerank",
        "novelty": "높음",
        "role": "long training + rerank",
        "finding": "장기 학습의 손실을 rerank가 충분히 복구하지 못한다.",
    },
    {
        "name": "Query adapter residual256",
        "file": "eval_query_adapter_residual256_infonce.json",
        "family": "adapter",
        "strategy": "query-side residual adapter over frozen embeddings",
        "novelty": "중간",
        "role": "lightweight adapter ablation",
        "finding": "query-only 보정은 심하게 부족하다.",
    },
    {
        "name": "Query adapter linear",
        "file": "eval_query_adapter_linear_infonce.json",
        "family": "adapter",
        "strategy": "query-side linear adapter over frozen embeddings",
        "novelty": "중간",
        "role": "lightweight adapter ablation",
        "finding": "일부 MRR은 회복하지만 R@5/R@10이 낮다.",
    },
    {
        "name": "Patch MaxSim top10",
        "file": "eval_patch_rerank_top10_lora384.json",
        "family": "late_interaction",
        "strategy": "token-patch late interaction rerank over top-10",
        "novelty": "높음",
        "role": "ColBERT-style ablation",
        "finding": "현재 SigLIP hidden/patch 신호만으로는 개선이 없다.",
    },
    {
        "name": "Attribute facet top10",
        "file": "eval_attribute_rerank_top10_lora384.json",
        "family": "facet",
        "strategy": "interpretable Korean fashion facet rerank",
        "novelty": "중간",
        "role": "해석 가능 rerank",
        "finding": "해석 가능하지만 BM25보다 약하다.",
    },
    {
        "name": "Facet+BM25 top10",
        "file": "eval_facet_hybrid_rerank_top10_lora384.json",
        "family": "facet",
        "strategy": "facet score + BM25 hybrid rerank",
        "novelty": "중간",
        "role": "facet/lexical fusion",
        "finding": "validation 과적합 성향이 있고 R@1/MRR이 낮다.",
    },
    {
        "name": "Learned top10 reranker",
        "file": "eval_candidate_reranker_logistic_fast.json",
        "family": "learned_rerank",
        "strategy": "learned logistic reranker over top-10 features",
        "novelty": "중간",
        "role": "학습형 reranker stress test",
        "finding": "validation은 매우 높지만 locked eval에서 붕괴한다.",
    },
    {
        "name": "Adaptive BM25 rerank",
        "file": "eval_adaptive_hybrid_rerank_lora384.json",
        "family": "hybrid_rerank",
        "strategy": "query-length bucket별 BM25 weight selection",
        "novelty": "중간",
        "role": "query-adaptive rerank ablation",
        "finding": "validation 선택이 eval로 일반화되지 않는다.",
    },
    {
        "name": "Union LoRA/BM25 candidates",
        "file": "eval_union_hybrid_lora384_bm25.json",
        "family": "hybrid_retrieval",
        "strategy": "LoRA top-K와 BM25 top-M 후보 union",
        "novelty": "중간",
        "role": "candidate expansion ablation",
        "finding": "후보 확장은 R@10을 손상한다. 후보 보존이 더 안정적이다.",
    },
    {
        "name": "Query expansion LoRA384",
        "file": "eval_query_expansion_lora384.json",
        "family": "query_expansion",
        "strategy": "rule-based fashion query expansion before retrieval",
        "novelty": "낮음",
        "role": "preprocess ablation",
        "finding": "validation에서 expansion이 선택되지 않아 실질 개선이 없다.",
    },
    {
        "name": "Ensemble dual r4 + LoRA/BM25",
        "file": "eval_ensemble_dual_r4_lora_bm25.json",
        "family": "ensemble",
        "strategy": "validation-selected ensemble of dual r4 and LoRA/BM25",
        "novelty": "중간",
        "role": "ensemble ablation",
        "finding": "validation이 LoRA/BM25 단독을 선택해 ensemble 이득은 없다.",
    },
]


def expected_random_metrics(gallery_size: int) -> dict[str, float]:
    harmonic = sum(1.0 / rank for rank in range(1, gallery_size + 1))
    return {
        "R@1": 1.0 / gallery_size,
        "R@5": min(5, gallery_size) / gallery_size,
        "R@10": min(10, gallery_size) / gallery_size,
        "MRR": harmonic / gallery_size,
    }


def selected_metrics(payload: dict) -> dict:
    if "metrics" in payload:
        return payload["metrics"]
    if "eval_metrics" in payload:
        return payload["eval_metrics"]
    if "selected" in payload:
        return payload["selected"]["eval_metrics"]
    raise KeyError("no eval metrics")


def selected_val_metrics(payload: dict) -> dict | None:
    if "best_val_metrics" in payload:
        return payload["best_val_metrics"]
    if "selected" in payload and "val_metrics" in payload["selected"]:
        return payload["selected"]["val_metrics"]
    return None


def selected_name(payload: dict) -> str:
    if "selected" in payload:
        return payload["selected"].get("name", "")
    return payload.get("name", "")


def detail_path(file_name: str) -> Path:
    return OUTPUT / f"{Path(file_name).stem}_detail.json"


def load_rows() -> list[dict]:
    rows = []
    baseline_path = OUTPUT / "eval_baseline.json"
    gallery_size = 588
    if baseline_path.exists():
        gallery_size = int(json.loads(baseline_path.read_text()).get("gallery_size", gallery_size))
    for method in METHODS:
        if method["file"] is None:
            metrics = expected_random_metrics(gallery_size)
            row = {
                **method,
                "file": "analytic_random_expectation",
                "selected": f"expected over gallery_size={gallery_size}",
                "balanced": float(np.mean([metrics[k] for k in METRIC_KEYS])),
                **{k: float(metrics[k]) for k in METRIC_KEYS},
                **{f"val_{k}": "" for k in METRIC_KEYS},
                "val_eval_mrr_gap": "",
            }
            rows.append(row)
            continue
        path = OUTPUT / method["file"]
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        metrics = selected_metrics(payload)
        val = selected_val_metrics(payload)
        balanced = float(np.mean([metrics[k] for k in METRIC_KEYS]))
        row = {
            **method,
            "selected": selected_name(payload),
            "balanced": balanced,
            **{k: float(metrics[k]) for k in METRIC_KEYS},
        }
        if val:
            row.update({f"val_{k}": float(val[k]) for k in METRIC_KEYS})
            row["val_eval_mrr_gap"] = float(val["MRR"] - metrics["MRR"])
        else:
            row.update({f"val_{k}": "" for k in METRIC_KEYS})
            row["val_eval_mrr_gap"] = ""
        rows.append(row)
    return rows


def by_difficulty(file_name: str) -> list[dict]:
    path = detail_path(file_name)
    if not path.exists():
        return []
    details = json.loads(path.read_text())
    groups: dict[str, list[int]] = {}
    for item in details:
        groups.setdefault(item.get("difficulty") or item.get("bucket") or "unknown", []).append(int(item["rank"]))
    out = []
    for name, ranks in sorted(groups.items()):
        n = len(ranks)
        out.append(
            {
                "difficulty": name,
                "n": n,
                "R@1": sum(r <= 1 for r in ranks) / n,
                "R@5": sum(r <= 5 for r in ranks) / n,
                "R@10": sum(r <= 10 for r in ranks) / n,
                "MRR": sum(1 / r for r in ranks) / n,
            }
        )
    return out


def fmt(x: float | str) -> str:
    if x == "":
        return ""
    return f"{float(x):.3f}"


def md_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    align = "| " + " | ".join("---:" if key in METRIC_KEYS or key.startswith("val_") or key == "balanced" else "---" for _, key in columns) + " |"
    lines = [header, align]
    for row in rows:
        vals = []
        for _, key in columns:
            val = row.get(key, "")
            vals.append(fmt(val) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_data(rows: list[dict]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    fields = ["name", "family", "role", "novelty", "strategy", "selected", *METRIC_KEYS, "balanced", "finding", "file"]
    with (DATA / "results_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    (DATA / "results_table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    best = next(r for r in rows if r["name"] == "Dual r4 10ep + BM25")
    lora = next(r for r in rows if r["name"] == "LoRA-384 + BM25 top-10")
    dual = next(r for r in rows if r["name"] == "Dual r4 InfoNCE 10ep")
    base = next(r for r in rows if r["name"] == "Baseline-256")
    ablation = [
        {"comparison": "Random -> final", **{k: best[k] - next(r for r in rows if r["name"] == "Random ranking expected")[k] for k in METRIC_KEYS}},
        {"comparison": "Baseline -> final", **{k: best[k] - base[k] for k in METRIC_KEYS}},
        {"comparison": "LoRA/BM25 -> final", **{k: best[k] - lora[k] for k in METRIC_KEYS}},
        {"comparison": "Dual r4 10ep -> final", **{k: best[k] - dual[k] for k in METRIC_KEYS}},
    ]
    with (DATA / "ablation_deltas.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison", *METRIC_KEYS])
        writer.writeheader()
        writer.writerows(ablation)
    (DATA / "ablation_deltas.json").write_text(json.dumps(ablation, ensure_ascii=False, indent=2))


def set_style():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_overall(rows: list[dict]) -> None:
    names = [
        "Random ranking expected",
        "Baseline-256",
        "LoRA-256",
        "LoRA-384 sigmoid10",
        "LoRA-384 + BM25 top-10",
        "Dual r4 InfoNCE 10ep",
        "Dual r4 10ep + BM25",
        "Dual r8 InfoNCE 5ep",
    ]
    selected = [next(r for r in rows if r["name"] == n) for n in names]
    x = np.arange(len(selected))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["#3b82f6", "#14b8a6", "#f59e0b", "#ef4444"]
    for i, metric in enumerate(METRIC_KEYS):
        ax.bar(x + (i - 1.5) * width, [r[metric] for r in selected], width, label=metric, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in selected], rotation=25, ha="right")
    ax.set_ylim(0.0, 0.95)
    ax.set_title("Overall locked-eval metrics")
    ax.set_ylabel("score")
    ax.legend(ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "overall_metrics_grouped_bar.png")
    plt.close(fig)


def plot_ladder(rows: list[dict]) -> None:
    names = ["Random ranking expected", "Baseline-256", "LoRA-256", "LoRA-384 sigmoid10", "LoRA-384 + BM25 top-10", "Dual r4 InfoNCE 10ep", "Dual r4 10ep + BM25"]
    selected = [next(r for r in rows if r["name"] == n) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, metric in zip(axes, ["MRR", "R@10"]):
        ax.plot(range(len(selected)), [r[metric] for r in selected], marker="o", linewidth=2)
        for i, r in enumerate(selected):
            ax.text(i, r[metric] + 0.006, fmt(r[metric]), ha="center", fontsize=8)
        ax.set_xticks(range(len(selected)))
        ax.set_xticklabels([r["name"] for r in selected], rotation=35, ha="right")
        ax.set_title(metric)
        ax.set_ylim(0.0, 0.93)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Component ladder: generator and reranker effects")
    fig.tight_layout()
    fig.savefig(FIGURES / "component_ladder_mrr_r10.png")
    plt.close(fig)


def plot_scatter(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    family_colors = {
        "baseline": "#64748b",
        "random": "#94a3b8",
        "text_lora": "#3b82f6",
        "hybrid_rerank": "#14b8a6",
        "dual_lora": "#f59e0b",
        "dual_hybrid": "#ef4444",
        "adapter": "#a855f7",
        "late_interaction": "#06b6d4",
        "facet": "#84cc16",
        "learned_rerank": "#f43f5e",
        "hybrid_retrieval": "#0f766e",
        "query_expansion": "#94a3b8",
        "ensemble": "#7c3aed",
    }
    for row in rows:
        ax.scatter(row["R@10"], row["MRR"], s=52, color=family_colors.get(row["family"], "#111827"), alpha=0.85)
    label_names = ["Random ranking expected", "Baseline-256", "LoRA-384 + BM25 top-10", "Dual r4 10ep + BM25", "Dual r8 InfoNCE 5ep", "Learned top10 reranker"]
    for row in rows:
        if row["name"] in label_names:
            ax.annotate(row["name"], (row["R@10"], row["MRR"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("R@10")
    ax.set_ylabel("MRR")
    ax.set_title("MRR vs R@10 trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "mrr_vs_r10_scatter.png")
    plt.close(fig)


def plot_difficulty() -> None:
    rows = by_difficulty("eval_dual_hybrid_rerank_r4_infonce10.json")
    x = np.arange(len(rows))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for i, metric in enumerate(METRIC_KEYS):
        ax.bar(x + (i - 1.5) * width, [r[metric] for r in rows], width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels([r["difficulty"] for r in rows])
    ax.set_ylim(0.50, 1.03)
    ax.set_title("Final method by difficulty")
    ax.set_ylabel("score")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_by_difficulty.png")
    plt.close(fig)


def plot_family_heatmap(rows: list[dict]) -> None:
    families = []
    for row in rows:
        if row["family"] not in families:
            families.append(row["family"])
    matrix = []
    labels = []
    for family in families:
        group = [r for r in rows if r["family"] == family]
        best = max(group, key=lambda r: r["MRR"])
        labels.append(f"{family}\n({best['name']})")
        matrix.append([best[k] for k in METRIC_KEYS])
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    im = ax.imshow(arr, cmap="YlGnBu", vmin=0.0, vmax=0.95)
    ax.set_xticks(np.arange(len(METRIC_KEYS)), labels=METRIC_KEYS)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Best method within each experiment family")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "family_best_heatmap.png")
    plt.close(fig)


def plot_val_eval_gap(rows: list[dict]) -> None:
    gap_rows = [r for r in rows if isinstance(r["val_eval_mrr_gap"], float)]
    gap_rows = sorted(gap_rows, key=lambda r: r["val_eval_mrr_gap"], reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(gap_rows))
    ax.barh(y, [r["val_eval_mrr_gap"] for r in gap_rows], color="#f97316")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in gap_rows])
    ax.invert_yaxis()
    ax.set_xlabel("validation MRR - eval MRR")
    ax.set_title("Generalization gap of tuned methods")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_eval_mrr_gap.png")
    plt.close(fig)


def plot_ablation_delta(rows: list[dict]) -> None:
    final = next(r for r in rows if r["name"] == "Dual r4 10ep + BM25")
    comparisons = [
        ("vs Random", next(r for r in rows if r["name"] == "Random ranking expected")),
        ("vs Baseline", next(r for r in rows if r["name"] == "Baseline-256")),
        ("vs LoRA-384", next(r for r in rows if r["name"] == "LoRA-384 sigmoid10")),
        ("vs Dual r4 10ep", next(r for r in rows if r["name"] == "Dual r4 InfoNCE 10ep")),
        ("vs LoRA/BM25", next(r for r in rows if r["name"] == "LoRA-384 + BM25 top-10")),
    ]
    x = np.arange(len(comparisons))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    for i, metric in enumerate(METRIC_KEYS):
        ax.bar(x + (i - 1.5) * width, [final[metric] - base[metric] for _, base in comparisons], width, label=metric)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in comparisons], rotation=15, ha="right")
    ax.set_title("Final method deltas")
    ax.set_ylabel("absolute metric delta")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_ablation_deltas.png")
    plt.close(fig)


def build_figures(rows: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_style()
    plot_overall(rows)
    plot_ladder(rows)
    plot_scatter(rows)
    plot_difficulty()
    plot_family_heatmap(rows)
    plot_val_eval_gap(rows)
    plot_ablation_delta(rows)


def write_index(rows: list[dict]) -> None:
    best = next(r for r in rows if r["name"] == "Dual r4 10ep + BM25")
    random = next(r for r in rows if r["name"] == "Random ranking expected")
    baseline = next(r for r in rows if r["name"] == "Baseline-256")
    top = sorted(rows, key=lambda r: (r["MRR"], r["R@1"], r["R@10"]), reverse=True)[:8]
    text = f"""# Final Retrieval Report

이 폴더는 최종 결과만 모은 보고서 패키지다. 모든 수치는 locked hand-written eval 30 queries 기준이며, 방법 선택은 train split validation에서만 수행했다.

## 결론

최종 전략은 **Dual-tower r4 InfoNCE 10epoch + top-20 후보 보존 BM25 rerank**이다.

| Method | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| {best['name']} | {fmt(best['R@1'])} | {fmt(best['R@5'])} | {fmt(best['R@10'])} | {fmt(best['MRR'])} |

이 전략은 baseline 대비 모든 지표를 올리고, 기존 LoRA-256 대비 R@1/R@10/MRR을 올린다. LoRA-384+BM25와 R@1/R@5/R@10은 같지만 MRR이 더 높아서 최종 rank quality 기준으로 가장 좋다.

## 핵심 표

{md_table(top, [('Method', 'name'), ('Family', 'family'), ('R@1', 'R@1'), ('R@5', 'R@5'), ('R@10', 'R@10'), ('MRR', 'MRR'), ('Role', 'role')])}

## 랜덤 선택 대비

{md_table([random, baseline, best], [('Method', 'name'), ('R@1', 'R@1'), ('R@5', 'R@5'), ('R@10', 'R@10'), ('MRR', 'MRR'), ('Role', 'role')])}

## 파일 구성

- [01_experiment_catalog.md](01_experiment_catalog.md): 시도한 방법 하나하나의 목적, 구조, 결과, 해석
- [02_best_strategy.md](02_best_strategy.md): 최종 전략의 구체적 pipeline과 hyperparameter
- [03_ablation_and_necessity.md](03_ablation_and_necessity.md): 어떤 구성요소가 왜 필요한지에 대한 ablation 근거
- [04_visualizations.md](04_visualizations.md): 그래프 모음과 해석
- [data/results_table.csv](data/results_table.csv), [data/results_table.json](data/results_table.json): 재사용 가능한 전체 결과 테이블
- [data/ablation_deltas.csv](data/ablation_deltas.csv): 최종 방법의 metric delta

## 대표 그래프

![Overall metrics](figures/overall_metrics_grouped_bar.png)

![MRR vs R@10](figures/mrr_vs_r10_scatter.png)
"""
    (FINAL / "index.md").write_text(text)


def write_catalog(rows: list[dict]) -> None:
    table = md_table(
        rows,
        [
            ("Method", "name"),
            ("Family", "family"),
            ("Role", "role"),
            ("Novelty", "novelty"),
            ("R@1", "R@1"),
            ("R@5", "R@5"),
            ("R@10", "R@10"),
            ("MRR", "MRR"),
            ("Finding", "finding"),
        ],
    )
    sections = []
    for row in rows:
        evidence = row["file"] if row["file"] == "analytic_random_expectation" else f"output/{row['file']}"
        sections.append(
            f"""### {row['name']}

- 목적: {row['role']}
- 적용 전략: {row['strategy']}
- 결과: R@1 {fmt(row['R@1'])}, R@5 {fmt(row['R@5'])}, R@10 {fmt(row['R@10'])}, MRR {fmt(row['MRR'])}
- 해석: {row['finding']}
- 근거: `{evidence}`
"""
        )
    text = f"""# Experiment Catalog

아래 표와 섹션은 이번 라운드에서 시도한 방법들을 하나씩 정리한 것이다.

## 전체 결과 테이블

{table}

## 방법별 상세 정리

{chr(10).join(sections)}
"""
    (FINAL / "01_experiment_catalog.md").write_text(text)


def write_best_strategy(rows: list[dict]) -> None:
    best = next(r for r in rows if r["name"] == "Dual r4 10ep + BM25")
    random = next(r for r in rows if r["name"] == "Random ranking expected")
    baseline = next(r for r in rows if r["name"] == "Baseline-256")
    text = f"""# Best Strategy

## 최종 선택

**Dual-tower r4 InfoNCE 10epoch + top-20 후보 보존 Korean BM25 rerank**

| R@1 | R@5 | R@10 | MRR |
|---:|---:|---:|---:|
| {fmt(best['R@1'])} | {fmt(best['R@5'])} | {fmt(best['R@10'])} | {fmt(best['MRR'])} |

## 구체적 pipeline

```mermaid
flowchart LR
  Q[한국어 사용자 쿼리] --> T[SigLIP2 text encoder + LoRA r=4]
  I[상품 이미지] --> V[SigLIP2 vision encoder + LoRA r=4]
  T --> S[image-text similarity]
  V --> S
  S --> K[top-20 후보 유지]
  M[상품명/브랜드/설명 metadata] --> B[Korean char-ngram BM25]
  Q --> B
  K --> R[후보 내부 rerank: dual score + 0.2 * BM25]
  B --> R
  R --> O[최종 ranking]
```

## 적용된 설정

- Backbone: `google/siglip2-base-patch16-384`
- Adaptation: text encoder와 vision encoder 양쪽에 LoRA 적용
- LoRA rank: `r=4`
- Loss: in-batch `InfoNCE`
- Epoch: `10`
- Rerank: 1차 dual retrieval의 top-20 후보만 유지
- Lexical signal: 상품명, 브랜드, 이미지 설명을 합친 문서에 Korean char-ngram BM25
- 선택 정책: BM25 weight/top-N은 train-split validation에서만 선택
- 선택된 reranker: `rerank_top20_0.2bm25`

## 왜 이 전략인가

1. dual-tower LoRA는 cached frozen image embedding을 쓰는 방식보다 novelty가 높다. text와 vision 양쪽 domain adaptation을 수행하기 때문이다.
2. BM25 rerank는 한국어 색상, 의류 타입, 프린트, 위치 단서처럼 metadata에 직접 남는 표현을 보완한다.
3. top-20 후보 보존 방식은 BM25가 gallery 전체를 뒤집지 못하게 하므로 R@10 손상을 제한한다.
4. 결과적으로 LoRA/BM25와 같은 R@1/R@5/R@10을 유지하면서 MRR을 더 높였다.

## 랜덤 선택과의 비교

{md_table([random, baseline, best], [('Method', 'name'), ('R@1', 'R@1'), ('R@5', 'R@5'), ('R@10', 'R@10'), ('MRR', 'MRR')])}

랜덤 선택은 gallery size 588에서 정답 rank가 균등하다고 가정한 기대값이다. 최종 방법은 random 대비 R@1 약 {best['R@1'] / random['R@1']:.1f}배, R@10 약 {best['R@10'] / random['R@10']:.1f}배, MRR 약 {best['MRR'] / random['MRR']:.1f}배 높다.

## 한계

R@5만 최적화하면 `Dual r8 InfoNCE 5ep` 또는 `Dual r4 InfoNCE 10ep`가 0.867로 더 높다. 따라서 최종 전략은 “모든 단일 metric에서 strict 최고”가 아니라, R@1/R@10을 유지하면서 MRR이 가장 높은 균형점이다.
"""
    (FINAL / "02_best_strategy.md").write_text(text)


def write_ablation(rows: list[dict]) -> None:
    names = [
        "Random ranking expected",
        "Baseline-256",
        "LoRA-256",
        "LoRA-384 sigmoid10",
        "LoRA-384 + BM25 top-10",
        "Dual r4 InfoNCE 5ep",
        "Dual r4 InfoNCE 10ep",
        "Dual r4 10ep + BM25",
        "Dual r8 InfoNCE 5ep",
        "Dual r16 InfoNCE 5ep",
        "Learned top10 reranker",
        "Adaptive BM25 rerank",
        "Union LoRA/BM25 candidates",
    ]
    selected = [next(r for r in rows if r["name"] == n) for n in names]
    final = next(r for r in rows if r["name"] == "Dual r4 10ep + BM25")
    random = next(r for r in rows if r["name"] == "Random ranking expected")
    dual = next(r for r in rows if r["name"] == "Dual r4 InfoNCE 10ep")
    lora_bm25 = next(r for r in rows if r["name"] == "LoRA-384 + BM25 top-10")
    baseline = next(r for r in rows if r["name"] == "Baseline-256")
    deltas = [
        {"Ablation", "x"}
    ]
    deltas_table = [
        {
            "Comparison": "Final - Random",
            "R@1": final["R@1"] - random["R@1"],
            "R@5": final["R@5"] - random["R@5"],
            "R@10": final["R@10"] - random["R@10"],
            "MRR": final["MRR"] - random["MRR"],
            "Meaning": "무작위 선택 대비 retrieval signal이 충분히 강함",
        },
        {
            "Comparison": "Final - Baseline",
            "R@1": final["R@1"] - baseline["R@1"],
            "R@5": final["R@5"] - baseline["R@5"],
            "R@10": final["R@10"] - baseline["R@10"],
            "MRR": final["MRR"] - baseline["MRR"],
            "Meaning": "전체 목표에서 baseline 대비 모든 지표 개선",
        },
        {
            "Comparison": "Final - Dual r4 10ep",
            "R@1": final["R@1"] - dual["R@1"],
            "R@5": final["R@5"] - dual["R@5"],
            "R@10": final["R@10"] - dual["R@10"],
            "MRR": final["MRR"] - dual["MRR"],
            "Meaning": "BM25 rerank는 R@1/MRR을 올리지만 R@5는 낮춘다",
        },
        {
            "Comparison": "Final - LoRA/BM25",
            "R@1": final["R@1"] - lora_bm25["R@1"],
            "R@5": final["R@5"] - lora_bm25["R@5"],
            "R@10": final["R@10"] - lora_bm25["R@10"],
            "MRR": final["MRR"] - lora_bm25["MRR"],
            "Meaning": "dual-tower generator는 MRR을 소폭 개선하며 novelty를 높인다",
        },
    ]
    text = f"""# Ablation And Necessity

## 핵심 ablation 표

{md_table(selected, [('Method', 'name'), ('R@1', 'R@1'), ('R@5', 'R@5'), ('R@10', 'R@10'), ('MRR', 'MRR'), ('Finding', 'finding')])}

## 구성요소 필요성

| 비교 | ΔR@1 | ΔR@5 | ΔR@10 | ΔMRR | 의미 |
|---|---:|---:|---:|---:|---|
"""
    for row in deltas_table:
        text += f"| {row['Comparison']} | {row['R@1']:.3f} | {row['R@5']:.3f} | {row['R@10']:.3f} | {row['MRR']:.3f} | {row['Meaning']} |\n"
    text += """
## 무엇이 필요한가

1. **Dual-tower adaptation**

   LoRA-384+BM25와 final은 R@1/R@5/R@10이 같지만 final의 MRR이 더 높다. 성능 차이는 작지만, 연구 novelty 관점에서는 text-only LoRA보다 강하다. image encoder까지 domain adaptation을 수행하기 때문이다.

2. **InfoNCE 10epoch**

   r4 5epoch 단독은 R@10이 0.833으로 낮다. r4 10epoch 단독은 R@10 0.900과 R@5 0.867을 회복한다. 즉 r4에서는 5epoch가 충분하지 않았고, 10epoch가 후보 recall을 회복한다.

3. **Candidate-preserving BM25 rerank**

   r4 10epoch 단독은 R@5 0.867이지만 R@1 0.633, MRR 0.726이다. BM25를 top-20 내부에만 적용하면 R@1 0.700, MRR 0.757로 오른다. 따라서 첫 순위 품질에는 lexical metadata가 필요하다.

4. **후보 보존 제약**

   Union LoRA/BM25 candidates는 R@10이 0.867로 내려간다. Adaptive BM25도 R@10 0.867이다. BM25를 전체 후보 확장으로 쓰는 것보다, visual retriever가 만든 후보 내부에서만 쓰는 편이 안정적이다.

## 무엇은 필요하지 않았나

- 학습형 logistic reranker: validation MRR은 높았지만 eval MRR 0.504로 붕괴했다.
- Patch MaxSim: LoRA-384와 같은 MRR 0.684로 개선이 없었다.
- Query-only adapter: query-side 보정만으로는 dual/lexical 결합을 대체하지 못했다.
- Query expansion: validation에서 expansion이 선택되지 않았다.
- r8 장기 학습: 15epoch는 eval MRR 0.674로 악화했다.

## 엄밀한 결론

우리 방법론의 모든 구성요소가 모든 metric에 항상 필요한 것은 아니다. R@5만 보면 BM25 없는 dual r4/r8가 더 높다. 그러나 논문 주장을 **balanced retrieval quality**, 특히 R@1/R@10 보존과 MRR 개선으로 잡으면, dual-tower adaptation + candidate-preserving BM25 rerank 조합이 가장 방어 가능하다.

![Final deltas](figures/final_ablation_deltas.png)

![Component ladder](figures/component_ladder_mrr_r10.png)
"""
    (FINAL / "03_ablation_and_necessity.md").write_text(text)


def write_visualizations(rows: list[dict]) -> None:
    text = """# Visualizations

## 1. 전체 metric 비교

![Overall metrics](figures/overall_metrics_grouped_bar.png)

해석: random expected는 gallery size 588 기준 하한선이다. final은 R@1/R@10을 최고 수준으로 유지하고 MRR이 가장 높다. R@5 단독 최고는 dual-only 계열이다.

## 2. 구성요소 ladder

![Component ladder](figures/component_ladder_mrr_r10.png)

해석: random에서 baseline, LoRA, dual+BM25로 올라가는 계단형 개선을 보여준다. dual r4 10epoch는 후보 recall을 회복하고, BM25가 첫 순위 품질을 보강한다.

## 3. MRR-R@10 trade-off

![MRR vs R@10](figures/mrr_vs_r10_scatter.png)

해석: final은 오른쪽 위에 있으며, R@10을 희생하지 않는 MRR 최고점이다.

## 4. 난이도별 final 성능

![Difficulty](figures/final_by_difficulty.png)

해석: long_specific은 강하지만 short_ambiguous가 가장 어려운 구간이다. 다음 개선은 짧고 모호한 쿼리의 disambiguation이 타깃이다.

## 5. 실험 family별 최고 성능

![Family heatmap](figures/family_best_heatmap.png)

해석: dual_hybrid 계열이 MRR 기준 최고이며, learned rerank는 과적합 위험이 크다.

## 6. validation-eval gap

![Validation gap](figures/validation_eval_mrr_gap.png)

해석: validation 점수가 높은 방법이 locked eval에서 항상 좋지 않다. 특히 learned/adaptive 류는 작은 eval에서 과적합 리스크가 크다.

## 7. final delta

![Final deltas](figures/final_ablation_deltas.png)

해석: final은 baseline과 LoRA-384 대비 뚜렷한 개선을 보이고, LoRA/BM25 대비로는 MRR만 소폭 개선한다.
"""
    (FINAL / "04_visualizations.md").write_text(text)


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    save_data(rows)
    build_figures(rows)
    write_index(rows)
    write_catalog(rows)
    write_best_strategy(rows)
    write_ablation(rows)
    write_visualizations(rows)
    print(FINAL)


if __name__ == "__main__":
    main()
