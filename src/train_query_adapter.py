"""Train query-side residual adapters on frozen SigLIP/LoRA embeddings.

The adapter is a small retrieval head on top of the LoRA text embedding:

    q' = normalize(q + W2 GELU(W1 LayerNorm(q)))

It is trained only on train-split augmented pairs and selected by an internal
train-product validation split. The locked hand-written eval set is used only
after model selection.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
TRAIN_PAIRS = Path(CONFIG["paths"]["train_pairs_final"])
EVAL_QUERIES = Path(CONFIG["paths"]["eval_queries"])
VISION_EMB = Path(CONFIG["paths"]["vision_emb"])
VISION_IDX = Path(CONFIG["paths"]["vision_idx"])
OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"])
CKPT_DIR = Path(CONFIG["paths"]["checkpoints_dir"])

MODEL_ID = CONFIG["backbone"]["model_id"]
MAX_LEN = CONFIG["backbone"]["text_max_length"]
SEED = CONFIG["project"]["seed"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_np(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def metrics_at_k(scores: np.ndarray, gt_indices: np.ndarray, k_list=(1, 5, 10)):
    ranks = []
    for s, gt in zip(scores, gt_indices):
        order = np.argsort(-s)
        rank = int(np.where(order == gt)[0][0]) + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    return {
        **{f"R@{k}": float((ranks <= k).mean()) for k in k_list},
        "MRR": float((1.0 / ranks).mean()),
    }, ranks.tolist()


def encode_text_cls(model, tokenizer, queries, device, batch_size=96):
    embs = []
    for i in range(0, len(queries), batch_size):
        toks = tokenizer(
            queries[i : i + batch_size],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            out = model.get_text_features(input_ids=toks["input_ids"])
        out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        embs.append(out.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def load_lora_model(lora_path: str, device: str):
    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(model, lora_path).eval().to(device)
    return model, processor.tokenizer


class PairEmbeddingDataset(Dataset):
    def __init__(self, q_emb: np.ndarray, img_emb: np.ndarray):
        self.q_emb = torch.from_numpy(q_emb).float()
        self.img_emb = torch.from_numpy(img_emb).float()

    def __len__(self):
        return self.q_emb.shape[0]

    def __getitem__(self, idx):
        return self.q_emb[idx], self.img_emb[idx]


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return F.normalize(x + self.net(x), dim=-1)


class LinearAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


def infonce_loss(q, img, temp):
    logits = q @ img.T / temp
    targets = torch.arange(q.shape[0], device=q.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def sigmoid_loss(q, img, scale=10.0, bias=-10.0):
    logits = scale * (q @ img.T) + bias
    labels = 2 * torch.eye(q.shape[0], device=q.device, dtype=q.dtype) - 1
    return -F.logsigmoid(labels * logits).sum() / q.shape[0]


def eval_adapter(adapter, q_emb, gallery_emb, gt_idx, device):
    adapter.eval()
    with torch.inference_mode():
        q = torch.from_numpy(q_emb).float().to(device)
        q = adapter(q).cpu().numpy()
    scores = q @ gallery_emb.T
    return metrics_at_k(scores, gt_idx), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default="checkpoints/lora_cls_384")
    parser.add_argument("--tag", default="query_adapter_lora384")
    parser.add_argument("--adapter", default="residual", choices=["residual", "linear"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--loss", default="infonce", choices=["infonce", "sigmoid"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-products", type=int, default=240)
    args = parser.parse_args()

    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_DIR / args.tag).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_split = json.loads(SPLIT_TRAIN.read_text())
    test_split = json.loads(SPLIT_TEST.read_text())
    train_pids = {it["product_id"] for it in train_split}
    test_pids = {it["product_id"] for it in test_split}
    train_pids = sorted(train_pids - test_pids)

    rng = random.Random(SEED)
    val_pids = set(rng.sample(train_pids, min(args.val_products, len(train_pids))))

    vision_emb_all = normalize_np(np.load(VISION_EMB).astype("float32"))
    vision_idx = json.loads(VISION_IDX.read_text())
    pid_to_row = {pid: i for i, pid in enumerate(vision_idx)}

    pairs = []
    with TRAIN_PAIRS.open() as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["product_id"]
            if rec.get("query") and pid in pid_to_row and pid in train_pids:
                pairs.append(rec)

    train_pairs = [p for p in pairs if p["product_id"] not in val_pids]
    val_pairs = [p for p in pairs if p["product_id"] in val_pids]
    eval_queries = [json.loads(l) for l in EVAL_QUERIES.read_text().splitlines() if l.strip()]

    model, tokenizer = load_lora_model(args.lora, device)
    train_q = encode_text_cls(model, tokenizer, [p["query"] for p in train_pairs], device)
    val_q = encode_text_cls(model, tokenizer, [p["query"] for p in val_pairs], device)
    eval_q = encode_text_cls(model, tokenizer, [q["query"] for q in eval_queries], device)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    train_img = vision_emb_all[[pid_to_row[p["product_id"]] for p in train_pairs]]
    val_products = sorted({p["product_id"] for p in val_pairs})
    val_gallery = vision_emb_all[[pid_to_row[pid] for pid in val_products]]
    val_pid_to_idx = {pid: i for i, pid in enumerate(val_products)}
    val_gt = np.array([val_pid_to_idx[p["product_id"]] for p in val_pairs], dtype=np.int64)

    test_products = [it["product_id"] for it in test_split if it["product_id"] in pid_to_row]
    seen = set()
    test_products = [pid for pid in test_products if not (pid in seen or seen.add(pid))]
    eval_gallery = vision_emb_all[[pid_to_row[pid] for pid in test_products]]
    eval_pid_to_idx = {pid: i for i, pid in enumerate(test_products)}
    eval_gt = np.array([eval_pid_to_idx[q["product_id"]] for q in eval_queries], dtype=np.int64)

    dim = train_q.shape[1]
    if args.adapter == "linear":
        adapter = LinearAdapter(dim)
    else:
        adapter = ResidualAdapter(dim, args.hidden, args.dropout)
    adapter.to(device)

    loader = DataLoader(
        PairEmbeddingDataset(train_q, train_img),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    optim = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)

    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        losses = []
        for q, img in loader:
            q = q.to(device)
            img = F.normalize(img.to(device), dim=-1)
            pred = adapter(q)
            loss = infonce_loss(pred, img, temp=0.05) if args.loss == "infonce" else sigmoid_loss(pred, img)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.detach().cpu()))
        (val_metrics, val_ranks), _ = eval_adapter(adapter, val_q, val_gallery, val_gt, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_metrics": val_metrics}
        history.append(row)
        key = (val_metrics["MRR"], val_metrics["R@1"], val_metrics["R@5"], val_metrics["R@10"])
        if best is None or key > best["key"]:
            best = {"key": key, "epoch": epoch, "state": {k: v.detach().cpu() for k, v in adapter.state_dict().items()}, "val_metrics": val_metrics}

    adapter.load_state_dict(best["state"])
    (eval_metrics, eval_ranks), eval_scores = eval_adapter(adapter, eval_q, eval_gallery, eval_gt, device)
    base_eval_metrics, _ = metrics_at_k(eval_q @ eval_gallery.T, eval_gt)

    ckpt_path = CKPT_DIR / args.tag / "adapter.pt"
    torch.save({"state_dict": best["state"], "args": vars(args), "best_epoch": best["epoch"]}, ckpt_path)

    summary = {
        "tag": args.tag,
        "method": "query-side residual adapter on frozen LoRA text embeddings",
        "selection_policy": "best epoch selected on train-split validation products only",
        "lora": args.lora,
        "backbone": MODEL_ID,
        "adapter": args.adapter,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "loss": args.loss,
        "epochs": args.epochs,
        "best_epoch": best["epoch"],
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "val_products": len(val_products),
        "eval_queries": len(eval_queries),
        "eval_gallery_size": len(test_products),
        "base_lora384_eval_metrics_recomputed": base_eval_metrics,
        "best_val_metrics": best["val_metrics"],
        "eval_metrics": eval_metrics,
        "history": history,
    }
    out_path = OUTPUT_DIR / f"eval_{args.tag}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    order = np.argsort(-eval_scores, axis=1)
    details = []
    for qi, q in enumerate(eval_queries):
        details.append(
            {
                **q,
                "rank": eval_ranks[qi],
                "top10_pids": [test_products[i] for i in order[qi, :10].tolist()],
            }
        )
    (OUTPUT_DIR / f"eval_{args.tag}_detail.json").write_text(json.dumps(details, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ["tag", "best_epoch", "best_val_metrics", "eval_metrics", "base_lora384_eval_metrics_recomputed"]}, ensure_ascii=False, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
