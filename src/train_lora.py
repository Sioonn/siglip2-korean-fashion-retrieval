"""Train text-only LoRA on top of frozen SigLIP 2 with cached vision features.

Supports two pooling modes:
  - cls: dot product between CLS-pooled text and image embeddings (legacy)
  - maxsim: ColBERT-style late interaction over per-token text and per-patch image vectors

And two losses:
  - sigmoid: SigLIP-style pairwise sigmoid loss
  - infonce: ColBERT-style softmax cross-entropy with in-batch negatives
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

from src.maxsim import maxsim_score  # noqa: E402

SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
TRAIN_PAIRS = Path(CONFIG["paths"]["train_pairs_final"])
VISION_EMB = Path(CONFIG["paths"]["vision_emb"])
VISION_IDX = Path(CONFIG["paths"]["vision_idx"])
VISION_PATCHES = Path(CONFIG["paths"]["vision_patches"])
VISION_PATCHES_IDX = Path(CONFIG["paths"]["vision_patches_idx"])
CKPT_DIR = Path(CONFIG["paths"]["checkpoints_dir"])
LOG_DIR = Path(CONFIG["paths"]["logs_dir"])

MODEL_ID = CONFIG["backbone"]["model_id"]
MAX_LEN = CONFIG["backbone"]["text_max_length"]
SEED = CONFIG["project"]["seed"]

LORA_R = CONFIG["lora"]["r"]
LORA_ALPHA = CONFIG["lora"]["alpha"]
LORA_DROPOUT = CONFIG["lora"]["dropout"]
LORA_TARGETS = CONFIG["lora"]["target_modules"]

BATCH = CONFIG["train"]["batch_size"]
LR = CONFIG["train"]["lr"]
EPOCHS = CONFIG["train"]["epochs"]
WARMUP_RATIO = CONFIG["train"]["warmup_ratio"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CLSPairDataset(Dataset):
    def __init__(self, pairs, vision_emb, pid_to_row):
        self.pairs = pairs
        self.vision_emb = vision_emb
        self.pid_to_row = pid_to_row

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        row = self.pid_to_row[rec["product_id"]]
        return {
            "query": rec["query"],
            "image_emb": torch.from_numpy(self.vision_emb[row]).float(),
        }


class MaxSimPairDataset(Dataset):
    def __init__(self, pairs, patch_emb, pid_to_row):
        self.pairs = pairs
        self.patch_emb = patch_emb  # (N, P, D) np.float16
        self.pid_to_row = pid_to_row

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        row = self.pid_to_row[rec["product_id"]]
        return {
            "query": rec["query"],
            "patches": torch.from_numpy(np.array(self.patch_emb[row])).float(),  # (P, D)
        }


def cls_collate_factory(tokenizer):
    def collate(batch):
        toks = tokenizer([b["query"] for b in batch], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return {
            "input_ids": toks["input_ids"],
            "image_embs": torch.stack([b["image_emb"] for b in batch]),
        }
    return collate


def maxsim_collate_factory(tokenizer):
    pad_id = tokenizer.pad_token_id
    def collate(batch):
        toks = tokenizer([b["query"] for b in batch], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return {
            "input_ids": toks["input_ids"],
            "text_mask": (toks["input_ids"] != pad_id),
            "patches": torch.stack([b["patches"] for b in batch]),
        }
    return collate


def sigmoid_loss(scores, logit_scale, logit_bias):
    """SigLIP-style pairwise sigmoid loss. scores: (B, B), logits = logit_scale * scores + logit_bias."""
    B = scores.shape[0]
    logits = logit_scale * scores + logit_bias
    labels = 2 * torch.eye(B, device=scores.device, dtype=scores.dtype) - 1
    return -F.logsigmoid(labels * logits).sum() / B


def infonce_loss(scores, logit_scale):
    """Symmetric InfoNCE. scores: (B, B). Use logit_scale (no bias) like CLIP."""
    B = scores.shape[0]
    logits = logit_scale * scores
    targets = torch.arange(B, device=scores.device)
    loss_t2i = F.cross_entropy(logits, targets)
    loss_i2t = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_t2i + loss_i2t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", default="cls", choices=["cls", "maxsim"])
    parser.add_argument("--loss", default="sigmoid", choices=["sigmoid", "infonce"])
    parser.add_argument("--out", default="lora_final")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    set_seed(SEED)
    from transformers import AutoModel, AutoProcessor

    epochs = args.epochs if args.epochs is not None else EPOCHS
    batch = args.batch_size if args.batch_size is not None else BATCH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if args.pooling == "cls":
        vision_emb = np.load(VISION_EMB)  # (N, D)
        vision_idx = json.loads(VISION_IDX.read_text())
    else:
        vision_emb = np.load(VISION_PATCHES, mmap_mode="r")  # (N, P, D) f16
        vision_idx = json.loads(VISION_PATCHES_IDX.read_text())
    pid_to_row = {pid: i for i, pid in enumerate(vision_idx)}

    pairs = []
    with TRAIN_PAIRS.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["product_id"] in pid_to_row and rec.get("query"):
                pairs.append(rec)
    print(f"loaded {len(pairs)} train pairs (after vision filter)")

    train_split = json.loads(SPLIT_TRAIN.read_text())
    train_pids = {it["product_id"] for it in train_split}
    pairs = [p for p in pairs if p["product_id"] in train_pids]
    print(f"after train-split filter: {len(pairs)} pairs")

    print("loading SigLIP 2...")
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    for p in model.vision_model.parameters():
        p.requires_grad = False

    text_lora_targets = [
        f"text_model.encoder.layers.{i}.self_attn.{m}"
        for i in range(model.config.text_config.num_hidden_layers)
        for m in LORA_TARGETS
    ]
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=text_lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    base = model.base_model.model
    # for maxsim: re-init logit_scale/bias since SigLIP's pretrained values
    # (scale≈4.72, bias≈-16.75) were calibrated for CLS-CLS dot product range
    # [-1,1], not for mean-MaxSim over per-token vectors which has a much
    # smaller dynamic range.
    if args.pooling == "maxsim":
        with torch.no_grad():
            if hasattr(base, "logit_scale"):
                base.logit_scale.fill_(math.log(10.0))  # scale = 10
            if hasattr(base, "logit_bias"):
                base.logit_bias.fill_(-2.0)
    if hasattr(base, "logit_scale"):
        base.logit_scale.requires_grad = True
    if hasattr(base, "logit_bias"):
        base.logit_bias.requires_grad = (args.loss == "sigmoid")

    if args.pooling == "cls":
        ds = CLSPairDataset(pairs, vision_emb, pid_to_row)
        collate = cls_collate_factory(tokenizer)
    else:
        ds = MaxSimPairDataset(pairs, vision_emb, pid_to_row)
        collate = maxsim_collate_factory(tokenizer)

    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2, collate_fn=collate, drop_last=True)

    total_steps = max(1, epochs * len(loader))
    warmup_steps = int(WARMUP_RATIO * total_steps)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=LR)

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    pad_id = tokenizer.pad_token_id
    losses = []
    step = 0
    model.train()
    for epoch in range(epochs):
        ep_losses = []
        pbar = tqdm(loader, ncols=80, desc=f"epoch {epoch+1}/{epochs}")
        for batch_dict in pbar:
            for g in optim.param_groups:
                g["lr"] = LR * lr_at(step)

            input_ids = batch_dict["input_ids"].to(device)
            logit_scale = base.logit_scale.exp() if hasattr(base, "logit_scale") else torch.tensor(1.0, device=device)
            logit_bias = base.logit_bias if hasattr(base, "logit_bias") else torch.tensor(0.0, device=device)

            if args.pooling == "cls":
                img_embs = batch_dict["image_embs"].to(device, dtype=torch.bfloat16)
                img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
                text_out = model.get_text_features(input_ids=input_ids)
                text_embs = text_out / text_out.norm(dim=-1, keepdim=True)
                scores = text_embs @ img_embs.T
            else:
                # maxsim path
                patches = batch_dict["patches"].to(device, dtype=torch.bfloat16)  # (B, P, D), pre-normalized
                # re-normalize defensively
                patches = patches / patches.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                text_mask = batch_dict["text_mask"].to(device)
                # forward through PEFT-wrapped base text model to get last_hidden_state
                base_text = base.text_model
                text_out = base_text(input_ids=input_ids)
                text_h = text_out.last_hidden_state  # (B, T, D)
                text_h = text_h / text_h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                scores = maxsim_score(text_h, text_mask, patches)  # (B, B)

            if args.loss == "sigmoid":
                loss = sigmoid_loss(scores, logit_scale, logit_bias)
            else:
                loss = infonce_loss(scores, logit_scale)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()

            step += 1
            ep_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}")
        losses.append(sum(ep_losses) / max(1, len(ep_losses)))
        print(f"[epoch {epoch+1}] mean loss = {losses[-1]:.4f}")

    out_dir = CKPT_DIR / args.out
    model.save_pretrained(out_dir)
    print(f"saved LoRA checkpoint to {out_dir}")

    extras = {
        "logit_scale": base.logit_scale.detach().float().cpu().item() if hasattr(base, "logit_scale") else None,
        "logit_bias": base.logit_bias.detach().float().cpu().item() if hasattr(base, "logit_bias") else None,
        "epoch_losses": losses,
        "total_steps": step,
        "pooling": args.pooling,
        "loss": args.loss,
        "epochs": epochs,
        "batch_size": batch,
        "backbone": MODEL_ID,
    }
    (out_dir / "extras.json").write_text(json.dumps(extras, ensure_ascii=False, indent=2))
    print("done.")


if __name__ == "__main__":
    main()
