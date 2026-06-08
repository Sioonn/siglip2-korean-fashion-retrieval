"""Train dual-tower LoRA on text and vision encoders.

Unlike src/train_lora.py, this does not use cached frozen image embeddings.
Images are forwarded through the PEFT-wrapped vision tower, allowing LoRA
updates on both query text and product images.
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

SPLIT_TRAIN = Path(CONFIG["paths"]["split_train"])
SPLIT_TEST = Path(CONFIG["paths"]["split_test"])
TRAIN_PAIRS = Path(CONFIG["paths"]["train_pairs_final"])
IMAGES_DIR = Path(CONFIG["paths"]["images_dir"])
CKPT_DIR = Path(CONFIG["paths"]["checkpoints_dir"])
LOG_DIR = Path(CONFIG["paths"]["logs_dir"])

MODEL_ID = CONFIG["backbone"]["model_id"]
MAX_LEN = CONFIG["backbone"]["text_max_length"]
SEED = CONFIG["project"]["seed"]
LORA_TARGETS = CONFIG["lora"]["target_modules"]
WARMUP_RATIO = CONFIG["train"]["warmup_ratio"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImagePairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        image = Image.open(IMAGES_DIR / f"{rec['product_id']}.jpg").convert("RGB")
        return {"query": rec["query"], "image": image}


def collate_factory(processor):
    tokenizer = processor.tokenizer

    def collate(batch):
        toks = tokenizer([b["query"] for b in batch], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        imgs = processor(images=[b["image"] for b in batch], return_tensors="pt")
        return {"input_ids": toks["input_ids"], "pixel_values": imgs["pixel_values"]}

    return collate


def sigmoid_loss(scores, logit_scale, logit_bias):
    bsz = scores.shape[0]
    logits = logit_scale * scores + logit_bias
    labels = 2 * torch.eye(bsz, device=scores.device, dtype=scores.dtype) - 1
    return -F.logsigmoid(labels * logits).sum() / bsz


def infonce_loss(scores, logit_scale):
    bsz = scores.shape[0]
    logits = logit_scale * scores
    targets = torch.arange(bsz, device=scores.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dual_lora_cls")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--loss", default="sigmoid", choices=["sigmoid", "infonce"])
    parser.add_argument("--r", type=int, default=8)
    args = parser.parse_args()

    set_seed(SEED)
    from transformers import AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_pids = {p["product_id"] for p in json.loads(SPLIT_TRAIN.read_text())}
    test_pids = {p["product_id"] for p in json.loads(SPLIT_TEST.read_text())}
    pairs = []
    for line in TRAIN_PAIRS.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        path = IMAGES_DIR / f"{rec['product_id']}.jpg"
        if rec.get("query") and rec["product_id"] in train_pids and rec["product_id"] not in test_pids and path.exists():
            pairs.append(rec)
    print(f"train pairs: {len(pairs)}")

    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    text_targets = [
        f"text_model.encoder.layers.{i}.self_attn.{m}"
        for i in range(model.config.text_config.num_hidden_layers)
        for m in LORA_TARGETS
    ]
    vision_targets = [
        f"vision_model.encoder.layers.{i}.self_attn.{m}"
        for i in range(model.config.vision_config.num_hidden_layers)
        for m in LORA_TARGETS
    ]
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.r * 2,
        lora_dropout=0.05,
        target_modules=text_targets + vision_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)
    base = model.base_model.model
    if hasattr(base, "logit_scale"):
        base.logit_scale.requires_grad = True
    if hasattr(base, "logit_bias"):
        base.logit_bias.requires_grad = args.loss == "sigmoid"

    ds = ImagePairDataset(pairs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_factory(processor), drop_last=True)
    total_steps = max(1, args.epochs * len(loader))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-3)

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    losses = []
    step = 0
    model.train()
    for epoch in range(args.epochs):
        ep_losses = []
        pbar = tqdm(loader, ncols=80, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            for group in optim.param_groups:
                group["lr"] = args.lr * lr_at(step)
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            text = model.get_text_features(input_ids=input_ids)
            image = model.get_image_features(pixel_values=pixel_values)
            text = text / text.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            image = image / image.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            scores = text @ image.T
            logit_scale = base.logit_scale.exp() if hasattr(base, "logit_scale") else torch.tensor(1.0, device=device)
            logit_bias = base.logit_bias if hasattr(base, "logit_bias") else torch.tensor(0.0, device=device)
            loss = sigmoid_loss(scores, logit_scale, logit_bias) if args.loss == "sigmoid" else infonce_loss(scores, logit_scale)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            step += 1
            ep_losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{ep_losses[-1]:.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}")
        losses.append(sum(ep_losses) / max(1, len(ep_losses)))
        print(f"[epoch {epoch + 1}] mean loss = {losses[-1]:.4f}")

    out_dir = CKPT_DIR / args.out
    model.save_pretrained(out_dir)
    (out_dir / "extras.json").write_text(json.dumps({"epoch_losses": losses, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "loss": args.loss, "r": args.r}, indent=2))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
