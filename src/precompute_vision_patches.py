"""Precompute SigLIP 2 vision patch tokens (per-patch features) for ColBERT-style MaxSim.

Saves last_hidden_state (post-layernorm, pre-attention-pool) so we have one
768-dim vector per spatial patch. Stored as float16 to fit ~2.6GB for 2,938 images.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

IMAGES_DIR = Path(CONFIG["paths"]["images_dir"])
EMB_PATH = Path(CONFIG["paths"]["vision_patches"])
IDX_PATH = Path(CONFIG["paths"]["vision_patches_idx"])
TOP_JSON = Path(CONFIG["paths"]["top_json"])

MODEL_ID = CONFIG["backbone"]["model_id"]
NUM_PATCHES = CONFIG["backbone"]["num_patches"]
BATCH = 32  # smaller because patch16-384 is larger


def main():
    from transformers import AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, model: {MODEL_ID}, patches/image: {NUM_PATCHES}")

    print("loading model...")
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    items = json.loads(TOP_JSON.read_text())
    import re
    def pid(link):
        m = re.search(r"/(?:goods|products)/(\d+)", link)
        return m.group(1) if m else None

    seen = set()
    pairs = []
    for it in items:
        p = pid(it["product_link"])
        if p is None or p in seen:
            continue
        path = IMAGES_DIR / f"{p}.jpg"
        if path.exists() and path.stat().st_size > 0:
            pairs.append((p, str(path)))
            seen.add(p)
    print(f"unique images to encode: {len(pairs)}")

    n = len(pairs)
    out = np.empty((n, NUM_PATCHES, 768), dtype=np.float16)
    product_ids = []
    write_row = 0

    with torch.inference_mode():
        for i in tqdm(range(0, len(pairs), BATCH), desc="encoding", ncols=80):
            chunk = pairs[i : i + BATCH]
            ids, paths = zip(*chunk)
            imgs = []
            kept_ids = []
            for pid_, p in chunk:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    kept_ids.append(pid_)
                except Exception:
                    continue
            if not imgs:
                continue
            inputs = processor(images=list(imgs), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
            outputs = model.vision_model(pixel_values=pixel_values)
            # last_hidden_state: (B, num_patches, 768) post-layernorm, pre-attention-pool
            patches = outputs.last_hidden_state  # already post post_layernorm
            # L2-normalize per patch token for cosine MaxSim
            patches = patches / patches.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            patches_np = patches.float().cpu().numpy().astype(np.float16)
            assert patches_np.shape[1] == NUM_PATCHES, f"expected {NUM_PATCHES} patches, got {patches_np.shape[1]}"
            out[write_row : write_row + patches_np.shape[0]] = patches_np
            product_ids.extend(kept_ids)
            write_row += patches_np.shape[0]

    out = out[:write_row]
    print(f"final shape: {out.shape}")

    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, out)
    IDX_PATH.write_text(json.dumps(product_ids, ensure_ascii=False))
    print(f"saved {EMB_PATH} ({out.nbytes / 1e9:.2f} GB) and {IDX_PATH}")


if __name__ == "__main__":
    main()
