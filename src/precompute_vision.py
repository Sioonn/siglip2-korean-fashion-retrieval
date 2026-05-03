"""Precompute SigLIP 2 vision embeddings for all downloaded product images.

Loads vision tower once, batches forward passes, saves to npy + idx json.
This runs once (vision tower is frozen for all training/eval).
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
EMB_PATH = Path(CONFIG["paths"]["vision_emb"])
IDX_PATH = Path(CONFIG["paths"]["vision_idx"])
TOP_JSON = Path(CONFIG["paths"]["top_json"])

MODEL_ID = CONFIG["backbone"]["model_id"]
BATCH = 64


def main():
    from transformers import AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("loading model...")
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # collect all images that exist
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

    embeddings = []
    product_ids = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(pairs), BATCH), desc="encoding", ncols=80):
            chunk = pairs[i : i + BATCH]
            ids, paths = zip(*chunk)
            try:
                imgs = [Image.open(p).convert("RGB") for p in paths]
            except Exception as e:
                # skip the entire chunk on read error and recover one-by-one
                imgs = []
                kept_ids = []
                kept_paths = []
                for pid_, p in chunk:
                    try:
                        imgs.append(Image.open(p).convert("RGB"))
                        kept_ids.append(pid_)
                        kept_paths.append(p)
                    except Exception:
                        continue
                if not imgs:
                    continue
                ids = kept_ids
                paths = kept_paths
            inputs = processor(images=list(imgs), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
            outputs = model.get_image_features(pixel_values=pixel_values)
            # SigLIP 2 returns image_embeds; normalize for cosine similarity
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            embeddings.append(outputs.float().cpu().numpy())
            product_ids.extend(ids)

    emb = np.concatenate(embeddings, axis=0)
    print(f"embedding shape: {emb.shape}")

    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb)
    IDX_PATH.write_text(json.dumps(product_ids, ensure_ascii=False))
    print(f"saved emb to {EMB_PATH} and idx to {IDX_PATH}")


if __name__ == "__main__":
    main()
