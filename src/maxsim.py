"""ColBERT-style late-interaction (Sum-of-MaxSim) score utilities.

Given query token vectors (B_q, T, D) with a padding mask, and image patch
token vectors (B_g, P, D), compute mean-normalized MaxSim score:

    s(q, g) = (1 / |valid_q|) * sum_{t in valid_q} max_{p in P} <q_t, g_p>

All token vectors are expected to be L2-normalized so the inner product is
cosine similarity.
"""
import torch


def maxsim_score(
    text_vecs: torch.Tensor,         # (B_q, T, D), L2-normalized along D
    text_mask: torch.Tensor,         # (B_q, T) bool, True for valid (non-padding)
    patch_vecs: torch.Tensor,        # (B_g, P, D), L2-normalized along D
) -> torch.Tensor:
    """Returns (B_q, B_g) mean-normalized MaxSim scores."""
    B_q, T, D = text_vecs.shape
    B_g, P, _ = patch_vecs.shape

    # (B_q, T, D) @ (D, B_g * P) -> (B_q, T, B_g, P)
    sim = torch.einsum("qtd,gpd->qtgp", text_vecs, patch_vecs)
    # max over patch dim P -> (B_q, T, B_g)
    sim_max = sim.amax(dim=-1)
    # mask invalid query tokens (padding) before sum
    mask = text_mask.unsqueeze(-1).to(sim_max.dtype)  # (B_q, T, 1)
    masked = sim_max * mask
    valid_count = mask.sum(dim=1).clamp(min=1)  # (B_q, 1)
    score = masked.sum(dim=1) / valid_count  # (B_q, B_g)
    return score


def maxsim_score_chunked(
    text_vecs: torch.Tensor,
    text_mask: torch.Tensor,
    patch_vecs: torch.Tensor,
    chunk_g: int = 64,
) -> torch.Tensor:
    """Chunked over gallery dim to avoid OOM for large galleries."""
    chunks = []
    for s in range(0, patch_vecs.shape[0], chunk_g):
        e = min(s + chunk_g, patch_vecs.shape[0])
        chunks.append(maxsim_score(text_vecs, text_mask, patch_vecs[s:e]))
    return torch.cat(chunks, dim=1)
