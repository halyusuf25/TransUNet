import torch
import torch.nn.functional as F
from typing import Tuple, Union

def topk_indices(x: torch.Tensor, attention_scores: torch.Tensor, keep_rate: float):
    """
    Select top-k token indices per batch based on keep_rate and attention_scores.

    Args:
        x: Tensor of shape [B, N, C] input features
        keep_rate: Fraction of tokens to keep (0 < keep_rate <= 1)
        attention_scores: Tensor of shape [B, H, N_q, N_k] = [B, H, N, N]

    Returns:
        idx: Tensor [B, K] of top-k indices where K = keep_rate * N
        index: Tensor [B, K, C] ready for torch.gather on dim=1 of x
    """
    B , N, C = x.shape
    attention_scores = attention_scores.mean(dim=1).mean(dim=1)  # [B, N]
    k = max(1, int(keep_rate * N))
    _ , idx = torch.topk(attention_scores, k, dim=1, largest=True, sorted=True)  # [B, K]
    index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, K, C]
    return idx, index


def compute_uncertainty(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalized entropy-based uncertainty map in [0, 1].

    Args:
        logits: Tensor of shape [B, C, H, W]
        eps: Small constant for numerical stability

    Returns:
        Tensor of shape [B, 1, H, W] with normalized entropy per pixel.
    """
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=1, keepdim=True)
    log_c = torch.log(torch.tensor(probs.shape[1], device=logits.device, dtype=logits.dtype))
    return (entropy / log_c.clamp_min(eps)).clamp_(0.0, 1.0)

import math
import torch
import torch.nn.functional as F

def compute_pixel_uncertainty(
    logits: torch.Tensor,
    dim: int = 1,
    normalize: bool = True,
    clamp: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute an entropy-based per-pixel uncertainty map from logits.

    Args:
        logits:   Tensor of shape [B, C, H, W] (or any shape where `dim` is the class dimension).
        dim:      Dimension over which to interpret logits as class scores (default: 1).
        normalize:
            If True, entropy is divided by log(C) so the result is in ~[0, 1],
            where C is the number of classes (size at `dim`).
        clamp:
            If True, clamp the output to [0, 1] when normalized,
            or to [0, +inf) when not normalized.
        eps:
            Small constant for numerical safety (mainly used when normalizing).

    Returns:
        Tensor of shape like `logits` with `dim` reduced and kept:
        shape will be `logits.shape` with size 1 at `dim`
        (e.g. [B, 1, H, W] for typical segmentation).
    """
    # 1) Stable log-softmax over class dimension (works directly on logits)
    log_probs = F.log_softmax(logits, dim=dim)    # log p(c|x)
    probs = log_probs.exp()                       # p(c|x)

    # 2) Entropy H = -sum_c p * log p
    entropy = -(probs * log_probs).sum(dim=dim, keepdim=True)

    # 3) Optional normalization by log(C) to get ~[0, 1]
    if normalize:
        num_classes = logits.size(dim)
        # log(C) as a tensor on the same device and dtype
        log_c = math.log(max(num_classes, 1))
        # avoid division by zero just in case
        log_c = max(log_c, eps)
        entropy = entropy / log_c

    # 4) Optional clamping to a clean range
    if clamp:
        if normalize:
            entropy = entropy.clamp_(0.0, 1.0)
        else:
            entropy = entropy.clamp_(min=0.0)

    return entropy


def compute_boundary_mask(labels: torch.Tensor, kernel_size: Union[int, Tuple[int, int]] = 3) -> torch.Tensor:
    """
    Binary boundary mask (1 on boundary pixels) derived from integer labels.

    Args:
        labels: Tensor of shape [B, H, W] with integer class ids
        kernel_size: Neighborhood size as int or (kh, kw); default 2 (2x2 neighborhood)

    Returns:
        Tensor of shape [B, 1, H, W] with boundary mask.
    """
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size

    lbl = labels.unsqueeze(1).float()  # [B, 1, H, W]

    pad_h = kh // 2
    pad_w = kw // 2
    lbl_padded = F.pad(lbl, (pad_w, pad_w, pad_h, pad_h), mode="replicate")

    max_pool = F.max_pool2d(lbl_padded, kernel_size=(kh, kw), stride=1)
    min_pool = -F.max_pool2d(-lbl_padded, kernel_size=(kh, kw), stride=1)

    # Crop back to original spatial size (needed for even kernel sizes)
    H, W = labels.shape[-2:]
    start_h = (max_pool.shape[-2] - H) // 2
    start_w = (max_pool.shape[-1] - W) // 2
    max_pool = max_pool[..., start_h:start_h + H, start_w:start_w + W]
    min_pool = min_pool[..., start_h:start_h + H, start_w:start_w + W]

    boundary = (max_pool != min_pool).float()
    return boundary
