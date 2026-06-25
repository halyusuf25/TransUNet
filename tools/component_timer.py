# component_timer.py
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, Optional, List

import torch
from torch import nn

# ----------------------------- helpers ---------------------------------

def _extract_images(batch):
    # Accept (images, labels), dicts with 'image'/'img', or raw tensor.
    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
        return batch[0]
    if isinstance(batch, dict):
        for k in ("image", "img", "images", "inputs", "x"):
            if k in batch:
                return batch[k]
    if torch.is_tensor(batch):
        return batch
    raise ValueError("Cannot extract images from batch")

def _ms_cpu(fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, out

def _ms_gpu(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()  # ensure accurate device time
    return float(start.elapsed_time(end)), out  # milliseconds

def _measure_ms(device_is_cuda: bool, fn):
    return _ms_gpu(fn) if device_is_cuda else _ms_cpu(fn)

def _percentiles(vals: List[float], ps=(50, 90, 95, 99)) -> Dict[str, float]:
    import numpy as np
    a = np.array(vals, dtype=np.float64)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}

# ----------------------------- results ---------------------------------

@dataclass
class ComponentTimes:
    # mean over measured batches (ms)
    mean: Dict[str, float] = field(default_factory=dict)
    # optional latency distribution for total forward (ms/img, B=1 path if you like)
    dist: Dict[str, float] = field(default_factory=dict)
    # extra notes (device, amp, batch size, etc.)
    notes: Dict[str, Any] = field(default_factory=dict)

    def pretty(self) -> str:
        lines = ["== Per-component inference time (ms) =="]
        for k, v in self.mean.items():
            lines.append(f"{k:22s}: {v:8.3f}")
        if self.dist:
            lines.append("== Total forward latency percentiles (ms) ==")
            for k, v in self.dist.items():
                lines.append(f"{k:22s}: {v:8.3f}")
        return "\n".join(lines)

# ----------------------------- public API ---------------------------------

@torch.inference_mode()
def time_model_components(
    model: nn.Module,
    test_loader: Iterable,
    device: str = "cuda",
    warmup_batches: int = 5,
    measure_batches: int = 25,
    autocast: bool = False,
    amp_dtype: Optional[torch.dtype] = torch.float16,
) -> ComponentTimes:
    """
    Measures (per batch) the forward-pass times of:
      - dataload_ms: time to fetch next batch from DataLoader (CPU side)
      - h2d_ms: host->device copy of the input tensor
      - embeddings_ms: model.transformer.embeddings(x)
      - encoder_ms: model.transformer.encoder(embeds)
      - prune_scatter_ms: scatter tokens back when pruning occurs (else ~0)
      - decoder_ms: model.decoder(tokens, features)
      - head_ms: model.segmentation_head(x)
      - total_forward_ms: sum of the above 'on device' parts (excl. dataload_ms)

    Uses CUDA events + synchronize on GPU (accurate device time), and perf_counter on CPU.
    """
    model = model.to(device).eval()
    device_is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if (device_is_cuda and autocast) else torch.no_grad()

    # Containers for aggregated times (ms)
    keys = ["dataload_ms", "h2d_ms", "embeddings_ms", "encoder_ms",
            "prune_scatter_ms", "decoder_ms", "head_ms", "total_forward_ms"]
    acc = {k: 0.0 for k in keys}
    counts = 0
    total_forward_samples_ms: List[float] = []  # optional latency distribution

    it = iter(test_loader)

    # Warm-up (fetch + move + single full forward, not recorded)
    for _ in range(max(0, warmup_batches)):
        try:
            t0 = time.perf_counter()
            batch = next(it)
            _ = time.perf_counter() - t0  # discard
        except StopIteration:
            it = iter(test_loader)
            batch = next(it)
        imgs = _extract_images(batch)
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)  # ensure B,C,H,W
        imgs = imgs.to(device, non_blocking=device_is_cuda)
        with ctx:
            # replicate your model's forward path without timing (warm caches, select kernels)
            x = imgs
            if x.size(1) == 1:  # match VisionTransformer.forward
                x = x.repeat(1, 3, 1, 1)
            embeds, features = model.transformer.embeddings(x)
            encoded, *_ = model.transformer.encoder(embeds)
            # pruning handled/pruned indices not needed in warm-up
            _ = model.decoder(encoded, features)
            _ = model.segmentation_head(_)

    # Measure
    it = iter(test_loader)
    for _ in range(measure_batches):
        # ---- dataload on CPU ----
        t_dl0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(test_loader)
            batch = next(it)
        t_dl1 = time.perf_counter()
        dataload_ms = (t_dl1 - t_dl0) * 1000.0

        # ---- extract + H2D ----
        imgs = _extract_images(batch)
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)  # (1,C,H,W)
        # measure H2D alone
        if device_is_cuda:
            h2d_ms, x = _measure_ms(True, lambda: imgs.to(device, non_blocking=True))
        else:
            h2d_ms, x = _measure_ms(False, lambda: imgs.to(device))
        B = x.shape[0]

        # ---- component timings (device) ----
        with ctx:
            # VisionTransformer.forward first line (grayscale -> 3ch)
            if x.size(1) == 1:
                ms, x = _measure_ms(device_is_cuda, lambda: x.repeat(1, 3, 1, 1))
                # treat this as part of 'embeddings_ms' (cheap view/expand normally)

            # Embeddings (aka backbone)
            embeddings_ms, out = _measure_ms(device_is_cuda, lambda: model.transformer.embeddings(x))
            embeds, features = out  # embeds: (B, N, C)

            # Encoder
            encoder_ms, enc_out = _measure_ms(device_is_cuda, lambda: model.transformer.encoder(embeds))
            encoded, attn_weights, kept_indices = enc_out  # (B, K, C) if pruned

            # Prune / scatter back to original sequence if needed
            def _scatter_tokens():
                if kept_indices is None:
                    return encoded
                B2, K, C2 = encoded.size()
                total_n = embeds.size(1)
                full = encoded.new_zeros(B2, total_n, C2)
                scatter_index = kept_indices.long().unsqueeze(-1).expand(-1, -1, C2)  # (B, K, C)
                full.scatter_(1, scatter_index, encoded)
                return full

            prune_scatter_ms, tokens = _measure_ms(device_is_cuda, _scatter_tokens)

            # Decoder
            decoder_ms, dec = _measure_ms(device_is_cuda, lambda: model.decoder(tokens, features))

            # Segmentation head
            head_ms, logits = _measure_ms(device_is_cuda, lambda: model.segmentation_head(dec))

            total_forward_ms = embeddings_ms + encoder_ms + prune_scatter_ms + decoder_ms + head_ms

        # accumulate
        acc["dataload_ms"] += dataload_ms
        acc["h2d_ms"] += h2d_ms
        acc["embeddings_ms"] += embeddings_ms
        acc["encoder_ms"] += encoder_ms
        acc["prune_scatter_ms"] += prune_scatter_ms
        acc["decoder_ms"] += decoder_ms
        acc["head_ms"] += head_ms
        acc["total_forward_ms"] += total_forward_ms
        counts += 1
        total_forward_samples_ms.append(total_forward_ms / max(1, B))  # per-image latency sample

    # means
    means = {k: (v / max(1, counts)) for k, v in acc.items()}
    dist = _percentiles(total_forward_samples_ms) if total_forward_samples_ms else {}

    notes = {
        "device": device,
        "warmup_batches": warmup_batches,
        "measure_batches": measure_batches,
        "autocast": bool(autocast),
    }
    return ComponentTimes(mean=means, dist=dist, notes=notes)
