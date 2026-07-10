# benchmark.py
# ---------------------------------------------------------------------------
# Segmentation benchmarking utility (PyTorch) with custom FLOPs for your ViT:
# - Uses real samples from your test DataLoader (no dummy tensors)
# - Throughput (img/s), Latency (mean & p50/p90/p95/p99 in ms)
# - #Params, FLOPs (GFLOPs)  --- includes custom counting for Attention/SHSA
# - Logical quantized-model counting for AWQ WQLinear and custom W4GroupedLinear layers
# - Clean, modular, easy to extend
# ---------------------------------------------------------------------------

from __future__ import annotations
import time
import copy
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Iterable, List, NamedTuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

from fvcore.nn import FlopCountAnalysis
from thop import profile
from src.quantize import W4GroupedLinear

try:
    from awq.quantize.qmodule import WQLinear
except Exception:  # pragma: no cover - optional dependency for quantized benchmarks
    WQLinear = None  # type: ignore[assignment]

# ============================= Safe collate =============================

def _extract_img_only(sample):
    # Accept dicts {'image': ...}, tuples/lists (image, label, ...), or raw tensors/arrays.
    if isinstance(sample, dict) and "image" in sample:
        return sample["image"]
    if isinstance(sample, (list, tuple)) and len(sample) >= 1:
        return sample[0]
    return sample  # assume it's the image tensor/array
        

def collate_for_benchmark(batch, img_size: int):
    """
    Returns a single tensor of shape (B, C, H, W) with uniform H=W=img_size.
    - Drops labels/metadata (not needed for speed).
    - Converts NumPy -> torch, fixes negative strides via copy/contiguous.
    - Resizes to (img_size, img_size) on CPU to make stacking safe.
    """
    imgs = []
    for item in batch:
        x = _extract_img_only(item)

        # To torch tensor (and fix NumPy negative-stride / non-contiguous cases)
        if isinstance(x, np.ndarray):
            # ensure CxHxW contiguous
            if not x.flags["C_CONTIGUOUS"]:
                x = np.ascontiguousarray(x)
            x = torch.from_numpy(x)
        elif not torch.is_tensor(x):
            # Last resort (shouldn't happen for your datasets)
            x = torch.as_tensor(x)

        x = x.float()
        # Normalize to CHW; handle HWC and DHW (3D volume) cases
        if x.ndim == 2:
            # H, W -> 1, H, W
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            is_chw = x.shape[0] in (1, 3, 4)
            is_hwc = x.shape[-1] in (1, 3, 4)
            if is_hwc and not is_chw:
                # H, W, C -> C, H, W
                x = x.permute(2, 0, 1).contiguous()
            elif is_chw:
                # already C, H, W
                pass
            else:
                # Likely a 3D volume (D, H, W) -> take center slice -> 1, H, W
                mid = x.shape[0] // 2
                x = x[mid].unsqueeze(0)
        else:
            raise ValueError(f"Unexpected image shape: {tuple(x.shape)}")
        
        # OPTIONAL: force 3-channel input (expand grayscale to 3 "RGB-like" channels)
        if x.shape[0] == 1:
            x = x.expand(3, *x.shape[1:])

        # Uniform spatial size for stacking
        if x.shape[-2:] != (img_size, img_size):
            x = F.interpolate(
                x.unsqueeze(0), size=(img_size, img_size),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        imgs.append(x.contiguous())

    return torch.stack(imgs, dim=0)  # (B,C,H,W)

def build_benchmark_loader(args, batch_size=1, num_workers=0, shuffle=False):
    """
    Recreate your test split but with a **custom collate** that returns only images
    resized to (args.img_size, args.img_size). This prevents default_collate shape
    issues and NumPy stride pitfalls.
    """
    if args.dataset in ['Synapse', 'ACDC']:
        db = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    elif args.dataset == "Cataract1k":
        db = args.Dataset(base_dir=args.volume_path, split="val")
    elif args.dataset == "EndoVis2018":
        db = args.Dataset(base_dir=args.volume_path, split="test")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # If the dataset is smaller than the requested batch size, repeat it
    # so the first batch (and subsequent ones) can reach `batch_size`.
    try:
        n_items = len(db)
        if n_items < batch_size and n_items > 0:
            reps = (batch_size + n_items - 1) // n_items
            db = ConcatDataset([db] * reps)
    except Exception:
        # If length is not available, just proceed without repeating
        pass

    def _collate(batch):
        return collate_for_benchmark(batch, img_size=args.img_size)

    # Start with safe settings to surface errors. You can increase workers later.
    return DataLoader(
        db,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,          # start at 0; raise after it works
        pin_memory=True,
        persistent_workers=False,
        collate_fn=_collate,
    )


# ============================== Core helpers ===============================

class ParameterCounts(NamedTuple):
    trainable: int
    total: int


def count_parameters(model: nn.Module) -> ParameterCounts:
    """Return trainable and total parameter counts."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return ParameterCounts(trainable=int(trainable_params), total=int(total_params))


def _linear_like_macs(x: torch.Tensor, in_features: int, out_features: int) -> int:
    n_instances = x.numel() // in_features
    return int(n_instances * in_features * out_features)


def count_parameters_quantized(model: nn.Module) -> ParameterCounts:
    """Count logical parameters for quantized linear modules as dense weights."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    def _logical_linear_params(module: nn.Module) -> int:
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None:
            return sum(p.numel() for p in module.parameters(recurse=False))
        count = int(in_features) * int(out_features)
        bias = getattr(module, "bias", None)
        if bias is not None:
            count += int(bias.numel())
        return count

    adjusted_trainable = int(trainable_params)
    adjusted_total = int(total_params)
    seen: set[int] = set()
    for mod in model.modules():
        if isinstance(mod, W4GroupedLinear):
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            direct_params = list(mod.parameters(recurse=False))
            stored = sum(p.numel() for p in direct_params)
            stored_trainable = sum(p.numel() for p in direct_params if p.requires_grad)
            logical = _logical_linear_params(mod)
            adjusted_total += logical - int(stored)
            if stored_trainable > 0:
                adjusted_trainable += logical - int(stored_trainable)
            continue

        wq = None
        if WQLinear is not None and isinstance(mod, WQLinear):
            wq = mod
        elif (
            WQLinear is not None
            and hasattr(mod, "inner")
            and isinstance(getattr(mod, "inner"), WQLinear)
        ):
            wq = getattr(mod, "inner")
        if wq is None or id(wq) in seen:
            continue
        seen.add(id(wq))
        direct_params = list(wq.parameters(recurse=False))
        stored = sum(p.numel() for p in direct_params)
        stored_trainable = sum(p.numel() for p in direct_params if p.requires_grad)
        logical = _logical_linear_params(wq)
        adjusted_total += logical - int(stored)
        if stored_trainable > 0:
            adjusted_trainable += logical - int(stored_trainable)

    return ParameterCounts(trainable=int(adjusted_trainable), total=int(adjusted_total))


def _percentiles(values: List[float], qs=(50, 90, 95, 99)) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    out = {}
    for q in qs:
        out[f"p{q}"] = float(np.percentile(arr, q))
    return out


def _set_cudnn_benchmark(enable: bool):
    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = bool(enable)
    except Exception:
        pass


@torch.inference_mode()
def _infer_one_batch(model: nn.Module, batch_imgs: torch.Tensor) -> Any:
    return model(batch_imgs)


def _extract_images(batch: Any) -> torch.Tensor:
    """
    Accepts (images, labels) tuples, dicts with 'image'/'img' keys, or raw tensors.
    Extend here if your dataset packs inputs differently.
    """
    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
        return batch[0]
    if isinstance(batch, dict):
        for k in ("image", "img", "images", "inputs", "x"):
            if k in batch:
                return batch[k]
    if torch.is_tensor(batch):
        return batch
    raise ValueError("Cannot extract images from the given batch structure.")

# ==================== Custom FLOPs/MACs for your ViT =======================

def _build_thop_custom_ops_for_vit(model: nn.Module):
    """
    Build THOP custom op dict for your ViT attention modules.
    This adds explicit counts for the QK^T and A*V matmuls in:
      - class Attention (full multi-head attention)
      - class SHSAttention (your partial-channel attention)
    Linear/Conv layers remain counted by THOP's defaults.
    """
    custom_ops = {}
    try:
        import torch  # noqa: F401  (required by THOP callback signature)
    except Exception:
        return custom_ops

    def _count_vit_attention_thop(m, x, y):
        # x[0]: hidden_states (B, S, H)
        B, S, _ = x[0].shape
        h = int(getattr(m, "num_attention_heads"))
        d = int(getattr(m, "attention_head_size"))
        # Per head: QK^T: S*S*d, AV: S*S*d  -> total 2*S*S*d; then * B * h
        macs = B * h * S * S * d * 2
        m.total_ops += torch.DoubleTensor([macs])

    def _count_shsa_attention_thop(m, x, y):
        # x[0]: hidden_states (B, S, H)
        B, S, _ = x[0].shape
        qk_dim = int(getattr(m, "qk_dim"))
        pdim   = int(getattr(m, "pdim"))
        # QK^T uses qk_dim channels; AV uses pdim channels
        macs = B * S * S * (qk_dim + pdim)
        m.total_ops += torch.DoubleTensor([macs])

    # Map by actual module types present in the model
    for mod in model.modules():
        cname = mod.__class__.__name__
        if cname == "Attention":
            custom_ops[type(mod)] = _count_vit_attention_thop
        elif cname == "SHSAttention":
            custom_ops[type(mod)] = _count_shsa_attention_thop

    return custom_ops


def _try_count_macs_and_params_with_thop(
    model: nn.Module,
    example: torch.Tensor,
) -> Optional[Tuple[int, ParameterCounts]]:
    """
    Prefer THOP with our custom ops so attention matmuls are counted correctly.
    Returns (MACs, ParameterCounts) or None if THOP is not available.
    """
    try:
        # Work on a copy to avoid altering the caller's model device
        model_cpu = copy.deepcopy(model).to("cpu").eval()
        custom_ops = _build_thop_custom_ops_for_vit(model_cpu)
        with torch.no_grad():
            macs, _ = profile(
                model_cpu,
                inputs=(example.to("cpu"),),
                custom_ops=custom_ops,
                verbose=False,
            )
        params = count_parameters(model_cpu)
        return int(macs), params
    except Exception:
        return None


def _try_count_macs_with_fvcore(
    model: nn.Module,
    example: torch.Tensor,
) -> Optional[Tuple[int, ParameterCounts]]:
    """
    fvcore fallback (Detectron2 tool). Counts MACs; params via PyTorch.
    """
    try:
        # Work on a copy to avoid altering the caller's model device
        model_cpu = copy.deepcopy(model).to("cpu").eval()
        with torch.no_grad():
            macs = FlopCountAnalysis(model_cpu, example.to("cpu")).total()
        params = count_parameters(model_cpu)
        return int(macs), params
    except Exception:
        return None


def _fallback_hook_macs_vit(
    model: nn.Module,
    example: torch.Tensor,
    quantized_model: bool = False,
) -> int:
    """
    Robust last-resort MACs counter via forward hooks.
    Covers:
      - Conv2d / ConvTranspose2d
      - Linear
      - WQLinear when quantized_model=True
      - Attention (full MHA): counts QK^T + A*V
      - SHSAttention: counts QK^T (qk_dim) + A*V (pdim)
    Pool/Norm/Act/Upsample are typically omitted in FLOPs tables.
    """
    macs_total = 0
    handles: List[Any] = []

    def conv_hook(m: nn.Conv2d, inp, out):
        nonlocal macs_total
        x = inp[0]                        # (N, Cin, Hin, Win)
        N, Cin, _, _ = x.shape
        _, Cout, Hout, Wout = out.shape
        Kh, Kw = m.kernel_size
        g = m.groups
        macs_total += N * (Cin // g) * Cout * Kh * Kw * Hout * Wout

    def linear_hook(m: nn.Linear, inp, out):
        nonlocal macs_total
        x = inp[0]                        # (..., in_features)
        n_instances = x.numel() // m.in_features
        macs_total += n_instances * m.in_features * m.out_features

    def wqlinear_hook(m, inp, out):
        nonlocal macs_total
        in_features = getattr(m, "in_features", None)
        out_features = getattr(m, "out_features", None)
        if in_features is None or out_features is None:
            return
        macs_total += _linear_like_macs(inp[0], int(in_features), int(out_features))

    def convt_hook(m: nn.ConvTranspose2d, inp, out):
        nonlocal macs_total
        x = inp[0]                        # (N, Cin, Hin, Win)
        N, Cin, _, _ = x.shape
        _, Cout, Hout, Wout = out.shape
        Kh, Kw = m.kernel_size
        g = m.groups
        macs_total += N * (Cout // g) * Cin * Kh * Kw * Hout * Wout

    def mha_hook(m, inp, out):
        nonlocal macs_total
        # inp[0]: hidden_states (B, S, H)
        B, S, _ = inp[0].shape
        h  = int(getattr(m, "num_attention_heads"))
        d  = int(getattr(m, "attention_head_size"))
        macs_total += B * h * S * S * d * 2  # QK^T + AV

    def shsa_hook(m, inp, out):
        nonlocal macs_total
        # inp[0]: hidden_states (B, S, H)
        B, S, _ = inp[0].shape
        qk_dim = int(getattr(m, "qk_dim"))
        pdim   = int(getattr(m, "pdim"))
        macs_total += B * S * S * (qk_dim + pdim)

    # Register hooks on a copy to avoid mutating the original model. Quantized
    # WQLinear kernels are CUDA-only, so keep that path on the example device.
    target_device = example.device if quantized_model else torch.device("cpu")
    model_copy = copy.deepcopy(model).to(target_device).eval()
    for mod in model_copy.modules():
        cname = mod.__class__.__name__
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, nn.ConvTranspose2d):
            handles.append(mod.register_forward_hook(convt_hook))
        elif quantized_model and WQLinear is not None and isinstance(mod, WQLinear):
            handles.append(mod.register_forward_hook(wqlinear_hook))
        elif cname == "Attention":
            handles.append(mod.register_forward_hook(mha_hook))
        elif cname == "SHSAttention":
            handles.append(mod.register_forward_hook(shsa_hook))

    with torch.no_grad():
        _ = model_copy(example.to(target_device))
    for h in handles:
        h.remove()
    return int(macs_total)


def count_flops_gflops(
    model: nn.Module,
    example: torch.Tensor,
    quantized_model: bool = False,
) -> Tuple[float, ParameterCounts]:
    """
    Return (GFLOPs, ParameterCounts) for ONE IMAGE in `example`.

    Counting order:
      1) THOP with custom ops (preferred)
      2) fvcore (good general fallback)
      3) Custom forward-hook counter (covers Attention/SHSA, Conv/Linear/ConvT)

    Convention: FLOPs ≈ 2 × MACs (multiply + add per MAC), which matches many CV papers.
    """
    # # THOP first
    # thop_res = _try_count_macs_and_params_with_thop(model, example)
    # if thop_res is not None:
    #     macs, params = thop_res
    #     return (2.0 * macs) / 1e9, params

    # # fvcore next
    # fv_res = _try_count_macs_with_fvcore(model, example)
    # if fv_res is not None:
    #     macs, params = fv_res
    #     return (2.0 * macs) / 1e9, params

    # hook fallback
    macs = _fallback_hook_macs_vit(model, example, quantized_model=quantized_model)
    parameter_counts = count_parameters_quantized(model) if quantized_model else count_parameters(model)
    return (2.0 * macs) / 1e9, parameter_counts


def _parameter_metrics(parameter_counts: ParameterCounts) -> Dict[str, Any]:
    total_params = int(parameter_counts.total)
    trainable_params = int(parameter_counts.trainable)
    return {
        "params": round(float(total_params) / 1e6, 3),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_params_m": round(float(trainable_params) / 1e6, 3),
        "total_params_m": round(float(total_params) / 1e6, 3),
    }


# ========================== Timing (GPU/CPU) ================================

def _timed_forward_gpu(model: nn.Module, images: torch.Tensor, stream_synchronize=True) -> float:
    """Elapsed milliseconds for one forward pass on CUDA."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = _infer_one_batch(model, images)
    end.record()
    if stream_synchronize:
        torch.cuda.synchronize()
    return float(start.elapsed_time(end))  # milliseconds


def _timed_forward_cpu(model: nn.Module, images: torch.Tensor) -> float:
    """Elapsed milliseconds for one forward pass on CPU."""
    t0 = time.perf_counter()
    _ = _infer_one_batch(model, images)
    t1 = time.perf_counter()
    return float((t1 - t0) * 1000.0)


# ============================ Public API ====================================

@dataclass
class BenchmarkResults:
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def pretty(self) -> str:
        lines = ["== Segmentation Inference Benchmark =="]
        for k, v in self.metrics.items():
            if isinstance(v, (list, tuple, dict)):
                lines.append(f"{k:28s}: {v}")
            elif "gflops" in k:
                lines.append(f"{k:28s}: {v:.3f}")
            elif "img_s" in k:
                lines.append(f"{k:28s}: {v:.2f}")
            elif "ms" in k:
                lines.append(f"{k:28s}: {v:.2f}")
            elif k == "params" or k.endswith("_params_m"):
                lines.append(f"{k:28s}: {v:.3f}")
            elif k.endswith("_params"):
                lines.append(f"{k:28s}: {int(v):,}")
            else:
                lines.append(f"{k:28s}: {v}")
        return "\n".join(lines)


def benchmark_segmentation_model(
    model: nn.Module,
    test_loader: Iterable,
    device: str = "cuda",
    warmup_steps: int = 20,
    measure_batches: int = 50,
    single_image_latency_samples: int = 1000,
    enable_cudnn_benchmark: bool = True,
    autocast: bool = False,
    amp_dtype: Optional[torch.dtype] = torch.float16,
    single_image_warmup_steps: int = 50,
    quantized_model: bool = False,
    args: Any = None,
) -> BenchmarkResults:
    """
    Benchmarks using real samples from `test_loader`.

    Reports:
      - throughput_img_s
      - latency_ms_mean (from throughput loop)
      - latency_ms_p50/p90/p95/p99 (from B=1 real images)
      - trainable_params / total_params
      - trainable_params_m / total_params_m
      - flops_gflops_per_image

    Notes include device, warmup, batch shape, etc.
    """
    assert measure_batches > 0
    device = str(device)
    if quantized_model:
        if not torch.cuda.is_available():
            raise RuntimeError("Quantized benchmarking requires CUDA.")
        if not device.startswith("cuda"):
            raise ValueError("Quantized benchmarking must run on a CUDA device.")

    model = model.to(device).eval()
    _set_cudnn_benchmark(enable_cudnn_benchmark)

    # Prepare one real sample for FLOPs and for shape discovery
    first_batch = next(iter(test_loader))
    first_imgs = _extract_images(first_batch)
    if not torch.is_tensor(first_imgs):
        raise TypeError("Extracted images are not a tensor.")
    B, C, H, W = first_imgs.shape[:4]

    # Count FLOPs & params on a real example (1 image)
    example = first_imgs[:1].contiguous()
    if quantized_model:
        example = example.to(device, non_blocking=True)
    gflops, parameter_counts = count_flops_gflops(
        model,
        example,
        quantized_model=quantized_model,
    )

    # ---------------- Warm-up ----------------
    n_warm = max(0, warmup_steps)
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (device.startswith("cuda") and autocast)
        else torch.no_grad()
    )
    repeated_runs = max(1, int(getattr(args, "repeated_runs", 1)))
    n_repeated = repeated_runs
    repeated_metrics: Dict[str, Any] = {}

    def _run_summary(values: List[float]) -> Dict[str, Any]:
        arr = np.array(values, dtype=np.float64)
        return {
            "runs": [float(v) for v in values],
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    with ctx:
        it = iter(test_loader)
        for _ in range(n_warm):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(test_loader)
                batch = next(it)
            imgs = _extract_images(batch).to(device, non_blocking=device.startswith("cuda"))
            if device.startswith("cuda"):
                _ = _timed_forward_gpu(model, imgs)
            else:
                _ = _timed_forward_cpu(model, imgs)

    # ---------------- Throughput loop ----------------
    throughput_img_s_runs: List[float] = []
    latency_ms_mean_runs: List[float] = []
    for _ in range(n_repeated):
        total_imgs = 0
        total_ms = 0.0
        with ctx:
            it = iter(test_loader)
            for _ in range(measure_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(test_loader)
                    batch = next(it)
                imgs = _extract_images(batch)
                bs = int(imgs.shape[0])
                imgs = imgs.to(device, non_blocking=device.startswith("cuda"))
                ms = _timed_forward_gpu(model, imgs) if device.startswith("cuda") else _timed_forward_cpu(model, imgs)
                total_imgs += bs
                total_ms += ms

        throughput_img_s_runs.append((total_imgs / (total_ms / 1000.0)) if total_ms > 0 else float("nan"))
        latency_ms_mean_runs.append(total_ms / max(1, total_imgs))

    throughput_img_s = throughput_img_s_runs[0]
    mean_latency_ms_per_image = latency_ms_mean_runs[0]
    if repeated_runs > 1:
        throughput_summary = _run_summary(throughput_img_s_runs)
        latency_mean_summary = _run_summary(latency_ms_mean_runs)
        throughput_img_s = throughput_summary["mean"]
        mean_latency_ms_per_image = latency_mean_summary["mean"]
        repeated_metrics.update({
            "repeated_runs": repeated_runs,
            "throughput_img_s_runs": throughput_summary["runs"],
            "throughput_img_s_mean": throughput_summary["mean"],
            "throughput_img_s_std": throughput_summary["std"],
            "latency_ms_mean_runs": latency_mean_summary["runs"],
            "latency_ms_mean_mean": latency_mean_summary["mean"],
            "latency_ms_mean_std": latency_mean_summary["std"],
        })

    # ---------------- Single-image warm-up for B=1 latency path ----------------
    
    #---single image warmup for B=1 latency path---
    n_single_warm = max(0, int(single_image_warmup_steps))
    if single_image_latency_samples > 0 and n_single_warm > 0:
        it = iter(test_loader)
        warmed = 0
        with ctx:
            while warmed < n_single_warm:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(test_loader)
                    batch = next(it)

                imgs = _extract_images(batch)
                for i in range(imgs.shape[0]):
                    x = imgs[i : i + 1].to(
                        device,
                        non_blocking=device.startswith("cuda"),
                    )
                    if device.startswith("cuda"):
                        _ = _timed_forward_gpu(model, x)
                    else:
                        _ = _timed_forward_cpu(model, x)

                    warmed += 1
                    if warmed >= n_single_warm:
                        break
    #------end of single image warmup for B=1 latency path------
    
                    
    #---------------- Single-image latency path ----------------
                    
    percentile_runs: List[Dict[str, float]] = []
    for _ in range(n_repeated):
        single_lat_ms: List[float] = []
        if single_image_latency_samples > 0:
            it = iter(test_loader)
            collected = 0
            with ctx:
                while collected < single_image_latency_samples:
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(test_loader)
                        batch = next(it)
                    imgs = _extract_images(batch)
                    for i in range(imgs.shape[0]):
                        x = imgs[i : i + 1].to(device, non_blocking=device.startswith("cuda"))
                        ms = _timed_forward_gpu(model, x) if device.startswith("cuda") else _timed_forward_cpu(model, x)
                        single_lat_ms.append(ms)
                        collected += 1
                        if collected >= single_image_latency_samples:
                            break
        percentile_runs.append(_percentiles(single_lat_ms) if len(single_lat_ms) else {})

    percentiles = percentile_runs[0]
    if repeated_runs > 1:
        for q in (50, 90, 95, 99):
            percentile_key = f"p{q}"
            metric_key = f"latency_ms_p{q}"
            summary = _run_summary([
                run.get(percentile_key, float("nan"))
                for run in percentile_runs
            ])
            percentiles[percentile_key] = summary["mean"]
            repeated_metrics.update({
                f"{metric_key}_runs": summary["runs"],
                f"{metric_key}_mean": summary["mean"],
                f"{metric_key}_std": summary["std"],
            })

    metrics = {
        "throughput_img_s": throughput_img_s,
        "latency_ms_mean": mean_latency_ms_per_image,
        "latency_ms_p50": percentiles.get("p50", float("nan")),
        "latency_ms_p90": percentiles.get("p90", float("nan")),
        "latency_ms_p95": percentiles.get("p95", float("nan")),
        "latency_ms_p99": percentiles.get("p99", float("nan")),
        **_parameter_metrics(parameter_counts),
        "flops_gflops_per_image": float(gflops),  # per single image in `example`
        "image_shape_HxW": float(H * W),
        "channels": float(C),
        "batch_size_first": float(B),
        **repeated_metrics,
    }
    notes = {
        "checkpoint" : args.ckpt if args is not None else "N/A",
        "device": device,
        "warmup_steps": n_warm,
        "measure_batches": measure_batches,
        "single_image_latency_samples": single_image_latency_samples,
        "cudnn_benchmark": enable_cudnn_benchmark,
        "amp_autocast": bool(autocast),
        "quantized_model": bool(quantized_model),
    }
    return BenchmarkResults(metrics=metrics, notes=notes)
