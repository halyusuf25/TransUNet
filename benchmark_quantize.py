# benchmark_quantize.py
# ---------------------------------------------------------------------------
# Quantized segmentation benchmarking utility (PyTorch) with GPU-only timing:
# - Mirrors benchmark_segmentation_model() metrics and reporting
# - Handles AWQ WQLinear layers during FLOPs/MACs counting
# - Keeps preprocessing and timing consistent for apples-to-apples comparison
# ---------------------------------------------------------------------------

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis
from thop import profile

from benchmark import (
    BenchmarkResults,
    _build_thop_custom_ops_for_vit,
    _extract_images,
    _percentiles,
    _set_cudnn_benchmark,
    _timed_forward_gpu,
    count_parameters,
    count_flops_gflops,
    )

# try:
#     from awq.quantize.qmodule import WQLinear
# except Exception:  # pragma: no cover - optional dependency
#     WQLinear = None  # type: ignore[assignment]
    
from awq.quantize.qmodule import WQLinear


def _linear_like_macs(x: torch.Tensor, in_features: int, out_features: int) -> int:
    n_instances = x.numel() // in_features
    return int(n_instances * in_features * out_features)


def _build_thop_custom_ops_for_vit_quantized(model: nn.Module) -> Dict[type, Any]:
    custom_ops = _build_thop_custom_ops_for_vit(model)
    if WQLinear is None:
        return custom_ops

    def _count_wqlinear_thop(m, x, y):
        in_features = getattr(m, "in_features", None)
        out_features = getattr(m, "out_features", None)
        if in_features is None or out_features is None:
            return
        macs = _linear_like_macs(x[0], int(in_features), int(out_features))
        m.total_ops += torch.DoubleTensor([macs])

    for mod in model.modules():
        if isinstance(mod, WQLinear):
            custom_ops[type(mod)] = _count_wqlinear_thop

    return custom_ops


def _try_count_macs_and_params_with_thop_quantized(
    model: nn.Module,
    example: torch.Tensor,
) -> Optional[Tuple[int, int]]:
    try:
        model_gpu = copy.deepcopy(model).to(example.device).eval()
        custom_ops = _build_thop_custom_ops_for_vit_quantized(model_gpu)
        with torch.no_grad():
            macs, params = profile(
                model_gpu,
                inputs=(example,),
                custom_ops=custom_ops,
                verbose=False,
            )
        return int(macs), int(params)
    except Exception:
        return None


def _try_count_macs_with_fvcore_quantized(
    model: nn.Module,
    example: torch.Tensor,
) -> Optional[Tuple[int, int]]:
    try:
        model_gpu = copy.deepcopy(model).to(example.device).eval()
        with torch.no_grad():
            macs = FlopCountAnalysis(model_gpu, example).total()
        params = count_parameters(model_gpu)
        return int(macs), int(params)
    except Exception:
        return None


def _fallback_hook_macs_vit_quantized(model: nn.Module, example: torch.Tensor) -> int:
    macs_total = 0
    handles: List[Any] = []

    def conv_hook(m: nn.Conv2d, inp, out):
        nonlocal macs_total
        x = inp[0]
        n, c_in, _, _ = x.shape
        _, c_out, h_out, w_out = out.shape
        k_h, k_w = m.kernel_size
        g = m.groups
        macs_total += n * (c_in // g) * c_out * k_h * k_w * h_out * w_out

    def linear_hook(m: nn.Linear, inp, out):
        nonlocal macs_total
        x = inp[0]
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
        x = inp[0]
        n, c_in, _, _ = x.shape
        _, c_out, h_out, w_out = out.shape
        k_h, k_w = m.kernel_size
        g = m.groups
        macs_total += n * (c_out // g) * c_in * k_h * k_w * h_out * w_out

    def mha_hook(m, inp, out):
        nonlocal macs_total
        b, s, _ = inp[0].shape
        h = int(getattr(m, "num_attention_heads"))
        d = int(getattr(m, "attention_head_size"))
        macs_total += b * h * s * s * d * 2

    def shsa_hook(m, inp, out):
        nonlocal macs_total
        b, s, _ = inp[0].shape
        qk_dim = int(getattr(m, "qk_dim"))
        pdim = int(getattr(m, "pdim"))
        macs_total += b * s * s * (qk_dim + pdim)

    model_gpu = copy.deepcopy(model).to(example.device).eval()
    for mod in model_gpu.modules():
        cname = mod.__class__.__name__
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, nn.ConvTranspose2d):
            handles.append(mod.register_forward_hook(convt_hook))
        elif WQLinear is not None and isinstance(mod, WQLinear):
            handles.append(mod.register_forward_hook(wqlinear_hook))
        elif cname == "Attention":
            handles.append(mod.register_forward_hook(mha_hook))
        elif cname == "SHSAttention":
            handles.append(mod.register_forward_hook(shsa_hook))

    with torch.no_grad():
        _ = model_gpu(example)
    for h in handles:
        h.remove()
    return int(macs_total)


def count_flops_gflops_quantized(model: nn.Module, example: torch.Tensor) -> Tuple[float, int]:
    # thop_res = _try_count_macs_and_params_with_thop_quantized(model, example)
    # if thop_res is not None:
    #     print(f"THOB result used for quantized model FLOPs and Params counting.")
    #     macs, params = thop_res
    #     return (2.0 * macs) / 1e9, params

    # fv_res = _try_count_macs_with_fvcore_quantized(model, example)
    # if fv_res is not None:
    #     print(f"FVCore result used for quantized model FLOPs and Params counting.")
    #     macs, params = fv_res
    #     return (2.0 * macs) / 1e9, params

    macs = _fallback_hook_macs_vit_quantized(model, example)
    params = count_parameters(model)
    return (2.0 * macs) / 1e9, params


def benchmark_segmentation_quantize_model(
    model: nn.Module,
    test_loader: Iterable,
    device: str = "cuda",
    warmup_steps: int = 20,
    measure_batches: int = 50,
    single_image_latency_samples: int = 200,
    enable_cudnn_benchmark: bool = True,
    autocast: bool = False,
    amp_dtype: Optional[torch.dtype] = torch.float16,
    args: Any = None,
) -> BenchmarkResults:
    """
    Benchmarks a quantized segmentation model on GPU using real samples from `test_loader`.

    Reports (same as benchmark_segmentation_model):
      - throughput_img_s
      - latency_ms_mean (from throughput loop)
      - latency_ms_p50/p90/p95/p99 (from B=1 real images)
      - params
      - flops_gflops_per_image
    """
    assert measure_batches > 0
    if not torch.cuda.is_available():
        raise RuntimeError("Quantized benchmarking requires CUDA.")
    device = str(device)
    if not device.startswith("cuda"):
        raise ValueError("Quantized benchmarking must run on a CUDA device.")

    model = model.to(device).eval()
    _set_cudnn_benchmark(enable_cudnn_benchmark)

    # Prepare one real sample for FLOPs and for shape discovery
    first_batch = next(iter(test_loader))
    first_imgs = _extract_images(first_batch)
    if not torch.is_tensor(first_imgs):
        raise TypeError("Extracted images are not a tensor.")
    b, c, h, w = first_imgs.shape[:4]

    # Count FLOPs & params on a real example (1 image)
    example = first_imgs[:1].contiguous().to(device, non_blocking=True)
    gflops, n_params = count_flops_gflops_quantized(model, example)
    
    # ---------------- Warm-up ----------------
    n_warm = max(0, warmup_steps)
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if autocast
        else torch.no_grad()
    )
    with ctx:
        it = iter(test_loader)
        for _ in range(n_warm):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(test_loader)
                batch = next(it)
            imgs = _extract_images(batch).to(device, non_blocking=True)
            _ = _timed_forward_gpu(model, imgs)

    # ---------------- Throughput loop ----------------
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
            imgs = imgs.to(device, non_blocking=True)
            ms = _timed_forward_gpu(model, imgs)
            total_imgs += bs
            total_ms += ms

    throughput_img_s = (total_imgs / (total_ms / 1000.0)) if total_ms > 0 else float("nan")
    mean_latency_ms_per_image = (total_ms / max(1, total_imgs))

    # ---------------- Single-image latency percentiles ----------------
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
                    x = imgs[i : i + 1].to(device, non_blocking=True)
                    ms = _timed_forward_gpu(model, x)
                    single_lat_ms.append(ms)
                    collected += 1
                    if collected >= single_image_latency_samples:
                        break

    percentiles = _percentiles(single_lat_ms) if len(single_lat_ms) else {}

    metrics = {
        "throughput_img_s": throughput_img_s,
        "latency_ms_mean": mean_latency_ms_per_image,
        "latency_ms_p50": percentiles.get("p50", float("nan")),
        "latency_ms_p90": percentiles.get("p90", float("nan")),
        "latency_ms_p95": percentiles.get("p95", float("nan")),
        "latency_ms_p99": percentiles.get("p99", float("nan")),
        "params": round(float(n_params) / 1e6, 3),
        "flops_gflops_per_image": float(gflops),
        "image_shape_HxW": float(h * w),
        "channels": float(c),
        "batch_size_first": float(b),
    }
    notes = {
        "checkpoint" : args.ckpt if args is not None else "N/A",
        "device": device,
        "warmup_steps": n_warm,
        "measure_batches": measure_batches,
        "single_image_latency_samples": single_image_latency_samples,
        "cudnn_benchmark": enable_cudnn_benchmark,
        "amp_autocast": bool(autocast),
    }
    return BenchmarkResults(metrics=metrics, notes=notes)
