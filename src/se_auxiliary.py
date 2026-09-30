"""Detached channelwise quantization targets and SE checkpoint configuration.

This module depends only on PyTorch and the standard library. Its asymmetric
weight QDQ mirrors AWQ's ``pseudo_quantize_tensor`` without importing the
deployment quantizer (or its CUDA/transformers dependencies) during training.
"""

import json
import math
from pathlib import Path

import torch


DEFAULT_SE_AUX_CONFIG = {
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "eps": 1e-12,
}


def _validate_target_config(bits, group_size, zero_point, eps):
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 16:
        raise ValueError("SE target bits must be an integer between 1 and 16.")
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise ValueError("SE target group_size must be an integer.")
    if zero_point is not True:
        raise ValueError("SE targets require zero_point=True, matching deployment AWQ.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("SE target eps must be finite and strictly positive.")


@torch.no_grad()
def build_quantization_targets(
    X, W, bits=4, group_size=128, zero_point=True, eps=1e-12
):
    """Return detached float32 ``[B, 1, C]`` quantization-error proxies.

    ``X`` is the actual linear projection input ``[B, N, C]`` and ``W`` its
    current full-precision weight ``[O, C]``. Groups are contiguous input-channel
    groups within each output row, exactly as in AWQ. Nonpositive group sizes
    select whole rows, also matching AWQ. Constant groups retain AWQ's 1e-5
    range clamp and clipped zero points; they are not special-cased to identity.
    """
    _validate_target_config(bits, group_size, zero_point, eps)
    if X.ndim != 3 or W.ndim != 2 or X.shape[-1] != W.shape[-1]:
        raise ValueError("SE targets require X[B,N,C] and W[O,C] with equal C.")
    if min(X.shape) == 0 or min(W.shape) == 0:
        raise ValueError("SE target inputs must have nonempty dimensions.")
    if X.device != W.device:
        raise ValueError("SE target activations and weights must share a device.")
    if group_size > 0 and W.shape[-1] % group_size:
        raise ValueError(
            f"SE target group_size={group_size} must divide input channels={W.shape[-1]}."
        )

    X_ref = X.detach().float()
    W_ref = W.detach().float()
    # A private copy ensures that model storage is never modified by QDQ.
    weight = W_ref.clone()
    if group_size > 0:
        weight = weight.reshape(-1, group_size)
    max_value = weight.amax(dim=1, keepdim=True)
    min_value = weight.amin(dim=1, keepdim=True)
    max_int = 2**bits - 1
    scales = (max_value - min_value).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_value / scales)).clamp(0, max_int)
    W_hat = (
        (torch.clamp(torch.round(weight / scales) + zeros, 0, max_int) - zeros)
        * scales
    ).reshape(W_ref.shape)

    weight_error_ms = (W_ref - W_hat).square().mean(dim=0)
    activation_ms = X_ref.square().mean(dim=1)
    channel_error_rms = (activation_ms * weight_error_ms.unsqueeze(0)).sqrt()
    mu = channel_error_rms.mean(dim=-1, keepdim=True)
    targets = (channel_error_rms + eps) / (channel_error_rms + mu + 2 * eps)
    targets = targets.unsqueeze(1)
    if not torch.isfinite(targets).all():
        raise ValueError("SE targets are nonfinite; check activation/weight values and scale.")
    return targets


def _encoder(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if hasattr(model, "transformer"):
        return model.transformer.encoder
    # Also permit the standalone Encoder used by focused tests and consumers.
    return model


def _metadata_path(checkpoint_path):
    return Path(str(checkpoint_path) + ".se_aux.json")


def save_se_metadata(model, checkpoint_path):
    """Save parameter-free SE configuration beside an unchanged state dict."""
    encoder = _encoder(model)
    layers = getattr(encoder, "SELayer", None)
    metadata_path = _metadata_path(checkpoint_path)
    if layers is None or len(layers) == 0:
        # Reusing a checkpoint filename must not retain obsolete SE metadata.
        if metadata_path.exists():
            metadata_path.unlink()
        return None
    modes = {getattr(layer, "pooling_mode", "mean") for layer in layers}
    if len(modes) != 1 or not modes <= {"mean", "rms"}:
        raise ValueError("All SE predictors must share a supported pooling mode.")
    config = dict(getattr(encoder, "se_aux_config", DEFAULT_SE_AUX_CONFIG))
    _validate_target_config(**config)
    metadata = {
        "version": 1,
        "pooling_mode": modes.pop(),
        "num_blocks": len(layers),
        "target_config": config,
        "target_definition": "mean_qkv_fc1_channel_error_rms_v1",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_se_metadata(model, checkpoint_path):
    """Restore predictor pooling for inference, always disabling auxiliary loss.

    A checkpoint without a sidecar is legacy and therefore uses mean pooling.
    Keep the sidecar with its associated ``.pth`` when copying checkpoints.
    """
    encoder = _encoder(model)
    metadata_path = _metadata_path(checkpoint_path)
    metadata = None
    pooling_mode = "mean"
    config = dict(DEFAULT_SE_AUX_CONFIG)
    layers = getattr(encoder, "SELayer", None)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("version") != 1:
            raise ValueError("Unsupported SE checkpoint metadata version.")
        pooling_mode = metadata.get("pooling_mode")
        if pooling_mode not in {"mean", "rms"}:
            raise ValueError("SE checkpoint pooling_mode must be 'mean' or 'rms'.")
        config = metadata.get("target_config")
        if not isinstance(config, dict) or set(config) != set(DEFAULT_SE_AUX_CONFIG):
            raise ValueError("SE checkpoint metadata has an invalid target_config.")
        _validate_target_config(**config)
        if layers is None or len(layers) != metadata.get("num_blocks"):
            raise ValueError(
                "SE checkpoint metadata does not match model predictors; use --use_se_block "
                "and the checkpoint's encoder architecture."
            )

    encoder.se_aux_loss = False
    encoder.se_aux_config = dict(config)
    for block in getattr(encoder, "layer", []):
        block.se_aux_config = dict(config)
    if layers is not None:
        for layer in layers:
            layer.pooling_mode = pooling_mode
    args = getattr(encoder, "args", None)
    if args is not None:
        args.se_aux_loss = False
        args.se_pooling_mode = pooling_mode
        for name, value in config.items():
            setattr(args, "se_aux_" + name, value)
    return metadata


def validate_se_calibration_config(model, bits=4, group_size=128, zero_point=True):
    """Reject deployment settings inconsistent with an RMS-trained predictor."""
    encoder = _encoder(model)
    layers = getattr(encoder, "SELayer", None)
    if layers is None or not any(getattr(layer, "pooling_mode", "mean") == "rms" for layer in layers):
        return
    target_config = getattr(encoder, "se_aux_config", DEFAULT_SE_AUX_CONFIG)
    deployment_config = dict(bits=bits, group_size=group_size, zero_point=zero_point)
    for name, value in deployment_config.items():
        if target_config[name] != value:
            raise ValueError(
                f"SE target {name}={target_config[name]} does not match calibration {name}={value}."
            )
