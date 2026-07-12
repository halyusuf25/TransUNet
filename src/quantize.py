from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


_OFFICIAL_AWQ_RUNTIME: Optional[Dict[str, Any]] = None
_OFFICIAL_AWQ_ERROR = (
    "The custom_w4 backend now requires the MIT-HAN-Lab AWQ W4A16 CUDA "
    "extension. Install llm-awq and build awq/kernels before running this "
    "backend. The reference eager-dequantization path is not used for "
    "deployment benchmarks."
)


def require_official_awq_runtime() -> Dict[str, Any]:
    """Load and validate the official MIT-HAN-Lab AWQ CUDA runtime."""
    global _OFFICIAL_AWQ_RUNTIME
    if _OFFICIAL_AWQ_RUNTIME is not None:
        return _OFFICIAL_AWQ_RUNTIME

    problems = []
    if not torch.cuda.is_available():
        problems.append("CUDA is not available")

    try:
        awq_module = importlib.import_module("awq")
    except Exception as exc:  # pragma: no cover - depends on deployment environment
        awq_module = None
        problems.append(f"could not import awq ({exc})")

    try:
        inference_engine = importlib.import_module("awq_inference_engine")
    except Exception as exc:  # pragma: no cover - depends on deployment environment
        inference_engine = None
        problems.append(f"could not import awq_inference_engine ({exc})")

    pseudo_quantize_tensor = None
    wqlinear = None
    auto_clip_layer = None
    qmodule = None
    if awq_module is not None and inference_engine is not None:
        try:
            quantizer_module = importlib.import_module("awq.quantize.quantizer")
            qmodule = importlib.import_module("awq.quantize.qmodule")
            auto_clip_module = importlib.import_module("awq.quantize.auto_clip")
            pseudo_quantize_tensor = quantizer_module.pseudo_quantize_tensor
            wqlinear = qmodule.WQLinear
            auto_clip_layer = auto_clip_module.auto_clip_layer
        except Exception as exc:  # pragma: no cover - depends on deployment environment
            problems.append(f"could not import official AWQ quantization modules ({exc})")

    for kernel_name in ("gemm_forward_cuda_new", "gemv_forward_cuda_new"):
        if inference_engine is not None and not hasattr(inference_engine, kernel_name):
            problems.append(f"awq_inference_engine is missing {kernel_name}")

    if problems:
        raise RuntimeError(f"{_OFFICIAL_AWQ_ERROR} Details: {'; '.join(problems)}.")

    _OFFICIAL_AWQ_RUNTIME = {
        "pseudo_quantize_tensor": pseudo_quantize_tensor,
        "WQLinear": wqlinear,
        "auto_clip_layer": auto_clip_layer,
        "awq_module_path": inspect.getfile(qmodule),
        "awq_inference_engine_path": getattr(inference_engine, "__file__", None),
        "awq_inference_engine": inference_engine,
    }
    return _OFFICIAL_AWQ_RUNTIME


def image_to_nchw_for_quant_calib(image, args):
    """Convert one calibration sample from repo dataset format to model-ready NCHW tensor."""
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)

    image = image.detach().cpu().float()
    img_size = int(args.img_size)

    if args.dataset in ["Synapse", "ACDC"]:
        # DataLoader usually gives [1, D, H, W] for volume tests.
        # Convert to [D, 1, H, W], so each slice becomes one calibration image.
        if image.dim() == 4 and image.shape[0] == 1:
            x = image.squeeze(0).unsqueeze(1)
        elif image.dim() == 3:
            x = image.unsqueeze(1)
        elif image.dim() == 4 and image.shape[1] in (1, 3):
            x = image
        else:
            raise ValueError(
                f"Unsupported {args.dataset} calibration image shape: {tuple(image.shape)}"
            )
    else:
        # Frame datasets: accept [1, H, W, 3], [1, 3, H, W], [H, W, 3], [3, H, W].
        if image.dim() == 4 and image.shape[1] in (1, 3):
            x = image
        elif image.dim() == 4 and image.shape[-1] in (1, 3):
            x = image.permute(0, 3, 1, 2)
        elif image.dim() == 3 and image.shape[0] in (1, 3):
            x = image.unsqueeze(0)
        elif image.dim() == 3 and image.shape[-1] in (1, 3):
            x = image.permute(2, 0, 1).unsqueeze(0)
        elif image.dim() == 2:
            x = image.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported calibration image shape: {tuple(image.shape)}")

    if x.shape[-2:] != (img_size, img_size):
        x = F.interpolate(
            x,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

    return x.contiguous()


def inc_awq_image_to_nchw(image, args):
    """Backward-compatible alias for the backend-neutral calibration image helper."""
    return image_to_nchw_for_quant_calib(image, args)


def collect_inc_awq_calib_inputs(args, calib_loader, max_forwards, chunk_size=8):
    """Collect a small list of tensors to run through INC AWQ calibration."""
    calib_inputs = []
    max_forwards = int(max_forwards)
    chunk_size = max(1, int(chunk_size))

    for sampled_batch in calib_loader:
        x = image_to_nchw_for_quant_calib(sampled_batch["image"], args)

        for chunk in x.split(chunk_size, dim=0):
            calib_inputs.append(chunk)
            if len(calib_inputs) >= max_forwards:
                return calib_inputs

    return calib_inputs


def _first_tensor_from_output(output):
    if isinstance(output, dict):
        for key in ("logits", "out", "output"):
            if key in output:
                return output[key]
        raise ValueError("Model output dict does not contain logits/out/output.")
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model output tuple/list is empty.")
        return output[0]
    if torch.is_tensor(output):
        return output
    raise ValueError(f"Unsupported model output type: {type(output).__name__}")


def _extract_logits_and_se_scales(output, expected_num_blocks):
    """Return logits and SE scales from tuple/list/dict model outputs."""
    if isinstance(output, dict):
        logits = None
        for key in ("logits", "out", "output"):
            if key in output:
                logits = output[key]
                break
        se_scales = None
        for key in ("se_scale", "se_scales", "se"):
            if key in output:
                se_scales = output[key]
                break
        if logits is None:
            raise ValueError("Model output dict does not contain logits/out/output.")
        if se_scales is None:
            raise ValueError("Model output dict does not contain se_scale/se_scales/se.")
    elif isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model output tuple/list is empty.")
        logits = output[0]
        se_scales = output[-1]
    else:
        raise ValueError(
            "SE-guided quantization requires model output with logits and SE scales; "
            f"got {type(output).__name__}."
        )

    if not isinstance(se_scales, (list, tuple)):
        raise ValueError(
            "SE scales must be a list/tuple with one tensor per encoder block; "
            f"got {type(se_scales).__name__}."
        )
    if len(se_scales) != int(expected_num_blocks):
        raise ValueError(
            "SE scales length mismatch: expected {} blocks, got {}.".format(
                expected_num_blocks, len(se_scales)
            )
        )
    return logits, se_scales


def _normalize_se_gate_tensor(gate, expected_channels=None):
    """Normalize SE gate tensors to shape (B, C)."""
    if not torch.is_tensor(gate):
        gate = torch.as_tensor(gate)

    if gate.dim() == 3 and gate.shape[1] == 1:
        gate = gate.squeeze(1)
    elif gate.dim() == 4 and gate.shape[2] == 1 and gate.shape[3] == 1:
        gate = gate.squeeze(-1).squeeze(-1)
    elif gate.dim() == 2:
        pass
    else:
        raise ValueError(
            "Unsupported SE gate shape {}; expected (B, 1, C), (B, C, 1, 1), "
            "or (B, C).".format(tuple(gate.shape))
        )

    if gate.dim() != 2:
        raise ValueError(f"SE gate normalization failed for shape {tuple(gate.shape)}.")
    if expected_channels is not None and gate.shape[1] != int(expected_channels):
        raise ValueError(
            "SE gate channel mismatch: expected {}, got {}.".format(
                expected_channels, gate.shape[1]
            )
        )
    if not torch.isfinite(gate).all():
        raise ValueError("SE gate contains NaN or Inf.")
    return gate.float()


class W4GroupedLinear(nn.Module):
    """Reference/debug W4 Linear; not used by custom_w4 deployment benchmarks."""

    pack_factor = 8

    def __init__(
        self,
        qweight,
        scales,
        bias,
        in_features,
        out_features,
        group_size=128,
        w_bit=4,
    ):
        super().__init__()
        if int(w_bit) != 4:
            raise ValueError(f"W4GroupedLinear only supports w_bit=4, got {w_bit}.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.w_bit = int(w_bit)
        self.register_buffer("qweight", qweight.contiguous().to(torch.int32))
        self.register_buffer("scales", scales.contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().contiguous())

    @classmethod
    def from_linear(cls, linear, group_size=128, w_bit=4):
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(linear).__name__}.")
        _validate_w4_grouped_linear(linear, group_size=group_size, w_bit=w_bit)
        qweight, scales = _pack_w4_grouped_weight(
            linear.weight.detach(),
            group_size=group_size,
            w_bit=w_bit,
        )
        bias = None if linear.bias is None else linear.bias.detach().clone()
        return cls(
            qweight=qweight,
            scales=scales,
            bias=bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            w_bit=w_bit,
        )

    def _dequantize_weight(self, dtype):
        device = self.qweight.device
        shifts = torch.arange(
            self.pack_factor,
            device=device,
            dtype=torch.int32,
        ).view(1, 1, self.pack_factor) * 4
        unpacked = ((self.qweight.unsqueeze(-1) >> shifts) & 0xF).to(torch.int16)
        q = unpacked.reshape(self.out_features, -1)[:, : self.in_features]
        q = q.to(dtype=dtype) - 8.0
        q = q.reshape(self.out_features, -1, self.group_size)
        scales = self.scales.to(device=device, dtype=dtype).unsqueeze(-1)
        return (q * scales).reshape(self.out_features, self.in_features)

    def forward(self, x):
        weight = self._dequantize_weight(x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, w_bit={self.w_bit}, group_size={self.group_size}"
        )


def _validate_w4_grouped_linear(linear, group_size=128, w_bit=4):
    if int(w_bit) != 4:
        raise ValueError(f"Only W4 quantization is supported, got w_bit={w_bit}.")
    in_features = int(linear.in_features)
    out_features = int(linear.out_features)
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if in_features % group_size != 0:
        raise ValueError(
            "{} in_features={} is not divisible by group_size={}.".format(
                linear.__class__.__name__, in_features, group_size
            )
        )
    if in_features % W4GroupedLinear.pack_factor != 0:
        raise ValueError(
            "in_features={} is not divisible by W4 pack factor {}.".format(
                in_features, W4GroupedLinear.pack_factor
            )
        )
    if out_features % W4GroupedLinear.pack_factor != 0:
        raise ValueError(
            "out_features={} is not divisible by W4 row pack factor {}.".format(
                out_features, W4GroupedLinear.pack_factor
            )
        )
    if out_features <= 0:
        raise ValueError(f"out_features must be positive, got {out_features}.")


def _pack_w4_grouped_weight(weight, group_size=128, w_bit=4):
    if int(w_bit) != 4:
        raise ValueError(f"Only W4 quantization is supported, got w_bit={w_bit}.")
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D Linear weight, got {tuple(weight.shape)}.")

    out_features, in_features = weight.shape
    if in_features % int(group_size) != 0:
        raise ValueError(
            "Linear in_features={} is not divisible by group_size={}.".format(
                in_features, group_size
            )
        )
    if in_features % W4GroupedLinear.pack_factor != 0:
        raise ValueError(
            "Linear in_features={} is not divisible by W4 pack factor {}.".format(
                in_features, W4GroupedLinear.pack_factor
            )
        )
    if out_features % W4GroupedLinear.pack_factor != 0:
        raise ValueError(
            "Linear out_features={} is not divisible by W4 row pack factor {}.".format(
                out_features, W4GroupedLinear.pack_factor
            )
        )
    if out_features <= 0:
        raise ValueError(f"Linear out_features must be positive, got {out_features}.")

    qmax = (1 << (int(w_bit) - 1)) - 1
    weight_fp = weight.detach().float().reshape(out_features, -1, int(group_size))
    scales = weight_fp.abs().amax(dim=-1).clamp(min=1e-8) / float(qmax)
    q = torch.round(weight_fp / scales.unsqueeze(-1)).clamp(-qmax, qmax)
    q = q.reshape(out_features, in_features).to(torch.int16)
    q = (q + 8).clamp(0, 15).to(torch.int32)

    packed_cols = in_features // W4GroupedLinear.pack_factor
    q = q.reshape(out_features, packed_cols, W4GroupedLinear.pack_factor)
    shifts = (
        torch.arange(W4GroupedLinear.pack_factor, device=q.device, dtype=torch.int32)
        .view(1, 1, W4GroupedLinear.pack_factor)
        * 4
    )
    qweight = torch.sum(q << shifts, dim=-1).to(torch.int32)
    return qweight, scales.to(dtype=weight.dtype)


def _activation_channel_mean(x):
    if not torch.is_tensor(x):
        raise TypeError(f"Expected tensor activation, got {type(x).__name__}.")
    if x.dim() < 2:
        raise ValueError(f"Expected activation with batch and channel dims, got {tuple(x.shape)}.")
    batch = x.shape[0]
    channels = x.shape[-1]
    return x.detach().float().reshape(batch, -1, channels).abs().mean(dim=1)


def _module_first_tensor(output):
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Module output tuple/list is empty.")
        return output[0]
    if isinstance(output, dict):
        for value in output.values():
            if torch.is_tensor(value):
                return value
        raise ValueError("Module output dict does not contain a tensor.")
    return output


def _get_submodule(root, path):
    module = root
    for part in path.split("."):
        if not hasattr(module, part):
            raise AttributeError(f"{module.__class__.__name__} has no submodule '{part}'.")
        module = getattr(module, part)
    return module


def _set_submodule(root, path, value):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            raise AttributeError(f"{parent.__class__.__name__} has no submodule '{part}'.")
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _fake_w4_quantize_weight_for_aux(weight, group_size=128, w_bit=4):
    if int(w_bit) != 4:
        raise ValueError(f"Only W4 quantization is supported for SE auxiliary loss, got w_bit={w_bit}.")
    if not torch.is_tensor(weight):
        raise TypeError(f"Expected tensor weight, got {type(weight).__name__}.")
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D Linear weight, got {tuple(weight.shape)}.")

    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            "Linear in_features={} is not divisible by group_size={} for SE auxiliary loss.".format(
                in_features, group_size
            )
        )

    qmax = (1 << (int(w_bit) - 1)) - 1
    weight_fp = weight.detach().float()
    grouped = weight_fp.reshape(out_features, -1, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / float(qmax)
    quant = torch.round(grouped / scales).clamp(-qmax, qmax)
    return (quant * scales).reshape_as(weight_fp)


def _input_channel_quant_error(weight, group_size=128, w_bit=4):
    fake_quant = _fake_w4_quantize_weight_for_aux(
        weight,
        group_size=group_size,
        w_bit=w_bit,
    )
    err = (weight.detach().float() - fake_quant).abs().mean(dim=0)
    if err.dim() != 1:
        raise ValueError(f"Expected per-input-channel error to be 1D, got {tuple(err.shape)}.")
    if not torch.isfinite(err).all():
        raise ValueError("Per-input-channel quantization error contains NaN or Inf.")
    return err.detach()


def _normalize_sensitivity_target(sensitivity):
    if not torch.is_tensor(sensitivity):
        sensitivity = torch.as_tensor(sensitivity)
    if sensitivity.dim() != 1:
        raise ValueError(
            "Expected 1D sensitivity target before normalization, got {}.".format(
                tuple(sensitivity.shape)
            )
        )
    sensitivity = sensitivity.detach().float()
    if not torch.isfinite(sensitivity).all():
        raise ValueError("Sensitivity target contains NaN or Inf before normalization.")

    std = sensitivity.std(unbiased=False).clamp(min=1e-6)
    target = torch.sigmoid((sensitivity - sensitivity.mean()) / std)
    if not torch.isfinite(target).all():
        raise ValueError("Sensitivity target contains NaN or Inf after normalization.")
    return target.detach()


def _linear_weight_for_aux(block, rel_path, block_idx):
    module = _get_submodule(block, rel_path)
    if not isinstance(module, nn.Linear):
        raise TypeError(
            "Block {} {} must be nn.Linear for SE auxiliary loss, got {}.".format(
                block_idx, rel_path, type(module).__name__
            )
        )
    return module.weight


def se_quant_sensitivity_aux_loss(
    model,
    se_scales,
    group_size=128,
    w_bit=4,
):
    """Train calibration-only SE gates to predict per-channel W4 sensitivity."""
    if int(w_bit) != 4:
        raise ValueError(f"se_quant_sensitivity_aux_loss currently supports only w_bit=4, got {w_bit}.")
    if not isinstance(se_scales, (list, tuple)):
        raise ValueError(
            "SE auxiliary loss requires se_scales as a list/tuple with one tensor per encoder block; "
            f"got {type(se_scales).__name__}."
        )

    core_model = model.module if hasattr(model, "module") else model
    transformer = getattr(core_model, "transformer", None)
    encoder = getattr(transformer, "encoder", None)
    blocks = getattr(encoder, "layer", None)
    if encoder is None or blocks is None:
        raise ValueError("Could not locate encoder blocks at model.transformer.encoder.layer.")
    blocks = list(blocks)
    if len(se_scales) != len(blocks):
        raise ValueError(
            "SE scales length mismatch for auxiliary loss: expected {} blocks, got {}.".format(
                len(blocks), len(se_scales)
            )
        )

    losses = []
    for block_idx, (block, gate) in enumerate(zip(blocks, se_scales)):
        hidden_size = int(getattr(block, "hidden_size", 0))
        if hidden_size <= 0:
            raise ValueError(f"Block {block_idx} does not expose a valid hidden_size.")

        gate_l = _normalize_se_gate_tensor(gate, expected_channels=hidden_size)

        with torch.no_grad():
            q_err = _input_channel_quant_error(
                _linear_weight_for_aux(block, "attn.query", block_idx),
                group_size=group_size,
                w_bit=w_bit,
            )
            k_err = _input_channel_quant_error(
                _linear_weight_for_aux(block, "attn.key", block_idx),
                group_size=group_size,
                w_bit=w_bit,
            )
            v_err = _input_channel_quant_error(
                _linear_weight_for_aux(block, "attn.value", block_idx),
                group_size=group_size,
                w_bit=w_bit,
            )
            fc1_err = _input_channel_quant_error(
                _linear_weight_for_aux(block, "ffn.fc1", block_idx),
                group_size=group_size,
                w_bit=w_bit,
            )
            for name, err in (
                ("attn.query", q_err),
                ("attn.key", k_err),
                ("attn.value", v_err),
                ("ffn.fc1", fc1_err),
            ):
                if err.numel() != hidden_size:
                    raise ValueError(
                        "Block {} {} input-channel error length {} does not match hidden_size {}.".format(
                            block_idx, name, err.numel(), hidden_size
                        )
                    )

            attn_err = (q_err + k_err + v_err) / 3.0
            combined_err = 0.5 * attn_err + 0.5 * fc1_err
            target = _normalize_sensitivity_target(combined_err)

        target = target.to(device=gate_l.device, dtype=gate_l.dtype).unsqueeze(0).expand_as(gate_l)
        losses.append(F.smooth_l1_loss(gate_l.float(), target.float()))

    if not losses:
        raise ValueError("No encoder blocks were available for SE auxiliary loss.")
    return torch.stack(losses).mean()


class SEViTSegQuantizer:
    """
    Official-AWQ-compatible W4A16 quantizer for ViT segmentation.

    The quantizer only converts Transformer encoder Linear layers
    (attn.query/key/value/out and ffn.fc1/fc2). LayerNorm, attention matmul,
    softmax, dropout, decoder layers, segmentation heads, SE MLPs, and embedding
    backbones remain unquantized. saliency_source='activation' provides vanilla
    mean-absolute activation saliency, while saliency_source='se_aux' weights
    that saliency by the trained calibration-only SE estimator. Both modes then
    share scale search, clipping, official WQLinear packing, and CUDA kernels.
    """

    target_linear_paths = (
        "attn.query",
        "attn.key",
        "attn.value",
        "attn.out",
        "ffn.fc1",
        "ffn.fc2",
    )
    clip_target_paths = (
        "attn.value",
        "attn.out",
        "ffn.fc1",
        "ffn.fc2",
    )

    def __init__(
        self,
        model,
        calib_loader,
        w_bit=4,
        q_group_size=128,
        n_calib_batches=8,
        device=None,
        args=None,
        saliency_source: str = "activation",
    ):
        if saliency_source not in ("activation", "se_aux"):
            raise ValueError(
                "saliency_source must be 'activation' or 'se_aux', got {!r}.".format(
                    saliency_source
                )
            )
        if int(w_bit) != 4:
            raise ValueError(f"The official custom_w4 backend requires w_bit=4, got {w_bit}.")
        if int(q_group_size) != 128:
            raise ValueError(
                "The official custom_w4 backend requires q_group_size=128, "
                f"got {q_group_size}."
            )

        official_awq = require_official_awq_runtime()
        self.model = model
        self.calib_loader = calib_loader
        self.w_bit = int(w_bit)
        self.q_group_size = int(q_group_size)
        self.q_config = {
            "zero_point": True,
            "q_group_size": self.q_group_size,
        }
        self.n_calib_batches = int(n_calib_batches)
        self.args = args
        self.saliency_source = saliency_source
        self.device = torch.device(device) if device is not None else self._model_device()
        if self.device.type != "cuda":
            raise RuntimeError(f"{_OFFICIAL_AWQ_ERROR} Details: selected device is {self.device}.")
        self._official_pseudo_quantize_tensor = official_awq["pseudo_quantize_tensor"]
        self._official_wqlinear = official_awq["WQLinear"]
        self._official_auto_clip_layer = official_awq["auto_clip_layer"]
        self._awq_module_path = official_awq["awq_module_path"]
        self._awq_inference_engine_path = official_awq["awq_inference_engine_path"]
        self.calib_chunk_size = min(max(1, int(getattr(args, "batch_size", 1))), 8)
        self.max_mse_samples = max(1, min(32, self.n_calib_batches * self.calib_chunk_size))
        self.max_clip_tokens = 512

        self.encoder = getattr(getattr(model, "transformer", None), "encoder", None)
        if self.encoder is None or not hasattr(self.encoder, "layer"):
            raise ValueError("Could not locate ViT encoder blocks at model.transformer.encoder.layer.")
        self.blocks = list(self.encoder.layer)
        if not self.blocks:
            raise ValueError("No Transformer encoder blocks found under model.transformer.encoder.layer.")

        self._validate_supported_blocks()
        self.total_encoder_linears = sum(
            1 for block in self.blocks for module in block.modules() if isinstance(module, nn.Linear)
        )
        logger.info("Official AWQ custom_w4 quantizer found %d encoder blocks.", len(self.blocks))
        logger.info(
            "Transformer encoder Linear layers found: %d; target layers per block: %s",
            self.total_encoder_linears,
            ", ".join(self.target_linear_paths),
        )

    def _model_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _validate_supported_blocks(self):
        for block_idx, block in enumerate(self.blocks):
            for rel_path in ("attention_norm", "ffn_norm"):
                module = _get_submodule(block, rel_path)
                if not isinstance(module, nn.LayerNorm):
                    raise TypeError(
                        f"Block {block_idx} {rel_path} must be nn.LayerNorm, got {type(module).__name__}."
                    )
            for rel_path in self.target_linear_paths:
                module = _get_submodule(block, rel_path)
                if not isinstance(module, nn.Linear):
                    raise TypeError(
                        f"Block {block_idx} {rel_path} must be nn.Linear, got {type(module).__name__}."
                    )
                in_features = int(module.in_features)
                out_features = int(module.out_features)
                if in_features % self.q_group_size != 0:
                    raise ValueError(
                        f"Block {block_idx} {rel_path} in_features={in_features} is not "
                        f"divisible by AWQ group_size={self.q_group_size}."
                    )
                if out_features % 8 != 0 or out_features % 4 != 0:
                    raise ValueError(
                        f"Block {block_idx} {rel_path} out_features={out_features} must "
                        "be divisible by both 8 and 4 for official AWQ WQLinear."
                    )

    def quantize(self):
        self.model.eval().to(self.device)
        encoder_args, had_drop_attr, old_drop = self._set_drop_se_block(False)
        try:
            logger.info("Calibration saliency source: %s", self.saliency_source)
            if self.saliency_source == "se_aux":
                self._validate_se_gates_available_and_mode_consistent()
            calib = self._collect_saliency_and_mse_inputs()
            scales_list = self._search_and_apply_scales(calib)
            clipped = self._search_and_apply_clipping(calib["input_feat"])
            quantized = self._convert_target_linears_to_wqlinear()
            self._enable_official_w4a16_runtime(quantized)
        finally:
            self._restore_drop_se_block(encoder_args, had_drop_attr, old_drop)
        logger.info("Official AWQ scale groups applied: %d", len(scales_list))
        logger.info("Official AWQ clipping targets applied: %d", clipped)
        logger.info(
            "Total Transformer encoder Linear layers: %d; official AWQ WQLinear modules: %d.",
            self.total_encoder_linears,
            quantized,
        )
        logger.info(
            "Official custom_w4 W4A16 quantization completed with saliency_source=%s.",
            self.saliency_source,
        )
        return self.model

    def _set_drop_se_block(self, value):
        encoder_args = getattr(self.encoder, "args", None)
        had_drop_attr = hasattr(encoder_args, "drop_se_block") if encoder_args is not None else False
        old_drop = getattr(encoder_args, "drop_se_block", False) if encoder_args is not None else False
        if encoder_args is not None:
            setattr(encoder_args, "drop_se_block", bool(value))
        return encoder_args, had_drop_attr, old_drop

    def _restore_drop_se_block(self, encoder_args, had_drop_attr, old_drop):
        if encoder_args is None:
            return
        if had_drop_attr:
            setattr(encoder_args, "drop_se_block", old_drop)
        else:
            delattr(encoder_args, "drop_se_block")

    def _first_calib_tensor(self):
        for sampled_batch in self.calib_loader:
            x = image_to_nchw_for_quant_calib(sampled_batch["image"], self.args)
            if x.numel() == 0:
                continue
            return x[: self.calib_chunk_size].to(self.device, non_blocking=True)
        raise RuntimeError("No calibration inputs were available for official AWQ quantization.")

    def _validate_se_gates_available_and_mode_consistent(self):
        se_layers = getattr(self.encoder, "SELayer", None)
        se_found = se_layers is not None and len(se_layers) == len(self.blocks)
        se_calib_only = bool(getattr(self.args, "se_calib_only", False))
        drop_se_block = bool(getattr(getattr(self.encoder, "args", None), "drop_se_block", False))
        tolerance = float(getattr(self.args, "se_logit_diff_tolerance", 1e-6))
        logger.info(
            "SE mode diagnostics: se_calib_only=%s drop_se_block=%s se_gates_found=%s.",
            se_calib_only,
            drop_se_block,
            bool(se_found),
        )
        logger.info(
            "Post-quantization SE drop expected: %s.",
            bool(se_calib_only),
        )
        if not se_calib_only:
            raise RuntimeError(
                "saliency_source='se_aux' requires calibration-only SE behavior; "
                "set --se_calib_only."
            )
        if not se_found:
            raise RuntimeError(
                "SE-guided quantization requires one SELayer per encoder block, but the "
                "model does not expose them at model.transformer.encoder.SELayer."
            )

        x = self._first_calib_tensor()
        encoder_args, had_drop_attr, old_drop = self._set_drop_se_block(False)
        try:
            with torch.no_grad():
                out_with_se = self.model(x)
                logits_with_se, se_scales = _extract_logits_and_se_scales(out_with_se, len(self.blocks))

                self._set_drop_se_block(True)
                out_without_se = self.model(x)
                logits_without_se = _first_tensor_from_output(out_without_se)
        finally:
            self._restore_drop_se_block(encoder_args, had_drop_attr, old_drop)

        for block_idx, gate in enumerate(se_scales):
            _normalize_se_gate_tensor(gate, expected_channels=self.blocks[block_idx].hidden_size)

        if logits_with_se.shape != logits_without_se.shape:
            raise RuntimeError(
                "SE mode consistency check failed because logits shape changed from {} to {}.".format(
                    tuple(logits_with_se.shape), tuple(logits_without_se.shape)
                )
            )
        max_diff = (logits_with_se.detach() - logits_without_se.detach()).abs().max().item()
        active = max_diff > tolerance
        logger.info(
            "SE forward path appears active: %s (max logit diff with drop_se_block toggle: %.8g; tolerance: %.8g).",
            bool(active),
            max_diff,
            tolerance,
        )
        if active:
            raise RuntimeError(
                "Expected calibration-only SE auxiliary block, but SE changes logits. "
                "max logit diff with drop_se_block toggle: {:.8g}".format(max_diff)
            )

    def _collect_saliency_and_mse_inputs(self):
        captured: Dict[Tuple[int, str], torch.Tensor] = {}
        hooks = []

        def make_hook(block_idx, module_path):
            def hook(_module, inputs):
                if not inputs:
                    raise RuntimeError(
                        f"Missing input for block {block_idx} {module_path} hook."
                    )
                captured[(block_idx, module_path)] = inputs[0].detach()

            return hook

        for block_idx, block in enumerate(self.blocks):
            for module_path in self.target_linear_paths:
                module = _get_submodule(block, module_path)
                hooks.append(
                    module.register_forward_pre_hook(make_hook(block_idx, module_path))
                )

        attn_sums: List[Optional[torch.Tensor]] = [None] * len(self.blocks)
        ffn_sums: List[Optional[torch.Tensor]] = [None] * len(self.blocks)
        attn_counts = [0] * len(self.blocks)
        ffn_counts = [0] * len(self.blocks)
        attn_mse_inputs: List[List[torch.Tensor]] = [[] for _ in self.blocks]
        ffn_mse_inputs: List[List[torch.Tensor]] = [[] for _ in self.blocks]
        attn_mse_counts = [0] * len(self.blocks)
        ffn_mse_counts = [0] * len(self.blocks)
        feature_cache_names = ("attn.qkv", "attn.out", "ffn.fc1", "ffn.fc2")
        feature_caches = [
            {name: [] for name in feature_cache_names} for _ in self.blocks
        ]
        feature_counts = [
            {name: 0 for name in feature_cache_names} for _ in self.blocks
        ]
        total_images = 0
        processed_batches = 0

        try:
            with torch.no_grad():
                for batch_idx, sampled_batch in enumerate(self.calib_loader):
                    if batch_idx >= self.n_calib_batches:
                        break
                    processed_batches += 1
                    x_all = image_to_nchw_for_quant_calib(sampled_batch["image"], self.args)
                    for chunk in x_all.split(self.calib_chunk_size, dim=0):
                        captured.clear()
                        chunk = chunk.to(self.device, non_blocking=True)
                        output = self.model(chunk)
                        if self.saliency_source == "se_aux":
                            _logits, se_scales = _extract_logits_and_se_scales(
                                output,
                                len(self.blocks),
                            )
                        else:
                            se_scales = None
                        total_images += int(chunk.shape[0])

                        for block_idx, block in enumerate(self.blocks):
                            block_inputs = {}
                            for module_path in self.target_linear_paths:
                                module_input = captured.get((block_idx, module_path))
                                if module_input is None:
                                    raise RuntimeError(
                                        "Calibration hook did not capture block {} {} input.".format(
                                            block_idx, module_path
                                        )
                                    )
                                expected_channels = _get_submodule(
                                    block, module_path
                                ).in_features
                                if module_input.shape[-1] != expected_channels:
                                    raise ValueError(
                                        "Block {} {} input has {} channels; expected {}.".format(
                                            block_idx,
                                            module_path,
                                            module_input.shape[-1],
                                            expected_channels,
                                        )
                                    )
                                block_inputs[module_path] = module_input

                            attn_in = block_inputs["attn.query"]
                            ffn_in = block_inputs["ffn.fc1"]
                            for shared_path in ("attn.key", "attn.value"):
                                if block_inputs[shared_path].shape != attn_in.shape:
                                    raise ValueError(
                                        "Block {} {} input shape {} does not match shared q/k/v shape {}.".format(
                                            block_idx,
                                            shared_path,
                                            tuple(block_inputs[shared_path].shape),
                                            tuple(attn_in.shape),
                                        )
                                    )

                            gate = None
                            if self.saliency_source == "se_aux":
                                gate = _normalize_se_gate_tensor(
                                    se_scales[block_idx],
                                    expected_channels=block.hidden_size,
                                ).to(device=attn_in.device)

                            attn_sum, attn_count = self._saliency_sum_and_count(
                                attn_in,
                                gate,
                                block_idx,
                                "attention",
                            )
                            ffn_sum, ffn_count = self._saliency_sum_and_count(
                                ffn_in,
                                gate,
                                block_idx,
                                "ffn",
                            )
                            attn_sums[block_idx] = attn_sum if attn_sums[block_idx] is None else attn_sums[block_idx] + attn_sum
                            ffn_sums[block_idx] = ffn_sum if ffn_sums[block_idx] is None else ffn_sums[block_idx] + ffn_sum
                            attn_counts[block_idx] += attn_count
                            ffn_counts[block_idx] += ffn_count

                            attn_mse_counts[block_idx] = self._cache_mse_input(
                                attn_mse_inputs[block_idx],
                                attn_in,
                                attn_mse_counts[block_idx],
                            )
                            ffn_mse_counts[block_idx] = self._cache_mse_input(
                                ffn_mse_inputs[block_idx],
                                ffn_in,
                                ffn_mse_counts[block_idx],
                            )

                            feature_sources = {
                                "attn.qkv": attn_in,
                                "attn.out": block_inputs["attn.out"],
                                "ffn.fc1": ffn_in,
                                "ffn.fc2": block_inputs["ffn.fc2"],
                            }
                            for cache_name, feature in feature_sources.items():
                                feature_counts[block_idx][cache_name] = self._cache_clip_tokens(
                                    feature_caches[block_idx][cache_name],
                                    feature,
                                    feature_counts[block_idx][cache_name],
                                )
        finally:
            for hook in hooks:
                hook.remove()

        if total_images <= 0:
            raise RuntimeError("No calibration images/slices were processed.")

        saliency_attn = []
        saliency_ffn = []
        input_feat = []
        for block_idx, block in enumerate(self.blocks):
            if (
                attn_counts[block_idx] <= 0
                or ffn_counts[block_idx] <= 0
                or attn_sums[block_idx] is None
                or ffn_sums[block_idx] is None
            ):
                raise RuntimeError(f"No calibration saliency accumulated for encoder block {block_idx}.")
            attn_sal = (attn_sums[block_idx] / float(attn_counts[block_idx])).float()
            ffn_sal = (ffn_sums[block_idx] / float(ffn_counts[block_idx])).float()
            self._validate_saliency(attn_sal, block.hidden_size, block_idx, "attention")
            self._validate_saliency(ffn_sal, block.hidden_size, block_idx, "ffn")
            saliency_attn.append(attn_sal)
            saliency_ffn.append(ffn_sal)
            self._log_saliency_stats(block_idx, "attention", attn_sal)
            self._log_saliency_stats(block_idx, "ffn", ffn_sal)

            cached = {}
            for cache_name in feature_cache_names:
                tensors = feature_caches[block_idx][cache_name]
                if not tensors:
                    raise RuntimeError(
                        f"No clipping inputs cached for block {block_idx} {cache_name}."
                    )
                cached[cache_name] = torch.cat(tensors, dim=0).contiguous()
            qkv_input = cached["attn.qkv"]
            input_feat.append(
                {
                    "attn.query": qkv_input,
                    "attn.key": qkv_input,
                    "attn.value": qkv_input,
                    "attn.out": cached["attn.out"],
                    "ffn.fc1": cached["ffn.fc1"],
                    "ffn.fc2": cached["ffn.fc2"],
                }
            )

        logger.info(
            "Calibration images/slices used: %d from %d dataloader batches.",
            total_images,
            processed_batches,
        )

        return {
            "saliency_attn": saliency_attn,
            "saliency_ffn": saliency_ffn,
            "attn_mse_inputs": attn_mse_inputs,
            "ffn_mse_inputs": ffn_mse_inputs,
            "input_feat": input_feat,
            "total_images": total_images,
            "processed_batches": processed_batches,
        }

    def _saliency_sum_and_count(self, activation, gate, block_idx, kind):
        if gate is None:
            flattened = activation.detach().float().reshape(-1, activation.shape[-1])
            return flattened.abs().sum(dim=0).cpu().double(), int(flattened.shape[0])

        per_sample = _activation_channel_mean(activation)
        if gate.shape != per_sample.shape:
            raise ValueError(
                "Block {} gate shape {} does not match {} activation shape {}.".format(
                    block_idx,
                    tuple(gate.shape),
                    kind,
                    tuple(per_sample.shape),
                )
            )
        weighted = per_sample * gate
        return weighted.sum(dim=0).detach().cpu().double(), int(gate.shape[0])

    def _cache_mse_input(self, cache, tensor, cached_count):
        remaining = self.max_mse_samples - int(cached_count)
        if remaining <= 0:
            return int(cached_count)
        take = min(remaining, int(tensor.shape[0]))
        cache.append(tensor[:take].detach().cpu().float().contiguous())
        return int(cached_count) + take

    def _cache_clip_tokens(self, cache, tensor, cached_count):
        remaining = self.max_clip_tokens - int(cached_count)
        if remaining <= 0:
            return int(cached_count)
        flattened = tensor.detach().reshape(-1, tensor.shape[-1])
        take = min(remaining, int(flattened.shape[0]))
        cache.append(flattened[:take].cpu().float().contiguous())
        return int(cached_count) + take

    def _validate_saliency(self, saliency, hidden_size, block_idx, kind):
        if saliency.numel() != int(hidden_size):
            raise ValueError(
                "Block {} {} saliency length {} does not match hidden size {}.".format(
                    block_idx, kind, saliency.numel(), hidden_size
                )
            )
        if not torch.isfinite(saliency).all():
            raise ValueError(f"Block {block_idx} {kind} saliency contains NaN or Inf.")

    def _log_saliency_stats(self, block_idx, kind, saliency):
        saliency = saliency.float()
        logger.info(
            "Block %d %s saliency stats: mean=%.8g min=%.8g max=%.8g std=%.8g",
            block_idx,
            kind,
            saliency.mean().item(),
            saliency.min().item(),
            saliency.max().item(),
            saliency.std(unbiased=False).item(),
        )

    def _search_and_apply_scales(self, calib):
        scales_list = []
        for block_idx, block in enumerate(self.blocks):
            attn_x = self._mse_inputs_to_device(calib["attn_mse_inputs"][block_idx], block_idx, "attention")
            ffn_x = self._mse_inputs_to_device(calib["ffn_mse_inputs"][block_idx], block_idx, "ffn")

            attn_scales, attn_ratio, attn_mse = self._search_best_scales_from_saliency(
                module2inspect=block.attn,
                linears2scale=[block.attn.query, block.attn.key, block.attn.value],
                x_for_mse=attn_x,
                saliency=calib["saliency_attn"][block_idx],
            )
            ffn_scales, ffn_ratio, ffn_mse = self._search_best_scales_from_saliency(
                module2inspect=block.ffn,
                linears2scale=[block.ffn.fc1],
                x_for_mse=ffn_x,
                saliency=calib["saliency_ffn"][block_idx],
            )

            logger.info(
                "Block %d selected alpha ratio: attention=%.4f mse=%.8g; ffn=%.4f mse=%.8g",
                block_idx,
                attn_ratio,
                attn_mse,
                ffn_ratio,
                ffn_mse,
            )

            self._apply_scale_to_norm_and_linears(
                block,
                "attention_norm",
                ("attn.query", "attn.key", "attn.value"),
                attn_scales,
                calib["input_feat"][block_idx],
            )
            self._apply_scale_to_norm_and_linears(
                block,
                "ffn_norm",
                ("ffn.fc1",),
                ffn_scales,
                calib["input_feat"][block_idx],
            )
            scales_list.extend(
                [
                    ("attention_norm", ("attn.query", "attn.key", "attn.value"), attn_scales),
                    ("ffn_norm", ("ffn.fc1",), ffn_scales),
                ]
            )
        return scales_list

    def _mse_inputs_to_device(self, cache, block_idx, kind):
        if not cache:
            raise RuntimeError(f"No cached {kind} MSE inputs for encoder block {block_idx}.")
        return torch.cat(cache, dim=0).to(self.device, non_blocking=True)

    def _search_best_scales_from_saliency(
        self,
        module2inspect,
        linears2scale,
        x_for_mse,
        saliency,
        n_grid=20,
    ):
        """TransUNet adapter of llm-awq auto_scale._search_module_scale."""
        saliency = saliency.detach().to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            fp_output = _module_first_tensor(module2inspect(x_for_mse))
            fp_output = fp_output.detach().float()

        original_weights = [linear.weight.detach().clone() for linear in linears2scale]
        best_mse = None
        best_ratio = None
        best_scales = None

        try:
            for grid_idx in range(int(n_grid)):
                ratio = float(grid_idx) / float(n_grid)
                scales = saliency.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / torch.sqrt(scales.max() * scales.min())
                scales = scales.to(dtype=linears2scale[0].weight.dtype)

                with torch.no_grad():
                    for linear, original in zip(linears2scale, original_weights):
                        if linear.in_features != scales.numel():
                            raise ValueError(
                                "Scale length {} does not match Linear in_features {}.".format(
                                    scales.numel(), linear.in_features
                                )
                            )
                        linear_scales = scales.view(1, -1).to(linear.weight.device)
                        scaled_weight = original.to(linear.weight.device) * linear_scales
                        quantized = self._pseudo_quantize_weight(scaled_weight)
                        linear.weight.copy_(quantized / linear_scales)

                    q_output = _module_first_tensor(module2inspect(x_for_mse)).detach().float()
                    mse = F.mse_loss(q_output, fp_output).item()

                    for linear, original in zip(linears2scale, original_weights):
                        linear.weight.copy_(original.to(linear.weight.device))

                if best_mse is None or mse < best_mse:
                    best_mse = float(mse)
                    best_ratio = float(ratio)
                    best_scales = scales.detach().clone()
        finally:
            with torch.no_grad():
                for linear, original in zip(linears2scale, original_weights):
                    linear.weight.copy_(original.to(device=linear.weight.device))

        if best_scales is None or best_ratio is None or best_mse is None:
            raise RuntimeError("AWQ-style alpha search failed to evaluate any candidate scales.")
        if not torch.isfinite(best_scales).all():
            raise RuntimeError("Selected AWQ scales contain NaN or Inf.")
        return best_scales, best_ratio, best_mse

    def _pseudo_quantize_weight(self, weight):
        if weight.dim() != 2:
            raise ValueError(f"Expected 2D Linear weight, got {tuple(weight.shape)}.")
        if weight.shape[-1] % self.q_group_size != 0:
            raise ValueError(
                "Linear in_features={} is not divisible by group_size={}.".format(
                    weight.shape[-1], self.q_group_size
                )
            )
        return self._official_pseudo_quantize_tensor(
            weight,
            n_bit=self.w_bit,
            **self.q_config,
        )

    def _apply_scale_to_norm_and_linears(
        self,
        block,
        norm_name,
        linear_names,
        scales,
        input_feat,
    ):
        norm = _get_submodule(block, norm_name)
        norm_scales = scales.detach().to(device=norm.weight.device, dtype=norm.weight.dtype)
        with torch.no_grad():
            norm.weight.div_(norm_scales)
            if norm.bias is not None:
                norm.bias.div_(norm_scales)
            for linear_name in linear_names:
                linear = _get_submodule(block, linear_name)
                linear_scales = scales.to(device=linear.weight.device, dtype=linear.weight.dtype)
                linear.weight.mul_(linear_scales.view(1, -1))

            seen_features = set()
            for linear_name in linear_names:
                feature = input_feat[linear_name]
                if id(feature) in seen_features:
                    continue
                seen_features.add(id(feature))
                feature.div_(scales.view(1, -1).to(feature.device, feature.dtype))

    def _search_and_apply_clipping(self, input_feat):
        clipped = 0
        for block_idx, block in enumerate(self.blocks):
            block_input_feat = input_feat[block_idx]
            for rel_path in self.clip_target_paths:
                linear = _get_submodule(block, rel_path)
                if not isinstance(linear, nn.Linear):
                    raise TypeError(
                        f"Block {block_idx} clipping target {rel_path} must be nn.Linear, "
                        f"got {type(linear).__name__}."
                    )
                if linear.out_features % 256 != 0 and linear.out_features % 64 != 0:
                    raise ValueError(
                        f"Block {block_idx} {rel_path} out_features={linear.out_features} "
                        "is incompatible with official auto_clip_layer output-channel batching."
                    )
                feature = block_input_feat[rel_path]
                if feature.numel() == 0:
                    raise RuntimeError(
                        f"No clipping input features for block {block_idx} {rel_path}."
                    )
                n_sample_token = min(self.max_clip_tokens, int(feature.shape[0]))
                max_val = self._official_auto_clip_layer(
                    linear.weight,
                    feature,
                    n_bit=self.w_bit,
                    q_config=self.q_config,
                    n_grid=20,
                    max_shrink=0.5,
                    n_sample_token=n_sample_token,
                )
                max_val = max_val.to(linear.weight.device, linear.weight.dtype)
                logger.info(
                    "Block %d clipping target %s selected max-value stats: "
                    "mean=%.8g min=%.8g max=%.8g",
                    block_idx,
                    rel_path,
                    max_val.float().mean().item(),
                    max_val.float().min().item(),
                    max_val.float().max().item(),
                )
                with torch.no_grad():
                    original_shape = linear.weight.shape
                    grouped_weight = linear.weight.data.reshape(*max_val.shape[:2], -1)
                    linear.weight.data = torch.clamp(
                        grouped_weight,
                        -max_val,
                        max_val,
                    ).reshape(original_shape)
                clipped += 1
        return clipped

    def _convert_target_linears_to_wqlinear(self):
        quantized = 0
        for block_idx, block in enumerate(self.blocks):
            for rel_path in self.target_linear_paths:
                module = _get_submodule(block, rel_path)
                if not isinstance(module, nn.Linear):
                    raise TypeError(
                        f"Expected block {block_idx} {rel_path} to be nn.Linear before quantization, "
                        f"got {type(module).__name__}."
                    )
                module.weight.data = module.weight.data.to(torch.float16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.float16)
                module.weight.data, scales, zeros = self._official_pseudo_quantize_tensor(
                    module.weight.data,
                    n_bit=self.w_bit,
                    get_scale_zp=True,
                    **self.q_config,
                )
                qmodule = self._official_wqlinear.from_linear(
                    module,
                    w_bit=self.w_bit,
                    group_size=self.q_group_size,
                    init_only=False,
                    scales=scales,
                    zeros=zeros,
                ).to(device=module.weight.device)
                _set_submodule(block, rel_path, qmodule)
                quantized += 1
                logger.info(
                    "Quantized encoder block %d %s to official AWQ WQLinear.",
                    block_idx,
                    rel_path,
                )

        expected = len(self.blocks) * len(self.target_linear_paths)
        if quantized != expected:
            raise RuntimeError(
                f"Expected {expected} official AWQ WQLinear modules, converted {quantized}."
            )
        return quantized

    def _enable_official_w4a16_runtime(self, quantized):
        self.encoder.half()
        self.encoder.awq_runtime_dtype = torch.float16
        self.encoder.awq_quant_backend = "custom_w4"
        self.encoder.awq_saliency_source = self.saliency_source
        self.encoder.awq_quantized_module_class = "awq.quantize.qmodule.WQLinear"
        self.encoder.awq_package_module_path = self._awq_module_path
        self.encoder.awq_inference_engine_path = self._awq_inference_engine_path
        self.encoder.awq_inference_engine_active = True
        self.encoder.awq_wqlinear_module_count = int(quantized)
        self.encoder.awq_zero_point = True
        self.encoder.awq_group_size = self.q_group_size
        self.encoder.awq_auto_clip = True

        actual = sum(
            1 for module in self.encoder.modules() if isinstance(module, self._official_wqlinear)
        )
        legacy = sum(
            1 for module in self.encoder.modules() if isinstance(module, W4GroupedLinear)
        )
        if actual != int(quantized):
            raise RuntimeError(
                f"Expected {quantized} official AWQ WQLinear modules after conversion, found {actual}."
            )
        if legacy:
            raise RuntimeError(
                "Official custom_w4 deployment unexpectedly contains W4GroupedLinear; "
                "the eager-dequantization fallback is forbidden."
            )
