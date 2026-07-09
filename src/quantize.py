from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


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
    """Packed W4 grouped weight-only Linear with floating point activations."""

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


class SEViTSegQuantizer:
    """
    SE-guided activation-aware W4 grouped weight-only quantizer for ViT segmentation.

    The quantizer only converts Transformer encoder Linear layers
    (attn.query/key/value/out and ffn.fc1/fc2). LayerNorm, attention matmul,
    softmax, dropout, decoder layers, segmentation heads, SE MLPs, and embedding
    backbones remain floating point. SE-guided saliency is the calibration-set
    average of activation magnitude multiplied by the trained SE gate:
    mean_x(g_l,c(x) * mean_tokens(abs(activation_l,c(x)))).
    """

    target_linear_paths = (
        "attn.query",
        "attn.key",
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
    ):
        self.model = model
        self.calib_loader = calib_loader
        self.w_bit = int(w_bit)
        self.q_group_size = int(q_group_size)
        self.n_calib_batches = int(n_calib_batches)
        self.args = args
        self.device = torch.device(device) if device is not None else self._model_device()
        self.calib_chunk_size = min(max(1, int(getattr(args, "batch_size", 1))), 8)
        self.max_mse_samples = max(1, min(32, self.n_calib_batches * self.calib_chunk_size))

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
        logger.info("SE-guided quantizer found %d encoder blocks.", len(self.blocks))
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
                _validate_w4_grouped_linear(
                    module,
                    group_size=self.q_group_size,
                    w_bit=self.w_bit,
                )

    def quantize(self):
        self.model.eval().to(self.device)
        encoder_args, had_drop_attr, old_drop = self._set_drop_se_block(False)
        try:
            self._validate_se_gates_available_and_active()
            calib = self._collect_saliency_and_mse_inputs()
            scales_list = self._search_and_apply_scales(calib)
            quantized = self._convert_target_linears_to_w4()
        finally:
            self._restore_drop_se_block(encoder_args, had_drop_attr, old_drop)
        logger.info("SE-guided scale groups applied: %d", len(scales_list))
        logger.info(
            "Total Transformer encoder Linear layers: %d; W4 grouped linears quantized: %d.",
            self.total_encoder_linears,
            quantized,
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
        raise RuntimeError("No calibration inputs were available for SE-guided quantization.")

    def _validate_se_gates_available_and_active(self):
        se_layers = getattr(self.encoder, "SELayer", None)
        se_found = se_layers is not None and len(se_layers) == len(self.blocks)
        logger.info("SE gates found: %s.", bool(se_found))
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
                "SE active-path check failed because logits shape changed from {} to {}.".format(
                    tuple(logits_with_se.shape), tuple(logits_without_se.shape)
                )
            )
        max_diff = (logits_with_se.detach() - logits_without_se.detach()).abs().max().item()
        active = max_diff > 1e-6
        logger.info(
            "SE forward path appears active: %s (max logit diff with drop_se_block toggle: %.8g).",
            bool(active),
            max_diff,
        )
        if not active:
            raise RuntimeError(
                "SE gates are returned by the model, but they do not affect segmentation logits. "
                "In networks/vit_seg_modeling.py Encoder.forward currently computes "
                "h_se, scale = se_layer(hidden_states) without assigning hidden_states = h_se. "
                "SE-guided quantization requires trained SE gates on the active segmentation "
                "forward path; retrain or load a checkpoint from an active-SE model before using "
                "--use_se_block --quantize."
            )

    def _collect_saliency_and_mse_inputs(self):
        captured: Dict[Tuple[int, str], torch.Tensor] = {}
        hooks = []

        def make_hook(block_idx, kind):
            def hook(_module, inputs):
                if not inputs:
                    raise RuntimeError(f"Missing input for block {block_idx} {kind} hook.")
                captured[(block_idx, kind)] = inputs[0].detach()

            return hook

        for block_idx, block in enumerate(self.blocks):
            hooks.append(block.attn.query.register_forward_pre_hook(make_hook(block_idx, "attn")))
            hooks.append(block.ffn.fc1.register_forward_pre_hook(make_hook(block_idx, "ffn")))

        attn_sums: List[Optional[torch.Tensor]] = [None] * len(self.blocks)
        ffn_sums: List[Optional[torch.Tensor]] = [None] * len(self.blocks)
        counts = [0] * len(self.blocks)
        attn_mse_inputs: List[List[torch.Tensor]] = [[] for _ in self.blocks]
        ffn_mse_inputs: List[List[torch.Tensor]] = [[] for _ in self.blocks]
        attn_mse_counts = [0] * len(self.blocks)
        ffn_mse_counts = [0] * len(self.blocks)
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
                        _logits, se_scales = _extract_logits_and_se_scales(output, len(self.blocks))
                        total_images += int(chunk.shape[0])

                        for block_idx, block in enumerate(self.blocks):
                            attn_in = captured.get((block_idx, "attn"))
                            ffn_in = captured.get((block_idx, "ffn"))
                            if attn_in is None or ffn_in is None:
                                raise RuntimeError(
                                    f"Calibration hooks did not capture both attn and ffn inputs for block {block_idx}."
                                )

                            gate = _normalize_se_gate_tensor(
                                se_scales[block_idx],
                                expected_channels=block.hidden_size,
                            ).to(device=attn_in.device)

                            attn_act = _activation_channel_mean(attn_in)
                            ffn_act = _activation_channel_mean(ffn_in)
                            if gate.shape != attn_act.shape:
                                raise ValueError(
                                    "Block {} gate shape {} does not match attn activation shape {}.".format(
                                        block_idx, tuple(gate.shape), tuple(attn_act.shape)
                                    )
                                )
                            if gate.shape != ffn_act.shape:
                                raise ValueError(
                                    "Block {} gate shape {} does not match ffn activation shape {}.".format(
                                        block_idx, tuple(gate.shape), tuple(ffn_act.shape)
                                    )
                                )

                            attn_weighted = attn_act * gate
                            ffn_weighted = ffn_act * gate

                            attn_sum = attn_weighted.sum(dim=0).detach().cpu().double()
                            ffn_sum = ffn_weighted.sum(dim=0).detach().cpu().double()
                            attn_sums[block_idx] = attn_sum if attn_sums[block_idx] is None else attn_sums[block_idx] + attn_sum
                            ffn_sums[block_idx] = ffn_sum if ffn_sums[block_idx] is None else ffn_sums[block_idx] + ffn_sum
                            counts[block_idx] += int(gate.shape[0])

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
        finally:
            for hook in hooks:
                hook.remove()

        if total_images <= 0:
            raise RuntimeError("No calibration images/slices were processed.")

        saliency_attn = []
        saliency_ffn = []
        for block_idx, block in enumerate(self.blocks):
            if counts[block_idx] <= 0 or attn_sums[block_idx] is None or ffn_sums[block_idx] is None:
                raise RuntimeError(f"No calibration saliency accumulated for encoder block {block_idx}.")
            attn_sal = (attn_sums[block_idx] / float(counts[block_idx])).float()
            ffn_sal = (ffn_sums[block_idx] / float(counts[block_idx])).float()
            self._validate_saliency(attn_sal, block.hidden_size, block_idx, "attention")
            self._validate_saliency(ffn_sal, block.hidden_size, block_idx, "ffn")
            saliency_attn.append(attn_sal)
            saliency_ffn.append(ffn_sal)
            self._log_saliency_stats(block_idx, "attention", attn_sal)
            self._log_saliency_stats(block_idx, "ffn", ffn_sal)

        logger.info(
            "SE-guided calibration images/slices used: %d from %d dataloader batches.",
            total_images,
            processed_batches,
        )

        return {
            "saliency_attn": saliency_attn,
            "saliency_ffn": saliency_ffn,
            "attn_mse_inputs": attn_mse_inputs,
            "ffn_mse_inputs": ffn_mse_inputs,
            "total_images": total_images,
            "processed_batches": processed_batches,
        }

    def _cache_mse_input(self, cache, tensor, cached_count):
        remaining = self.max_mse_samples - int(cached_count)
        if remaining <= 0:
            return int(cached_count)
        take = min(remaining, int(tensor.shape[0]))
        cache.append(tensor[:take].detach().cpu().float())
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
                w_quantize_func=self._pseudo_quantize_weight,
            )
            ffn_scales, ffn_ratio, ffn_mse = self._search_best_scales_from_saliency(
                module2inspect=block.ffn,
                linears2scale=[block.ffn.fc1],
                x_for_mse=ffn_x,
                saliency=calib["saliency_ffn"][block_idx],
                w_quantize_func=self._pseudo_quantize_weight,
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
            )
            self._apply_scale_to_norm_and_linears(
                block,
                "ffn_norm",
                ("ffn.fc1",),
                ffn_scales,
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
        w_quantize_func,
        kwargs=None,
        n_grid=20,
    ):
        kwargs = kwargs or {}
        saliency = saliency.detach().to(device=self.device, dtype=torch.float32)
        base = saliency.clamp(min=1e-4)

        with torch.no_grad():
            fp_output = _module_first_tensor(module2inspect(x_for_mse))
            fp_output = fp_output.detach().float()

        original_weights = [linear.weight.detach().clone() for linear in linears2scale]
        best_mse = None
        best_ratio = None
        best_scales = None

        try:
            for grid_idx in range(int(n_grid)):
                ratio = 0.0 if n_grid <= 1 else float(grid_idx) / float(n_grid - 1)
                scales = base.pow(ratio)
                scales = scales / torch.sqrt(scales.max() * scales.min()).clamp(min=1e-8)
                scales = scales.to(dtype=linears2scale[0].weight.dtype)

                with torch.no_grad():
                    for linear, original in zip(linears2scale, original_weights):
                        if linear.in_features != scales.numel():
                            raise ValueError(
                                "Scale length {} does not match Linear in_features {}.".format(
                                    scales.numel(), linear.in_features
                                )
                            )
                        scaled_weight = original.to(device=linear.weight.device) * scales.view(1, -1).to(linear.weight.device)
                        quantized = w_quantize_func(scaled_weight, **kwargs)
                        linear.weight.copy_(quantized / scales.view(1, -1).to(linear.weight.device))

                    q_output = _module_first_tensor(module2inspect(x_for_mse)).detach().float()
                    mse = F.mse_loss(q_output, fp_output).item()

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

    def _pseudo_quantize_weight(self, weight, group_size=None):
        group_size = self.q_group_size if group_size is None else int(group_size)
        if weight.dim() != 2:
            raise ValueError(f"Expected 2D Linear weight, got {tuple(weight.shape)}.")
        out_features, in_features = weight.shape
        if in_features % group_size != 0:
            raise ValueError(
                "Linear in_features={} is not divisible by group_size={}.".format(
                    in_features, group_size
                )
            )
        qmax = (1 << (self.w_bit - 1)) - 1
        grouped = weight.float().reshape(out_features, -1, group_size)
        scales = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / float(qmax)
        quant = torch.round(grouped / scales).clamp(-qmax, qmax)
        return (quant * scales).reshape_as(weight).to(dtype=weight.dtype)

    def _apply_scale_to_norm_and_linears(self, block, norm_name, linear_names, scales):
        norm = _get_submodule(block, norm_name)
        scales = scales.detach().to(device=norm.weight.device, dtype=norm.weight.dtype)
        with torch.no_grad():
            norm.weight.div_(scales)
            if norm.bias is not None:
                norm.bias.div_(scales)
            for linear_name in linear_names:
                linear = _get_submodule(block, linear_name)
                linear_scales = scales.to(device=linear.weight.device, dtype=linear.weight.dtype)
                linear.weight.mul_(linear_scales.view(1, -1))

    def _convert_target_linears_to_w4(self):
        quantized = 0
        for block_idx, block in enumerate(self.blocks):
            for rel_path in self.target_linear_paths:
                module = _get_submodule(block, rel_path)
                if not isinstance(module, nn.Linear):
                    raise TypeError(
                        f"Expected block {block_idx} {rel_path} to be nn.Linear before quantization, "
                        f"got {type(module).__name__}."
                    )
                qmodule = W4GroupedLinear.from_linear(
                    module,
                    group_size=self.q_group_size,
                    w_bit=self.w_bit,
                ).to(device=module.weight.device)
                _set_submodule(block, rel_path, qmodule)
                quantized += 1
                logger.info("Quantized encoder block %d %s to W4 grouped weight-only.", block_idx, rel_path)
        return quantized
