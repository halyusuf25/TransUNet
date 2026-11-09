"""Floating-point baseline for the default packed path's encoder projections."""

import logging

import torch
from torch import nn


class FP16Projection(nn.Module):
    """Use FP16 linear storage/compute and let the existing residual restore FP32."""

    def __init__(self, linear):
        super().__init__()
        # Replace the original storage; retain no master weights or scales.
        self.linear = linear.to(dtype=torch.float16)
        self.train(linear.training)

    def forward(self, x):
        weight = self.linear.weight
        # Follow the weight dtype so an isolated CPU profiling copy can use FP32.
        x = x.to(device=weight.device, dtype=weight.dtype).contiguous()
        return self.linear(x)


def convert_fp16_projections(model):
    """Convert only targets eligible for legacy W4/group-128 packed deployment.

    Mirror AWQViTSegQuantizer._attention_layout and
    _convert_selected_linears_in_block without importing its optional backend.
    No calibration, rescaling, quantization, or changes outside these linears.
    """
    encoder = getattr(getattr(model, "transformer", None), "encoder", None)
    blocks = getattr(encoder, "layer", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError("FP16 projections require a nonempty transformer.encoder.layer ModuleList")

    selected, skipped = [], {}
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        projections = ("qkv", "proj") if (
            hasattr(attention, "qkv") and hasattr(attention, "proj")
        ) else ("query", "key", "value", "out")
        targets = [f"attn.{name}" for name in projections] + ["ffn.fc1", "ffn.fc2"]
        for target in targets:
            name = f"transformer.encoder.layer.{index}.{target}"
            parent, leaf = target.split(".")
            module = getattr(getattr(block, parent, None), leaf, None)
            if not isinstance(module, nn.Linear):
                skipped[name] = "missing or not nn.Linear"
            elif module.in_features % 128:
                skipped[name] = "input width is not divisible by 128 (packed group size)"
            elif module.out_features % 8:
                skipped[name] = "output width is not divisible by 8 (packed INT4 eligibility)"
            else:
                selected.append(name)

    for name, reason in skipped.items():
        logging.warning("FP16 projections skipped %s: %s", name, reason)
    if not selected:
        raise ValueError("No encoder projections are eligible for the FP16 projections baseline")
    for index, block in enumerate(blocks):
        prefix = f"transformer.encoder.layer.{index}.ffn."
        if prefix + "fc1" in selected and prefix + "fc2" in skipped:
            fc2 = getattr(block.ffn, "fc2", None)
            if isinstance(fc2, nn.Linear) and fc2.weight.dtype != torch.float16:
                raise ValueError(f"FP16 FC1 output cannot feed skipped {prefix}fc2; "
                                 "use a configuration where both FFN projections are eligible")

    for name in selected:
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, FP16Projection(model.get_submodule(name)))

    # Verify storage before assigning the benchmark label.
    for name in selected:
        linear = model.get_submodule(name).linear
        if linear.weight.dtype != torch.float16 or (
            linear.bias is not None and linear.bias.dtype != torch.float16
        ):
            raise RuntimeError(f"FP16 projection storage verification failed: {name}")
    logging.info("FP16 Transformer projections: converted %d modules:\n%s",
                 len(selected), "\n".join(selected))
    logging.info("FP16 projections skipped targets: %s", skipped or "none")
    return {
        "label": "FP16 Transformer projections",
        "scope": "legacy packed encoder projections (input width % 128 = 0, output width % 8 = 0)",
        "converted_modules": selected,
        "skipped_targets": skipped,
        "projection_weight_dtype": "torch.float16",
        "projection_bias_dtype": "torch.float16 (when present)",
        "projection_input_dtype": "torch.float16 (contiguous, on weight device)",
        "projection_output_dtype": "torch.float16",
        "other_parameters": "original dtypes preserved",
    }
