"""Validation and gathered-batch loss reduction for optional SE training."""

from collections import Counter
import math

import torch
import torch.nn.functional as F


def validate_se_aux_args(args):
    """Validate only the opt-in auxiliary path; leave legacy runs unchanged."""
    if not bool(getattr(args, "se_aux_loss", False)):
        return
    if not bool(getattr(args, "use_se_block", False)):
        raise ValueError("--se_aux_loss requires --use_se_block.")
    if bool(getattr(args, "use_kd", False)):
        raise ValueError("--se_aux_loss is not supported with --use_kd; auxiliary SE training must run without KD.")
    weight = float(getattr(args, "se_aux_weight", 0.1))
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("--se_aux_weight must be finite and strictly positive when --se_aux_loss is enabled.")


def compute_se_aux_loss(gates, targets, expected_blocks=None):
    """Average per-block MSE after DataParallel has gathered image tensors.

    Both lists follow encoder block order. Reducing the gathered [B,1,C]
    tensors here weights every image equally, including when replicas receive
    different batch sizes; averaging replica-local scalar losses would not.
    """
    if not isinstance(gates, (list, tuple)) or not isinstance(targets, (list, tuple)):
        raise ValueError("SE gates and targets must be ordered lists with one tensor per encoder block.")
    if not gates or len(gates) != len(targets):
        raise ValueError("SE gate/target block correspondence failed: nonempty lists must have equal length.")
    if expected_blocks is not None and len(gates) != expected_blocks:
        raise ValueError(f"Expected SE gates and targets for {expected_blocks} blocks, got {len(gates)}.")

    block_losses = []
    batch_channels = None
    for block_index, (gate, target) in enumerate(zip(gates, targets)):
        context = f"SE block {block_index}"
        if not isinstance(gate, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError(f"{context}: gates and targets must be tensors.")
        if gate.shape != target.shape:
            raise ValueError(f"{context}: exact gate/target shape equality required, got {tuple(gate.shape)} and {tuple(target.shape)}.")
        if gate.ndim != 3 or gate.shape[1] != 1 or gate.shape[0] < 1 or gate.shape[2] < 1:
            raise ValueError(f"{context}: expected nonempty [B,1,C] gates and targets, got {tuple(gate.shape)}.")
        if batch_channels is None:
            batch_channels = (gate.shape[0], gate.shape[2])
        elif batch_channels != (gate.shape[0], gate.shape[2]):
            raise ValueError(f"{context}: image and channel dimensions must agree across encoder blocks.")
        if target.requires_grad or target.grad_fn is not None:
            raise ValueError(f"{context}: targets must be detached.")
        if target.dtype != torch.float32:
            raise ValueError(f"{context}: targets must be float32.")
        if not gate.is_floating_point() or gate.device != target.device:
            raise ValueError(f"{context}: floating-point gates and targets must be on the same device.")
        if not torch.isfinite(gate).all() or not torch.isfinite(target).all():
            raise ValueError(f"{context}: gates and targets must be finite.")
        block_losses.append(F.mse_loss(gate.float(), target))
    return torch.stack(block_losses).mean()


def verify_se_optimizer(model, optimizer):
    """Check one registered predictor per block and one optimizer entry per SE parameter."""
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    transformer = getattr(core_model, "transformer", None)
    encoder = getattr(transformer, "encoder", None)
    blocks = getattr(encoder, "layer", None)
    predictors = getattr(encoder, "SELayer", None)
    if not isinstance(predictors, torch.nn.ModuleList) or blocks is None or len(predictors) != len(blocks) or not predictors:
        raise ValueError("Auxiliary SE training requires one registered SE predictor per encoder block.")
    se_parameters = [parameter for predictor in predictors for parameter in predictor.parameters() if parameter.requires_grad]
    if not se_parameters or len({id(parameter) for parameter in se_parameters}) != len(se_parameters):
        raise ValueError("Auxiliary SE predictors must have distinct trainable parameters.")
    optimizer_counts = Counter(id(parameter) for group in optimizer.param_groups for parameter in group["params"])
    if any(optimizer_counts[id(parameter)] != 1 for parameter in se_parameters):
        raise ValueError("Every trainable SE parameter must appear exactly once in the existing optimizer.")
    return len(predictors)
