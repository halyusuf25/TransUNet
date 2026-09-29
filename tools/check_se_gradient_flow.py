#!/usr/bin/env python3
"""Measure SE's connection to the real TransUNet segmentation computation graph.

Examples (run with the repository's Python environment):
  python tools/check_se_gradient_flow.py --device auto --steps 3
  python tools/check_se_gradient_flow.py --checkpoint model.pth --num-classes 9
  python tools/check_se_gradient_flow.py --se-calib-only --verbose \
      --json-output se_calibration.json --log-file se_calibration.log

No pretrained weights are downloaded. Only this process's model is modified.
Exit codes: 0 = completed with valid controls, 1 = execution/checkpoint error,
2 = inconclusive controls. Either connected or disconnected SE can be a valid result.

# Without a checkpoint
python tools/check_se_gradient_flow.py --device auto --steps 3 \
    --json-output diagnostics/se_check.json \
    --log-file diagnostics/se_check.log \
    --verbose
        

# With a checkpoint; match its class count
python tools/check_se_gradient_flow.py --device auto --steps 3 \
  --checkpoint ckpt/ckpt_Synapse/SEBlockFixed_Synapse__best_val_dice_0.7925063371658325_epoch_103_20260709_010335.pth --num-classes 9 \
  --json-output diagnostics/se_checkpoint_9classes_pretrainedmodel.json \
  --log-file diagnostics/se_checkpoint_9classes_pretrainedmodel.log \
  --verbose

# Verbose console logging, also saved to a file
python tools/check_se_gradient_flow.py --device auto --steps 3 --verbose \
  --json-output diagnostics/se_verbose.json \
  --log-file diagnostics/se_verbose.log \
  --verbose
  
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("se_gradient_flow")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=9, help="Synapse default; match checkpoint's output channels")
    parser.add_argument("--img-size", type=int, default=224, help="Default training resolution; must be a multiple of 16")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--json-output", type=Path, default=Path("se_gradient_flow.json"))
    parser.add_argument("--log-file", type=Path, help="Save the console diagnostic log as well")
    parser.add_argument("--verbose", action="store_true", help="Log each SE parameter, control, and returned gate per step")
    parser.add_argument("--se-calib-only", action="store_true", help="Use the existing calibration-only encoder mode, without auxiliary loss")
    parser.add_argument("--base-lr", type=float, default=0.01, help="train.py default")
    parser.add_argument("--ce-weight", type=float, default=0.5, help="trainer lambda_: CE weight; Dice weight is 1 minus this")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads (also used for CUDA host work)")
    parser.add_argument("--atol", type=float, default=1e-6, help="FP32 absolute tolerance for logits, losses, and gates")
    parser.add_argument("--rtol", type=float, default=1e-5, help="FP32 relative tolerance: abs(candidate-reference) <= atol + rtol*abs(reference)")
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.num_classes < 2 or args.threads < 1:
        parser.error("batch-size, steps, and threads must be positive; num-classes must be at least 2")
    if args.img_size < 32 or args.img_size % 16:
        parser.error("img-size must be at least 32 and divisible by 16")
    if not math.isfinite(args.base_lr) or args.base_lr <= 0:
        parser.error("base-lr must be finite and positive")
    if not 0 <= args.ce_weight <= 1:
        parser.error("ce-weight must be between 0 and 1")
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        parser.error("atol and rtol must be finite and nonnegative")
    for key in ("checkpoint", "json_output", "log_file"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    destinations = [p for p in (args.json_output, args.log_file) if p is not None]
    if len(set(destinations)) != len(destinations) or args.checkpoint in destinations:
        parser.error("checkpoint, JSON output, and log file must have distinct paths")
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    files = ["networks/se_block.py", "networks/vit_seg_modeling.py", "networks/vit_seg_configs.py",
             "train.py", "trainer.py", "utils.py", "tools/check_se_gradient_flow.py"]
    return {"commit_sha": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain=v1"),
            "source_sha256": {name: file_sha256(ROOT / name) for name in files}}


def finite_float(value):
    result = float(value)
    return result if math.isfinite(result) else None


def gradient_stats(tensor):
    grad = tensor.grad
    if grad is None:
        return {"gradient_status": "None", "gradient_norm": None, "gradient_finite": None}
    return {"gradient_status": "nonzero" if bool(grad.count_nonzero()) else "all_zero",
            "gradient_norm": finite_float(grad.norm().item()), "gradient_finite": bool(grad.isfinite().all())}


def gate_stats(gate, index):
    detached = gate.detach()
    return {"index": index, "shape": list(gate.shape), "requires_grad": gate.requires_grad,
            "grad_fn": type(gate.grad_fn).__name__ if gate.grad_fn is not None else None,
            "min": finite_float(detached.min().item()), "max": finite_float(detached.max().item()),
            "mean": finite_float(detached.mean().item()), "finite": bool(detached.isfinite().all())}


def load_checkpoint(model, path, report):
    """Check every key/shape before loading; record all supported normalization."""
    info = report["checkpoint"]
    info.update(path=str(path), sha256=file_sha256(path), size_bytes=path.stat().st_size,
                loaded=False, weights_only=True)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a state_dict or a dict wrapping one")
    candidates = [key for key in ("state_dict", "model_state_dict", "model") if isinstance(payload.get(key), dict)]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous checkpoint state dictionaries: {candidates}")
    wrapper = candidates[0] if candidates else None
    state = payload[wrapper] if wrapper else payload
    info["wrapper"] = wrapper
    info["other_top_level_keys"] = sorted(str(key) for key in payload if key != wrapper) if wrapper else []
    if not all(isinstance(key, str) for key in state):
        raise ValueError("Checkpoint state_dict keys must be strings")
    prefixed = bool(state) and all(key.startswith("module.") for key in state)
    info["removed_prefix"] = "module." if prefixed else None
    if prefixed:
        state = {key[len("module."):]: value for key, value in state.items()}
    expected = model.state_dict()
    info["missing_keys"] = sorted(set(expected) - set(state))
    info["unexpected_keys"] = sorted(set(state) - set(expected))
    info["shape_mismatches"] = []
    info["invalid_tensors"] = []
    info["dtype_conversions"] = []
    for key in sorted(set(state) & set(expected)):
        value, target = state[key], expected[key]
        if not torch.is_tensor(value) or value.is_quantized or value.layout != torch.strided:
            info["invalid_tensors"].append(key)
        elif value.shape != target.shape:
            info["shape_mismatches"].append({"name": key, "checkpoint": list(value.shape), "model": list(target.shape)})
        elif value.dtype != target.dtype:
            info["dtype_conversions"].append({"name": key, "from": str(value.dtype), "to": str(target.dtype)})
            if value.is_complex() or value.is_floating_point() != target.is_floating_point():
                info["invalid_tensors"].append(key)
    incompatible = any(info[key] for key in ("missing_keys", "unexpected_keys", "shape_mismatches", "invalid_tensors"))
    info["compatible"] = not incompatible
    LOG.info("Checkpoint compatibility: %s", json.dumps(info, sort_keys=True))
    if incompatible:
        raise ValueError("Checkpoint incompatible with the selected standard R50-ViT-B/16 model; see checkpoint details in JSON/log")
    model.load_state_dict(state, strict=True)
    info["loaded"] = True
    info["initialization"] = "strictly loaded checkpoint model weights; fresh optimizer"


def segmentation_loss(logits, labels, ce_loss, dice_loss, ce_weight):
    ce = ce_loss(logits, labels.long())
    dice = dice_loss(logits, labels, softmax=True)
    total = ce_weight * ce + (1 - ce_weight) * dice
    return total, {"total": float(total.detach()), "ce": float(ce.detach()), "dice": float(dice.detach())}


def train_steps(model, images, labels, se_params, control_params, ce_loss, dice_loss, args, report):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in model.parameters()}:
        raise RuntimeError("Optimizer does not contain every model parameter")
    tracked = {**se_params, **control_params}
    report["training"] = []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        before = {name: param.detach().clone() for name, param in tracked.items()}
        logits, _, _, gates = model(images)
        if tuple(logits.shape) != (args.batch_size, args.num_classes, args.img_size, args.img_size):
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        for gate in gates:
            if gate.requires_grad:
                gate.retain_grad()
        loss, values = segmentation_loss(logits, labels, ce_loss, dice_loss, args.ce_weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite segmentation loss in step {step + 1}")
        loss.backward()
        rows = {name: {"name": name, "shape": list(param.shape), "requires_grad": param.requires_grad,
                       **gradient_stats(param)} for name, param in tracked.items()}
        gate_rows = [{**gate_stats(gate, index), **gradient_stats(gate)} for index, gate in enumerate(gates)]
        used_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        for name, param in tracked.items():
            rows[name]["max_abs_parameter_change"] = finite_float((param.detach() - before[name]).abs().max().item())
        # Match trainer.py's post-step schedule, using this run's steps as its horizon.
        next_lr = args.base_lr * (1 - step / args.steps) ** 0.9
        for group in optimizer.param_groups:
            group["lr"] = next_lr
        controls_ok = any(row["gradient_status"] == "nonzero" and row["gradient_finite"]
                          and row["max_abs_parameter_change"] is not None and row["max_abs_parameter_change"] > 0
                          for name, row in rows.items() if name in control_params)
        record = {"step": step + 1, "loss": values, "learning_rate": used_lr, "next_learning_rate": next_lr,
                  "se_parameters": [rows[name] for name in se_params],
                  "transformer_controls": [rows[name] for name in control_params], "gates": gate_rows,
                  "positive_control_passed": controls_ok}
        report["training"].append(record)
        counts = {state: sum(rows[name]["gradient_status"] == state for name in se_params)
                  for state in ("None", "all_zero", "nonzero")}
        LOG.info("Step %d/%d loss=%.8g SE grads=%s transformer control=%s", step + 1, args.steps, values["total"], counts, controls_ok)
        for row in [*rows.values(), *gate_rows]:
            LOG.debug("Step %d %s", step + 1, json.dumps(row, sort_keys=True))
        del before, logits, gates, loss
    optimizer.zero_grad(set_to_none=True)


def compare_tensors(reference, candidate, args):
    if reference.shape != candidate.shape:
        return {"shape_mismatch": [list(reference.shape), list(candidate.shape)], "within_tolerance": False,
                "max_abs_difference": None, "exactly_equal": False, "finite": False}
    return {"max_abs_difference": finite_float((candidate - reference).abs().max().item()),
            "exactly_equal": torch.equal(reference, candidate),
            "within_tolerance": torch.allclose(candidate, reference, atol=args.atol, rtol=args.rtol),
            "finite": bool(reference.isfinite().all() and candidate.isfinite().all())}


def compare_outputs(reference, candidate, args):
    gate_rows = [{"index": index, **compare_tensors(left, right, args)}
                 for index, (left, right) in enumerate(zip(reference["gates"], candidate["gates"]))]
    same_count = len(reference["gates"]) == len(candidate["gates"])
    loss_diffs = {key: {"absolute_difference": abs(candidate["loss"][key] - reference["loss"][key]),
                       "signed_difference": candidate["loss"][key] - reference["loss"][key],
                       "within_tolerance": abs(candidate["loss"][key] - reference["loss"][key])
                       <= args.atol + args.rtol * abs(reference["loss"][key])}
                  for key in reference["loss"]}
    return {"logits": compare_tensors(reference["logits"], candidate["logits"], args), "loss": loss_diffs,
            "reference_loss": reference["loss"], "candidate_loss": candidate["loss"],
            "gates": {"reference_count": len(reference["gates"]), "candidate_count": len(candidate["gates"]),
                      "same_count": same_count, "per_gate": gate_rows,
                      "any_changed_exactly": same_count and any(not row["exactly_equal"] for row in gate_rows),
                      "any_changed_beyond_tolerance": same_count and any(not row["within_tolerance"] for row in gate_rows),
                      "all_within_tolerance": same_count and all(row["within_tolerance"] for row in gate_rows)}}


def eval_comparisons(model, images, labels, se_params, ce_loss, dice_loss, args, report):
    """Use one model, fully restoring parameters, buffers, flags and RNG per trial."""
    snapshot = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    modes = [(module, module.training) for module in model.modules()]
    config = model.transformer.encoder.args
    had_drop_flag = "drop_se_block" in config
    drop_flag = getattr(config, "drop_se_block", False)
    py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def restore():
        model.load_state_dict(snapshot, strict=True)
        config.drop_se_block = drop_flag
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.eval()

    @torch.no_grad()
    def forward():
        logits, _, _, gates = model(images)
        _, values = segmentation_loss(logits, labels, ce_loss, dice_loss, args.ce_weight)
        if not bool(logits.isfinite().all()) or not all(math.isfinite(v) for v in values.values()):
            raise RuntimeError("Non-finite evaluation logits/loss")
        return {"logits": logits.cpu(), "loss": values, "gates": [gate.detach().cpu() for gate in gates]}

    result = report["evaluation"] = {}
    try:
        restore()
        reference = forward()
        result["baseline"] = {"loss": reference["loss"], "logits_shape": list(reference["logits"].shape),
                              "gates": [gate_stats(gate, i) for i, gate in enumerate(reference["gates"])]}
        # Do not restore model state for the first repeat: detect eval-time state drift too.
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        result["unchanged_repeat"] = compare_outputs(reference, forward(), args)
        restore()
        torch.manual_seed(args.seed + 100)
        changes = []
        with torch.no_grad():
            for name, param in se_params.items():
                sigma = max(float(param.square().mean().sqrt()), 0.1)
                param.add_(torch.randn_like(param) * sigma)
                changes.append({"name": name, "noise_std": sigma,
                                "max_abs_parameter_change": float((param.cpu() - snapshot[name]).abs().max())})
        # Perturbation's RNG consumption must not affect the forward comparison.
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        result["perturbation"] = {"method": "add independent Gaussian noise only to SE-owned parameters; std=max(parameter RMS, 0.1)",
                                  "seed": args.seed + 100, "parameters": changes}
        result["se_perturbed"] = compare_outputs(reference, forward(), args)
        restore()
        config.drop_se_block = True
        result["se_bypassed"] = compare_outputs(reference, forward(), args)
        result["bypass_mechanism"] = "model.transformer.encoder.args.drop_se_block=True"
        restore()
        result["restored_repeat"] = compare_outputs(reference, forward(), args)
    finally:
        restore()
        if not had_drop_flag:
            del config["drop_se_block"]
        for module, training in modes:
            module.training = training
        result["model_state_restored_exactly"] = all(torch.equal(value.detach().cpu(), snapshot[name])
                                                    for name, value in model.state_dict().items())
        result["training_flags_restored"] = all(module.training == training for module, training in modes)


def interpret(report):
    training, evaluation = report["training"], report["evaluation"]
    se_rows = [row for step in training for row in step["se_parameters"]]
    gate_rows = [row for step in training for row in step["gates"]]
    controls = {
        "se_parameters_found": bool(se_rows), "returned_gates_found": bool(gate_rows),
        "transformer_positive_control_every_step": all(step["positive_control_passed"] for step in training),
        "finite_gradients_and_updates": all(row["gradient_finite"] is not False
                                           and row["max_abs_parameter_change"] is not None
                                           for step in training for row in step["se_parameters"] + step["transformer_controls"]),
        "finite_gates_and_gate_gradients": all(row["finite"] and row["gradient_finite"] is not False for row in gate_rows),
        "perturbation_changed_gates": evaluation["se_perturbed"]["gates"]["any_changed_beyond_tolerance"],
        "bypass_removed_gates": evaluation["se_bypassed"]["gates"]["candidate_count"] == 0,
        "model_state_restored": evaluation["model_state_restored_exactly"] and evaluation["training_flags_restored"],
        "finite_evaluation_outputs": all(evaluation[name]["logits"]["finite"]
                                         and all(row["finite"] for row in evaluation[name]["gates"]["per_gate"])
                                         for name in ("unchanged_repeat", "se_perturbed", "se_bypassed", "restored_repeat")),
    }
    for name in ("unchanged_repeat", "restored_repeat"):
        comparison = evaluation[name]
        controls[name] = (comparison["logits"]["within_tolerance"] and comparison["loss"]["total"]["within_tolerance"]
                          and comparison["gates"]["all_within_tolerance"])
    findings = {
        "se_parameter_gradient_counts": {state: sum(row["gradient_status"] == state for row in se_rows)
                                         for state in ("None", "all_zero", "nonzero")},
        "se_parameter_updates_observed": any((row["max_abs_parameter_change"] or 0) > 0 for row in se_rows),
        "gate_gradient_counts": {state: sum(row["gradient_status"] == state for row in gate_rows)
                                 for state in ("None", "all_zero", "nonzero")},
        "output_effects": {name: {"logits_changed": not evaluation[name]["logits"]["within_tolerance"],
                                  "loss_changed": not evaluation[name]["loss"]["total"]["within_tolerance"]}
                           for name in ("se_perturbed", "se_bypassed")},
    }
    valid = all(controls.values())
    gradients_observed = any(row["gradient_status"] == "nonzero" for row in se_rows + gate_rows)
    effects_observed = any(any(values.values()) for values in findings["output_effects"].values())
    all_none = all(row["gradient_status"] == "None" for row in se_rows + gate_rows)
    if not valid:
        conclusion = "Inconclusive: one or more experimental controls failed. Inspect individual measurements."
    elif gradients_observed or effects_observed:
        conclusion = "SE has a measured segmentation-loss gradient and/or prediction/loss effect in this configuration; the disconnected-branch hypothesis is not supported."
    elif all_none and not findings["se_parameter_updates_observed"]:
        conclusion = "Measurements support a disconnected SE branch for this configuration's segmentation-only training graph."
    else:
        conclusion = "No effect above tolerance was measured, but computed zero gradients or updates prevent concluding that SE is disconnected."
    report.update(status="completed" if valid else "inconclusive", controls=controls, findings=findings, conclusion=conclusion)
    report["interpretation_notes"] = [
        "grad=None means no gradient was produced for that tensor in this backward pass; an all-zero tensor is a computed gradient with zero entries.",
        "requires_grad=True and a grad_fn indicate autograd tracking, not necessarily a connection to the segmentation loss.",
        "SGD weight decay/momentum can update parameters with computed zero gradients; a fresh optimizer skips parameters whose grad is None.",
        "Unchanged SE weights do not imply constant gates: the upstream backbone/features can change during training.",
        "A branch disconnected from segmentation loss may still affect subsequent quantization when calibration consumes its gates; quantization was not run here.",
        "Synthetic inputs establish behavior of this tested computation graph, not historical training or quantization results.",
        "Current source has normal SE and explicit se_calib_only modes. The trainer also has an optional SE auxiliary loss; this experiment uses only CE + Dice.",
        "Checkpoint state_dicts do not establish historical flags or optimizer state; this run uses the reported CLI configuration and a fresh SGD optimizer.",
    ]


def print_summary(report):
    LOG.info("%-20s %8s %8s %8s %12s", "Training step", "SE None", "SE zero", "SE nonzero", "SE updated")
    for step in report["training"]:
        rows = step["se_parameters"]
        counts = [sum(row["gradient_status"] == state for row in rows) for state in ("None", "all_zero", "nonzero")]
        LOG.info("%-20s %8d %8d %8d %12d", str(step["step"]), *counts,
                 sum((row["max_abs_parameter_change"] or 0) > 0 for row in rows))
    LOG.info("%-20s %16s %16s %12s", "Eval comparison", "max |logit delta|", "|loss delta|", "both close")
    for name in ("unchanged_repeat", "se_perturbed", "se_bypassed", "restored_repeat"):
        item = report["evaluation"][name]
        LOG.info("%-20s %16.8g %16.8g %12s", name, item["logits"]["max_abs_difference"],
                 item["loss"]["total"]["absolute_difference"],
                 item["logits"]["within_tolerance"] and item["loss"]["total"]["within_tolerance"])
        for row in item["gates"]["per_gate"]:
            LOG.debug("%s gate: %s", name, json.dumps(row, sort_keys=True))
    gate_changes = report["evaluation"]["se_perturbed"]["gates"]
    LOG.info("Perturbation changed %d/%d gates beyond tolerance; largest absolute gate change=%s",
             sum(not row["within_tolerance"] for row in gate_changes["per_gate"]), gate_changes["reference_count"],
             max((row["max_abs_difference"] or 0 for row in gate_changes["per_gate"]), default=None))
    LOG.info("Controls: %s", json.dumps(report["controls"], sort_keys=True))
    LOG.info("%s", report["conclusion"])
    for note in report["interpretation_notes"]:
        LOG.info("%s", note)


def run(args, report):
    # Delayed imports let --help work and let missing dependencies produce error JSON.
    global torch, np
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT))
    from networks.se_block import SELayer
    from networks.vit_seg_modeling import CONFIGS, VisionTransformer
    from utils import DiceLoss

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # CUDA CE lacks strict deterministic kernels in some PyTorch versions.
    # Prefer deterministic kernels, then measure repeatability explicitly.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA execution are supported")
    report["environment"] = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                             "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                             "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                             "threads": torch.get_num_threads(), "dtype": "torch.float32", "autocast": False,
                             "deterministic_algorithms": True, "deterministic_warn_only": True, "tf32": False,
                             "reproducibility": "Seeded RNG and preferred deterministic kernels; evaluation repeats are measured controls"}
    config = copy.deepcopy(CONFIGS["R50-ViT-B_16"])
    config.n_classes = args.num_classes
    config.n_skip = 3
    config.patches.grid = (args.img_size // 16, args.img_size // 16)
    for flag in ("use_swin", "use_efficientnet", "use_shsa", "use_alternate_shsa", "use_ats", "use_gumbel_topk", "verbose", "drop_se_block"):
        setattr(config, flag, False)
    config.topk_attn = 0.0
    config.gumbel_sampling_mode = "dist"
    config.use_se_block = True
    config.se_calib_only = args.se_calib_only
    config.pretrained_path = None
    report["configuration"] = {"model_name": "R50-ViT-B_16", "model": config.to_dict(),
                               "image_shape": [args.batch_size, 3, args.img_size, args.img_size],
                               "mask_shape": [args.batch_size, args.img_size, args.img_size],
                               "loss": {"CE": "torch.nn.CrossEntropyLoss", "Dice": "utils.DiceLoss", "softmax_dice": True,
                                        "ce_weight": args.ce_weight, "dice_weight": 1 - args.ce_weight, "auxiliary_loss": False},
                               "optimizer": {"type": "SGD", "base_lr": args.base_lr, "momentum": 0.9, "weight_decay": 0.0001,
                                             "all_model_parameters": True, "zero_grad_set_to_none": True,
                                             "schedule": "post-step base_lr * (1 - zero_based_step / steps)**0.9", "horizon": args.steps}}
    LOG.info("Building full R50-ViT-B/16: device=%s, image=%d, classes=%d, calibration_only=%s", device, args.img_size, args.num_classes, args.se_calib_only)
    # Embeddings reads a repo-relative YAML even when Swin is disabled.
    previous_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        model = VisionTransformer(config, img_size=args.img_size, num_classes=args.num_classes).float()
    finally:
        os.chdir(previous_cwd)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, report)
    model.to(device=device, dtype=torch.float32)
    se_modules = {name: module for name, module in model.named_modules() if isinstance(module, SELayer)}
    se_ids = {id(param) for module in se_modules.values() for param in module.parameters()}
    se_params = {name: param for name, param in model.named_parameters() if id(param) in se_ids}
    candidates = [(name, param) for name, param in model.named_parameters()
                  if name.startswith("transformer.encoder.layer.") and id(param) not in se_ids
                  and name.endswith(("attn.query.weight", "ffn.fc2.weight"))]
    control_params = dict(candidates[:2] + candidates[-2:])
    report["discovery"] = {"se_modules": list(se_modules), "se_parameter_tensor_count": len(se_params),
                           "se_parameter_count": sum(param.numel() for param in se_params.values()),
                           "model_parameter_count": sum(param.numel() for param in model.parameters()),
                           "transformer_control_names": list(control_params)}
    LOG.info("Discovered %d SE modules, %d SE parameter tensors, %d transformer controls", len(se_modules), len(se_params), len(control_params))
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    images = torch.randn(args.batch_size, 3, args.img_size, args.img_size, generator=generator).to(device)
    labels = torch.randint(args.num_classes, (args.batch_size, args.img_size, args.img_size), generator=generator).to(device)
    report["synthetic_data"] = {"seed": args.seed + 1, "images": "standard normal FP32", "masks": "uniform integer class IDs",
                                "class_histogram": torch.bincount(labels.flatten(), minlength=args.num_classes).cpu().tolist(),
                                "same_batch_each_step_and_comparison": True}
    ce_loss, dice_loss = torch.nn.CrossEntropyLoss(), DiceLoss(args.num_classes)
    train_steps(model, images, labels, se_params, control_params, ce_loss, dice_loss, args, report)
    LOG.info("Running FP32 evaluation comparisons (atol=%g, rtol=%g)", args.atol, args.rtol)
    eval_comparisons(model, images, labels, se_params, ce_loss, dice_loss, args, report)
    interpret(report)
    print_summary(report)
    return 0 if report["status"] == "completed" else 2


def main():
    args = parse_args()
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s", handlers=handlers)
    logging.captureWarnings(True)
    # Required by deterministic CUDA GEMM, before importing/initializing PyTorch.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    started = time.perf_counter()
    report = {"schema_version": 1, "status": "running", "provenance": provenance(),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "seed": args.seed, "checkpoint": {"path": None, "loaded": False, "initialization": "local random initialization"},
              "tolerances": {"atol": args.atol, "rtol": args.rtol,
                             "rule": "abs(candidate - reference) <= atol + rtol * abs(reference); applied to each logit, gate, and scalar loss"}}
    exit_code = 1
    try:
        exit_code = run(args, report)
    except Exception as exc:
        report.update(status="error", error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        LOG.exception("Diagnostic could not complete; no successful conclusion is claimed")
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        LOG.info("Detailed JSON: %s", args.json_output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
