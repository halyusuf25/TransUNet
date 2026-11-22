#!/usr/bin/env python3
"""Standalone TransUNet se_loss profiling (no production-file modifications).

Place this file in the repository root or pass --repo-root /path/to/TransUNet.
Run --help for configuration; --check-environment needs no dataset/checkpoint.
Example:
  python tools/profile_quantization.py --dataset Synapse \
    --checkpoint ckpt/ckpt_Synapse/SELoss_Synapse__best_val_dice_checkpoint.pth \
    --quick --output-dir profiling/synapse_smoke
    
    
CUDA_VISIBLE_DEVICES=1 python tools/profile_quantization.py \
  --dataset Synapse \
  --checkpoint ckpt/ckpt_Synapse/SELoss_Synapse__best_val_dice_0.7899280190467834_epoch_100_20260930_173838.pth \
  --methods fp32 fp16 awq se_aux \
  --batch-sizes 1 36 \
  --calibration-batches 8 \
  --quick --profile-stacks \
  --output-dir profiling/synapse_smoke_quick
  
CUDA_VISIBLE_DEVICES=1 python tools/profile_quantization.py \
  --dataset Synapse \
  --checkpoint ckpt/ckpt_Synapse/SELoss_Synapse__best_val_dice_0.7899280190467834_epoch_100_20260930_173838.pth \
  --methods fp32 fp16 awq se_aux \
  --batch-sizes 1 36 \
  --calibration-batches 8 \
  --profile-stacks \
  --output-dir profiling/synapse_smoke_NotQuick

Inspected source: halyusuf25/TransUNet se_loss, commit
770a6a87f8a9746ab879409acd25ee4761a640a6. This script does not change a branch,
train a model, evaluate segmentation scores, or replace an inference backend.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import statistics
import subprocess
import sys
import time
import traceback

SCHEMA_VERSION = "1.0"
INSPECTED_COMMIT = "770a6a87f8a9746ab879409acd25ee4761a640a6"
METHODS = ("fp32", "fp16", "awq", "se_aux")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_json(value):
    """Normalize optional backend metadata; reject nonfinite observations upstream."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            return safe_json(value.item())
        except (ValueError, RuntimeError, TypeError):
            pass
    return str(value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(safe_json(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, q):
    ordered = sorted(float(x) for x in values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def describe(values):
    xs = [float(x) for x in values]
    if any(not math.isfinite(x) for x in xs):
        raise ValueError("Nonfinite timing or throughput observation")
    return {
        "n": len(xs), "mean": statistics.fmean(xs) if xs else None,
        "sample_std": statistics.stdev(xs) if len(xs) > 1 else None,
        "sample_std_unavailable_reason": None if len(xs) > 1 else "Requires at least two observations",
        "median": percentile(xs, 50), "p90": percentile(xs, 90),
        "p95": percentile(xs, 95), "p99": percentile(xs, 99),
        "min": min(xs) if xs else None, "max": max(xs) if xs else None,
        "percentile_definition": "Linear interpolation at (n-1)*q/100 on individual observations",
    }


def command_result(argv, cwd=None):
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=15)
        return {"status": "ok" if result.returncode == 0 else "unavailable",
                "returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def environment_metadata(repo_root, with_torch=False):
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "neural-compressor", "awq",
                 "autoawq", "triton", "fvcore", "thop", "ml-collections"):
        try:
            packages[name] = {"version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"version": None, "reason": "Distribution metadata not installed"}
    source_hashes = {}
    for relative in ("test.py", "src/test_helpers.py", "src/inc_awq.py", "src/fp16_projections.py",
                     "src/benchmark.py", "src/se_auxiliary.py", "networks/quantizer.py",
                     "networks/vit_seg_modeling.py"):
        path = Path(repo_root) / relative
        source_hashes[relative] = file_sha256(path) if path.is_file() else None
    result = {
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "repo_root": str(repo_root), "inspected_source_commit": INSPECTED_COMMIT,
        "git_head": command_result(["git", "rev-parse", "HEAD"], repo_root),
        "git_branch": command_result(["git", "branch", "--show-current"], repo_root),
        "git_status": command_result(["git", "status", "--porcelain"], repo_root),
        "source_sha256": source_hashes,
        "script_sha256": file_sha256(__file__),
        "execution_environment": {key: os.environ.get(key) for key in
            ("CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "CUBLAS_WORKSPACE_CONFIG",
             "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "OMP_NUM_THREADS", "INC_TARGET_DEVICE")},
        "nvidia_smi": command_result(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,utilization.gpu,temperature.gpu,clocks.sm,power.draw", "--format=csv,noheader,nounits"]),
    }
    if with_torch:
        try:
            import torch
            result["torch_runtime"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available()}
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                prop = torch.cuda.get_device_properties(device)
                result["torch_runtime"].update({"device_index": device, "gpu_name": prop.name,
                    "compute_capability": [prop.major, prop.minor], "total_memory_bytes": prop.total_memory,
                    "gpu_uuid": str(getattr(prop, "uuid", "unavailable")),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "default_dtype": str(torch.get_default_dtype()),
                    "stream": "default" if torch.cuda.current_stream() == torch.cuda.default_stream() else "nondefault",
                    "autocast": False, "torch_compile": False, "cuda_graphs": False,
                    "inference_mode_for_measured_forwards": True})
        except Exception as exc:
            result["torch_runtime"] = {"status": "unavailable", "reason": str(exc)}
    return result


class PhaseRecorder:
    def __init__(self, records):
        self.records = records

    @contextlib.contextmanager
    def __call__(self, name):
        import torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        row = {"phase": name, "status": "running", "started_at": utc_now()}
        self.records.append(row)
        print(f"  {name} ...", flush=True)
        try:
            yield
            torch.cuda.synchronize()
            row["status"] = "ok"
        except BaseException as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            raise
        finally:
            row["wall_time_ms"] = (time.perf_counter() - start) * 1000.0


def tensor_outputs(value):
    import torch
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensor_outputs(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in tensor_outputs(v)]
    return []


def check_output(output):
    import torch
    tensors = tensor_outputs(output)
    if not tensors:
        raise RuntimeError("Forward returned no tensors")
    rows = [{"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device),
             "finite": bool(torch.isfinite(t).all().item())} for t in tensors]
    if not all(row["finite"] for row in rows):
        raise RuntimeError("Nonfinite model output; refusing to report valid benchmark")
    return rows


def payload_measurement(model):
    from utils import model_size_mb_benchmark
    repository_metric = model_size_mb_benchmark(model)
    storages = {}
    for name, tensor in model.state_dict().items():
        try:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            storages.setdefault(key, {"bytes": storage.nbytes(), "device": str(tensor.device), "aliases": []})["aliases"].append(name)
        except (AttributeError, RuntimeError):
            continue
    return {"repository_persistent_payload": repository_metric,
            "unique_backing_storage_bytes": sum(row["bytes"] for row in storages.values()),
            "shared_storage_groups": [row for row in storages.values() if len(row["aliases"]) > 1],
            "note": "Repository metric counts persistent state_dict tensor elements. Unique backing storage counts entire storages, including unused portions of views; neither is peak runtime memory."}


def prepare_device_inputs(bank, args):
    import torch
    n = len(bank)
    if not n:
        raise ValueError("Empty benchmark input bank")
    if bank.device.type != "cpu":
        raise ValueError("Repository adapter must return a CPU input bank")
    digest = hashlib.sha256(bank.contiguous().numpy().tobytes()).hexdigest()
    batches, indices = [], []
    for i in range(args.resident_batches):
        selection = [(i * args.batch_size + j) % n for j in range(args.batch_size)]
        batch = bank[selection].contiguous().to("cuda")
        batches.append(batch)
        indices.append(selection)
    torch.cuda.synchronize()
    return batches, {"cpu_bank_shape": list(bank.shape), "cpu_bank_sha256": digest,
                     "resident_batch_indices": indices, "resident_batches": len(batches),
                     "resident_input_bytes": sum(t.numel() * t.element_size() for t in batches),
                     "input_dtype": str(batches[0].dtype),
                     "policy": "Fixed preprocessed CPU bank; selected full batches uploaded once and cycled; no timed loading, copies or input allocations"}


def clean_benchmarks(model, inputs, args):
    import torch
    runs = []
    for run in range(args.runs):
        gpu_ms, host_ms = [], []
        # Preinitialize events before measuring. Synchronized calls measure isolated latency.
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record(); end_event.record(); end_event.synchronize()
        for i in range(args.iterations):
            torch.cuda.synchronize()
            start_host = time.perf_counter()
            start_event.record()
            output = model(inputs[i % len(inputs)])
            end_event.record()
            torch.cuda.synchronize()
            host_ms.append((time.perf_counter() - start_host) * 1000.0)
            gpu_ms.append(start_event.elapsed_time(end_event))
            del output
        # This independent loop has no per-forward synchronization or events.
        torch.cuda.synchronize()
        start_host = time.perf_counter()
        for i in range(args.iterations):
            output = model(inputs[i % len(inputs)])
            del output
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_host
        images = args.iterations * args.batch_size
        if elapsed <= 0:
            raise RuntimeError("Nonpositive throughput duration")
        row = {"run": run + 1, "latency_cuda_event_ms": gpu_ms,
               "latency_synchronized_host_ms": host_ms,
               "latency_cuda_event_summary_ms": describe(gpu_ms),
               "latency_synchronized_host_summary_ms": describe(host_ms),
               "throughput_loop_wall_ms": elapsed * 1000.0,
               "throughput_images": images, "throughput_images_per_s": images / elapsed}
        runs.append(row)
        print(f"  Run {run + 1}/{args.runs}: host batch latency {statistics.fmean(host_ms):.4f} ms; {images/elapsed:.2f} img/s", flush=True)
    return {"batch_size": args.batch_size, "latency_unit": "ms per model call (single image only when batch_size=1)",
            "raw_runs": runs,
            "latency_cuda_event_pooled_ms": describe([x for row in runs for x in row["latency_cuda_event_ms"]]),
            "latency_host_pooled_ms": describe([x for row in runs for x in row["latency_synchronized_host_ms"]]),
            "latency_host_run_means_ms": describe([row["latency_synchronized_host_summary_ms"]["mean"] for row in runs]),
            "throughput_run_summary_images_per_s": describe([row["throughput_images_per_s"] for row in runs]),
            "definitions": {"pooled_std": "Across individual forward calls", "run_mean_std": "Across repeated-run mean latency values",
                "throughput_std": "Across repeated independently timed loop throughputs",
                "cuda_events": "Elapsed interval on current/default CUDA stream; not a sum of active kernel times",
                "host": "perf_counter with device synchronization before/after each call; includes event-recording and host launch overhead",
                "throughput": "Independent perf_counter loop, device synchronization only at boundaries; output released after every call"}}


def memory_measurement(model, inputs, args):
    import torch
    gc.collect()
    torch.cuda.synchronize()
    rows = []
    for repeat in range(args.memory_runs):
        torch.cuda.synchronize()
        baseline_alloc = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        # Release each previous output before the next call, matching the throughput loop.
        for i in range(args.memory_steps):
            output = model(inputs[i % len(inputs)])
            torch.cuda.synchronize()
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            after_alloc, after_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            del output
        torch.cuda.synchronize()
        rows.append({"repeat": repeat + 1, "baseline_allocated_bytes": baseline_alloc,
            "baseline_reserved_bytes": baseline_reserved, "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_reserved, "incremental_peak_allocated_bytes": peak_alloc-baseline_alloc,
            "incremental_peak_reserved_bytes": peak_reserved-baseline_reserved,
            "after_forward_output_live_allocated_bytes": after_alloc,
            "after_forward_output_live_reserved_bytes": after_reserved,
            "after_output_release_allocated_bytes": torch.cuda.memory_allocated(),
            "after_output_release_reserved_bytes": torch.cuda.memory_reserved()})
    return {"raw_runs": rows, "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
            "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rows),
            "boundary": "Reset after setup, warm-up and clean timing. Model and all resident GPU inputs live; previous outputs and calibration intermediates released. Caching allocator not emptied inside measurement.",
            "limitations": "PyTorch allocator statistics may exclude external CUDA allocations. Reserved peaks reflect warmed caching allocator state. Multiple resident inputs change absolute memory relative to a one-input protocol."}


def source_findings(repo_root):
    findings = []
    for relative, needle, meaning in (
        ("src/inc_awq.py", "padded_rows = ((rows + 63) // 64) * 64", "AWQ wrapper contains row padding to a multiple of 64"),
        ("src/inc_awq.py", "if self.inner.in_features == 256 and padded_rows == 128:", "AWQ wrapper contains a K=256 special padding branch"),
        ("networks/quantizer.py", "class _InputCastWQLinear", "Legacy SE deployment uses its own input-cast wrapper"),
    ):
        path = Path(repo_root) / relative
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        matches = [{"line": i+1, "text": line.strip()} for i, line in enumerate(lines) if needle in line]
        findings.append({"hypothesis": meaning, "source": relative, "matches": matches,
                         "status": "observed_in_source" if matches else "not_established",
                         "interpretation": "Static source evidence only; execution frequency and cost require runtime diagnostics."})
    return findings


def worker(config_path):
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    args = argparse.Namespace(**payload["args"])
    result_path = Path(payload["result_path"])
    report = {"case_id": payload["case_id"], "method": args.method, "batch_size": args.batch_size,
              "status": "running", "started_at": utc_now(), "configuration": vars(args),
              "phases": [], "warnings": [], "errors": [], "checks": []}
    atomic_json(result_path, report)
    try:
        import torch
        import numpy as np
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. This profiling script does not substitute a CPU or fake-quantization backend.")
        torch.cuda.set_device(torch.device(args.device))
        if torch.cuda.current_stream() != torch.cuda.default_stream():
            raise RuntimeError("The packed WQLinear wrappers require the default CUDA stream")
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cudnn.deterministic = args.cudnn_deterministic
        report["environment"] = environment_metadata(args.repo_root, with_torch=True)
        phase = PhaseRecorder(report["phases"])
        model, bank, metadata = build_repository_case(args, phase)
        report["repository"] = metadata
        requested_device = torch.device(args.device)
        requested_index = requested_device.index if requested_device.index is not None else torch.cuda.current_device()
        if torch.cuda.current_device() != requested_index or next(model.parameters()).device.index != requested_index:
            raise RuntimeError("Conversion changed the selected CUDA device; refusing a mismatched benchmark")
        report["persistent_storage"] = payload_measurement(model)
        with phase("prepare_fixed_device_inputs"):
            inputs, input_metadata = prepare_device_inputs(bank, args)
        report["benchmark_inputs"] = input_metadata
        del bank
        gc.collect()
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        model.eval()
        with torch.inference_mode():
            with phase("first_deployed_forward_after_conversion"):
                output = model(inputs[0])
            report["output_validation"] = check_output(output)
            del output
            with phase("warmup"):
                for i in range(args.warmup):
                    output = model(inputs[i % len(inputs)])
                    del output
            report["benchmark"] = clean_benchmarks(model, inputs, args)
            report["memory"] = memory_measurement(model, inputs, args)
            atomic_json(result_path, report)
            if args.skip_profile:
                report["diagnostics"] = {"status": "skipped", "reason": "--skip-profile supplied"}
            else:
                try:
                    report["diagnostics"] = collect_diagnostics(model, inputs, args, result_path.parent)
                except Exception as exc:
                    report["diagnostics"] = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
                    report["warnings"].append("Detailed profiling failed; clean benchmark results are retained.")
        report["source_findings"] = source_findings(args.repo_root)
        report["checks"] = [
            {"check": "shared_checkpoint_strict_load", "status": "passed", "sha256": metadata.get("checkpoint_sha256")},
            {"check": "full_token_sequence", "status": "passed"},
            {"check": "auxiliary_se_removed", "status": "passed"},
            {"check": "finite_output", "status": "passed", "details": report["output_validation"]},
            {"check": "uninstrumented_latency_and_throughput", "status": "completed"},
            {"check": "uninstrumented_memory", "status": "completed"},
            {"check": "detailed_profiler", "status": report.get("diagnostics", {}).get("status")},
            {"check": "controlled_causal_ablation", "status": "not_performed", "reason": "Production paths preserved; no kernel/padding behavior is modified."},
        ]
        report["warnings"].extend([
            "First deployed forward follows calibration/conversion; it is not a cold-process CUDA first-use measurement.",
            "Timing repetitions repeat inference on one converted model; they do not estimate calibration or training-seed variability.",
            "This compares complete deployed implementations, not the scaling algorithm in isolation.",
        ])
        diagnostic_status = report.get("diagnostics", {}).get("status")
        report["status"] = "partial" if diagnostic_status in ("failed", "partial", "completed_with_warnings") else "ok"
        validation = report.get("diagnostics", {}).get("instrumentation_validation") or {}
        if validation.get("passed") is False:
            report["status"] = "partial"
            report["warnings"].append("Instrumented output validation failed; do not interpret the detailed profile as equivalent.")
    except BaseException as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
    finally:
        report["finished_at"] = utc_now()
        atomic_json(result_path, report)
    return 0 if report["status"] == "ok" else 1


def compare_cases(cases):
    comparisons = []
    for batch_size in sorted({c["batch_size"] for c in cases}):
        available = {c["method"]: c for c in cases if c["batch_size"] == batch_size and "benchmark" in c and "memory" in c}
        if "se_aux" not in available:
            continue
        se = available["se_aux"]
        for baseline in ("awq", "fp16", "fp32"):
            if baseline not in available:
                continue
            other = available[baseline]
            def known_equal(a, b):
                return a == b if a not in (None, "", "unavailable") and b not in (None, "", "unavailable") else None
            same_inputs = known_equal(se.get("benchmark_inputs", {}).get("cpu_bank_sha256"), other.get("benchmark_inputs", {}).get("cpu_bank_sha256"))
            metrics = {}
            for label, getter in (
                ("host_batch_latency_ms", lambda c: c["benchmark"]["latency_host_pooled_ms"]["mean"]),
                ("throughput_images_per_s", lambda c: c["benchmark"]["throughput_run_summary_images_per_s"]["mean"]),
                ("peak_allocated_bytes", lambda c: c["memory"]["peak_allocated_bytes"]),
            ):
                value, reference = getter(se), getter(other)
                metrics[label] = {"se_aux": value, "baseline": reference, "difference_se_minus_baseline": value-reference,
                                  "percent_change_from_baseline": 100*(value-reference)/reference if reference else None}
            conditions = {"same_input_bank": same_inputs,
                "same_checkpoint_sha256": known_equal(se.get("repository", {}).get("checkpoint_sha256"), other.get("repository", {}).get("checkpoint_sha256")),
                "same_resident_batch_indices": known_equal(se.get("benchmark_inputs", {}).get("resident_batch_indices"), other.get("benchmark_inputs", {}).get("resident_batch_indices")),
                "same_gpu_uuid": known_equal(se.get("environment", {}).get("torch_runtime", {}).get("gpu_uuid"), other.get("environment", {}).get("torch_runtime", {}).get("gpu_uuid")),
                "same_repository_source": known_equal(se.get("environment", {}).get("source_sha256"), other.get("environment", {}).get("source_sha256")),
                "same_software_versions": known_equal(se.get("environment", {}).get("packages"), other.get("environment", {}).get("packages"))}
            if baseline == "awq":
                conditions["same_calibration_images"] = known_equal(se.get("repository", {}).get("calibration", {}).get("image_tensor_sha256"), other.get("repository", {}).get("calibration", {}).get("image_tensor_sha256"))
            operations = []
            other_checks = {r["check"]: r for r in other.get("diagnostics", {}).get("operation_checks", [])}
            se_steps = se.get("diagnostics", {}).get("profiler_configuration", {}).get("steps")
            other_steps = other.get("diagnostics", {}).get("profiler_configuration", {}).get("steps")
            for check in se.get("diagnostics", {}).get("operation_checks", []):
                old = other_checks.get(check["check"])
                if old is None or not se_steps or not other_steps:
                    continue
                entry = {"check": check["check"], "causal_status": "not_established"}
                for field in ("calls", "cpu_self_us", "device_self_us"):
                    a, b = check.get(field), old.get(field)
                    entry[field + "_per_profiled_forward"] = {"se_aux": a/se_steps if a is not None else None,
                        "baseline": b/other_steps if b is not None else None,
                        "difference": a/se_steps-b/other_steps if a is not None and b is not None else None}
                operations.append(entry)
            comparisons.append({"batch_size": batch_size, "baseline": baseline, "same_input_bank": same_inputs,
                "comparison_conditions": conditions,
                "status": "observed" if all(v is True for v in conditions.values()) else "not_comparable" if any(v is False for v in conditions.values()) else "conditions_unverified", "metrics": metrics,
                "diagnostic_operation_comparisons": operations,
                "causal_explanation": "not_established", "limitation": "Separate processes and actual production backends; operator presence or summed durations alone do not establish cause."})
    return comparisons


def print_report(report):
    print("\n" + "=" * 78)
    print("TransUNet quantization profiling — " + report["status"])
    print(f"{'Method':<10} {'Batch':>5} {'Status':<9} {'Host ms/call':>13} {'img/s':>12} {'Peak MiB':>12}")
    for case in report.get("cases", []):
        benchmark = case.get("benchmark")
        if benchmark:
            latency = benchmark["latency_host_pooled_ms"]["mean"]
            throughput = benchmark["throughput_run_summary_images_per_s"]["mean"]
            peak = case.get("memory", {}).get("peak_allocated_bytes")
            peak_text = f"{peak / 2**20:.2f}" if peak is not None else "unavailable"
            print(f"{case['method']:<10} {case['batch_size']:>5} {case['status']:<9} {latency:>13.4f} {throughput:>12.3f} {peak_text:>12}")
        else:
            print(f"{case['method']:<10} {case['batch_size']:>5} {case['status']:<9} {'—':>13} {'—':>12} {'—':>12}")
        for error in case.get("errors", []):
            print("  Error:", error.get("message"))
        diagnostic = case.get("diagnostics", {})
        backend = case.get("repository", {}).get("conversion", {}).get("backend", "unresolved")
        print("  Backend:", backend)
        if case.get("phases"):
            print("  Setup/measurement phases (ms):", ", ".join(f"{p['phase']}={p.get('wall_time_ms', 0):.2f}" for p in case["phases"]))
        for warning in case.get("repository", {}).get("warnings", []):
            print("  Repository warning:", warning)
        operators = sorted(diagnostic.get("operator_aggregates", []),
                           key=lambda row: row.get("device_self_us") or row.get("cpu_self_us") or 0, reverse=True)[:5]
        for op in operators:
            print(f"  Operator {op['name']}: calls={op['count']}, CPU self us={op.get('cpu_self_us')}, device self us={op.get('device_self_us')}")
        kernels = sorted(diagnostic.get("kernel_summary", {}).get("kernels", []),
                         key=lambda row: row.get("summed_duration_us", 0), reverse=True)[:5]
        for kernel in kernels:
            print(f"  Kernel {kernel['name']}: module={kernel.get('module')}, calls={kernel['count']}, summed diagnostic us={kernel['summed_duration_us']:.2f}")
        for warning in diagnostic.get("warnings", []):
            print("  Diagnostic warning:", warning)
        # Console diagnostic rendering is deliberately derived from the saved object.
        for check in diagnostic.get("operation_checks", []):
            if isinstance(check, dict):
                summary = {key: check.get(key) for key in ("check", "status", "calls", "cpu_self_us", "device_self_us", "causal_status")}
                print("  Check:", json.dumps(summary, ensure_ascii=False))
        shapes = diagnostic.get("projection_wrapper_shapes", [])
        expanded = [row for row in shapes if row.get("additional_rows", 0) > 0]
        if expanded:
            first = expanded[0]
            print(f"  Row expansion observed in {len(expanded)} wrappers; first: {first['wrapper']} {first['wrapper_logical_rows']} -> {first['inner_logical_rows']} rows")
    print("Latency is per model call; only batch 1 is single-image latency.")
    print("Detailed traces are diagnostic runs; headline timings and memory are unprofiled.")
    for comparison in report.get("comparisons", []):
        print(f"SE-AUX vs {comparison['baseline']} batch {comparison['batch_size']} [{comparison['status']}]:")
        for metric, values in comparison["metrics"].items():
            percent = values["percent_change_from_baseline"]
            print(f"  {metric}: delta={values['difference_se_minus_baseline']:.4f}; change={percent:.2f}%" if percent is not None else f"  {metric}: percent unavailable")
    print("Causal attribution of the runtime gap remains not established by operator presence alone.")
    print("JSON:", report["result_path"])


def controller(args):
    out = Path(args.output_dir).expanduser().resolve()
    if (out / "profiling_results.json").exists():
        raise ValueError(f"Refusing to overwrite existing report: {out}. Choose a new --output-dir.")
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "profiling_results.json"
    report = {"schema_version": SCHEMA_VERSION, "status": "running", "started_at": utc_now(),
        "invocation": {"argv": sys.argv, "shell_display": shlex.join([sys.executable] + sys.argv)},
        "configuration": vars(args), "environment": environment_metadata(args.repo_root),
        "result_path": str(report_path), "cases": [], "comparisons": [], "warnings": [],
        "protocol": {"process_isolation": "Sequential fresh subprocess per method and batch size",
            "order": "Batch size order followed by supplied method order", "measurement_scope": "Forward only",
            "calibration_repeated_per_case": True,
            "no_accuracy_metrics": "Dice/IoU/HD95 are not evaluated; this script profiles inference."}}
    atomic_json(report_path, report)
    try:
        for batch_size in args.batch_sizes:
            for method in args.methods:
                case_id = f"{method}_b{batch_size}"
                case_dir = out / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                resolved = dict(vars(args), method=method, batch_size=batch_size)
                config_path, result_path = case_dir / "worker_config.json", case_dir / "result.json"
                atomic_json(config_path, {"args": resolved, "result_path": str(result_path), "case_id": case_id})
                command = [sys.executable, str(Path(__file__).resolve()), "--_worker-config", str(config_path)]
                print(f"\n[{case_id}] launching isolated worker", flush=True)
                with (case_dir / "console.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, bufsize=1, cwd=args.repo_root)
                    try:
                        for line in process.stdout:
                            print(line, end="", flush=True); log.write(line); log.flush()
                        returncode = process.wait()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
                        raise
                if result_path.is_file():
                    case = json.loads(result_path.read_text(encoding="utf-8"))
                    if case.get("status") == "running":
                        case["status"] = "failed"
                        case.setdefault("errors", []).append({"message": "Worker exited before finalizing; see console.log"})
                else:
                    case = {"case_id": case_id, "method": method, "batch_size": batch_size,
                            "status": "failed", "errors": [{"message": "Worker produced no JSON; see console.log"}]}
                case["worker_returncode"] = returncode
                case["console_log"] = str(case_dir / "console.log")
                report["cases"].append(case)
                report["comparisons"] = compare_cases(report["cases"])
                atomic_json(report_path, report)
        statuses = [c["status"] for c in report["cases"]]
        report["status"] = "ok" if all(s == "ok" for s in statuses) else "failed" if all(s == "failed" for s in statuses) else "partial"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["finished_at"] = utc_now()
        atomic_json(report_path, report)
        print_report(report)
    return 0 if report["status"] == "ok" else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=".", help="TransUNet root containing test.py and src/")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 36])
    parser.add_argument("--device", default="cuda:0", help="Logical CUDA device after CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--resident-batches", type=int, default=1)
    parser.add_argument("--memory-runs", type=int, default=3)
    parser.add_argument("--memory-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-stacks", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cudnn-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic", action="store_true", help="Explicit synthetic benchmark input smoke test, still requires trained checkpoint")
    parser.add_argument("--quick", action="store_true", help="Override timing to warmup5/iterations10/runs2, profile3, memory1x2; calibration unchanged")
    parser.add_argument("--check-environment", action="store_true", help="Write environment JSON without loading a model")
    parser.add_argument("--output-dir", default="profiling/" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--_worker-config", help=argparse.SUPPRESS)
    add_repository_arguments(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args._worker_config:
        return worker(args._worker_config)
    args.repo_root = str(Path(args.repo_root).expanduser().resolve())
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    if args.check_environment:
        path = Path(args.output_dir) / "environment.json"
        metadata = environment_metadata(args.repo_root, with_torch=True)
        atomic_json(path, metadata)
        print(json.dumps(safe_json(metadata), indent=2, allow_nan=False))
        print("Environment report:", path)
        return 0
    if not args.checkpoint:
        parser.error("--checkpoint is required for model profiling")
    checkpoint = Path(args.checkpoint).expanduser()
    args.checkpoint = str((Path(args.repo_root) / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve())
    if not args.device.startswith("cuda"):
        parser.error("--device must be a CUDA device; no CPU substitution is performed")
    if args.synthetic_calibration and not args.synthetic:
        parser.error("--synthetic-calibration requires --synthetic")
    if args.quick:
        args.warmup, args.iterations, args.runs = 5, 10, 2
        args.profile_steps, args.profile_warmup = 3, 1
        args.memory_runs, args.memory_steps = 1, 2
    for name in ("iterations", "runs", "resident_batches", "memory_runs", "memory_steps", "profile_steps", "calibration_batches", "input_samples", "img_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if min(args.batch_sizes) < 1 or args.warmup < 0 or args.profile_warmup < 0:
        parser.error("batch sizes must be positive; warmup counts must be nonnegative")
    if len(set(args.methods)) != len(args.methods) or len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("methods and batch sizes must not contain duplicates")
    if not (Path(args.repo_root) / "test.py").is_file():
        parser.error("--repo-root must point to the TransUNet repository")
    return controller(args)


def add_repository_arguments(parser):
    """Repository-specific flags; no optional dependency is imported for --help."""
    parser.add_argument("--dataset", choices=("Synapse", "ACDC", "Cataract1k"), default="Synapse")
    parser.add_argument("--checkpoint", help="Path to the shared FP checkpoint; keep its .se_aux.json sidecar alongside it")
    parser.add_argument("--volume-path", "--volume_path", dest="volume_path", default=None)
    parser.add_argument("--list-dir", "--list_dir", dest="list_dir", default=None)
    parser.add_argument("--calibration-batches", type=int, default=8,
                        help="Raw dataset items: a complete volume or one RGB frame, not a count of slices")
    parser.add_argument("--input-samples", type=int, default=36,
                        help="CPU benchmark bank size; dataset items repeat deterministically when necessary")
    parser.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    parser.add_argument("--vit-name", "--vit_name", dest="vit_name", default="R50-ViT-B_16")
    parser.add_argument("--n-skip", type=int, default=3)
    parser.add_argument("--vit-patches-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--synthetic-calibration", action="store_true",
                        help="Explicit smoke-test-only permission to use random calibration; requires --synthetic")
    parser.add_argument("--normalize-present-class-eval", action="store_true",
                        help="Match the existing INC RGB-calibration normalization flag; does not change legacy SE or benchmark collation")
    return parser


def _repo_sample_identity(sample, index):
    """Keep only identifiers and shapes, never raw medical images, in JSON."""
    image = sample["image"]
    name = sample.get("case_name", "dataset_index_{}".format(index))
    if isinstance(name, (tuple, list)):
        name = name[0] if name else "dataset_index_{}".format(index)
    return {"dataset_index": int(index), "case_name": str(name),
            "raw_image_shape": [int(v) for v in image.shape]}


def _repo_digest_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_repository_case(args, phase):
    """Build the real se_loss deployment path and its fixed CPU benchmark bank.

    The caller selects the CUDA device, seeds RNGs, and supplies a phase(name)
    context manager. Returns (model_on_current_cuda_device, cpu_bank, metadata).
    The raw calibration iterable is cached on CPU and shared across repeated
    passes by the existing quantizers; it is released before this function exits.
    """
    import copy
    import gc
    import inspect
    import os
    import sys
    from collections.abc import Mapping
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    from torch.utils.data import DataLoader, Subset

    root = Path(args.repo_root).expanduser().resolve()
    if not (root / "networks" / "vit_seg_modeling.py").is_file():
        raise FileNotFoundError("--repo-root must identify the TransUNet repository root")
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    # Production modules use relative paths for optional resources and lists.
    os.chdir(root)
    if not torch.cuda.is_available():
        raise RuntimeError("Real deployment profiling requires CUDA; no CPU/backend substitution is performed")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required, including when synthetic benchmark inputs are requested")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(checkpoint))
    if args.calibration_batches < 1 or args.input_samples < 1:
        raise ValueError("--calibration-batches and --input-samples must be positive")
    if args.img_size < 1 or args.vit_patches_size < 1:
        raise ValueError("Image and patch sizes must be positive")
    if args.synthetic_calibration and not args.synthetic:
        raise ValueError("--synthetic-calibration requires --synthetic; this is a smoke test, not a research comparison")

    from networks.vit_seg_modeling import CONFIGS, VisionTransformer
    from src.se_auxiliary import DEFAULT_SE_AUX_CONFIG, load_se_metadata, validate_se_calibration_config

    metadata = {
        "adapter_reference_commit": "770a6a87f8a9746ab879409acd25ee4761a640a6",
        "checkpoint": str(checkpoint),
        "method": args.method,
        "warnings": [],
        "full_token_sequence": True,
        "benchmark_input_source": "synthetic" if args.synthetic else "real_dataset",
        "synthetic_calibration": bool(args.synthetic_calibration),
        "setup_phase_limitation": "Quantizer public APIs combine calibration, scale search, and conversion; these are timed as one phase without modifying production code.",
        "source_paths": {
            "model": "networks/vit_seg_modeling.py",
            "benchmark_preprocessing": "src/benchmark.py:collate_for_benchmark",
            "awq": "src/inc_awq.py:quantize_inc_awq",
            "se_aux": "networks/quantizer.py:AWQViTSegQuantizer.quantize",
            "fp16": "src/fp16_projections.py:convert_fp16_projections",
        },
    }
    with phase("checkpoint_cpu_load"):
        metadata["checkpoint_sha256"] = file_sha256(checkpoint)
        sidecar = Path(str(checkpoint) + ".se_aux.json")
        metadata["se_sidecar_sha256"] = file_sha256(sidecar) if sidecar.is_file() else None
        load_kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = True
        state = torch.load(str(checkpoint), **load_kwargs)
        if not isinstance(state, Mapping) or not state or not all(
            isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()
        ):
            raise ValueError("Expected the repository's raw tensor state_dict checkpoint; wrapped/serialized models are not silently unpacked")
        if all(name.startswith("module.") for name in state):
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, "module.")
            metadata["removed_dataparallel_key_prefix"] = True
        else:
            metadata["removed_dataparallel_key_prefix"] = False
        has_se = any(name.startswith("transformer.encoder.SELayer.") for name in state)
        metadata["checkpoint_contains_se_parameters"] = has_se
        if args.method == "se_aux" and not has_se:
            raise ValueError("SE-AUX requires checkpoint SE predictor parameters; random SE gates will not be created")

    classes = {"Synapse": 9, "ACDC": 4, "Cataract1k": 5}
    default_roots = {
        "Synapse": "/data/halyusuf/data/Synapse/test_vol_h5",
        "ACDC": "/data/halyusuf/data/ACDC/",
        "Cataract1k": "/data/halyusuf/data/CataractData/",
    }
    volume_path = str(Path(args.volume_path or default_roots[args.dataset]).expanduser().resolve())
    list_dir = str(Path(args.list_dir or root / "lists" / "lists_Synapse").expanduser().resolve()) if args.dataset == "Synapse" else None
    local_args = SimpleNamespace(
        dataset=args.dataset, volume_path=volume_path, list_dir=list_dir,
        img_size=args.img_size, num_classes=classes[args.dataset],
        use_se_block=has_se, drop_se_block=False,
        quantize=args.method in ("awq", "se_aux"),
        quantize_backend="inc_awq_packed" if args.method == "awq" else "legacy",
        quantize_calibrate_batch_size=args.calibration_batches,
        fp16_projections=args.method == "fp16", verbose=False,
        normalize_present_class_eval=args.normalize_present_class_eval,
        normalize_endovis_eval=args.normalize_present_class_eval,
    )
    with phase("model_construction_and_checkpoint_assignment"):
        if args.vit_name not in CONFIGS:
            raise ValueError("Unknown --vit-name: {}".format(args.vit_name))
        config = copy.deepcopy(CONFIGS[args.vit_name])
        settings = {
            "verbose": False, "n_classes": classes[args.dataset], "n_skip": args.n_skip,
            "use_se_block": has_se, "drop_se_block": False, "se_aux_loss": False,
            "gumbel_sampling_mode": "dist", "topk_attn": 0.0,
            "use_ats": False, "use_gumbel_topk": False,
            "use_shsa": False, "use_alternate_shsa": False,
            "use_efficientnet": False, "use_swin": False,
        }
        for name, value in settings.items():
            setattr(config, name, value)
        config.patches.size = (args.vit_patches_size, args.vit_patches_size)
        if args.num_heads is not None:
            config.transformer.num_heads = args.num_heads
        if args.num_layers is not None:
            config.transformer.num_layers = args.num_layers
        if "R50" in args.vit_name:
            config.patches.grid = (args.img_size // args.vit_patches_size,) * 2
        model = VisionTransformer(config, img_size=args.img_size, num_classes=classes[args.dataset]).cuda()
        model.load_state_dict(state, strict=True)
        del state
        se_metadata = load_se_metadata(model, str(checkpoint))
        model.eval()
    metadata["architecture"] = dict(settings, vit_name=args.vit_name, img_size=args.img_size,
                                    patch_size=args.vit_patches_size,
                                    num_heads=int(config.transformer.num_heads),
                                    num_layers=int(config.transformer.num_layers))
    metadata["se_checkpoint_metadata"] = se_metadata
    metadata["se_metadata_sidecar"] = str(checkpoint) + ".se_aux.json"
    if has_se and se_metadata is None:
        metadata["warnings"].append("SE sidecar is absent: the production loader selects legacy mean pooling. Preserve the original .se_aux.json sidecar for RMS-trained checkpoints.")
    metadata["resolved_dataset"] = {"name": args.dataset, "volume_path": volume_path,
                                    "list_dir": list_dir, "num_classes": classes[args.dataset]}

    # Get actual repository datasets lazily, only when real data are needed.
    needs_calibration = args.method in ("awq", "se_aux")
    needs_real_data = not args.synthetic or (needs_calibration and not args.synthetic_calibration)
    if needs_real_data:
        if args.dataset == "Synapse":
            from datasets.dataset_synapse import Synapse_dataset
            local_args.Dataset = Synapse_dataset
        elif args.dataset == "ACDC":
            from datasets.dataset_acdc import ACDC_Dataset
            local_args.Dataset = ACDC_Dataset
        else:
            from datasets.dataset_cataract import Cataract1kDataset
            local_args.Dataset = Cataract1kDataset

    with phase("benchmark_input_preparation"):
        if args.synthetic:
            generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
            bank = torch.randn(args.input_samples, 3, args.img_size, args.img_size, generator=generator)
            identities = [{"synthetic_index": i} for i in range(args.input_samples)]
            metadata["benchmark_preprocessing"] = {"kind": "synthetic_standard_normal", "shape": list(bank.shape)}
        else:
            from src.benchmark import build_benchmark_loader
            loader = build_benchmark_loader(local_args, batch_size=args.input_samples, num_workers=0, shuffle=False)
            dataset = loader.dataset
            if len(dataset) == 0:
                raise ValueError("The benchmark dataset is empty")
            samples = [dataset[i % len(dataset)] for i in range(args.input_samples)]
            identities = [_repo_sample_identity(item, i % len(dataset)) for i, item in enumerate(samples)]
            bank = loader.collate_fn(samples).detach().cpu().contiguous()
            metadata["benchmark_preprocessing"] = {
                "kind": "existing_repository_benchmark_collate",
                "function": "src.benchmark.collate_for_benchmark",
                "split": "val" if args.dataset == "Cataract1k" else "test_vol",
                "resize": "bilinear, align_corners=False, only when H/W differ",
                "grayscale": "expand singleton channel to three channels",
                "volume_selection": "existing shape heuristic: center slice unless first dimension is 1, 3, or 4",
                "normalization": "none added by benchmark collate",
                "sample_count": len(identities),
                "bounded_bank_reused": True,
            }
            if args.dataset == "Cataract1k":
                metadata["warnings"].append("Existing benchmark asks for split='val'; current Cataract1kDataset maps all non-train splits, including 'val' and 'test', to test.csv.")
            del loader, dataset, samples
        if bank.ndim != 4 or not torch.isfinite(bank).all():
            raise ValueError("Prepared benchmark input bank must be finite NCHW")
        metadata["benchmark_inputs"] = {"identifiers": identities, "shape": list(bank.shape),
                                        "dtype": str(bank.dtype), "sha256": _repo_digest_tensor(bank)}

    calibration_batches = None
    if needs_calibration:
        with phase("calibration_input_preparation"):
            if args.synthetic_calibration:
                generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1)
                calibration_batches = []
                for i in range(args.calibration_batches):
                    if args.dataset in ("Synapse", "ACDC"):
                        raw = torch.randn(1, 1, args.img_size, args.img_size, generator=generator)
                    else:
                        raw = torch.rand(1, args.img_size, args.img_size, 3, generator=generator) * 255.0
                    calibration_batches.append({"image": raw, "case_name": ["synthetic_calibration_{}".format(i)]})
                metadata["warnings"].append("Synthetic calibration is explicitly enabled. Results are smoke-test diagnostics and must not be used as real-data accuracy or calibration evidence.")
            else:
                split = "test_vol" if args.dataset in ("Synapse", "ACDC") else "test"
                kwargs = {"base_dir": volume_path, "split": split}
                if args.dataset in ("Synapse", "ACDC"):
                    kwargs["list_dir"] = list_dir
                raw_dataset = local_args.Dataset(**kwargs)
                if len(raw_dataset) < args.calibration_batches:
                    raise ValueError("Requested {} complete calibration cases but only {} exist".format(args.calibration_batches, len(raw_dataset)))
                raw_loader = DataLoader(Subset(raw_dataset, range(args.calibration_batches)),
                                        batch_size=1, shuffle=False, num_workers=0)
                calibration_batches = list(raw_loader)
                del raw_loader, raw_dataset
            metadata["calibration"] = {
                "requested_loader_batches": args.calibration_batches,
                "actual_loader_batches": len(calibration_batches),
                "loader_batch_size": 1,
                "case_unit": "entire_volume_all_slices" if args.dataset in ("Synapse", "ACDC") else "one_HWC_frame",
                "identifiers": [_repo_sample_identity(batch, i) for i, batch in enumerate(calibration_batches)],
                "image_tensor_sha256": [_repo_digest_tensor(batch["image"]) for batch in calibration_batches],
                "total_images_after_volume_slicing": sum(int(batch["image"].shape[1]) if args.dataset in ("Synapse", "ACDC") else 1 for batch in calibration_batches),
                "resize": "existing quantizer-specific scipy.ndimage.zoom order=3 path",
                "sampling": "first N raw dataset items, shuffle=False; identifiers and hashes recorded",
                "split": "synthetic" if args.synthetic_calibration else ("test_vol" if args.dataset in ("Synapse", "ACDC") else "test"),
            }
            if args.normalize_present_class_eval and args.dataset == "Cataract1k":
                metadata["warnings"].append("The existing INC path applies requested ImageNet normalization in calibration; legacy SE and benchmark collation do not. This inherited difference is preserved and disclosed.")

        with phase("calibration_scale_search_and_packing_combined"):
            with torch.no_grad():
                if args.method == "awq":
                    from src.inc_awq import quantize_inc_awq
                    from neural_compressor.torch.utils import get_accelerator
                    inc_device = str(get_accelerator().current_device_name())
                    metadata["inc_calibration_device"] = inc_device
                    if torch.device(inc_device) != next(model.parameters()).device:
                        raise RuntimeError("INC selected {} but the model is on {}; check INC_TARGET_DEVICE before profiling".format(inc_device, next(model.parameters()).device))
                    model = quantize_inc_awq(model, calibration_batches, args.calibration_batches, local_args)
                    metadata["conversion"] = {
                        "function": "src.inc_awq.quantize_inc_awq", "backend": "inc_awq_packed",
                        "bits": 4, "group_size": 128, "group_dim": 1, "use_sym": False,
                        "use_auto_scale": True, "use_auto_clip": True, "folding": False,
                        "scope": "all selected full-channel attention and FFN projections",
                        "scaling_scope": "INC automatic scaling; inspect deployed MulLinear wrappers and source for exact sites",
                    }
                else:
                    from networks.quantizer import AWQViTSegQuantizer
                    validate_se_calibration_config(model, bits=DEFAULT_SE_AUX_CONFIG["bits"],
                                                   group_size=DEFAULT_SE_AUX_CONFIG["group_size"],
                                                   zero_point=DEFAULT_SE_AUX_CONFIG["zero_point"])
                    quantizer = AWQViTSegQuantizer(model=model, calib_loader=calibration_batches,
                                                  w_bit=DEFAULT_SE_AUX_CONFIG["bits"],
                                                  q_group_size=DEFAULT_SE_AUX_CONFIG["group_size"],
                                                  n_calib_batches=args.calibration_batches,
                                                  device=torch.device("cuda", torch.cuda.current_device()),
                                                  args=local_args)
                    model = quantizer.quantize()
                    del quantizer
                    metadata["conversion"] = {
                        "function": "networks.quantizer.AWQViTSegQuantizer.quantize", "backend": "legacy_se_guided_packed",
                        "bits": 4, "group_size": 128, "zero_point": True,
                        "scope": "eligible Q/K/V, attention-output, FC1 and FC2 projections",
                        "scaling_scope": "Q/K/V and FC1; inverse scales folded into preceding LayerNorm",
                        "selection": "one case's gates chosen by mean full-model logits MSE over all cached calibration cases",
                    }
        del calibration_batches

    with phase("deployment_cleanup_and_precision_conversion"):
        encoder = model.transformer.encoder
        if getattr(encoder, "SELayer", None) is not None:
            del encoder.SELayer
        encoder.args.drop_se_block = True
        encoder.se_aux_loss = False
        if args.method == "fp16":
            from src.fp16_projections import convert_fp16_projections
            metadata["conversion"] = convert_fp16_projections(model)
            metadata["conversion"]["backend"] = "fp16_transformer_projections"
        elif args.method == "fp32":
            metadata["conversion"] = {"backend": "native_fp32", "scope": "none"}
        model.eval()
        metadata["se_modules_present_at_inference"] = getattr(encoder, "SELayer", None) is not None
        if metadata["se_modules_present_at_inference"]:
            raise RuntimeError("Auxiliary SE modules remain in the deployed model")
        if any(bool(getattr(model.args, flag, False)) for flag in ("use_ats", "use_gumbel_topk")) or float(model.args.topk_attn) != 0.0:
            raise RuntimeError("Token reduction was unexpectedly enabled")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return model, bank, metadata


def _diag_scalar(value):
    """JSON-safe metadata only; never copy a tensor to the host from a hook."""
    import math
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_diag_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _diag_scalar(v) for k, v in value.items()}
    return str(value)


def _diag_tensor_metadata(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape), "dtype": str(value.dtype),
            "device": str(value.device), "stride": list(value.stride()),
            "contiguous": value.is_contiguous(), "numel": value.numel(),
            "logical_bytes": value.numel() * value.element_size(),
        }
    if isinstance(value, dict):
        return {str(k): _diag_tensor_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diag_tensor_metadata(v) for v in value]
    return {"type": type(value).__name__}


def _diag_projection(name):
    import re
    return bool(re.search(
        r"(?:^|\.)(?:attn\.(?:query|key|value|out|qkv|proj)|ffn\.fc[12])(?:\.|$)", name))


def _diag_source(obj):
    import inspect
    try:
        lines, start = inspect.getsourcelines(obj)
        return {"file": inspect.getsourcefile(obj), "line": start,
                "source": "".join(lines), "unavailable_reason": None}
    except (OSError, TypeError) as exc:
        return {"file": None, "line": None, "source": None,
                "unavailable_reason": str(exc)}


def _diag_inventory(model):
    import torch
    modules, sources = [], {}
    attributes = ("in_features", "out_features", "w_bit", "bits", "weight_bits",
                  "group_size", "q_group_size", "interleave", "kstride",
                  "zero_point", "num_heads", "head_dim")
    for name, module in model.named_modules():
        cls = type(module)
        qualified = cls.__module__ + "." + cls.__qualname__
        row = {"name": name, "class": qualified, "projection": _diag_projection(name),
               "training": bool(module.training), "attributes": {},
               "parameters": {}, "buffers": {}, "unregistered_tensor_attributes": {}}
        for key in attributes:
            value = getattr(module, key, None)
            if value is not None and not isinstance(value, torch.Tensor):
                row["attributes"][key] = _diag_scalar(value)
        registered = set()
        for kind, collection in (("parameters", module.named_parameters(recurse=False)),
                                 ("buffers", module.named_buffers(recurse=False))):
            for key, tensor in collection:
                row[kind][key] = _diag_tensor_metadata(tensor)
                registered.add(key)
        for key, value in vars(module).items():
            if isinstance(value, torch.Tensor) and key not in registered:
                row["unregistered_tensor_attributes"][key] = _diag_tensor_metadata(value)
        if qualified not in sources:
            sources[qualified] = _diag_source(cls.forward) if row["projection"] else {
                "file": None, "line": None, "source": None,
                "unavailable_reason": "Forward source captured for projection implementations only."}
        elif row["projection"] and sources[qualified]["source"] is None:
            sources[qualified] = _diag_source(cls.forward)
        modules.append(row)
    return {"modules": modules, "implementation_sources": sources,
            "notes": ["Tensor byte counts are logical per-attribute sizes, not a deduplicated model payload.",
                      "Unregistered direct tensor attributes are inspected; arbitrary external CUDA allocations are not."]}


def _diag_cpu_tensors(value, prefix="output"):
    import torch
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu().clone()}
    if isinstance(value, (tuple, list)):
        result = {}
        for i, child in enumerate(value):
            result.update(_diag_cpu_tensors(child, "%s[%d]" % (prefix, i)))
        return result
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_diag_cpu_tensors(child, "%s[%s]" % (prefix, key)))
        return result
    return {}


def _diag_attr(event, names):
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return _diag_scalar(value)
    return None


def _diag_operator_rows(profiler, cuda_traced):
    rows = []
    for index, event in enumerate(profiler.key_averages(group_by_input_shape=True)):
        rows.append({
            "id": "op%06d" % index, "name": event.key, "count": event.count,
            "input_shapes": _diag_scalar(getattr(event, "input_shapes", None)),
            "cpu_self_us": _diag_attr(event, ("self_cpu_time_total",)),
            "cpu_total_us": _diag_attr(event, ("cpu_time_total",)),
            "device_self_us": _diag_attr(event, ("self_device_time_total", "self_cuda_time_total")) if cuda_traced else None,
            "device_total_us": _diag_attr(event, ("device_time_total", "cuda_time_total")) if cuda_traced else None,
            "cpu_self_memory_bytes": _diag_attr(event, ("self_cpu_memory_usage",)),
            "cpu_memory_bytes": _diag_attr(event, ("cpu_memory_usage",)),
            "device_self_memory_bytes": _diag_attr(event, ("self_device_memory_usage", "self_cuda_memory_usage")) if cuda_traced else None,
            "device_memory_bytes": _diag_attr(event, ("device_memory_usage", "cuda_memory_usage")) if cuda_traced else None,
            "source_stack": _diag_scalar(getattr(event, "stack", None)),
            "device_values_unavailable_reason": None if cuda_traced else "CUDA kernel tracing was unavailable or no CUDA kernels were captured.",
        })
    return rows


def _diag_parse_trace(path):
    """Keep actual trace identifiers; correlate launches, never kernel-name guesses."""
    import bisect
    import collections
    import json
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", [])
    ranges, starts = collections.defaultdict(list), {}
    for index, event in enumerate(events):
        if event.get("ph") == "X" and str(event.get("name", "")).startswith("module::"):
            ranges[(event.get("pid"), event.get("tid"))].append((event["ts"], event["ts"] + event.get("dur", 0), event["name"][8:], index))
    for thread, records in ranges.items():
        records.sort()
        starts[thread] = [r[0] for r in records]

    def parent_module(event):
        thread = (event.get("pid"), event.get("tid"))
        records = ranges.get(thread, [])
        if not records:
            return None
        pos = bisect.bisect_right(starts[thread], event.get("ts", -1))
        end = event.get("ts", -1) + event.get("dur", 0)
        for start, stop, name, index in reversed(records[:pos]):
            if stop >= end - 0.001:
                return {"module": name, "range_event_index": index}
        return None

    def arg(event, keys):
        values = event.get("args", {})
        return next((str(values[k]) for k in keys if k in values), None)

    external, launches, cpu_links = {}, {}, collections.defaultdict(list)
    for index, event in enumerate(events):
        category = str(event.get("cat", "")).lower()
        if event.get("ph") != "X":
            continue
        if category == "cpu_op":
            linked = {"event_index": index, "operator": event.get("name"),
                      "module_range": parent_module(event)}
            ext = arg(event, ("External id", "external id", "External ID"))
            if ext is not None:
                external[ext] = linked
            cpu_links[event.get("name")].append(linked)
        if category in ("cuda_runtime", "cuda_driver"):
            correlation = arg(event, ("correlation", "Correlation id", "correlation_id"))
            if correlation is not None:
                launches[correlation] = {"event_index": index,
                    "external_id": arg(event, ("External id", "external id", "External ID")),
                    "module_range": parent_module(event)}
    grouped = {}
    for index, event in enumerate(events):
        category = str(event.get("cat", "")).lower()
        if event.get("ph") != "X" or category not in ("kernel", "cuda_kernel", "gpu_kernel"):
            continue
        ext = arg(event, ("External id", "external id", "External ID"))
        correlation = arg(event, ("correlation", "Correlation id", "correlation_id"))
        launch = launches.get(correlation)
        op = external.get(ext) or (external.get(launch["external_id"]) if launch else None)
        module = op.get("module_range") if op else (launch.get("module_range") if launch else None)
        key = (event.get("name"), module["module"] if module else None)
        if key not in grouped:
            grouped[key] = {"name": key[0], "module": key[1], "count": 0,
                            "summed_duration_us": 0.0, "min_duration_us": None,
                            "max_duration_us": None, "events": []}
        item = grouped[key]
        duration = float(event.get("dur", 0))
        item["count"] += 1
        item["summed_duration_us"] += duration
        item["min_duration_us"] = duration if item["min_duration_us"] is None else min(item["min_duration_us"], duration)
        item["max_duration_us"] = duration if item["max_duration_us"] is None else max(item["max_duration_us"], duration)
        item["events"].append({"trace_event_index": index, "duration_us": duration,
            "correlation_id": correlation, "external_id": ext,
            "cpu_operator": op, "launch": launch,
            "attribution": "trace_external_id_or_launch_correlation" if op or module else "unresolved"})
    return {"trace_event_count": len(events), "kernels": list(grouped.values()),
            "cpu_operator_links": dict(cpu_links), "kernel_duration_note":
            "Summed kernel durations are diagnostic device work, not additive end-to-end wall time; kernels may overlap."}


def _diag_checks(operators, trace):
    patterns = {
        "padding": ("aten::pad", "aten::constant_pad_nd", "aten::reflection_pad", "aten::replication_pad"),
        "shape_or_view": ("aten::view", "aten::_unsafe_view", "aten::reshape", "aten::transpose", "aten::permute", "aten::slice", "aten::narrow"),
        "materialization_or_copy": ("aten::contiguous", "aten::clone", "aten::copy_"),
        "dtype_or_device_conversion": ("aten::to", "aten::_to_copy", "aten::type_as"),
        "explicit_dequantization_or_unpack": ("dequant", "unpack"),
        "explicit_packing": ("pack",),
        "matrix_multiplication": ("aten::linear", "aten::mm", "aten::bmm", "aten::matmul", "aten::addmm", "gemm", "gemv"),
    }
    checks = []
    for category, terms in patterns.items():
        matched = [r for r in operators if any(term in r["name"].lower() for term in terms)
                   and not (category == "explicit_packing" and "unpack" in r["name"].lower())]
        kernel_terms = {"padding": ("pad",), "explicit_dequantization_or_unpack": ("dequant", "unpack"),
                        "explicit_packing": ("pack",), "matrix_multiplication": ("gemm", "gemv", "matmul")}.get(category, ())
        kernel_candidates = [{"kernel_summary_index": index, "name": kernel["name"],
                              "module": kernel["module"], "count": kernel["count"]}
                             for index, kernel in enumerate(trace.get("kernels", []))
                             if any(term in str(kernel["name"]).lower() for term in kernel_terms)
                             and not (category == "explicit_packing" and "unpack" in str(kernel["name"]).lower())]
        links = []
        for name in sorted({row["name"] for row in matched}):
            links.extend(trace.get("cpu_operator_links", {}).get(name, []))
        checks.append({"check": category, "status": "observed" if matched or kernel_candidates else "not_observable",
            "operator_aggregate_ids": [r["id"] for r in matched], "trace_operator_links": links,
            "candidate_kernel_groups": kernel_candidates,
            "calls": sum(r["count"] for r in matched),
            "cpu_self_us": sum(r["cpu_self_us"] or 0 for r in matched) if matched else None,
            "device_self_us": sum(r["device_self_us"] or 0 for r in matched) if matched and any(r["device_self_us"] is not None for r in matched) else None,
            "causal_status": "not_established",
            "limitation": "Name filtering identifies candidates only; use linked module ranges, source, shapes and kernel evidence. Absence of a named operation does not exclude fused work. Self-time sums are not wall-clock contributions. Views do not imply a copy; casts may be no-ops."})
    return checks


def _diag_wrapper_shapes(observations):
    import math
    rows = []
    for name, wrapper in observations.items():
        for suffix in (".inner", ".linear"):
            inner = observations.get(name + suffix)
            if inner is None:
                continue
            outer_inputs, inner_inputs = wrapper.get("first_input"), inner.get("first_input")
            if not isinstance(outer_inputs, list) or not outer_inputs or not isinstance(inner_inputs, list) or not inner_inputs:
                continue
            outer, inside = outer_inputs[0], inner_inputs[0]
            outer_shape, inner_shape = outer.get("shape", []), inside.get("shape", [])
            if not outer_shape or not inner_shape:
                continue
            before, after = math.prod(outer_shape[:-1]), math.prod(inner_shape[:-1])
            rows.append({"wrapper": name, "inner": name + suffix,
                "wrapper_input": outer, "inner_input": inside,
                "wrapper_logical_rows": before, "inner_logical_rows": after,
                "additional_rows": after - before,
                "row_expansion_percent": 100.0 * (after - before) / before if before else None,
                "status": "observed", "causal_status": "not_established",
                "interpretation": "Logical row counts come from observed tensor shapes. Use wrapper source and correlated operations to determine whether expansion is padding; this does not measure its wall-clock contribution."})
    return rows


def collect_diagnostics(model, inputs, args, output_dir):
    """Separate diagnostic pass. Never use its durations as headline benchmarks."""
    import gc
    import json
    from pathlib import Path
    import torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "warnings": [], "trace_files": [],
              "module_inventory": _diag_inventory(model), "module_observations": {},
              "instrumentation_validation": None, "operator_aggregates": [],
              "kernel_summary": {}, "operation_checks": [],
              "measurement_note": "Diagnostic tracing includes instrumentation overhead; use separate unprofiled benchmarks for performance and memory comparisons."}
    if not inputs:
        raise ValueError("collect_diagnostics requires at least one input batch")
    hooks, active_ranges = [], {}
    observed = result["module_observations"]

    def pre(name):
        def enter(module, arguments):
            context = torch.profiler.record_function("module::" + name)
            context.__enter__()
            active_ranges.setdefault(name, []).append(context)
            if name not in observed:
                observed[name] = {"calls": 0, "first_input": _diag_tensor_metadata(arguments), "first_output": None}
            observed[name]["calls"] += 1
        return enter

    def post(name):
        def leave(module, arguments, output):
            if name in observed and observed[name]["first_output"] is None:
                observed[name]["first_output"] = _diag_tensor_metadata(output)
            stack = active_ranges.get(name, [])
            if stack:
                stack.pop().__exit__(None, None, None)
        return leave

    try:
        with torch.inference_mode():
            baseline = _diag_cpu_tensors(model(inputs[0]))
        for name, module in model.named_modules():
            if _diag_projection(name):
                hooks.append(module.register_forward_pre_hook(pre(name)))
                try:
                    hooks.append(module.register_forward_hook(post(name), always_call=True))
                except TypeError:
                    hooks.append(module.register_forward_hook(post(name)))
        with torch.inference_mode():
            instrumented = _diag_cpu_tensors(model(inputs[0]))
        comparisons = []
        for name in sorted(set(baseline) | set(instrumented)):
            left, right = baseline.get(name), instrumented.get(name)
            same_shape = left is not None and right is not None and left.shape == right.shape
            passed = bool(same_shape and torch.allclose(left, right,
                rtol=getattr(args, "diag_rtol", 1e-4), atol=getattr(args, "diag_atol", 1e-5), equal_nan=False))
            comparisons.append({"output": name, "shape_matches": same_shape, "allclose": passed,
                "finite": bool(torch.isfinite(right).all()) if right is not None else False,
                "max_abs_difference": float((left.double() - right.double()).abs().max()) if same_shape and left.numel() else None})
        result["instrumentation_validation"] = {"passed": bool(comparisons) and all(c["allclose"] and c["finite"] for c in comparisons),
            "rtol": getattr(args, "diag_rtol", 1e-4), "atol": getattr(args, "diag_atol", 1e-5), "outputs": comparisons}
        del baseline, instrumented
        if comparisons:
            del left, right
        gc.collect()
        if not result["instrumentation_validation"]["passed"]:
            raise RuntimeError("Diagnostic instrumentation output validation failed")
        supported = torch.profiler.supported_activities() if hasattr(torch.profiler, "supported_activities") else set()
        cuda_requested = bool(inputs[0].is_cuda)
        cuda_supported = cuda_requested and torch.profiler.ProfilerActivity.CUDA in supported
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda_supported:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        elif cuda_requested:
            result["warnings"].append("CUDA profiler activity is unavailable; CPU diagnostic trace only.")
        with torch.inference_mode():
            for index in range(args.profile_warmup):
                model(inputs[index % len(inputs)])
            if cuda_requested:
                torch.cuda.synchronize(inputs[0].device)
            observed.clear()
            with torch.profiler.profile(activities=activities, record_shapes=True,
                    profile_memory=True, with_stack=args.profile_stacks) as profiler:
                for index in range(args.profile_steps):
                    with torch.profiler.record_function("profile_forward_%d" % index):
                        model(inputs[index % len(inputs)])
                    profiler.step()
                if cuda_requested:
                    torch.cuda.synchronize(inputs[0].device)
        path = output_dir / "inference_trace.json"
        profiler.export_chrome_trace(str(path))
        result["trace_files"].append(str(path.resolve()))
        try:
            trace = _diag_parse_trace(path)
        except Exception as exc:
            trace = {"kernels": [], "cpu_operator_links": {}, "parse_error": repr(exc)}
            result["warnings"].append("Trace parsing failed: " + repr(exc))
        cuda_traced = bool(trace.get("kernels"))
        if cuda_requested and not cuda_traced:
            result["warnings"].append("No CUDA kernel events could be extracted. Kernel costs are unavailable, not zero.")
        result["operator_aggregates"] = _diag_operator_rows(profiler, cuda_traced)
        result["kernel_summary"] = trace
        result["operation_checks"] = _diag_checks(result["operator_aggregates"], trace)
        result["projection_wrapper_shapes"] = _diag_wrapper_shapes(observed)
        result["profiler_configuration"] = {"steps": args.profile_steps, "warmup_steps": args.profile_warmup,
            "record_shapes": True, "profile_memory": True, "with_stack": args.profile_stacks,
            "cuda_requested": cuda_requested, "cuda_activity_supported": cuda_supported,
            "cuda_kernel_events_observed": cuda_traced, "module_observation_scope": "active profiling forwards only"}
        result["status"] = "completed" if not result["warnings"] else "completed_with_warnings"
    except Exception as exc:
        import traceback
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        for hook in hooks:
            hook.remove()
        for stack in active_ranges.values():
            while stack:
                stack.pop().__exit__(None, None, None)
    path = output_dir / "diagnostics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_diag_scalar(result), handle, indent=2, allow_nan=False)
        handle.write("\n")
    result["diagnostics_file"] = str(path.resolve())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
