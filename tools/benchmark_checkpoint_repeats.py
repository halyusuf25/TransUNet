#!/usr/bin/env python3
"""Repeat checkpoint throughput and p99-latency benchmarking."""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark import (  # noqa: E402
    _extract_images,
    _set_cudnn_benchmark,
    _timed_forward_cpu,
    _timed_forward_gpu,
    build_benchmark_loader,
)
from datasets.dataset_acdc import ACDC_Dataset  # noqa: E402
from datasets.dataset_cataract import Cataract1kDataset  # noqa: E402
from datasets.dataset_endovis2018 import EndoVis2018Dataset  # noqa: E402
from datasets.dataset_synapse import Synapse_dataset  # noqa: E402
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg  # noqa: E402
from networks.vit_seg_modeling import VisionTransformer as ViT_seg  # noqa: E402
from src.test_helpers import build_test_arg_parser  # noqa: E402
from utils import _make_json_safe, _sanitize_name  # noqa: E402


TEST_DATASET_CONFIG = {
    "Synapse": {
        "Dataset": Synapse_dataset,
        "volume_path": "/data/halyusuf/data/Synapse/test_vol_h5",
        "list_dir": "./lists/lists_Synapse",
        "num_classes": 9,
        "z_spacing": 1,
        "class_names": [
            "Background",
            "Aorta",
            "Gallbladder",
            "Kidney(L)",
            "Kidney(R)",
            "Liver",
            "Pancreas",
            "Spleen",
            "Stomach",
        ],
    },
    "ACDC": {
        "Dataset": ACDC_Dataset,
        "volume_path": "/data/halyusuf/data/ACDC/",
        "list_dir": None,
        "num_classes": 4,
        "z_spacing": 5,
        "class_names": [
            "Background",
            "Right Ventricle",
            "Myocardium",
            "Left Ventricle",
        ],
    },
    "Cataract1k": {
        "Dataset": Cataract1kDataset,
        "volume_path": "/data/halyusuf/data/CataractData/",
        "list_dir": None,
        "num_classes": 5,
        "z_spacing": 1,
        "class_names": [
            "Background",
            "Pupil",
            "Cornea",
            "Lens",
            "Instruments",
        ],
    },
    "EndoVis2018": {
        "Dataset": EndoVis2018Dataset,
        "volume_path": "/data/halyusuf/data/EndoVis_2018",
        "list_dir": None,
        "num_classes": 12,
        "z_spacing": 1,
        "class_names": [
            "background-tissue",
            "instrument-shaft",
            "instrument-clasper",
            "instrument-wrist",
            "kidney-parenchyma",
            "covered-kidney",
            "thread",
            "clamps",
            "suturing-needle",
            "suction-instrument",
            "small-intestine",
            "ultrasound-probe",
        ],
    },
}


def prepare_test_args(args: Any) -> str:
    dataset_name = args.dataset
    if dataset_name not in TEST_DATASET_CONFIG:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    dataset_config = TEST_DATASET_CONFIG[dataset_name]
    if args.volume_path is None:
        args.volume_path = dataset_config["volume_path"]
    args.num_classes = dataset_config["num_classes"]
    args.Dataset = dataset_config["Dataset"]
    args.z_spacing = dataset_config["z_spacing"]
    args.class_names = dataset_config.get("class_names")
    args.list_dir = dataset_config["list_dir"]
    args.is_pretrain = True

    if args.fold_id < 0 or args.fold_id >= 5:
        raise ValueError("fold_id must be between 0 and 4")

    args.exp = "TU_" + dataset_name + str(args.img_size)
    return dataset_name


def build_test_model(args: Any, device: str = "cuda", map_location: str | None = None) -> nn.Module:
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.verbose = args.verbose
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.use_se_block = args.use_se_block
    config_vit.drop_se_block = args.drop_se_block

    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.num_heads is not None:
        config_vit.transformer.num_heads = args.num_heads
    if args.num_layers is not None:
        config_vit.transformer.num_layers = args.num_layers

    if args.use_alternate_shsa and not args.use_shsa:
        raise ValueError("The --use_alternate_shsa flag requires --use_shsa to be set as well.")
    if args.use_ats and args.topk_attn <= 0.0:
        raise ValueError("The --use_ats flag requires --topk_attn to be greater than 0.0.")
    if args.use_shsa and args.topk_attn > 0.0:
        raise ValueError("The --use_shsa flag is mutually exclusive with --topk_attn > 0.0.")
    if args.use_gumbel_topk and args.topk_attn <= 0.0:
        raise ValueError("The --use_gumbel_topk flag requires --topk_attn to be greater than 0.0.")
    if args.use_ats and args.use_gumbel_topk:
        raise ValueError("--use_ats and --use_gumbel_topk are mutually exclusive.")

    config_vit.topk_attn = args.topk_attn
    config_vit.use_ats = args.use_ats
    config_vit.use_gumbel_topk = args.use_gumbel_topk
    config_vit.use_shsa = args.use_shsa
    config_vit.use_alternate_shsa = args.use_alternate_shsa
    config_vit.use_efficientnet = args.use_efficientnet
    config_vit.use_swin = args.use_swin

    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )

    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)

    checkpoint_path = Path(args.ckpt_dir) / args.ckpt
    load_kwargs = {}
    if map_location is not None:
        load_kwargs["map_location"] = map_location
    model.load_state_dict(torch.load(checkpoint_path, **load_kwargs))
    return model


def parse_args() -> Any:
    parser = build_test_arg_parser()
    parser.description = (
        "Run the benchmark.py throughput and p99 latency timing for one checkpoint "
        "multiple times."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Overrides --ckpt_dir and --ckpt when provided.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of repeated benchmark runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output JSON file. Defaults to benchmark/<dataset>/.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Benchmark device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--benchmark_batch_size",
        type=int,
        default=36,
        help="Batch size for the throughput loop.",
    )
    parser.add_argument(
        "--benchmark_num_workers",
        type=int,
        default=0,
        help="DataLoader workers for benchmark samples.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=20,
        help="Warmup batches before each measured run.",
    )
    parser.add_argument(
        "--measure_batches",
        type=int,
        default=50,
        help="Measured batches for throughput.",
    )
    parser.add_argument(
        "--single_image_latency_samples",
        type=int,
        default=200,
        help="Single-image forward passes used for p99 latency.",
    )
    parser.add_argument(
        "--autocast",
        action="store_true",
        help="Use CUDA autocast during timing.",
    )
    args = parser.parse_args()
    args.normalize_endovis_eval = args.normalize_present_class_eval
    return args


def _apply_checkpoint_override(args: Any) -> Path:
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        if not checkpoint.is_absolute():
            checkpoint = ROOT_DIR / checkpoint
        args.ckpt_dir = str(checkpoint.parent)
        args.ckpt = checkpoint.name

    checkpoint_path = Path(args.ckpt_dir) / args.ckpt
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT_DIR / checkpoint_path
    return checkpoint_path


def _set_reproducibility(args: Any) -> None:
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


def _validate_args(args: Any, device: str) -> None:
    if args.runs <= 0:
        raise ValueError("--runs must be greater than 0")
    if args.benchmark_batch_size <= 0:
        raise ValueError("--benchmark_batch_size must be greater than 0")
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be non-negative")
    if args.measure_batches <= 0:
        raise ValueError("--measure_batches must be greater than 0")
    if args.single_image_latency_samples <= 0:
        raise ValueError("--single_image_latency_samples must be greater than 0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.quantize:
        raise ValueError(
            "This tool uses benchmark.py timing only and does not support --quantize."
        )


def _move_images(batch: Any, device: str) -> torch.Tensor:
    images = _extract_images(batch)
    if not torch.is_tensor(images):
        raise TypeError("Extracted images are not a tensor.")
    return images.to(device, non_blocking=device.startswith("cuda"))


def benchmark_throughput_and_latency99(
    model: nn.Module,
    test_loader: Iterable,
    device: str = "cuda",
    warmup_steps: int = 20,
    measure_batches: int = 50,
    single_image_latency_samples: int = 200,
    enable_cudnn_benchmark: bool = True,
    autocast: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, float]:
    """Use benchmark.py timing logic but return only throughput and p99 latency."""
    model = model.to(device).eval()
    _set_cudnn_benchmark(enable_cudnn_benchmark)

    first_batch = next(iter(test_loader))
    first_imgs = _extract_images(first_batch)
    if not torch.is_tensor(first_imgs):
        raise TypeError("Extracted images are not a tensor.")

    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (device.startswith("cuda") and autocast)
        else torch.no_grad()
    )

    with ctx:
        it = iter(test_loader)
        for _ in range(max(0, warmup_steps)):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(test_loader)
                batch = next(it)
            imgs = _move_images(batch, device)
            if device.startswith("cuda"):
                _timed_forward_gpu(model, imgs)
            else:
                _timed_forward_cpu(model, imgs)

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
            ms = (
                _timed_forward_gpu(model, imgs)
                if device.startswith("cuda")
                else _timed_forward_cpu(model, imgs)
            )
            total_imgs += bs
            total_ms += ms

    single_lat_ms = []
    with ctx:
        it = iter(test_loader)
        collected = 0
        while collected < single_image_latency_samples:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(test_loader)
                batch = next(it)
            imgs = _extract_images(batch)
            for index in range(imgs.shape[0]):
                x = imgs[index:index + 1].to(
                    device,
                    non_blocking=device.startswith("cuda"),
                )
                ms = (
                    _timed_forward_gpu(model, x)
                    if device.startswith("cuda")
                    else _timed_forward_cpu(model, x)
                )
                single_lat_ms.append(ms)
                collected += 1
                if collected >= single_image_latency_samples:
                    break

    throughput_img_s = (total_imgs / (total_ms / 1000.0)) if total_ms > 0 else float("nan")
    latency_ms_p99 = float(np.percentile(np.array(single_lat_ms, dtype=np.float64), 99))
    return {
        "throughput_img_s": float(throughput_img_s),
        "latency_ms_p99": latency_ms_p99,
    }


def _metric_summary(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return _make_json_safe(value)


def _output_path(args: Any, timestamp: str) -> Path:
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = ROOT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    output_dir = ROOT_DIR / args.benchmark_dict / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = _sanitize_name(Path(args.ckpt).name)
    return output_dir / f"{ckpt_name}_{args.img_size}_throughput_latency99_{timestamp}.json"


def main() -> int:
    args = parse_args()
    checkpoint_path = _apply_checkpoint_override(args)
    dataset_name = prepare_test_args(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _validate_args(args, device)
    _set_reproducibility(args)

    model = build_test_model(args, device=device, map_location=device)
    test_loader = build_benchmark_loader(
        args,
        batch_size=args.benchmark_batch_size,
        num_workers=args.benchmark_num_workers,
        shuffle=False,
    )

    run_results = []
    for run_index in range(1, args.runs + 1):
        metrics = benchmark_throughput_and_latency99(
            model=model,
            test_loader=test_loader,
            device=device,
            warmup_steps=args.warmup_steps,
            measure_batches=args.measure_batches,
            single_image_latency_samples=args.single_image_latency_samples,
            enable_cudnn_benchmark=True,
            autocast=args.autocast,
        )
        result = {
            "run": run_index,
            "metrics": metrics,
        }
        run_results.append(result)
        print(
            "run {}/{} throughput_img_s={:.4f} latency_ms_p99={:.4f}".format(
                run_index,
                args.runs,
                metrics["throughput_img_s"],
                metrics["latency_ms_p99"],
            )
        )

    throughput_values = [
        run["metrics"]["throughput_img_s"]
        for run in run_results
    ]
    latency_values = [
        run["metrics"]["latency_ms_p99"]
        for run in run_results
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date, time = timestamp.split("_")
    output_path = _output_path(args, timestamp)

    payload = {
        "description": args.description,
        "date_time": date + " " + time,
        "dataset": dataset_name,
        "checkpoint": str(checkpoint_path),
        "runs": run_results,
        "summary": {
            "throughput_img_s": _metric_summary(throughput_values),
            "latency_ms_p99": _metric_summary(latency_values),
        },
        "arguments": {k: _json_safe(v) for k, v in vars(args).items()},
    }
    with output_path.open("w") as handle:
        json.dump(_json_safe(payload), handle, indent=2)
    print("saved {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
