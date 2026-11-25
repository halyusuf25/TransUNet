#!/usr/bin/env python3
"""Compare learned SE gates with their direct training targets after packed INT4 conversion."""


"""
command line examples:
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python tools/compare_se_proxy.py \
  --dataset ACDC \
  --checkpoint ckpt/ckpt_ACDC/SELoss_ACDC__best_val_dice_0.8992649912834167_epoch_76_20260930_153247.pth \
  --multiple_seeds --case_by_case \
  --calibration-seeds 1234 1235 1236 1237 1238 \
  --calibration-size 8 \
  --output-dir benchmark/ACDC/compare_se_proxy/multiple_seeds/case_by_case
  
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python tools/compare_se_proxy.py \
  --dataset Cataract1k \
  --checkpoint ckpt/ckpt_Cataract1k/SELoss_Cataract1k__best_val_dice_0.942674332857132_epoch_259_20260930_200930.pth \
  --multiple_seeds \
  --calibration-seeds 1234 1235 1236 1237 1238 \
  --calibration-size 8 \
  --normalize_present_class_eval \
  --output-dir benchmark/Cataract1k/compare_se_proxy/multiple_seeds
  
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python tools/compare_se_proxy.py \
  --dataset Synapse \
  --checkpoint ckpt/ckpt_Synapse/SELoss_Synapse__best_val_dice_0.7899280190467834_epoch_100_20260930_173838.pth \
  --multiple_seeds \
  --calibration-seeds 1234 1235 1236 1237 1238 \
  --calibration-size 8 \
  --output-dir benchmark/Synapse/compare_se_proxy/multiple_seeds
  
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python tools/compare_se_proxy.py \
  --dataset ACDC \
  --checkpoint ckpt/ckpt_ACDC/SELoss_ACDC__best_val_dice_0.8992649912834167_epoch_76_20260930_153247.pth \
  --multiple_seeds --case_by_case \
  --calibration-seeds 1234 1235 1236 1237 1238 \
  --calibration-size 8 \
  --output-dir benchmark/ACDC/compare_se_proxy/multiple_seeds/case_by_case
  
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python tools/compare_se_proxy.py \
  --dataset Cataract1k \
  --checkpoint ckpt/ckpt_Cataract1k/SELoss_Cataract1k__best_val_dice_0.942674332857132_epoch_259_20260930_200930.pth \
  --multiple_seeds --case_by_case \
  --calibration-seeds 1234 1235 1236 1237 1238 \
  --calibration-size 8 \
  --normalize-present-class-eval \
  --output-dir benchmark/Cataract1k/compare_se_proxy/multiple_seeds/case_by_case
  
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python tools/compare_se_proxy.py \
  --dataset Synapse \
  --checkpoint ckpt/ckpt_Synapse/SELoss_Synapse__best_val_dice_0.7899280190467834_epoch_100_20260930_173838.pth \
  --multiple_seeds --case_by_case \
  --calibration-seeds 1234 1235 1236 1237 1238 \
  --calibration-size 8 \
  --output-dir benchmark/Synapse/compare_se_proxy/multiple_seeds/case_by_case
"""


import argparse
from copy import copy, deepcopy
from functools import partial
from pathlib import Path
from pickle import UnpicklingError
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("Synapse", "ACDC", "Cataract1k"))
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Full .pth path with its matching .pth.se_aux.json sidecar")
    parser.add_argument("--calibration-size", type=int, default=8, help="Unique training images (default 8)")
    parser.add_argument("--calibration-seed", type=int, default=1234, help="Sampling seed for single-draw runs (default 1234)")
    parser.add_argument("--calibration-seeds", type=int, nargs="+", metavar="SEED",
                        default=[1234, 1235, 1236, 1237, 1238],
                        help="Ordered sampling seeds for --multiple_seeds (default: %(default)s)")
    parser.add_argument("--case_by_case", action="store_true",
                        help="Also evaluate and compare both packed models for every matched calibration image")
    parser.add_argument("--multiple_seeds", action="store_true",
                        help="Repeat the comparison for each seed in --calibration-seeds")
    parser.add_argument("--output-dir", type=Path, help="Report directory (default benchmark/<dataset>/compare_se_proxy, "
                        "with multiple_seeds and/or case_by_case subdirectories for those options)")
    parser.add_argument("--train-root", help="Training data root (Synapse train_npz, ACDC or Cataract1k data root)")
    parser.add_argument("--volume-path", help="Test volume or frame root")
    parser.add_argument("--list-dir", help="Directory with train.txt and test_vol.txt for Synapse or ACDC, "
                                          "or train.csv and test.csv for Cataract1k. ACDC accepts slice/volume "
                                          "IDs with or without .h5 within its built-in patient splits.")
    parser.add_argument("--img-size", "--image-size", "--img_size", "--image_size", dest="img_size",
                        type=int, default=224, help="Model input size (default 224)")
    parser.add_argument("--vit-name", "--vit_name", choices=("R50-ViT-B_16", "R50-ViT-L_16"),
                        default="R50-ViT-B_16", help="Full-token segmentation backbone (default R50-ViT-B_16)")
    parser.add_argument("--normalize-present-class-eval", "--normalize_present_class_eval",
                        "--normalize-endovis-eval", "--normalize_endovis_eval", action="store_true",
                        help="Use ImageNet normalization for Cataract1k calibration and evaluation")
    args = parser.parse_args(argv)
    if args.calibration_size < 1:
        parser.error("--calibration-size must be positive")
    if args.img_size < 16 or args.img_size % 16:
        parser.error("--img-size must be a positive multiple of 16")
    if args.multiple_seeds and len(set(args.calibration_seeds)) != len(args.calibration_seeds):
        parser.error("--calibration-seeds must be distinct so each draw has its own seed directory")
    return args


def listed_acdc_dataset(*, base_dir, list_dir, split, transform=None):
    from datasets.dataset_acdc import ACDC_Dataset

    dataset = ACDC_Dataset(base_dir=base_dir, split=split, transform=transform)
    manifest = Path(list_dir) / (split + ".txt")
    available = set(dataset.sample_list)
    selected = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        filename = name if name.endswith(".h5") else name + ".h5"
        if filename not in available:
            raise ValueError(f"{manifest}: {name!r} is outside the available ACDC {split} split at {base_dir}")
        selected.append(filename)
    dataset.sample_list = list(dict.fromkeys(selected))
    return dataset


def experiment_args(options):
    from src.test_helpers import build_test_arg_parser
    from test import get_test_dataset_config

    # Inherit the evaluator's model and metric settings, including full attention.
    args = build_test_arg_parser().parse_args([])
    vars(args).update(vars(options))
    config = get_test_dataset_config()[args.dataset]
    args.Dataset = config["Dataset"]
    args.num_classes = config["num_classes"]
    args.class_names = config["class_names"]
    args.z_spacing = config["z_spacing"]
    args.volume_path = args.volume_path or config["volume_path"]
    if args.train_root is None:
        args.train_root = ("/data/halyusuf/data/Synapse/train_npz" if args.dataset == "Synapse"
                           else config["volume_path"])
    if args.list_dir is None and config["list_dir"] is not None:
        args.list_dir = str(ROOT / config["list_dir"])
    if args.dataset == "Cataract1k" and args.list_dir is not None:
        list_dir = Path(args.list_dir).resolve()
        args.Dataset = partial(args.Dataset, train_csv=list_dir / "train.csv", test_csv=list_dir / "test.csv")
    if args.dataset == "ACDC" and args.list_dir is not None:
        args.Dataset = listed_acdc_dataset
    if args.output_dir is None:
        args.output_dir = ROOT / args.benchmark_dict / args.dataset / "compare_se_proxy"
        if args.multiple_seeds:
            args.output_dir /= "multiple_seeds"
        if args.case_by_case:
            args.output_dir /= "case_by_case"
    args.use_se_block = True
    return args


def prepare_calibration(args):
    """Sample unique training IDs, then cache one evaluation-ready NCHW input per ID."""
    import torch
    from src.inc_awq import _calibration_images

    if args.dataset == "Cataract1k":
        dataset = args.Dataset(base_dir=args.train_root, split="train")
        sample_ids = [Path(path).name for path in dataset.image_files]
    else:
        # ACDC calls its training transform unconditionally. Keep each raw 2-D slice.
        dataset = args.Dataset(base_dir=args.train_root, list_dir=args.list_dir, split="train",
                               transform=lambda sample: sample)
        sample_ids = [name.strip() for name in dataset.sample_list]

    indices = {name: index for index, name in enumerate(sample_ids) if name}
    if not 1 <= args.calibration_size <= len(indices):
        raise ValueError(f"--calibration-size must be between 1 and {len(indices)} "
                         f"({args.dataset} unique training images), got {args.calibration_size}")
    ids = random.Random(args.calibration_seed).sample(sorted(indices), args.calibration_size)
    inputs = []
    for name in ids:
        image = torch.as_tensor(dataset[indices[name]]["image"])
        if args.dataset != "Cataract1k":
            if image.ndim != 2:
                raise ValueError(f"Expected a 2-D training slice for {name}, got {tuple(image.shape)}")
            image = image.unsqueeze(0)  # One-slice volume for the shared preprocessor.
        inputs.append(_calibration_images(image.unsqueeze(0), args))
    return tuple(ids), tuple(inputs)


def load_rms_model(args):
    import torch
    from src.se_auxiliary import DEFAULT_SE_AUX_CONFIG, validate_se_calibration_config
    from test import build_test_model, load_test_checkpoint

    checkpoint = args.checkpoint
    sidecar = Path(str(checkpoint) + ".se_aux.json")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Full-precision checkpoint does not exist: {checkpoint}")
    if not sidecar.is_file():
        raise ValueError(f"RMS-trained SE-AUX comparison requires the checkpoint sidecar: {sidecar}")

    model = build_test_model(args)
    try:
        metadata = load_test_checkpoint(model, checkpoint, require_fp32=True)
        if (metadata["pooling_mode"] != "rms" or
                metadata.get("target_definition") != "mean_qkv_fc1_channel_error_rms_v1"):
            raise ValueError(f"{sidecar} must specify RMS pooling and the mean QKV/FC1 v1 target")
        se_layers = model.transformer.encoder.SELayer
        if not all(torch.isfinite(p).all().item() for p in se_layers.parameters()):
            raise ValueError("checkpoint contains nonfinite SE weights")
        settings = DEFAULT_SE_AUX_CONFIG
        validate_se_calibration_config(model, bits=settings["bits"], group_size=settings["group_size"],
                                       zero_point=settings["zero_point"])
    except (RuntimeError, ValueError, KeyError, TypeError, AttributeError, EOFError, UnpicklingError) as exc:
        raise ValueError(f"{checkpoint} cannot support the RMS-trained SE-AUX comparison: {exc}") from exc
    return model.eval()


def make_quantizer(args, source, calibration_inputs, model):
    from networks.quantizer import AWQViTSegQuantizer

    settings = model.transformer.encoder.se_aux_config
    return AWQViTSegQuantizer(model.cuda(), calibration_inputs, w_bit=settings["bits"],
                              q_group_size=settings["group_size"], n_calib_batches=len(calibration_inputs),
                              device="cuda", args=args, candidate_source=source)


def evaluate_packed(args, packed):
    from test import drop_se_blocks, inference

    drop_se_blocks(packed)
    return inference(args, packed)


def run_method(args, source, calibration_inputs, model):
    settings = model.transformer.encoder.se_aux_config
    quantizer = make_quantizer(args, source, calibration_inputs, model)
    packed = quantizer.quantize()
    index, mse = quantizer.winning_candidate_index, quantizer.winning_calibration_mse
    return index, mse, evaluate_packed(args, packed), settings


def run_paired_method(args, source, calibration_inputs, model):
    """Evaluate every scored candidate once and return the winner's row alongside all rows."""
    from src.se_auxiliary import lowest_mse_index

    settings = model.transformer.encoder.se_aux_config
    candidates = make_quantizer(args, source, calibration_inputs, model).score_candidates()
    winner = lowest_mse_index(candidate.mean_mse for candidate in candidates)
    results = []
    for index, candidate in enumerate(candidates):
        print(f"Evaluating {source} candidate {index + 1}/{len(candidates)}", flush=True)
        quantizer = make_quantizer(args, source, calibration_inputs, deepcopy(model))
        packed = quantizer.quantize(candidate_scales=candidate.scales)
        results.append((index, candidate.mean_mse, evaluate_packed(args, packed), settings))
        del packed, quantizer
    return results[winner], results


def report_header(args, ids, settings):
    cataract = args.dataset == "Cataract1k"
    unit = "RGB training frame" if cataract else "2-D training slice"
    normalization = "stored slice intensities"
    if cataract:
        normalization = "ImageNet mean/std" if args.normalize_present_class_eval else "raw RGB (0–255)"
    return [
        "# SE-AUX and direct proxy comparison",
        "",
        f"- Checkpoint: `{args.checkpoint.resolve()}`",
        f"- Dataset: {args.dataset}, full test split",
        f"- Model: {args.vit_name}, full tokens, {args.img_size} × {args.img_size}",
        f"- Training root: `{args.train_root}`",
        f"- Test root: `{args.volume_path}`",
        f"- Split list directory: {args.list_dir or '(dataset default)'}",
        f"- Calibration seed: {args.calibration_seed}",
        f"- Sampling unit: {unit}, {len(ids)} sampled without replacement",
        "- Ordered calibration IDs: " + ", ".join(f"`{name}`" for name in ids),
        f"- Input normalization (calibration and test): {normalization}",
        f"- Quantization: packed W{settings['bits']}A16 (AWQ WQLinear), group size {settings['group_size']}, "
        f"zero point {settings['zero_point']}, RMS target epsilon {settings['eps']:.8g}. "
        "SE modules removed after folding.",
        "",
    ]


def format_report(args, ids, se, proxy):
    se_index, se_mse, se_metrics, settings = se
    proxy_index, proxy_mse, proxy_metrics, _ = proxy
    lines = report_header(args, ids, settings) + [
        "| Method | Winning calibration image ID | Mean calibration logit MSE | Test Dice (%) | Test IoU (%) | HD95 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, index, mse, metrics in (("SE-AUX", se_index, se_mse, se_metrics),
                                       ("Direct proxy", proxy_index, proxy_mse, proxy_metrics)):
        lines.append(f"| {label} | `{ids[index]}` | {mse:.8g} | {100 * metrics['mean_dice']:.4f} | "
                     f"{100 * metrics['mean_iou']:.4f} | {metrics['mean_hd95']:.4f} |")
    dice_pp = 100 * (se_metrics["mean_dice"] - proxy_metrics["mean_dice"])
    iou_pp = 100 * (se_metrics["mean_iou"] - proxy_metrics["mean_iou"])
    hd95_delta = se_metrics["mean_hd95"] - proxy_metrics["mean_hd95"]
    lines.append(f"| SE-AUX minus direct proxy | — | {se_mse - proxy_mse:+.8g} | {dice_pp:+.4f} pp | "
                 f"{iou_pp:+.4f} pp | {hd95_delta:+.4f} |")
    return "\n".join(lines) + "\n"


def format_seed_summary(args, runs):
    """Summarize the selected models, giving each calibration seed equal weight."""
    import numpy as np

    if not runs:
        raise ValueError("At least one calibration draw is required for a seed summary")

    def table_row(label, values, *, signed_deltas=True):
        cells = [label]
        for se_value, proxy_value, delta in values:
            for value, fmt in ((se_value, ".4f"), (proxy_value, ".4f"),
                               (delta, "+.4f" if signed_deltas else ".4f")):
                cells.append(format(value, fmt) if np.isfinite(value) else "N/A")
        return "| " + " | ".join(cells) + " |"

    lines = [
        "# SE-AUX and direct proxy across calibration seeds",
        "",
        f"Each of the {len(runs)} draws uses {args.calibration_size} training images. "
        "Both methods receive the same images in the same order and each selects its lowest-MSE candidate.",
        f"All selected models start from `{args.checkpoint.resolve()}` and are evaluated on the "
        f"full {args.dataset} test split with {args.vit_name} at {args.img_size} × {args.img_size}.",
        "",
        "Each Δ is SE-AUX minus direct proxy within a seed. Dice and IoU differences and SDs are in "
        "percentage points. HD95 uses the evaluator's native units. Sample SD uses the n − 1 denominator "
        "and requires at least two seeds.",
        "",
        "| Calibration seed | SE-AUX Dice (%) | Proxy Dice (%) | Δ Dice (pp) | SE-AUX IoU (%) | "
        "Proxy IoU (%) | Δ IoU (pp) | SE-AUX HD95 | Proxy HD95 | Δ HD95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    seed_values = []
    for seed, se_result, proxy_result in runs:
        se_metrics, proxy_metrics = se_result[2], proxy_result[2]
        values = []
        for metric, scale in (("mean_dice", 100), ("mean_iou", 100), ("mean_hd95", 1)):
            se, proxy = float(se_metrics[metric]), float(proxy_metrics[metric])
            values.append((scale * se, scale * proxy, scale * (se - proxy)))
        lines.append(table_row(f"[{seed}](seed_{seed}/results.md)", values))
        seed_values.append(values)

    values = np.asarray(seed_values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        mean = values.mean(axis=0)
        sample_sd = values.std(axis=0, ddof=1) if len(runs) > 1 else np.full_like(mean, np.nan)
    lines.append(table_row("**Mean**", mean))
    lines.append(table_row("**Sample SD**", sample_sd, signed_deltas=False))
    return "\n".join(lines) + "\n"


def format_matched_seed_summary(args, runs):
    """Summarize matched-image differences with the calibration seed as the sampling unit."""
    import numpy as np

    if not runs:
        raise ValueError("At least one calibration draw is required for a matched-image summary")

    def format_values(values, signed):
        formats = (".8g", ".4f", ".4f", ".4f")
        return [format(value, ("+" if signed else "") + fmt) if np.isfinite(value) else "N/A"
                for value, fmt in zip(values, formats)]

    lines = [
        "# Matched calibration images across seeds",
        "",
        f"Each seed pairs the two methods on {args.calibration_size} scale-generating training images. "
        f"Each packed candidate is evaluated on the full {args.dataset} test split.",
        "Each Δ is SE-AUX minus direct proxy. Each row averages the matched-image differences within "
        "one seed. The last row gives the mean and sample SD of those seed means, weighting seeds equally. "
        "Sample SD requires at least two seeds. HD95 uses the evaluator's native units.",
        "",
        "| Calibration seed | Δ logit MSE | Δ Dice (pp) | Δ IoU (pp) | Δ HD95 |",
        "|---|---:|---:|---:|---:|",
    ]
    image_deltas = []
    seed_deltas = []
    for seed, se_results, proxy_results in runs:
        rows = matched_metric_rows(se_results, proxy_results, args.calibration_size)
        deltas = [delta for _, _, delta in rows]
        seed_mean = mean_metric_values(deltas)
        lines.append(f"| {seed} | " + " | ".join(format_values(seed_mean, signed=True)) + " |")
        image_deltas.extend(deltas)
        seed_deltas.append(seed_mean)

    with np.errstate(invalid="ignore"):
        mean = np.mean(seed_deltas, axis=0)
        sd = np.std(seed_deltas, axis=0, ddof=1) if len(runs) > 1 else np.full_like(mean, np.nan)
    cells = [f"{m} ± {s}" for m, s in zip(format_values(mean, signed=True), format_values(sd, signed=False))]
    lines.extend([
        "| **Mean ± sample SD** | " + " | ".join(cells) + " |",
        "",
        "Win counts are SE-AUX / direct proxy / ties. Higher Dice and IoU and lower MSE and HD95 win. "
        "Matched images count each image within each draw. Seed means count each draw once. "
        "Counts include finite differences only.",
        "",
        "| Comparison unit | Logit MSE | Dice | IoU | HD95 |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, deltas in (("Matched images", image_deltas), ("Seed means", seed_deltas)):
        counts = [f"{se} / {proxy} / {ties}" for se, proxy, ties in metric_win_counts(deltas)]
        lines.append(f"| {label} | " + " | ".join(counts) + " |")
    return "\n".join(lines) + "\n"


def matched_metric_rows(se_results, proxy_results, expected_count):
    """Pair SE-AUX values, proxy values and differences by scale-generating image index."""
    def ordered_values(results):
        by_index = sorted(results, key=lambda r: r[0])
        if not expected_count or [r[0] for r in by_index] != list(range(expected_count)):
            raise ValueError("Expected one result per calibration image for each method")
        return [(mse, 100 * metrics["mean_dice"], 100 * metrics["mean_iou"], metrics["mean_hd95"])
                for _, mse, metrics, _ in by_index]

    return [(se, proxy, tuple(a - b for a, b in zip(se, proxy)))
            for se, proxy in zip(ordered_values(se_results), ordered_values(proxy_results))]


def mean_metric_values(rows):
    from math import isfinite, nan
    from statistics import fmean

    return tuple(fmean(column) if all(map(isfinite, column)) else nan for column in zip(*rows))


def metric_win_counts(delta_rows):
    """Count SE wins, proxy wins and ties from unrounded SE-minus-proxy differences."""
    from math import isfinite

    counts = []
    # In column order, lower MSE, higher Dice, higher IoU and lower HD95 win.
    for deltas, direction in zip(zip(*delta_rows), (-1, 1, 1, -1)):
        margins = [delta * direction for delta in deltas if isfinite(delta)]
        counts.append((sum(v > 0 for v in margins), sum(v < 0 for v in margins), margins.count(0)))
    return counts


def format_paired_report(args, ids, se_results, proxy_results):
    """Show test metrics for each scale-generating image, paired across methods."""
    from math import isfinite

    rows = matched_metric_rows(se_results, proxy_results, len(ids))
    lines = report_header(args, ids, se_results[0][3])
    lines.extend([
        "## Matched calibration images",
        "",
        f"Each row uses scales from the same training image for both methods. Calibration logit MSE "
        f"is averaged over all {len(ids)} calibration images. Dice, IoU, and HD95 use the full test split.",
        "Each Δ is SE-AUX minus direct proxy. The mean gives each calibration image equal weight.",
        "",
        "| Scale-generating training image | SE-AUX MSE | Proxy MSE | Δ MSE | SE-AUX Dice (%) | Proxy Dice (%) | "
        "Δ Dice (pp) | SE-AUX IoU (%) | Proxy IoU (%) | Δ IoU (pp) | SE-AUX HD95 | Proxy HD95 | Δ HD95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    labels = [f"`{name}`" for name in ids] + ["**Paired mean**"]
    mean_row = tuple(mean_metric_values(group) for group in zip(*rows))
    for label, row in zip(labels, [*rows, mean_row]):
        cells = [label]
        for precision, (se, proxy, delta) in zip((".8g", ".4f", ".4f", ".4f"), zip(*row)):
            for value, fmt in ((se, precision), (proxy, precision), (delta, "+" + precision)):
                cells.append(format(value, fmt) if isfinite(value) else "N/A")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "", "### Matched-image win counts", "",
        "Lower calibration MSE and HD95 win. Higher Dice and IoU win. Counts use finite, unrounded differences.",
        "", "| Metric | SE-AUX wins | Direct proxy wins | Ties |", "|---|---:|---:|---:|",
    ])
    counts = metric_win_counts([delta for _, _, delta in rows])
    for metric, (se_wins, proxy_wins, ties) in zip(("Calibration logit MSE", "Test Dice", "Test IoU", "HD95"), counts):
        lines.append(f"| {metric} | {se_wins} | {proxy_wins} | {ties} |")
    return "\n".join(lines) + "\n"


def write_report(path, report):
    print(report, end="")
    path.write_text(report, encoding="utf-8")


def run_seed(args):
    """Compare the selected models and optionally retain all matched-image evaluations."""
    import torch

    model = load_rms_model(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Packed W4A16 evaluation requires a visible CUDA GPU.")
    ids, inputs = prepare_calibration(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(source, model):
        if args.case_by_case:
            return run_paired_method(args, source, inputs, model)
        return run_method(args, source, inputs, model), None

    print(f"Calibrating SE-AUX on {len(ids)} training images", flush=True)
    se, se_cases = evaluate("se", model)
    del model
    print(f"Calibrating direct proxy on the same {len(ids)} training images", flush=True)
    proxy, proxy_cases = evaluate("direct_proxy", load_rms_model(args))

    write_report(args.output_dir / "results.md", format_report(args, ids, se, proxy))
    if args.case_by_case:
        write_report(args.output_dir / "case_by_case_results.md", format_paired_report(args, ids, se_cases, proxy_cases))
    return se, proxy, se_cases, proxy_cases


def main(argv=None):
    options = parse_args(argv)
    import torch

    args = experiment_args(options)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not args.multiple_seeds:
        run_seed(args)
        return

    selected_runs = []
    matched_runs = []
    for seed in args.calibration_seeds:
        draw = copy(args)
        draw.calibration_seed = seed
        draw.output_dir = args.output_dir / f"seed_{seed}"
        print(f"Running calibration seed {seed}", flush=True)
        se, proxy, se_cases, proxy_cases = run_seed(draw)
        selected_runs.append((seed, se, proxy))
        if args.case_by_case:
            matched_runs.append((seed, se_cases, proxy_cases))

    write_report(args.output_dir / "summary.md", format_seed_summary(args, selected_runs))
    if args.case_by_case:
        write_report(args.output_dir / "case_by_case_summary.md", format_matched_seed_summary(args, matched_runs))


if __name__ == "__main__":
    main()
