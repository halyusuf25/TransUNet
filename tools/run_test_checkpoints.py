#!/usr/bin/env python3
"""Run test.py for a list of checkpoint files.

Edit CHECKPOINTS and BASE_TEST_ARGS below for your usual command, or pass
checkpoint names and extra test.py arguments from the command line.

Examples:
    python tools/run_test_checkpoints.py

    python tools/run_test_checkpoints.py \
        --ckpt model_a.pth model_b.pth \
        --cuda-visible-devices 4 \
        -- --img_size 224 --topk_attn 0.50 --use_gumbel_topk
"""

from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple


# ---------------------------------------------------------------------------
# Edit these defaults for the command you run most often.
# ---------------------------------------------------------------------------
DEFAULT_CUDA_VISIBLE_DEVICES = "4"

# Each item can be either:
#   "model.pth"
#   ("model.pth", "description for this checkpoint")
#   {"ckpt": "model.pth", "description": "description for this checkpoint"}
CHECKPOINTS = [
    (
        "GumbelTopK_Synapse__best_val_dice_0.7858279943466187_epoch_103_20260703_152127.pth",
        "TransUNet with Gumbel Top-K on Synapse Dataset (512x512) epoch 103 - dice 0.785",
    ),
    (
        "GumbelTopK_Synapse__best_val_dice_0.7897534370422363_epoch_121_20260703_153117.pth",
        "TransUNet with Gumbel Top-K on Synapse Dataset (512x512) epoch 121 - dice 0.789",
    ),
    (
        "GumbelTopK_Synapse__best_val_dice_0.7908648252487183_epoch_184_20260703_160541.pth",
        "TransUNet with Gumbel Top-K on Synapse Dataset (512x512) epoch 184 - dice 0.790",
    ),
    (
        "GumbelTopK_Synapse__best_val_dice_0.7911930084228516_epoch_190_20260703_160858.pth",
        "TransUNet with Gumbel Top-K on Synapse Dataset (512x512) epoch 190 - dice 0.791",
    ),
    (
        "GumbelTopK_Synapse__epoch_250_dice_0.7876302003860474_20260703_164143.pth",
        "TransUNet with Gumbel Top-K on Synapse Dataset (512x512) epoch 250 - dice 0.787",
    ),
    (
        "GumbelTopK_Synapse__epoch_270_LastEpoch_20260703_165205_dice_0.7854651808738708.pth",
        "TransUNet with Gumbel Top-K on Synapse Dataset (512x512) epoch 270 - dice 0.785",
    ),
]

BASE_TEST_ARGS = [
    "--dataset", "Synapse",
    "--vit_name", "R50-ViT-B_16",
    "--benchmark_dict", "benchmarkTest",
    "--ckpt_dir", "ckpt/ckpt_Synapse",
    "--topk_attn", "0.50",
    "--use_gumbel_topk",
    "--repeated_runs", "10",
    "--gumbel_sampling_mode", "manual",
]


class CheckpointRun(NamedTuple):
    ckpt: str
    description: str | None = None


def _strip_separator(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _remove_ckpt_args(args: list[str]) -> list[str]:
    """Remove --ckpt from a test.py argument list.

    The runner owns --ckpt so each loop iteration can replace it safely.
    """
    cleaned = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--ckpt":
            index += 2
            continue
        if arg.startswith("--ckpt="):
            index += 1
            continue
        cleaned.append(arg)
        index += 1
    return cleaned


def _option_value(args: list[str], option: str, default: str | None = None) -> str | None:
    value = default
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == option and index + 1 < len(args):
            value = args[index + 1]
            index += 2
            continue
        prefix = option + "="
        if arg.startswith(prefix):
            value = arg[len(prefix):]
        index += 1
    return value


def _read_ckpt_file(path: Path) -> list[Any]:
    checkpoints = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            ckpt, description = line.split("\t", 1)
            checkpoints.append(CheckpointRun(ckpt.strip(), description.strip()))
        else:
            checkpoints.append(line)
    return checkpoints


def _normalize_checkpoint_run(spec: Any) -> CheckpointRun:
    if isinstance(spec, CheckpointRun):
        return spec
    if isinstance(spec, (str, Path)):
        return CheckpointRun(str(spec))
    if isinstance(spec, dict):
        ckpt = spec.get("ckpt", spec.get("checkpoint"))
        if ckpt is None:
            raise ValueError(f"Checkpoint dictionary is missing 'ckpt': {spec!r}")
        description = spec.get("description")
        return CheckpointRun(str(ckpt), None if description is None else str(description))
    if isinstance(spec, (tuple, list)):
        if len(spec) == 1:
            return CheckpointRun(str(spec[0]))
        if len(spec) == 2:
            description = spec[1]
            return CheckpointRun(str(spec[0]), None if description is None else str(description))
    raise TypeError(f"Unsupported checkpoint entry: {spec!r}")


def _checkpoint_from_match(match: str, ckpt_dir: str | None, repo_root: Path) -> str:
    match_path = Path(match)
    if ckpt_dir is None:
        return str(match_path)

    ckpt_dir_path = Path(ckpt_dir)
    if not ckpt_dir_path.is_absolute():
        ckpt_dir_path = repo_root / ckpt_dir_path

    try:
        return str(match_path.resolve().relative_to(ckpt_dir_path.resolve()))
    except ValueError:
        return str(match_path)


def _expand_ckpt_globs(
    patterns: list[str],
    ckpt_dir: str | None,
    repo_root: Path,
) -> list[CheckpointRun]:
    checkpoints = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and ckpt_dir and not Path(pattern).parent.parts:
            matches = sorted(glob.glob(str(repo_root / ckpt_dir / pattern)))
        if not matches:
            raise FileNotFoundError(f"No checkpoint files matched pattern: {pattern}")
        checkpoints.extend(
            CheckpointRun(_checkpoint_from_match(match, ckpt_dir, repo_root))
            for match in matches
        )
    return checkpoints


def _dedupe_keep_order(values: list[CheckpointRun]) -> list[CheckpointRun]:
    seen = set()
    result = []
    for value in values:
        key = (value.ckpt, value.description)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run test.py once for each checkpoint in a list.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        nargs="+",
        action="append",
        default=[],
        help="Checkpoint file names to test. These override the CHECKPOINTS list when provided.",
    )
    parser.add_argument(
        "--ckpt-file",
        type=Path,
        action="append",
        default=[],
        help="Text file with one checkpoint per line. Use ckpt<TAB>description for per-checkpoint descriptions.",
    )
    parser.add_argument(
        "--ckpt-glob",
        action="append",
        default=[],
        help="Glob pattern for checkpoint files, for example '*.pth' or 'ckpt/ckpt_ACDC/*best*.pth'.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=DEFAULT_CUDA_VISIBLE_DEVICES,
        help="Value to set for CUDA_VISIBLE_DEVICES. Use an empty string to keep the current environment unchanged.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch test.py.",
    )
    parser.add_argument(
        "--test-script",
        default="test.py",
        help="Path to the test script, relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed checkpoint run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "test_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments appended to BASE_TEST_ARGS and passed to test.py. Put them after --.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    extra_test_args = _strip_separator(args.test_args)
    base_test_args = _remove_ckpt_args(BASE_TEST_ARGS + extra_test_args)
    ckpt_dir = _option_value(base_test_args, "--ckpt_dir", "ckpt/")

    checkpoint_specs = []
    cli_checkpoints = [item for group in args.ckpt for item in group]
    checkpoint_specs.extend(cli_checkpoints or CHECKPOINTS)

    for ckpt_file in args.ckpt_file:
        checkpoint_specs.extend(_read_ckpt_file(ckpt_file))

    checkpoint_specs.extend(_expand_ckpt_globs(args.ckpt_glob, ckpt_dir, repo_root))
    checkpoints = _dedupe_keep_order(
        [_normalize_checkpoint_run(spec) for spec in checkpoint_specs]
    )

    if not checkpoints:
        print("No checkpoints were provided. Add names to CHECKPOINTS or pass --ckpt.", file=sys.stderr)
        return 2

    test_script = Path(args.test_script)
    if not test_script.is_absolute():
        test_script = repo_root / test_script

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    failures = []
    for run_number, checkpoint in enumerate(checkpoints, start=1):
        command = [
            args.python,
            str(test_script),
            *base_test_args,
            "--ckpt",
            checkpoint.ckpt,
        ]
        if checkpoint.description is not None:
            command.extend(["--description", checkpoint.description])

        env_prefix = ""
        if args.cuda_visible_devices:
            env_prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_visible_devices)} "

        print(f"\n[{run_number}/{len(checkpoints)}] {checkpoint.ckpt}")
        if checkpoint.description is not None:
            print(f"Description: {checkpoint.description}")
        print(env_prefix + shlex.join(command))

        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=repo_root, env=env)
        if completed.returncode != 0:
            failures.append((checkpoint.ckpt, completed.returncode))
            print(f"FAILED: {checkpoint.ckpt} exited with code {completed.returncode}", file=sys.stderr)
            if args.fail_fast:
                break

    if failures:
        print("\nFailed checkpoint runs:", file=sys.stderr)
        for checkpoint, returncode in failures:
            print(f"  {checkpoint}: exit code {returncode}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
