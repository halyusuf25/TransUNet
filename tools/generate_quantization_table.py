#!/usr/bin/env python3
"""Generate a LaTeX table from per-dataset evaluation JSON files (stdlib only).

Run from the repository root::

    python tools/generate_quantization_table.py benchmark_quantize_newSELoss \
        --output_directory tables --output performance_quantization.tex

The parent LaTeX document needs graphicx, array, makecell, and multirow.
Classification follows test.py and networks/quantizer.py: INC backends use AWQ
even with SE checkpoints; legacy quantization uses SE guidance only when
use_se_block is true. notes.quantized_model describes packed-kernel counting,
so it is false for the INC fake-quantized AWQ results. Checkpoint names and
free-text descriptions are never used to infer a method.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, NamedTuple


METHODS = ("Non-quantized", "AWQ", "SE-AUX (ours)")
ARCHITECTURE_FIELDS = (
    "vit_name", "vit_patches_size", "n_skip", "num_heads", "num_layers",
    "num_classes", "use_swin", "use_efficientnet", "use_shsa",
    "use_alternate_shsa", "use_ats", "use_gumbel_topk", "topk_attn",
)
TABLE_START = r"""\begin{table*}[!t]
\caption{QUANTIZATION PERFORMANCE COMPARISON ACROSS MEDICAL SEGMENTATION DATASETS}
\label{tab:quantization_performance}
\centering
\setlength{\extrarowheight}{2pt}
\renewcommand{\arraystretch}{1.15}
\resizebox{\textwidth}{!}{%
\begin{tabular}{clccccccc}
\hline
\multicolumn{1}{c}{\makecell[c]{Dataset}} &
\multicolumn{1}{c}{\makecell[c]{Method}} &
\makecell[c]{Dice (\%) $\uparrow$} &
\makecell[c]{IoU (\%) $\uparrow$} &
\makecell[c]{HD95 $\downarrow$} &
\makecell[c]{Throughput\\(img/s) $\uparrow$} &
\makecell[c]{Latency\\(ms) $\downarrow$} &
\makecell[c]{Compression\\Ratio $\uparrow$} &
\makecell[c]{Memory\\(MiB)$^{*}$ $\downarrow$} \\[2pt]
\hline

"""
TABLE_END = r"""
\multicolumn{9}{l}{\scriptsize
$^{*}$ Peak allocated runtime memory. Values in parentheses denote standard deviation.}
\end{tabular}%
}
\end{table*}
"""


class InputError(ValueError):
    """An input cannot be interpreted without guessing."""


class Result(NamedTuple):
    path: Path
    data: dict[str, Any]

    def get(self, field: str) -> Any:
        value = self.data
        for key in field.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value


def warn(result: Result, message: str) -> None:
    print(f"warning: {result.path}: {message}", file=sys.stderr)


def metadata(result: Result) -> str:
    fields = (
        "arguments.quantize", "arguments.quantize_backend",
        "arguments.use_se_block", "arguments.drop_se_block",
        "arguments.se_aux_loss", "notes.quantized_model",
    )
    return json.dumps({key: result.get(key) for key in fields}, sort_keys=True)


def boolean(result: Result, field: str) -> bool | None:
    value = result.get(field)
    if value is not None and not isinstance(value, bool):
        raise InputError(f"{field} must be a JSON boolean, got {value!r}")
    return value


def identify_method(result: Result) -> str:
    quantize = boolean(result, "arguments.quantize")
    packed = boolean(result, "notes.quantized_model")
    backend = result.get("arguments.quantize_backend")
    if quantize is False:
        if packed is True:
            raise InputError("conflicting quantize=false and quantized_model=true")
        return METHODS[0]
    if quantize is None and packed is not True:
        raise InputError("missing quantization status; quantized_model=false alone "
                         "does not distinguish INC AWQ from a non-quantized model")
    if backend in ("inc_awq", "inc_awq_packed"):
        return METHODS[1]
    if backend == "legacy":
        if packed is False:
            raise InputError("legacy quantization conflicts with quantized_model=false")
        use_se = boolean(result, "arguments.use_se_block")
        if use_se is not None:
            if use_se and boolean(result, "arguments.drop_se_block") is True:
                raise InputError("legacy SE guidance conflicts with arguments.drop_se_block=true")
            # se_aux_loss is a training flag, disabled at inference; it does
            # not select the quantizer's scaling method.
            return METHODS[2] if use_se else METHODS[1]
        raise InputError("legacy quantization requires arguments.use_se_block "
                         "to distinguish AWQ from SE-AUX")
    raise InputError(f"missing or unsupported arguments.quantize_backend={backend!r}")


def dataset_name(path: Path) -> str:
    return "Cataract-1K" if path.name == "Cataract1k" else path.name


def dataset_order(path: Path) -> tuple[int, str, str]:
    name = dataset_name(path)
    priority = {"Cataract-1K": 0, "Synapse": 1, "ACDC": 2}
    return priority.get(name, 3), name.casefold(), name


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently choosing the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key {key!r}: {result[key]!r} and {value!r}")
        result[key] = value
    return result


def load_datasets(directory: Path) -> list[tuple[str, dict[str, Result]]]:
    if not directory.is_dir():
        raise InputError(f"{directory}: input must be an existing directory")
    folders = sorted((p for p in directory.iterdir() if p.is_dir()), key=dataset_order)
    if not folders:
        raise InputError(f"{directory}: no dataset subdirectories found")
    datasets = []
    errors = []
    for folder in folders:
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json")
        grouped: dict[str, list[Result]] = {method: [] for method in METHODS}
        inventory = []
        problems = []
        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
                if not isinstance(data, dict):
                    raise InputError("expected a JSON object at the top level")
                result = Result(path, data)
                inventory.append(f"  {path}: {metadata(result)}")
                grouped[identify_method(result)].append(result)
            except (OSError, UnicodeError, ValueError) as exc:
                problems.append(f"{path}: {exc}")
        for method, records in grouped.items():
            if len(records) != 1:
                problems.append(f"{folder}: expected exactly one {method} result; "
                                f"found {len(records)}" +
                                (": " + ", ".join(str(r.path) for r in records) if records else ""))
        if problems:
            errors.extend(problems)
            if inventory:
                errors.append(f"{folder}: method metadata:\n" + "\n".join(inventory))
        else:
            datasets.append((dataset_name(folder), {m: grouped[m][0] for m in METHODS}))
    if errors:
        raise InputError("\n".join(errors))
    return datasets


def finite_number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def number(result: Result, field: str, *, positive: bool = False) -> int | float | None:
    value = result.get(field)
    if not finite_number(value) or (value <= 0 if positive else value < 0):
        warn(result, f"{field} is missing, null, non-finite, or invalid ({value!r})")
        return None
    return value


def format_metric(result: Result, field: str, digits: int, scale: float = 1) -> str:
    value = number(result, field)
    if value is None:
        return "--"
    value *= scale
    if not finite_number(value):
        warn(result, f"{field} is non-finite after unit conversion; displaying --")
        return "--"
    return f"{value:.{digits}f}"


def format_statistic(result: Result, mean_field: str, std_field: str) -> str:
    mean = number(result, mean_field)
    std = number(result, std_field)
    if mean is None:
        return "--"
    return f"{mean:.3f}" + (f" ({std:.3f})" if std is not None else "")


def throughput(result: Result) -> str:
    metrics = result.get("metrics")
    mean_field = "metrics.throughput_img_s_mean"
    if isinstance(metrics, dict) and "throughput_img_s_mean" not in metrics:
        candidates = sorted(key for key in metrics if re.fullmatch(r"throughput_img_s_batch[1-9]\d*", key))
        if len(candidates) > 1:
            warn(result, "ambiguous throughput fallback fields: " + ", ".join(candidates) + "; displaying --")
            return "--"
        if len(candidates) == 1:
            mean_field = "metrics." + candidates[0]
    return format_statistic(result, mean_field, "metrics.throughput_img_s_std")


def memory(result: Result) -> str:
    raw = result.get("model_runtime_memory.peak_allocated_bytes")
    if finite_number(raw) and raw >= 0:
        return f"{raw / 2**20:.1f}"
    warn(result, f"model_runtime_memory.peak_allocated_bytes is missing or invalid ({raw!r})")
    if result.get("model_runtime_memory.unit") == "MiB":
        return format_metric(result, "model_runtime_memory.peak_allocated_mb", 1)
    warn(result, "peak_allocated_mb fallback requires model_runtime_memory.unit='MiB'; displaying --")
    return "--"


def input_shape(result: Result) -> tuple[int | None, int, int]:
    """Cross-check reported runtime C,H,W against the configured image size."""
    spatial = []
    channels = []
    image_size = result.get("arguments.img_size")
    runtime_size = result.get("model_runtime_memory.input_size")
    if image_size is not None:
        size = [image_size, image_size] if isinstance(image_size, (int, float)) else image_size
        if not isinstance(size, list) or len(size) != 2:
            raise InputError(f"invalid arguments.img_size={image_size!r}")
        spatial.append(tuple(size))
    if runtime_size is not None:
        if not isinstance(runtime_size, list) or len(runtime_size) != 3:
            raise InputError(f"invalid model_runtime_memory.input_size={runtime_size!r}")
        channels.append(runtime_size[0])
        spatial.append(tuple(runtime_size[1:]))
    metric_channels = result.get("metrics.channels")
    if metric_channels is not None:
        channels.append(metric_channels)
    if not spatial:
        raise InputError("missing input size (arguments.img_size or model_runtime_memory.input_size)")
    dimensions = [v for size in spatial for v in size] + channels
    if any(not finite_number(v) or v <= 0 or int(v) != v for v in dimensions):
        raise InputError(f"invalid input dimensions: {dimensions!r}")
    if any(size != spatial[0] for size in spatial) or any(c != channels[0] for c in channels):
        raise InputError("conflicting input sizes or channels in the same result")
    return (int(channels[0]) if channels else None, int(spatial[0][0]), int(spatial[0][1]))


def compression(baseline: Result, quantized: Result) -> str:
    issues = []
    for result in (baseline, quantized):
        backbone = result.get("arguments.vit_name")
        if not isinstance(backbone, str) or not backbone.strip():
            issues.append(f"{result.path}: missing or invalid arguments.vit_name")
    for field in ARCHITECTURE_FIELDS:
        left, right = baseline.get("arguments." + field), quantized.get("arguments." + field)
        if left != right:
            issues.append(f"arguments.{field}: baseline={left!r}, quantized={right!r}")
    shapes = []
    for result in (baseline, quantized):
        try:
            shapes.append(input_shape(result))
        except InputError as exc:
            issues.append(f"{result.path}: {exc}")
    if len(shapes) == 2 and shapes[0] != shapes[1]:
        issues.append(f"input sizes: baseline={shapes[0]!r}, quantized={shapes[1]!r}")
    baseline_bytes = number(baseline, "model_size.total_bytes", positive=True)
    quantized_bytes = number(quantized, "model_size.total_bytes", positive=True)
    if issues:
        warn(quantized, f"cannot verify compatible configurations with baseline {baseline.path}: "
             + "; ".join(issues) + "; compression displayed as --")
    if issues or baseline_bytes is None or quantized_bytes is None:
        return "--"
    ratio = baseline_bytes / quantized_bytes
    if not finite_number(ratio) or ratio <= 0:
        warn(quantized, f"invalid compression ratio against baseline {baseline.path}; displaying --")
        return "--"
    return f"{ratio:.3f}" + r"$\times$"


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "\n": " ", "\r": " ", "\t": " ",
    }
    return "".join(replacements.get(char, char) for char in text)


def render_table(datasets: list[tuple[str, dict[str, Result]]]) -> str:
    rows = []
    for name, methods in datasets:
        rows.append(r"\multirow{3}{*}{" + escape_latex(name) + "}")
        for method in METHODS:
            result = methods[method]
            cells = [
                escape_latex(method),
                format_metric(result, "accuracy.mean_dice", 2, 100),
                format_metric(result, "accuracy.mean_iou", 2, 100),
                format_metric(result, "accuracy.mean_hd95", 3),
                throughput(result),
                format_statistic(result, "metrics.latency_ms_mean_single", "metrics.latency_ms_std_single"),
                "--" if method == METHODS[0] else compression(methods[METHODS[0]], result),
                memory(result),
            ]
            rows.append("& " + " & ".join(cells) + r" \\")
        rows.append(r"\hline")
    return TABLE_START + "\n".join(rows) + "\n" + TABLE_END


def write_table(output: Path, table: str, sources: list[Path]) -> None:
    for source in sources:
        if output.resolve() == source.resolve() or (output.exists() and output.samefile(source)):
            raise InputError(f"{output}: refusing to overwrite source JSON {source}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".quantization_table_", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(table)
        os.replace(temporary, output)
    except OSError as exc:
        raise InputError(f"{output}: cannot write table: {exc}") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_directory", type=Path, help="directory containing dataset subdirectories")
    parser.add_argument("--output", type=Path, default=Path("performance_quantization.tex"),
                        help="output LaTeX fragment (default: %(default)s)")
    parser.add_argument("--output_directory", type=Path,
                        help="destination directory (created if needed); --output must be relative to it")
    args = parser.parse_args(argv)
    output = args.output
    if args.output_directory is not None:
        if output.is_absolute():
            parser.error("--output must be relative when --output_directory is provided")
        output = args.output_directory / output
    try:
        datasets = load_datasets(args.input_directory)
        table = render_table(datasets)
        if args.output_directory is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
        write_table(output, table, [r.path for _, methods in datasets for r in methods.values()])
    except (InputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Generated {output.resolve()}", file=sys.stderr)
    print("Parent LaTeX document requires: graphicx, array, makecell, multirow.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
