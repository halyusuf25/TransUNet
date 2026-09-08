#!/usr/bin/env python3
"""Plot Dice score versus inference throughput for the models in ``sota.csv``.

Examples
--------
Default (Synapse)::

    python tools/plot_dice_vs_throughput.py

ACDC only::

    python tools/plot_dice_vs_throughput.py --dataset ACDC

Cataract-1k with memory-scaled bubbles::

    python tools/plot_dice_vs_throughput.py \
        --dataset Cataract-1k --memorey --center_x_marker

All datasets::

    python tools/plot_dice_vs_throughput.py \
        --multiple_dataset Synapse ACDC Cataract-1k

Publication-ready output::

    python tools/plot_dice_vs_throughput.py \
        --multiple_dataset Synapse ACDC Cataract-1k \
        --memorey --paper_style

Proportional bubbles with exact-size memory references and highlighting::

    python tools/plot_dice_vs_throughput.py \
        --dataset Synapse --memorey --buble-size_legened \
        --model_center_markers --model_center_colors --model_marker_legend \
        --highlight_ours --paper_style

Omit the title when the manuscript caption supplies the context::

    python tools/plot_dice_vs_throughput.py \
        --dataset Synapse --memorey --highlight_ours --remove_title --paper_style

Right-side memory scale key (the two legend modifiers also work separately)::

    python tools/plot_dice_vs_throughput.py \
        --dataset Synapse --memorey --buble-size_legened \
        --right_memory_legend --scaled_memorey_legened \
        --model_center_markers --model_marker_legend --paper_style

Calibrate all bubbles to a physical reference in the lower-right corner::

    python tools/plot_dice_vs_throughput.py \
        --dataset Synapse --add_memory_legende --radius 1 --runtime_memory 1024 \
        --model_center_markers --model_center_colors --model_marker_legend --paper_style

Only --add_memory_legende activates the radius/MiB calibration. Without it,
the existing bubble scale is unchanged, even if --radius or --runtime_memory
is supplied. The calibrated circle radius is measured to the outline center.

The physical area key describes nominal circle area, excluding the outline,
at the exported dimensions. Resizing the figure changes its mm² calibration.

Custom labels::

    python tools/plot_dice_vs_throughput.py \
        --dataset Synapse \
        --title "Accuracy–Efficiency Trade-off on Synapse" \
        --xlabel "Throughput (img/sec)" \
        --ylabel "Dice (%)" \
        --dislay_model_names \
        --model_name_font_size 7 \
        --output_name synapse_accuracy_efficiency \
        --output_dir qualitative/sota_plots

Model-specific bubble-center symbols and a bottom legend::

    python tools/plot_dice_vs_throughput.py \
        --dataset Synapse --memorey \
        --model_center_markers --model_center_colors \
        --model_marker_legend --axis_label_font_size 9 \
        --paper_style

The misspelled options ``--memorey``, ``--dislay_model_names``, and
``--buble-size_legened`` are retained intentionally for compatibility with the
requested command-line interface.
"""

from __future__ import annotations

import argparse
import csv
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

# A non-interactive backend makes command-line execution reliable on servers.
matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.path import Path as PlotPath
from matplotlib.transforms import Bbox


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_COLUMNS: Mapping[str, str] = {
    "Synapse": "Synapse Dice (%)",
    "ACDC": "ACDC Dice (%)",
    "Cataract-1k": "Cataract-1K Dice (%)",
}

DATASET_STYLES: Mapping[str, Mapping[str, str]] = {
    "Synapse": {"marker": "o", "color": "#0072B2"},
    "ACDC": {"marker": "s", "color": "#D55E00"},
    "Cataract-1k": {"marker": "^", "color": "#009E73"},
}

MODEL_DISPLAY_NAMES: Mapping[str, str] = {
    "SwinUNet (Swin-Base)": "SwinUNet",
    "Lightweight-TransUNet (Ours)": "Lightweight-TransUNet",
}

# Filled silhouettes remain recognizable when reduced or printed in grayscale.
MODEL_MARKERS: Tuple[str, ...] = (
    "o",
    "s",
    "^",
    "v",
    "P",
    "X",
    "D",
    "d",
    "<",
    ">",
    "8",
)

# This palette intentionally excludes every outer-bubble dataset color so center
# symbols remain visually separate from their surrounding circles.
MODEL_COLORS: Tuple[str, ...] = (
    "#F0E442",
    "#CC79A7",
    "#6F4C9B",
    "#A6761D",
    "#E7298A",
    "#17BECF",
    "#BDBDBD",
    "#252525",
)
DEFAULT_MODEL_MARKER_COLOR = "#252525"
MODEL_CENTER_AREA_SCALE = 0.70
OURS_OUTLINE_COLOR = "#B2182B"
OURS_OUTLINE_WIDTH = 0.85
OURS_HIGHLIGHT_FILL = "#E69F00"
OURS_HIGHLIGHT_WIDTH = 1.5
OURS_HIGHLIGHT_LABEL = "Lightweight TransUNet (ours)"
DEFAULT_MODEL_LEGEND_FONT_SIZE = 7.0
MAX_MEMORY_MARKER_AREA = 900.0

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "Model",
    "Synapse Dice (%)",
    "Cataract-1K Dice (%)",
    "ACDC Dice (%)",
    "Throughput (img/s)",
    "Throughput Std",
    "Memory (MiB)",
    "GFLOPs",
)

NUMERIC_COLUMNS: Tuple[str, ...] = REQUIRED_COLUMNS[1:]
THROUGHPUT_COLUMN = "Throughput (img/s)"
THROUGHPUT_STD_COLUMN = "Throughput Std"
MEMORY_COLUMN = "Memory (MiB)"
OURS_MODEL = "Lightweight-TransUNet (Ours)"

DEFAULT_TITLE = "Comparision between SOTA and our Lightweight Model"
DEFAULT_XLABEL = "Throughput (images/s)"
DEFAULT_YLABEL = "Dice (%)"

FONT_PREFERENCES: Tuple[str, ...] = (
    "Times New Roman",
    "Arial",
    "Helvetica",
    "DejaVu Serif",
)


@dataclass(frozen=True)
class PlotStyle:
    """Size and typography values for one output style."""

    figure_size: Tuple[float, float]
    title_size: float
    label_size: float
    tick_size: float
    annotation_size: float
    legend_size: float
    fixed_marker_area: float
    center_marker_area: float
    axis_line_width: float


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    examples = """examples:
  %(prog)s
  %(prog)s --dataset ACDC
  %(prog)s --dataset Cataract-1k --memorey --center_x_marker
  %(prog)s --multiple_dataset Synapse ACDC Cataract-1k
  %(prog)s --multiple_dataset Synapse ACDC Cataract-1k --memorey --paper_style
  %(prog)s --dataset Synapse --dislay_model_names --model_name_font_size 7
  %(prog)s --dataset Synapse --memorey --buble-size_legened --highlight_ours --paper_style
  %(prog)s --dataset Synapse --memorey --remove_title --paper_style
  %(prog)s --dataset Synapse --add_memory_legende --radius 1 --runtime_memory 1024 --paper_style
  %(prog)s --dataset Synapse --memorey --buble-size_legened \
      --right_memory_legend --scaled_memorey_legened --paper_style
  %(prog)s --dataset Synapse --memorey --model_center_markers \
      --model_center_colors --model_marker_legend --axis_label_font_size 9 \
      --paper_style
"""
    parser = argparse.ArgumentParser(
        description=(
            "Create PDF and PNG Dice-versus-throughput plots from sota.csv. "
            "Outputs are written to --output_dir."
        ),
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_COLUMNS),
        default="Synapse",
        help="dataset used for a single-series plot (default: %(default)s)",
    )
    parser.add_argument(
        "--multiple_dataset",
        nargs="+",
        choices=tuple(DATASET_COLUMNS),
        default=None,
        metavar="DATASET",
        help=(
            "plot multiple datasets in the supplied order; this takes "
            "precedence over --dataset"
        ),
    )
    parser.add_argument(
        "--memorey",
        action="store_true",
        help="make bubble area proportional to memory in MiB (spelling is intentional)",
    )
    parser.add_argument(
        "--add_memory_legende",
        action="store_true",
        help=(
            "enable physically calibrated memory bubbles (implies --memorey) and a "
            "compact lower-right legend; uses --radius and --runtime_memory"
        ),
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        metavar="MM",
        help=(
            "reference-circle radius in millimeters at the exported size "
            "(default: %(default)s); used only with --add_memory_legende"
        ),
    )
    parser.add_argument(
        "--runtime_memory",
        type=float,
        default=1024.0,
        metavar="MIB",
        help=(
            "memory represented by the reference circle (default: %(default)s MiB); "
            "used only with --add_memory_legende"
        ),
    )
    parser.add_argument(
        "--buble-size_legened",
        action="store_true",
        help=(
            "show distinct min/median/max MiB references at the plotted bubble sizes; "
            "requires --memorey "
            "(spelling is intentional)"
        ),
    )
    parser.add_argument(
        "--right_memory_legend",
        action="store_true",
        help=(
            "place the memory legend outside the plot on the right in one column, "
            "reserving extra figure width; used only with --buble-size_legened"
        ),
    )
    parser.add_argument(
        "--scaled_memorey_legened",
        action="store_true",
        help=(
            "replace memory-value labels with a computed bubble-area scale in mm² "
            "at the exported physical size (outline excluded); used only with "
            "--buble-size_legened"
        ),
    )
    parser.add_argument(
        "--highlight_ours",
        action="store_true",
        help=(
            "highlight Lightweight TransUNet with a direct label and clearer outline; "
            "also use a distinct fill for a single dataset"
        ),
    )
    parser.add_argument(
        "--dislay_model_names",
        action="store_true",
        help="annotate model names beside their points (spelling is intentional)",
    )
    parser.add_argument(
        "--model_name_font_size",
        type=float,
        default=None,
        metavar="POINTS",
        help=(
            "model-annotation font size in points; used with "
            "--dislay_model_names (default: style-specific)"
        ),
    )
    parser.add_argument(
        "--throughput_std",
        action="store_true",
        help="show Throughput Std as horizontal error bars",
    )
    parser.add_argument(
        "--center_x_marker",
        action="store_true",
        help=(
            "use a thin x instead of a dot at each bubble center; requires "
            "--memorey and a single selected dataset"
        ),
    )
    parser.add_argument(
        "--model_center_markers",
        action="store_true",
        help=(
            "use a distinct bubble-center marker for each model; requires "
            "--memorey and cannot be combined with --center_x_marker"
        ),
    )
    parser.add_argument(
        "--model_center_colors",
        action="store_true",
        help=(
            "give every model-center marker a distinct color; requires "
            "--model_center_markers"
        ),
    )
    parser.add_argument(
        "--model_marker_legend",
        type=float,
        nargs="?",
        const=DEFAULT_MODEL_LEGEND_FONT_SIZE,
        default=None,
        metavar="POINTS",
        help=(
            "show a model-marker legend below the plot, adding rows as needed; "
            "optional font size in points (a bare flag uses 7 pt); requires "
            "--model_center_markers"
        ),
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help=(
            "centered plot title; use \\n for an explicit line break; long titles "
            "automatically wrap to two lines (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--remove_title",
        action="store_true",
        help="hide the title and reclaim its space, overriding --title in all styles",
    )
    parser.add_argument(
        "--xlabel",
        default=DEFAULT_XLABEL,
        help="x-axis label (default: %(default)s)",
    )
    parser.add_argument(
        "--ylabel",
        default=DEFAULT_YLABEL,
        help="y-axis label (default: %(default)s)",
    )
    parser.add_argument(
        "--title_font_size",
        type=float,
        default=None,
        metavar="POINTS",
        help="plot-title font size in points (default: style-specific)",
    )
    parser.add_argument(
        "--axis_label_font_size",
        type=float,
        default=None,
        metavar="POINTS",
        help=(
            "shared font size for both axis labels and both axes' tick "
            "numbers, in points "
            "(default: style-specific)"
        ),
    )
    parser.add_argument(
        "--center_icon_size",
        type=float,
        default=None,
        metavar="POINTS_SQUARED",
        help=(
            "area of bubble-center icons in points squared; requires "
            "--memorey (default: style-specific)"
        ),
    )
    parser.add_argument(
        "--paper_style",
        action="store_true",
        help="use IEEE-sized typography and save the PNG at 1800 dpi",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="show a light major grid (enabled automatically by --paper_style)",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help=(
            "custom filename stem shared by the PDF and PNG; an optional "
            ".pdf or .png suffix is removed (default: dataset-based name)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("qualitative"),
        help="directory for generated PDF and PNG files (default: %(default)s)",
    )
    return parser


def parse_selected_datasets(
    dataset: str,
    multiple_dataset: Optional[Sequence[str]],
) -> List[str]:
    """Resolve dataset selection, preserving order while removing duplicates."""

    requested = list(multiple_dataset) if multiple_dataset is not None else [dataset]
    selected: List[str] = []
    for name in requested:
        if name not in DATASET_COLUMNS:
            choices = ", ".join(DATASET_COLUMNS)
            raise ValueError("Unknown dataset {!r}; choose from {}.".format(name, choices))
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("At least one dataset must be selected.")
    return selected


def display_model_name(model_name: str) -> str:
    """Return the publication-facing name without changing the source CSV."""

    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def build_model_marker_styles(
    model_names: Sequence[str],
    use_colors: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Assign deterministic, distinct center-marker styles to model names."""

    unique_names = list(dict.fromkeys(str(name) for name in model_names))
    if not unique_names:
        raise ValueError("At least one model is required for model-center markers.")
    if len(unique_names) > len(MODEL_MARKERS):
        raise ValueError(
            "Model-center markers support at most {} unique models; found {}.".format(
                len(MODEL_MARKERS), len(unique_names)
            )
        )
    dataset_colors = {
        style["color"].lower() for style in DATASET_STYLES.values()
    }
    reused_colors = [
        color for color in MODEL_COLORS if color.lower() in dataset_colors
    ]
    if use_colors and reused_colors:
        raise ValueError(
            "Model-center colors must not reuse an outer-circle color: {}.".format(
                ", ".join(reused_colors)
            )
        )
    if use_colors and len(unique_names) > len(MODEL_COLORS):
        raise ValueError(
            "Distinct model-center colors support at most {} unique models; found {}.".format(
                len(MODEL_COLORS), len(unique_names)
            )
        )

    return {
        model_name: {
            "marker": MODEL_MARKERS[index],
            "color": MODEL_COLORS[index] if use_colors else DEFAULT_MODEL_MARKER_COLOR,
        }
        for index, model_name in enumerate(unique_names)
    }


def resolve_csv_path() -> Path:
    """Locate ``sota.csv`` beside the script or at the repository root."""

    candidates = (SCRIPT_DIR / "sota.csv", ROOT_DIR / "sota.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError("Could not find sota.csv. Searched: {}".format(searched))


def _csv_row_numbers(mask: pd.Series) -> str:
    """Return compact, one-based CSV row numbers for a validation mask."""

    rows = [str(int(index) + 2) for index in mask.index[mask]]
    if len(rows) > 8:
        return ", ".join(rows[:8]) + ", ..."
    return ", ".join(rows)


def load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    """Load the source CSV and validate its schema and required values."""

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError("CSV file not found: {}".format(path.resolve()))

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = csv.reader(csv_file, strict=True)
            try:
                header = next(rows)
            except StopIteration as exc:
                raise ValueError("CSV file is empty: {}".format(path.resolve())) from exc
            expected_fields = len(header)
            for row_number, row in enumerate(rows, start=2):
                if not row or all(not field.strip() for field in row):
                    continue
                if len(row) != expected_fields:
                    raise ValueError(
                        "Malformed CSV file {}: row {} has {} field(s); expected {}.".format(
                            path.resolve(), row_number, len(row), expected_fields
                        )
                    )
        data = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="error")
    except csv.Error as exc:
        raise ValueError("Malformed CSV file {}: {}".format(path.resolve(), exc)) from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError("CSV file is empty: {}".format(path.resolve())) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            "Malformed CSV file {}: {}".format(path.resolve(), exc)
        ) from exc
    except UnicodeError as exc:
        raise ValueError(
            "Could not decode CSV file {} as UTF-8: {}".format(
                path.resolve(), exc
            )
        ) from exc
    except OSError as exc:
        raise OSError("Could not read CSV file {}: {}".format(path.resolve(), exc)) from exc

    if data.empty:
        raise ValueError("CSV file contains no model rows: {}".format(path.resolve()))

    expected_index = pd.RangeIndex(start=0, stop=len(data), step=1)
    if not data.index.equals(expected_index):
        raise ValueError(
            "Malformed CSV file {}: data rows do not match the header field count.".format(
                path.resolve()
            )
        )

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "CSV file is missing required column(s): {}.".format(
                ", ".join(missing_columns)
            )
        )

    # Work on a private copy and retain only the documented schema.
    validated = data.loc[:, REQUIRED_COLUMNS].copy().reset_index(drop=True)

    model_text = validated["Model"].astype("string").str.strip()
    missing_models = model_text.isna() | model_text.eq("")
    if bool(missing_models.any()):
        raise ValueError(
            "Column 'Model' contains a missing or blank value at CSV row(s): {}.".format(
                _csv_row_numbers(missing_models)
            )
        )
    validated["Model"] = model_text.astype(str)

    for column in NUMERIC_COLUMNS:
        raw = validated[column]
        text_values = raw.astype("string").str.strip()
        missing = raw.isna() | text_values.isna() | text_values.eq("")
        numeric = pd.to_numeric(raw, errors="coerce")
        nonnumeric = ~missing & numeric.isna()
        if bool(nonnumeric.any()):
            raise ValueError(
                "Column {!r} contains nonnumeric value(s) at CSV row(s): {}.".format(
                    column, _csv_row_numbers(nonnumeric)
                )
            )
        if column != THROUGHPUT_STD_COLUMN and bool(missing.any()):
            raise ValueError(
                "Column {!r} contains missing value(s) at CSV row(s): {}.".format(
                    column, _csv_row_numbers(missing)
                )
            )

        finite_values = numeric.notna() & ~np.isfinite(numeric.astype(float))
        if bool(finite_values.any()):
            raise ValueError(
                "Column {!r} contains non-finite value(s) at CSV row(s): {}.".format(
                    column, _csv_row_numbers(finite_values)
                )
            )
        validated[column] = numeric.astype(float)

    negative_std = validated[THROUGHPUT_STD_COLUMN].notna() & (
        validated[THROUGHPUT_STD_COLUMN] < 0
    )
    if bool(negative_std.any()):
        raise ValueError(
            "Column 'Throughput Std' contains negative value(s) at CSV row(s): {}.".format(
                _csv_row_numbers(negative_std)
            )
        )

    return validated


def scale_memory_to_marker_area(
    memory: pd.Series,
    max_area: float = MAX_MEMORY_MARKER_AREA,
    scale_factor: Optional[float] = None,
) -> np.ndarray:
    """Map positive memory to proportional marker areas in points squared.

    By default the largest memory value receives ``max_area``. Supplying the
    resulting points-squared-per-MiB factor reuses that exact mapping for other
    series or memory-legend references, including equal-valued inputs.
    """

    if not np.isfinite(max_area) or max_area <= 0:
        raise ValueError("Maximum marker area must be finite and greater than zero.")
    if scale_factor is not None and (
        not np.isfinite(scale_factor) or scale_factor <= 0
    ):
        raise ValueError("Memory scale factor must be finite and greater than zero.")

    try:
        values = pd.to_numeric(memory, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Memory values must be numeric.") from exc

    if values.size == 0:
        raise ValueError("At least one memory value is required for bubble scaling.")
    if not bool(np.isfinite(values).all()):
        raise ValueError("Memory values must all be finite.")
    if bool((values <= 0).any()):
        raise ValueError("Memory values must all be greater than zero MiB.")

    if scale_factor is None:
        scale_factor = max_area / float(values.max())
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        areas = values * scale_factor
    if not bool(np.isfinite(areas).all()) or bool((areas <= 0).any()):
        raise ValueError("Memory scaling must produce finite, positive marker areas.")
    return areas


def _reference_memory_scale(radius: float, runtime_memory: float) -> float:
    """Convert the reference radius/MiB pair to scatter points squared per MiB."""

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("--radius must be a finite value greater than zero millimeters.")
    if not np.isfinite(runtime_memory) or runtime_memory <= 0:
        raise ValueError("--runtime_memory must be a finite value greater than zero MiB.")
    # The circular scatter path has unit diameter, so sqrt(s) is its diameter
    # in points, not its radius or its disk area. The outline is decorative;
    # the calibrated radius is measured to the outline's centerline.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        diameter_points = np.float64(radius) * (2.0 * 72.0 / 25.4)
        scale_factor = np.square(diameter_points) / runtime_memory
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("The radius/memory reference must give a finite, positive bubble scale.")
    return float(scale_factor)


def _select_font() -> str:
    """Select the first installed publication-compatible font."""

    available = {font.name for font in font_manager.fontManager.ttflist}
    for preferred in FONT_PREFERENCES:
        if preferred in available:
            return preferred
    return "DejaVu Serif"


def apply_paper_style(
    paper_style: bool,
    dataset_count: int,
    bottom_model_legend: bool = False,
) -> PlotStyle:
    """Apply consistent Matplotlib settings and return plot dimensions."""

    if dataset_count < 1:
        raise ValueError("dataset_count must be at least one.")

    selected_font = _select_font()
    matplotlib.rcParams.update(
        {
            "font.family": selected_font,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    if paper_style:
        if dataset_count == 1:
            # Legend space is measured and added after the plot layout is known.
            figure_size = (3.5, 2.8)
        else:
            figure_size = (7.16, 3.8)
        style = PlotStyle(
            figure_size=figure_size,
            title_size=9.5,
            label_size=9.0,
            tick_size=8.0,
            annotation_size=7.0,
            legend_size=8.0,
            fixed_marker_area=55.0,
            center_marker_area=20.0,
            axis_line_width=0.7,
        )
    else:
        style = PlotStyle(
            figure_size=(8.0, 5.0),
            title_size=13.0,
            label_size=11.0,
            tick_size=9.5,
            annotation_size=8.5,
            legend_size=9.0,
            fixed_marker_area=75.0,
            center_marker_area=28.0,
            axis_line_width=0.9,
        )

    matplotlib.rcParams.update(
        {
            "axes.titlesize": style.title_size,
            "axes.labelsize": style.label_size,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "legend.fontsize": style.legend_size,
            "axes.linewidth": style.axis_line_width,
        }
    )
    return style


def _padded_limits(
    values: np.ndarray,
    errors: Optional[np.ndarray] = None,
    padding_fraction: float = 0.08,
) -> Tuple[float, float]:
    """Return finite, data-derived limits with proportional padding."""

    numeric_values = np.asarray(values, dtype=float)
    if numeric_values.size == 0 or not bool(np.isfinite(numeric_values).all()):
        raise ValueError("Axis values must be a non-empty finite array.")

    if errors is None:
        finite_errors = np.zeros_like(numeric_values)
    else:
        supplied_errors = np.asarray(errors, dtype=float)
        if supplied_errors.shape != numeric_values.shape:
            raise ValueError("Axis errors must have the same shape as axis values.")
        finite_errors = np.where(np.isfinite(supplied_errors), supplied_errors, 0.0)

    lower = float(np.min(numeric_values - finite_errors))
    upper = float(np.max(numeric_values + finite_errors))
    span = upper - lower
    if span == 0:
        padding = max(abs(lower) * padding_fraction, 1.0)
    else:
        padding = span * padding_fraction
    return lower - padding, upper + padding


def _draw_error_bars(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    standard_deviations: np.ndarray,
) -> None:
    """Draw horizontal errors for rows that contain a standard deviation."""

    present = np.isfinite(standard_deviations)
    if not bool(present.any()):
        return
    ax.errorbar(
        x_values[present],
        y_values[present],
        xerr=standard_deviations[present],
        fmt="none",
        ecolor="#707070",
        elinewidth=0.7,
        capsize=2.0,
        capthick=0.7,
        alpha=0.75,
        zorder=1,
        label="_nolegend_",
    )


def _pad_for_marker_extents(ax: plt.Axes, bottom_padding_points: float = 0.0) -> None:
    """Expand limits using physical marker/outline sizes at the final axes size."""

    figure = ax.figure
    figure.canvas.draw()
    points_to_pixels = figure.dpi / 72.0
    coordinates = []
    lower_extents = []
    upper_extents = []
    for collection in ax.collections:
        if not isinstance(collection, PathCollection):
            continue
        paths = collection.get_paths()
        sizes = collection.get_sizes()
        widths = collection.get_linewidths()
        for index, offset in enumerate(collection.get_offsets()):
            bounds = paths[index % len(paths)].get_extents()
            size = float(np.sqrt(sizes[index % len(sizes)]))
            # Include half the stroke and a small clearance from the axis spine.
            clearance = widths[index % len(widths)] / 2.0 + 1.5
            coordinates.append(offset)
            lower_extents.append(
                (clearance - bounds.x0 * size, clearance - bounds.y0 * size)
            )
            upper_extents.append(
                (clearance + bounds.x1 * size, clearance + bounds.y1 * size)
            )

    # Error-bar caps are Line2D markers in physical points, too. Their data
    # endpoints already incorporate the optional throughput standard deviation.
    for line in ax.lines:
        if line.get_marker() != "|":
            continue
        clearance = line.get_markeredgewidth() / 2.0 + 1.5
        radius = line.get_markersize() / 2.0 + clearance
        for x_value, y_value in zip(line.get_xdata(), line.get_ydata()):
            coordinates.append((x_value, y_value))
            lower_extents.append((clearance, radius))
            upper_extents.append((clearance, radius))

    if not coordinates:
        return
    values = np.asarray(coordinates, dtype=float)
    lower_pixels = np.asarray(lower_extents) * points_to_pixels
    lower_pixels[:, 1] += bottom_padding_points * points_to_pixels
    upper_pixels = np.asarray(upper_extents) * points_to_pixels
    for dimension, (get_limits, set_limits, pixel_span) in enumerate((
        (ax.get_xlim, ax.set_xlim, ax.bbox.width),
        (ax.get_ylim, ax.set_ylim, ax.bbox.height),
    )):
        low_fraction = lower_pixels[:, dimension] / pixel_span
        high_fraction = upper_pixels[:, dimension] / pixel_span
        if np.any(low_fraction + high_fraction >= 1.0):
            raise ValueError(
                "A marker is larger than the plot; reduce --center_icon_size or --radius, "
                "or increase --runtime_memory for calibrated bubbles."
            )
        lower, upper = get_limits()
        # Expanding the data span changes the data-to-point conversion. Iterate
        # to convergence without moving any coordinates or changing any areas.
        for _ in range(100):
            span = upper - lower
            new_lower = min(lower, float(np.min(values[:, dimension] - low_fraction * span)))
            new_upper = max(upper, float(np.max(values[:, dimension] + high_fraction * span)))
            change_pixels = max(lower - new_lower, new_upper - upper) / span * pixel_span
            lower, upper = new_lower, new_upper
            if change_pixels < 1e-7:
                break
        set_limits(lower, upper)


class _MemoryLegendHandler(HandlerBase):
    """Pack each reference at its real diameter, centered beside its label."""

    def __init__(self, handle_width: float = 0.0):
        super().__init__()
        self.handle_width = handle_width

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        diameter = orig_handle.get_markersize() + orig_handle.get_markeredgewidth()
        handlebox.width = max(diameter + 4.0, self.handle_width)
        handlebox.height = max(diameter + 4.0, fontsize)
        # Align marker centers to the text's approximate midline. Each drawing
        # box includes the full outline, even for a reference larger than text.
        center_y = fontsize * 0.35
        handlebox.ydescent = handlebox.height / 2.0 - center_y
        marker = Line2D([handlebox.width / 2.0], [center_y], linestyle="none")
        self.update_prop(marker, orig_handle, legend)
        marker.set_transform(handlebox.get_transform())
        handlebox.add_artist(marker)
        return marker


def _memory_scale_label(scale_factor: float, vertical: bool = False) -> str:
    """Describe nominal circle area in physical units, independent of export DPI."""

    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("Memory scale factor must be finite and greater than zero.")
    # For scatter(marker='o'), sqrt(s) is the diameter in points. The disk
    # area is therefore pi/4 * s, not s itself. Outlines are decorative and
    # excluded from this geometric area; 72 points = 1 inch = 25.4 mm.
    square_mm_per_mib = (np.pi / 4.0) * scale_factor * (25.4 / 72.0) ** 2
    mib_per_square_mm = 1.0 / square_mm_per_mib
    separator = "\n" if vertical else " "
    return "Bubble area:{}1 mm² ≈ {:.4g} MiB\n(outline excluded)".format(
        separator, mib_per_square_mm
    )


def _create_memory_legend(
    figure: Figure,
    handles: Sequence[Line2D],
    font_size: float,
    right_side: bool = False,
    scale_factor: Optional[float] = None,
) -> Legend:
    """Reserve real circle diameters in the legend's horizontal/vertical cells."""

    title = "Memory (MiB)"
    if scale_factor is not None:
        title = _memory_scale_label(scale_factor, vertical=right_side)
        for handle in handles:
            handle.set_label("")
    handle_width = (
        max(handle.get_markersize() + handle.get_markeredgewidth() + 4.0 for handle in handles)
        if right_side else 0.0
    )
    column_counts = (1,) if right_side else range(len(handles), 0, -1)
    for columns in column_counts:
        ordered_handles = [
            handles[index]
            for column in range(columns)
            for index in range(column, len(handles), columns)
        ]
        legend = figure.legend(
            handles=ordered_handles,
            handler_map={Line2D: _MemoryLegendHandler(handle_width=handle_width)},
            markerscale=1.0,
            title=title,
            loc="upper left" if right_side else "upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=columns,
            fontsize=font_size,
            title_fontsize=font_size,
            frameon=False,
            borderaxespad=0.0,
            borderpad=0.4,
            handletextpad=0.0 if scale_factor is not None else 0.7,
            columnspacing=1.2,
            labelspacing=0.8,
        )
        legend.get_title().set_multialignment("center")
        figure.canvas.draw()
        if right_side or legend.get_window_extent().width <= figure.bbox.width * 0.96:
            return legend
        legend.remove()
    raise ValueError("Memory-legend labels do not fit within the figure width.")


def _create_reference_memory_legend(
    ax: plt.Axes,
    radius: float,
    runtime_memory: float,
    scale_factor: float,
    font_size: float,
    obstacles: Sequence[object],
) -> Legend:
    """Place the calibrated reference near the lower-right, clear of data."""

    reference_area = scale_memory_to_marker_area(
        pd.Series([runtime_memory]), scale_factor=scale_factor
    )[0]
    reference = Line2D(
        [], [], linestyle="none", marker="o",
        markersize=float(np.sqrt(reference_area)),
        markerfacecolor="#B8B8B8", markeredgecolor="#303030", markeredgewidth=0.75,
    )
    legend = Legend(
        ax,
        handles=[reference],
        labels=["{:.12g} mm radius = {:.12g} MiB".format(radius, runtime_memory)],
        handler_map={Line2D: _MemoryLegendHandler()},
        markerscale=1.0,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        bbox_transform=ax.transAxes,
        fontsize=font_size,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#B8B8B8",
        borderpad=0.5,
        borderaxespad=0.6,
        handletextpad=0.6,
    )
    legend.get_frame().set_linewidth(0.6)
    ax.add_artist(legend)
    # The key is inside the already-laid-out axes; it must not reserve another
    # external legend row or change the panel's physical dimensions.
    legend.set_in_layout(False)
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    pixels_per_point = figure.dpi / 72.0
    axes_box = ax.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer)
    inset = font_size * 0.6 * pixels_per_point
    if (
        legend_box.width + 2 * inset > axes_box.width
        or legend_box.height + 2 * inset > axes_box.height
    ):
        legend.remove()
        raise ValueError(
            "The memory reference legend does not fit inside the plot; "
            "choose a smaller --radius or shorter reference values."
        )

    def data_boxes() -> List[Bbox]:
        boxes = [artist.get_window_extent(renderer) for artist in obstacles]
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                centers = collection.get_offset_transform().transform(collection.get_offsets())
                sizes = collection.get_sizes()
                widths = collection.get_linewidths()
                paths = collection.get_paths()
                for index, (x_value, y_value) in enumerate(centers):
                    bounds = paths[index % len(paths)].get_extents()
                    scale = np.sqrt(sizes[index % len(sizes)]) * pixels_per_point
                    margin = (widths[index % len(widths)] / 2.0 + 1.5) * pixels_per_point
                    boxes.append(Bbox.from_extents(
                        x_value + bounds.x0 * scale - margin,
                        y_value + bounds.y0 * scale - margin,
                        x_value + bounds.x1 * scale + margin,
                        y_value + bounds.y1 * scale + margin,
                    ))
            elif isinstance(collection, LineCollection):
                for segment in collection.get_segments():
                    points = collection.get_transform().transform(segment)
                    if len(points):
                        boxes.append(Bbox.from_extents(
                            *points.min(axis=0), *points.max(axis=0)
                        ).padded(2.0 * pixels_per_point))
        for line in ax.lines:
            if line.get_marker() == "|":
                margin = (line.get_markeredgewidth() / 2.0 + 1.5) * pixels_per_point
                half_height = line.get_markersize() / 2.0 * pixels_per_point + margin
                for x_value, y_value in line.get_transform().transform(line.get_xydata()):
                    boxes.append(Bbox.from_extents(
                        x_value - margin, y_value - half_height,
                        x_value + margin, y_value + half_height,
                    ))
        return boxes

    occupied = data_boxes()
    # Prefer the corner itself. If needed, move only slightly upward within the
    # lower-right portion of the axes, keeping the reference's physical size.
    step = max(2.0, font_size / 2.0) * pixels_per_point
    for offset in np.arange(0.0, axes_box.height * 0.35 + step, step):
        legend.set_bbox_to_anchor((1.0, offset / axes_box.height), transform=ax.transAxes)
        candidate_box = legend.get_window_extent(renderer)
        if candidate_box.y1 > axes_box.y1 - inset:
            break
        if not any(candidate_box.overlaps(box) for box in occupied):
            return legend

    # A crowded lower-right region needs a small empty band below the data.
    # Reserve it through limits, keeping all coordinates and marker areas intact.
    legend.set_bbox_to_anchor((1.0, 0.0), transform=ax.transAxes)
    reserved_points = (legend_box.height + inset) / pixels_per_point + 3.0
    _pad_for_marker_extents(ax, bottom_padding_points=reserved_points)
    return legend


def _layout_plot_legends(
    figure: Figure,
    ax: plt.Axes,
    legends: Sequence[Legend],
    paper_style: bool,
    right_legend: Optional[Legend] = None,
) -> None:
    """Reserve measured bottom/right legend space without scaling the artwork."""

    plot_width = figure.get_figwidth()
    for legend in legends:
        legend.set_in_layout(False)
    legends = [legend for legend in legends if legend is not right_legend]
    figure.tight_layout(pad=0.8 if paper_style else 1.0)
    if legends:
        figure.canvas.draw()
        heights = [legend.get_window_extent().height / figure.dpi for legend in legends]
        gap = 6.0 / 72.0
        extra_height = sum(heights) + gap * (len(legends) + 1)
        width, height = figure.get_size_inches()
        position = ax.get_position()
        figure.set_size_inches(width, height + extra_height)
        ax.set_position((
            position.x0,
            (position.y0 * height + extra_height) / (height + extra_height),
            position.width,
            position.height * height / (height + extra_height),
        ))
        top = extra_height - gap
        for legend, legend_height in zip(legends, heights):
            legend.set_bbox_to_anchor(
                (0.5, top / (height + extra_height)), transform=figure.transFigure
            )
            legend.set_in_layout(True)
            top -= legend_height + gap
    if right_legend is not None:
        figure.canvas.draw()
        legend_box = right_legend.get_window_extent()
        legend_width = legend_box.width / figure.dpi
        legend_height = legend_box.height / figure.dpi
        width, height = figure.get_size_inches()
        position = ax.get_position()
        axes_bottom = position.y0 * height
        axes_height = position.height * height
        extra_height = max(0.0, legend_height - axes_height)
        final_width = width + legend_width + 6.0 / 72.0
        final_height = height + extra_height
        bottom_anchors = [legend.get_bbox_to_anchor().y0 / figure.dpi for legend in legends]
        figure.set_size_inches(final_width, final_height)
        ax.set_position((
            position.x0 * width / final_width,
            axes_bottom / final_height,
            position.width * width / final_width,
            (axes_height + extra_height) / final_height,
        ))
        for legend, anchor_y in zip(legends, bottom_anchors):
            legend.set_bbox_to_anchor(
                (plot_width / (2.0 * final_width), anchor_y / final_height),
                transform=figure.transFigure,
            )
        top = axes_bottom + (axes_height + extra_height + legend_height) / 2.0
        right_legend.set_bbox_to_anchor(
            (width / final_width, top / final_height), transform=figure.transFigure
        )
        right_legend.set_in_layout(True)
    position = ax.get_position()
    panel_center = plot_width / (2.0 * figure.get_figwidth())
    ax.title.set_x((panel_center - position.x0) / position.width)


def _dataset_legend_handles(
    datasets: Sequence[str],
    marker_size: float,
    use_bubble_shape: bool = False,
) -> List[Line2D]:
    """Create deterministic legend handles for dataset encodings."""

    handles: List[Line2D] = []
    for dataset in datasets:
        style = DATASET_STYLES[dataset]
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o" if use_bubble_shape else style["marker"],
                markersize=marker_size,
                markerfacecolor=style["color"],
                markeredgecolor="#202020",
                markeredgewidth=0.8,
                label=dataset,
            )
        )
    return handles


def _model_legend_handles(
    model_names: Sequence[str],
    model_styles: Mapping[str, Mapping[str, str]],
    marker_size: float,
) -> List[Line2D]:
    """Create model handles in their first-occurrence order."""

    handles: List[Line2D] = []
    for model_name in dict.fromkeys(str(name) for name in model_names):
        model_style = model_styles[model_name]
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=model_style["marker"],
                markersize=marker_size,
                markerfacecolor=model_style["color"],
                markeredgecolor="#202020",
                markeredgewidth=0.5,
                label=display_model_name(model_name),
            )
        )
    return handles


def _create_model_legend(
    ax: plt.Axes,
    figure: Figure,
    handles: Sequence[Line2D],
    font_size: float,
    anchor_y: float,
) -> Tuple[Legend, int]:
    """Fit a model legend to the figure width, adding rows as necessary."""

    if not handles:
        raise ValueError("At least one model handle is required for the model legend.")
    if not np.isfinite(font_size) or font_size <= 0:
        raise ValueError("Model-legend font size must be finite and greater than zero.")

    def create(
        legend_handles: Sequence[Line2D],
        column_count: int,
    ) -> Legend:
        legend = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, anchor_y),
            bbox_transform=figure.transFigure,
            ncol=column_count,
            fontsize=font_size,
            frameon=False,
            borderaxespad=0.0,
            borderpad=0.1,
            labelspacing=0.6,
            handlelength=1.0,
            handletextpad=0.5,
            columnspacing=1.0,
        )
        return legend

    maximum_width = figure.bbox.width * 0.96
    for column_count in range(len(handles), 0, -1):
        # Matplotlib fills columns top-to-bottom. Reorder only the handles so
        # readers encounter models left-to-right in their CSV order.
        ordered_handles = [
            handles[index]
            for column_index in range(column_count)
            for index in range(column_index, len(handles), column_count)
        ]
        legend = create(ordered_handles, column_count)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        if legend.get_window_extent(renderer=renderer).width <= maximum_width:
            return legend, (len(handles) + column_count - 1) // column_count
        legend.remove()
    raise ValueError(
        "--model_marker_legend font size {:g} pt leaves a model entry wider "
        "than the selected figure; choose a smaller value.".format(
            font_size
        )
    )


def _memory_legend_handles(
    memory: pd.Series,
    scale_factor: Optional[float] = None,
) -> List[Line2D]:
    """Create distinct min/median/max references at the plotted physical sizes."""

    # Validate all source values before choosing representatives, and derive the
    # shared factor from the full source rather than independently scaling keys.
    source_areas = scale_memory_to_marker_area(memory, scale_factor=scale_factor)
    values = memory.to_numpy(dtype=float)
    if scale_factor is None:
        scale_factor = float(source_areas[np.argmax(values)]) / float(values.max())
    representatives = np.unique(
        [float(np.min(values)), float(np.median(values)), float(np.max(values))],
    )
    areas = scale_memory_to_marker_area(
        pd.Series(representatives), scale_factor=scale_factor
    )
    # Increase decimal precision only when rounding would conflate references.
    for decimals in range(13):
        labels = ["{:,.{}f} MiB".format(value, decimals) for value in representatives]
        if len(set(labels)) == len(labels) and all(
            round(float(value), decimals) > 0 for value in representatives
        ):
            break
    else:
        labels = ["{:.17g} MiB".format(value) for value in representatives]
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=float(np.sqrt(area)),
            markerfacecolor="#B8B8B8",
            markeredgecolor="#303030",
            markeredgewidth=0.75,
            alpha=0.8,
            label=label,
        )
        for label, area in zip(labels, areas)
    ]


def _annotate_models(
    ax: plt.Axes,
    data: pd.DataFrame,
    datasets: Sequence[str],
    annotation_size: float,
    marker_areas: np.ndarray,
    obstacles: Sequence[object],
    paper_style: bool,
    highlight_ours: bool = False,
    display_all: bool = True,
) -> None:
    """Annotate each model once, avoiding earlier labels and plot legends."""

    x_values = data[THROUGHPUT_COLUMN].to_numpy(dtype=float)
    y_matrix = np.column_stack(
        [data[DATASET_COLUMNS[dataset]].to_numpy(dtype=float) for dataset in datasets]
    )
    y_values = np.max(y_matrix, axis=1)
    x_midpoint = float(np.median(x_values))
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer=renderer)
    occupied_boxes = [
        artist.get_window_extent(renderer=renderer).expanded(1.03, 1.08)
        for artist in obstacles
    ]
    label_boxes = []
    connector_paths = []
    pixels_per_point = figure.dpi / 72.0
    for dataset in datasets:
        dataset_y = data[DATASET_COLUMNS[dataset]].to_numpy(dtype=float)
        for row_index, (x_value, y_value) in enumerate(zip(x_values, dataset_y)):
            center_x, center_y = ax.transData.transform((x_value, y_value))
            marker_radius = max(
                3.0,
                float(np.sqrt(marker_areas[row_index]) * 0.55 * pixels_per_point),
            )
            occupied_boxes.append(
                Bbox.from_extents(
                    center_x - marker_radius,
                    center_y - marker_radius,
                    center_x + marker_radius,
                    center_y + marker_radius,
                )
            )

    def overlap_area(first: object, second: object) -> float:
        overlap_width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
        overlap_height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
        return overlap_width * overlap_height

    def add_annotation(
        row_index: int,
        horizontal_offset: float,
        vertical_offset: float,
        connector: bool = False,
    ) -> object:
        model_name = str(data.iloc[row_index]["Model"])
        highlighted = highlight_ours and model_name == OURS_MODEL
        displayed_name = OURS_HIGHLIGHT_LABEL if highlighted else display_model_name(model_name)
        if paper_style and len(datasets) == 1 and len(displayed_name) > 18 and not highlighted:
            if " (" in displayed_name and len(displayed_name.split(" (", 1)[0]) <= 14:
                displayed_name = displayed_name.replace(" (", "\n(", 1)
            else:
                displayed_name = textwrap.fill(
                    displayed_name,
                    width=18,
                    break_long_words=False,
                    break_on_hyphens=True,
                )
        alignment = "left" if horizontal_offset > 0 else "right"
        return ax.annotate(
            displayed_name,
            xy=(float(x_values[row_index]), float(y_values[row_index])),
            xytext=(horizontal_offset, vertical_offset),
            textcoords="offset points",
            ha=alignment,
            va="center",
            multialignment=alignment,
            fontsize=annotation_size,
            fontweight="semibold" if model_name == OURS_MODEL else "normal",
            color=OURS_OUTLINE_COLOR if highlighted else "#111111",
            annotation_clip=False,
            arrowprops=(
                {
                    "arrowstyle": "-",
                    "color": "#888888",
                    "linewidth": 0.35 if paper_style else 0.45,
                    "shrinkA": 1.0,
                    "shrinkB": 4.0,
                }
                if connector
                else None
            ),
            zorder=6,
        )

    # Place long labels first because they have the fewest collision-free options.
    sorted_indices = np.asarray(
        sorted(
            [
                index for index in range(len(data))
                if display_all or (highlight_ours and str(data.iloc[index]["Model"]) == OURS_MODEL)
            ],
            key=lambda index: (
                0 if highlight_ours and str(data.iloc[index]["Model"]) == OURS_MODEL else 1,
                -len(display_model_name(str(data.iloc[index]["Model"]))),
                x_values[index],
            ),
        ),
        dtype=int,
    )
    for row_index in sorted_indices:
        x_value = float(x_values[row_index])
        preferred_side = 1.0 if x_value <= x_midpoint else -1.0
        marker_clearance = float(np.sqrt(marker_areas[row_index]) / 2.0 + 4.0)
        vertical_offsets = (
            0.0,
            3.0,
            -3.0,
            6.0,
            -6.0,
            10.0,
            -10.0,
            14.0,
            -14.0,
            20.0,
            -20.0,
            28.0,
            -28.0,
            36.0,
            -36.0,
            44.0,
            -44.0,
        )
        candidates = [
            (side * (marker_clearance + extra_clearance), vertical_offset)
            for vertical_offset in vertical_offsets
            for extra_clearance in (0.0, 8.0, 16.0, 24.0, 32.0)
            for side in (preferred_side, -preferred_side)
        ]

        best_candidate = candidates[0]
        best_box = None
        best_key = None
        best_connector_path = None
        for horizontal_offset, vertical_offset in candidates:
            annotation = add_annotation(row_index, horizontal_offset, vertical_offset)
            candidate_box = annotation.get_window_extent(renderer=renderer).expanded(
                1.02, 1.20
            )
            overlap_score = sum(
                overlap_area(candidate_box, occupied) for occupied in occupied_boxes
            )
            # A connector must not run through another model's label, and later
            # labels must not be placed over connectors already selected.
            needs_connector = (
                abs(vertical_offset) > 11.0
                or abs(horizontal_offset) > marker_clearance + 4.0
            )
            connector_path = None
            connector_collisions = sum(
                path.intersects_bbox(candidate_box, filled=False) for path in connector_paths
            )
            if needs_connector:
                connector_path = PlotPath([
                    ((candidate_box.x0 + candidate_box.x1) / 2.0,
                     (candidate_box.y0 + candidate_box.y1) / 2.0),
                    ax.transData.transform((x_value, float(y_values[row_index]))),
                ])
                connector_collisions += sum(
                    connector_path.intersects_bbox(box, filled=False) for box in label_boxes
                )
            overlap_score += connector_collisions * axes_box.width * axes_box.height
            outside_distance = max(axes_box.x0 - candidate_box.x0, 0.0)
            outside_distance += max(candidate_box.x1 - axes_box.x1, 0.0)
            outside_distance += max(axes_box.y0 - candidate_box.y0, 0.0)
            outside_distance += max(candidate_box.y1 - axes_box.y1, 0.0)
            extra_clearance = max(abs(horizontal_offset) - marker_clearance, 0.0)
            opposite_side_penalty = 1.0 if horizontal_offset * preferred_side < 0 else 0.0
            placement_distance = (
                abs(vertical_offset)
                + 0.25 * extra_clearance
                + opposite_side_penalty
            )
            feasible = outside_distance == 0.0 and overlap_score == 0.0
            candidate_key = (
                0 if feasible else 1,
                outside_distance * 1000.0 + overlap_score,
                placement_distance,
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_candidate = (horizontal_offset, vertical_offset)
                best_box = candidate_box
                best_connector_path = connector_path
            annotation.remove()

        needs_connector = abs(best_candidate[1]) > 11.0
        needs_connector = (
            needs_connector
            or abs(best_candidate[0]) > marker_clearance + 4.0
        )
        add_annotation(row_index, *best_candidate, connector=needs_connector)
        occupied_boxes.append(best_box)
        label_boxes.append(best_box)
        if best_connector_path is not None:
            connector_paths.append(best_connector_path)


def plot_dice_vs_throughput(
    data: pd.DataFrame,
    datasets: Sequence[str],
    title: str = DEFAULT_TITLE,
    xlabel: str = DEFAULT_XLABEL,
    ylabel: str = DEFAULT_YLABEL,
    memory_scaling: bool = False,
    display_model_names: bool = False,
    show_throughput_std: bool = False,
    show_bubble_legend: bool = False,
    center_x_marker: bool = False,
    paper_style: bool = False,
    show_grid: bool = False,
    model_name_font_size: Optional[float] = None,
    model_center_markers: bool = False,
    model_center_colors: bool = False,
    show_model_marker_legend: bool = False,
    axis_label_font_size: Optional[float] = None,
    model_marker_legend_font_size: Optional[float] = None,
    title_font_size: Optional[float] = None,
    center_icon_size: Optional[float] = None,
    highlight_ours: bool = False,
    remove_title: bool = False,
    right_memory_legend: bool = False,
    scaled_memorey_legened: bool = False,
    add_memory_legende: bool = False,
    radius: float = 1.0,
    runtime_memory: float = 1024.0,
) -> Figure:
    """Create the requested figure without mutating the source dataframe."""

    memory_scaling = memory_scaling or add_memory_legende
    reference_scale_factor = (
        _reference_memory_scale(radius, runtime_memory) if add_memory_legende else None
    )
    if not datasets:
        raise ValueError("At least one dataset must be selected.")
    selected = parse_selected_datasets(datasets[0], datasets)
    if model_marker_legend_font_size is not None:
        if (
            not np.isfinite(model_marker_legend_font_size)
            or model_marker_legend_font_size <= 0
        ):
            raise ValueError(
                "--model_marker_legend must be a finite font size greater than zero."
            )
        show_model_marker_legend = True
    if center_icon_size is not None:
        if not np.isfinite(center_icon_size) or center_icon_size <= 0:
            raise ValueError(
                "--center_icon_size must be a finite value greater than zero."
            )
        if not memory_scaling:
            raise ValueError("--center_icon_size requires --memorey.")
    if show_bubble_legend and not memory_scaling:
        raise ValueError("--buble-size_legened requires --memorey.")
    if center_x_marker and not memory_scaling:
        raise ValueError("--center_x_marker requires --memorey.")
    if model_center_markers and not memory_scaling:
        raise ValueError("--model_center_markers requires --memorey.")
    if center_x_marker and model_center_markers:
        raise ValueError(
            "--center_x_marker and --model_center_markers cannot be used together."
        )
    if center_x_marker and len(selected) > 1:
        raise ValueError(
            "--center_x_marker supports only one selected dataset because "
            "multi-dataset plots use distinct center-marker shapes."
        )
    if model_center_colors and not model_center_markers:
        raise ValueError("--model_center_colors requires --model_center_markers.")
    if show_model_marker_legend and not model_center_markers:
        raise ValueError("--model_marker_legend requires --model_center_markers.")

    style = apply_paper_style(
        paper_style,
        len(selected),
        bottom_model_legend=show_model_marker_legend,
    )
    annotation_size = style.annotation_size
    if model_name_font_size is not None:
        if not np.isfinite(model_name_font_size) or model_name_font_size <= 0:
            raise ValueError("--model_name_font_size must be a finite value greater than zero.")
        annotation_size = float(model_name_font_size)
    title_size = style.title_size
    if title_font_size is not None:
        if not np.isfinite(title_font_size) or title_font_size <= 0:
            raise ValueError("--title_font_size must be a finite value greater than zero.")
        title_size = float(title_font_size)
    axis_label_size = style.label_size
    axis_tick_size = style.tick_size
    if axis_label_font_size is not None:
        if not np.isfinite(axis_label_font_size) or axis_label_font_size <= 0:
            raise ValueError(
                "--axis_label_font_size must be a finite value greater than zero."
            )
        axis_label_size = float(axis_label_font_size)
        axis_tick_size = float(axis_label_font_size)
    x_values = data[THROUGHPUT_COLUMN].to_numpy(dtype=float)
    standard_deviations = data[THROUGHPUT_STD_COLUMN].to_numpy(dtype=float)
    model_names = data["Model"].astype(str).tolist()
    memory_areas = (
        scale_memory_to_marker_area(data[MEMORY_COLUMN], scale_factor=reference_scale_factor)
        if memory_scaling
        else np.full(len(data), style.fixed_marker_area, dtype=float)
    )
    # Reuse the scaler's factor after it has validated all memory inputs.
    memory_scale_factor = reference_scale_factor
    if memory_scaling and memory_scale_factor is None:
        memory_scale_factor = (
            MAX_MEMORY_MARKER_AREA / float(pd.to_numeric(data[MEMORY_COLUMN]).max())
        )
    ours = data["Model"].eq(OURS_MODEL).to_numpy()
    edge_colors = np.where(ours, OURS_OUTLINE_COLOR, "#383838")
    edge_widths = np.where(ours, OURS_OUTLINE_WIDTH, 0.75)
    if highlight_ours:
        edge_widths = np.where(ours, OURS_HIGHLIGHT_WIDTH, edge_widths)
    model_center_area = (
        float(center_icon_size)
        if center_icon_size is not None
        else style.center_marker_area * MODEL_CENTER_AREA_SCALE
    )
    dataset_center_area = (
        float(center_icon_size)
        if center_icon_size is not None
        else style.center_marker_area
    )
    x_center_area = (
        float(center_icon_size)
        if center_icon_size is not None
        else style.center_marker_area * 0.6
    )
    dot_center_area = (
        float(center_icon_size)
        if center_icon_size is not None
        else style.center_marker_area * 0.35
    )
    model_styles: Dict[str, Dict[str, str]] = {}
    if model_center_markers:
        model_styles = build_model_marker_styles(
            model_names,
            use_colors=model_center_colors,
        )

    figure, ax = plt.subplots(figsize=style.figure_size)

    for dataset in selected:
        dataset_style = DATASET_STYLES[dataset]
        y_values = data[DATASET_COLUMNS[dataset]].to_numpy(dtype=float)
        face_colors = np.where(
            ours & highlight_ours & (len(selected) == 1),
            OURS_HIGHLIGHT_FILL,
            dataset_style["color"],
        )

        if show_throughput_std:
            _draw_error_bars(ax, x_values, y_values, standard_deviations)

        if memory_scaling:
            ax.scatter(
                x_values,
                y_values,
                s=memory_areas,
                marker="o",
                facecolor=face_colors,
                edgecolors=edge_colors,
                linewidths=edge_widths,
                alpha=0.38 if len(selected) > 1 else 0.72,
                zorder=3,
            )
            if model_center_markers:
                for row_index, model_name in enumerate(model_names):
                    model_style = model_styles[model_name]
                    ax.scatter(
                        [x_values[row_index]],
                        [y_values[row_index]],
                        s=model_center_area,
                        marker=model_style["marker"],
                        facecolor=model_style["color"],
                        edgecolor="#202020",
                        linewidth=0.45,
                        alpha=1.0,
                        zorder=4,
                    )
            elif len(selected) > 1:
                ax.scatter(
                    x_values,
                    y_values,
                    s=dataset_center_area,
                    marker=dataset_style["marker"],
                    facecolor=dataset_style["color"],
                    edgecolor="#111111",
                    linewidth=0.55,
                    alpha=1.0,
                    zorder=4,
                )
            else:
                # Mark the exact coordinate independently of the bubble area.
                if center_x_marker:
                    ax.scatter(
                        x_values,
                        y_values,
                        s=x_center_area,
                        marker="x",
                        color="#333333",
                        linewidths=0.55 if paper_style else 0.7,
                        alpha=0.95,
                        zorder=4,
                    )
                else:
                    ax.scatter(
                        x_values,
                        y_values,
                        s=dot_center_area,
                        marker="o",
                        facecolor="#252525",
                        edgecolors="none",
                        linewidths=0.0,
                        alpha=0.9,
                        zorder=4,
                    )
        else:
            ax.scatter(
                x_values,
                y_values,
                s=np.full(len(data), style.fixed_marker_area, dtype=float),
                marker=dataset_style["marker"],
                facecolor=face_colors,
                edgecolors=edge_colors,
                linewidths=edge_widths,
                alpha=0.9,
                zorder=3,
            )

    dataset_legend = None
    if len(selected) > 1:
        dataset_legend = figure.legend(
            handles=_dataset_legend_handles(
                selected,
                marker_size=6.5,
                use_bubble_shape=model_center_markers,
            ),
            title="Dataset",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(selected),
            frameon=False,
            borderaxespad=0.0,
            borderpad=0.45,
            labelspacing=0.6,
            handletextpad=0.7,
        )
        dataset_legend.get_title().set_fontsize(style.legend_size)

    model_legend = None
    if show_model_marker_legend:
        model_legend_size = (
            float(model_marker_legend_font_size)
            if model_marker_legend_font_size is not None
            else DEFAULT_MODEL_LEGEND_FONT_SIZE
        )
        model_legend, _ = _create_model_legend(
            ax,
            figure,
            _model_legend_handles(
                model_names,
                model_styles,
                marker_size=max(3.8, float(np.sqrt(model_center_area))),
            ),
            model_legend_size,
            0.0,
        )

    memory_legend = None
    if show_bubble_legend:
        memory_legend = _create_memory_legend(
            figure,
            _memory_legend_handles(
                data[MEMORY_COLUMN],
                scale_factor=memory_scale_factor,
            ),
            style.legend_size,
            right_side=right_memory_legend,
            scale_factor=memory_scale_factor if scaled_memorey_legened else None,
        )

    all_y_values = np.concatenate(
        [data[DATASET_COLUMNS[dataset]].to_numpy(dtype=float) for dataset in selected]
    )
    x_errors = standard_deviations if show_throughput_std else None
    ax.set_xlim(_padded_limits(x_values, x_errors, padding_fraction=0.09))
    ax.set_ylim(_padded_limits(all_y_values, padding_fraction=0.09))

    title_text = "" if remove_title else "\n".join(
        " ".join(line.split())
        for line in str(title).replace(r"\n", "\n").splitlines()
    ).strip()
    ax.set_title(
        title_text,
        pad=7.0,
        fontsize=title_size,
        loc="center",
        horizontalalignment="center",
        multialignment="center",
    )
    ax.title.set_visible(not remove_title)
    ax.set_xlabel(xlabel, fontsize=axis_label_size)
    ax.set_ylabel(ylabel, fontsize=axis_label_size)
    ax.tick_params(
        axis="both",
        which="major",
        width=style.axis_line_width,
        labelsize=axis_tick_size,
    )
    ax.minorticks_off()
    ax.set_axisbelow(True)
    if show_grid or paper_style:
        ax.grid(
            True,
            which="major",
            color="#D6D6D6",
            linewidth=0.55,
            alpha=0.65,
        )
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_line_width)
        spine.set_color("#303030")

    if title_text:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        title_width = ax.title.get_window_extent(renderer=renderer).width
        maximum_title_width = figure.bbox.width * 0.96
        if title_width > maximum_title_width and "\n" not in title_text:
            # Balance two lines using rendered widths, preserving whole words.
            # Explicit line breaks take precedence over automatic wrapping.
            words = title_text.split()
            best_title = title_text
            best_width = title_width
            for split_index in range(1, len(words)):
                candidate = "{}\n{}".format(
                    " ".join(words[:split_index]),
                    " ".join(words[split_index:]),
                )
                ax.title.set_text(candidate)
                candidate_width = ax.title.get_window_extent(renderer=renderer).width
                if candidate_width < best_width:
                    best_title, best_width = candidate, candidate_width
            ax.title.set_text(best_title)
            title_width = best_width
        if paper_style and title_width > maximum_title_width:
            fitted_title_size = title_size * maximum_title_width / title_width
            if title_font_size is not None:
                raise ValueError(
                    "--title_font_size {:g} pt is too large to fit the title "
                    "lines within the selected paper width; choose {:.1f} pt "
                    "or smaller, or shorten the title.".format(
                        title_font_size, fitted_title_size
                    )
                )
            ax.title.set_fontsize(fitted_title_size)

    legends = [
        legend for legend in (dataset_legend, memory_legend, model_legend)
        if legend is not None
    ]
    _layout_plot_legends(
        figure, ax, legends, paper_style,
        right_legend=memory_legend if right_memory_legend else None,
    )
    _pad_for_marker_extents(ax)
    if add_memory_legende:
        legends.append(_create_reference_memory_legend(
            ax, radius, runtime_memory, memory_scale_factor, style.legend_size, legends
        ))

    if display_model_names or highlight_ours:
        _annotate_models(
            ax,
            data,
            selected,
            annotation_size,
            memory_areas,
            legends,
            paper_style,
            highlight_ours=highlight_ours,
            display_all=display_model_names,
        )
    return figure


def sanitize_filename_component(value: str, fallback: str = "dataset") -> str:
    """Convert text to a safe, deterministic filename component."""

    sanitized = re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
    return sanitized or fallback


def normalize_output_name(output_name: str) -> str:
    """Validate and sanitize a custom PDF/PNG filename stem."""

    candidate = output_name.strip()
    if candidate.lower().endswith((".pdf", ".png")):
        candidate = candidate[:-4]
    sanitized = sanitize_filename_component(candidate, fallback="")
    if not sanitized:
        raise ValueError(
            "--output_name must contain at least one letter, number, underscore, or hyphen."
        )
    return sanitized


def create_output_filenames(
    datasets: Sequence[str],
    output_directory: Path,
    output_name: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Create PDF and PNG paths using a custom or dataset-derived stem."""

    if not datasets:
        raise ValueError("At least one dataset must be selected.")
    selected = parse_selected_datasets(datasets[0], datasets)
    if output_name is None:
        suffix = "_".join(sanitize_filename_component(name) for name in selected)
        stem = "dice_vs_throughput_{}".format(suffix)
    else:
        stem = normalize_output_name(output_name)
    directory = Path(output_directory).resolve()
    return directory / "{}.pdf".format(stem), directory / "{}.png".format(stem)


def save_figure(
    figure: Figure,
    pdf_path: Path,
    png_path: Path,
    paper_style: bool,
) -> None:
    """Save vector and raster outputs with publication-safe bounds."""

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    common_options = {
        "bbox_inches": "tight",
        # Paper canvases already include safe internal margins; extra padding
        # would turn an IEEE 3.5-inch column into a 3.54-inch exported file.
        "pad_inches": 0.0 if paper_style else 0.02,
        # Preserve the requested physical figure width when tight bounding is used.
        "bbox_extra_artists": (figure.patch,),
    }
    figure.savefig(pdf_path, format="pdf", **common_options)
    figure.savefig(
        png_path,
        format="png",
        dpi=1800 if paper_style else 300,
        **common_options,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, build the plot, and save both requested formats."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    figure: Optional[Figure] = None

    try:
        selected = parse_selected_datasets(args.dataset, args.multiple_dataset)
        memory_scaling = args.memorey or args.add_memory_legende
        if args.buble_size_legened and not memory_scaling:
            raise ValueError("--buble-size_legened requires --memorey.")

        csv_path = resolve_csv_path()
        data = load_and_validate_csv(csv_path)
        pdf_path, png_path = create_output_filenames(
            selected,
            args.output_dir,
            args.output_name,
        )

        with matplotlib.rc_context():
            figure = plot_dice_vs_throughput(
                data=data,
                datasets=selected,
                title=args.title,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                memory_scaling=memory_scaling,
                display_model_names=args.dislay_model_names,
                model_name_font_size=args.model_name_font_size,
                show_throughput_std=args.throughput_std,
                show_bubble_legend=args.buble_size_legened,
                center_x_marker=args.center_x_marker,
                paper_style=args.paper_style,
                show_grid=args.grid,
                model_center_markers=args.model_center_markers,
                model_center_colors=args.model_center_colors,
                show_model_marker_legend=args.model_marker_legend is not None,
                axis_label_font_size=args.axis_label_font_size,
                model_marker_legend_font_size=args.model_marker_legend,
                title_font_size=args.title_font_size,
                center_icon_size=args.center_icon_size,
                highlight_ours=args.highlight_ours,
                remove_title=args.remove_title,
                right_memory_legend=args.right_memory_legend,
                scaled_memorey_legened=args.scaled_memorey_legened,
                add_memory_legende=args.add_memory_legende,
                radius=args.radius,
                runtime_memory=args.runtime_memory,
            )
            save_figure(figure, pdf_path, png_path, args.paper_style)

        print("Selected dataset(s): {}".format(", ".join(selected)))
        print("Memory scaling enabled: {}".format(memory_scaling))
        print("Paper style enabled: {}".format(args.paper_style))
        print("Generated files:")
        print("  {}".format(pdf_path.resolve()))
        print("  {}".format(png_path.resolve()))
        return 0
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    finally:
        if figure is not None:
            plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
