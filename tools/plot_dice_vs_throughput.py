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
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
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
DEFAULT_XLABEL = "Throughput (img/sec)"
DEFAULT_YLABEL = "Dice(%)"

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
        help="encode model memory as bounded bubble area (spelling is intentional)",
    )
    parser.add_argument(
        "--buble-size_legened",
        action="store_true",
        help=(
            "show a min/median/max memory-size legend; requires --memorey "
            "(spelling is intentional)"
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
        const=5.0,
        default=None,
        metavar="POINTS",
        help=(
            "show a one- or two-row model-marker legend below the plot using "
            "an optional font size in points; a bare flag uses 5 pt; requires "
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
    min_area: float = 70.0,
    max_area: float = 900.0,
) -> np.ndarray:
    """Log-scale positive memory values to finite marker areas in points squared."""

    if not np.isfinite(min_area) or not np.isfinite(max_area):
        raise ValueError("Marker-area bounds must be finite.")
    if min_area <= 0 or max_area <= min_area:
        raise ValueError("Marker-area bounds must satisfy 0 < min_area < max_area.")

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

    minimum = float(values.min())
    maximum = float(values.max())
    if minimum == maximum:
        return np.full(values.shape, (min_area + max_area) / 2.0, dtype=float)

    logged = np.log(values)
    logged_span = float(np.log(maximum) - np.log(minimum))
    if not np.isfinite(logged_span) or logged_span <= 0.0:
        unique_values, inverse = np.unique(values, return_inverse=True)
        normalized = inverse.astype(float) / float(len(unique_values) - 1)
    else:
        normalized = (logged - np.log(minimum)) / logged_span
    areas = min_area + normalized * (max_area - min_area)
    sorted_indices = np.argsort(values, kind="stable")
    sorted_values = values[sorted_indices]
    sorted_areas = areas[sorted_indices]
    distinct_values = np.diff(sorted_values) > 0
    if bool((np.diff(sorted_areas)[distinct_values] <= 0).any()):
        unique_values, inverse = np.unique(values, return_inverse=True)
        areas = min_area + inverse.astype(float) / float(len(unique_values) - 1) * (
            max_area - min_area
        )
    if not bool(np.isfinite(areas).all()):
        raise ValueError("Memory scaling produced a non-finite marker area.")
    return np.clip(areas, min_area, max_area)


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
            # Retain one-column width and add vertical room for the bottom key.
            figure_size = (3.5, 3.6 if bottom_model_legend else 2.8)
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
    """Create the widest model legend that fits in at most two rows."""

    if not handles:
        raise ValueError("At least one model handle is required for the model legend.")

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
            labelspacing=0.25,
            handlelength=0.6,
            handletextpad=0.2,
            columnspacing=0.3,
        )
        return legend

    maximum_width = figure.bbox.width * 0.96
    legend = create(handles, len(handles))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    if legend.get_window_extent(renderer=renderer).width <= maximum_width:
        return legend, 1

    legend.remove()
    two_row_columns = max(1, (len(handles) + 1) // 2)
    # Matplotlib fills columns top-to-bottom. Reorder only the supplied handles
    # so readers still encounter models left-to-right in their CSV order.
    two_row_handles = [
        handles[index]
        for column_index in range(two_row_columns)
        for index in (column_index, column_index + two_row_columns)
        if index < len(handles)
    ]
    legend = create(two_row_handles, two_row_columns)
    figure.canvas.draw()
    if legend.get_window_extent(renderer=renderer).width > maximum_width:
        legend.remove()
        raise ValueError(
            "--model_marker_legend font size {:g} pt is too large for a "
            "two-row legend at the selected figure width; choose a smaller "
            "value (5 pt is recommended for a JBHI one-column figure).".format(
                font_size
            )
        )
    return legend, 2


def _memory_legend_handles(
    memory: pd.Series,
    area_scale: float = 1.0,
) -> List[Line2D]:
    """Create min/median/max memory handles using the plot's area scaler."""

    if not np.isfinite(area_scale) or area_scale <= 0:
        raise ValueError("Memory-legend area scale must be finite and greater than zero.")

    values = memory.to_numpy(dtype=float)
    representatives = np.asarray(
        [float(np.min(values)), float(np.median(values)), float(np.max(values))],
        dtype=float,
    )
    representatives = np.asarray(list(dict.fromkeys(representatives.tolist())), dtype=float)
    areas = scale_memory_to_marker_area(pd.Series(representatives))
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=float(np.sqrt(area * area_scale)),
            markerfacecolor="#B8B8B8",
            markeredgecolor="#303030",
            markeredgewidth=0.7,
            alpha=0.8,
            label="{:,.0f}".format(value),
        )
        for value, area in zip(representatives, areas)
    ]


def _annotate_models(
    ax: plt.Axes,
    data: pd.DataFrame,
    datasets: Sequence[str],
    annotation_size: float,
    marker_areas: np.ndarray,
    obstacles: Sequence[object],
    paper_style: bool,
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
        displayed_name = display_model_name(model_name)
        if paper_style and len(datasets) == 1 and len(displayed_name) > 18:
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
            color="#111111",
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
            range(len(data)),
            key=lambda index: (
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
            4.0,
            -5.0,
            10.0,
            -11.0,
            17.0,
            -18.0,
            25.0,
            -26.0,
            34.0,
            -35.0,
            44.0,
            -45.0,
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
        for horizontal_offset, vertical_offset in candidates:
            annotation = add_annotation(row_index, horizontal_offset, vertical_offset)
            candidate_box = annotation.get_window_extent(renderer=renderer).expanded(
                1.02, 1.20
            )
            overlap_score = sum(
                overlap_area(candidate_box, occupied) for occupied in occupied_boxes
            )
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
            annotation.remove()

        needs_connector = abs(best_candidate[1]) > 11.0
        needs_connector = (
            needs_connector
            or abs(best_candidate[0]) > marker_clearance + 4.0
        )
        add_annotation(row_index, *best_candidate, connector=needs_connector)
        occupied_boxes.append(best_box)


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
) -> Figure:
    """Create the requested figure without mutating the source dataframe."""

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
        scale_memory_to_marker_area(data[MEMORY_COLUMN])
        if memory_scaling
        else np.full(len(data), style.fixed_marker_area, dtype=float)
    )
    ours = data["Model"].eq(OURS_MODEL).to_numpy()
    edge_colors = np.where(ours, OURS_OUTLINE_COLOR, "#383838")
    edge_widths = np.where(ours, OURS_OUTLINE_WIDTH, 0.75)
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

        if show_throughput_std:
            _draw_error_bars(ax, x_values, y_values, standard_deviations)

        if memory_scaling:
            ax.scatter(
                x_values,
                y_values,
                s=memory_areas,
                marker="o",
                facecolor=dataset_style["color"],
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
                facecolor=dataset_style["color"],
                edgecolors=edge_colors,
                linewidths=edge_widths,
                alpha=0.9,
                zorder=3,
            )

    dataset_legend = None
    if len(selected) > 1:
        dataset_legend = ax.legend(
            handles=_dataset_legend_handles(
                selected,
                marker_size=6.5,
                use_bubble_shape=model_center_markers,
            ),
            title="Dataset",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=not show_model_marker_legend,
            framealpha=0.94,
            borderpad=0.45,
            labelspacing=0.35,
            handletextpad=0.45,
        )
        dataset_legend.get_title().set_fontsize(style.legend_size)

    model_legend = None
    model_legend_rows = 0
    if show_model_marker_legend:
        model_legend_size = (
            float(model_marker_legend_font_size)
            if model_marker_legend_font_size is not None
            else 5.0
        )
        model_legend_anchor_y = (
            0.14 if paper_style and len(selected) == 1 else 0.10
        )
        if dataset_legend is not None:
            ax.add_artist(dataset_legend)
        model_legend, model_legend_rows = _create_model_legend(
            ax,
            figure,
            _model_legend_handles(
                model_names,
                model_styles,
                marker_size=max(3.8, float(np.sqrt(model_center_area))),
            ),
            model_legend_size,
            model_legend_anchor_y,
        )

    memory_legend = None
    if show_bubble_legend:
        if dataset_legend is not None and model_legend is None:
            ax.add_artist(dataset_legend)
        if model_legend is not None:
            ax.add_artist(model_legend)
        memory_legend = ax.legend(
            handles=_memory_legend_handles(
                data[MEMORY_COLUMN],
                area_scale=0.5 if model_legend is not None else 1.0,
            ),
            title="Memory (MiB)",
            loc=(
                "lower left"
                if dataset_legend is not None or model_legend is not None
                else "center left"
            ),
            bbox_to_anchor=(
                1.01,
                0.0
                if dataset_legend is not None or model_legend is not None
                else 0.5,
            ),
            frameon=model_legend is None,
            framealpha=0.94,
            borderpad=0.35 if model_legend is not None else 1.25,
            labelspacing=0.55 if model_legend is not None else 1.25,
            handletextpad=0.6 if model_legend is not None else 0.7,
            handleheight=1.4 if model_legend is not None else 2.1,
        )
        memory_legend.get_title().set_fontsize(style.legend_size)

    all_y_values = np.concatenate(
        [data[DATASET_COLUMNS[dataset]].to_numpy(dtype=float) for dataset in selected]
    )
    x_errors = standard_deviations if show_throughput_std else None
    ax.set_xlim(_padded_limits(x_values, x_errors, padding_fraction=0.09))
    ax.set_ylim(_padded_limits(all_y_values, padding_fraction=0.09))

    title_text = "\n".join(
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

    legends = []
    if dataset_legend is not None:
        legends.append(dataset_legend)
    if model_legend is not None:
        legends.append(model_legend)
    if memory_legend is not None:
        legends.append(memory_legend)
    current_legend = ax.get_legend()
    if current_legend is not None and current_legend not in legends:
        legends.append(current_legend)

    if legends:
        for legend in legends:
            legend.set_in_layout(False)
        figure.tight_layout(pad=0.5 if paper_style else 1.0)
        has_right_legend = dataset_legend is not None or memory_legend is not None
        if has_right_legend:
            reserved_right = 0.64 if paper_style and len(selected) == 1 else 0.79
            figure.subplots_adjust(right=reserved_right)
        if show_model_marker_legend:
            if paper_style and len(selected) == 1:
                reserved_bottom = 0.27
            elif paper_style:
                reserved_bottom = 0.23 if model_legend_rows == 1 else 0.28
            else:
                reserved_bottom = 0.22 if model_legend_rows == 1 else 0.27
            figure.subplots_adjust(bottom=reserved_bottom)
        for legend in legends:
            legend.set_in_layout(True)
    else:
        figure.tight_layout(pad=0.5 if paper_style else 1.0)

    axes_position = ax.get_position()
    figure_center_in_axes = (0.5 - axes_position.x0) / axes_position.width
    ax.title.set_x(figure_center_in_axes)

    if display_model_names:
        _annotate_models(
            ax,
            data,
            selected,
            annotation_size,
            memory_areas,
            legends,
            paper_style,
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
        if args.buble_size_legened and not args.memorey:
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
                memory_scaling=args.memorey,
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
            )
            save_figure(figure, pdf_path, png_path, args.paper_style)

        print("Selected dataset(s): {}".format(", ".join(selected)))
        print("Memory scaling enabled: {}".format(args.memorey))
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
