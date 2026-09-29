#!/usr/bin/env python3
"""Plot Dice score versus inference throughput for the models in ``sota.csv``.

Examples
--------
Default (Synapse)::

    python tools/plot_dice_vs_throughput.py

ACDC only::

    python tools/plot_dice_vs_throughput.py --dataset ACDC

Cataract-1k with memory-scaled bubbles::

    python tools/plot_dice_vs_throughput.py --dataset Cataract-1k --memorey

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
        --ylabel "Dice (%)"

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
from typing import List, Mapping, Optional, Sequence, Tuple

import matplotlib

# A non-interactive backend makes command-line execution reliable on servers.
matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
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
  %(prog)s --dataset Cataract-1k --memorey
  %(prog)s --multiple_dataset Synapse ACDC Cataract-1k
  %(prog)s --multiple_dataset Synapse ACDC Cataract-1k --memorey --paper_style
  %(prog)s --dataset Synapse --title "Accuracy–Efficiency Trade-off on Synapse"
"""
    parser = argparse.ArgumentParser(
        description=(
            "Create PDF and PNG Dice-versus-throughput plots from sota.csv. "
            "Outputs are written beside the resolved CSV file."
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
        "--throughput_std",
        action="store_true",
        help="show Throughput Std as horizontal error bars",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="plot title (default: %(default)s)",
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
        "--paper_style",
        action="store_true",
        help="use IEEE-sized typography and save the PNG at 1800 dpi",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="show a light major grid (enabled automatically by --paper_style)",
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


def apply_paper_style(paper_style: bool, dataset_count: int) -> PlotStyle:
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
        figure_size = (3.5, 2.8) if dataset_count == 1 else (7.16, 3.8)
        style = PlotStyle(
            figure_size=figure_size,
            title_size=9.5,
            label_size=9.0,
            tick_size=8.0,
            annotation_size=8.0,
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


def _dataset_legend_handles(datasets: Sequence[str], marker_size: float) -> List[Line2D]:
    """Create deterministic legend handles for dataset encodings."""

    handles: List[Line2D] = []
    for dataset in datasets:
        style = DATASET_STYLES[dataset]
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=style["marker"],
                markersize=marker_size,
                markerfacecolor=style["color"],
                markeredgecolor="#202020",
                markeredgewidth=0.8,
                label=dataset,
            )
        )
    return handles


def _memory_legend_handles(memory: pd.Series) -> List[Line2D]:
    """Create min/median/max memory handles using the plot's area scaler."""

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
            markersize=float(np.sqrt(area)),
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
        return ax.annotate(
            model_name,
            xy=(float(x_values[row_index]), float(y_values[row_index])),
            xytext=(horizontal_offset, vertical_offset),
            textcoords="offset points",
            ha="left" if horizontal_offset > 0 else "right",
            va="center",
            fontsize=annotation_size,
            fontweight="bold" if model_name == OURS_MODEL else "normal",
            color="#111111",
            annotation_clip=False,
            arrowprops=(
                {
                    "arrowstyle": "-",
                    "color": "#777777",
                    "linewidth": 0.45,
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
            key=lambda index: (-len(str(data.iloc[index]["Model"])), x_values[index]),
        ),
        dtype=int,
    )
    for rank, row_index in enumerate(sorted_indices):
        x_value = float(x_values[row_index])
        preferred_side = 1.0 if x_value <= x_midpoint else -1.0
        marker_clearance = float(np.sqrt(marker_areas[row_index]) / 2.0 + 4.0)
        vertical_offsets = (6.0, -9.0, 20.0, -23.0, 34.0, -37.0, 48.0, -51.0)
        # Rotate the search order so adjacent points do not all prefer one height.
        shift = rank % 4
        vertical_offsets = vertical_offsets[shift:] + vertical_offsets[:shift]
        candidates = [
            (side * (marker_clearance + extra_clearance), vertical_offset)
            for vertical_offset in vertical_offsets
            for extra_clearance in (0.0, 16.0, 32.0)
            for side in (preferred_side, -preferred_side)
        ]

        chosen = None
        chosen_box = None
        chosen_candidate = candidates[0]
        best_candidate = candidates[0]
        best_score = float("inf")
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
            score = overlap_score + outside_distance * 1000.0
            if score < best_score:
                best_score = score
                best_candidate = (horizontal_offset, vertical_offset)
            if score == 0.0:
                chosen = annotation
                chosen_box = candidate_box
                chosen_candidate = (horizontal_offset, vertical_offset)
                break
            annotation.remove()

        if chosen is None:
            chosen = add_annotation(row_index, *best_candidate)
            chosen_box = chosen.get_window_extent(renderer=renderer).expanded(1.02, 1.20)
            chosen_candidate = best_candidate
        needs_connector = abs(chosen_candidate[1]) > 12.0
        needs_connector = (
            needs_connector
            or abs(chosen_candidate[0]) > marker_clearance + 4.0
        )
        if needs_connector:
            chosen.remove()
            chosen = add_annotation(row_index, *chosen_candidate, connector=True)
        occupied_boxes.append(chosen_box)


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
    paper_style: bool = False,
    show_grid: bool = False,
) -> Figure:
    """Create the requested figure without mutating the source dataframe."""

    if not datasets:
        raise ValueError("At least one dataset must be selected.")
    selected = parse_selected_datasets(datasets[0], datasets)
    if show_bubble_legend and not memory_scaling:
        raise ValueError("--buble-size_legened requires --memorey.")

    style = apply_paper_style(paper_style, len(selected))
    x_values = data[THROUGHPUT_COLUMN].to_numpy(dtype=float)
    standard_deviations = data[THROUGHPUT_STD_COLUMN].to_numpy(dtype=float)
    memory_areas = (
        scale_memory_to_marker_area(data[MEMORY_COLUMN])
        if memory_scaling
        else np.full(len(data), style.fixed_marker_area, dtype=float)
    )
    ours = data["Model"].eq(OURS_MODEL).to_numpy()
    edge_colors = np.where(ours, "#111111", "#383838")
    edge_widths = np.where(ours, 2.0, 0.75)

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
            if len(selected) > 1:
                ax.scatter(
                    x_values,
                    y_values,
                    s=style.center_marker_area,
                    marker=dataset_style["marker"],
                    facecolor=dataset_style["color"],
                    edgecolor="#111111",
                    linewidth=0.55,
                    alpha=1.0,
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
            handles=_dataset_legend_handles(selected, marker_size=6.5),
            title="Dataset",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            framealpha=0.94,
            borderpad=0.45,
            labelspacing=0.35,
            handletextpad=0.45,
        )
        dataset_legend.get_title().set_fontsize(style.legend_size)

    if show_bubble_legend:
        if dataset_legend is not None:
            ax.add_artist(dataset_legend)
        memory_legend = ax.legend(
            handles=_memory_legend_handles(data[MEMORY_COLUMN]),
            title="Memory (MiB)",
            loc="lower left" if dataset_legend is not None else "center left",
            bbox_to_anchor=(1.01, 0.0 if dataset_legend is not None else 0.5),
            frameon=True,
            framealpha=0.94,
            borderpad=1.25,
            labelspacing=1.25,
            handletextpad=0.7,
            handleheight=2.1,
        )
        memory_legend.get_title().set_fontsize(style.legend_size)

    all_y_values = np.concatenate(
        [data[DATASET_COLUMNS[dataset]].to_numpy(dtype=float) for dataset in selected]
    )
    x_errors = standard_deviations if show_throughput_std else None
    ax.set_xlim(_padded_limits(x_values, x_errors, padding_fraction=0.09))
    ax.set_ylim(_padded_limits(all_y_values, padding_fraction=0.09))

    displayed_title = title
    if paper_style:
        displayed_title = textwrap.fill(
            title,
            width=42 if len(selected) == 1 else 88,
            break_long_words=False,
            break_on_hyphens=False,
        )
    ax.set_title(displayed_title, pad=7.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major", width=style.axis_line_width)
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

    legends = []
    if dataset_legend is not None:
        legends.append(dataset_legend)
    current_legend = ax.get_legend()
    if current_legend is not None and current_legend not in legends:
        legends.append(current_legend)

    if legends:
        for legend in legends:
            legend.set_in_layout(False)
        figure.tight_layout(pad=0.5 if paper_style else 1.0)
        reserved_right = 0.64 if paper_style and len(selected) == 1 else 0.79
        figure.subplots_adjust(right=reserved_right)
        for legend in legends:
            legend.set_in_layout(True)
    else:
        figure.tight_layout(pad=0.5 if paper_style else 1.0)

    if display_model_names:
        _annotate_models(
            ax,
            data,
            selected,
            style.annotation_size,
            memory_areas,
            legends,
        )
    return figure


def sanitize_filename_component(value: str) -> str:
    """Convert text to a safe, deterministic filename component."""

    sanitized = re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
    return sanitized or "dataset"


def create_output_filenames(
    datasets: Sequence[str],
    output_directory: Path,
) -> Tuple[Path, Path]:
    """Create PDF and PNG output paths from ordered dataset names."""

    if not datasets:
        raise ValueError("At least one dataset must be selected.")
    selected = parse_selected_datasets(datasets[0], datasets)
    suffix = "_".join(sanitize_filename_component(name) for name in selected)
    stem = "dice_vs_throughput_{}".format(suffix)
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
        "pad_inches": 0.02,
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
        pdf_path, png_path = create_output_filenames(selected, csv_path.parent)

        with matplotlib.rc_context():
            figure = plot_dice_vs_throughput(
                data=data,
                datasets=selected,
                title=args.title,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                memory_scaling=args.memorey,
                display_model_names=args.dislay_model_names,
                show_throughput_std=args.throughput_std,
                show_bubble_legend=args.buble_size_legened,
                paper_style=args.paper_style,
                show_grid=args.grid,
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
