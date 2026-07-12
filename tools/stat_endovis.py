#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_ROOT = "/data/halyusuf/data/EndoVis_2018"
SEQUENCE_RE = re.compile(r"^(seq\d+)_")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute EndoVis2018 class pixel distributions per sequence for "
            "the train and test splits."
        )
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="EndoVis dataset root containing train/ and test/ directories.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Dataset splits to summarize.",
    )
    parser.add_argument(
        "--label-json",
        default="labels.json",
        help=(
            "Label JSON filename or absolute path. Relative paths are resolved "
            "like EndoVis2018Dataset: dataset root first, then train/labels, "
            "then split/labels."
        ),
    )
    parser.add_argument(
        "--count-all-labels",
        action="store_true",
        help=(
            "Count every PNG in split/labels. By default, only labels with a "
            "matching image in split/imgs are counted, matching the dataset loader."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional debug limit per split.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save bar charts for the class distribution.",
    )
    parser.add_argument(
        "--plot-dir",
        default="endovis_distribution_plots",
        help="Output directory for --plot charts.",
    )
    parser.add_argument(
        "--ignore_background",
        "--ignore-background",
        action="store_true",
        help="Exclude class 0/background-tissue from percentages, tables, and plots.",
    )
    return parser.parse_args()


def resolve_label_json(root, split, label_json):
    if os.path.isabs(label_json):
        if os.path.exists(label_json):
            return label_json
        raise FileNotFoundError("Could not find {}".format(label_json))

    root_candidate = os.path.join(root, label_json)
    if os.path.exists(root_candidate):
        return root_candidate

    train_candidate = os.path.join(root, "train", "labels", label_json)
    if os.path.exists(train_candidate):
        return train_candidate

    split_candidate = os.path.join(root, split, "labels", label_json)
    if os.path.exists(split_candidate):
        return split_candidate

    raise FileNotFoundError("Could not find {}".format(label_json))


def load_label_spec(label_json_path):
    with open(label_json_path, "r") as f:
        labels = json.load(f)

    required_keys = {"name", "color", "classid"}
    for item in labels:
        if not required_keys.issubset(item):
            raise ValueError(
                "Each label entry must contain name, color, and classid"
            )
        item["classid"] = int(item["classid"])
        item["color"] = [int(value) for value in item["color"]]

    labels = sorted(labels, key=lambda item: item["classid"])
    num_classes = max(item["classid"] for item in labels) + 1
    class_names = [
        next(
            (item["name"] for item in labels if item["classid"] == class_id),
            "class_{}".format(class_id),
        )
        for class_id in range(num_classes)
    ]
    color_to_class = {
        rgb_to_key(item["color"]): item["classid"]
        for item in labels
    }
    return labels, class_names, color_to_class, num_classes


def rgb_to_key(color):
    red, green, blue = [int(value) for value in color]
    return (red << 16) + (green << 8) + blue


def build_color_lut(color_to_class):
    color_lut = np.full(256 ** 3, -1, dtype=np.int16)
    for color_key, class_id in color_to_class.items():
        color_lut[int(color_key)] = int(class_id)
    return color_lut


def list_png_names(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError("Directory not found: {}".format(directory))

    return sorted(
        name
        for name in os.listdir(directory)
        if name.lower().endswith(".png")
    )


def sequence_name(filename):
    match = SEQUENCE_RE.match(filename)
    if match:
        return match.group(1)
    return os.path.splitext(filename)[0].split("_", 1)[0]


def sequence_sort_key(name):
    if name.startswith("seq") and name[3:].isdigit():
        return 0, int(name[3:])
    return 1, name


def load_label_counts(label_path, color_lut, num_classes):
    label_image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    if label_image is None:
        raise FileNotFoundError("Could not read label: {}".format(label_path))

    if label_image.ndim == 2:
        values = label_image.astype(np.int64, copy=False)
        valid = values < num_classes
        counts = np.bincount(
            values[valid].ravel(),
            minlength=num_classes,
        )[:num_classes]
        unknown_pixels = int(np.count_nonzero(~valid))
        return counts.astype(np.int64), unknown_pixels

    if label_image.shape[2] == 4:
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGRA2RGB)
    else:
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

    encoded = (
        (label_image[:, :, 0].astype(np.uint32) << 16)
        + (label_image[:, :, 1].astype(np.uint32) << 8)
        + label_image[:, :, 2].astype(np.uint32)
    )
    class_ids = color_lut[encoded]
    valid = class_ids >= 0
    counts = np.bincount(
        class_ids[valid].ravel(),
        minlength=num_classes,
    )[:num_classes]
    unknown_pixels = int(np.count_nonzero(~valid))
    return counts.astype(np.int64), unknown_pixels


def collect_split_stats(root, split, label_json, count_all_labels, max_files, show_progress):
    split_dir = os.path.join(root, split)
    image_dir = os.path.join(split_dir, "imgs")
    label_dir = os.path.join(split_dir, "labels")

    label_json_path = resolve_label_json(root, split, label_json)
    labels, class_names, color_to_class, num_classes = load_label_spec(label_json_path)
    color_lut = build_color_lut(color_to_class)

    image_names = set(list_png_names(image_dir))
    label_names = list_png_names(label_dir)
    if count_all_labels:
        candidate_sample_names = label_names
    else:
        candidate_sample_names = [name for name in label_names if name in image_names]

    sample_names = candidate_sample_names
    if max_files is not None:
        sample_names = candidate_sample_names[:max_files]

    stats = defaultdict(
        lambda: {
            "counts": np.zeros(num_classes, dtype=np.int64),
            "frames": 0,
            "unknown_pixels": 0,
        }
    )
    iterator = sample_names
    if show_progress:
        iterator = tqdm(sample_names, desc="{} labels".format(split), unit="file")

    for name in iterator:
        label_path = os.path.join(label_dir, name)
        counts, unknown_pixels = load_label_counts(
            label_path,
            color_lut,
            num_classes,
        )
        seq_name = sequence_name(name)
        stats[seq_name]["counts"] += counts
        stats[seq_name]["frames"] += 1
        stats[seq_name]["unknown_pixels"] += unknown_pixels

    total = {
        "counts": np.zeros(num_classes, dtype=np.int64),
        "frames": 0,
        "unknown_pixels": 0,
    }
    for seq_stats in stats.values():
        total["counts"] += seq_stats["counts"]
        total["frames"] += seq_stats["frames"]
        total["unknown_pixels"] += seq_stats["unknown_pixels"]

    unmatched_labels = sorted(set(label_names) - image_names)
    missing_labels = sorted(image_names - set(label_names))

    return {
        "labels": labels,
        "class_names": class_names,
        "num_classes": num_classes,
        "label_json_path": label_json_path,
        "image_count": len(image_names),
        "label_count": len(label_names),
        "candidate_sample_count": len(candidate_sample_names),
        "sample_count": len(sample_names),
        "unmatched_label_count": len(unmatched_labels),
        "missing_label_count": len(missing_labels),
        "max_files": max_files,
        "count_all_labels": count_all_labels,
        "stats": dict(stats),
        "total": total,
    }


def format_int(value):
    return "{:,}".format(int(value))


def format_percent(value):
    return "{:6.2f}%".format(float(value))


def distribution_start_index(ignore_background):
    return 1 if ignore_background else 0


def print_distribution(title, stats, class_names, ignore_background=False):
    counts = stats["counts"]
    start_index = distribution_start_index(ignore_background)
    displayed_counts = counts[start_index:]
    total_pixels = int(displayed_counts.sum())
    pixel_label = "foreground pixels" if ignore_background else "labeled pixels"

    print(title)
    print("  frames: {}".format(format_int(stats["frames"])))
    print("  {}: {}".format(pixel_label, format_int(total_pixels)))
    if ignore_background and len(counts) > 0:
        print("  ignored background pixels: {}".format(format_int(counts[0])))
    if stats["unknown_pixels"]:
        print("  unknown-color pixels: {}".format(format_int(stats["unknown_pixels"])))

    header = "{:<5} {:<28} {:>14} {:>10}".format(
        "id",
        "class",
        "pixels",
        "percent",
    )
    print("  " + header)
    print("  " + "-" * len(header))
    for class_id in range(start_index, len(class_names)):
        class_name = class_names[class_id]
        pixels = int(counts[class_id])
        percent = 0.0 if total_pixels == 0 else (pixels / total_pixels) * 100.0
        print(
            "  {:<5} {:<28} {:>14} {:>10}".format(
                class_id,
                class_name,
                format_int(pixels),
                format_percent(percent),
            )
        )
    print()


def print_split_report(split, split_stats, ignore_background=False):
    print("=" * 80)
    print("Split: {}".format(split))
    print("Label spec: {}".format(split_stats["label_json_path"]))
    counted_label = "counted labels" if split_stats["count_all_labels"] else "counted image-label pairs"
    print(
        "Images: {images} | labels: {labels} | {counted_label}: {samples}".format(
            images=format_int(split_stats["image_count"]),
            labels=format_int(split_stats["label_count"]),
            counted_label=counted_label,
            samples=format_int(split_stats["sample_count"]),
        )
    )
    if split_stats["max_files"] is not None:
        print(
            "Applied --max-files: {limit} of {candidates} candidate files counted".format(
                limit=format_int(split_stats["sample_count"]),
                candidates=format_int(split_stats["candidate_sample_count"]),
            )
        )
    if split_stats["unmatched_label_count"]:
        prefix = "Labels without matching images"
        if not split_stats["count_all_labels"]:
            prefix += " (not counted)"
        print(
            "{}: {}".format(
                prefix,
                format_int(split_stats["unmatched_label_count"])
            )
        )
    if split_stats["missing_label_count"]:
        print(
            "Images without labels: {}".format(
                format_int(split_stats["missing_label_count"])
            )
        )
    print()

    class_names = split_stats["class_names"]
    for seq_name in sorted(split_stats["stats"], key=sequence_sort_key):
        print_distribution(
            "Sequence {}".format(seq_name),
            split_stats["stats"][seq_name],
            class_names,
            ignore_background=ignore_background,
        )

    print_distribution(
        "Split total",
        split_stats["total"],
        class_names,
        ignore_background=ignore_background,
    )


def _load_matplotlib(plot_dir):
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = os.path.join(plot_dir, ".matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_config_dir

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _class_percentages(counts, ignore_background=False):
    start_index = distribution_start_index(ignore_background)
    displayed_counts = counts[start_index:]
    total_pixels = int(displayed_counts.sum())
    if total_pixels == 0:
        return np.zeros_like(displayed_counts, dtype=np.float64)
    return displayed_counts.astype(np.float64) * 100.0 / float(total_pixels)


def plot_split_distribution(split, split_stats, plot_dir, ignore_background=False):
    os.makedirs(plot_dir, exist_ok=True)
    plt = _load_matplotlib(plot_dir)

    all_class_names = split_stats["class_names"]
    start_index = distribution_start_index(ignore_background)
    class_ids = list(range(start_index, len(all_class_names)))
    class_names = all_class_names[start_index:]
    counts = split_stats["total"]["counts"]
    percentages = _class_percentages(counts, ignore_background=ignore_background)
    colors = plt.get_cmap("tab20").colors
    bar_colors = [colors[class_id % len(colors)] for class_id in class_ids]
    title_suffix = "foreground class distribution" if ignore_background else "class distribution"

    fig_width = max(10.0, len(class_names) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)
    ax.bar(class_names, percentages, color=bar_colors)
    ax.set_title("{} split {}".format(split, title_suffix))
    ax.set_ylabel("Pixels (%)")
    ax.set_ylim(0, max(100.0, float(percentages.max()) * 1.15))
    ax.tick_params(axis="x", labelrotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    total_path = os.path.join(plot_dir, "{}_class_distribution.png".format(split))
    fig.savefig(total_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    seq_names = sorted(split_stats["stats"], key=sequence_sort_key)
    seq_values = np.array(
        [
            _class_percentages(
                split_stats["stats"][seq_name]["counts"],
                ignore_background=ignore_background,
            )
            for seq_name in seq_names
        ],
        dtype=np.float64,
    )

    fig_width = max(10.0, len(seq_names) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0), constrained_layout=True)
    bottom = np.zeros(len(seq_names), dtype=np.float64)
    for display_idx, class_name in enumerate(class_names):
        class_id = class_ids[display_idx]
        values = seq_values[:, display_idx] if len(seq_names) else []
        ax.bar(
            seq_names,
            values,
            bottom=bottom,
            label=class_name,
            color=colors[class_id % len(colors)],
        )
        if len(seq_names):
            bottom += values

    ax.set_title("{} split per-sequence {}".format(split, title_suffix))
    ax.set_ylabel("Pixels (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelrotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    sequence_path = os.path.join(
        plot_dir,
        "{}_per_sequence_distribution.png".format(split),
    )
    fig.savefig(sequence_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return [total_path, sequence_path]


def main():
    args = parse_args()
    root = os.path.abspath(args.root)

    for split in args.splits:
        split_stats = collect_split_stats(
            root=root,
            split=split,
            label_json=args.label_json,
            count_all_labels=args.count_all_labels,
            max_files=args.max_files,
            show_progress=not args.no_progress,
        )
        print_split_report(
            split,
            split_stats,
            ignore_background=args.ignore_background,
        )
        if args.plot:
            plot_paths = plot_split_distribution(
                split,
                split_stats,
                args.plot_dir,
                ignore_background=args.ignore_background,
            )
            for plot_path in plot_paths:
                print("Saved plot: {}".format(plot_path))


if __name__ == "__main__":
    main()
