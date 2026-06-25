#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_ROOT = "/data/halyusuf/data/CataractData"
CASE_RE = re.compile(r"^(case_?\d+)_")

CLASS_NAMES = [
    "Background",
    "Pupil",
    "Cornea",
    "Lens",
    "Instruments",
]
CLASS_MAP = {
    "Pupil": 1,
    "Cornea": 2,
    "Lens": 3,
    "Instruments": 4,
}
INSTRUMENT_CLASSES = {
    "Slit Knife",
    "Gauge",
    "Capsulorhexis Cystotome",
    "Spatula",
    "Phacoemulsification Tip",
    "Irrigation-Aspiration",
    "Lens Injector",
    "Incision Knife",
    "Katena Forceps",
    "Capsulorhexis Forceps",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Cataract1k/Catarakt1k class pixel distributions per "
            "sequence for train and test splits."
        )
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="Dataset root containing img/, ann/, train.csv, and test.csv.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Dataset splits to summarize.",
    )
    parser.add_argument(
        "--train-csv",
        default="train.csv",
        help="Train split CSV filename or absolute path.",
    )
    parser.add_argument(
        "--test-csv",
        default="test.csv",
        help="Test split CSV filename or absolute path.",
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
    return parser.parse_args()


def resolve_csv_path(root, csv_path):
    if os.path.isabs(csv_path):
        return csv_path
    return os.path.join(root, csv_path)


def split_csv_path(root, split, train_csv, test_csv):
    csv_path = train_csv if split == "train" else test_csv
    resolved = resolve_csv_path(root, csv_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError("CSV not found: {}".format(resolved))
    return resolved


def load_split_rows(root, csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "imgs" not in reader.fieldnames:
            raise ValueError("CSV must contain an 'imgs' column: {}".format(csv_path))

        for row in reader:
            filename = os.path.basename(row["imgs"])
            image_path = os.path.join(root, "img", filename)
            annotation_path = os.path.join(root, "ann", filename + ".json")
            rows.append(
                {
                    "filename": filename,
                    "image_path": image_path,
                    "annotation_path": annotation_path,
                }
            )
    return rows


def sequence_name(filename):
    match = CASE_RE.match(filename)
    if match:
        return match.group(1)
    return os.path.splitext(filename)[0].split("_", 1)[0]


def sequence_sort_key(name):
    normalized = name.replace("_", "")
    if normalized.startswith("case") and normalized[4:].isdigit():
        return 0, int(normalized[4:])
    return 1, name


def annotation_shape(annotation, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is not None:
        return image.shape[:2]

    size = annotation.get("size", {})
    height = size.get("height")
    width = size.get("width")
    if height is None or width is None:
        raise FileNotFoundError(
            "Could not read image and annotation has no size: {}".format(image_path)
        )
    return int(height), int(width)


def annotation_to_counts(annotation, image_path):
    height, width = annotation_shape(annotation, image_path)
    mask = np.zeros((height, width), dtype=np.uint8)
    ignored_titles = Counter()

    for obj in annotation.get("objects", []):
        class_title = obj.get("classTitle")
        if class_title in CLASS_MAP:
            class_id = CLASS_MAP[class_title]
        elif class_title in INSTRUMENT_CLASSES:
            class_id = 4
        else:
            ignored_titles[class_title or "<missing>"] += 1
            continue

        exterior = obj.get("points", {}).get("exterior")
        if not exterior:
            ignored_titles["<missing polygon>"] += 1
            continue

        exterior_points = np.array(exterior, dtype=np.int32)
        cv2.fillPoly(mask, [exterior_points], class_id)

    counts = np.bincount(mask.ravel(), minlength=len(CLASS_NAMES))[:len(CLASS_NAMES)]
    return counts.astype(np.int64), ignored_titles


def collect_split_stats(root, split, train_csv, test_csv, max_files, show_progress):
    csv_path = split_csv_path(root, split, train_csv, test_csv)
    rows = load_split_rows(root, csv_path)
    if max_files is not None:
        rows = rows[:max_files]

    stats = defaultdict(
        lambda: {
            "counts": np.zeros(len(CLASS_NAMES), dtype=np.int64),
            "frames": 0,
        }
    )
    total = {
        "counts": np.zeros(len(CLASS_NAMES), dtype=np.int64),
        "frames": 0,
    }
    ignored_titles = Counter()
    missing_images = 0
    missing_annotations = 0

    iterator = rows
    if show_progress:
        iterator = tqdm(rows, desc="{} annotations".format(split), unit="file")

    for row in iterator:
        if not os.path.exists(row["image_path"]):
            missing_images += 1
        if not os.path.exists(row["annotation_path"]):
            missing_annotations += 1
            continue

        with open(row["annotation_path"], "r") as f:
            annotation = json.load(f)

        counts, ignored = annotation_to_counts(annotation, row["image_path"])
        seq_name = sequence_name(row["filename"])
        stats[seq_name]["counts"] += counts
        stats[seq_name]["frames"] += 1
        total["counts"] += counts
        total["frames"] += 1
        ignored_titles.update(ignored)

    return {
        "csv_path": csv_path,
        "row_count": len(rows),
        "stats": dict(stats),
        "total": total,
        "ignored_titles": ignored_titles,
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
        "max_files": max_files,
    }


def format_int(value):
    return "{:,}".format(int(value))


def format_percent(value):
    return "{:6.2f}%".format(float(value))


def print_distribution(title, stats):
    counts = stats["counts"]
    total_pixels = int(counts.sum())
    print(title)
    print("  frames: {}".format(format_int(stats["frames"])))
    print("  pixels: {}".format(format_int(total_pixels)))

    header = "{:<5} {:<16} {:>14} {:>10}".format(
        "id",
        "class",
        "pixels",
        "percent",
    )
    print("  " + header)
    print("  " + "-" * len(header))
    for class_id, class_name in enumerate(CLASS_NAMES):
        pixels = int(counts[class_id])
        percent = 0.0 if total_pixels == 0 else (pixels / total_pixels) * 100.0
        print(
            "  {:<5} {:<16} {:>14} {:>10}".format(
                class_id,
                class_name,
                format_int(pixels),
                format_percent(percent),
            )
        )
    print()


def print_split_report(split, split_stats):
    print("=" * 80)
    print("Split: {}".format(split))
    print("CSV: {}".format(split_stats["csv_path"]))
    print("Rows counted: {}".format(format_int(split_stats["row_count"])))
    if split_stats["max_files"] is not None:
        print("Applied --max-files: {}".format(format_int(split_stats["row_count"])))
    if split_stats["missing_images"]:
        print("Missing images: {}".format(format_int(split_stats["missing_images"])))
    if split_stats["missing_annotations"]:
        print(
            "Missing annotations: {}".format(
                format_int(split_stats["missing_annotations"])
            )
        )
    if split_stats["ignored_titles"]:
        preview = ", ".join(
            "{} ({})".format(title, count)
            for title, count in split_stats["ignored_titles"].most_common(10)
        )
        print("Ignored annotation titles: {}".format(preview))
    print()

    for seq_name in sorted(split_stats["stats"], key=sequence_sort_key):
        print_distribution(
            "Sequence {}".format(seq_name),
            split_stats["stats"][seq_name],
        )

    print_distribution("Split total", split_stats["total"])


def main():
    args = parse_args()
    root = os.path.abspath(args.root)

    for split in args.splits:
        split_stats = collect_split_stats(
            root=root,
            split=split,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            max_files=args.max_files,
            show_progress=not args.no_progress,
        )
        print_split_report(split, split_stats)


if __name__ == "__main__":
    main()
