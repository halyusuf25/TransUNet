#!/usr/bin/env python3
"""
check_dataset_labels.py

Standalone label-value checker for the segmentation datasets used in your codebase:
Synapse, ACDC, Cataract1k, and EndoVis2018.

What it checks
--------------
For CrossEntropyLoss-style segmentation training, every target pixel should be an
integer class id in [0, num_classes - 1]. In your current trainer, 255 is NOT
ignored, so any 255 pixel is flagged as a problem for the current training code.

The script scans raw labels BEFORE augmentation/resizing, because the issue is in
raw target values.

Examples
--------
# Synapse, using the same default paths as your train.py config:
python check_dataset_labels.py --dataset Synapse \
  --root_path /data/halyusuf/data/Synapse/train_npz \
  --list_dir ./lists/lists_Synapse \
  --num_classes 9

# ACDC:
python check_dataset_labels.py --dataset ACDC \
  --root_path /data/halyusuf/data/ACDC \
  --num_classes 4

# Cataract1k:
python check_dataset_labels.py --dataset Cataract1k \
  --root_path /data/halyusuf/data/CataractData \
  --num_classes 5

# EndoVis2018:
python check_dataset_labels.py --dataset EndoVis2018 \
  --root_path /data/halyusuf/data/EndoVis_2018 \
  --num_classes 12

# Make the command fail with a non-zero exit code if problems are found:
python check_dataset_labels.py --dataset ACDC --root_path /data/halyusuf/data/ACDC --fail-on-problem
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DATASET_CONFIG = {
    "Synapse": {
        "root_path": "/data/halyusuf/data/Synapse/train_npz/",
        "list_dir": "./lists/lists_Synapse",
        "num_classes": 9,
    },
    "Cataract1k": {
        "root_path": "/data/halyusuf/data/CataractData/",
        "list_dir": None,
        "num_classes": 5,
    },
    "ACDC": {
        "root_path": "/data/halyusuf/data/ACDC",
        "list_dir": None,
        "num_classes": 4,
    },
    "EndoVis2018": {
        "root_path": "/data/halyusuf/data/EndoVis_2018",
        "list_dir": None,
        "num_classes": 12,
    },
}

CATARACT_CLASS_MAP = {
    "Pupil": 1,
    "pupil1": 1,
    "Cornea": 2,
    "cornea1": 2,
    "Lens": 3,
    "Instruments": 4,
}

CATARACT_INSTRUMENTS = {
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

ACDC_ALL_CASES = [f"patient{i:03d}" for i in range(1, 101)]
ACDC_TEST_OR_VAL_CASES = {f"patient{i:03d}" for i in range(1, 21)}
ACDC_TRAIN_CASES = set(ACDC_ALL_CASES) - ACDC_TEST_OR_VAL_CASES


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def import_or_raise(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing Python dependency '{module_name}'. {install_hint}"
        ) from exc


def json_safe_value(value: Any) -> Any:
    """Convert NumPy/Python scalars into safe JSON values."""
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if value.is_integer():
            return int(value)
        return float(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return json_safe_value(float(value))

    if isinstance(value, (str, bool)) or value is None:
        return value

    return str(value)


def value_sort_key(entry: Dict[str, Any]) -> Tuple[int, float, str]:
    value = entry.get("value")
    if isinstance(value, int):
        return (0, float(value), "")
    if isinstance(value, float):
        return (0, float(value), "")
    return (1, 0.0, str(value))


def path_to_str(path: Path) -> str:
    return str(path.expanduser())


def read_text_lines(path: Path) -> List[str]:
    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def deduplicate_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    result = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def limit_sequence(seq: Sequence[Any], max_items: int) -> List[Any]:
    if max_items is None or max_items <= 0:
        return list(seq)
    return list(seq[:max_items])


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)


def write_problem_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "split",
        "path",
        "kind",
        "problem_for_current_training",
        "warning_only",
        "error",
        "shape",
        "dtype",
        "total_elements",
        "unique_value_count",
        "valid_pixel_count",
        "outside_valid_pixel_count",
        "ignore_pixel_count",
        "non_integer_pixel_count",
        "non_finite_pixel_count",
        "unknown_color_pixel_count",
        "unknown_cataract_class_titles",
        "values_outside_valid_range",
        "ignore_value_counts",
        "unique_values_preview",
        "message",
    ]

    problemish = [
        r for r in records
        if r.get("problem_for_current_training") or r.get("warning_only") or r.get("error")
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in problemish:
            row = {}
            for field in fields:
                value = record.get(field, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                row[field] = value
            writer.writerow(row)


# -----------------------------------------------------------------------------
# Label value analysis
# -----------------------------------------------------------------------------

def classify_numeric_value(
    raw_value: Any,
    count: int,
    num_classes: int,
    ignore_index: Optional[int],
    integer_tolerance: float,
) -> Dict[str, Any]:
    """Classify one unique label value."""
    v_json = json_safe_value(raw_value)

    try:
        value_float = float(raw_value)
    except Exception:
        return {
            "value": v_json,
            "count": int(count),
            "category": "non_numeric",
            "problem_for_current_training": True,
        }

    if not math.isfinite(value_float):
        return {
            "value": v_json,
            "count": int(count),
            "category": "non_finite",
            "problem_for_current_training": True,
        }

    rounded = round(value_float)
    is_integer_like = abs(value_float - rounded) <= integer_tolerance
    if not is_integer_like:
        return {
            "value": v_json,
            "count": int(count),
            "category": "non_integer",
            "problem_for_current_training": True,
        }

    value_int = int(rounded)
    if 0 <= value_int < num_classes:
        return {
            "value": value_int,
            "count": int(count),
            "category": "valid_class_id",
            "problem_for_current_training": False,
        }

    if ignore_index is not None and value_int == int(ignore_index):
        return {
            "value": value_int,
            "count": int(count),
            "category": "ignore_index_value",
            # Current trainer uses CrossEntropyLoss() with no ignore_index and DiceLoss has no ignore support.
            "problem_for_current_training": True,
        }

    return {
        "value": value_int,
        "count": int(count),
        "category": "outside_valid_range",
        "problem_for_current_training": True,
    }


def analyze_label_array(
    array: np.ndarray,
    *,
    dataset: str,
    split: str,
    source_path: Path,
    kind: str,
    num_classes: int,
    ignore_index: Optional[int],
    integer_tolerance: float,
    preview_values: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    arr = np.asarray(array)
    flat = arr.reshape(-1)

    record: Dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "path": path_to_str(source_path),
        "kind": kind,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "total_elements": int(flat.size),
        "num_classes": int(num_classes),
        "valid_class_id_range": [0, int(num_classes) - 1],
        "ignore_index_checked": None if ignore_index is None else int(ignore_index),
        "error": None,
        "warning_only": False,
    }
    if extra:
        record.update(extra)

    if flat.size == 0:
        record.update(
            {
                "unique_value_count": 0,
                "unique_values_preview": [],
                "valid_pixel_count": 0,
                "outside_valid_pixel_count": 0,
                "ignore_pixel_count": 0,
                "non_integer_pixel_count": 0,
                "non_finite_pixel_count": 0,
                "values_outside_valid_range": [],
                "ignore_value_counts": [],
                "problem_for_current_training": True,
                "message": "Empty label array.",
            }
        )
        return record

    try:
        unique_values, unique_counts = np.unique(flat, return_counts=True)
    except Exception as exc:
        record.update(
            {
                "problem_for_current_training": True,
                "error": repr(exc),
                "message": "Could not compute unique label values.",
            }
        )
        return record

    classified = [
        classify_numeric_value(v, c, num_classes, ignore_index, integer_tolerance)
        for v, c in zip(unique_values, unique_counts)
    ]

    valid_entries = [x for x in classified if x["category"] == "valid_class_id"]
    outside_entries = [
        x for x in classified
        if x["category"] in {"outside_valid_range", "ignore_index_value", "non_integer", "non_finite", "non_numeric"}
    ]
    outside_excluding_ignore = [x for x in outside_entries if x["category"] != "ignore_index_value"]
    ignore_entries = [x for x in classified if x["category"] == "ignore_index_value"]
    non_integer_entries = [x for x in classified if x["category"] == "non_integer"]
    non_finite_entries = [x for x in classified if x["category"] == "non_finite"]

    sorted_classified = sorted(classified, key=value_sort_key)
    if len(sorted_classified) <= preview_values:
        preview = sorted_classified
    else:
        # Show the most frequent values if there are too many distinct values.
        preview = sorted(classified, key=lambda x: int(x.get("count", 0)), reverse=True)[:preview_values]

    record.update(
        {
            "unique_value_count": int(len(classified)),
            "unique_values_preview": preview,
            "valid_pixel_count": int(sum(x["count"] for x in valid_entries)),
            "outside_valid_pixel_count": int(sum(x["count"] for x in outside_entries)),
            "outside_valid_excluding_ignore_pixel_count": int(sum(x["count"] for x in outside_excluding_ignore)),
            "ignore_pixel_count": int(sum(x["count"] for x in ignore_entries)),
            "non_integer_pixel_count": int(sum(x["count"] for x in non_integer_entries)),
            "non_finite_pixel_count": int(sum(x["count"] for x in non_finite_entries)),
            "values_outside_valid_range": outside_entries,
            "values_outside_valid_range_excluding_ignore": outside_excluding_ignore,
            "ignore_value_counts": ignore_entries,
            "problem_for_current_training": bool(any(x["problem_for_current_training"] for x in classified)),
        }
    )

    if record["problem_for_current_training"]:
        bits = []
        if record["ignore_pixel_count"] > 0:
            bits.append(
                f"found {record['ignore_pixel_count']} pixels equal to ignore_index={ignore_index}; current trainer does not ignore them"
            )
        if record["outside_valid_excluding_ignore_pixel_count"] > 0:
            bad_preview = [x["value"] for x in outside_excluding_ignore[:10]]
            bits.append(f"found values outside 0..{num_classes - 1}: {bad_preview}")
        if record["non_integer_pixel_count"] > 0:
            bits.append(f"found {record['non_integer_pixel_count']} non-integer label pixels")
        if record["non_finite_pixel_count"] > 0:
            bits.append(f"found {record['non_finite_pixel_count']} NaN/Inf label pixels")
        record["message"] = "; ".join(bits) if bits else "Invalid label values found."
    else:
        record["message"] = f"OK: all label values are integer ids in 0..{num_classes - 1}."

    return record


def make_error_record(
    *,
    dataset: str,
    split: str,
    source_path: Path,
    kind: str,
    num_classes: int,
    message: str,
    exc: Optional[BaseException] = None,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "path": path_to_str(source_path),
        "kind": kind,
        "shape": None,
        "dtype": None,
        "total_elements": 0,
        "unique_value_count": 0,
        "unique_values_preview": [],
        "valid_pixel_count": 0,
        "outside_valid_pixel_count": 0,
        "ignore_pixel_count": 0,
        "non_integer_pixel_count": 0,
        "non_finite_pixel_count": 0,
        "unknown_color_pixel_count": 0,
        "values_outside_valid_range": [],
        "ignore_value_counts": [],
        "num_classes": int(num_classes),
        "valid_class_id_range": [0, int(num_classes) - 1],
        "problem_for_current_training": True,
        "warning_only": False,
        "error": repr(exc) if exc is not None else message,
        "message": message,
        "traceback": traceback.format_exc() if exc is not None else None,
    }


# -----------------------------------------------------------------------------
# EndoVis2018 scanner
# -----------------------------------------------------------------------------

def resolve_endovis_label_json(root_path: Path, label_json: str) -> Path:
    candidate = Path(label_json)
    if candidate.is_absolute():
        return candidate
    return root_path / label_json


def load_endovis_labels(label_json_path: Path) -> Tuple[Dict[Tuple[int, int, int], int], Optional[int], List[Dict[str, Any]]]:
    with label_json_path.open("r") as f:
        labels = json.load(f)

    color_to_class: Dict[Tuple[int, int, int], int] = {}
    class_ids = []
    for item in labels:
        color = tuple(int(v) for v in item["color"])
        class_id = int(item["classid"])
        if len(color) != 3:
            raise ValueError(f"Invalid RGB color in labels.json entry: {item}")
        color_to_class[color] = class_id
        class_ids.append(class_id)

    inferred_num_classes = max(class_ids) + 1 if class_ids else None
    return color_to_class, inferred_num_classes, labels


def discover_endovis_paths(root_path: Path, split: str) -> List[Tuple[str, Path]]:
    splits = ["train", "test"] if split == "all" else [split]
    items: List[Tuple[str, Path]] = []
    for sp in splits:
        label_dir = root_path / sp / "labels"
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.glob("*.png")):
            items.append((sp, path))
    return items


def read_endovis_label_as_ids(
    path: Path,
    color_to_class: Dict[Tuple[int, int, int], int],
    max_unknown_colors: int,
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    cv2 = import_or_raise("cv2", "Install opencv-python, or run inside your existing training environment.")

    label_image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label_image is None:
        raise FileNotFoundError(f"Could not read label image: {path}")

    extra: Dict[str, Any] = {
        "endovis_original_label_shape": list(label_image.shape),
        "endovis_original_label_dtype": str(label_image.dtype),
        "unknown_color_pixel_count": 0,
        "unknown_colors_preview": [],
    }

    if label_image.ndim == 2:
        return label_image, "png_grayscale_label", extra

    if label_image.ndim != 3 or label_image.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported EndoVis label shape {label_image.shape} for {path}")

    if label_image.shape[2] == 4:
        label_rgb = cv2.cvtColor(label_image, cv2.COLOR_BGRA2RGB)
    else:
        label_rgb = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

    mask = np.full(label_rgb.shape[:2], 255, dtype=np.int64)
    for color, class_id in color_to_class.items():
        color_arr = np.asarray(color, dtype=label_rgb.dtype)
        matches = np.all(label_rgb == color_arr, axis=-1)
        mask[matches] = int(class_id)

    unknown = mask == 255
    unknown_count = int(np.sum(unknown))
    if unknown_count > 0:
        unknown_colors, counts = np.unique(label_rgb[unknown].reshape(-1, 3), axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        preview = [
            {"rgb": unknown_colors[i].astype(int).tolist(), "count": int(counts[i])}
            for i in order[:max_unknown_colors]
        ]
        extra["unknown_color_pixel_count"] = unknown_count
        extra["unknown_colors_preview"] = preview
        extra["unknown_color_message"] = "This color label would raise ValueError in EndoVis2018Dataset._color_label_to_class_ids."

    return mask, "png_color_label_mapped_to_ids", extra


def scan_endovis(args: argparse.Namespace, num_classes: int) -> List[Dict[str, Any]]:
    root = Path(args.root_path)
    label_json_path = resolve_endovis_label_json(root, args.label_json)
    records: List[Dict[str, Any]] = []

    try:
        color_to_class, inferred_num_classes, _labels = load_endovis_labels(label_json_path)
    except Exception as exc:
        return [
            make_error_record(
                dataset="EndoVis2018",
                split=args.split,
                source_path=label_json_path,
                kind="labels_json",
                num_classes=num_classes,
                message=f"Could not load EndoVis labels.json: {label_json_path}",
                exc=exc,
            )
        ]

    if inferred_num_classes is not None and int(inferred_num_classes) != int(num_classes):
        records.append(
            {
                "dataset": "EndoVis2018",
                "split": "all",
                "path": path_to_str(label_json_path),
                "kind": "labels_json_num_classes_warning",
                "shape": None,
                "dtype": None,
                "total_elements": 0,
                "unique_value_count": 0,
                "unique_values_preview": [],
                "valid_pixel_count": 0,
                "outside_valid_pixel_count": 0,
                "ignore_pixel_count": 0,
                "non_integer_pixel_count": 0,
                "non_finite_pixel_count": 0,
                "unknown_color_pixel_count": 0,
                "values_outside_valid_range": [],
                "ignore_value_counts": [],
                "num_classes": int(num_classes),
                "inferred_num_classes_from_labels_json": int(inferred_num_classes),
                "valid_class_id_range": [0, int(num_classes) - 1],
                "problem_for_current_training": inferred_num_classes > num_classes,
                "warning_only": inferred_num_classes <= num_classes,
                "error": None,
                "message": (
                    f"labels.json implies {inferred_num_classes} classes, but --num_classes={num_classes}. "
                    "If these differ from train.py, model channels and labels can mismatch."
                ),
            }
        )

    paths = discover_endovis_paths(root, args.split)
    paths = paths[: args.max_files] if args.max_files else paths
    if not paths:
        records.append(
            make_error_record(
                dataset="EndoVis2018",
                split=args.split,
                source_path=root,
                kind="dataset_discovery",
                num_classes=num_classes,
                message=f"No EndoVis label PNG files found under {root}/<split>/labels for split={args.split}.",
            )
        )
        return records

    for sp, path in paths:
        try:
            label_ids, kind, extra = read_endovis_label_as_ids(path, color_to_class, args.max_unknown_colors)
            rec = analyze_label_array(
                label_ids,
                dataset="EndoVis2018",
                split=sp,
                source_path=path,
                kind=kind,
                num_classes=num_classes,
                ignore_index=args.ignore_index,
                integer_tolerance=args.integer_tolerance,
                preview_values=args.preview_values,
                extra=extra,
            )
            if rec.get("unknown_color_pixel_count", 0) > 0:
                rec["problem_for_current_training"] = True
                rec["message"] = (
                    f"Unknown RGB colors found in {rec['unknown_color_pixel_count']} pixels. "
                    "Your dataset loader would raise an error before training this sample."
                )
            records.append(rec)
        except Exception as exc:
            records.append(
                make_error_record(
                    dataset="EndoVis2018",
                    split=sp,
                    source_path=path,
                    kind="png_label",
                    num_classes=num_classes,
                    message=f"Could not inspect EndoVis label file: {path}",
                    exc=exc,
                )
            )
    return records


# -----------------------------------------------------------------------------
# Synapse scanner
# -----------------------------------------------------------------------------

def infer_synapse_volume_root(root_path: Path, volume_path: Optional[str]) -> Path:
    if volume_path:
        return Path(volume_path)
    root_clean = Path(str(root_path).rstrip(os.sep))
    if root_clean.name == "train_npz":
        return root_clean.parent / "test_vol_h5"
    return root_clean


def discover_synapse_items(args: argparse.Namespace) -> List[Tuple[str, Path, str]]:
    root = Path(args.root_path)
    items: List[Tuple[str, Path, str]] = []

    if args.list_dir:
        list_dir = Path(args.list_dir)
        if args.split in {"all", "train"}:
            train_list = list_dir / "train.txt"
            if train_list.exists():
                for name in read_text_lines(train_list):
                    items.append(("train", root / f"{name}.npz", "npz"))
            else:
                items.append(("train", train_list, "missing_list_file"))

        if args.split in {"all", "test", "test_vol", "val"}:
            test_list = list_dir / "test_vol.txt"
            if not test_list.exists():
                alt = list_dir / "test.txt"
                test_list = alt if alt.exists() else test_list
            if test_list.exists():
                volume_root = infer_synapse_volume_root(root, args.volume_path)
                for name in read_text_lines(test_list):
                    items.append(("test_vol", volume_root / f"{name}.npy.h5", "h5"))
            else:
                items.append(("test_vol", test_list, "missing_list_file"))
    else:
        # Fallback: scan whatever files exist below root_path.
        for path in sorted(root.rglob("*.npz")):
            items.append(("unknown", path, "npz"))
        for path in sorted(root.rglob("*.h5")):
            items.append(("unknown", path, "h5"))

    if args.max_files:
        items = items[: args.max_files]
    return items


def read_npz_label(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        if "label" not in data:
            raise KeyError(f"No 'label' key in {path}. Available keys: {list(data.keys())}")
        return np.asarray(data["label"])


def read_h5_label(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    h5py = import_or_raise("h5py", "Install h5py, or run inside your existing training environment.")
    with h5py.File(path, "r") as h5f:
        if "label" not in h5f:
            raise KeyError(f"No 'label' dataset in {path}. Available keys: {list(h5f.keys())}")
        return np.asarray(h5f["label"][:])


def scan_synapse(args: argparse.Namespace, num_classes: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    items = discover_synapse_items(args)
    if not items:
        return [
            make_error_record(
                dataset="Synapse",
                split=args.split,
                source_path=Path(args.root_path),
                kind="dataset_discovery",
                num_classes=num_classes,
                message="No Synapse .npz/.h5 label files discovered. Provide --list_dir or point --root_path at the dataset files.",
            )
        ]

    for sp, path, kind in items:
        try:
            if kind == "missing_list_file":
                raise FileNotFoundError(path)
            label = read_npz_label(path) if kind == "npz" else read_h5_label(path)
            records.append(
                analyze_label_array(
                    label,
                    dataset="Synapse",
                    split=sp,
                    source_path=path,
                    kind=kind,
                    num_classes=num_classes,
                    ignore_index=args.ignore_index,
                    integer_tolerance=args.integer_tolerance,
                    preview_values=args.preview_values,
                )
            )
        except Exception as exc:
            records.append(
                make_error_record(
                    dataset="Synapse",
                    split=sp,
                    source_path=path,
                    kind=kind,
                    num_classes=num_classes,
                    message=f"Could not inspect Synapse {kind} label file/list: {path}",
                    exc=exc,
                )
            )
    return records


# -----------------------------------------------------------------------------
# ACDC scanner
# -----------------------------------------------------------------------------

def acdc_patient_id_from_filename(path: Path) -> str:
    # Example: patient021_frame01_slice_1.h5 or patient001_frame01.h5
    name = path.name
    return name.split("_")[0].replace(".h5", "")


def discover_acdc_items(args: argparse.Namespace) -> List[Tuple[str, Path, str]]:
    root = Path(args.root_path)
    items: List[Tuple[str, Path, str]] = []

    if args.split in {"all", "train"}:
        slice_dir = root / "ACDC_training_slices"
        if slice_dir.is_dir():
            for path in sorted(slice_dir.glob("*.h5")):
                patient_id = acdc_patient_id_from_filename(path)
                if args.split == "all" or patient_id in ACDC_TRAIN_CASES:
                    items.append(("train", path, "h5_slice"))
        else:
            items.append(("train", slice_dir, "missing_directory"))

    if args.split in {"all", "val", "test"}:
        volume_dir = root / "ACDC_training_volumes"
        if volume_dir.is_dir():
            for path in sorted(volume_dir.glob("*.h5")):
                patient_id = acdc_patient_id_from_filename(path)
                if args.split == "all" or patient_id in ACDC_TEST_OR_VAL_CASES:
                    split_name = "test" if args.split in {"all", "test"} else "val"
                    items.append((split_name, path, "h5_volume"))
        else:
            items.append((args.split if args.split != "all" else "test", volume_dir, "missing_directory"))

    if args.max_files:
        items = items[: args.max_files]
    return items


def scan_acdc(args: argparse.Namespace, num_classes: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    items = discover_acdc_items(args)
    if not items:
        return [
            make_error_record(
                dataset="ACDC",
                split=args.split,
                source_path=Path(args.root_path),
                kind="dataset_discovery",
                num_classes=num_classes,
                message="No ACDC H5 files discovered under ACDC_training_slices or ACDC_training_volumes.",
            )
        ]

    for sp, path, kind in items:
        try:
            if kind == "missing_directory":
                raise FileNotFoundError(path)
            label = read_h5_label(path)
            records.append(
                analyze_label_array(
                    label,
                    dataset="ACDC",
                    split=sp,
                    source_path=path,
                    kind=kind,
                    num_classes=num_classes,
                    ignore_index=args.ignore_index,
                    integer_tolerance=args.integer_tolerance,
                    preview_values=args.preview_values,
                )
            )
        except Exception as exc:
            records.append(
                make_error_record(
                    dataset="ACDC",
                    split=sp,
                    source_path=path,
                    kind=kind,
                    num_classes=num_classes,
                    message=f"Could not inspect ACDC label file/directory: {path}",
                    exc=exc,
                )
            )
    return records


# -----------------------------------------------------------------------------
# Cataract1k scanner
# -----------------------------------------------------------------------------

def read_cataract_csv_annotation_paths(root: Path, csv_name: str) -> List[Path]:
    csv_path = root / csv_name
    if not csv_path.exists():
        return []

    result: List[Path] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "imgs" in fieldnames:
            img_field = "imgs"
        elif "img" in fieldnames:
            img_field = "img"
        elif fieldnames:
            img_field = fieldnames[0]
        else:
            return []

        for row in reader:
            raw_img = row.get(img_field, "")
            if not raw_img:
                continue
            file_name = os.path.basename(raw_img)
            result.append(root / "ann" / f"{file_name}.json")
    return result


def discover_cataract_items(args: argparse.Namespace) -> List[Tuple[str, Path, str]]:
    root = Path(args.root_path)
    items: List[Tuple[str, Path, str]] = []

    if args.split in {"all", "train"}:
        train_paths = read_cataract_csv_annotation_paths(root, args.train_csv)
        if train_paths:
            items.extend(("train", p, "json_annotation") for p in train_paths)
        elif args.split == "train":
            # Fallback to all annotations if the CSV is absent.
            items.extend(("train", p, "json_annotation") for p in sorted((root / "ann").glob("*.json")))

    if args.split in {"all", "test", "val"}:
        test_paths = read_cataract_csv_annotation_paths(root, args.test_csv)
        split_name = "test" if args.split != "val" else "val"
        if test_paths:
            items.extend((split_name, p, "json_annotation") for p in test_paths)
        elif args.split in {"test", "val"}:
            items.extend((split_name, p, "json_annotation") for p in sorted((root / "ann").glob("*.json")))

    if args.split == "all" and not items:
        items.extend(("all", p, "json_annotation") for p in sorted((root / "ann").glob("*.json")))

    # Avoid duplicates when train/test CSVs overlap.
    seen = set()
    deduped: List[Tuple[str, Path, str]] = []
    for sp, path, kind in items:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append((sp, path, kind))

    if args.max_files:
        deduped = deduped[: args.max_files]
    return deduped


def analyze_cataract_annotation(
    path: Path,
    *,
    split: str,
    num_classes: int,
    strict_unknown: bool,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r") as f:
        annotation = json.load(f)

    objects = annotation.get("objects", [])
    generated_label_values = {0}
    mapped_object_counts: Counter = Counter()
    unknown_titles: Counter = Counter()
    malformed_objects: List[Dict[str, Any]] = []

    for index, obj in enumerate(objects):
        class_title = obj.get("classTitle")
        class_id = None
        if class_title in CATARACT_CLASS_MAP:
            class_id = CATARACT_CLASS_MAP[class_title]
        elif class_title in CATARACT_INSTRUMENTS and len(CATARACT_CLASS_MAP) > 3:
            class_id = 4
        else:
            unknown_titles[str(class_title)] += 1
            continue

        generated_label_values.add(int(class_id))
        mapped_object_counts[str(class_title)] += 1

        exterior = obj.get("points", {}).get("exterior")
        if not isinstance(exterior, list) or len(exterior) < 3:
            malformed_objects.append(
                {
                    "object_index": int(index),
                    "classTitle": str(class_title),
                    "reason": "points.exterior missing or has fewer than 3 points",
                }
            )

    outside_values = sorted(v for v in generated_label_values if not (0 <= int(v) < int(num_classes)))
    unknown_title_entries = [
        {"classTitle": title, "count": int(count)}
        for title, count in unknown_titles.most_common()
    ]

    warning_only = bool(unknown_title_entries) and not strict_unknown
    problem = bool(outside_values or malformed_objects or (strict_unknown and unknown_title_entries))

    if problem:
        message_bits = []
        if outside_values:
            message_bits.append(f"generated label ids outside 0..{num_classes - 1}: {outside_values}")
        if malformed_objects:
            message_bits.append(f"found {len(malformed_objects)} malformed mapped polygon objects")
        if strict_unknown and unknown_title_entries:
            message_bits.append(f"found unknown/ignored classTitle values: {unknown_title_entries[:10]}")
        message = "; ".join(message_bits)
    elif warning_only:
        message = (
            "Unknown Cataract classTitle values were found. Your current dataset code silently leaves "
            "these objects as background, so this is not a CrossEntropy out-of-range crash, but it may be an annotation issue."
        )
    else:
        message = f"OK: Cataract1k annotation can only generate ids {sorted(generated_label_values)} within 0..{num_classes - 1}."

    # This is not pixel-level because Cataract labels are generated from polygons at load time.
    return {
        "dataset": "Cataract1k",
        "split": split,
        "path": path_to_str(path),
        "kind": "json_annotation",
        "shape": None,
        "dtype": None,
        "total_elements": 0,
        "unique_value_count": int(len(generated_label_values)),
        "unique_values_preview": [
            {"value": int(v), "count": None, "category": "generated_class_id"}
            for v in sorted(generated_label_values)
        ],
        "valid_pixel_count": 0,
        "outside_valid_pixel_count": 0,
        "ignore_pixel_count": 0,
        "non_integer_pixel_count": 0,
        "non_finite_pixel_count": 0,
        "unknown_color_pixel_count": 0,
        "values_outside_valid_range": [
            {"value": int(v), "count": None, "category": "outside_valid_range"}
            for v in outside_values
        ],
        "ignore_value_counts": [],
        "unknown_cataract_class_titles": unknown_title_entries,
        "mapped_object_counts": dict(mapped_object_counts),
        "malformed_objects": malformed_objects[:20],
        "num_objects": int(len(objects)),
        "generated_label_values": sorted(int(v) for v in generated_label_values),
        "num_classes": int(num_classes),
        "valid_class_id_range": [0, int(num_classes) - 1],
        "problem_for_current_training": bool(problem),
        "warning_only": bool(warning_only),
        "error": None,
        "message": message,
    }


def scan_cataract(args: argparse.Namespace, num_classes: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    items = discover_cataract_items(args)
    if not items:
        return [
            make_error_record(
                dataset="Cataract1k",
                split=args.split,
                source_path=Path(args.root_path),
                kind="dataset_discovery",
                num_classes=num_classes,
                message="No Cataract1k annotation JSON files discovered. Check --root_path, --train_csv, --test_csv, and root/ann.",
            )
        ]

    for sp, path, kind in items:
        try:
            records.append(
                analyze_cataract_annotation(
                    path,
                    split=sp,
                    num_classes=num_classes,
                    strict_unknown=args.strict_cataract_unknown,
                )
            )
        except Exception as exc:
            records.append(
                make_error_record(
                    dataset="Cataract1k",
                    split=sp,
                    source_path=path,
                    kind=kind,
                    num_classes=num_classes,
                    message=f"Could not inspect Cataract1k annotation: {path}",
                    exc=exc,
                )
            )
    return records


# -----------------------------------------------------------------------------
# Summary and CLI
# -----------------------------------------------------------------------------

def summarize_records(records: List[Dict[str, Any]], *, dataset: str, root_path: str, num_classes: int, args: argparse.Namespace) -> Dict[str, Any]:
    total_records = len(records)
    problem_records = [r for r in records if r.get("problem_for_current_training")]
    warning_records = [r for r in records if r.get("warning_only")]
    error_records = [r for r in records if r.get("error")]

    value_hist = Counter()
    split_counts = Counter()
    kind_counts = Counter()
    values_outside = Counter()
    ignore_pixels = 0
    outside_valid_pixels = 0
    unknown_color_pixels = 0
    non_integer_pixels = 0
    non_finite_pixels = 0
    valid_pixels = 0
    cataract_unknown_titles = Counter()

    for rec in records:
        split_counts[str(rec.get("split", "unknown"))] += 1
        kind_counts[str(rec.get("kind", "unknown"))] += 1
        valid_pixels += int(rec.get("valid_pixel_count") or 0)
        outside_valid_pixels += int(rec.get("outside_valid_pixel_count") or 0)
        ignore_pixels += int(rec.get("ignore_pixel_count") or 0)
        unknown_color_pixels += int(rec.get("unknown_color_pixel_count") or 0)
        non_integer_pixels += int(rec.get("non_integer_pixel_count") or 0)
        non_finite_pixels += int(rec.get("non_finite_pixel_count") or 0)

        for entry in rec.get("unique_values_preview", []) or []:
            # Only global-count values that have true counts.
            value = entry.get("value")
            count = entry.get("count")
            if isinstance(count, int):
                value_hist[str(value)] += int(count)
        for entry in rec.get("values_outside_valid_range", []) or []:
            values_outside[str(entry.get("value"))] += int(entry.get("count") or 0)
        for entry in rec.get("unknown_cataract_class_titles", []) or []:
            cataract_unknown_titles[str(entry.get("classTitle"))] += int(entry.get("count") or 0)

    return {
        "dataset": dataset,
        "root_path": root_path,
        "num_classes": int(num_classes),
        "valid_class_id_range": [0, int(num_classes) - 1],
        "ignore_index_checked": None if args.ignore_index is None else int(args.ignore_index),
        "split_requested": args.split,
        "total_records_scanned": int(total_records),
        "problem_records_for_current_training": int(len(problem_records)),
        "warning_only_records": int(len(warning_records)),
        "error_records": int(len(error_records)),
        "valid_pixel_count": int(valid_pixels),
        "outside_valid_pixel_count": int(outside_valid_pixels),
        "ignore_pixel_count": int(ignore_pixels),
        "unknown_color_pixel_count": int(unknown_color_pixels),
        "non_integer_pixel_count": int(non_integer_pixels),
        "non_finite_pixel_count": int(non_finite_pixels),
        "split_counts": dict(split_counts),
        "kind_counts": dict(kind_counts),
        "values_outside_valid_range_counts": dict(values_outside.most_common()),
        "cataract_unknown_class_title_counts": dict(cataract_unknown_titles.most_common()),
        "problem_examples": [
            {
                "path": r.get("path"),
                "split": r.get("split"),
                "kind": r.get("kind"),
                "message": r.get("message"),
                "values_outside_valid_range": r.get("values_outside_valid_range"),
                "unknown_colors_preview": r.get("unknown_colors_preview"),
                "unknown_cataract_class_titles": r.get("unknown_cataract_class_titles"),
                "error": r.get("error"),
            }
            for r in problem_records[: args.max_problem_examples]
        ],
        "warning_examples": [
            {
                "path": r.get("path"),
                "split": r.get("split"),
                "kind": r.get("kind"),
                "message": r.get("message"),
                "unknown_cataract_class_titles": r.get("unknown_cataract_class_titles"),
            }
            for r in warning_records[: args.max_problem_examples]
        ],
    }


def print_summary(summary: Dict[str, Any], report_json: Path, problem_csv: Path) -> None:
    dataset = summary["dataset"]
    problems = summary["problem_records_for_current_training"]
    warnings = summary["warning_only_records"]
    errors = summary["error_records"]
    total = summary["total_records_scanned"]
    num_classes = summary["num_classes"]

    print("\n" + "=" * 80)
    print(f"Label check summary: {dataset}")
    print("=" * 80)
    print(f"Root path: {summary['root_path']}")
    print(f"Requested split: {summary['split_requested']}")
    print(f"Expected valid class ids: 0..{num_classes - 1}")
    print(f"Ignore index checked/informative: {summary['ignore_index_checked']}")
    print(f"Records scanned: {total}")
    print(f"Problems for CURRENT training code: {problems}")
    print(f"Warnings only: {warnings}")
    print(f"Read/discovery errors: {errors}")
    print(f"Outside-valid pixels/voxels: {summary['outside_valid_pixel_count']}")
    print(f"Pixels/voxels equal to ignore_index: {summary['ignore_pixel_count']}")
    print(f"Unknown EndoVis color pixels: {summary['unknown_color_pixel_count']}")
    print(f"Non-integer pixels/voxels: {summary['non_integer_pixel_count']}")
    print(f"NaN/Inf pixels/voxels: {summary['non_finite_pixel_count']}")

    if problems:
        print("\nSTATUS: PROBLEM FOUND for your current trainer.")
        print("Reason: your current trainer expects every target value to be an integer class id in the valid range.")
        print("Top problem examples:")
        for i, ex in enumerate(summary["problem_examples"], start=1):
            print(f"  {i}. {ex.get('path')}")
            print(f"     {ex.get('message')}")
    else:
        print("\nSTATUS: OK for the current trainer: no out-of-range label values were found in scanned records.")

    if warnings:
        print("\nWarnings:")
        for i, ex in enumerate(summary["warning_examples"], start=1):
            print(f"  {i}. {ex.get('path')}")
            print(f"     {ex.get('message')}")

    if summary["values_outside_valid_range_counts"]:
        print("\nValues outside valid range, aggregated counts:")
        for value, count in list(summary["values_outside_valid_range_counts"].items())[:20]:
            print(f"  value {value}: {count}")

    if summary["cataract_unknown_class_title_counts"]:
        print("\nCataract unknown/ignored classTitle counts:")
        for title, count in list(summary["cataract_unknown_class_title_counts"].items())[:20]:
            print(f"  {title}: {count}")

    print(f"\nJSON report written to: {report_json}")
    print(f"Problem/warning CSV written to: {problem_csv}")
    print("=" * 80 + "\n")


def resolve_num_classes(args: argparse.Namespace) -> int:
    if args.num_classes is not None:
        return int(args.num_classes)

    default = DEFAULT_DATASET_CONFIG[args.dataset]["num_classes"]

    # For EndoVis, try to infer from labels.json when possible.
    if args.dataset == "EndoVis2018":
        root = Path(args.root_path or DEFAULT_DATASET_CONFIG["EndoVis2018"]["root_path"])
        label_json_path = resolve_endovis_label_json(root, args.label_json)
        if label_json_path.exists():
            try:
                _color_to_class, inferred, _labels = load_endovis_labels(label_json_path)
                if inferred is not None:
                    return int(inferred)
            except Exception:
                pass

    return int(default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check segmentation dataset labels for values outside 0..num_classes-1, including 255 ignore labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["Synapse", "ACDC", "Cataract1k", "EndoVis2018"],
        help="Dataset to scan.",
    )
    parser.add_argument(
        "--root_path",
        default=None,
        help="Dataset root path. If omitted, uses the default path from your train.py-style config.",
    )
    parser.add_argument(
        "--list_dir",
        default=None,
        help="Synapse list directory. If omitted for Synapse, uses default ./lists/lists_Synapse; if empty string, scans root recursively.",
    )
    parser.add_argument(
        "--volume_path",
        default=None,
        help="Optional Synapse test_vol_h5 root. If omitted and root_path ends with train_npz, uses ../test_vol_h5.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Expected number of classes. Defaults: Synapse=9, ACDC=4, Cataract1k=5, EndoVis2018=labels.json or 12.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "test", "test_vol", "val"],
        help="Split to scan. For ACDC, val/test both use volume files in your current dataset code.",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=255,
        help=(
            "Value to count separately as a likely ignore/void label. It is STILL reported as a problem "
            "for your current trainer because CrossEntropyLoss() and DiceLoss do not ignore it. Use --ignore-index -1 "
            "if you do not want special counting for 255."
        ),
    )
    parser.add_argument(
        "--integer-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for deciding whether float labels are integer-like.",
    )
    parser.add_argument(
        "--preview-values",
        type=int,
        default=50,
        help="Maximum unique values to preview per record.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit for quick tests. 0 means scan all discovered files.",
    )
    parser.add_argument(
        "--max-problem-examples",
        type=int,
        default=10,
        help="Maximum examples printed/stored in summary.",
    )
    parser.add_argument(
        "--output_dir",
        default="label_check_reports",
        help="Directory for JSON and CSV reports.",
    )
    parser.add_argument(
        "--fail-on-problem",
        action="store_true",
        help="Exit with code 2 if any problem_for_current_training is found.",
    )

    # EndoVis-specific
    parser.add_argument(
        "--label_json",
        default="labels.json",
        help="EndoVis labels JSON path or filename relative to root_path.",
    )
    parser.add_argument(
        "--max-unknown-colors",
        type=int,
        default=20,
        help="Maximum unknown EndoVis RGB colors to preview per file.",
    )

    # Cataract-specific
    parser.add_argument(
        "--train_csv",
        default="train.csv",
        help="Cataract1k train CSV filename relative to root_path.",
    )
    parser.add_argument(
        "--test_csv",
        default="test.csv",
        help="Cataract1k test CSV filename relative to root_path.",
    )
    parser.add_argument(
        "--strict-cataract-unknown",
        action="store_true",
        help="Treat unknown Cataract classTitle values as problems instead of warnings.",
    )

    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    default_cfg = DEFAULT_DATASET_CONFIG[args.dataset]

    if args.root_path is None:
        args.root_path = default_cfg["root_path"]

    # argparse cannot distinguish omitted from empty string easily in a defaulted field.
    # For Synapse, use default list_dir unless the user explicitly passes --list_dir "".
    if args.dataset == "Synapse" and args.list_dir is None:
        args.list_dir = default_cfg.get("list_dir")
    elif args.list_dir == "":
        args.list_dir = None

    if args.max_files is not None and args.max_files <= 0:
        args.max_files = None

    # Let --ignore-index -1 mean "do not special-case an ignore index".
    if args.ignore_index is not None and int(args.ignore_index) < 0:
        args.ignore_index = None

    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = normalize_args(args)
    num_classes = resolve_num_classes(args)

    if num_classes <= 0:
        parser.error("--num_classes must be positive")

    scanner_map: Dict[str, Callable[[argparse.Namespace, int], List[Dict[str, Any]]]] = {
        "Synapse": scan_synapse,
        "ACDC": scan_acdc,
        "Cataract1k": scan_cataract,
        "EndoVis2018": scan_endovis,
    }

    records = scanner_map[args.dataset](args, num_classes)
    summary = summarize_records(
        records,
        dataset=args.dataset,
        root_path=args.root_path,
        num_classes=num_classes,
        args=args,
    )

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset = args.dataset.replace("/", "_")
    output_dir = Path(args.output_dir)
    report_json = output_dir / f"label_check_{safe_dataset}_{timestamp}.json"
    problem_csv = output_dir / f"label_check_{safe_dataset}_{timestamp}_problems.csv"

    payload = {
        "summary": summary,
        "records": records,
    }
    write_json(report_json, payload)
    write_problem_csv(problem_csv, records)
    print_summary(summary, report_json, problem_csv)

    if args.fail_on_problem and summary["problem_records_for_current_training"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
