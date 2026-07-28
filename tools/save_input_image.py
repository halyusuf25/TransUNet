#!/usr/bin/env python3
"""Save one original dataset image (or volume slice) as a PNG.

Examples:
    python tools/save_input_image.py \
        --dataset Synapse --image-index 8 --slice-index 115 \
        --include-ground-truth

    python tools/save_input_image.py \
        --dataset ACDC --case-name patient001_frame01 --slice-index 4

    python tools/save_input_image.py \
        --dataset Cataract1k --image-index 490

The output directory defaults to ``viz/input_images`` at the repository root.
The script intentionally loads samples without augmentation, resizing, or model
normalization. RGB uint8 images are written without intensity conversion. Other
numeric images are converted to uint8 only because PNG cannot store arbitrary
floating-point medical-image intensities.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]


DATASET_CONFIG = {
    "Synapse": {
        "volume_path": "/data/halyusuf/data/Synapse/test_vol_h5",
        "list_dir": str(ROOT_DIR / "lists" / "lists_Synapse"),
        "is_volume": True,
        "num_classes": 9,
    },
    "ACDC": {
        "volume_path": "/data/halyusuf/data/ACDC/",
        "list_dir": None,
        "is_volume": True,
        "num_classes": 4,
    },
    "Cataract1k": {
        "volume_path": "/data/halyusuf/data/CataractData/",
        "list_dir": None,
        "is_volume": False,
        "num_classes": 5,
    },
    "EndoVis2018": {
        "volume_path": "/data/halyusuf/data/EndoVis_2018",
        "list_dir": None,
        "is_volume": False,
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

# RGB values from Matplotlib's tab10/tab20 palettes, matching the repository's
# other segmentation visualizations. Background (class 0) is not overlaid.
TAB10_COLORS = np.asarray(
    [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ],
    dtype=np.uint8,
)

TAB20_COLORS = np.asarray(
    [
        (31, 119, 180),
        (174, 199, 232),
        (255, 127, 14),
        (255, 187, 120),
        (44, 160, 44),
        (152, 223, 138),
        (214, 39, 40),
        (255, 152, 150),
        (148, 103, 189),
        (197, 176, 213),
        (140, 86, 75),
        (196, 156, 148),
        (227, 119, 194),
        (247, 182, 210),
        (127, 127, 127),
        (199, 199, 199),
        (188, 189, 34),
        (219, 219, 141),
        (23, 190, 207),
        (158, 218, 229),
    ],
    dtype=np.uint8,
)

GROUND_TRUTH_ALPHA = 0.5
GROUND_TRUTH_CONTOUR_COLOR = (255, 255, 0)  # RGB yellow
GROUND_TRUTH_CONTOUR_THICKNESS = 1
GROUND_TRUTH_DASH_PIXELS = 6
GROUND_TRUTH_GAP_PIXELS = 4


class OriginalImageDataset:
    """Small read-only dataset that mirrors the visualization test-set ordering."""

    def __init__(self, dataset_name: str, base_dir: str, list_dir: Optional[str]) -> None:
        self.dataset_name = dataset_name
        self.base_dir = Path(base_dir)
        self.sample_list = []
        self.image_files = []
        self.label_files = []

        if dataset_name == "Synapse":
            self._index_synapse(list_dir)
        elif dataset_name == "ACDC":
            self._index_acdc()
        elif dataset_name == "Cataract1k":
            self._index_cataract()
        elif dataset_name == "EndoVis2018":
            self._index_endovis()
        else:
            raise ValueError("Unsupported dataset: {}.".format(dataset_name))

    def _index_synapse(self, list_dir: Optional[str]) -> None:
        if list_dir is None:
            raise ValueError("A list directory is required for Synapse.")
        list_path = Path(list_dir) / "test_vol.txt"
        with list_path.open("r") as handle:
            self.sample_list = [line.strip() for line in handle if line.strip()]
        self.image_files = [
            self.base_dir / "{}.npy.h5".format(case_name)
            for case_name in self.sample_list
        ]
        self.label_files = list(self.image_files)

    def _index_acdc(self) -> None:
        volume_dir = self.base_dir / "ACDC_training_volumes"
        volume_names = [name for name in os.listdir(volume_dir) if name.endswith(".h5")]
        for patient_id in ("patient{:0>3}".format(i) for i in range(1, 21)):
            self.sample_list.extend(
                name for name in volume_names if name.startswith(patient_id)
            )
        self.image_files = [volume_dir / name for name in self.sample_list]
        self.label_files = list(self.image_files)

    def _index_cataract(self) -> None:
        csv_path = self.base_dir / "test.csv"
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            image_column = "imgs" if "imgs" in fieldnames else "img"
            if image_column not in fieldnames:
                raise ValueError(
                    "Expected an 'imgs' or 'img' column in {}.".format(csv_path)
                )
            for row in reader:
                filename = Path(row[image_column]).name
                self.sample_list.append(filename)
                self.image_files.append(self.base_dir / "img" / filename)
                self.label_files.append(
                    self.base_dir / "ann" / "{}.json".format(filename)
                )

    def _index_endovis(self) -> None:
        image_dir = self.base_dir / "test" / "imgs"
        label_dir = self.base_dir / "test" / "labels"
        image_names = sorted(
            path.name for path in image_dir.iterdir() if path.suffix.lower() == ".png"
        )
        label_names = {
            path.name for path in label_dir.iterdir() if path.suffix.lower() == ".png"
        }
        self.sample_list = [name for name in image_names if name in label_names]
        self.image_files = [image_dir / name for name in self.sample_list]
        self.label_files = [label_dir / name for name in self.sample_list]

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, index: int):
        case_name = _case_key(self.sample_list[index])
        image_path = self.image_files[index]
        if self.dataset_name in ("Synapse", "ACDC"):
            with h5py.File(image_path, "r") as data:
                image = data["image"][:]
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError("Could not read image: {}.".format(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {"image": image, "case_name": case_name}

    def load_label(self, index: int, image_shape: Tuple[int, ...]) -> np.ndarray:
        label_path = self.label_files[index]
        if self.dataset_name in ("Synapse", "ACDC"):
            with h5py.File(label_path, "r") as data:
                return data["label"][:]
        if self.dataset_name == "Cataract1k":
            return _load_cataract_label(label_path, image_shape[:2])
        if self.dataset_name == "EndoVis2018":
            return _load_endovis_label(label_path, self.base_dir / "labels.json")
        raise ValueError("Unsupported dataset: {}.".format(self.dataset_name))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save one unmodified dataset image, or one slice from a 3D sample, "
            "under viz/input_images/."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name: {}.".format(", ".join(DATASET_CONFIG)),
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--image-index",
        "--index",
        "--viz_index",
        dest="image_index",
        type=int,
        help="Zero-based dataset sample index (same meaning as tools/visualize.py --viz_index).",
    )
    selection.add_argument(
        "--case-name",
        type=str,
        help="Select a sample by case name instead of its numeric index.",
    )

    parser.add_argument(
        "--slice-index",
        "--slice",
        "--viz_slice",
        dest="slice_index",
        type=int,
        default=None,
        help=(
            "Zero-based slice index for Synapse or ACDC. If omitted for a volume, "
            "the middle slice is saved."
        ),
    )
    parser.add_argument(
        "--volume-path",
        "--volume_path",
        dest="volume_path",
        type=str,
        default=None,
        help="Override the configured dataset root directory.",
    )
    parser.add_argument(
        "--list-dir",
        "--list_dir",
        dest="list_dir",
        type=str,
        default=None,
        help="Override the Synapse list directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "viz" / "input_images",
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output filename. A .png suffix is added when missing.",
    )
    parser.add_argument(
        "--include-ground-truth",
        "--include_ground_truth",
        action="store_true",
        help=(
            "Overlay non-background ground-truth segmentation classes and draw "
            "yellow dashed contour lines on the saved input image."
        ),
    )
    return parser.parse_args()


def _normalize_dataset_name(value: str) -> str:
    aliases = {name.lower(): name for name in DATASET_CONFIG}
    key = value.strip().lower()
    if key not in aliases:
        raise ValueError(
            "Unsupported dataset '{}'. Choose one of: {}.".format(
                value,
                ", ".join(DATASET_CONFIG),
            )
        )
    return aliases[key]


def _build_dataset(dataset_name: str, volume_path: Optional[str], list_dir: Optional[str]):
    config = DATASET_CONFIG[dataset_name]
    base_dir = volume_path or config["volume_path"]
    resolved_list_dir = list_dir or config["list_dir"]
    return OriginalImageDataset(dataset_name, base_dir, resolved_list_dir)


def _load_cataract_label(
    annotation_path: Path,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    with annotation_path.open("r") as handle:
        annotation = json.load(handle)

    mask = np.zeros(image_shape, dtype=np.uint8)
    for obj in annotation.get("objects", []):
        class_title = obj.get("classTitle")
        if class_title in CATARACT_CLASS_MAP:
            class_id = CATARACT_CLASS_MAP[class_title]
        elif class_title in CATARACT_INSTRUMENTS:
            class_id = CATARACT_CLASS_MAP["Instruments"]
        else:
            continue

        exterior = obj.get("points", {}).get("exterior", [])
        if len(exterior) >= 3:
            cv2.fillPoly(
                mask,
                [np.asarray(exterior, dtype=np.int32)],
                int(class_id),
            )
    return mask


def _load_endovis_label(label_path: Path, labels_json_path: Path) -> np.ndarray:
    with labels_json_path.open("r") as handle:
        label_definitions = json.load(handle)
    color_to_class = {
        tuple(int(value) for value in item["color"]): int(item["classid"])
        for item in label_definitions
    }

    label_image = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label_image is None:
        raise FileNotFoundError("Could not read label: {}.".format(label_path))
    if label_image.ndim == 2:
        return label_image.astype(np.uint8)
    if label_image.shape[2] == 4:
        label_rgb = cv2.cvtColor(label_image, cv2.COLOR_BGRA2RGB)
    else:
        label_rgb = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

    mask = np.full(label_rgb.shape[:2], 255, dtype=np.uint8)
    for color, class_id in color_to_class.items():
        matches = np.all(label_rgb == np.asarray(color, dtype=np.uint8), axis=-1)
        mask[matches] = class_id
    if np.any(mask == 255):
        unknown_count = int(np.count_nonzero(mask == 255))
        raise ValueError(
            "Found {} pixels with unknown colors in {}.".format(
                unknown_count,
                label_path,
            )
        )
    return mask


def _case_key(value: object) -> str:
    name = Path(str(value).strip()).name
    if name.lower().endswith(".h5"):
        name = name[:-3]
    else:
        name = Path(name).stem
    return name.lower()


def _available_case_names(dataset) -> Sequence[object]:
    if hasattr(dataset, "sample_list"):
        return dataset.sample_list
    if hasattr(dataset, "image_files"):
        return dataset.image_files
    return ()


def _resolve_sample_index(dataset, image_index: Optional[int], case_name: Optional[str]) -> int:
    total = len(dataset)
    if total == 0:
        raise ValueError("The selected dataset split contains no samples.")

    if image_index is not None:
        if image_index < 0 or image_index >= total:
            raise IndexError(
                "--image-index must be in [0, {}], got {}.".format(total - 1, image_index)
            )
        return image_index

    requested_key = _case_key(case_name)
    matches = [
        index
        for index, candidate in enumerate(_available_case_names(dataset))
        if _case_key(candidate) == requested_key
    ]
    if not matches:
        raise ValueError("Case '{}' was not found in the selected dataset split.".format(case_name))
    if len(matches) > 1:
        raise ValueError(
            "Case name '{}' is ambiguous; use --image-index instead.".format(case_name)
        )
    return matches[0]


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _select_image(
    image: np.ndarray,
    dataset_name: str,
    requested_slice: Optional[int],
) -> Tuple[np.ndarray, Optional[int]]:
    image = np.squeeze(image)
    is_volume = bool(DATASET_CONFIG[dataset_name]["is_volume"])

    if not is_volume:
        if requested_slice is not None:
            raise ValueError(
                "--slice-index is only valid for Synapse and ACDC; {} contains 2D frames.".format(
                    dataset_name
                )
            )
        if image.ndim not in (2, 3):
            raise ValueError("Expected a 2D image or HWC image, got shape {}.".format(image.shape))
        return image, None

    if image.ndim != 3:
        raise ValueError(
            "Expected a depth-first 3D {} volume, got shape {}.".format(dataset_name, image.shape)
        )
    depth = image.shape[0]
    slice_index = depth // 2 if requested_slice is None else requested_slice
    if slice_index < 0 or slice_index >= depth:
        raise IndexError(
            "--slice-index must be in [0, {}] for this volume, got {}.".format(
                depth - 1,
                slice_index,
            )
        )
    return image[slice_index], slice_index


def _to_png_uint8(image: np.ndarray) -> Tuple[np.ndarray, str]:
    if image.dtype == np.uint8:
        return image, "none (already uint8)"

    values = image.astype(np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("The selected image contains no finite pixel values.")

    finite_values = values[finite]
    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    values = np.nan_to_num(values, nan=minimum, posinf=maximum, neginf=minimum)

    if minimum >= 0.0 and maximum <= 1.0:
        values = values * 255.0
        conversion = "[0, 1] to [0, 255]"
    elif minimum >= 0.0 and maximum <= 255.0:
        conversion = "cast from [{:.6g}, {:.6g}] to uint8".format(minimum, maximum)
    elif maximum > minimum:
        values = (values - minimum) * (255.0 / (maximum - minimum))
        conversion = "min-max [{:.6g}, {:.6g}] to [0, 255]".format(minimum, maximum)
    else:
        values = np.zeros_like(values)
        conversion = "constant value {:.6g} to 0".format(minimum)

    return np.clip(np.rint(values), 0, 255).astype(np.uint8), conversion


def _overlay_ground_truth(
    image: np.ndarray,
    label: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    label = np.squeeze(np.asarray(label))
    if label.ndim != 2:
        raise ValueError("Expected a 2D ground-truth mask, got shape {}.".format(label.shape))
    if image.shape[:2] != label.shape:
        raise ValueError(
            "Image shape {} and ground-truth shape {} do not match.".format(
                image.shape[:2],
                label.shape,
            )
        )
    if not np.all(np.isfinite(label)):
        raise ValueError("The ground-truth mask contains non-finite values.")

    label = np.rint(label).astype(np.int32)
    if np.any(label < 0):
        raise ValueError("The ground-truth mask contains negative class IDs.")
    if np.any(label >= num_classes):
        invalid_ids = np.unique(label[label >= num_classes]).tolist()
        raise ValueError(
            "Ground-truth class IDs {} exceed the configured {} classes.".format(
                invalid_ids,
                num_classes,
            )
        )

    if image.ndim == 2:
        image_rgb = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 3:
        image_rgb = image
    elif image.ndim == 3 and image.shape[2] == 4:
        image_rgb = image[..., :3]
    else:
        raise ValueError("Could not overlay a mask on image shape {}.".format(image.shape))

    palette = TAB20_COLORS if num_classes - 1 > 10 else TAB10_COLORS
    foreground = label > 0
    class_colors = palette[(np.maximum(label, 1) - 1) % len(palette)]
    result = image_rgb.astype(np.float32)
    result[foreground] = (
        (1.0 - GROUND_TRUTH_ALPHA) * result[foreground]
        + GROUND_TRUTH_ALPHA * class_colors[foreground]
    )
    result = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    _draw_ground_truth_contours(result, label)
    return result


def _draw_ground_truth_contours(image: np.ndarray, label: np.ndarray) -> None:
    """Draw the existing yellow dashed per-class boundary style in-place."""
    for class_id in np.unique(label):
        if class_id == 0:
            continue
        class_mask = (label == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            class_mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )
        for contour in contours:
            points = contour.reshape(-1, 2)
            if len(points) < 2:
                continue
            points = np.concatenate((points, points[:1]), axis=0)
            step = GROUND_TRUTH_DASH_PIXELS + GROUND_TRUTH_GAP_PIXELS
            for start in range(0, len(points) - 1, step):
                dash = points[start:min(start + GROUND_TRUTH_DASH_PIXELS + 1, len(points))]
                if len(dash) >= 2:
                    cv2.polylines(
                        image,
                        [dash.reshape(-1, 1, 2)],
                        isClosed=False,
                        color=GROUND_TRUTH_CONTOUR_COLOR,
                        thickness=GROUND_TRUTH_CONTOUR_THICKNESS,
                        lineType=cv2.LINE_AA,
                    )


def _safe_filename_component(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("._") or "sample"


def _output_path(
    output_dir: Path,
    output_name: Optional[str],
    dataset_name: str,
    case_name: object,
    slice_index: Optional[int],
    include_ground_truth: bool,
) -> Path:
    if output_name:
        filename = Path(output_name).name
        if Path(filename).suffix.lower() != ".png":
            filename += ".png"
    else:
        parts = [dataset_name, _safe_filename_component(case_name)]
        if slice_index is not None:
            parts.append("slice_{:03d}".format(slice_index))
        if include_ground_truth:
            parts.append("ground_truth")
        filename = "_".join(parts) + ".png"
    return output_dir / filename


def _write_png(path: Path, image: np.ndarray) -> None:
    if image.ndim == 3:
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        else:
            raise ValueError("Expected 3 or 4 image channels, got shape {}.".format(image.shape))

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError("OpenCV failed to save {}.".format(path))


def main() -> None:
    args = _parse_args()
    dataset_name = _normalize_dataset_name(args.dataset)
    dataset = _build_dataset(dataset_name, args.volume_path, args.list_dir)
    sample_index = _resolve_sample_index(dataset, args.image_index, args.case_name)
    sample = dataset[sample_index]

    original = _to_numpy(sample["image"])
    selected, slice_index = _select_image(original, dataset_name, args.slice_index)
    png_image, conversion = _to_png_uint8(selected)
    if args.include_ground_truth:
        original_label = dataset.load_label(sample_index, tuple(original.shape))
        selected_label, _ = _select_image(
            _to_numpy(original_label),
            dataset_name,
            slice_index,
        )
        png_image = _overlay_ground_truth(
            png_image,
            selected_label,
            int(DATASET_CONFIG[dataset_name]["num_classes"]),
        )
    case_name = sample.get("case_name", "sample_{:04d}".format(sample_index))
    output_path = _output_path(
        args.output_dir,
        args.output_name,
        dataset_name,
        case_name,
        slice_index,
        args.include_ground_truth,
    )
    _write_png(output_path, png_image)

    description = "input image with ground-truth overlay" if args.include_ground_truth else "original input image"
    print("Saved {} to {}".format(description, output_path))
    print(
        "Dataset: {} | sample index: {} | case: {} | source shape: {} | PNG conversion: {}".format(
            dataset_name,
            sample_index,
            case_name,
            tuple(original.shape),
            conversion,
        )
    )
    if DATASET_CONFIG[dataset_name]["is_volume"]:
        print("Slice index: {} | saved shape: {}".format(slice_index, tuple(selected.shape)))


if __name__ == "__main__":
    main()
