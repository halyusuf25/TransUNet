#!/usr/bin/env python3
"""
Save ground-truth mask overlays for a dataset test split.


example of command-line usage:
    CUDA_VISIBLE_DEVICES=0 python tools/save_test_mask_overlays.py \
        --dataset ACDC \
        --volume_path /data/halyusuf/data/ACDC \
        --frame_per_file \
        --alpha 0.6 \
        --dpi 180 \
        --legend 
        
        
    CUDA_VISIBLE_DEVICES=2 python tools/save_test_mask_overlays.py \
        --dataset Synapse \
        --alpha 0.75 \
        --dpi 180 \
        --images_per_file 9 \
        --skip_empty_synapse_slices \
        --legend 


    CUDA_VISIBLE_DEVICES=3 python tools/save_test_mask_overlays.py \
        --dataset EndoVis2018 \
        --alpha 0.45 \
        --dpi 150 \
        --images_per_file 9 \
        --sample_limit 90 \
        --skip_empty_synapse_slices \
        --legend 


"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DATASET_CONFIG = {
    "Synapse": {
        "volume_path": "/data/halyusuf/data/Synapse/test_vol_h5",
        "list_dir": "./lists/lists_Synapse",
        "split": "test_vol",
        "num_classes": 9,
        "class_names": [
            "Background",
            "Aorta",
            "Gallbladder",
            "Kidney(L)",
            "Kidney(R)",
            "Liver",
            "Pancreas",
            "Spleen",
            "Stomach",
        ],
    },
    "ACDC": {
        "volume_path": "/data/halyusuf/data/ACDC/",
        "list_dir": None,
        "split": "test_vol",
        "num_classes": 4,
        "class_names": [
            "Background",
            "Right Ventricle",
            "Myocardium",
            "Left Ventricle",
        ],
    },
    "Cataract1k": {
        "volume_path": "/data/halyusuf/data/CataractData/",
        "list_dir": None,
        "split": "test",
        "num_classes": 5,
        "class_names": [
            "Background",
            "Pupil",
            "Cornea",
            "Lens",
            "Instruments",
        ],
    },
    "EndoVis2018": {
        "volume_path": "/data/halyusuf/data/EndoVis_2018",
        "list_dir": None,
        "split": "test",
        "num_classes": 12,
        "class_names": [
            "background-tissue",
            "instrument-shaft",
            "instrument-clasper",
            "instrument-wrist",
            "kidney-parenchyma",
            "covered-kidney",
            "thread",
            "clamps",
            "suturing-needle",
            "suction-instrument",
            "small-intestine",
            "ultrasound-probe",
        ],
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


class SynapseTestDataset:
    def __init__(self, base_dir: str, list_dir: str, split: str = "test_vol") -> None:
        self.base_dir = Path(base_dir)
        list_path = Path(list_dir) / "{}.txt".format(split)
        with open(list_path, "r") as f:
            self.sample_list = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        vol_name = self.sample_list[index]
        data_path = self.base_dir / "{}.npy.h5".format(vol_name)
        with h5py.File(data_path, "r") as data:
            image = data["image"][:]
            label = data["label"][:]
        return {"image": image, "label": label, "case_name": vol_name}


class ACDCTestDataset:
    def __init__(self, base_dir: str) -> None:
        self.volume_dir = Path(base_dir) / "ACDC_training_volumes"
        test_ids = ["patient{:0>3}".format(i) for i in range(1, 21)]
        all_volumes = sorted(name for name in os.listdir(self.volume_dir) if name.endswith(".h5"))
        self.sample_list = []
        for patient_id in test_ids:
            self.sample_list.extend(
                name for name in all_volumes if name.startswith(patient_id)
            )

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        case = self.sample_list[index]
        with h5py.File(self.volume_dir / case, "r") as data:
            image = data["image"][:]
            label = data["label"][:]
        return {"image": image, "label": label, "case_name": case.replace(".h5", "")}


class Cataract1kTestDataset:
    def __init__(self, base_dir: str, test_csv: str = "test.csv") -> None:
        self.base_dir = Path(base_dir)
        self.image_files = []
        self.annotation_files = []
        csv_path = self.base_dir / test_csv
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV has no header: {}".format(csv_path))
            image_column = "imgs" if "imgs" in reader.fieldnames else "img"
            if image_column not in reader.fieldnames:
                raise ValueError(
                    "Expected an 'imgs' or 'img' column in {}".format(csv_path)
                )
            for row in reader:
                file_name = os.path.basename(row[image_column])
                self.image_files.append(self.base_dir / "img" / file_name)
                self.annotation_files.append(self.base_dir / "ann" / "{}.json".format(file_name))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        image_path = self.image_files[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("Could not read image: {}".format(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = _cataract_annotation_to_mask(
            self.annotation_files[index],
            image.shape[:2],
        )
        return {
            "image": image,
            "label": label,
            "case_name": image_path.stem,
        }


class EndoVis2018TestDataset:
    def __init__(self, base_dir: str, label_json: str = "labels.json") -> None:
        self.base_dir = Path(base_dir)
        self.image_dir = self.base_dir / "test" / "imgs"
        self.label_dir = self.base_dir / "test" / "labels"
        label_path = self.base_dir / label_json
        self.color_to_class = _load_endovis_color_map(label_path)

        image_names = _list_png_names(self.image_dir)
        label_names = set(_list_png_names(self.label_dir))
        self.sample_list = [name for name in image_names if name in label_names]
        self.image_files = [self.image_dir / name for name in self.sample_list]
        self.label_files = [self.label_dir / name for name in self.sample_list]

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("Could not read image: {}".format(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = _load_endovis_label(label_path, self.color_to_class)
        return {
            "image": image,
            "label": label,
            "case_name": image_path.stem,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate through a dataset test set and save visualizations with "
            "the ground-truth segmentation mask overlaid on the original image."
        )
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        help="Dataset name. Supported values: {}".format(", ".join(DATASET_CONFIG)),
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_flag",
        help="Dataset name; equivalent to the positional dataset_name argument.",
    )
    parser.add_argument(
        "--volume_path",
        type=str,
        default=None,
        help="Override the dataset root path. Defaults to the repo's test.py paths.",
    )
    parser.add_argument(
        "--list_dir",
        type=str,
        default=None,
        help="Override list directory for datasets that use list files, e.g. Synapse.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_images",
        help="Output root directory. A dataset subdirectory is created inside it.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.50,
        help="Segmentation-mask overlay opacity in [0, 1].",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="DPI for saved matplotlib figures.",
    )
    parser.add_argument(
        "--images_per_file",
        type=int,
        default=1,
        help="Number of overlaid images/slices to place in each saved PNG.",
    )
    parser.add_argument(
        "--frame_per_file",
        action="store_true",
        help="For ACDC only, save each patient frame volume as one PNG containing its slices.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Optional maximum number of dataset samples to process.",
    )
    parser.add_argument(
        "--slice_limit",
        type=int,
        default=None,
        help="Optional maximum number of slices to process per 3D sample.",
    )
    parser.add_argument(
        "--skip_empty_synapse_slices",
        action="store_true",
        help="For Synapse only, skip slices whose ground-truth mask contains only background.",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Include a class colorbar legend in each saved visualization.",
    )
    args = parser.parse_args()

    requested = args.dataset_flag or args.dataset_name
    if requested is None:
        parser.error("Provide a dataset name, e.g. `ACDC` or `--dataset ACDC`.")
    if args.dataset_flag and args.dataset_name and args.dataset_flag != args.dataset_name:
        parser.error("Positional dataset_name and --dataset disagree.")
    args.dataset = _normalize_dataset_name(requested)

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1.")
    if args.frame_per_file and args.dataset != "ACDC":
        parser.error("--frame_per_file can only be used with --dataset ACDC.")
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    if args.images_per_file < 1:
        parser.error("--images_per_file must be at least 1.")
    if args.sample_limit is not None and args.sample_limit < 1:
        parser.error("--sample_limit must be positive when provided.")
    if args.slice_limit is not None and args.slice_limit < 1:
        parser.error("--slice_limit must be positive when provided.")

    return args


def _normalize_dataset_name(name: str) -> str:
    aliases = {dataset.lower(): dataset for dataset in DATASET_CONFIG}
    key = str(name).lower()
    if key not in aliases:
        raise SystemExit(
            "Unsupported dataset '{}'. Supported values: {}".format(
                name,
                ", ".join(DATASET_CONFIG),
            )
        )
    return aliases[key]


def _build_dataset(dataset_name: str, volume_path: Optional[str], list_dir: Optional[str]):
    config = DATASET_CONFIG[dataset_name]
    base_dir = volume_path or config["volume_path"]
    if dataset_name == "Synapse":
        resolved_list_dir = list_dir if list_dir is not None else config["list_dir"]
        return SynapseTestDataset(
            base_dir=base_dir,
            list_dir=resolved_list_dir,
            split=config["split"],
        )
    if dataset_name == "ACDC":
        return ACDCTestDataset(base_dir=base_dir)
    if dataset_name == "Cataract1k":
        return Cataract1kTestDataset(base_dir=base_dir)
    if dataset_name == "EndoVis2018":
        return EndoVis2018TestDataset(base_dir=base_dir)
    raise ValueError("Unsupported dataset: {}".format(dataset_name))


def _cataract_annotation_to_mask(annotation_path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    with open(annotation_path, "r") as f:
        annotation = json.load(f)

    for obj in annotation.get("objects", []):
        class_title = obj.get("classTitle")
        if class_title in CATARACT_CLASS_MAP:
            class_id = CATARACT_CLASS_MAP[class_title]
        elif class_title in CATARACT_INSTRUMENTS:
            class_id = CATARACT_CLASS_MAP["Instruments"]
        else:
            continue

        exterior_points = obj.get("points", {}).get("exterior", [])
        if len(exterior_points) < 3:
            continue
        points = np.asarray(exterior_points, dtype=np.int32)
        cv2.fillPoly(mask, [points], int(class_id))

    return mask


def _list_png_names(directory: Path) -> List[str]:
    if not directory.is_dir():
        raise FileNotFoundError("Directory not found: {}".format(directory))
    return sorted(
        name
        for name in os.listdir(directory)
        if name.lower().endswith(".png")
    )


def _load_endovis_color_map(label_json_path: Path) -> Dict[Tuple[int, int, int], int]:
    with open(label_json_path, "r") as f:
        labels = json.load(f)

    color_to_class = {}
    for item in labels:
        color = tuple(int(value) for value in item["color"])
        class_id = int(item["classid"])
        color_to_class[color] = class_id
    return color_to_class


def _load_endovis_label(
    label_path: Path,
    color_to_class: Dict[Tuple[int, int, int], int],
) -> np.ndarray:
    label_image = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label_image is None:
        raise FileNotFoundError("Could not read label: {}".format(label_path))

    if label_image.ndim == 2:
        return label_image.astype(np.uint8)

    if label_image.shape[2] == 4:
        label_rgb = cv2.cvtColor(label_image, cv2.COLOR_BGRA2RGB)
    else:
        label_rgb = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

    mask = np.full(label_rgb.shape[:2], 255, dtype=np.uint8)
    for color, class_id in color_to_class.items():
        color_arr = np.asarray(color, dtype=np.uint8)
        matches = np.all(label_rgb == color_arr, axis=-1)
        mask[matches] = class_id

    unknown = mask == 255
    if np.any(unknown):
        unknown_colors, counts = np.unique(
            label_rgb[unknown].reshape(-1, 3),
            axis=0,
            return_counts=True,
        )
        order = np.argsort(counts)[::-1]
        preview = [
            {
                "rgb": unknown_colors[i].tolist(),
                "count": int(counts[i]),
            }
            for i in order[:20]
        ]
        raise ValueError(
            "Unknown RGB colors found in {}. Most frequent unknown colors: {}".format(
                label_path,
                preview,
            )
        )

    return mask


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_grayscale(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _normalize_color(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    v_min = float(np.min(finite))
    v_max = float(np.max(finite))
    if v_min >= 0.0 and v_max <= 1.0:
        return np.clip(arr, 0.0, 1.0)
    if v_min >= 0.0 and v_max <= 255.0:
        return np.clip(arr / 255.0, 0.0, 1.0)

    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _prepare_display_image(image) -> np.ndarray:
    arr = np.squeeze(_to_numpy(image))

    if arr.ndim == 2:
        gray = _normalize_grayscale(arr)
        return np.repeat(gray[..., None], 3, axis=2)

    if arr.ndim != 3:
        raise ValueError("Expected a 2D image or 3D color image, got shape {}".format(arr.shape))

    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        gray = _normalize_grayscale(arr[..., 0])
        return np.repeat(gray[..., None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        gray = _normalize_grayscale(arr[..., 0])
        return np.repeat(gray[..., None], 3, axis=2)

    return _normalize_color(arr)


def _prepare_label_map(label, target_shape: Tuple[int, int]) -> np.ndarray:
    arr = np.squeeze(_to_numpy(label))

    while arr.ndim > 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] <= 32 and arr.shape[0] > 32 and arr.shape[1] > 32:
            arr = np.argmax(arr, axis=-1)
        elif arr.shape[0] <= 32 and arr.shape[1] > 32 and arr.shape[2] > 32:
            arr = np.argmax(arr, axis=0)
        else:
            arr = arr[0]

    if arr.ndim != 2:
        raise ValueError("Expected a 2D label map, got shape {}".format(arr.shape))

    target_h, target_w = target_shape
    if arr.shape != (target_h, target_w):
        arr = cv2.resize(
            arr.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
    return arr.astype(np.int32, copy=False)


def _class_count(label: np.ndarray, configured_classes: int) -> int:
    if label.size == 0:
        return configured_classes
    label_max = int(np.nanmax(label))
    return max(configured_classes, label_max + 1, 2)


def _class_labels(class_names: List[str], num_classes: int) -> List[str]:
    labels = list(class_names)
    while len(labels) < num_classes:
        labels.append("class_{}".format(len(labels)))
    return labels[:num_classes]


def _discrete_cmap(num_classes: int) -> Tuple[ListedColormap, BoundaryNorm]:
    base_cmap = plt.get_cmap("tab20") if num_classes - 1 > 10 else plt.get_cmap("tab10")
    colors = [(0.0, 0.0, 0.0, 1.0)]
    for class_id in range(1, num_classes):
        color = base_cmap((class_id - 1) % base_cmap.N)
        colors.append((color[0], color[1], color[2], 1.0))

    cmap = ListedColormap(colors, N=num_classes)
    bounds = np.arange(-0.5, num_classes + 0.5, 1)
    return cmap, BoundaryNorm(bounds, cmap.N)


def _prepare_overlay_panel(image, label) -> Tuple[np.ndarray, np.ndarray]:
    image_rgb = _prepare_display_image(image)
    label_map = _prepare_label_map(label, image_rgb.shape[:2])
    return image_rgb, label_map


def _draw_overlay_panel(
    ax,
    image_rgb: np.ndarray,
    label_map: np.ndarray,
    title: str,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    alpha: float,
):
    ax.imshow(image_rgb, interpolation="nearest")
    visible_mask = np.ma.masked_where(label_map == 0, label_map)
    overlay = ax.imshow(
        visible_mask.astype(np.int32),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        alpha=alpha,
    )
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return overlay


def _save_overlay_group(
    items: List[Tuple[object, object, str]],
    output_path: Path,
    configured_classes: int,
    configured_labels: List[str],
    alpha: float,
    dpi: int,
    legend: bool,
) -> None:
    if not items:
        return

    panels = [
        (_prepare_overlay_panel(image, label), title)
        for image, label, title in items
    ]
    label_maps = [panel[0][1] for panel in panels]
    num_classes = max(
        _class_count(label_map, configured_classes)
        for label_map in label_maps
    )
    cmap, norm = _discrete_cmap(num_classes)

    panel_shapes = [panel[0][0].shape[:2] for panel in panels]
    max_h = max(shape[0] for shape in panel_shapes)
    max_w = max(shape[1] for shape in panel_shapes)

    item_count = len(items)
    cols = int(math.ceil(math.sqrt(item_count)))
    rows = int(math.ceil(item_count / float(cols)))
    panel_w = max(4.0, max_w / float(dpi))
    panel_h = max(4.0, max_h / float(dpi))
    fig_w = panel_w * cols
    fig_h = panel_h * rows
    if legend:
        fig_w += 1.5

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        constrained_layout=True,
        squeeze=False,
    )

    overlay = None
    for index, ((image_rgb, label_map), title) in enumerate(panels):
        row = index // cols
        col = index % cols
        overlay = _draw_overlay_panel(
            axes[row][col],
            image_rgb,
            label_map,
            title,
            cmap,
            norm,
            alpha,
        )

    for empty_index in range(item_count, rows * cols):
        row = empty_index // cols
        col = empty_index % cols
        axes[row][col].axis("off")

    if legend and overlay is not None:
        labels = _class_labels(configured_labels, num_classes)
        colorbar = fig.colorbar(
            overlay,
            ax=axes.ravel().tolist(),
            ticks=np.arange(0, num_classes, 1),
            fraction=0.046,
            pad=0.02,
        )
        colorbar.ax.set_yticklabels(labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _save_overlay(
    image,
    label,
    title: str,
    output_path: Path,
    configured_classes: int,
    configured_labels: List[str],
    alpha: float,
    dpi: int,
    legend: bool,
) -> None:
    _save_overlay_group(
        [(image, label, title)],
        output_path,
        configured_classes,
        configured_labels,
        alpha,
        dpi,
        legend,
    )


def _safe_stem(name: object) -> str:
    stem = str(name)
    stem = stem.replace(os.sep, "_").replace("/", "_")
    return Path(stem).stem if Path(stem).suffix else stem


def _sample_case_name(sample: Dict, index: int) -> str:
    case_name = sample.get("case_name", "sample_{:04d}".format(index))
    if isinstance(case_name, (list, tuple)):
        case_name = case_name[0]
    return _safe_stem(case_name)


def _label_has_foreground(label) -> bool:
    return bool(np.any(np.asarray(label) != 0))


def _group_output_path(output_dir: Path, group_index: int, items: List[Tuple[object, object, str]]) -> Path:
    first_title = _safe_stem(items[0][2])
    last_title = _safe_stem(items[-1][2])
    if first_title == last_title:
        filename = "group_{:04d}_{}.png".format(group_index, first_title)
    else:
        filename = "group_{:04d}_{}_to_{}.png".format(
            group_index,
            first_title,
            last_title,
        )
    return output_dir / filename


def _is_volume_sample(image: np.ndarray, label: np.ndarray) -> bool:
    label_arr = np.squeeze(label)
    image_arr = np.squeeze(image)
    if label_arr.ndim >= 3:
        return True
    if image_arr.ndim == 3 and label_arr.ndim == 3:
        return True
    return False


def _iter_sample_slices(image: np.ndarray, label: np.ndarray) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    image_arr = np.squeeze(image)
    label_arr = np.squeeze(label)
    if label_arr.ndim != 3:
        raise ValueError("Expected a 3D label volume, got shape {}".format(label_arr.shape))

    depth = label_arr.shape[0]
    if image_arr.ndim == 3 and image_arr.shape[0] == depth:
        for slice_idx in range(depth):
            yield slice_idx, image_arr[slice_idx], label_arr[slice_idx]
    elif image_arr.ndim == 4 and image_arr.shape[0] == depth:
        for slice_idx in range(depth):
            yield slice_idx, image_arr[slice_idx], label_arr[slice_idx]
    else:
        raise ValueError(
            "Could not align image volume shape {} with label volume shape {}".format(
                image_arr.shape,
                label_arr.shape,
            )
        )


def main() -> None:
    args = _parse_args()
    dataset = _build_dataset(args.dataset, args.volume_path, args.list_dir)
    output_dir = Path(args.output_dir) / args.dataset
    config = DATASET_CONFIG[args.dataset]
    configured_classes = int(config["num_classes"])
    configured_labels = list(config.get("class_names", []))

    total_samples = len(dataset)
    sample_count = min(total_samples, args.sample_limit) if args.sample_limit else total_samples
    saved_count = 0
    file_count = 0
    skipped_count = 0
    pending_items = []

    def flush_pending_items() -> None:
        nonlocal file_count, pending_items, saved_count
        if not pending_items:
            return
        output_path = _group_output_path(output_dir, file_count, pending_items)
        _save_overlay_group(
            pending_items,
            output_path,
            configured_classes,
            configured_labels,
            args.alpha,
            args.dpi,
            args.legend,
        )
        saved_count += len(pending_items)
        file_count += 1
        pending_items = []

    def save_or_queue_overlay(image, label, title: str, output_path: Path) -> None:
        nonlocal file_count, saved_count
        if args.images_per_file == 1:
            _save_overlay(
                image,
                label,
                title,
                output_path,
                configured_classes,
                configured_labels,
                args.alpha,
                args.dpi,
                args.legend,
            )
            saved_count += 1
            file_count += 1
            return

        pending_items.append((image, label, title))
        if len(pending_items) >= args.images_per_file:
            flush_pending_items()

    for sample_idx in tqdm(range(sample_count), desc="Saving overlays"):
        sample = dataset[sample_idx]
        image = _to_numpy(sample["image"])
        label = _to_numpy(sample["label"])
        case_name = _sample_case_name(sample, sample_idx)

        if _is_volume_sample(image, label):
            if args.dataset == "ACDC" and args.frame_per_file:
                frame_items = []
                for slice_counter, (slice_idx, image_slice, label_slice) in enumerate(
                    _iter_sample_slices(image, label)
                ):
                    if args.slice_limit is not None and slice_counter >= args.slice_limit:
                        break
                    slide_name = "{}_slice_{:03d}".format(case_name, slice_idx)
                    frame_items.append((image_slice, label_slice, slide_name))

                if frame_items:
                    _save_overlay_group(
                        frame_items,
                        output_dir / "{}.png".format(case_name),
                        configured_classes,
                        configured_labels,
                        args.alpha,
                        args.dpi,
                        args.legend,
                    )
                    saved_count += len(frame_items)
                    file_count += 1
                continue

            for slice_counter, (slice_idx, image_slice, label_slice) in enumerate(
                _iter_sample_slices(image, label)
            ):
                if args.slice_limit is not None and slice_counter >= args.slice_limit:
                    break
                if (
                    args.dataset == "Synapse"
                    and args.skip_empty_synapse_slices
                    and not _label_has_foreground(label_slice)
                ):
                    skipped_count += 1
                    continue
                slide_name = "{}_slice_{:03d}".format(case_name, slice_idx)
                output_path = output_dir / "{}.png".format(slide_name)
                save_or_queue_overlay(
                    image_slice,
                    label_slice,
                    slide_name,
                    output_path,
                )
        else:
            if (
                args.dataset == "Synapse"
                and args.skip_empty_synapse_slices
                and not _label_has_foreground(label)
            ):
                skipped_count += 1
                continue
            output_path = output_dir / "{}.png".format(case_name)
            save_or_queue_overlay(
                image,
                label,
                case_name,
                output_path,
            )

    flush_pending_items()

    print(
        "Saved {} overlay image(s) in {} file(s) for {} to {}".format(
            saved_count,
            file_count,
            args.dataset,
            output_dir,
        )
    )
    if skipped_count:
        print("Skipped {} empty Synapse slice(s).".format(skipped_count))


if __name__ == "__main__":
    main()
