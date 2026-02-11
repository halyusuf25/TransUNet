#!/usr/bin/env python3
import argparse
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from visualize import _discrete_cmap as _visualize_discrete_cmap
    from visualize import _draw_gt_boundaries as _visualize_draw_gt_boundaries
except Exception:
    _visualize_discrete_cmap = None
    _visualize_draw_gt_boundaries = None


FILE_PATTERN = re.compile(
    r"^epoch_(?P<epoch>\d+)_step_(?P<step>\d+)_sample_(?P<sample>.+)\.(?P<ext>png|npy)$"
)
CASE_SLICE_PATTERN = re.compile(r"(?P<case>case\d+)[_-]?(?P<slice>slice\d+)", re.IGNORECASE)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Create qualitative epoch-comparison panels from saved heatmap overlays."
    )
    parser.add_argument(
        "--heatmaps_dir",
        type=str,
        required=True,
        help="Directory containing saved heatmap PNG/NPY files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="qualitative",
        help="Output directory for the comparison figure.",
    )
    return parser.parse_args()


def _discrete_cmap(n_classes: int):
    if _visualize_discrete_cmap is not None:
        cmap, norm, _ = _visualize_discrete_cmap(n_classes)
        return cmap, norm

    from matplotlib.colors import BoundaryNorm, ListedColormap

    base_cmap = plt.get_cmap("tab20") if n_classes - 1 > 10 else plt.get_cmap("tab10")
    colors = [(0.0, 0.0, 0.0, 1.0)]
    for i in range(1, n_classes):
        c = base_cmap((i - 1) % base_cmap.N)
        colors.append((c[0], c[1], c[2], 1.0))
    cmap = ListedColormap(colors, N=n_classes)
    bounds = np.arange(-0.5, n_classes + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _draw_gt_boundaries(ax, gt: np.ndarray):
    if _visualize_draw_gt_boundaries is not None:
        _visualize_draw_gt_boundaries(ax, gt, color="white", linewidth=1.0, linestyle=":")
        return

    gt_arr = np.asarray(gt)
    if gt_arr.ndim != 2:
        gt_arr = np.squeeze(gt_arr)
    if gt_arr.ndim != 2:
        return
    classes = np.unique(gt_arr.astype(int))
    classes = classes[classes != 0]
    for cls in classes:
        mask = gt_arr == cls
        if mask.any():
            ax.contour(
                mask.astype(float),
                levels=[0.5],
                colors="white",
                linewidths=1.0,
                linestyles=":",
            )


def _parse_case_slice(sample_name: str) -> Optional[Tuple[str, str]]:
    match = CASE_SLICE_PATTERN.search(sample_name)
    if not match:
        return None
    return match.group("case").lower(), match.group("slice").lower()


def _row_sort_key(row_key: Tuple[str, str]):
    case_id, slice_id = row_key
    case_match = re.search(r"(\d+)", case_id)
    slice_match = re.search(r"(\d+)", slice_id)
    case_num = int(case_match.group(1)) if case_match else -1
    slice_num = int(slice_match.group(1)) if slice_match else -1
    return case_num, slice_num, case_id, slice_id


def _index_heatmap_files(
    heatmaps_dir: str,
) -> Tuple[Dict[Tuple[str, str], Dict[int, Dict[str, str]]], List[int]]:
    grouped: Dict[Tuple[str, str], Dict[int, Dict[str, str]]] = {}
    all_epochs = set()

    with os.scandir(heatmaps_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            match = FILE_PATTERN.match(entry.name)
            if not match:
                continue

            epoch = int(match.group("epoch"))
            sample_name = match.group("sample")
            ext = match.group("ext").lower()

            row_key = _parse_case_slice(sample_name)
            if row_key is None:
                warnings.warn(f"Skipping unrecognized sample naming format: {sample_name}")
                continue

            row_bucket = grouped.setdefault(row_key, {})
            epoch_bucket = row_bucket.setdefault(epoch, {"sample": sample_name, "png": None, "npy": None})
            epoch_bucket[ext] = entry.path

    cleaned: Dict[Tuple[str, str], Dict[int, Dict[str, str]]] = {}
    for row_key, epoch_map in grouped.items():
        valid_epoch_map = {}
        for epoch in sorted(epoch_map.keys()):
            item = epoch_map[epoch]
            if item["png"] and item["npy"]:
                valid_epoch_map[epoch] = item
                all_epochs.add(epoch)
            else:
                missing = "png" if not item["png"] else "npy"
                warnings.warn(
                    f"Skipping row {row_key} epoch {epoch}: missing {missing} pair file."
                )
        if valid_epoch_map:
            cleaned[row_key] = valid_epoch_map

    return cleaned, sorted(all_epochs)


def _candidate_synapse_roots() -> List[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    candidates = [
        os.environ.get("SYNAPSE_TRAIN_NPZ"),
        "/data/shared/project_TransUNet/data/Synapse/train_npz/",
        os.path.join(script_dir, "data", "Synapse", "train_npz"),
        os.path.abspath(os.path.join(script_dir, "..", "..", "data", "Synapse", "train_npz")),
        os.path.abspath(os.path.join(cwd, "..", "..", "data", "Synapse", "train_npz")),
    ]
    deduped = []
    seen = set()
    for path in candidates:
        if not path:
            continue
        norm_path = os.path.normpath(path)
        if norm_path in seen:
            continue
        seen.add(norm_path)
        deduped.append(norm_path)
    return deduped


def _load_synapse_sample(
    sample_name: str,
    roots: List[str],
    preferred_root: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    search_roots = [preferred_root] if preferred_root else []
    search_roots.extend([root for root in roots if root and root != preferred_root])

    for root in search_roots:
        if not root:
            continue
        npz_path = os.path.join(root, f"{sample_name}.npz")
        if os.path.isfile(npz_path):
            with np.load(npz_path) as data:
                image = data["image"]
                label = data["label"]
            return image, label, root

    raise FileNotFoundError(f"Could not locate GT sample file for {sample_name}")


def _normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape for display: {arr.shape}")
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v - min_v < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v)


def _build_output_filename(heatmaps_dir: str) -> str:
    run_name = os.path.basename(os.path.normpath(heatmaps_dir))
    ckpt_name = re.sub(r"[_-]?heatmap$", "", run_name, flags=re.IGNORECASE)
    ckpt_name = ckpt_name or run_name
    return f"Heatmap_{ckpt_name}.png"


def _render_comparison_figure(
    rows: List[Dict],
    epochs: List[int],
    out_path: str,
):
    n_rows = len(rows)
    n_cols = 1 + len(epochs)
    max_class = max(int(np.max(row["label"])) for row in rows) if rows else 1
    n_classes = max(2, max_class + 1)
    cmap, norm = _discrete_cmap(n_classes)

    fig_w = max(3.0 * n_cols, 8.0)
    fig_h = max(3.0 * n_rows, 4.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in enumerate(rows):
        image = _normalize_image(row["image"])
        label = np.asarray(row["label"])
        row_name = f"{row['case_id']}_{row['slice_id']}"

        gt_ax = axes[row_idx, 0]
        if image.ndim == 2:
            gt_ax.imshow(image, cmap="gray")
        else:
            gt_ax.imshow(image)
        gt_mask = np.ma.masked_where(label == 0, label)
        gt_ax.imshow(gt_mask.astype(int), cmap=cmap, norm=norm, interpolation="nearest", alpha=0.45)
        _draw_gt_boundaries(gt_ax, label)
        gt_ax.set_axis_off()
        gt_ax.set_ylabel(row_name, fontsize=9)
        if row_idx == 0:
            gt_ax.set_title("GT", fontsize=11)

        for col_idx, epoch in enumerate(epochs, start=1):
            ax = axes[row_idx, col_idx]
            epoch_item = row["epochs"].get(epoch)
            if epoch_item is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9)
                ax.set_facecolor("black")
            else:
                overlay = plt.imread(epoch_item["png"])
                ax.imshow(overlay)
            ax.set_axis_off()
            if row_idx == 0:
                ax.set_title(f"Epoch {epoch}", fontsize=11)

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    args = _parse_args()
    heatmaps_dir = os.path.normpath(args.heatmaps_dir)
    if not os.path.isdir(heatmaps_dir):
        raise FileNotFoundError(f"Heatmap directory does not exist: {heatmaps_dir}")

    grouped, all_epochs = _index_heatmap_files(heatmaps_dir)
    if not grouped:
        raise RuntimeError(f"No valid heatmap PNG/NPY pairs found in: {heatmaps_dir}")
    if not all_epochs:
        raise RuntimeError(f"No epochs discovered from valid pairs in: {heatmaps_dir}")

    candidate_roots = _candidate_synapse_roots()
    preferred_root = None
    rows = []
    for row_key in sorted(grouped.keys(), key=_row_sort_key):
        epoch_map = grouped[row_key]
        first_epoch = min(epoch_map.keys())
        sample_name = epoch_map[first_epoch]["sample"]
        try:
            image, label, preferred_root = _load_synapse_sample(
                sample_name=sample_name,
                roots=candidate_roots,
                preferred_root=preferred_root,
            )
        except FileNotFoundError:
            warnings.warn(
                f"Skipping row {row_key}: unable to locate Synapse sample file for {sample_name}. "
                "Set SYNAPSE_TRAIN_NPZ env var if your dataset root is custom."
            )
            continue

        rows.append(
            {
                "case_id": row_key[0],
                "slice_id": row_key[1],
                "sample_name": sample_name,
                "image": image,
                "label": label,
                "epochs": epoch_map,
            }
        )

    if not rows:
        raise RuntimeError(
            "No rows with GT found. Ensure Synapse train_npz is available (or set SYNAPSE_TRAIN_NPZ)."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = _build_output_filename(heatmaps_dir)
    out_path = os.path.join(args.out_dir, out_file)
    _render_comparison_figure(rows=rows, epochs=all_epochs, out_path=out_path)
    print(f"Saved qualitative comparison to: {out_path}")


if __name__ == "__main__":
    main()
