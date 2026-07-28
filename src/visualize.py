import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom


GRID_ONLY_ARG_NAMES = (
    'viz_view',
    'viz_title',
    'viz_legend_mode',
    'viz_publication_style',
    'viz_prediction_alpha',
    'viz_gt_boundary_on_ground_truth',
    'viz_boundary_linewidth',
    'viz_dataset_label_fontsize',
    'viz_dataset_label_bold',
    'viz_dataset_label_orientation',
    'viz_header_fontsize',
    'viz_header_bold',
    'viz_legend_fontsize',
    'viz_row_spacing',
    'viz_legend_position',
    'viz_zoom_bboxes',
    'viz_zoom_inset',
    'viz_dpi',
    'grid_rows',
    'grid_indices',
    'viz_min_classes',
    'grid_slice_ranges',
    'grid_case_slides',
    'grid_cols',
    'grid_ckpts',
    'grid_model_names',
    'multi_datasets',
    'multi_dataset_args',
    'multi_model_names',
    'multi_ckpts',
)


def add_visualization_args(parser):
    parser.add_argument('--viz', action='store_true', help='show qualitative visualization for a sample')
    parser.add_argument('--viz_view', type=str, default='default', choices=['default', 'grid', 'multiple_datasets'],
                        help='visualization view mode')
    parser.add_argument('--viz_title', type=str, default=None,
                        help='optional title for grid or multiple_datasets visualization')
    parser.add_argument('--viz_legend_mode', type=str, default='global', choices=['global', 'per_row'],
                        help='legend placement for grid visualization; multiple_datasets requires per_row')
    parser.add_argument('--viz_legend_position', type=str, default='right', choices=['left', 'right', 'bottom'],
                        help='publication-style categorical legend position (default: right)')
    parser.add_argument('--viz_publication_style', action='store_true',
                        help='use the compact IEEE publication layout for grid or multiple_datasets views')
    parser.add_argument('--viz_prediction_alpha', type=float, default=0.55,
                        help='prediction-mask opacity in publication style (default: 0.55)')
    parser.add_argument('--viz_gt_boundary_on_ground_truth', action='store_true',
                        help='draw the yellow dashed ground-truth boundary on the ground-truth panel in publication style')
    parser.add_argument('--viz_boundary_linewidth', type=float, default=0.25,
                        help='line width for yellow dashed ground-truth boundaries (default: 0.25)')
    parser.add_argument('--viz_dataset_label_fontsize', type=float, default=9.0,
                        help='publication-style dataset label size in points; reasonable values are 7-12 (default: 9)')
    parser.add_argument('--viz_dataset_label_bold', action='store_true',
                        help='use bold dataset labels in publication style')
    parser.add_argument('--viz_dataset_label_orientation', type=str, default='horizontal',
                        choices=['horizontal', 'vertical'],
                        help='publication-style dataset label orientation (default: horizontal)')
    parser.add_argument('--viz_header_fontsize', type=float, default=9.0,
                        help='publication-style column header size in points; reasonable values are 7-11 (default: 9)')
    parser.add_argument('--viz_header_bold', action='store_true',
                        help='use bold column headers in publication style')
    parser.add_argument('--viz_legend_fontsize', type=float, default=9.0,
                        help='publication-style categorical legend size in points; reasonable values are 6-10 (default: 9)')
    parser.add_argument('--viz_row_spacing', type=float, default=0.0,
                        help='additional publication-style gap between dataset rows in inches; 0 uses the minimum non-overlapping gap (default: 0.0)')
    parser.add_argument('--viz_zoom_bboxes', nargs='*', default=None, metavar='BBOX',
                        help='optional pixel ROI per multiple_datasets row, in --multi_datasets order, '
                             'as "[center_x,center_y,height,width]"; auto uses the centered 25%% region')
    parser.add_argument('--viz_zoom_inset', action='store_true',
                        help='show --viz_zoom_bboxes crops as upper-right panel insets instead of separate rows')
    parser.add_argument('--viz_dpi', type=int, default=600,
                        help='PNG resolution in publication style (default: 600 dpi)')
    parser.add_argument('--viz_index', type=int, default=0, help='dataset index to visualize')
    parser.add_argument('--viz_slice', type=int, default=None, help='slice index for Synapse volumes (default: middle slice)')
    parser.add_argument('--viz_save', type=str, default='viz/', help='path to save figure (file or directory)')
    parser.add_argument('--viz_out', type=str, default=None,
                        help='output filename for the saved figure (used if --viz_save is a directory or not provided)')
    parser.add_argument('--viz_count', type=int, default=4, help='number of samples to visualize (default: 4)')
    parser.add_argument('--viz_suffix', type=str, default=None,
                        help='suffix to append to the output filename (before extension)')
    parser.add_argument('--viz_hide_input', action='store_true',
                        help='hide input column in visualization (show only prediction and ground truth)')
    parser.add_argument('--num_slices_to_overlay', type=int, default=None,
                        help='number of slices to overlay for visualization; values <=1 show a single selected slice, values >=2 are clamped to [2,20]')
    parser.add_argument('--grid_rows', type=int, default=3,
                        help='number of image/sample rows in grid visualization; must match --multi_datasets in multiple_datasets mode')
    parser.add_argument('--grid_indices', type=int, nargs='*', default=None,
                        help='optional dataset indices for each grid row; omitted means random samples')
    parser.add_argument('--viz_min_classes', type=int, default=None,
                        help='minimum number of foreground ground-truth classes required for randomly selected grid images (default: no minimum)')
    parser.add_argument('--grid_slice_ranges', nargs='*', default=None,
                        help='optional volume slice ranges for grid rows, e.g. case0008:103-111 case0022:54-62')
    parser.add_argument('--grid_case_slides', nargs='*', default=None,
                        help='optional Cataract case slide specs for grid rows, e.g. case5057:42,62,69 case5334:04,08')
    parser.add_argument('--grid_cols', type=int, default=3,
                        help='number of model-output columns in grid visualization; must match --multi_ckpts in multiple_datasets mode')
    parser.add_argument('--grid_ckpts', nargs='*', default=None,
                        help='checkpoint paths or names to render as model-output columns; append ::args for per-checkpoint model args')
    parser.add_argument('--grid_model_names', nargs='*', default=None,
                        help='optional display names for grid model-output columns')
    parser.add_argument('--multi_datasets', nargs='*', default=None,
                        help='dataset name per row in multiple_datasets visualization; names may repeat')
    parser.add_argument('--multi_dataset_args', nargs='*', default=None,
                        help='optional per-row visualization args, matched to repeated dataset occurrences in order, e.g. Dataset::--viz_index 0 --viz_slice 60; omitted rows use a random sample/slice with foreground labels')
    parser.add_argument('--multi_model_names', nargs='*', default=None,
                        help='optional display names for multiple_datasets model-output columns')
    parser.add_argument('--multi_ckpts', nargs='*', default=None,
                        help='one semicolon-separated Dataset=checkpoint::args mapping per multiple_datasets model column')
    return parser


def _ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: F401
        return True
    except Exception as e:
        print(
            "Matplotlib is required for visualization but is not available.\n"
            "Please install it (e.g., `pip install matplotlib`) and retry.\n"
            f"Error: {e}"
        )
        return False


def _discrete_cmap(n_classes: int, class_labels: Optional[List[str]] = None):
    """
    Create a discrete colormap and norm with distinct colors where index 0 is background.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Start with black for background, then use tab10/tab20 for others
    base_cmap = plt.get_cmap('tab20') if n_classes - 1 > 10 else plt.get_cmap('tab10')
    colors = [(0.0, 0.0, 0.0, 1.0)]  # background as black
    for i in range(1, n_classes):
        c = base_cmap((i - 1) % base_cmap.N)
        colors.append((c[0], c[1], c[2], 1.0))

    cmap = ListedColormap(colors, N=n_classes)
    bounds = np.arange(-0.5, n_classes + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # Labels: default to indices if not provided
    if class_labels is None or len(class_labels) != n_classes:
        class_labels = [str(i) for i in range(n_classes)]

    return cmap, norm, class_labels


def _imshow_input(ax, image: np.ndarray) -> None:
    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    elif image.ndim == 3 and image.shape[2] in (1, 3):
        if image.shape[2] == 1:
            ax.imshow(image[:, :, 0], cmap='gray')
        else:
            ax.imshow(image)
    else:
        # Fall back to showing the first slice/channel
        ax.imshow(np.squeeze(image), cmap='gray')


def _draw_gt_boundaries(
    ax,
    gt: np.ndarray,
    color: str = "yellow",
    linewidth: float = 0.25,
    linestyle: str = "--",
) -> None:
    if linewidth < 0:
        raise ValueError("boundary linewidth must be non-negative, got {}".format(linewidth))
    gt_arr = np.asarray(gt)
    if gt_arr.ndim != 2:
        gt_arr = np.squeeze(gt_arr)
    if gt_arr.ndim != 2:
        return
    classes = np.unique(gt_arr.astype(int))
    classes = classes[classes != 0]
    if classes.size == 0:
        return
    for cls in classes:
        mask = gt_arr == cls
        if mask.any():
            ax.contour(
                mask.astype(float),
                levels=[0.5],
                colors=color,
                linewidths=linewidth,
                linestyles=linestyle,
            )


def _to_numpy_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_float_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if v_max - v_min < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - v_min) / (v_max - v_min)


def _prepare_display_image(image: np.ndarray) -> np.ndarray:
    img = np.squeeze(np.asarray(image))
    if img.ndim == 2:
        img = img[..., None]
    elif img.ndim == 3:
        # Support both CHW and HWC.
        if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[-1] not in (1, 3) and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[-1] not in (1, 3):
            img = img[..., :1]
    else:
        raise ValueError(f"Unsupported image shape for visualization: {img.shape}")

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[-1] > 3:
        img = img[..., :3]

    return _normalize_float_map(img)


def _prepare_weight_map(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights)
    arr = np.squeeze(arr)
    if arr.ndim < 2:
        raise ValueError(f"weights_tensor must have at least 2 dims after squeeze, got {arr.shape}")
    while arr.ndim > 2:
        arr = arr[0]
    return arr.astype(np.float32, copy=False)


def _resize_float_map(float_map: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if float_map.shape == (out_h, out_w):
        return float_map.astype(np.float32, copy=False)
    tensor = torch.from_numpy(float_map).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _prepare_label_map(label: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    arr = np.squeeze(np.asarray(label))
    if arr.ndim == 3:
        # Handle one-hot style label maps if they appear.
        if arr.shape[0] <= 32 and arr.shape[1] > 32 and arr.shape[2] > 32:
            arr = np.argmax(arr, axis=0)
        elif arr.shape[-1] <= 32 and arr.shape[0] > 32 and arr.shape[1] > 32:
            arr = np.argmax(arr, axis=-1)
        else:
            arr = arr[0]
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"gt_mask_or_label must resolve to 2D, got {arr.shape}")
    if arr.shape != (out_h, out_w):
        tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        arr = F.interpolate(tensor, size=(out_h, out_w), mode="nearest").squeeze(0).squeeze(0).cpu().numpy()
    return arr


def save_weight_heatmap_overlay(
    image_tensor,
    weights_tensor,
    gt_mask_or_label,
    out_path_png,
    out_path_npy,
    normalize: bool = True,
):
    """
    Save a BU-loss weight heatmap overlaid on the input image with GT boundaries.
    """
    if not _ensure_matplotlib():
        raise RuntimeError("Matplotlib is not available. Please install it to save heatmaps.")
    import matplotlib.pyplot as plt

    image_np = _to_numpy_array(image_tensor)
    weights_np = _to_numpy_array(weights_tensor)
    gt_np = _to_numpy_array(gt_mask_or_label)

    out_dir_png = os.path.dirname(out_path_png)
    out_dir_npy = os.path.dirname(out_path_npy)
    if out_dir_png:
        os.makedirs(out_dir_png, exist_ok=True)
    if out_dir_npy:
        os.makedirs(out_dir_npy, exist_ok=True)

    np.save(out_path_npy, weights_np)

    image_rgb = _prepare_display_image(image_np)
    out_h, out_w = image_rgb.shape[:2]

    weight_map = _prepare_weight_map(weights_np)
    weight_map = _resize_float_map(weight_map, out_h, out_w)
    if normalize:
        weight_map = _normalize_float_map(weight_map)

    heat_rgb = plt.get_cmap("jet")(np.clip(weight_map, 0.0, 1.0))[..., :3]
    overlay_alpha = 0.45
    overlay_rgb = np.clip((1.0 - overlay_alpha) * image_rgb + overlay_alpha * heat_rgb, 0.0, 1.0)

    gt_map = _prepare_label_map(gt_np, out_h, out_w)

    dpi = 100
    fig = plt.figure(figsize=(max(out_w, 1) / dpi, max(out_h, 1) / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(overlay_rgb, interpolation="nearest")
    _draw_gt_boundaries(ax, gt_map, color="white", linewidth=1.0, linestyle=":")
    ax.set_axis_off()
    fig.savefig(out_path_png, dpi=dpi)
    plt.close(fig)


def _sanitize_sample_id(sample_id: str) -> str:
    return str(sample_id).replace("/", "-").replace("\\", "-").replace(" ", "_")


def save_pending_weight_heatmaps(
    should_save_heatmap_this_epoch: bool,
    details,
    pending_heatmap_samples,
    case_names,
    image_batch,
    label_batch,
    epoch_index: int,
    iter_num: int,
    heatmaps_dir: str,
    logger=None,
) -> int:
    """
    Save heatmaps for pending samples that appear in the current batch.
    Returns how many samples were saved in this call.
    """
    if (
        not should_save_heatmap_this_epoch
        or details is None
        or "weights" not in details
        or not pending_heatmap_samples
    ):
        return 0

    matched_batch_indices = []
    if case_names:
        for sample_index, sample_name in enumerate(case_names):
            if sample_name in pending_heatmap_samples:
                matched_batch_indices.append((sample_index, sample_name))

    saved_count = 0
    for sample_index, sample_name in matched_batch_indices:
        sample_id_safe = _sanitize_sample_id(sample_name)
        file_stem = f"epoch_{epoch_index:04d}_step_{iter_num:06d}_sample_{sample_id_safe}"
        out_png = os.path.join(heatmaps_dir, f"{file_stem}.png")
        out_npy = os.path.join(heatmaps_dir, f"{file_stem}.npy")
        try:
            save_weight_heatmap_overlay(
                image_tensor=image_batch[sample_index],
                weights_tensor=details["weights"][sample_index],
                gt_mask_or_label=label_batch[sample_index],
                out_path_png=out_png,
                out_path_npy=out_npy,
                normalize=True,
            )
            pending_heatmap_samples.discard(sample_name)
            saved_count += 1
            if logger is not None:
                logger.info(
                    "Saved BU-loss heatmap | epoch %d step %d sample %s -> %s",
                    epoch_index,
                    iter_num,
                    sample_name,
                    out_png,
                )
        except Exception:
            if logger is not None:
                logger.exception(
                    "Failed to save BU-loss heatmap | epoch %d step %d sample %s",
                    epoch_index,
                    iter_num,
                    sample_name,
                )

    return saved_count


def _case_id_from_sample_name(sample_name: str) -> str:
    name = str(sample_name)
    lowered = name.lower()
    for token in ("_slice", "-slice", "_frame", "-frame", "_z", "-z"):
        idx = lowered.rfind(token)
        if idx > 0:
            return name[:idx]
    match = re.match(r"^(.*?)[_-]\d+$", name)
    if match:
        return match.group(1)
    return name


def _slice_number_from_sample_name(sample_name: str):
    name = str(sample_name)
    lowered = name.lower()
    match = re.search(r"(?:slice|frame|z)[_-]?(\d+)(?!.*\d)", lowered)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)(?!.*\d)", name)
    if match:
        return int(match.group(1))
    return None


def _evenly_spaced_positions(length: int, count: int):
    if length <= 0:
        return []
    count = max(1, min(int(count), length))
    indices = np.linspace(0, length - 1, num=count)
    indices = np.round(indices).astype(int).tolist()
    indices = [max(0, min(length - 1, idx)) for idx in indices]
    ordered = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    if len(ordered) < count:
        for idx in range(length):
            if idx not in seen:
                ordered.append(idx)
                seen.add(idx)
                if len(ordered) >= count:
                    break
    return ordered[:count]


def _get_dataset_sample_names(dataset):
    if hasattr(dataset, "sample_list"):
        return [str(name).strip() for name in dataset.sample_list]
    if hasattr(dataset, "image_files"):
        sample_names = []
        for image_path in dataset.image_files:
            filename = os.path.basename(str(image_path))
            sample_names.append(os.path.splitext(filename)[0])
        return sample_names
    return []


def _select_evenly_spaced_samples(sample_names, count):
    if not sample_names:
        return []
    sortable = []
    for name in sample_names:
        slice_number = _slice_number_from_sample_name(name)
        if slice_number is None:
            sortable.append((1, str(name), str(name)))
        else:
            sortable.append((0, int(slice_number), str(name)))
    sortable.sort()
    ordered_names = [entry[2] for entry in sortable]
    chosen_positions = _evenly_spaced_positions(len(ordered_names), count)
    return [ordered_names[pos] for pos in chosen_positions]


def _resolve_heatmap_sample_targets(dataset_sample_names, anchor_sample_name: str, num_heatmap_slices: int):
    case_id = _case_id_from_sample_name(anchor_sample_name)
    case_samples = [name for name in dataset_sample_names if _case_id_from_sample_name(name) == case_id]
    if not case_samples:
        return [anchor_sample_name]
    selected = _select_evenly_spaced_samples(case_samples, num_heatmap_slices)
    return selected if selected else [anchor_sample_name]


def _should_save_heatmap_epoch(epoch_index: int) -> bool:
    # Baseline at epoch 1, then every 70 epochs (70, 140, 210, ...).
    return epoch_index == 1 or (epoch_index % 70 == 0)


def _clamp_overlay_slices(num_slices_to_overlay: Optional[int]) -> Optional[int]:
    if num_slices_to_overlay is None:
        return None
    try:
        value = int(num_slices_to_overlay)
    except (TypeError, ValueError):
        return None
    if value <= 1:
        return None
    return max(2, min(20, value))


def _evenly_spaced_indices(depth: int, count: int) -> List[int]:
    if depth <= 0:
        return []
    count = max(1, min(count, depth))
    indices = np.linspace(0, depth - 1, num=count)
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, depth - 1)
    seen = set()
    ordered = []
    for idx in indices.tolist():
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    return ordered


def _resolve_overlay_indices(depth: int, num_slices_to_overlay: Optional[int]) -> Optional[List[int]]:
    count = _clamp_overlay_slices(num_slices_to_overlay)
    if count is None:
        return None
    indices = _evenly_spaced_indices(depth, count)
    return indices if indices else None


def _overlay_class_maps(maps: List[np.ndarray]) -> np.ndarray:
    if not maps:
        return np.array([])
    composite = np.zeros_like(maps[0])
    for m in maps:
        mask = m != 0
        composite[mask] = m[mask]
    return composite


def _infer_slice_prediction(
    model: torch.nn.Module,
    img2d: np.ndarray,
    img_size: int,
    device: str,
) -> np.ndarray:
    h, w = img2d.shape
    if (h, w) != (img_size, img_size):
        img_resized = zoom(img2d, (img_size / h, img_size / w), order=3)
    else:
        img_resized = img2d

    with torch.no_grad():
        input_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float().to(device)
        outputs = model(input_tensor)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        pred_small = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

    if (h, w) != (img_size, img_size):
        pred = zoom(pred_small, (h / img_size, w / img_size), order=0)
    else:
        pred = pred_small
    return pred


def _plot_triplet(
    image: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    n_classes: int,
    class_labels: Optional[List[str]] = None,
    figure_title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
    include_input: bool = True,
    boundary_linewidth: float = 0.25,
):
    """
    Plot a side-by-side triplet: Input | Prediction | Ground Truth
    """
    if not _ensure_matplotlib():
        raise RuntimeError("Matplotlib is not available. Please install it to use this function.")

    import matplotlib.pyplot as plt

    cmap, norm, labels = _discrete_cmap(n_classes, class_labels)

    ncols = 3 if include_input else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)

    if include_input:
        input_ax = axes[0]
        pred_ax = axes[1]
        gt_ax = axes[2]

        # Input image (handle grayscale or RGB)
        _imshow_input(input_ax, image)
        input_ax.set_title('Input', fontsize=12)
        input_ax.set_xlabel('X (px)')
        input_ax.set_ylabel('Y (px)')
    else:
        pred_ax = axes[0]
        gt_ax = axes[1]
        pred_ax.set_ylabel('Y (px)')

    # Prediction
    _imshow_input(pred_ax, image)
    pred_mask = np.ma.masked_where(pred == 0, pred)
    im1 = pred_ax.imshow(pred_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
    _draw_gt_boundaries(pred_ax, gt, linewidth=boundary_linewidth)
    pred_ax.set_title('Prediction', fontsize=12)
    pred_ax.set_xlabel('X (px)')
    if include_input:
        pred_ax.set_yticklabels([])

    # Ground truth
    _imshow_input(gt_ax, image)
    gt_mask = np.ma.masked_where(gt == 0, gt)
    im2 = gt_ax.imshow(gt_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
    _draw_gt_boundaries(gt_ax, gt, linewidth=boundary_linewidth)
    gt_ax.set_title('Ground Truth', fontsize=12)
    gt_ax.set_xlabel('X (px)')
    gt_ax.set_yticklabels([])

    # Common colorbar with class labels (narrow width)
    cbar = fig.colorbar(
        im2,
        ax=axes.ravel().tolist(),
        ticks=np.arange(0, n_classes, 1),
        fraction=0.03,
        pad=0.02,
    )
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('Classes', rotation=270, labelpad=15)

    if figure_title:
        fig.suptitle(figure_title, fontsize=13)

    if save_path is not None:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def _plot_triplet_grid(
    triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_classes: int,
    class_labels: Optional[List[str]] = None,
    row_titles: Optional[List[str]] = None,
    figure_title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    include_input: bool = True,
    boundary_linewidth: float = 0.25,
):
    """
    Plot a grid with rows of (Input | Prediction | Ground Truth).
    """
    if not _ensure_matplotlib():
        raise RuntimeError("Matplotlib is not available. Please install it to use this function.")

    import matplotlib.pyplot as plt

    num_rows = len(triplets)
    if num_rows == 0:
        return

    # Default figure height scales with number of rows
    if figsize is None:
        figsize = (12, 3.5 * num_rows)

    cmap, norm, labels = _discrete_cmap(n_classes, class_labels)

    ncols = 3 if include_input else 2
    fig, axes = plt.subplots(num_rows, ncols, figsize=figsize, constrained_layout=True)
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (image, pred, gt) in enumerate(triplets):
        if include_input:
            input_ax = axes[r, 0]
            pred_ax = axes[r, 1]
            gt_ax = axes[r, 2]

            _imshow_input(input_ax, image)
        else:
            pred_ax = axes[r, 0]
            gt_ax = axes[r, 1]

        _imshow_input(pred_ax, image)
        pred_mask = np.ma.masked_where(pred == 0, pred)
        im_pred = pred_ax.imshow(pred_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
        _draw_gt_boundaries(pred_ax, gt, linewidth=boundary_linewidth)

        _imshow_input(gt_ax, image)
        gt_mask = np.ma.masked_where(gt == 0, gt)
        im_gt = gt_ax.imshow(gt_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
        _draw_gt_boundaries(gt_ax, gt, linewidth=boundary_linewidth)

        # Titles on top row only
        if r == 0:
            if include_input:
                input_ax.set_title('Input', fontsize=12)
            pred_ax.set_title('Prediction', fontsize=12)
            gt_ax.set_title('Ground Truth', fontsize=12)

        # Row titles (case names)
        if row_titles and r < len(row_titles):
            label_ax = input_ax if include_input else pred_ax
            label_ax.set_ylabel(row_titles[r], fontsize=10)

        # Axes labels
        if include_input:
            input_ax.set_xlabel('X (px)')
        pred_ax.set_xlabel('X (px)')
        gt_ax.set_xlabel('X (px)')
        # Hide y tick labels for middle and right columns to reduce clutter
        if include_input:
            pred_ax.set_yticklabels([])
        gt_ax.set_yticklabels([])

    # Colorbar with labels (narrow width)
    cbar = fig.colorbar(
        im_gt,
        ax=axes.ravel().tolist(),
        ticks=np.arange(0, n_classes, 1),
        fraction=0.03,
        pad=0.02,
    )
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('Classes', rotation=270, labelpad=15)

    if figure_title:
        fig.suptitle(figure_title, fontsize=13)

    if save_path is not None:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization grid to: {save_path}")

    plt.show()


def visualize_synapse_sample(
    model: torch.nn.Module,
    volume: np.ndarray,
    label: np.ndarray,
    slice_index: Optional[int] = None,
    img_size: int = 224,
    class_labels: Optional[List[str]] = None,
    figure_title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
    num_slices_to_overlay: Optional[int] = None,
    include_input: bool = True,
    device: Optional[str] = None,
    boundary_linewidth: float = 0.25,
):
    """
    Visualize a Synapse sample: Input slice, Predicted mask, Ground truth mask.

    Args:
        model: Trained segmentation model.
        volume: 3D numpy array [D, H, W] or 2D [H, W].
        label: 3D [D, H, W] or 2D [H, W] ground truth labels.
        slice_index: If volume is 3D, pick this slice (defaults to middle).
        img_size: Model input size (square) for resizing during inference.
        class_labels: Optional list of class names (length n_classes).
        figure_title: Optional title for the figure.
        figsize: Matplotlib figure size.
        save_path: Optional path to save the figure.
        device: Optional device override ('cuda' or 'cpu').
    """
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    overlay_indices = None
    if volume.ndim == 3:
        D, H, W = volume.shape
        overlay_indices = _resolve_overlay_indices(D, num_slices_to_overlay)
        if overlay_indices and label.ndim != 3:
            overlay_indices = None
        if slice_index is None:
            if overlay_indices:
                slice_index = overlay_indices[len(overlay_indices) // 2]
            else:
                slice_index = D // 2
        img2d = volume[slice_index]
        if overlay_indices:
            gt_slices = [label[i] for i in overlay_indices]
            gt2d = _overlay_class_maps(gt_slices)
        else:
            gt2d = label[slice_index] if label.ndim == 3 else label
    elif volume.ndim == 2:
        img2d = volume
        gt2d = label
        H, W = img2d.shape
    else:
        raise ValueError("Synapse volume must be 2D or 3D array")

    if overlay_indices:
        pred_slices = [
            _infer_slice_prediction(model, volume[i], img_size, device) for i in overlay_indices
        ]
        pred = _overlay_class_maps(pred_slices)
    else:
        pred = _infer_slice_prediction(model, img2d, img_size, device)

    # Determine n_classes from prediction/labels
    n_classes = int(max(np.max(pred), np.max(gt2d)) + 1)
    if n_classes < 2:
        n_classes = 2  # at least background + one class

    # Default class labels for Synapse if none provided
    default_labels = [
        'Background', 'Aorta', 'Gallbladder', 'Kidney(L)', 'Kidney(R)',
        'Liver', 'Pancreas', 'Spleen', 'Stomach'
    ]
    if class_labels is None:
        class_labels = default_labels[:n_classes]

    _plot_triplet(
        image=img2d,
        pred=pred,
        gt=gt2d,
        n_classes=n_classes,
        class_labels=class_labels,
        figure_title=figure_title,
        figsize=figsize,
        save_path=save_path,
        include_input=include_input,
        boundary_linewidth=boundary_linewidth,
    )


def visualize_cataract_sample(
    model: torch.nn.Module,
    image: np.ndarray,
    label: np.ndarray,
    img_size: int = 224,
    class_labels: Optional[List[str]] = None,
    figure_title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
    num_slices_to_overlay: Optional[int] = None,
    include_input: bool = True,
    device: Optional[str] = None,
    boundary_linewidth: float = 0.25,
):
    """
    Visualize a Cataract-101K sample: Input image, Predicted mask, Ground truth mask.

    Args:
        model: Trained segmentation model.
        image: 2D/3D numpy array [H, W, 3] RGB expected.
        label: 2D numpy array [H, W] ground truth labels.
        img_size: Model input size (square) for resizing during inference.
        class_labels: Optional list of class names (length n_classes).
        figure_title: Optional title for the figure.
        figsize: Matplotlib figure size.
        save_path: Optional path to save the figure.
        device: Optional device override ('cuda' or 'cpu').
    """
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Cataract sample expects an RGB image with shape [H, W, 3]")

    H, W, _ = image.shape

    # Resize RGB channels to model input size
    resized_channels = []
    for ch in range(3):
        resized_channels.append(zoom(image[:, :, ch], (img_size / H, img_size / W), order=3))
    img_resized = np.stack(resized_channels, axis=0)  # (3, img_size, img_size)

    with torch.no_grad():
        input_tensor = torch.from_numpy(img_resized).unsqueeze(0).float().to(device)
        outputs = model(input_tensor)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        pred_small = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

    # Resize prediction back to original size
    pred = zoom(pred_small, (H / img_size, W / img_size), order=0)

    # Determine n_classes from prediction/labels
    n_classes = int(max(np.max(pred), np.max(label)) + 1)
    if n_classes < 2:
        n_classes = 2

    # Default class labels for Cataract-101K if none provided
    default_labels = ['Background', 'Pupil', 'Cornea', 'Lens', 'Instruments']
    if class_labels is None:
        class_labels = default_labels[:n_classes]

    _plot_triplet(
        image=image,
        pred=pred,
        gt=label,
        n_classes=n_classes,
        class_labels=class_labels,
        figure_title=figure_title,
        figsize=figsize,
        save_path=save_path,
        include_input=include_input,
        boundary_linewidth=boundary_linewidth,
    )


def visualize_synapse_batch(
    model: torch.nn.Module,
    volumes: List[np.ndarray],
    labels: List[np.ndarray],
    slice_indices: Optional[List[Optional[int]]] = None,
    img_size: int = 224,
    class_labels: Optional[List[str]] = None,
    row_titles: Optional[List[str]] = None,
    figure_title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    num_slices_to_overlay: Optional[int] = None,
    include_input: bool = True,
    device: Optional[str] = None,
    boundary_linewidth: float = 0.25,
):
    """
    Visualize multiple Synapse samples as a single grid figure.
    Each row is: Input | Prediction | Ground Truth.
    """
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    triplets = []
    max_class = 1
    for idx, (volume, label) in enumerate(zip(volumes, labels)):
        overlay_indices = None
        s_idx = None
        if slice_indices is not None and idx < len(slice_indices):
            s_idx = slice_indices[idx]

        if volume.ndim == 3:
            D, H, W = volume.shape
            overlay_indices = _resolve_overlay_indices(D, num_slices_to_overlay)
            if overlay_indices and label.ndim != 3:
                overlay_indices = None
            if s_idx is None:
                if overlay_indices:
                    s_idx = overlay_indices[len(overlay_indices) // 2]
                else:
                    s_idx = D // 2
            img2d = volume[s_idx]
            if overlay_indices:
                gt_slices = [label[i] for i in overlay_indices]
                gt2d = _overlay_class_maps(gt_slices)
            else:
                gt2d = label[s_idx] if label.ndim == 3 else label
        elif volume.ndim == 2:
            img2d = volume
            gt2d = label
            H, W = img2d.shape
        else:
            raise ValueError("Synapse volume must be 2D or 3D array")

        if overlay_indices:
            pred_slices = [
                _infer_slice_prediction(model, volume[i], img_size, device) for i in overlay_indices
            ]
            pred = _overlay_class_maps(pred_slices)
        else:
            pred = _infer_slice_prediction(model, img2d, img_size, device)

        max_class = max(max_class, int(max(np.max(pred), np.max(gt2d)) + 1))
        triplets.append((img2d, pred, gt2d))

    if max_class < 2:
        max_class = 2

    default_labels = [
        'Background', 'Aorta', 'Gallbladder', 'Kidney(L)', 'Kidney(R)',
        'Liver', 'Pancreas', 'Spleen', 'Stomach'
    ]
    if class_labels is None:
        class_labels = default_labels[:max_class]

    _plot_triplet_grid(
        triplets=triplets,
        n_classes=max_class,
        class_labels=class_labels,
        row_titles=row_titles,
        figure_title=figure_title,
        figsize=figsize,
        save_path=save_path,
        include_input=include_input,
        boundary_linewidth=boundary_linewidth,
    )


def visualize_cataract_batch(
    model: torch.nn.Module,
    images: List[np.ndarray],
    labels: List[np.ndarray],
    img_size: int = 224,
    class_labels: Optional[List[str]] = None,
    row_titles: Optional[List[str]] = None,
    figure_title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    num_slices_to_overlay: Optional[int] = None,
    include_input: bool = True,
    device: Optional[str] = None,
    boundary_linewidth: float = 0.25,
):
    """
    Visualize multiple Cataract-101K samples as a single grid figure.
    Each row is: Input | Prediction | Ground Truth.
    """
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    triplets = []
    max_class = 1
    for image, label in zip(images, labels):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Cataract samples must be RGB images with shape [H, W, 3]")
        H, W, _ = image.shape

        # Resize RGB channels
        resized_channels = []
        for ch in range(3):
            resized_channels.append(zoom(image[:, :, ch], (img_size / H, img_size / W), order=3))
        img_resized = np.stack(resized_channels, axis=0)

        with torch.no_grad():
            input_tensor = torch.from_numpy(img_resized).unsqueeze(0).float().to(device)
            outputs = model(input_tensor)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            pred_small = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

        pred = zoom(pred_small, (H / img_size, W / img_size), order=0)

        max_class = max(max_class, int(max(np.max(pred), np.max(label)) + 1))
        triplets.append((image, pred, label))

    if max_class < 2:
        max_class = 2

    default_labels = ['Background', 'Pupil', 'Cornea', 'Lens', 'Instruments']
    if class_labels is None:
        class_labels = default_labels[:max_class]

    _plot_triplet_grid(
        triplets=triplets,
        n_classes=max_class,
        class_labels=class_labels,
        row_titles=row_titles,
        figure_title=figure_title,
        figsize=figsize,
        save_path=save_path,
        include_input=include_input,
        boundary_linewidth=boundary_linewidth,
    )


def resolve_visualization_save_path(args, default_name: str):
    img_exts = {'.png', '.jpg', '.jpeg', '.pdf', '.svg', '.tif', '.tiff'}
    base = args.viz_save
    name = args.viz_out or default_name
    if args.viz_suffix:
        base_name, ext = os.path.splitext(name)
        name = f"{base_name}{args.viz_suffix}{ext}"
    if base is None:
        return os.path.join('qualitative', name)
    ext = os.path.splitext(base)[1].lower()
    if ext in img_exts:
        return base
    return os.path.join(base, name)


def _prediction_triplet_for_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    label: np.ndarray,
    slice_index: Optional[int],
    img_size: int,
    num_slices_to_overlay: Optional[int],
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlay_indices = None
    if volume.ndim == 3:
        D, H, W = volume.shape
        overlay_indices = _resolve_overlay_indices(D, num_slices_to_overlay)
        if overlay_indices and label.ndim != 3:
            overlay_indices = None
        if slice_index is None:
            if overlay_indices:
                slice_index = overlay_indices[len(overlay_indices) // 2]
            else:
                slice_index = D // 2
        img2d = volume[slice_index]
        if overlay_indices:
            gt_slices = [label[i] for i in overlay_indices]
            gt2d = _overlay_class_maps(gt_slices)
        else:
            gt2d = label[slice_index] if label.ndim == 3 else label
    elif volume.ndim == 2:
        img2d = volume
        gt2d = label
    else:
        raise ValueError("Volume sample must be 2D or 3D array")

    if overlay_indices:
        pred_slices = [
            _infer_slice_prediction(model, volume[i], img_size, device) for i in overlay_indices
        ]
        pred = _overlay_class_maps(pred_slices)
    else:
        pred = _infer_slice_prediction(model, img2d, img_size, device)
    return img2d, pred, gt2d


def _prediction_triplet_for_rgb_image(
    model: torch.nn.Module,
    image: np.ndarray,
    label: np.ndarray,
    img_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Frame sample must be an RGB image with shape [H, W, 3]")

    H, W, _ = image.shape
    resized_channels = []
    for ch in range(3):
        resized_channels.append(zoom(image[:, :, ch], (img_size / H, img_size / W), order=3))
    img_resized = np.stack(resized_channels, axis=0)

    with torch.no_grad():
        input_tensor = torch.from_numpy(img_resized).unsqueeze(0).float().to(device)
        outputs = model(input_tensor)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        pred_small = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

    pred = zoom(pred_small, (H / img_size, W / img_size), order=0)
    return image, pred, label


def build_grid_prediction_entry(
    model: torch.nn.Module,
    sample: Dict[str, np.ndarray],
    args,
    dataset_name: str,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name in ['Synapse', 'ACDC']:
        image, pred, gt = _prediction_triplet_for_volume(
            model=model,
            volume=sample['image'],
            label=sample['label'],
            slice_index=args.viz_slice,
            img_size=args.img_size,
            num_slices_to_overlay=args.num_slices_to_overlay,
            device=device,
        )
    elif dataset_name in ['Cataract1k', 'EndoVis2018']:
        image, pred, gt = _prediction_triplet_for_rgb_image(
            model=model,
            image=sample['image'],
            label=sample['label'],
            img_size=args.img_size,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported dataset for grid visualization: {dataset_name}")

    return {
        'image': image,
        'pred': pred,
        'gt': gt,
    }


def _publication_dataset_label(dataset_name: str) -> str:
    labels = {
        'Synapse': 'Synapse',
        'ACDC': 'ACDC',
        'Cataract1k': 'Cataract-1K',
        'EndoVis2018': 'EndoVis2018',
    }
    return labels.get(dataset_name, str(dataset_name))


def _publication_column_header(label: str) -> str:
    label = str(label)
    if len(label) <= 9:
        return label
    if '+' in label:
        left, right = label.split('+', 1)
        if 'TopK' in left and not left.startswith('TopK'):
            left = left.replace('TopK', '\nTopK', 1)
        return "{}+\n{}".format(left, right)
    if 'TopK' in label and not label.startswith('TopK'):
        return label.replace('TopK', '\nTopK', 1)
    midpoint = len(label) // 2
    return label[:midpoint] + '\n' + label[midpoint:]


def _save_publication_grid(fig, save_path: str, dpi: int, save_pdf_and_png: bool) -> None:
    root, extension = os.path.splitext(save_path)
    if save_pdf_and_png:
        output_root = root if extension else save_path
        output_paths = [output_root + '.pdf', output_root + '.png']
    else:
        output_paths = [save_path]

    for output_path in output_paths:
        dirpath = os.path.dirname(output_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        output_extension = os.path.splitext(output_path)[1].lower()
        save_kwargs = {
            'facecolor': 'white',
            'edgecolor': 'white',
        }
        if output_extension != '.pdf':
            save_kwargs['dpi'] = dpi
        fig.savefig(output_path, **save_kwargs)
        print(f"Saved publication visualization grid to: {output_path}")


def _publication_image_aspect(image: np.ndarray) -> float:
    image_array = np.squeeze(np.asarray(image))
    if image_array.ndim == 2:
        height, width = image_array.shape
    elif image_array.ndim == 3 and image_array.shape[-1] in (1, 3):
        height, width = image_array.shape[:2]
    elif image_array.ndim == 3 and image_array.shape[0] in (1, 3):
        height, width = image_array.shape[1:]
    else:
        return 1.0
    if height <= 0 or width <= 0:
        return 1.0
    return float(height) / float(width)


def _publication_legend_geometry(
    labels: List[str],
    legend_fontsize: float,
    position: str,
) -> Tuple[int, float]:
    foreground_labels = labels[1:]
    if not foreground_labels:
        return 1, 0.0

    line_height = legend_fontsize / 72.0 * 1.25
    if position in ('left', 'right'):
        return 1, len(foreground_labels) * line_height + 0.04

    return len(foreground_labels), line_height + 0.04


def _resolve_publication_zoom_bbox(spec, image: np.ndarray, row_index: int, dataset_name: str):
    image_array = np.squeeze(np.asarray(image))
    if image_array.ndim == 2:
        image_height, image_width = image_array.shape
    elif image_array.ndim == 3 and image_array.shape[-1] in (1, 3):
        image_height, image_width = image_array.shape[:2]
    else:
        raise ValueError(
            "Cannot resolve zoom ROI for row {} ({}) from image shape {}.".format(
                row_index + 1,
                dataset_name,
                image_array.shape,
            )
        )

    if spec == 'auto':
        center_x = image_width / 2.0
        center_y = image_height / 2.0
        height = image_height * 0.25
        width = image_width * 0.25
    else:
        center_x, center_y, height, width = spec

    crop_height = min(image_height, max(1, int(np.ceil(height))))
    crop_width = min(image_width, max(1, int(np.ceil(width))))
    x0 = int(np.floor(center_x - width / 2.0))
    y0 = int(np.floor(center_y - height / 2.0))
    x0 = max(0, min(x0, image_width - crop_width))
    y0 = max(0, min(y0, image_height - crop_height))
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    final_center_x = (x0 + x1) / 2.0
    final_center_y = (y0 + y1) / 2.0
    print(
        "Resolved zoom ROI row {} ({}): "
        "[center_x,center_y,height,width]=[{:.1f},{:.1f},{},{}], "
        "bounds=[x0={},y0={},x1={},y1={}]".format(
            row_index + 1,
            dataset_name,
            final_center_x,
            final_center_y,
            crop_height,
            crop_width,
            x0,
            y0,
            x1,
            y1,
        )
    )
    return x0, y0, x1, y1


def _crop_publication_array(array: np.ndarray, bounds) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    array = np.asarray(array)
    if array.ndim == 2:
        return array[y0:y1, x0:x1]
    if array.ndim == 3:
        return array[y0:y1, x0:x1, ...]
    raise ValueError("Zoom ROI arrays must be 2D or HWC, got shape {}.".format(array.shape))


def _draw_publication_roi_rectangle(ax, bounds, linewidth: float) -> None:
    from matplotlib.patches import Rectangle

    x0, y0, x1, y1 = bounds
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor='red',
            linestyle='--',
            linewidth=linewidth,
        )
    )


def _draw_publication_panel_row(
    axes,
    image: np.ndarray,
    gt: np.ndarray,
    predictions: List[np.ndarray],
    cmap,
    norm,
    prediction_alpha: float,
    draw_gt_boundary_on_ground_truth: bool,
    boundary_linewidth: float,
    roi_bounds=None,
) -> None:
    column_index = 0
    if len(axes) == len(predictions) + 2:
        _imshow_input(axes[column_index], image)
        column_index += 1

    gt_ax = axes[column_index]
    _imshow_input(gt_ax, image)
    gt_mask = np.ma.masked_where(gt == 0, gt)
    gt_ax.imshow(gt_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
    if draw_gt_boundary_on_ground_truth:
        _draw_gt_boundaries(gt_ax, gt, linewidth=boundary_linewidth)
    column_index += 1

    for prediction in predictions:
        pred_ax = axes[column_index]
        _imshow_input(pred_ax, image)
        pred_mask = np.ma.masked_where(prediction == 0, prediction)
        pred_ax.imshow(
            pred_mask.astype(int),
            cmap=cmap,
            norm=norm,
            interpolation='nearest',
            alpha=prediction_alpha,
        )
        _draw_gt_boundaries(pred_ax, gt, linewidth=boundary_linewidth)
        column_index += 1

    for image_ax in axes:
        if roi_bounds is not None:
            _draw_publication_roi_rectangle(image_ax, roi_bounds, boundary_linewidth)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_visible(False)


def _draw_publication_zoom_insets(
    parent_axes,
    image: np.ndarray,
    gt: np.ndarray,
    predictions: List[np.ndarray],
    bounds,
    cmap,
    norm,
    prediction_alpha: float,
    draw_gt_boundary_on_ground_truth: bool,
    boundary_linewidth: float,
) -> None:
    x0, y0, x1, y1 = bounds
    crop_aspect = (y1 - y0) / float(x1 - x0)
    image_aspect = _publication_image_aspect(image)
    inset_extent = 0.42
    inset_padding = 0.02
    inset_width = inset_extent
    inset_height = inset_width * crop_aspect / image_aspect
    if inset_height > inset_extent:
        inset_width *= inset_extent / inset_height
        inset_height = inset_extent

    inset_axes = [
        parent_ax.inset_axes(
            [
                1.0 - inset_padding - inset_width,
                1.0 - inset_padding - inset_height,
                inset_width,
                inset_height,
            ],
            zorder=5,
        )
        for parent_ax in parent_axes
    ]
    zoom_image = _crop_publication_array(image, bounds)
    zoom_gt = _crop_publication_array(gt, bounds)
    zoom_predictions = [
        _crop_publication_array(prediction, bounds)
        for prediction in predictions
    ]
    _draw_publication_panel_row(
        inset_axes,
        zoom_image,
        zoom_gt,
        zoom_predictions,
        cmap,
        norm,
        prediction_alpha,
        draw_gt_boundary_on_ground_truth,
        boundary_linewidth,
    )
    for inset_ax in inset_axes:
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_color('red')
            spine.set_linestyle('-')
            spine.set_linewidth(boundary_linewidth)


def _plot_publication_grid_with_zoom(
    rows: List[Dict],
    model_names: List[str],
    include_input: bool,
    save_path: Optional[str],
    prediction_alpha: float,
    draw_gt_boundary_on_ground_truth: bool,
    boundary_linewidth: float,
    dataset_label_fontsize: float,
    header_fontsize: float,
    legend_fontsize: float,
    row_spacing: float,
    legend_position: str,
    dpi: int,
    save_pdf_and_png: bool,
    zoom_bboxes,
    dataset_label_bold: bool = False,
    header_bold: bool = False,
    dataset_label_orientation: str = 'horizontal',
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    num_rows = len(rows)
    image_column_count = len(model_names) + 1 + int(include_input)
    figure_width = 7.16
    longest_row_label = max(
        len("({}) {}".format(chr(ord('a') + index), _publication_dataset_label(row['dataset_name'])))
        for index, row in enumerate(rows)
    )
    if dataset_label_orientation == 'vertical':
        left_margin = max(0.32, 0.12 + dataset_label_fontsize / 72.0 * 1.2)
    else:
        left_margin = max(
            1.04,
            0.12 + longest_row_label * dataset_label_fontsize / 72.0 * 0.5,
        )
    right_margin = 0.04
    available_width = figure_width - left_margin - right_margin
    side_legend_ratio = 1.65
    if legend_position in ('left', 'right'):
        panel_width = available_width / (image_column_count + side_legend_ratio)
        width_ratios = (
            [side_legend_ratio] + [1.0] * image_column_count
            if legend_position == 'left'
            else [1.0] * image_column_count + [side_legend_ratio]
        )
    else:
        panel_width = available_width / image_column_count
        width_ratios = [1.0] * image_column_count

    headers = ([] if not include_input else ['Input']) + ['Ground Truth'] + list(model_names)
    formatted_headers = [_publication_column_header(header) for header in headers]
    header_line_count = max(header.count('\n') + 1 for header in formatted_headers)
    header_height = header_line_count * header_fontsize / 72.0 * 1.2 + 0.05
    bottom_margin = 0.04
    panel_gap = 0.02
    group_gap = 0.02 + row_spacing

    resolved_bounds = []
    row_styles = []
    zoom_heights = []
    full_heights = []
    legend_heights = []
    legend_columns = []
    for row_index, (row, bbox_spec) in enumerate(zip(rows, zoom_bboxes)):
        bounds = _resolve_publication_zoom_bbox(
            bbox_spec,
            row['image'],
            row_index,
            row['dataset_name'],
        )
        resolved_bounds.append(bounds)
        cmap, norm, labels = _discrete_cmap(
            int(row['num_classes']),
            row.get('class_labels'),
        )
        legend_ncol, legend_height = _publication_legend_geometry(
            labels,
            legend_fontsize,
            legend_position,
        )
        x0, y0, x1, y1 = bounds
        zoom_height = panel_width * (y1 - y0) / float(x1 - x0)
        full_height = panel_width * _publication_image_aspect(row['image'])
        if legend_position in ('left', 'right'):
            extra_height = max(0.0, legend_height - (zoom_height + panel_gap + full_height))
            zoom_height += extra_height / 2.0
            full_height += extra_height / 2.0
        zoom_heights.append(zoom_height)
        full_heights.append(full_height)
        legend_heights.append(legend_height)
        legend_columns.append(legend_ncol)
        row_styles.append((cmap, norm, labels))

    height_ratios = []
    zoom_slots = []
    full_slots = []
    legend_slots = []
    for row_index in range(num_rows):
        zoom_slots.append(len(height_ratios))
        height_ratios.append(zoom_heights[row_index])
        height_ratios.append(panel_gap)
        full_slots.append(len(height_ratios))
        height_ratios.append(full_heights[row_index])
        if legend_position == 'bottom':
            legend_slots.append(len(height_ratios))
            height_ratios.append(legend_heights[row_index])
        else:
            legend_slots.append(None)
        if row_index < num_rows - 1:
            height_ratios.append(group_gap)

    content_height = sum(height_ratios)
    figure_height = max(1.0, header_height + content_height + bottom_margin)
    fig = plt.figure(figsize=(figure_width, figure_height))
    fig.patch.set_facecolor('white')
    grid = fig.add_gridspec(
        len(height_ratios),
        len(width_ratios),
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        left=left_margin / figure_width,
        right=1.0 - right_margin / figure_width,
        bottom=bottom_margin / figure_height,
        top=1.0 - header_height / figure_height,
        wspace=0.04,
        hspace=0.0,
    )

    image_column_start = 1 if legend_position == 'left' else 0
    zoom_axes = []
    full_axes = []
    legend_axes = []
    for row_index in range(num_rows):
        zoom_axes.append([
            fig.add_subplot(grid[zoom_slots[row_index], image_column_start + column_index])
            for column_index in range(image_column_count)
        ])
        full_axes.append([
            fig.add_subplot(grid[full_slots[row_index], image_column_start + column_index])
            for column_index in range(image_column_count)
        ])
        if legend_position == 'bottom':
            legend_axes.append(fig.add_subplot(grid[legend_slots[row_index], :]))
        else:
            legend_column = 0 if legend_position == 'left' else image_column_count
            legend_axes.append(
                fig.add_subplot(grid[zoom_slots[row_index]:full_slots[row_index] + 1, legend_column])
            )

    for column_index, header in enumerate(formatted_headers):
        zoom_axes[0][column_index].set_title(
            header,
            pad=2.0,
            fontsize=header_fontsize,
            fontweight='bold' if header_bold else 'normal',
        )

    for row_index, row in enumerate(rows):
        bounds = resolved_bounds[row_index]
        cmap, norm, labels = row_styles[row_index]
        zoom_image = _crop_publication_array(row['image'], bounds)
        zoom_gt = _crop_publication_array(row['gt'], bounds)
        zoom_predictions = [
            _crop_publication_array(prediction, bounds)
            for prediction in row['predictions']
        ]
        _draw_publication_panel_row(
            zoom_axes[row_index],
            zoom_image,
            zoom_gt,
            zoom_predictions,
            cmap,
            norm,
            prediction_alpha,
            draw_gt_boundary_on_ground_truth,
            boundary_linewidth,
        )
        _draw_publication_panel_row(
            full_axes[row_index],
            row['image'],
            row['gt'],
            row['predictions'],
            cmap,
            norm,
            prediction_alpha,
            draw_gt_boundary_on_ground_truth,
            boundary_linewidth,
            roi_bounds=bounds,
        )

        group_bounds = grid[
            zoom_slots[row_index]:full_slots[row_index] + 1,
            image_column_start,
        ].get_position(fig)
        row_letter = chr(ord('a') + row_index)
        fig.text(
            (left_margin - 0.07) / figure_width,
            (group_bounds.y0 + group_bounds.y1) / 2.0,
            "({}) {}".format(row_letter, _publication_dataset_label(row['dataset_name'])),
            ha='center' if dataset_label_orientation == 'vertical' else 'right',
            va='center',
            fontsize=dataset_label_fontsize,
            fontweight='bold' if dataset_label_bold else 'normal',
            rotation=90 if dataset_label_orientation == 'vertical' else 0,
        )

        legend_ax = legend_axes[row_index]
        legend_ax.axis('off')
        handles = [
            Patch(facecolor=cmap(class_index), edgecolor='none', label=labels[class_index])
            for class_index in range(1, int(row['num_classes']))
        ]
        if handles:
            legend_ax.legend(
                handles=handles,
                loc='center' if legend_position == 'bottom' else 'center left',
                frameon=False,
                ncol=legend_columns[row_index],
                borderaxespad=0.0,
                handlelength=0.9,
                handleheight=0.8,
                handletextpad=0.35,
                columnspacing=0.65,
                labelspacing=0.25,
                fontsize=legend_fontsize,
            )

    if save_path is not None:
        _save_publication_grid(fig, save_path, dpi, save_pdf_and_png)
    plt.show()


def _plot_publication_grid(
    rows: List[Dict],
    model_names: List[str],
    include_input: bool,
    save_path: Optional[str],
    prediction_alpha: float,
    draw_gt_boundary_on_ground_truth: bool,
    boundary_linewidth: float,
    dataset_label_fontsize: float,
    header_fontsize: float,
    legend_fontsize: float,
    row_spacing: float,
    legend_position: str,
    zoom_bboxes,
    dpi: int,
    save_pdf_and_png: bool,
    dataset_label_bold: bool = False,
    header_bold: bool = False,
    dataset_label_orientation: str = 'horizontal',
    zoom_inset: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not 0.0 <= prediction_alpha <= 1.0:
        raise ValueError(
            "prediction_alpha must be between 0.0 and 1.0, got {}".format(prediction_alpha)
        )
    if dpi < 1:
        raise ValueError("dpi must be a positive integer, got {}".format(dpi))
    if row_spacing < 0.0:
        raise ValueError("row_spacing must be non-negative, got {}".format(row_spacing))
    if legend_position not in ('left', 'right', 'bottom'):
        raise ValueError(
            "legend_position must be one of left, right, or bottom, got '{}'".format(legend_position)
        )
    if dataset_label_orientation not in ('horizontal', 'vertical'):
        raise ValueError(
            "dataset_label_orientation must be horizontal or vertical, got '{}'".format(
                dataset_label_orientation
            )
        )
    font_sizes = {
        'dataset_label_fontsize': dataset_label_fontsize,
        'header_fontsize': header_fontsize,
        'legend_fontsize': legend_fontsize,
    }
    invalid_font_sizes = [name for name, value in font_sizes.items() if value <= 0]
    if invalid_font_sizes:
        raise ValueError("{} must be positive".format(", ".join(invalid_font_sizes)))

    num_rows = len(rows)
    if zoom_bboxes is not None and len(zoom_bboxes) != num_rows:
        raise ValueError(
            "zoom_bboxes must contain exactly one entry per row ({} entries for {} rows).".format(
                len(zoom_bboxes),
                num_rows,
            )
        )
    if zoom_inset and zoom_bboxes is None:
        raise ValueError("zoom_inset requires zoom_bboxes.")
    image_column_count = len(model_names) + 1 + int(include_input)
    font_settings = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Nimbus Sans', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': header_fontsize,
        'axes.labelsize': 9,
        'legend.fontsize': legend_fontsize,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }

    with plt.rc_context(font_settings):
        if zoom_bboxes is not None and not zoom_inset:
            _plot_publication_grid_with_zoom(
                rows=rows,
                model_names=model_names,
                include_input=include_input,
                save_path=save_path,
                prediction_alpha=prediction_alpha,
                draw_gt_boundary_on_ground_truth=draw_gt_boundary_on_ground_truth,
                boundary_linewidth=boundary_linewidth,
                dataset_label_fontsize=dataset_label_fontsize,
                header_fontsize=header_fontsize,
                legend_fontsize=legend_fontsize,
                row_spacing=row_spacing,
                legend_position=legend_position,
                dpi=dpi,
                save_pdf_and_png=save_pdf_and_png,
                zoom_bboxes=zoom_bboxes,
                dataset_label_bold=dataset_label_bold,
                header_bold=header_bold,
                dataset_label_orientation=dataset_label_orientation,
            )
            return

        inset_bounds = None
        if zoom_inset:
            inset_bounds = [
                _resolve_publication_zoom_bbox(
                    bbox_spec,
                    row['image'],
                    row_index,
                    row['dataset_name'],
                )
                for row_index, (row, bbox_spec) in enumerate(zip(rows, zoom_bboxes))
            ]

        figure_width = 7.16
        longest_row_label = max(
            len("({}) {}".format(chr(ord('a') + index), _publication_dataset_label(row['dataset_name'])))
            for index, row in enumerate(rows)
        )
        if dataset_label_orientation == 'vertical':
            left_margin = max(0.32, 0.12 + dataset_label_fontsize / 72.0 * 1.2)
        else:
            estimated_row_label_width = (
                0.12 + longest_row_label * dataset_label_fontsize / 72.0 * 0.5
            )
            left_margin = max(1.04, estimated_row_label_width)
        right_margin = 0.04
        available_width = figure_width - left_margin - right_margin
        side_legend_ratio = 1.65
        if legend_position in ('left', 'right'):
            panel_width = available_width / (image_column_count + side_legend_ratio)
            width_ratios = (
                [side_legend_ratio] + [1.0] * image_column_count
                if legend_position == 'left'
                else [1.0] * image_column_count + [side_legend_ratio]
            )
        else:
            panel_width = available_width / image_column_count
            width_ratios = [1.0] * image_column_count

        headers = ([] if not include_input else ['Input']) + ['Ground Truth'] + list(model_names)
        formatted_headers = [_publication_column_header(header) for header in headers]
        header_line_count = max(header.count('\n') + 1 for header in formatted_headers)
        header_height = header_line_count * header_fontsize / 72.0 * 1.2 + 0.05
        bottom_margin = 0.04
        minimum_row_gap = 0.02
        effective_row_gap = minimum_row_gap + row_spacing

        row_styles = []
        row_heights = []
        legend_heights = []
        legend_columns = []
        for row in rows:
            cmap, norm, labels = _discrete_cmap(
                int(row['num_classes']),
                row.get('class_labels'),
            )
            legend_ncol, legend_height = _publication_legend_geometry(
                labels,
                legend_fontsize,
                legend_position,
            )
            image_height = panel_width * _publication_image_aspect(row['image'])
            if legend_position in ('left', 'right'):
                row_heights.append(max(image_height, legend_height))
                legend_heights.append(0.0)
            else:
                row_heights.append(image_height)
                legend_heights.append(legend_height)
            legend_columns.append(legend_ncol)
            row_styles.append((cmap, norm, labels))

        height_ratios = []
        image_slots = []
        legend_slots = []
        for row_index in range(num_rows):
            image_slots.append(len(height_ratios))
            height_ratios.append(row_heights[row_index])
            if legend_position == 'bottom':
                legend_slots.append(len(height_ratios))
                height_ratios.append(legend_heights[row_index])
            else:
                legend_slots.append(None)
            if row_index < num_rows - 1:
                height_ratios.append(effective_row_gap)

        content_height = sum(height_ratios)
        figure_height = max(1.0, header_height + content_height + bottom_margin)
        fig = plt.figure(figsize=(figure_width, figure_height))
        fig.patch.set_facecolor('white')
        grid = fig.add_gridspec(
            len(height_ratios),
            len(width_ratios),
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            left=left_margin / figure_width,
            right=1.0 - right_margin / figure_width,
            bottom=bottom_margin / figure_height,
            top=1.0 - header_height / figure_height,
            wspace=0.04,
            hspace=0.0,
        )

        image_column_start = 1 if legend_position == 'left' else 0
        image_axes = []
        legend_axes = []
        for row_index in range(num_rows):
            row_image_axes = [
                fig.add_subplot(grid[image_slots[row_index], image_column_start + column_index])
                for column_index in range(image_column_count)
            ]
            image_axes.append(row_image_axes)
            if legend_position == 'bottom':
                legend_axes.append(fig.add_subplot(grid[legend_slots[row_index], :]))
            else:
                legend_column = 0 if legend_position == 'left' else image_column_count
                legend_axes.append(fig.add_subplot(grid[image_slots[row_index], legend_column]))

        for column_index, header in enumerate(formatted_headers):
            image_axes[0][column_index].set_title(
                header,
                pad=2.0,
                fontsize=header_fontsize,
                fontweight='bold' if header_bold else 'normal',
            )

        for row_index, row in enumerate(rows):
            image = row['image']
            gt = row['gt']
            predictions = row['predictions']
            cmap, norm, labels = row_styles[row_index]

            column_index = 0
            if include_input:
                _imshow_input(image_axes[row_index][column_index], image)
                column_index += 1

            gt_ax = image_axes[row_index][column_index]
            _imshow_input(gt_ax, image)
            gt_mask = np.ma.masked_where(gt == 0, gt)
            gt_ax.imshow(
                gt_mask.astype(int),
                cmap=cmap,
                norm=norm,
                interpolation='nearest',
            )
            if draw_gt_boundary_on_ground_truth:
                _draw_gt_boundaries(gt_ax, gt, linewidth=boundary_linewidth)
            column_index += 1

            for prediction in predictions:
                pred_ax = image_axes[row_index][column_index]
                _imshow_input(pred_ax, image)
                pred_mask = np.ma.masked_where(prediction == 0, prediction)
                pred_ax.imshow(
                    pred_mask.astype(int),
                    cmap=cmap,
                    norm=norm,
                    interpolation='nearest',
                    alpha=prediction_alpha,
                )
                _draw_gt_boundaries(pred_ax, gt, linewidth=boundary_linewidth)
                column_index += 1

            for image_ax in image_axes[row_index]:
                image_ax.set_xticks([])
                image_ax.set_yticks([])
                for spine in image_ax.spines.values():
                    spine.set_visible(False)

            if inset_bounds is not None:
                bounds = inset_bounds[row_index]
                for image_ax in image_axes[row_index]:
                    _draw_publication_roi_rectangle(
                        image_ax,
                        bounds,
                        boundary_linewidth,
                    )
                _draw_publication_zoom_insets(
                    image_axes[row_index],
                    image,
                    gt,
                    predictions,
                    bounds,
                    cmap,
                    norm,
                    prediction_alpha,
                    draw_gt_boundary_on_ground_truth,
                    boundary_linewidth,
                )

            row_bounds = grid[image_slots[row_index], image_column_start].get_position(fig)
            row_center = (row_bounds.y0 + row_bounds.y1) / 2.0
            row_letter = chr(ord('a') + row_index)
            fig.text(
                (left_margin - 0.07) / figure_width,
                row_center,
                "({}) {}".format(row_letter, _publication_dataset_label(row['dataset_name'])),
                ha='center' if dataset_label_orientation == 'vertical' else 'right',
                va='center',
                fontsize=dataset_label_fontsize,
                fontweight='bold' if dataset_label_bold else 'normal',
                rotation=90 if dataset_label_orientation == 'vertical' else 0,
            )

            legend_ax = legend_axes[row_index]
            legend_ax.axis('off')
            handles = [
                Patch(facecolor=cmap(class_index), edgecolor='none', label=labels[class_index])
                for class_index in range(1, int(row['num_classes']))
            ]
            if handles:
                legend_location = 'center' if legend_position == 'bottom' else 'center left'
                legend_ax.legend(
                    handles=handles,
                    loc=legend_location,
                    frameon=False,
                    ncol=legend_columns[row_index],
                    borderaxespad=0.0,
                    handlelength=0.9,
                    handleheight=0.8,
                    handletextpad=0.35,
                    columnspacing=0.65,
                    labelspacing=0.25,
                    fontsize=legend_fontsize,
                )

        if save_path is not None:
            _save_publication_grid(fig, save_path, dpi, save_pdf_and_png)

        plt.show()


def plot_grid_qualitative_visualization(
    rows: List[Dict],
    model_names: List[str],
    include_input: bool = True,
    figure_title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    legend_mode: str = 'global',
    publication_style: bool = False,
    prediction_alpha: float = 0.55,
    draw_gt_boundary_on_ground_truth: bool = False,
    boundary_linewidth: float = 0.25,
    dataset_label_fontsize: float = 9.0,
    header_fontsize: float = 9.0,
    legend_fontsize: float = 9.0,
    row_spacing: float = 0.0,
    legend_position: str = 'right',
    zoom_bboxes=None,
    publication_dpi: int = 600,
    save_pdf_and_png: bool = False,
    dataset_label_bold: bool = False,
    header_bold: bool = False,
    dataset_label_orientation: str = 'horizontal',
    zoom_inset: bool = False,
):
    if not _ensure_matplotlib():
        raise RuntimeError("Matplotlib is not available. Please install it to use this function.")

    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No rows were provided for grid visualization.")
    if legend_mode not in ('global', 'per_row'):
        raise ValueError("legend_mode must be 'global' or 'per_row', got {}".format(legend_mode))
    if publication_style:
        _plot_publication_grid(
            rows=rows,
            model_names=model_names,
            include_input=include_input,
            save_path=save_path,
            prediction_alpha=prediction_alpha,
            draw_gt_boundary_on_ground_truth=draw_gt_boundary_on_ground_truth,
            boundary_linewidth=boundary_linewidth,
            dataset_label_fontsize=dataset_label_fontsize,
            header_fontsize=header_fontsize,
            legend_fontsize=legend_fontsize,
            row_spacing=row_spacing,
            legend_position=legend_position,
            zoom_bboxes=zoom_bboxes,
            dpi=publication_dpi,
            save_pdf_and_png=save_pdf_and_png,
            dataset_label_bold=dataset_label_bold,
            header_bold=header_bold,
            dataset_label_orientation=dataset_label_orientation,
            zoom_inset=zoom_inset,
        )
        return

    num_rows = len(rows)
    num_models = len(model_names)
    ncols = num_models + 2 if include_input else num_models + 1
    if figsize is None:
        figsize = (max(4.0 * ncols, 8.0), max(3.4 * num_rows, 3.4))

    fig, axes = plt.subplots(num_rows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    last_im = None
    if legend_mode == 'global':
        global_n_classes = max(int(row['num_classes']) for row in rows)
        class_labels = rows[0].get('class_labels')
        cmap, norm, labels = _discrete_cmap(global_n_classes, class_labels)

    for row_index, row in enumerate(rows):
        dataset_name = row['dataset_name']
        row_title = row.get('row_title', dataset_name)
        image = row['image']
        gt = row['gt']
        predictions = row['predictions']
        col_index = 0
        if legend_mode == 'per_row':
            cmap, norm, labels = _discrete_cmap(int(row['num_classes']), row.get('class_labels'))

        if include_input:
            input_ax = axes[row_index, col_index]
            _imshow_input(input_ax, image)
            input_ax.set_ylabel(row_title, fontsize=11)
            if row_index == num_rows - 1:
                input_ax.set_xlabel('Input', fontsize=12)
            input_ax.set_xticks([])
            input_ax.set_yticks([])
            col_index += 1
        else:
            axes[row_index, col_index].set_ylabel(row_title, fontsize=11)

        for model_index, pred in enumerate(predictions):
            pred_ax = axes[row_index, col_index]
            _imshow_input(pred_ax, image)
            pred_mask = np.ma.masked_where(pred == 0, pred)
            last_im = pred_ax.imshow(pred_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
            _draw_gt_boundaries(pred_ax, gt, linewidth=boundary_linewidth)
            if row_index == num_rows - 1:
                pred_ax.set_xlabel(model_names[model_index], fontsize=12)
            pred_ax.set_xticks([])
            pred_ax.set_yticks([])
            col_index += 1

        gt_ax = axes[row_index, col_index]
        _imshow_input(gt_ax, image)
        gt_mask = np.ma.masked_where(gt == 0, gt)
        last_im = gt_ax.imshow(gt_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
        _draw_gt_boundaries(gt_ax, gt, linewidth=boundary_linewidth)
        if row_index == num_rows - 1:
            gt_ax.set_xlabel('Ground Truth', fontsize=12)
        gt_ax.set_xticks([])
        gt_ax.set_yticks([])

        if legend_mode == 'per_row':
            cbar = fig.colorbar(
                last_im,
                ax=axes[row_index, :].ravel().tolist(),
                ticks=np.arange(0, int(row['num_classes']), 1),
                fraction=0.025,
                pad=0.01,
            )
            cbar.ax.set_yticklabels(labels)

    if legend_mode == 'global':
        cbar = fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            ticks=np.arange(0, global_n_classes, 1),
            fraction=0.025,
            pad=0.01,
        )
        cbar.ax.set_yticklabels(labels)

    if figure_title:
        fig.suptitle(figure_title, fontsize=13)

    if save_path is not None:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization grid to: {save_path}")

    plt.show()


def generate_qualitative_visualization(args, model, dataset_name):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if dataset_name == 'Synapse':
            ds_viz = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
            total = len(ds_viz)
            start = max(0, min(args.viz_index, total - 1))
            count = max(1, args.viz_count)
            end = min(total, start + count)

            if count > 1:
                vols, labs, titles = [], [], []
                for i in range(start, end):
                    s = ds_viz[i]
                    vols.append(s['image'])
                    labs.append(s['label'])
                    titles.append(s['case_name'])
                slice_indices = None if args.viz_slice is None else [args.viz_slice] * len(vols)
                default_name = f"Synapse_grid_{start}-{end-1}_{timestamp}.png"
                save_path = resolve_visualization_save_path(args, default_name)
                figure_title = f"Synapse | cases {start}-{end-1}"
                visualize_synapse_batch(
                    model,
                    volumes=vols,
                    labels=labs,
                    slice_indices=slice_indices,
                    img_size=args.img_size,
                    row_titles=titles,
                    figure_title=figure_title,
                    include_input=not args.viz_hide_input,
                    num_slices_to_overlay=args.num_slices_to_overlay,
                    save_path=save_path,
                    boundary_linewidth=args.viz_boundary_linewidth,
                )
            else:
                sample = ds_viz[start]
                title = f"Synapse | case: {sample['case_name']}"
                default_name = f"Synapse_{sample['case_name']}_{timestamp}.png"
                save_path = resolve_visualization_save_path(args, default_name)
                visualize_synapse_sample(
                    model,
                    volume=sample['image'],
                    label=sample['label'],
                    slice_index=args.viz_slice,
                    img_size=args.img_size,
                    figure_title=title,
                    include_input=not args.viz_hide_input,
                    num_slices_to_overlay=args.num_slices_to_overlay,
                    save_path=save_path,
                    boundary_linewidth=args.viz_boundary_linewidth,
                )
        elif dataset_name in ['Cataract1k', 'EndoVis2018']:
            viz_split = "val" if dataset_name == 'Cataract1k' else "test"
            ds_viz = args.Dataset(base_dir=args.volume_path, split=viz_split)
            total = len(ds_viz)
            start = max(0, min(args.viz_index, total - 1))
            count = max(1, args.viz_count)
            end = min(total, start + count)
            viz_dataset_label = 'Cataract1K' if dataset_name == 'Cataract1k' else 'EndoVis2018'

            if count > 1:
                imgs, labs, titles = [], [], []
                for i in range(start, end):
                    s = ds_viz[i]
                    imgs.append(s['image'])
                    labs.append(s['label'])
                    titles.append(s['case_name'])
                default_name = f"{dataset_name}_grid_{start}-{end-1}_{timestamp}.png"
                save_path = resolve_visualization_save_path(args, default_name)
                figure_title = f"{viz_dataset_label} | cases {start}-{end-1}"
                visualize_cataract_batch(
                    model,
                    images=imgs,
                    labels=labs,
                    img_size=args.img_size,
                    class_labels=args.class_names,
                    row_titles=titles,
                    figure_title=figure_title,
                    include_input=not args.viz_hide_input,
                    num_slices_to_overlay=args.num_slices_to_overlay,
                    save_path=save_path,
                    boundary_linewidth=args.viz_boundary_linewidth,
                )
            else:
                sample = ds_viz[start]
                title = f"{viz_dataset_label} | case: {sample['case_name']}"
                default_name = f"{dataset_name}_{sample['case_name']}_{timestamp}.png"
                save_path = resolve_visualization_save_path(args, default_name)
                visualize_cataract_sample(
                    model,
                    image=sample['image'],
                    label=sample['label'],
                    img_size=args.img_size,
                    class_labels=args.class_names,
                    figure_title=title,
                    include_input=not args.viz_hide_input,
                    num_slices_to_overlay=args.num_slices_to_overlay,
                    save_path=save_path,
                    boundary_linewidth=args.viz_boundary_linewidth,
                )
    except Exception as e:
        raise RuntimeError(f"Visualization failed due to: {e}")
