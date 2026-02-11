import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom


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
    linewidth: float = 1.0,
    linestyle: str = "--",
) -> None:
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
    _draw_gt_boundaries(pred_ax, gt)
    pred_ax.set_title('Prediction', fontsize=12)
    pred_ax.set_xlabel('X (px)')
    if include_input:
        pred_ax.set_yticklabels([])

    # Ground truth
    _imshow_input(gt_ax, image)
    gt_mask = np.ma.masked_where(gt == 0, gt)
    im2 = gt_ax.imshow(gt_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
    _draw_gt_boundaries(gt_ax, gt)
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
        _draw_gt_boundaries(pred_ax, gt)

        _imshow_input(gt_ax, image)
        gt_mask = np.ma.masked_where(gt == 0, gt)
        im_gt = gt_ax.imshow(gt_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
        _draw_gt_boundaries(gt_ax, gt)

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
    )
