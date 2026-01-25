import os
from typing import List, Optional, Tuple

import numpy as np
import torch
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


def _plot_triplet(
    image: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    n_classes: int,
    class_labels: Optional[List[str]] = None,
    figure_title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
):
    """
    Plot a side-by-side triplet: Input | Prediction | Ground Truth
    """
    if not _ensure_matplotlib():
        raise RuntimeError("Matplotlib is not available. Please install it to use this function.")

    import matplotlib.pyplot as plt

    cmap, norm, labels = _discrete_cmap(n_classes, class_labels)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # Input image (handle grayscale or RGB)
    _imshow_input(axes[0], image)
    axes[0].set_title('Input', fontsize=12)
    axes[0].set_xlabel('X (px)')
    axes[0].set_ylabel('Y (px)')

    # Prediction
    _imshow_input(axes[1], image)
    pred_mask = np.ma.masked_where(pred == 0, pred)
    im1 = axes[1].imshow(pred_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
    _draw_gt_boundaries(axes[1], gt)
    axes[1].set_title('Prediction', fontsize=12)
    axes[1].set_xlabel('X (px)')
    axes[1].set_yticklabels([])

    # Ground truth
    _imshow_input(axes[2], image)
    gt_mask = np.ma.masked_where(gt == 0, gt)
    im2 = axes[2].imshow(gt_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
    _draw_gt_boundaries(axes[2], gt)
    axes[2].set_title('Ground Truth', fontsize=12)
    axes[2].set_xlabel('X (px)')
    axes[2].set_yticklabels([])

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

    fig, axes = plt.subplots(num_rows, 3, figsize=figsize, constrained_layout=True)
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (image, pred, gt) in enumerate(triplets):
        # Input
        _imshow_input(axes[r, 0], image)

        # Prediction and GT
        _imshow_input(axes[r, 1], image)
        pred_mask = np.ma.masked_where(pred == 0, pred)
        im_pred = axes[r, 1].imshow(pred_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
        _draw_gt_boundaries(axes[r, 1], gt)
        _imshow_input(axes[r, 2], image)
        gt_mask = np.ma.masked_where(gt == 0, gt)
        im_gt = axes[r, 2].imshow(gt_mask.astype(int), cmap=cmap, norm=norm, interpolation='nearest')
        _draw_gt_boundaries(axes[r, 2], gt)

        # Titles on top row only
        if r == 0:
            axes[r, 0].set_title('Input', fontsize=12)
            axes[r, 1].set_title('Prediction', fontsize=12)
            axes[r, 2].set_title('Ground Truth', fontsize=12)

        # Row titles (case names)
        if row_titles and r < len(row_titles):
            axes[r, 0].set_ylabel(row_titles[r], fontsize=10)

        # Axes labels
        axes[r, 0].set_xlabel('X (px)')
        axes[r, 1].set_xlabel('X (px)')
        axes[r, 2].set_xlabel('X (px)')
        # Hide y tick labels for middle and right columns to reduce clutter
        axes[r, 1].set_yticklabels([])
        axes[r, 2].set_yticklabels([])

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

    # Determine dimensionality
    if volume.ndim == 3:
        D, H, W = volume.shape
        if slice_index is None:
            slice_index = D // 2
        img2d = volume[slice_index]
        gt2d = label[slice_index] if label.ndim == 3 else label
    elif volume.ndim == 2:
        img2d = volume
        gt2d = label
        H, W = img2d.shape
    else:
        raise ValueError("Synapse volume must be 2D or 3D array")

    # Resize to model input
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

    # Resize prediction back to original size
    if (h, w) != (img_size, img_size):
        pred = zoom(pred_small, (h / img_size, w / img_size), order=0)
    else:
        pred = pred_small

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
        # Determine slice for this volume
        s_idx = None
        if slice_indices is not None and idx < len(slice_indices):
            s_idx = slice_indices[idx]

        if volume.ndim == 3:
            D, H, W = volume.shape
            if s_idx is None:
                s_idx = D // 2
            img2d = volume[s_idx]
            gt2d = label[s_idx] if label.ndim == 3 else label
        elif volume.ndim == 2:
            img2d = volume
            gt2d = label
            H, W = img2d.shape
        else:
            raise ValueError("Synapse volume must be 2D or 3D array")

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
    )
