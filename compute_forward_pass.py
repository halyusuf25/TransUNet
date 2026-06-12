#!/usr/bin/env python3
# compute_forward_pass.py
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent
_tmp_candidates = [
    ROOT_DIR / ".tmp",
    Path.home() / ".cache" / "transunet_tmp",
]
for candidate in _tmp_candidates:
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        continue
    else:
        os.environ.setdefault("TMPDIR", str(candidate))
        break

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

DATASETS_DIR = ROOT_DIR / "datasets"
NETWORKS_DIR = ROOT_DIR / "networks"

for path in (ROOT_DIR, DATASETS_DIR, NETWORKS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from component_timer import time_model_components  # noqa: E402

try:
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover
    from dataset_synapse import Synapse_dataset, RandomGenerator  # noqa: E402

try:
    from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg  # noqa: E402
    from networks.vit_seg_modeling import VisionTransformer as ViT_seg  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover
    from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg  # noqa: E402
    from vit_seg_modeling import VisionTransformer as ViT_seg  # noqa: E402


def build_synapse_model(
    device: torch.device,
    vit_name: str,
    img_size: int,
    num_classes: int,
    n_skip: int,
    vit_patch_size: int,
    use_shsa: bool = False,
    use_swin: bool = False,
    use_efficientnet: bool = False,
    topk_attn: float = 0.0,
    num_heads: Optional[int] = None,
    num_layers: Optional[int] = None,
    checkpoint: Optional[Path] = None,
) -> ViT_seg:
    """Instantiate TransUNet for Synapse and (optionally) load weights."""
    config = CONFIGS_ViT_seg[vit_name]
    config.n_classes = num_classes
    config.n_skip = n_skip
    config.use_shsa = use_shsa
    config.use_swin = use_swin
    config.use_efficientnet = use_efficientnet
    config.use_alternate_shsa = False  # expose via CLI if you need it
    config.topk_attn = topk_attn
    config.patches.size = (vit_patch_size, vit_patch_size)
    if vit_name.find('R50') != -1:
        grid = img_size // vit_patch_size
        config.patches.grid = (grid, grid)
    if num_heads is not None:
        config.transformer.num_heads = num_heads
    if num_layers is not None:
        config.transformer.num_layers = num_layers

    model = ViT_seg(config, img_size=img_size, num_classes=num_classes).to(device)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] missing keys while loading checkpoint: {missing}")
        if unexpected:
            print(f"[warn] unexpected keys while loading checkpoint: {unexpected}")

    model.eval()
    return model


def build_synapse_loader(
    root_path: Path,
    list_dir: Path,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Return a DataLoader that yields Synapse slices as tensors."""
    if split != "train":
        raise ValueError(
            "Synapse_dataset exposes full volumes for non-train splits. "
            "Use split='train' (2D slices) or add your own volume-to-slice pipeline."
        )

    transform = transforms.Compose([RandomGenerator(output_size=[img_size, img_size])])
    dataset = Synapse_dataset(
        base_dir=str(root_path),
        list_dir=str(list_dir),
        split=split,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure per-component forward times on Synapse.")
    parser.add_argument("--root_path", type=Path,
                        default=Path("/data/halyusuf/data/Synapse/train_npz/"),
                        help="Root directory with Synapse .npz slices.")
    parser.add_argument("--list_dir", type=Path,
                        default=Path("./lists/lists_Synapse"),
                        help="Directory containing train.txt/val.txt lists.")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to benchmark (train uses 2D slices).")
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16",
                        help="Backbone identifier (must exist in CONFIGS).")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input resolution expected by the model.")
    parser.add_argument("--vit_patch_size", type=int, default=16,
                        help="Patch size passed to the ViT config.")
    parser.add_argument("--n_skip", type=int, default=3,
                        help="Number of decoder skip connections.")
    parser.add_argument("--num_classes", type=int, default=9,
                        help="Output channel count for the segmentation head.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for benchmarking.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="DataLoader worker processes.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional path to a trained checkpoint (.pth).")
    parser.add_argument("--warmup_batches", type=int, default=5,
                        help="How many warm-up iterations to discard.")
    parser.add_argument("--measure_batches", type=int, default=25,
                        help="How many batches to time.")
    parser.add_argument("--autocast", action="store_true",
                        help="Enable CUDA autocast during timing.")
    parser.add_argument("--amp_dtype", type=str, default="float16",
                        help="AMP dtype to use when --autocast is set.")
    parser.add_argument("--num_heads", type=int, default=None,
                        help="Override ViT attention heads.")
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Override number of transformer layers.")
    parser.add_argument("--use_shsa", action="store_true",
                        help="Enable single-head self-attention in the config.")
    parser.add_argument("--use_swin", action="store_true",
                        help="Use Swin Transformer backbone.")
    parser.add_argument("--use_efficientnet", action="store_true",
                        help="Use EfficientNet decoder blocks.")
    parser.add_argument("--topk_attn", type=float, default=0.0,
                        help="Top-k attention keep ratio (0 disables).")
    return parser.parse_args()


def resolve_amp_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported amp_dtype '{name}'")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp_dtype)

    loader = build_synapse_loader(
        root_path=args.root_path,
        list_dir=args.list_dir,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_synapse_model(
        device=device,
        vit_name=args.vit_name,
        img_size=args.img_size,
        num_classes=args.num_classes,
        n_skip=args.n_skip,
        vit_patch_size=args.vit_patch_size,
        use_shsa=args.use_shsa,
        use_swin=args.use_swin,
        use_efficientnet=args.use_efficientnet,
        topk_attn=args.topk_attn,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        checkpoint=args.checkpoint,
    )

    timings = time_model_components(
        model=model,
        test_loader=loader,
        device=str(device),
        warmup_batches=args.warmup_batches,
        measure_batches=args.measure_batches,
        autocast=args.autocast,
        amp_dtype=amp_dtype,
    )
    print(timings.pretty())


if __name__ == "__main__":
    main()
