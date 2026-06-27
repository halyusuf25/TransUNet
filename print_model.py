import argparse
import os

import torch

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Synapse",
        choices=["Synapse", "Cataract1k"],
        help="dataset name (used to set default num_classes)",
    )
    parser.add_argument("--num_classes", type=int, default=None, help="output channels")
    parser.add_argument("--img_size", type=int, default=224, help="input patch size")
    parser.add_argument("--n_skip", type=int, default=3, help="number of skip connections")
    parser.add_argument("--vit_name", type=str, default="ViT-B_16", help="vit model name")
    parser.add_argument("--vit_patches_size", type=int, default=16, help="vit patches size")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt/", help="checkpoint dir")
    parser.add_argument("--ckpt", type=str, default="epoch_29.pth", help="checkpoint file name")
    parser.add_argument("--num_heads", type=int, default=None, help="transformer heads override")
    parser.add_argument("--num_layers", type=int, default=None, help="transformer layers override")
    parser.add_argument("--use_shsa", action="store_true", help="use SHSA")
    parser.add_argument("--use_swin", action="store_true", help="use Swin backbone")
    parser.add_argument("--use_efficientnet", action="store_true", help="use EfficientNet decoder")
    parser.add_argument("--use_alternate_shsa", action="store_true", help="use alternate partial attention")
    parser.add_argument("--topk_attn", type=float, default=0.0, help="top-k attention fraction")

    parser.add_argument("--use_se_block", action="store_true", help="use SE block")
    parser.add_argument("--verbose", action="store_true", help="enable verbose config")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device for model (default: auto)",
    )
    return parser.parse_args()


def main():
    args = build_args()

    dataset_config = {
        "Synapse": 9,
        "Cataract1k": 5,
    }
    if args.num_classes is None:
        args.num_classes = dataset_config[args.dataset]

    if args.use_alternate_shsa and not args.use_shsa:
        raise ValueError("The --use_alternate_shsa flag requires --use_shsa to be set as well.")

    if args.use_shsa and args.topk_attn > 0.0:
        raise ValueError("The --use_shsa flag is mutually exclusive with --topk_attn > 0.0.")

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.verbose = args.verbose
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.use_se_block = args.use_se_block
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.num_heads is not None:
        config_vit.transformer.num_heads = args.num_heads
    if args.num_layers is not None:
        config_vit.transformer.num_layers = args.num_layers
    config_vit.topk_attn = args.topk_attn
    config_vit.use_shsa = args.use_shsa
    config_vit.use_alternate_shsa = args.use_alternate_shsa
    config_vit.use_efficientnet = args.use_efficientnet
    config_vit.use_swin = args.use_swin
    if args.vit_name.find("R50") != -1:
        grid_size = int(args.img_size / args.vit_patches_size)
        config_vit.patches.grid = (grid_size, grid_size)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt)
    net.load_state_dict(torch.load(ckpt_path))
    print(net)
    se_layers = getattr(net.transformer.encoder, "SELayer", None)
    if se_layers is not None:
        del net.transformer.encoder.SELayer
        print("Deleted SELayer from the transformer encoder.")
        print("Model architecture after modifications")
        print(net)





if __name__ == "__main__":
    main()
