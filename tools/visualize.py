#!/usr/bin/env python3
"""Generate qualitative visualizations for a trained checkpoint."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.dataset_acdc import ACDC_Dataset  # noqa: E402
from datasets.dataset_cataract import Cataract1kDataset  # noqa: E402
from datasets.dataset_endovis2018 import EndoVis2018Dataset  # noqa: E402
from datasets.dataset_synapse import Synapse_dataset  # noqa: E402
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg  # noqa: E402
from networks.vit_seg_modeling import VisionTransformer as ViT_seg  # noqa: E402
from src.test_helpers import parse_test_args  # noqa: E402
from src.visualize import generate_qualitative_visualization  # noqa: E402


DATASET_CONFIG = {
    'Synapse': {
        'Dataset': Synapse_dataset,
        'volume_path': '/data/halyusuf/data/Synapse/test_vol_h5',
        'list_dir': './lists/lists_Synapse',
        'num_classes': 9,
        'z_spacing': 1,
        'class_names': [
            'Background',
            'Aorta',
            'Gallbladder',
            'Kidney(L)',
            'Kidney(R)',
            'Liver',
            'Pancreas',
            'Spleen',
            'Stomach',
        ],
    },
    'ACDC': {
        'Dataset': ACDC_Dataset,
        'volume_path': '/data/halyusuf/data/ACDC/',
        'list_dir': None,
        'num_classes': 4,
        'z_spacing': 5,
        'info': '3D',
        'class_names': [
            'Background',
            'Right Ventricle',
            'Myocardium',
            'Left Ventricle',
        ],
    },
    'Cataract1k': {
        'Dataset': Cataract1kDataset,
        'volume_path': '/data/halyusuf/data/CataractData/',
        'list_dir': None,
        'num_classes': 5,
        'z_spacing': 1,
        'class_names': [
            'Background',
            'Pupil',
            'Cornea',
            'Lens',
            'Instruments',
        ],
    },
    'EndoVis2018': {
        'Dataset': EndoVis2018Dataset,
        'volume_path': '/data/halyusuf/data/EndoVis_2018',
        'list_dir': None,
        'num_classes': 12,
        'z_spacing': 1,
        'class_names': [
            'background-tissue',
            'instrument-shaft',
            'instrument-clasper',
            'instrument-wrist',
            'kidney-parenchyma',
            'covered-kidney',
            'thread',
            'clamps',
            'suturing-needle',
            'suction-instrument',
            'small-intestine',
            'ultrasound-probe',
        ],
    },
}


def _set_reproducibility(args):
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def _prepare_args(args):
    dataset_name = args.dataset
    if args.volume_path is None:
        args.volume_path = DATASET_CONFIG[dataset_name]['volume_path']
    args.num_classes = DATASET_CONFIG[dataset_name]['num_classes']
    args.Dataset = DATASET_CONFIG[dataset_name]['Dataset']
    args.z_spacing = DATASET_CONFIG[dataset_name]['z_spacing']
    args.class_names = DATASET_CONFIG[dataset_name].get('class_names')
    args.list_dir = DATASET_CONFIG[dataset_name]['list_dir']
    args.is_pretrain = True

    if args.fold_id < 0 or args.fold_id >= 5:
        raise ValueError("fold_id must be between 0 and 4")

    args.exp = 'TU_' + dataset_name + str(args.img_size)
    return dataset_name


def _build_model(args):
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.verbose = args.verbose
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.use_se_block = args.use_se_block
    config_vit.drop_se_block = args.drop_se_block
    config_vit.gumbel_sampling_mode = args.gumbel_sampling_mode

    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.num_heads is not None:
        config_vit.transformer.num_heads = args.num_heads
    if args.num_layers is not None:
        config_vit.transformer.num_layers = args.num_layers

    if args.use_alternate_shsa and not args.use_shsa:
        raise ValueError("The --use_alternate_shsa flag requires --use_shsa to be set as well.")

    if args.use_ats and args.topk_attn <= 0.0:
        raise ValueError("The --use_ats flag requires --topk_attn to be greater than 0.0.")

    if args.use_shsa and args.topk_attn > 0.0:
        raise ValueError("The --use_shsa flag is mutually exclusive with --topk_attn > 0.0.")

    if args.use_gumbel_topk and args.topk_attn <= 0.0:
        raise ValueError("The --use_gumbel_topk flag requires --topk_attn to be greater than 0.0.")

    if args.use_ats and args.use_gumbel_topk:
        raise ValueError("--use_ats and --use_gumbel_topk are mutually exclusive.")

    if not isinstance(args.repeated_runs, int) or args.repeated_runs < 1 or args.repeated_runs > 10:
        raise ValueError("The --repeated_runs argument must be an integer between 1 and 10.")

    config_vit.topk_attn = args.topk_attn
    config_vit.use_ats = args.use_ats
    config_vit.use_gumbel_topk = args.use_gumbel_topk
    config_vit.use_shsa = args.use_shsa
    config_vit.use_alternate_shsa = args.use_alternate_shsa
    config_vit.use_efficientnet = args.use_efficientnet
    config_vit.use_swin = args.use_swin

    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt)
    net.load_state_dict(torch.load(ckpt_path))
    return net


def _configure_logging(args):
    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + args.ckpt + ".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(args.ckpt)


def _quantize_if_requested(args, net):
    if args.quantize:
        from networks.quantizer import AWQViTSegQuantizer
        if args.dataset == 'Synapse':
            db_calib = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
        elif args.dataset in ['Cataract1k', 'EndoVis2018']:
            db_calib = args.Dataset(base_dir=args.volume_path, split="test",)

        calib_loader = DataLoader(db_calib, batch_size=1, shuffle=False, num_workers=1)

        quantizer = AWQViTSegQuantizer(
            model=net,
            calib_loader=calib_loader,
            w_bit=4,
            q_group_size=128,
            n_calib_batches=args.quantize_calibrate_batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            args=args,
        )

        logging.info(f"Calibrating model on {len(calib_loader)} batches from test set.")
        net = quantizer.quantize()
        logging.info(f"Model quantized successfully.")

        se_layers = getattr(net.transformer.encoder, "SELayer", None)
        if se_layers is not None:
            del net.transformer.encoder.SELayer
            net.transformer.encoder.args.drop_se_block = True
            logging.info("Dropped SE-blocks after quantization.")
    return net


def main():
    args = parse_test_args(include_visualization_args=True)
    _set_reproducibility(args)
    dataset_name = _prepare_args(args)
    net = _build_model(args)
    _configure_logging(args)
    net = _quantize_if_requested(args, net)
    generate_qualitative_visualization(args, net, dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
