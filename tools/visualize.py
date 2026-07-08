#!/usr/bin/env python3
"""Generate qualitative visualizations for a trained checkpoint."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import shlex
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.dataset_acdc import ACDC_Dataset  # noqa: E402
from datasets.dataset_cataract import Cataract1kDataset  # noqa: E402
from datasets.dataset_endovis2018 import EndoVis2018Dataset  # noqa: E402
from datasets.dataset_synapse import Synapse_dataset  # noqa: E402
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg  # noqa: E402
from networks.vit_seg_modeling import VisionTransformer as ViT_seg  # noqa: E402
from src.test_helpers import build_test_arg_parser, parse_test_args  # noqa: E402
from src.visualize import (  # noqa: E402
    GRID_ONLY_ARG_NAMES,
    build_grid_prediction_entry,
    generate_qualitative_visualization,
    plot_grid_qualitative_visualization,
    resolve_visualization_save_path,
)


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

VALID_GRID_DATASETS = tuple(DATASET_CONFIG.keys())


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
    config_vit.drop_se_block = getattr(args, 'drop_se_block', False)
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


def _strip_grid_only_args(args):
    for name in GRID_ONLY_ARG_NAMES:
        if hasattr(args, name):
            delattr(args, name)


def _validate_grid_args(args):
    if args.dataset not in VALID_GRID_DATASETS:
        raise ValueError(
            "Invalid --dataset '{}' for grid visualization. Valid choices are: {}.".format(
                args.dataset,
                ", ".join(VALID_GRID_DATASETS),
            )
        )
    if args.grid_rows < 1 or args.grid_rows > 20:
        raise ValueError("--grid_rows must be between 1 and 20.")
    if args.grid_cols < 1 or args.grid_cols > 8:
        raise ValueError("--grid_cols must be between 1 and 8.")
    if args.grid_indices is not None:
        if len(args.grid_indices) != args.grid_rows:
            raise ValueError(
                "--grid_indices must contain exactly one index per grid row "
                "({} indices for {} rows).".format(len(args.grid_indices), args.grid_rows)
            )
        negative_indices = [index for index in args.grid_indices if index < 0]
        if negative_indices:
            raise ValueError(
                "--grid_indices values must be non-negative, got: {}.".format(
                    ", ".join(str(index) for index in negative_indices)
                )
            )

    ckpts = list(args.grid_ckpts or [])
    if len(ckpts) < 1:
        raise ValueError("--grid_ckpts is required when --viz_view grid is used.")
    if len(ckpts) > 8:
        raise ValueError("--grid_ckpts must contain at most 8 checkpoints.")
    if len(ckpts) != args.grid_cols:
        raise ValueError(
            "--grid_cols ({}) must match the number of --grid_ckpts entries ({}).".format(
                args.grid_cols,
                len(ckpts),
            )
        )

    if args.grid_model_names is None:
        model_names = ["model{}".format(index + 1) for index in range(len(ckpts))]
    else:
        model_names = list(args.grid_model_names)
        if len(model_names) != len(ckpts):
            raise ValueError(
                "--grid_model_names must contain exactly one name per --grid_ckpts entry "
                "({} names for {} checkpoints).".format(len(model_names), len(ckpts))
            )

    return args.dataset, ckpts, model_names


def _validate_multiple_datasets_args(args):
    if args.viz_legend_mode != 'per_row':
        raise ValueError("--viz_view multiple_datasets requires --viz_legend_mode per_row.")
    if args.viz_hide_input:
        raise ValueError("--viz_hide_input cannot be used with --viz_view multiple_datasets; the input column is required.")

    datasets = list(args.multi_datasets or [])
    if len(datasets) < 1:
        raise ValueError("--multi_datasets is required when --viz_view multiple_datasets is used.")
    if len(datasets) > 4:
        raise ValueError("--multi_datasets must contain at most 4 datasets.")

    invalid_datasets = [dataset for dataset in datasets if dataset not in VALID_GRID_DATASETS]
    if invalid_datasets:
        raise ValueError(
            "Invalid --multi_datasets value(s): {}. Valid choices are: {}.".format(
                ", ".join(invalid_datasets),
                ", ".join(VALID_GRID_DATASETS),
            )
        )

    duplicate_datasets = sorted({dataset for dataset in datasets if datasets.count(dataset) > 1})
    if duplicate_datasets:
        raise ValueError(
            "--multi_datasets must not contain duplicates, got: {}.".format(
                ", ".join(duplicate_datasets)
            )
        )

    if args.grid_rows != len(datasets):
        raise ValueError(
            "--grid_rows ({}) must match the number of --multi_datasets entries ({}).".format(
                args.grid_rows,
                len(datasets),
            )
        )

    ckpts = list(args.multi_ckpts or [])
    if len(ckpts) < 1:
        raise ValueError("--multi_ckpts is required when --viz_view multiple_datasets is used.")
    if len(ckpts) > 8:
        raise ValueError("--multi_ckpts must contain at most 8 model columns.")
    if args.grid_cols != len(ckpts):
        raise ValueError(
            "--grid_cols ({}) must match the number of --multi_ckpts entries ({}).".format(
                args.grid_cols,
                len(ckpts),
            )
        )

    if args.multi_model_names is None:
        model_names = ["model{}".format(index + 1) for index in range(len(ckpts))]
    else:
        model_names = list(args.multi_model_names)
        if len(model_names) != len(ckpts):
            raise ValueError(
                "--multi_model_names must contain exactly one name per --multi_ckpts entry "
                "({} names for {} checkpoint columns).".format(len(model_names), len(ckpts))
            )

    dataset_arg_map = _parse_multi_dataset_args(args.multi_dataset_args, datasets)
    checkpoint_maps = [
        _parse_multi_checkpoint_entry(raw_spec, datasets, model_name)
        for raw_spec, model_name in zip(ckpts, model_names)
    ]
    return datasets, checkpoint_maps, model_names, dataset_arg_map


def _split_checkpoint_spec(raw_spec):
    if "::" not in raw_spec:
        return raw_spec.strip(), ""
    checkpoint_part, override_part = raw_spec.split("::", 1)
    return checkpoint_part.strip(), override_part.strip()


def _parse_dataset_checkpoint_map(checkpoint_part):
    pieces = [piece.strip() for piece in checkpoint_part.split(",") if piece.strip()]
    if len(pieces) <= 1 or not all("=" in piece for piece in pieces):
        return None

    checkpoint_map = {}
    for piece in pieces:
        dataset_name, checkpoint_value = piece.split("=", 1)
        dataset_name = dataset_name.strip()
        checkpoint_value = checkpoint_value.strip()
        if dataset_name not in VALID_GRID_DATASETS:
            raise ValueError(
                "Invalid dataset key '{}' in --grid_ckpts mapping '{}'. Valid choices are: {}.".format(
                    dataset_name,
                    checkpoint_part,
                    ", ".join(VALID_GRID_DATASETS),
                )
            )
        if not checkpoint_value:
            raise ValueError("Empty checkpoint value for dataset '{}' in --grid_ckpts.".format(dataset_name))
        checkpoint_map[dataset_name] = checkpoint_value
    return checkpoint_map


def _parse_multi_checkpoint_entry(raw_spec, requested_datasets, model_name):
    pieces = [piece.strip() for piece in raw_spec.split(";") if piece.strip()]
    if not pieces:
        raise ValueError("Empty --multi_ckpts entry for model '{}'.".format(model_name))

    checkpoint_map = {}
    for piece in pieces:
        if "=" not in piece:
            raise ValueError(
                "Invalid --multi_ckpts entry for model '{}': '{}'. Expected Dataset=checkpoint::args.".format(
                    model_name,
                    piece,
                )
            )
        dataset_name, checkpoint_spec = piece.split("=", 1)
        dataset_name = dataset_name.strip()
        checkpoint_spec = checkpoint_spec.strip()
        if dataset_name not in VALID_GRID_DATASETS:
            raise ValueError(
                "Invalid dataset key '{}' in --multi_ckpts entry for model '{}'. Valid choices are: {}.".format(
                    dataset_name,
                    model_name,
                    ", ".join(VALID_GRID_DATASETS),
                )
            )
        if dataset_name not in requested_datasets:
            raise ValueError(
                "--multi_ckpts entry for model '{}' includes dataset '{}' that is not in --multi_datasets.".format(
                    model_name,
                    dataset_name,
                )
            )
        if dataset_name in checkpoint_map:
            raise ValueError(
                "--multi_ckpts entry for model '{}' defines dataset '{}' more than once.".format(
                    model_name,
                    dataset_name,
                )
            )
        checkpoint_value, override_text = _split_checkpoint_spec(checkpoint_spec)
        if not checkpoint_value:
            raise ValueError(
                "--multi_ckpts entry for model '{}' has an empty checkpoint for dataset '{}'.".format(
                    model_name,
                    dataset_name,
                )
            )
        checkpoint_map[dataset_name] = (checkpoint_value, override_text)

    missing_datasets = [dataset for dataset in requested_datasets if dataset not in checkpoint_map]
    if missing_datasets:
        raise ValueError(
            "--multi_ckpts entry for model '{}' is missing dataset checkpoint(s): {}.".format(
                model_name,
                ", ".join(missing_datasets),
            )
        )
    return checkpoint_map


def _parse_multi_dataset_arg_entry(raw_spec):
    if "::" not in raw_spec:
        raise ValueError(
            "Invalid --multi_dataset_args entry '{}'. Expected Dataset::--viz_index ...".format(raw_spec)
        )
    dataset_name, override_text = raw_spec.split("::", 1)
    dataset_name = dataset_name.strip()
    override_text = override_text.strip()
    if dataset_name not in VALID_GRID_DATASETS:
        raise ValueError(
            "Invalid dataset key '{}' in --multi_dataset_args. Valid choices are: {}.".format(
                dataset_name,
                ", ".join(VALID_GRID_DATASETS),
            )
        )
    if not override_text:
        raise ValueError("--multi_dataset_args entry for '{}' has no arguments.".format(dataset_name))
    return dataset_name, override_text


def _parse_multi_dataset_args(raw_specs, requested_datasets):
    dataset_args = {}
    for raw_spec in raw_specs or []:
        dataset_name, override_text = _parse_multi_dataset_arg_entry(raw_spec)
        if dataset_name not in requested_datasets:
            raise ValueError(
                "--multi_dataset_args includes dataset '{}' that is not in --multi_datasets.".format(
                    dataset_name
                )
            )
        if dataset_name in dataset_args:
            raise ValueError("--multi_dataset_args defines dataset '{}' more than once.".format(dataset_name))
        dataset_args[dataset_name] = override_text
    return dataset_args


def _apply_dataset_visualization_overrides(dataset_args, override_text, dataset_name):
    if not override_text:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--viz_index', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--viz_slice', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--num_slices_to_overlay', type=int, default=argparse.SUPPRESS)
    try:
        overrides = parser.parse_args(shlex.split(override_text))
    except (SystemExit, ValueError) as exc:
        raise ValueError(
            "Invalid --multi_dataset_args for dataset '{}': {}".format(dataset_name, override_text)
        ) from exc
    for name, value in vars(overrides).items():
        setattr(dataset_args, name, value)


def _apply_checkpoint_overrides(model_args, override_text, model_name):
    if not override_text:
        return
    try:
        override_tokens = shlex.split(override_text)
    except ValueError as exc:
        raise ValueError(
            "Could not parse checkpoint override arguments for '{}': {}".format(model_name, exc)
        ) from exc
    if not override_tokens:
        return

    parser = build_test_arg_parser(
        include_visualization_args=False,
        include_quantization_args=False,
    )
    try:
        parser.parse_args(override_tokens, namespace=model_args)
    except SystemExit as exc:
        raise ValueError(
            "Invalid checkpoint override arguments for '{}': {}".format(
                model_name,
                " ".join(override_tokens),
            )
        ) from exc


def _select_checkpoint_for_dataset(raw_spec, dataset_name):
    checkpoint_part, override_text = _split_checkpoint_spec(raw_spec)
    checkpoint_map = _parse_dataset_checkpoint_map(checkpoint_part)
    if checkpoint_map is None:
        return checkpoint_part, override_text
    if dataset_name not in checkpoint_map:
        raise ValueError(
            "Checkpoint mapping '{}' does not define a checkpoint for dataset '{}'.".format(
                checkpoint_part,
                dataset_name,
            )
        )
    return checkpoint_map[dataset_name], override_text


def _set_checkpoint_args(model_args, checkpoint_value):
    checkpoint_path = Path(checkpoint_value)
    if checkpoint_path.is_absolute() or checkpoint_path.parent != Path("."):
        model_args.ckpt_dir = str(checkpoint_path.parent)
        model_args.ckpt = checkpoint_path.name
    else:
        model_args.ckpt = checkpoint_value


def _prepare_grid_dataset_args(base_args, dataset_name):
    dataset_args = copy.deepcopy(base_args)
    dataset_args.dataset = dataset_name
    dataset_args.volume_path = None
    _prepare_args(dataset_args)
    return dataset_args


def _build_visualization_dataset(dataset_args, dataset_name):
    if dataset_name in ['Synapse', 'ACDC']:
        return dataset_args.Dataset(
            base_dir=dataset_args.volume_path,
            split="test_vol",
            list_dir=dataset_args.list_dir,
        )
    elif dataset_name == 'Cataract1k':
        return dataset_args.Dataset(base_dir=dataset_args.volume_path, split="val")
    elif dataset_name == 'EndoVis2018':
        return dataset_args.Dataset(base_dir=dataset_args.volume_path, split="test")
    raise ValueError("Unsupported dataset for visualization: {}".format(dataset_name))


def _load_grid_samples(dataset_args, dataset_name, row_count, sample_indices=None):
    ds_viz = _build_visualization_dataset(dataset_args, dataset_name)

    total = len(ds_viz)
    if total <= 0:
        raise ValueError("Dataset '{}' has no samples to visualize.".format(dataset_name))
    if total < row_count:
        raise ValueError(
            "Dataset '{}' has only {} samples, but --grid_rows requested {}.".format(
                dataset_name,
                total,
                row_count,
            )
        )
    if sample_indices is None:
        selected_indices = random.sample(range(total), row_count)
    else:
        invalid_indices = [index for index in sample_indices if index >= total]
        if invalid_indices:
            raise ValueError(
                "--grid_indices contains index values outside dataset '{}' range [0, {}]: {}.".format(
                    dataset_name,
                    total - 1,
                    ", ".join(str(index) for index in invalid_indices),
                )
            )
        selected_indices = list(sample_indices)
    return [ds_viz[index] for index in selected_indices]


def _load_single_grid_sample(dataset_args, dataset_name):
    ds_viz = _build_visualization_dataset(dataset_args, dataset_name)
    total = len(ds_viz)
    if total <= 0:
        raise ValueError("Dataset '{}' has no samples to visualize.".format(dataset_name))

    sample_index = int(dataset_args.viz_index)
    if sample_index < 0 or sample_index >= total:
        raise ValueError(
            "--viz_index for dataset '{}' must be in range [0, {}], got {}.".format(
                dataset_name,
                total - 1,
                sample_index,
            )
        )
    return ds_viz[sample_index]


def _observed_num_classes(num_classes, predictions, gt):
    max_class = int(np.max(gt)) if np.size(gt) else 0
    for pred in predictions:
        if np.size(pred):
            max_class = max(max_class, int(np.max(pred)))
    return max(int(num_classes), max_class + 1, 2)


def _sanitize_name(value):
    return str(value).replace("/", "-").replace("\\", "-").replace(" ", "_")


def _grid_default_output_name(dataset_name, model_names):
    dataset_part = _sanitize_name(dataset_name)
    model_part = "-".join(_sanitize_name(name) for name in model_names)
    return "grid_{}_{}.png".format(dataset_part, model_part)


def _multiple_datasets_default_output_name(dataset_names, model_names):
    dataset_part = "-".join(_sanitize_name(name) for name in dataset_names)
    model_part = "-".join(_sanitize_name(name) for name in model_names)
    return "multiple_datasets_{}_{}.png".format(dataset_part, model_part)


def _build_model_for_checkpoint(base_args, dataset_name, checkpoint_value, override_text, model_name):
    model_args = copy.deepcopy(base_args)
    _apply_checkpoint_overrides(model_args, override_text, model_name)
    model_args.dataset = dataset_name
    model_args.volume_path = None
    _prepare_args(model_args)
    _set_checkpoint_args(model_args, checkpoint_value)
    try:
        return model_args, _build_model(model_args)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load checkpoint '{}' for dataset '{}' as model '{}'. "
            "Check that the checkpoint matches dataset num_classes={}, vit_name='{}', "
            "img_size={}, n_skip={}, topk_attn={}, use_gumbel_topk={}, use_ats={}, "
            "use_shsa={}, use_swin={}, and use_efficientnet={}. Original error: {}".format(
                checkpoint_value,
                dataset_name,
                model_name,
                getattr(model_args, 'num_classes', None),
                getattr(model_args, 'vit_name', None),
                getattr(model_args, 'img_size', None),
                getattr(model_args, 'n_skip', None),
                getattr(model_args, 'topk_attn', None),
                getattr(model_args, 'use_gumbel_topk', None),
                getattr(model_args, 'use_ats', None),
                getattr(model_args, 'use_shsa', None),
                getattr(model_args, 'use_swin', None),
                getattr(model_args, 'use_efficientnet', None),
                exc,
            )
        ) from exc


def _build_grid_model(base_args, dataset_name, raw_checkpoint_spec, model_name):
    checkpoint_value, override_text = _select_checkpoint_for_dataset(raw_checkpoint_spec, dataset_name)
    return _build_model_for_checkpoint(
        base_args=base_args,
        dataset_name=dataset_name,
        checkpoint_value=checkpoint_value,
        override_text=override_text,
        model_name=model_name,
    )


def generate_grid_visualization(args):
    dataset_name, checkpoint_specs, model_names = _validate_grid_args(args)
    dataset_args = _prepare_grid_dataset_args(args, dataset_name)
    samples = _load_grid_samples(
        dataset_args,
        dataset_name,
        args.grid_rows,
        sample_indices=args.grid_indices,
    )
    rows = [
        {
            'dataset_name': dataset_name,
            'row_title': sample.get('case_name', '{}_{}'.format(dataset_name, index)),
            'image': None,
            'gt': None,
            'predictions': [],
            'class_labels': dataset_args.class_names,
            'num_classes': dataset_args.num_classes,
        }
        for index, sample in enumerate(samples)
    ]

    for raw_checkpoint_spec, model_name in zip(checkpoint_specs, model_names):
        model_args, model = _build_grid_model(
            base_args=args,
            dataset_name=dataset_name,
            raw_checkpoint_spec=raw_checkpoint_spec,
            model_name=model_name,
        )
        try:
            for row, sample in zip(rows, samples):
                try:
                    entry = build_grid_prediction_entry(
                        model=model,
                        sample=sample,
                        args=model_args,
                        dataset_name=dataset_name,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to generate prediction for dataset '{}' sample '{}' with grid model '{}'. "
                        "Original error: {}".format(dataset_name, row['row_title'], model_name, exc)
                    ) from exc

                if row['image'] is None:
                    row['image'] = entry['image']
                    row['gt'] = entry['gt']
                row['predictions'].append(entry['pred'])
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for row in rows:
        row['num_classes'] = _observed_num_classes(
            dataset_args.num_classes,
            row['predictions'],
            row['gt'],
        )

    default_name = _grid_default_output_name(dataset_name, model_names)
    save_path = resolve_visualization_save_path(args, default_name)
    plot_grid_qualitative_visualization(
        rows=rows,
        model_names=model_names,
        include_input=not args.viz_hide_input,
        figure_title=args.viz_title or "{} qualitative comparison".format(dataset_name),
        save_path=save_path,
        legend_mode=args.viz_legend_mode,
    )


def generate_multiple_datasets_visualization(args):
    dataset_names, checkpoint_maps, model_names, dataset_arg_map = _validate_multiple_datasets_args(args)

    rows = []
    row_payloads = []
    for dataset_name in dataset_names:
        dataset_args = _prepare_grid_dataset_args(args, dataset_name)
        _apply_dataset_visualization_overrides(
            dataset_args,
            dataset_arg_map.get(dataset_name, ""),
            dataset_name,
        )
        sample = _load_single_grid_sample(dataset_args, dataset_name)
        sample_name = sample.get('case_name', '{}_{}'.format(dataset_name, dataset_args.viz_index))
        row = {
            'dataset_name': dataset_name,
            'row_title': "{}: {}".format(dataset_name, sample_name),
            'image': None,
            'gt': None,
            'predictions': [],
            'class_labels': dataset_args.class_names,
            'num_classes': dataset_args.num_classes,
        }
        rows.append(row)
        row_payloads.append((dataset_name, dataset_args, sample, row))

    for checkpoint_map, model_name in zip(checkpoint_maps, model_names):
        for dataset_name, dataset_args, sample, row in row_payloads:
            checkpoint_value, override_text = checkpoint_map[dataset_name]
            model_args, model = _build_model_for_checkpoint(
                base_args=args,
                dataset_name=dataset_name,
                checkpoint_value=checkpoint_value,
                override_text=override_text,
                model_name=model_name,
            )
            try:
                model_args.viz_slice = dataset_args.viz_slice
                model_args.num_slices_to_overlay = dataset_args.num_slices_to_overlay
                try:
                    entry = build_grid_prediction_entry(
                        model=model,
                        sample=sample,
                        args=model_args,
                        dataset_name=dataset_name,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to generate prediction for dataset '{}' sample '{}' with model '{}'. "
                        "Original error: {}".format(dataset_name, row['row_title'], model_name, exc)
                    ) from exc

                if row['image'] is None:
                    row['image'] = entry['image']
                    row['gt'] = entry['gt']
                row['predictions'].append(entry['pred'])
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    for dataset_name, dataset_args, _sample, row in row_payloads:
        row['num_classes'] = _observed_num_classes(
            dataset_args.num_classes,
            row['predictions'],
            row['gt'],
        )

    default_name = _multiple_datasets_default_output_name(dataset_names, model_names)
    save_path = resolve_visualization_save_path(args, default_name)
    plot_grid_qualitative_visualization(
        rows=rows,
        model_names=model_names,
        include_input=True,
        figure_title=args.viz_title or "Multiple dataset qualitative comparison",
        save_path=save_path,
        legend_mode=args.viz_legend_mode,
    )


def main():
    args = parse_test_args(
        include_visualization_args=True,
        include_quantization_args=False,
    )
    _set_reproducibility(args)

    if args.viz_view == 'grid':
        generate_grid_visualization(args)
        return 0

    if args.viz_view == 'multiple_datasets':
        generate_multiple_datasets_visualization(args)
        return 0

    _strip_grid_only_args(args)
    dataset_name = _prepare_args(args)
    net = _build_model(args)
    _configure_logging(args)
    generate_qualitative_visualization(args, net, dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
