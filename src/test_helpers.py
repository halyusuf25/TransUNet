import argparse
import logging
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import torch

from networks.attention import ATSAttention, TopkAttention
from utils import _safe_nanmean


logger = logging.getLogger(__name__)


ACDC_SPACING_KEYS = (
    "voxelspacing_zyx",
    "spacing_zyx",
    "voxelspacing",
    "spacing",
    "spacing_mm",
    "pixdim",
    "zooms",
)
ACDC_SPACING_ZYX_KEYS = {"voxelspacing_zyx", "spacing_zyx"}


def convert_token_reduction_checkpoint_state_dict(model, state_dict):
    """Convert legacy packed Top-K/Gumbel/ATS projections for strict loading."""
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "Checkpoint state_dict must be a mapping, got {}.".format(
                type(state_dict).__name__
            )
        )

    checkpoint_keys = list(state_dict.keys())
    if any(not isinstance(key, str) for key in checkpoint_keys):
        raise TypeError("Checkpoint state_dict keys must all be strings.")
    prefixed = [key.startswith("module.") for key in checkpoint_keys]
    if any(prefixed) and not all(prefixed):
        raise ValueError(
            "Checkpoint contains a mixed set of DataParallel 'module.'-prefixed "
            "and unprefixed keys."
        )
    checkpoint_has_module_prefix = bool(prefixed) and all(prefixed)
    converted_state = OrderedDict(
        (
            key[len("module."):] if checkpoint_has_module_prefix else key,
            value,
        )
        for key, value in state_dict.items()
    )
    checkpoint_metadata = getattr(state_dict, "_metadata", None)
    normalized_metadata = None
    if checkpoint_metadata is not None:
        normalized_metadata = OrderedDict()
        for key, value in checkpoint_metadata.items():
            if checkpoint_has_module_prefix:
                if key == "":
                    continue
                if key == "module":
                    key = ""
                elif key.startswith("module."):
                    key = key[len("module."):]
                else:
                    raise ValueError(
                        "Checkpoint metadata contains a mixed set of DataParallel "
                        "'module.'-prefixed and unprefixed module paths."
                    )
            normalized_metadata[key] = value

    core_model = model.module if hasattr(model, "module") else model
    transformer = getattr(core_model, "transformer", None)
    encoder = getattr(transformer, "encoder", None)
    blocks = getattr(encoder, "layer", ())
    token_blocks = [
        (block_idx, block.attn)
        for block_idx, block in enumerate(blocks)
        if isinstance(getattr(block, "attn", None), (TopkAttention, ATSAttention))
    ]

    block_modes = []
    for block_idx, attention in token_blocks:
        prefix = "transformer.encoder.layer.{}.attn.".format(block_idx)
        packed_suffixes = ("qkv.weight", "qkv.bias", "proj.weight", "proj.bias")
        separate_suffixes = (
            "query.weight",
            "query.bias",
            "key.weight",
            "key.bias",
            "value.weight",
            "value.bias",
            "out.weight",
            "out.bias",
        )
        packed = {suffix for suffix in packed_suffixes if prefix + suffix in converted_state}
        separate = {suffix for suffix in separate_suffixes if prefix + suffix in converted_state}
        if packed and separate:
            raise ValueError(
                "Token-reduction encoder block {} contains ambiguous mixed packed "
                "and separate attention projection keys.".format(block_idx)
            )

        has_bias = attention.query.bias is not None
        required_packed = {"qkv.weight", "proj.weight"}
        required_separate = {
            "query.weight",
            "key.weight",
            "value.weight",
            "out.weight",
        }
        if has_bias:
            required_packed.update(("qkv.bias", "proj.bias"))
            required_separate.update(
                ("query.bias", "key.bias", "value.bias", "out.bias")
            )

        if packed:
            missing = sorted(required_packed - packed)
            unexpected_bias = sorted(packed - required_packed)
            if missing or unexpected_bias:
                raise ValueError(
                    "Incomplete packed attention projections for token-reduction encoder "
                    "block {}: missing={}, unexpected={}.".format(
                        block_idx, missing, unexpected_bias
                    )
                )
            mode = "packed"
        else:
            missing = sorted(required_separate - separate)
            unexpected_bias = sorted(separate - required_separate)
            if missing or unexpected_bias:
                raise ValueError(
                    "Incomplete separate attention projections for token-reduction encoder "
                    "block {}: missing={}, unexpected={}.".format(
                        block_idx, missing, unexpected_bias
                    )
                )
            mode = "separate"
        block_modes.append(mode)

        hidden_size = int(attention.embed_dim)

        def validate_shape(suffix, expected_shape):
            tensor = converted_state[prefix + suffix]
            if not torch.is_tensor(tensor) or tuple(tensor.shape) != tuple(expected_shape):
                actual_shape = tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor).__name__
                raise ValueError(
                    "Token-reduction encoder block {} {} must have shape {}, got {}.".format(
                        block_idx, suffix, tuple(expected_shape), actual_shape
                    )
                )

        if mode == "packed":
            validate_shape("qkv.weight", (3 * hidden_size, hidden_size))
            validate_shape("proj.weight", (hidden_size, hidden_size))
            if has_bias:
                validate_shape("qkv.bias", (3 * hidden_size,))
                validate_shape("proj.bias", (hidden_size,))
        else:
            for name in ("query", "key", "value", "out"):
                validate_shape(name + ".weight", (hidden_size, hidden_size))
                if has_bias:
                    validate_shape(name + ".bias", (hidden_size,))

    if len(set(block_modes)) > 1:
        raise ValueError(
            "Checkpoint contains ambiguous mixed packed and separate projection "
            "representations across token-reduction encoder blocks."
        )

    converted_blocks = 0
    if block_modes and block_modes[0] == "packed":
        for block_idx, attention in token_blocks:
            prefix = "transformer.encoder.layer.{}.attn.".format(block_idx)
            q_weight, k_weight, v_weight = converted_state.pop(
                prefix + "qkv.weight"
            ).chunk(3, dim=0)
            converted_state[prefix + "query.weight"] = q_weight
            converted_state[prefix + "key.weight"] = k_weight
            converted_state[prefix + "value.weight"] = v_weight

            if attention.query.bias is not None:
                q_bias, k_bias, v_bias = converted_state.pop(
                    prefix + "qkv.bias"
                ).chunk(3, dim=0)
                converted_state[prefix + "query.bias"] = q_bias
                converted_state[prefix + "key.bias"] = k_bias
                converted_state[prefix + "value.bias"] = v_bias

            converted_state[prefix + "out.weight"] = converted_state.pop(
                prefix + "proj.weight"
            )
            if attention.out.bias is not None:
                converted_state[prefix + "out.bias"] = converted_state.pop(
                    prefix + "proj.bias"
                )
            converted_blocks += 1

    model_state = model.state_dict()
    model_keys = list(model_state.keys())
    model_expects_module_prefix = bool(model_keys) and all(
        key.startswith("module.") for key in model_keys
    )
    if model_expects_module_prefix:
        converted_state = OrderedDict(
            ("module." + key, value) for key, value in converted_state.items()
        )
    if normalized_metadata is not None:
        if model_expects_module_prefix:
            converted_metadata = OrderedDict()
            model_metadata = getattr(model_state, "_metadata", {})
            if "" in model_metadata:
                converted_metadata[""] = model_metadata[""]
            for key, value in normalized_metadata.items():
                prefixed_key = "module" if key == "" else "module." + key
                converted_metadata[prefixed_key] = value
        else:
            converted_metadata = normalized_metadata
        converted_state._metadata = converted_metadata

    logger.info(
        "Converted %d token-reduction encoder block(s) from packed checkpoint projections.",
        converted_blocks,
    )
    return converted_state


def remove_calibration_only_se_branch(model, se_calib_only):
    """Remove the calibration-only SE auxiliary branch before deployment evaluation."""
    core_model = model.module if hasattr(model, "module") else model
    transformer = getattr(core_model, "transformer", None)
    encoder = getattr(transformer, "encoder", None)
    if encoder is None:
        return False

    se_layers = getattr(encoder, "SELayer", None)
    if se_layers is None:
        return False

    encoder_args = getattr(encoder, "args", None)
    model_calibration_only = bool(
        getattr(encoder_args, "se_calib_only", False)
    )
    if not bool(se_calib_only) or not model_calibration_only:
        raise RuntimeError(
            "Refusing to remove transformer.encoder.SELayer because it is not "
            "configured as a calibration-only SE branch. Removing a functionally "
            "active SE block would change the trained segmentation function."
        )

    setattr(encoder_args, "drop_se_block", True)
    del encoder.SELayer

    assert not hasattr(encoder, "SELayer"), (
        "transformer.encoder.SELayer is still present after deployment cleanup."
    )
    module_names = [
        name[len("module."):] if name.startswith("module.") else name
        for name, _module in model.named_modules()
    ]
    assert "transformer.encoder.SELayer" not in module_names, (
        "transformer.encoder.SELayer remains registered after deployment cleanup."
    )
    remaining_state_keys = [
        key
        for key in model.state_dict()
        if (key[len("module."):] if key.startswith("module.") else key).startswith(
            "transformer.encoder.SELayer."
        )
    ]
    assert not remaining_state_keys, (
        "SE auxiliary state_dict entries remain after deployment cleanup: {}".format(
            remaining_state_keys
        )
    )
    logger.info(
        "Removed the calibration-only SE auxiliary branch before deployment evaluation."
    )
    return True


def _spacing_value_to_vector(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().reshape(-1)
    if isinstance(value, np.ndarray):
        return value.reshape(-1)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _spacing_value_to_vector(value[0])
        return np.concatenate([_spacing_value_to_vector(item) for item in value])
    return np.asarray(value).reshape(-1)


def _normalize_acdc_spacing_zyx(value, key, case_name):
    spacing = _spacing_value_to_vector(value).astype(np.float32)
    if key == "pixdim" and spacing.size >= 4:
        spacing = spacing[1:4]
    else:
        spacing = spacing[:3]
    if spacing.size != 3:
        raise ValueError(
            "ACDC case {} spacing key {} must provide 3 spatial values, got {}".format(
                case_name, key, spacing.tolist()
            )
        )
    if key not in ACDC_SPACING_ZYX_KEYS:
        spacing = spacing[::-1]
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(
            "ACDC case {} spacing key {} must be positive finite values, got {}".format(
                case_name, key, spacing.tolist()
            )
        )
    return tuple(float(v) for v in spacing)


def _extract_acdc_voxelspacing_zyx(sampled_batch, case_name):
    for key in ACDC_SPACING_KEYS:
        if key in sampled_batch:
            return _normalize_acdc_spacing_zyx(sampled_batch[key], key, case_name)
    return None


def _fallback_acdc_voxelspacing_zyx(args, case_name):
    z_spacing = float(getattr(args, "acdc_zspacing", 5.0))
    if not np.isfinite(z_spacing) or z_spacing <= 0:
        raise ValueError(
            "ACDC case {} fallback --acdc_zspacing must be a positive finite value, got {}".format(
                case_name, z_spacing
            )
        )
    return (z_spacing, 1.0, 1.0)


def add_quantization_args(parser):
    parser.add_argument('--quantize', action='store_true', help='whether to quantize the model')
    parser.add_argument('--quantize_calibrate_batch_size', type=int, default=8,
                        help='batch size for calibration (default: 8)')
    parser.add_argument(
        '--quant_backend',
        type=str,
        default='custom_w4',
        choices=['custom_w4', 'inc_awq'],
        help=(
            'Quantization implementation. custom_w4 uses the official MIT-HAN-Lab '
            'AWQ WQLinear W4A16 CUDA backend; inc_awq uses Intel Neural Compressor AWQ.'
        ),
    )
    parser.add_argument(
        '--saliency_source',
        type=str,
        default='activation',
        choices=['activation', 'se_aux'],
        help=(
            'Channel saliency source for custom_w4 quantization. '
            'activation uses ordinary activation magnitude; '
            'se_aux uses activation magnitude weighted by the trained '
            'calibration-only SE estimator.'
        ),
    )
    parser.add_argument('--drop_se_block', action='store_true', help='whether to drop SE block during quantization')
    return parser


def build_test_arg_parser(include_visualization_args=False, include_quantization_args=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_path', type=str,
                        default=None,
                        help='root dir for validation volume data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--num_classes', type=int,
                        default=None, help='output channel of network')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')

    parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size per gpu')
    parser.add_argument('--img_size', '--image_size', dest='img_size',
                        type=int, default=224, help='input patch size of network input')
    parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
    parser.add_argument('--fold_id', type=int, default=0,
                        help='ACDC fold id to use when --random_split is set; valid values are 0-4')
    parser.add_argument('--acdc_zspacing', type=float, default=5.0,
                        help='fallback ACDC z-spacing in mm when per-case spacing metadata is unavailable')

    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

    parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt/', help='directory to save trained model')
    parser.add_argument('--ckpt', type=str, default='epoch_29.pth', help='name of the checkpoint file')
    parser.add_argument('--num_heads', type=int,
                        default=None, help='number of attention heads (default value sets in the imported CONFIGS_ViT_seg)')
    parser.add_argument('--num_layers', type=int,
                        default=None, help='number of transformer layers (default value sets in the imported CONFIGS_ViT_seg)')
    parser.add_argument('--use_shsa', action='store_true',
                        help='whether to use single-head self-attention (SHSA) or the default multi-head self-attention')
    parser.add_argument('--use_swin', action='store_true',
                        help='whether to use Swin Transformer as the backbone')
    parser.add_argument('--use_efficientnet', action='store_true',
                        help='whether to use EfficientNet as the decoder')
    parser.add_argument('--use_alternate_shsa', action='store_true',
                        help='whether to use alternate partial attention')
    parser.add_argument('--topk_attn', type=float, default=0.0,
                        help='if >0.0, use top-k attention (fraction of k) instead of full attention (mutually exclusive with --use_shsa)')
    parser.add_argument('--use_gumbel_topk', action='store_true',
                        help='whether to use Gumbel-Softmax sampling for Top-k attention (it has to be used with --topk_attn > 0.0)')
    parser.add_argument('--gumbel_sampling_mode', type=str, default='dist', choices=['dist', 'manual'],
                        help='mode for Gumbel-Softmax sampling (default: dist, which uses the distribution directly; sample uses sampling from the distribution)')
    parser.add_argument('--use_ats', action='store_true',
                        help='whether to use Adaptive Token Sampling (ATS) for attention')
    parser.add_argument('--benchmark_dict', type=str, default='benchmark/', help='directory to save benchmark results')
    parser.add_argument('--repeated_runs', type=int, default=1,
                        help='number of repeated throughput/latency benchmark runs')

    if include_visualization_args:
        from src.visualize import add_visualization_args
        add_visualization_args(parser)

    parser.add_argument('--swin_pretrained_path', type=str,
                        default='/data/halyusuf/data/pretrained_backbones/swin/swin_large_patch4_window7_224_22k.pth',
                        help='path to swin pretrained model')

    if include_quantization_args:
        add_quantization_args(parser)
    parser.add_argument('--use_se_block', action='store_true', help='whether to use SE block in the encoder')
    parser.add_argument('--se_calib_only', action='store_true',
                        help='use SE as a calibration-only auxiliary block that emits gates without modifying hidden states')

    parser.add_argument('--description', type=str, default='no description for this test run',
                        help='description for the experiment')
    parser.add_argument('--verbose', action='store_true',
                        help='whether to print detailed debug information during inference')
    parser.add_argument(
        "--normalize_present_class_eval",
        "--normalize_endovis_eval",
        dest="normalize_present_class_eval",
        action="store_true",
        help="Apply ImageNet normalization during present-class frame evaluation. Use only if training used the same normalization.",
    )
    return parser


def parse_test_args(include_visualization_args=False, include_quantization_args=True):
    args = build_test_arg_parser(
        include_visualization_args=include_visualization_args,
        include_quantization_args=include_quantization_args,
    ).parse_args()
    args.normalize_endovis_eval = args.normalize_present_class_eval
    return args


def _case_group_name(case_name):
    return str(case_name).split('_', 1)[0]


def _case_group_sort_key(case_group_name):
    prefix = "seq"
    if case_group_name.startswith(prefix) and case_group_name[len(prefix):].isdigit():
        return 0, int(case_group_name[len(prefix):])
    return 1, case_group_name


def _endovis_sequence_name(case_name):
    return _case_group_name(case_name)


def _endovis_sequence_sort_key(sequence_name):
    return _case_group_sort_key(sequence_name)


def _class_label(class_names, class_id):
    if class_names is not None and class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"


def _mean_metric_array(metric_stack):
    metric_stack = np.asarray(metric_stack, dtype=np.float32)
    mean_metrics = np.full(metric_stack.shape[1:], np.nan, dtype=np.float32)
    for index in np.ndindex(mean_metrics.shape):
        mean_metrics[index] = _safe_nanmean(metric_stack[(slice(None),) + index])
    return mean_metrics
