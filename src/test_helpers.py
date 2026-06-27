import argparse

import numpy as np

from utils import _safe_nanmean


def build_test_arg_parser():
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
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
    parser.add_argument('--fold_id', type=int, default=0,
                        help='ACDC fold id to use when --random_split is set; valid values are 0-4')

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
    parser.add_argument('--use_ats', action='store_true',
                        help='whether to use Adaptive Token Sampling (ATS) for attention')
    parser.add_argument('--benchmark_dict', type=str, default='benchmark/', help='directory to save benchmark results')

    parser.add_argument('--viz', action='store_true', help='show qualitative visualization for a sample')
    parser.add_argument('--viz_index', type=int, default=0, help='dataset index to visualize')
    parser.add_argument('--viz_slice', type=int, default=None, help='slice index for Synapse volumes (default: middle slice)')
    parser.add_argument('--viz_save', type=str, default=None, help='path to save figure (file or directory)')
    parser.add_argument('--viz_out', type=str, default=None,
                        help='output filename for the saved figure (used if --viz_save is a directory or not provided)')
    parser.add_argument('--viz_count', type=int, default=4, help='number of samples to visualize (default: 4)')
    parser.add_argument('--viz_suffix', type=str, default=None,
                        help='suffix to append to the output filename (before extension)')
    parser.add_argument('--viz_hide_input', action='store_true',
                        help='hide input column in visualization (show only prediction and ground truth)')
    parser.add_argument('--num_slices_to_overlay', type=int, default=None,
                        help='number of slices to overlay for visualization (clamped to [2,20])')

    parser.add_argument('--swin_pretrained_path', type=str,
                        default='/data/halyusuf/data/pretrained_backbones/swin/swin_large_patch4_window7_224_22k.pth',
                        help='path to swin pretrained model')

    parser.add_argument('--quantize', action='store_true', help='whether to quantize the model')
    parser.add_argument('--quantize_calibrate_batch_size', type=int, default=8,
                        help='batch size for calibration (default: 8)')
    parser.add_argument('--use_se_block', action='store_true', help='whether to use SE block in the encoder')
    parser.add_argument('--drop_se_block', action='store_true', help='whether to drop SE block during quantization')

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


def parse_test_args():
    args = build_test_arg_parser().parse_args()
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
