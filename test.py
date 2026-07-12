import logging
import os
import json
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from src.benchmark import benchmark_segmentation_model, build_benchmark_loader
from src.quantize import (
    SEViTSegQuantizer,
    collect_inc_awq_calib_inputs,
    require_official_awq_runtime,
)
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_cataract import Cataract1kDataset
from datasets.dataset_acdc import ACDC_Dataset
from datasets.dataset_endovis2018 import EndoVis2018Dataset
from utils import (
    test_single_volume,
    test_single_frame_present_classes,
    _safe_nanmean,
    _make_json_safe,
    model_size_mb_benchmark,
    runtime_memory_mb_benchmark,
)
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datetime import datetime
from src.test_helpers import (
    parse_test_args,
    _case_group_name,
    _case_group_sort_key,
    _class_label,
    _extract_acdc_voxelspacing_zyx,
    _fallback_acdc_voxelspacing_zyx,
    _mean_metric_array,
)


PRESENT_CLASS_FRAME_EVAL_DATASETS = {'EndoVis2018', 'Cataract1k'}


def inference(args, model, test_save_path=None):
    if args.dataset in ['Synapse', 'ACDC']:
        db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    elif args.dataset in PRESENT_CLASS_FRAME_EVAL_DATASETS:
        db_test = args.Dataset(base_dir=args.volume_path, split="test",)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    all_metrics = []
    class_names = getattr(args, "class_names", None)

    if args.dataset in PRESENT_CLASS_FRAME_EVAL_DATASETS:
        frame_metrics_all = []
        per_class_metrics_all = []
        case_group_frame_metrics = {}
        case_group_per_class_metrics = {}
        class_presence_counts = np.zeros(args.num_classes, dtype=np.int64)
        false_positive_absent_class_counts = np.zeros(args.num_classes, dtype=np.int64)
        discounted_frames = 0
        normalize_present_class_eval = getattr(
            args,
            "normalize_present_class_eval",
            getattr(args, "normalize_endovis_eval", False),
        )

        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            result_i = test_single_frame_present_classes(
                image,
                label,
                model,
                classes=args.num_classes,
                patch_size=[args.img_size, args.img_size],
                test_save_path=None,
                case=case_name,
                z_spacing=args.z_spacing,
                normalize=normalize_present_class_eval,
            )
            metric_i = result_i["per_class_metrics"]
            frame_metrics_i = result_i["frame_metrics"]
            present_class_ids_i = result_i["present_class_ids"]
            false_positive_absent_class_ids_i = result_i["false_positive_absent_class_ids"]

            frame_metrics_all.append(frame_metrics_i)
            per_class_metrics_all.append(metric_i)
            if result_i["discounted_frame"]:
                discounted_frames += 1
            for class_id in present_class_ids_i:
                class_presence_counts[class_id] += 1
            for class_id in false_positive_absent_class_ids_i:
                false_positive_absent_class_counts[class_id] += 1

            case_group_name = _case_group_name(case_name)
            case_group_frame_metrics.setdefault(case_group_name, []).append(frame_metrics_i)
            case_group_per_class_metrics.setdefault(case_group_name, []).append(metric_i)

            logging.info(
                'idx %d case %s frame_present_mean_dice %f frame_present_mean_hd95 %f frame_present_mean_iou %f present_class_ids %s' %
                (
                    i_batch,
                    case_name,
                    frame_metrics_i[0],
                    frame_metrics_i[1],
                    frame_metrics_i[2],
                    present_class_ids_i,
                )
            )
            if args.verbose:
                break

        if not frame_metrics_all:
            raise RuntimeError("No metrics were collected during inference.")

        official_frame_stack = np.stack(frame_metrics_all, axis=0)
        performance = _safe_nanmean(official_frame_stack[:, 0])
        mean_hd95 = _safe_nanmean(official_frame_stack[:, 1])
        mean_iou = _safe_nanmean(official_frame_stack[:, 2])

        per_class_stack = np.stack(per_class_metrics_all, axis=0)
        per_class_mean = _mean_metric_array(per_class_stack)
        per_class_metrics = {}
        logging.info("Diagnostic per-class means are not the official headline metric.")
        for i in range(1, args.num_classes):
            class_metrics = per_class_mean[i - 1]
            class_label = _class_label(class_names, i)
            logging.info(
                'Diagnostic Mean class %s (idx %d) mean_dice %f mean_hd95 %f mean_iou %f' %
                (class_label, i, class_metrics[0], class_metrics[1], class_metrics[2])
            )
            per_class_metrics[class_label] = {
                'dice': float(class_metrics[0]),
                'hd95': float(class_metrics[1]),
                'iou': float(class_metrics[2]),
            }

        per_case_group_metrics = {}
        for case_group_name in sorted(case_group_frame_metrics, key=_case_group_sort_key):
            case_group_stack = np.stack(case_group_frame_metrics[case_group_name], axis=0)
            case_group_dice = _safe_nanmean(case_group_stack[:, 0])
            case_group_hd95 = _safe_nanmean(case_group_stack[:, 1])
            case_group_iou = _safe_nanmean(case_group_stack[:, 2])
            case_group_per_class_mean = _mean_metric_array(
                np.stack(case_group_per_class_metrics[case_group_name], axis=0)
            )
            case_group_per_class = {}
            for i in range(1, args.num_classes):
                class_metrics = case_group_per_class_mean[i - 1]
                class_label = _class_label(class_names, i)
                case_group_per_class[class_label] = {
                    'dice': float(class_metrics[0]),
                    'hd95': float(class_metrics[1]),
                    'iou': float(class_metrics[2]),
                }

            logging.info(
                'Mean case group %s frames %d mean_dice %f mean_hd95 %f mean_iou %f' %
                (
                    case_group_name,
                    len(case_group_frame_metrics[case_group_name]),
                    case_group_dice,
                    case_group_hd95,
                    case_group_iou,
                )
            )
            per_case_group_metrics[case_group_name] = {
                'mean_dice': float(case_group_dice),
                'mean_hd95': float(case_group_hd95),
                'mean_iou': float(case_group_iou),
                'num_frames': len(case_group_frame_metrics[case_group_name]),
                'per_class_diagnostic': case_group_per_class,
            }

        class_presence_counts_dict = {
            _class_label(class_names, i): int(class_presence_counts[i])
            for i in range(1, args.num_classes)
        }
        false_positive_absent_class_counts_dict = {
            _class_label(class_names, i): int(false_positive_absent_class_counts[i])
            for i in range(1, args.num_classes)
        }
        num_frames = len(frame_metrics_all)
        evaluated_frames = num_frames - discounted_frames
        protocol = (
            "EndoVis2018_RSS_frame_present_background_excluded"
            if args.dataset == 'EndoVis2018'
            else "{}_frame_present_background_excluded".format(args.dataset)
        )
        logging.info(
            "%s present-class official-style performance: mean_dice %f, mean_hd95 %f, mean_iou %f",
            args.dataset,
            performance,
            mean_hd95,
            mean_iou,
        )
        logging.info(
            "%s present-class evaluated frames: %d | discounted frames: %d | total frames: %d",
            args.dataset,
            evaluated_frames,
            discounted_frames,
            num_frames,
        )
        logging.info("%s class presence counts: %s", args.dataset, json.dumps(class_presence_counts_dict))
        logging.info(
            "%s false-positive absent-class counts: %s",
            args.dataset,
            json.dumps(false_positive_absent_class_counts_dict),
        )
        print("Testing Finished!")
        results = {
            "mean_dice": float(performance),
            "mean_hd95": float(mean_hd95),
            "mean_iou": float(mean_iou),
            "protocol": protocol,
            "per_class_diagnostic": per_class_metrics,
            "per_case_group": per_case_group_metrics,
            "class_presence_counts": class_presence_counts_dict,
            "false_positive_absent_class_counts": false_positive_absent_class_counts_dict,
            "discounted_frames": int(discounted_frames),
            "evaluated_frames": int(evaluated_frames),
            "num_frames": int(num_frames),
        }
        if args.dataset == 'EndoVis2018':
            results["per_sequence"] = per_case_group_metrics
        return results

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        case_voxelspacing_zyx = None
        if args.dataset == "ACDC":
            case_voxelspacing_zyx = _fallback_acdc_voxelspacing_zyx(args, case_name)
            logging.info("ACDC case %s HD95 voxelspacing_zyx=%s", case_name, case_voxelspacing_zyx)
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=None, case=case_name, z_spacing=args.z_spacing,
                                      dataset=args.dataset, voxelspacing=case_voxelspacing_zyx)
        metric_i = np.array(metric_i, dtype=np.float32)
        all_metrics.append(metric_i)
        with np.errstate(invalid="ignore"):
            case_dice = np.nanmean(metric_i[:, 0])
            case_hd95 = np.nanmean(metric_i[:, 1])
            case_iou = np.nanmean(metric_i[:, 2])
        logging.info(
            'idx %d case %s mean_dice %f mean_hd95 %f mean_iou %f' %
            (i_batch, case_name, case_dice, case_hd95, case_iou)
        )
        if args.verbose:
            break;  # for debugging, run only one batch
        
        

    if not all_metrics:
        raise RuntimeError("No metrics were collected during inference.")

    metrics_stack = np.stack(all_metrics, axis=0)
    with np.errstate(invalid="ignore"):
        mean_metrics = np.nanmean(metrics_stack, axis=0)

    class_names = getattr(args, "class_names", None)
    per_class_metrics = {}
    for i in range(1, args.num_classes):
        class_metrics = mean_metrics[i - 1]
        class_label = (
            class_names[i]
            if class_names is not None and i < len(class_names)
            else f"class_{i}"
        )
        logging.info(
            'Mean class %s (idx %d) mean_dice %f mean_hd95 %f mean_iou %f' %
            (class_label, i, class_metrics[0], class_metrics[1], class_metrics[2])
        )
        per_class_metrics[class_label] = {
            'dice': float(class_metrics[0]),
            'hd95': float(class_metrics[1]),
            'iou': float(class_metrics[2]),
        }
    with np.errstate(invalid="ignore"):
        performance = np.nanmean(mean_metrics[:, 0])
        mean_hd95 = np.nanmean(mean_metrics[:, 1])
        mean_iou = np.nanmean(mean_metrics[:, 2])
    logging.info(
        'Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_iou : %f' %
        (performance, mean_hd95, mean_iou)
    )
    print("Testing Finished!")
    return {
        'mean_dice': float(performance),
        'mean_hd95': float(mean_hd95),
        'mean_iou': float(mean_iou),
        'per_class': per_class_metrics,
    }


def main():
    args = parse_test_args()

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

    dataset_config = {
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
            'list_dir': None,  # Not needed for ACDC
            'num_classes': 4,  # Background (0), RV (1), Myo (2), LV (3)
            'z_spacing': 5,  # NIfTI save metadata only; HD95 uses per-case spacing or --acdc_zspacing fallback.
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
            'list_dir': None,  # Not needed for Cataract1k
            'num_classes': 5,  # Background (0), Pupil (1), Cornea (2), Lens (3), Instruments (4)
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
    
    dataset_name = args.dataset
    if args.volume_path is None:
        args.volume_path = dataset_config[dataset_name]['volume_path']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.class_names = dataset_config[dataset_name].get('class_names')
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    if args.fold_id < 0 or args.fold_id >= 5:
        raise ValueError("fold_id must be between 0 and 4")
    if args.se_calib_only and not args.use_se_block:
        raise ValueError("--se_calib_only requires --use_se_block.")
    if args.quant_backend == "inc_awq" and args.saliency_source != "activation":
        raise ValueError(
            "--quant_backend inc_awq requires --saliency_source activation; "
            "SE-auxiliary saliency is supported only by custom_w4."
        )
    if args.saliency_source == "se_aux":
        if args.quant_backend != "custom_w4":
            raise ValueError("--saliency_source se_aux requires --quant_backend custom_w4.")
        if not args.use_se_block:
            raise ValueError("--saliency_source se_aux requires --use_se_block.")
        if not args.se_calib_only:
            raise ValueError("--saliency_source se_aux requires --se_calib_only.")
    if args.quantize and args.quant_backend == "custom_w4":
        require_official_awq_runtime()
    
    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.verbose = args.verbose
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.use_se_block = args.use_se_block
    config_vit.se_calib_only = args.se_calib_only
    config_vit.drop_se_block = args.drop_se_block
    config_vit.gumbel_sampling_mode = args.gumbel_sampling_mode
    
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.num_heads is not None:
        config_vit.transformer.num_heads = args.num_heads
    if args.num_layers is not None:
        config_vit.transformer.num_layers = args.num_layers
    
    # pass the args use_shsa to the config_vit
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
    
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    #get checkpoint path
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt)
    net.load_state_dict(torch.load(ckpt_path))

    fp32_reference_size = model_size_mb_benchmark(
        net,
        exclude_prefixes=("transformer.encoder.SELayer",)
        if args.se_calib_only
        else None,
    )

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+args.ckpt+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(args.ckpt)
    if args.use_se_block:
        se_layers = getattr(net.transformer.encoder, "SELayer", None)
        encoder_layers = getattr(net.transformer.encoder, "layer", [])
        se_gates_found = se_layers is not None and len(se_layers) == len(encoder_layers)
        logging.info(
            "SE diagnostics: se_calib_only=%s drop_se_block=%s se_gates_found=%s.",
            bool(args.se_calib_only),
            bool(args.drop_se_block),
            bool(se_gates_found),
        )
        if args.quantize:
            logging.info(
                "SE post-quantization drop planned: %s.",
                bool(args.se_calib_only),
            )


    if args.quantize:
        if args.dataset in ["Synapse", "ACDC"]:
            db_calib = args.Dataset(
                base_dir=args.volume_path,
                split="test_vol",
                list_dir=args.list_dir,
            )
        elif args.dataset in ["Cataract1k", "EndoVis2018"]:
            db_calib = args.Dataset(
                base_dir=args.volume_path,
                split="test",
            )
        else:
            raise ValueError(f"Unsupported dataset for AWQ calibration: {args.dataset}")

        calib_loader = DataLoader(
            db_calib,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )
        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            ValueError("No CUDA device available for quantization. Please ensure that a CUDA-capable GPU is available.")
            
        net.eval().to(device)

        logging.info("Quantization backend: %s", args.quant_backend)
        logging.info("Saliency source: %s", args.saliency_source)

        if args.quant_backend == "custom_w4":
            logging.info(
                "Running official AWQ WQLinear W4A16 quantization: "
                "n_calib_batches=%d group_size=%d saliency_source=%s",
                args.quantize_calibrate_batch_size,
                128,
                args.saliency_source,
            )
            quantizer = SEViTSegQuantizer(
                model=net,
                calib_loader=calib_loader,
                w_bit=4,
                q_group_size=128,
                n_calib_batches=args.quantize_calibrate_batch_size,
                device=device,
                args=args,
                saliency_source=args.saliency_source,
            )
            net = quantizer.quantize()
            net.eval().to(device)
            logging.info(
                "Official AWQ WQLinear W4A16 quantization finished with saliency_source=%s.",
                args.saliency_source,
            )
            logging.info("Model quantized successfully.")
        elif args.quant_backend == "inc_awq":
            from neural_compressor.torch.quantization import AWQConfig, prepare, convert

            calib_inputs = collect_inc_awq_calib_inputs(
                args=args,
                calib_loader=calib_loader,
                max_forwards=args.quantize_calibrate_batch_size,
                chunk_size=min(max(1, int(args.batch_size)), 8),
            )

            if not calib_inputs:
                raise RuntimeError("No calibration inputs were collected for INC AWQ.")

            quant_config = AWQConfig(
                dtype="int",
                bits=4,
                group_size=128,
                use_sym=False,
                use_auto_scale=True,
                use_auto_clip=True,
                folding=False,
            )

            # Minimal/safe first pass:
            # Quantize only Linear layers inside transformer.encoder.layer.
            # This avoids accidentally quantizing SE MLPs or other helper Linear layers.
            for name, module in net.named_modules():
                if isinstance(module, nn.Linear) and not name.startswith("transformer.encoder.layer."):
                    quant_config.set_local(name, AWQConfig(dtype="fp32"))

            example_inputs = calib_inputs[0].to(device, non_blocking=True)

            logging.info(
                "Running Intel Neural Compressor AWQ: %d calibration forwards, example input shape=%s",
                len(calib_inputs),
                tuple(example_inputs.shape),
            )

            net = prepare(
                net,
                quant_config,
                example_inputs=example_inputs,
            )

            with torch.no_grad():
                for x in calib_inputs:
                    net(x.to(device, non_blocking=True))

            net = convert(net)
            net.eval().to(device)

            logging.info("Intel Neural Compressor AWQ quantization finished.")
            logging.info("Model quantized successfully.")
        else:
            raise ValueError(f"Unsupported quantization backend: {args.quant_backend}")
        
        # Drop only calibration-only SE auxiliary blocks after quantization.
        se_layers = getattr(net.transformer.encoder, "SELayer", None)
        if se_layers is not None:
            if getattr(args, "se_calib_only", False):
                del net.transformer.encoder.SELayer
                net.transformer.encoder.args.drop_se_block = True
                logging.info("Dropped calibration-only SE-auxiliary-block after quantization.")
            else:
                logging.warning(
                    "Keeping active SE blocks after quantization because --se_calib_only is not set. "
                    "Dropping active SE would change the trained segmentation function."
                )

    deployed_model_size = model_size_mb_benchmark(net)

    performance = inference(args, net, test_save_path=None)
    
    # ---------------- Benchmark (runs AFTER inference is done) ----------------
    # For throughput, use a reasonable batch size (fixed HxW works best with cudnn.benchmark)
    # bench_bs = 1 if args.dataset == 'Synapse' else max(1, min(args.batch_size, 16))
    test_loader_bench = build_benchmark_loader(args, batch_size=36, num_workers=0, shuffle=False)

    results = benchmark_segmentation_model(
        model=net,
        test_loader=test_loader_bench,                 # real test samples
        device="cuda" if args.quantize else ("cuda" if torch.cuda.is_available() else "cpu"),
        warmup_steps=20,                               # stabilize kernels
        measure_batches=50,                            # how many batches to time
        single_image_latency_samples=1000,              # B=1 latency percentiles
        enable_cudnn_benchmark=True,                   # True if fixed image size
        autocast=False,                                # set True to benchmark AMP
        quantized_model=args.quantize,
        args=args,
        fp32_reference_size_bytes=fp32_reference_size["total_bytes"],
        deployed_model_size_bytes=deployed_model_size["total_bytes"],
    )

    pretty = "\n" + results.pretty()
    # print(pretty)
    logging.info(pretty)
    
    model_runtime_memory = runtime_memory_mb_benchmark(
        net,
        test_loader=test_loader_bench,
    )
    
    benchmark_dir = args.benchmark_dict + args.dataset
    os.makedirs(benchmark_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date, time = timestamp.split("_")
    with open(os.path.join(benchmark_dir, f"{args.ckpt}_{args.img_size}_{timestamp}.json"), "w") as f:
        accuracy_for_json = {k: _make_json_safe(v) for k, v in performance.items()}
        metrics_for_json = {k: _make_json_safe(v) for k, v in results.metrics.items()}
        notes_for_json = {k: _make_json_safe(v) for k, v in results.notes.items()}
        arguments_for_json = {k: _make_json_safe(v) for k, v in vars(args).items()}  # Include all arguments as a dictionary
        model_size_for_json = {k: _make_json_safe(v) for k, v in deployed_model_size.items()}  # Include model size info
        model_runtime_memory_for_json = {k: _make_json_safe(v) for k, v in model_runtime_memory.items()}  # Include runtime memory info

        json.dump(
            {
                "description": args.description,
                "date_time": date + " " + time,
                "accuracy": accuracy_for_json,
                "metrics": metrics_for_json,
                "notes": notes_for_json,
                "arguments": arguments_for_json,
                "model_size": model_size_for_json,
                "model_runtime_memory": model_runtime_memory_for_json,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
