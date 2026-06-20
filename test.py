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
from benchmark import benchmark_segmentation_model, build_benchmark_loader
from benchmark_quantize import benchmark_segmentation_quantize_model
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_cataract import Cataract1kDataset
from datasets.dataset_acdc import ACDC_Dataset
from datasets.dataset_endovis2018 import EndoVis2018Dataset
from utils import (
    test_single_volume,
    test_single_volume_endovis2018_rss,
    _safe_nanmean,
    _make_json_safe,
    model_size_mb_benchmark,
    runtime_memory_mb_benchmark,
)
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.quantizer import AWQViTSegQuantizer
from datetime import datetime
from src.test_helpers import (
    parse_test_args,
    _endovis_sequence_name,
    _endovis_sequence_sort_key,
    _class_label,
    _mean_metric_array,
)
from src.visualize import (
    generate_qualitative_visualization,
)


def inference(args, model, test_save_path=None):
    if args.dataset in ['Synapse', 'ACDC']:
        db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, fold_id=args.fold_id)
    elif args.dataset in ['Cataract1k', 'EndoVis2018']:
        db_test = args.Dataset(base_dir=args.volume_path, split="test",)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    all_metrics = []
    class_names = getattr(args, "class_names", None)

    if args.dataset == 'EndoVis2018':
        endovis_frame_metrics = []
        endovis_per_class_metrics = []
        sequence_frame_metrics = {}
        sequence_per_class_metrics = {}
        class_presence_counts = np.zeros(args.num_classes, dtype=np.int64)
        false_positive_absent_class_counts = np.zeros(args.num_classes, dtype=np.int64)
        discounted_frames = 0

        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            result_i = test_single_volume_endovis2018_rss(
                image,
                label,
                model,
                classes=args.num_classes,
                patch_size=[args.img_size, args.img_size],
                test_save_path=test_save_path,
                case=case_name,
                z_spacing=args.z_spacing,
                normalize=getattr(args, "normalize_endovis_eval", False),
            )
            metric_i = result_i["per_class_metrics"]
            frame_metrics_i = result_i["frame_metrics"]
            present_class_ids_i = result_i["present_class_ids"]
            false_positive_absent_class_ids_i = result_i["false_positive_absent_class_ids"]

            endovis_frame_metrics.append(frame_metrics_i)
            endovis_per_class_metrics.append(metric_i)
            if result_i["discounted_frame"]:
                discounted_frames += 1
            for class_id in present_class_ids_i:
                class_presence_counts[class_id] += 1
            for class_id in false_positive_absent_class_ids_i:
                false_positive_absent_class_counts[class_id] += 1

            sequence_name = _endovis_sequence_name(case_name)
            sequence_frame_metrics.setdefault(sequence_name, []).append(frame_metrics_i)
            sequence_per_class_metrics.setdefault(sequence_name, []).append(metric_i)

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

        if not endovis_frame_metrics:
            raise RuntimeError("No metrics were collected during inference.")

        official_frame_stack = np.stack(endovis_frame_metrics, axis=0)
        performance = _safe_nanmean(official_frame_stack[:, 0])
        mean_hd95 = _safe_nanmean(official_frame_stack[:, 1])
        mean_iou = _safe_nanmean(official_frame_stack[:, 2])

        per_class_stack = np.stack(endovis_per_class_metrics, axis=0)
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

        per_sequence_metrics = {}
        for sequence_name in sorted(sequence_frame_metrics, key=_endovis_sequence_sort_key):
            sequence_stack = np.stack(sequence_frame_metrics[sequence_name], axis=0)
            sequence_dice = _safe_nanmean(sequence_stack[:, 0])
            sequence_hd95 = _safe_nanmean(sequence_stack[:, 1])
            sequence_iou = _safe_nanmean(sequence_stack[:, 2])
            sequence_per_class_mean = _mean_metric_array(
                np.stack(sequence_per_class_metrics[sequence_name], axis=0)
            )
            sequence_per_class = {}
            for i in range(1, args.num_classes):
                class_metrics = sequence_per_class_mean[i - 1]
                class_label = _class_label(class_names, i)
                sequence_per_class[class_label] = {
                    'dice': float(class_metrics[0]),
                    'hd95': float(class_metrics[1]),
                    'iou': float(class_metrics[2]),
                }

            logging.info(
                'Mean sequence %s frames %d mean_dice %f mean_hd95 %f mean_iou %f' %
                (
                    sequence_name,
                    len(sequence_frame_metrics[sequence_name]),
                    sequence_dice,
                    sequence_hd95,
                    sequence_iou,
                )
            )
            per_sequence_metrics[sequence_name] = {
                'mean_dice': float(sequence_dice),
                'mean_hd95': float(sequence_hd95),
                'mean_iou': float(sequence_iou),
                'num_frames': len(sequence_frame_metrics[sequence_name]),
                'per_class_diagnostic': sequence_per_class,
            }

        class_presence_counts_dict = {
            _class_label(class_names, i): int(class_presence_counts[i])
            for i in range(1, args.num_classes)
        }
        false_positive_absent_class_counts_dict = {
            _class_label(class_names, i): int(false_positive_absent_class_counts[i])
            for i in range(1, args.num_classes)
        }
        num_frames = len(endovis_frame_metrics)
        evaluated_frames = num_frames - discounted_frames
        logging.info(
            "EndoVis2018 RSS official-style performance: mean_dice %f, mean_hd95 %f, mean_iou %f",
            performance,
            mean_hd95,
            mean_iou,
        )
        logging.info(
            "EndoVis2018 RSS evaluated frames: %d | discounted frames: %d | total frames: %d",
            evaluated_frames,
            discounted_frames,
            num_frames,
        )
        logging.info("EndoVis2018 class presence counts: %s", json.dumps(class_presence_counts_dict))
        logging.info(
            "EndoVis2018 false-positive absent-class counts: %s",
            json.dumps(false_positive_absent_class_counts_dict),
        )
        print("Testing Finished!")
        return {
            "mean_dice": float(performance),
            "mean_hd95": float(mean_hd95),
            "mean_iou": float(mean_iou),
            "protocol": "EndoVis2018_RSS_frame_present_background_excluded",
            "per_class_diagnostic": per_class_metrics,
            "per_sequence": per_sequence_metrics,
            "class_presence_counts": class_presence_counts_dict,
            "false_positive_absent_class_counts": false_positive_absent_class_counts_dict,
            "discounted_frames": int(discounted_frames),
            "evaluated_frames": int(evaluated_frames),
            "num_frames": int(num_frames),
        }

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, dataset=args.dataset)
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
            'z_spacing': 5, #TODO: confirm the z-spacing for ACDC and update accordingly.
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
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.class_names = dataset_config[dataset_name].get('class_names')
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    if args.fold_id < 0 or args.fold_id >= 5:
        raise ValueError("fold_id must be between 0 and 4")
    
    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.verbose = args.verbose
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.use_se_block = args.use_se_block
    config_vit.drop_se_block = args.drop_se_block
    
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
    
    if args.adaptive_attn_threshold > 0.0 and (args.use_shsa or args.topk_attn > 0.0):
        raise ValueError("The --adaptive_attn_threshold argument is mutually exclusive with --use_shsa and --topk_attn > 0.0.")
    
    config_vit.topk_attn = args.topk_attn
    config_vit.use_ats = args.use_ats
    config_vit.use_gumbel_topk = args.use_gumbel_topk
    config_vit.use_shsa = args.use_shsa
    config_vit.use_alternate_shsa = args.use_alternate_shsa
    config_vit.adaptive_attn_threshold = args.adaptive_attn_threshold
    config_vit.use_efficientnet = args.use_efficientnet
    config_vit.use_swin = args.use_swin
    
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    # net.load_state_dict(torch.load(snapshot, weights_only=True))
    # net = nn.DataParallel(net) #added by me to overcome testing error problem
    #get checkpoint path
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt)
    net.load_state_dict(torch.load(ckpt_path))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)


    if args.quantize:
        if args.dataset == 'Synapse':
            db_calib = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
        elif args.dataset in ['Cataract1k', 'EndoVis2018']:
            db_calib = args.Dataset(base_dir=args.volume_path, split="test",)

        calib_loader = DataLoader(db_calib, batch_size=1, shuffle=False, num_workers=1)
 
        quantizer = AWQViTSegQuantizer(
            model = net,
            calib_loader = calib_loader,
            w_bit=4,
            q_group_size=128,
            n_calib_batches=args.quantize_calibrate_batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            args=args,
        )
        
        logging.info(f"Calibrating model on {len(calib_loader)} batches from test set.")
        net = quantizer.quantize()  
        logging.info(f"Model quantized successfully.")
        
        #drop SE block after quantization
        se_layers = getattr(net.transformer.encoder, "SELayer", None)
        if se_layers is not None:
            del net.transformer.encoder.SELayer
            net.transformer.encoder.args.drop_se_block = True
            logging.info("Dropped SE-blocks after quantization.")

        
    # Optional qualitative visualization before running full inference
    if args.viz:
        generate_qualitative_visualization(args, net, dataset_name)
        sys.exit(0)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    

    performance = inference(args, net, test_save_path)
    
    # ---------------- Benchmark (runs AFTER inference is done) ----------------
    # For throughput, use a reasonable batch size (fixed HxW works best with cudnn.benchmark)
    # bench_bs = 1 if args.dataset == 'Synapse' else max(1, min(args.batch_size, 16))
    test_loader_bench = build_benchmark_loader(args, batch_size=36, num_workers=0, shuffle=False)

    if args.quantize:
        results = benchmark_segmentation_quantize_model(
            model=net,
            test_loader=test_loader_bench,                 # real test samples
            device="cuda",
            warmup_steps=20,                               # stabilize kernels
            measure_batches=50,                            # how many batches to time
            single_image_latency_samples=200,              # B=1 latency percentiles
            enable_cudnn_benchmark=True,                   # True if fixed image size
            autocast=False,                                 # set True to benchmark AMP
            args=args,
        )
    else:
        results = benchmark_segmentation_model(
            model=net,
            test_loader=test_loader_bench,                 # real test samples
            device="cuda" if torch.cuda.is_available() else "cpu",
            warmup_steps=20,                               # stabilize kernels
            measure_batches=50,                            # how many batches to time
            single_image_latency_samples=200,              # B=1 latency percentiles
            enable_cudnn_benchmark=True,                   # True if fixed image size
            autocast=False,                        # set True to benchmark AMP      
            args=args,    
        )

    pretty = "\n" + results.pretty()
    # print(pretty)
    logging.info(pretty)
    
    model_size = model_size_mb_benchmark(net)
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
        model_size_for_json = {k: _make_json_safe(v) for k, v in model_size.items()}  # Include model size info
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
