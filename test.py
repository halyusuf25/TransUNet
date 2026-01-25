import argparse
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
from utils import test_single_volume, _make_json_safe, model_size_mb_benchmark, runtime_memory_mb_benchmark
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.quantizer import AWQViTSegQuantizer
from datetime import datetime
from visualize import (
    visualize_synapse_sample,
    visualize_cataract_sample,
    visualize_synapse_batch,
    visualize_cataract_batch,
)

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=None, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
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
parser.add_argument('--adaptive_attn_threshold', type=float,
                    default=0.0, help='threshold for adaptive attention to select tokens (0.0 means not using adaptive attention)')
parser.add_argument('--use_se_block', action='store_true', help='whether to use SE block in the encoder')

##################### visualization arguments ####################
parser.add_argument('--viz', action='store_true', help='show qualitative visualization for a sample')
parser.add_argument('--viz_index', type=int, default=0, help='dataset index to visualize')
parser.add_argument('--viz_slice', type=int, default=None, help='slice index for Synapse volumes (default: middle slice)')
parser.add_argument('--viz_save', type=str, default=None, help='path to save figure (file or directory)')
parser.add_argument('--viz_out', type=str, default=None, help='output filename for the saved figure (used if --viz_save is a directory or not provided)')
parser.add_argument('--viz_count', type=int, default=4, help='number of samples to visualize (default: 4)')
parser.add_argument('--viz_suffix', type=str, default=None, help='suffix to append to the output filename (before extension)')

##########swin config arguments##########
parser.add_argument('--swin_pretrained_path', type=str,
                    default='/data/shared/pretrained_backbones/swin/swin_large_patch4_window7_224_22k.pth', help='path to swin pretrained model')
#########################################

################ Quantization arguments ################
parser.add_argument('--quantize', action='store_true', help='whether to quantize the model')
parser.add_argument('--quantize_calibrate_batch_size', type=int, default=8, help='batch size for calibration (default: 8)')
#####################################################

#######additional arguments for debugging#########
parser.add_argument('--description', type=str, default='no description for this test run', help='description for the experiment')
parser.add_argument('--verbose', action='store_true', 
                    help='whether to print detailed debug information during inference')
###############################################
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    if args.dataset == 'Synapse':
        db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    elif args.dataset == 'Cataract1k':
        db_test = args.Dataset(base_dir=args.volume_path, split="test",)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    all_metrics = []
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


if __name__ == "__main__":

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
            'volume_path': '/data/shared/project_TransUNet/data/Synapse/test_vol_h5',
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
        'Cataract1k': {
            'Dataset': Cataract1kDataset,
            'volume_path': '/data/shared/CataractData/',
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
    }
    
    dataset_name = args.dataset
    args.volume_path = dataset_config[dataset_name]['volume_path']
    if args.num_classes is None:
        args.num_classes = dataset_config[dataset_name]['num_classes']
    
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.class_names = dataset_config[dataset_name].get('class_names')
    if dataset_name == 'Synapse':
        args.list_dir = dataset_config[dataset_name]['list_dir']
        


    args.is_pretrain = True

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
    
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.num_heads is not None:
        config_vit.transformer.num_heads = args.num_heads
    if args.num_layers is not None:
        config_vit.transformer.num_layers = args.num_layers
    
    # pass the args use_shsa to the config_vit
    if args.use_alternate_shsa and not args.use_shsa:
        raise ValueError("The --use_alternate_shsa flag requires --use_shsa to be set as well.")
    
    if args.use_shsa and args.topk_attn > 0.0:
        raise ValueError("The --use_shsa flag is mutually exclusive with --topk_attn > 0.0.")
    
    if args.adaptive_attn_threshold > 0.0 and (args.use_shsa or args.topk_attn > 0.0):
        raise ValueError("The --adaptive_attn_threshold argument is mutually exclusive with --use_shsa and --topk_attn > 0.0.")
    
    config_vit.topk_attn = args.topk_attn
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
        elif args.dataset == 'Cataract1k':
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
        
    # Optional qualitative visualization before running full inference
    if args.viz:
        try:
            # Helper to resolve save path from args.viz_save and args.viz_out
            def resolve_save_path(default_name: str):
                img_exts = {'.png', '.jpg', '.jpeg', '.pdf', '.svg', '.tif', '.tiff'}
                base = args.viz_save
                name = args.viz_out or default_name
                if args.viz_suffix:
                    base_name, ext = os.path.splitext(name)
                    name = f"{base_name}{args.viz_suffix}{ext}"
                if base is None:
                    return os.path.join('qualitative', name)
                ext = os.path.splitext(base)[1].lower()
                if ext in img_exts:
                    return base
                return os.path.join(base, name)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if dataset_name == 'Synapse':
                ds_viz = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
                total = len(ds_viz)
                start = max(0, min(args.viz_index, total - 1))
                count = max(1, args.viz_count)
                end = min(total, start + count)

                if count > 1:
                    vols, labs, titles = [], [], []
                    for i in range(start, end):
                        s = ds_viz[i]
                        vols.append(s['image'])
                        labs.append(s['label'])
                        titles.append(s['case_name'])
                    slice_indices = None if args.viz_slice is None else [args.viz_slice] * len(vols)
                    default_name = f"Synapse_grid_{start}-{end-1}_{timestamp}.png"
                    save_path = resolve_save_path(default_name)
                    figure_title = f"Synapse | cases {start}-{end-1}"
                    visualize_synapse_batch(
                        net,
                        volumes=vols,
                        labels=labs,
                        slice_indices=slice_indices,
                        img_size=args.img_size,
                        row_titles=titles,
                        figure_title=figure_title,
                        save_path=save_path,
                    )
                else:
                    sample = ds_viz[start]
                    title = f"Synapse | case: {sample['case_name']}"
                    default_name = f"Synapse_{sample['case_name']}_{timestamp}.png"
                    save_path = resolve_save_path(default_name)
                    visualize_synapse_sample(
                        net,
                        volume=sample['image'],
                        label=sample['label'],
                        slice_index=args.viz_slice,
                        img_size=args.img_size,
                        figure_title=title,
                        save_path=save_path,
                    )
            elif dataset_name == 'Cataract1k':
                ds_viz = Cataract1kDataset(base_dir=args.volume_path, split="val")
                total = len(ds_viz)
                start = max(0, min(args.viz_index, total - 1))
                count = max(1, args.viz_count)
                end = min(total, start + count)

                if count > 1:
                    imgs, labs, titles = [], [], []
                    for i in range(start, end):
                        s = ds_viz[i]
                        imgs.append(s['image'])
                        labs.append(s['label'])
                        titles.append(s['case_name'])
                    default_name = f"Cataract_grid_{start}-{end-1}_{timestamp}.png"
                    save_path = resolve_save_path(default_name)
                    figure_title = f"Cataract-101K | cases {start}-{end-1}"
                    visualize_cataract_batch(
                        net,
                        images=imgs,
                        labels=labs,
                        img_size=args.img_size,
                        row_titles=titles,
                        figure_title=figure_title,
                        save_path=save_path,
                    )
                else:
                    sample = ds_viz[start]
                    title = f"Cataract-101K | case: {sample['case_name']}"
                    default_name = f"Cataract_{sample['case_name']}_{timestamp}.png"
                    save_path = resolve_save_path(default_name)
                    visualize_cataract_sample(
                        net,
                        image=sample['image'],
                        label=sample['label'],
                        img_size=args.img_size,
                        figure_title=title,
                        save_path=save_path,
                    )
        except Exception as e:
            raise RuntimeError(f"Visualization failed due to: {e}")
        # Do not continue with testing when --viz is set
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
    model_runtime_memory = runtime_memory_mb_benchmark(net)
    
    os.makedirs("bench_logs_", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date, time = timestamp.split("_")
    with open(os.path.join("bench_logs_", f"bench_{args.ckpt}_{timestamp}.json"), "w") as f:
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
