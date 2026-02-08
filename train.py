import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse, trainer_acdc
from datasets.dataset_cataract import  Cataract1kDataset

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='dataset name, and possible values are Synapse, ACDC, and Cataract1k')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--ckpt_dir', type=str, 
                    default='ckpt/', help='directory to save trained model')
parser.add_argument('--ckpt', type=str, 
                    default='default_ckpt_name', help='name of the checkpoint file (dont add .pth)')
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
parser.add_argument('--topk_attn', type=float,
                    default=0.0, help='keep rate for Top-k attention (0.0 means not using Top-k attention)')
parser.add_argument('--adaptive_attn_threshold', type=float,
                    default=0.0, help='threshold for adaptive attention to select tokens (0.0 means not using adaptive attention)')
parser.add_argument('--use_se_block', action='store_true', help='whether to use SE block in the encoder')

###Teacher Model Argument:#####
parser.add_argument('--use_kd', action='store_true', 
                    help='whether to use knowledge distillation (kd) training')
parser.add_argument('--teacher_vit_name', type=str,
                    default='R50-ViT-B_16', help='load teacher model for kd-training')
parser.add_argument('--teacher_ckpt', type=str,
                    default=None, help='load teacher model checkpoints for kd-training')
parser.add_argument('--teacher_num_heads', type=int,
                    default=None, help='number of attention heads for the teacher model (default value sets in the imported CONFIGS_ViT_seg)')
parser.add_argument('--teacher_num_layers', type=int,
                    default=None, help='number of layers for the teacher model')
parser.add_argument('--teacher_ckpt_path', type=str,
                    default='ckpt/', help='path for teacher pretrained checkpoints')
parser.add_argument('--kd_temperature', type=float,
                    default=1.0, help='temperature for kd training (default 1.0 means no temperature scaling)')
parser.add_argument('--kd_points' , type=str,
                    default='logits', help='"logits", "intermediate", "features", "logits+intermediate", and "all" are options for kd training')
###########################################

##########LOSS FUNCTION arguments##########
parser.add_argument('--lambda_' , type=float, default=0.5, help='weighting factor for the loss function')
parser.add_argument('--use_bu_loss', action='store_true', 
                    help='whether to use Boundary-Uncertainty (BU) loss for training')
parser.add_argument('--tau', type=float, default=1.0, help='Boundry Decay parameter for BU loss')
parser.add_argument('--alpha', type=float, default=10, help='maximum value for the Weight Map in BU loss')
parser.add_argument('--bm_min', type=float, default=0.2, help='Minimum value for the Boundary Map in BU loss')
parser.add_argument('--bm_max', type=float, default=3.0, help='Maximum value for the Boundary Map in BU loss')
parser.add_argument('--distance_map_type', type=str, 
                    default='unsigned', help='Type of Distance Map for BU loss: "dtm", "signed" or "unsigned"')
parser.add_argument('--buloss_option', type=str, default='C', help='Options for BU loss: "A, B, C"')
#OPTION A : \mathcal{L}_{total} = \mathcal{L}_{Dice}^w+ \mathcal{L}_{CE}^w
#OPTION B : \mathcal{L}_{total} = \mathcal{L}_{Dice}+ \mathcal{L}_{CE}^w
#OPTION C : \mathcal{L}_{total} = \mathcal{L}_{Dice}+ \mathcal{L}_{CE}
#########################################

##########swin config arguments##########
parser.add_argument('--swin_pretrained_path', type=str,
                    default='/data/shared/pretrained_backbones/swin/swin_large_patch4_window7_224_22k.pth', help='path to swin pretrained model')
#########################################


#########addtional arguments for debugging#########
parser.add_argument('--verbose', action='store_true', 
                    help='whether to print detailed debug information during training')
###################################################

args = parser.parse_args()


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
    if args.dataset not in ['Synapse', 'Cataract1k', 'ACDC']:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets are: Synapse, Cataract1k, and ACDC.")
    
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            # 'root_path': '../../data/Synapse/train_npz',
            'root_path': '/data/shared/project_TransUNet/data/Synapse/train_npz/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'Cataract1k': {
            'root_path': '/data/shared/CataractData/',
            'list_dir': None,  # Not needed for Cataract1k
            'num_classes': 5,  # Background (0), Pupil (1), Cornea (2), Lens (3), Instruments (4)
        },
        'ACDC': {
            'root_path': '/data/shared/project_TransUNet/data/ACDC',
            'list_dir': None,
            'num_classes': 4,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.ckpt_filename = args.ckpt 
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.verbose = args.verbose
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.num_heads is not None:
        config_vit.transformer.num_heads = args.num_heads
        args.ckpt_filename +='_head'+str(args.num_heads)
    if args.num_layers is not None:
        config_vit.transformer.num_layers = args.num_layers
        args.ckpt_filename +='_layer'+str(args.num_layers)

    #pass the args use_shsa to the config_vit
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
    config_vit.use_se_block = args.use_se_block
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))
    
    if args.verbose:
        print(f"Model parameters: {sum(p.numel() for p in net.parameters())}")
        for name, param in net.named_parameters():
            print(f"{name}: shape={param.shape}, dtype={param.dtype}")
            if param.numel() < 20:  # Only print values for small parameters
                print(f"  values: {param.data}")

    
    ### Load Teacher Model for KD-Training: ###
    if args.use_kd:
        if args.teacher_ckpt is None:
            raise ValueError("The --teacher_ckpt argument must be provided when --use_kd is set.")
        
        if not os.path.isfile(os.path.join(args.teacher_ckpt_path, args.teacher_ckpt)):
            raise ValueError(f"The specified teacher checkpoint file does not exist: {os.path.join(args.teacher_ckpt_path, args.teacher_ckpt)}")
        teacher_config_vit = CONFIGS_ViT_seg[args.teacher_vit_name] 
        teacher_config_vit.n_classes = args.num_classes
        teacher_config_vit.n_skip = args.n_skip
        # teacher_config_vit.kd_points = args.kd_points
        if args.teacher_num_heads is not None:
            teacher_config_vit.transformer.num_heads = args.teacher_num_heads
        if args.teacher_num_layers is not None:
            teacher_config_vit.transformer.num_layers = args.teacher_num_layers
        teacher_config_vit.use_shsa = args.use_shsa
        teacher_config_vit.use_alternate_shsa = args.use_alternate_shsa
        teacher_config_vit.topk_attn = args.topk_attn
        teacher_config_vit.use_efficientnet = args.use_efficientnet
        teacher_config_vit.use_swin = args.use_swin
        if args.teacher_vit_name.find('R50') != -1:
            teacher_config_vit.patches.grid = config_vit.patches.grid
        teacher_net = ViT_seg(teacher_config_vit, img_size=args.img_size, num_classes=teacher_config_vit.n_classes).cuda()
        teacher_net.load_state_dict(torch.load(os.path.join(args.teacher_ckpt_path, args.teacher_ckpt)))
        teacher_net.eval()  # Set teacher to evaluation mode
    else:
        Warning("Knowledge Distillation (KD) training is not enabled. Proceeding without a teacher model.")
            

    # print(f"arguments for training: {args}")
    # print(f"configuration of the vit model for training: {config_vit}") 
    trainer = {'Synapse': trainer_synapse, 'Cataract1k': trainer_synapse, 'ACDC': trainer_acdc}
    if args.use_kd:
        trainer[dataset_name](args, net, snapshot_path, teacher_model=teacher_net)
    else:
        trainer[dataset_name](args, net, snapshot_path)