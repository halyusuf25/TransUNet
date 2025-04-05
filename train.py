import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
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
parser.add_argument('--teacher_vit_name', type=str,
                    default=None, help='load teacher model for kd-training')
parser.add_argument('--teacher_num_heads', type=int,
                    default=None, help='number of attention heads for the teacher model (default value sets in the imported CONFIGS_ViT_seg)')
parser.add_argument('--teacher_num_layers', type=int,
                    default=None, help='number of layers for the teacher model')
parser.add_argument('--teacher_pretrained_path', type=str,
                    default=None, help='load teacher pretrained checkpoint')
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
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../../model/{}/{}".format(args.exp, 'TU')
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
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.num_heads is not None:
        config_vit.transformer.num_heads = args.num_heads
    if args.num_layers is not None:
        config_vit.transformer.num_layers = args.num_layers

    #pass the args use_shsa to the config_vit
    config_vit.use_shsa = args.use_shsa
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))
    
    if args.teacher_vit_name is not None:
        teacher_config_vit = CONFIGS_ViT_seg[args.teacher_vit_name]
        teacher_config_vit.n_classes = args.num_classes
        teacher_config_vit.n_skip = args.n_skip
        teacher_config_vit.use_shsa = args.use_shsa
        
        if args.teacher_num_heads is not None:
            teacher_config_vit.transformer.num_heads = args.teacher_num_heads
            
        if args.teacher_num_layers is not None:
            teacher_config_vit.transformer.num_layers = args.teacher_num_layers
            
        if args.teacher_vit_name.find('R50') != -1:
            teacher_config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
                
        if args.teacher_pretrained_path is not None:
            ckpt_path = os.path.join(args.ckpt_dir, args.teacher_pretrained_path)
        else:
            raise ValueError("Teacher pretrained checkpoint path is not provided. Please specify --teacher_pretrained_path.")

        net_teacher = ViT_seg(teacher_config_vit, img_size=args.img_size, num_classes=teacher_config_vit.n_classes).cuda()
        net_teacher.load_state_dict(torch.load(ckpt_path))
    
    # print(f"arguments for training: {args}")
    # print(f"configuration of the vit model for training: {config_vit}") 
    trainer = {'Synapse': trainer_synapse,}
    if args.teacher_vit_name is not None:
        trainer[dataset_name](args, net, snapshot_path, teacher_model=net_teacher)
    else:
        trainer[dataset_name](args, net, snapshot_path)