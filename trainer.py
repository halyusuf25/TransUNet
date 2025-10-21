import argparse
import datetime
import logging
import os
from pyexpat import features
import random
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from networks.distillation import compute_kd_loss, MGD, KDTarget, KDWeights
from torchvision import transforms
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from datasets.dataset_cataract import Cataract1kDataset, RandomGenerator4Cataract

def trainer_synapse(args, model, snapshot_path, teacher_model=None):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    if args.dataset == 'Synapse':
        db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                transform=transforms.Compose(
                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
        print("The length of train set is: {}".format(len(db_train)))
    elif args.dataset == 'Cataract1k':
        db_train = Cataract1kDataset(base_dir=args.root_path, split="train",
                                    transform=transforms.Compose(
                                        [RandomGenerator4Cataract(output_size=[args.img_size, args.img_size])]))
        print("The length of train set is: {}".format(len(db_train)))
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset)) 

    args.ckpt_filename += '_' + args.dataset + '_'

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    gamma = 0.2 # distillation loss weight
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs, _ , features = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            
            if args.use_kd and teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs, _ , teacher_features = teacher_model(image_batch)
                    if args.kd_points in {'backbone', 'all'}:
                        s_last = features[-1]
                        t_last = teacher_features[-1]
                        _mgd_predictor = MGD(c_s=s_last.shape[1], c_t=t_last.shape[1]).to(s_last.device)
                        _backbone_pair = (s_last, t_last)
                        optimizer.add_param_group({'params': _mgd_predictor.parameters(), 'lr': base_lr})
                    else:
                        _mgd_predictor = None
                        _backbone_pair = None
                           
                kd_loss, _ = compute_kd_loss(
                    kd_points=args.kd_points,
                    weights=KDWeights(logits=1.0, intermediate=1.0, backbone=1.0),
                    student_logits=outputs,
                    teacher_logits=teacher_outputs,
                    student_features=features,
                    teacher_features=teacher_features,
                    backbone_pair=_backbone_pair, # (student_feat, teacher_feat) for MGD
                    mgd_predictor=_mgd_predictor,
                    temperature=args.kd_temperature,
                )
                loss = (1-gamma) * (0.5 * loss_ce + 0.5 * loss_dice) + gamma * kd_loss
            else:
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            if args.use_kd and teacher_model is not None:
                writer.add_scalar('info/loss_kd', kd_loss, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_interval = 70  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            local_path = os.path.join(args.ckpt_dir, args.ckpt_filename + '_epoch_' + str(epoch_num) + '_' + str(timestamp) + '.pth')
            torch.save(model.state_dict(), local_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            local_path = os.path.join(args.ckpt_dir, args.ckpt_filename + '_epoch_' + str(epoch_num) + '_' + str(timestamp) + '.pth')
            torch.save(model.state_dict(), local_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"