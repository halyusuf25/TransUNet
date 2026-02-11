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
from src.loss_bu import BULoss
from networks.distillation import compute_kd_loss, MGD, KDTarget, KDWeights
from torchvision import transforms
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from datasets.dataset_cataract import Cataract1kDataset, RandomGenerator4Cataract
from datasets.dataset_acdc import ACDC_Dataset, RandomGenerator4ACDC
from utils import test_single_volume

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
    lambda_ = args.lambda_
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    bu_loss = BULoss(loss_option=args.buloss_option, args=args)
    bu_loss = bu_loss.to(next(model.parameters()).device)
    optimizer_params = list(model.parameters())
    if args.use_bu_loss:
        optimizer_params.extend(list(bu_loss.parameters()))
    optimizer = optim.SGD(optimizer_params, lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(args.tensorboard_run_dir)
    logging.info("TensorBoard run dir: %s", args.tensorboard_run_dir)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        epoch_total_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_batch_count = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs, _ , features, _ = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            
            if args.use_kd and teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs, _ , teacher_features, _ = teacher_model(image_batch)
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
                loss = (1-gamma) * ((1-lambda_) * loss_dice + lambda_ * loss_ce) + gamma * kd_loss
            elif args.use_bu_loss:
                loss, details = bu_loss(outputs, label_batch, return_details=True)
            else:
                loss = (1-lambda_) * loss_dice + lambda_ * loss_ce

            epoch_total_loss += loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_dice_loss += loss_dice.item()
            epoch_batch_count += 1
                
            optimizer.zero_grad()
            loss.backward()
            # To inspect learnable tau gradients after backward:
            # if args.use_bu_loss and getattr(args, "learn_tau", False): print(bu_loss.rho.grad)
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            if args.verbose and iter_num >= 2:
                print("Verbose mode is ON. Detailed training information were printed and training is stopped after two iterations.")
                sys.exit(0)

            tau_value = float(bu_loss.get_tau().detach().item())
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/tau', tau_value, iter_num)
            
            if args.use_kd and teacher_model is not None:
                writer.add_scalar('info/loss_kd', kd_loss, iter_num)
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_kd: %f' % (iter_num, loss.item(), loss_ce.item(), kd_loss.item()))
            else:
                logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        if epoch_batch_count > 0:
            mean_total_loss = epoch_total_loss / epoch_batch_count
            mean_ce_loss = epoch_ce_loss / epoch_batch_count
            mean_dice_loss = epoch_dice_loss / epoch_batch_count
            tau_value = float(bu_loss.get_tau().detach().item())
            epoch_index = epoch_num + 1
            writer.add_scalar('epoch/total_loss', mean_total_loss, epoch_index)
            writer.add_scalar('epoch/loss_ce', mean_ce_loss, epoch_index)
            writer.add_scalar('epoch/loss_dice', mean_dice_loss, epoch_index)
            writer.add_scalar('epoch/tau', tau_value, epoch_index)
            if args.learn_tau:
                writer.add_scalar('epoch/mean_weights', details["w_mean"], epoch_index)
                writer.add_scalar('epoch/max_weights', details["w_max"], epoch_index)
                writer.add_scalar('epoch/mean_UM', details["UM_mean"], epoch_index)
                writer.add_scalar('epoch/max_UM', details["UM_max"], epoch_index)
                writer.add_scalar('epoch/mean_BM', details["BM_mean"], epoch_index)
                writer.add_scalar('epoch/max_BM', details["BM_max"], epoch_index)
                
            logging.info(
                'epoch %d : total_loss : %f, loss_ce : %f, loss_dice : %f, tau : %f',
                epoch_index, mean_total_loss, mean_ce_loss, mean_dice_loss, tau_value
            )
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_interval = 70  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     local_path = os.path.join(args.ckpt_dir, args.ckpt_filename + '_epoch_' + str(epoch_num) + '_' + str(timestamp) + '.pth')
        #     torch.save(model.state_dict(), local_path)
        #     logging.info("save model to {}".format(save_mode_path))

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


def trainer_acdc(args, model, snapshot_path, teacher_model=None):
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    db_train = ACDC_Dataset(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator4ACDC([args.img_size, args.img_size])]))
    db_val = ACDC_Dataset(base_dir=args.root_path, split="val")
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = DiceLoss(num_classes)

    writer = SummaryWriter(args.tensorboard_run_dir)
    logging.info("TensorBoard run dir: %s", args.tensorboard_run_dir)
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("{} val iterations per epoch".format(len(valloader)))
    # logging.info("{} test iterations per epoch".format(len(testloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs, _ , features, _ = model(volume_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            if args.verbose and iter_num >= 2:
                print("Verbose mode is ON. Detailed training information were printed and training is stopped after two iterations.")
                sys.exit(0)
                
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 500 == 0:  # 500
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    metric_i = test_single_volume(image, label, model, classes=num_classes,
                                                  patch_size=[args.img_size, args.img_size])
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_iteration, best_performance, best_hd95 = iter_num, performance, mean_hd95
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_best)
                    logging.info('Best model | iteration %d : mean_dice : %f mean_hd95 : %f' % (
                    iter_num, performance, mean_hd95))
                    
                    save_checkpoint(model, args, epoch_num, performance)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                save_checkpoint(model, args, epoch_num, performance)
                break
            
            

def save_checkpoint(model, args, epoch_num, mean_dice):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = os.path.join(args.ckpt_dir, args.ckpt_filename + '_dice_'+ str(mean_dice) + '_epoch_' + str(epoch_num) + '_' + str(timestamp) + '.pth')
    torch.save(model.state_dict(), local_path)
    logging.info("save model to {}".format(local_path))
