import logging
import os
import random
from datetime import datetime

import numpy as np
import torch


def make_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)

    return worker_init_fn


def _endovis_class_dice(outputs, labels, num_classes):
    predictions = torch.argmax(outputs, dim=1)
    dice_values = []
    for class_i in range(1, num_classes):
        pred_mask = predictions == class_i
        label_mask = labels == class_i
        pred_sum = pred_mask.sum().float()
        label_sum = label_mask.sum().float()

        if pred_sum.item() == 0 and label_sum.item() == 0:
            dice_values.append(np.nan)
            continue

        intersection = (pred_mask & label_mask).sum().float()
        dice = (2.0 * intersection) / (pred_sum + label_sum + 1e-5)
        dice_values.append(float(dice.detach().cpu().item()))

    return np.array(dice_values, dtype=np.float32)


def _validate_endovis(model, valloader, ce_loss, dice_loss, num_classes, lambda_):
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_dice_loss = 0.0
    batch_count = 0
    dice_sum = np.zeros(num_classes - 1, dtype=np.float64)
    dice_count = np.zeros(num_classes - 1, dtype=np.float64)

    with torch.no_grad():
        for sampled_batch in valloader:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs, _, _, _ = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = (1 - lambda_) * loss_dice + lambda_ * loss_ce

            total_loss += loss.item()
            total_ce_loss += loss_ce.item()
            total_dice_loss += loss_dice.item()
            batch_count += 1

            class_dice = _endovis_class_dice(outputs, label_batch, num_classes)
            valid_classes = ~np.isnan(class_dice)
            dice_sum[valid_classes] += class_dice[valid_classes]
            dice_count[valid_classes] += 1

    mean_class_dice = np.divide(
        dice_sum,
        dice_count,
        out=np.full_like(dice_sum, np.nan, dtype=np.float64),
        where=dice_count > 0,
    )
    if np.all(np.isnan(mean_class_dice)):
        mean_dice = 0.0
    else:
        mean_dice = float(np.nanmean(mean_class_dice))

    return {
        'loss': total_loss / max(batch_count, 1),
        'loss_ce': total_ce_loss / max(batch_count, 1),
        'loss_dice': total_dice_loss / max(batch_count, 1),
        'class_dice': mean_class_dice,
        'mean_dice': mean_dice,
    }


def save_checkpoint(model, args, epoch_num, mean_dice):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = os.path.join(
        args.ckpt_dir,
        args.ckpt_filename + '_dice_' + str(mean_dice) + '_epoch_' + str(epoch_num) + '_' + str(timestamp) + '.pth',
    )
    torch.save(model.state_dict(), local_path)
    logging.info("save model to {}".format(local_path))
