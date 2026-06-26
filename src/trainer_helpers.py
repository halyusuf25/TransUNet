import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from datasets.dataset_acdc import ACDC_Dataset, RandomGenerator4ACDC
from datasets.dataset_cataract import Cataract1kDataset, RandomGenerator4Cataract
from datasets.dataset_endovis2018 import EndoVis2018Dataset, RandomGenerator4EndoVis2018
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator


def _fast_dice(pred, gt):
    pred_sum = int(np.sum(pred))
    gt_sum = int(np.sum(gt))

    if pred_sum > 0 and gt_sum > 0:
        intersection = int(np.logical_and(pred, gt).sum())
        dice = (2.0 * intersection) / float(pred_sum + gt_sum)
        return float(dice)

    if pred_sum == 0 and gt_sum == 0:
        return 1.0

    return 0.0


def make_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        worker_seed = torch.utils.data.get_worker_info().seed % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


def _make_random_half_validation_subset(dataset, seed):
    dataset_size = len(dataset)
    if dataset_size <= 1:
        return dataset

    subset_size = max(1, dataset_size // 2)
    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected_indices = sorted(indices[:subset_size])

    subset = Subset(dataset, selected_indices)
    subset.full_validation_size = dataset_size
    subset.validation_subset_indices = selected_indices
    return subset


def _synapse_validation_root(args):
    if getattr(args, "volume_path", None):
        return args.volume_path

    root_path = str(args.root_path).rstrip(os.sep)
    if os.path.basename(root_path) == "train_npz":
        return os.path.join(os.path.dirname(root_path), "test_vol_h5")
    return args.root_path


def _build_datasets(args):
    output_size = [args.img_size, args.img_size]

    if args.dataset == "Synapse":
        db_train = Synapse_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split="train",
            transform=transforms.Compose([RandomGenerator(output_size=output_size)]),
        )
        db_val = Synapse_dataset(
            base_dir=_synapse_validation_root(args),
            list_dir=args.list_dir,
            split="test_vol",
        )
        validation_protocol = "volume"
    elif args.dataset == "Cataract1k":
        db_train = Cataract1kDataset(
            base_dir=args.root_path,
            split="train",
            transform=transforms.Compose(
                [RandomGenerator4Cataract(output_size=output_size)]
            ),
        )
        db_val = Cataract1kDataset(
            base_dir=args.root_path,
            split="test",
            transform=transforms.Compose(
                [RandomGenerator4Cataract(output_size=output_size, augment=False)]
            ),
        )
        validation_protocol = "image"
    elif args.dataset == "EndoVis2018":
        db_train = EndoVis2018Dataset(
            base_dir=args.root_path,
            split="train",
            transform=transforms.Compose(
                [RandomGenerator4EndoVis2018(output_size, augment=True)]
            ),
        )
        if hasattr(db_train, "num_classes") and db_train.num_classes != args.num_classes:
            raise ValueError(
            f"Dataset labels.json has {db_train.num_classes} classes, "
            f"but args.num_classes={args.num_classes}"
            )
            
        db_val = EndoVis2018Dataset(
            base_dir=args.root_path,
            split="test",
            transform=transforms.Compose(
                [RandomGenerator4EndoVis2018(output_size, augment=False)]
            ),
        )
        if hasattr(db_val, "num_classes") and db_val.num_classes != args.num_classes:
            raise ValueError(
            f"Dataset labels.json has {db_val.num_classes} classes, "
            f"but args.num_classes={args.num_classes}"
            )
        validation_protocol = "image"
    elif args.dataset == "ACDC":
        db_train = ACDC_Dataset(
            base_dir=args.root_path,
            split="train",
            transform=transforms.Compose([RandomGenerator4ACDC(output_size)]),
            fold_id=args.fold_id,
        )
        db_val = ACDC_Dataset(
            base_dir=args.root_path,
            split="test",
            fold_id=args.fold_id,
        )
        validation_protocol = "volume"
    else:
        raise ValueError(
            "Unsupported dataset: {}. Supported datasets are: Synapse, "
            "Cataract1k, ACDC, and EndoVis2018.".format(args.dataset)
        )

    db_val = _make_random_half_validation_subset(db_val, args.seed)

    return db_train, db_val, validation_protocol


def _make_validation_loader(db_val, batch_size, validation_protocol):
    if validation_protocol == "volume":
        return DataLoader(
            db_val,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

    return DataLoader(
        db_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


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


def _validate_volume_dataset(model, valloader, ce_loss, dice_loss, num_classes, lambda_, args):
    model.eval()
    device = next(model.parameters()).device
    patch_h, patch_w = int(args.img_size), int(args.img_size)
    total_loss = 0.0
    total_ce_loss = 0.0
    total_dice_loss = 0.0
    slice_count = 0
    all_dice = []

    with torch.no_grad():
        for sampled_batch in valloader:
            image = sampled_batch["image"].squeeze(0).cpu().detach().numpy()
            label = sampled_batch["label"].squeeze(0).cpu().detach().numpy()
            if image.ndim == 2:
                image = image[np.newaxis, ...]
                label = label[np.newaxis, ...]

            prediction = np.zeros_like(label, dtype=np.uint8)
            for slice_index in range(image.shape[0]):
                image_slice = image[slice_index]
                label_slice = label[slice_index]
                height, width = image_slice.shape
                if height != patch_h or width != patch_w:
                    image_for_model = zoom(
                        image_slice,
                        (patch_h / height, patch_w / width),
                        order=3,
                    )
                    label_for_loss = zoom(
                        label_slice,
                        (patch_h / height, patch_w / width),
                        order=0,
                    )
                else:
                    image_for_model = image_slice
                    label_for_loss = label_slice

                input_tensor = (
                    torch.from_numpy(image_for_model)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )
                target_tensor = (
                    torch.from_numpy(label_for_loss.astype(np.int64))
                    .unsqueeze(0)
                    .to(device)
                )

                outputs, _, _, _ = model(input_tensor)
                loss_ce = ce_loss(outputs, target_tensor.long())
                loss_dice = dice_loss(outputs, target_tensor, softmax=True)
                loss = (1 - lambda_) * loss_dice + lambda_ * loss_ce

                total_loss += loss.item()
                total_ce_loss += loss_ce.item()
                total_dice_loss += loss_dice.item()
                slice_count += 1

                output_slice = (
                    torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                    .squeeze(0)
                    .cpu()
                    .detach()
                    .numpy()
                )
                if height != patch_h or width != patch_w:
                    output_slice = zoom(
                        output_slice,
                        (height / patch_h, width / patch_w),
                        order=0,
                    )
                prediction[slice_index] = output_slice

            case_dice = [
                _fast_dice(prediction == class_i, label == class_i)
                for class_i in range(1, num_classes)
            ]
            all_dice.append(np.asarray(case_dice, dtype=np.float32))

    if all_dice:
        dice_stack = np.stack(all_dice, axis=0)
        class_dice = np.mean(dice_stack, axis=0)
        mean_dice = float(np.mean(class_dice))
    else:
        class_dice = np.full(num_classes - 1, np.nan, dtype=np.float32)
        mean_dice = 0.0

    return {
        "loss": total_loss / max(slice_count, 1),
        "loss_ce": total_ce_loss / max(slice_count, 1),
        "loss_dice": total_dice_loss / max(slice_count, 1),
        "class_dice": class_dice,
        "mean_dice": mean_dice,
    }


def _validate(model, valloader, ce_loss, dice_loss, num_classes, lambda_, args, validation_protocol):
    if validation_protocol == "volume":
        return _validate_volume_dataset(
            model,
            valloader,
            ce_loss,
            dice_loss,
            num_classes,
            lambda_,
            args,
        )

    return _validate_endovis(model, valloader, ce_loss, dice_loss, num_classes, lambda_)


def _log_validation(writer, val_metrics, iter_num):
    writer.add_scalar("info/val_loss", val_metrics["loss"], iter_num)
    writer.add_scalar("info/val_loss_ce", val_metrics["loss_ce"], iter_num)
    writer.add_scalar("info/val_loss_dice", val_metrics["loss_dice"], iter_num)
    writer.add_scalar("info/val_mean_dice", val_metrics["mean_dice"], iter_num)
    for class_i, class_dice in enumerate(val_metrics["class_dice"], start=1):
        if not np.isnan(class_dice):
            writer.add_scalar("info/val_{}_dice".format(class_i), class_dice, iter_num)


def _save_periodic_checkpoint(model, args, performance, epoch_index):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = os.path.join(
        args.ckpt_dir,
        args.ckpt_filename + "_epoch_" + str(epoch_index) + "_dice_" + str(performance) + "_" + str(timestamp) + ".pth",
    )
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), local_path)
    logging.info("save model to {}".format(local_path))


def _save_last_epoch_checkpoint(model, args, epoch_index, performance):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    local_path = os.path.join(
        args.ckpt_dir,
        args.ckpt_filename
        + "_epoch_"
        + str(epoch_index)
        + "_LastEpoch_"
        + str(timestamp)
        + "_dice_"
        + str(performance)
        + ".pth",
    )
    torch.save(model.state_dict(), local_path)
    logging.info("save model to {}".format(local_path))


def _append_dataset_to_checkpoint_name(args):
    suffix = "_" + args.dataset + "_"
    if not str(args.ckpt_filename).endswith(suffix):
        args.ckpt_filename += suffix


def _log_train_images(writer, image_batch, label_batch, outputs, iter_num):
    if image_batch.size(0) == 0:
        return
    sample_index = min(1, image_batch.size(0) - 1)
    image = image_batch[sample_index]
    if image.dim() == 3 and image.size(0) > 1:
        image = image[0:1, :, :]
    elif image.dim() == 2:
        image = image.unsqueeze(0)

    image_min = image.min()
    image_max = image.max()
    denom = image_max - image_min
    if float(denom.detach().item()) > 0:
        image = (image - image_min) / denom

    predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
    labels = label_batch[sample_index, ...].unsqueeze(0)
    writer.add_image("train/Image", image, iter_num)
    writer.add_image("train/Prediction", predictions[sample_index, ...] * 50, iter_num)
    writer.add_image("train/GroundTruth", labels * 50, iter_num)


def _log_bu_epoch_details(writer, details, epoch_index):
    for tag, key in (
        ("epoch/mean_weights", "w_mean"),
        ("epoch/max_weights", "w_max"),
        ("epoch/mean_UM", "UM_mean"),
        ("epoch/max_UM", "UM_max"),
        ("epoch/mean_BM", "BM_mean"),
        ("epoch/max_BM", "BM_max"),
    ):
        if key in details:
            writer.add_scalar(tag, details[key], epoch_index)


def save_checkpoint(model, args, epoch_index, mean_dice):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = os.path.join(
        args.ckpt_dir,
        args.ckpt_filename + '_best_val_dice_' + str(mean_dice) + '_epoch_' + str(epoch_index) + '_' + str(timestamp) + '.pth',
    )
    torch.save(model.state_dict(), local_path)
    logging.info("save model to {}".format(local_path))
