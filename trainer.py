import logging
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from networks.distillation import KDWeights, MGD, compute_kd_loss
from src.loss_bu import BULoss
from src.trainer_helpers import (
    _append_dataset_to_checkpoint_name,
    _build_datasets,
    _log_bu_epoch_details,
    _log_train_images,
    _log_validation,
    _make_validation_loader,
    _save_last_epoch_checkpoint,
    _save_periodic_checkpoint,
    _validate,
    make_worker_init_fn,
    save_checkpoint,
)
from src.visualize import (
    _get_dataset_sample_names,
    _resolve_heatmap_sample_targets,
    _should_save_heatmap_epoch,
    save_pending_weight_heatmaps,
)
from utils import (
    DiceLoss,
    _extract_case_names,
    _is_primary_process,
)


def trainer(args, model, snapshot_path, teacher_model=None):
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train, db_val, validation_protocol = _build_datasets(args)
    full_validation_size = getattr(db_val, "full_validation_size", len(db_val))

    print("The length of train set is: {}".format(len(db_train)))
    print(
        "The length of validataion set is: {} / {}".format(
            len(db_val),
            full_validation_size,
        )
    )
    logging.info(
        "%s train samples: %d | val samples: %d/%d | validation subset seed: %d",
        args.dataset,
        len(db_train),
        len(db_val),
        full_validation_size,
        args.seed,
    )
    _append_dataset_to_checkpoint_name(args)

    worker_init_fn = make_worker_init_fn(args.seed)
    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = _make_validation_loader(db_val, batch_size, validation_protocol)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    gamma = 0.2
    lambda_ = args.lambda_
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    
    if args.verbose:
        print(f"Verbose mode is ON. Training will be stopped after {args.verbose_iterations} for debugging purposes.")
        
    if args.use_bu_loss:
        bu_loss = BULoss(loss_option=args.buloss_option, args=args)
        bu_loss = bu_loss.to(next(model.parameters()).device)
        if args.verbose:
            print(f"BU Loss parameters: {list(bu_loss.parameters())}")


    optimizer_param_groups = [
        {
            "params": list(model.parameters()),
            "lr": base_lr,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "lr_mult": 1.0,
        },
    ]
    if args.use_bu_loss and args.learn_tau:
        optimizer_param_groups.append(
            {
                "params": [bu_loss.rho],
                "lr": base_lr * 0.5,
                "momentum": 0.9,
                "weight_decay": 0.0,
                "lr_mult": 0.5 ,
            }
        )
    optimizer = optim.SGD(
        optimizer_param_groups,
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    writer = SummaryWriter(args.tensorboard_run_dir)
    logging.info("TensorBoard run dir: %s", args.tensorboard_run_dir)
    logging.info("{} val iterations per validation".format(len(valloader)))

    is_primary_process = _is_primary_process()
    heatmaps_enabled = (
        bool(getattr(args, "create_heatmaps", False))
        and args.use_bu_loss
        and is_primary_process
    )
    fixed_heatmap_sample_id = None
    fixed_heatmap_target_samples = []
    dataset_sample_names = _get_dataset_sample_names(db_train) if heatmaps_enabled else []
    if heatmaps_enabled:
        os.makedirs(args.heatmaps_dir, exist_ok=True)
        logging.info(
            "BU-loss heatmap generation enabled. Output dir: %s | slices per target epoch: %d",
            args.heatmaps_dir,
            int(getattr(args, "num_heatmap_slices", 1)),
        )
    elif bool(getattr(args, "create_heatmaps", False)) and not is_primary_process:
        logging.info("BU-loss heatmap generation disabled on non-primary process.")

    iter_num = 0
    max_iterations = args.max_iterations
    iterations_per_epoch = len(trainloader)
    max_epoch = (max_iterations + iterations_per_epoch - 1) // iterations_per_epoch
    logging.info(
        "%d iterations per epoch. %d max iterations. %d total epochs.",
        iterations_per_epoch,
        max_iterations,
        max_epoch,
    )

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    last_bu_details = None
    stop_training = False

    for epoch_num in iterator:
        epoch_index = epoch_num + 1
        should_save_heatmap_this_epoch = heatmaps_enabled and _should_save_heatmap_epoch(epoch_index)
        pending_heatmap_samples = set(fixed_heatmap_target_samples) if should_save_heatmap_this_epoch else set()
        saved_heatmap_count_this_epoch = 0
        epoch_total_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_batch_count = 0

        for i_batch, sampled_batch in enumerate(trainloader):
            case_names = _extract_case_names(sampled_batch)
            if heatmaps_enabled and fixed_heatmap_sample_id is None:
                if case_names and len(case_names) > 0:
                    fixed_heatmap_sample_id = case_names[0]
                else:
                    fixed_heatmap_sample_id = "sample_0"
                fixed_heatmap_target_samples = _resolve_heatmap_sample_targets(
                    dataset_sample_names=dataset_sample_names,
                    anchor_sample_name=fixed_heatmap_sample_id,
                    num_heatmap_slices=int(getattr(args, "num_heatmap_slices", 1)),
                )
                if should_save_heatmap_this_epoch:
                    pending_heatmap_samples = set(fixed_heatmap_target_samples)
                logging.info(
                    "Fixed BU-loss heatmap anchor sample: %s | selected target slices (%d): %s",
                    fixed_heatmap_sample_id,
                    len(fixed_heatmap_target_samples),
                    ", ".join(fixed_heatmap_target_samples),
                )

            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs, _, features, _ = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            details = None
            kd_loss = None

            if args.use_kd and teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs, _, teacher_features, _ = teacher_model(image_batch)
                    if args.kd_points in {"backbone", "all"}:
                        s_last = features[-1]
                        t_last = teacher_features[-1]
                        _mgd_predictor = MGD(c_s=s_last.shape[1], c_t=t_last.shape[1]).to(s_last.device)
                        _backbone_pair = (s_last, t_last)
                        optimizer.add_param_group({"params": _mgd_predictor.parameters(), "lr": base_lr})
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
                    backbone_pair=_backbone_pair,
                    mgd_predictor=_mgd_predictor,
                    temperature=args.kd_temperature,
                )
                loss = (1 - gamma) * ((1 - lambda_) * loss_dice + lambda_ * loss_ce) + gamma * kd_loss
            elif args.use_bu_loss:
                loss, details = bu_loss(outputs, label_batch, return_details=True)
                last_bu_details = details
            else:
                loss = (1 - lambda_) * loss_dice + lambda_ * loss_ce

            epoch_total_loss += loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_dice_loss += loss_dice.item()
            epoch_batch_count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_ * float(param_group.get("lr_mult", 1.0))

            iter_num = iter_num + 1
            saved_heatmap_count_this_epoch += save_pending_weight_heatmaps(
                should_save_heatmap_this_epoch=should_save_heatmap_this_epoch,
                details=details,
                pending_heatmap_samples=pending_heatmap_samples,
                case_names=case_names,
                image_batch=image_batch,
                label_batch=label_batch,
                epoch_index=epoch_index,
                iter_num=iter_num,
                heatmaps_dir=args.heatmaps_dir,
                logger=logging,
            )

            if args.verbose and iter_num >= args.verbose_iterations:
                print(
                    f"Verbose mode is ON. Detailed training information were printed "
                    f"and training is stopped after {args.verbose_iterations} iterations."
                )
                sys.exit(0)

            
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/loss_ce", loss_ce, iter_num)
            writer.add_scalar("info/loss_dice", loss_dice, iter_num)
            
            if args.use_bu_loss:
                tau_value = float(bu_loss.get_tau().detach().item())
                writer.add_scalar("info/tau", tau_value, iter_num)

            if args.use_kd and teacher_model is not None:
                writer.add_scalar("info/loss_kd", kd_loss, iter_num)
                logging.info(
                    "epoch %d iteration %d : loss : %f, loss_dice: %f, loss_ce: %f, loss_kd: %f",
                    epoch_index,
                    iter_num,
                    loss.item(),
                    loss_dice.item(),
                    loss_ce.item(),
                    kd_loss.item(),
                )
            else:
                logging.info(
                    "epoch %d iteration %d : loss : %f, loss_dice: %f, loss_ce: %f",
                    epoch_index,
                    iter_num,
                    loss.item(),
                    loss_dice.item(),
                    loss_ce.item(),
                )

            # if iter_num % 20 == 0:
            #     _log_train_images(writer, image_batch, label_batch, outputs, iter_num)

            if iter_num >= max_iterations:
                stop_training = True
                break

        if epoch_batch_count > 0:
            mean_total_loss = epoch_total_loss / epoch_batch_count
            mean_ce_loss = epoch_ce_loss / epoch_batch_count
            mean_dice_loss = epoch_dice_loss / epoch_batch_count
            writer.add_scalar("epoch/total_loss", mean_total_loss, epoch_index)
            writer.add_scalar("epoch/loss_ce", mean_ce_loss, epoch_index)
            writer.add_scalar("epoch/loss_dice", mean_dice_loss, epoch_index)
            
            if args.use_bu_loss:
                tau_value = float(bu_loss.get_tau().detach().item())
                writer.add_scalar("epoch/tau", tau_value, epoch_index)
                if last_bu_details is not None:
                    _log_bu_epoch_details(writer, last_bu_details, epoch_index)

            epoch_end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logging.info(
                "epoch %d : total_loss : %f, loss_ce : %f, loss_dice : %f, timestamp: %s",
                epoch_index,
                mean_total_loss,
                mean_ce_loss,
                mean_dice_loss,
                epoch_end_timestamp,
            )

        if should_save_heatmap_this_epoch and pending_heatmap_samples:
            missing_samples = sorted(list(pending_heatmap_samples))
            logging.warning(
                "Heatmap target epoch reached (epoch=%d) and saved %d/%d slices. Missing samples: %s",
                epoch_index,
                saved_heatmap_count_this_epoch,
                len(fixed_heatmap_target_samples),
                ", ".join(missing_samples),
            )

        if epoch_batch_count > 0:
            val_metrics = _validate(
                model,
                valloader,
                ce_loss,
                dice_loss,
                num_classes,
                lambda_,
                args,
                validation_protocol,
            )
            # _log_validation(writer, val_metrics, iter_num)
            writer.add_scalar("epoch/val_loss", val_metrics["loss"], iter_num)
            writer.add_scalar("epoch/val_loss_ce", val_metrics["loss_ce"], iter_num)
            writer.add_scalar("epoch/val_loss_dice", val_metrics["loss_dice"], iter_num)
            writer.add_scalar("epoch/val_mean_dice", val_metrics["mean_dice"], iter_num)

            performance = val_metrics["mean_dice"]
            if epoch_index > args.best_checkpoint_start_epoch and performance > best_performance:
                best_performance = performance
                logging.info(
                    "Best validation model | epoch %d iteration %d : mean_dice : %f val_loss : %f",
                    epoch_index,
                    iter_num,
                    performance,
                    val_metrics["loss"],
                )
                save_checkpoint(model, args, epoch_index, performance)

            val_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logging.info(
                "epoch %d iteration %d : val_loss : %f mean_dice : %f, timestamp: %s",
                epoch_index,
                iter_num,
                val_metrics["loss"],
                performance,
                val_timestamp,
            )
            model.train()

        if epoch_index % 50 == 0:
            _save_periodic_checkpoint(model, args, performance, epoch_index)

        if stop_training:
            _save_last_epoch_checkpoint(model, args, epoch_index, performance)
            iterator.close()
            break

        if epoch_num >= max_epoch - 1:
            _save_last_epoch_checkpoint(model, args, epoch_index, performance)
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
