"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import math
from models import postprocessors
import os
import sys
from typing import Iterable

import torch
import torch.distributed as dist

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.refexp_eval import RefExpEvaluator

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets.a2d_eval import calculate_precision_at_k_and_iou_metrics, calculate_bbox_precision_at_k_and_iou_metrics


import random
import numpy as np
import torch


from torch.cuda.amp import GradScaler, autocast

import copy
from copy import deepcopy
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # print("samples, targets device:",samples, targets,device)
        # print("device:", device)
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device) 

        outputs = model(samples, captions, targets) #######debug outputs and mid-layer outputs
        loss_dict = criterion(outputs, targets)### debug   # last layer and mid-layer losses

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # print("loss_value = ",loss_value)
        if not math.isfinite(loss_value):## loss is abnormally large
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()# backward pass
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_elr_orgsize_mask(model: torch.nn.Module, criterion: torch.nn.Module,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, max_norm: float = 0, start_epoch: int = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.9f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        # video_ids = [t["video_id"] for t in targets]
        clip_idxs = [t["clip_idx"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs = model(samples, captions, targets)
        loss_dict = criterion(clip_idxs, outputs, targets, epoch, start_epoch)  # pass epoch and start_epoch

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_ws(model: torch.nn.Module, criterion: torch.nn.Module,
                       data_loader: Iterable, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, max_norm: float = 0,
                       warmup_epochs: int = 3,):
    """
    Train for one epoch using SetCriterionNPN loss function
    Parameters:
        model: model
        criterion: SetCriterionNPN instance
        data_loader: data loader
        optimizer: optimizer
        device: device
        epoch: current epoch
        max_norm: maximum norm for gradient clipping
        warmup_epochs: number of warm-up epochs
    """
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.9f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000
    for samples_w, samples_s, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples_w = samples_w.to(device)
        samples_s =  samples_s.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)
        clip_indices = torch.tensor([t["clip_idx"] for t in targets], device=device)

        # Warm-up phase
        if epoch < warmup_epochs:
            outputs_w = model(samples_w, captions, targets)
            # Use empty dictionary to simulate outputs_s, as warm-up doesn't need strong augmentation output
            loss_dict = criterion(clip_indices, outputs_w, {}, targets, epoch)
        else:
            # Robust Training phase
            outputs_w = model(samples_w, captions, targets)
            outputs_s = model(samples_s, captions, targets)
            loss_dict = criterion(clip_indices, outputs_w, outputs_s, targets, epoch)

        # Calculate total loss
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Optimization
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        metric_logger.update(loss=losses.item(), grad_norm=grad_total_norm, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, evaluator_list, device, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    predictions = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        dataset_name = targets[0]["dataset_name"]
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs = model(samples, captions, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        for evaluator in evaluator_list:
            evaluator.update(res)

        # REC & RES predictions
        for p, target in zip(results, targets):
            for s, b, m in zip(p['scores'], p['boxes'], p['rle_masks']):
                    predictions.append({'image_id': target['image_id'].item(),
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'bbox': b.tolist(),
                                        'segmentation': m,
                                        'score': s.item()})


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    refexp_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            evaluator.accumulate()
            evaluator.summarize()
        elif isinstance(evaluator, RefExpEvaluator):
            refexp_res = evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # update stats
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()
    if refexp_res is not None:
        stats.update(refexp_res)

    # evaluate RES 
    # gather and merge predictions from all gpus
    gathered_pred_lists = utils.all_gather(predictions)
    predictions = [p for p_list in gathered_pred_lists for p in p_list]

    eval_metrics = {}
    if utils.is_main_process():
        if dataset_name == 'refcoco':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcoco/instances_refcoco_val.json'))
        elif dataset_name == 'refcoco+':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcoco+/instances_refcoco+_val.json'))
        elif dataset_name == 'refcocog':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcocog/instances_refcocog_val.json'))
        else:
            raise NotImplementedError
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        # ap_metrics = coco_eval.stats[:6]
        # eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
        # Precision and IOU
        # bbox 
        precision_at_k, overall_iou, mean_iou = calculate_bbox_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'bbox P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'bbox overall_iou': overall_iou, 'bbox mean_iou': mean_iou})
        # mask
        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'segm P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'segm overall_iou': overall_iou, 'segm mean_iou': mean_iou})
        print(eval_metrics)
        stats.update(eval_metrics)
        
    return stats


@torch.no_grad()
def evaluate_a2d(model, data_loader, postprocessor, device, args):
    model.eval()
    predictions = []
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        image_ids = [t['image_id'] for t in targets]

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs = model(samples, captions, targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        processed_outputs = postprocessor(outputs, orig_target_sizes, target_sizes)

        for p, image_id in zip(processed_outputs, image_ids):
            for s, m in zip(p['scores'], p['rle_masks']):
                    predictions.append({'image_id': image_id,
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'segmentation': m,
                                        'score': s.item()})
    
    # gather and merge predictions from all gpus
    gathered_pred_lists = utils.all_gather(predictions)
    predictions = [p for p_list in gathered_pred_lists for p in p_list]
    # evaluation
    eval_metrics = {}
    if utils.is_main_process():
        if args.dataset_file == 'a2d':
            coco_gt = COCO(os.path.join(args.a2d_path, 'a2d_sentences_test_annotations_in_coco_format.json'))
        elif args.dataset_file == 'jhmdb':
            coco_gt = COCO(os.path.join(args.jhmdb_path, 'jhmdb_sentences_gt_annotations_in_coco_format.json'))
        else:
            raise NotImplementedError
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        ap_metrics = coco_eval.stats[:6]
        eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
        # Precision and IOU
        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
        print(eval_metrics)

    # sync all processes before starting a new epoch or exiting
    dist.barrier()
    return eval_metrics
# Co-teaching training function
def train_one_epoch_coteaching(model1: torch.nn.Module, model2: torch.nn.Module,
                               criterion: torch.nn.Module, data_loader: Iterable,
                               optimizer1: torch.optim.Optimizer, optimizer2: torch.optim.Optimizer,
                               device: torch.device, epoch: int, max_norm: float = 0,
                               Rt: float = 0.6):

    print("Rt:", Rt)
    # Ensure models are in training mode and criterion is also in training mode.
    model1.train()
    model2.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.9f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000  # only affects print frequency; does not change training logic

    mini_batch_size = 128
    mini_batch_d_ls = []
    model1_loss_ls = []
    model2_loss_ls = []
    current_idx = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        current_idx += 1
        mini_batch_d_ls.append((samples, targets))

        # Copy input for model2 (keep original logic)
        samples2, targets2 = samples, targets

        # Device transfer and targets processing
        samples2 = samples2.to(device)
        targets2 = utils.targets_to(targets2, device)

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        weight_dict = criterion.weight_dict

        # ===== Selection phase: forward (no backprop) =====
        with torch.no_grad():
            # model1
            outputs1 = model1(samples, captions, targets)
            loss_dict1 = criterion(outputs1, targets)
            losses1 = sum(loss_dict1[k] * weight_dict[k] for k in loss_dict1.keys() if k in weight_dict)

        # Reduce & aggregate (model1, selection phase)
        loss_dict_reduced1 = utils.reduce_dict(loss_dict1)
        loss_value1 = sum(
            loss_dict_reduced1[k] * weight_dict[k] for k in loss_dict_reduced1.keys() if k in weight_dict
        ).item()

        # ---- New: itemized logging (model1, selection phase) ----
        loss_dict_reduced1_scaled = {
            f"m1_{k}": (loss_dict_reduced1[k] * weight_dict[k])
            for k in loss_dict_reduced1.keys() if k in weight_dict
        }
        loss_dict_reduced1_unscaled = {
            f"m1_{k}_unscaled": v for k, v in loss_dict_reduced1.items()
        }
        metric_logger.update(**loss_dict_reduced1_scaled, **loss_dict_reduced1_unscaled)

        with torch.no_grad():
            # model2
            outputs2 = model2(samples2, captions, targets)
            loss_dict2 = criterion(outputs2, targets2)
            losses2 = sum(loss_dict2[k] * weight_dict[k] for k in loss_dict2.keys() if k in weight_dict)

        # Reduce & aggregate (model2, selection phase)
        loss_dict_reduced2 = utils.reduce_dict(loss_dict2)
        loss_value2 = sum(
            loss_dict_reduced2[k] * weight_dict[k] for k in loss_dict_reduced2.keys() if k in weight_dict
        ).item()

        # ---- New: itemized logging (model2, selection phase) ----
        loss_dict_reduced2_scaled = {
            f"m2_{k}": (loss_dict_reduced2[k] * weight_dict[k])
            for k in loss_dict_reduced2.keys() if k in weight_dict
        }
        loss_dict_reduced2_unscaled = {
            f"m2_{k}_unscaled": v for k, v in loss_dict_reduced2.items()
        }
        metric_logger.update(**loss_dict_reduced2_scaled, **loss_dict_reduced2_unscaled)

        # Keep original: total loss (for display and global average)
        model1_loss_ls.append(losses1.item())
        model2_loss_ls.append(losses2.item())
        metric_logger.update(loss1=loss_value1, loss2=loss_value2)
        metric_logger.update(lr=optimizer1.param_groups[0]["lr"])

        # ===== After reaching cache threshold, perform Co-teaching sample selection and update (keep original logic) =====
        if len(mini_batch_d_ls) == mini_batch_size or current_idx == len(data_loader):
            retain_num = int(len(model1_loss_ls) * Rt)
            selected_indices1 = np.argsort(model1_loss_ls)[:retain_num]
            selected_indices2 = np.argsort(model2_loss_ls)[:retain_num]

            # Cross-select clean samples
            clean_batch_d_model1_ls = [mini_batch_d_ls[i] for i in selected_indices2]  # for model1
            clean_batch_d_model2_ls = [mini_batch_d_ls[i] for i in selected_indices1]  # for model2

            # ===== Update model2 (using clean_batch_d_model2_ls) =====
            for samples, targets in clean_batch_d_model2_ls:
                samples = samples.to(device)
                captions = [t["caption"] for t in targets]
                targets = utils.targets_to(targets, device)

                outputs2 = model2(samples, captions, targets)
                loss_dict2 = criterion(outputs2, targets)
                losses2 = sum(loss_dict2[k] * weight_dict[k] for k in loss_dict2.keys() if k in weight_dict)

                optimizer2.zero_grad()
                losses2.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm)
                optimizer2.step()

            # ===== Update model1 (using clean_batch_d_model1_ls) =====
            for samples, targets in clean_batch_d_model1_ls:
                samples = samples.to(device)
                captions = [t["caption"] for t in targets]
                targets = utils.targets_to(targets, device)

                outputs1 = model1(samples, captions, targets)
                loss_dict1 = criterion(outputs1, targets)
                losses1 = sum(loss_dict1[k] * weight_dict[k] for k in loss_dict1.keys() if k in weight_dict)

                optimizer1.zero_grad()
                losses1.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm)
                optimizer1.step()

            # Clear cache (keep original logic)
            mini_batch_d_ls = []
            model1_loss_ls = []
            model2_loss_ls = []

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
