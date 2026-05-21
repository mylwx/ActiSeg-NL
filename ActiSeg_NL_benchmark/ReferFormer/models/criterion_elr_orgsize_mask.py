import torch
import torch.nn as nn
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .segmentation import (dice_loss, sigmoid_focal_loss, sigmoid_focal_loss_weighted, sce_loss, gce_loss)
from einops import rearrange
from .criterion import SetCriterion
import torch.distributed as dist
import os
import tempfile
import shutil

class SetCriterionELR(SetCriterion):
    def __init__(self, num_examp, num_classes, matcher, weight_dict, eos_coef, losses, 
                 num_frames, focal_alpha=0.25, num_category_classes=400, beta=0.1, beta_seg=0.01,##########Adjust beta cls to be smaller for later testing
                 cls_lambda=1e-2, seg_lambda=5e-4, device='cpu',
                 cls_target_dir="cls_target", seg_target_dir="seg_target"):
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses, 
                         focal_alpha)
        
        self.num_examp = num_examp
        self.num_frames = num_frames
        self.beta = beta
        self.beta_seg = beta_seg
        self.cls_lambda = cls_lambda
        self.seg_lambda = seg_lambda
        self.device = device
        self.cls_target_dir = cls_target_dir
        self.seg_target_dir = seg_target_dir

    def initialize_cls_targets(self, clip_indices):
        """Initialize cls_target for current batch"""
        os.makedirs(self.cls_target_dir, exist_ok=True)
        for clip_idx in clip_indices:
            cls_path = os.path.join(self.cls_target_dir, f"cls_target_{clip_idx.item()}.pt")
            if not os.path.exists(cls_path):
                torch.save(torch.zeros(1, device=self.device), cls_path)

    def initialize_seg_targets(self, clip_indices, num_frames, curr_height, curr_width):
        """Initialize seg_target for current batch"""
        os.makedirs(self.seg_target_dir, exist_ok=True)
        for clip_idx in clip_indices:
            seg_path = os.path.join(self.seg_target_dir, f"seg_target_{clip_idx.item()}.pt")
            if not os.path.exists(seg_path):
                torch.save(torch.zeros(num_frames, curr_height, curr_width, device=self.device), seg_path)

    def load_cls_target(self, clip_indices):
        batch_size = len(clip_indices)
        cls_target = torch.zeros(batch_size, device=self.device)
        for i, clip_idx in enumerate(clip_indices):
            cls_path = os.path.join(self.cls_target_dir, f"cls_target_{clip_idx.item()}.pt")
            if os.path.exists(cls_path):
                try:
                    cls_target[i] = torch.load(cls_path, map_location=self.device)
                except (RuntimeError, EOFError) as e:
                    print(f"Warning: Failed to load cls_target at {cls_path}: {e}. Using zeros.")
        return cls_target

    def load_seg_target(self, clip_indices, mask_height, mask_width):
        batch_size = len(clip_indices)
        seg_target = torch.zeros(batch_size, self.num_frames, mask_height, mask_width, device=self.device)
        for i, clip_idx in enumerate(clip_indices):
            seg_path = os.path.join(self.seg_target_dir, f"seg_target_{clip_idx.item()}.pt")
            if os.path.exists(seg_path):
                try:
                    seg_target[i] = torch.load(seg_path, map_location=self.device)
                except (RuntimeError, EOFError) as e:
                    print(f"Warning: Failed to load seg_target at {seg_path}: {e}. Using zeros.")
        return seg_target

    def save_targets(self, clip_indices, cls_target=None, seg_target=None):
        if cls_target is not None:
            for i, clip_idx in enumerate(clip_indices):
                cls_path = os.path.join(self.cls_target_dir, f"cls_target_{clip_idx.item()}.pt")
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    torch.save(cls_target[i].cpu(), tmp_file.name)
                    shutil.move(tmp_file.name, cls_path)
        if seg_target is not None:
            for i, clip_idx in enumerate(clip_indices):
                seg_path = os.path.join(self.seg_target_dir, f"seg_target_{clip_idx.item()}.pt")
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    torch.save(seg_target[i].cpu(), tmp_file.name)
                    shutil.move(tmp_file.name, seg_path)

    def loss_weighted_masks(self, clip_indices, outputs, targets, indices, num_boxes, epoch, start_epoch):
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks.transpose(1, 2)

        target_masks, _ = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                        size_divisibility=32, split=False).decompose()
        weights, _ = nested_tensor_from_tensor_list([t["weights"] for t in targets], 
                                                   size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks)
        weights = weights.to(src_masks)

        batch_size = len(targets)
        num_frames = self.num_frames

        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        weights = weights[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        curr_height, curr_width = target_masks.shape[-2], target_masks.shape[-1]

        # Initialize seg_target in the first training epoch
        if epoch == start_epoch:
            self.initialize_seg_targets(clip_indices, num_frames, curr_height, curr_width)

        src_masks = src_masks[src_idx]
        src_masks_flat = src_masks.flatten(1)
        target_masks_flat = target_masks.flatten(1)
        weights_flat = weights.flatten(1)

        frame_offsets = torch.arange(num_frames, device=self.device).repeat(batch_size)
        seg_indices = clip_indices.repeat_interleave(num_frames) * num_frames + frame_offsets

        seg_target = self.load_seg_target(clip_indices, curr_height, curr_width)
        seg_target_current = seg_target
        seg_target_current_flat = seg_target_current.flatten(1)

        seg_pred_pos = torch.sigmoid(src_masks_flat)
        seg_pred_pos = torch.clamp(seg_pred_pos, 1e-4, 1.0 - 1e-4)
        seg_pred_neg = 1 - seg_pred_pos

        seg_pred_detach = seg_pred_pos.data.detach()
        # if weights_flat is not None:
        #     weighted_seg_pred = seg_pred_detach * weights_flat
        #     seg_target_update = (self.beta_seg * seg_target_current_flat + 
        #                         (1 - self.beta_seg) * weighted_seg_pred)
        # else:
        #     seg_target_update = (self.beta_seg * seg_target_current_flat + 
        #                         (1 - self.beta_seg) * seg_pred_detach)
        seg_target_update = (self.beta_seg * seg_target_current_flat + 
                            (1 - self.beta_seg) * seg_pred_detach)
        seg_target = seg_target_update.view(batch_size, num_frames, curr_height, curr_width)

        loss_mask = sigmoid_focal_loss_weighted(src_masks_flat, target_masks_flat, num_boxes, weights_flat,
                                               alpha=self.focal_alpha, gamma=2)
        loss_dice = dice_loss(src_masks_flat, target_masks_flat, num_boxes)

        seg_target_pos = seg_target_current_flat
        seg_target_neg = 1 - seg_target_pos
        seg_pred_binary = torch.stack([seg_pred_neg, seg_pred_pos], dim=-1)
        seg_target_binary = torch.stack([1 - target_masks_flat, target_masks_flat], dim=-1)
        ############elr_reg = torch.relu(-((1 - (seg_pred_binary * seg_target_binary).sum(dim=-1)).log().mean()))
        elr_reg = (1 - (seg_pred_binary * seg_target_binary).sum(dim=-1)).log().mean()
        final_loss = loss_mask + self.seg_lambda * elr_reg
        losses = {
            "loss_mask": final_loss,
            "loss_dice": loss_dice,
        }

        self.save_targets(clip_indices, seg_target=seg_target)

        return losses

    def loss_masks(self, clip_indices, outputs, targets, indices, num_boxes, epoch, start_epoch):
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks.transpose(1, 2)

        target_masks, _ = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                        size_divisibility=32, split=False).decompose()
        # weights, _ = nested_tensor_from_tensor_list([t["weights"] for t in targets], 
        #                                            size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks)
        # weights = weights.to(src_masks)

        batch_size = len(targets)
        num_frames = self.num_frames

        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        # weights = weights[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        curr_height, curr_width = target_masks.shape[-2], target_masks.shape[-1]

        # Initialize seg_target in the first training epoch
        if epoch == start_epoch:
            self.initialize_seg_targets(clip_indices, num_frames, curr_height, curr_width)

        src_masks = src_masks[src_idx]
        src_masks_flat = src_masks.flatten(1)
        target_masks_flat = target_masks.flatten(1)
        # weights_flat = weights.flatten(1)

        frame_offsets = torch.arange(num_frames, device=self.device).repeat(batch_size)
        seg_indices = clip_indices.repeat_interleave(num_frames) * num_frames + frame_offsets

        seg_target = self.load_seg_target(clip_indices, curr_height, curr_width)
        seg_target_current = seg_target
        seg_target_current_flat = seg_target_current.flatten(1)

        seg_pred_pos = torch.sigmoid(src_masks_flat)
        seg_pred_pos = torch.clamp(seg_pred_pos, 1e-4, 1.0 - 1e-4)
        seg_pred_neg = 1 - seg_pred_pos

        seg_pred_detach = seg_pred_pos.data.detach()
        # if weights_flat is not None:
        #     weighted_seg_pred = seg_pred_detach * weights_flat
        #     seg_target_update = (self.beta_seg * seg_target_current_flat + 
        #                         (1 - self.beta_seg) * weighted_seg_pred)
        # else:
        #     seg_target_update = (self.beta_seg * seg_target_current_flat + 
        #                         (1 - self.beta_seg) * seg_pred_detach)
        seg_target_update = (self.beta_seg * seg_target_current_flat + 
                            (1 - self.beta_seg) * seg_pred_detach)
        seg_target = seg_target_update.view(batch_size, num_frames, curr_height, curr_width)
        #######sigmoid_focal_loss(src_masks, target_masks, num_boxes),
        loss_mask = sigmoid_focal_loss(src_masks_flat, target_masks_flat, num_boxes)
        # loss_mask = sigmoid_focal_loss_weighted(src_masks_flat, target_masks_flat, num_boxes,
        #                                        alpha=self.focal_alpha, gamma=2)
        loss_dice = dice_loss(src_masks_flat, target_masks_flat, num_boxes)

        seg_target_pos = seg_target_current_flat
        seg_target_neg = 1 - seg_target_pos
        seg_pred_binary = torch.stack([seg_pred_neg, seg_pred_pos], dim=-1)
        seg_target_binary = torch.stack([1 - target_masks_flat, target_masks_flat], dim=-1)
        ############elr_reg = torch.relu(-((1 - (seg_pred_binary * seg_target_binary).sum(dim=-1)).log().mean()))
        elr_reg = (1 - (seg_pred_binary * seg_target_binary).sum(dim=-1)).log().mean()
        final_loss = loss_mask + self.seg_lambda * elr_reg
        losses = {
            "loss_mask": final_loss,
            "loss_dice": loss_dice,
        }

        self.save_targets(clip_indices, seg_target=seg_target)

        return losses

    def get_loss(self, loss, clip_indices, outputs, targets, indices, num_boxes, epoch, start_epoch, **kwargs):
        loss_map = {
            # 'category_labels': self.loss_category_labels,
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'positive_labels': self.loss_positive_labels,
            'weighted_masks': self.loss_weighted_masks,
            # 'weighted_sce_masks': self.loss_weighted_sce_masks,
            # 'weighted_gce_masks': self.loss_weighted_gce_masks,
            'sce_masks': self.loss_sce_masks,
            'gce_masks': self.loss_gce_masks,
            'loss_active_passive_masks': self.loss_active_passive_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if loss in [ 'weighted_masks','masks']:##'positive_labels',
            return loss_map[loss](clip_indices, outputs, targets, indices, num_boxes, epoch, start_epoch, **kwargs)
        else:
            return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, clip_idx, outputs, targets, epoch, start_epoch):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1)
        num_boxes = target_valid.sum().item()
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        clip_indices = torch.tensor([idx.item() for idx in clip_idx], device=self.device)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, clip_indices, outputs, targets, indices, num_boxes, epoch, start_epoch))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, clip_indices, aux_outputs, targets, indices, num_boxes, epoch, start_epoch, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def to(self, device):
        self.device = device
        super().to(device)