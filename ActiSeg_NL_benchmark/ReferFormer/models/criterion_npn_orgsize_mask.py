import torch
import torch.nn as nn
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .segmentation import (dice_loss, sigmoid_focal_loss, sigmoid_focal_loss_weighted)
from einops import rearrange
from .criterion import SetCriterion
import torch.distributed as dist
import os
import tempfile
import shutil

class SetCriterionNPN(SetCriterion):
    def __init__(self, num_examp, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_frames, focal_alpha=0.25, num_category_classes=310,
                 alpha=1e-1, beta=2e-1, topk=1, warmup_epochs=2, start_epoch=0,#alpha=1.0, beta=2.0, topk=1, warmup_epochs=6, start_epoch=0,
                 device='cpu', candidate_dir_pos="candidate_pos", candidate_dir_masks="candidate_masks"):
        """Initialize SetCriterionNPN, inherited from SetCriterion"""
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha)
        
        self.num_examp = num_examp
        self.num_frames = num_frames
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.warmup_epochs = warmup_epochs
        self.start_epoch = start_epoch
        self.device = device
        self.candidate_dir_pos = candidate_dir_pos
        self.candidate_dir_masks = candidate_dir_masks

    def initialize_cls_candidate_targets(self, clip_indices, num_classes_pos):
        """Initialize candidate_count for cls_pos"""
        os.makedirs(self.candidate_dir_pos, exist_ok=True)
        for clip_idx in clip_indices:
            pos_path = os.path.join(self.candidate_dir_pos, f"candidate_pos_{clip_idx.item()}.pt")
            if not os.path.exists(pos_path):
                torch.save(torch.zeros(num_classes_pos, device=self.device), pos_path)

    def initialize_masks_candidate_targets(self, clip_indices, num_frames, curr_height, curr_width):
        """Initialize candidate_count for masks"""
        os.makedirs(self.candidate_dir_masks, exist_ok=True)
        for clip_idx in clip_indices:
            masks_path = os.path.join(self.candidate_dir_masks, f"candidate_masks_{clip_idx.item()}.pt")
            if not os.path.exists(masks_path):
                torch.save(torch.zeros(num_frames, curr_height, curr_width, 2, device=self.device), masks_path)

    def load_candidate_pos(self, clip_indices, num_classes_pos):
        """Load cls_pos candidate_count for current batch"""
        batch_size = len(clip_indices)
        candidate_pos = torch.zeros(batch_size, num_classes_pos, device=self.device)
        for i, clip_idx in enumerate(clip_indices):
            pos_path = os.path.join(self.candidate_dir_pos, f"candidate_pos_{clip_idx.item()}.pt")
            if os.path.exists(pos_path):
                try:
                    candidate_pos[i] = torch.load(pos_path, map_location=self.device)
                except Exception as e:
                    print(f"Warning: Failed to load candidate_pos at {pos_path}: {e}. Using zeros.")
        return candidate_pos

    def load_candidate_masks(self, clip_indices, num_frames, mask_height, mask_width):
        """Load masks candidate_count for current batch"""
        batch_size = len(clip_indices)
        candidate_masks = torch.zeros(batch_size, num_frames, mask_height, mask_width, 2, device=self.device)
        for i, clip_idx in enumerate(clip_indices):
            masks_path = os.path.join(self.candidate_dir_masks, f"candidate_masks_{clip_idx.item()}.pt")
            if os.path.exists(masks_path):
                try:
                    candidate_masks[i] = torch.load(masks_path, map_location=self.device)
                except Exception as e:
                    print(f"Warning: Failed to load candidate_masks at {masks_path}: {e}. Using zeros.")
        return candidate_masks

    def save_candidate_targets(self, clip_indices, candidate_pos=None, candidate_masks=None):
        """Save candidate_count for current batch"""
        if candidate_pos is not None:
            for i, clip_idx in enumerate(clip_indices):
                pos_path = os.path.join(self.candidate_dir_pos, f"candidate_pos_{clip_idx.item()}.pt")
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    torch.save(candidate_pos[i].cpu().to(torch.int8), tmp_file.name)
                    shutil.move(tmp_file.name, pos_path)
        if candidate_masks is not None:
            for i, clip_idx in enumerate(clip_indices):
                masks_path = os.path.join(self.candidate_dir_masks, f"candidate_masks_{clip_idx.item()}.pt")
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    torch.save(candidate_masks[i].cpu().to(torch.int8), tmp_file.name)
                    shutil.move(tmp_file.name, masks_path)

    def _update_candidate_masks(self, src_masks_flat, candidate_masks, clip_indices, batch_size, target_masks_flat=None):
        num_frames, height, width = candidate_masks.shape[1:4]
        num_pixels = height * width
        total_pixels = num_frames * num_pixels

        pred_masks_prob = src_masks_flat.sigmoid().clamp(min=1e-4, max=1.0)  # [B * num_frames, num_pixels]
        prob_class_1 = pred_masks_prob
        prob_class_0 = 1 - pred_masks_prob
        pred_masks_prob_full = torch.stack([prob_class_0, prob_class_1], dim=-1)  # [B * num_frames, num_pixels, 2]
        pred_masks_prob_full = pred_masks_prob_full.view(batch_size, total_pixels, 2)  # [B, num_frames * num_pixels, 2]

        max_probs, pred_masks_topk = pred_masks_prob_full.max(dim=-1)  # [B, num_frames * num_pixels]
        tau = 0.7  # Dynamic threshold
        high_conf_mask = max_probs > tau  # [B, num_frames * num_pixels]

        batch_idx = torch.arange(batch_size).view(-1, 1).expand(batch_size, total_pixels)
        pixel_idx = torch.arange(total_pixels).view(1, -1).expand(batch_size, total_pixels)

        candidate_masks_reshaped = candidate_masks.view(batch_size, total_pixels, 2)
        if target_masks_flat is not None:
            target_masks_flat_reshaped = target_masks_flat.view(batch_size, total_pixels).long()
            candidate_masks_reshaped.scatter_(2, target_masks_flat_reshaped.unsqueeze(-1), 1)
        candidate_masks_reshaped[batch_idx[high_conf_mask], pixel_idx[high_conf_mask], pred_masks_topk[high_conf_mask]] += 1
        candidate_masks = candidate_masks_reshaped.view(batch_size, num_frames, height, width, 2)

        self.save_candidate_targets(clip_indices, candidate_masks=candidate_masks)

    def loss_masks(self, clip_indices, outputs_w, outputs_s, targets, indices, num_boxes, epoch):
        assert "pred_masks" in outputs_w
        src_idx = self._get_src_permutation_idx(indices)
        src_masks_w = outputs_w["pred_masks"].transpose(1, 2)

        target_masks, _ = nested_tensor_from_tensor_list([t["masks"] for t in targets], size_divisibility=32, split=False).decompose()
        weights, _ = nested_tensor_from_tensor_list([t["weights"] for t in targets], size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks_w)
        weights = weights.to(src_masks_w)

        batch_size = len(targets)
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        curr_height, curr_width = target_masks.shape[-2], target_masks.shape[-1]
        num_pixels = curr_height * curr_width
        num_frames = self.num_frames
        total_pixels = self.num_frames * num_pixels

        src_masks_w = src_masks_w[src_idx]
        src_masks_flat_w = src_masks_w.flatten(1)  # [B * num_frames, num_pixels]
        target_masks_flat = target_masks.flatten(1)  # [B, num_frames * num_pixels]

        # Initialize masks candidate set
        if epoch == self.start_epoch:
            self.initialize_masks_candidate_targets(clip_indices, self.num_frames, curr_height, curr_width)

        # Load candidate_masks
        candidate_masks = self.load_candidate_masks(clip_indices, self.num_frames, curr_height, curr_width)

        if epoch < self.warmup_epochs:
            loss_mask = sigmoid_focal_loss(src_masks_flat_w, target_masks_flat, num_boxes)            
            loss_dice = dice_loss(src_masks_flat_w, target_masks_flat, num_boxes)
            self._update_candidate_masks(src_masks_flat_w, candidate_masks, clip_indices, batch_size, target_masks_flat=target_masks_flat)
            return {"loss_mask": loss_mask, "loss_dice": loss_dice}

        # Robust Training phase
        assert "pred_masks" in outputs_s
        src_masks_s = outputs_s["pred_masks"].transpose(1, 2)
        src_masks_s = src_masks_s[src_idx]
        src_masks_flat_s = src_masks_s.flatten(1)  # [B * num_frames, num_pixels]

        # Convert to 2D probabilities
        pred_masks_prob_w = src_masks_flat_w.sigmoid().clamp(min=1e-4, max=1.0)  # [B * num_frames, num_pixels]
        pred_masks_neg_prob_w = 1 - pred_masks_prob_w
        pred_masks_prob_w_2d = torch.stack([pred_masks_neg_prob_w, pred_masks_prob_w], dim=2)  # [B * num_frames, num_pixels, 2]
        pred_masks_prob_w_2d = pred_masks_prob_w_2d.view(batch_size, total_pixels, 2)  # [B, num_frames * num_pixels, 2]

        pred_masks_prob_s = src_masks_flat_s.sigmoid().clamp(min=1e-4, max=1.0)
        pred_masks_neg_prob_s = 1 - pred_masks_prob_s
        pred_masks_prob_s_2d = torch.stack([pred_masks_neg_prob_s, pred_masks_prob_s], dim=2)  # [B * num_frames, num_pixels, 2]
        pred_masks_prob_s_2d = pred_masks_prob_s_2d.view(batch_size, total_pixels, 2)  # [B, num_frames * num_pixels, 2]

        # PLL
        tau = 0.7
        max_probs, pesudo_masks = pred_masks_prob_w_2d.max(dim=-1)  # [B, num_frames * num_pixels]
        candidate_label_masks = torch.zeros(batch_size, total_pixels, 2, device=self.device)
        target_masks_flat_reshaped = target_masks_flat.unsqueeze(-1).long()
        candidate_label_masks.scatter_(2, target_masks_flat_reshaped, 1)
        high_conf_mask = max_probs > tau
        batch_idx = torch.arange(batch_size).view(-1, 1).expand(batch_size, total_pixels)
        pixel_idx = torch.arange(total_pixels).view(1, -1).expand(batch_size, total_pixels)
        candidate_label_masks[batch_idx[high_conf_mask], pixel_idx[high_conf_mask], pesudo_masks[high_conf_mask]] = 1

        # Update complete candidate_label_masks
        candidate_masks_reshaped = candidate_masks.view(batch_size, total_pixels, 2)
        candidate_masks_reshaped += candidate_label_masks
        candidate_masks = candidate_masks_reshaped.view(batch_size, num_frames, curr_height, curr_width, 2)
        self.save_candidate_targets(clip_indices, candidate_masks=candidate_masks)

        # Calculate PLL loss
        _, y_ce_masks = torch.max(candidate_masks.view(batch_size, total_pixels, 2), dim=-1)  # [B, num_frames * num_pixels]
        src_masks_flat_w_2d = torch.stack([-src_masks_flat_w, src_masks_flat_w], dim=2).view(batch_size, total_pixels, 2)
        L_PLL_masks = F.cross_entropy(src_masks_flat_w_2d.transpose(1, 2), y_ce_masks, reduction='none')
        weights_masks = candidate_masks.max(dim=-1)[0] / (candidate_masks.sum(dim=-1) + 1e-10)
        weights_masks = weights_masks.view(batch_size, total_pixels)
        L_PLL_masks = (L_PLL_masks * weights_masks).mean()

        # NL
        complementary_label_masks = 1 - candidate_label_masks
        L_NL_masks = -torch.mean(torch.sum(torch.log(1 - pred_masks_prob_w_2d + 1e-7) * complementary_label_masks, dim=-1))

        # CR
        eta = 0.7
        high_conf_cr_mask = max_probs > eta
        src_masks_flat_s_2d = torch.stack([-src_masks_flat_s, src_masks_flat_s], dim=2).view(batch_size, total_pixels, 2)
        L_CR_masks = F.cross_entropy(src_masks_flat_s_2d[high_conf_cr_mask], 
                                    pesudo_masks[high_conf_cr_mask], reduction='mean') if high_conf_cr_mask.any() else torch.tensor(0.0, device=self.device)

        # Total loss
        final_loss = L_PLL_masks + self.alpha * L_NL_masks + self.beta * L_CR_masks
        return {"loss_mask": final_loss, "loss_dice": dice_loss(src_masks_flat_w, target_masks_flat, num_boxes)}

    def get_loss(self, loss, clip_indices, outputs_w, outputs_s, targets, indices, num_boxes, epoch, **kwargs):
        loss_map = {
            # 'category_labels': self.loss_category_labels,
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'positive_labels': self.loss_positive_labels,
            'weighted_masks': self.loss_weighted_masks,
            'sce_masks': self.loss_sce_masks,
            'gce_masks': self.loss_gce_masks,
            'loss_active_passive_masks': self.loss_active_passive_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if loss in ['masks']:###'weighted_masks',###'positive_labels', 
            return loss_map[loss](clip_indices, outputs_w, outputs_s, targets, indices, num_boxes, epoch, **kwargs)
        return loss_map[loss](outputs_w, targets, indices, num_boxes, **kwargs)

    def forward(self, clip_idx, outputs_w, outputs_s, targets, epoch):
        """Compute all losses"""
        outputs_without_aux_w = {k: v for k, v in outputs_w.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux_w, targets)

        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1)
        num_boxes = target_valid.sum().item()
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs_w.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        clip_indices = torch.tensor([idx.item() for idx in clip_idx], device=self.device)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, clip_indices, outputs_w, outputs_s, targets, indices, num_boxes, epoch))

        if 'aux_outputs' in outputs_w and (epoch >= self.warmup_epochs or 'aux_outputs' not in outputs_s):
            aux_outputs_s = outputs_s.get('aux_outputs', [{}] * len(outputs_w['aux_outputs']))
            for i, (aux_outputs_w, aux_outputs_s_i) in enumerate(zip(outputs_w['aux_outputs'], aux_outputs_s)):
                indices = self.matcher(aux_outputs_w, targets)
                for loss in self.losses:
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, clip_indices, aux_outputs_w, aux_outputs_s_i, targets, indices, num_boxes, epoch, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def to(self, device):
        self.device = device
        super().to(device)