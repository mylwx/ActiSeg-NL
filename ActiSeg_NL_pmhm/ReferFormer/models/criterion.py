import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .segmentation import (dice_loss, sigmoid_focal_loss, sigmoid_focal_loss_weighted)

from einops import rearrange
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


########## Visualization
try:
    import torchvision.utils as vutils
    TORCHVISION_OK = True
except Exception:
    TORCHVISION_OK = False

##############################################
# ===== PMHM helpers =====
def symmetric_kl(p, q, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    kl_pq = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    kl_qp = q * (q / p).log() + (1 - q) * ((1 - q) / (1 - p)).log()
    return kl_pq + kl_qp

def tversky_loss(p, y, alpha=0.7, beta=0.5, eps=1e-6):
    tp = (p * y).sum()
    fp = (p * (1 - y)).sum()
    fn = ((1 - p) * y).sum()
    t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - t

def sobel_grad(prob):
    device = prob.device
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=prob.dtype, device=device).view(1,1,3,3)/8.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=prob.dtype, device=device).view(1,1,3,3)/8.0
    x = prob.unsqueeze(-3)  # (..., 1, H, W)
    gx = torch.conv2d(x, kx, padding=1)
    gy = torch.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy).squeeze(-3)
    return g


##############################################

class SetCriterion(nn.Module):
    """ This class computes the loss for ReferFormer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    # Add PMHM to SetCriterion.init 
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25,
            # ===== PMHM =====
            use_pmhm=True,
            k_agree=2,
            tau_m_p=0.20,
            tau_e_p=0.85,
            tau_h_p=0.60,
            tv_alpha_start=0.7,
            tv_alpha_end=0.5,
            tv_beta=0.5,
            use_proto=False,
            # ---- NEW: stats  ----#######Intermediate results
             pmhm_log_stats=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_alpha = focal_alpha
        self.mask_out_stride = 4
        # And after super().__init__() and existing member assignments, add these members
        # NEW PMHM cfg
        self.use_pmhm = bool(use_pmhm)
        self.k_agree = int(k_agree)
        self.tau_m_p = float(tau_m_p)
        self.tau_e_p = float(tau_e_p)
        self.tau_h_p = float(tau_h_p)
        self.tv_alpha_start = float(tv_alpha_start)
        self.tv_alpha_end = float(tv_alpha_end)
        self.tv_beta = float(tv_beta)
        self.use_proto = bool(use_proto)
        self._epoch = 0
        self._max_epoch = 1

        ####Intermediate results
        #################
        self.pmhm_log_stats = bool(pmhm_log_stats)
        #################


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] 
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')

        # judge the valid frames
        valid_indices = []
        valids = [target['valid'] for target in targets]
        for valid, (indice_i, indice_j) in zip(valids, indices): 
            valid_ind = valid.nonzero().flatten() 
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices) # NOTE: use valid indices
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) 
        if self.num_classes == 1: # binary referred, positive
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        #print('target_loss_label',target_classes_onehot)
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            pass
        return losses

    def loss_positive_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        edited label loss to classify pos/neg
        """
        assert 'pred_positives' in outputs
        src_logits = outputs['pred_positives'] 
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.float, device=src_logits.device) 
        for id,t in enumerate(targets):
            if t['positive'].any():
                target_classes[id] = 1
        loss_ce = sigmoid_focal_loss(src_logits, target_classes, num_boxes=src_logits.shape[0], alpha=self.focal_alpha, gamma=2)
        losses = {'loss_positive_labels': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            pass
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']  
        bs, nf, nq = src_boxes.shape[:3]
        src_boxes = src_boxes.transpose(1, 2)  

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]  
        src_boxes = src_boxes.flatten(0, 1)  # [b*t, 4]

        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)  # [b*t, 4]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def loss_weighted_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
           targets dicts must contain the key "weights" containing a tensor of dim [nb_target_boxes, h, w]
           weights are for focal loss, should be the same shape as target['masks']
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"] 
        src_masks = src_masks.transpose(1, 2) 

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                              size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks)

        # what is this step ?
        #print('before:')
        #print([torch.unique(t['weights']) for t in targets]) 
        weights, valid = nested_tensor_from_tensor_list([t["weights"] for t in targets], 
                                                              size_divisibility=32, split=False).decompose() 
        weights = weights.to(src_masks)
        #print('after:')
        #print(torch.unique(weights))

        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        weights = weights[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        src_masks = src_masks[src_idx] 
        # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False) 
        src_masks = src_masks.flatten(1) # [b, thw]
    
        target_masks = target_masks.flatten(1) # [b, thw]
        weights = weights.flatten(1)
        
        losses = {
            "loss_mask": sigmoid_focal_loss_weighted(src_masks, target_masks, num_boxes, weights),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"] 
        src_masks = src_masks.transpose(1, 2) 

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                              size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks) 

        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        src_masks = src_masks[src_idx] 
        # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False) 
        src_masks = src_masks.flatten(1) # [b, thw]

        target_masks = target_masks.flatten(1) # [b, thw]

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    @torch.no_grad()
    def _build_regions(self, main_prob):  # [B,T,Q,H',W']
        M = (main_prob - 0.5).abs()
        E = sobel_grad(main_prob.flatten(0,2)).view_as(main_prob)  # edge mag

        B = main_prob.shape[0]
        U_masks = []
        tau_h_list = []
        for b in range(B):
            m = M[b].flatten()
            e = E[b].flatten()
            tau_m = torch.quantile(m, self.tau_m_p).item()
            tau_e = torch.quantile(e, self.tau_e_p).item()
            tau_h = torch.quantile(m, self.tau_h_p).item()
            U_b = ((M[b] < tau_m) | (E[b] > tau_e))
            U_masks.append(U_b)
            tau_h_list.append(tau_h)
        U = torch.stack(U_masks, dim=0)  # bool
        tau_h_tensor = main_prob.new_tensor(tau_h_list).view(B,1,1,1,1)
        return M, U, tau_h_tensor

    def masks_robust(self, outputs, targets, indices, num_boxes):
        assert "pred_masks" in outputs
        main = outputs["pred_masks"]                      # [B,T,Q,H',W']
        main_prob = main.sigmoid()
        src_idx = self._get_src_permutation_idx(indices)

        aux_list = outputs.get("pred_masks_heads", None)
        n_aux = 0 if aux_list is None else len(aux_list)
        aux_probs = [m.sigmoid() for m in aux_list] if n_aux > 0 else []

        M, U, tau_h = self._build_regions(main_prob)

        votes = [ (main_prob > 0.5).float() ]
        for p in aux_probs:
            votes.append( (p > 0.5).float() )
        votes = torch.stack(votes, dim=0)                 # [H_total,B,T,Q,H',W']
        vote_sum = votes.sum(dim=0)
        H_total = votes.shape[0]
        k_agree = min(self.k_agree, H_total)

        highconf = (M >= tau_h)
        S_fg = (vote_sum >= k_agree) & highconf
        S_bg = ((H_total - vote_sum) >= k_agree) & highconf
        S = S_fg | S_bg

        ################
        # downsample GT
        target_masks, _ = nested_tensor_from_tensor_list([t["masks"] for t in targets],
                                                         size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(main)
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        # src_idx = self._get_src_permutation_idx(indices)

        # Main head matched logits
        main_logit = outputs["pred_masks"].transpose(1,2)[src_idx]   # [B*T,H',W']
        main_logit = main_logit.flatten(1)                            # [B*T,THW]
        tgt = target_masks.flatten(1)                                 # [B,THW]

        # Region weights S
        S_all = S.transpose(1,2)                # [B,Q,T,H',W']
        S_sel = S_all[src_idx].flatten(1).float()
        S_weight = (S_sel > 0).float()

        # Strong supervision on S
        loss_mask = sigmoid_focal_loss_weighted(main_logit, tgt, num_boxes, S_weight)
        loss_dice = dice_loss(main_logit * S_weight, tgt * S_weight, num_boxes)

        # Tversky annealing on U
        if self._max_epoch > 1:
            a = self.tv_alpha_start + (self.tv_alpha_end - self.tv_alpha_start) * float(self._epoch) / float(self._max_epoch - 1)
        else:
            a = self.tv_alpha_start
        b = self.tv_beta

        U_all = U.transpose(1,2)                    # [B,Q,T,H',W']
        U_sel = U_all[src_idx]                      # [B*T,H',W']
        main_prob_sel = main_prob.transpose(1,2)[src_idx]
        tgt_sel = target_masks.flatten(0,1).view_as(main_prob_sel)
        loss_tv = tversky_loss(main_prob_sel * U_sel.float(), tgt_sel * U_sel.float(), alpha=a, beta=b)

        # Head consistency on U
        loss_head = main_prob.new_tensor(0.0)
        if n_aux > 0:
            for p_aux in aux_probs:
                aux_sel = p_aux.transpose(1,2)[src_idx]
                kl = symmetric_kl(main_prob_sel, aux_sel)
                mask = U_sel.float()
                denom = mask.sum() + 1e-6
                loss_head = loss_head + (kl * mask).sum() / denom
            loss_head = loss_head / float(n_aux)

        # Layer consistency on U
        loss_layer = main_prob.new_tensor(0.0)
        if 'aux_outputs' in outputs:
            aux_layers = [a['pred_masks'] for a in outputs['aux_outputs']]
            num = 0
            for m in aux_layers:
                p_l = m.sigmoid().transpose(1,2)[src_idx]
                kl = symmetric_kl(main_prob_sel, p_l)
                mask = U_sel.float()
                denom = mask.sum() + 1e-6
                loss_layer = loss_layer + (kl * mask).sum() / denom
                num += 1
            if num > 0:
                loss_layer = loss_layer / float(num)

        losses = {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "loss_tv": loss_tv,
            "loss_head": loss_head,
            "loss_layer": loss_layer
        }

        ################
        if self.pmhm_log_stats:
            # Select matched queries consistent with main supervision
            S_all = S.transpose(1,2)               # [B,Q,T,H',W']
            U_all = U.transpose(1,2)               # [B,Q,T,H',W']

            S_sel = S_all[src_idx].float().flatten(1)  # [B*T, THW]
            U_sel = U_all[src_idx].float().flatten(1)

            # Ratio
            S_ratio = (S_sel.mean()).detach()
            U_ratio = (U_sel.mean()).detach()

            # Quantile threshold (batch average)
            tau_m_e_h = torch.tensor([
                torch.quantile((M).flatten(1), self.tau_m_p, dim=1).mean().item(),
                torch.quantile((sobel_grad(main_prob.flatten(0,2)).view_as(main_prob)).flatten(1),
                            self.tau_e_p, dim=1).mean().item(),
                tau_h.mean().item()
            ], device=main_prob.device)

            # Voting mean (with k_agree context)
            H_total = 1 + (len(aux_probs) if len(aux_probs) > 0 else 0)
            vote_sum = ((main_prob > 0.5).float()
                        + sum([(p > 0.5).float() for p in aux_probs]))  # [B,T,Q,H',W']
            vote_sum = vote_sum.transpose(1,2)                          # [B,Q,T,H',W']
            vote_sel_mean = vote_sum[src_idx].float().mean().detach()

            # KL magnitude (if computed)
            head_kl = loss_head.detach() if 'loss_head' in locals() else main_prob.new_tensor(0.0)
            layer_kl = loss_layer.detach() if 'loss_layer' in locals() else main_prob.new_tensor(0.0)

            # Write back to losses
            losses.update({
                'stat_pmhm_S_ratio': S_ratio,
                'stat_pmhm_U_ratio': U_ratio,
                'stat_pmhm_tau_m': main_prob.new_tensor(tau_m_e_h[0]),
                'stat_pmhm_tau_e': main_prob.new_tensor(tau_m_e_h[1]),
                'stat_pmhm_tau_h': main_prob.new_tensor(tau_m_e_h[2]),
                'stat_pmhm_vote_mean': vote_sel_mean,
                'stat_pmhm_head_kl': head_kl,
                'stat_pmhm_layer_kl': layer_kl,
                'stat_pmhm_H_total': main_prob.new_tensor(float(H_total)),
                'stat_pmhm_k_agree': main_prob.new_tensor(float(self.k_agree)),
            })
        return losses

    #####################################################
    # Robust mask loss masks_robust

    #####################################################
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'positive_labels': self.loss_positive_labels,
            'weighted_masks': self.loss_weighted_masks,
            'masks_robust': self.masks_robust,#### Register new loss name in get_loss
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k not in ('aux_outputs', 'viz_rgb')}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1) # [B, T] -> [B*T]
        num_boxes = target_valid.sum().item() 
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # Change to: follow a tensor that is definitely on GPU, such as pred_masks
        ref_device = outputs['pred_masks'].device
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=ref_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


