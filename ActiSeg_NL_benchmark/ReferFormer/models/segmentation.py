"""
Segmentaion Part 
Modified from DETR (https://github.com/facebookresearch/detr)
"""
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

from einops import rearrange, repeat

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

import fvcore.nn.weight_init as weight_init

from .position_encoding import PositionEmbeddingSine1D

BN_MOMENTUM = 0.1

def get_norm(norm, out_channels): # only support GN or LN
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(8, channels),
            "LN": lambda channels: nn.LayerNorm(channels)
        }[norm]
    return norm(out_channels)

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# FPN structure
class CrossModalFPNDecoder(nn.Module):
    def __init__(self, feature_channels: List, conv_dim: int, mask_dim: int, dim_feedforward: int = 2048, norm=None):
        """
        Args:
            feature_channels: list of fpn feature channel numbers.
            conv_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            dim_feedforward: number of vision-language fusion module ffn channel numbers.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.feature_channels = feature_channels

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            # in_channels: 4x -> 32x
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = idx+1
            self.add_module("adapter_{}".format(stage), lateral_conv)
            self.add_module("layer_{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
            
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        # vision-language cross-modal fusion
        self.text_pos = PositionEmbeddingSine1D(conv_dim, normalize=True)
        sr_ratios = [8, 4, 2, 1]
        cross_attns = []
        for idx in range(len(feature_channels)): # res2 -> res5
            cross_attn = VisionLanguageBlock(conv_dim, dim_feedforward=dim_feedforward,
                                             nhead=8, sr_ratio=sr_ratios[idx])
            for p in cross_attn.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            stage = int(idx + 1)
            self.add_module("cross_attn_{}".format(stage), cross_attn)
            cross_attns.append(cross_attn)
        # place cross-attn in top-down order (from low to high resolution)
        self.cross_attns = cross_attns[::-1]


    def forward_features(self, features, text_features, poses, memory, nf):
        # nf: num_frames
        text_pos = self.text_pos(text_features).permute(2, 0, 1)   # [length, batch_size, c]  
        text_features, text_masks = text_features.decompose()      
        text_features = text_features.permute(1, 0, 2)   

        for idx, (mem, f, pos) in enumerate(zip(memory[::-1], features[1:][::-1], poses[1:][::-1])): # 32x -> 8x
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cross_attn = self.cross_attns[idx]
            
            _, x_mask = f.decompose()
            n, c, h, w = pos.shape
            b = n // nf
            t = nf

            # NOTE: here the (h, w) is the size for current fpn layer
            vision_features = lateral_conv(mem)  # [b*t, c, h, w]
            vision_features = rearrange(vision_features, '(b t) c h w -> (t h w) b c', b=b, t=t)
            vision_pos = rearrange(pos, '(b t) c h w -> (t h w) b c', b=b, t=t)
            vision_masks = rearrange(x_mask, '(b t) h w -> b (t h w)', b=b, t=t)

            cur_fpn = cross_attn(tgt=vision_features,
                                 memory=text_features,
                                 t=t, h=h, w=w,
                                 tgt_key_padding_mask=vision_masks,
                                 memory_key_padding_mask=text_masks,
                                 pos=text_pos,
                                 query_pos=vision_pos
            ) # [t*h*w, b, c]
            cur_fpn = rearrange(cur_fpn, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            # upsample
            if idx == 0: # top layer
                y = output_conv(cur_fpn)
            else:
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        
        # 4x level
        lateral_conv = self.lateral_convs[-1]
        output_conv = self.output_convs[-1]
        cross_attn = self.cross_attns[-1]
        
        x, x_mask = features[0].decompose()
        pos = poses[0]
        n, c, h, w = pos.shape
        b = n // nf
        t = nf

        vision_features = lateral_conv(x)  # [b*t, c, h, w]
        vision_features = rearrange(vision_features, '(b t) c h w -> (t h w) b c', b=b, t=t)
        vision_pos = rearrange(pos, '(b t) c h w -> (t h w) b c', b=b, t=t)
        vision_masks = rearrange(x_mask, '(b t) h w -> b (t h w)', b=b, t=t)

        cur_fpn = cross_attn(tgt=vision_features,
                             memory=text_features,
                             t=t, h=h, w=w,
                             tgt_key_padding_mask=vision_masks,
                             memory_key_padding_mask=text_masks,
                             pos=text_pos,
                             query_pos=vision_pos
        ) # [t*h*w, b, c]
        cur_fpn = rearrange(cur_fpn, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
        # Following FPN implementation, we use nearest upsampling here
        y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
        y = output_conv(y)
        return y   # [b*t, c, h, w], the spatial stride is 4x

    def forward(self, features, text_features, pos, memory, nf):
        """The forward function receives the vision and language features, 
            and outputs the mask features with the spatial stride of 4x.

        Args:
            features (list[NestedTensor]): backbone features (vision), length is number of FPN layers
                tensors: [b*t, ci, hi, wi], mask: [b*t, hi, wi]
            text_features (NestedTensor): text features (language)
                tensors: [b, length, c], mask: [b, length]
            pos (list[Tensor]): position encoding of vision features, length is number of FPN layers
                tensors: [b*t, c, hi, wi]
            memory (list[Tensor]): features from encoder output. from 8x -> 32x
            NOTE: the layer orders of both features and pos are res2 -> res5

        Returns:
            mask_features (Tensor): [b*t, mask_dim, h, w], with the spatial stride of 4x.
        """
        y = self.forward_features(features, text_features, pos, memory, nf)
        return self.mask_features(y)


class VisionLanguageBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, sr_ratio=1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # for downsample
        self.sr_ratio = sr_ratio

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, t, h, w,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        b = tgt.size(1)
        # self attn
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.sr_ratio > 1: # downsample
            q = rearrange(q, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            k = rearrange(k, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            v = rearrange(tgt, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            # downsample
            new_h = int(h * 1./self.sr_ratio)
            new_w = int(w * 1./self.sr_ratio)
            size = (new_h, new_w)
            q = F.interpolate(q, size=size, mode='nearest')
            k = F.interpolate(k, size=size, mode='nearest')
            v = F.interpolate(v, size=size, mode='nearest')
            # shape for transformer
            q = rearrange(q, '(b t) c h w -> (t h w) b c', t=t)
            k = rearrange(k, '(b t) c h w -> (t h w) b c', t=t)
            v = rearrange(v, '(b t) c h w -> (t h w) b c', t=t)
            # downsample mask
            tgt_key_padding_mask = tgt_key_padding_mask.reshape(b*t, h, w)
            tgt_key_padding_mask = F.interpolate(tgt_key_padding_mask[None].float(), size=(new_h, new_w), mode='nearest').bool()[0] 
            tgt_key_padding_mask = tgt_key_padding_mask.reshape(b, t, new_h, new_w).flatten(1)
        else:
            v = tgt
        tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=tgt_key_padding_mask)[0] # [H*W, B*T, C]
        if self.sr_ratio > 1:
            tgt2 = rearrange(tgt2, '(t h w) b c -> (b t) c h w', t=t, h=new_h, w=new_w)
            size = (h, w)  # recover to origin size
            tgt2 = F.interpolate(tgt2, size=size, mode='bilinear', align_corners=False) # [B*T, C, H, W]
            tgt2 = rearrange(tgt2, '(b t) c h w -> (t h w) b c', t=t)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attn
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, t, h, w,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        b = tgt.size(1)
        # self attn
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        if self.sr_ratio > 1: # downsample
            q = rearrange(q, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            k = rearrange(k, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            v = rearrange(tgt, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            # downsample
            new_h = int(h * 1./self.sr_ratio)
            new_w = int(w * 1./self.sr_ratio)
            size = (new_h, new_w)
            q = F.interpolate(q, size=size, mode='nearest')
            k = F.interpolate(k, size=size, mode='nearest')
            v = F.interpolate(v, size=size, mode='nearest')
            # shape for transformer
            q = rearrange(q, '(b t) c h w -> (t h w) b c', t=t)
            k = rearrange(k, '(b t) c h w -> (t h w) b c', t=t)
            v = rearrange(v, '(b t) c h w -> (t h w) b c', t=t)
            # downsample mask
            tgt_key_padding_mask = tgt_key_padding_mask.reshape(b*t, h, w)
            tgt_key_padding_mask = F.interpolate(tgt_key_padding_mask[None].float(), size=(new_h, new_w), mode='nearest').bool()[0] 
            tgt_key_padding_mask = tgt_key_padding_mask.reshape(b, t, new_h, new_w).flatten(1)
        else:
            v = tgt2
        tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=tgt_key_padding_mask)[0] # [T*H*W, B, C]
        if self.sr_ratio > 1:
            tgt2 = rearrange(tgt2, '(t h w) b c -> (b t) c h w', t=t, h=new_h, w=new_w)
            size = (h, w)  # recover to origin size
            tgt2 = F.interpolate(tgt2, size=size, mode='bilinear', align_corners=False) # [B*T, C, H, W]
            tgt2 = rearrange(tgt2, '(b t) c h w -> (t h w) b c', t=t)
        tgt = tgt + self.dropout1(tgt2)

        # cross attn
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)

        # ffn
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, t, h, w,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, t, h, w,
                                    tgt_key_padding_mask, memory_key_padding_mask, 
                                    pos, query_pos)
        return self.forward_post(tgt, memory, t, h, w,
                                 tgt_key_padding_mask, memory_key_padding_mask, 
                                 pos, query_pos)


class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * tgt2
        return tgt


def dice_loss_actionvos(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_boxes: The number of boxes to normalize the loss.
    Returns:
        A scalar tensor representing the normalized DICE loss.
    """
    # Convert to FP32 for better numerical stability
    inputs = inputs.to(torch.float32)##[1, 129600]#[b,thw]
    targets = targets.to(torch.float32)########This does not need conversion
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)##can be removed###[1, 129600]#[b,thw]
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_boxes
    return loss


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()#[1, 25, 1]#[1, 1]#input and target shapes are consistent##if using weighted, there are###[1, 129600]#[b,thw]
    #print('w/o weight', inputs.shape)
    # print(f"input = {inputs}, targets = {targets}")
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes

def sigmoid_focal_loss_weighted(inputs, targets, num_boxes, weights, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()###[1, 129600]#input and target shapes are consistent
    # print(f"520 inputs.shape = {inputs.shape}, targets.shape = {targets.shape}")
    #print('w/ weight', weights.shape, torch.unique(weights))
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=weights)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes

def sigmoid_binary_ce_loss_weighted(inputs, targets, num_boxes, weights):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    # prob = inputs.sigmoid()
    #print('w/ weight', weights.shape, torch.unique(weights))
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=weights)
#     p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss# * ((1 - p_t) ** gamma)

#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def sce_loss(inputs, targets, num_boxes, weights=None, alpha: float = 0.1, beta: float = 1.0):
    """
    Symmetric Cross Entropy Loss (SL form), adapted for binary classification with weights.
    Args:
        inputs: Logits tensor of shape [b, thw].
        targets: Binary label tensor of shape [b, thw] (0 or 1).
        num_boxes: Normalization factor, typically number of positive samples.
        weights: Weight tensor of shape [b, thw], default None (no weighting).
        alpha: Weight for CE term, default 0.1.
        beta: Weight for RCE term, default 1.0.
        A: Constant for log(0) in RCE, default -6.0 (negative constant).
    Returns:
        Loss tensor (scalar).
    """
    assert inputs.shape == targets.shape, f"Inputs shape {inputs.shape} must match targets shape {targets.shape}"

    probs = inputs.sigmoid()  # [b, thw]

    # CE term: -q * log(p)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    if weights is not None:
        # print("weights for sce ce")
        ce = ce * weights  # element-wise weighting
    # else:
    #     print("weights not for sce ce")
    # RCE term: -p * log(q)
    # probs = torch.clamp(probs, min=1e-7, max=1.0)
    targets = torch.clamp(targets, min=1e-4, max=1.0)
    targets_neg = 1 - targets
    targets_neg = torch.clamp(targets_neg, min=1e-4, max=1.0)
    rce = -probs * torch.log(targets) - (1 - probs) * torch.log(targets_neg)  # [b, thw]
    if weights is not None:
        # print("weights for sce rce")
        rce = rce * weights  # element-wise weighting
    # else:
    #     print("weights not for sce rce")

    # SL loss
    sl = alpha * ce + beta * rce
    return sl.mean(1).sum() / num_boxes  # normalization

# class GeneralizedCrossEntropy(nn.Module):
#     def __init__(self, q: float = 0.7):
#         super().__init__()
#         self.q = q
#         self.epsilon = 1e-6
#         self.softmax = nn.Softmax(dim=1)
#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         p = self.softmax(input)
#         p = p[torch.arange(p.shape[0]), target]
#         p += self.epsilon
#         loss = (1 - p ** self.q) / self.q
#         return torch.mean(loss)

def gce_loss(inputs, targets, num_boxes, weights=None, q: float = 0.7):
    """
    Generalized Cross Entropy Loss, adapted for binary classification with weights.
    Args:
        inputs: Logits tensor of shape [b, thw].
        targets: Binary label tensor of shape [b, thw] (0 or 1).
        num_boxes: Normalization factor, typically number of positive samples.
        weights: Weight tensor of shape [b, thw], default None (no weighting).
        q: GCE hyperparameter in (0, 1], default 0.7.
    Returns:
        Loss tensor (scalar).
    """
    assert inputs.shape == targets.shape, f"Inputs shape {inputs.shape} must match targets shape {targets.shape}"
    # inputs = torch.clamp(inputs, min=-10.0, max=10.0)  # 限制 logits
    probs = inputs.sigmoid()  # [b, thw]
    p_y = probs * targets + (1 - probs) * (1 - targets)  # [b, thw], binary p_t
    epsilon = 1e-6
    p_y = p_y + epsilon
    # GCE loss
    loss = (1 - p_y.pow(q)) / q  # [b, thw]
    if weights is not None:
        # print("weights for gce")
        loss = loss * weights  # element-wise weighting
    # else:
    #     print("weights not for gce")

    return loss.mean(1).sum() / num_boxes  # normalization

def active_passive_loss(inputs, targets, num_boxes, weights=None, alpha=10.0, beta=1.0, 
                        active='gce', passive='mae', q: float = 0.7):
    """
    Active-Passive Loss adapted for binary mask prediction, inspired by ICML 2020 paper
    "Normalized Loss Functions for Deep Learning with Noisy Labels".
    
    Args:
        inputs: Logits tensor of shape [b, thw], predicted mask logits.
        targets: Binary label tensor of shape [b, thw] (0 or 1).
        num_boxes: Normalization factor, typically number of positive samples.
        weights: Weight tensor of shape [b, thw], default None (no weighting).
        alpha: Weight for active loss, default 10.0.
        beta: Weight for passive loss, default 1.0.
        active: Active loss type, options: 'ce', 'gce', 'ngce', default 'gce'.
        passive: Passive loss type, options: 'mae', 'mse', 'rce', default 'mae'.
        q: GCE/NGCE hyperparameter in (0, 1], default 0.7.
    Returns:
        Loss tensor (scalar), normalized by num_boxes.
    """
    assert inputs.shape == targets.shape, f"Inputs shape {inputs.shape} must match targets shape {targets.shape}"
    if weights is not None:
        assert weights.shape == inputs.shape, f"Weights shape {weights.shape} must match inputs shape {inputs.shape}"

    # Clamp logits range and compute probabilities
    inputs = torch.clamp(inputs, min=-10.0, max=10.0)  # Same as gce_loss
    probs = inputs.sigmoid()  # [b, thw]
    targets = targets.float()  # Ensure targets is float type

    # Active loss
    if active == 'ce':
        # Binary cross-entropy loss
        active_loss = - (targets * torch.log(probs + 1e-7) + (1 - targets) * torch.log(1 - probs + 1e-7))
    elif active == 'gce':
        # GCE loss, refer to gce_loss
        p_y = probs * targets + (1 - probs) * (1 - targets)  # [b, thw], binary p_t
        active_loss = (1 - p_y.pow(q)) / q  # [b, thw]
    elif active == 'ngce':
        # NGCE adapted for binary case: normalized GCE
        p_y = probs * targets + (1 - probs) * (1 - targets)  # [b, thw]
        gce_loss = (1 - p_y.pow(q)) / q  # [b, thw]
        norm_factor = (2 - torch.pow(probs, q) - torch.pow(1 - probs, q)) / q  # Normalization factor for binary case
        active_loss = gce_loss / norm_factor.clamp(min=1e-7)  # Avoid division by zero
    else:
        raise AssertionError(f'Active loss: {active} is not supported yet')

    # Passive loss
    if passive == 'mae':
        passive_loss = torch.abs(probs - targets)  # [b, thw]
    elif passive == 'mse':
        passive_loss = (probs - targets) ** 2  # [b, thw]
    elif passive == 'rce':
        # RCE adapted for binary case: reverse cross-entropy
        targets_clamped = torch.clamp(targets, min=1e-4, max=1.0)  # Avoid log(0)
        passive_loss = - (probs * torch.log(targets_clamped + 1e-7) + (1 - probs) * torch.log(1 - targets_clamped + 1e-7))
    else:
        raise AssertionError(f'Passive loss: {passive} is not supported yet')

    # Apply weights
    if weights is not None:
        active_loss = active_loss * weights
        passive_loss = passive_loss * weights

    # Combine losses and normalize
    loss = alpha * active_loss + beta * passive_loss  # [b, thw]
    return loss.mean(1).sum() / num_boxes  # Scalar, normalized

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


