# ActiSeg-NL-PMHM

## 概述

**PMHM** 方法，一种多副头协同挖掘的噪声标签鲁棒学习方法，专门针对动作指代视频物体分割任务中的掩码级标签噪声问题。

## PMHM方法训练

### 掩码噪声训练 (mask9)

```bash
bash scripts/train_actionvos.sh \
  actionvos_dirs/r101/pmhm_step_twoheads_mask9/ \
  pretrained_weights/r101_refytvos_joint.pth \
  4 0,1,2,3 31900 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_weights --use_positive_cls \
  --actionvos_path ../dataset_visor \
  --imagesets_path ../dataset_visor/ImageSets/ \
  --annotations_path ../noise_data/mask_9_15_21/Annotations_Sparse_dilation_1227_rate100_9by9/ \
  --epochs 6 --lr_drop 3 5 --save_interval 3 --num_workers 2 \
  --use_pmhm 1 --n_aux_heads 1 --init_mode_single copy_perturb \
  --copy_sigma 1e-3 --aux_p_drop 0.2 --aux_gamma_noise 0.2 \
  --aux_freeze_mode step --aux_freeze_p0 0.3 --aux_freeze_p1 0.1 \
  --k_agree 2 --tau_m_p 0.20 --tau_e_p 0.85 --tau_h_p 0.60 \
  --tv_loss_coef 0.3 --head_loss_coef 0.1 --layer_loss_coef 0.1 \
  --tv_alpha_start 0.7 --tv_alpha_end 0.5 --tv_beta 0.5
```

## 推理评估

### PMHM模型推理

```bash
bash scripts/test_actionvos.sh \
  actionvos_dirs/r101_refytvos_joint/pmhm_model/ \
  path/to/checkpoint.pth \
  0 29500 \
  --backbone resnet101 \
  --expression_file val_human_meta_expressions_promptaction.json \
  --use_positive_cls \
  --pos_cls_thres 0.75 \
  --actionvos_path ../dataset_visor
```

### 评估指标计算

```bash
# 计算评估指标
cd ..
python actionvos_metrics.py \
  --gt_path ./dataset_visor/Annotations_Sparse/val \
  --split_json ./dataset_visor/ImageSets/val_human.json \
  --pred_path ./ReferFormer/actionvos_dirs/r101_refytvos_joint/pmhm_model/val
cd ReferFormer
```
