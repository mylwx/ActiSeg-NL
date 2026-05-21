# ActiSeg-NL-Benchmark

## Overview

A benchmark of various noisy label robust learning methods on Referring Video Object Segmentation tasks, including:

- **Co-teaching** - Co-teaching noisy label learning method
- **SCE (Symmetric Cross Entropy)** - Symmetric cross entropy loss function
- **GCE (Generalized Cross Entropy)** - Generalized cross entropy loss function
- **APL (Active Passive Loss)** - Active passive loss function
- **ELR** ([Early-learning regularization prevents memorization of noisy labels](https://arxiv.org/pdf/2007.00151))
- **NPN** ([Adaptive integration of partial label learning and negative learning for enhanced noisy label learning](https://arxiv.org/pdf/2312.09505))

## Training Methods

### 1. Co-teaching Training

```bash
bash scripts/train_actionvos_coteaching.sh \
  actionvos_dirs/r101/coteaching/ \
  pretrained_weights/r101_refytvos_joint.pth \
  4 0,1,2,3 29500 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_positive_cls \
  --actionvos_path ../dataset_visor \
  --imagesets_path ../dataset_visor/ImageSets/ \
  --epochs 6 \
  --lr_drop 3 5 \
  --save_interval 3 \
  --num_workers 2
```

### 2. SCE Loss Function Training

#### SCE Method - mask9 mask noise training

```bash
bash scripts/train_actionvos.sh \
  actionvos_dirs/r101/sce_mask9_noamp/ \
  pretrained_weights/r101_refytvos_joint.pth \
  4 0,1,2,3 29000 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_weights --use_positive_cls \
  --actionvos_path ../dataset_visor_1122_latest_0119 \
  --imagesets_path ../dataset_visor/ImageSets/ \
  --annotations_path ../noise_data/mask_9_15_21/Annotations_Sparse_dilation_1227_rate100_9by9/ \
  --epochs 6 \
  --lr_drop 3 5 \
  --save_interval 3 \
  --num_workers 2 \
  --use_sce
```

#### SCE Method - noise40 expression noise training

```bash
bash scripts/train_actionvos.sh \
  actionvos_dirs/r101/sce_noise40_noamp/ \
  pretrained_weights/r101_refytvos_joint.pth \
  1 0 29000 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_weights --use_positive_cls \
  --actionvos_path ../dataset_visor \
  --imagesets_path ../noise_data_enable/prompt_name_20_40_60/ImageSets_noise40_classid_prompt/ \
  --epochs 6 \
  --lr_drop 3 5 \
  --save_interval 3 \
  --num_workers 2 \
  --use_sce
```

### 3. GCE Loss Function Training

#### GCE Method - mask9 mask noise training

```bash
bash scripts/train_actionvos.sh \
  actionvos_dirs/r101/gce_mask9_noamp/ \
  pretrained_weights/r101_refytvos_joint.pth \
  4 0,1,2,3 29000 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_weights --use_positive_cls \
  --actionvos_path ../dataset_visor_1122_latest_0119 \
  --imagesets_path ../dataset_visor/ImageSets/ \
  --annotations_path ../noise_data/mask_9_15_21/Annotations_Sparse_dilation_1227_rate100_9by9/ \
  --epochs 6 \
  --lr_drop 3 5 \
  --save_interval 3 \
  --num_workers 2 \
  --use_gce \
  --gce_q 0.7
```

#### GCE Method - noise40 expression noise training

```bash
bash scripts/train_actionvos.sh \
  actionvos_dirs/r101/gce_noise40_noamp/ \
  pretrained_weights/r101_refytvos_joint.pth \
  1 0 29000 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_weights --use_positive_cls \
  --actionvos_path ../dataset_visor \
  --imagesets_path ../noise_data_enable/prompt_name_20_40_60/ImageSets_noise40_classid_prompt/ \
  --epochs 6 \
  --lr_drop 3 5 \
  --save_interval 3 \
  --num_workers 2 \
  --use_gce \
  --gce_q 0.7
```

### 4. APL Loss Function Training

#### APL Method - mask9 mask noise training

```bash
bash scripts/train_actionvos.sh \
  actionvos_dirs/r101/apl_mask9_noamp/ \
  pretrained_weights/r101_refytvos_joint.pth \
  4 0,1,2,3 29000 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_weights --use_positive_cls \
  --actionvos_path ../dataset_visor_1122_latest_0119 \
  --imagesets_path ../dataset_visor/ImageSets/ \
  --annotations_path ../noise_data/mask_9_15_21/Annotations_Sparse_dilation_1227_rate100_9by9/ \
  --epochs 6 \
  --lr_drop 3 5 \
  --save_interval 3 \
  --num_workers 2 \
  --use_active_passive
```

#### APL Method - noise40 expression noise training

```bash
bash scripts/train_actionvos.sh \
  actionvos_dirs/r101/apl_noise40_noamp/ \
  pretrained_weights/r101_refytvos_joint.pth \
  1 0 29000 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_weights --use_positive_cls \
  --actionvos_path ../dataset_visor \
  --imagesets_path ../noise_data_enable/prompt_name_20_40_60/ImageSets_noise40_classid_prompt/ \
  --epochs 6 \
  --lr_drop 3 5 \
  --save_interval 3 \
  --num_workers 2 \
  --use_active_passive
```

### 5. ELR Training

#### ELR Method - mask9 mask noise training

```bash
bash scripts/train_actionvos_elr_orgsize_mask.sh \
  actionvos_dirs/r101/0805_elr_mask9_noamp/ \
  pretrained_weights/r101_refytvos_joint.pth \
  4 4,5,6,7 25900 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_positive_cls \
  --actionvos_path ../dataset_visor_1122_latest_0119 \
  --imagesets_path ../dataset_visor/ImageSets/ \
  --annotations_path ../noise_data/mask_9_15_21/Annotations_Sparse_dilation_1227_rate100_9by9/ \
  --epochs 6 \
  --lr_drop 3 5 \
  --save_interval 3 \
  --num_workers 2 \
  --use_elr_loss_mask \
  --index_map_path actionvos_dirs/r101/0805_elr_mask9_noamp/video_index_map.json \
  --sample_map_path actionvos_dirs/r101/0805_elr_mask9_noamp/clip_sample_map.json
```

#### ELR Method - noise20 expression noise training

```bash
bash scripts/train_actionvos_elr_orgsize_mask.sh \
  actionvos_dirs/r101/0805_elr_noise20_noamp/ \
  pretrained_weights/r101_refytvos_joint.pth \
  4 0,1,2,3 39000 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_positive_cls \
  --actionvos_path ../dataset_visor \
  --imagesets_path ../noise_data_enable/prompt_name_20_40_60/ImageSets_noise20_classid_prompt/ \
  --epochs 6 \
  --lr_drop 3 5 \
  --save_interval 3 \
  --num_workers 2 \
  --use_elr_loss_mask \
  --index_map_path actionvos_dirs/r101/0805_elr_noise20_noamp/video_index_map.json \
  --sample_map_path actionvos_dirs/r101/0805_elr_noise20_noamp/clip_sample_map.json
```

### 6. NPN Training

#### NPN Method - mask9 mask noise training

```bash
bash scripts/train_actionvos_npn_orgsize_mask.sh \
  actionvos_dirs/r101/0805_npn_mask9_noamp/ \
  pretrained_weights/r101_refytvos_joint.pth \
  1 0 29000 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_positive_cls \
  --actionvos_path ../dataset_visor_1122_latest_0119 \
  --imagesets_path ../dataset_visor/ImageSets/ \
  --annotations_path ../noise_data/mask_9_15_21/Annotations_Sparse_dilation_1227_rate100_9by9/ \
  --epochs 6 \
  --lr_drop 3 5 \
  --save_interval 3 \
  --num_workers 2 \
  --use_npn_loss_mask \
  --dataset_file actionvos_ws \
  --warmup_epochs_npn 2 \
  --index_map_path actionvos_dirs/r101/0805_npn_mask9_noamp/video_index_map.json \
  --sample_map_path actionvos_dirs/r101/0805_npn_mask9_noamp/clip_sample_map.json
```

#### NPN Method - noise20 expression noise training

```bash
bash scripts/train_actionvos_npn_orgsize_mask.sh \
  actionvos_dirs/r101/0805_npn_noise20_noamp/ \
  pretrained_weights/r101_refytvos_joint.pth \
  4 0,1,2,3 29000 \
  --backbone resnet101 \
  --expression_file train_meta_expressions_promptaction.json \
  --use_positive_cls \
  --actionvos_path ../dataset_visor \
  --imagesets_path ../noise_data_enable/prompt_name_20_40_60/ImageSets_noise20_classid_prompt/ \
  --epochs 6 \
  --lr_drop 3 5 \
  --save_interval 3 \
  --num_workers 2 \
  --use_npn_loss_mask \
  --dataset_file actionvos_ws \
  --warmup_epochs_npn 2 \
  --index_map_path actionvos_dirs/r101/0805_npn_noise20_noamp/video_index_map.json \
  --sample_map_path actionvos_dirs/r101/0805_npn_noise20_noamp/clip_sample_map.json
```

## Inference and Evaluation

### Inference

```bash
bash scripts/test_actionvos.sh \
  actionvos_dirs/r101_refytvos_joint/model_name/ \
  path/to/checkpoint.pth \
  0 29500 \
  --backbone resnet101 \
  --expression_file val_human_meta_expressions_promptaction.json \
  --use_positive_cls \
  --pos_cls_thres 0.75 \
  --actionvos_path ../dataset_visor
```

### Evaluation Metrics Calculation

```bash
# Calculate evaluation metrics
cd ..
python actionvos_metrics.py \
  --gt_path ./dataset_visor/Annotations_Sparse/val \
  --split_json ./dataset_visor/ImageSets/val_human.json \
  --pred_path ./ReferFormer/actionvos_dirs/r101_refytvos_joint/model_name/val
cd ReferFormer
```
