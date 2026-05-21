# ActiSeg-NL-Benchmark 噪声标签鲁棒视频物体分割基准测试

## 概述

多种噪声标签鲁棒学习方法在动作指代视频物体分割任务上的基准测试，包括：

- **Co-teaching** - 协同教学噪声标签学习方法
- **SCE (Symmetric Cross Entropy)** - 对称交叉熵损失函数
- **GCE (Generalized Cross Entropy)** - 广义交叉熵损失函数
- **APL (Active Passive Loss)** - 主动被动损失函数
- **ELR ([Early-learning regularization prevents memorization of noisy labels](https://arxiv.org/pdf/2007.00151))**
- **NPN ([Adaptive integration of partial label learning and negative learning for enhanced noisy label learning](https://arxiv.org/pdf/2312.09505))**

## 训练方法

### 1. Co-teaching训练

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

### 2. SCE损失函数训练

#### SCE方法 - mask9掩码噪声训练

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

#### SCE方法 - noise40表达式噪声训练

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

### 3. GCE损失函数训练

#### GCE方法 - mask9掩码噪声训练

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

#### GCE方法 - noise40表达式噪声训练

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

### 4. APL损失函数训练

#### APL方法 - mask9掩码噪声训练

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

#### APL方法 - noise40表达式噪声训练

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

### 5. ELR训练

#### ELR方法 - mask9掩码噪声训练

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

#### ELR方法 - noise20表达式噪声训练

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

### 6. NPN训练

#### NPN方法 - mask9掩码噪声训练

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

#### NPN方法 - noise20表达式噪声训练

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

## 推理评估

### 推理

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

### 评估指标计算

```bash
# 计算评估指标
cd ..
python actionvos_metrics.py \
  --gt_path ./dataset_visor/Annotations_Sparse/val \
  --split_json ./dataset_visor/ImageSets/val_human.json \
  --pred_path ./ReferFormer/actionvos_dirs/r101_refytvos_joint/model_name/val
cd ReferFormer
```
