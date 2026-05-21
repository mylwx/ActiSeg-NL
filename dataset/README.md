# Dataset Noise Generation Instructions

This directory contains scripts for generating training data noise and pre-generated noisy datasets.

## Noise Generation Scripts

### 1. Prompt Name Noise Generation Script

**File**: `generate_prompt_name_noise.py`

**Function**: This script generates prompt-level noise by randomly replacing the target object's category name and description text to create noisy training data. Specific features include:

- Randomly selects categories different from the original category as noisy labels
- Generates corresponding noise prompts based on noise categories (format: `{category} used in the action of {narration}`)
- Supports custom noise rate (rate_noise parameter) to control the proportion of noisy data generated
- Preserves original label information (class_id_org, category_org, exp_org) for comparative analysis

**Main Parameters**:

- `--actionvos_path_input`: Input data path
- `--actionvos_output_path_noise`: Noise data output path
- `--rate_noise`: Noise rate (default 0.2, i.e., 20% of data with noise added)

### 2. Annotation Mask Noise Generation Script

**File**: `generate_annotations_mask_noise.py`

**Function**: This script generates annotation mask-level noise by performing morphological dilation operations on target object segmentation masks to create noisy annotations. Specific features include:

- Performs dilation operations on positive sample object masks to expand annotation regions
- Supports custom dilation kernel size (ksize parameter) and noise rate (rate_noise parameter)
- Keeps negative sample masks unchanged, only processes positive samples
- Saves original mask and noisy mask comparison files

**Main Parameters**:

- `--root_annotations_save`: Noise annotation save path
- `--ksize`: Dilation kernel size (default 9, optional 9/15/21, corresponding to mask9/15/21)
- `--split`: Dataset split (train/val)
- `--rate_noise`: Noise rate (default 1.0, i.e., add noise to all data)

## Prompt Referencing Noise Dataset

**Directory**: `prompt_name_noise/`

This directory provides pre-generated prompt noise datasets with different noise rates for training and evaluation models.

### Dataset Structure

```
prompt_name_noise/
└── prompt_name_20_40_60/
    ├── ImageSets_noise20_classid_prompt/    # 20% noise rate dataset
    │   ├── train_meta_expressions_promptaction.json
    │   └── train_objects_category.json
    ├── ImageSets_noise40_classid_prompt/    # 40% noise rate dataset
    │   ├── train_meta_expressions_promptaction.json
    │   └── train_objects_category.json
    └── ImageSets_noise60_classid_prompt/    # 60% noise rate dataset
        ├── train_meta_expressions_promptaction.json
        └── train_objects_category.json
```

### File Description

- `train_meta_expressions_promptaction.json`: Expression annotation file containing noisy prompts, where the exp field of each expression has been replaced with noisy category names
- `train_objects_category.json`: Object category file containing noisy category information, where class_id and category fields have been replaced with noisy categories

### Noise Rate Description

- **noise20**: 20% of training data with prompt noise added
- **noise40**: 40% of training data with prompt noise added
- **noise60**: 60% of training data with prompt noise added
