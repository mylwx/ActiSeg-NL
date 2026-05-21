# Dataset 噪声生成说明

本目录包含用于生成训练数据噪声的脚本和预生成的噪声数据集。

## 噪声生成脚本

### 1. 提示词名称噪声生成脚本

**文件**: `generate_prompt_name_noise.py`

**功能**: 该脚本用于生成提示词级别的噪声，通过随机替换目标对象的类别名称和描述文本来创建噪声训练数据。具体功能包括：

- 从所有类别中随机选择不同于原始类别的类别作为噪声标签
- 根据噪声类别生成对应的噪声提示词（格式：`{category} used in the action of {narration}`）
- 支持自定义噪声率（rate_noise参数），控制生成噪声数据的比例
- 保留原始标签信息（class_id_org、category_org、exp_org）用于对比分析

**主要参数**:

- `--actionvos_path_input`: 输入数据路径
- `--actionvos_output_path_noise`: 噪声数据输出路径
- `--rate_noise`: 噪声率（默认0.2，即20%的数据添加噪声）

### 2. 标注掩码噪声生成脚本

**文件**: `generate_annotations_mask_noise.py`

**功能**: 该脚本用于生成标注掩码级别的噪声，通过对目标对象的分割掩码进行形态学膨胀操作来创建噪声标注。具体功能包括：

- 对正样本对象的掩码进行膨胀操作，扩大标注区域
- 支持自定义膨胀核大小（ksize参数）和噪声率（rate_noise参数）
- 保持负样本掩码不变，仅处理正样本
- 保存原始掩码和噪声掩码对比文件

**主要参数**:

- `--root_annotations_save`: 噪声标注保存路径
- `--ksize`: 膨胀核大小（默认9，可选9/15/21，对应mask9/15/21）
- `--split`: 数据集划分（train/val）
- `--rate_noise`: 噪声率（默认1.0，即全部添加噪声）

## 提示词指代噪声数据集

**目录**: `prompt_name_noise/`

本目录提供了预生成的提示词噪声数据集，包含不同噪声率的训练数据，可直接用于模型训练和评估。

### 数据集结构

```
prompt_name_noise/
└── prompt_name_20_40_60/
    ├── ImageSets_noise20_classid_prompt/    # 20%噪声率数据集
    │   ├── train_meta_expressions_promptaction.json
    │   └── train_objects_category.json
    ├── ImageSets_noise40_classid_prompt/    # 40%噪声率数据集
    │   ├── train_meta_expressions_promptaction.json
    │   └── train_objects_category.json
    └── ImageSets_noise60_classid_prompt/    # 60%噪声率数据集
        ├── train_meta_expressions_promptaction.json
        └── train_objects_category.json
```

### 文件说明

- `train_meta_expressions_promptaction.json`: 包含噪声提示词的表达式标注文件，每个表达式的exp字段已被替换为噪声类别名称
- `train_objects_category.json`: 包含噪声类别信息的对象类别文件，class_id和category字段已被替换为噪声类别

### 噪声率说明

- **noise20**: 20%的训练数据添加了提示词噪声
- **noise40**: 40%的训练数据添加了提示词噪声
- **noise60**: 60%的训练数据添加了提示词噪声
