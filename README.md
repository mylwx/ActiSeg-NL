# ActiSeg-NL  🏷️🎲🤖

**Segment-to-Act: Label-Noise-Robust Action-Prompted Video Segmentation Towards Embodied Intelligence**

[![arXiv](https://img.shields.io/badge/arXiv-2509.16677-b31b1b.svg)](https://arxiv.org/abs/2509.16677)

https://github.com/user-attachments/assets/9b30862f-e6ab-44a6-8cbc-b55891a3da90

### TODO List  ✅

* [X] 📄 Release the preprint on arXiv.
* [X] 🧰 Release data preparation scripts.
* [X] 🏗️ Release training and evaluation code.

### Abstract  📚

Embodied intelligence relies on accurately segmenting objects actively involved in interactions. Action-based video object segmentation addresses this by linking segmentation with action semantics, but it depends on large-scale annotations and prompts that are costly, inconsistent, and prone to multimodal noise such as imprecise masks and referential ambiguity. To date, this challenge remains unexplored. In this work, we take the first step by studying action-based video object segmentation under label noise, focusing on two sources: textual prompt noise (category flips and within-category noun substitutions) and mask annotation noise (perturbed object boundaries to mimic imprecise supervision). Our contributions are threefold. First, we introduce two types of label noises for the action-based video object segmentation task. Second, we build up the first action-based video object segmentation under a label noise benchmark ActiSeg-NL and adapt six label-noise learning strategies to this setting, and establish protocols for evaluating them under textual, boundary, and mixed noise. Third, we provide a comprehensive analysis linking noise types to failure modes and robustness gains, and we introduce a Parallel Mask Head Mechanism (PMHM) to address mask annotation noise. Qualitative evaluations further reveal characteristic failure modes, including boundary leakage and mislocalization under boundary perturbations, as well as occasional identity substitutions under textual flips. Our comparative analysis reveals that different learning strategies exhibit distinct robustness profiles, governed by a foreground-background trade-off where some achieve balanced performance while others prioritize foreground accuracy at the cost of background precision. These results establish a clear sensitivity profile of action-based video object segmentation to imperfect annotations and set a benchmark for studying noise-robust learning in embodied perception.

## Resources  🔗

Material related to our project is available via the following links:

- 📄 [**Paper**](https://arxiv.org/pdf/2509.16677)
- 🛠️ [**ActionVOS**](https://github.com/ut-vision/ActionVOS)
- 📊 [**VISOR Dataset**](https://epic-kitchens.github.io/VISOR/)

## Requirements

* Our experiment environment follows [ActionVOS](https://github.com/ut-vision/ActionVOS) requirements.
* Tested with Python 3.8, PyTorch 1.11.0
* Please refer to [ActionVOS](https://github.com/ut-vision/ActionVOS) for detailed environment setup and installation instructions.

## Dataset Preparation

### **Base Dataset Download**

Please download the VISOR dataset from the following link:

- [**VISOR-VOS (28.4GB)**](https://data.bris.ac.uk/data/dataset/2v6cgv1x04ol22qp9rm9x2j6a7)

For detailed data preparation instructions, please follow the [ActionVOS](https://github.com/ut-vision/ActionVOS) data preparation guide.

### **Noise Label Generation**

Our benchmark provides comprehensive noise generation scripts and pre-generated noise datasets in the `dataset/` directory.

For detailed documentation on noise generation methods and pre-generated datasets, please refer to [dataset/README.md](dataset/README.md).

## Training & Evaluation

For detailed training and evaluation instructions, please refer to:

- **[ActiSeg_NL_benchmark/](ActiSeg_NL_benchmark/)** - Benchmark implementation with training and evaluation scripts
- **[ActiSeg_NL_pmhm/](ActiSeg_NL_pmhm/)** - PMHM method implementation with experimental setup

Both directories contain complete instructions for:

- Training with noise-contaminated datasets
- Evaluation using standard ActionVOS metrics (p-mIoU, n-mIoU, p-cIoU, n-cIoU, gIoU, Accuracy)
- Reproducing experimental results from our paper

## Citation

If this benchmark or code is helpful in your research, please cite our paper:

```bibtex
@article{li2025segment,
  title={Segment-to-Act: Label-Noise-Robust Action-Prompted Video Segmentation Towards Embodied Intelligence},
  author={Li, Wenxin and Peng, Kunyu and Wen, Di and Liu, Ruiping and Duan, Mengfei and Luo, Kai and Yang, Kailun},
  journal={arXiv preprint arXiv:2509.16677},
  year={2025}
}
```

If you are using the base ActionVOS framework, datasets, or evaluation metrics, please cite the [ActionVOS paper](https://arxiv.org/abs/2407.07402) and related works:

```bibtex
@inproceedings{ouyang2024actionvos,
  title={ActionVOS: Actions as Prompts for Video Object Segmentation},
  author={Ouyang, Liangyang and Liu, Ruicong and Huang, Yifei and Furuta, Ryosuke and Sato, Yoichi},
  booktitle={European Conference on Computer Vision},
  pages={216--235},
  year={2024}
}
```

## Acknowledgments

This benchmark is built upon the excellent [ActionVOS](https://github.com/ut-vision/ActionVOS). We thank the original authors for their open-source contributions.
