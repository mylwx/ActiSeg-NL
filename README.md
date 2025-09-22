
# ActiSeg-NL  🏷️🎲🤖

This repository is the repository of ActiSeg-NL.  🔬

https://github.com/user-attachments/assets/d5b8d6f0-9c5a-4436-83ed-36293db144b8



### Update  🗓️

* 2025.09.21 Init repository.  ✨



### TODO List  ✅

* [ ] 📄 Release the preprint on arXiv.
* [ ] 🧰 Release data preparation scripts.
* [ ] 🏗️ Release training and evaluation code.
* [ ] 💾 Checkpoints will be released.

### Abstract  📚

Embodied intelligence relies on accurately segmenting objects actively involved in interactions. Action-based video object segmentation addresses this by linking segmentation with action semantics, but it depends on large-scale annotations and prompts that are costly, inconsistent, and prone to multimodal noise such as imprecise masks and referential ambiguity. To date, this challenge remains unexplored. In this work, we take the first step by studying action-based video object segmentation under label noise, focusing on two sources: textual prompt noise (category flips and within-category noun substitutions) and mask annotation noise (perturbed object boundaries to mimic imprecise supervision). Our contributions are threefold. First, we introduce two types of label noises for the action-based video object segmentation task. Second, we build up the first action-based video object segmentation under a label noise benchmark ActiSeg-NL and adapt six label-noise learning strategies to this setting, and establish protocols for evaluating them under textual, boundary, and mixed noise. Third, we provide a comprehensive analysis linking noise types to failure modes and robustness gains, and we introduce a Parallel Mask Head Mechanism (PMHM) to address mask annotation noise. Qualitative evaluations further reveal characteristic failure modes, including boundary leakage and mislocalization under boundary perturbations, as well as occasional identity substitutions under textual flips. Our comparative analysis reveals that different learning strategies exhibit distinct robustness profiles, governed by a foreground-background trade-off where some achieve balanced performance while others prioritize foreground accuracy at the cost of background precision. These results establish a clear sensitivity profile of action-based video object segmentation to imperfect annotations and set a benchmark for studying noise-robust learning in embodied perception.

### Benchmark  📊

Todo

### 🤝 Publication  🧾

If you find ActiSeg-NL useful for your research, please cite it.

A BibTeX entry will be provided after the preprint is available.

Todo
