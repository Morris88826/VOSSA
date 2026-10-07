# VOSSA: Voiceprint Optimization for Streaming Speech Architectures (Interspeech 2026)

<a href="https://arxiv.org/abs/2609.38887">
  <img src="https://img.shields.io/badge/arXiv-2609.38887-%23B31B1B">
</a>
<a href="https://morris88826.github.io/VOSSA/"><img src="https://img.shields.io/badge/Demo%20Page-online-brightgreen"></a>
<br>

This is the official repository for the paper

"VOSSA: Voiceprint Optimization for Streaming Speech Architectures"

by [Mu-Ruei Tseng](https://github.com/Morris88826), [Waris Quamer](https://github.com/warisqr007), [Ghady Nasrallah](https://github.com/Ghadynasrallah), [Ricardo Gutierrez-Osuna](https://scholar.google.com/citations?user=UnuQfEwAAAAJ&hl=en)

Department of Computer Science & Engineering, Texas A&M University

## News
**Sep. 2026:** VOSSA received the Best Student Paper Award at Interspeech 2026.

**Jun. 2026:** VOSSA accepted at Interspeech 2026.

## Introduction

![Architecture](demo/public/figures/architecture.png)

Real-time voice conversion (VC) systems commonly rely on pretrained speaker embeddings from automatic speaker verification (ASV) models. While effective for speaker discrimination, these embeddings are trained to remain stable across phonetic and prosodic variations within-speaker, which may conflict with frame-level acoustic generation in streaming constraints. To address this issue, we propose VOSSA, a speaker representation framework that extracts speaker information from intermediate content encoder layers and aggregates using attentive statistics pooling. The embedding is trained jointly with VC objectives, removing the need for a separate speaker encoder.

For more information, please check out our [Demo Page](https://morris88826.github.io/VOSSA/).

## Highlights

- **19% fewer parameters** than TVTSyn (132.4M vs. 162.8M) by eliminating the external speaker encoder
- **Real-time streaming** with RTF ≈ 0.25 and end-to-end latency ≈ 73 ms
- **Best normalized target-speaker similarity** across six evaluation datasets
- **Improved pitch accuracy** — lowest pitch MAE and highest Pearson CC among all baselines
- **Better vowel preservation** — lowest Wasserstein distance to ground-truth F1 distributions for high, mid, and low vowels

## Code

Source code coming soon.

## Citation
```
@inproceedings{tseng26c_interspeech,
  title     = {{VOSSA: Voiceprint Optimization for Streaming Speech Architectures}},
  author    = {Mu-Ruei Tseng and Waris Quamer and Ghady Nasrallah and Ricardo Gutierrez-Osuna},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {4716--4720},
  doi       = {10.21437/Interspeech.2026-2763},
  issn      = {2958-1796},
}
```
