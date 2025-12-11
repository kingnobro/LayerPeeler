# LayerPeeler: Autoregressive Peeling for Layer-wise Image Vectorization

[![arXiv](https://img.shields.io/badge/arXiv-2505.23740-b31b1b.svg)](https://arxiv.org/abs/2505.23740)
[![website](https://img.shields.io/badge/Website-Gitpage-4CCD99)](https://layerpeeler.github.io/)

![title](./assets/teaser.png)

## Overview

LayerPeeler is a framework for layer-wise image vectorization that decomposes images into structured vector representations. The system uses a vision-language model to analyze the image and construct a hierarchical layer graph, identifying the topmost visual elements. These detected elements are then processed by a fine-tuned image diffusion model to generate clean background images with the specified elements removed, enabling precise layer-by-layer vectorization.

## Updates
- [x] **[2025.12.11]**: Dataset Released
- [x] **[2025.12.11]**: Training Code Released
- [ ] Inference Code Released
- [ ] Vectorization Released

## Setup

### 1. Environment Setup
```shell
conda create -n layerpeeler python=3.10
conda activate layerpeeler

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Merge the Pretrained Model
As described in the paper, we use a pretrained LoRA model trained on the `SeedEdit` dataset. You must merge this pretrained LoRA with the base model before training. 

```bash
python merge_pretrain.py
```
> [!TIP]  
> You can update the path to the pretrained model inside `merge_pretrain.py` if necessary.

## Inference
TBD

## Training
You can find the training code and dataset in the [`train`](./train) directory, and detailed training instructions in the corresponding [`README`](./train/README.md) file.