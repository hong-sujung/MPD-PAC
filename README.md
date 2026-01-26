# Mitigating Mask Prior Drift and Positional Attention Collapse in Large Diffusion Vision-Language Models 

## Abstract
Large diffusion vision–language models (LDVLMs) enable parallel decoding and
bidirectional attention, but their behavior under long-form generation remains
underexplored.
We show that existing LDVLMs suffer from repetitive generation and degraded visual
grounding, caused by a shared mask token prior and misaligned positional attention
during iterative unmasking.
To address these issues, we propose a training-free, inference-time approach
consisting of **Mask Prior Suppression** and **Monotonic RoPE Scaling**, which jointly
mitigate mask prior bias and positional attention collapse.
Experiments on multimodal benchmarks demonstrate consistent improvements in visual
grounding and long-form description quality, without requiring any additional
training.

## Method Overview

Our approach consists of two inference-time techniques:

- **Mask Prior Suppression**: suppresses the shared mask token prior in the final hidden states.
- **Monotonic RoPE Scaling**: stabilizes long-range attention by emphasizing low-frequency RoPE components.

<p align="center">
  <img src="assets/overview.png" width="80%">
</p>

## Installation

### LLaDA-V
Please follow the environment setup instructions provided in the official
[LLaDA-V repository](https://github.com/ML-GSAI/LLaDA-V).

Specifically, clone the repository and initialize the environment as follows:

```bash
git clone https://github.com/ML-GSAI/LLaDA-V.git
cd LLaDA-V/train
bash init_env.sh
```
### LaViDa
Please follow the environment setup instructions provided in the official
[LaViDa repository](https://github.com/jacklishufan/LaViDa.git).

The environment can be set up using the following steps.

First, create and activate a conda environment:
```bash
conda create --name lavida python=3.13
conda activate lavida
```
Then, clone the repository and install the required dependencies:
```bash
git clone https://github.com/jacklishufan/LaViDa.git
pip install -e .[train]
cd eval
pip install -e .
cd ../
pip install trl==0.17.0 
```

## Inference
Below we provide example commands for running inference with our method.

### LLaDA-V + Ours
```bash
cd path/to/icml2026_code/LLaDA-V
python generate_demo_ours.py
```

### LaViDa + Ours
```bash
cd path/to/icml2026_code/LaViDa
python predict_ours.py
```