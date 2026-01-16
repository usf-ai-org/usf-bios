# USF BIOS - AI Training & Fine-tuning Platform

<p align="center">
  <br>
  <img src="asset/banner.svg" alt="USF BIOS Banner" width="700"/>
  <br>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-5be.svg">
  <img src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange.svg">
  <img src="https://img.shields.io/badge/license-Proprietary-red.svg">
  <img src="https://img.shields.io/badge/owner-UltraSafe%20AI-blue.svg">
</p>

---

> **PROPRIETARY SOFTWARE**
> 
> This software is the exclusive property of **US Inc**. Unauthorized use, reproduction, modification, or distribution is strictly prohibited without explicit written permission from US Inc. All rights reserved.

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Supported Models](#-supported-models)
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-start)
- [Fine-tuning Guide](#-fine-tuning-guide)
  - [SFT (Supervised Fine-Tuning)](#sft-supervised-fine-tuning)
  - [LoRA Fine-tuning](#lora-fine-tuning)
  - [QLoRA Fine-tuning](#qlora-fine-tuning)
  - [RLHF (Reinforcement Learning from Human Feedback)](#rlhf-reinforcement-learning-from-human-feedback)
  - [GRPO (Group Relative Policy Optimization)](#grpo-group-relative-policy-optimization)
  - [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
- [Multimodal Training](#-multimodal-training)
- [Inference & Deployment](#-inference--deployment)
- [Web UI](#-web-ui)
- [License](#-license)
- [Contact](#-contact)

---

## 📝 Introduction

**USF BIOS** (UltraSafe Fine-tuning Basic Input/Output System) is an enterprise-grade AI training and fine-tuning platform developed by **US Inc**. It provides a comprehensive, production-ready infrastructure for training, fine-tuning, evaluating, and deploying large language models (LLMs) and multimodal models.

### Key Features

- 🚀 **Enterprise-Ready**: Production-grade architecture designed for scalability and reliability
- 🔒 **Secure**: Built with security-first principles by US Inc
- 🎯 **Optimized for USF Models**: First-class support for USF Mini, USF Omega, and USF ASI1
- 📊 **Full Pipeline**: Training → Fine-tuning → Evaluation → Deployment
- 🔧 **Multiple Training Methods**: SFT, LoRA, QLoRA, RLHF, GRPO, DPO, and more
- 🖼️ **Multimodal Support**: Train with text, images, video, and audio
- ⚡ **High Performance**: Megatron parallelism, DeepSpeed, FSDP support
- 🌐 **Web UI**: User-friendly interface for training and inference

---

## 🤖 Supported Models

### US Inc Models (Recommended)

| Model | Parameters | Type | Description |
|-------|------------|------|-------------|
| **USF Mini** | upto 40B | LLM | Compact, efficient model for edge deployment |
| **USF Omega** | upto 40B | LLM | High-performance flagship model |
| **USF ASI1** | upto 40B | LLM | Advanced superintelligent reasoning model |
| **USF Mini-VL** | upto 48B | Multimodal | Vision-language model for multimodal tasks |
| **USF Omega-VL** | upto 48B | Multimodal | Advanced multimodal understanding |

### Other Supported Models

USF BIOS also supports 600+ text-only large models and 300+ multimodal large models:

**Text Models**: Qwen3, Llama4, Mistral, DeepSeek-R1, InternLM3, GLM4.5, Yi, Baichuan, and more.

**Multimodal Models**: Qwen3-VL, Llava, InternVL3.5, MiniCPM-V, DeepSeek-VL2, and more.

---

## 🛠️ Installation

### Prerequisites

- Python 3.9+ (recommended: 3.10 or 3.11)
- PyTorch 2.0+
- CUDA 12.0+ (for GPU training)

### Install via pip

```bash
pip install usf-bios -U
```

### Install from Source

```bash
git clone https://github.com/us-inc/usf-bios.git
cd usf-bios
pip install -e .
```

### Environment Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Python | >=3.9 | Recommended: 3.10/3.11 |
| PyTorch | >=2.0 | Recommended: 2.8.0+ |
| CUDA | 12.x | Required for GPU training |
| transformers | >=4.33 | Recommended: 4.57+ |
| deepspeed | >=0.14 | For distributed training |
| vllm | >=0.5.1 | For inference acceleration |

---

## 🚀 Quick Start

### Fine-tune USF Omega with LoRA (Single GPU)

```bash
CUDA_VISIBLE_DEVICES=0 \
usf_bios sft \
    --model arpitsh018/usf-omega-40b-instruct \
    --train_type lora \
    --dataset your_dataset.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --output_dir output/usf-omega-lora
```

### Inference with Trained Model

```bash
CUDA_VISIBLE_DEVICES=0 \
usf_bios infer \
    --adapters output/usf-omega-lora/checkpoint-xxx \
    --stream true \
    --temperature 0.7 \
    --max_new_tokens 2048
```

---

## 📚 Fine-tuning Guide

USF BIOS supports multiple fine-tuning methods for different use cases and hardware configurations.

### SFT (Supervised Fine-Tuning)

Full-parameter supervised fine-tuning for maximum model adaptation.

```bash
# Full Fine-tuning (requires significant GPU memory)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
usf_bios sft \
    --model arpitsh018/usf-omega-40b-instruct \
    --train_type full \
    --dataset your_dataset.jsonl \
    --deepspeed zero3 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --max_length 4096 \
    --output_dir output/usf-omega-full
```

### LoRA Fine-tuning

Low-Rank Adaptation for efficient fine-tuning with minimal GPU memory.

```bash
# LoRA Fine-tuning (memory efficient)
CUDA_VISIBLE_DEVICES=0 \
usf_bios sft \
    --model arpitsh018/usf-omega-40b-instruct \
    --train_type lora \
    --dataset your_dataset.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --output_dir output/usf-omega-lora
```

**LoRA Parameters:**
- `lora_rank`: Rank of the low-rank matrices (8-64 recommended)
- `lora_alpha`: Scaling factor (typically 2x rank)
- `target_modules`: Which layers to apply LoRA (use `all-linear` for best results)

### QLoRA Fine-tuning

Quantized LoRA for training large models on consumer GPUs.

```bash
# QLoRA Fine-tuning (4-bit quantization)
CUDA_VISIBLE_DEVICES=0 \
usf_bios sft \
    --model arpitsh018/usf-omega-40b-instruct \
    --train_type lora \
    --quant_bits 4 \
    --dataset your_dataset.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --output_dir output/usf-omega-qlora
```

**Memory Requirements:**
- 7B model: ~6GB VRAM
- 14B model: ~10GB VRAM
- 40B model: ~24GB VRAM
- 70B model: ~40GB VRAM

### RLHF (Reinforcement Learning from Human Feedback)

Train models to align with human preferences using reward models.

```bash
# DPO Training
CUDA_VISIBLE_DEVICES=0 \
usf_bios rlhf \
    --rlhf_type dpo \
    --model arpitsh018/usf-omega-40b-instruct \
    --dataset your_preference_dataset.jsonl \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-5 \
    --beta 0.1 \
    --max_length 2048 \
    --output_dir output/usf-omega-dpo
```

### GRPO (Group Relative Policy Optimization)

Advanced reinforcement learning with group-based optimization.

```bash
# GRPO Training with vLLM acceleration
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
usf_bios rlhf \
    --rlhf_type grpo \
    --model arpitsh018/usf-omega-40b-instruct \
    --train_type lora \
    --use_vllm true \
    --vllm_mode colocate \
    --dataset your_rl_dataset.jsonl \
    --num_generations 4 \
    --max_completion_length 512 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --output_dir output/usf-omega-grpo
```

**GRPO Variants Supported:**
- GRPO (base algorithm)
- DAPO, GSPO, SAPO, CISPO
- CHORD, RLOO, Reinforce++

### DPO (Direct Preference Optimization)

Simple and effective preference learning without reward models.

```bash
# DPO Training
CUDA_VISIBLE_DEVICES=0 \
usf_bios rlhf \
    --rlhf_type dpo \
    --model arpitsh018/usf-omega-40b-instruct \
    --dataset preference_data.jsonl \
    --train_type lora \
    --beta 0.1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --output_dir output/usf-omega-dpo
```

**Other RLHF Methods:**
- **KTO**: Knowledge Transfer Optimization
- **CPO**: Contrastive Preference Optimization
- **SimPO**: Simple Preference Optimization
- **ORPO**: Odds Ratio Preference Optimization
- **PPO**: Proximal Policy Optimization

---

## 🖼️ Multimodal Training

Train vision-language models with images, video, and audio.

### Vision-Language Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 \
usf_bios sft \
    --model arpitsh018/usf-omega-vl-40b \
    --train_type lora \
    --dataset your_multimodal_dataset.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --freeze_vit true \
    --max_length 4096 \
    --output_dir output/usf-omega-vl-lora
```

**Dataset Format for Multimodal:**
```json
{"messages": [{"role": "user", "content": "<image>Describe this image."}, {"role": "assistant", "content": "This image shows..."}], "images": ["path/to/image.jpg"]}
```

---

## 🚀 Inference & Deployment

### Local Inference

```bash
# Interactive inference
CUDA_VISIBLE_DEVICES=0 \
usf_bios infer \
    --model arpitsh018/usf-omega-40b-instruct \
    --stream true \
    --temperature 0.7 \
    --max_new_tokens 2048

# With LoRA adapter
CUDA_VISIBLE_DEVICES=0 \
usf_bios infer \
    --adapters output/usf-omega-lora/checkpoint-xxx \
    --stream true \
    --merge_lora true \
    --infer_backend vllm
```

### Deploy as API Server

```bash
# Deploy with vLLM backend
CUDA_VISIBLE_DEVICES=0 \
usf_bios deploy \
    --model arpitsh018/usf-omega-40b-instruct \
    --infer_backend vllm \
    --port 8000
```

### Model Export & Quantization

```bash
# Export with AWQ quantization
CUDA_VISIBLE_DEVICES=0 \
usf_bios export \
    --model arpitsh018/usf-omega-40b-instruct \
    --quant_bits 4 \
    --quant_method awq \
    --output_dir usf-omega-40b-awq
```

---

## 🌐 Web UI

Launch the USF BIOS Web UI for a user-friendly training experience.

```bash
usf_bios app --server_port 7860
```

Features:
- Model selection and configuration
- Dataset upload and preview
- Training parameter tuning
- Real-time training monitoring
- Inference testing

---

## 📋 Dataset Format

### Chat Format (Recommended)

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi! How can I help you today?"}]}
```

### Instruction Format

```json
{"instruction": "Translate to French", "input": "Hello, how are you?", "output": "Bonjour, comment allez-vous?"}
```

### Preference Format (for RLHF)

```json
{"prompt": "Write a poem about AI", "chosen": "In circuits deep...", "rejected": "AI is cool..."}
```

---

## 🏛 License

### Proprietary License

**Copyright (c) 2024-2026 US Inc. All Rights Reserved.**

This software and associated documentation files (the "Software") are the exclusive proprietary property of **US Inc**.

**TERMS OF USE:**

1. **No Unauthorized Use**: You may not use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software without explicit written permission from US Inc.

2. **No Ownership Claims**: No individual or organization may claim ownership, authorship, or any intellectual property rights over this Software.

3. **Authorized Use Only**: Use of this Software is permitted only under a valid license agreement with US Inc.

4. **No Reverse Engineering**: You may not reverse engineer, decompile, or disassemble the Software.

5. **Confidentiality**: The Software contains trade secrets and proprietary information of US Inc. You agree to maintain the confidentiality of the Software.

For licensing inquiries, please contact: **licensing@us.inc**

---

## 📞 Contact

**US Inc**

- **Website**: [https://us.inc](https://us.inc)
- **Email**: contact@us.inc
- **Licensing**: licensing@us.inc
- **Technical Support**: support@us.inc

---

<p align="center">
  <strong>USF BIOS - Powered by US Inc</strong>
  <br>
  <em>Enterprise-Grade AI Training Infrastructure</em>
</p>
