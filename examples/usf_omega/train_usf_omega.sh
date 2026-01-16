#!/bin/bash
# USF Omega Fine-tuning Script
# Dataset: TeichAI/glm-4.7-2000x

# Single GPU LoRA (~22GB VRAM)
CUDA_VISIBLE_DEVICES=0 \
usf sft \
    --model arpitsh018/usf-omega-40b-base \
    --train_type lora \
    --dataset 'TeichAI/glm-4.7-2000x' \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 4096 \
    --output_dir output/usf_omega_finetuned \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
