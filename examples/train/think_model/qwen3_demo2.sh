# use `usf_bios/self-cognition:qwen3`
# Avoid losing the thinking capability by appending `/no_think` to the dataset query.
# https://github.com/us-inc/usf-bios/blob/77985c2ccdac8ed4037174ee222e79d1f1d5059d/usf_bios/llm/dataset/dataset/llm.py#L835
CUDA_VISIBLE_DEVICES=0 \
usf sft \
    --model Qwen/Qwen3-8B \
    --train_type lora \
    --dataset 'usf_bios/Qwen3-SFT-Mixin#2000' \
              'usf_bios/self-cognition:qwen3#600' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_liger_kernel true \
    --load_from_cache_file false \
    --model_author usf \
    --model_name usf-robot
