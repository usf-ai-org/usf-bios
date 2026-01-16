# 76GiB
CUDA_VISIBLE_DEVICES=0 \
usf sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --dataset 'AI-HuggingFace/alpaca-gpt4-data-zh#500' \
              'AI-HuggingFace/alpaca-gpt4-data-en#500' \
              'usf_bios/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author usf \
    --model_name usf-robot
