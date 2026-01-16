# Here is the command-line style training code.
# 22GB
CUDA_VISIBLE_DEVICES=0 \
usf sft \
 --model Qwen/Qwen2.5-3B-Instruct \
 --train_type lora \
 --dataset 'AI-HuggingFace/alpaca-gpt4-data-zh#500' \
 'AI-HuggingFace/alpaca-gpt4-data-en#500' \
 'usf_bios/self-cognition#500' \
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
 --system 'You are a helpful assistant.' \
 --warmup_ratio 0.05 \
 --dataloader_num_workers 4 \
 --dataset_num_proc 4 \
 --model_name 'Xiao Huang' \
 --model_author '' 'HuggingFace'
