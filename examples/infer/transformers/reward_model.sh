CUDA_VISIBLE_DEVICES=0 \
usf infer \
    --model Shanghai_AI_Laboratory/internlm2-1_8b-reward \
    --val_dataset AI-HuggingFace/alpaca-gpt4-data-zh#1000 \
    --max_batch_size 64
