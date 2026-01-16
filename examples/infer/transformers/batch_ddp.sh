# 18GB
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
usf infer \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --infer_backend transformers \
    --val_dataset AI-HuggingFace/alpaca-gpt4-data-zh#1000 \
    --max_batch_size 16 \
    --max_new_tokens 512
