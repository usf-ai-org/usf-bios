NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
usf infer \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --infer_backend transformers \
    --val_dataset AI-HuggingFace/LaTeX_OCR#1000 \
    --max_batch_size 16 \
    --max_new_tokens 512
