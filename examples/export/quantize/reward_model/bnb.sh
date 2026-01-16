# bnb quantize
CUDA_VISIBLE_DEVICES=0 usf export \
    --model Shanghai_AI_Laboratory/internlm2-1_8b-reward \
    --output_dir output/internlm2-1_8b-reward-bnb-int4 \
    --quant_bits 4 \
    --quant_method bnb

# infer
CUDA_VISIBLE_DEVICES=0 usf infer \
    --model output/internlm2-1_8b-reward-bnb-int4 \
    --val_dataset 'AI-HuggingFace/alpaca-gpt4-data-zh#1000' \
    --max_batch_size 16
