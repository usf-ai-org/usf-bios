# 2 * 80GB
OMP_NUM_THREADS=14 \
CUDA_VISIBLE_DEVICES=0,1 \
usf export \
    --model Qwen/Qwen2-57B-A14B-Instruct \
    --dataset 'AI-HuggingFace/alpaca-gpt4-data-zh#1000' \
              'AI-HuggingFace/alpaca-gpt4-data-en#1000' \
    --quant_n_samples 512 \
    --quant_batch_size 1 \
    --max_length 4096 \
    --quant_method gptq \
    --quant_bits 4 \
    --output_dir Qwen2-57B-A14B-Instruct-GPTQ-Int4
