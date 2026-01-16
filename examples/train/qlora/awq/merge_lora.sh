# This will store the full, unquantized weights.
usf export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapters output/vx-xxx/checkpoint-xxx \
    --merge_lora true
