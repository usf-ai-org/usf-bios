CUDA_VISIBLE_DEVICES=0 \
usf infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --max_batch_size 16 \
    --load_data_args true \
    --temperature 0
