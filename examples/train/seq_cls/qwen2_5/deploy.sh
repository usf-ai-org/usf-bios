CUDA_VISIBLE_DEVICES=0 \
usf deploy \
 --adapters output/vx-xxx/checkpoint-xxx

# curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
# "model": "Qwen2.5-0.5B",
# "messages": [{"role": "user", "content": "，。"}]
# }'
