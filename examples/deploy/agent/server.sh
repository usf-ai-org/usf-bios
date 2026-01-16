CUDA_VISIBLE_DEVICES=0 usf deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 2048 \
    --agent_template hermes \
    --served_model_name Qwen2.5-7B-Instruct
