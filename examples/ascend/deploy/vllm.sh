ASCEND_RT_VISIBLE_DEVICES=0 usf deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm \
    --served_model_name Qwen2.5-7B-Instruct
