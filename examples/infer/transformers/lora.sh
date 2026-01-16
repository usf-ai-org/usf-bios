# Since `usf_bios/test_lora` is trained by usf and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
# To disable this behavior, please set `--load_args false`.
CUDA_VISIBLE_DEVICES=0 \
usf infer \
    --adapters usf_bios/test_lora \
    --infer_backend transformers \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
