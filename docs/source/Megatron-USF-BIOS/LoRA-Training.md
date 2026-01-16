# LoRA Training

Best practice reference for single-node 8xH20 LoRA training with Qwen3-235B-A22B-Instruct-250718: https://github.com/us-inc/usf-bios/pull/5033.

For environment setup, please refer to the [Quick Start Guide](./Quick-start.md) of Megatron-USF BIOS.

## Traditional Method

### Converting HF to Mcore

Below, we introduce weight conversion using the `usf export` and `megatron export` commands respectively. Compared to `usf export`, `megatron export` supports multi-node and LoRA incremental weight conversion, but is also more complex, requiring additional specification of parallelism parameters during export, such as `--tensor_model_parallel_size` and `--export_model_parallel_size`. For details, refer to the [Mcore-Bridge Documentation](./Mcore-Bridge.md). To use the `usf export` command, refer to the [Quick Start Documentation](./Quick-start.md).
- `usf export` uses a single process, places HF weights on the GPU, and uses device_map for parallelization; mcore weights are placed on the CPU without enabling parallelization. This approach is very easy to debug and test the precision alignment between HF and mcore.
- `megatron export` uses torchrun to launch multiple processes, places mcore weights on the GPU, supports enabling various parallelization methods, fp8, mtp, etc., with comprehensive functionality. If precision alignment testing is needed, the last rank will load HF weights and place them on the CPU.

```shell
# megatron export
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor_model_parallel_size 2 \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --save Qwen2.5-7B-Instruct-mcore \
    --test_convert_precision true

# usf export
# CUDA_VISIBLE_DEVICES=0 \
# usf export \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --to_mcore true \
#     --torch_dtype bfloat16 \
#     --output_dir Qwen2.5-7B-Instruct-mcore \
#     --test_convert_precision true
```

### LoRA Training

Training Script:

```bash
# full: 2 * 70GiB 0.61s/it
# lora: 2 * 14GiB 0.45s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --load Qwen2.5-7B-Instruct-mcore \
    --save_safetensors false \
    --dataset 'tatsu-lab/alpaca#500' \
              'tatsu-lab/alpaca#500' \
              'usf_bios/self-cognition#500' \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/Qwen2.5-7B-Instruct \
    --save_interval 100 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author usf \
    --model_name usf-robot
```
- For LoRA training scripts of MoE models, please refer to [here](https://github.com/us-inc/usf-bios/tree/main/examples/megatron/lora).

### Converting MCore to HF

```bash
# megatron export
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron export \
    --adapter_load megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --merge_lora false \
    --torch_dtype bfloat16 \
    --save megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-hf \
    --test_convert_precision true

# usf export
# CUDA_VISIBLE_DEVICES=0 \
# usf export \
#     --mcore_adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx \
#     --to_hf true \
#     --torch_dtype bfloat16 \
#     --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-hf \
#     --test_convert_precision true
```

- Note: The `--adapter_load/--mcore_adapters` folder contains an `args.json` file. The conversion process will read the `--model/--mcore_model` and LoRA-related parameter information from this file. `usf export` does not currently support conversion of LoRA incremental weights. With `megatron export`, you can use the `--merge_lora` parameter to control whether to merge weights.

### Inference

```shell
# If using full weights, replace `--adapters` with `--model`
CUDA_VISIBLE_DEVICES=0 \
usf infer \
    --adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-hf \
    --stream true
```

### Merge-LoRA

If you only want to merge the LoRA weights without converting them to Hugging Face format, for subsequent DPO training, you can use the following script:

```shell
# megatron export
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron export \
    --adapter_load megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx \
    --tensor_model_parallel_size 2 \
    --to_mcore true \
    --merge_lora true \
    --torch_dtype bfloat16 \
    --save megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-mcore \
    --test_convert_precision true

# usf export
# CUDA_VISIBLE_DEVICES=0 \
# usf export \
#     --mcore_adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx \
#     --to_mcore true \
#     --torch_dtype bfloat16 \
#     --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-mcore \
#     --test_convert_precision true
```

## Mcore-Bridge [Recommended]

### Training

```shell
# full: 2 * 70GiB 0.61s/it
# lora: 2 * 14GiB 0.45s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'tatsu-lab/alpaca#500' \
              'tatsu-lab/alpaca#500' \
              'usf_bios/self-cognition#500' \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/Qwen2.5-7B-Instruct \
    --save_interval 100 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author usf \
    --model_name usf-robot
```

### Inference

```shell
# If using full weights, replace `--adapters` with `--model`
CUDA_VISIBLE_DEVICES=0 \
usf infer \
    --adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-hf \
    --stream true
```
