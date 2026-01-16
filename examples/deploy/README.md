Please refer to the examples in [examples/infer](../../infer/) and change `usf infer` to `usf deploy` to start the service. (You need to additionally remove `--val_dataset`)

e.g.
```shell
CUDA_VISIBLE_DEVICES=0 \
usf deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm
```
