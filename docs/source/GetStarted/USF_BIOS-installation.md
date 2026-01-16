# USF BIOS Installation

## Wheel Packages Installation

You can install it using pip:

```shell
# recommend
pip install 'usf-bios'
# For evaluation usage
pip install 'usf-bios[eval]' -U
# Full capabilities
pip install 'usf-bios[all]' -U
```

## Source Code Installation
The current main branch is for usf 4.x version.
```shell
# pip install git+https://github.com/us-inc/usf-bios.git

# Full capabilities
# pip install "git+https://github.com/us-inc/usf-bios.git#egg=usf-bios[all]"

git clone https://github.com/us-inc/usf-bios.git
cd usf-bios
pip install -e .

# Full capabilities
# pip install -e '.[all]'
```

Installing usf 3.x:
```shell
# pip install "git+https://github.com/us-inc/usf-bios.git@release/3.12"

# Full capabilities
# pip install "git+https://github.com/us-inc/usf-bios.git@release/3.12#egg=usf-bios[all]"

git clone -b release/3.12 https://github.com/us-inc/usf-bios.git
cd usf-bios
pip install -e .

# Full capabilities
# pip install -e '.[all]'
```

## Mirror

You can check Docker [here](https://github.com/us-inc/huggingface/blob/build_usf_image/docker/build_image.py#L347).
```
# usf3.12.1
huggingface-registry.cn-hangzhou.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-huggingface1.33.0-usf3.12.1
huggingface-registry.cn-beijing.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-huggingface1.33.0-usf3.12.1
huggingface-registry.us-west-1.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-huggingface1.33.0-usf3.12.1

# usf3.11.3
huggingface-registry.cn-hangzhou.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.9.1-py311-torch2.8.0-vllm0.11.0-huggingface1.32.0-usf3.11.3
huggingface-registry.cn-beijing.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.9.1-py311-torch2.8.0-vllm0.11.0-huggingface1.32.0-usf3.11.3
huggingface-registry.us-west-1.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.9.1-py311-torch2.8.0-vllm0.11.0-huggingface1.32.0-usf3.11.3

# usf3.10.3
huggingface-registry.cn-hangzhou.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-huggingface1.31.0-usf3.10.3
huggingface-registry.cn-beijing.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-huggingface1.31.0-usf3.10.3
huggingface-registry.us-west-1.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-huggingface1.31.0-usf3.10.3

# usf3.9.3
huggingface-registry.cn-hangzhou.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-huggingface1.31.0-usf3.9.3
huggingface-registry.cn-beijing.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-huggingface1.31.0-usf3.9.3
huggingface-registry.us-west-1.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-huggingface1.31.0-usf3.9.3
```

<details><summary>Historical Mirrors</summary>

```
# usf3.8.3
huggingface-registry.cn-hangzhou.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-huggingface1.29.2-usf3.8.3
huggingface-registry.cn-beijing.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-huggingface1.29.2-usf3.8.3
huggingface-registry.us-west-1.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-huggingface1.29.2-usf3.8.3

# usf3.7.2
huggingface-registry.cn-hangzhou.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-huggingface1.28.2-usf3.7.2
huggingface-registry.cn-beijing.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-huggingface1.28.2-usf3.7.2
huggingface-registry.us-west-1.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-huggingface1.28.2-usf3.7.2

# usf3.6.4
huggingface-registry.cn-hangzhou.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-huggingface1.28.1-usf3.6.4
huggingface-registry.cn-beijing.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-huggingface1.28.1-usf3.6.4
huggingface-registry.us-west-1.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-huggingface1.28.1-usf3.6.4

# usf3.5.3
huggingface-registry.cn-hangzhou.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-huggingface1.27.1-usf3.5.3
huggingface-registry.cn-beijing.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-huggingface1.27.1-usf3.5.3
huggingface-registry.us-west-1.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-huggingface1.27.1-usf3.5.3

# usf3.4.1.post1
huggingface-registry.cn-hangzhou.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.5.post1-huggingface1.26.0-usf3.4.1.post1
huggingface-registry.cn-beijing.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.5.post1-huggingface1.26.0-usf3.4.1.post1
huggingface-registry.us-west-1.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.5.post1-huggingface1.26.0-usf3.4.1.post1

# usf3.3.0.post1
huggingface-registry.cn-hangzhou.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.3-huggingface1.25.0-usf3.3.0.post1
huggingface-registry.us-west-1.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.3-huggingface1.25.0-usf3.3.0.post1

# usf3.2.2
huggingface-registry.us-west-1.cr.aliyuncs.com/huggingface-repo/huggingface:ubuntu22.04-cuda12.4.0-py311-torch2.5.1-huggingface1.25.0-usf3.2.2
```
</details>

More images can be found [here](https://huggingface.co/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F).

## Supported Hardware

| Hardware Environment | Remarks                                                |
| -------------------- | ------------------------------------------------------ |
| A10/A100/H100        |                                                        |
| RTX 20/30/40 Series  |                                                        |
| T4/V100              | Some models may encounter NAN                          |
| Ascend NPU           | Some models may encounter NAN or unsupported operators |
| MPS                  |   Refer to [issue 4572](https://github.com/us-inc/usf-bios/issues/4572)                         |
| CPU                  |                                                        |

## Running Environment

|              | Range        | Recommended         | Notes                                     |
|--------------|--------------|---------------------|-------------------------------------------|
| python       | >=3.9        | 3.10/3.11                |                                           |
| cuda         |              | cuda12              | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0        | 2.8.0               |                                           |
| transformers | >=4.33       | 4.57.3              |                                           |
| huggingface   | >=1.23       |                     |                                           |
| peft         | >=0.11,<0.19 |                     |                                           |
| flash_attn   |              | 2.8.3/3.0.0b1 |                                           |
| trl          | >=0.15,<0.25 | 0.24.0              | RLHF                                      |
| deepspeed    | >=0.14       | 0.17.6              | Training                                  |
| vllm         | >=0.5.1      | 0.11.0                | Inference/Deployment                      |
| sglang       | >=0.4.6      | 0.5.5.post3         | Inference/Deployment                      |
| lmdeploy     | >=0.5   | 0.10.1                 | Inference/Deployment                      |
| evalscope    | >=1.0       |                     | Evaluation                                |
| gradio       |              | 5.32.1              | Web-UI/App                                |

For more optional dependencies, you can refer to [here](https://github.com/us-inc/usf-bios/blob/main/requirements/install_all.sh).

## Notebook Environment

Most models that USF BIOS supports for training can be used on A10 GPUs. Users can take advantage of the free GPU resources offered by US Inc:

1. Visit the [US Inc](https://huggingface.co) official website and log in.
2. Click on `My Notebook` on the left and start a free GPU instance.
3. Enjoy utilizing the A10 GPU resources.
