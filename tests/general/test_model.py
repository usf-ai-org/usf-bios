import os

import torch

from usf_bios.utils import get_device

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def test_qwen2():
    import os
    from usf_bios.model import get_model_processor
    model, tokenizer = get_model_processor('Qwen/Qwen2-7B-Instruct', load_model=False)
    print(f'model: {model}, tokenizer: {tokenizer}')
    # test hf
    model, tokenizer = get_model_processor('Qwen/Qwen2-7B-Instruct', load_model=False, use_hf=True)

    model, tokenizer = get_model_processor(
        'Qwen/Qwen2-7B-Instruct', torch_dtype=torch.float32, device_map=get_device(), attn_impl='flash_attn')
    print(f'model: {model}, tokenizer: {tokenizer}')


def test_huggingface_hub():
    from usf_bios.model import get_model_processor
    model, tokenizer = get_model_processor('Qwen/Qwen2___5-Math-1___5B-Instruct/', load_model=False)


if __name__ == '__main__':
    test_qwen2()
    # test_huggingface_hub()
