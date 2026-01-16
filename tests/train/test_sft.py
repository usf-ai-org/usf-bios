import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm_ddp():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            split_dataset_ratio=0.01,
            # ddp_find_unused_parameters=False,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            target_modules=['all-linear', 'all-embedding'],
            modules_to_save=['all-embedding', 'all-norm'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_unsloth():
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            split_dataset_ratio=0.01,
            max_steps=5,
            tuner_backend='unsloth',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    result = sft_main(SftArguments(resume_from_checkpoint=last_model_checkpoint, load_data_args=True, max_steps=10))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_mllm_mp():
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2.5-VL-7B-Instruct',
            dataset=['detection-datasets/coco:validation#20'],
            # dataset=['detection-datasets/coco:validation#20', 'tatsu-lab/alpaca#20'],
            split_dataset_ratio=0.01,
            train_type='lora',
            target_modules=['all-linear'],
            freeze_aligner=False,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_llm_streaming():
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct', dataset=['usf_bios/chinese-c4'], streaming=True, max_steps=16, **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm_streaming():
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['detection-datasets/coco:validation', 'tatsu-lab/alpaca'],
            streaming=True,
            max_steps=16,
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm_zero3():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    sft_main(
        SftArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['detection-datasets/coco:validation#100', 'tatsu-lab/alpaca#100'],  #
            split_dataset_ratio=0.01,
            deepspeed='zero3',
            **kwargs))


def test_qwen_vl():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    sft_main(
        SftArguments(
            model='Qwen/Qwen-VL-Chat',
            dataset=['linxy/LaTeX_OCR#40', 'detection-datasets/coco:validation#40'],
            split_dataset_ratio=0.01,
            **kwargs))


def test_qwen2_audio():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    sft_main(
        SftArguments(
            model='Qwen/Qwen2-Audio-7B-Instruct',
            dataset=['speech_asr/speech_asr_aishell1_trainsets:validation#200'],
            split_dataset_ratio=0.01,
            freeze_parameters_ratio=1,
            trainable_parameters=['audio_tower'],
            train_type='full',
            **kwargs))


def test_llm_gptq():
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct-GPTQ-Int4',
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True))


def test_llm_awq():
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct-AWQ',
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True))


def test_mllm_streaming_zero3():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    sft_main(
        SftArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['detection-datasets/coco:validation', 'tatsu-lab/alpaca'],
            streaming=True,
            max_steps=16,
            deepspeed='zero3',
            **kwargs))


def test_mllm_streaming_mp_ddp():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    sft_main(
        SftArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['detection-datasets/coco:validation', 'tatsu-lab/alpaca'],
            streaming=True,
            max_steps=16,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            **kwargs))


def test_llm_hqq():
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            split_dataset_ratio=0.01,
            quant_method='hqq',
            quant_bits=4,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True))


def test_llm_bnb():
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            split_dataset_ratio=0.01,
            quant_method='bnb',
            quant_bits=4,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True))


def test_moe():
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4',
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_resume_from_checkpoint():
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            max_steps=5,
            streaming=True,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-0.5B',
            resume_from_checkpoint=last_model_checkpoint,
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            streaming=True,
            load_data_args=True,
            max_steps=10,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_resume_only_model():
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            max_steps=5,
            save_only_model=True,
            deepspeed='zero3',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-0.5B',
            resume_from_checkpoint=last_model_checkpoint,
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            resume_only_model=True,
            save_only_model=True,
            load_data_args=True,
            max_steps=10,
            deepspeed='zero3',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    print(f'last_model_checkpoint: {last_model_checkpoint}')


def test_llm_transformers_4_33():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    sft_main(
        SftArguments(
            model='Qwen/Qwen-7B-Chat',
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            split_dataset_ratio=0.01,
            **kwargs))


def test_predict_with_generate():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    # 'detection-datasets/coco:validation#100',
    sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['tatsu-lab/alpaca#400'],
            predict_with_generate=True,
            # padding_free=True,
            max_length=512,
            packing=True,
            attn_impl='flash_attn',
            split_dataset_ratio=0.01,
            **kwargs))


def test_predict_with_generate_zero3():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    # 'detection-datasets/coco:validation#100',
    sft_main(
        SftArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['linxy/LaTeX_OCR#40'],
            split_dataset_ratio=0.01,
            predict_with_generate=True,
            freeze_vit=False,
            deepspeed='zero3',
            **kwargs))


def test_template():
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    global kwargs
    kwargs = kwargs.copy()
    kwargs['num_train_epochs'] = 3
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['usf_bios/self-cognition#200'],
            split_dataset_ratio=0.01,
            model_name=['小黄'],
            model_author=['usf_bios'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_emu3_gen():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['max_position_embeddings'] = '10240'
    os.environ['image_area'] = '518400'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    kwargs['num_train_epochs'] = 100
    result = sft_main(
        SftArguments(model='BAAI/Emu3-Gen', dataset=['usf_bios/TextCaps#2'], split_dataset_ratio=0.01, **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    args = InferArguments(
        ckpt_dir=last_model_checkpoint,
        infer_backend='transformers',
        stream=False,
        use_chat_template=False,
        top_k=2048,
        max_new_tokens=40960)
    infer_main(args)


def test_eval_strategy():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct',
            eval_strategy='no',
            dataset=['tatsu-lab/alpaca#100', 'tatsu-lab/alpaca#100'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_epoch():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments

    train_kwargs = kwargs.copy()
    train_kwargs['num_train_epochs'] = 3
    # train_kwargs['save_steps'] = 2  # not use
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['tatsu-lab/alpaca#50', 'tatsu-lab/alpaca#50'],
            split_dataset_ratio=0.01,
            save_strategy='epoch',
            **train_kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_agent():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments

    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['usf_bios/ToolBench#500'],
            split_dataset_ratio=0.01,
            loss_scale='react',
            agent_template='toolbench',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_grounding():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from usf_bios import sft_main, SftArguments, infer_main, InferArguments

    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2.5-VL-7B-Instruct',
            dataset=['detection-datasets/coco#200'],
            split_dataset_ratio=0.01,
            dataset_num_proc=4,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, stream=True, max_new_tokens=2048))


if __name__ == '__main__':
    # test_llm_ddp()
    # test_mllm_mp()
    # test_llm_streaming()
    # test_mllm_streaming()
    # test_mllm_zero3()
    # test_llm_gptq()
    # test_llm_awq()
    # test_mllm_streaming_zero3()
    # test_mllm_streaming_mp_ddp()
    # test_llm_bnb()
    # test_llm_hqq()
    # test_moe()
    # test_resume_from_checkpoint()
    test_resume_only_model()
    # test_llm_transformers_4_33()
    # test_predict_with_generate()
    # test_predict_with_generate_zero3()
    # test_template()
    # test_qwen_vl()
    # test_qwen2_audio()
    # test_emu3_gen()
    # test_unsloth()
    # test_eval_strategy()
    # test_epoch()
    # test_agent()
    # test_grounding()
