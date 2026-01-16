import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm():
    from usf_bios import pretrain_main, PretrainArguments, infer_main, InferArguments
    result = pretrain_main(
        PretrainArguments(
            model='Qwen/Qwen2-7B-Instruct', dataset=['usf_bios/sharegpt:all#100'], split_dataset_ratio=0.01, **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm():
    from usf_bios import pretrain_main, PretrainArguments, infer_main, InferArguments
    result = pretrain_main(
        PretrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['detection-datasets/coco:validation#20', 'tatsu-lab/alpaca#20'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


if __name__ == '__main__':
    # test_llm()
    test_mllm()
