import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def test_sft():
    from usf_bios.megatron import megatron_sft_main, MegatronSftArguments
    megatron_sft_main(
        MegatronSftArguments(
            load='Qwen2-7B-Instruct-mcore',
            dataset=[
                'tatsu-lab/alpaca#500', 'usf_bios/self-cognition#500',
                'tatsu-lab/alpaca#500'
            ],
            split_dataset_ratio=0.01,
            tensor_model_parallel_size=2,
            train_iters=100,
            model_author=['usf_bios'],
            model_name=['usf_bios-robot'],
            sequence_parallel=True,
            finetune=True))


def test_pt():
    from usf_bios.megatron import megatron_pretrain_main, MegatronPretrainArguments
    megatron_pretrain_main(
        MegatronPretrainArguments(
            load='Qwen2-7B-mcore',
            dataset=['tatsu-lab/alpaca#500', 'tatsu-lab/alpaca#500'],
            split_dataset_ratio=0.01,
            tensor_model_parallel_size=2,
            train_iters=200,
            eval_iters=5,
            finetune=True))


if __name__ == '__main__':
    test_sft()
    # test_pt()
