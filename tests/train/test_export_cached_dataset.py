def test_export_cached_dataset():
    from usf_bios import export_main, ExportArguments
    export_main(
        ExportArguments(
            model='Qwen/Qwen2.5-7B-Instruct',
            dataset='usf_bios/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT',
            to_cached_dataset=True,
            dataset_num_proc=4,
        ))
    print()


def test_sft():
    from usf_bios import sft_main, SftArguments
    sft_main(
        SftArguments(
            model='Qwen/Qwen2.5-7B-Instruct',
            dataset='liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT#1000',
            dataset_num_proc=2,
            packing=True,
            attn_impl='flash_attn',
        ))


if __name__ == '__main__':
    # test_export_cached_dataset()
    test_sft()
