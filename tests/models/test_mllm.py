import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_cogvlm():
    from usf_bios import infer_main, InferArguments, sft_main, SftArguments
    # infer_main(InferArguments(model='ZhipuAI/cogvlm2-video-llama3-chat'))
    sft_main(
        SftArguments(
            model='ZhipuAI/cogvlm2-video-llama3-chat',
            dataset=['tatsu-lab/alpaca#200', 'usf_bios/VideoChatGPT:Generic#200'],
            split_dataset_ratio=0.01))


if __name__ == '__main__':
    test_cogvlm()
