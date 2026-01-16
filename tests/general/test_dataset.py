from typing import List

from usf_bios.dataset import load_dataset


def _test_dataset(datasets: List[str], num_proc: int = 1, strict: bool = False, **kwargs):
    dataset = load_dataset(datasets, num_proc=num_proc, strict=strict, **kwargs)
    print(f'dataset[0]: {dataset[0]}')
    print(f'dataset[1]: {dataset[1]}')


def test_sft():
    # usf_bios/SlimOrca  usf_bios/cosmopedia-100k
    # _test_dataset(['lvjianjin/AdvertiseGen'])
    # _test_dataset(['OpenGVLab/Duet-v0.5'])
    # _test_dataset(['usf_bios/SlimOrca', 'usf_bios/cosmopedia-100k'])
    # _test_dataset(['OmniData/Zhihu-KOL-More-Than-100-Upvotes'])
    # _test_dataset(['OmniData/Zhihu-KOL'])
    _test_dataset([
        'tatsu-lab/alpaca#1000', 'tatsu-lab/alpaca#1000',
        'Yukang/LongAlpaca-12k#1000'
    ])
    # _test_dataset(['usf_bios/Infinity-Instruct:all'])
    # _test_dataset(['usf_bios/sharegpt:all'])
    # _test_dataset(['anon8231489123/ShareGPT_Vicuna_unfiltered:all'])
    # _test_dataset(['iic/ms_bench'])
    # _test_dataset(['usf_bios/tagengo-gpt4'])


def test_mllm():
    # _test_dataset(['Lin-Chen/ShareGPT4V:all'])
    # _test_dataset(['liuhaotian/LLaVA-Pretrain'])
    # _test_dataset(['usf_bios/TextCaps'])
    # _test_dataset(['usf_bios/RLAIF-V-Dataset:all'])
    # _test_dataset(['usf_bios/OK-VQA_train'])
    # _test_dataset(['usf_bios/OCR-VQA'])
    # _test_dataset(['usf_bios/A-OKVQA'])
    # _test_dataset(['Enxin/MovieChat-1K-test'])
    _test_dataset([
        'linxy/LaTeX_OCR:all', 'detection-datasets/coco:validation',
        'speech_asr/speech_asr_aishell1_trainsets:validation'
    ],
                  strict=False)
    # _test_dataset(['usf_bios/VideoChatGPT:all'])
    # _test_dataset(['speech_asr/speech_asr_aishell1_trainsets:validation'])
    # _test_dataset(['project-sloth/captcha-images'])
    # _test_dataset(['usf_bios/gpt4v-dataset:all'])
    # _test_dataset(['detection-datasets/coco:validation'])
    # _test_dataset(['liuhaotian/LLaVA-Instruct-150K'], num_proc=16)


def test_agent():
    _test_dataset(['usf_bios/ToolBench'])
    # _test_dataset(['ms::usf-bios/ms_agent_for_agentfabric:all'])


def test_dpo():
    _test_dataset(['mlabonne/orpo-dpo-mix-40k'])
    _test_dataset(['Anthropic/hh-rlhf:all'])
    _test_dataset(['Anthropic/hh-rlhf:all'])
    _test_dataset(['hjh0119/shareAI-Llama3-DPO-zh-en-emoji:all'])


def test_kto():
    _test_dataset(['argilla/ultrafeedback-binarized-preferences-cleaned'])


def test_pretrain():
    _test_dataset(['LooksJuicy/ruozhiba:all'])


def test_dataset_info():
    _test_dataset(['usf_bios/self-cognition#500'], model_name='xiao huang', model_author='usf_bios')
    # _test_dataset(['codefuse-ai/CodeExercise-Python-27k'])


def test_cls():
    _test_dataset(['simpleai/HC3-Chinese:baike'])
    _test_dataset(['simpleai/HC3-Chinese:baike_cls'])


if __name__ == '__main__':
    # test_sft()
    # test_agent()
    # test_dpo()
    # test_kto()
    test_mllm()
    # test_pretrain()
    # test_dataset_info()
    # test_cls()
