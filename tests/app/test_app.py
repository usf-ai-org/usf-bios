def test_llm():
    from usf_bios import app_main, AppArguments
    app_main(AppArguments(model='Qwen/Qwen2.5-0.5B-Instruct'))


def test_lora():
    from usf_bios import app_main, AppArguments
    app_main(AppArguments(adapters='usf_bios/test_lora', lang='en', studio_title='小黄'))


def test_mllm():
    from usf_bios import app_main, AppArguments
    app_main(AppArguments(model='Qwen/Qwen2-VL-7B-Instruct', stream=True))


def test_audio():
    from usf_bios import AppArguments, app_main, DeployArguments, run_deploy
    deploy_args = DeployArguments(model='Qwen/Qwen2-Audio-7B-Instruct', infer_backend='transformers', verbose=False)

    with run_deploy(deploy_args, return_url=True) as url:
        app_main(AppArguments(model='Qwen2-Audio-7B-Instruct', base_url=url, stream=True))


if __name__ == '__main__':
    test_mllm()
