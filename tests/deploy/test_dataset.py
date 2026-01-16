def _test_client(port=8000):
    import time
    from usf_bios.dataset import load_dataset
    from usf_bios.infer_engine import InferClient, InferRequest, RequestConfig
    dataset = load_dataset(['tatsu-lab/alpaca#1000'], num_proc=4)
    infer_client = InferClient(port=port)
    while True:
        try:
            infer_client.models
            break
        except Exception:
            time.sleep(1)
            pass
    infer_requests = []
    for data in dataset[0]:
        infer_requests.append(InferRequest(**data))
    request_config = RequestConfig(seed=42, max_tokens=256, temperature=0.8)

    resp = infer_client.infer(infer_requests, request_config=request_config, use_tqdm=False)
    print(len(resp))


def _test(infer_backend):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from usf_bios.pipelines import run_deploy
    from usf_bios.arguments import DeployArguments
    args = DeployArguments(model='Qwen/Qwen2-7B-Instruct', infer_backend=infer_backend, verbose=False)
    with run_deploy(args) as port:
        _test_client(port)


def test_vllm():
    _test('vllm')


def test_lmdeploy():
    _test('lmdeploy')


def test_pt():
    _test('transformers')


def test_vllm_origin():
    import subprocess
    import sys
    from usf_bios.hub import snapshot_download
    model_dir = snapshot_download('Qwen/Qwen2-7B-Instruct')
    args = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', model_dir]
    process = subprocess.Popen(args)
    _test_client()
    process.terminate()


if __name__ == '__main__':
    # test_vllm_origin()
    # test_vllm()
    test_lmdeploy()
    # test_pt()
