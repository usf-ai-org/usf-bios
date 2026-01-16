# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
import os

if __name__ == '__main__':
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    from usf_bios.megatron import megatron_pretrain_main
    megatron_pretrain_main()
