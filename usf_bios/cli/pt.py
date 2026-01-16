# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform

if __name__ == '__main__':
    from usf_bios.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from usf_bios.pipelines import pretrain_main
    pretrain_main()
