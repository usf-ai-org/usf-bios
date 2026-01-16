# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform

if __name__ == '__main__':
    from usf_bios.ray import try_init_ray
    try_init_ray()
    from usf_bios.pipelines import sampling_main
    sampling_main()
