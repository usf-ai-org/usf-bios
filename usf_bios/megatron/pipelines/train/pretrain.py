# Copyright (c) US Inc. All rights reserved.
from typing import List, Optional, Union

from usf_bios.megatron.arguments import MegatronPretrainArguments
from usf_bios.utils import get_logger
from .sft import MegatronSft

logger = get_logger()


class MegatronPretrain(MegatronSft):
    args_class = MegatronPretrainArguments
    args: args_class


def megatron_pretrain_main(args: Optional[Union[List[str], MegatronPretrainArguments]] = None):
    from usf_bios.system_guard import guard_with_integrity
    guard_with_integrity()
    return MegatronPretrain(args).main()
