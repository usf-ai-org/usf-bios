# Copyright (c) US Inc. All rights reserved.
from typing import List, Optional, Union

from usf_bios.arguments import PretrainArguments
from usf_bios.utils import get_logger
from .sft import USFSft

logger = get_logger()


class USFPretrain(USFSft):
    args_class = PretrainArguments
    args: args_class


def pretrain_main(args: Optional[Union[List[str], PretrainArguments]] = None):
    return USFPretrain(args).main()
