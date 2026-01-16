# Copyright (c) US Inc. All rights reserved.
import os
from typing import List, Optional, Union

import torch

from usf_bios.arguments import ExportArguments
from usf_bios.template import TEMPLATE_MAPPING
from usf_bios.utils import is_debug_mode, get_logger
from ..train import USFSft

logger = get_logger()


class ExportCachedDataset(USFSft):
    args_class = ExportArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], ExportArguments]] = None) -> None:
        super(USFSft, self).__init__(args)
        args = self.args
        self.train_msg = {}  # dummy
        template_cls = TEMPLATE_MAPPING[args.template].template_cls
        if template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self._prepare_model_tokenizer(**kwargs)
        self._prepare_template()
        self.template.set_mode(args.template_mode)

    def _post_process_datasets(self, datasets: List) -> List:
        return datasets

    def main(self):
        train_dataset, val_dataset = self._prepare_dataset()
        train_data_dir = os.path.join(self.args.output_dir, 'train')
        val_data_dir = os.path.join(self.args.output_dir, 'val')
        train_dataset.save_to_disk(train_data_dir)
        if val_dataset is not None:
            val_dataset.save_to_disk(val_data_dir)
        if is_debug_mode():
            logger.info_debug(f'cached_dataset: `{train_data_dir}`')
        if val_dataset is not None:
            if is_debug_mode():
                logger.info_debug(f'cached_val_dataset: `{val_data_dir}`')


def export_cached_dataset(args: Optional[Union[List[str], ExportArguments]] = None):
    return ExportCachedDataset(args).main()
