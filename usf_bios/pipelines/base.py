# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
import datetime as dt
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import usf_bios
from usf_bios.arguments import AppArguments, BaseArguments, WebUIArguments
from usf_bios.utils import (ProcessorMixin, get_logger, is_master, parse_args, seed_everything,
                         setup_graceful_exit, show_startup_banner, show_success, show_training_complete,
                         show_training_failed, start_training_timer)

logger = get_logger()


class USFPipeline(ABC, ProcessorMixin):
    args_class = BaseArguments

    def __init__(self, args: Optional[Union[List[str], args_class]] = None):
        self.args = self._parse_args(args)
        args = self.args
        if hasattr(args, 'seed'):
            seed = args.seed + max(getattr(args, 'rank', -1), 0)
            seed_everything(seed)
        logger.info_debug(f'args: {args}')
        self._compat_dsw_gradio(args)

    def _parse_args(self, args: Optional[Union[List[str], args_class]] = None) -> args_class:
        if isinstance(args, self.args_class):
            return args
        assert self.args_class is not None
        args, remaining_argv = parse_args(self.args_class, args)
        if len(remaining_argv) > 0:
            if getattr(args, 'ignore_args_error', False):
                logger.warning(f'remaining_argv: {remaining_argv}')
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')
        return args

    @staticmethod
    def _compat_dsw_gradio(args) -> None:
        if (isinstance(args, (WebUIArguments, AppArguments)) and 'JUPYTER_NAME' in os.environ
                and 'dsw-' in os.environ['JUPYTER_NAME'] and 'GRADIO_ROOT_PATH' not in os.environ):
            os.environ['GRADIO_ROOT_PATH'] = f"/{os.environ['JUPYTER_NAME']}/proxy/{args.server_port}"

    def main(self):
        logger.info_debug(f'[USF BIOS] Start time: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
        logger.info_debug(f'[USF BIOS] Version: {usf_bios.__version__}')
        
        # Setup graceful exit handling
        setup_graceful_exit()
        
        # Show startup banner for training pipelines (only on master)
        args = self.args
        if is_master() and hasattr(args, 'train_type'):
            model_name = getattr(args, 'model', None)
            train_type = getattr(args, 'train_type', 'sft')
            use_hf = getattr(args, 'use_hf', True)
            show_startup_banner(
                model=model_name,
                train_type=train_type,
                use_hf=use_hf
            )
        
        # Start training timer
        start_training_timer()
        
        try:
            result = self.run()
            
            # Show completion summary for training pipelines
            if is_master() and hasattr(args, 'output_dir') and result:
                output_dir = getattr(args, 'output_dir', None)
                final_loss = None
                total_steps = None
                
                if isinstance(result, dict):
                    log_history = result.get('log_history', [])
                    if log_history:
                        for entry in reversed(log_history):
                            if 'loss' in entry and final_loss is None:
                                final_loss = entry.get('loss')
                            if 'step' in entry and total_steps is None:
                                total_steps = entry.get('step')
                            if final_loss is not None and total_steps is not None:
                                break
                
                show_training_complete(
                    output_dir=output_dir,
                    final_loss=final_loss,
                    total_steps=total_steps
                )
            
            logger.info_debug(f'[USF BIOS] End time: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
            return result
            
        except KeyboardInterrupt:
            if is_master():
                show_training_failed("Training interrupted by user (Ctrl+C)")
            raise
        except Exception as e:
            if is_master():
                show_training_failed(str(e))
            raise

    @abstractmethod
    def run(self):
        pass


# Alias for backward compatibility
USFPipeline = USFPipeline
