# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class Target(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'target_params': {
            'label': {
                'en': 'Tuner modules params'
            }
        },
        'freeze_llm': {
            'label': {
                'en': 'Freeze LLM'
            },
        },
        'freeze_aligner': {
            'label': {
                'en': 'Freeze aligner'
            },
        },
        'freeze_vit': {
            'label': {
                'en': 'Freeze ViT'
            },
        },
        'target_modules': {
            'label': {
                'en': 'Specify the tuner module'
            }
        },
        'target_regex': {
            'label': {
                'en': 'Tuner module regex expression'
            }
        },
        'modules_to_save': {
            'label': {
                'en': 'Original model modules to train and save'
            }
        },
        'init_weights': {
            'label': {
                'en': 'Init tuner weights'
            },
            'info': {
                'en': ('LoRA: gaussian/pissa/pissa_niter_[n]/olora/loftq/lora-ga/true/false, '
                       'Bone: bat/true/false')
            }
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Blocks():
            with gr.Row():
                gr.Textbox(elem_id='target_modules', lines=1, value='all-linear', is_list=True, scale=5)
                gr.Checkbox(elem_id='freeze_llm', scale=5)
                gr.Checkbox(elem_id='freeze_aligner', scale=5)
                gr.Checkbox(elem_id='freeze_vit', scale=5)
            with gr.Row():
                gr.Textbox(elem_id='target_regex', scale=5)
                gr.Textbox(elem_id='modules_to_save', scale=5)
                gr.Textbox(elem_id='init_weights', scale=5)
