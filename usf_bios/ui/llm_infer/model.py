# Copyright (c) US Inc. All rights reserved.
from functools import partial
from typing import Type

import gradio as gr

from usf_bios.arguments import DeployArguments
from usf_bios.model import ModelType, get_model_list
from usf_bios.template import TEMPLATE_MAPPING
from ..base import BaseUI
from .generate import Generate


class Model(BaseUI):

    llm_train = 'llm_infer'

    sub_ui = [Generate]

    locale_dict = {
        'model_type': {
            'label': {
                'en': 'Select Model Type'
            },
            'info': {
                'en': 'Base model type supported by USF BIOS'
            }
        },
        'load_checkpoint': {
            'value': {
                'en': 'Deploy model',
            }
        },
        'model': {
            'label': {
                'en': 'Model id or path'
            },
            'info': {
                'en': 'The actual model id or path, if is a trained model, please fill in the checkpoint-xxx dir'
            }
        },
        'template': {
            'label': {
                'en': 'Prompt template type'
            },
            'info': {
                'en': 'Choose the template type of the model'
            }
        },
        'merge_lora': {
            'label': {
                'en': 'Merge LoRA'
            },
            'info': {
                'en': 'Only available when `sft_type=lora`'
            }
        },
        'lora_modules': {
            'label': {
                'en': 'More LoRA modules'
            },
            'info': {
                'en': 'name=/path1/path2 split by blanks'
            }
        },
        'more_params': {
            'label': {
                'en': 'More params'
            },
            'info': {
                'en': 'Fill in with json format or --xxx xxx cmd format'
            }
        },
        'reset': {
            'value': {
                'en': 'Reset to default'
            },
        },
        'infer_backend': {
            'label': {
                'en': 'Infer backend'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row(equal_height=True):
            gr.Dropdown(
                elem_id='model',
                scale=20,
                choices=get_model_list(),
                value='Qwen/Qwen2.5-7B-Instruct',
                allow_custom_value=True)
            gr.Dropdown(elem_id='model_type', choices=ModelType.get_model_name_list(), scale=20)
            gr.Dropdown(elem_id='template', choices=list(TEMPLATE_MAPPING.keys()), scale=20)
            gr.Checkbox(elem_id='merge_lora', scale=4)
            gr.Button(elem_id='reset', scale=2)
        with gr.Row():
            gr.Dropdown(elem_id='infer_backend', value='transformers', scale=5)
        Generate.set_lang(cls.lang)
        Generate.build_ui(base_tab)
        with gr.Row(equal_height=True):
            gr.Textbox(elem_id='lora_modules', lines=1, is_list=True, scale=40)
            gr.Textbox(elem_id='more_params', lines=1, scale=20)
            gr.Button(elem_id='load_checkpoint', scale=2, variant='primary')

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('model').change(
            partial(cls.update_input_model, arg_cls=DeployArguments, has_record=False),
            inputs=[cls.element('model')],
            outputs=list(cls.valid_elements().values()))
