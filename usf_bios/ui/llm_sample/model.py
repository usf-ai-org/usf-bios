# Copyright (c) US Inc. All rights reserved.
from functools import partial
from typing import Type

import gradio as gr

from usf_bios.arguments import SamplingArguments
from usf_bios.model import ModelType, get_model_list
from usf_bios.template import TEMPLATE_MAPPING
from ..base import BaseUI


class Model(BaseUI):

    group = 'llm_sample'

    locale_dict = {
        'model_type': {
            'label': {
                'en': 'Select Model Type'
            },
            'info': {
                'en': 'Base model type supported by USF BIOS, Please leave it blank if model is the service name'
            }
        },
        'model': {
            'label': {
                'en': 'Model id, path or server name'
            },
            'info': {
                'en': ('The actual model id or path, if is a trained model, please fill in the checkpoint-xxx dir, '
                       'if is a model service, please fill in the server name')
            }
        },
        'template': {
            'label': {
                'en': 'Prompt template type'
            },
            'info': {
                'en': 'Choose the template type of the model, Please leave it blank if model is the service name'
            }
        },
        'system': {
            'label': {
                'en': 'System'
            },
            'info': {
                'en': 'System can be modified after the model weights loaded'
            }
        },
        'prm_model': {
            'label': {
                'en': 'Process Reward Model'
            },
            'info': {
                'en': 'It can be a model id, or a prm key defined in the plugin'
            }
        },
        'orm_model': {
            'label': {
                'en': 'Outcome Reward Model'
            },
            'info': {
                'en': 'Usually a wildcard or test case, etc., defined in the plugin'
            }
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
        with gr.Row():
            gr.Textbox(elem_id='system', lines=1)
        with gr.Row():
            gr.Textbox(elem_id='prm_model', scale=20)
            gr.Textbox(elem_id='orm_model', scale=20)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('model').change(
            partial(cls.update_input_model, arg_cls=SamplingArguments, has_record=False),
            inputs=[cls.element('model')],
            outputs=list(cls.valid_elements().values()))
