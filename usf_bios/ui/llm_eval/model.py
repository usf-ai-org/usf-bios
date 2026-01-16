# Copyright (c) US Inc. All rights reserved.
from functools import partial
from typing import Type

import gradio as gr

from usf_bios.arguments import EvalArguments
from usf_bios.model import ModelType, get_model_list
from usf_bios.template import TEMPLATE_MAPPING
from ..base import BaseUI


class Model(BaseUI):

    group = 'llm_eval'

    locale_dict = {
        'checkpoint': {
            'value': {
                'en': 'Trained model'
            }
        },
        'model_type': {
            'label': {
                'en': 'Select Model Type'
            },
            'info': {
                'en': 'Base model type supported by USF BIOS'
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
        'reset': {
            'value': {
                'en': 'Reset to default'
            },
        },
        'template': {
            'label': {
                'en': 'Prompt template type'
            },
            'info': {
                'en': 'Choose the template type of the model'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            gr.Dropdown(
                elem_id='model',
                scale=20,
                choices=get_model_list(),
                value='Qwen/Qwen2.5-7B-Instruct',
                allow_custom_value=True)
            gr.Dropdown(elem_id='model_type', choices=ModelType.get_model_name_list(), scale=20)
            gr.Dropdown(elem_id='template', choices=list(TEMPLATE_MAPPING.keys()), scale=20)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('model').change(
            partial(cls.update_input_model, arg_cls=EvalArguments, has_record=False),
            inputs=[cls.element('model')],
            outputs=list(cls.valid_elements().values()))
