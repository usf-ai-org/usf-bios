# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class SelfCog(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'selfcog_tab': {
            'label': {
                'en': 'Self cognition settings'
            },
        },
        'model_name': {
            'label': {
                'en': 'Model name'
            },
            'info': {
                'en': 'Set the name of the model think itself of, the format is Chinesename Englishname, split by space'
            }
        },
        'model_author': {
            'label': {
                'en': 'Model author'
            },
            'info': {
                'en': 'Set the author of the model, the format is Chineseauthor Englishauthor, split by space'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='selfcog_tab'):
            with gr.Row():
                gr.Textbox(elem_id='model_name', scale=20, is_list=True)
                gr.Textbox(elem_id='model_author', scale=20, is_list=True)
