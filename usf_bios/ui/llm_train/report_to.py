# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class ReportTo(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'reporter_tab': {
            'label': {
                'en': 'Training report'
            },
        },
        'report_to': {
            'label': {
                'en': 'Report to'
            },
        },
        'swanlab_token': {
            'label': {
                'en': 'The login token of SwanLab'
            },
        },
        'swanlab_project': {
            'label': {
                'en': 'Project of SwanLab'
            },
        },
        'swanlab_workspace': {
            'label': {
                'en': 'Workspace of SwanLab'
            },
        },
        'swanlab_exp_name': {
            'label': {
                'en': 'Experiment of SwanLab'
            },
        },
        'swanlab_mode': {
            'label': {
                'en': 'Work mode of SwanLab'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='reporter_tab'):
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(
                        elem_id='report_to',
                        multiselect=True,
                        is_list=True,
                        choices=['tensorboard', 'wandb', 'swanlab'],
                        allow_custom_value=True,
                        scale=20)
                    gr.Textbox(elem_id='swanlab_token', lines=1, scale=20)
                    gr.Textbox(elem_id='swanlab_project', lines=1, scale=20)
                with gr.Row():
                    gr.Textbox(elem_id='swanlab_workspace', lines=1, scale=20)
                    gr.Textbox(elem_id='swanlab_exp_name', lines=1, scale=20)
                    gr.Dropdown(elem_id='swanlab_mode', scale=20)
