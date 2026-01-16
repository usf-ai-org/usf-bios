# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr

from ..base import BaseUI


class Task(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'embed_tab': {
            'label': {
                'en': 'Embedding'
            },
        },
        'loss_type': {
            'label': {
                'en': 'Loss type'
            }
        },
        'seq_cls_tab': {
            'label': {
                'en': 'Sequence Classification'
            },
        },
        'num_labels': {
            'label': {
                'en': 'Number of labels'
            }
        },
        'use_chat_template': {
            'label': {
                'en': 'use chat template'
            },
            'info': {
                'en': 'Use the chat template or generation template'
            }
        },
        'task_type': {
            'label': {
                'en': 'Task type'
            },
        },
        'task_params': {
            'label': {
                'en': 'Task params'
            },
        }
    }

    tabs_to_filter = {'embedding': ['loss_type'], 'seq_cls': ['num_labels', 'use_chat_template']}

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='task_params', open=False):
            gr.Dropdown(elem_id='task_type', choices=['causal_lm', 'seq_cls', 'embedding'])
            with gr.Tabs():
                with gr.TabItem(elem_id='embed_tab'):
                    with gr.Row():
                        gr.Dropdown(
                            elem_id='loss_type',
                            choices=['cosine_similarity', 'contrastive', 'online_contrastive', 'infonce'])
                with gr.TabItem(elem_id='seq_cls_tab'):
                    with gr.Row():
                        gr.Textbox(elem_id='num_labels', scale=4)
                        gr.Checkbox(elem_id='use_chat_template', value=True, scale=4)
