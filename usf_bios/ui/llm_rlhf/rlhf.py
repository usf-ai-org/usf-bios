# Copyright (c) US Inc. All rights reserved.
from functools import partial
from typing import Type

import gradio as gr

from usf_bios.model import ModelType, get_model_list
from ..base import BaseUI


class RLHF(BaseUI):

    group = 'llm_rlhf'

    locale_dict = {
        'rlhf_tab': {
            'label': {
                'en': 'Alignment params settings'
            },
        },
        'ref_model': {
            'label': {
                'en': 'Ref model id or path'
            },
            'info': {
                'en': 'The actual model id or path'
            }
        },
        'ref_model_type': {
            'label': {
                'en': 'Ref model type'
            },
            'info': {
                'en': 'Model type supported by USF BIOS'
            }
        },
        'reward_model': {
            'label': {
                'en': 'Reward model id or path'
            },
            'info': {
                'en': 'The actual model id or path'
            }
        },
        'reward_model_type': {
            'label': {
                'en': 'Reward model type'
            },
            'info': {
                'en': 'Model type supported by USF BIOS'
            }
        },
        'teacher_model': {
            'label': {
                'en': 'Teacher model id or path'
            },
            'info': {
                'en': 'The actual model id or path'
            }
        },
        'teacher_model_type': {
            'label': {
                'en': 'Teacher model type'
            },
            'info': {
                'en': 'Model type supported by USF BIOS'
            }
        },
        'beta': {
            'label': {
                'en': 'KL regression ratio'
            },
        },
        'max_completion_length': {
            'label': {
                'en': 'Max completion length'
            },
        },
        'loss_scale': {
            'label': {
                'en': 'Loss weights setting'
            },
        },
        'lmbda': {
            'label': {
                'en': 'GKD student data ratio'
            },
        },
        'cpo_alpha': {
            'label': {
                'en': 'CPO/SimPO NLL loss coefficient'
            },
        },
        'rpo_alpha': {
            'label': {
                'en': 'DPO Cross Entropy ratio'
            },
        },
        'simpo_gamma': {
            'label': {
                'en': 'SimPO reward margin'
            },
        },
        'desirable_weight': {
            'label': {
                'en': 'KTO desirable ratio'
            },
        },
        'undesirable_weight': {
            'label': {
                'en': 'KTO undesirable ratio'
            },
        }
    }

    rlhf_args_dict = {
        'dpo': ['rpo_alpha', 'ref_model', 'ref_model_type'],
        'cpo': ['cpo_alpha'],
        'kto': ['desirable_weight', 'undesirable_weight', 'ref_model', 'ref_model_type'],
        'simpo': ['simpo_gamma', 'cpo_alpha'],
        'gkd': ['teacher_model', 'teacher_model_type', 'max_completion_length', 'lmbda'],
        'ppo': ['reward_model', 'reward_model_type', 'max_completion_length', 'ref_model', 'ref_model_type']
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='rlhf_tab', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Slider(elem_id='beta', minimum=0., maximum=5.0, step=0.1, value=0.1, scale=10)
                    gr.Slider(elem_id='rpo_alpha', minimum=0., maximum=2, step=0.1, scale=10)
                    gr.Slider(elem_id='lmbda', minimum=0., maximum=1.0, step=0.1, scale=10)
                    gr.Slider(elem_id='simpo_gamma', minimum=0., maximum=2.0, step=0.1, scale=10)
                    gr.Slider(elem_id='desirable_weight', minimum=0., maximum=2.0, step=0.1, scale=10)
                    gr.Slider(elem_id='undesirable_weight', minimum=0., maximum=2.0, step=0.1, scale=10)
                with gr.Row():
                    gr.Textbox(elem_id='max_completion_length', scale=10)
                    gr.Textbox(elem_id='loss_scale', scale=10)
                    gr.Slider(elem_id='cpo_alpha', minimum=0., maximum=1, step=0.1, scale=10)
                    gr.Dropdown(
                        elem_id='teacher_model',
                        scale=20,
                        value=None,
                        choices=get_model_list(),
                        allow_custom_value=True)
                    gr.Dropdown(
                        elem_id='teacher_model_type',
                        choices=ModelType.get_model_name_list(),
                        value=None,
                        scale=10,
                        allow_custom_value=True)
                with gr.Row():
                    gr.Dropdown(
                        elem_id='ref_model', scale=20, value=None, choices=get_model_list(), allow_custom_value=True)
                    gr.Dropdown(
                        elem_id='ref_model_type',
                        choices=ModelType.get_model_name_list(),
                        value=None,
                        scale=10,
                        allow_custom_value=True)
                    gr.Dropdown(
                        elem_id='reward_model', scale=20, value=None, choices=get_model_list(), allow_custom_value=True)
                    gr.Dropdown(
                        elem_id='reward_model_type',
                        choices=ModelType.get_model_name_list(),
                        value=None,
                        scale=10,
                        allow_custom_value=True)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('ref_model').change(
            partial(cls.update_input_model, allow_keys=['ref_model_type'], has_record=False, is_ref_model=True),
            inputs=[cls.element('ref_model')],
            outputs=[cls.element('ref_model_type')])
        cls.element('reward_model').change(
            partial(cls.update_input_model, allow_keys=['reward_model_type'], has_record=False, is_ref_model=True),
            inputs=[cls.element('reward_model')],
            outputs=[cls.element('reward_model_type')])
        cls.element('teacher_model').change(
            partial(cls.update_input_model, allow_keys=['teacher_model_type'], has_record=False, is_ref_model=True),
            inputs=[cls.element('teacher_model')],
            outputs=[cls.element('teacher_model_type')])

    @staticmethod
    def update_beta(rlhf_type):
        beta_value_dict = {'simpo': 2., 'gkd': 0.5, 'grpo': 0.04}
        return beta_value_dict.get(rlhf_type, 0.1) if rlhf_type else 0.1
