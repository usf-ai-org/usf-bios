# Copyright (c) US Inc. All rights reserved.
from typing import Type

import gradio as gr
from packaging import version

from usf_bios.utils import get_logger
from ..base import BaseUI
from ..llm_infer import Runtime

logger = get_logger()


class EvalRuntime(Runtime):

    group = 'llm_eval'

    cmd = 'eval'

    locale_dict = {
        'runtime_tab': {
            'label': {
                'en': 'Runtime'
            },
        },
        'running_cmd': {
            'label': {
                'en': 'Command line'
            },
            'info': {
                'en': 'The actual command'
            }
        },
        'show_log': {
            'value': {
                'en': 'Show eval status'
            },
        },
        'stop_show_log': {
            'value': {
                'en': 'Stop showing running status'
            },
        },
        'log': {
            'label': {
                'en': 'Logging content'
            },
            'info': {
                'en': 'Please press "Show eval status" if the log content is not updating'
            }
        },
        'running_tasks': {
            'label': {
                'en': 'Running evaluation'
            },
            'info': {
                'en': 'All tasks started by usf eval'
            }
        },
        'refresh_tasks': {
            'value': {
                'en': 'Find evaluation'
            },
        },
        'kill_task': {
            'value': {
                'en': 'Kill evaluation'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='runtime_tab', open=False, visible=True):
            with gr.Blocks():
                with gr.Row(equal_height=True):
                    gr.Dropdown(elem_id='running_tasks', scale=10)
                    gr.Button(elem_id='refresh_tasks', scale=1, variant='primary')
                    gr.Button(elem_id='show_log', scale=1, variant='primary')
                    gr.Button(elem_id='stop_show_log', scale=1)
                    gr.Button(elem_id='kill_task', scale=1, size='lg')
                with gr.Row():
                    gr.Textbox(elem_id='log', lines=6, visible=False)

                concurrency_limit = {}
                if version.parse(gr.__version__) >= version.parse('4.0.0'):
                    concurrency_limit = {'concurrency_limit': 5}
                cls.log_event = base_tab.element('show_log').click(cls.update_log, [], [cls.element('log')]).then(
                    cls.wait, [base_tab.element('running_tasks')], [cls.element('log')], **concurrency_limit)

                base_tab.element('stop_show_log').click(cls.break_log_event, [cls.element('running_tasks')], [])

                base_tab.element('refresh_tasks').click(
                    cls.refresh_tasks,
                    [base_tab.element('running_tasks')],
                    [base_tab.element('running_tasks')],
                )
