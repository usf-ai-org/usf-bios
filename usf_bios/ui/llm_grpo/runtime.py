# Copyright (c) US Inc. All rights reserved.
import os

import gradio as gr

from usf_bios.utils import get_logger
from ..llm_train import Runtime

logger = get_logger()


class GRPORuntime(Runtime):

    group = 'llm_grpo'

    locale_dict = {
        'runtime_tab': {
            'label': {
                'en': 'Runtime'
            },
        },
        'tb_not_found': {
            'value': {
                'en': 'tensorboard not found, install it by `pip install tensorboard`',
            }
        },
        'running_cmd': {
            'label': {
                'en': 'Command line'
            },
            'info': {
                'en': 'The actual command'
            }
        },
        'show_running_cmd': {
            'value': {
                'en': 'Show running command line'
            },
        },
        'show_sh': {
            'label': {
                'en': 'Show sh command line'
            },
        },
        'cmd_sh': {
            'label': {
                'en': 'Training command line'
            },
            'info': {
                'en': ('Please press "Show running command line" if the content is none, '
                       'click the "Save training command" below to save the sh script')
            }
        },
        'save_cmd_as_sh': {
            'value': {
                'en': 'Save training command'
            }
        },
        'save_cmd_alert': {
            'value': {
                'en': 'The training command line will be saved in: {}'
            }
        },
        'close_cmd_show': {
            'value': {
                'en': 'Close training command show'
            }
        },
        'show_log': {
            'value': {
                'en': 'Show running status'
            },
        },
        'stop_show_log': {
            'value': {
                'en': 'Stop showing running status'
            },
        },
        'logging_dir': {
            'label': {
                'en': 'Logging dir'
            },
            'info': {
                'en': 'Support fill custom path in'
            }
        },
        'log': {
            'label': {
                'en': 'Logging content'
            },
            'info': {
                'en': 'Please press "Show running status" if the log content is not updating'
            }
        },
        'running_tasks': {
            'label': {
                'en': 'Running Tasks'
            },
            'info': {
                'en': 'All running tasks(started by `usf rlhf --rlhf_type grpo`)'
            }
        },
        'refresh_tasks': {
            'value': {
                'en': 'Find running tasks'
            },
        },
        'kill_task': {
            'value': {
                'en': 'Kill running task'
            },
        },
        'tb_url': {
            'label': {
                'en': 'Tensorboard URL'
            },
            'info': {
                'en': 'Not editable'
            }
        },
        'start_tb': {
            'value': {
                'en': 'Start TensorBoard'
            },
        },
        'close_tb': {
            'value': {
                'en': 'Close TensorBoard'
            },
        },
    }

    @classmethod
    def save_cmd(cls, cmd):
        if len(cmd) > 0:
            cmd_sh, output_dir = cls.cmd_to_sh_format(cmd)
            os.makedirs(output_dir, exist_ok=True)
            sh_file_path = os.path.join(output_dir, 'grpo.sh')
            gr.Info(cls.locale('save_cmd_alert', cls.lang)['value'].format(sh_file_path))
            with open(sh_file_path, 'w', encoding='utf-8') as f:
                f.write(cmd_sh)
