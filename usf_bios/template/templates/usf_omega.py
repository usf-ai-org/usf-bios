# Copyright (c) Ultrasafe AI. All rights reserved.
# USF Omega Template Registration
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from transformers.utils import strtobool

from ..constant import LLMTemplateType
from ..register import Template, TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Prompt, Word

# Special tokens - BOS/EOS from tokenizer_config.json
BOS_TOKEN = '<|startoftext|>'
EOS_TOKEN = '<|endoftext|>'
PAD_TOKEN = '<|padding|>'

# Message boundary tokens
START_TOKEN = '<|:@:|start|:@:|>'
END_TOKEN = '<|:@::|end|:@::|>'

# Assistant internal tokens
REASONING_START = '<|:@:|reasoning_start|:@:|>'
REASONING_END = '<|:@::|reasoning_end|:@::|>'
MESSAGE_START = '<|:@:|message|:@:|>'
MESSAGE_END = '<|:@::|message|:@::|>'
FUNCTIONS_START = '<|:@:|functions_start|:@:|>'
FUNCTIONS_END = '<|:@::|functions_end|:@::|>'
INVOKE_START = '<|:@:|invoke_start|:@:|>'
INVOKE_END = '<|:@::|invoke_end|:@::|>'
CONSTRAIN_TOKEN = '<|:@:|constrain|:@:|>'
CONSTRAIN_END = '<|:@::|constrain|:@::|>'
FUNCTION_RESULTS_START = '<|:@:|function_results|:@:|>'
FUNCTION_RESULTS_END = '<|:@::|function_results|:@::|>'


class UsfOmegaTemplate(Template):

    def _usf_prepare_inputs(self, inputs: StdTemplateInputs):
        """Prepare inputs for USF Omega template with reasoning support."""
        super()._usf_prepare_inputs(inputs)
        # Handle reasoning/thinking content in assistant messages
        for message in inputs.messages:
            if message['role'] == 'assistant' and message.get('content'):
                content = message['content']
                # Convert <think> tags to USF Omega reasoning format if present
                if '<think>' in content and '</think>' in content:
                    pre, rest = content.split('<think>', maxsplit=1)
                    think, post = rest.split('</think>', maxsplit=1)
                    message['content'] = (
                        pre + REASONING_START + think + REASONING_END + 
                        MESSAGE_START + post + MESSAGE_END
                    )

    def _jinja_encode(self, inputs: StdTemplateInputs):
        return super()._jinja_encode(inputs)


@dataclass
class UsfOmegaTemplateMeta(TemplateMeta):
    template_type: str = 'usf_omega'
    prefix: Prompt = field(default_factory=lambda: ['<|startoftext|>'])
    prompt: Prompt = field(default_factory=lambda: [
        '<|:@:|start|:@:|>user\n{{QUERY}}<|:@::|end|:@::|>\n<|:@:|start|:@:|>assistant\n'
    ])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: [
        '<|:@:|start|:@:|>system\n{{SYSTEM}}<|:@::|end|:@::|>\n'
    ])
    auto_add_bos: bool = True
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|:@::|end|:@::|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|:@::|end|:@::|><|endoftext|>'])
    template_cls: Type[Template] = UsfOmegaTemplate
    default_system: Optional[str] = None
    stop_words: List[Word] = field(default_factory=lambda: ['<|:@::|end|:@::|>', '<|endoftext|>'])
    agent_template: str = 'react_en'
    is_thinking: bool = True
    thinking_prefix: str = '<|:@:|reasoning_start|:@:|>'


register_template(UsfOmegaTemplateMeta(LLMTemplateType.usf_omega, default_system=None, template_cls=UsfOmegaTemplate))
