# Copyright (c) Ultrasafe AI. All rights reserved.
# USF Omega Template Registration
#
# This module provides template support for USF Omega models with two variants:
# 1. usf_omega - Instruct model (non-reasoning) - uses chat_template.jinja2
# 2. usf_omega_reasoning - Reasoning model (with thinking) - uses chat_template_reasoning.jinja2
#
# Features supported via jinja2 templates:
# - System/Developer messages with model_identity
# - Function definitions and calls (single and parallel)
# - Tool responses (batched)
# - Response schema (structured output)
# - Citations
# - Self-reflection
# - Constrained output (json mode)
# - Reasoning/thinking blocks (reasoning variant only)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from ..constant import LLMTemplateType
from ..register import Template, TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Prompt, Word

# ==================== SPECIAL TOKENS ====================
# Matching jinja2 templates: chat_template.jinja2 and chat_template_reasoning.jinja2
# Token IDs from added_tokens_decoder in tokenizer_config.json

# BOS/EOS/PAD tokens (IDs 0-2)
BOS_TOKEN = '<|:@:|startoftext|:@:|>'      # ID 0
PAD_TOKEN = '<|:@:|padding|:@:|>'          # ID 1
EOS_TOKEN = '<|:@:|endoftext|:@:|>'        # ID 2

# Message boundary tokens (IDs 3-4)
START_TOKEN = '<|:@:|start|:@:|>'          # ID 3
END_TOKEN = '<|:@:|end|:@:|>'              # ID 4

# Constrain tokens (IDs 5-6)
CONSTRAIN_START = '<|:@:|constrain_start|:@:|>'  # ID 5
CONSTRAIN_END = '<|:@:|constrain_end|:@:|>'      # ID 6

# Reasoning tokens (IDs 7-8)
REASONING_START = '<|:@:|reasoning_start|:@:|>'  # ID 7
REASONING_END = '<|:@:|reasoning_end|:@:|>'      # ID 8

# Message content tokens (IDs 9-10)
MESSAGE_START = '<|:@:|message_start|:@:|>'      # ID 9
MESSAGE_END = '<|:@:|message_end|:@:|>'          # ID 10

# Function/Tool call tokens (IDs 11-16)
FUNCTIONS_START = '<|:@:|functions_start|:@:|>'  # ID 11
INVOKE_START = '<||invoke_start to='             # ID 12
PARAMETER_START = '<||parameter_start name='     # ID 13
PARAMETER_END = '<||parameter_end||>'            # ID 14
INVOKE_END = '<||invoke_end||>'                  # ID 15
FUNCTIONS_END = '<|:@:|functions_end|:@:|>'      # ID 16

# Tool response tokens (IDs 17-20)
FUNCTION_RESULTS_START = '<|:@:|function_results_start|:@:|>'  # ID 17
FUNCTION_RESPONSE_START = '<||function_response_start to='     # ID 18
FUNCTION_RESPONSE_END = '<||function_response_end||>'          # ID 19
FUNCTION_RESULTS_END = '<|:@:|function_results_end|:@:|>'      # ID 20

# Self-reflection tokens (IDs 21-22)
SELF_REFLECTION_START = '<|:@:|self_reflection_start|:@:|>'  # ID 21
SELF_REFLECTION_END = '<|:@:|self_reflection_end|:@:|>'      # ID 22

# Citations tokens (IDs 23-24)
CITATIONS_START = '<||citations_start ref='      # ID 23
CITATIONS_END = '<||citations_end||>'            # ID 24

# Search tokens (IDs 25-26)
SEARCH_START = '<|:@:|search_start|:@:|>'        # ID 25
SEARCH_END = '<|:@:|search_end|:@:|>'            # ID 26

# Wait token (ID 27)
WAIT_TOKEN = '<|:@:|wait|:@:|>'                  # ID 27


class UsfOmegaTemplate(Template):
    """Template for USF Omega models (both reasoning and non-reasoning variants).
    
    This template relies on jinja2 chat templates for full feature support:
    - chat_template.jinja2 for non-reasoning (instruct) models
    - chat_template_reasoning.jinja2 for reasoning models
    
    The jinja2 templates handle all complex features including:
    - Function/tool calling with parallel execution support
    - Response schema with TypeScript-like type conversion
    - Citations with source references
    - Self-reflection loops
    - Developer/system message separation
    
    Supported extra_kwargs parameters:
    - model_identity: str - Custom system identity message
    - functions: list - Function definitions (USF format)
    - response_schema: dict - Structured output schema
    - search_enabled: bool - Enable built-in search function
    - citations_enabled: bool - Enable citation instructions
    - self_reflection_enabled: bool - Enable self-reflection
    - json_mode: bool - Enable JSON constrained output
    - reasoning_effort: str - Reasoning effort level (none/minimal/low/medium/high/max)
    
    Message roles supported:
    - system: System message (combined with model_identity)
    - developer: Developer instructions (combined after system)
    - user: User messages
    - assistant: Assistant responses (can contain message, function_calls, reasoning)
    - tool: Tool/function results
    - functions: Function definitions (rendered as namespace)
    """
    
    jinja_enable_thinking_key = 'enable_reasoning'

    def _usf_prepare_inputs(self, inputs: StdTemplateInputs):
        """Prepare inputs for USF Omega template.
        
        Handles:
        - Merging consecutive messages from same role
        - Processing tool messages following assistant
        - Preserving function_calls and tool_calls in messages
        """
        messages = inputs.messages
        if len(messages) < 2:
            return
        
        i = 1
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            pre_role = pre_message['role']
            role = message['role']
            
            # Handle tool messages after assistant - use agent template
            if pre_role == 'assistant' and role == 'tool' and self.template_backend == 'usf_bios':
                i_start = i
                while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool':
                    i += 1
                pre_content = pre_message.get('content', '')
                pre_message['content'], tool_content = self.agent_template._format_tool_responses(
                    pre_content, messages[i_start:i + 1])
                messages[i_start:i + 1] = [{'role': 'tool', 'content': tool_content}]
                i = i_start + 1
            # Merge consecutive assistant or user messages
            elif (pre_role == 'assistant' and role == 'assistant') or (pre_role == 'user' and role == 'user'):
                pre_content = pre_message.get('content', '') or ''
                content = message.get('content', '') or ''
                pre_message['content'] = pre_content + content
                messages.pop(i)
            else:
                i += 1

    def _jinja_encode(self, inputs: StdTemplateInputs):
        """Override jinja encoding to pass USF Omega specific parameters.
        
        Handles:
        - model_identity for system message customization
        - functions/tools for function calling
        - response_schema/response_format for structured output
        - search_enabled, citations_enabled, self_reflection_enabled flags
        - reasoning_effort for reasoning models
        - Developer role messages
        - Proper training boundaries (loss_scale=0 for prompt, 1 for response)
        """
        messages = inputs.messages.copy()
        extra = inputs.extra_kwargs
        
        # Handle system message with model_identity support
        # If model_identity is provided, it becomes the system message
        # Original system content becomes part of developer block (handled by jinja2)
        if inputs.system is not None:
            messages.insert(0, {'role': 'system', 'content': inputs.system})
        
        # Remove trailing None content message
        if messages and messages[-1].get('content') is None:
            messages.pop()
        
        add_generation_prompt = not messages or messages[-1]['role'] != 'assistant'
        
        # Build kwargs for jinja2 template
        kwargs: Dict[str, Any] = {}
        
        # Function/tool definitions - support both formats
        if inputs.tools:
            kwargs['tools'] = inputs.tools
        if extra.get('functions'):
            kwargs['functions'] = extra['functions']
        
        # Response schema/format for structured output
        if extra.get('response_schema'):
            kwargs['response_schema'] = extra['response_schema']
        if extra.get('response_format'):
            kwargs['response_format'] = extra['response_format']
        
        # Model identity (custom system message)
        if extra.get('model_identity'):
            kwargs['model_identity'] = extra['model_identity']
        
        # Feature flags
        for flag in ['search_enabled', 'citations_enabled', 'self_reflection_enabled', 'json_mode']:
            if extra.get(flag) is not None:
                kwargs[flag] = extra[flag]
        
        # Reasoning parameters
        if extra.get('reasoning_effort'):
            kwargs['reasoning_effort'] = extra['reasoning_effort']
        if extra.get('thinking_budget') is not None:
            kwargs['thinking_budget'] = extra['thinking_budget']
        
        # Enable thinking/reasoning for reasoning variant
        if self.template_meta.is_thinking or self.enable_thinking:
            kwargs[self.jinja_enable_thinking_key] = self.enable_thinking
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)
        
        # For inference, return single block
        if not self.is_training:
            return [text], [1.], 0
        
        # For training, split into prompt (loss_scale=0) and response (loss_scale=1)
        # Only train on ASSISTANT role, not on: system, developer, user, tool, functions
        
        # Role markers in jinja2 output
        assistant_marker = START_TOKEN + 'assistant\n'
        
        # All non-trainable role markers
        non_train_markers = [
            START_TOKEN + 'user\n',
            START_TOKEN + 'tool\n', 
            START_TOKEN + 'system\n',
            START_TOKEN + 'developer\n',
            START_TOKEN + 'functions\n',
        ]
        
        # Count assistant responses in original messages
        assistant_count = sum(1 for m in messages if m['role'] == 'assistant')
        
        if assistant_count == 0:
            # No assistant response - don't train on anything
            return [text], [0.], 0
        
        # Split text into segments and assign loss_scale based on role
        context_list = []
        loss_scale_list = []
        
        # Track current position and whether we're in assistant content
        remaining = text
        in_assistant = False
        
        while remaining:
            # Find the next role marker (any role)
            next_pos = len(remaining)
            next_marker = None
            next_is_assistant = False
            
            # Check for assistant marker
            pos = remaining.find(assistant_marker)
            if pos >= 0 and pos < next_pos:
                next_pos = pos
                next_marker = assistant_marker
                next_is_assistant = True
            
            # Check for all non-trainable markers
            for marker in non_train_markers:
                pos = remaining.find(marker)
                if pos >= 0 and pos < next_pos:
                    next_pos = pos
                    next_marker = marker
                    next_is_assistant = False
            
            if next_pos > 0:
                # Add content before the marker
                content = remaining[:next_pos]
                if content:
                    context_list.append(content)
                    loss_scale_list.append(1. if in_assistant else 0.)
            
            if next_marker:
                # Add the marker itself
                context_list.append(next_marker)
                # Marker belongs to the role it introduces
                loss_scale_list.append(1. if next_is_assistant else 0.)
                remaining = remaining[next_pos + len(next_marker):]
                in_assistant = next_is_assistant
            else:
                # No more markers - add remaining content
                if remaining:
                    context_list.append(remaining)
                    loss_scale_list.append(1. if in_assistant else 0.)
                break
        
        answer_len = sum(1 for ls in loss_scale_list if ls > 0)
        return context_list, loss_scale_list, answer_len


class UsfOmegaReasoningTemplate(UsfOmegaTemplate):
    """Template for USF Omega reasoning models with thinking/reasoning blocks.
    
    Uses chat_template_reasoning.jinja2 which supports:
    - Reasoning effort levels: none, minimal, low, medium, high, max
    - enable_reasoning parameter to control reasoning mode
    - Per-message reasoning_effort override
    """
    pass


@dataclass
class UsfOmegaTemplateMeta(TemplateMeta):
    """Template metadata for USF Omega instruct model (non-reasoning).
    
    IMPORTANT: USF Omega requires jinja2 template backend for full feature support.
    The jinja2 template (chat_template.jinja2) handles ALL complex formatting:
    - Multi-iteration contents array (text → tool_calls → self_reflection cycles)
    - Parallel tool calls at different positions
    - Reasoning blocks (reasoning variant only)
    - Citations, self-reflection, structured output
    
    The prompt/system_prefix fields are minimal placeholders - actual formatting
    is done by the jinja2 template.
    
    Use usf_omega_reasoning for models with thinking capability.
    """
    template_type: str = 'usf_omega'
    # Jinja2 adds BOS - prefix not used but required by base class
    prefix: Prompt = field(default_factory=lambda: [])
    # Jinja2 handles formatting - minimal placeholder
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}}'])
    # System prefix for support_system=True (jinja2 handles actual formatting)
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['{{SYSTEM}}'])
    # Jinja2 adds BOS itself
    auto_add_bos: bool = False
    # Chat separator - MUST NOT be None for multi-turn support (support_multi_round=True)
    # Jinja2 handles actual formatting, this is just for metadata
    chat_sep: Optional[Prompt] = field(default_factory=lambda: [END_TOKEN + '\n'])
    # Suffix for STOP DETECTION ONLY (not appended to output)
    # Jinja2 output ends with: end_token (training) or message_start (inference)
    suffix: Prompt = field(default_factory=lambda: [END_TOKEN])
    template_cls: Type[Template] = UsfOmegaTemplate
    default_system: Optional[str] = 'You are a helpful assistant.'
    # Stop words for generation
    stop_words: List[Word] = field(default_factory=lambda: [
        EOS_TOKEN,
        END_TOKEN,
    ])
    # USF Omega agent template for function call parsing
    agent_template: Optional[str] = 'usf_omega'
    # Non-reasoning variant
    is_thinking: bool = False
    thinking_prefix: Optional[str] = None


@dataclass  
class UsfOmegaReasoningTemplateMeta(UsfOmegaTemplateMeta):
    """Template metadata for USF Omega reasoning model (with thinking).
    
    IMPORTANT: Requires jinja2 template backend (chat_template_reasoning.jinja2).
    
    Features handled by jinja2:
    - Reasoning blocks with effort levels (none/minimal/low/medium/high/max)
    - Multi-iteration contents array with reasoning at start
    - All instruct features (tool calls, self-reflection, citations, etc.)
    
    Supports reasoning_effort parameter and enable_reasoning flag.
    """
    template_type: str = 'usf_omega_reasoning'
    # Jinja2 handles reasoning format
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}}'])
    # Chat separator - MUST NOT be None for multi-turn support
    chat_sep: Optional[Prompt] = field(default_factory=lambda: [END_TOKEN + '\n'])
    # For stop detection only
    suffix: Prompt = field(default_factory=lambda: [END_TOKEN])
    template_cls: Type[Template] = UsfOmegaReasoningTemplate
    stop_words: List[Word] = field(default_factory=lambda: [
        EOS_TOKEN,
        END_TOKEN,
    ])
    # Reasoning variant
    is_thinking: bool = True
    thinking_prefix: str = REASONING_START
    # For hybrid models that may have non-thinking samples
    non_thinking_prefix: str = REASONING_START + REASONING_END + MESSAGE_START
    history_thinking_prefix: str = REASONING_END + MESSAGE_START


# ==================== REGISTER TEMPLATES ====================

# Register non-reasoning (instruct) variant
register_template(UsfOmegaTemplateMeta(
    LLMTemplateType.usf_omega,
    template_cls=UsfOmegaTemplate,
))

# Register reasoning variant  
register_template(UsfOmegaReasoningTemplateMeta(
    LLMTemplateType.usf_omega_reasoning,
    template_cls=UsfOmegaReasoningTemplate,
))
