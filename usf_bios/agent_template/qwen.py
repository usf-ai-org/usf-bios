# Copyright (c) US Inc. All rights reserved.
from typing import List, Optional, Union

from .base import AgentKeyword, BaseAgentTemplate

keyword = AgentKeyword(
    action='✿FUNCTION✿:',
    action_input='✿ARGS✿:',
    observation='✿RESULT✿:',
)


class QwenEnAgentTemplate(BaseAgentTemplate):
    keyword = keyword

    def _get_tool_names_descs(self, tools):
        tool_names = []
        tool_descs = []
        for tool in tools:
            tool_desc = self._parse_tool(tool, 'en')
            tool_names.append(tool_desc.name_for_model)
            tool_descs.append(f'### {tool_desc.name_for_human}\n\n'
                              f'{tool_desc.name_for_model}: {tool_desc.description_for_model} '
                              f'Parameters: {tool_desc.parameters} {tool_desc.args_format}')
        return tool_names, tool_descs

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names, tool_descs = self._get_tool_names_descs(tools)
        system = system or ''
        return f"""{system}

# Tools

## You have access to the following tools:

""" + '\n\n'.join(tool_descs) + f"""

## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

✿FUNCTION✿: The tool to use, should be one of [{','.join(tool_names)}]
✿ARGS✿: The input of the tool
✿RESULT✿: Tool results
✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)"""  # noqa


class QwenZhAgentTemplate(BaseAgentTemplate):
    keyword = keyword

    def _get_tool_names_descs(self, tools):
        tool_names = []
        tool_descs = []
        for tool in tools:
            tool_desc = self._parse_tool(tool, 'zh')
            tool_names.append(tool_desc.name_for_model)
            tool_descs.append(f'### {tool_desc.name_for_human}\n\n'
                              f'{tool_desc.name_for_model}: {tool_desc.description_for_model} '
                              f'Input parameters: {tool_desc.parameters} {tool_desc.args_format}')
        return tool_names, tool_descs

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names, tool_descs = self._get_tool_names_descs(tools)
        system = system or ''
        return f"""{system}

# Tools

## You have access to the following tools:

""" + '\n\n'.join(tool_descs) + f"""

## You can insert the following commands zero, one, or multiple times in your reply to call tools:

✿FUNCTION✿: Tool name, must be one of [{','.join(tool_names)}]。
✿ARGS✿: Tool input
✿RESULT✿: Tool result
✿RETURN✿: Reply based on tool results, render images using ![](url)"""  # noqa


class QwenEnParallelAgentTemplate(QwenEnAgentTemplate):

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names, tool_descs = self._get_tool_names_descs(tools)
        system = system or ''
        return f"""{system}

# Tools

## You have access to the following tools:

""" + '\n\n'.join(tool_descs) + f"""

## Insert the following command in your reply when you need to call N tools in parallel:

✿FUNCTION✿: The name of tool 1, should be one of [{','.join(tool_names)}]
✿ARGS✿: The input of tool 1
✿FUNCTION✿: The name of tool 2
✿ARGS✿: The input of tool 2
...
✿FUNCTION✿: The name of tool N
✿ARGS✿: The input of tool N
✿RESULT✿: The result of tool 1
✿RESULT✿: The result of tool 2
...
✿RESULT✿: he result of tool N
✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)"""  # noqa


class QwenZhParallelAgentTemplate(QwenZhAgentTemplate):

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names, tool_descs = self._get_tool_names_descs(tools)
        system = system or ''
        return f"""{system}

# Tools

## You have access to the following tools:

""" + '\n\n'.join(tool_descs) + f"""

## You can insert the following commands in your reply to call N tools in parallel:

✿FUNCTION✿: Name of tool 1, must be one of [{','.join(tool_names)}]
✿ARGS✿: Input for tool 1
✿FUNCTION✿: Name of tool 2
✿ARGS✿: Input for tool 2
...
✿FUNCTION✿: Name of tool N
✿ARGS✿: Input for tool N
✿RESULT✿: Result of tool 1
✿RESULT✿: Result of tool 2
...
✿RESULT✿: Result of tool N
✿RETURN✿: Reply based on tool results, render images using ![](url)"""  # noqa
