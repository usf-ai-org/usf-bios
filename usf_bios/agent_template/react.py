# Copyright (c) US Inc. All rights reserved.
from typing import List, Optional, Union

from .base import BaseAgentTemplate


class ReactEnAgentTemplate(BaseAgentTemplate):

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names = []
        tool_descs = []
        for tool in tools:
            tool_desc = self._parse_tool(tool, 'en')
            tool_names.append(tool_desc.name_for_model)
            tool_descs.append(
                f'{tool_desc.name_for_model}: Call this tool to interact with the {tool_desc.name_for_human} API. '
                f'What is the {tool_desc.name_for_human} API useful for? {tool_desc.description_for_model} '
                f'Parameters: {tool_desc.parameters} {tool_desc.args_format}')

        return """Answer the following questions as best you can. You have access to the following tools:

""" + '\n\n'.join(tool_descs) + f"""

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{','.join(tool_names)}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""


class ReactZnAgentTemplate(BaseAgentTemplate):

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names = []
        tool_descs = []
        for tool in tools:
            tool_desc = self._parse_tool(tool, 'zh')
            tool_names.append(tool_desc.name_for_model)
            tool_descs.append(f'{tool_desc.name_for_model}: Call this tool to interact with  {tool_desc.name_for_human}  API.'
                              f'{tool_desc.name_for_human} What is it useful for? {tool_desc.description_for_model} '
                              f'Input parameters: {tool_desc.parameters} {tool_desc.args_format}')
        return """Answer the following questions as best you can. You have access to the following tools:

""" + '\n\n'.join(tool_descs) + f"""

Please follow the format below:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the tool to use, should be one of[{','.join(tool_names)}]
Action Input: the input to the tool
Observation: the result of the action
... (thisThought/Action/Action Input/Observationcan be repeatedN times)
Thought: I now at know the final answer
Final Answer: the final answer to the original input question

Now atbegin！
"""
