# Copyright (c) Ultrasafe AI. All rights reserved.
"""
Comprehensive tests for USF Omega Agent Template.

Tests all possible combinations of:
- Simple format (content, tool_calls, self_reflection)
- Contents array format with type field
- Multi-iteration cycles (text → tool_calls → self_reflection)
- Parallel tool calls at different positions
- Reasoning blocks (optional, once, first)
- Edge cases and error handling

Total: 90+ test cases covering all variations.
"""
import pytest
import json
from usf_bios.agent_template.usf_omega import (
    UsfOmegaAgentTemplate,
    ParsedContentItem,
    ParsedAssistantResponse,
    # Tokens
    START_TOKEN, END_TOKEN, BOS_TOKEN, EOS_TOKEN,
    MESSAGE_START, MESSAGE_END,
    CONSTRAIN_START, CONSTRAIN_END,
    REASONING_START, REASONING_END,
    SELF_REFLECTION_START, SELF_REFLECTION_END,
    FUNCTIONS_START, FUNCTIONS_END,
    INVOKE_START, INVOKE_END,
    PARAMETER_START, PARAMETER_END,
    FUNCTION_RESULTS_START, FUNCTION_RESULTS_END,
    FUNCTION_RESPONSE_START, FUNCTION_RESPONSE_END,
    CITATIONS_START, CITATIONS_END,
)


@pytest.fixture
def agent():
    """Create agent template instance."""
    return UsfOmegaAgentTemplate()


# ==============================================================================
# SECTION 1: TOKEN CONSTANTS TESTS (5 tests)
# ==============================================================================

class TestTokenConstants:
    """Test that all token constants are correctly defined."""
    
    def test_message_boundary_tokens(self):
        """Test message boundary tokens."""
        assert START_TOKEN == '<|:@:|start|:@:|>'
        assert END_TOKEN == '<|:@:|end|:@:|>'
        assert BOS_TOKEN == '<|:@:|startoftext|:@:|>'
        assert EOS_TOKEN == '<|:@:|endoftext|:@:|>'
    
    def test_message_content_tokens(self):
        """Test message content tokens."""
        assert MESSAGE_START == '<|:@:|message_start|:@:|>'
        assert MESSAGE_END == '<|:@:|message_end|:@:|>'
    
    def test_reasoning_tokens(self):
        """Test reasoning tokens."""
        assert REASONING_START == '<|:@:|reasoning_start|:@:|>'
        assert REASONING_END == '<|:@:|reasoning_end|:@:|>'
    
    def test_self_reflection_tokens(self):
        """Test self-reflection tokens."""
        assert SELF_REFLECTION_START == '<|:@:|self_reflection_start|:@:|>'
        assert SELF_REFLECTION_END == '<|:@:|self_reflection_end|:@:|>'
    
    def test_function_tokens(self):
        """Test function call tokens."""
        assert FUNCTIONS_START == '<|:@:|functions_start|:@:|>'
        assert FUNCTIONS_END == '<|:@:|functions_end|:@:|>'
        assert INVOKE_START == '<||invoke_start to='
        assert INVOKE_END == '<||invoke_end||>'
        assert PARAMETER_START == '<||parameter_start name='
        assert PARAMETER_END == '<||parameter_end||>'


# ==============================================================================
# SECTION 2: SIMPLE FORMAT PARSING TESTS (15 tests)
# ==============================================================================

class TestSimpleFormatParsing:
    """Test parsing of simple assistant message format."""
    
    def test_simple_text_only(self, agent):
        """Test: text only."""
        text = f"{MESSAGE_START}Hello, world!{MESSAGE_END}"
        result = agent.parse_assistant_response(text)
        
        assert len(result.message_contents) == 1
        assert result.message_contents[0] == "Hello, world!"
        assert not result.has_function_calls
        assert not result.has_reasoning
        assert not result.has_self_reflection
    
    def test_simple_text_with_self_reflection(self, agent):
        """Test: text + self_reflection."""
        text = f"{MESSAGE_START}Answer{MESSAGE_END}{SELF_REFLECTION_START}Verified.{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert result.message_contents[0] == "Answer"
        assert result.self_reflections[0] == "Verified."
        assert result.has_self_reflection
    
    def test_simple_text_with_single_tool_call(self, agent):
        """Test: text + single tool_call."""
        text = f"""{MESSAGE_START}Let me search.{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.search||><||parameter_start name=query||>weather{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}"""
        result = agent.parse_assistant_response(text)
        
        assert result.message_contents[0] == "Let me search."
        assert result.has_function_calls
        assert len(result.function_calls) == 1
        assert result.function_calls[0]['function']['name'] == 'search'
    
    def test_simple_text_with_parallel_tool_calls(self, agent):
        """Test: text + parallel tool_calls (2 calls in one block)."""
        text = f"""{MESSAGE_START}Searching...{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.search||><||parameter_start name=q||>a{PARAMETER_END}{INVOKE_END}<||invoke_start to=function.lookup||><||parameter_start name=id||>123{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}"""
        result = agent.parse_assistant_response(text)
        
        assert result.has_function_calls
        assert len(result.function_calls) == 2
        assert result.function_calls[0]['function']['name'] == 'search'
        assert result.function_calls[1]['function']['name'] == 'lookup'
    
    def test_simple_text_with_tool_call_and_reflection(self, agent):
        """Test: text + tool_call + self_reflection."""
        text = f"""{MESSAGE_START}Answer{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.get_time||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert result.message_contents[0] == "Answer"
        assert result.has_function_calls
        assert result.has_self_reflection
    
    def test_simple_reasoning_with_text(self, agent):
        """Test: reasoning + text."""
        text = f"{REASONING_START}Let me think...{REASONING_END}{MESSAGE_START}Here's my answer.{MESSAGE_END}"
        result = agent.parse_assistant_response(text)
        
        assert result.has_reasoning
        assert result.reasoning == "Let me think..."
        assert result.message_contents[0] == "Here's my answer."
    
    def test_simple_reasoning_text_tool_reflection(self, agent):
        """Test: reasoning + text + tool_call + self_reflection."""
        text = f"""{REASONING_START}Thinking...{REASONING_END}{MESSAGE_START}Answer{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.calc||><||parameter_start name=x||>5{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}Verified{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert result.has_reasoning
        assert result.reasoning == "Thinking..."
        assert result.message_contents[0] == "Answer"
        assert result.has_function_calls
        assert result.has_self_reflection
    
    def test_simple_empty_text(self, agent):
        """Test: empty text content."""
        text = f"{MESSAGE_START}{MESSAGE_END}"
        result = agent.parse_assistant_response(text)
        
        assert len(result.message_contents) == 1
        assert result.message_contents[0] == ""
    
    def test_simple_text_with_json_parameter(self, agent):
        """Test: tool_call with JSON parameter."""
        json_arg = json.dumps({"key": "value", "nested": {"a": 1}})
        text = f"""{MESSAGE_START}Calling{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.process||><||parameter_start name=data||>{json_arg}{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}"""
        result = agent.parse_assistant_response(text)
        
        assert result.function_calls[0]['function']['arguments']['data'] == {"key": "value", "nested": {"a": 1}}
    
    def test_simple_text_with_array_parameter(self, agent):
        """Test: tool_call with array parameter."""
        arr_arg = json.dumps([1, 2, 3, "four"])
        text = f"""{MESSAGE_START}Processing{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.sum||><||parameter_start name=numbers||>{arr_arg}{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}"""
        result = agent.parse_assistant_response(text)
        
        assert result.function_calls[0]['function']['arguments']['numbers'] == [1, 2, 3, "four"]
    
    def test_simple_text_with_multiple_parameters(self, agent):
        """Test: tool_call with multiple parameters."""
        text = f"""{MESSAGE_START}Call{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.api||><||parameter_start name=url||>https://api.com{PARAMETER_END}<||parameter_start name=method||>POST{PARAMETER_END}<||parameter_start name=timeout||>30{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}"""
        result = agent.parse_assistant_response(text)
        
        args = result.function_calls[0]['function']['arguments']
        assert args['url'] == 'https://api.com'
        assert args['method'] == 'POST'
        assert args['timeout'] == 30
    
    def test_simple_text_with_no_parameter_tool_call(self, agent):
        """Test: tool_call with no parameters."""
        text = f"""{MESSAGE_START}Getting time{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.get_time||>{INVOKE_END}{FUNCTIONS_END}"""
        result = agent.parse_assistant_response(text)
        
        assert result.function_calls[0]['function']['name'] == 'get_time'
        assert result.function_calls[0]['function']['arguments'] == {}
    
    def test_simple_constrain_block(self, agent):
        """Test: constrain block parsing."""
        text = f"{CONSTRAIN_START}response.format={{type:json, strict:true}}{CONSTRAIN_END}{MESSAGE_START}{{\"answer\": 42}}{MESSAGE_END}"
        result = agent.parse_assistant_response(text)
        
        assert result.constrain == "response.format={type:json, strict:true}"
    
    def test_simple_constrain_with_reasoning_effort(self, agent):
        """Test: constrain block with reasoning effort."""
        text = f"{CONSTRAIN_START}reasoning={{effort: high}}{CONSTRAIN_END}{REASONING_START}Deep thinking...{REASONING_END}{MESSAGE_START}Answer{MESSAGE_END}"
        result = agent.parse_assistant_response(text)
        
        assert "reasoning={effort: high}" in result.constrain
        assert result.has_reasoning


# ==============================================================================
# SECTION 3: CONTENTS ARRAY - SINGLE ITERATION TESTS (15 tests)
# ==============================================================================

class TestContentsArraySingleIteration:
    """Test contents array format with single iteration (text → [tool_calls] → self_reflection)."""
    
    def test_contents_text_self_reflection(self, agent):
        """Test: [text, self_reflection]."""
        text = f"{MESSAGE_START}Answer{MESSAGE_END}{SELF_REFLECTION_START}Verified{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert len(result.contents) == 2
        assert result.contents[0].type == 'text'
        assert result.contents[0].content == 'Answer'
        assert result.contents[1].type == 'self_reflection'
        assert result.contents[1].content == 'Verified'
    
    def test_contents_text_tool_calls_self_reflection(self, agent):
        """Test: [text, tool_calls, self_reflection]."""
        text = f"""{MESSAGE_START}Let me check{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.search||><||parameter_start name=q||>test{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert len(result.contents) == 3
        assert result.contents[0].type == 'text'
        assert result.contents[1].type == 'tool_calls'
        assert result.contents[2].type == 'self_reflection'
    
    def test_contents_reasoning_text_self_reflection(self, agent):
        """Test: [reasoning, text, self_reflection]."""
        text = f"{REASONING_START}Thinking{REASONING_END}{MESSAGE_START}Answer{MESSAGE_END}{SELF_REFLECTION_START}Verified{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert len(result.contents) == 3
        assert result.contents[0].type == 'reasoning'
        assert result.contents[0].content == 'Thinking'
        assert result.contents[1].type == 'text'
        assert result.contents[2].type == 'self_reflection'
    
    def test_contents_reasoning_text_tool_calls_self_reflection(self, agent):
        """Test: [reasoning, text, tool_calls, self_reflection]."""
        text = f"""{REASONING_START}Let me think{REASONING_END}{MESSAGE_START}I'll search{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.search||><||parameter_start name=q||>query{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}Confirmed{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert len(result.contents) == 4
        assert result.contents[0].type == 'reasoning'
        assert result.contents[1].type == 'text'
        assert result.contents[2].type == 'tool_calls'
        assert result.contents[3].type == 'self_reflection'
    
    def test_contents_text_parallel_tool_calls_self_reflection(self, agent):
        """Test: [text, parallel tool_calls (3 calls), self_reflection]."""
        text = f"""{MESSAGE_START}Calling APIs{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.api1||>{INVOKE_END}<||invoke_start to=function.api2||>{INVOKE_END}<||invoke_start to=function.api3||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}All done{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert len(result.contents) == 3
        assert result.contents[1].type == 'tool_calls'
        assert len(result.contents[1].content) == 3  # 3 parallel calls
        assert result.contents[1].content[0]['function']['name'] == 'api1'
        assert result.contents[1].content[1]['function']['name'] == 'api2'
        assert result.contents[1].content[2]['function']['name'] == 'api3'
    
    def test_contents_single_iteration_to_dict(self, agent):
        """Test converting parsed contents to dict format."""
        text = f"{MESSAGE_START}Answer{MESSAGE_END}{SELF_REFLECTION_START}Verified{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        contents_array = result.to_contents_array()
        assert len(contents_array) == 2
        assert contents_array[0] == {'type': 'text', 'text': 'Answer'}
        assert contents_array[1] == {'type': 'self_reflection', 'self_reflection': 'Verified'}
    
    def test_contents_single_iteration_with_all_elements(self, agent):
        """Test: [reasoning, text, tool_calls, self_reflection] with full content."""
        text = f"""{REASONING_START}Step 1: Understand the problem
Step 2: Search for information
Step 3: Provide answer{REASONING_END}{MESSAGE_START}Based on my analysis, I need to search for more information.{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.web_search||><||parameter_start name=query||>latest AI news{PARAMETER_END}<||parameter_start name=limit||>5{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}The search query is relevant and the limit is appropriate.{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert result.contents[0].type == 'reasoning'
        assert 'Step 1' in result.contents[0].content
        assert result.contents[1].type == 'text'
        assert 'Based on my analysis' in result.contents[1].content
        assert result.contents[2].type == 'tool_calls'
        assert result.contents[2].content[0]['function']['arguments']['query'] == 'latest AI news'
        assert result.contents[2].content[0]['function']['arguments']['limit'] == 5
        assert result.contents[3].type == 'self_reflection'
    
    def test_contents_text_only_long_content(self, agent):
        """Test: long text content."""
        long_text = "A" * 10000
        text = f"{MESSAGE_START}{long_text}{MESSAGE_END}{SELF_REFLECTION_START}OK{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert len(result.contents[0].content) == 10000
    
    def test_contents_text_with_special_characters(self, agent):
        """Test: text with special characters."""
        special = "Hello <world> & \"quotes\" 'apostrophe' |pipe| [brackets]"
        text = f"{MESSAGE_START}{special}{MESSAGE_END}{SELF_REFLECTION_START}OK{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert result.contents[0].content == special
    
    def test_contents_text_with_unicode(self, agent):
        """Test: text with unicode characters."""
        unicode_text = "Hello 世界 🌍 مرحبا שלום"
        text = f"{MESSAGE_START}{unicode_text}{MESSAGE_END}{SELF_REFLECTION_START}✓{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert result.contents[0].content == unicode_text
        assert result.contents[1].content == "✓"
    
    def test_contents_text_with_newlines(self, agent):
        """Test: text with newlines and formatting."""
        formatted = "Line 1\nLine 2\n\nParagraph 2\n\t- Item 1\n\t- Item 2"
        text = f"{MESSAGE_START}{formatted}{MESSAGE_END}{SELF_REFLECTION_START}Formatted correctly{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert "Line 1\nLine 2" in result.contents[0].content
    
    def test_contents_tool_call_with_complex_json(self, agent):
        """Test: tool_call with deeply nested JSON."""
        complex_json = json.dumps({
            "level1": {
                "level2": {
                    "level3": {
                        "array": [1, 2, {"nested": True}],
                        "string": "value"
                    }
                }
            }
        })
        text = f"""{MESSAGE_START}Processing{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.process||><||parameter_start name=data||>{complex_json}{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}OK{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        data = result.contents[1].content[0]['function']['arguments']['data']
        assert data['level1']['level2']['level3']['array'][2]['nested'] is True
    
    def test_contents_position_preserved(self, agent):
        """Test that position is correctly tracked."""
        text = f"{REASONING_START}R{REASONING_END}{MESSAGE_START}T{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}S{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert result.contents[0].position == 0
        assert result.contents[1].position == 1
        assert result.contents[2].position == 2
        assert result.contents[3].position == 3
    
    def test_contents_to_simple_format(self, agent):
        """Test converting to simple format."""
        text = f"{MESSAGE_START}Answer{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        simple = result.to_simple_format()
        assert simple['role'] == 'assistant'
        assert simple['content'] == 'Answer'
        assert 'tool_calls' in simple
        assert simple['self_reflection'] == 'Done'


# ==============================================================================
# SECTION 4: CONTENTS ARRAY - MULTI-ITERATION TESTS (20 tests)
# ==============================================================================

class TestContentsArrayMultiIteration:
    """Test contents array format with multiple iterations."""
    
    def test_two_iterations_text_reflection_only(self, agent):
        """Test: [text, self_reflection, text, self_reflection]."""
        text = f"{MESSAGE_START}First answer{MESSAGE_END}{SELF_REFLECTION_START}Check 1{SELF_REFLECTION_END}{MESSAGE_START}Revised answer{MESSAGE_END}{SELF_REFLECTION_START}Check 2{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert len(result.contents) == 4
        assert result.contents[0].type == 'text'
        assert result.contents[0].content == 'First answer'
        assert result.contents[1].type == 'self_reflection'
        assert result.contents[2].type == 'text'
        assert result.contents[2].content == 'Revised answer'
        assert result.contents[3].type == 'self_reflection'
        assert result.has_multiple_iterations
    
    def test_two_iterations_with_tool_calls(self, agent):
        """Test: [text, tool_calls, self_reflection, text, self_reflection]."""
        text = f"""{MESSAGE_START}First{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.search||><||parameter_start name=q||>query1{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}Reflect1{SELF_REFLECTION_END}{MESSAGE_START}Second{MESSAGE_END}{SELF_REFLECTION_START}Reflect2{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert len(result.contents) == 5
        assert result.contents[1].type == 'tool_calls'
        assert len(result.tool_call_blocks) == 1  # Only one tool_calls block
    
    def test_two_iterations_both_with_tool_calls(self, agent):
        """Test: [text, tool_calls, self_reflection, text, tool_calls, self_reflection]."""
        text = f"""{MESSAGE_START}First{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.api1||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}Second{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.api2||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert len(result.contents) == 6
        assert len(result.tool_call_blocks) == 2
        assert result.tool_call_blocks[0][0]['function']['name'] == 'api1'
        assert result.tool_call_blocks[1][0]['function']['name'] == 'api2'
    
    def test_three_iterations(self, agent):
        """Test: three complete iterations."""
        text = f"""{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert len(result.message_contents) == 3
        assert len(result.self_reflections) == 3
        assert result.get_iteration_count(text) == 3
    
    def test_three_iterations_with_reasoning(self, agent):
        """Test: [reasoning, text, self_reflection, text, self_reflection, text, self_reflection]."""
        text = f"""{REASONING_START}Think{REASONING_END}{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert result.contents[0].type == 'reasoning'
        assert len(result.message_contents) == 3
        assert len(result.self_reflections) == 3
    
    def test_three_iterations_all_with_tool_calls(self, agent):
        """Test: three iterations all with tool_calls."""
        text = f"""{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f1||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f2||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f3||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert len(result.tool_call_blocks) == 3
        assert result.tool_call_blocks[0][0]['function']['name'] == 'f1'
        assert result.tool_call_blocks[1][0]['function']['name'] == 'f2'
        assert result.tool_call_blocks[2][0]['function']['name'] == 'f3'
    
    def test_mixed_iterations_some_with_tool_calls(self, agent):
        """Test: [text, tool_calls, self_reflection, text, self_reflection, text, tool_calls, self_reflection]."""
        text = f"""{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f1||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f3||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        # Iterations 1 and 3 have tool_calls, iteration 2 doesn't
        assert len(result.tool_call_blocks) == 2
    
    def test_parallel_tool_calls_in_multiple_iterations(self, agent):
        """Test: parallel tool_calls in multiple iterations."""
        text = f"""{MESSAGE_START}First{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.a||>{INVOKE_END}<||invoke_start to=function.b||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}Second{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.c||>{INVOKE_END}<||invoke_start to=function.d||>{INVOKE_END}<||invoke_start to=function.e||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        assert len(result.tool_call_blocks) == 2
        assert len(result.tool_call_blocks[0]) == 2  # a, b
        assert len(result.tool_call_blocks[1]) == 3  # c, d, e
    
    def test_extract_iteration_0(self, agent):
        """Test extracting first iteration."""
        text = f"""{MESSAGE_START}First{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f1||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}Second{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"""
        
        iteration_0 = agent.extract_iteration(text, 0)
        assert iteration_0['text'] == 'First'
        assert iteration_0['tool_calls'] is not None
        assert iteration_0['self_reflection'] == 'R1'
    
    def test_extract_iteration_1(self, agent):
        """Test extracting second iteration."""
        text = f"""{MESSAGE_START}First{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}Second{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f2||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"""
        
        iteration_1 = agent.extract_iteration(text, 1)
        assert iteration_1['text'] == 'Second'
        assert iteration_1['tool_calls'] is not None
        assert iteration_1['self_reflection'] == 'R2'
    
    def test_extract_iteration_out_of_range(self, agent):
        """Test extracting non-existent iteration."""
        text = f"{MESSAGE_START}Only{MESSAGE_END}{SELF_REFLECTION_START}One{SELF_REFLECTION_END}"
        
        iteration_5 = agent.extract_iteration(text, 5)
        assert iteration_5['text'] is None
        assert iteration_5['tool_calls'] is None
        assert iteration_5['self_reflection'] is None
    
    def test_get_all_tool_calls_flattened(self, agent):
        """Test getting all tool calls as flat list."""
        text = f"""{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.a||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.b||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        # Flattened function_calls
        assert len(result.function_calls) == 2
        assert result.function_calls[0]['function']['name'] == 'a'
        assert result.function_calls[1]['function']['name'] == 'b'
    
    def test_get_all_tool_calls_grouped(self, agent):
        """Test getting tool calls grouped by position."""
        text = f"""{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.a||>{INVOKE_END}<||invoke_start to=function.b||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.c||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"""
        
        grouped = agent.get_all_tool_calls(text)
        assert len(grouped) == 2
        assert len(grouped[0]) == 2  # a, b
        assert len(grouped[1]) == 1  # c
        assert grouped[0][0].name == 'a'
        assert grouped[0][1].name == 'b'
        assert grouped[1][0].name == 'c'
    
    def test_iteration_count(self, agent):
        """Test counting iterations."""
        text = f"""{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"""
        
        count = agent.get_iteration_count(text)
        assert count == 3
    
    def test_has_multiple_iterations_true(self, agent):
        """Test has_multiple_iterations property when true."""
        text = f"{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert result.has_multiple_iterations is True
    
    def test_has_multiple_iterations_false(self, agent):
        """Test has_multiple_iterations property when false."""
        text = f"{MESSAGE_START}Only one{MESSAGE_END}{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert result.has_multiple_iterations is False
    
    def test_five_iterations_stress(self, agent):
        """Test: five iterations (stress test)."""
        parts = []
        for i in range(5):
            parts.append(f"{MESSAGE_START}Text {i+1}{MESSAGE_END}")
            parts.append(f"{FUNCTIONS_START}<||invoke_start to=function.func{i+1}||>{INVOKE_END}{FUNCTIONS_END}")
            parts.append(f"{SELF_REFLECTION_START}Reflection {i+1}{SELF_REFLECTION_END}")
        text = ''.join(parts)
        result = agent.parse_assistant_response(text)
        
        assert len(result.message_contents) == 5
        assert len(result.self_reflections) == 5
        assert len(result.tool_call_blocks) == 5
    
    def test_reasoning_only_at_start(self, agent):
        """Test reasoning appears only at start, not in iterations."""
        text = f"""{REASONING_START}Initial thinking{REASONING_END}{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        # Only one reasoning block
        assert result.reasoning == "Initial thinking"
        # First content should be reasoning
        assert result.contents[0].type == 'reasoning'
        # Count reasoning in contents
        reasoning_count = sum(1 for c in result.contents if c.type == 'reasoning')
        assert reasoning_count == 1
    
    def test_multi_iteration_to_contents_array(self, agent):
        """Test converting multi-iteration to contents array."""
        text = f"""{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"""
        result = agent.parse_assistant_response(text)
        
        contents = result.to_contents_array()
        assert len(contents) == 5
        assert contents[0]['type'] == 'text'
        assert contents[1]['type'] == 'tool_calls'
        assert contents[2]['type'] == 'self_reflection'
        assert contents[3]['type'] == 'text'
        assert contents[4]['type'] == 'self_reflection'


# ==============================================================================
# SECTION 5: FORMATTING TESTS (15 tests)
# ==============================================================================

class TestFormatting:
    """Test formatting methods."""
    
    def test_format_message_block(self, agent):
        """Test formatting message block."""
        result = agent.format_message_block("Hello")
        assert result == f"{MESSAGE_START}Hello{MESSAGE_END}"
    
    def test_format_reasoning_block(self, agent):
        """Test formatting reasoning block."""
        result = agent.format_reasoning_block("Thinking...")
        assert result == f"{REASONING_START}Thinking...{REASONING_END}"
    
    def test_format_self_reflection_block(self, agent):
        """Test formatting self-reflection block."""
        result = agent.format_self_reflection_block("Verified")
        assert result == f"{SELF_REFLECTION_START}Verified{SELF_REFLECTION_END}"
    
    def test_format_citation(self, agent):
        """Test formatting citation."""
        result = agent.format_citation(["s001", "s002"], "cited text")
        assert CITATIONS_START in result
        assert '"s001"' in result
        assert '"s002"' in result
        assert "cited text" in result
        assert CITATIONS_END in result
    
    def test_format_constrain_block_json(self, agent):
        """Test formatting constrain block with JSON."""
        result = agent.format_constrain_block(response_format={'strict': True})
        assert "response.format={type:json, strict:true}" in result
    
    def test_format_constrain_block_reasoning(self, agent):
        """Test formatting constrain block with reasoning."""
        result = agent.format_constrain_block(reasoning_effort='high')
        assert "reasoning={effort: high}" in result
    
    def test_format_constrain_block_both(self, agent):
        """Test formatting constrain block with both."""
        result = agent.format_constrain_block(
            response_format={'strict': False},
            reasoning_effort='medium'
        )
        assert "response.format={type:json, strict:false}" in result
        assert "reasoning={effort: medium}" in result
    
    def test_format_tool_call_simple(self, agent):
        """Test formatting single tool call."""
        result = agent._format_tool_call("search", {"query": "test"})
        assert "function.search" in result
        assert "query" in result
        assert "test" in result
    
    def test_format_tool_call_no_params(self, agent):
        """Test formatting tool call without params."""
        result = agent._format_tool_call("get_time", {})
        assert "function.get_time" in result
        assert INVOKE_END in result
    
    def test_format_tool_call_json_param(self, agent):
        """Test formatting tool call with JSON param."""
        result = agent._format_tool_call("process", {"data": {"key": "value"}})
        assert '{"key": "value"}' in result or "{'key': 'value'}" in result
    
    def test_format_contents_array_simple(self, agent):
        """Test formatting simple contents array."""
        contents = [
            {"type": "text", "text": "Hello"},
            {"type": "self_reflection", "self_reflection": "Done"}
        ]
        result = agent.format_contents_array(contents)
        
        assert f"{MESSAGE_START}Hello{MESSAGE_END}" in result
        assert f"{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}" in result
    
    def test_format_contents_array_with_reasoning(self, agent):
        """Test formatting contents array with reasoning."""
        contents = [
            {"type": "reasoning", "reasoning": "Thinking"},
            {"type": "text", "text": "Answer"},
            {"type": "self_reflection", "self_reflection": "OK"}
        ]
        result = agent.format_contents_array(contents, include_reasoning=True)
        
        assert f"{REASONING_START}Thinking{REASONING_END}" in result
        assert f"{MESSAGE_START}Answer{MESSAGE_END}" in result
    
    def test_format_contents_array_skip_reasoning(self, agent):
        """Test formatting contents array skipping reasoning."""
        contents = [
            {"type": "reasoning", "reasoning": "Thinking"},
            {"type": "text", "text": "Answer"},
            {"type": "self_reflection", "self_reflection": "OK"}
        ]
        result = agent.format_contents_array(contents, include_reasoning=False)
        
        assert REASONING_START not in result
        assert f"{MESSAGE_START}Answer{MESSAGE_END}" in result
    
    def test_format_contents_array_with_tool_calls(self, agent):
        """Test formatting contents array with tool_calls."""
        contents = [
            {"type": "text", "text": "Calling"},
            {"type": "tool_calls", "tool_calls": [
                {"function": {"name": "search", "arguments": {"q": "test"}}}
            ]},
            {"type": "self_reflection", "self_reflection": "Done"}
        ]
        result = agent.format_contents_array(contents)
        
        assert FUNCTIONS_START in result
        assert "function.search" in result
        assert FUNCTIONS_END in result
    
    def test_format_tool_calls_list(self, agent):
        """Test formatting list of tool calls."""
        tool_calls = [
            {"function": {"name": "api1", "arguments": {"x": 1}}},
            {"function": {"name": "api2", "arguments": {"y": 2}}}
        ]
        result = agent._format_tool_calls_list(tool_calls)
        
        assert "function.api1" in result
        assert "function.api2" in result


# ==============================================================================
# SECTION 6: TOOL RESPONSE PARSING TESTS (10 tests)
# ==============================================================================

class TestToolResponseParsing:
    """Test parsing of tool responses."""
    
    def test_parse_single_tool_response(self, agent):
        """Test parsing single tool response."""
        text = f"{FUNCTION_RESULTS_START}<||function_response_start to=function.search id=\"r1\"||>Result data<||function_response_end||>{FUNCTION_RESULTS_END}"
        results = agent.parse_tool_responses_from_text(text)
        
        assert len(results) == 1
        assert results[0]['name'] == 'search'
        assert results[0]['id'] == 'r1'
        assert results[0]['content'] == 'Result data'
    
    def test_parse_multiple_tool_responses(self, agent):
        """Test parsing multiple tool responses."""
        text = f"""{FUNCTION_RESULTS_START}<||function_response_start to=function.api1 id="r1"||>Result 1<||function_response_end||><||function_response_start to=function.api2 id="r2"||>Result 2<||function_response_end||>{FUNCTION_RESULTS_END}"""
        results = agent.parse_tool_responses_from_text(text)
        
        assert len(results) == 2
        assert results[0]['name'] == 'api1'
        assert results[1]['name'] == 'api2'
    
    def test_parse_tool_response_without_id(self, agent):
        """Test parsing tool response without ID."""
        text = f"{FUNCTION_RESULTS_START}<||function_response_start to=function.test||>Data<||function_response_end||>{FUNCTION_RESULTS_END}"
        results = agent.parse_tool_responses_from_text(text)
        
        assert results[0]['id'] is None
        assert results[0]['name'] == 'test'
    
    def test_parse_tool_response_with_json_content(self, agent):
        """Test parsing tool response with JSON content."""
        json_content = json.dumps({"status": "ok", "data": [1, 2, 3]})
        text = f'{FUNCTION_RESULTS_START}<||function_response_start to=function.api id="r1"||>{json_content}<||function_response_end||>{FUNCTION_RESULTS_END}'
        results = agent.parse_tool_responses_from_text(text)
        
        assert json_content.strip() == results[0]['content']
    
    def test_format_tool_response(self, agent):
        """Test formatting tool response."""
        result = agent._format_tool_response("search", "Result data", "call_1")
        
        assert "function.search" in result
        assert 'id="call_1"' in result
        assert "Result data" in result
    
    def test_format_tool_responses(self, agent):
        """Test formatting multiple tool responses."""
        assistant_content = f"{MESSAGE_START}Called{MESSAGE_END}"
        tool_messages = [
            {"name": "api1", "content": "Result 1", "id": "r1"},
            {"name": "api2", "content": "Result 2", "id": "r2"}
        ]
        
        new_assistant, tool_content = agent._format_tool_responses(assistant_content, tool_messages)
        
        assert new_assistant == assistant_content
        assert FUNCTION_RESULTS_START in tool_content[0]
        assert "api1" in tool_content[0]
        assert "api2" in tool_content[0]
    
    def test_has_tool_results_true(self, agent):
        """Test has_tool_results when true."""
        text = f"{FUNCTION_RESULTS_START}..content..{FUNCTION_RESULTS_END}"
        assert agent.has_tool_results(text) is True
    
    def test_has_tool_results_false(self, agent):
        """Test has_tool_results when false."""
        text = f"{MESSAGE_START}No tool results{MESSAGE_END}"
        assert agent.has_tool_results(text) is False
    
    def test_parse_tool_response_multiline(self, agent):
        """Test parsing tool response with multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        text = f'{FUNCTION_RESULTS_START}<||function_response_start to=function.read id="r1"||>{content}<||function_response_end||>{FUNCTION_RESULTS_END}'
        results = agent.parse_tool_responses_from_text(text)
        
        assert "Line 1\nLine 2\nLine 3" in results[0]['content']
    
    def test_parse_tool_response_special_chars(self, agent):
        """Test parsing tool response with special characters."""
        content = '<html><body>"Hello & World"</body></html>'
        text = f'{FUNCTION_RESULTS_START}<||function_response_start to=function.scrape id="r1"||>{content}<||function_response_end||>{FUNCTION_RESULTS_END}'
        results = agent.parse_tool_responses_from_text(text)
        
        assert '"Hello & World"' in results[0]['content']


# ==============================================================================
# SECTION 7: CITATION PARSING TESTS (5 tests)
# ==============================================================================

class TestCitationParsing:
    """Test citation parsing."""
    
    def test_parse_single_citation(self, agent):
        """Test parsing single citation."""
        text = f'{CITATIONS_START}["s001"]||>cited text{CITATIONS_END}'
        citations = agent.parse_citations(text)
        
        assert len(citations) == 1
        assert citations[0]['refs'] == ['s001']
        assert citations[0]['content'] == 'cited text'
    
    def test_parse_multiple_refs(self, agent):
        """Test parsing citation with multiple refs."""
        text = f'{CITATIONS_START}["s001","s002","s003"]||>cited text{CITATIONS_END}'
        citations = agent.parse_citations(text)
        
        assert citations[0]['refs'] == ['s001', 's002', 's003']
    
    def test_parse_multiple_citations(self, agent):
        """Test parsing multiple citations in text."""
        text = f'{CITATIONS_START}["s001"]||>first{CITATIONS_END} and {CITATIONS_START}["s002"]||>second{CITATIONS_END}'
        citations = agent.parse_citations(text)
        
        assert len(citations) == 2
    
    def test_has_citations_true(self, agent):
        """Test has_citations when true."""
        text = f'{CITATIONS_START}["s001"]||>text{CITATIONS_END}'
        assert agent.has_citations(text) is True
    
    def test_has_citations_false(self, agent):
        """Test has_citations when false."""
        text = f"{MESSAGE_START}No citations{MESSAGE_END}"
        assert agent.has_citations(text) is False


# ==============================================================================
# SECTION 8: CHECKING METHODS TESTS (10 tests)
# ==============================================================================

class TestCheckingMethods:
    """Test checking/detection methods."""
    
    def test_has_function_calls_true(self, agent):
        """Test has_function_calls when true."""
        text = f"{FUNCTIONS_START}<||invoke_start to=function.test||>{INVOKE_END}{FUNCTIONS_END}"
        assert agent.has_function_calls(text) is True
    
    def test_has_function_calls_false(self, agent):
        """Test has_function_calls when false."""
        text = f"{MESSAGE_START}No function calls{MESSAGE_END}"
        assert agent.has_function_calls(text) is False
    
    def test_has_reasoning_true(self, agent):
        """Test has_reasoning when true."""
        text = f"{REASONING_START}Thinking{REASONING_END}"
        assert agent.has_reasoning(text) is True
    
    def test_has_reasoning_false(self, agent):
        """Test has_reasoning when false."""
        text = f"{MESSAGE_START}No reasoning{MESSAGE_END}"
        assert agent.has_reasoning(text) is False
    
    def test_has_self_reflection_true(self, agent):
        """Test has_self_reflection when true."""
        text = f"{SELF_REFLECTION_START}Verifying{SELF_REFLECTION_END}"
        assert agent.has_self_reflection(text) is True
    
    def test_has_self_reflection_false(self, agent):
        """Test has_self_reflection when false."""
        text = f"{MESSAGE_START}No reflection{MESSAGE_END}"
        assert agent.has_self_reflection(text) is False
    
    def test_has_constrain_true(self, agent):
        """Test has_constrain when true."""
        text = f"{CONSTRAIN_START}config{CONSTRAIN_END}"
        assert agent.has_constrain(text) is True
    
    def test_has_constrain_false(self, agent):
        """Test has_constrain when false."""
        text = f"{MESSAGE_START}No constrain{MESSAGE_END}"
        assert agent.has_constrain(text) is False
    
    def test_parsed_response_has_function_calls(self, agent):
        """Test ParsedAssistantResponse.has_function_calls."""
        text = f"{MESSAGE_START}Call{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||>{INVOKE_END}{FUNCTIONS_END}"
        result = agent.parse_assistant_response(text)
        assert result.has_function_calls is True
    
    def test_parsed_response_has_self_reflection(self, agent):
        """Test ParsedAssistantResponse.has_self_reflection."""
        text = f"{MESSAGE_START}Answer{MESSAGE_END}{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        assert result.has_self_reflection is True


# ==============================================================================
# SECTION 9: EDGE CASES AND ERROR HANDLING (10 tests)
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_text(self, agent):
        """Test parsing empty text."""
        result = agent.parse_assistant_response("")
        
        assert len(result.contents) == 0
        assert len(result.message_contents) == 0
        assert not result.has_function_calls
    
    def test_text_without_tokens(self, agent):
        """Test parsing text without any tokens."""
        result = agent.parse_assistant_response("Plain text without tokens")
        
        assert len(result.contents) == 0
    
    def test_incomplete_message_block(self, agent):
        """Test incomplete message block (no end token)."""
        text = f"{MESSAGE_START}Incomplete"
        result = agent.parse_assistant_response(text)
        
        assert len(result.message_contents) == 0
    
    def test_incomplete_function_block(self, agent):
        """Test incomplete function block."""
        text = f"{FUNCTIONS_START}<||invoke_start to=function.test||>"
        result = agent.parse_assistant_response(text)
        
        assert len(result.function_calls) == 0
    
    def test_nested_special_chars_in_content(self, agent):
        """Test content with characters similar to token delimiters."""
        content = "Content with ||| and <|> and :@: characters"
        text = f"{MESSAGE_START}{content}{MESSAGE_END}"
        result = agent.parse_assistant_response(text)
        
        assert result.message_contents[0] == content
    
    def test_very_long_content(self, agent):
        """Test very long content."""
        long_content = "X" * 100000
        text = f"{MESSAGE_START}{long_content}{MESSAGE_END}"
        result = agent.parse_assistant_response(text)
        
        assert len(result.message_contents[0]) == 100000
    
    def test_many_iterations_stress(self, agent):
        """Test many iterations (stress test)."""
        parts = []
        for i in range(10):
            parts.append(f"{MESSAGE_START}T{i}{MESSAGE_END}")
            parts.append(f"{SELF_REFLECTION_START}R{i}{SELF_REFLECTION_END}")
        text = ''.join(parts)
        result = agent.parse_assistant_response(text)
        
        assert len(result.message_contents) == 10
        assert len(result.self_reflections) == 10
    
    def test_many_parallel_tool_calls_stress(self, agent):
        """Test many parallel tool calls (stress test)."""
        calls = ''.join([f"<||invoke_start to=function.f{i}||>{INVOKE_END}" for i in range(20)])
        text = f"{MESSAGE_START}Calling{MESSAGE_END}{FUNCTIONS_START}{calls}{FUNCTIONS_END}"
        result = agent.parse_assistant_response(text)
        
        assert len(result.function_calls) == 20
    
    def test_whitespace_handling(self, agent):
        """Test whitespace is preserved/handled correctly."""
        text = f"{MESSAGE_START}  Content with spaces  {MESSAGE_END}"
        result = agent.parse_assistant_response(text)
        
        # Content is stripped
        assert result.message_contents[0] == "Content with spaces"
    
    def test_final_message_property(self, agent):
        """Test final_message property."""
        text = f"{MESSAGE_START}First{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}Last{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
        result = agent.parse_assistant_response(text)
        
        assert result.final_message == "Last"


# ==============================================================================
# SECTION 10: PARSED CONTENT ITEM TESTS (5 tests)
# ==============================================================================

class TestParsedContentItem:
    """Test ParsedContentItem dataclass."""
    
    def test_reasoning_to_dict(self):
        """Test reasoning item to dict."""
        item = ParsedContentItem(type='reasoning', content='Thinking...')
        d = item.to_dict()
        assert d == {'type': 'reasoning', 'reasoning': 'Thinking...'}
    
    def test_text_to_dict(self):
        """Test text item to dict."""
        item = ParsedContentItem(type='text', content='Hello')
        d = item.to_dict()
        assert d == {'type': 'text', 'text': 'Hello'}
    
    def test_tool_calls_to_dict(self):
        """Test tool_calls item to dict."""
        tool_calls = [{'function': {'name': 'f', 'arguments': {}}}]
        item = ParsedContentItem(type='tool_calls', content=tool_calls)
        d = item.to_dict()
        assert d == {'type': 'tool_calls', 'tool_calls': tool_calls}
    
    def test_self_reflection_to_dict(self):
        """Test self_reflection item to dict."""
        item = ParsedContentItem(type='self_reflection', content='Verified')
        d = item.to_dict()
        assert d == {'type': 'self_reflection', 'self_reflection': 'Verified'}
    
    def test_unknown_type_to_dict(self):
        """Test unknown type to dict."""
        item = ParsedContentItem(type='custom', content='data')
        d = item.to_dict()
        assert d == {'type': 'custom', 'content': 'data'}


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
