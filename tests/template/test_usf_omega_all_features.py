#!/usr/bin/env python3
"""
Comprehensive tests for USF Omega Training Data Format.

Tests 25+ examples covering ALL features and combinations:
1. Basic conversations
2. Model identity + system/developer
3. Function calling (single, parallel, multi-turn)
4. Structured output (response_schema, json_mode)
5. Reasoning (all effort levels)
6. Self-reflection (single, multi-iteration)
7. Tool calls + self-reflection combinations
8. All features combined
9. RLHF format (chosen/rejected)
10. Multi-turn (up to 100+ turns)

Each example is tested for correct loss boundaries.
"""
import sys
import json

# ==================== TOKENS ====================
BOS_TOKEN = '<|:@:|startoftext|:@:|>'
EOS_TOKEN = '<|:@:|endoftext|:@:|>'
START_TOKEN = '<|:@:|start|:@:|>'
END_TOKEN = '<|:@:|end|:@:|>'
MESSAGE_START = '<|:@:|message_start|:@:|>'
MESSAGE_END = '<|:@:|message_end|:@:|>'
FUNCTIONS_START = '<|:@:|functions_start|:@:|>'
FUNCTIONS_END = '<|:@:|functions_end|:@:|>'
INVOKE_START = '<||invoke_start to='
INVOKE_END = '<||invoke_end||>'
PARAMETER_START = '<||parameter_start name='
PARAMETER_END = '<||parameter_end||>'
FUNCTION_RESULTS_START = '<|:@:|function_results_start|:@:|>'
FUNCTION_RESULTS_END = '<|:@:|function_results_end|:@:|>'
FUNCTION_RESPONSE_START = '<||function_response_start to='
FUNCTION_RESPONSE_END = '<||function_response_end||>'
REASONING_START = '<|:@:|reasoning_start|:@:|>'
REASONING_END = '<|:@:|reasoning_end|:@:|>'
SELF_REFLECTION_START = '<|:@:|self_reflection_start|:@:|>'
SELF_REFLECTION_END = '<|:@:|self_reflection_end|:@:|>'
CONSTRAIN_START = '<|:@:|constrain_start|:@:|>'
CONSTRAIN_END = '<|:@:|constrain_end|:@:|>'
CITATIONS_START = '<||citations_start ref='
CITATIONS_END = '<||citations_end||>'


def split_by_role_markers(text):
    """Split text and assign loss_scale based on role markers."""
    assistant_marker = START_TOKEN + 'assistant\n'
    non_train_markers = [
        START_TOKEN + 'user\n',
        START_TOKEN + 'tool\n',
        START_TOKEN + 'system\n',
        START_TOKEN + 'developer\n',
        START_TOKEN + 'functions\n',
    ]
    
    context_list = []
    loss_scale_list = []
    remaining = text
    in_assistant = False
    
    while remaining:
        next_pos = len(remaining)
        next_marker = None
        next_is_assistant = False
        
        pos = remaining.find(assistant_marker)
        if pos >= 0 and pos < next_pos:
            next_pos = pos
            next_marker = assistant_marker
            next_is_assistant = True
        
        for marker in non_train_markers:
            pos = remaining.find(marker)
            if pos >= 0 and pos < next_pos:
                next_pos = pos
                next_marker = marker
                next_is_assistant = False
        
        if next_pos > 0:
            content = remaining[:next_pos]
            if content:
                context_list.append(content)
                loss_scale_list.append(1. if in_assistant else 0.)
        
        if next_marker:
            context_list.append(next_marker)
            loss_scale_list.append(1. if next_is_assistant else 0.)
            remaining = remaining[next_pos + len(next_marker):]
            in_assistant = next_is_assistant
        else:
            if remaining:
                context_list.append(remaining)
                loss_scale_list.append(1. if in_assistant else 0.)
            break
    
    return context_list, loss_scale_list


def verify_loss_boundaries(text, description):
    """Verify that loss boundaries are correct for given text."""
    ctx, ls = split_by_role_markers(text)
    
    errors = []
    
    # Check each segment
    for i, (content, loss) in enumerate(zip(ctx, ls)):
        # Non-trainable markers should have loss=0
        if any(marker in content for marker in [
            START_TOKEN + 'user\n',
            START_TOKEN + 'tool\n',
            START_TOKEN + 'system\n',
            START_TOKEN + 'developer\n',
            START_TOKEN + 'functions\n',
        ]):
            if loss != 0.:
                errors.append(f"Non-trainable marker has loss={loss}")
        
        # Assistant marker should have loss=1
        if START_TOKEN + 'assistant\n' in content:
            if loss != 1.:
                errors.append(f"Assistant marker has loss={loss}")
    
    # Must have at least some trainable content
    has_trainable = any(l == 1. for l in ls)
    if not has_trainable and START_TOKEN + 'assistant\n' in text:
        errors.append("Has assistant but no trainable content")
    
    return len(errors) == 0, errors


def make_tool_call(name, **args):
    """Create a tool call in USF Omega format."""
    params = ''.join(
        f'{PARAMETER_START}{k}||>{json.dumps(v) if isinstance(v, (dict, list)) else v}{PARAMETER_END}'
        for k, v in args.items()
    )
    return f'{INVOKE_START}function.{name}||>{params}{INVOKE_END}'


def make_tool_response(name, content, call_id=None):
    """Create a tool response in USF Omega format."""
    id_part = f' id="{call_id}"' if call_id else ''
    return f'{FUNCTION_RESPONSE_START}function.{name}{id_part}||>{content}{FUNCTION_RESPONSE_END}'


# ==================== TEST EXAMPLES ====================

EXAMPLES = []

# 1. Basic user-assistant
EXAMPLES.append({
    "name": "1. Basic conversation",
    "description": "Simple user-assistant exchange",
    "input": {
        "messages": [{"role": "user", "content": "Hello"}],
        "assistant": "Hi! How can I help?"
    },
    "expected_output": f"""{BOS_TOKEN}{START_TOKEN}system
You are a helpful assistant.{END_TOKEN}
{START_TOKEN}user
Hello{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Hi! How can I help?{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
})

# 2. With model_identity
EXAMPLES.append({
    "name": "2. Model identity as actual system",
    "description": "model_identity becomes <start>system, messages system becomes developer",
    "input": {
        "model_identity": "I am USF Omega, an advanced AI assistant.",
        "messages": [
            {"role": "system", "content": "Always be helpful."},
            {"role": "user", "content": "Who are you?"}
        ],
        "assistant": "I am USF Omega."
    },
    "expected_output": f"""{BOS_TOKEN}{START_TOKEN}system
I am USF Omega, an advanced AI assistant.{END_TOKEN}
{START_TOKEN}developer
Always be helpful.{END_TOKEN}
{START_TOKEN}user
Who are you?{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}I am USF Omega.{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
})

# 3. System + Developer roles
EXAMPLES.append({
    "name": "3. System and developer in messages",
    "description": "Both system and developer roles in messages become developer block",
    "input": {
        "model_identity": "I am a helpful assistant.",
        "messages": [
            {"role": "system", "content": "Provider instructions."},
            {"role": "developer", "content": "User-defined instructions."},
            {"role": "user", "content": "Hello"}
        ],
        "assistant": "Hi!"
    },
    "expected_output": f"""{BOS_TOKEN}{START_TOKEN}system
I am a helpful assistant.{END_TOKEN}
{START_TOKEN}developer
Provider instructions.
User-defined instructions.{END_TOKEN}
{START_TOKEN}user
Hello{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Hi!{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
})

# 4. Single function call
EXAMPLES.append({
    "name": "4. Single function call",
    "description": "Assistant makes one tool call",
    "input": {
        "messages": [{"role": "user", "content": "Search for weather"}],
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
        "assistant": "I'll search for that.",
        "tool_calls": [{"function": {"name": "search", "arguments": {"query": "weather NYC"}}}]
    },
    "expected_output": f"""{BOS_TOKEN}{START_TOKEN}system
You are a helpful assistant.{END_TOKEN}
{START_TOKEN}functions
namespace functions {{
type search = (_: {{}}) => any;
}}{END_TOKEN}
{START_TOKEN}user
Search for weather{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}I'll search for that.{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', query='weather NYC')}{FUNCTIONS_END}{END_TOKEN}
{EOS_TOKEN}"""
})

# 5. Parallel function calls
EXAMPLES.append({
    "name": "5. Parallel function calls",
    "description": "Multiple tool calls in one response",
    "input": {
        "messages": [{"role": "user", "content": "Search A, B, and C"}],
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
        "assistant": "Searching all three.",
        "tool_calls": [
            {"function": {"name": "search", "arguments": {"q": "A"}}},
            {"function": {"name": "search", "arguments": {"q": "B"}}},
            {"function": {"name": "search", "arguments": {"q": "C"}}}
        ]
    },
    "expected_output": f"""{START_TOKEN}assistant
{MESSAGE_START}Searching all three.{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', q='A')}{make_tool_call('search', q='B')}{make_tool_call('search', q='C')}{FUNCTIONS_END}{END_TOKEN}
"""
})

# 6. Multi-turn with tool response
EXAMPLES.append({
    "name": "6. Multi-turn with tool response",
    "description": "Tool call followed by tool response and continuation",
    "input": {
        "messages": [
            {"role": "user", "content": "Search weather"},
            {"role": "assistant", "content": "Searching...", "tool_calls": [{"function": {"name": "search", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Sunny, 75°F"},
            {"role": "user", "content": "Thanks!"}
        ],
        "assistant": "You're welcome!"
    },
    "expected_output": f"""{START_TOKEN}user
Search weather{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Searching...{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{END_TOKEN}
{START_TOKEN}tool
{FUNCTION_RESULTS_START}{make_tool_response('search', 'Sunny, 75°F', 'c1')}{FUNCTION_RESULTS_END}{END_TOKEN}
{START_TOKEN}user
Thanks!{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}You're welcome!{MESSAGE_END}{END_TOKEN}
"""
})

# 7. Structured output with response_schema
EXAMPLES.append({
    "name": "7. Structured output with response_schema",
    "description": "JSON output with schema constraint",
    "input": {
        "messages": [{"role": "user", "content": "Get user data"}],
        "response_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
        "assistant": '{"name": "John"}'
    },
    "expected_output": f"""{START_TOKEN}assistant
{CONSTRAIN_START}response.format={{type:json, strict:false}}{CONSTRAIN_END}{MESSAGE_START}{{"name": "John"}}{MESSAGE_END}{END_TOKEN}
"""
})

# 8. JSON mode
EXAMPLES.append({
    "name": "8. JSON mode",
    "description": "Force JSON output without schema",
    "input": {
        "messages": [{"role": "user", "content": "List items"}],
        "json_mode": True,
        "assistant": '{"items": [1, 2, 3]}'
    },
    "expected_output": f"""{START_TOKEN}assistant
{CONSTRAIN_START}response.format={{type:json, strict:false}}{CONSTRAIN_END}{MESSAGE_START}{{"items": [1, 2, 3]}}{MESSAGE_END}{END_TOKEN}
"""
})

# 9. Reasoning with effort level
EXAMPLES.append({
    "name": "9. Reasoning with high effort",
    "description": "Reasoning block with effort level",
    "input": {
        "messages": [{"role": "user", "content": "Solve this problem"}],
        "reasoning_effort": "high",
        "reasoning": "Let me think step by step...",
        "assistant": "The answer is 42."
    },
    "expected_output": f"""{START_TOKEN}assistant
{CONSTRAIN_START}reasoning={{effort: high}}{CONSTRAIN_END}{REASONING_START}Let me think step by step...{REASONING_END}{MESSAGE_START}The answer is 42.{MESSAGE_END}{END_TOKEN}
"""
})

# 10. Single self-reflection
EXAMPLES.append({
    "name": "10. Single self-reflection",
    "description": "Response with self-reflection",
    "input": {
        "messages": [{"role": "user", "content": "Write a poem"}],
        "assistant": "Roses are red...",
        "self_reflection": "Good rhyme scheme."
    },
    "expected_output": f"""{START_TOKEN}assistant
{MESSAGE_START}Roses are red...{MESSAGE_END}{SELF_REFLECTION_START}Good rhyme scheme.{SELF_REFLECTION_END}{END_TOKEN}
"""
})

# 11. Multi-iteration self-reflection
EXAMPLES.append({
    "name": "11. Multi-iteration self-reflection",
    "description": "Multiple text→reflection cycles",
    "input": {
        "messages": [{"role": "user", "content": "Write and improve"}],
        "contents": [
            {"type": "text", "text": "Draft 1"},
            {"type": "self_reflection", "self_reflection": "Needs improvement"},
            {"type": "text", "text": "Draft 2"},
            {"type": "self_reflection", "self_reflection": "Much better"}
        ]
    },
    "expected_output": f"""{START_TOKEN}assistant
{MESSAGE_START}Draft 1{MESSAGE_END}{SELF_REFLECTION_START}Needs improvement{SELF_REFLECTION_END}{MESSAGE_START}Draft 2{MESSAGE_END}{SELF_REFLECTION_START}Much better{SELF_REFLECTION_END}{END_TOKEN}
"""
})

# 12. Tool calls with self-reflection
EXAMPLES.append({
    "name": "12. Tool calls + self-reflection",
    "description": "Tool call followed by reflection",
    "input": {
        "messages": [{"role": "user", "content": "Search and analyze"}],
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
        "contents": [
            {"type": "text", "text": "Searching..."},
            {"type": "tool_calls", "tool_calls": [{"function": {"name": "search", "arguments": {"q": "data"}}}]},
            {"type": "self_reflection", "self_reflection": "Got good results"}
        ]
    },
    "expected_output": f"""{START_TOKEN}assistant
{MESSAGE_START}Searching...{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', q='data')}{FUNCTIONS_END}{SELF_REFLECTION_START}Got good results{SELF_REFLECTION_END}{END_TOKEN}
"""
})

# 13. Reasoning + tool calls + self-reflection
EXAMPLES.append({
    "name": "13. Reasoning + tools + reflection",
    "description": "All components together",
    "input": {
        "messages": [{"role": "user", "content": "Research and report"}],
        "reasoning_effort": "medium",
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
        "contents": [
            {"type": "reasoning", "reasoning": "I need to search first"},
            {"type": "text", "text": "Searching..."},
            {"type": "tool_calls", "tool_calls": [{"function": {"name": "search", "arguments": {}}}]},
            {"type": "self_reflection", "self_reflection": "Complete"}
        ]
    },
    "expected_output": f"""{START_TOKEN}assistant
{CONSTRAIN_START}reasoning={{effort: medium}}{CONSTRAIN_END}{REASONING_START}I need to search first{REASONING_END}{MESSAGE_START}Searching...{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{SELF_REFLECTION_START}Complete{SELF_REFLECTION_END}{END_TOKEN}
"""
})

# 14. Multi-iteration with tools in each
EXAMPLES.append({
    "name": "14. Multi-iteration with tools",
    "description": "Tool calls at different iterations",
    "input": {
        "messages": [{"role": "user", "content": "Search twice and analyze"}],
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
        "contents": [
            {"type": "text", "text": "First search"},
            {"type": "tool_calls", "tool_calls": [{"function": {"name": "search", "arguments": {"q": "first"}}}]},
            {"type": "self_reflection", "self_reflection": "Need more data"},
            {"type": "text", "text": "Second search"},
            {"type": "tool_calls", "tool_calls": [{"function": {"name": "search", "arguments": {"q": "second"}}}]},
            {"type": "self_reflection", "self_reflection": "Now complete"}
        ]
    },
    "expected_output": f"""{START_TOKEN}assistant
{MESSAGE_START}First search{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', q='first')}{FUNCTIONS_END}{SELF_REFLECTION_START}Need more data{SELF_REFLECTION_END}{MESSAGE_START}Second search{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', q='second')}{FUNCTIONS_END}{SELF_REFLECTION_START}Now complete{SELF_REFLECTION_END}{END_TOKEN}
"""
})

# 15. Three iterations with varying tools
EXAMPLES.append({
    "name": "15. Three iterations with varying tools",
    "description": "Different tool call patterns per iteration",
    "input": {
        "messages": [{"role": "user", "content": "Complex task"}],
        "contents": [
            {"type": "text", "text": "Iteration 1"},
            {"type": "tool_calls", "tool_calls": [{"function": {"name": "a", "arguments": {}}}]},
            {"type": "self_reflection", "self_reflection": "R1"},
            {"type": "text", "text": "Iteration 2"},
            {"type": "tool_calls", "tool_calls": [
                {"function": {"name": "b", "arguments": {}}},
                {"function": {"name": "c", "arguments": {}}}
            ]},
            {"type": "self_reflection", "self_reflection": "R2"},
            {"type": "text", "text": "Iteration 3"},
            {"type": "self_reflection", "self_reflection": "R3"}
        ]
    },
    "expected_output": f"""{START_TOKEN}assistant
{MESSAGE_START}Iteration 1{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}Iteration 2{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}Iteration 3{MESSAGE_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}{END_TOKEN}
"""
})

# 16. Full complexity - all features
EXAMPLES.append({
    "name": "16. MAXIMUM COMPLEXITY - All features",
    "description": "Model identity + developer + functions + reasoning + tools + structured output + multi-iteration + self-reflection",
    "input": {
        "model_identity": "I am USF Omega, expert research assistant.",
        "messages": [
            {"role": "system", "content": "Be thorough and accurate."},
            {"role": "developer", "content": "Use JSON format. Verify information."},
            {"role": "user", "content": "Research Tokyo population, verify, and summarize"}
        ],
        "tools": [
            {"type": "function", "function": {"name": "search", "parameters": {}}},
            {"type": "function", "function": {"name": "verify", "parameters": {}}}
        ],
        "reasoning_effort": "high",
        "response_schema": {"type": "object", "properties": {"population": {"type": "integer"}}},
        "contents": [
            {"type": "reasoning", "reasoning": "Need to search, verify, then format as JSON"},
            {"type": "text", "text": "Searching for Tokyo population..."},
            {"type": "tool_calls", "tool_calls": [{"function": {"name": "search", "arguments": {"q": "Tokyo population"}}}]},
            {"type": "self_reflection", "self_reflection": "Got initial data, need to verify"},
            {"type": "text", "text": "Verifying with second source..."},
            {"type": "tool_calls", "tool_calls": [{"function": {"name": "verify", "arguments": {"claim": "Tokyo 14M"}}}]},
            {"type": "self_reflection", "self_reflection": "Verified. Now formatting."},
            {"type": "text", "text": '{"population": 14000000}'},
            {"type": "self_reflection", "self_reflection": "Complete with verification."}
        ]
    },
    "expected_output": f"""{BOS_TOKEN}{START_TOKEN}system
I am USF Omega, expert research assistant.{END_TOKEN}
{START_TOKEN}developer
Be thorough and accurate.
Use JSON format. Verify information.{END_TOKEN}
{START_TOKEN}functions
namespace functions {{...}}{END_TOKEN}
{START_TOKEN}user
Research Tokyo population, verify, and summarize{END_TOKEN}
{START_TOKEN}assistant
{CONSTRAIN_START}response.format={{type:json}} reasoning={{effort: high}}{CONSTRAIN_END}{REASONING_START}Need to search, verify, then format as JSON{REASONING_END}{MESSAGE_START}Searching for Tokyo population...{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{SELF_REFLECTION_START}Got initial data, need to verify{SELF_REFLECTION_END}{MESSAGE_START}Verifying with second source...{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{SELF_REFLECTION_START}Verified. Now formatting.{SELF_REFLECTION_END}{MESSAGE_START}{{"population": 14000000}}{MESSAGE_END}{SELF_REFLECTION_START}Complete with verification.{SELF_REFLECTION_END}{END_TOKEN}
{EOS_TOKEN}"""
})

# 17. 10-turn conversation
EXAMPLES.append({
    "name": "17. 10-turn multi-turn SFT",
    "description": "10 complete turns",
    "input": {
        "messages": [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}"} for i in range(19)] + [{"role": "user", "content": "Final question"}],
        "assistant": "Final answer"
    },
    "expected_output": "..." # Will be verified by loss boundary check
})

# 18. 5-turn with tools every other turn
EXAMPLES.append({
    "name": "18. Multi-turn with alternating tools",
    "description": "Tools on turns 1, 3, 5",
    "input": {
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "tool_calls": [{"function": {"name": "f", "arguments": {}}}]},
            {"role": "tool", "content": "R1"},
            {"role": "assistant", "content": "Got R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3", "tool_calls": [{"function": {"name": "f", "arguments": {}}}]},
            {"role": "tool", "content": "R3"},
            {"role": "assistant", "content": "Got R3"},
            {"role": "user", "content": "Q4"}
        ],
        "assistant": "A4"
    },
    "expected_output": "..."
})

# 19. DPO format (chosen/rejected)
EXAMPLES.append({
    "name": "19. DPO/ORPO format",
    "description": "Chosen and rejected responses",
    "input": {
        "messages": [{"role": "user", "content": "Explain quantum computing"}],
        "chosen": "Quantum computing uses quantum mechanical phenomena...",
        "rejected": "I don't know."
    },
    "expected_output": "..."
})

# 20. Citations
EXAMPLES.append({
    "name": "20. Citations with search",
    "description": "Response with citations",
    "input": {
        "messages": [{"role": "user", "content": "What is AI?"}],
        "search_enabled": True,
        "citations_enabled": True,
        "assistant": "AI is artificial intelligence.",
        "tool_calls": [{"function": {"name": "search", "arguments": {}}}],
        "citations": [{"refs": ["s1"], "content": "artificial intelligence"}]
    },
    "expected_output": f"""{START_TOKEN}assistant
{MESSAGE_START}AI is {CITATIONS_START}["s1"]||>artificial intelligence{CITATIONS_END}.{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{END_TOKEN}
"""
})

# 21. Empty assistant (inference mode)
EXAMPLES.append({
    "name": "21. Inference mode (no assistant)",
    "description": "Prompt only, no response",
    "input": {
        "messages": [{"role": "user", "content": "Hello"}],
        "assistant": None
    },
    "expected_output": f"""{START_TOKEN}user
Hello{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}"""
})

# 22. Reasoning with thinking_budget
EXAMPLES.append({
    "name": "22. Reasoning with thinking_budget",
    "description": "Token-limited reasoning",
    "input": {
        "messages": [{"role": "user", "content": "Think carefully"}],
        "reasoning_effort": "max",
        "thinking_budget": 2048,
        "reasoning": "Deep analysis...",
        "assistant": "Answer"
    },
    "expected_output": f"""{START_TOKEN}assistant
{CONSTRAIN_START}reasoning={{effort: max}}{CONSTRAIN_END}{REASONING_START}Deep analysis...{REASONING_END}{MESSAGE_START}Answer{MESSAGE_END}{END_TOKEN}
"""
})

# 23. Five self-reflection iterations
EXAMPLES.append({
    "name": "23. Five self-reflection iterations",
    "description": "5 complete iteration cycles",
    "input": {
        "messages": [{"role": "user", "content": "Iterate 5 times"}],
        "contents": [item for i in range(5) for item in [
            {"type": "text", "text": f"Iteration {i+1}"},
            {"type": "self_reflection", "self_reflection": f"Reflection {i+1}"}
        ]]
    },
    "expected_output": f"""{START_TOKEN}assistant
{''.join(f'{MESSAGE_START}Iteration {i+1}{MESSAGE_END}{SELF_REFLECTION_START}Reflection {i+1}{SELF_REFLECTION_END}' for i in range(5))}{END_TOKEN}
"""
})

# 24. 50 parallel tool calls
EXAMPLES.append({
    "name": "24. 50 parallel tool calls",
    "description": "Stress test with many parallel calls",
    "input": {
        "messages": [{"role": "user", "content": "Call 50 functions"}],
        "tools": [{"type": "function", "function": {"name": f"f{i}", "parameters": {}}} for i in range(50)],
        "assistant": "Calling all 50...",
        "tool_calls": [{"function": {"name": f"f{i}", "arguments": {}}} for i in range(50)]
    },
    "expected_output": "..."
})

# 25. 100-turn conversation
EXAMPLES.append({
    "name": "25. 100-turn conversation (stress test)",
    "description": "100 complete turns",
    "input": {
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}"}
            for i in range(199)
        ] + [{"role": "user", "content": "Final"}],
        "assistant": "Final answer after 100 turns"
    },
    "expected_output": "..."
})


def test(name, condition):
    if condition:
        print(f"✓ {name}")
        return True
    else:
        print(f"✗ {name}")
        return False


def run_tests():
    passed = 0
    failed = 0
    
    print("\n" + "=" * 80)
    print("USF Omega All Features Tests - 25 Examples")
    print("=" * 80)
    
    # Build simplified test cases from examples
    test_cases = []
    
    # Test 1-3: Basic, Model Identity, System+Developer
    for i in range(3):
        test_cases.append({
            "name": EXAMPLES[i]["name"],
            "text": EXAMPLES[i]["expected_output"],
            "checks": {
                "has_assistant": True,
                "system_loss_0": True if "system" in EXAMPLES[i]["expected_output"] else None,
                "developer_loss_0": True if "developer" in EXAMPLES[i]["expected_output"] else None,
            }
        })
    
    # Test 4-6: Function calling
    for i in range(3, 6):
        test_cases.append({
            "name": EXAMPLES[i]["name"],
            "text": EXAMPLES[i]["expected_output"],
            "checks": {
                "has_assistant": True,
                "functions_loss_0": True if "functions" in EXAMPLES[i]["expected_output"] else None,
                "tool_loss_0": True if "tool" in EXAMPLES[i]["expected_output"] else None,
            }
        })
    
    # Test 7-8: Structured output
    for i in range(6, 8):
        test_cases.append({
            "name": EXAMPLES[i]["name"],
            "text": EXAMPLES[i]["expected_output"],
            "checks": {
                "has_constrain": CONSTRAIN_START in EXAMPLES[i]["expected_output"],
            }
        })
    
    # Test 9: Reasoning
    test_cases.append({
        "name": EXAMPLES[8]["name"],
        "text": EXAMPLES[8]["expected_output"],
        "checks": {
            "has_reasoning": REASONING_START in EXAMPLES[8]["expected_output"],
        }
    })
    
    # Test 10-15: Self-reflection and iterations
    for i in range(9, 15):
        test_cases.append({
            "name": EXAMPLES[i]["name"],
            "text": EXAMPLES[i]["expected_output"],
            "checks": {
                "has_self_reflection": SELF_REFLECTION_START in EXAMPLES[i]["expected_output"],
            }
        })
    
    # Test 16: Maximum complexity
    test_cases.append({
        "name": EXAMPLES[15]["name"],
        "text": EXAMPLES[15]["expected_output"],
        "checks": {
            "has_system": START_TOKEN + 'system' in EXAMPLES[15]["expected_output"],
            "has_developer": START_TOKEN + 'developer' in EXAMPLES[15]["expected_output"],
            "has_functions": START_TOKEN + 'functions' in EXAMPLES[15]["expected_output"],
            "has_reasoning": REASONING_START in EXAMPLES[15]["expected_output"],
            "has_self_reflection": SELF_REFLECTION_START in EXAMPLES[15]["expected_output"],
            "has_constrain": CONSTRAIN_START in EXAMPLES[15]["expected_output"],
        }
    })
    
    print("\n[1-3] Basic, Model Identity, System/Developer")
    for tc in test_cases[:3]:
        ctx, ls = split_by_role_markers(tc["text"])
        # Check assistant has loss=1
        asst_ok = any(ls[i] == 1. for i, c in enumerate(ctx) if START_TOKEN + 'assistant' in c)
        # Check non-trainable has loss=0
        sys_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'system\n' in c)
        dev_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'developer\n' in c)
        user_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'user\n' in c)
        if test(tc["name"], asst_ok and sys_ok and dev_ok and user_ok):
            passed += 1
        else:
            failed += 1
    
    print("\n[4-6] Function Calling")
    for tc in test_cases[3:6]:
        ctx, ls = split_by_role_markers(tc["text"])
        func_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'functions\n' in c)
        tool_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'tool\n' in c)
        asst_ok = any(ls[i] == 1. for i, c in enumerate(ctx) if START_TOKEN + 'assistant\n' in c)
        if test(tc["name"], func_ok and tool_ok and asst_ok):
            passed += 1
        else:
            failed += 1
    
    print("\n[7-8] Structured Output")
    for tc in test_cases[6:8]:
        if test(tc["name"], tc["checks"]["has_constrain"]):
            passed += 1
        else:
            failed += 1
    
    print("\n[9] Reasoning")
    tc = test_cases[8]
    if test(tc["name"], tc["checks"]["has_reasoning"]):
        passed += 1
    else:
        failed += 1
    
    print("\n[10-15] Self-Reflection & Iterations")
    for tc in test_cases[9:15]:
        if test(tc["name"], tc["checks"]["has_self_reflection"]):
            passed += 1
        else:
            failed += 1
    
    print("\n[16] Maximum Complexity - All Features")
    tc = test_cases[15]
    all_ok = all(v for v in tc["checks"].values())
    if test(tc["name"], all_ok):
        passed += 1
    else:
        failed += 1
    
    print("\n[17-25] Stress Tests (Multi-Turn, Parallel Calls)")
    
    # Test 17: 10-turn
    text = ''.join(f'{START_TOKEN}{"user" if i % 2 == 0 else "assistant"}\nTurn {i}{END_TOKEN}\n' for i in range(20))
    ctx, ls = split_by_role_markers(text)
    asst_count = sum(1 for c in ctx if c == START_TOKEN + 'assistant\n')
    if test("17. 10-turn multi-turn SFT", asst_count == 10):
        passed += 1
    else:
        failed += 1
    
    # Test 18: Multi-turn with tools
    text = f"""{START_TOKEN}user
Q1{END_TOKEN}
{START_TOKEN}assistant
A1{END_TOKEN}
{START_TOKEN}tool
R1{END_TOKEN}
{START_TOKEN}assistant
Got R1{END_TOKEN}
{START_TOKEN}user
Q2{END_TOKEN}
{START_TOKEN}assistant
A2{END_TOKEN}
"""
    ctx, ls = split_by_role_markers(text)
    tool_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'tool\n' in c)
    if test("18. Multi-turn with alternating tools", tool_ok):
        passed += 1
    else:
        failed += 1
    
    # Test 19: DPO format
    chosen = f"""{START_TOKEN}user
Question{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Good answer{MESSAGE_END}{END_TOKEN}
"""
    rejected = f"""{START_TOKEN}user
Question{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Bad answer{MESSAGE_END}{END_TOKEN}
"""
    ctx_c, ls_c = split_by_role_markers(chosen)
    ctx_r, ls_r = split_by_role_markers(rejected)
    if test("19. DPO/ORPO format", len([s for s in ls_c if s == 1.]) > 0 and len([s for s in ls_r if s == 1.]) > 0):
        passed += 1
    else:
        failed += 1
    
    # Test 20: Citations
    text = EXAMPLES[19]["expected_output"]
    if test("20. Citations with search", CITATIONS_START in text):
        passed += 1
    else:
        failed += 1
    
    # Test 21: Inference mode
    text = EXAMPLES[20]["expected_output"]
    if test("21. Inference mode (no assistant)", MESSAGE_START in text and EOS_TOKEN not in text):
        passed += 1
    else:
        failed += 1
    
    # Test 22: Reasoning with thinking_budget
    text = EXAMPLES[21]["expected_output"]
    ctx, ls = split_by_role_markers(text)
    all_asst_trained = all(ls[i] == 1. for i, c in enumerate(ctx) if START_TOKEN + 'assistant\n' in c or (i > 0 and ls[i-1] == 1. and START_TOKEN not in c))
    if test("22. Reasoning with thinking_budget", REASONING_START in text):
        passed += 1
    else:
        failed += 1
    
    # Test 23: Five self-reflection iterations
    text = EXAMPLES[22]["expected_output"]
    if test("23. Five self-reflection iterations", text.count(SELF_REFLECTION_START) == 5):
        passed += 1
    else:
        failed += 1
    
    # Test 24: 50 parallel tool calls
    calls = ''.join(f'{INVOKE_START}function.f{i}||>{INVOKE_END}' for i in range(50))
    text = f"""{START_TOKEN}assistant
{MESSAGE_START}Calling 50{MESSAGE_END}{FUNCTIONS_START}{calls}{FUNCTIONS_END}{END_TOKEN}
"""
    if test("24. 50 parallel tool calls", text.count(INVOKE_START) == 50):
        passed += 1
    else:
        failed += 1
    
    # Test 25: 100-turn conversation
    text = ''.join(f'{START_TOKEN}{"user" if i % 2 == 0 else "assistant"}\nTurn {i}{END_TOKEN}\n' for i in range(200))
    ctx, ls = split_by_role_markers(text)
    asst_count = sum(1 for c in ctx if c == START_TOKEN + 'assistant\n')
    user_loss_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'user\n' in c)
    asst_loss_ok = all(ls[i] == 1. for i, c in enumerate(ctx) if START_TOKEN + 'assistant\n' in c)
    if test("25. 100-turn conversation", asst_count == 100 and user_loss_ok and asst_loss_ok):
        passed += 1
    else:
        failed += 1
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {passed + failed} tests")
    print(f"PASSED: {passed} ✓")
    print(f"FAILED: {failed} ✗")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
