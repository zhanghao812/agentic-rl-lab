"""定义 Qwen3.5 使用的代码解释器（code_interpreter）工具协议。

模型用 Qwen 原生 <tool_call> 格式调用
code_interpreter（参数就一个 code 字符串），沙箱执行结果以 role="tool" 消息返回；
最终答案用 \boxed{} 给出，且不允许同一轮里既调工具又给答案。
"""

# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 collections.abc 导入 Mapping。
from collections.abc import Mapping
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 typing 导入 Any。
from typing import Any


# 逐行讲解：构造字典 {'type': 'function', 'function': {'name': 'code_interpreter', 'description': 'A Python code execution envir…，赋给 CODE_TOOL。
CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        # 工具描述参照官方复现指南中实测"能涨点"的写法：
        # 强调 print 输出、每次执行独立无状态。
        "description": (
            "A Python code execution environment that allows you to:\n"
            "- Run Python code for calculations, data analysis, and other computational tasks\n"
            "- Get results through the `print()` function output\n"
            "- Execute code in an isolated environment (each execution starts fresh)\n\n"
            "Important notes:\n"
            "- Results are captured from `print()` statements\n"
            "- Returns empty string if no output is printed\n"
            "- Each execution is independent (no state persistence between runs)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to be executed.",
                }
            },
            "required": ["code"],
        },
    },
}

# 逐行讲解：取值 'You solve math problems step by step with help from a Python code interpreter.\nUse the code_interpreter t…，赋给 SYSTEM_PROMPT。
SYSTEM_PROMPT = """You solve math problems step by step with help from a Python code interpreter.
Use the code_interpreter tool when calculation, symbolic manipulation, or enumeration helps you solve the problem accurately and quickly.

How to use the code_interpreter tool:
- Call code_interpreter with a `code` string containing Python code. Call it at most once per assistant turn, then wait for the execution result.
- Results are captured from what your code prints with print(). Always print the values you want to see.
- Each execution is independent: no variables, files, or state carry over between calls. Redefine everything you need in each piece of code.
- Code must finish within a few seconds and use little memory. Do not read or write files. If you enumerate or brute-force, keep the search space small.
- If the execution returns an error, analyze it and retry with corrected code when useful.

When you have the final answer, end with exactly one line in this format:
\\boxed{<your final answer>}
Do not call the tool and give the final answer in the same turn."""

# 逐行讲解：调用 re.compile（re.compile('<tool_call>\\s*<function=code_interpreter>\\s*<parameter=code>\\s*(.*?)\\s*</parameter>\\s*</fu…），赋给 TOOL_CALL_PATTERN。
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=code_interpreter>\s*<parameter=code>\s*(.*?)\s*"
    r"</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)


# 逐行讲解：定义类 ParsedAssistant：保存一次 assistant 输出的协议解析结果。
@dataclass(frozen=True)
class ParsedAssistant:
    """保存一次 assistant 输出的协议解析结果。"""

    # 逐行讲解：声明 kind 的类型为 str。
    kind: str  # 解析类型："tool"、"answer" 或 "invalid"
    # 逐行讲解：声明 content 的类型为 str。
    content: str  # 普通文本：工具调用前的推理，或完整答案/非法输出
    # 逐行讲解：声明 code 的类型为 str | None；取值 None。
    code: str | None = None  # kind 为 "tool" 时提取出的待执行 Python 代码


# 逐行讲解：定义函数 initial_messages：为一道数学题创建初始对话。
def initial_messages(question: str) -> list[dict[str, Any]]:
    """为一道数学题创建初始对话。"""
    # 逐行讲解：返回结果：[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': question}]。
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# 逐行讲解：定义函数 _render_chat：渲染消息并把 tokenizer 的不同返回类型统一成一维 token 列表。
def _render_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """渲染消息并把 tokenizer 的不同返回类型统一成一维 token 列表。"""
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template(messages, tools=[CODE_TOOL], tokenize=True, add_generation_prompt=add_generat…），赋给 rendered。
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[CODE_TOOL],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    # 逐行讲解：检查条件 isinstance(rendered, Mapping)；成立时执行下方分支。
    if isinstance(rendered, Mapping):
        # 逐行讲解：按索引或切片读取 rendered['input_ids']，赋给 rendered。
        rendered = rendered["input_ids"]
    # 逐行讲解：检查条件 hasattr(rendered, 'tolist')；成立时执行下方分支。
    if hasattr(rendered, "tolist"):
        # 逐行讲解：调用 rendered.tolist（rendered.tolist()），赋给 rendered。
        rendered = rendered.tolist()
    # 逐行讲解：检查条件 rendered and isinstance(rendered[0], list)；成立时执行下方分支。
    if rendered and isinstance(rendered[0], list):
        # 逐行讲解：按索引或切片读取 rendered[0]，赋给 rendered。
        rendered = rendered[0]
    # 逐行讲解：返回结果：[int(token) for token in rendered]。
    return [int(token) for token in rendered]


# 逐行讲解：定义函数 build_prompt：用模型原生 chat template 构建带工具定义的生成 prompt。
def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """用模型原生 chat template 构建带工具定义的生成 prompt。"""
    # 逐行讲解：返回结果：_render_chat(tokenizer, messages, add_generation_prompt=True)。
    return _render_chat(tokenizer, messages, add_generation_prompt=True)


# 逐行讲解：定义函数 _encoded_text_tokens：把普通文本编码结果统一成一维 token 列表。
def _encoded_text_tokens(tokenizer: Any, text: str) -> list[int]:
    """把普通文本编码结果统一成一维 token 列表。"""
    # 逐行讲解：将文本编码为 token（tokenizer.encode(text, add_special_tokens=False)），赋给 encoded。
    encoded = tokenizer.encode(text, add_special_tokens=False)
    # 逐行讲解：检查条件 hasattr(encoded, 'tolist')；成立时执行下方分支。
    if hasattr(encoded, "tolist"):
        # 逐行讲解：调用 encoded.tolist（encoded.tolist()），赋给 encoded。
        encoded = encoded.tolist()
    # 逐行讲解：检查条件 encoded and isinstance(encoded[0], list)；成立时执行下方分支。
    if encoded and isinstance(encoded[0], list):
        # 逐行讲解：按索引或切片读取 encoded[0]，赋给 encoded。
        encoded = encoded[0]
    # 逐行讲解：返回结果：[int(token) for token in encoded]。
    return [int(token) for token in encoded]


# 逐行讲解：定义函数 _suffix_prefix_overlap：返回 tokens 末尾与 suffix 开头的最长重叠长度。
def _suffix_prefix_overlap(tokens: list[int], suffix: list[int]) -> int:
    """返回 tokens 末尾与 suffix 开头的最长重叠长度。"""
    # 逐行讲解：遍历 range(min(len(tokens), len(suffix)), 0, -1)，将每项绑定到 length。
    for length in range(min(len(tokens), len(suffix)), 0, -1):
        # 逐行讲解：检查条件 tokens[-length:] == suffix[:length]；成立时执行下方分支。
        if tokens[-length:] == suffix[:length]:
            # 逐行讲解：返回结果：length。
            return length
    # 逐行讲解：返回结果：0。
    return 0


# 逐行讲解：定义函数 build_next_prompt：用真实采样 token 接上 assistant 结束符和新的 tool observation。
def build_next_prompt(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    """用真实采样 token 接上 assistant 结束符和新的 tool observation。

    与 search-r1 相同：不重渲染整段历史（token-in token-out），
    只拼接"assistant 结束符 + tool observation"的增量 token，
    避免 text↔token 重编码不可逆导致的训练错位（官方实测的坑）。

    增量片段只与 tool 消息有关、与 assistant 内容无关，因此 canonical 计算
    一律使用占位内容：采样文本可能含 </think> 等会被模板重构为
    reasoning/content 两段的内容（smoke test 实测踩坑），用真实文本
    定位结束边界不可靠。
    """
    # 逐行讲解：调用 build_prompt（build_prompt(tokenizer, messages_before_assistant)），赋给 canonical_prompt。
    canonical_prompt = build_prompt(tokenizer, messages_before_assistant)
    # 逐行讲解：构造字典 {'role': 'assistant', 'content': 'x'}，赋给 placeholder_message。
    placeholder_message = {"role": "assistant", "content": "x"}
    # 逐行讲解：构造列表 [*messages_before_assistant, placeholder_message]，赋给 messages_with_assistant。
    messages_with_assistant = [*messages_before_assistant, placeholder_message]
    # 逐行讲解：调用 _render_chat（_render_chat(tokenizer, messages_with_assistant, add_generation_prompt=False)），赋给 canonical_assistant_end。
    canonical_assistant_end = _render_chat(
        tokenizer,
        messages_with_assistant,
        add_generation_prompt=False,
    )
    # 逐行讲解：调用 _encoded_text_tokens（_encoded_text_tokens(tokenizer, 'x')），赋给 placeholder_tokens。
    placeholder_tokens = _encoded_text_tokens(tokenizer, "x")
    # 逐行讲解：构造列表 [*canonical_prompt, *placeholder_tokens]，赋给 canonical_action。
    canonical_action = [*canonical_prompt, *placeholder_tokens]
    # 逐行讲解：检查条件 canonical_assistant_end[:len(canonical_action)] != canonical_action；成立时执行下方分支。
    if canonical_assistant_end[: len(canonical_action)] != canonical_action:
        # 逐行讲解：抛出异常：ValueError('chat template 无法定位 assistant 结束边界（占位内容也不匹配）')。
        raise ValueError("chat template 无法定位 assistant 结束边界（占位内容也不匹配）")
    # 逐行讲解：按索引或切片读取 canonical_assistant_end[len(canonical_action):]，赋给 assistant_closing_tokens。
    assistant_closing_tokens = canonical_assistant_end[len(canonical_action) :]

    # 逐行讲解：调用 build_prompt（build_prompt(tokenizer, [*messages_with_assistant, next_tool_message])），赋给 canonical_next_prompt。
    canonical_next_prompt = build_prompt(
        tokenizer,
        [*messages_with_assistant, next_tool_message],
    )
    # 逐行讲解：检查条件 canonical_next_prompt[:len(canonical_assistant_end)] != canonical_assistant_end；成立时执行下方分支。
    if canonical_next_prompt[: len(canonical_assistant_end)] != canonical_assistant_end:
        # 逐行讲解：抛出异常：ValueError('加入 tool observation 后 chat template 改写了历史消息')。
        raise ValueError("加入 tool observation 后 chat template 改写了历史消息")
    # 逐行讲解：按索引或切片读取 canonical_next_prompt[len(canonical_assistant_end):]，赋给 observation_tokens。
    observation_tokens = canonical_next_prompt[len(canonical_assistant_end) :]

    # sampler 可能已经返回部分或全部 assistant 结束符，只补尚未包含的部分。
    overlap = _suffix_prefix_overlap(completion_tokens, assistant_closing_tokens)
    # 逐行讲解：返回结果：[*previous_prompt_tokens, *completion_tokens, *assistant_closing_tokens[overlap:], *observation_tokens]。
    return [
        *previous_prompt_tokens,
        *completion_tokens,
        *assistant_closing_tokens[overlap:],
        *observation_tokens,
    ]


# 逐行讲解：定义函数 parse_assistant：把 assistant 文本识别成代码调用、最终回答或非法输出。
def parse_assistant(text: str) -> ParsedAssistant:
    """把 assistant 文本识别成代码调用、最终回答或非法输出。

    合法工具调用要求：恰好一个 <tool_call>，且其后不再有其他内容。
    代码里允许出现任意字符（Python 本就有 <、>、引号），只校验非空。
    """
    # 逐行讲解：调用 list（list(TOOL_CALL_PATTERN.finditer(text))），赋给 matches。
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    # 逐行讲解：检查条件 not matches；成立时执行下方分支。
    if not matches:
        # 逐行讲解：按条件选择结果：'invalid' if '<tool_call>' in text else 'answer'，赋给 kind。
        kind = "invalid" if "<tool_call>" in text else "answer"
        # 逐行讲解：返回结果：ParsedAssistant(kind=kind, content=text.strip())。
        return ParsedAssistant(kind=kind, content=text.strip())
    # 逐行讲解：检查条件 len(matches) != 1 or text[matches[0].end():].strip()；成立时执行下方分支。
    if len(matches) != 1 or text[matches[0].end() :].strip():
        # 逐行讲解：返回结果：ParsedAssistant(kind='invalid', content=text.strip())。
        return ParsedAssistant(kind="invalid", content=text.strip())
    # 逐行讲解：调用 matches[0].group(1).strip（matches[0].group(1).strip()），赋给 code。
    code = matches[0].group(1).strip()
    # 逐行讲解：检查条件 not code；成立时执行下方分支。
    if not code:
        # 逐行讲解：返回结果：ParsedAssistant(kind='invalid', content=text.strip())。
        return ParsedAssistant(kind="invalid", content=text.strip())
    # 逐行讲解：调用 text[:matches[0].start()].strip（text[:matches[0].start()].strip()），赋给 content。
    content = text[: matches[0].start()].strip()
    # 逐行讲解：返回结果：ParsedAssistant(kind='tool', content=content, code=code)。
    return ParsedAssistant(kind="tool", content=content, code=code)


# 逐行讲解：定义函数 tool_message：构造一条结构化代码执行结果消息。
def tool_message(call_id: str, content: str) -> dict[str, Any]:
    """构造一条结构化代码执行结果消息。"""
    # 逐行讲解：返回结果：{'role': 'tool', 'tool_call_id': call_id, 'name': 'code_interpreter', 'content': content}。
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "code_interpreter",
        "content": content,
    }


# 逐行讲解：定义函数 stop_sequences：返回模型结束一轮 assistant 输出时使用的停止字符串。
def stop_sequences(tokenizer: Any) -> list[str]:
    """返回模型结束一轮 assistant 输出时使用的停止字符串。"""
    # 逐行讲解：调用 getattr（getattr(tokenizer, 'eos_token', None)），赋给 eos_token。
    eos_token = getattr(tokenizer, "eos_token", None)
    # 逐行讲解：返回结果：[eos_token] if eos_token else []。
    return [eos_token] if eos_token else []
