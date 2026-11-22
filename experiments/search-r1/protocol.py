"""定义 Qwen3.5 使用的搜索工具协议。"""

# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 collections.abc 导入 Mapping。
from collections.abc import Mapping
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 typing 导入 Any。
from typing import Any


# 逐行讲解：构造字典 {'type': 'function', 'function': {'name': 'search', 'description': 'Search the web for evidence. Use a conc…，赋给 SEARCH_TOOL。
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for evidence. Use a concise English query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A concise English search query."}
            },
            "required": ["query"],
        },
    },
}

# 逐行讲解：取值 'You answer factual questions with help from a search tool.\nSearch when you need evidence. You may call se…，赋给 SYSTEM_PROMPT。
SYSTEM_PROMPT = """You answer factual questions with help from a search tool.
Search when you need evidence. You may call search several times with concise English queries.
Call search exactly once per assistant turn. Wait for the tool result before making another search call.
When ready, end with exactly one non-empty line in this format:
Answer: <your short answer>
Do not call a tool and give the final answer in the same turn.

Example of a question that needs two searches:
Question: What country was the author of The Little Prince born in?
1. Call search with query "The Little Prince author".
2. From the result, identify Antoine de Saint-Exupery.
3. Call search with query "Antoine de Saint-Exupery birthplace country".
4. From the result, identify France.
5. In a new assistant turn, give the final answer:
Answer: France"""

# 逐行讲解：调用 re.compile（re.compile('<tool_call>\\s*<function=search>\\s*<parameter=query>\\s*(.*?)\\s*</parameter>\\s*</function>\\…），赋给 TOOL_CALL_PATTERN。
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=search>\s*<parameter=query>\s*(.*?)\s*"
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
    content: str  # 普通文本：工具调用前的规划，或完整答案/非法输出
    # 逐行讲解：声明 query 的类型为 str | None；取值 None。
    query: str | None = None


# 逐行讲解：定义函数 initial_messages：为一道问题创建初始对话。
def initial_messages(question: str) -> list[dict[str, Any]]:
    """为一道问题创建初始对话。"""
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
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template(messages, tools=[SEARCH_TOOL], tokenize=True, add_generation_prompt=add_gener…），赋给 rendered。
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[SEARCH_TOOL],
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
    assistant_text: str,
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    """用真实采样 token 接上 assistant 结束符和新的 tool observation。"""
    # 逐行讲解：调用 build_prompt（build_prompt(tokenizer, messages_before_assistant)），赋给 canonical_prompt。
    canonical_prompt = build_prompt(tokenizer, messages_before_assistant)
    # Qwen3.5 会 trim 消息 content，不能靠重新编码采样文本定位结束符。
    # 用空 assistant 单独提取模板闭合 token，真实采样 token 始终原样保留。
    empty_assistant_end = _render_chat(
        tokenizer,
        [*messages_before_assistant, {"role": "assistant", "content": ""}],
        add_generation_prompt=False,
    )
    # 逐行讲解：检查条件 empty_assistant_end[:len(canonical_prompt)] != canonical_prompt；成立时执行下方分支。
    if empty_assistant_end[: len(canonical_prompt)] != canonical_prompt:
        # 逐行讲解：抛出异常：ValueError('chat template 无法从空 assistant 提取结束边界')。
        raise ValueError("chat template 无法从空 assistant 提取结束边界")
    # 逐行讲解：按索引或切片读取 empty_assistant_end[len(canonical_prompt):]，赋给 assistant_closing_tokens。
    assistant_closing_tokens = empty_assistant_end[len(canonical_prompt) :]

    # 逐行讲解：构造字典 {'role': 'assistant', 'content': assistant_text}，赋给 assistant_message。
    assistant_message = {"role": "assistant", "content": assistant_text}
    # 逐行讲解：构造列表 [*messages_before_assistant, assistant_message]，赋给 messages_with_assistant。
    messages_with_assistant = [*messages_before_assistant, assistant_message]
    # 逐行讲解：调用 _render_chat（_render_chat(tokenizer, messages_with_assistant, add_generation_prompt=False)），赋给 canonical_assistant_end。
    canonical_assistant_end = _render_chat(
        tokenizer,
        messages_with_assistant,
        add_generation_prompt=False,
    )

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


# 逐行讲解：定义函数 parse_assistant：把 assistant 文本识别成搜索调用、最终回答或非法调用。
def parse_assistant(text: str) -> ParsedAssistant:
    """把 assistant 文本识别成搜索调用、最终回答或非法调用。"""
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
    # 逐行讲解：调用 matches[0].group(1).strip（matches[0].group(1).strip()），赋给 query。
    query = matches[0].group(1).strip()
    # 逐行讲解：检查条件 not query or '<' in query or '>' in query；成立时执行下方分支。
    if not query or "<" in query or ">" in query:
        # 逐行讲解：返回结果：ParsedAssistant(kind='invalid', content=text.strip())。
        return ParsedAssistant(kind="invalid", content=text.strip())
    # 逐行讲解：调用 text[:matches[0].start()].strip（text[:matches[0].start()].strip()），赋给 content。
    content = text[: matches[0].start()].strip()
    # 逐行讲解：返回结果：ParsedAssistant(kind='tool', content=content, query=query)。
    return ParsedAssistant(kind="tool", content=content, query=query)


# 逐行讲解：定义函数 tool_message：构造一条结构化搜索结果消息。
def tool_message(call_id: str, content: str) -> dict[str, Any]:
    """构造一条结构化搜索结果消息。"""
    # 逐行讲解：返回结果：{'role': 'tool', 'tool_call_id': call_id, 'name': 'search', 'content': content}。
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "search",
        "content": content,
    }


# 逐行讲解：定义函数 stop_sequences：返回模型结束一轮 assistant 输出时使用的停止字符串。
def stop_sequences(tokenizer: Any) -> list[str]:
    """返回模型结束一轮 assistant 输出时使用的停止字符串。"""
    # 逐行讲解：调用 getattr（getattr(tokenizer, 'eos_token', None)），赋给 eos_token。
    eos_token = getattr(tokenizer, "eos_token", None)
    # 逐行讲解：返回结果：[eos_token] if eos_token else []。
    return [eos_token] if eos_token else []
