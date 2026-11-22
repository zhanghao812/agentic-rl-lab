"""Tournament-GRPO 的研究 prompt 与 XML 工具协议。

Adapted from PuffYang/Tournament_GRPO (eea826cf), Apache-2.0.
"""

# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any


# 作者 prompt.txt 原文，按作者代码读取后 strip。
SYSTEM_PROMPT = (Path(__file__).resolve().parent / "prompt.txt").read_text(encoding="utf-8").strip()
# 逐行讲解：取值 4096，赋给 MAX_PROMPT_LENGTH。
MAX_PROMPT_LENGTH = 4096
# 逐行讲解：取值 '</call_tool>'，赋给 TOOL_STOP。
TOOL_STOP = "</call_tool>"
# 低成本配置的收尾提示，放在最后一次工具 observation 内，不作为模型输出训练。
TOOL_BUDGET_NOTICE = (
    "Environment notice: tool budget exhausted; no tool calls remain. "
    "Do not call google_search or browse_webpage again. "
    "Use the evidence already collected to give your final response now: "
    "a think block followed by one answer block, following the required XML format. "
    "If evidence is incomplete, state the limitations rather than inventing facts."
)
# 逐行讲解：调用 re.compile（re.compile('<call_tool\\s+name="([^"]+)"([^>]*)>(.*?)</call_tool>', re.DOTALL)），赋给 TOOL_CALL_PATTERN。
TOOL_CALL_PATTERN = re.compile(r'<call_tool\s+name="([^"]+)"([^>]*)>(.*?)</call_tool>', re.DOTALL)
# 逐行讲解：调用 re.compile（re.compile('([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')），赋给 ATTRIBUTE_PATTERN。
ATTRIBUTE_PATTERN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
# 作者工具的 OpenAI schema。rollout 走 XML 协议不传 schema，这里只用于复现作者的数据筛长。
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "General web search for relevant webpages and snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_webpage",
            "description": "Open a specific URL and extract readable page text.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The webpage URL to browse."}},
                "required": ["url"],
            },
        },
    },
]


# 逐行讲解：定义类 ToolCall：一次 XML 工具调用，参数名与作者一致（query / url）。
@dataclass(frozen=True)
class ToolCall:
    """一次 XML 工具调用，参数名与作者一致（query / url）。"""

    # 逐行讲解：声明 name 的类型为 str。
    name: str
    # 逐行讲解：声明 arguments 的类型为 dict[str, str]。
    arguments: dict[str, str]


# 逐行讲解：定义函数 render_chat：用 chat template 渲染消息，并追加 assistant 生成前缀。
def render_chat(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> list[int]:
    """用 chat template 渲染消息，并追加 assistant 生成前缀。"""
    # 逐行讲解：返回结果：tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True, return_dict…。
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True, return_dict=False,
    )


# 逐行讲解：定义函数 build_prompt：一道研究题的初始 prompt：研究 system prompt + 原 user 问题。
def build_prompt(tokenizer: Any, query: str) -> list[int]:
    """一道研究题的初始 prompt：研究 system prompt + 原 user 问题。"""
    # 逐行讲解：构造列表 [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': query}]，赋给 messages。
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    # 逐行讲解：返回结果：render_chat(tokenizer, messages)。
    return render_chat(tokenizer, messages)


# 逐行讲解：定义函数 parse_tool_call：只执行第一个调用，对应作者的 max_parallel_calls=1。
def parse_tool_call(text: str) -> ToolCall | None:
    """只执行第一个调用，对应作者的 max_parallel_calls=1。"""
    # 逐行讲解：调用 TOOL_CALL_PATTERN.search（TOOL_CALL_PATTERN.search(text)），赋给 match。
    match = TOOL_CALL_PATTERN.search(text)
    # 逐行讲解：检查条件 match is None；成立时执行下方分支。
    if match is None:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：调用 match.groups（match.groups()），赋给 (name, attributes, body)。
    name, attributes, body = match.groups()
    # 逐行讲解：调用 name.strip（name.strip()），赋给 name。
    name = name.strip()
    # 逐行讲解：调用 dict（dict(ATTRIBUTE_PATTERN.findall(attributes))），赋给 arguments。
    arguments = dict(ATTRIBUTE_PATTERN.findall(attributes))
    # 逐行讲解：检查条件 body.strip()；成立时执行下方分支。
    if body.strip():
        # 逐行讲解：调用 body.strip（body.strip()），赋给 arguments['url' if name == 'browse_webpage' else 'query']。
        arguments["url" if name == "browse_webpage" else "query"] = body.strip()
    # 逐行讲解：返回结果：ToolCall(name, arguments)。
    return ToolCall(name, arguments)


# 逐行讲解：定义函数 truncate_tool_text：作者的按字符中间截断：超过 3000 字符时保留首尾各 1500。
def truncate_tool_text(text: str) -> str:
    """作者的按字符中间截断：超过 3000 字符时保留首尾各 1500。"""
    # 逐行讲解：返回结果：text if len(text) <= 3000 else text[:1500] + '...(truncated)...' + text[-1500:]。
    return text if len(text) <= 3000 else text[:1500] + "...(truncated)..." + text[-1500:]


# 逐行讲解：定义函数 tool_observation：observation 直接接在 assistant 后面；预算用尽时在同一 tool_output 内提示收尾。
def tool_observation(text: str, *, tools_exhausted: bool = False) -> str:
    """observation 直接接在 assistant 后面；预算用尽时在同一 tool_output 内提示收尾。"""
    # 逐行讲解：调用 text.strip（text.strip()），赋给 text。
    text = text.strip()
    # 逐行讲解：检查条件 tools_exhausted；成立时执行下方分支。
    if tools_exhausted:
        # 保留一对调用/返回，不增加新的顶层块；提示不用 XML 标签，避免干扰格式奖励。
        if text.endswith("</tool_output>"):
            # 逐行讲解：调用 text.removesuffix('</tool_output>').rstrip（text.removesuffix('</tool_output>').rstrip()），赋给 text。
            text = text.removesuffix("</tool_output>").rstrip()
        else:
            # 逐行讲解：取值 f'<tool_output>\n{text}'，赋给 text。
            text = f"<tool_output>\n{text}"  # 工具失败时返回的是没有标签的错误文本。
        # 逐行讲解：更新 text：text += f'\n\n{TOOL_BUDGET_NOTICE}\n</tool_output>'。
        text += f"\n\n{TOOL_BUDGET_NOTICE}\n</tool_output>"
    # 逐行讲解：返回结果：f'\n{text}\n'。
    return f"\n{text}\n"


# 逐行讲解：定义函数 transcript_text：还原送去格式检查的完整文本；Qwen3.5 把 <think> 放在 prompt 里，这里补回开头。
def transcript_text(tokenizer: Any, prompt_tokens: list[int], response_tokens: list[int]) -> str:
    """还原送去格式检查的完整文本；Qwen3.5 把 <think> 放在 prompt 里，这里补回开头。"""
    # 逐行讲解：调用 re.search（re.search('(<think>\\s*)$', tokenizer.decode(prompt_tokens))），赋给 prefix。
    prefix = re.search(r"(<think>\s*)$", tokenizer.decode(prompt_tokens))
    # 逐行讲解：返回结果：(prefix.group(1) if prefix else '') + tokenizer.decode(response_tokens, skip_special_tokens=True)。
    return (prefix.group(1) if prefix else "") + tokenizer.decode(response_tokens, skip_special_tokens=True)
