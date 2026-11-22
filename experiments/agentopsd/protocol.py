"""定义 ALFWorld 单工具协议和完整多轮工具上下文。"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 collections.abc 导入 Mapping, Sequence。
from collections.abc import Mapping, Sequence
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 typing 导入 Any。
from typing import Any


# 逐行讲解：构造字典 {'type': 'function', 'function': {'name': 'alfworld_step', 'description': "Execute exactly one text action …，赋给 ALFWORLD_TOOL。
ALFWORLD_TOOL = {
    "type": "function",
    "function": {
        "name": "alfworld_step",
        "description": (
            "Execute exactly one text action in the current ALFWorld state. "
            "Copy object and receptacle names exactly, including numeric suffixes such as "
            "'apple 1' and 'fridge 1'. Common command forms are: 'go to <receptacle>', "
            "'open <receptacle>', 'close <receptacle>', "
            "'take <object> from <receptacle>', 'move <object> to <receptacle>', "
            "'inventory', 'examine <thing>', 'use <object>', "
            "'heat <object> with <receptacle>', "
            "'clean <object> with <receptacle>', 'cool <object> with <receptacle>', "
            "'slice <object> with <object>', and 'look'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "One ALFWorld command. Use the exact spelling and numbering shown "
                        "by the environment. Do not include explanations in this string."
                    ),
                }
            },
            "required": ["action"],
        },
    },
}

# 逐行讲解：取值 'You are an autonomous agent in the text-only ALFWorld environment.\nComplete the household task using as f…，赋给 SYSTEM_PROMPT。
SYSTEM_PROMPT = """You are an autonomous agent in the text-only ALFWorld environment.
Complete the household task using as few valid steps as you can.

At every turn:
1. Reason from the task, complete tool history, current observation, and available actions.
2. Call alfworld_step exactly once with one action string.
3. Wait for the next environment observation before choosing another action.

Object names include instance numbers. For example, "apple 1" and "fridge 1" are complete
environment identifiers; copy them exactly. Never invent abbreviations such as "fridage 1".
Do not claim the task is complete yourself. The environment decides completion."""

# 逐行讲解：取值 '[Privileged Skill Information]'，赋给 PRIVILEGED_SKILL_HEADER。
PRIVILEGED_SKILL_HEADER = "[Privileged Skill Information]"

# 逐行讲解：调用 re.compile（re.compile('<tool_call>\\s*<function=alfworld_step>\\s*<parameter=action>\\s*(.*?)\\s*</parameter>\\s*</fun…），赋给 XML_TOOL_CALL_PATTERN。
XML_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=alfworld_step>\s*"
    r"<parameter=action>\s*(.*?)\s*</parameter>\s*"
    r"</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
# 逐行讲解：调用 re.compile（re.compile('<tool_call>\\s*(\\{.*\\})\\s*</tool_call>', re.DOTALL | re.IGNORECASE)），赋给 GENERIC_TOOL_CALL_PATTERN。
GENERIC_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


# 逐行讲解：定义类 ParsedAction：一次 assistant 输出的工具协议解析结果。
@dataclass(frozen=True)
class ParsedAction:
    """一次 assistant 输出的工具协议解析结果。"""

    # 逐行讲解：声明 valid_format 的类型为 bool。
    valid_format: bool
    # 逐行讲解：声明 action 的类型为 str | None。
    action: str | None
    # 逐行讲解：声明 reasoning 的类型为 str。
    reasoning: str


# 逐行讲解：定义函数 canonical_action：匹配官方 projection：转小写并压平多余空白。
def canonical_action(action: str) -> str:
    """匹配官方 projection：转小写并压平多余空白。"""
    # 逐行讲解：返回结果：' '.join(action.strip().lower().split())。
    return " ".join(action.strip().lower().split())


# 逐行讲解：定义函数 _parse_json_tool_call：兼容部分 tokenizer 输出的 JSON 版 ``<tool_call>``。
def _parse_json_tool_call(body: str) -> str | None:
    """兼容部分 tokenizer 输出的 JSON 版 ``<tool_call>``。"""
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(body)），赋给 value。
        value = json.loads(body)
    except json.JSONDecodeError:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：检查条件 not isinstance(value, dict) or value.get('name') != 'alfworld_step'；成立时执行下方分支。
    if not isinstance(value, dict) or value.get("name") != "alfworld_step":
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：查询指定键，缺失时使用默认值（value.get('arguments')），赋给 arguments。
    arguments = value.get("arguments")
    # 逐行讲解：检查条件 isinstance(arguments, str)；成立时执行下方分支。
    if isinstance(arguments, str):
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(arguments)），赋给 arguments。
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            # 逐行讲解：返回结果：None。
            return None
    # 逐行讲解：检查条件 not isinstance(arguments, dict) or not isinstance(arguments.get('action'), str)；成立时执行下方分支。
    if not isinstance(arguments, dict) or not isinstance(arguments.get("action"), str):
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：arguments['action']。
    return arguments["action"]


# 逐行讲解：定义函数 parse_assistant：要求恰好一个位于输出末尾的 ``alfworld_step`` 工具调用。
def parse_assistant(text: str) -> ParsedAction:
    """要求恰好一个位于输出末尾的 ``alfworld_step`` 工具调用。"""
    # 逐行讲解：检查条件 len(re.findall('<tool_call>', text, re.IGNORECASE)) != 1；成立时执行下方分支。
    if len(re.findall(r"<tool_call>", text, re.IGNORECASE)) != 1:
        # 逐行讲解：返回结果：ParsedAction(False, None, text.strip())。
        return ParsedAction(False, None, text.strip())
    # 逐行讲解：检查条件 len(re.findall('</tool_call>', text, re.IGNORECASE)) != 1；成立时执行下方分支。
    if len(re.findall(r"</tool_call>", text, re.IGNORECASE)) != 1:
        # 逐行讲解：返回结果：ParsedAction(False, None, text.strip())。
        return ParsedAction(False, None, text.strip())

    # 逐行讲解：调用 list（list(XML_TOOL_CALL_PATTERN.finditer(text))），赋给 xml_matches。
    xml_matches = list(XML_TOOL_CALL_PATTERN.finditer(text))
    # 逐行讲解：调用 list（list(GENERIC_TOOL_CALL_PATTERN.finditer(text))），赋给 json_matches。
    json_matches = list(GENERIC_TOOL_CALL_PATTERN.finditer(text))

    # 逐行讲解：声明 matches 的类型为 list[tuple[re.Match[str], str]]；构造列表 []。
    matches: list[tuple[re.Match[str], str]] = []
    # 逐行讲解：把一组元素依次加入列表（matches.extend(((match, match.group(1)) for match in xml_matches))）。
    matches.extend((match, match.group(1)) for match in xml_matches)
    # 逐行讲解：遍历 json_matches，将每项绑定到 match。
    for match in json_matches:
        # XML 参数体也会被宽松 JSON 正则忽略；这里只接收真正可解析的
        # JSON。
        action = _parse_json_tool_call(match.group(1))
        # 逐行讲解：检查条件 action is not None；成立时执行下方分支。
        if action is not None:
            # 逐行讲解：向列表末尾追加一个元素（matches.append((match, action))）。
            matches.append((match, action))

    # 逐行讲解：检查条件 len(matches) != 1；成立时执行下方分支。
    if len(matches) != 1:
        # 逐行讲解：返回结果：ParsedAction(False, None, text.strip())。
        return ParsedAction(False, None, text.strip())
    # 逐行讲解：按索引或切片读取 matches[0]，赋给 (match, raw_action)。
    match, raw_action = matches[0]
    # 逐行讲解：检查条件 text[match.end():].strip()；成立时执行下方分支。
    if text[match.end() :].strip():
        # 逐行讲解：返回结果：ParsedAction(False, None, text.strip())。
        return ParsedAction(False, None, text.strip())
    # 逐行讲解：调用 canonical_action（canonical_action(raw_action)），赋给 action。
    action = canonical_action(raw_action)
    # 逐行讲解：检查条件 not action or '\n' in raw_action.strip() or '\r' in raw_action.strip()；成立时执行下方分支。
    if not action or "\n" in raw_action.strip() or "\r" in raw_action.strip():
        # 逐行讲解：返回结果：ParsedAction(False, None, text.strip())。
        return ParsedAction(False, None, text.strip())
    # 逐行讲解：返回结果：ParsedAction(True, action, text[:match.start()].strip())。
    return ParsedAction(True, action, text[: match.start()].strip())


# 逐行讲解：定义函数 _available_actions：规范化环境动作，并排除仅用于调试的 ``help``。
def _available_actions(
    admissible_actions: Sequence[str],
) -> list[str]:
    """规范化环境动作，并排除仅用于调试的 ``help``。"""
    # 逐行讲解：返回结果：[canonical_action(action) for action in admissible_actions if action != 'help']。
    return [
        canonical_action(action)
        for action in admissible_actions
        if action != "help"
    ]


# 逐行讲解：定义函数 initial_user_prompt：构造一条轨迹的初始 user 消息。
def initial_user_prompt(
    *,
    task: str,
    observation: str,
    admissible_actions: Sequence[str],
    include_admissible_actions: bool,
) -> str:
    """构造一条轨迹的初始 user 消息。"""
    # 逐行讲解：构造列表 [f'Task: {task}', 'Current step: 1', 'Current observation:', observation]，赋给 lines。
    lines = [
        f"Task: {task}",
        "Current step: 1",
        "Current observation:",
        observation,
    ]
    # 逐行讲解：检查条件 include_admissible_actions；成立时执行下方分支。
    if include_admissible_actions:
        # 逐行讲解：调用 _available_actions（_available_actions(admissible_actions)），赋给 actions。
        actions = _available_actions(admissible_actions)
        # 逐行讲解：向列表末尾追加一个元素（lines.append('Available actions (copy one exactly):')）。
        lines.append("Available actions (copy one exactly):")
        # 逐行讲解：把一组元素依次加入列表（lines.extend((f'- {action}' for action in actions))）。
        lines.extend(f"- {action}" for action in actions)
    # 逐行讲解：向列表末尾追加一个元素（lines.append('Choose the next action by calling alfworld_step exactly once.')）。
    lines.append("Choose the next action by calling alfworld_step exactly once.")
    # 逐行讲解：返回结果：'\n'.join(lines)。
    return "\n".join(lines)


# 逐行讲解：定义函数 initial_messages：返回一条轨迹起点的 system/user 消息。
def initial_messages(**kwargs: Any) -> list[dict[str, Any]]:
    """返回一条轨迹起点的 system/user 消息。"""
    # 逐行讲解：返回结果：[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': initial_user_prompt(**kwargs)}]。
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_user_prompt(**kwargs)},
    ]


# 逐行讲解：定义函数 teacher_initial_messages：Return the same initial task with Teacher-only privileged skills.
def teacher_initial_messages(
    *,
    skill_text: str,
    task: str,
    observation: str,
    admissible_actions: Sequence[str],
    include_admissible_actions: bool,
) -> list[dict[str, Any]]:
    """Return the same initial task with Teacher-only privileged skills.

    The Student never calls this helper. The skill is prepended to the system
    content while the user task, environment observation and tool schema remain
    exactly the same.
    """

    # 逐行讲解：调用 str(skill_text).strip（str(skill_text).strip()），赋给 normalized_skill。
    normalized_skill = str(skill_text).strip()
    # 逐行讲解：取值 f'{PRIVILEGED_SKILL_HEADER}\n{normalized_skill}\n\n{SYSTEM_PROMPT}'，赋给 teacher_system。
    teacher_system = (
        f"{PRIVILEGED_SKILL_HEADER}\n{normalized_skill}\n\n{SYSTEM_PROMPT}"
    )
    # 逐行讲解：返回结果：[{'role': 'system', 'content': teacher_system}, {'role': 'user', 'content': initial_user_prompt(task=task, …。
    return [
        {"role": "system", "content": teacher_system},
        {
            "role": "user",
            "content": initial_user_prompt(
                task=task,
                observation=observation,
                admissible_actions=admissible_actions,
                include_admissible_actions=include_admissible_actions,
            ),
        },
    ]


# 逐行讲解：定义函数 environment_tool_content：把一次环境执行结果渲染为下一轮可见的 tool observation。
def environment_tool_content(
    *,
    step: int,
    observation: str,
    admissible_actions: Sequence[str],
    valid_format: bool,
    admissible: bool,
    done: bool,
    won: bool,
    include_admissible_actions: bool,
) -> str:
    """把一次环境执行结果渲染为下一轮可见的 tool observation。"""
    # 逐行讲解：检查条件 won；成立时执行下方分支。
    if won:
        # 逐行讲解：取值 'Task completed successfully.'，赋给 status。
        status = "Task completed successfully."
    # 逐行讲解：检查条件 done；成立时执行下方分支。
    elif done:
        # 逐行讲解：取值 'Episode ended without success.'，赋给 status。
        status = "Episode ended without success."
    # 逐行讲解：检查条件 not valid_format；成立时执行下方分支。
    elif not valid_format:
        # 逐行讲解：取值 'Invalid tool-call format; the environment state did not advance.'，赋给 status。
        status = "Invalid tool-call format; the environment state did not advance."
    # 逐行讲解：检查条件 not admissible；成立时执行下方分支。
    elif not admissible:
        # 逐行讲解：取值 'The action was not admissible; choose an available action.'，赋给 status。
        status = "The action was not admissible; choose an available action."
    else:
        # 逐行讲解：取值 'Action executed.'，赋给 status。
        status = "Action executed."

    # 逐行讲解：构造列表 [f'ALFWorld step {step} result: {status}', 'Current observation:', observation]，赋给 lines。
    lines = [
        f"ALFWorld step {step} result: {status}",
        "Current observation:",
        observation,
    ]
    # 逐行讲解：检查条件 include_admissible_actions and (not done)；成立时执行下方分支。
    if include_admissible_actions and not done:
        # 逐行讲解：向列表末尾追加一个元素（lines.append('Available actions (copy one exactly):')）。
        lines.append("Available actions (copy one exactly):")
        # 逐行讲解：把一组元素依次加入列表（lines.extend((f'- {action}' for action in _available_actions(admissible_actions)))）。
        lines.extend(
            f"- {action}" for action in _available_actions(admissible_actions)
        )
    # 逐行讲解：返回结果：'\n'.join(lines)。
    return "\n".join(lines)


# 逐行讲解：定义函数 tool_message：构造一条结构化 ALFWorld tool observation。
def tool_message(call_id: str, content: str) -> dict[str, Any]:
    """构造一条结构化 ALFWorld tool observation。"""
    # 逐行讲解：返回结果：{'role': 'tool', 'tool_call_id': call_id, 'name': 'alfworld_step', 'content': content}。
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "alfworld_step",
        "content": content,
    }


# 逐行讲解：定义函数 _one_dimensional_tokens：封装下方逻辑，供调用方复用。
def _one_dimensional_tokens(value: Any) -> list[int]:
    # 逐行讲解：检查条件 isinstance(value, Mapping)；成立时执行下方分支。
    if isinstance(value, Mapping):
        # 逐行讲解：按索引或切片读取 value['input_ids']，赋给 value。
        value = value["input_ids"]
    # 逐行讲解：检查条件 hasattr(value, 'tolist')；成立时执行下方分支。
    if hasattr(value, "tolist"):
        # 逐行讲解：调用 value.tolist（value.tolist()），赋给 value。
        value = value.tolist()
    # 逐行讲解：检查条件 value and isinstance(value[0], list)；成立时执行下方分支。
    if value and isinstance(value[0], list):
        # 逐行讲解：按索引或切片读取 value[0]，赋给 value。
        value = value[0]
    # 逐行讲解：返回结果：[int(token) for token in value]。
    return [int(token) for token in value]


# 逐行讲解：定义函数 _render_chat：调用模型原生 chat template，并统一为一维 token。
def _render_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """调用模型原生 chat template，并统一为一维 token。"""
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template(messages, tools=[ALFWORLD_TOOL], tokenize=True, add_generation_prompt=add_gen…），赋给 rendered。
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[ALFWORLD_TOOL],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    # 逐行讲解：返回结果：_one_dimensional_tokens(rendered)。
    return _one_dimensional_tokens(rendered)


# 逐行讲解：定义函数 build_prompt：渲染包含完整历史、工具定义和下一轮 assistant 起点的 prompt。
def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """渲染包含完整历史、工具定义和下一轮 assistant 起点的 prompt。"""
    # 逐行讲解：返回结果：_render_chat(tokenizer, messages, add_generation_prompt=True)。
    return _render_chat(tokenizer, messages, add_generation_prompt=True)


# 逐行讲解：定义函数 _encoded_text_tokens：把普通文本编码结果统一为一维 token。
def _encoded_text_tokens(tokenizer: Any, text: str) -> list[int]:
    """把普通文本编码结果统一为一维 token。"""
    # 逐行讲解：返回结果：_one_dimensional_tokens(tokenizer.encode(text, add_special_tokens=False))。
    return _one_dimensional_tokens(
        tokenizer.encode(text, add_special_tokens=False)
    )


# 逐行讲解：定义函数 _suffix_prefix_overlap：返回采样结尾与 assistant 闭合 token 开头的最长重叠长度。
def _suffix_prefix_overlap(tokens: list[int], suffix: list[int]) -> int:
    """返回采样结尾与 assistant 闭合 token 开头的最长重叠长度。"""
    # 逐行讲解：遍历 range(min(len(tokens), len(suffix)), 0, -1)，将每项绑定到 length。
    for length in range(min(len(tokens), len(suffix)), 0, -1):
        # 逐行讲解：检查条件 tokens[-length:] == suffix[:length]；成立时执行下方分支。
        if tokens[-length:] == suffix[:length]:
            # 逐行讲解：返回结果：length。
            return length
    # 逐行讲解：返回结果：0。
    return 0


# 逐行讲解：定义函数 build_next_prompt：在真实采样 token 后追加 assistant 闭合与 tool observation。
def build_next_prompt(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    """在真实采样 token 后追加 assistant 闭合与 tool observation。

    历史 assistant token 始终沿用 sampler 返回值，不把文本重新 tokenize，
    从而保证下一轮 prompt 是上一轮真实 token 序列的严格前缀扩展。
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
    # 逐行讲解：按索引或切片读取 canonical_assistant_end[len(canonical_action):]，赋给 assistant_closing_tokens。
    assistant_closing_tokens = canonical_assistant_end[len(canonical_action) :]

    # 逐行讲解：调用 build_prompt（build_prompt(tokenizer, [*messages_with_assistant, next_tool_message])），赋给 canonical_next_prompt。
    canonical_next_prompt = build_prompt(
        tokenizer,
        [*messages_with_assistant, next_tool_message],
    )
    # 逐行讲解：按索引或切片读取 canonical_next_prompt[len(canonical_assistant_end):]，赋给 observation_tokens。
    observation_tokens = canonical_next_prompt[len(canonical_assistant_end) :]

    # 逐行讲解：调用 _suffix_prefix_overlap（_suffix_prefix_overlap(completion_tokens, assistant_closing_tokens)），赋给 overlap。
    overlap = _suffix_prefix_overlap(completion_tokens, assistant_closing_tokens)
    # 逐行讲解：返回结果：[*previous_prompt_tokens, *completion_tokens, *assistant_closing_tokens[overlap:], *observation_tokens]。
    return [
        *previous_prompt_tokens,
        *completion_tokens,
        *assistant_closing_tokens[overlap:],
        *observation_tokens,
    ]


# 逐行讲解：定义函数 stop_sequences：返回一轮 assistant 生成使用的停止字符串。
def stop_sequences(tokenizer: Any) -> list[str]:
    """返回一轮 assistant 生成使用的停止字符串。"""
    # 逐行讲解：调用 getattr（getattr(tokenizer, 'eos_token', None)），赋给 eos_token。
    eos_token = getattr(tokenizer, "eos_token", None)
    # 逐行讲解：返回结果：[eos_token] if eos_token else []。
    return [eos_token] if eos_token else []
