"""生成式 critic：把 macro-step 边界状态的价值估计建模为生成任务。

critic 与 actor 共享同一份采样权重，仅靠 prompt、上下文和奖励区分角色。
它可以看到 actor 看不到的特权信息（专家 walkthrough），先推理再在结尾
给出 <value>0.42</value> 形式的数值估计。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：math。
import math
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 collections.abc 导入 Mapping, Sequence。
from collections.abc import Mapping, Sequence
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio

# 逐行讲解：从 protocol 导入 stop_sequences。
from protocol import stop_sequences


# 逐行讲解：取值 'You are a value estimator for an autonomous agent in the text-only ALFWorld environment.\nThe agent is mid…，赋给 CRITIC_SYSTEM_PROMPT。
CRITIC_SYSTEM_PROMPT = """You are a value estimator for an autonomous agent in the text-only ALFWorld environment.
The agent is mid-episode. You see its task, its full interaction history, and privileged
reference information (an expert walkthrough) that the agent itself cannot see.

Estimate the probability that this agent will eventually complete the task from the
current state. Judge:
- what the agent has already accomplished and understood so far,
- whether its current hypothesis and plan still lead to a valid solution (check it
  against the walkthrough),
- what obstacles remain.

Be concise. End your reply with exactly one line of the form
<value>0.42</value>
where the number is your probability estimate between 0.00 and 1.00. Do not call tools."""

# 逐行讲解：调用 re.compile（re.compile('<value>\\s*([0-9]*\\.?[0-9]+)\\s*</value>', re.IGNORECASE)），赋给 VALUE_PATTERN。
VALUE_PATTERN = re.compile(
    r"<value>\s*([0-9]*\.?[0-9]+)\s*</value>",
    re.IGNORECASE,
)


# 逐行讲解：定义类 CriticRequest：一个状态的 K 次独立估值请求。
@dataclass(frozen=True)
class CriticRequest:
    """一个状态的 K 次独立估值请求。"""

    # 逐行讲解：声明 prompt_tokens 的类型为 list[int]。
    prompt_tokens: list[int]
    # 逐行讲解：声明 num_samples 的类型为 int。
    num_samples: int
    # 逐行讲解：声明 seed 的类型为 int。
    seed: int


# 逐行讲解：定义类 CriticSample：一次估值生成及其解析结果。
@dataclass(frozen=True)
class CriticSample:
    """一次估值生成及其解析结果。"""

    # 逐行讲解：声明 prompt_tokens 的类型为 list[int]。
    prompt_tokens: list[int]
    # 逐行讲解：声明 completion_tokens 的类型为 list[int]。
    completion_tokens: list[int]
    # 逐行讲解：声明 logprobs 的类型为 list[float]。
    logprobs: list[float]
    # 逐行讲解：声明 text 的类型为 str。
    text: str
    # 逐行讲解：声明 value 的类型为 float | None。
    value: float | None


# 逐行讲解：定义函数 parse_value：取最后一个 <value>...</value>，收敛到 [0, 1]；没有则返回 None。
def parse_value(text: str) -> float | None:
    """取最后一个 <value>...</value>，收敛到 [0, 1]；没有则返回 None。"""
    # 逐行讲解：调用 VALUE_PATTERN.findall（VALUE_PATTERN.findall(text)），赋给 matches。
    matches = VALUE_PATTERN.findall(text)
    # 逐行讲解：检查条件 not matches；成立时执行下方分支。
    if not matches:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：转换为浮点表示（float(matches[-1])），赋给 value。
    value = float(matches[-1])
    # 逐行讲解：检查条件 not math.isfinite(value)；成立时执行下方分支。
    if not math.isfinite(value):
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：min(1.0, max(0.0, value))。
    return min(1.0, max(0.0, value))


# 逐行讲解：定义函数 condensed_history：把 actor 的 chat 历史压成 critic 可读的紧凑文本。
def condensed_history(messages: Sequence[dict[str, Any]]) -> str:
    """把 actor 的 chat 历史压成 critic 可读的紧凑文本。

    tool observation 里的 Available actions 列表只服务 actor 选动作，
    对估值没有信息量，直接裁掉以控制 critic prompt 长度。
    """
    # 逐行讲解：声明 lines 的类型为 list[str]；构造列表 []。
    lines: list[str] = []
    # 逐行讲解：遍历 messages，将每项绑定到 message。
    for message in messages:
        # 逐行讲解：查询指定键，缺失时使用默认值（message.get('role')），赋给 role。
        role = message.get("role")
        # 逐行讲解：调用 str(message.get('content', '')).strip（str(message.get('content', '')).strip()），赋给 content。
        content = str(message.get("content", "")).strip()
        # 逐行讲解：检查条件 role == 'system'；成立时执行下方分支。
        if role == "system":
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：检查条件 role == 'user'；成立时执行下方分支。
        if role == "user":
            # 逐行讲解：向列表末尾追加一个元素（lines.append(f'Initial situation:\n{content}')）。
            lines.append(f"Initial situation:\n{content}")
        # 逐行讲解：检查条件 role == 'assistant'；成立时执行下方分支。
        elif role == "assistant":
            # 逐行讲解：向列表末尾追加一个元素（lines.append(f'Actor: {content}')）。
            lines.append(f"Actor: {content}")
        # 逐行讲解：检查条件 role == 'tool'；成立时执行下方分支。
        elif role == "tool":
            # 逐行讲解：调用 content.find（content.find('Available actions')），赋给 cut。
            cut = content.find("Available actions")
            # 逐行讲解：检查条件 cut >= 0；成立时执行下方分支。
            if cut >= 0:
                # 逐行讲解：调用 content[:cut].rstrip（content[:cut].rstrip()），赋给 content。
                content = content[:cut].rstrip()
            # 逐行讲解：向列表末尾追加一个元素（lines.append(f'Environment: {content}')）。
            lines.append(f"Environment: {content}")
    # 逐行讲解：返回结果：'\n\n'.join(lines)。
    return "\n\n".join(lines)


# 逐行讲解：定义函数 critic_messages：构造 critic 的一次估值 prompt。
def critic_messages(
    *,
    task: str,
    walkthrough: Sequence[str],
    history: str,
) -> list[dict[str, Any]]:
    """构造 critic 的一次估值 prompt。"""
    # 逐行讲解：取值 [f'  {index + 1}. {step}' for index, step in enumerate(walkthrough)] or ['  (empty)']，赋给 walkthrough_lines。
    walkthrough_lines = [
        f"  {index + 1}. {step}" for index, step in enumerate(walkthrough)
    ] or ["  (empty)"]
    # 逐行讲解：构造列表 ['An ALFWorld episode is in progress.', f'Task: {task}', '', 'Privileged reference (expert walkthrough, hid…，赋给 user_lines。
    user_lines = [
        "An ALFWorld episode is in progress.",
        f"Task: {task}",
        "",
        "Privileged reference (expert walkthrough, hidden from the agent):",
        *walkthrough_lines,
        "",
        "Interaction history:",
        history if history else "(The episode has just started.)",
        "",
        "Estimate the probability that the agent eventually completes the task "
        "from the current state.",
    ]
    # 逐行讲解：返回结果：[{'role': 'system', 'content': CRITIC_SYSTEM_PROMPT}, {'role': 'user', 'content': '\n'.join(user_lines)}]。
    return [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


# 逐行讲解：定义函数 _one_dimensional_tokens：封装下方逻辑，供调用方复用。
def _one_dimensional_tokens(value: Any) -> list[int]:
    # 与 protocol.py 相同：必须用 Mapping 而不是 dict，
    # 否则接不住 transformers 的 BatchEncoding（UserDict 子类）。
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


# 逐行讲解：定义函数 render_critic_prompt：渲染 critic prompt；与 actor 不同，不携带工具定义。
def render_critic_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
) -> list[int]:
    """渲染 critic prompt；与 actor 不同，不携带工具定义。"""
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)），赋给 rendered。
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # 逐行讲解：返回结果：_one_dimensional_tokens(rendered)。
    return _one_dimensional_tokens(rendered)


# 逐行讲解：定义函数 critic_prompt_for_state：从 actor 侧状态一步得到 critic prompt token。
def critic_prompt_for_state(
    tokenizer: Any,
    state_messages: Sequence[dict[str, Any]],
    *,
    task: str,
    walkthrough: Sequence[str],
) -> list[int]:
    """从 actor 侧状态一步得到 critic prompt token。"""
    # 逐行讲解：返回结果：render_critic_prompt(tokenizer, critic_messages(task=task, walkthrough=walkthrough, history=condensed_histo…。
    return render_critic_prompt(
        tokenizer,
        critic_messages(
            task=task,
            walkthrough=walkthrough,
            history=condensed_history(state_messages),
        ),
    )


# 逐行讲解：定义异步函数 _sample_async：封装下方逻辑，供调用方复用。
async def _sample_async(
    sampling_client: Any,
    request: CriticRequest,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
) -> Any:
    # 逐行讲解：返回结果：await sampling_client.sample_async(prompt=trio.ModelInput.from_ints(request.prompt_tokens), num_samples=req…。
    return await sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(request.prompt_tokens),
        num_samples=request.num_samples,
        sampling_params=trio.SamplingParams(
            max_tokens=max_tokens,
            seed=request.seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        ),
        return_text=True,
    )


# 逐行讲解：定义函数 sample_critic_values：并发采样所有请求；每个请求返回 num_samples 个估值样本。
def sample_critic_values(
    sampling_client: Any,
    tokenizer: Any,
    requests: Sequence[CriticRequest],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[list[CriticSample]]:
    """并发采样所有请求；每个请求返回 num_samples 个估值样本。"""
    # 逐行讲解：检查条件 not requests；成立时执行下方分支。
    if not requests:
        # 逐行讲解：返回结果：[]。
        return []
    # 逐行讲解：调用 stop_sequences（stop_sequences(tokenizer)），赋给 stop。
    stop = stop_sequences(tokenizer)

    # 逐行讲解：定义异步函数 run：封装下方逻辑，供调用方复用。
    async def run() -> list[Any]:
        # 逐行讲解：通过推导式逐项生成结果：[_sample_async(sampling_client, request, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=…，赋给 calls。
        calls = [
            _sample_async(
                sampling_client,
                request,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            for request in requests
        ]
        # 逐行讲解：返回结果：list(await asyncio.gather(*calls))。
        return list(await asyncio.gather(*calls))

    # 逐行讲解：调用 asyncio.run（asyncio.run(run())），赋给 responses。
    responses = asyncio.run(run())
    # 逐行讲解：声明 batches 的类型为 list[list[CriticSample]]；构造列表 []。
    batches: list[list[CriticSample]] = []
    # 逐行讲解：遍历 zip(requests, responses, strict=True)，将每项绑定到 (request, response)。
    for request, response in zip(requests, responses, strict=True):
        # 逐行讲解：调用 list（list(response.sequences)），赋给 sequences。
        sequences = list(response.sequences)
        # 逐行讲解：检查条件 len(sequences) != request.num_samples；成立时执行下方分支。
        if len(sequences) != request.num_samples:
            # 逐行讲解：抛出异常：ValueError(f'critic 采样数量 {len(sequences)} != 请求数量 {request.num_samples}')。
            raise ValueError(
                f"critic 采样数量 {len(sequences)} != 请求数量 {request.num_samples}"
            )
        # 逐行讲解：声明 samples 的类型为 list[CriticSample]；构造列表 []。
        samples: list[CriticSample] = []
        # 逐行讲解：遍历 sequences，将每项绑定到 sequence。
        for sequence in sequences:
            # 逐行讲解：通过推导式逐项生成结果：[int(token) for token in sequence.tokens]，赋给 tokens。
            tokens = [int(token) for token in sequence.tokens]
            # 逐行讲解：通过推导式逐项生成结果：[float(value) for value in sequence.logprobs]，赋给 logprobs。
            logprobs = [float(value) for value in sequence.logprobs]
            # 逐行讲解：检查条件 len(tokens) != len(logprobs)；成立时执行下方分支。
            if len(tokens) != len(logprobs):
                # 逐行讲解：抛出异常：ValueError('critic completion token 与 old logprob 长度不一致')。
                raise ValueError("critic completion token 与 old logprob 长度不一致")
            # 逐行讲解：取值 sequence.text，赋给 text。
            text = sequence.text
            # 逐行讲解：检查条件 text is None；成立时执行下方分支。
            if text is None:
                # 逐行讲解：将 token 解码为文本（tokenizer.decode(tokens, skip_special_tokens=True)），赋给 text。
                text = tokenizer.decode(tokens, skip_special_tokens=True)
            # 逐行讲解：调用 str（str(text)），赋给 text。
            text = str(text)
            # 逐行讲解：向列表末尾追加一个元素（samples.append(CriticSample(prompt_tokens=request.prompt_tokens, completion_tokens=tokens, logprobs=logprob…）。
            samples.append(
                CriticSample(
                    prompt_tokens=request.prompt_tokens,
                    completion_tokens=tokens,
                    logprobs=logprobs,
                    text=text,
                    value=parse_value(text),
                )
            )
        # 逐行讲解：向列表末尾追加一个元素（batches.append(samples)）。
        batches.append(samples)
    # 逐行讲解：返回结果：batches。
    return batches


# 逐行讲解：定义函数 endpoint_value：终点估值：解析成功的样本取均值；全部失败时悲观取 0.0。
def endpoint_value(samples: Sequence[CriticSample]) -> tuple[float, int]:
    """终点估值：解析成功的样本取均值；全部失败时悲观取 0.0。

    返回 (估值, 解析失败数)。
    """
    # 逐行讲解：通过推导式逐项生成结果：[sample.value for sample in samples if sample.value is not None]，赋给 values。
    values = [sample.value for sample in samples if sample.value is not None]
    # 逐行讲解：计算 len(samples) - len(values)，赋给 failures。
    failures = len(samples) - len(values)
    # 逐行讲解：检查条件 not values；成立时执行下方分支。
    if not values:
        # 逐行讲解：返回结果：(0.0, failures)。
        return 0.0, failures
    # 逐行讲解：返回结果：(sum(values) / len(values), failures)。
    return sum(values) / len(values), failures
