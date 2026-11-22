"""执行 Search-R1 的多轮搜索轨迹。"""

# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：copy。
import copy
# 逐行讲解：从 concurrent.futures 导入 ThreadPoolExecutor。
from concurrent.futures import ThreadPoolExecutor
# 逐行讲解：从 dataclasses 导入 dataclass, field, replace。
from dataclasses import dataclass, field, replace
# 逐行讲解：从 typing 导入 Any, Callable。
from typing import Any, Callable

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio

# 逐行讲解：从 data 导入 SearchExample。
from data import SearchExample
# 逐行讲解：从 protocol 导入 build_next_prompt, build_prompt, initial_messages, parse_assistant, stop_sequences, tool_message。
from protocol import (
    build_next_prompt,
    build_prompt,
    initial_messages,
    parse_assistant,
    stop_sequences,
    tool_message,
)
# 逐行讲解：从 reward 导入 score_answer。
from reward import score_answer
# 逐行讲解：从 search 导入 SearchClient, SearchResult, format_item。
from search import SearchClient, SearchResult, format_item


# 逐行讲解：定义类 RolloutConfig：保存轨迹长度和采样参数。
@dataclass(frozen=True)
class RolloutConfig:
    """保存轨迹长度和采样参数。"""

    # 逐行讲解：声明 group_size 的类型为 int；取值 8。
    group_size: int = 8  # 同一问题一次采样的轨迹数。
    # 逐行讲解：声明 max_search_calls 的类型为 int；取值 4。
    max_search_calls: int = 4  # 每条轨迹最多调用搜索的次数。
    # 逐行讲解：声明 max_assistant_turns 的类型为 int；取值 6。
    max_assistant_turns: int = 6  # 每条轨迹最多生成的 assistant 轮数。
    # 逐行讲解：声明 max_trajectory_tokens 的类型为 int；取值 8192。
    max_trajectory_tokens: int = 8192  # 整条轨迹允许的最大 token 数。
    # 逐行讲解：声明 max_assistant_tokens 的类型为 int；取值 1024。
    max_assistant_tokens: int = 1024  # assistant 单轮最多生成的 token 数。
    # 逐行讲解：声明 max_tool_response_tokens 的类型为 int；取值 1024。
    max_tool_response_tokens: int = 1024  # 单次搜索结果允许的最大 token 数。
    # 逐行讲解：声明 search_concurrency 的类型为 int；取值 16。
    search_concurrency: int = 16  # 不同轨迹之间并发执行的搜索请求数。
    # 逐行讲解：声明 temperature 的类型为 float；取值 1.0。
    temperature: float = 1.0  # 控制采样随机性。
    # 逐行讲解：声明 top_p 的类型为 float；取值 1.0。
    top_p: float = 1.0  # 核采样的累积概率上限。
    # 逐行讲解：声明 seed 的类型为 int；取值 42。
    seed: int = 42  # 采样使用的随机种子。


# 逐行讲解：定义类 AssistantTurn：保存一个可训练的 assistant 动作。
@dataclass
class AssistantTurn:
    """保存一个可训练的 assistant 动作。"""

    # 逐行讲解：声明 prompt_tokens 的类型为 list[int]。
    prompt_tokens: list[int]  # 本轮生成前的完整输入 token。
    # 逐行讲解：声明 completion_tokens 的类型为 list[int]。
    completion_tokens: list[int]  # 本轮 assistant 生成的 token。
    # 逐行讲解：声明 logprobs 的类型为 list[float]。
    logprobs: list[float]  # 采样时每个生成 token 的旧策略 logprob。
    # 逐行讲解：声明 text 的类型为 str。
    text: str  # 本轮生成 token 解码后的文本。


# 逐行讲解：定义类 Trajectory：保存一条完整搜索轨迹及其训练信号。
@dataclass
class Trajectory:
    """保存一条完整搜索轨迹及其训练信号。"""

    # 逐行讲解：声明 example 的类型为 SearchExample。
    example: SearchExample  # 当前轨迹对应的问题和参考答案。
    # 逐行讲解：声明 group_index 的类型为 int。
    group_index: int  # 当前轨迹在同题采样组中的编号。
    # 逐行讲解：声明 messages 的类型为 list[dict[str, Any]]。
    messages: list[dict[str, Any]]  # 当前轨迹的完整多轮消息。
    # 工具返回后供下一轮直接使用的连续 token。
    next_prompt_tokens: list[int] | None = None
    # 逐行讲解：声明 question_index 的类型为 int；取值 0。
    question_index: int = 0  # 当前问题在 rollout batch 中的编号。
    # 逐行讲解：声明 turns 的类型为 list[AssistantTurn]；调用 field（field(default_factory=list)）。
    turns: list[AssistantTurn] = field(default_factory=list)  # 已生成的 assistant 轮次。
    # 逐行讲解：声明 search_calls 的类型为 int；取值 0。
    search_calls: int = 0  # 已经调用搜索的次数。
    # 逐行讲解：声明 final_text 的类型为 str；取值 ''。
    final_text: str = ""  # 轨迹结束时的最终 assistant 输出。
    # 逐行讲解：声明 reward 的类型为 float；取值 -0.1。
    reward: float = -0.1  # 最终答案获得的奖励。
    # 逐行讲解：声明 advantage 的类型为 float；取值 0.0。
    advantage: float = 0.0  # 减去同题组平均奖励后的 advantage。
    # 逐行讲解：声明 valid_format 的类型为 bool；取值 False。
    valid_format: bool = False  # 最终答案是否符合 Answer 格式。
    # 逐行讲解：声明 exact_match 的类型为 bool；取值 False。
    exact_match: bool = False  # 最终答案是否精确命中参考答案。
    # 逐行讲解：声明 done 的类型为 bool；取值 False。
    done: bool = False  # 当前轨迹是否已经结束。


# 逐行讲解：定义类 SampleRequest：描述一批共享 prompt 的采样请求。
@dataclass(frozen=True)
class SampleRequest:
    """描述一批共享 prompt 的采样请求。"""

    # 逐行讲解：声明 trajectory_index 的类型为 int。
    trajectory_index: int  # 采样结果需要写回的轨迹编号。
    # 逐行讲解：声明 prompt_tokens 的类型为 list[int]。
    prompt_tokens: list[int]  # 本次采样的模型输入 token。
    # 逐行讲解：声明 num_samples 的类型为 int。
    num_samples: int  # 共需要采样的候选数量。
    # 逐行讲解：声明 max_tokens 的类型为 int。
    max_tokens: int  # 每个候选最多生成的 token 数。
    # 逐行讲解：声明 seed 的类型为 int。
    seed: int  # 本次采样使用的随机种子。


# 逐行讲解：定义类 PendingSearch：保存一条等待并发执行的轨迹搜索。
@dataclass(frozen=True)
class PendingSearch:
    """保存一条等待并发执行的轨迹搜索。"""

    # 逐行讲解：声明 trajectory 的类型为 Trajectory。
    trajectory: Trajectory
    # 逐行讲解：声明 messages_before_assistant 的类型为 list[dict[str, Any]]。
    messages_before_assistant: list[dict[str, Any]]
    # 逐行讲解：声明 assistant_text 的类型为 str。
    assistant_text: str
    # 逐行讲解：声明 prompt_tokens 的类型为 list[int]。
    prompt_tokens: list[int]
    # 逐行讲解：声明 completion_tokens 的类型为 list[int]。
    completion_tokens: list[int]
    # 逐行讲解：声明 call_id 的类型为 str。
    call_id: str
    # 逐行讲解：声明 query 的类型为 str。
    query: str


# 逐行讲解：定义异步函数 sample_requests_async：只并发执行 PyTRIO 的 sample_async 请求。
async def sample_requests_async(
    sampling_client: Any,
    requests: list[SampleRequest],
    config: RolloutConfig,
    tokenizer: Any,
) -> list[Any]:
    """只并发执行 PyTRIO 的 sample_async 请求。"""
    # 逐行讲解：构造列表 []，赋给 tasks。
    tasks = []
    # 逐行讲解：遍历 requests，将每项绑定到 request。
    for request in requests:
        # 逐行讲解：调用 trio.SamplingParams（trio.SamplingParams(max_tokens=request.max_tokens, seed=request.seed, stop=stop_sequences(tokenizer), tempe…），赋给 params。
        params = trio.SamplingParams(
            max_tokens=request.max_tokens,
            seed=request.seed,
            stop=stop_sequences(tokenizer),
            temperature=config.temperature,
            top_p=config.top_p,
        )
        # 逐行讲解：向列表末尾追加一个元素（tasks.append(sampling_client.sample_async(prompt=trio.ModelInput.from_ints(request.prompt_tokens), num_samp…）。
        tasks.append(
            sampling_client.sample_async(
                prompt=trio.ModelInput.from_ints(request.prompt_tokens),
                num_samples=request.num_samples,
                sampling_params=params,
                return_text=True,
            )
        )
    # 逐行讲解：返回结果：list(await asyncio.gather(*tasks))。
    return list(await asyncio.gather(*tasks))


# 逐行讲解：定义函数 token_count：统计一段普通文本的 token 数。
def token_count(tokenizer: Any, text: str) -> int:
    """统计一段普通文本的 token 数。"""
    # 逐行讲解：将文本编码为 token（tokenizer.encode(text, add_special_tokens=False)），赋给 tokens。
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # 逐行讲解：返回结果：len(tokens)。
    return len(tokens)


# 逐行讲解：定义函数 fit_tool_content：优先保留完整条目；首条过长时缩短正文，保留标题和来源。
def fit_tool_content(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    assistant_text: str,
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    call_id: str,
    result: SearchResult,
    config: RolloutConfig,
) -> tuple[str, list[int]] | None:
    """优先保留完整条目；首条过长时缩短正文，保留标题和来源。"""
    # 逐行讲解：检查条件 not result.ok；成立时执行下方分支。
    if not result.ok:
        # 逐行讲解：构造列表 [f"Search error: {result.error or 'unknown error'}"]，赋给 candidates。
        candidates = [f"Search error: {result.error or 'unknown error'}"]
    # 逐行讲解：检查条件 not result.items；成立时执行下方分支。
    elif not result.items:
        # 逐行讲解：构造列表 ['Search returned no results.']，赋给 candidates。
        candidates = ["Search returned no results."]
    else:
        # 逐行讲解：通过推导式逐项生成结果：[format_item(item, index) for index, item in enumerate(result.items, 1)]，赋给 candidates。
        candidates = [format_item(item, index) for index, item in enumerate(result.items, 1)]

    # 逐行讲解：定义函数 fitting_prompt：封装下方逻辑，供调用方复用。
    def fitting_prompt(content: str) -> list[int] | None:
        # 逐行讲解：检查条件 token_count(tokenizer, content) > config.max_tool_response_tokens；成立时执行下方分支。
        if token_count(tokenizer, content) > config.max_tool_response_tokens:
            # 逐行讲解：返回结果：None。
            return None
        # 逐行讲解：调用 tool_message（tool_message(call_id, content)），赋给 next_tool_message。
        next_tool_message = tool_message(call_id, content)
        # 逐行讲解：调用 build_next_prompt（build_next_prompt(tokenizer, messages_before_assistant, assistant_text, previous_prompt_tokens, completion_…），赋给 next_prompt。
        next_prompt = build_next_prompt(
            tokenizer,
            messages_before_assistant,
            assistant_text,
            previous_prompt_tokens,
            completion_tokens,
            next_tool_message,
        )
        # 逐行讲解：检查条件 len(next_prompt) > config.max_trajectory_tokens；成立时执行下方分支。
        if len(next_prompt) > config.max_trajectory_tokens:
            # 逐行讲解：返回结果：None。
            return None
        # 逐行讲解：返回结果：next_prompt。
        return next_prompt

    # 逐行讲解：声明 accepted 的类型为 list[str]；构造列表 []。
    accepted: list[str] = []
    # 逐行讲解：声明 accepted_prompt 的类型为 list[int] | None；取值 None。
    accepted_prompt: list[int] | None = None
    # 逐行讲解：遍历 candidates，将每项绑定到 candidate。
    for candidate in candidates:
        # 逐行讲解：调用 '\n\n'.join（'\n\n'.join([*accepted, candidate])），赋给 content。
        content = "\n\n".join([*accepted, candidate])
        # 逐行讲解：调用 fitting_prompt（fitting_prompt(content)），赋给 next_prompt。
        next_prompt = fitting_prompt(content)
        # 逐行讲解：检查条件 next_prompt is None；成立时执行下方分支。
        if next_prompt is None:
            # 逐行讲解：结束当前循环。
            break
        # 逐行讲解：向列表末尾追加一个元素（accepted.append(candidate)）。
        accepted.append(candidate)
        # 逐行讲解：取值 next_prompt，赋给 accepted_prompt。
        accepted_prompt = next_prompt
    # 逐行讲解：检查条件 accepted and accepted_prompt is not None；成立时执行下方分支。
    if accepted and accepted_prompt is not None:
        # 逐行讲解：返回结果：('\n\n'.join(accepted), accepted_prompt)。
        return "\n\n".join(accepted), accepted_prompt

    # 缺少 snippet 时回退的 summary 可能很长，不能因此丢掉整次搜索。
    if result.ok and result.items:
        # 逐行讲解：按索引或切片读取 result.items[0]，赋给 item。
        item = result.items[0]
        # 逐行讲解：将文本编码为 token（tokenizer.encode(item.content, add_special_tokens=False)），赋给 tokens。
        tokens = tokenizer.encode(item.content, add_special_tokens=False)
        # 逐行讲解：构造元组 (1, len(tokens))，赋给 (low, high)。
        low, high = 1, len(tokens)
        # 逐行讲解：声明 fitted 的类型为 tuple[str, list[int]] | None；取值 None。
        fitted: tuple[str, list[int]] | None = None
        # 逐行讲解：当 low <= high 成立时继续循环。
        while low <= high:
            # 逐行讲解：计算 (low + high) // 2，赋给 middle。
            middle = (low + high) // 2
            # 逐行讲解：调用 replace（replace(item, content=tokenizer.decode(tokens[:middle]) + '\n[truncated]')），赋给 shortened。
            shortened = replace(
                item,
                content=tokenizer.decode(tokens[:middle]) + "\n[truncated]",
            )
            # 逐行讲解：调用 format_item（format_item(shortened, 1)），赋给 content。
            content = format_item(shortened, 1)
            # 逐行讲解：调用 fitting_prompt（fitting_prompt(content)），赋给 next_prompt。
            next_prompt = fitting_prompt(content)
            # 逐行讲解：检查条件 next_prompt is None；成立时执行下方分支。
            if next_prompt is None:
                # 逐行讲解：计算 middle - 1，赋给 high。
                high = middle - 1
            else:
                # 逐行讲解：构造元组 (content, next_prompt)，赋给 fitted。
                fitted = content, next_prompt
                # 逐行讲解：计算 middle + 1，赋给 low。
                low = middle + 1
        # 逐行讲解：返回结果：fitted。
        return fitted
    # 逐行讲解：返回结果：None。
    return None


# 逐行讲解：定义函数 make_request：按当前真实 prompt 长度创建动态采样请求。
def make_request(
    tokenizer: Any,
    trajectory: Trajectory,
    trajectory_index: int,
    num_samples: int,
    seed: int,
    config: RolloutConfig,
) -> SampleRequest | None:
    """按当前真实 prompt 长度创建动态采样请求。"""
    # 逐行讲解：按条件选择结果：trajectory.next_prompt_tokens if trajectory.next_prompt_tokens is not None else build_prompt(tokenizer, tra…，赋给 prompt_tokens。
    prompt_tokens = (
        trajectory.next_prompt_tokens
        if trajectory.next_prompt_tokens is not None
        else build_prompt(tokenizer, trajectory.messages)
    )
    # 逐行讲解：调用 min（min(config.max_assistant_tokens, config.max_trajectory_tokens - len(prompt_tokens))），赋给 max_tokens。
    max_tokens = min(
        config.max_assistant_tokens,
        config.max_trajectory_tokens - len(prompt_tokens),
    )
    # 逐行讲解：检查条件 max_tokens <= 0；成立时执行下方分支。
    if max_tokens <= 0:
        # 逐行讲解：取值 True，赋给 trajectory.done。
        trajectory.done = True
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：SampleRequest(trajectory_index, prompt_tokens, num_samples, max_tokens, seed)。
    return SampleRequest(trajectory_index, prompt_tokens, num_samples, max_tokens, seed)


# 逐行讲解：定义函数 read_sequence：读取采样序列并校验 token 与旧策略 logprob 对齐。
def read_sequence(sequence: Any, tokenizer: Any) -> tuple[list[int], list[float], str]:
    """读取采样序列并校验 token 与旧策略 logprob 对齐。"""
    # 逐行讲解：通过推导式逐项生成结果：[int(token) for token in sequence.tokens]，赋给 tokens。
    tokens = [int(token) for token in sequence.tokens]
    # 逐行讲解：通过推导式逐项生成结果：[float(value) for value in sequence.logprobs]，赋给 logprobs。
    logprobs = [float(value) for value in sequence.logprobs]
    # 逐行讲解：检查条件 len(tokens) != len(logprobs)；成立时执行下方分支。
    if len(tokens) != len(logprobs):
        # 逐行讲解：抛出异常：ValueError('采样 token 与 logprob 长度不一致')。
        raise ValueError("采样 token 与 logprob 长度不一致")
    # 逐行讲解：取值 sequence.text，赋给 text。
    text = sequence.text
    # 逐行讲解：检查条件 text is None；成立时执行下方分支。
    if text is None:
        # 逐行讲解：将 token 解码为文本（tokenizer.decode(tokens, skip_special_tokens=True)），赋给 text。
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    # 逐行讲解：返回结果：(tokens, logprobs, text)。
    return tokens, logprobs, text


# 逐行讲解：定义函数 consume_assistant：消费一次 assistant 输出，返回需要执行的搜索或直接结束轨迹。
def consume_assistant(
    trajectory: Trajectory,
    prompt_tokens: list[int],
    sequence: Any,
    tokenizer: Any,
    config: RolloutConfig,
) -> PendingSearch | None:
    """消费一次 assistant 输出，返回需要执行的搜索或直接结束轨迹。"""
    # 逐行讲解：调用 read_sequence（read_sequence(sequence, tokenizer)），赋给 (tokens, logprobs, text)。
    tokens, logprobs, text = read_sequence(sequence, tokenizer)
    # 逐行讲解：向列表末尾追加一个元素（trajectory.turns.append(AssistantTurn(prompt_tokens, tokens, logprobs, text))）。
    trajectory.turns.append(AssistantTurn(prompt_tokens, tokens, logprobs, text))
    # 逐行讲解：调用 parse_assistant（parse_assistant(text)），赋给 parsed。
    parsed = parse_assistant(text)

    # 逐行讲解：取值 parsed.kind == 'tool' and trajectory.search_calls < config.max_search_calls and (len(trajectory.turns) < co…，赋给 can_search。
    can_search = (
        parsed.kind == "tool"
        and trajectory.search_calls < config.max_search_calls
        and len(trajectory.turns) < config.max_assistant_turns
    )
    # 逐行讲解：检查条件 not can_search；成立时执行下方分支。
    if not can_search:
        # 逐行讲解：向列表末尾追加一个元素（trajectory.messages.append({'role': 'assistant', 'content': text})）。
        trajectory.messages.append({"role": "assistant", "content": text})
        # 逐行讲解：取值 text，赋给 trajectory.final_text。
        trajectory.final_text = text
        # 逐行讲解：取值 True，赋给 trajectory.done。
        trajectory.done = True
        # 逐行讲解：返回结果：None。
        return None

    # 逐行讲解：取值 f'search-{trajectory.question_index}-{trajectory.group_index}-{trajectory.search_calls + 1}'，赋给 call_id。
    call_id = (
        f"search-{trajectory.question_index}-{trajectory.group_index}-"
        f"{trajectory.search_calls + 1}"
    )
    # 逐行讲解：调用 list（list(trajectory.messages)），赋给 messages_before_assistant。
    messages_before_assistant = list(trajectory.messages)
    # messages 用于协议和结果记录；下一轮模型输入则由真实采样 token 连续构造。
    trajectory.messages.append({"role": "assistant", "content": text})
    # 逐行讲解：返回结果：PendingSearch(trajectory=trajectory, messages_before_assistant=messages_before_assistant, assistant_text=te…。
    return PendingSearch(
        trajectory=trajectory,
        messages_before_assistant=messages_before_assistant,
        assistant_text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=tokens,
        call_id=call_id,
        query=parsed.query or "",
    )


# 逐行讲解：定义函数 finish_search：把一条搜索 observation 接回对应轨迹，并返回轨迹是否结束。
def finish_search(
    pending: PendingSearch,
    result: SearchResult,
    tokenizer: Any,
    config: RolloutConfig,
) -> bool:
    """把一条搜索 observation 接回对应轨迹，并返回轨迹是否结束。"""
    # 逐行讲解：调用 fit_tool_content（fit_tool_content(tokenizer, pending.messages_before_assistant, pending.assistant_text, pending.prompt_token…），赋给 fitted。
    fitted = fit_tool_content(
        tokenizer,
        pending.messages_before_assistant,
        pending.assistant_text,
        pending.prompt_tokens,
        pending.completion_tokens,
        pending.call_id,
        result,
        config,
    )
    # 逐行讲解：检查条件 fitted is None；成立时执行下方分支。
    if fitted is None:
        # 逐行讲解：取值 pending.assistant_text，赋给 pending.trajectory.final_text。
        pending.trajectory.final_text = pending.assistant_text
        # 逐行讲解：取值 True，赋给 pending.trajectory.done。
        pending.trajectory.done = True
        # 逐行讲解：返回结果：True。
        return True
    # 逐行讲解：取值 fitted，赋给 (content, next_prompt_tokens)。
    content, next_prompt_tokens = fitted
    # 逐行讲解：向列表末尾追加一个元素（pending.trajectory.messages.append(tool_message(pending.call_id, content))）。
    pending.trajectory.messages.append(tool_message(pending.call_id, content))
    # 逐行讲解：取值 next_prompt_tokens，赋给 pending.trajectory.next_prompt_tokens。
    pending.trajectory.next_prompt_tokens = next_prompt_tokens
    # 逐行讲解：更新 pending.trajectory.search_calls：pending.trajectory.search_calls += 1。
    pending.trajectory.search_calls += 1
    # 逐行讲解：返回结果：False。
    return False


# 逐行讲解：定义函数 resolve_searches：并发执行不同轨迹的搜索，再按原顺序把 observation 接回轨迹。
def resolve_searches(
    pending_searches: list[PendingSearch],
    search_client: SearchClient,
    tokenizer: Any,
    config: RolloutConfig,
) -> int:
    """并发执行不同轨迹的搜索，再按原顺序把 observation 接回轨迹。"""
    # 逐行讲解：检查条件 not pending_searches；成立时执行下方分支。
    if not pending_searches:
        # 逐行讲解：返回结果：0。
        return 0
    # 逐行讲解：调用 min（min(config.search_concurrency, len(pending_searches))），赋给 workers。
    workers = min(config.search_concurrency, len(pending_searches))
    # 逐行讲解：进入上下文管理器 ThreadPoolExecutor(max_workers=workers)，退出时执行其收尾逻辑。
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # 逐行讲解：调用 list（list(pool.map(search_client.search, [pending.query for pending in pending_searches]))），赋给 results。
        results = list(
            pool.map(
                search_client.search,
                [pending.query for pending in pending_searches],
            )
        )
    # 逐行讲解：返回结果：sum((finish_search(pending, result, tokenizer, config) for pending, result in zip(pending_searches, results…。
    return sum(
        finish_search(pending, result, tokenizer, config)
        for pending, result in zip(pending_searches, results, strict=True)
    )


# 逐行讲解：定义函数 score_trajectory：给完成的轨迹写入格式与精确匹配奖励。
def score_trajectory(trajectory: Trajectory) -> None:
    """给完成的轨迹写入格式与精确匹配奖励。"""
    # 逐行讲解：调用 score_answer（score_answer(trajectory.final_text, trajectory.example.answers)），赋给 result。
    result = score_answer(trajectory.final_text, trajectory.example.answers)
    # 逐行讲解：取值 result.reward，赋给 trajectory.reward。
    trajectory.reward = result.reward
    # 逐行讲解：取值 result.valid_format，赋给 trajectory.valid_format。
    trajectory.valid_format = result.valid_format
    # 逐行讲解：取值 result.exact_match，赋给 trajectory.exact_match。
    trajectory.exact_match = result.exact_match


# 逐行讲解：定义函数 assign_group_advantages：按同题 group 的平均 reward 计算中心化 advantage。
def assign_group_advantages(trajectories: list[Trajectory]) -> int:
    """按同题 group 的平均 reward 计算中心化 advantage。"""
    # 逐行讲解：声明 groups 的类型为 dict[int, list[Trajectory]]；构造字典 {}。
    groups: dict[int, list[Trajectory]] = {}
    # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
    for trajectory in trajectories:
        # 逐行讲解：向列表末尾追加一个元素（groups.setdefault(trajectory.question_index, []).append(trajectory)）。
        groups.setdefault(trajectory.question_index, []).append(trajectory)
    # 逐行讲解：取值 0，赋给 degenerate。
    degenerate = 0
    # 逐行讲解：遍历 groups.values()，将每项绑定到 group。
    for group in groups.values():
        # 逐行讲解：计算 sum((item.reward for item in group)) / len(group)，赋给 mean_reward。
        mean_reward = sum(item.reward for item in group) / len(group)
        # 逐行讲解：遍历 group，将每项绑定到 item。
        for item in group:
            # 逐行讲解：计算 item.reward - mean_reward，赋给 item.advantage。
            item.advantage = item.reward - mean_reward
        # 逐行讲解：检查条件 all((item.advantage == 0.0 for item in group))；成立时执行下方分支。
        if all(item.advantage == 0.0 for item in group):
            # 逐行讲解：更新 degenerate：degenerate += 1。
            degenerate += 1
    # 逐行讲解：返回结果：degenerate。
    return degenerate


# 逐行讲解：定义函数 rollout_batch：同步控制多轮状态机，并在轨迹结束时可选上报完成数量。
def rollout_batch(
    sampling_client: Any,
    tokenizer: Any,
    search_client: SearchClient,
    examples: list[SearchExample],
    config: RolloutConfig,
    progress_callback: Callable[[int], None] | None = None,
) -> list[Trajectory]:
    """同步控制多轮状态机，并在轨迹结束时可选上报完成数量。"""
    # 逐行讲解：检查条件 config.search_concurrency < 1；成立时执行下方分支。
    if config.search_concurrency < 1:
        # 逐行讲解：抛出异常：ValueError('search_concurrency 必须大于等于 1')。
        raise ValueError("search_concurrency 必须大于等于 1")
    # 每个问题先创建一条共享初始 prompt 的根轨迹。
    roots = [
        Trajectory(
            example=example,
            group_index=0,
            messages=initial_messages(example.question),
            question_index=question_index,
        )
        for question_index, example in enumerate(examples)
    ]

    # 首轮每个问题只发一个请求，用 num_samples=group_size 生成多个分支。
    first_requests: list[SampleRequest] = []
    # 逐行讲解：遍历 enumerate(roots)，将每项绑定到 (index, trajectory)。
    for index, trajectory in enumerate(roots):
        # 逐行讲解：调用 make_request（make_request(tokenizer, trajectory, index, config.group_size, config.seed + index, config)），赋给 request。
        request = make_request(
            tokenizer,
            trajectory,
            index,
            config.group_size,
            config.seed + index,
            config,
        )
        # 逐行讲解：检查条件 request；成立时执行下方分支。
        if request:
            # 逐行讲解：向列表末尾追加一个元素（first_requests.append(request)）。
            first_requests.append(request)

    # 逐行讲解：声明 trajectories 的类型为 list[Trajectory]；构造列表 []。
    trajectories: list[Trajectory] = []
    # 逐行讲解：检查条件 first_requests；成立时执行下方分支。
    if first_requests:
        # 不同问题的首轮模型采样并发执行。
        responses = asyncio.run(
            sample_requests_async(sampling_client, first_requests, config, tokenizer)
        )
        # 逐行讲解：声明 pending_searches 的类型为 list[PendingSearch]；构造列表 []。
        pending_searches: list[PendingSearch] = []
        # 逐行讲解：遍历 zip(first_requests, responses, strict=True)，将每项绑定到 (request, response)。
        for request, response in zip(first_requests, responses, strict=True):
            # 逐行讲解：按索引或切片读取 roots[request.trajectory_index]，赋给 root。
            root = roots[request.trajectory_index]
            # 逐行讲解：检查条件 len(response.sequences) != config.group_size；成立时执行下方分支。
            if len(response.sequences) != config.group_size:
                # 逐行讲解：抛出异常：ValueError('首轮采样数量与 group_size 不一致')。
                raise ValueError("首轮采样数量与 group_size 不一致")
            # 逐行讲解：遍历 enumerate(response.sequences)，将每项绑定到 (group_index, sequence)。
            for group_index, sequence in enumerate(response.sequences):
                # 深拷贝保证每个分支拥有独立的 messages 和搜索历史。
                branch = copy.deepcopy(root)
                # 逐行讲解：取值 group_index，赋给 branch.group_index。
                branch.group_index = group_index
                # 先解析全部分支，再并发执行其中需要的搜索。
                pending = consume_assistant(
                    branch,
                    request.prompt_tokens,
                    sequence,
                    tokenizer,
                    config,
                )
                # 逐行讲解：向列表末尾追加一个元素（trajectories.append(branch)）。
                trajectories.append(branch)
                # 逐行讲解：检查条件 pending is not None；成立时执行下方分支。
                if pending is not None:
                    # 逐行讲解：向列表末尾追加一个元素（pending_searches.append(pending)）。
                    pending_searches.append(pending)
                # 逐行讲解：检查条件 branch.done and progress_callback is not None；成立时执行下方分支。
                if branch.done and progress_callback is not None:
                    # 逐行讲解：调用 progress_callback（progress_callback(1)）。
                    progress_callback(1)
        # 逐行讲解：调用 resolve_searches（resolve_searches(pending_searches, search_client, tokenizer, config)），赋给 completed。
        completed = resolve_searches(
            pending_searches,
            search_client,
            tokenizer,
            config,
        )
        # 逐行讲解：检查条件 completed and progress_callback is not None；成立时执行下方分支。
        if completed and progress_callback is not None:
            # 逐行讲解：调用 progress_callback（progress_callback(completed)）。
            progress_callback(completed)

    # 首轮分叉后 prompt 已经不同，后续每条未结束轨迹单独采样一个结果。
    while any(not trajectory.done for trajectory in trajectories):
        # 逐行讲解：声明 requests 的类型为 list[SampleRequest]；构造列表 []。
        requests: list[SampleRequest] = []
        # 逐行讲解：遍历 enumerate(trajectories)，将每项绑定到 (index, trajectory)。
        for index, trajectory in enumerate(trajectories):
            # 逐行讲解：检查条件 trajectory.done；成立时执行下方分支。
            if trajectory.done:
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：调用 make_request（make_request(tokenizer, trajectory, index, 1, config.seed + index + len(trajectory.turns) * 10000, config)），赋给 request。
            request = make_request(
                tokenizer,
                trajectory,
                index,
                1,
                config.seed + index + len(trajectory.turns) * 10_000,
                config,
            )
            # 逐行讲解：检查条件 request；成立时执行下方分支。
            if request:
                # 逐行讲解：向列表末尾追加一个元素（requests.append(request)）。
                requests.append(request)
            # 逐行讲解：检查条件 trajectory.done and progress_callback is not None；成立时执行下方分支。
            elif trajectory.done and progress_callback is not None:
                # make_request 会在轨迹 token 预算耗尽时直接将其标记为结束。
                progress_callback(1)
        # 逐行讲解：检查条件 not requests；成立时执行下方分支。
        if not requests:
            # 逐行讲解：结束当前循环。
            break
        # 各轨迹 prompt 不同，但仍通过多个 sample_async 并发采样。
        responses = asyncio.run(sample_requests_async(sampling_client, requests, config, tokenizer))
        # 逐行讲解：构造列表 []，赋给 pending_searches。
        pending_searches = []
        # 逐行讲解：遍历 zip(requests, responses, strict=True)，将每项绑定到 (request, response)。
        for request, response in zip(requests, responses, strict=True):
            # 逐行讲解：检查条件 len(response.sequences) != 1；成立时执行下方分支。
            if len(response.sequences) != 1:
                # 逐行讲解：抛出异常：ValueError('后续轮次每条轨迹必须只采样一个分支')。
                raise ValueError("后续轮次每条轨迹必须只采样一个分支")
            # 逐行讲解：按索引或切片读取 trajectories[request.trajectory_index]，赋给 trajectory。
            trajectory = trajectories[request.trajectory_index]
            # 逐行讲解：调用 consume_assistant（consume_assistant(trajectory, request.prompt_tokens, response.sequences[0], tokenizer, config)），赋给 pending。
            pending = consume_assistant(
                trajectory,
                request.prompt_tokens,
                response.sequences[0],
                tokenizer,
                config,
            )
            # 逐行讲解：检查条件 pending is not None；成立时执行下方分支。
            if pending is not None:
                # 逐行讲解：向列表末尾追加一个元素（pending_searches.append(pending)）。
                pending_searches.append(pending)
            # 逐行讲解：检查条件 trajectory.done and progress_callback is not None；成立时执行下方分支。
            if trajectory.done and progress_callback is not None:
                # 逐行讲解：调用 progress_callback（progress_callback(1)）。
                progress_callback(1)
        # 逐行讲解：调用 resolve_searches（resolve_searches(pending_searches, search_client, tokenizer, config)），赋给 completed。
        completed = resolve_searches(
            pending_searches,
            search_client,
            tokenizer,
            config,
        )
        # 逐行讲解：检查条件 completed and progress_callback is not None；成立时执行下方分支。
        if completed and progress_callback is not None:
            # 逐行讲解：调用 progress_callback（progress_callback(completed)）。
            progress_callback(completed)

    # 所有轨迹结束后先计算奖励，再按同题 group 计算中心化 advantage。
    for trajectory in trajectories:
        # 逐行讲解：调用 score_trajectory（score_trajectory(trajectory)）。
        score_trajectory(trajectory)
    # 逐行讲解：调用 assign_group_advantages（assign_group_advantages(trajectories)）。
    assign_group_advantages(trajectories)
    # 逐行讲解：返回结果：trajectories。
    return trajectories
