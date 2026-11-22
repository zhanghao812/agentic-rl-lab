"""多轮采样 Tournament-GRPO 轨迹：工具 observation 直接接在 assistant 文本后面。

Adapted from experiments/search-r1 and PuffYang/Tournament_GRPO (eea826cf), Apache-2.0.
"""

# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：从 concurrent.futures 导入 ThreadPoolExecutor。
from concurrent.futures import ThreadPoolExecutor
# 逐行讲解：从 dataclasses 导入 asdict, dataclass, field。
from dataclasses import asdict, dataclass, field
# 逐行讲解：从 typing 导入 Any, Callable。
from typing import Any, Callable

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio

# 逐行讲解：从 data 导入 ResearchExample。
from data import ResearchExample
# 逐行讲解：从 protocol 导入 MAX_PROMPT_LENGTH, TOOL_STOP, ToolCall, build_prompt, parse_tool_call, tool_observation。
from protocol import MAX_PROMPT_LENGTH, TOOL_STOP, ToolCall, build_prompt, parse_tool_call, tool_observation
# 逐行讲解：从 search 导入 ResearchTools。
from search import ResearchTools


# 逐行讲解：定义类 RolloutConfig：封装下方逻辑，供调用方复用。
@dataclass(frozen=True)
class RolloutConfig:
    # 逐行讲解：声明 group_size 的类型为 int；取值 8。
    group_size: int = 8
    # 逐行讲解：声明 max_response_tokens 的类型为 int；取值 18000。
    max_response_tokens: int = 18000  # 和作者一样，response 长度包含 observation
    # 逐行讲解：声明 max_model_tokens 的类型为 int；取值 25000。
    max_model_tokens: int = 25000
    # 逐行讲解：声明 max_assistant_turns 的类型为 int；取值 6。
    max_assistant_turns: int = 6
    # 逐行讲解：声明 max_tool_calls 的类型为 int；取值 4。
    max_tool_calls: int = 4  # 搜索和浏览合计；最后一次工具返回时提示模型收尾
    # 逐行讲解：声明 tool_concurrency 的类型为 int；取值 16。
    tool_concurrency: int = 16
    # 逐行讲解：声明 temperature 的类型为 float；取值 1.0。
    temperature: float = 1.0
    # 逐行讲解：声明 top_p 的类型为 float；取值 1.0。
    top_p: float = 1.0
    # 逐行讲解：声明 top_k 的类型为 int；取值 -1。
    top_k: int = -1
    # 逐行讲解：声明 seed 的类型为 int；取值 42。
    seed: int = 42


# 逐行讲解：定义类 Trajectory：封装下方逻辑，供调用方复用。
@dataclass
class Trajectory:
    # 逐行讲解：声明 example 的类型为 ResearchExample。
    example: ResearchExample
    # 逐行讲解：声明 question_index 的类型为 int。
    question_index: int
    # 逐行讲解：声明 group_index 的类型为 int。
    group_index: int
    # 逐行讲解：声明 prompt_tokens 的类型为 list[int]。
    prompt_tokens: list[int]
    # 逐行讲解：声明 response_tokens 的类型为 list[int]；调用 field（field(default_factory=list)）。
    response_tokens: list[int] = field(default_factory=list)
    # 逐行讲解：声明 response_mask 的类型为 list[int]；调用 field（field(default_factory=list)）。
    response_mask: list[int] = field(default_factory=list)  # 1 = 模型生成，0 = 工具 observation
    # 逐行讲解：声明 logprobs 的类型为 list[float]；调用 field（field(default_factory=list)）。
    logprobs: list[float] = field(default_factory=list)  # 采样端 logprob，observation 记 0
    # 逐行讲解：声明 assistant_turns 的类型为 int；取值 0。
    assistant_turns: int = 0
    # 逐行讲解：声明 tool_calls 的类型为 list[ToolCall]；调用 field（field(default_factory=list)）。
    tool_calls: list[ToolCall] = field(default_factory=list)
    # 逐行讲解：声明 tool_errors 的类型为 int；取值 0。
    tool_errors: int = 0
    # 逐行讲解：声明 finish_reason 的类型为 str；取值 ''。
    finish_reason: str = ""  # 非空表示轨迹结束：no_tool_call / max_turns / max_tool_calls / max_response_tokens
    # 逐行讲解：声明 reward 的类型为 float；取值 0.0。
    reward: float = 0.0
    # 逐行讲解：声明 advantage 的类型为 float；取值 0.0。
    advantage: float = 0.0

    # 逐行讲解：定义函数 done：封装下方逻辑，供调用方复用。
    @property
    def done(self) -> bool:
        # 逐行讲解：返回结果：bool(self.finish_reason)。
        return bool(self.finish_reason)


# 逐行讲解：定义函数 consume_assistant：记录本轮生成的 token 和 logprob，返回要执行的工具调用；轨迹结束时返回 None。
def consume_assistant(trajectory: Trajectory, sequence: Any, tokenizer: Any, config: RolloutConfig) -> ToolCall | None:
    """记录本轮生成的 token 和 logprob，返回要执行的工具调用；轨迹结束时返回 None。"""
    # PyTRIO 的 text 会去掉停止符，但 tokens 和 logprobs 保留了 </call_tool>，和作者 include_stop_str_in_output 一致。
    trajectory.response_tokens.extend(sequence.tokens)
    # 逐行讲解：把一组元素依次加入列表（trajectory.response_mask.extend([1] * len(sequence.tokens))）。
    trajectory.response_mask.extend([1] * len(sequence.tokens))
    # 逐行讲解：把一组元素依次加入列表（trajectory.logprobs.extend(sequence.logprobs)）。
    trajectory.logprobs.extend(sequence.logprobs)
    # 逐行讲解：更新 trajectory.assistant_turns：trajectory.assistant_turns += 1。
    trajectory.assistant_turns += 1
    # 先检查长度和轮数上限：达到上限时即使生成了工具调用也不再执行。
    if len(trajectory.response_tokens) >= config.max_response_tokens:
        # 逐行讲解：取值 'max_response_tokens'，赋给 trajectory.finish_reason。
        trajectory.finish_reason = "max_response_tokens"
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：检查条件 trajectory.assistant_turns >= config.max_assistant_turns；成立时执行下方分支。
    if trajectory.assistant_turns >= config.max_assistant_turns:
        # 逐行讲解：取值 'max_turns'，赋给 trajectory.finish_reason。
        trajectory.finish_reason = "max_turns"
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：调用 parse_tool_call（parse_tool_call(tokenizer.decode(sequence.tokens))），赋给 call。
    call = parse_tool_call(tokenizer.decode(sequence.tokens))
    # 逐行讲解：检查条件 call is None；成立时执行下方分支。
    if call is None:
        # 逐行讲解：取值 'no_tool_call'，赋给 trajectory.finish_reason。
        trajectory.finish_reason = "no_tool_call"
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：检查条件 len(trajectory.tool_calls) >= config.max_tool_calls；成立时执行下方分支。
    if len(trajectory.tool_calls) >= config.max_tool_calls:
        # 逐行讲解：取值 'max_tool_calls'，赋给 trajectory.finish_reason。
        trajectory.finish_reason = "max_tool_calls"
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：call。
    return call


# 逐行讲解：定义函数 append_observation：observation 和预算提示均不参与训练（mask 0）；放不下时直接结束。
def append_observation(trajectory: Trajectory, text: str, tokenizer: Any, config: RolloutConfig) -> None:
    """observation 和预算提示均不参与训练（mask 0）；放不下时直接结束。"""
    # 逐行讲解：调用 tool_observation（tool_observation(text, tools_exhausted=len(trajectory.tool_calls) >= config.max_tool_calls)），赋给 observation。
    observation = tool_observation(text, tools_exhausted=len(trajectory.tool_calls) >= config.max_tool_calls)
    # 逐行讲解：将文本编码为 token（tokenizer.encode(observation, add_special_tokens=False)），赋给 tokens。
    tokens = tokenizer.encode(observation, add_special_tokens=False)
    # 逐行讲解：检查条件 len(trajectory.response_tokens) + len(tokens) >= config.max_response_tokens；成立时执行下方分支。
    if len(trajectory.response_tokens) + len(tokens) >= config.max_response_tokens:
        # 逐行讲解：取值 'max_response_tokens'，赋给 trajectory.finish_reason。
        trajectory.finish_reason = "max_response_tokens"
        # 逐行讲解：返回，结束当前函数。
        return
    # 逐行讲解：把一组元素依次加入列表（trajectory.response_tokens.extend(tokens)）。
    trajectory.response_tokens.extend(tokens)
    # 逐行讲解：把一组元素依次加入列表（trajectory.response_mask.extend([0] * len(tokens))）。
    trajectory.response_mask.extend([0] * len(tokens))
    # 逐行讲解：把一组元素依次加入列表（trajectory.logprobs.extend([0.0] * len(tokens))）。
    trajectory.logprobs.extend([0.0] * len(tokens))


# 逐行讲解：定义函数 token_budget：作者的每轮生成上限：response 余量、prompt+response 余量和模型上下文余量取最小。
def token_budget(context_length: int, config: RolloutConfig) -> int:
    """作者的每轮生成上限：response 余量、prompt+response 余量和模型上下文余量取最小。"""
    # 逐行讲解：返回结果：min(config.max_response_tokens, MAX_PROMPT_LENGTH + config.max_response_tokens - context_length, config.max…。
    return min(
        config.max_response_tokens,
        MAX_PROMPT_LENGTH + config.max_response_tokens - context_length,
        config.max_model_tokens - context_length,
    )


# 逐行讲解：定义函数 sample_turn：接着当前上下文采样下一段 assistant 文本，遇到 </call_tool> 停止。
def sample_turn(sampling_client: Any, tokens: list[int], config: RolloutConfig, num_samples: int, seed: int) -> Any:
    """接着当前上下文采样下一段 assistant 文本，遇到 </call_tool> 停止。"""
    # 逐行讲解：返回结果：sampling_client.sample_async(prompt=trio.ModelInput.from_ints(tokens), num_samples=num_samples, sampling_pa…。
    return sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(tokens),
        num_samples=num_samples,
        sampling_params=trio.SamplingParams(
            max_tokens=token_budget(len(tokens), config), stop=[TOOL_STOP],
            temperature=config.temperature, top_p=config.top_p, top_k=config.top_k, seed=seed,
        ),
        return_text=False,
    )


# 逐行讲解：定义异步函数 rollout_batch_async：第一轮对共享 prompt 一次采样 group_size 条，之后每条轨迹各自继续。
async def rollout_batch_async(
    sampling_client: Any,
    tokenizer: Any,
    tools: ResearchTools,
    examples: list[ResearchExample],
    config: RolloutConfig,
    on_finished: Callable[[int], None],
) -> list[Trajectory]:
    """第一轮对共享 prompt 一次采样 group_size 条，之后每条轨迹各自继续。

    on_finished(n) 在每轮结束时报告新完成的轨迹数，供进度条使用。
    """
    # 逐行讲解：通过推导式逐项生成结果：[build_prompt(tokenizer, example.query) for example in examples]，赋给 prompts。
    prompts = [build_prompt(tokenizer, example.query) for example in examples]
    # 逐行讲解：等待异步操作 asyncio.gather(*[sample_turn(sampling_client, prompt, config, config.group_size, config.seed + index) for i… 完成并取得结果，赋给 responses。
    responses = await asyncio.gather(*[
        sample_turn(sampling_client, prompt, config, config.group_size, config.seed + index)
        for index, prompt in enumerate(prompts)
    ])
    # 逐行讲解：通过推导式逐项生成结果：[Trajectory(example, question_index, group_index, prompt) for question_index, (example, prompt) in enumerat…，赋给 trajectories。
    trajectories = [
        Trajectory(example, question_index, group_index, prompt)
        for question_index, (example, prompt) in enumerate(zip(examples, prompts, strict=True))
        for group_index in range(config.group_size)
    ]
    # sequences[i] 是第 i 条轨迹本轮新生成的内容；已结束的轨迹为 None。
    sequences = [sequence for response in responses for sequence in response.sequences]
    # 逐行讲解：取值 0，赋给 finished。
    finished = 0
    # 逐行讲解：调用 asyncio.get_running_loop（asyncio.get_running_loop()），赋给 loop。
    loop = asyncio.get_running_loop()
    # 逐行讲解：进入上下文管理器 ThreadPoolExecutor(max_workers=config.tool_concurrency)，退出时执行其收尾逻辑。
    with ThreadPoolExecutor(max_workers=config.tool_concurrency) as pool:
        # 逐行讲解：遍历 range(1, config.max_assistant_turns + 1)，将每项绑定到 turn。
        for turn in range(1, config.max_assistant_turns + 1):
            # 1. 记录本轮生成，收集需要执行的工具调用。
            pending = []
            # 逐行讲解：遍历 zip(trajectories, sequences, strict=True)，将每项绑定到 (trajectory, sequence)。
            for trajectory, sequence in zip(trajectories, sequences, strict=True):
                # 逐行讲解：检查条件 sequence is None；成立时执行下方分支。
                if sequence is None:
                    # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                    continue
                # 逐行讲解：调用 consume_assistant（consume_assistant(trajectory, sequence, tokenizer, config)），赋给 call。
                call = consume_assistant(trajectory, sequence, tokenizer, config)
                # 逐行讲解：检查条件 call is not None；成立时执行下方分支。
                if call is not None:
                    # 逐行讲解：向列表末尾追加一个元素（pending.append((trajectory, call))）。
                    pending.append((trajectory, call))

            # 2. 并发执行工具，把 observation 接到各自的上下文后面。
            results = await asyncio.gather(*[loop.run_in_executor(pool, tools.execute, call) for _, call in pending])
            # 逐行讲解：遍历 zip(pending, results, strict=True)，将每项绑定到 ((trajectory, call), result)。
            for (trajectory, call), result in zip(pending, results, strict=True):
                # 逐行讲解：向列表末尾追加一个元素（trajectory.tool_calls.append(call)）。
                trajectory.tool_calls.append(call)
                # 逐行讲解：更新 trajectory.tool_errors：trajectory.tool_errors += int(not result.ok)。
                trajectory.tool_errors += int(not result.ok)
                # 逐行讲解：调用 append_observation（append_observation(trajectory, result.text, tokenizer, config)）。
                append_observation(trajectory, result.text, tokenizer, config)

            # 逐行讲解：求和（sum((t.done for t in trajectories))），赋给 done。
            done = sum(t.done for t in trajectories)
            # 逐行讲解：调用 on_finished（on_finished(done - finished)）。
            on_finished(done - finished)
            # 逐行讲解：取值 done，赋给 finished。
            finished = done

            # 3. 未结束的轨迹各自采样下一轮。
            active = [index for index, trajectory in enumerate(trajectories) if not trajectory.done]
            # 逐行讲解：检查条件 not active；成立时执行下方分支。
            if not active:
                # 逐行讲解：结束当前循环。
                break
            # 逐行讲解：等待异步操作 asyncio.gather(*[sample_turn(sampling_client, trajectories[index].prompt_tokens + trajectories[index].respo… 完成并取得结果，赋给 responses。
            responses = await asyncio.gather(*[
                sample_turn(sampling_client, trajectories[index].prompt_tokens + trajectories[index].response_tokens,
                            config, 1, config.seed + index + turn * 10000)
                for index in active
            ])
            # 逐行讲解：计算 [None] * len(trajectories)，赋给 sequences。
            sequences = [None] * len(trajectories)
            # 逐行讲解：遍历 zip(active, responses, strict=True)，将每项绑定到 (index, response)。
            for index, response in zip(active, responses, strict=True):
                # 逐行讲解：按索引或切片读取 response.sequences[0]，赋给 sequences[index]。
                sequences[index] = response.sequences[0]

    # 每轮预算按 prompt + response 总长计算，prompt 不足 4096 时 response 可能越过上限；
    # 和作者一样，最后统一截断到 max_response_tokens。
    for trajectory in trajectories:
        # 逐行讲解：按索引或切片读取 trajectory.response_tokens[:config.max_response_tokens]，赋给 trajectory.response_tokens。
        trajectory.response_tokens = trajectory.response_tokens[:config.max_response_tokens]
        # 逐行讲解：按索引或切片读取 trajectory.response_mask[:config.max_response_tokens]，赋给 trajectory.response_mask。
        trajectory.response_mask = trajectory.response_mask[:config.max_response_tokens]
        # 逐行讲解：按索引或切片读取 trajectory.logprobs[:config.max_response_tokens]，赋给 trajectory.logprobs。
        trajectory.logprobs = trajectory.logprobs[:config.max_response_tokens]
    # 逐行讲解：返回结果：trajectories。
    return trajectories


# 逐行讲解：定义函数 rollout_batch：同步入口，和其他章节的训练循环保持一致。
def rollout_batch(
    sampling_client: Any, tokenizer: Any, tools: ResearchTools,
    examples: list[ResearchExample], config: RolloutConfig, on_finished: Callable[[int], None],
) -> list[Trajectory]:
    """同步入口，和其他章节的训练循环保持一致。"""
    # 逐行讲解：返回结果：asyncio.run(rollout_batch_async(sampling_client, tokenizer, tools, examples, config, on_finished))。
    return asyncio.run(rollout_batch_async(sampling_client, tokenizer, tools, examples, config, on_finished))


# 逐行讲解：定义函数 trajectory_record：保存一条完整轨迹，附解码后的 response 文本，方便人工复盘。
def trajectory_record(trajectory: Trajectory, tokenizer: Any, **extra: Any) -> dict[str, Any]:
    """保存一条完整轨迹，附解码后的 response 文本，方便人工复盘。"""
    # 逐行讲解：返回结果：{**asdict(trajectory), **extra, 'response_text': tokenizer.decode(trajectory.response_tokens, skip_special_…。
    return {
        **asdict(trajectory), **extra,
        "response_text": tokenizer.decode(trajectory.response_tokens, skip_special_tokens=True),
    }
