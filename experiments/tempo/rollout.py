"""从保存的 macro-step 起点采样 N 条长度为 H 的分支轨迹。

与第 8 篇完整轨迹 rollout 的两点区别：
1. 每条分支只向前走 H 轮，停在 macro_boundary 而不是终局；
2. 起点可以是一个保存的中间状态，环境通过重放动作历史恢复，
   续跑 prompt 使用保存的真实 token 前缀，保证严格前缀扩展。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：从 concurrent.futures 导入 ThreadPoolExecutor。
from concurrent.futures import ThreadPoolExecutor
# 逐行讲解：从 dataclasses 导入 dataclass, field。
from dataclasses import dataclass, field
# 逐行讲解：从 typing 导入 Any, Callable。
from typing import Any, Callable

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio

# 逐行讲解：从 data 导入 GameExample。
from data import GameExample
# 逐行讲解：从 environment 导入 ALFWorldGroup, EnvironmentStep。
from environment import ALFWorldGroup, EnvironmentStep
# 逐行讲解：从 protocol 导入 build_next_prompt, build_prompt, environment_tool_content, initial_messages, parse_assistant, stop_sequences, tool_message。
from protocol import (
    build_next_prompt,
    build_prompt,
    environment_tool_content,
    initial_messages,
    parse_assistant,
    stop_sequences,
    tool_message,
)
# 逐行讲解：从 states 导入 MacroState。
from states import MacroState


# 逐行讲解：取值 '__invalid_tool_call__'，赋给 INVALID_ENV_ACTION。
INVALID_ENV_ACTION = "__invalid_tool_call__"
# 逐行讲解：取值 12000，赋给 MAX_SEQUENCE_TOKENS。
MAX_SEQUENCE_TOKENS = 12_000


# 逐行讲解：定义类 RolloutConfig：保存环境、macro-step 和采样参数。
@dataclass(frozen=True)
class RolloutConfig:
    """保存环境、macro-step 和采样参数。"""

    # 逐行讲解：声明 branches 的类型为 int；取值 4。
    branches: int = 4                 # N：每个起点的分支数（actor 的 group 大小）
    # 逐行讲解：声明 macro_rounds 的类型为 int；取值 10。
    macro_rounds: int = 10            # H：一个 macro-step 的交互轮数
    # 逐行讲解：声明 max_episode_steps 的类型为 int；取值 50。
    max_episode_steps: int = 50       # T：单局交互上限
    # 逐行讲解：声明 max_trajectory_tokens 的类型为 int；取值 MAX_SEQUENCE_TOKENS。
    max_trajectory_tokens: int = MAX_SEQUENCE_TOKENS
    # 逐行讲解：声明 max_assistant_tokens 的类型为 int；取值 512。
    max_assistant_tokens: int = 512
    # 逐行讲解：声明 temperature 的类型为 float；取值 1.0。
    temperature: float = 1.0
    # 逐行讲解：声明 top_p 的类型为 float；取值 1.0。
    top_p: float = 1.0
    # 逐行讲解：声明 seed 的类型为 int；取值 42。
    seed: int = 42
    # 逐行讲解：声明 include_admissible_actions 的类型为 bool；取值 True。
    include_admissible_actions: bool = True
    # 逐行讲解：声明 environment_asynchronous 的类型为 bool；取值 True。
    environment_asynchronous: bool = True

    # 逐行讲解：定义函数 __post_init__：封装下方逻辑，供调用方复用。
    def __post_init__(self) -> None:
        # 逐行讲解：检查条件 self.branches < 1；成立时执行下方分支。
        if self.branches < 1:
            # 逐行讲解：抛出异常：ValueError('branches 必须大于等于 1')。
            raise ValueError("branches 必须大于等于 1")
        # 逐行讲解：检查条件 self.macro_rounds < 1 or self.max_episode_steps < 1；成立时执行下方分支。
        if self.macro_rounds < 1 or self.max_episode_steps < 1:
            # 逐行讲解：抛出异常：ValueError('macro_rounds 和 max_episode_steps 必须大于等于 1')。
            raise ValueError("macro_rounds 和 max_episode_steps 必须大于等于 1")
        # 逐行讲解：检查条件 self.max_episode_steps % self.macro_rounds != 0；成立时执行下方分支。
        if self.max_episode_steps % self.macro_rounds != 0:
            # 逐行讲解：抛出异常：ValueError('max_episode_steps 必须是 macro_rounds 的整数倍')。
            raise ValueError("max_episode_steps 必须是 macro_rounds 的整数倍")
        # 逐行讲解：检查条件 self.max_trajectory_tokens < 1 or self.max_assistant_tokens < 1；成立时执行下方分支。
        if self.max_trajectory_tokens < 1 or self.max_assistant_tokens < 1:
            # 逐行讲解：抛出异常：ValueError('token 上限必须大于等于 1')。
            raise ValueError("token 上限必须大于等于 1")
        # 逐行讲解：检查条件 self.temperature < 0.0；成立时执行下方分支。
        if self.temperature < 0.0:
            # 逐行讲解：抛出异常：ValueError('temperature 不能为负数')。
            raise ValueError("temperature 不能为负数")
        # 逐行讲解：检查条件 not 0.0 < self.top_p <= 1.0；成立时执行下方分支。
        if not 0.0 < self.top_p <= 1.0:
            # 逐行讲解：抛出异常：ValueError('top_p 必须位于 (0, 1]')。
            raise ValueError("top_p 必须位于 (0, 1]")


# 逐行讲解：定义类 StepRecord：一次 action 前后的完整训练和环境信息。
@dataclass
class StepRecord:
    """一次 action 前后的完整训练和环境信息。"""

    # 逐行讲解：声明 index 的类型为 int。
    index: int
    # 逐行讲解：声明 observation 的类型为 str。
    observation: str
    # 逐行讲解：声明 admissible_actions 的类型为 tuple[str, ...]。
    admissible_actions: tuple[str, ...]
    # 逐行讲解：声明 prompt_tokens 的类型为 list[int]。
    prompt_tokens: list[int]
    # 逐行讲解：声明 completion_tokens 的类型为 list[int]。
    completion_tokens: list[int]
    # 逐行讲解：声明 logprobs 的类型为 list[float]。
    logprobs: list[float]
    # 逐行讲解：声明 assistant_text 的类型为 str。
    assistant_text: str
    # 逐行讲解：声明 call_id 的类型为 str。
    call_id: str
    # 逐行讲解：声明 action 的类型为 str。
    action: str
    # 逐行讲解：声明 valid_format 的类型为 bool。
    valid_format: bool
    # 逐行讲解：声明 admissible 的类型为 bool。
    admissible: bool
    # 逐行讲解：声明 valid_action 的类型为 bool。
    valid_action: bool
    # 逐行讲解：声明 tool_response 的类型为 str。
    tool_response: str
    # 逐行讲解：声明 next_observation 的类型为 str。
    next_observation: str
    # 逐行讲解：声明 score 的类型为 float。
    score: float
    # 逐行讲解：声明 done 的类型为 bool。
    done: bool
    # 逐行讲解：声明 won 的类型为 bool。
    won: bool


# 逐行讲解：定义类 Trajectory：从某个起点出发的一条分支。
@dataclass
class Trajectory:
    """从某个起点出发的一条分支。"""

    # 逐行讲解：声明 example 的类型为 GameExample。
    example: GameExample
    # 逐行讲解：声明 start_state 的类型为 MacroState。
    start_state: MacroState
    # 逐行讲解：声明 group_id 的类型为 str。
    group_id: str
    # 逐行讲解：声明 game_index 的类型为 int。
    game_index: int
    # 逐行讲解：声明 group_index 的类型为 int。
    group_index: int
    # 逐行讲解：声明 task 的类型为 str。
    task: str
    # 逐行讲解：声明 initial_observation 的类型为 str。
    initial_observation: str
    # 逐行讲解：声明 start_round 的类型为 int。
    start_round: int
    # 逐行讲解：声明 current_observation 的类型为 str。
    current_observation: str
    # 逐行讲解：声明 current_admissible_actions 的类型为 tuple[str, ...]。
    current_admissible_actions: tuple[str, ...]
    # 逐行讲解：声明 messages 的类型为 list[dict[str, Any]]。
    messages: list[dict[str, Any]]
    # 逐行讲解：声明 next_prompt_tokens 的类型为 list[int] | None；取值 None。
    next_prompt_tokens: list[int] | None = None
    # 逐行讲解：声明 steps 的类型为 list[StepRecord]；调用 field（field(default_factory=list)）。
    steps: list[StepRecord] = field(default_factory=list)
    # 逐行讲解：声明 done 的类型为 bool；取值 False。
    done: bool = False
    # 逐行讲解：声明 won 的类型为 bool；取值 False。
    won: bool = False
    # 逐行讲解：声明 truncated 的类型为 bool；取值 False。
    truncated: bool = False
    # 逐行讲解：声明 stop_reason 的类型为 str；取值 ''。
    stop_reason: str = ""
    # TEMPO 信号字段，由 tempo.py 拼装。
    endpoint_value: float = 0.0
    # 逐行讲解：声明 segment_reward 的类型为 float；取值 0.0。
    segment_reward: float = 0.0
    # 逐行讲解：声明 branch_return 的类型为 float；取值 0.0。
    branch_return: float = 0.0
    # 逐行讲解：声明 advantage 的类型为 float；取值 0.0。
    advantage: float = 0.0
    # 逐行讲解：声明 progress_reported 的类型为 bool；调用 field（field(default=False, repr=False)）。
    progress_reported: bool = field(default=False, repr=False)


# 逐行讲解：定义类 MacroGroup：同一个 macro-step 起点的 N 条分支。
@dataclass(frozen=True)
class MacroGroup:
    """同一个 macro-step 起点的 N 条分支。"""

    # 逐行讲解：声明 start_state 的类型为 MacroState。
    start_state: MacroState
    # 逐行讲解：声明 trajectories 的类型为 list[Trajectory]。
    trajectories: list[Trajectory]


# 逐行讲解：定义类 SampleRequest：描述一个共享 prompt 或一条分叉轨迹的采样请求。
@dataclass(frozen=True)
class SampleRequest:
    """描述一个共享 prompt 或一条分叉轨迹的采样请求。"""

    # 逐行讲解：声明 targets 的类型为 tuple[tuple[int, int], ...]。
    targets: tuple[tuple[int, int], ...]
    # 逐行讲解：声明 prompt_tokens 的类型为 list[int]。
    prompt_tokens: list[int]
    # 逐行讲解：声明 num_samples 的类型为 int。
    num_samples: int
    # 逐行讲解：声明 max_tokens 的类型为 int。
    max_tokens: int
    # 逐行讲解：声明 seed 的类型为 int。
    seed: int


# 逐行讲解：定义类 GeneratedAction：从 sampler response 中读取的一次待执行动作。
@dataclass(frozen=True)
class GeneratedAction:
    """从 sampler response 中读取的一次待执行动作。"""

    # 逐行讲解：声明 prompt_tokens 的类型为 list[int]。
    prompt_tokens: list[int]
    # 逐行讲解：声明 completion_tokens 的类型为 list[int]。
    completion_tokens: list[int]
    # 逐行讲解：声明 logprobs 的类型为 list[float]。
    logprobs: list[float]
    # 逐行讲解：声明 text 的类型为 str。
    text: str
    # 逐行讲解：声明 action 的类型为 str。
    action: str
    # 逐行讲解：声明 valid_format 的类型为 bool。
    valid_format: bool


# 逐行讲解：定义类 GroupRuntime：一个起点组的环境和 N 条分支。
@dataclass
class GroupRuntime:
    """一个起点组的环境和 N 条分支。"""

    # 逐行讲解：声明 environment 的类型为 ALFWorldGroup。
    environment: ALFWorldGroup
    # 逐行讲解：声明 start_state 的类型为 MacroState。
    start_state: MacroState
    # 逐行讲解：声明 trajectories 的类型为 list[Trajectory]。
    trajectories: list[Trajectory]


# 逐行讲解：定义函数 prompt_for_trajectory：返回某条分支包含完整工具历史的真实 token 前缀。
def prompt_for_trajectory(
    tokenizer: Any,
    trajectory: Trajectory,
) -> list[int]:
    """返回某条分支包含完整工具历史的真实 token 前缀。"""
    # 逐行讲解：检查条件 trajectory.next_prompt_tokens is not None；成立时执行下方分支。
    if trajectory.next_prompt_tokens is not None:
        # 逐行讲解：返回结果：trajectory.next_prompt_tokens。
        return trajectory.next_prompt_tokens
    # 逐行讲解：返回结果：build_prompt(tokenizer, trajectory.messages)。
    return build_prompt(tokenizer, trajectory.messages)


# 逐行讲解：定义函数 _generation_budget：在轨迹 token 上限内计算本轮最多可采样的 token 数。
def _generation_budget(
    prompt_tokens: list[int],
    config: RolloutConfig,
) -> int:
    """在轨迹 token 上限内计算本轮最多可采样的 token 数。"""
    # 逐行讲解：返回结果：min(config.max_assistant_tokens, config.max_trajectory_tokens - len(prompt_tokens))。
    return min(
        config.max_assistant_tokens,
        config.max_trajectory_tokens - len(prompt_tokens),
    )


# 逐行讲解：定义函数 _request_for_trajectory：封装下方逻辑，供调用方复用。
def _request_for_trajectory(
    tokenizer: Any,
    trajectory: Trajectory,
    target: tuple[int, int],
    config: RolloutConfig,
    *,
    seed: int,
) -> SampleRequest | None:
    # 逐行讲解：调用 prompt_for_trajectory（prompt_for_trajectory(tokenizer, trajectory)），赋给 prompt_tokens。
    prompt_tokens = prompt_for_trajectory(tokenizer, trajectory)
    # 逐行讲解：调用 _generation_budget（_generation_budget(prompt_tokens, config)），赋给 max_tokens。
    max_tokens = _generation_budget(prompt_tokens, config)
    # 逐行讲解：检查条件 max_tokens < 1；成立时执行下方分支。
    if max_tokens < 1:
        # 逐行讲解：取值 True，赋给 trajectory.done。
        trajectory.done = True
        # 逐行讲解：取值 True，赋给 trajectory.truncated。
        trajectory.truncated = True
        # 逐行讲解：取值 'trajectory_too_long'，赋给 trajectory.stop_reason。
        trajectory.stop_reason = "trajectory_too_long"
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：SampleRequest((target,), prompt_tokens, 1, max_tokens, seed)。
    return SampleRequest((target,), prompt_tokens, 1, max_tokens, seed)


# 逐行讲解：定义异步函数 _sample_requests_async：并发提交不同 prompt；同一轨迹内部仍保持严格顺序。
async def _sample_requests_async(
    sampling_client: Any,
    requests: list[SampleRequest],
    tokenizer: Any,
    config: RolloutConfig,
) -> list[Any]:
    """并发提交不同 prompt；同一轨迹内部仍保持严格顺序。"""
    # 逐行讲解：调用 stop_sequences（stop_sequences(tokenizer)），赋给 stop。
    stop = stop_sequences(tokenizer)
    # 逐行讲解：通过推导式逐项生成结果：[sampling_client.sample_async(prompt=trio.ModelInput.from_ints(request.prompt_tokens), num_samples=request.…，赋给 calls。
    calls = [
        sampling_client.sample_async(
            prompt=trio.ModelInput.from_ints(request.prompt_tokens),
            num_samples=request.num_samples,
            sampling_params=trio.SamplingParams(
                max_tokens=request.max_tokens,
                seed=request.seed,
                stop=stop,
                temperature=config.temperature,
                top_p=config.top_p,
            ),
            return_text=True,
        )
        for request in requests
    ]
    # 逐行讲解：返回结果：list(await asyncio.gather(*calls))。
    return list(await asyncio.gather(*calls))


# 逐行讲解：定义函数 _read_sequence：封装下方逻辑，供调用方复用。
def _read_sequence(
    sequence: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
) -> GeneratedAction:
    # 逐行讲解：通过推导式逐项生成结果：[int(token) for token in sequence.tokens]，赋给 tokens。
    tokens = [int(token) for token in sequence.tokens]
    # 逐行讲解：通过推导式逐项生成结果：[float(value) for value in sequence.logprobs]，赋给 logprobs。
    logprobs = [float(value) for value in sequence.logprobs]
    # 逐行讲解：检查条件 len(tokens) != len(logprobs)；成立时执行下方分支。
    if len(tokens) != len(logprobs):
        # 逐行讲解：抛出异常：ValueError('采样 completion token 与 old logprob 长度不一致')。
        raise ValueError("采样 completion token 与 old logprob 长度不一致")
    # 逐行讲解：取值 sequence.text，赋给 text。
    text = sequence.text
    # 逐行讲解：检查条件 text is None；成立时执行下方分支。
    if text is None:
        # 逐行讲解：将 token 解码为文本（tokenizer.decode(tokens, skip_special_tokens=True)），赋给 text。
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    # 逐行讲解：调用 parse_assistant（parse_assistant(str(text))），赋给 parsed。
    parsed = parse_assistant(str(text))
    # 逐行讲解：返回结果：GeneratedAction(prompt_tokens=prompt_tokens, completion_tokens=tokens, logprobs=logprobs, text=str(text), a…。
    return GeneratedAction(
        prompt_tokens=prompt_tokens,
        completion_tokens=tokens,
        logprobs=logprobs,
        text=str(text),
        action=parsed.action if parsed.action is not None else INVALID_ENV_ACTION,
        valid_format=parsed.valid_format,
    )


# 逐行讲解：定义函数 _consume_responses：将 response 中的候选按 (game, branch) 放回原位置。
def _consume_responses(
    requests: list[SampleRequest],
    responses: list[Any],
    tokenizer: Any,
) -> dict[tuple[int, int], GeneratedAction]:
    """将 response 中的候选按 (game, branch) 放回原位置。"""
    # 逐行讲解：声明 pending 的类型为 dict[tuple[int, int], GeneratedAction]；构造字典 {}。
    pending: dict[tuple[int, int], GeneratedAction] = {}
    # 逐行讲解：遍历 zip(requests, responses, strict=True)，将每项绑定到 (request, response)。
    for request, response in zip(requests, responses, strict=True):
        # 逐行讲解：调用 list（list(response.sequences)），赋给 sequences。
        sequences = list(response.sequences)
        # 逐行讲解：检查条件 len(sequences) != request.num_samples；成立时执行下方分支。
        if len(sequences) != request.num_samples:
            # 逐行讲解：抛出异常：ValueError(f'采样数量 {len(sequences)} != 请求数量 {request.num_samples}')。
            raise ValueError(
                f"采样数量 {len(sequences)} != 请求数量 {request.num_samples}"
            )
        # 逐行讲解：检查条件 len(request.targets) != request.num_samples；成立时执行下方分支。
        if len(request.targets) != request.num_samples:
            # 逐行讲解：抛出异常：ValueError('SampleRequest targets 与 num_samples 不一致')。
            raise ValueError("SampleRequest targets 与 num_samples 不一致")
        # 逐行讲解：遍历 zip(request.targets, sequences, strict=True)，将每项绑定到 (target, sequence)。
        for target, sequence in zip(request.targets, sequences, strict=True):
            # 逐行讲解：调用 _read_sequence（_read_sequence(sequence, tokenizer, request.prompt_tokens)），赋给 pending[target]。
            pending[target] = _read_sequence(
                sequence,
                tokenizer,
                request.prompt_tokens,
            )
    # 逐行讲解：返回结果：pending。
    return pending


# 逐行讲解：定义函数 _step_environments：不同起点组可并发执行；组内一次提交 N 个动作。
def _step_environments(
    runtimes: list[GroupRuntime],
    pending: dict[tuple[int, int], GeneratedAction],
) -> dict[int, list[EnvironmentStep]]:
    """不同起点组可并发执行；组内一次提交 N 个动作。"""
    # 逐行讲解：声明 calls 的类型为 list[tuple[int, ALFWorldGroup, list[str]]]；构造列表 []。
    calls: list[tuple[int, ALFWorldGroup, list[str]]] = []
    # 逐行讲解：遍历 enumerate(runtimes)，将每项绑定到 (game_index, runtime)。
    for game_index, runtime in enumerate(runtimes):
        # 逐行讲解：通过推导式逐项生成结果：[(branch_index, pending.get((game_index, branch_index))) for branch_index in range(len(runtime.trajectories…，赋给 targets。
        targets = [
            (branch_index, pending.get((game_index, branch_index)))
            for branch_index in range(len(runtime.trajectories))
        ]
        # 逐行讲解：检查条件 not any((generated is not None for _, generated in targets))；成立时执行下方分支。
        if not any(generated is not None for _, generated in targets):
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：通过推导式逐项生成结果：[generated.action if generated is not None else 'look' for _, generated in targets]，赋给 actions。
        actions = [
            generated.action if generated is not None else "look"
            for _, generated in targets
        ]
        # 逐行讲解：向列表末尾追加一个元素（calls.append((game_index, runtime.environment, actions))）。
        calls.append((game_index, runtime.environment, actions))

    # 逐行讲解：检查条件 not calls；成立时执行下方分支。
    if not calls:
        # 逐行讲解：返回结果：{}。
        return {}
    # 逐行讲解：声明 results 的类型为 dict[int, list[EnvironmentStep]]；构造字典 {}。
    results: dict[int, list[EnvironmentStep]] = {}
    # 逐行讲解：进入上下文管理器 ThreadPoolExecutor(max_workers=min(len(calls), 32))，退出时执行其收尾逻辑。
    with ThreadPoolExecutor(max_workers=min(len(calls), 32)) as pool:
        # 逐行讲解：通过推导式逐项生成结果：[(game_index, pool.submit(environment.step, actions)) for game_index, environment, actions in calls]，赋给 futures。
        futures = [
            (game_index, pool.submit(environment.step, actions))
            for game_index, environment, actions in calls
        ]
        # 逐行讲解：遍历 futures，将每项绑定到 (game_index, future)。
        for game_index, future in futures:
            # 逐行讲解：等待任务完成并读取结果（future.result()），赋给 results[game_index]。
            results[game_index] = future.result()
    # 逐行讲解：返回结果：results。
    return results


# 逐行讲解：定义函数 _finish_trajectory：按终局 / T 上限 / H 边界 / token 预算的优先级结束一条分支。
def _finish_trajectory(
    trajectory: Trajectory,
    environment_step: EnvironmentStep,
    *,
    prompt_fits: bool,
    config: RolloutConfig,
) -> None:
    """按终局 / T 上限 / H 边界 / token 预算的优先级结束一条分支。"""
    # 逐行讲解：调用 len（len(trajectory.steps)），赋给 segment_rounds。
    segment_rounds = len(trajectory.steps)
    # 逐行讲解：计算 trajectory.start_round + segment_rounds，赋给 absolute_round。
    absolute_round = trajectory.start_round + segment_rounds
    # 逐行讲解：检查条件 environment_step.done；成立时执行下方分支。
    if environment_step.done:
        # 逐行讲解：取值 True，赋给 trajectory.done。
        trajectory.done = True
        # 逐行讲解：取值 not environment_step.won，赋给 trajectory.truncated。
        trajectory.truncated = not environment_step.won
        # 逐行讲解：按条件选择结果：'won' if environment_step.won else 'environment_done'，赋给 trajectory.stop_reason。
        trajectory.stop_reason = "won" if environment_step.won else "environment_done"
    # 逐行讲解：检查条件 absolute_round >= config.max_episode_steps；成立时执行下方分支。
    elif absolute_round >= config.max_episode_steps:
        # 逐行讲解：取值 True，赋给 trajectory.done。
        trajectory.done = True
        # 逐行讲解：取值 not environment_step.won，赋给 trajectory.truncated。
        trajectory.truncated = not environment_step.won
        # 逐行讲解：取值 'max_steps'，赋给 trajectory.stop_reason。
        trajectory.stop_reason = "max_steps"
    # 逐行讲解：检查条件 segment_rounds >= config.macro_rounds；成立时执行下方分支。
    elif segment_rounds >= config.macro_rounds:
        # 逐行讲解：取值 True，赋给 trajectory.done。
        trajectory.done = True
        # 逐行讲解：取值 False，赋给 trajectory.truncated。
        trajectory.truncated = False
        # 逐行讲解：取值 'macro_boundary'，赋给 trajectory.stop_reason。
        trajectory.stop_reason = "macro_boundary"
    # 逐行讲解：检查条件 not prompt_fits；成立时执行下方分支。
    elif not prompt_fits:
        # 逐行讲解：取值 True，赋给 trajectory.done。
        trajectory.done = True
        # 逐行讲解：取值 True，赋给 trajectory.truncated。
        trajectory.truncated = True
        # 逐行讲解：取值 'trajectory_too_long'，赋给 trajectory.stop_reason。
        trajectory.stop_reason = "trajectory_too_long"


# 逐行讲解：定义函数 _apply_environment_results：写入 StepRecord，并更新各分支的下一状态。
def _apply_environment_results(
    runtimes: list[GroupRuntime],
    pending: dict[tuple[int, int], GeneratedAction],
    results: dict[int, list[EnvironmentStep]],
    tokenizer: Any,
    config: RolloutConfig,
) -> None:
    """写入 StepRecord，并更新各分支的下一状态。"""
    # 逐行讲解：遍历 pending.items()，将每项绑定到 ((game_index, branch_index), generated)。
    for (game_index, branch_index), generated in pending.items():
        # 逐行讲解：按索引或切片读取 runtimes[game_index].trajectories[branch_index]，赋给 trajectory。
        trajectory = runtimes[game_index].trajectories[branch_index]
        # 逐行讲解：按索引或切片读取 results[game_index][branch_index]，赋给 environment_step。
        environment_step = results[game_index][branch_index]
        # 逐行讲解：取值 generated.valid_format and environment_step.admissible，赋给 valid_action。
        valid_action = generated.valid_format and environment_step.admissible
        # 逐行讲解：计算 trajectory.start_round + len(trajectory.steps) + 1，赋给 step_index。
        step_index = trajectory.start_round + len(trajectory.steps) + 1
        # 逐行讲解：取值 f'tempo-{trajectory.game_index}-{trajectory.group_index}-{step_index}'，赋给 call_id。
        call_id = (
            f"tempo-{trajectory.game_index}-{trajectory.group_index}-{step_index}"
        )
        # 逐行讲解：调用 environment_tool_content（environment_tool_content(step=step_index, observation=environment_step.observation, admissible_actions=envi…），赋给 content。
        content = environment_tool_content(
            step=step_index,
            observation=environment_step.observation,
            admissible_actions=environment_step.admissible_actions,
            valid_format=generated.valid_format,
            admissible=environment_step.admissible,
            done=environment_step.done,
            won=environment_step.won,
            include_admissible_actions=config.include_admissible_actions,
        )
        # 逐行讲解：调用 tool_message（tool_message(call_id, content)），赋给 next_tool_message。
        next_tool_message = tool_message(call_id, content)
        # 逐行讲解：调用 build_next_prompt（build_next_prompt(tokenizer, list(trajectory.messages), generated.prompt_tokens, generated.completion_token…），赋给 next_prompt_tokens。
        next_prompt_tokens = build_next_prompt(
            tokenizer,
            list(trajectory.messages),
            generated.prompt_tokens,
            generated.completion_tokens,
            next_tool_message,
        )
        # 逐行讲解：向列表末尾追加一个元素（trajectory.messages.append({'role': 'assistant', 'content': generated.text})）。
        trajectory.messages.append(
            {"role": "assistant", "content": generated.text}
        )
        # 逐行讲解：向列表末尾追加一个元素（trajectory.messages.append(next_tool_message)）。
        trajectory.messages.append(next_tool_message)
        # 逐行讲解：取值 len(next_prompt_tokens) <= config.max_trajectory_tokens，赋给 prompt_fits。
        prompt_fits = len(next_prompt_tokens) <= config.max_trajectory_tokens
        # 逐行讲解：按条件选择结果：next_prompt_tokens if prompt_fits else None，赋给 trajectory.next_prompt_tokens。
        trajectory.next_prompt_tokens = next_prompt_tokens if prompt_fits else None

        # 逐行讲解：向列表末尾追加一个元素（trajectory.steps.append(StepRecord(index=step_index, observation=trajectory.current_observation, admissible…）。
        trajectory.steps.append(
            StepRecord(
                index=step_index,
                observation=trajectory.current_observation,
                admissible_actions=trajectory.current_admissible_actions,
                prompt_tokens=generated.prompt_tokens,
                completion_tokens=generated.completion_tokens,
                logprobs=generated.logprobs,
                assistant_text=generated.text,
                call_id=call_id,
                action=generated.action,
                valid_format=generated.valid_format,
                admissible=environment_step.admissible,
                valid_action=valid_action,
                tool_response=content,
                next_observation=environment_step.observation,
                score=environment_step.score,
                done=environment_step.done,
                won=environment_step.won,
            )
        )
        # 逐行讲解：取值 environment_step.observation，赋给 trajectory.current_observation。
        trajectory.current_observation = environment_step.observation
        # 逐行讲解：取值 environment_step.admissible_actions，赋给 trajectory.current_admissible_actions。
        trajectory.current_admissible_actions = environment_step.admissible_actions
        # 逐行讲解：取值 trajectory.won or environment_step.won，赋给 trajectory.won。
        trajectory.won = trajectory.won or environment_step.won

        # 逐行讲解：检查条件 not trajectory.done；成立时执行下方分支。
        if not trajectory.done:
            # 逐行讲解：调用 _finish_trajectory（_finish_trajectory(trajectory, environment_step, prompt_fits=prompt_fits, config=config)）。
            _finish_trajectory(
                trajectory,
                environment_step,
                prompt_fits=prompt_fits,
                config=config,
            )


# 逐行讲解：定义函数 _report_finished：封装下方逻辑，供调用方复用。
def _report_finished(
    trajectories: list[Trajectory],
    callback: Callable[[int], None] | None,
) -> None:
    # 逐行讲解：取值 0，赋给 newly_finished。
    newly_finished = 0
    # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
    for trajectory in trajectories:
        # 逐行讲解：检查条件 trajectory.done and (not trajectory.progress_reported)；成立时执行下方分支。
        if trajectory.done and not trajectory.progress_reported:
            # 逐行讲解：取值 True，赋给 trajectory.progress_reported。
            trajectory.progress_reported = True
            # 逐行讲解：更新 newly_finished：newly_finished += 1。
            newly_finished += 1
    # 逐行讲解：检查条件 newly_finished and callback is not None；成立时执行下方分支。
    if newly_finished and callback is not None:
        # 逐行讲解：调用 callback（callback(newly_finished)）。
        callback(newly_finished)


# 逐行讲解：定义函数 _replay_environment：reset 后按保存的动作历史重放，并校验恢复到的 observation。
def _replay_environment(
    environment: ALFWorldGroup,
    state: MacroState,
    config: RolloutConfig,
) -> None:
    """reset 后按保存的动作历史重放，并校验恢复到的 observation。"""
    # 逐行讲解：调用 environment.reset（environment.reset()），赋给 env_states。
    env_states = environment.reset()
    # 逐行讲解：检查条件 len({item.observation for item in env_states}) != 1；成立时执行下方分支。
    if len({item.observation for item in env_states}) != 1:
        # 逐行讲解：抛出异常：RuntimeError('同组 ALFWorld 环境的初始 observation 不一致')。
        raise RuntimeError("同组 ALFWorld 环境的初始 observation 不一致")

    # 逐行讲解：声明 last 的类型为 list[EnvironmentStep] | None；取值 None。
    last: list[EnvironmentStep] | None = None
    # 逐行讲解：遍历 state.action_history，将每项绑定到 action。
    for action in state.action_history:
        # 逐行讲解：调用 environment.step（environment.step([action] * config.branches)），赋给 last。
        last = environment.step([action] * config.branches)

    # 逐行讲解：检查条件 last is None；成立时执行下方分支。
    if last is None:
        # 逐行讲解：取值 env_states[0].observation，赋给 restored。
        restored = env_states[0].observation
    else:
        # 重放是组内锁步的，所有分支必然落在同一 observation 上。
        observations = {step.observation for step in last}
        # 逐行讲解：检查条件 len(observations) != 1；成立时执行下方分支。
        if len(observations) != 1:
            # 逐行讲解：抛出异常：RuntimeError('重放后分支 observation 不一致')。
            raise RuntimeError("重放后分支 observation 不一致")
        # 逐行讲解：调用 observations.pop（observations.pop()），赋给 restored。
        restored = observations.pop()
    # 逐行讲解：检查条件 restored != state.current_observation；成立时执行下方分支。
    if restored != state.current_observation:
        # 逐行讲解：抛出异常：RuntimeError(f'重放后的 observation 与保存状态不一致，环境无法恢复: expected={state.current_observation!r}, actual={restored!r…。
        raise RuntimeError(
            "重放后的 observation 与保存状态不一致，环境无法恢复: "
            f"expected={state.current_observation!r}, actual={restored!r}"
        )


# 逐行讲解：定义函数 _trajectories_from_state：封装下方逻辑，供调用方复用。
def _trajectories_from_state(
    state: MacroState,
    game_index: int,
    config: RolloutConfig,
) -> list[Trajectory]:
    # 逐行讲解：取值 f'{game_index}:{state.example.id}'，赋给 group_id。
    group_id = f"{game_index}:{state.example.id}"
    # 逐行讲解：返回结果：[Trajectory(example=state.example, start_state=state, group_id=group_id, game_index=game_index, group_index…。
    return [
        Trajectory(
            example=state.example,
            start_state=state,
            group_id=group_id,
            game_index=game_index,
            group_index=branch_index,
            task=state.task,
            initial_observation=state.initial_observation,
            start_round=state.round_index,
            current_observation=state.current_observation,
            current_admissible_actions=state.current_admissible_actions,
            messages=state.fork_messages(),
            next_prompt_tokens=(
                list(state.token_prefix) if state.token_prefix is not None else None
            ),
        )
        for branch_index in range(config.branches)
    ]


# 逐行讲解：定义函数 _restored_runtime：封装下方逻辑，供调用方复用。
def _restored_runtime(
    state: MacroState,
    game_index: int,
    config: RolloutConfig,
) -> GroupRuntime:
    # 逐行讲解：调用 ALFWorldGroup（ALFWorldGroup(state.example, config.branches, config.max_episode_steps, seed=state.environment_seed, asynch…），赋给 environment。
    environment = ALFWorldGroup(
        state.example,
        config.branches,
        config.max_episode_steps,
        seed=state.environment_seed,
        asynchronous=config.environment_asynchronous,
    )
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：调用 _replay_environment（_replay_environment(environment, state, config)）。
        _replay_environment(environment, state, config)
    except Exception:
        # 逐行讲解：调用 environment.close（environment.close()）。
        environment.close()
        # 逐行讲解：抛出异常：重新抛出当前异常。
        raise
    # 逐行讲解：返回结果：GroupRuntime(environment, state, _trajectories_from_state(state, game_index, config))。
    return GroupRuntime(
        environment,
        state,
        _trajectories_from_state(state, game_index, config),
    )


# 逐行讲解：定义函数 _fresh_runtime：封装下方逻辑，供调用方复用。
def _fresh_runtime(
    example: GameExample,
    game_index: int,
    config: RolloutConfig,
    *,
    seed: int,
) -> GroupRuntime:
    # 逐行讲解：调用 ALFWorldGroup（ALFWorldGroup(example, config.branches, config.max_episode_steps, seed=seed, asynchronous=config.environmen…），赋给 environment。
    environment = ALFWorldGroup(
        example,
        config.branches,
        config.max_episode_steps,
        seed=seed,
        asynchronous=config.environment_asynchronous,
    )
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：调用 environment.reset（environment.reset()），赋给 env_states。
        env_states = environment.reset()
    except Exception:
        # 逐行讲解：调用 environment.close（environment.close()）。
        environment.close()
        # 逐行讲解：抛出异常：重新抛出当前异常。
        raise
    # 逐行讲解：按索引或切片读取 env_states[0]，赋给 state。
    state = env_states[0]
    # 逐行讲解：调用 initial_messages（initial_messages(task=state.task, observation=state.observation, admissible_actions=state.admissible_action…），赋给 messages。
    messages = initial_messages(
        task=state.task,
        observation=state.observation,
        admissible_actions=state.admissible_actions,
        include_admissible_actions=config.include_admissible_actions,
    )
    # 逐行讲解：调用 MacroState（MacroState(example=example, task=state.task, initial_observation=state.observation, action_history=(), mess…），赋给 macro_state。
    macro_state = MacroState(
        example=example,
        task=state.task,
        initial_observation=state.observation,
        action_history=(),
        messages=messages,
        token_prefix=None,
        round_index=0,
        macro_index=0,
        environment_seed=seed,
        current_observation=state.observation,
        current_admissible_actions=state.admissible_actions,
    )
    # 逐行讲解：返回结果：GroupRuntime(environment, macro_state, _trajectories_from_state(macro_state, game_index, config))。
    return GroupRuntime(
        environment,
        macro_state,
        _trajectories_from_state(macro_state, game_index, config),
    )


# 逐行讲解：定义函数 _initialize_runtimes：封装下方逻辑，供调用方复用。
def _initialize_runtimes(
    start_states: list[MacroState],
    fresh_examples: list[GameExample],
    config: RolloutConfig,
    *,
    seed_offset: int,
) -> list[GroupRuntime]:
    # 逐行讲解：声明 runtimes 的类型为 list[GroupRuntime]；构造列表 []。
    runtimes: list[GroupRuntime] = []
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：遍历 start_states，将每项绑定到 state。
        for state in start_states:
            # 逐行讲解：向列表末尾追加一个元素（runtimes.append(_restored_runtime(state, len(runtimes), config))）。
            runtimes.append(_restored_runtime(state, len(runtimes), config))
        # 逐行讲解：遍历 fresh_examples，将每项绑定到 example。
        for example in fresh_examples:
            # 逐行讲解：向列表末尾追加一个元素（runtimes.append(_fresh_runtime(example, len(runtimes), config, seed=config.seed + seed_offset + len(runtime…）。
            runtimes.append(
                _fresh_runtime(
                    example,
                    len(runtimes),
                    config,
                    seed=config.seed + seed_offset + len(runtimes),
                )
            )
    except Exception:
        # 逐行讲解：遍历 runtimes，将每项绑定到 runtime。
        for runtime in runtimes:
            # 逐行讲解：调用 runtime.environment.close（runtime.environment.close()）。
            runtime.environment.close()
        # 逐行讲解：抛出异常：重新抛出当前异常。
        raise
    # 逐行讲解：返回结果：runtimes。
    return runtimes


# 逐行讲解：定义函数 rollout_macro_steps：对每个起点恢复环境并采满一个 macro-step，返回按起点分组的分支。
def rollout_macro_steps(
    sampling_client: Any,
    tokenizer: Any,
    start_states: list[MacroState],
    fresh_examples: list[GameExample],
    config: RolloutConfig,
    *,
    seed_offset: int = 0,
    progress_callback: Callable[[int], None] | None = None,
) -> list[MacroGroup]:
    """对每个起点恢复环境并采满一个 macro-step，返回按起点分组的分支。

    warm-up 阶段传入 macro_rounds == max_episode_steps 的 config 和
    全 fresh 游戏，即退化为第 8 篇的完整轨迹 rollout。
    """
    # 逐行讲解：检查条件 not start_states and (not fresh_examples)；成立时执行下方分支。
    if not start_states and not fresh_examples:
        # 逐行讲解：返回结果：[]。
        return []
    # 逐行讲解：声明 runtimes 的类型为 list[GroupRuntime]；构造列表 []。
    runtimes: list[GroupRuntime] = []
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：调用 _initialize_runtimes（_initialize_runtimes(start_states, fresh_examples, config, seed_offset=seed_offset)），赋给 runtimes。
        runtimes = _initialize_runtimes(
            start_states,
            fresh_examples,
            config,
            seed_offset=seed_offset,
        )
        # 逐行讲解：通过推导式逐项生成结果：[trajectory for runtime in runtimes for trajectory in runtime.trajectories]，赋给 trajectories。
        trajectories = [
            trajectory
            for runtime in runtimes
            for trajectory in runtime.trajectories
        ]

        # 逐行讲解：遍历 range(config.macro_rounds)，将每项绑定到 round_index。
        for round_index in range(config.macro_rounds):
            # 逐行讲解：检查条件 all((trajectory.done for trajectory in trajectories))；成立时执行下方分支。
            if all(trajectory.done for trajectory in trajectories):
                # 逐行讲解：结束当前循环。
                break

            # 逐行讲解：声明 requests 的类型为 list[SampleRequest]；构造列表 []。
            requests: list[SampleRequest] = []
            # 逐行讲解：检查条件 round_index == 0；成立时执行下方分支。
            if round_index == 0:
                # 同一起点的 N 条分支共享 prompt，一次请求分叉出 N 个候选。
                for game_index, runtime in enumerate(runtimes):
                    # 逐行讲解：按索引或切片读取 runtime.trajectories[0]，赋给 root。
                    root = runtime.trajectories[0]
                    # 逐行讲解：调用 prompt_for_trajectory（prompt_for_trajectory(tokenizer, root)），赋给 prompt_tokens。
                    prompt_tokens = prompt_for_trajectory(tokenizer, root)
                    # 逐行讲解：调用 _generation_budget（_generation_budget(prompt_tokens, config)），赋给 max_tokens。
                    max_tokens = _generation_budget(prompt_tokens, config)
                    # 逐行讲解：检查条件 max_tokens < 1；成立时执行下方分支。
                    if max_tokens < 1:
                        # 逐行讲解：遍历 runtime.trajectories，将每项绑定到 trajectory。
                        for trajectory in runtime.trajectories:
                            # 逐行讲解：取值 True，赋给 trajectory.done。
                            trajectory.done = True
                            # 逐行讲解：取值 True，赋给 trajectory.truncated。
                            trajectory.truncated = True
                            # 逐行讲解：取值 'trajectory_too_long'，赋给 trajectory.stop_reason。
                            trajectory.stop_reason = "trajectory_too_long"
                        # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                        continue
                    # 逐行讲解：遍历 runtime.trajectories[1:]，将每项绑定到 trajectory。
                    for trajectory in runtime.trajectories[1:]:
                        # 逐行讲解：检查条件 prompt_for_trajectory(tokenizer, trajectory) != prompt_tokens；成立时执行下方分支。
                        if prompt_for_trajectory(tokenizer, trajectory) != prompt_tokens:
                            # 逐行讲解：抛出异常：RuntimeError('同组分支的续跑 prompt 不一致')。
                            raise RuntimeError("同组分支的续跑 prompt 不一致")
                    # 逐行讲解：调用 tuple（tuple(((game_index, branch_index) for branch_index in range(config.branches)))），赋给 targets。
                    targets = tuple(
                        (game_index, branch_index)
                        for branch_index in range(config.branches)
                    )
                    # 逐行讲解：向列表末尾追加一个元素（requests.append(SampleRequest(targets=targets, prompt_tokens=prompt_tokens, num_samples=config.branches, ma…）。
                    requests.append(
                        SampleRequest(
                            targets=targets,
                            prompt_tokens=prompt_tokens,
                            num_samples=config.branches,
                            max_tokens=max_tokens,
                            seed=config.seed + seed_offset + game_index,
                        )
                    )
            else:
                # 分叉后每条分支状态不同，各自请求一个候选；
                # 不同分支的请求之间并发。
                for game_index, runtime in enumerate(runtimes):
                    # 逐行讲解：遍历 enumerate(runtime.trajectories)，将每项绑定到 (branch_index, trajectory)。
                    for branch_index, trajectory in enumerate(runtime.trajectories):
                        # 逐行讲解：检查条件 trajectory.done；成立时执行下方分支。
                        if trajectory.done:
                            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                            continue
                        # 逐行讲解：调用 _request_for_trajectory（_request_for_trajectory(tokenizer, trajectory, (game_index, branch_index), config, seed=config.seed + seed_…），赋给 request。
                        request = _request_for_trajectory(
                            tokenizer,
                            trajectory,
                            (game_index, branch_index),
                            config,
                            seed=(
                                config.seed
                                + seed_offset
                                + game_index * 1_000_000
                                + branch_index * 10_000
                                + round_index
                            ),
                        )
                        # 逐行讲解：检查条件 request is not None；成立时执行下方分支。
                        if request is not None:
                            # 逐行讲解：向列表末尾追加一个元素（requests.append(request)）。
                            requests.append(request)

            # 逐行讲解：调用 _report_finished（_report_finished(trajectories, progress_callback)）。
            _report_finished(trajectories, progress_callback)
            # 逐行讲解：检查条件 not requests；成立时执行下方分支。
            if not requests:
                # 逐行讲解：结束当前循环。
                break
            # 逐行讲解：调用 asyncio.run（asyncio.run(_sample_requests_async(sampling_client, requests, tokenizer, config))），赋给 responses。
            responses = asyncio.run(
                _sample_requests_async(sampling_client, requests, tokenizer, config)
            )
            # 逐行讲解：调用 _consume_responses（_consume_responses(requests, responses, tokenizer)），赋给 pending。
            pending = _consume_responses(requests, responses, tokenizer)
            # 逐行讲解：调用 _step_environments（_step_environments(runtimes, pending)），赋给 environment_results。
            environment_results = _step_environments(runtimes, pending)
            # 逐行讲解：调用 _apply_environment_results（_apply_environment_results(runtimes, pending, environment_results, tokenizer, config)）。
            _apply_environment_results(
                runtimes,
                pending,
                environment_results,
                tokenizer,
                config,
            )
            # 逐行讲解：调用 _report_finished（_report_finished(trajectories, progress_callback)）。
            _report_finished(trajectories, progress_callback)

        # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
        for trajectory in trajectories:
            # 逐行讲解：检查条件 not trajectory.done；成立时执行下方分支。
            if not trajectory.done:
                # 逐行讲解：取值 True，赋给 trajectory.done。
                trajectory.done = True
                # 逐行讲解：取值 not trajectory.won，赋给 trajectory.truncated。
                trajectory.truncated = not trajectory.won
                # 逐行讲解：取值 'max_steps'，赋给 trajectory.stop_reason。
                trajectory.stop_reason = "max_steps"
            # 逐行讲解：按条件选择结果：1.0 if trajectory.won else 0.0，赋给 trajectory.segment_reward。
            trajectory.segment_reward = 1.0 if trajectory.won else 0.0
        # 逐行讲解：调用 _report_finished（_report_finished(trajectories, progress_callback)）。
        _report_finished(trajectories, progress_callback)
    finally:
        # 逐行讲解：遍历 runtimes，将每项绑定到 runtime。
        for runtime in runtimes:
            # 逐行讲解：调用 runtime.environment.close（runtime.environment.close()）。
            runtime.environment.close()

    # 逐行讲解：返回结果：[MacroGroup(runtime.start_state, runtime.trajectories) for runtime in runtimes]。
    return [
        MacroGroup(runtime.start_state, runtime.trajectories)
        for runtime in runtimes
    ]
