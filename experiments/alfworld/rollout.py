"""采样并执行 text-only ALFWorld 的同游戏分组轨迹。"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：从 concurrent.futures 导入 ThreadPoolExecutor。
from concurrent.futures import ThreadPoolExecutor
# 逐行讲解：从 dataclasses 导入 dataclass, field。
from dataclasses import dataclass, field
# 逐行讲解：从 typing 导入 Any, Callable, Protocol。
from typing import Any, Callable, Protocol

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio

# 逐行讲解：从 advantages 导入 AdvantageStats, assign_advantages。
from advantages import AdvantageStats, assign_advantages
# 逐行讲解：从 data 导入 GameExample。
from data import GameExample
# 逐行讲解：从 environment 导入 ALFWorldGroup, EnvironmentState, EnvironmentStep。
from environment import ALFWorldGroup, EnvironmentState, EnvironmentStep
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


# 逐行讲解：取值 '__invalid_tool_call__'，赋给 INVALID_ENV_ACTION。
INVALID_ENV_ACTION = "__invalid_tool_call__"
# 逐行讲解：取值 0.1，赋给 INVALID_ACTION_PENALTY。
INVALID_ACTION_PENALTY = 0.1
# 逐行讲解：取值 12000，赋给 MAX_SEQUENCE_TOKENS。
MAX_SEQUENCE_TOKENS = 12_000


# 逐行讲解：定义类 EnvironmentGroup：rollout 所需的最小环境接口，便于用 fake environment 做单元测试。
class EnvironmentGroup(Protocol):
    """rollout 所需的最小环境接口，便于用 fake environment 做单元测试。"""

    # 逐行讲解：定义函数 reset：封装下方逻辑，供调用方复用。
    def reset(self) -> list[EnvironmentState]: ...

    # 逐行讲解：定义函数 step：封装下方逻辑，供调用方复用。
    def step(self, actions: list[str]) -> list[EnvironmentStep]: ...

    # 逐行讲解：定义函数 close：封装下方逻辑，供调用方复用。
    def close(self) -> None: ...


# 逐行讲解：按索引或切片读取 Callable[..., EnvironmentGroup]，赋给 EnvironmentFactory。
EnvironmentFactory = Callable[..., EnvironmentGroup]


# 逐行讲解：定义类 RolloutConfig：保存环境和采样参数。
@dataclass(frozen=True)
class RolloutConfig:
    """保存环境和采样参数。"""

    # 逐行讲解：声明 group_size 的类型为 int；取值 8。
    group_size: int = 8
    # 逐行讲解：声明 max_steps 的类型为 int；取值 50。
    max_steps: int = 50
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
        # 逐行讲解：检查条件 self.group_size < 1；成立时执行下方分支。
        if self.group_size < 1:
            # 逐行讲解：抛出异常：ValueError('group_size 必须大于等于 1')。
            raise ValueError("group_size 必须大于等于 1")
        # 逐行讲解：检查条件 self.max_steps < 1；成立时执行下方分支。
        if self.max_steps < 1:
            # 逐行讲解：抛出异常：ValueError('max_steps 必须大于等于 1')。
            raise ValueError("max_steps 必须大于等于 1")
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


# 逐行讲解：定义类 Trajectory：一条独立环境轨迹。
@dataclass
class Trajectory:
    """一条独立环境轨迹。"""

    # 逐行讲解：声明 example 的类型为 GameExample。
    example: GameExample
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
    # 逐行讲解：声明 invalid_action_count 的类型为 int；取值 0。
    invalid_action_count: int = 0
    # 逐行讲解：声明 reward 的类型为 float；取值 0.0。
    reward: float = 0.0
    # 逐行讲解：声明 advantage 的类型为 float；取值 0.0。
    advantage: float = 0.0
    # 逐行讲解：声明 progress_reported 的类型为 bool；调用 field（field(default=False, repr=False)）。
    progress_reported: bool = field(default=False, repr=False)


# 逐行讲解：定义类 RolloutBatch：一批轨迹及其 advantage 分组统计。
@dataclass(frozen=True)
class RolloutBatch:
    """一批轨迹及其 advantage 分组统计。"""

    # 逐行讲解：声明 trajectories 的类型为 list[Trajectory]。
    trajectories: list[Trajectory]
    # 逐行讲解：声明 advantage_stats 的类型为 AdvantageStats。
    advantage_stats: AdvantageStats


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


# 逐行讲解：定义类 GroupRuntime：一个游戏组的环境和 K 条分支。
@dataclass
class GroupRuntime:
    """一个游戏组的环境和 K 条分支。"""

    # 逐行讲解：声明 environment 的类型为 EnvironmentGroup。
    environment: EnvironmentGroup
    # 逐行讲解：声明 trajectories 的类型为 list[Trajectory]。
    trajectories: list[Trajectory]


# 逐行讲解：定义函数 _default_environment_factory：封装下方逻辑，供调用方复用。
def _default_environment_factory(
    example: GameExample,
    group_size: int,
    max_steps: int,
    *,
    seed: int,
    asynchronous: bool,
) -> EnvironmentGroup:
    # 逐行讲解：返回结果：ALFWorldGroup(example, group_size, max_steps, seed=seed, asynchronous=asynchronous)。
    return ALFWorldGroup(
        example,
        group_size,
        max_steps,
        seed=seed,
        asynchronous=asynchronous,
    )


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


# 逐行讲解：定义函数 _generation_budget：在 16K 轨迹上限内计算本轮最多可采样的 token 数。
def _generation_budget(
    prompt_tokens: list[int],
    config: RolloutConfig,
) -> int:
    """在 16K 轨迹上限内计算本轮最多可采样的 token 数。"""
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


# 逐行讲解：定义函数 _step_environments：不同游戏组可并发执行；组内一次提交 K 个动作。
def _step_environments(
    runtimes: list[GroupRuntime],
    pending: dict[tuple[int, int], GeneratedAction],
) -> dict[int, list[EnvironmentStep]]:
    """不同游戏组可并发执行；组内一次提交 K 个动作。"""
    # 逐行讲解：声明 calls 的类型为 list[tuple[int, EnvironmentGroup, list[str]]]；构造列表 []。
    calls: list[tuple[int, EnvironmentGroup, list[str]]] = []
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
        # 逐行讲解：计算 len(trajectory.steps) + 1，赋给 step_index。
        step_index = len(trajectory.steps) + 1
        # 逐行讲解：取值 f'alfworld-{trajectory.game_index}-{trajectory.group_index}-{step_index}'，赋给 call_id。
        call_id = (
            f"alfworld-{trajectory.game_index}-{trajectory.group_index}-{step_index}"
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
        # 逐行讲解：取值 environment_step.won，赋给 trajectory.won。
        trajectory.won = environment_step.won

        # 逐行讲解：取值 len(trajectory.steps) >= config.max_steps，赋给 reached_limit。
        reached_limit = len(trajectory.steps) >= config.max_steps
        # 逐行讲解：检查条件 environment_step.done or reached_limit；成立时执行下方分支。
        if environment_step.done or reached_limit:
            # 逐行讲解：取值 True，赋给 trajectory.done。
            trajectory.done = True
            # 逐行讲解：取值 not trajectory.won，赋给 trajectory.truncated。
            trajectory.truncated = not trajectory.won
            # 逐行讲解：检查条件 trajectory.won；成立时执行下方分支。
            if trajectory.won:
                # 逐行讲解：取值 'won'，赋给 trajectory.stop_reason。
                trajectory.stop_reason = "won"
            # 逐行讲解：检查条件 environment_step.done；成立时执行下方分支。
            elif environment_step.done:
                # 逐行讲解：取值 'environment_done'，赋给 trajectory.stop_reason。
                trajectory.stop_reason = "environment_done"
            else:
                # 逐行讲解：取值 'max_steps'，赋给 trajectory.stop_reason。
                trajectory.stop_reason = "max_steps"
        # 逐行讲解：检查条件 not prompt_fits；成立时执行下方分支。
        elif not prompt_fits:
            # 逐行讲解：取值 True，赋给 trajectory.done。
            trajectory.done = True
            # 逐行讲解：取值 True，赋给 trajectory.truncated。
            trajectory.truncated = True
            # 逐行讲解：取值 'trajectory_too_long'，赋给 trajectory.stop_reason。
            trajectory.stop_reason = "trajectory_too_long"


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


# 逐行讲解：定义函数 _initialize_runtimes：封装下方逻辑，供调用方复用。
def _initialize_runtimes(
    examples: list[GameExample],
    config: RolloutConfig,
    environment_factory: EnvironmentFactory,
) -> list[GroupRuntime]:
    # 逐行讲解：声明 runtimes 的类型为 list[GroupRuntime]；构造列表 []。
    runtimes: list[GroupRuntime] = []
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：遍历 enumerate(examples)，将每项绑定到 (game_index, example)。
        for game_index, example in enumerate(examples):
            # 逐行讲解：调用 environment_factory（environment_factory(example, config.group_size, config.max_steps, seed=config.seed + game_index, asynchrono…），赋给 environment。
            environment = environment_factory(
                example,
                config.group_size,
                config.max_steps,
                seed=config.seed + game_index,
                asynchronous=config.environment_asynchronous,
            )
            # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
            try:
                # 逐行讲解：调用 environment.reset（environment.reset()），赋给 states。
                states = environment.reset()
                # 逐行讲解：检查条件 len(states) != config.group_size；成立时执行下方分支。
                if len(states) != config.group_size:
                    # 逐行讲解：抛出异常：ValueError('环境 reset 返回数量与 group_size 不一致')。
                    raise ValueError("环境 reset 返回数量与 group_size 不一致")
                # 逐行讲解：检查条件 len({state.observation for state in states}) != 1；成立时执行下方分支。
                if len({state.observation for state in states}) != 1:
                    # 逐行讲解：抛出异常：RuntimeError('同游戏 group 的初始 observation 不一致')。
                    raise RuntimeError("同游戏 group 的初始 observation 不一致")
                # 逐行讲解：检查条件 len({state.task for state in states}) != 1；成立时执行下方分支。
                if len({state.task for state in states}) != 1:
                    # 逐行讲解：抛出异常：RuntimeError('同游戏 group 的初始 task 不一致')。
                    raise RuntimeError("同游戏 group 的初始 task 不一致")
            except Exception:
                # 逐行讲解：调用 environment.close（environment.close()）。
                environment.close()
                # 逐行讲解：抛出异常：重新抛出当前异常。
                raise

            # 逐行讲解：取值 f'{game_index}:{example.id}'，赋给 group_id。
            group_id = f"{game_index}:{example.id}"
            # 逐行讲解：通过推导式逐项生成结果：[Trajectory(example=example, group_id=group_id, game_index=game_index, group_index=branch_index, task=state…，赋给 trajectories。
            trajectories = [
                Trajectory(
                    example=example,
                    group_id=group_id,
                    game_index=game_index,
                    group_index=branch_index,
                    task=state.task,
                    initial_observation=state.observation,
                    current_observation=state.observation,
                    current_admissible_actions=state.admissible_actions,
                    messages=initial_messages(
                        task=state.task,
                        observation=state.observation,
                        admissible_actions=state.admissible_actions,
                        include_admissible_actions=config.include_admissible_actions,
                    ),
                )
                for branch_index, state in enumerate(states)
            ]
            # 逐行讲解：向列表末尾追加一个元素（runtimes.append(GroupRuntime(environment, trajectories))）。
            runtimes.append(GroupRuntime(environment, trajectories))
    except Exception:
        # 逐行讲解：遍历 runtimes，将每项绑定到 runtime。
        for runtime in runtimes:
            # 逐行讲解：调用 runtime.environment.close（runtime.environment.close()）。
            runtime.environment.close()
        # 逐行讲解：抛出异常：重新抛出当前异常。
        raise
    # 逐行讲解：返回结果：runtimes。
    return runtimes


# 逐行讲解：定义函数 rollout_batch：完成一批同游戏分组轨迹，再统一计算轨迹级 advantage。
def rollout_batch(
    sampling_client: Any,
    tokenizer: Any,
    examples: list[GameExample],
    config: RolloutConfig,
    *,
    environment_factory: EnvironmentFactory | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> RolloutBatch:
    """完成一批同游戏分组轨迹，再统一计算轨迹级 advantage。"""
    # 逐行讲解：检查条件 not examples；成立时执行下方分支。
    if not examples:
        # 逐行讲解：调用 AdvantageStats（AdvantageStats(0, 0)），赋给 empty_stats。
        empty_stats = AdvantageStats(0, 0)
        # 逐行讲解：返回结果：RolloutBatch([], empty_stats)。
        return RolloutBatch([], empty_stats)
    # 逐行讲解：取值 environment_factory or _default_environment_factory，赋给 factory。
    factory = environment_factory or _default_environment_factory
    # 逐行讲解：声明 runtimes 的类型为 list[GroupRuntime]；构造列表 []。
    runtimes: list[GroupRuntime] = []
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：调用 _initialize_runtimes（_initialize_runtimes(examples, config, factory)），赋给 runtimes。
        runtimes = _initialize_runtimes(examples, config, factory)
        # 逐行讲解：通过推导式逐项生成结果：[trajectory for runtime in runtimes for trajectory in runtime.trajectories]，赋给 trajectories。
        trajectories = [
            trajectory
            for runtime in runtimes
            for trajectory in runtime.trajectories
        ]

        # 逐行讲解：遍历 range(config.max_steps)，将每项绑定到 round_index。
        for round_index in range(config.max_steps):
            # 逐行讲解：检查条件 all((trajectory.done for trajectory in trajectories))；成立时执行下方分支。
            if all(trajectory.done for trajectory in trajectories):
                # 逐行讲解：结束当前循环。
                break

            # 逐行讲解：声明 requests 的类型为 list[SampleRequest]；构造列表 []。
            requests: list[SampleRequest] = []
            # 逐行讲解：检查条件 round_index == 0；成立时执行下方分支。
            if round_index == 0:
                # 同游戏 K 条根轨迹共享 prompt，一次请求分叉出 K 个候选。
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
                            # 逐行讲解：抛出异常：RuntimeError('同游戏 group 的首轮 prompt 不一致')。
                            raise RuntimeError("同游戏 group 的首轮 prompt 不一致")
                    # 逐行讲解：调用 tuple（tuple(((game_index, branch_index) for branch_index in range(config.group_size)))），赋给 targets。
                    targets = tuple(
                        (game_index, branch_index)
                        for branch_index in range(config.group_size)
                    )
                    # 逐行讲解：向列表末尾追加一个元素（requests.append(SampleRequest(targets=targets, prompt_tokens=prompt_tokens, num_samples=config.group_size, …）。
                    requests.append(
                        SampleRequest(
                            targets=targets,
                            prompt_tokens=prompt_tokens,
                            num_samples=config.group_size,
                            max_tokens=max_tokens,
                            seed=config.seed + game_index,
                        )
                    )
            else:
                # 分叉后每条轨迹状态不同，各自请求一个候选；
                # 不同轨迹的请求之间并发。
                for game_index, runtime in enumerate(runtimes):
                    # 逐行讲解：遍历 enumerate(runtime.trajectories)，将每项绑定到 (branch_index, trajectory)。
                    for branch_index, trajectory in enumerate(runtime.trajectories):
                        # 逐行讲解：检查条件 trajectory.done；成立时执行下方分支。
                        if trajectory.done:
                            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                            continue
                        # 逐行讲解：调用 _request_for_trajectory（_request_for_trajectory(tokenizer, trajectory, (game_index, branch_index), config, seed=config.seed + game_…），赋给 request。
                        request = _request_for_trajectory(
                            tokenizer,
                            trajectory,
                            (game_index, branch_index),
                            config,
                            seed=(
                                config.seed
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
                # 逐行讲解：取值 True，赋给 trajectory.truncated。
                trajectory.truncated = True
                # 逐行讲解：取值 'max_steps'，赋给 trajectory.stop_reason。
                trajectory.stop_reason = "max_steps"
            # 逐行讲解：求和（sum((not step.valid_action for step in trajectory.steps))），赋给 trajectory.invalid_action_count。
            trajectory.invalid_action_count = sum(
                not step.valid_action for step in trajectory.steps
            )
            # 逐行讲解：计算 (1.0 if trajectory.won else 0.0) - INVALID_ACTION_PENALTY * trajectory.invalid_action_count，赋给 trajectory.reward。
            trajectory.reward = (
                (1.0 if trajectory.won else 0.0)
                - INVALID_ACTION_PENALTY * trajectory.invalid_action_count
            )
        # 逐行讲解：调用 _report_finished（_report_finished(trajectories, progress_callback)）。
        _report_finished(trajectories, progress_callback)
    finally:
        # 逐行讲解：遍历 runtimes，将每项绑定到 runtime。
        for runtime in runtimes:
            # 逐行讲解：调用 runtime.environment.close（runtime.environment.close()）。
            runtime.environment.close()

    # 逐行讲解：调用 assign_advantages（assign_advantages(trajectories)），赋给 advantage_stats。
    advantage_stats = assign_advantages(trajectories)
    # 逐行讲解：返回结果：RolloutBatch(trajectories, advantage_stats)。
    return RolloutBatch(trajectories, advantage_stats)


# 逐行讲解：定义函数 trajectory_record：把轨迹转成适合 JSONL 落盘和人工 review 的结构。
def trajectory_record(trajectory: Trajectory) -> dict[str, Any]:
    """把轨迹转成适合 JSONL 落盘和人工 review 的结构。"""
    # 逐行讲解：返回结果：{'game_id': trajectory.example.id, 'game_file': str(trajectory.example.game_file), 'split': trajectory.exam…。
    return {
        "game_id": trajectory.example.id,
        "game_file": str(trajectory.example.game_file),
        "split": trajectory.example.split,
        "task_type": trajectory.example.task_type,
        "group_id": trajectory.group_id,
        "group_index": trajectory.group_index,
        "task": trajectory.task,
        "initial_observation": trajectory.initial_observation,
        "messages": trajectory.messages,
        "won": trajectory.won,
        "truncated": trajectory.truncated,
        "stop_reason": trajectory.stop_reason,
        "invalid_action_count": trajectory.invalid_action_count,
        "reward": trajectory.reward,
        "advantage": trajectory.advantage,
        "steps": [
            {
                "step": step.index,
                "observation": step.observation,
                "admissible_actions": list(step.admissible_actions),
                "assistant_text": step.assistant_text,
                "call_id": step.call_id,
                "action": step.action,
                "valid_format": step.valid_format,
                "admissible": step.admissible,
                "valid_action": step.valid_action,
                "tool_response": step.tool_response,
                "next_observation": step.next_observation,
                "environment_score": step.score,
                "done": step.done,
                "won": step.won,
                "prompt_tokens": len(step.prompt_tokens),
                "completion_tokens": len(step.completion_tokens),
                "mean_old_logprob": (
                    sum(step.logprobs) / len(step.logprobs)
                    if step.logprobs
                    else 0.0
                ),
            }
            for step in trajectory.steps
        ],
    }
