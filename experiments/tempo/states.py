"""TEMPO macro-step 起点状态的表示、保存与仓库管理。

一个 MacroState 是"环境 + 交互历史"的完整快照：
- 环境侧通过重放 action_history 恢复（TextWorld 确定性）；
- token 侧通过保存的 token_prefix 续跑，保证下一轮 prompt
  仍然是历史真实 token 序列的严格前缀扩展。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：random。
import random
# 逐行讲解：从 collections 导入 deque。
from collections import deque
# 逐行讲解：从 dataclasses 导入 dataclass, field。
from dataclasses import dataclass, field
# 逐行讲解：从 typing 导入 TYPE_CHECKING, Any。
from typing import TYPE_CHECKING, Any

# 逐行讲解：从 data 导入 GameExample。
from data import GameExample
# 逐行讲解：从 environment 导入 EnvironmentState。
from environment import EnvironmentState

# 逐行讲解：检查条件 TYPE_CHECKING；成立时执行下方分支。
if TYPE_CHECKING:
    # 逐行讲解：从 rollout 导入 RolloutConfig, Trajectory。
    from rollout import RolloutConfig, Trajectory


# 逐行讲解：定义类 MacroState：一个可续跑的 macro-step 起点。
@dataclass
class MacroState:
    """一个可续跑的 macro-step 起点。"""

    # 逐行讲解：声明 example 的类型为 GameExample。
    example: GameExample
    # 逐行讲解：声明 task 的类型为 str。
    task: str
    # 逐行讲解：声明 initial_observation 的类型为 str。
    initial_observation: str
    # 环境侧快照：按顺序重放即可恢复到 current_observation。
    action_history: tuple[str, ...]
    # actor 侧快照：完整 chat 历史；初始状态时为 initial_messages。
    messages: list[dict[str, Any]] = field(default_factory=list)
    # 续跑用的真实 token 前缀；None 表示尚未生成（初始状态，rollout 现算）。
    token_prefix: list[int] | None = None
    # 该状态之前已消耗的交互轮数（用于 T 上限与 macro_index 推进）。
    round_index: int = 0
    # 逐行讲解：声明 macro_index 的类型为 int；取值 0。
    macro_index: int = 0
    # 重放时必须使用生成该前缀时的同一环境种子。
    environment_seed: int = 0
    # 逐行讲解：声明 current_observation 的类型为 str；取值 ''。
    current_observation: str = ""
    # 逐行讲解：声明 current_admissible_actions 的类型为 tuple[str, ...]；构造元组 ()。
    current_admissible_actions: tuple[str, ...] = ()

    # 逐行讲解：定义函数 fork_messages：分支各自追加消息，不能共享同一个 list。
    def fork_messages(self) -> list[dict[str, Any]]:
        """分支各自追加消息，不能共享同一个 list。"""
        # 逐行讲解：返回结果：[dict(message) for message in self.messages]。
        return [dict(message) for message in self.messages]


# 逐行讲解：定义函数 initial_macro_state：从一次环境 reset 的结果构造第 0 个 macro-step 起点。
def initial_macro_state(
    example: GameExample,
    state: EnvironmentState,
    *,
    messages: list[dict[str, Any]],
    environment_seed: int,
) -> MacroState:
    """从一次环境 reset 的结果构造第 0 个 macro-step 起点。"""
    # 逐行讲解：返回结果：MacroState(example=example, task=state.task, initial_observation=state.observation, action_history=(), mess…。
    return MacroState(
        example=example,
        task=state.task,
        initial_observation=state.observation,
        action_history=(),
        messages=messages,
        token_prefix=None,
        round_index=0,
        macro_index=0,
        environment_seed=environment_seed,
        current_observation=state.observation,
        current_admissible_actions=state.admissible_actions,
    )


# 逐行讲解：定义类 StateStore：endpoint 状态的定长仓库，跨训练轮次均匀采样消费。
class StateStore:
    """endpoint 状态的定长仓库，跨训练轮次均匀采样消费。

    每轮 TD 更新存入 batch 内所有非终局分支的终点，并按容量
    淘汰最旧的状态；这近似博客中"部分段末状态被保存，供后续
    rollout 从这里继续执行"的设定。
    """

    # 逐行讲解：定义函数 __init__：封装下方逻辑，供调用方复用。
    def __init__(self, capacity: int, seed: int) -> None:
        # 逐行讲解：检查条件 capacity < 1；成立时执行下方分支。
        if capacity < 1:
            # 逐行讲解：抛出异常：ValueError('capacity 必须大于等于 1')。
            raise ValueError("capacity 必须大于等于 1")
        # 逐行讲解：声明 self._states 的类型为 deque[MacroState]；调用 deque（deque(maxlen=capacity)）。
        self._states: deque[MacroState] = deque(maxlen=capacity)
        # 逐行讲解：调用 random.Random（random.Random(seed)），赋给 self._rng。
        self._rng = random.Random(seed)

    # 逐行讲解：定义函数 add：封装下方逻辑，供调用方复用。
    def add(self, states: list[MacroState]) -> None:
        # 逐行讲解：把一组元素依次加入列表（self._states.extend(states)）。
        self._states.extend(states)

    # 逐行讲解：定义函数 sample：均匀抽出 count 个互不重复的状态；不足时有多少返回多少。
    def sample(self, count: int) -> list[MacroState]:
        """均匀抽出 count 个互不重复的状态；不足时有多少返回多少。"""
        # 逐行讲解：调用 min（min(count, len(self._states))），赋给 count。
        count = min(count, len(self._states))
        # 逐行讲解：提交生成采样请求（self._rng.sample(range(len(self._states)), count)），赋给 indices。
        indices = self._rng.sample(range(len(self._states)), count)
        # 逐行讲解：返回结果：[self._states[index] for index in indices]。
        return [self._states[index] for index in indices]

    # 逐行讲解：定义函数 __len__：封装下方逻辑，供调用方复用。
    def __len__(self) -> int:
        # 逐行讲解：返回结果：len(self._states)。
        return len(self._states)


# 逐行讲解：定义函数 endpoint_state：把一条停在 macro_boundary 的分支转成下一轮起点；其余情况返回 None。
def endpoint_state(
    start_state: MacroState,
    trajectory: "Trajectory",
    *,
    config: "RolloutConfig",
) -> MacroState | None:
    """把一条停在 macro_boundary 的分支转成下一轮起点；其余情况返回 None。

    终局（won / 环境结束 / 轮数用尽 / token 超限）分支没有未来，
    不入库；macro_boundary 但 token 前缀已无续跑空间的分支同样丢弃。
    """
    # 逐行讲解：检查条件 trajectory.stop_reason != 'macro_boundary'；成立时执行下方分支。
    if trajectory.stop_reason != "macro_boundary":
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：检查条件 trajectory.next_prompt_tokens is None；成立时执行下方分支。
    if trajectory.next_prompt_tokens is None:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：计算 start_state.round_index + len(trajectory.steps)，赋给 round_index。
    round_index = start_state.round_index + len(trajectory.steps)
    # 逐行讲解：检查条件 round_index >= config.max_episode_steps；成立时执行下方分支。
    if round_index >= config.max_episode_steps:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：检查条件 len(trajectory.next_prompt_tokens) >= config.max_trajectory_tokens；成立时执行下方分支。
    if len(trajectory.next_prompt_tokens) >= config.max_trajectory_tokens:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：MacroState(example=start_state.example, task=start_state.task, initial_observation=start_state.initial_obse…。
    return MacroState(
        example=start_state.example,
        task=start_state.task,
        initial_observation=start_state.initial_observation,
        action_history=start_state.action_history
        + tuple(step.action for step in trajectory.steps),
        messages=[dict(message) for message in trajectory.messages],
        token_prefix=list(trajectory.next_prompt_tokens),
        round_index=round_index,
        macro_index=start_state.macro_index + 1,
        environment_seed=start_state.environment_seed,
        current_observation=trajectory.current_observation,
        current_admissible_actions=trajectory.current_admissible_actions,
    )


# 逐行讲解：定义函数 boundary_states：提取轨迹在每个 H 轮边界上的非终局快照（warm-up 的 critic 训练点）。
def boundary_states(
    start_state: MacroState,
    trajectory: "Trajectory",
    *,
    macro_rounds: int,
    max_trajectory_tokens: int,
) -> list[MacroState]:
    """提取轨迹在每个 H 轮边界上的非终局快照（warm-up 的 critic 训练点）。

    warm-up 阶段 rollout 会跑到终局；这里把 H、2H、... 轮处的中间状态
    重建出来，配合轨迹最终结局（MC return）构造无需 bootstrap 的 target。
    """
    # 逐行讲解：从 protocol 导入 tool_message。
    from protocol import tool_message

    # 逐行讲解：声明 snapshots 的类型为 list[MacroState]；构造列表 []。
    snapshots: list[MacroState] = []
    # 逐行讲解：调用 len（len(trajectory.steps)），赋给 total_steps。
    total_steps = len(trajectory.steps)
    # 逐行讲解：遍历 range(macro_rounds, total_steps + 1, macro_rounds)，将每项绑定到 rounds。
    for rounds in range(macro_rounds, total_steps + 1, macro_rounds):
        # 轨迹恰好在该边界终止则没有"从该状态继续"的语义。
        if total_steps == rounds and trajectory.stop_reason != "macro_boundary":
            # 逐行讲解：结束当前循环。
            break

        # 逐行讲解：调用 start_state.fork_messages（start_state.fork_messages()），赋给 messages。
        messages = start_state.fork_messages()
        # 逐行讲解：遍历 trajectory.steps[:rounds]，将每项绑定到 step。
        for step in trajectory.steps[:rounds]:
            # 逐行讲解：向列表末尾追加一个元素（messages.append({'role': 'assistant', 'content': step.assistant_text})）。
            messages.append({"role": "assistant", "content": step.assistant_text})
            # 逐行讲解：向列表末尾追加一个元素（messages.append(tool_message(step.call_id, step.tool_response))）。
            messages.append(tool_message(step.call_id, step.tool_response))

        # 逐行讲解：检查条件 total_steps > rounds；成立时执行下方分支。
        if total_steps > rounds:
            # 逐行讲解：调用 list（list(trajectory.steps[rounds].prompt_tokens)），赋给 token_prefix。
            token_prefix = list(trajectory.steps[rounds].prompt_tokens)
            # 逐行讲解：取值 trajectory.steps[rounds].admissible_actions，赋给 admissible。
            admissible = trajectory.steps[rounds].admissible_actions
        else:
            # 逐行讲解：检查条件 trajectory.next_prompt_tokens is None；成立时执行下方分支。
            if trajectory.next_prompt_tokens is None:
                # 逐行讲解：结束当前循环。
                break
            # 逐行讲解：调用 list（list(trajectory.next_prompt_tokens)），赋给 token_prefix。
            token_prefix = list(trajectory.next_prompt_tokens)
            # 逐行讲解：取值 trajectory.current_admissible_actions，赋给 admissible。
            admissible = trajectory.current_admissible_actions
        # 逐行讲解：检查条件 len(token_prefix) >= max_trajectory_tokens；成立时执行下方分支。
        if len(token_prefix) >= max_trajectory_tokens:
            # 逐行讲解：结束当前循环。
            break

        # 逐行讲解：向列表末尾追加一个元素（snapshots.append(MacroState(example=start_state.example, task=start_state.task, initial_observation=start_s…）。
        snapshots.append(
            MacroState(
                example=start_state.example,
                task=start_state.task,
                initial_observation=start_state.initial_observation,
                action_history=start_state.action_history
                + tuple(step.action for step in trajectory.steps[:rounds]),
                messages=messages,
                token_prefix=token_prefix,
                round_index=start_state.round_index + rounds,
                macro_index=start_state.macro_index + rounds // macro_rounds,
                environment_seed=start_state.environment_seed,
                current_observation=trajectory.steps[rounds - 1].next_observation,
                current_admissible_actions=admissible,
            )
        )
    # 逐行讲解：返回结果：snapshots。
    return snapshots
