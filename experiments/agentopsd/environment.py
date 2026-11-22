"""将同一个 ALFWorld 游戏实例化为一组独立的 TextWorld 环境。"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 threading 导入 RLock。
from threading import RLock
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：导入模块：textworld。
import textworld
# 逐行讲解：导入模块：textworld.envs.pddl.textgen，别名 textgen。
import textworld.envs.pddl.textgen as textgen
# 逐行讲解：导入模块：textworld.gym。
import textworld.gym
# 逐行讲解：从 alfworld.agents.environment.alfred_tw_env 导入 AlfredDemangler, AlfredInfos。
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

# 逐行讲解：从 data 导入 GameExample。
from data import GameExample
# 逐行讲解：从 protocol 导入 canonical_action。
from protocol import canonical_action


# 逐行讲解：调用 RLock（RLock()），赋给 _TEXTWORLD_SYNC_LOCK。
_TEXTWORLD_SYNC_LOCK = RLock()


# 逐行讲解：定义类 EnvironmentState：reset 后一条分支可见的初始状态。
@dataclass(frozen=True)
class EnvironmentState:
    """reset 后一条分支可见的初始状态。"""

    # 逐行讲解：声明 observation 的类型为 str。
    observation: str
    # 逐行讲解：声明 task 的类型为 str。
    task: str
    # 逐行讲解：声明 admissible_actions 的类型为 tuple[str, ...]。
    admissible_actions: tuple[str, ...]


# 逐行讲解：定义类 EnvironmentStep：执行一次文本动作后的环境结果。
@dataclass(frozen=True)
class EnvironmentStep:
    """执行一次文本动作后的环境结果。"""

    # 逐行讲解：声明 observation 的类型为 str。
    observation: str
    # 逐行讲解：声明 score 的类型为 float。
    score: float
    # 逐行讲解：声明 done 的类型为 bool。
    done: bool
    # 逐行讲解：声明 won 的类型为 bool。
    won: bool
    # 逐行讲解：声明 admissible 的类型为 bool。
    admissible: bool
    # 逐行讲解：声明 admissible_actions 的类型为 tuple[str, ...]。
    admissible_actions: tuple[str, ...]


# 逐行讲解：定义函数 extract_task：从 ALFWorld 初始 observation 中提取目标描述。
def extract_task(observation: str) -> str:
    """从 ALFWorld 初始 observation 中提取目标描述。"""
    # 逐行讲解：取值 'Your task is to: '，赋给 marker。
    marker = "Your task is to: "
    # 逐行讲解：返回结果：observation.split(marker, 1)[1].strip()。
    return observation.split(marker, 1)[1].strip()


# 逐行讲解：定义函数 _column：封装下方逻辑，供调用方复用。
def _column(infos: dict[str, Any], key: str, size: int, default: Any) -> list[Any]:
    # 逐行讲解：查询指定键，缺失时使用默认值（infos.get(key)），赋给 value。
    value = infos.get(key)
    # 逐行讲解：检查条件 value is None；成立时执行下方分支。
    if value is None:
        # 逐行讲解：返回结果：[default for _ in range(size)]。
        return [default for _ in range(size)]
    # 逐行讲解：检查条件 hasattr(value, 'tolist')；成立时执行下方分支。
    if hasattr(value, "tolist"):
        # 逐行讲解：调用 value.tolist（value.tolist()），赋给 value。
        value = value.tolist()
    # 逐行讲解：检查条件 isinstance(value, (list, tuple))；成立时执行下方分支。
    if isinstance(value, (list, tuple)):
        # 逐行讲解：调用 list（list(value)），赋给 values。
        values = list(value)
    else:
        # 逐行讲解：构造列表 [value]，赋给 values。
        values = [value]
    # 逐行讲解：返回结果：values。
    return values


# 逐行讲解：定义函数 _textworld_eval_scope_compatibility：修复 CPython 3.13+ 中 TextWorld 动态变量不可见的问题。
def _textworld_eval_scope_compatibility(environment: Any) -> Any:
    """修复 CPython 3.13+ 中 TextWorld 动态变量不可见的问题。

    TextWorld 1.7.0 通过 ``locals().update(...)`` 把 ``r``、``o`` 等游戏
    变量注入 ``eval``。CPython 3.13 起函数局部变量采用已定义的新语义，
    这种写法不会再改变后续 ``eval`` 可见的作用域。该 wrapper 会在同步
    环境以及每个异步子进程内部执行，因此无需修改虚拟环境中的第三方包。
    """
    # 逐行讲解：取值 textgen.EvalSymbol.derive，赋给 current_derive。
    current_derive = textgen.EvalSymbol.derive
    # 逐行讲解：检查条件 getattr(current_derive, '_alfworld_scope_compatible', False)；成立时执行下方分支。
    if getattr(current_derive, "_alfworld_scope_compatible", False):
        # 逐行讲解：返回结果：environment。
        return environment

    # 逐行讲解：定义函数 derive：封装下方逻辑，供调用方复用。
    def derive(symbol: Any, context: dict[str, Any] | None = None) -> list[Any]:
        # 逐行讲解：取值 context or symbol.context，赋给 context。
        context = context or symbol.context
        # 逐行讲解：调用 vars(textgen).copy（vars(textgen).copy()），赋给 namespace。
        namespace = vars(textgen).copy()
        # 逐行讲解：调用 namespace.update（namespace.update(context['variables'])）。
        namespace.update(context["variables"])
        # 逐行讲解：调用 eval（eval(symbol.expression, namespace, namespace)），赋给 value。
        value = eval(symbol.expression, namespace, namespace)
        # 逐行讲解：返回结果：[textgen.TerminalSymbol(value)]。
        return [textgen.TerminalSymbol(value)]

    # 逐行讲解：取值 True，赋给 derive._alfworld_scope_compatible。
    derive._alfworld_scope_compatible = True  # type: ignore[attr-defined]
    # 逐行讲解：取值 derive，赋给 textgen.EvalSymbol.derive。
    textgen.EvalSymbol.derive = derive
    # 逐行讲解：返回结果：environment。
    return environment


# 逐行讲解：定义函数 _create_textworld_batch：使用 ALFWorld 官方 demangler/infos wrapper 注册固定游戏。
def _create_textworld_batch(
    game_file: Path,
    *,
    batch_size: int,
    max_turns: int,
    asynchronous: bool,
) -> Any:
    """使用 ALFWorld 官方 demangler/infos wrapper 注册固定游戏。"""
    # 逐行讲解：调用 textworld.EnvInfos（textworld.EnvInfos(won=True, admissible_commands=True)），赋给 request_infos。
    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
    )
    # 逐行讲解：构造列表 [_textworld_eval_scope_compatibility, AlfredDemangler(shuffle=False), AlfredInfos]，赋给 wrappers。
    wrappers = [
        _textworld_eval_scope_compatibility,
        AlfredDemangler(shuffle=False),
        AlfredInfos,
    ]
    # 逐行讲解：调用 textworld.gym.register_games（textworld.gym.register_games([str(game_file)], request_infos, batch_size=batch_size, asynchronous=asynchron…），赋给 env_id。
    env_id = textworld.gym.register_games(
        [str(game_file)],
        request_infos,
        batch_size=batch_size,
        asynchronous=asynchronous,
        auto_reset=False,
        max_episode_steps=max_turns,
        wrappers=wrappers,
        name="agentopsd-alfworld",
    )
    # 逐行讲解：返回结果：textworld.gym.make(env_id)。
    return textworld.gym.make(env_id)


# 逐行讲解：定义函数 _close_async_workers：让 TextWorld 1.7.0 的异步 worker 正常退出，而不是在析构时强杀。
def _close_async_workers(environment: Any) -> bool:
    """让 TextWorld 1.7.0 的异步 worker 正常退出，而不是在析构时强杀。"""
    # 逐行讲解：调用 getattr（getattr(environment, 'batch_env', None)），赋给 batch。
    batch = getattr(environment, "batch_env", None)
    # 逐行讲解：调用 getattr（getattr(batch, 'envs', None)），赋给 children。
    children = getattr(batch, "envs", None)
    # 逐行讲解：检查条件 not children or not all((hasattr(child, '_pipe') and hasattr(child, '_process') for child in children))；成立时执行下方分支。
    if not children or not all(
        hasattr(child, "_pipe") and hasattr(child, "_process")
        for child in children
    ):
        # 逐行讲解：返回结果：False。
        return False

    # 逐行讲解：调用 type（type(children[0])），赋给 child_type。
    child_type = type(children[0])
    # 逐行讲解：取值 child_type.__del__，赋给 current_destructor。
    current_destructor = child_type.__del__
    # 逐行讲解：检查条件 not getattr(current_destructor, '_alfworld_close_compatible', False)；成立时执行下方分支。
    if not getattr(current_destructor, "_alfworld_close_compatible", False):
        # 逐行讲解：取值 current_destructor，赋给 original_destructor。
        original_destructor = current_destructor

        # 逐行讲解：定义函数 safe_destructor：封装下方逻辑，供调用方复用。
        def safe_destructor(child: Any) -> None:
            # 逐行讲解：检查条件 getattr(child, '_alfworld_closed', False)；成立时执行下方分支。
            if getattr(child, "_alfworld_closed", False):
                # 逐行讲解：返回，结束当前函数。
                return
            # 逐行讲解：调用 original_destructor（original_destructor(child)）。
            original_destructor(child)

        # 逐行讲解：取值 True，赋给 safe_destructor._alfworld_close_compatible。
        safe_destructor._alfworld_close_compatible = True  # type: ignore[attr-defined]
        # 逐行讲解：取值 safe_destructor，赋给 child_type.__del__。
        child_type.__del__ = safe_destructor

    # 逐行讲解：遍历 children，将每项绑定到 child。
    for child in children:
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：调用 child._pipe.send（child._pipe.send(('close', '', ()))）。
            child._pipe.send(("close", "", ()))
        except (BrokenPipeError, EOFError, OSError):
            # 逐行讲解：保留空语句块，此处不执行操作。
            pass

    # 逐行讲解：遍历 children，将每项绑定到 child。
    for child in children:
        # 逐行讲解：调用 child._process.join（child._process.join(timeout=2.0)）。
        child._process.join(timeout=2.0)
        # 逐行讲解：检查条件 child._process.is_alive()；成立时执行下方分支。
        if child._process.is_alive():
            # 逐行讲解：调用 child._process.terminate（child._process.terminate()）。
            child._process.terminate()
            # 逐行讲解：调用 child._process.join（child._process.join()）。
            child._process.join()
        # 逐行讲解：取值 True，赋给 child._alfworld_closed。
        child._alfworld_closed = True
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：调用 child._pipe.close（child._pipe.close()）。
            child._pipe.close()
        except OSError:
            # 逐行讲解：保留空语句块，此处不执行操作。
            pass

    # 避免 TextWorld 的 __del__ 再次向已经关闭的 pipe 发送 close。
    batch.envs.clear()
    # 逐行讲解：取值 None，赋给 environment.batch_env。
    environment.batch_env = None
    # 逐行讲解：返回结果：True。
    return True


# 逐行讲解：定义类 ALFWorldGroup：持有同一游戏的 K 个独立状态，供组内 rollout 使用。
class ALFWorldGroup:
    """持有同一游戏的 K 个独立状态，供组内 rollout 使用。"""

    # 逐行讲解：定义函数 __init__：封装下方逻辑，供调用方复用。
    def __init__(
        self,
        example: GameExample,
        group_size: int,
        max_turns: int,
        *,
        seed: int = 0,
        asynchronous: bool = True,
    ) -> None:
        # 逐行讲解：取值 example，赋给 self.example。
        self.example = example
        # 逐行讲解：取值 group_size，赋给 self.group_size。
        self.group_size = group_size
        # TextWorld 即使 asynchronous=True，也会让 batch_size=1 使用进程内
        # SyncBatchEnv；其中多个 TatSu parser 是模块级单例，不能跨线程调用。
        self._uses_in_process_textworld = group_size <= 1 or not asynchronous
        # 逐行讲解：调用 _create_textworld_batch（_create_textworld_batch(example.game_file, batch_size=group_size, max_turns=max_turns, asynchronous=asynchr…），赋给 self._env。
        self._env = _create_textworld_batch(
            example.game_file,
            batch_size=group_size,
            max_turns=max_turns,
            asynchronous=asynchronous,
        )
        # 逐行讲解：设置对应随机数生成器的种子（self._env.seed(seed)）。
        self._env.seed(seed)
        # 逐行讲解：声明 self._admissible_actions 的类型为 list[tuple[str, ...]]；计算 [tuple()] * group_size。
        self._admissible_actions: list[tuple[str, ...]] = [tuple()] * group_size
        # 逐行讲解：取值 False，赋给 self._closed。
        self._closed = False

    # 逐行讲解：定义函数 reset：重置整组环境并返回每条分支的初始状态。
    def reset(self) -> list[EnvironmentState]:
        """重置整组环境并返回每条分支的初始状态。"""
        # 逐行讲解：检查条件 self._uses_in_process_textworld；成立时执行下方分支。
        if self._uses_in_process_textworld:
            # 逐行讲解：进入上下文管理器 _TEXTWORLD_SYNC_LOCK，退出时执行其收尾逻辑。
            with _TEXTWORLD_SYNC_LOCK:
                # 逐行讲解：调用 self._env.reset（self._env.reset()），赋给 (observations, infos)。
                observations, infos = self._env.reset()
        else:
            # 逐行讲解：调用 self._env.reset（self._env.reset()），赋给 (observations, infos)。
            observations, infos = self._env.reset()
        # 逐行讲解：通过推导式逐项生成结果：[str(value) for value in observations]，赋给 observations。
        observations = [str(value) for value in observations]
        # 逐行讲解：调用 _column（_column(infos, 'admissible_commands', self.group_size, [])），赋给 commands。
        commands = _column(infos, "admissible_commands", self.group_size, [])
        # 逐行讲解：通过推导式逐项生成结果：[tuple((canonical_action(str(action)) for action in actions)) for actions in commands]，赋给 self._admissible_actions。
        self._admissible_actions = [
            tuple(canonical_action(str(action)) for action in actions)
            for actions in commands
        ]

        # 逐行讲解：调用 extract_task（extract_task(observations[0])），赋给 task。
        task = extract_task(observations[0])
        # 逐行讲解：返回结果：[EnvironmentState(observation=observation, task=task, admissible_actions=self._admissible_actions[index]) f…。
        return [
            EnvironmentState(
                observation=observation,
                task=task,
                admissible_actions=self._admissible_actions[index],
            )
            for index, observation in enumerate(observations)
        ]

    # 逐行讲解：定义函数 step：每个分支执行一个动作；score 仅记录，不参与自定义奖励。
    def step(self, actions: list[str]) -> list[EnvironmentStep]:
        """每个分支执行一个动作；score 仅记录，不参与自定义奖励。"""
        # 逐行讲解：通过推导式逐项生成结果：[canonical_action(action) for action in actions]，赋给 normalized。
        normalized = [canonical_action(action) for action in actions]
        # 逐行讲解：通过推导式逐项生成结果：[action in current for action, current in zip(normalized, self._admissible_actions)]，赋给 admissible。
        admissible = [
            action in current
            for action, current in zip(normalized, self._admissible_actions)
        ]
        # 逐行讲解：检查条件 self._uses_in_process_textworld；成立时执行下方分支。
        if self._uses_in_process_textworld:
            # 逐行讲解：进入上下文管理器 _TEXTWORLD_SYNC_LOCK，退出时执行其收尾逻辑。
            with _TEXTWORLD_SYNC_LOCK:
                # 逐行讲解：调用 self._env.step（self._env.step(normalized)），赋给 (observations, scores, dones, infos)。
                observations, scores, dones, infos = self._env.step(normalized)
        else:
            # 逐行讲解：调用 self._env.step（self._env.step(normalized)），赋给 (observations, scores, dones, infos)。
            observations, scores, dones, infos = self._env.step(normalized)
        # 逐行讲解：通过推导式逐项生成结果：[str(value) for value in observations]，赋给 observations。
        observations = [str(value) for value in observations]
        # 逐行讲解：通过推导式逐项生成结果：[bool(value) for value in _column(infos, 'won', self.group_size, False)]，赋给 won。
        won = [bool(value) for value in _column(infos, "won", self.group_size, False)]
        # 逐行讲解：调用 _column（_column(infos, 'admissible_commands', self.group_size, [])），赋给 commands。
        commands = _column(infos, "admissible_commands", self.group_size, [])
        # 逐行讲解：通过推导式逐项生成结果：[tuple((canonical_action(str(action)) for action in values)) for values in commands]，赋给 self._admissible_actions。
        self._admissible_actions = [
            tuple(canonical_action(str(action)) for action in values)
            for values in commands
        ]

        # 逐行讲解：返回结果：[EnvironmentStep(observation=observations[index], score=float(scores[index]), done=bool(dones[index]), won=…。
        return [
            EnvironmentStep(
                observation=observations[index],
                score=float(scores[index]),
                done=bool(dones[index]),
                won=won[index],
                admissible=admissible[index],
                admissible_actions=self._admissible_actions[index],
            )
            for index in range(self.group_size)
        ]

    # 逐行讲解：定义函数 close：释放 TextWorld 子进程。
    def close(self) -> None:
        """释放 TextWorld 子进程。"""
        # 逐行讲解：检查条件 not self._closed；成立时执行下方分支。
        if not self._closed:
            # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
            try:
                # 逐行讲解：检查条件 not _close_async_workers(self._env)；成立时执行下方分支。
                if not _close_async_workers(self._env):
                    # 逐行讲解：调用 self._env.close（self._env.close()）。
                    self._env.close()
            finally:
                # 逐行讲解：取值 True，赋给 self._closed。
                self._closed = True

    # 逐行讲解：定义函数 __enter__：封装下方逻辑，供调用方复用。
    def __enter__(self) -> "ALFWorldGroup":
        # 逐行讲解：返回结果：self。
        return self

    # 逐行讲解：定义函数 __exit__：封装下方逻辑，供调用方复用。
    def __exit__(self, *_: object) -> None:
        # 逐行讲解：调用 self.close（self.close()）。
        self.close()
