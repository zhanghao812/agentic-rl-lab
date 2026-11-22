"""TEMPO 的信号拼装与训练 Datum 构造。

本文件对应博客中 TEMPO 区别于普通 GRPO 的全部本地逻辑：
- actor 的 return = 段内环境奖励 + 终点 critic 估值（截断尾部由 V̂ 补齐）；
- G（TD target）= 同一起点 N 条分支 return 的均值；
- critic 的 reward = -|V̂ - G| / R_max（解析失败按最差处理）；
- 两侧都做组内中心化，用同一个 importance_sampling loss 更新。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 typing 导入 Sequence。
from typing import Sequence

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio

# 逐行讲解：从 critic 导入 CriticSample。
from critic import CriticSample
# 逐行讲解：从 rollout 导入 MacroGroup, Trajectory。
from rollout import MacroGroup, Trajectory
# 逐行讲解：从 states 导入 MacroState。
from states import MacroState


# 逐行讲解：取值 1.0，赋给 R_MAX。
R_MAX = 1.0
# 逐行讲解：取值 1e-12，赋给 ADVANTAGE_EPSILON。
ADVANTAGE_EPSILON = 1e-12


# 逐行讲解：定义类 ActorSignals：一个起点组内 N 条分支的 return 与 advantage。
@dataclass(frozen=True)
class ActorSignals:
    """一个起点组内 N 条分支的 return 与 advantage。"""

    # 逐行讲解：声明 segment_rewards 的类型为 list[float]。
    segment_rewards: list[float]
    # 逐行讲解：声明 endpoint_values 的类型为 list[float]。
    endpoint_values: list[float]
    # 逐行讲解：声明 returns 的类型为 list[float]。
    returns: list[float]
    # 逐行讲解：声明 value_target 的类型为 float。
    value_target: float
    # 逐行讲解：声明 advantages 的类型为 list[float]。
    advantages: list[float]
    # 逐行讲解：声明 degenerate 的类型为 bool。
    degenerate: bool
    # 段内奖励相同（全 0）且非终局分支间的估值差 —— 博客"放置骑士"指标。
    zero_reward_value_std: float


# 逐行讲解：定义类 CriticSignals：同一起点上 K 次独立估值的误差奖励与 advantage。
@dataclass(frozen=True)
class CriticSignals:
    """同一起点上 K 次独立估值的误差奖励与 advantage。"""

    # 逐行讲解：声明 rewards 的类型为 list[float]。
    rewards: list[float]
    # 逐行讲解：声明 advantages 的类型为 list[float]。
    advantages: list[float]
    # 逐行讲解：声明 errors 的类型为 list[float]。
    errors: list[float]
    # 逐行讲解：声明 degenerate 的类型为 bool。
    degenerate: bool
    # 逐行讲解：声明 parse_failures 的类型为 int。
    parse_failures: int
    # 逐行讲解：声明 mean_estimate 的类型为 float | None。
    mean_estimate: float | None


# 逐行讲解：定义函数 _terminal：macro_boundary 之外的停机原因都视为终局（没有可 bootstrap 的未来）。
def _terminal(trajectory: Trajectory) -> bool:
    """macro_boundary 之外的停机原因都视为终局（没有可 bootstrap 的未来）。"""
    # 逐行讲解：返回结果：trajectory.stop_reason != 'macro_boundary'。
    return trajectory.stop_reason != "macro_boundary"


# 逐行讲解：定义函数 assemble_actor_group：计算 N 条分支的 return、G 与组内中心化 advantage。
def assemble_actor_group(group: MacroGroup) -> ActorSignals:
    """计算 N 条分支的 return、G 与组内中心化 advantage。"""
    # 逐行讲解：取值 group.trajectories，赋给 trajectories。
    trajectories = group.trajectories
    # 逐行讲解：通过推导式逐项生成结果：[trajectory.segment_reward for trajectory in trajectories]，赋给 rewards。
    rewards = [trajectory.segment_reward for trajectory in trajectories]
    # 终局分支没有未来，终点估值恒为 0；只有停在 macro_boundary 的分支
    # 由 critic 估出的 endpoint_value 参与 bootstrap。
    values = [
        0.0 if _terminal(trajectory) else trajectory.endpoint_value
        for trajectory in trajectories
    ]
    # 逐行讲解：通过推导式逐项生成结果：[reward + value for reward, value in zip(rewards, values, strict=True)]，赋给 returns。
    returns = [reward + value for reward, value in zip(rewards, values, strict=True)]
    # 逐行讲解：计算 sum(returns) / len(returns)，赋给 value_target。
    value_target = sum(returns) / len(returns)
    # 逐行讲解：取值 value_target，赋给 mean_return。
    mean_return = value_target
    # 逐行讲解：通过推导式逐项生成结果：[item - mean_return for item in returns]，赋给 advantages。
    advantages = [item - mean_return for item in returns]
    # 逐行讲解：遍历 zip(trajectories, advantages, strict=True)，将每项绑定到 (trajectory, advantage)。
    for trajectory, advantage in zip(trajectories, advantages, strict=True):
        # 逐行讲解：计算 trajectory.segment_reward + (0.0 if _terminal(trajectory) else trajectory.endpoint_value)，赋给 trajectory.branch_return。
        trajectory.branch_return = (
            trajectory.segment_reward
            + (0.0 if _terminal(trajectory) else trajectory.endpoint_value)
        )
        # 逐行讲解：取值 advantage，赋给 trajectory.advantage。
        trajectory.advantage = advantage

    # 逐行讲解：通过推导式逐项生成结果：[value for reward, value, trajectory in zip(rewards, values, trajectories, strict=True) if abs(reward) <= A…，赋给 zero_reward_values。
    zero_reward_values = [
        value
        for reward, value, trajectory in zip(
            rewards, values, trajectories, strict=True
        )
        if abs(reward) <= ADVANTAGE_EPSILON and not _terminal(trajectory)
    ]
    # 逐行讲解：按条件选择结果：float(np.std(np.asarray(zero_reward_values, dtype=np.float64))) if len(zero_reward_values) >= 2 else 0.0，赋给 zero_reward_value_std。
    zero_reward_value_std = (
        float(np.std(np.asarray(zero_reward_values, dtype=np.float64)))
        if len(zero_reward_values) >= 2
        else 0.0
    )

    # 逐行讲解：返回结果：ActorSignals(segment_rewards=rewards, endpoint_values=values, returns=returns, value_target=value_target, a…。
    return ActorSignals(
        segment_rewards=rewards,
        endpoint_values=values,
        returns=returns,
        value_target=value_target,
        advantages=advantages,
        degenerate=all(
            abs(advantage) <= ADVANTAGE_EPSILON for advantage in advantages
        ),
        zero_reward_value_std=zero_reward_value_std,
    )


# 逐行讲解：定义函数 assemble_critic_group：估值误差转奖励并组内中心化；解析失败的样本按最差 reward 处理。
def assemble_critic_group(
    samples: Sequence[CriticSample],
    value_target: float,
) -> CriticSignals:
    """估值误差转奖励并组内中心化；解析失败的样本按最差 reward 处理。"""
    # 逐行讲解：声明 rewards 的类型为 list[float]；构造列表 []。
    rewards: list[float] = []
    # 逐行讲解：声明 errors 的类型为 list[float]；构造列表 []。
    errors: list[float] = []
    # 逐行讲解：取值 0，赋给 parse_failures。
    parse_failures = 0
    # 逐行讲解：遍历 samples，将每项绑定到 sample。
    for sample in samples:
        # 逐行讲解：检查条件 sample.value is None；成立时执行下方分支。
        if sample.value is None:
            # 逐行讲解：更新 parse_failures：parse_failures += 1。
            parse_failures += 1
            # 逐行讲解：向列表末尾追加一个元素（rewards.append(-R_MAX)）。
            rewards.append(-R_MAX)
            # 逐行讲解：向列表末尾追加一个元素（errors.append(R_MAX)）。
            errors.append(R_MAX)
        else:
            # 逐行讲解：计算 abs(sample.value - value_target) / R_MAX，赋给 error。
            error = abs(sample.value - value_target) / R_MAX
            # 逐行讲解：向列表末尾追加一个元素（errors.append(error)）。
            errors.append(error)
            # 逐行讲解：向列表末尾追加一个元素（rewards.append(-error)）。
            rewards.append(-error)

    # 逐行讲解：按条件选择结果：sum(rewards) / len(rewards) if rewards else 0.0，赋给 mean_reward。
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    # 逐行讲解：通过推导式逐项生成结果：[reward - mean_reward for reward in rewards]，赋给 advantages。
    advantages = [reward - mean_reward for reward in rewards]

    # 逐行讲解：通过推导式逐项生成结果：[sample.value for sample in samples if sample.value is not None]，赋给 valid。
    valid = [sample.value for sample in samples if sample.value is not None]
    # 逐行讲解：返回结果：CriticSignals(rewards=rewards, advantages=advantages, errors=errors, degenerate=all((abs(advantage) <= ADVA…。
    return CriticSignals(
        rewards=rewards,
        advantages=advantages,
        errors=errors,
        degenerate=all(
            abs(advantage) <= ADVANTAGE_EPSILON for advantage in advantages
        ),
        parse_failures=parse_failures,
        mean_estimate=(sum(valid) / len(valid)) if valid else None,
    )


# 逐行讲解：定义类 TrainingDatum：PyTRIO Datum 及其装箱、追踪元数据。
@dataclass(frozen=True)
class TrainingDatum:
    """PyTRIO Datum 及其装箱、追踪元数据。"""

    # 逐行讲解：声明 datum 的类型为 trio.Datum。
    datum: trio.Datum
    # 逐行讲解：声明 num_tokens 的类型为 int。
    num_tokens: int
    # 逐行讲解：声明 loss_tokens 的类型为 int。
    loss_tokens: int
    # 逐行讲解：声明 kind 的类型为 str。
    kind: str  # "actor" | "critic"
    # 逐行讲解：声明 game_id 的类型为 str。
    game_id: str
    # 逐行讲解：声明 group_index 的类型为 int。
    group_index: int


# 逐行讲解：定义函数 build_segment_datum：把一个 macro-step 分支构造成右移对齐的 Datum。
def build_segment_datum(
    start_state: MacroState,
    trajectory: Trajectory,
    *,
    max_sequence_tokens: int,
) -> TrainingDatum:
    """把一个 macro-step 分支构造成右移对齐的 Datum。

    基座是保存的 token_prefix（初始状态时为空，由第一步 prompt 兜底）。
    每轮 prompt 相对上一轮只增加 assistant 闭合符和 tool observation，
    这些环境 token 的 old logprob/advantage 均为 0；本段所有 assistant
    completion 使用采样时的 old logprob 和该分支的组内中心化 advantage。
    """
    # 逐行讲解：检查条件 not trajectory.steps；成立时执行下方分支。
    if not trajectory.steps:
        # 逐行讲解：抛出异常：ValueError('不能用空分支构造 Datum')。
        raise ValueError("不能用空分支构造 Datum")

    # 逐行讲解：取值 start_state.token_prefix，赋给 base。
    base = start_state.token_prefix
    # 逐行讲解：声明 full_tokens 的类型为 list[int]；构造列表 []。
    full_tokens: list[int] = []
    # 逐行讲解：声明 old_logprobs_by_token 的类型为 list[float]；构造列表 []。
    old_logprobs_by_token: list[float] = []
    # 逐行讲解：声明 advantages_by_token 的类型为 list[float]；构造列表 []。
    advantages_by_token: list[float] = []
    # 逐行讲解：取值 0，赋给 assistant_token_count。
    assistant_token_count = 0

    # 逐行讲解：遍历 trajectory.steps，将每项绑定到 step。
    for step in trajectory.steps:
        # 逐行讲解：检查条件 not step.prompt_tokens；成立时执行下方分支。
        if not step.prompt_tokens:
            # 逐行讲解：抛出异常：ValueError(f'第 {step.index} 步 prompt 不能为空')。
            raise ValueError(f"第 {step.index} 步 prompt 不能为空")
        # 逐行讲解：检查条件 not step.completion_tokens；成立时执行下方分支。
        if not step.completion_tokens:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：检查条件 len(step.completion_tokens) != len(step.logprobs)；成立时执行下方分支。
        if len(step.completion_tokens) != len(step.logprobs):
            # 逐行讲解：抛出异常：ValueError(f'第 {step.index} 步 completion token 与 old logprob 长度不一致')。
            raise ValueError(
                f"第 {step.index} 步 completion token 与 old logprob 长度不一致"
            )

        # 逐行讲解：检查条件 not full_tokens；成立时执行下方分支。
        if not full_tokens:
            # 逐行讲解：检查条件 base is None；成立时执行下方分支。
            if base is None:
                # 逐行讲解：取值 step.prompt_tokens，赋给 delta_environment。
                delta_environment = step.prompt_tokens
            # 逐行讲解：检查条件 step.prompt_tokens[:len(base)] == base；成立时执行下方分支。
            elif step.prompt_tokens[: len(base)] == base:
                # 保存的 token 前缀是上下文的一部分，必须完整保留，
                # 训练信号为 0。
                full_tokens.extend(base)
                # 逐行讲解：把一组元素依次加入列表（old_logprobs_by_token.extend([0.0] * len(base))）。
                old_logprobs_by_token.extend([0.0] * len(base))
                # 逐行讲解：把一组元素依次加入列表（advantages_by_token.extend([0.0] * len(base))）。
                advantages_by_token.extend([0.0] * len(base))
                # 逐行讲解：按索引或切片读取 step.prompt_tokens[len(base):]，赋给 delta_environment。
                delta_environment = step.prompt_tokens[len(base) :]
            else:
                # 逐行讲解：抛出异常：ValueError('分支首步 prompt 不是保存前缀的扩展')。
                raise ValueError("分支首步 prompt 不是保存前缀的扩展")
        # 逐行讲解：检查条件 step.prompt_tokens[:len(full_tokens)] == full_tokens；成立时执行下方分支。
        elif step.prompt_tokens[: len(full_tokens)] == full_tokens:
            # 逐行讲解：按索引或切片读取 step.prompt_tokens[len(full_tokens):]，赋给 delta_environment。
            delta_environment = step.prompt_tokens[len(full_tokens) :]
        else:
            # 逐行讲解：抛出异常：ValueError(f'第 {step.index} 步 prompt 不是已有轨迹的前缀扩展')。
            raise ValueError(f"第 {step.index} 步 prompt 不是已有轨迹的前缀扩展")

        # 逐行讲解：把一组元素依次加入列表（full_tokens.extend(delta_environment)）。
        full_tokens.extend(delta_environment)
        # 逐行讲解：把一组元素依次加入列表（old_logprobs_by_token.extend([0.0] * len(delta_environment))）。
        old_logprobs_by_token.extend([0.0] * len(delta_environment))
        # 逐行讲解：把一组元素依次加入列表（advantages_by_token.extend([0.0] * len(delta_environment))）。
        advantages_by_token.extend([0.0] * len(delta_environment))

        # 逐行讲解：把一组元素依次加入列表（full_tokens.extend(step.completion_tokens)）。
        full_tokens.extend(step.completion_tokens)
        # 逐行讲解：把一组元素依次加入列表（old_logprobs_by_token.extend(step.logprobs)）。
        old_logprobs_by_token.extend(step.logprobs)
        # 逐行讲解：把一组元素依次加入列表（advantages_by_token.extend([trajectory.advantage] * len(step.completion_tokens))）。
        advantages_by_token.extend(
            [trajectory.advantage] * len(step.completion_tokens)
        )
        # 逐行讲解：更新 assistant_token_count：assistant_token_count += len(step.completion_tokens)。
        assistant_token_count += len(step.completion_tokens)

    # macro_boundary 分支保留最后一段 tool observation 作为上下文，
    # 训练信号明确为 0。
    if trajectory.next_prompt_tokens is not None:
        # 逐行讲解：检查条件 trajectory.next_prompt_tokens[:len(full_tokens)] != full_tokens；成立时执行下方分支。
        if trajectory.next_prompt_tokens[: len(full_tokens)] != full_tokens:
            # 逐行讲解：抛出异常：ValueError('最终 tool observation 不是已有轨迹的前缀扩展')。
            raise ValueError("最终 tool observation 不是已有轨迹的前缀扩展")
        # 逐行讲解：按索引或切片读取 trajectory.next_prompt_tokens[len(full_tokens):]，赋给 final_environment。
        final_environment = trajectory.next_prompt_tokens[len(full_tokens) :]
        # 逐行讲解：把一组元素依次加入列表（full_tokens.extend(final_environment)）。
        full_tokens.extend(final_environment)
        # 逐行讲解：把一组元素依次加入列表（old_logprobs_by_token.extend([0.0] * len(final_environment))）。
        old_logprobs_by_token.extend([0.0] * len(final_environment))
        # 逐行讲解：把一组元素依次加入列表（advantages_by_token.extend([0.0] * len(final_environment))）。
        advantages_by_token.extend([0.0] * len(final_environment))

    # 逐行讲解：检查条件 assistant_token_count == 0；成立时执行下方分支。
    if assistant_token_count == 0:
        # 逐行讲解：抛出异常：ValueError('不能用没有 assistant token 的分支构造 Datum')。
        raise ValueError("不能用没有 assistant token 的分支构造 Datum")
    # 逐行讲解：检查条件 not len(full_tokens) == len(old_logprobs_by_token) == len(advantages_by_token)；成立时执行下方分支。
    if not (
        len(full_tokens)
        == len(old_logprobs_by_token)
        == len(advantages_by_token)
    ):
        # 逐行讲解：抛出异常：ValueError('分支的 token、logprob 和 advantage 长度不一致')。
        raise ValueError("分支的 token、logprob 和 advantage 长度不一致")
    # 逐行讲解：检查条件 len(full_tokens) > max_sequence_tokens；成立时执行下方分支。
    if len(full_tokens) > max_sequence_tokens:
        # 逐行讲解：抛出异常：ValueError(f'分支为 {len(full_tokens)} tokens，超过上限 {max_sequence_tokens}')。
        raise ValueError(
            f"分支为 {len(full_tokens)} tokens，超过上限 {max_sequence_tokens}"
        )

    # 逐行讲解：按索引或切片读取 full_tokens[:-1]，赋给 input_tokens。
    input_tokens = full_tokens[:-1]
    # 逐行讲解：按索引或切片读取 full_tokens[1:]，赋给 target_tokens。
    target_tokens = full_tokens[1:]
    # 逐行讲解：按索引或切片读取 old_logprobs_by_token[1:]，赋给 old_logprobs。
    old_logprobs = old_logprobs_by_token[1:]
    # 逐行讲解：按索引或切片读取 advantages_by_token[1:]，赋给 advantages。
    advantages = advantages_by_token[1:]
    # 逐行讲解：检查条件 not len(input_tokens) == len(target_tokens) == len(old_logprobs) == len(advantages)；成立时执行下方分支。
    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(old_logprobs)
        == len(advantages)
    ):
        # 逐行讲解：抛出异常：ValueError('Datum input/target/logprobs/advantages 长度不一致')。
        raise ValueError("Datum input/target/logprobs/advantages 长度不一致")

    # 逐行讲解：调用 trio.Datum（trio.Datum(model_input=trio.ModelInput.from_ints(input_tokens), loss_fn_inputs={'target_tokens': np.asarray…），赋给 datum。
    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(old_logprobs, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
        },
    )
    # 逐行讲解：返回结果：TrainingDatum(datum=datum, num_tokens=len(input_tokens), loss_tokens=assistant_token_count, kind='actor', g…。
    return TrainingDatum(
        datum=datum,
        num_tokens=len(input_tokens),
        loss_tokens=assistant_token_count,
        kind="actor",
        game_id=trajectory.example.id,
        group_index=trajectory.group_index,
    )


# 逐行讲解：定义函数 build_critic_datum：把一次估值样本构造成 Datum：prompt 仅作上下文，整段 completion 训练。
def build_critic_datum(
    sample: CriticSample,
    advantage: float,
    *,
    game_id: str,
    group_index: int,
) -> TrainingDatum:
    """把一次估值样本构造成 Datum：prompt 仅作上下文，整段 completion 训练。"""
    # 逐行讲解：取值 sample.prompt_tokens，赋给 prompt_tokens。
    prompt_tokens = sample.prompt_tokens
    # 逐行讲解：取值 sample.completion_tokens，赋给 completion_tokens。
    completion_tokens = sample.completion_tokens
    # 逐行讲解：检查条件 not completion_tokens；成立时执行下方分支。
    if not completion_tokens:
        # 逐行讲解：抛出异常：ValueError('critic 样本没有 completion token')。
        raise ValueError("critic 样本没有 completion token")
    # 逐行讲解：检查条件 len(completion_tokens) != len(sample.logprobs)；成立时执行下方分支。
    if len(completion_tokens) != len(sample.logprobs):
        # 逐行讲解：抛出异常：ValueError('critic completion token 与 old logprob 长度不一致')。
        raise ValueError("critic completion token 与 old logprob 长度不一致")

    # 逐行讲解：构造列表 [*prompt_tokens, *completion_tokens]，赋给 full_tokens。
    full_tokens = [*prompt_tokens, *completion_tokens]
    # 逐行讲解：计算 [0.0] * len(prompt_tokens) + list(sample.logprobs)，赋给 old_logprobs_by_token。
    old_logprobs_by_token = [0.0] * len(prompt_tokens) + list(sample.logprobs)
    # 逐行讲解：计算 [0.0] * len(prompt_tokens) + [advantage] * len(completion_tokens)，赋给 advantages_by_token。
    advantages_by_token = [0.0] * len(prompt_tokens) + [advantage] * len(
        completion_tokens
    )

    # 逐行讲解：按索引或切片读取 full_tokens[:-1]，赋给 input_tokens。
    input_tokens = full_tokens[:-1]
    # 逐行讲解：按索引或切片读取 full_tokens[1:]，赋给 target_tokens。
    target_tokens = full_tokens[1:]
    # 逐行讲解：按索引或切片读取 old_logprobs_by_token[1:]，赋给 old_logprobs。
    old_logprobs = old_logprobs_by_token[1:]
    # 逐行讲解：按索引或切片读取 advantages_by_token[1:]，赋给 advantages。
    advantages = advantages_by_token[1:]

    # 逐行讲解：调用 trio.Datum（trio.Datum(model_input=trio.ModelInput.from_ints(input_tokens), loss_fn_inputs={'target_tokens': np.asarray…），赋给 datum。
    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(old_logprobs, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
        },
    )
    # 逐行讲解：返回结果：TrainingDatum(datum=datum, num_tokens=len(input_tokens), loss_tokens=len(completion_tokens), kind='critic',…。
    return TrainingDatum(
        datum=datum,
        num_tokens=len(input_tokens),
        loss_tokens=len(completion_tokens),
        kind="critic",
        game_id=game_id,
        group_index=group_index,
    )


# 逐行讲解：定义函数 build_actor_datums：跳过零 advantage 分支后，为起点组构造 actor Datum。
def build_actor_datums(
    group: MacroGroup,
    signals: ActorSignals,
    *,
    max_sequence_tokens: int,
) -> list[TrainingDatum]:
    """跳过零 advantage 分支后，为起点组构造 actor Datum。"""
    # 逐行讲解：声明 datums 的类型为 list[TrainingDatum]；构造列表 []。
    datums: list[TrainingDatum] = []
    # 逐行讲解：遍历 group.trajectories，将每项绑定到 trajectory。
    for trajectory in group.trajectories:
        # 逐行讲解：调用 any（any((step.completion_tokens for step in trajectory.steps))），赋给 trainable。
        trainable = any(step.completion_tokens for step in trajectory.steps)
        # 逐行讲解：检查条件 not trainable or abs(trajectory.advantage) <= ADVANTAGE_EPSILON；成立时执行下方分支。
        if not trainable or abs(trajectory.advantage) <= ADVANTAGE_EPSILON:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：向列表末尾追加一个元素（datums.append(build_segment_datum(group.start_state, trajectory, max_sequence_tokens=max_sequence_tokens))）。
        datums.append(
            build_segment_datum(
                group.start_state,
                trajectory,
                max_sequence_tokens=max_sequence_tokens,
            )
        )
    # 逐行讲解：返回结果：datums。
    return datums


# 逐行讲解：定义函数 build_critic_datums：跳过退化组与零 advantage 样本后，构造 critic Datum。
def build_critic_datums(
    samples: Sequence[CriticSample],
    signals: CriticSignals,
    *,
    game_id: str,
    group_index: int,
) -> list[TrainingDatum]:
    """跳过退化组与零 advantage 样本后，构造 critic Datum。"""
    # 逐行讲解：检查条件 signals.degenerate or not samples；成立时执行下方分支。
    if signals.degenerate or not samples:
        # 逐行讲解：返回结果：[]。
        return []
    # 逐行讲解：声明 datums 的类型为 list[TrainingDatum]；构造列表 []。
    datums: list[TrainingDatum] = []
    # 逐行讲解：遍历 zip(samples, signals.advantages, strict=True)，将每项绑定到 (sample, advantage)。
    for sample, advantage in zip(samples, signals.advantages, strict=True):
        # 逐行讲解：检查条件 abs(advantage) <= ADVANTAGE_EPSILON or not sample.completion_tokens；成立时执行下方分支。
        if abs(advantage) <= ADVANTAGE_EPSILON or not sample.completion_tokens:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：向列表末尾追加一个元素（datums.append(build_critic_datum(sample, advantage, game_id=game_id, group_index=group_index))）。
        datums.append(
            build_critic_datum(
                sample,
                advantage,
                game_id=game_id,
                group_index=group_index,
            )
        )
    # 逐行讲解：返回结果：datums。
    return datums
