"""计算 AgentOPSD 的轨迹级优势和轮级信用。

本模块不访问远程服务，只使用 NumPy/Python 对终局奖励以及已经对齐的
Teacher/Student 对数概率证据进行确定性计算。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：从 collections 导入 defaultdict。
from collections import defaultdict
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 math 导入 exp, log。
from math import exp, log
# 逐行讲解：从 typing 导入 Any, Sequence。
from typing import Any, Sequence

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np


# 逐行讲解：定义类 AgentOPSDConfig：Paper-aligned hyperparameters for turn advantage reshaping.
@dataclass(frozen=True)
class AgentOPSDConfig:
    """Paper-aligned hyperparameters for turn advantage reshaping."""

    # 逐行讲解：声明 gamma 的类型为 float；取值 0.95。
    gamma: float = 0.95
    # 逐行讲解：声明 reshape_lambda 的类型为 float；取值 0.5。
    reshape_lambda: float = 0.5
    # 逐行讲解：声明 weight_bound 的类型为 float；取值 0.2。
    weight_bound: float = 0.2
    # 逐行讲解：声明 epsilon 的类型为 float；取值 0.0001。
    epsilon: float = 1e-4


# 逐行讲解：定义类 TurnCredit：Every intermediate value used to reshape one turn's advantage.
@dataclass(frozen=True)
class TurnCredit:
    """Every intermediate value used to reshape one turn's advantage."""

    # 逐行讲解：声明 turn_index 的类型为 int。
    turn_index: int
    # 逐行讲解：声明 evidence 的类型为 float。
    evidence: float
    # 逐行讲解：声明 cumulative_evidence 的类型为 float。
    cumulative_evidence: float
    # 逐行讲解：声明 belief_before 的类型为 float。
    belief_before: float
    # 逐行讲解：声明 belief_after 的类型为 float。
    belief_after: float
    # 逐行讲解：声明 delta_belief 的类型为 float。
    delta_belief: float
    # 逐行讲解：声明 outcome_credit 的类型为 float。
    outcome_credit: float
    # 逐行讲解：声明 normalized_credit 的类型为 float。
    normalized_credit: float
    # 逐行讲解：声明 weight 的类型为 float。
    weight: float
    # 逐行讲解：声明 advantage 的类型为 float。
    advantage: float


# 逐行讲解：定义类 TrajectoryCredit：Sequence-level GRPO advantage plus all per-turn credits.
@dataclass(frozen=True)
class TrajectoryCredit:
    """Sequence-level GRPO advantage plus all per-turn credits."""

    # 逐行讲解：声明 sequence_advantage 的类型为 float。
    sequence_advantage: float
    # 逐行讲解：声明 initial_belief 的类型为 float。
    initial_belief: float
    # 逐行讲解：声明 turns 的类型为 tuple[TurnCredit, ...]。
    turns: tuple[TurnCredit, ...]


# 逐行讲解：定义类 AdvantageStats：Group-level diagnostics returned after assigning all trajectories.
@dataclass(frozen=True)
class AdvantageStats:
    """Group-level diagnostics returned after assigning all trajectories."""

    # 逐行讲解：声明 groups 的类型为 int。
    groups: int
    # 逐行讲解：声明 degenerate_groups 的类型为 int。
    degenerate_groups: int


# 逐行讲解：定义函数 group_sequence_advantages：计算组内 GRPO advantage 和初始 belief。
def group_sequence_advantages(
    rewards: Sequence[float],
    *,
    epsilon: float,
) -> tuple[list[float], float]:
    """计算组内 GRPO advantage 和初始 belief。"""

    # 逐行讲解：计算 sum(rewards) / len(rewards)，赋给 mean_reward。
    mean_reward = sum(rewards) / len(rewards)
    # 逐行讲解：转换为浮点表示（float(np.std(rewards, ddof=0))），赋给 std_reward。
    std_reward = float(np.std(rewards, ddof=0))
    # 逐行讲解：通过推导式逐项生成结果：[(reward - mean_reward) / (std_reward + epsilon) for reward in rewards]，赋给 advantages。
    advantages = [
        (reward - mean_reward) / (std_reward + epsilon)
        for reward in rewards
    ]
    # 逐行讲解：返回结果：(advantages, mean_reward)。
    return advantages, mean_reward


# 逐行讲解：定义函数 _sigmoid：计算 sigmoid。
def _sigmoid(value: float) -> float:
    """计算 sigmoid。"""

    # 逐行讲解：检查条件 value >= 0.0；成立时执行下方分支。
    if value >= 0.0:
        # 逐行讲解：通过指数运算将对数值转换回原尺度（exp(-value)），赋给 inverse。
        inverse = exp(-value)
        # 逐行讲解：返回结果：1.0 / (1.0 + inverse)。
        return 1.0 / (1.0 + inverse)
    # 逐行讲解：通过指数运算将对数值转换回原尺度（exp(value)），赋给 forward。
    forward = exp(value)
    # 逐行讲解：返回结果：forward / (1.0 + forward)。
    return forward / (1.0 + forward)


# 逐行讲解：定义函数 reshape_turn_advantages：根据每轮证据重塑 advantage。
def reshape_turn_advantages(
    sequence_advantage: float,
    initial_belief: float,
    evidences: Sequence[float],
    config: AgentOPSDConfig = AgentOPSDConfig(),
) -> TrajectoryCredit:
    """根据每轮证据重塑 advantage。"""

    # 逐行讲解：调用 min（min(max(initial_belief, config.epsilon), 1.0 - config.epsilon)），赋给 b0。
    b0 = min(max(initial_belief, config.epsilon), 1.0 - config.epsilon)
    # 逐行讲解：计算自然对数（log(b0 / (1.0 - b0))），赋给 logit_b0。
    logit_b0 = log(b0 / (1.0 - b0))
    # 逐行讲解：取值 0.0，赋给 cumulative。
    cumulative = 0.0
    # 逐行讲解：取值 b0，赋给 belief_before。
    belief_before = b0
    # 逐行讲解：声明 raw 的类型为 list[tuple[float, float, float, float]]；构造列表 []。
    raw: list[tuple[float, float, float, float]] = []

    # 逐行讲解：遍历 evidences，将每项绑定到 evidence。
    for evidence in evidences:
        # 逐行讲解：计算 config.gamma * cumulative + evidence，赋给 cumulative。
        cumulative = config.gamma * cumulative + evidence
        # 逐行讲解：调用 _sigmoid（_sigmoid(logit_b0 + cumulative)），赋给 belief_after。
        belief_after = _sigmoid(logit_b0 + cumulative)
        # 逐行讲解：向列表末尾追加一个元素（raw.append((evidence, cumulative, belief_before, belief_after))）。
        raw.append((evidence, cumulative, belief_before, belief_after))
        # 逐行讲解：取值 belief_after，赋给 belief_before。
        belief_before = belief_after

    # 逐行讲解：按条件选择结果：1.0 if sequence_advantage > 0.0 else -1.0 if sequence_advantage < 0.0 else 0.0，赋给 sign。
    sign = 1.0 if sequence_advantage > 0.0 else (
        -1.0 if sequence_advantage < 0.0 else 0.0
    )
    # 逐行讲解：通过推导式逐项生成结果：[sign * (belief_after - belief_before_value) for _, _, belief_before_value, belief_after in raw]，赋给 outcome_credits。
    outcome_credits = [
        sign * (belief_after - belief_before_value)
        for _, _, belief_before_value, belief_after in raw
    ]
    # 逐行讲解：计算 sum(outcome_credits) / len(outcome_credits)，赋给 credit_mean。
    credit_mean = sum(outcome_credits) / len(outcome_credits)
    # 逐行讲解：转换为浮点表示（float(np.std(outcome_credits, ddof=0))），赋给 credit_std。
    credit_std = float(np.std(outcome_credits, ddof=0))
    # 逐行讲解：通过推导式逐项生成结果：[(credit - credit_mean) / (credit_std + config.epsilon) for credit in outcome_credits]，赋给 normalized。
    normalized = [
        (credit - credit_mean) / (credit_std + config.epsilon)
        for credit in outcome_credits
    ]

    # 逐行讲解：计算 1.0 - config.weight_bound，赋给 lower_weight。
    lower_weight = 1.0 - config.weight_bound
    # 逐行讲解：计算 1.0 + config.weight_bound，赋给 upper_weight。
    upper_weight = 1.0 + config.weight_bound
    # 逐行讲解：声明 turns 的类型为 list[TurnCredit]；构造列表 []。
    turns: list[TurnCredit] = []
    # 逐行讲解：遍历 enumerate(zip(raw, outcome_credits, normalized), start=1)，将每项绑定到 (index, ((evidence, cumulative_value, before, after), outcome_credit, normalized_credit))。
    for index, (
        (evidence, cumulative_value, before, after),
        outcome_credit,
        normalized_credit,
    ) in enumerate(zip(raw, outcome_credits, normalized), start=1):
        # 逐行讲解：调用 min（min(max(1.0 + config.weight_bound * normalized_credit, lower_weight), upper_weight)），赋给 weight。
        weight = min(
            max(1.0 + config.weight_bound * normalized_credit, lower_weight),
            upper_weight,
        )
        # 逐行讲解：计算 1.0 - config.reshape_lambda + config.reshape_lambda * weight，赋给 multiplier。
        multiplier = (
            (1.0 - config.reshape_lambda)
            + config.reshape_lambda * weight
        )
        # 逐行讲解：计算 sequence_advantage * multiplier，赋给 turn_advantage。
        turn_advantage = sequence_advantage * multiplier
        # 逐行讲解：向列表末尾追加一个元素（turns.append(TurnCredit(turn_index=index, evidence=evidence, cumulative_evidence=cumulative_value, belief_b…）。
        turns.append(
            TurnCredit(
                turn_index=index,
                evidence=evidence,
                cumulative_evidence=cumulative_value,
                belief_before=before,
                belief_after=after,
                delta_belief=after - before,
                outcome_credit=outcome_credit,
                normalized_credit=normalized_credit,
                weight=weight,
                advantage=turn_advantage,
            )
        )

    # 逐行讲解：返回结果：TrajectoryCredit(sequence_advantage=sequence_advantage, initial_belief=b0, turns=tuple(turns))。
    return TrajectoryCredit(
        sequence_advantage=sequence_advantage,
        initial_belief=b0,
        turns=tuple(turns),
    )


# 逐行讲解：定义函数 assign_agentopsd_advantages：为每组轨迹写入序列和 turn advantage。
def assign_agentopsd_advantages(
    trajectories: Sequence[Any],
    config: AgentOPSDConfig = AgentOPSDConfig(),
) -> AdvantageStats:
    """为每组轨迹写入序列和 turn advantage。"""

    # 逐行讲解：声明 groups 的类型为 dict[str, list[Any]]；调用 defaultdict（defaultdict(list)）。
    groups: dict[str, list[Any]] = defaultdict(list)
    # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
    for trajectory in trajectories:
        # 逐行讲解：向列表末尾追加一个元素（groups[str(trajectory.group_id)].append(trajectory)）。
        groups[str(trajectory.group_id)].append(trajectory)

    # 逐行讲解：取值 0，赋给 degenerate_groups。
    degenerate_groups = 0
    # 逐行讲解：遍历 groups.values()，将每项绑定到 group。
    for group in groups.values():
        # 逐行讲解：调用 group_sequence_advantages（group_sequence_advantages([trajectory.reward for trajectory in group], epsilon=config.epsilon)），赋给 (sequence_advantages, initial_belief)。
        sequence_advantages, initial_belief = group_sequence_advantages(
            [trajectory.reward for trajectory in group],
            epsilon=config.epsilon,
        )
        # 逐行讲解：检查条件 all((abs(value) <= config.epsilon for value in sequence_advantages))；成立时执行下方分支。
        if all(abs(value) <= config.epsilon for value in sequence_advantages):
            # 逐行讲解：更新 degenerate_groups：degenerate_groups += 1。
            degenerate_groups += 1

        # 逐行讲解：遍历 zip(group, sequence_advantages)，将每项绑定到 (trajectory, sequence_advantage)。
        for trajectory, sequence_advantage in zip(group, sequence_advantages):
            # 逐行讲解：调用 reshape_turn_advantages（reshape_turn_advantages(sequence_advantage, initial_belief, trajectory.turn_evidences, config)），赋给 credit。
            credit = reshape_turn_advantages(
                sequence_advantage,
                initial_belief,
                trajectory.turn_evidences,
                config,
            )
            # 逐行讲解：取值 sequence_advantage，赋给 trajectory.advantage。
            trajectory.advantage = sequence_advantage
            # 逐行讲解：取值 sequence_advantage，赋给 trajectory.sequence_advantage。
            trajectory.sequence_advantage = sequence_advantage
            # 逐行讲解：取值 credit.initial_belief，赋给 trajectory.initial_belief。
            trajectory.initial_belief = credit.initial_belief
            # 逐行讲解：调用 list（list(credit.turns)），赋给 trajectory.turn_credits。
            trajectory.turn_credits = list(credit.turns)
            # 逐行讲解：遍历 zip(trajectory.steps, credit.turns)，将每项绑定到 (step, turn)。
            for step, turn in zip(trajectory.steps, credit.turns):
                # 逐行讲解：取值 turn.evidence，赋给 step.teacher_evidence。
                step.teacher_evidence = turn.evidence
                # 逐行讲解：取值 turn.advantage，赋给 step.turn_advantage。
                step.turn_advantage = turn.advantage

    # 逐行讲解：返回结果：AdvantageStats(groups=len(groups), degenerate_groups=degenerate_groups)。
    return AdvantageStats(
        groups=len(groups),
        degenerate_groups=degenerate_groups,
    )
