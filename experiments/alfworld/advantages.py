"""根据同一游戏的终局奖励计算轨迹级 group-relative advantage。"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：从 collections 导入 defaultdict。
from collections import defaultdict
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 typing 导入 Any。
from typing import Any


# 逐行讲解：定义类 AdvantageStats：记录本次轨迹级 advantage 的分组统计。
@dataclass(frozen=True)
class AdvantageStats:
    """记录本次轨迹级 advantage 的分组统计。"""

    # 逐行讲解：声明 groups 的类型为 int。
    groups: int
    # 逐行讲解：声明 degenerate_groups 的类型为 int。
    degenerate_groups: int


# 逐行讲解：定义函数 assign_advantages：在相同游戏的 K 条轨迹内执行 reward 减均值。
def assign_advantages(trajectories: list[Any]) -> AdvantageStats:
    """在相同游戏的 K 条轨迹内执行 reward 减均值。"""
    # 逐行讲解：声明 groups 的类型为 dict[str, list[Any]]；调用 defaultdict（defaultdict(list)）。
    groups: dict[str, list[Any]] = defaultdict(list)
    # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
    for trajectory in trajectories:
        # 逐行讲解：向列表末尾追加一个元素（groups[trajectory.group_id].append(trajectory)）。
        groups[trajectory.group_id].append(trajectory)

    # 逐行讲解：取值 0，赋给 degenerate_groups。
    degenerate_groups = 0
    # 逐行讲解：遍历 groups.values()，将每项绑定到 group。
    for group in groups.values():
        # 逐行讲解：计算 sum((trajectory.reward for trajectory in group)) / len(group)，赋给 mean_reward。
        mean_reward = (
            sum(trajectory.reward for trajectory in group) / len(group)
        )
        # 逐行讲解：通过推导式逐项生成结果：[float(trajectory.reward - mean_reward) for trajectory in group]，赋给 advantages。
        advantages = [
            float(trajectory.reward - mean_reward)
            for trajectory in group
        ]
        # 逐行讲解：遍历 zip(group, advantages, strict=True)，将每项绑定到 (trajectory, advantage)。
        for trajectory, advantage in zip(group, advantages, strict=True):
            # 逐行讲解：取值 advantage，赋给 trajectory.advantage。
            trajectory.advantage = advantage
        # 逐行讲解：检查条件 all((abs(value) <= 1e-12 for value in advantages))；成立时执行下方分支。
        if all(abs(value) <= 1e-12 for value in advantages):
            # 逐行讲解：更新 degenerate_groups：degenerate_groups += 1。
            degenerate_groups += 1

    # 逐行讲解：返回结果：AdvantageStats(groups=len(groups), degenerate_groups=degenerate_groups)。
    return AdvantageStats(
        groups=len(groups),
        degenerate_groups=degenerate_groups,
    )
