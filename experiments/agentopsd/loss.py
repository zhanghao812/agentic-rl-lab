"""为 AgentOPSD 构造只训练 action token 的 PyTRIO 内置 PPO Datum。"""

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

# 逐行讲解：从 rollout 导入 Trajectory。
from rollout import Trajectory


# 逐行讲解：定义类 TrainingDatum：保存 PPO 数据和 token 统计信息。
@dataclass(frozen=True)
class TrainingDatum:
    """保存 PPO 数据和 token 统计信息。"""

    # 逐行讲解：声明 datum 的类型为 trio.Datum。
    datum: trio.Datum
    # 逐行讲解：声明 num_tokens 的类型为 int。
    num_tokens: int
    # 逐行讲解：声明 action_tokens 的类型为 int。
    action_tokens: int
    # 逐行讲解：声明 game_id 的类型为 str。
    game_id: str
    # 逐行讲解：声明 group_index 的类型为 int。
    group_index: int
    # 逐行讲解：声明 policy_snapshot_id 的类型为 str。
    policy_snapshot_id: str


# 逐行讲解：定义类 BuiltinPPOConfig：保存 PyTRIO 内置 PPO 的裁剪参数。
@dataclass(frozen=True)
class BuiltinPPOConfig:
    """保存 PyTRIO 内置 PPO 的裁剪参数。"""

    # 逐行讲解：声明 clip_low 的类型为 float；取值 0.8。
    clip_low: float = 0.8
    # 逐行讲解：声明 clip_high 的类型为 float；取值 1.24。
    clip_high: float = 1.24

    # 逐行讲解：定义函数 as_loss_fn_config：转换为 PyTRIO PPO 配置。
    def as_loss_fn_config(self) -> dict[str, float]:
        """转换为 PyTRIO PPO 配置。"""

        # 逐行讲解：返回结果：{'clip_low_threshold': self.clip_low, 'clip_high_threshold': self.clip_high}。
        return {
            "clip_low_threshold": self.clip_low,
            "clip_high_threshold": self.clip_high,
        }


# 逐行讲解：定义函数 build_agentopsd_ppo_datum：构造只训练 action token 的 PPO 数据。
def build_agentopsd_ppo_datum(trajectory: Trajectory) -> TrainingDatum:
    """构造只训练 action token 的 PPO 数据。"""

    # 逐行讲解：取值 trajectory.full_student_tokens，赋给 full_tokens。
    full_tokens = trajectory.full_student_tokens
    # 逐行讲解：计算 [0.0] * len(full_tokens)，赋给 old_logprobs_by_token。
    old_logprobs_by_token = [0.0] * len(full_tokens)
    # 逐行讲解：计算 [0.0] * len(full_tokens)，赋给 advantages_by_token。
    advantages_by_token = [0.0] * len(full_tokens)
    # 逐行讲解：取值 0，赋给 action_tokens。
    action_tokens = 0

    # 逐行讲解：遍历 zip(trajectory.steps, trajectory.turn_credits)，将每项绑定到 (step, credit)。
    for step, credit in zip(trajectory.steps, trajectory.turn_credits):
        # 逐行讲解：遍历 enumerate(step.logprobs)，将每项绑定到 (offset, old_logprob)。
        for offset, old_logprob in enumerate(step.logprobs):
            # 逐行讲解：计算 step.action_start + offset，赋给 position。
            position = step.action_start + offset
            # 逐行讲解：取值 old_logprob，赋给 old_logprobs_by_token[position]。
            old_logprobs_by_token[position] = old_logprob
            # 逐行讲解：取值 credit.advantage，赋给 advantages_by_token[position]。
            advantages_by_token[position] = credit.advantage
        # 逐行讲解：更新 action_tokens：action_tokens += len(step.logprobs)。
        action_tokens += len(step.logprobs)

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
    # 逐行讲解：返回结果：TrainingDatum(datum=datum, num_tokens=len(input_tokens), action_tokens=action_tokens, game_id=trajectory.ex…。
    return TrainingDatum(
        datum=datum,
        num_tokens=len(input_tokens),
        action_tokens=action_tokens,
        game_id=trajectory.example.id,
        group_index=trajectory.group_index,
        policy_snapshot_id=trajectory.policy_snapshot_id,
    )


# 逐行讲解：定义函数 build_training_datums：批量构造非零 advantage 的 PPO 数据。
def build_training_datums(
    trajectories: Sequence[Trajectory],
) -> list[TrainingDatum]:
    """批量构造非零 advantage 的 PPO 数据。"""

    # 逐行讲解：返回结果：[build_agentopsd_ppo_datum(trajectory) for trajectory in trajectories if trajectory.sequence_advantage != 0…。
    return [
        build_agentopsd_ppo_datum(trajectory)
        for trajectory in trajectories
        if trajectory.sequence_advantage != 0.0
    ]
