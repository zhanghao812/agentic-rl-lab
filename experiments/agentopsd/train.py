"""使用 PyTRIO 内置 PPO 在纯文本 ALFWorld 上训练 AgentOPSD。

下载 ALFWorld 数据后，在当前目录运行：

uv run --extra alfworld python train.py \
    --max-steps 10 \
    --tasks-per-update 8 \
    --group-size 8 \
    --reshape-lambda 0.5 \
    --save-every 5 \
    --swanlab-mode disabled

同一个训练 step 中的 Student rollout、特权 Teacher 打分和旧策略对数概率始终共享
同一个冻结的 SamplingClient snapshot。Student rollout 永远不会看到技能。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：importlib.metadata。
import importlib.metadata
# 逐行讲解：导入模块：os。
import os
# 逐行讲解：从 collections 导入 defaultdict。
from collections import defaultdict
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 time 导入 perf_counter。
from time import perf_counter
# 逐行讲解：从 typing 导入 Any, Iterable, Sequence。
from typing import Any, Iterable, Sequence

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm

# 逐行讲解：从 advantages 导入 AgentOPSDConfig, AdvantageStats, assign_agentopsd_advantages。
from advantages import AgentOPSDConfig, AdvantageStats, assign_agentopsd_advantages
# 逐行讲解：从 data 导入 SPLIT_DIRECTORIES, default_data_root, shuffled_games, take_batch。
from data import SPLIT_DIRECTORIES, default_data_root, shuffled_games, take_batch
# 逐行讲解：从 loss 导入 BuiltinPPOConfig, TrainingDatum, build_training_datums。
from loss import BuiltinPPOConfig, TrainingDatum, build_training_datums
# 逐行讲解：从 rollout 导入 MAX_STUDENT_TRAJECTORY_TOKENS, RolloutBatch, RolloutConfig, Trajectory, rollout_batch。
from rollout import (
    MAX_STUDENT_TRAJECTORY_TOKENS,
    RolloutBatch,
    RolloutConfig,
    Trajectory,
    rollout_batch,
)
# 逐行讲解：从 skills 导入 DEFAULT_SKILLS_DIR, SKILL_SOURCE_COMMIT, SkillProvider。
from skills import (
    DEFAULT_SKILLS_DIR,
    SKILL_SOURCE_COMMIT,
    SkillProvider,
)
# 逐行讲解：从 teacher 导入 TeacherBatchStats, score_trajectories_async。
from teacher import TeacherBatchStats, score_trajectories_async


# 逐行讲解：调用 trio.configure（trio.configure(request_timeout_seconds=1800)）。
trio.configure(request_timeout_seconds=1800)

# 逐行讲解：取值 1，赋给 PPO_EPOCHS。
PPO_EPOCHS = 1


# 逐行讲解：定义函数 _path：封装下方逻辑，供调用方复用。
def _path(value: str | Path) -> Path:
    # 逐行讲解：返回结果：Path(os.path.expandvars(str(value))).expanduser().resolve()。
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


# 逐行讲解：定义函数 parse_args：解析 AgentOPSD 训练和 PyTRIO 运行参数。
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 AgentOPSD 训练和 PyTRIO 运行参数。"""

    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__, add_help=False)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('-h', '--help', action='help', help='显示帮助信息并退出。')）。
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="显示帮助信息并退出。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data-root', type=Path, default=default_data_root(), help='ALFWorld 数据根目录。')）。
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="ALFWorld 数据根目录。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--train-split', choices=list(SPLIT_DIRECTORIES), default='train', help='训练使用的数据划分。')）。
    parser.add_argument(
        "--train-split",
        choices=list(SPLIT_DIRECTORIES),
        default="train",
        help="训练使用的数据划分。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-train-games', type=int, default=0, help='打乱后最多使用多少个游戏；0 表示使用整个数据划分。')）。
    parser.add_argument(
        "--max-train-games",
        type=int,
        default=0,
        help="打乱后最多使用多少个游戏；0 表示使用整个数据划分。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-steps', type=int, default=200, help='训练更新次数，不控制单条轨迹的环境交互轮数。')）。
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="训练更新次数，不控制单条轨迹的环境交互轮数。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--tasks-per-update', type=int, default=16, help='每次训练更新抽取的 ALFWorld 游戏数量。')）。
    parser.add_argument(
        "--tasks-per-update",
        type=int,
        default=16,
        help="每次训练更新抽取的 ALFWorld 游戏数量。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--group-size', type=int, default=8, help='每个游戏并行采样的轨迹数量。')）。
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="每个游戏并行采样的轨迹数量。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-turns', type=int, default=50, help='单条轨迹最多与环境交互的轮数。')）。
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="单条轨迹最多与环境交互的轮数。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--include-admissible-actions', action=argparse.BooleanOptionalAction, default=True, he…）。
    parser.add_argument(
        "--include-admissible-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在 prompt 中展示环境当前允许执行的动作。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--environment-asynchronous', action=argparse.BooleanOptionalAction, default=True, help…）。
    parser.add_argument(
        "--environment-asynchronous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用 TextWorld 异步子进程环境。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-trajectory-tokens', type=int, default=MAX_STUDENT_TRAJECTORY_TOKENS, help='单条 St…）。
    parser.add_argument(
        "--max-trajectory-tokens",
        type=int,
        default=MAX_STUDENT_TRAJECTORY_TOKENS,
        help="单条 Student 完整轨迹允许的最大 token 数。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-action-tokens', type=int, default=512, help='每轮 assistant action 最多生成的 token 数。')）。
    parser.add_argument(
        "--max-action-tokens",
        type=int,
        default=512,
        help="每轮 assistant action 最多生成的 token 数。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0, help='Student rollout 的采样温度。')）。
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Student rollout 的采样温度。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0, help='Student rollout 的 nucleus sampling 概率阈值。')）。
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Student rollout 的 nucleus sampling 概率阈值。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--teacher-concurrency', type=int, default=16, help='Teacher 并发计算整条轨迹 logprob 的最大请求数。')）。
    parser.add_argument(
        "--teacher-concurrency",
        type=int,
        default=16,
        help="Teacher 并发计算整条轨迹 logprob 的最大请求数。",
    )

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--gamma', type=float, default=0.95, help='递归累计 Teacher 证据时的历史衰减系数。')）。
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="递归累计 Teacher 证据时的历史衰减系数。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--reshape-lambda', type=float, default=0.5, help='控制 Teacher 的逐轮证据对原始 GRPO advantage 的…）。
    parser.add_argument(
        "--reshape-lambda",
        type=float,
        default=0.5,
        help=(
            "控制 Teacher 的逐轮证据对原始 GRPO advantage 的影响强度；"
            "0 表示所有轮次完全使用原始轨迹 advantage，1 表示完全使用"
            "轮级重塑结果，实际调整幅度仍受 --weight-bound 限制。"
        ),
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--weight-bound', type=float, default=0.2, help='每轮重塑权重相对 1.0 的最大偏移量。')）。
    parser.add_argument(
        "--weight-bound",
        type=float,
        default=0.2,
        help="每轮重塑权重相对 1.0 的最大偏移量。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--epsilon', type=float, default=0.0001, help='优势标准化和 belief 裁剪使用的小常数。')）。
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="优势标准化和 belief 裁剪使用的小常数。",
    )

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--loss-mode', choices=['builtin_ppo'], default='builtin_ppo', help='训练损失实现，目前仅支持 PyTRI…）。
    parser.add_argument(
        "--loss-mode",
        choices=["builtin_ppo"],
        default="builtin_ppo",
        help="训练损失实现，目前仅支持 PyTRIO 内置 PPO。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--ppo-clip-low', type=float, default=0.8, help='PPO importance ratio 的下裁剪阈值。')）。
    parser.add_argument(
        "--ppo-clip-low",
        type=float,
        default=0.8,
        help="PPO importance ratio 的下裁剪阈值。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--ppo-clip-high', type=float, default=1.24, help='PPO importance ratio 的上裁剪阈值。')）。
    parser.add_argument(
        "--ppo-clip-high",
        type=float,
        default=1.24,
        help="PPO importance ratio 的上裁剪阈值。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B', help='创建 PyTRIO LoRA 训练客户端使用的基座模型。')）。
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="创建 PyTRIO LoRA 训练客户端使用的基座模型。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=32, help='LoRA 的秩。')）。
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA 的秩。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--train-unembed', action=argparse.BooleanOptionalAction, default=False, help='是否训练输出词表…）。
    parser.add_argument(
        "--train-unembed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否训练输出词表映射层。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=4e-06, help='Adam 优化器学习率。')）。
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-6,
        help="Adam 优化器学习率。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta1', type=float, default=0.9, help='Adam 优化器的一阶动量系数。')）。
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam 优化器的一阶动量系数。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta2', type=float, default=0.95, help='Adam 优化器的二阶动量系数。')）。
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="Adam 优化器的二阶动量系数。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--weight-decay', type=float, default=0.0, help='Adam 优化器的权重衰减系数。')）。
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam 优化器的权重衰减系数。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--grad-clip-norm', type=float, default=1.0, help='梯度范数裁剪上限。')）。
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="梯度范数裁剪上限。",
    )

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--skills-dir', type=Path, default=DEFAULT_SKILLS_DIR, help='Teacher 使用的固定 ALFWorld Ski…）。
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help="Teacher 使用的固定 ALFWorld SkillBank 目录。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every', type=int, default=25, help='每隔多少个训练 step 保存状态和采样权重；0 表示只保存最终结果。')）。
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="每隔多少个训练 step 保存状态和采样权重；0 表示只保存最终结果。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42, help='数据打乱、环境和模型采样使用的随机种子。')）。
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="数据打乱、环境和模型采样使用的随机种子。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--run-name', default='agentopsd-alfworld-qwen35-4b', help='checkpoint 和 SwanLab 实验使用的运…）。
    parser.add_argument(
        "--run-name",
        default="agentopsd-alfworld-qwen35-4b",
        help="checkpoint 和 SwanLab 实验使用的运行名称。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-agentopsd', help='记录实验使用的 SwanLab 项目名称。')）。
    parser.add_argument(
        "--swanlab-project",
        default="agentic-rl-lab-agentopsd",
        help="记录实验使用的 SwanLab 项目名称。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=['online', 'local', 'offline', 'disabled'], default='online',…）。
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="online",
        help="SwanLab 记录模式。",
    )
    # 逐行讲解：解析实际命令行参数（parser.parse_args(argv)），赋给 args。
    args = parser.parse_args(argv)
    # 逐行讲解：调用 _path（_path(args.data_root)），赋给 args.data_root。
    args.data_root = _path(args.data_root)
    # 逐行讲解：调用 _path（_path(args.skills_dir)），赋给 args.skills_dir。
    args.skills_dir = _path(args.skills_dir)
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 mean：封装下方逻辑，供调用方复用。
def mean(values: Iterable[float]) -> float:
    # 逐行讲解：调用 list（list(values)），赋给 items。
    items = list(values)
    # 逐行讲解：返回结果：sum(items) / len(items)。
    return sum(items) / len(items)


# 逐行讲解：定义函数 population_std：封装下方逻辑，供调用方复用。
def population_std(values: Iterable[float]) -> float:
    # 逐行讲解：调用 list（list(values)），赋给 items。
    items = list(values)
    # 逐行讲解：返回结果：float(np.std(np.asarray(items, dtype=np.float64), ddof=0))。
    return float(np.std(np.asarray(items, dtype=np.float64), ddof=0))


# 逐行讲解：定义函数 logical_step_count：封装下方逻辑，供调用方复用。
def logical_step_count(trajectories: Sequence[Trajectory]) -> int:
    # 逐行讲解：返回结果：sum((bool(step.completion_tokens) for trajectory in trajectories for step in trajectory.steps))。
    return sum(
        bool(step.completion_tokens)
        for trajectory in trajectories
        for step in trajectory.steps
    )


# 逐行讲解：定义函数 rollout_metrics：封装下方逻辑，供调用方复用。
def rollout_metrics(
    batch: RolloutBatch,
    advantage_stats: AdvantageStats,
    teacher_stats: TeacherBatchStats,
    datums: Sequence[TrainingDatum],
) -> dict[str, float]:
    # 逐行讲解：取值 batch.trajectories，赋给 trajectories。
    trajectories = batch.trajectories
    # 逐行讲解：通过推导式逐项生成结果：[step for trajectory in trajectories for step in trajectory.steps]，赋给 steps。
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    # 逐行讲解：通过推导式逐项生成结果：[credit for trajectory in trajectories for credit in trajectory.turn_credits]，赋给 credits。
    credits = [credit for trajectory in trajectories for credit in trajectory.turn_credits]
    # 逐行讲解：通过推导式逐项生成结果：[float(trajectory.reward) for trajectory in trajectories]，赋给 rewards。
    rewards = [float(trajectory.reward) for trajectory in trajectories]
    # 逐行讲解：通过推导式逐项生成结果：[float(trajectory.sequence_advantage) for trajectory in trajectories]，赋给 sequence_advantages。
    sequence_advantages = [
        float(trajectory.sequence_advantage) for trajectory in trajectories
    ]
    # 逐行讲解：通过推导式逐项生成结果：[float(sum((len(step.completion_tokens) for step in trajectory.steps))) for trajectory in trajectories]，赋给 action_tokens_by_trajectory。
    action_tokens_by_trajectory = [
        float(sum(len(step.completion_tokens) for step in trajectory.steps))
        for trajectory in trajectories
    ]
    # 逐行讲解：声明 grouped_rewards 的类型为 dict[str, list[float]]；调用 defaultdict（defaultdict(list)）。
    grouped_rewards: dict[str, list[float]] = defaultdict(list)
    # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
    for trajectory in trajectories:
        # 逐行讲解：向列表末尾追加一个元素（grouped_rewards[trajectory.group_id].append(float(trajectory.reward))）。
        grouped_rewards[trajectory.group_id].append(float(trajectory.reward))
    # 逐行讲解：求和（sum((all((value == 1.0 for value in values)) for values in grouped_rewards.values()))），赋给 all_success。
    all_success = sum(all(value == 1.0 for value in values) for values in grouped_rewards.values())
    # 逐行讲解：求和（sum((all((value == 0.0 for value in values)) for values in grouped_rewards.values()))），赋给 all_failure。
    all_failure = sum(all(value == 0.0 for value in values) for values in grouped_rewards.values())
    # 逐行讲解：调用 len（len(grouped_rewards)），赋给 group_count。
    group_count = len(grouped_rewards)
    # 逐行讲解：通过推导式逐项生成结果：[credit.advantage / trajectory.sequence_advantage for trajectory in trajectories if abs(float(trajectory.se…，赋给 multipliers。
    multipliers = [
        credit.advantage / trajectory.sequence_advantage
        for trajectory in trajectories
        if abs(float(trajectory.sequence_advantage)) > 1e-12
        for credit in trajectory.turn_credits
    ]
    # 逐行讲解：构造字典 {'reward/mean': mean(rewards), 'reward/std': population_std(rewards), 'reward/success_rate': mean((float(tr…，赋给 metrics。
    metrics = {
        "reward/mean": mean(rewards),
        "reward/std": population_std(rewards),
        "reward/success_rate": mean(float(trajectory.won) for trajectory in trajectories),
        "reward/all_success_group_rate": all_success / group_count,
        "reward/all_failure_group_rate": all_failure / group_count,
        "trajectory/turns_mean": mean(float(len(item.steps)) for item in trajectories),
        "trajectory/action_tokens_mean": mean(action_tokens_by_trajectory),
        "trajectory/truncated_rate": mean(
            float(item.truncated) for item in trajectories
        ),
        "trajectory/valid_tool_call_rate": mean(
            float(step.valid_format) for step in steps
        ),
        "trajectory/admissible_action_rate": mean(
            float(step.admissible) for step in steps
        ),
        "trajectory/valid_action_rate": mean(
            float(step.valid_action) for step in steps
        ),
        "trajectory/invalid_actions_mean": mean(
            float(item.invalid_action_count) for item in trajectories
        ),
        "advantage/sequence_std": population_std(sequence_advantages),
        "advantage/groups": float(advantage_stats.groups),
        "advantage/degenerate_group_rate": (
            advantage_stats.degenerate_groups / advantage_stats.groups
        ),
        "opsd/evidence_mean": mean(credit.evidence for credit in credits),
        "opsd/evidence_abs_mean": mean(abs(credit.evidence) for credit in credits),
        "opsd/delta_b_mean": mean(credit.delta_belief for credit in credits),
        "opsd/delta_b_abs_mean": mean(abs(credit.delta_belief) for credit in credits),
        "opsd/weight_mean": mean(credit.weight for credit in credits),
        "opsd/weight_min": min(credit.weight for credit in credits),
        "opsd/weight_max": max(credit.weight for credit in credits),
        "opsd/adv_multiplier_min": min(multipliers, default=1.0),
        "opsd/adv_multiplier_max": max(multipliers, default=1.0),
        "requests/teacher_count": float(teacher_stats.requests),
        "requests/reference_count": 0.0,
        "tokens/teacher_scored": float(teacher_stats.teacher_input_tokens),
        "tokens/teacher_action": float(teacher_stats.action_tokens),
        "train/logical_steps": float(logical_step_count(trajectories)),
        "train/nonzero_trajectory_datums": float(len(datums)),
        "train/batch_size": float(len(datums)),
        "train/tokens": float(sum(item.num_tokens for item in datums)),
        "train/action_tokens": float(sum(item.action_tokens for item in datums)),
        "train/max_sequence_tokens": float(
            max((item.num_tokens for item in datums), default=0)
        ),
    }
    # 逐行讲解：声明 by_task 的类型为 dict[str, list[float]]；调用 defaultdict（defaultdict(list)）。
    by_task: dict[str, list[float]] = defaultdict(list)
    # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
    for trajectory in trajectories:
        # 逐行讲解：向列表末尾追加一个元素（by_task[trajectory.example.task_type].append(float(trajectory.won))）。
        by_task[trajectory.example.task_type].append(float(trajectory.won))
    # 逐行讲解：遍历 sorted(by_task.items())，将每项绑定到 (task_type, values)。
    for task_type, values in sorted(by_task.items()):
        # 逐行讲解：计算均值（mean(values)），赋给 metrics[f'reward/success_rate/{task_type}']。
        metrics[f"reward/success_rate/{task_type}"] = mean(values)
    # 逐行讲解：返回结果：metrics。
    return metrics


# 逐行讲解：定义函数 merge_result_metrics：封装下方逻辑，供调用方复用。
def merge_result_metrics(result: Any, prefix: str) -> dict[str, float]:
    # 逐行讲解：返回结果：{f'{prefix}/{key}': float(value) for key, value in dict(result.metrics).items()}。
    return {
        f"{prefix}/{key}": float(value)
        for key, value in dict(result.metrics).items()
    }


# 逐行讲解：定义函数 pick_mean_loss：封装下方逻辑，供调用方复用。
def pick_mean_loss(metrics: dict[str, float]) -> float | None:
    # 逐行讲解：返回结果：metrics.get('trainer/loss_mean')。
    return metrics.get("trainer/loss_mean")


# 逐行讲解：定义函数 save_checkpoint：封装下方逻辑，供调用方复用。
def save_checkpoint(training_client: Any, name: str) -> None:
    # 逐行讲解：等待任务完成并读取结果（training_client.save_state(name=f'{name}-state').result()），赋给 state。
    state = training_client.save_state(name=f"{name}-state").result()
    # 逐行讲解：等待任务完成并读取结果（training_client.save_weights_for_sampler(name=f'{name}-weights').result()），赋给 weights。
    weights = training_client.save_weights_for_sampler(
        name=f"{name}-weights"
    ).result()
    # 逐行讲解：调用 tqdm.write（tqdm.write(f'Saved state: {state.path}')）。
    tqdm.write(f"Saved state: {state.path}")
    # 逐行讲解：调用 tqdm.write（tqdm.write(f'Saved sampler weights: {weights.path}')）。
    tqdm.write(f"Saved sampler weights: {weights.path}")


# 逐行讲解：定义函数 serializable_args：封装下方逻辑，供调用方复用。
def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    # 逐行讲解：返回结果：{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}。
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


# 逐行讲解：定义函数 main：Run snapshot -> rollout -> Teacher -> credit -> PPO -> optimizer.
def main(args: argparse.Namespace) -> None:
    """Run snapshot -> rollout -> Teacher -> credit -> PPO -> optimizer."""

    # 逐行讲解：调用 shuffled_games（shuffled_games(args.data_root, args.train_split, args.seed, max_games=args.max_train_games)），赋给 games。
    games = shuffled_games(
        args.data_root,
        args.train_split,
        args.seed,
        max_games=args.max_train_games,
    )
    # 逐行讲解：调用 SkillProvider（SkillProvider(args.skills_dir)），赋给 skill_provider。
    skill_provider = SkillProvider(args.skills_dir)
    # 逐行讲解：调用 AgentOPSDConfig（AgentOPSDConfig(gamma=args.gamma, reshape_lambda=args.reshape_lambda, weight_bound=args.weight_bound, epsil…），赋给 agentopsd_config。
    agentopsd_config = AgentOPSDConfig(
        gamma=args.gamma,
        reshape_lambda=args.reshape_lambda,
        weight_bound=args.weight_bound,
        epsilon=args.epsilon,
    )
    # 逐行讲解：调用 BuiltinPPOConfig（BuiltinPPOConfig(args.ppo_clip_low, args.ppo_clip_high)），赋给 ppo_config。
    ppo_config = BuiltinPPOConfig(args.ppo_clip_low, args.ppo_clip_high)
    # 逐行讲解：调用 RolloutConfig（RolloutConfig(group_size=args.group_size, max_turns=args.max_turns, max_trajectory_tokens=args.max_trajecto…），赋给 rollout_config。
    rollout_config = RolloutConfig(
        group_size=args.group_size,
        max_turns=args.max_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_action_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        include_admissible_actions=args.include_admissible_actions,
        environment_asynchronous=args.environment_asynchronous,
    )
    # 逐行讲解：调用 trio.AdamParams（trio.AdamParams(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, weight_decay=args.wei…），赋给 adam_params。
    adam_params = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
    )

    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：调用 service_client.create_lora_training_client（service_client.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank, seed=args.seed,…），赋给 training_client。
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
        train_mlp=True,
        train_attn=True,
        train_unembed=args.train_unembed,
    )
    # 逐行讲解：调用 training_client.get_tokenizer（training_client.get_tokenizer()），赋给 tokenizer。
    tokenizer = training_client.get_tokenizer()

    # 逐行讲解：调用 swanlab.init（swanlab.init(project=args.swanlab_project, name=args.run_name, mode=args.swanlab_mode, config={**serializab…）。
    swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config={
            **serializable_args(args),
            "ppo_epochs": PPO_EPOCHS,
            "pytrio_version": importlib.metadata.version("pytrio"),
            "skill_source_commit": SKILL_SOURCE_COMMIT,
        },
        tags=["pytrio", "alfworld", "agentopsd", "builtin-ppo"],
        job_type="train",
    )
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：进入上下文管理器 tqdm(total=args.max_steps, desc='AgentOPSD training', unit='step')，退出时执行其收尾逻辑。
        with tqdm(
            total=args.max_steps,
            desc="AgentOPSD training",
            unit="step",
        ) as progress:
            # 逐行讲解：遍历 range(args.max_steps)，将每项绑定到 step_index。
            for step_index in range(args.max_steps):
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 step_started。
                step_started = perf_counter()
                # 逐行讲解：调用 take_batch（take_batch(games, step_index * args.tasks_per_update, args.tasks_per_update)），赋给 games_batch。
                games_batch = take_batch(
                    games,
                    step_index * args.tasks_per_update,
                    args.tasks_per_update,
                )

                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='snapshot', refresh=True)）。
                progress.set_postfix(phase="snapshot", refresh=True)
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 phase_started。
                phase_started = perf_counter()
                # 逐行讲解：保存当前权重并创建对应采样客户端（training_client.save_weights_and_get_sampling_client()），赋给 sampling_client。
                sampling_client = (
                    training_client.save_weights_and_get_sampling_client()
                )
                # 逐行讲解：计算 perf_counter() - phase_started，赋给 snapshot_seconds。
                snapshot_seconds = perf_counter() - phase_started
                # 逐行讲解：调用 str（str(sampling_client.task_id)），赋给 snapshot_id。
                snapshot_id = str(sampling_client.task_id)

                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='student-rollout', refresh=True)）。
                progress.set_postfix(phase="student-rollout", refresh=True)
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 phase_started。
                phase_started = perf_counter()
                # 逐行讲解：进入上下文管理器 tqdm(total=len(games_batch) * args.group_size, desc=f'Step {step_index + 1} rollout', unit='trajectory', po…，退出时执行其收尾逻辑。
                with tqdm(
                    total=len(games_batch) * args.group_size,
                    desc=f"Step {step_index + 1} rollout",
                    unit="trajectory",
                    position=1,
                    leave=False,
                ) as rollout_progress:
                    # 逐行讲解：调用 rollout_batch（rollout_batch(sampling_client, tokenizer, games_batch, rollout_config, policy_snapshot_id=snapshot_id, prog…），赋给 batch。
                    batch = rollout_batch(
                        sampling_client,
                        tokenizer,
                        games_batch,
                        rollout_config,
                        policy_snapshot_id=snapshot_id,
                        progress_callback=rollout_progress.update,
                    )
                # 逐行讲解：计算 perf_counter() - phase_started，赋给 rollout_seconds。
                rollout_seconds = perf_counter() - phase_started

                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='teacher', refresh=True)）。
                progress.set_postfix(phase="teacher", refresh=True)
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 phase_started。
                phase_started = perf_counter()
                # 逐行讲解：调用 asyncio.run（asyncio.run(score_trajectories_async(sampling_client, tokenizer, batch.trajectories, skill_provider, max_co…），赋给 (_, teacher_stats)。
                _, teacher_stats = asyncio.run(
                    score_trajectories_async(
                        sampling_client,
                        tokenizer,
                        batch.trajectories,
                        skill_provider,
                        max_concurrency=args.teacher_concurrency,
                    )
                )
                # 逐行讲解：计算 perf_counter() - phase_started，赋给 teacher_seconds。
                teacher_seconds = perf_counter() - phase_started

                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='credit', refresh=True)）。
                progress.set_postfix(phase="credit", refresh=True)
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 phase_started。
                phase_started = perf_counter()
                # 逐行讲解：调用 assign_agentopsd_advantages（assign_agentopsd_advantages(batch.trajectories, agentopsd_config)），赋给 advantage_stats。
                advantage_stats = assign_agentopsd_advantages(
                    batch.trajectories,
                    agentopsd_config,
                )
                # 逐行讲解：调用 build_training_datums（build_training_datums(batch.trajectories)），赋给 datums。
                datums = build_training_datums(batch.trajectories)
                # 逐行讲解：计算 perf_counter() - phase_started，赋给 credit_seconds。
                credit_seconds = perf_counter() - phase_started

                # 逐行讲解：取值 None，赋给 backward_result。
                backward_result = None
                # 逐行讲解：取值 None，赋给 optimizer_result。
                optimizer_result = None
                # 逐行讲解：取值 0.0，赋给 backward_seconds。
                backward_seconds = 0.0
                # 逐行讲解：取值 0.0，赋给 optimizer_seconds。
                optimizer_seconds = 0.0
                # 逐行讲解：检查条件 datums；成立时执行下方分支。
                if datums:
                    # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='backward', refresh=True)）。
                    progress.set_postfix(phase="backward", refresh=True)
                    # 逐行讲解：调用 perf_counter（perf_counter()），赋给 phase_started。
                    phase_started = perf_counter()
                    # 逐行讲解：调用 training_client.forward_backward([item.datum for item in datums], loss_fn='ppo', loss_fn_config=ppo_config.…（training_client.forward_backward([item.datum for item in datums], loss_fn='ppo', loss_fn_config=ppo_config.…），赋给 backward_result。
                    backward_result = training_client.forward_backward(
                        [item.datum for item in datums],
                        loss_fn="ppo",
                        loss_fn_config=ppo_config.as_loss_fn_config(),
                    ).result()
                    # 逐行讲解：计算 perf_counter() - phase_started，赋给 backward_seconds。
                    backward_seconds = perf_counter() - phase_started

                    # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='optimizer', refresh=True)）。
                    progress.set_postfix(phase="optimizer", refresh=True)
                    # 逐行讲解：调用 perf_counter（perf_counter()），赋给 phase_started。
                    phase_started = perf_counter()
                    # 逐行讲解：等待任务完成并读取结果（training_client.optim_step(adam_params).result()），赋给 optimizer_result。
                    optimizer_result = training_client.optim_step(
                        adam_params
                    ).result()
                    # 逐行讲解：计算 perf_counter() - phase_started，赋给 optimizer_seconds。
                    optimizer_seconds = perf_counter() - phase_started

                # 逐行讲解：取值 0.0，赋给 checkpoint_seconds。
                checkpoint_seconds = 0.0
                # 逐行讲解：检查条件 args.save_every > 0 and (step_index + 1) % args.save_every == 0；成立时执行下方分支。
                if (
                    args.save_every > 0
                    and (step_index + 1) % args.save_every == 0
                ):
                    # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='checkpoint', refresh=True)）。
                    progress.set_postfix(phase="checkpoint", refresh=True)
                    # 逐行讲解：调用 perf_counter（perf_counter()），赋给 phase_started。
                    phase_started = perf_counter()
                    # 逐行讲解：调用 save_checkpoint（save_checkpoint(training_client, f'{args.run_name}-step-{step_index + 1}')）。
                    save_checkpoint(
                        training_client,
                        f"{args.run_name}-step-{step_index + 1}",
                    )
                    # 逐行讲解：计算 perf_counter() - phase_started，赋给 checkpoint_seconds。
                    checkpoint_seconds = perf_counter() - phase_started

                # 逐行讲解：调用 rollout_metrics（rollout_metrics(batch, advantage_stats, teacher_stats, datums)），赋给 metrics。
                metrics = rollout_metrics(
                    batch,
                    advantage_stats,
                    teacher_stats,
                    datums,
                )
                # 逐行讲解：检查条件 backward_result is not None；成立时执行下方分支。
                if backward_result is not None:
                    # 逐行讲解：调用 metrics.update（metrics.update(merge_result_metrics(backward_result, 'trainer'))）。
                    metrics.update(
                        merge_result_metrics(backward_result, "trainer")
                    )
                # 逐行讲解：检查条件 optimizer_result is not None；成立时执行下方分支。
                if optimizer_result is not None:
                    # 逐行讲解：调用 metrics.update（metrics.update(merge_result_metrics(optimizer_result, 'optimizer'))）。
                    metrics.update(
                        merge_result_metrics(optimizer_result, "optimizer")
                    )
                # 逐行讲解：调用 metrics.update（metrics.update({'train/step_skipped': float(not datums), 'time/snapshot_seconds': snapshot_seconds, 'time/r…）。
                metrics.update(
                    {
                        "train/step_skipped": float(not datums),
                        "time/snapshot_seconds": snapshot_seconds,
                        "time/rollout_seconds": rollout_seconds,
                        "time/teacher_seconds": teacher_seconds,
                        "time/reference_seconds": 0.0,
                        "time/credit_seconds": credit_seconds,
                        "time/forward_backward_seconds": backward_seconds,
                        "time/optimizer_seconds": optimizer_seconds,
                        "time/checkpoint_seconds": checkpoint_seconds,
                        "time/step_seconds": perf_counter() - step_started,
                    }
                )
                # 逐行讲解：计算自然对数（swanlab.log(metrics, step=step_index)）。
                swanlab.log(metrics, step=step_index)

                # 逐行讲解：调用 pick_mean_loss（pick_mean_loss(metrics)），赋给 loss。
                loss = pick_mean_loss(metrics)
                # 逐行讲解：按条件选择结果：'skipped' if not datums else f'{loss:.4f}' if loss is not None else 'missing'，赋给 loss_text。
                loss_text = "skipped" if not datums else (
                    f"{loss:.4f}" if loss is not None else "missing"
                )
                # 逐行讲解：调用 progress.update（progress.update(1)）。
                progress.update(1)
                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(success=f"{metrics['reward/success_rate']:.3f}", loss=loss_text, step_s=f"{metrics['ti…）。
                progress.set_postfix(
                    success=f"{metrics['reward/success_rate']:.3f}",
                    loss=loss_text,
                    step_s=f"{metrics['time/step_seconds']:.1f}",
                    refresh=True,
                )
                # 逐行讲解：调用 tqdm.write（tqdm.write(f"Step {step_index + 1}/{args.max_steps} success={metrics['reward/success_rate']:.3f} datums={le…）。
                tqdm.write(
                    f"Step {step_index + 1}/{args.max_steps} "
                    f"success={metrics['reward/success_rate']:.3f} "
                    f"datums={len(datums)} loss={loss_text}"
                )

        # 逐行讲解：调用 save_checkpoint（save_checkpoint(training_client, f'{args.run_name}-final')）。
        save_checkpoint(training_client, f"{args.run_name}-final")
    except KeyboardInterrupt:
        # 逐行讲解：调用 swanlab.finish（swanlab.finish(state='aborted')）。
        swanlab.finish(state="aborted")
        # 逐行讲解：抛出异常：重新抛出当前异常。
        raise
    except Exception as error:
        # 逐行讲解：调用 swanlab.finish（swanlab.finish(state='crashed', error=f'{type(error).__name__}: {error}')）。
        swanlab.finish(state="crashed", error=f"{type(error).__name__}: {error}")
        # 逐行讲解：抛出异常：重新抛出当前异常。
        raise
    else:
        # 逐行讲解：调用 swanlab.finish（swanlab.finish()）。
        swanlab.finish()


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main(parse_args())）。
    main(parse_args())
