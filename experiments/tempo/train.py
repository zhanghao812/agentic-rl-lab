"""使用 PyTRIO + TEMPO 在 ALFWorld 上做 macro-step 粒度的 actor-critic 训练。

训练分两阶段：
1. warm-up：完整轨迹 rollout，Monte Carlo return 作为 critic 的 value
   target（无需 bootstrap），只更新 critic；
2. TD 阶段：从 StateStore 取保存的 macro-step 起点续跑 H 轮，生成式
   critic 估终点价值，actor return = 段内奖励 + V̂(终点)，actor 与
   critic 的 Datum 合并成 batch 更新，非终局终点重新入库。

先在本目录下载数据（同第 8 篇）：

uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/datasets/alfworld"

冒烟测试：
uv run --extra alfworld python train.py \
    --warmup-updates 4 \
    --td-updates 4 \
    --warmup-games-per-batch 4 \
    --states-per-batch 4 \
    --branches 4 \
    --critic-samples 4 \
    --endpoint-samples 1 \
    --swanlab-mode disabled

正式训练：
uv run --extra alfworld python train.py \
    --warmup-updates 4 --td-updates 40 \
    --swanlab-mode online
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：os。
import os
# 逐行讲解：从 collections 导入 defaultdict。
from collections import defaultdict
# 逐行讲解：从 dataclasses 导入 replace。
from dataclasses import replace
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 time 导入 perf_counter。
from time import perf_counter
# 逐行讲解：从 typing 导入 Any, Sequence。
from typing import Any, Sequence

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm

# 逐行讲解：从 critic 导入 CriticRequest, critic_prompt_for_state, endpoint_value, sample_critic_values。
from critic import (
    CriticRequest,
    critic_prompt_for_state,
    endpoint_value,
    sample_critic_values,
)
# 逐行讲解：从 data 导入 SPLIT_DIRECTORIES, default_data_root, load_walkthrough, shuffled_games, take_batch。
from data import (
    SPLIT_DIRECTORIES,
    default_data_root,
    load_walkthrough,
    shuffled_games,
    take_batch,
)
# 逐行讲解：从 rollout 导入 MAX_SEQUENCE_TOKENS, MacroGroup, RolloutConfig, Trajectory, rollout_macro_steps。
from rollout import (
    MAX_SEQUENCE_TOKENS,
    MacroGroup,
    RolloutConfig,
    Trajectory,
    rollout_macro_steps,
)
# 逐行讲解：从 states 导入 MacroState, StateStore, boundary_states, endpoint_state。
from states import (
    MacroState,
    StateStore,
    boundary_states,
    endpoint_state,
)
# 逐行讲解：从 tempo 导入 ActorSignals, CriticSignals, TrainingDatum, assemble_actor_group, assemble_critic_group, build_actor_datums, build_critic_datums。
from tempo import (
    ActorSignals,
    CriticSignals,
    TrainingDatum,
    assemble_actor_group,
    assemble_critic_group,
    build_actor_datums,
    build_critic_datums,
)


# 逐行讲解：取值 128，赋给 MAX_TRAIN_BATCH_SIZE。
MAX_TRAIN_BATCH_SIZE = 128


# 逐行讲解：定义函数 _path：封装下方逻辑，供调用方复用。
def _path(value: str | Path) -> Path:
    # 逐行讲解：返回结果：Path(os.path.expandvars(str(value))).expanduser().resolve()。
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


# 逐行讲解：定义函数 parse_args：解析两阶段训练、rollout、critic 和日志参数。
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析两阶段训练、rollout、critic 和日志参数。"""
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data-root', type=Path, default=default_data_root())）。
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--train-split', choices=list(SPLIT_DIRECTORIES), default='train')）。
    parser.add_argument(
        "--train-split", choices=list(SPLIT_DIRECTORIES), default="train"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-train-games', type=int, default=0, help='最多使用多少局；0 表示使用完整训练 split')）。
    parser.add_argument(
        "--max-train-games",
        type=int,
        default=0,
        help="最多使用多少局；0 表示使用完整训练 split",
    )
    # ---- 两阶段 ----
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=4,
        help="warm-up 更新次数（完整轨迹 + MC target，只训 critic）",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--td-updates', type=int, default=40, help='TD 阶段更新次数')）。
    parser.add_argument(
        "--td-updates", type=int, default=40, help="TD 阶段更新次数"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--warmup-games-per-batch', type=int, default=4, help='每次 warm-up 更新采多少局完整轨迹')）。
    parser.add_argument(
        "--warmup-games-per-batch",
        type=int,
        default=4,
        help="每次 warm-up 更新采多少局完整轨迹",
    )
    # ---- macro-step / 采样 ----
    parser.add_argument("--states-per-batch", type=int, default=4)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--branches', type=int, default=4, help='每个起点的分支数 N（actor 的 group 大小）')）。
    parser.add_argument(
        "--branches",
        type=int,
        default=4,
        help="每个起点的分支数 N（actor 的 group 大小）",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--critic-samples', type=int, default=4, help='同一起点独立估值次数 K（critic 的 group 大小）')）。
    parser.add_argument(
        "--critic-samples",
        type=int,
        default=4,
        help="同一起点独立估值次数 K（critic 的 group 大小）",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--endpoint-samples', type=int, default=2, help='每个非终局终点采几次估值取均值')）。
    parser.add_argument(
        "--endpoint-samples",
        type=int,
        default=2,
        help="每个非终局终点采几次估值取均值",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--macro-rounds', type=int, default=10, help='macro-step 长度 H')）。
    parser.add_argument(
        "--macro-rounds",
        type=int,
        default=10,
        help="macro-step 长度 H",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-episode-steps', type=int, default=50, help='单局交互上限 T')）。
    parser.add_argument(
        "--max-episode-steps", type=int, default=50, help="单局交互上限 T"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--state-store-capacity', type=int, default=256)）。
    parser.add_argument("--state-store-capacity", type=int, default=256)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--include-admissible-actions', action=argparse.BooleanOptionalAction, default=True)）。
    parser.add_argument(
        "--include-admissible-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--environment-asynchronous', action=argparse.BooleanOptionalAction, default=True)）。
    parser.add_argument(
        "--environment-asynchronous",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-trajectory-tokens', type=int, default=MAX_SEQUENCE_TOKENS)）。
    parser.add_argument("--max-trajectory-tokens", type=int, default=MAX_SEQUENCE_TOKENS)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-assistant-tokens', type=int, default=512)）。
    parser.add_argument("--max-assistant-tokens", type=int, default=512)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--critic-max-tokens', type=int, default=512)）。
    parser.add_argument("--critic-max-tokens", type=int, default=512)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0)）。
    parser.add_argument("--temperature", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0)）。
    parser.add_argument("--top-p", type=float, default=1.0)
    # ---- 优化器 / 模型 ----
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=32)）。
    parser.add_argument("--lora-rank", type=int, default=32)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=1e-06)）。
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta1', type=float, default=0.9)）。
    parser.add_argument("--beta1", type=float, default=0.9)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta2', type=float, default=0.95)）。
    parser.add_argument("--beta2", type=float, default=0.95)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--weight-decay', type=float, default=0.0)）。
    parser.add_argument("--weight-decay", type=float, default=0.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--grad-clip-norm', type=float, default=1.0)）。
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every', type=int, default=10, help='每隔多少次 update 保存；0 关闭中间保存，最终产物始终保存')）。
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="每隔多少次 update 保存；0 关闭中间保存，最终产物始终保存",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--run-name', default='tempo-alfworld-qwen35-4b')）。
    parser.add_argument("--run-name", default="tempo-alfworld-qwen35-4b")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-tempo')）。
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-tempo")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=['online', 'local', 'offline', 'disabled'], default='online')）。
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="online",
    )
    # 逐行讲解：解析实际命令行参数（parser.parse_args(argv)），赋给 args。
    args = parser.parse_args(argv)
    # 逐行讲解：调用 _path（_path(args.data_root)），赋给 args.data_root。
    args.data_root = _path(args.data_root)
    # 逐行讲解：调用 _validate_args（_validate_args(args)）。
    _validate_args(args)
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 _validate_args：封装下方逻辑，供调用方复用。
def _validate_args(args: argparse.Namespace) -> None:
    # 逐行讲解：构造字典 {'warmup_updates': args.warmup_updates, 'td_updates': args.td_updates, 'warmup_games_per_batch': args.warmu…，赋给 positive。
    positive = {
        "warmup_updates": args.warmup_updates,
        "td_updates": args.td_updates,
        "warmup_games_per_batch": args.warmup_games_per_batch,
        "states_per_batch": args.states_per_batch,
        "branches": args.branches,
        "critic_samples": args.critic_samples,
        "endpoint_samples": args.endpoint_samples,
        "macro_rounds": args.macro_rounds,
        "max_episode_steps": args.max_episode_steps,
        "state_store_capacity": args.state_store_capacity,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "max_assistant_tokens": args.max_assistant_tokens,
        "critic_max_tokens": args.critic_max_tokens,
        "lora_rank": args.lora_rank,
    }
    # 逐行讲解：遍历 positive.items()，将每项绑定到 (name, value)。
    for name, value in positive.items():
        # 逐行讲解：检查条件 value < 1；成立时执行下方分支。
        if value < 1:
            # 逐行讲解：抛出异常：ValueError(f'{name} 必须大于等于 1')。
            raise ValueError(f"{name} 必须大于等于 1")
    # 逐行讲解：检查条件 args.max_episode_steps % args.macro_rounds != 0；成立时执行下方分支。
    if args.max_episode_steps % args.macro_rounds != 0:
        # 逐行讲解：抛出异常：ValueError('max_episode_steps 必须是 macro_rounds 的整数倍')。
        raise ValueError("max_episode_steps 必须是 macro_rounds 的整数倍")
    # 逐行讲解：检查条件 args.max_train_games < 0 or args.save_every < 0；成立时执行下方分支。
    if args.max_train_games < 0 or args.save_every < 0:
        # 逐行讲解：抛出异常：ValueError('max_train_games 和 save_every 不能为负数')。
        raise ValueError("max_train_games 和 save_every 不能为负数")
    # 逐行讲解：检查条件 args.max_trajectory_tokens > MAX_SEQUENCE_TOKENS；成立时执行下方分支。
    if args.max_trajectory_tokens > MAX_SEQUENCE_TOKENS:
        # 逐行讲解：抛出异常：ValueError(f'max_trajectory_tokens 不能超过平台单序列上限 {MAX_SEQUENCE_TOKENS}')。
        raise ValueError(
            f"max_trajectory_tokens 不能超过平台单序列上限 {MAX_SEQUENCE_TOKENS}"
        )
    # 逐行讲解：检查条件 args.states_per_batch * (args.branches + args.critic_samples) > MAX_TRAIN_BATCH_SIZE；成立时执行下方分支。
    if args.states_per_batch * (args.branches + args.critic_samples) > MAX_TRAIN_BATCH_SIZE:
        # 逐行讲解：抛出异常：ValueError(f'states_per_batch * (branches + critic_samples) 超过平台 batch 上限 {MAX_TRAIN_BATCH_SIZE}')。
        raise ValueError(
            f"states_per_batch * (branches + critic_samples) 超过平台 batch 上限 "
            f"{MAX_TRAIN_BATCH_SIZE}"
        )
    # 逐行讲解：检查条件 args.temperature < 0.0 or not 0.0 < args.top_p <= 1.0；成立时执行下方分支。
    if args.temperature < 0.0 or not 0.0 < args.top_p <= 1.0:
        # 逐行讲解：抛出异常：ValueError('temperature 不能为负数，top_p 必须位于 (0, 1]')。
        raise ValueError("temperature 不能为负数，top_p 必须位于 (0, 1]")
    # 逐行讲解：检查条件 args.learning_rate <= 0.0；成立时执行下方分支。
    if args.learning_rate <= 0.0:
        # 逐行讲解：抛出异常：ValueError('learning_rate 必须大于 0')。
        raise ValueError("learning_rate 必须大于 0")


# 逐行讲解：定义函数 mean：封装下方逻辑，供调用方复用。
def mean(values: list[float]) -> float:
    # 逐行讲解：返回结果：sum(values) / len(values) if values else 0.0。
    return sum(values) / len(values) if values else 0.0


# 逐行讲解：定义函数 population_std：封装下方逻辑，供调用方复用。
def population_std(values: list[float]) -> float:
    # 逐行讲解：返回结果：float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0。
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0


# 逐行讲解：定义函数 pearson：封装下方逻辑，供调用方复用。
def pearson(xs: list[float], ys: list[float]) -> float | None:
    # 逐行讲解：检查条件 len(xs) < 2 or population_std(xs) <= 0.0 or population_std(ys) <= 0.0；成立时执行下方分支。
    if len(xs) < 2 or population_std(xs) <= 0.0 or population_std(ys) <= 0.0:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])。
    return float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])


# 逐行讲解：定义函数 merge_trainer_metrics：把 PyTRIO 单次 batch 返回的数值指标统一加上 trainer 前缀。
def merge_trainer_metrics(results: list[Any]) -> dict[str, float]:
    """把 PyTRIO 单次 batch 返回的数值指标统一加上 trainer 前缀。"""
    # 逐行讲解：声明 values 的类型为 dict[str, list[float]]；调用 defaultdict（defaultdict(list)）。
    values: dict[str, list[float]] = defaultdict(list)
    # 逐行讲解：遍历 results，将每项绑定到 result。
    for result in results:
        # 逐行讲解：遍历 dict(result.metrics).items()，将每项绑定到 (key, value)。
        for key, value in dict(result.metrics).items():
            # 逐行讲解：检查条件 isinstance(value, (int, float, np.number))；成立时执行下方分支。
            if isinstance(value, (int, float, np.number)):
                # 逐行讲解：向列表末尾追加一个元素（values[key].append(float(value))）。
                values[key].append(float(value))
    # 逐行讲解：返回结果：{f'trainer/{key}': mean(items) for key, items in values.items()}。
    return {f"trainer/{key}": mean(items) for key, items in values.items()}


# 逐行讲解：定义函数 pick_mean_loss：封装下方逻辑，供调用方复用。
def pick_mean_loss(metrics: dict[str, float]) -> float | None:
    # 逐行讲解：遍历 ('trainer/loss_mean', 'trainer/loss/mean')，将每项绑定到 key。
    for key in ("trainer/loss_mean", "trainer/loss/mean"):
        # 逐行讲解：检查条件 key in metrics；成立时执行下方分支。
        if key in metrics:
            # 逐行讲解：返回结果：metrics[key]。
            return metrics[key]
    # 逐行讲解：返回结果：None。
    return None


# 逐行讲解：定义函数 forward_backward_chunked：按平台 batch 上限分块前向反向，梯度累积后由调用方统一 optim_step。
def forward_backward_chunked(
    training_client: Any,
    datums: list[TrainingDatum],
) -> list[Any]:
    """按平台 batch 上限分块前向反向，梯度累积后由调用方统一 optim_step。"""
    # 逐行讲解：声明 results 的类型为 list[Any]；构造列表 []。
    results: list[Any] = []
    # 逐行讲解：遍历 range(0, len(datums), MAX_TRAIN_BATCH_SIZE)，将每项绑定到 start。
    for start in range(0, len(datums), MAX_TRAIN_BATCH_SIZE):
        # 逐行讲解：按索引或切片读取 datums[start:start + MAX_TRAIN_BATCH_SIZE]，赋给 chunk。
        chunk = datums[start : start + MAX_TRAIN_BATCH_SIZE]
        # 逐行讲解：向列表末尾追加一个元素（results.append(training_client.forward_backward([item.datum for item in chunk], loss_fn='importance_samplin…）。
        results.append(
            training_client.forward_backward(
                [item.datum for item in chunk],
                loss_fn="importance_sampling",
            ).result()
        )
    # 逐行讲解：返回结果：results。
    return results


# 逐行讲解：定义函数 _critic_prompt_budget：封装下方逻辑，供调用方复用。
def _critic_prompt_budget(args: argparse.Namespace) -> int:
    # 逐行讲解：返回结果：args.max_trajectory_tokens - args.critic_max_tokens。
    return args.max_trajectory_tokens - args.critic_max_tokens


# 逐行讲解：定义函数 _trajectory_steps：封装下方逻辑，供调用方复用。
def _trajectory_steps(trajectories: Sequence[Trajectory]) -> list[Any]:
    # 逐行讲解：返回结果：[step for item in trajectories for step in item.steps]。
    return [step for item in trajectories for step in item.steps]


# 逐行讲解：定义函数 _rollout_quality_metrics：封装下方逻辑，供调用方复用。
def _rollout_quality_metrics(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    # 逐行讲解：调用 _trajectory_steps（_trajectory_steps(trajectories)），赋给 steps。
    steps = _trajectory_steps(trajectories)
    # 逐行讲解：返回结果：{'rollout/valid_tool_call_rate': mean([float(step.valid_format) for step in steps]), 'rollout/admissible_ac…。
    return {
        "rollout/valid_tool_call_rate": mean(
            [float(step.valid_format) for step in steps]
        ),
        "rollout/admissible_action_rate": mean(
            [float(step.admissible) for step in steps]
        ),
        "rollout/valid_action_rate": mean(
            [float(step.valid_action) for step in steps]
        ),
    }


# 逐行讲解：定义函数 save_checkpoint：同时保存训练 state 和 sampler weights。
def save_checkpoint(training_client: Any, name: str) -> None:
    """同时保存训练 state 和 sampler weights。"""
    # 逐行讲解：等待任务完成并读取结果（training_client.save_state(name=f'{name}-state').result()），赋给 state。
    state = training_client.save_state(name=f"{name}-state").result()
    # 逐行讲解：等待任务完成并读取结果（training_client.save_weights_for_sampler(name=f'{name}-weights').result()），赋给 weights。
    weights = training_client.save_weights_for_sampler(
        name=f"{name}-weights"
    ).result()
    # 逐行讲解：调用 print（print(f'Saved state: {state.path}')）。
    print(f"Saved state: {state.path}")
    # 逐行讲解：调用 print（print(f'Saved sampler weights: {weights.path}')）。
    print(f"Saved sampler weights: {weights.path}")


# 逐行讲解：定义函数 serializable_config：封装下方逻辑，供调用方复用。
def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    # 逐行讲解：返回结果：{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}。
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


# 逐行讲解：定义类 WarmupRunner：warm-up 阶段：完整轨迹 + Monte Carlo target，只更新 critic。
class WarmupRunner:
    """warm-up 阶段：完整轨迹 + Monte Carlo target，只更新 critic。"""

    # 逐行讲解：定义函数 __init__：封装下方逻辑，供调用方复用。
    def __init__(self, args: argparse.Namespace, rollout_config: RolloutConfig) -> None:
        # 逐行讲解：取值 args，赋给 self.args。
        self.args = args
        # warm-up 让 macro 边界退化为终局，即完整轨迹 rollout。
        self.config = replace(rollout_config, macro_rounds=args.max_episode_steps)

    # 逐行讲解：定义函数 run_update：一次 warm-up 更新；返回 (指标, 本轮消耗的游戏数)。
    def run_update(
        self,
        training_client: Any,
        sampling_client: Any,
        tokenizer: Any,
        games: list[Any],
        cursor: int,
        *,
        update_index: int,
        seed_offset: int,
    ) -> tuple[dict[str, float], int]:
        """一次 warm-up 更新；返回 (指标, 本轮消耗的游戏数)。"""
        # 逐行讲解：取值 self.args，赋给 args。
        args = self.args
        # 逐行讲解：调用 take_batch（take_batch(games, cursor, args.warmup_games_per_batch)），赋给 fresh。
        fresh = take_batch(games, cursor, args.warmup_games_per_batch)
        # 逐行讲解：进入上下文管理器 tqdm(total=args.warmup_games_per_batch * args.branches, desc=f'Update {update_index + 1} warmup rollout', u…，退出时执行其收尾逻辑。
        with tqdm(
            total=args.warmup_games_per_batch * args.branches,
            desc=f"Update {update_index + 1} warmup rollout",
            unit="trajectory",
            position=1,
            leave=False,
        ) as rollout_progress:
            # 逐行讲解：调用 rollout_macro_steps（rollout_macro_steps(sampling_client, tokenizer, [], fresh, self.config, seed_offset=seed_offset, progress_c…），赋给 groups。
            groups = rollout_macro_steps(
                sampling_client,
                tokenizer,
                [],
                fresh,
                self.config,
                seed_offset=seed_offset,
                progress_callback=rollout_progress.update,
            )

        # 轨迹终局即 MC return；边界状态由 states.boundary_states 重建。
        # 注意局部变量不要也叫 boundary_states，会遮蔽导入的函数。
        snapshots: list[MacroState] = []
        # 逐行讲解：声明 mc_returns 的类型为 list[float]；构造列表 []。
        mc_returns: list[float] = []
        # 逐行讲解：遍历 groups，将每项绑定到 group。
        for group in groups:
            # 逐行讲解：遍历 group.trajectories，将每项绑定到 trajectory。
            for trajectory in group.trajectories:
                # 逐行讲解：按条件选择结果：1.0 if trajectory.won else 0.0，赋给 won。
                won = 1.0 if trajectory.won else 0.0
                # 逐行讲解：遍历 boundary_states(group.start_state, trajectory, macro_rounds=args.macro_rounds, max_trajectory_tokens=args.m…，将每项绑定到 state。
                for state in boundary_states(
                    group.start_state,
                    trajectory,
                    macro_rounds=args.macro_rounds,
                    max_trajectory_tokens=args.max_trajectory_tokens,
                ):
                    # 逐行讲解：向列表末尾追加一个元素（snapshots.append(state)）。
                    snapshots.append(state)
                    # 逐行讲解：向列表末尾追加一个元素（mc_returns.append(won)）。
                    mc_returns.append(won)

        # 逐行讲解：声明 requests 的类型为 list[CriticRequest]；构造列表 []。
        requests: list[CriticRequest] = []
        # 逐行讲解：声明 kept 的类型为 list[tuple[MacroState, float]]；构造列表 []。
        kept: list[tuple[MacroState, float]] = []
        # 逐行讲解：取值 0，赋给 prompt_too_long。
        prompt_too_long = 0
        # 逐行讲解：遍历 enumerate(zip(snapshots, mc_returns, strict=True))，将每项绑定到 (index, (state, mc_return))。
        for index, (state, mc_return) in enumerate(
            zip(snapshots, mc_returns, strict=True)
        ):
            # 逐行讲解：调用 critic_prompt_for_state（critic_prompt_for_state(tokenizer, state.messages, task=state.task, walkthrough=load_walkthrough(state.exam…），赋给 prompt。
            prompt = critic_prompt_for_state(
                tokenizer,
                state.messages,
                task=state.task,
                walkthrough=load_walkthrough(state.example.game_file),
            )
            # 逐行讲解：检查条件 len(prompt) > _critic_prompt_budget(args)；成立时执行下方分支。
            if len(prompt) > _critic_prompt_budget(args):
                # 逐行讲解：更新 prompt_too_long：prompt_too_long += 1。
                prompt_too_long += 1
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：向列表末尾追加一个元素（kept.append((state, mc_return))）。
            kept.append((state, mc_return))
            # 逐行讲解：向列表末尾追加一个元素（requests.append(CriticRequest(prompt_tokens=prompt, num_samples=args.critic_samples, seed=args.seed + seed_…）。
            requests.append(
                CriticRequest(
                    prompt_tokens=prompt,
                    num_samples=args.critic_samples,
                    seed=args.seed + seed_offset + index,
                )
            )

        # 逐行讲解：调用 sample_critic_values（sample_critic_values(sampling_client, tokenizer, requests, max_tokens=args.critic_max_tokens, temperature=a…），赋给 batches。
        batches = sample_critic_values(
            sampling_client,
            tokenizer,
            requests,
            max_tokens=args.critic_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # 逐行讲解：声明 datums 的类型为 list[TrainingDatum]；构造列表 []。
        datums: list[TrainingDatum] = []
        # 逐行讲解：声明 abs_errors 的类型为 list[float]；构造列表 []。
        abs_errors: list[float] = []
        # 逐行讲解：取值 0，赋给 parse_failures。
        parse_failures = 0
        # 逐行讲解：取值 0，赋给 total_samples。
        total_samples = 0
        # 逐行讲解：声明 estimates 的类型为 list[float]；构造列表 []。
        estimates: list[float] = []
        # 逐行讲解：声明 outcomes 的类型为 list[float]；构造列表 []。
        outcomes: list[float] = []
        # 逐行讲解：遍历 zip(kept, batches, strict=True)，将每项绑定到 ((state, mc_return), samples)。
        for (state, mc_return), samples in zip(kept, batches, strict=True):
            # 逐行讲解：调用 assemble_critic_group（assemble_critic_group(samples, mc_return)），赋给 signals。
            signals = assemble_critic_group(samples, mc_return)
            # 逐行讲解：把一组元素依次加入列表（abs_errors.extend(signals.errors)）。
            abs_errors.extend(signals.errors)
            # 逐行讲解：更新 parse_failures：parse_failures += signals.parse_failures。
            parse_failures += signals.parse_failures
            # 逐行讲解：更新 total_samples：total_samples += len(samples)。
            total_samples += len(samples)
            # 逐行讲解：检查条件 signals.mean_estimate is not None；成立时执行下方分支。
            if signals.mean_estimate is not None:
                # 逐行讲解：向列表末尾追加一个元素（estimates.append(signals.mean_estimate)）。
                estimates.append(signals.mean_estimate)
                # 逐行讲解：向列表末尾追加一个元素（outcomes.append(mc_return)）。
                outcomes.append(mc_return)
            # 逐行讲解：把一组元素依次加入列表（datums.extend(build_critic_datums(samples, signals, game_id=state.example.id, group_index=state.macro_index))）。
            datums.extend(
                build_critic_datums(
                    samples,
                    signals,
                    game_id=state.example.id,
                    group_index=state.macro_index,
                )
            )

        # 逐行讲解：声明 trainer_results 的类型为 list[Any]；构造列表 []。
        trainer_results: list[Any] = []
        # 逐行讲解：检查条件 datums；成立时执行下方分支。
        if datums:
            # 逐行讲解：调用 forward_backward_chunked（forward_backward_chunked(training_client, datums)），赋给 trainer_results。
            trainer_results = forward_backward_chunked(training_client, datums)
            # 逐行讲解：调用 training_client.optim_step(trio.AdamParams(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.b…（training_client.optim_step(trio.AdamParams(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.b…）。
            training_client.optim_step(
                trio.AdamParams(
                    learning_rate=args.learning_rate,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    weight_decay=args.weight_decay,
                    grad_clip_norm=args.grad_clip_norm,
                )
            ).result()

        # 逐行讲解：通过推导式逐项生成结果：[t for group in groups for t in group.trajectories]，赋给 trajectories。
        trajectories = [t for group in groups for t in group.trajectories]
        # 逐行讲解：声明 metrics 的类型为 dict[str, float]；构造字典 {'phase': 0.0, 'warmup/success_rate': mean([float(item.won) for item in trajectories]), 'warmup/steps_mean'…。
        metrics: dict[str, float] = {
            "phase": 0.0,
            "warmup/success_rate": mean(
                [float(item.won) for item in trajectories]
            ),
            "warmup/steps_mean": mean(
                [float(len(item.steps)) for item in trajectories]
            ),
            "warmup/boundary_states": float(len(snapshots)),
            "target/mc_return_mean": mean(mc_returns),
            "critic/abs_error_mean": mean(abs_errors),
            "critic/parse_failure_rate": (
                parse_failures / total_samples if total_samples else 0.0
            ),
            "critic/prompt_too_long_rate": (
                prompt_too_long / len(snapshots) if snapshots else 0.0
            ),
            "critic/mean_estimate_mean": mean(estimates),
            "train/batch_size": float(len(datums)),
            "train/tokens": float(sum(item.num_tokens for item in datums)),
            "train/loss_tokens": float(sum(item.loss_tokens for item in datums)),
        }
        # 逐行讲解：调用 pearson（pearson(estimates, outcomes)），赋给 correlation。
        correlation = pearson(estimates, outcomes)
        # 逐行讲解：检查条件 correlation is not None；成立时执行下方分支。
        if correlation is not None:
            # 逐行讲解：取值 correlation，赋给 metrics['critic/value_corr']。
            metrics["critic/value_corr"] = correlation
        # 逐行讲解：调用 metrics.update（metrics.update(_rollout_quality_metrics(trajectories))）。
        metrics.update(_rollout_quality_metrics(trajectories))
        # 逐行讲解：调用 metrics.update（metrics.update(merge_trainer_metrics(trainer_results))）。
        metrics.update(merge_trainer_metrics(trainer_results))
        # 逐行讲解：返回结果：(metrics, len(fresh))。
        return metrics, len(fresh)


# 逐行讲解：定义类 TDRunner：TD 阶段：macro-step rollout + 生成式 critic bootstrap。
class TDRunner:
    """TD 阶段：macro-step rollout + 生成式 critic bootstrap。"""

    # 逐行讲解：定义函数 __init__：封装下方逻辑，供调用方复用。
    def __init__(self, args: argparse.Namespace, rollout_config: RolloutConfig) -> None:
        # 逐行讲解：取值 args，赋给 self.args。
        self.args = args
        # 逐行讲解：取值 rollout_config，赋给 self.config。
        self.config = rollout_config
        # 逐行讲解：调用 StateStore（StateStore(args.state_store_capacity, seed=args.seed + 1)），赋给 self.store。
        self.store = StateStore(args.state_store_capacity, seed=args.seed + 1)

    # 逐行讲解：定义函数 run_update：一次 TD 更新：rollout -> 估终点 -> 拼 return/G -> 更新 -> 入库。
    def run_update(
        self,
        training_client: Any,
        sampling_client: Any,
        tokenizer: Any,
        games: list[Any],
        cursor: int,
        *,
        update_index: int,
        seed_offset: int,
    ) -> dict[str, float]:
        """一次 TD 更新：rollout -> 估终点 -> 拼 return/G -> 更新 -> 入库。"""
        # 逐行讲解：取值 self.args，赋给 args。
        args = self.args
        # 逐行讲解：提交生成采样请求（self.store.sample(args.states_per_batch)），赋给 start_states。
        start_states = self.store.sample(args.states_per_batch)
        # 逐行讲解：计算 args.states_per_batch - len(start_states)，赋给 fresh_count。
        fresh_count = args.states_per_batch - len(start_states)
        # 逐行讲解：按条件选择结果：take_batch(games, cursor, fresh_count) if fresh_count > 0 else []，赋给 fresh。
        fresh = take_batch(games, cursor, fresh_count) if fresh_count > 0 else []
        # 逐行讲解：进入上下文管理器 tqdm(total=(len(start_states) + len(fresh)) * args.branches, desc=f'Update {update_index + 1} TD rollout', …，退出时执行其收尾逻辑。
        with tqdm(
            total=(len(start_states) + len(fresh)) * args.branches,
            desc=f"Update {update_index + 1} TD rollout",
            unit="branch",
            position=1,
            leave=False,
        ) as rollout_progress:
            # 逐行讲解：调用 rollout_macro_steps（rollout_macro_steps(sampling_client, tokenizer, start_states, fresh, self.config, seed_offset=seed_offset, …），赋给 groups。
            groups = rollout_macro_steps(
                sampling_client,
                tokenizer,
                start_states,
                fresh,
                self.config,
                seed_offset=seed_offset,
                progress_callback=rollout_progress.update,
            )

        # ---- 终点估值：非终局分支的截断尾部由 critic 补齐 ----
        endpoint_requests: list[tuple[int, int, CriticRequest]] = []
        # 逐行讲解：取值 0，赋给 prompt_too_long。
        prompt_too_long = 0
        # 逐行讲解：遍历 enumerate(groups)，将每项绑定到 (game_index, group)。
        for game_index, group in enumerate(groups):
            # 逐行讲解：遍历 enumerate(group.trajectories)，将每项绑定到 (branch_index, trajectory)。
            for branch_index, trajectory in enumerate(group.trajectories):
                # 逐行讲解：检查条件 trajectory.stop_reason == 'macro_boundary'；成立时执行下方分支。
                if trajectory.stop_reason == "macro_boundary":
                    # 逐行讲解：调用 critic_prompt_for_state（critic_prompt_for_state(tokenizer, trajectory.messages, task=trajectory.task, walkthrough=load_walkthrough(…），赋给 prompt。
                    prompt = critic_prompt_for_state(
                        tokenizer,
                        trajectory.messages,
                        task=trajectory.task,
                        walkthrough=load_walkthrough(trajectory.example.game_file),
                    )
                    # 逐行讲解：检查条件 len(prompt) > _critic_prompt_budget(args)；成立时执行下方分支。
                    if len(prompt) > _critic_prompt_budget(args):
                        # 逐行讲解：更新 prompt_too_long：prompt_too_long += 1。
                        prompt_too_long += 1
                        # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                        continue
                    # 逐行讲解：向列表末尾追加一个元素（endpoint_requests.append((game_index, branch_index, CriticRequest(prompt_tokens=prompt, num_samples=args.en…）。
                    endpoint_requests.append(
                        (
                            game_index,
                            branch_index,
                            CriticRequest(
                                prompt_tokens=prompt,
                                num_samples=args.endpoint_samples,
                                seed=(
                                    args.seed
                                    + seed_offset
                                    + game_index * 10_000
                                    + branch_index
                                ),
                            ),
                        )
                    )
        # 逐行讲解：调用 sample_critic_values（sample_critic_values(sampling_client, tokenizer, [request for _, _, request in endpoint_requests], max_toke…），赋给 endpoint_batches。
        endpoint_batches = sample_critic_values(
            sampling_client,
            tokenizer,
            [request for _, _, request in endpoint_requests],
            max_tokens=args.critic_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        # 逐行讲解：取值 0，赋给 endpoint_parse_failures。
        endpoint_parse_failures = 0
        # 逐行讲解：声明 endpoint_values 的类型为 list[float]；构造列表 []。
        endpoint_values: list[float] = []
        # 逐行讲解：遍历 zip(endpoint_requests, endpoint_batches, strict=True)，将每项绑定到 ((game_index, branch_index, _), samples)。
        for (game_index, branch_index, _), samples in zip(
            endpoint_requests, endpoint_batches, strict=True
        ):
            # 逐行讲解：调用 endpoint_value（endpoint_value(samples)），赋给 (value, failures)。
            value, failures = endpoint_value(samples)
            # 逐行讲解：取值 value，赋给 groups[game_index].trajectories[branch_index].endpoint_value。
            groups[game_index].trajectories[branch_index].endpoint_value = value
            # 逐行讲解：更新 endpoint_parse_failures：endpoint_parse_failures += failures。
            endpoint_parse_failures += failures
            # 逐行讲解：向列表末尾追加一个元素（endpoint_values.append(value)）。
            endpoint_values.append(value)

        # ---- 信号拼装：actor return/G，随后同一起点的 critic 训练组 ----
        actor_signals: list[ActorSignals] = [
            assemble_actor_group(group) for group in groups
        ]
        # 逐行讲解：声明 critic_requests 的类型为 list[CriticRequest]；构造列表 []。
        critic_requests: list[CriticRequest] = []
        # 逐行讲解：声明 critic_states 的类型为 list[tuple[MacroGroup, float]]；构造列表 []。
        critic_states: list[tuple[MacroGroup, float]] = []
        # 逐行讲解：取值 0，赋给 start_prompt_too_long。
        start_prompt_too_long = 0
        # 逐行讲解：遍历 enumerate(groups)，将每项绑定到 (index, group)。
        for index, group in enumerate(groups):
            # 逐行讲解：调用 critic_prompt_for_state（critic_prompt_for_state(tokenizer, group.start_state.messages, task=group.start_state.task, walkthrough=loa…），赋给 prompt。
            prompt = critic_prompt_for_state(
                tokenizer,
                group.start_state.messages,
                task=group.start_state.task,
                walkthrough=load_walkthrough(group.start_state.example.game_file),
            )
            # 逐行讲解：检查条件 len(prompt) > _critic_prompt_budget(args)；成立时执行下方分支。
            if len(prompt) > _critic_prompt_budget(args):
                # 逐行讲解：更新 start_prompt_too_long：start_prompt_too_long += 1。
                start_prompt_too_long += 1
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：向列表末尾追加一个元素（critic_states.append((group, actor_signals[index].value_target))）。
            critic_states.append((group, actor_signals[index].value_target))
            # 逐行讲解：向列表末尾追加一个元素（critic_requests.append(CriticRequest(prompt_tokens=prompt, num_samples=args.critic_samples, seed=args.seed …）。
            critic_requests.append(
                CriticRequest(
                    prompt_tokens=prompt,
                    num_samples=args.critic_samples,
                    seed=args.seed + seed_offset + 100_000 + index,
                )
            )
        # 逐行讲解：调用 sample_critic_values（sample_critic_values(sampling_client, tokenizer, critic_requests, max_tokens=args.critic_max_tokens, temper…），赋给 critic_batches。
        critic_batches = sample_critic_values(
            sampling_client,
            tokenizer,
            critic_requests,
            max_tokens=args.critic_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # 逐行讲解：声明 datums 的类型为 list[TrainingDatum]；构造列表 []。
        datums: list[TrainingDatum] = []
        # 逐行讲解：遍历 zip(groups, actor_signals, strict=True)，将每项绑定到 (group, signals)。
        for group, signals in zip(groups, actor_signals, strict=True):
            # 逐行讲解：把一组元素依次加入列表（datums.extend(build_actor_datums(group, signals, max_sequence_tokens=args.max_trajectory_tokens))）。
            datums.extend(
                build_actor_datums(
                    group,
                    signals,
                    max_sequence_tokens=args.max_trajectory_tokens,
                )
            )

        # 逐行讲解：声明 critic_signals 的类型为 list[CriticSignals]；构造列表 []。
        critic_signals: list[CriticSignals] = []
        # 逐行讲解：声明 abs_errors 的类型为 list[float]；构造列表 []。
        abs_errors: list[float] = []
        # 逐行讲解：取值 0，赋给 parse_failures。
        parse_failures = 0
        # 逐行讲解：取值 0，赋给 total_samples。
        total_samples = 0
        # 逐行讲解：遍历 zip(critic_states, critic_batches, strict=True)，将每项绑定到 ((group, value_target), samples)。
        for (group, value_target), samples in zip(
            critic_states, critic_batches, strict=True
        ):
            # 逐行讲解：调用 assemble_critic_group（assemble_critic_group(samples, value_target)），赋给 signals。
            signals = assemble_critic_group(samples, value_target)
            # 逐行讲解：向列表末尾追加一个元素（critic_signals.append(signals)）。
            critic_signals.append(signals)
            # 逐行讲解：把一组元素依次加入列表（abs_errors.extend(signals.errors)）。
            abs_errors.extend(signals.errors)
            # 逐行讲解：更新 parse_failures：parse_failures += signals.parse_failures。
            parse_failures += signals.parse_failures
            # 逐行讲解：更新 total_samples：total_samples += len(samples)。
            total_samples += len(samples)
            # 逐行讲解：把一组元素依次加入列表（datums.extend(build_critic_datums(samples, signals, game_id=group.start_state.example.id, group_index=group…）。
            datums.extend(
                build_critic_datums(
                    samples,
                    signals,
                    game_id=group.start_state.example.id,
                    group_index=group.start_state.macro_index,
                )
            )

        # 逐行讲解：声明 trainer_results 的类型为 list[Any]；构造列表 []。
        trainer_results: list[Any] = []
        # 逐行讲解：检查条件 datums；成立时执行下方分支。
        if datums:
            # 逐行讲解：调用 forward_backward_chunked（forward_backward_chunked(training_client, datums)），赋给 trainer_results。
            trainer_results = forward_backward_chunked(training_client, datums)
            # 逐行讲解：调用 training_client.optim_step(trio.AdamParams(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.b…（training_client.optim_step(trio.AdamParams(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.b…）。
            training_client.optim_step(
                trio.AdamParams(
                    learning_rate=args.learning_rate,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    weight_decay=args.weight_decay,
                    grad_clip_norm=args.grad_clip_norm,
                )
            ).result()

        # ---- 终点入库，供后续训练继续向后推进 ----
        saved_endpoints: list[MacroState] = []
        # 逐行讲解：遍历 groups，将每项绑定到 group。
        for group in groups:
            # 逐行讲解：遍历 group.trajectories，将每项绑定到 trajectory。
            for trajectory in group.trajectories:
                # 逐行讲解：调用 endpoint_state（endpoint_state(group.start_state, trajectory, config=self.config)），赋给 state。
                state = endpoint_state(
                    group.start_state,
                    trajectory,
                    config=self.config,
                )
                # 逐行讲解：检查条件 state is not None；成立时执行下方分支。
                if state is not None:
                    # 逐行讲解：向列表末尾追加一个元素（saved_endpoints.append(state)）。
                    saved_endpoints.append(state)
        # 逐行讲解：调用 self.store.add（self.store.add(saved_endpoints)）。
        self.store.add(saved_endpoints)

        # 逐行讲解：通过推导式逐项生成结果：[t for group in groups for t in group.trajectories]，赋给 trajectories。
        trajectories = [t for group in groups for t in group.trajectories]
        # 逐行讲解：求和（sum((trajectory.stop_reason == 'macro_boundary' for trajectory in trajectories))），赋给 total_macro_boundary。
        total_macro_boundary = sum(
            trajectory.stop_reason == "macro_boundary" for trajectory in trajectories
        )
        # 逐行讲解：声明 metrics 的类型为 dict[str, float]；构造字典 {'phase': 1.0, 'td/store_size': float(len(self.store)), 'td/new_endpoints': float(len(saved_endpoints)), 't…。
        metrics: dict[str, float] = {
            "phase": 1.0,
            "td/store_size": float(len(self.store)),
            "td/new_endpoints": float(len(saved_endpoints)),
            "td/fresh_states": float(len(fresh)),
            "td/macro_index_mean": mean(
                [float(group.start_state.macro_index) for group in groups]
            ),
            "actor/segment_reward_mean": mean(
                [item.segment_reward for item in trajectories]
            ),
            "actor/terminal_won_rate": mean(
                [float(item.won) for item in trajectories]
            ),
            "actor/macro_boundary_rate": (
                total_macro_boundary / len(trajectories) if trajectories else 0.0
            ),
            "actor/degenerate_group_rate": mean(
                [float(signal.degenerate) for signal in actor_signals]
            ),
            "actor/value_target_mean": mean(
                [signal.value_target for signal in actor_signals]
            ),
            "actor/value_target_std": population_std(
                [signal.value_target for signal in actor_signals]
            ),
            "actor/value_gap_zero_reward": mean(
                [signal.zero_reward_value_std for signal in actor_signals]
            ),
            "critic/abs_error_mean": mean(abs_errors),
            "critic/parse_failure_rate": (
                parse_failures / total_samples if total_samples else 0.0
            ),
            "critic/endpoint_value_mean": mean(endpoint_values),
            "critic/endpoint_parse_failures": float(endpoint_parse_failures),
            "critic/prompt_too_long_rate": (
                (prompt_too_long + start_prompt_too_long)
                / max(2 * len(groups), 1)
            ),
            "train/actor_datums": float(
                sum(item.kind == "actor" for item in datums)
            ),
            "train/critic_datums": float(
                sum(item.kind == "critic" for item in datums)
            ),
            "train/batch_size": float(len(datums)),
            "train/tokens": float(sum(item.num_tokens for item in datums)),
            "train/loss_tokens": float(sum(item.loss_tokens for item in datums)),
        }
        # 逐行讲解：调用 metrics.update（metrics.update(_rollout_quality_metrics(trajectories))）。
        metrics.update(_rollout_quality_metrics(trajectories))
        # 逐行讲解：调用 metrics.update（metrics.update(merge_trainer_metrics(trainer_results))）。
        metrics.update(merge_trainer_metrics(trainer_results))
        # 逐行讲解：返回结果：metrics。
        return metrics


# 逐行讲解：定义函数 _log_update：封装下方逻辑，供调用方复用。
def _log_update(
    progress: tqdm,
    label: str,
    metrics: dict[str, float],
    started: float,
) -> None:
    # 逐行讲解：调用 progress.update（progress.update(1)）。
    progress.update(1)
    # 逐行讲解：调用 pick_mean_loss（pick_mean_loss(metrics)），赋给 loss。
    loss = pick_mean_loss(metrics)
    # 逐行讲解：按条件选择结果：'skipped' if metrics.get('train/batch_size', 0.0) == 0.0 else f'{loss:.4f}' if loss is not None else 'missi…，赋给 loss_text。
    loss_text = (
        "skipped"
        if metrics.get("train/batch_size", 0.0) == 0.0
        else (f"{loss:.4f}" if loss is not None else "missing")
    )
    # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(success=f"{metrics.get('actor/segment_reward_mean', metrics.get('warmup/success_rate',…）。
    progress.set_postfix(
        success=f"{metrics.get('actor/segment_reward_mean', metrics.get('warmup/success_rate', 0.0)):.3f}",
        loss=loss_text,
        step_s=f"{perf_counter() - started:.1f}",
        refresh=True,
    )
    # 逐行讲解：调用 tqdm.write（tqdm.write(f'{label} ' + ' '.join((f'{key}={value:.3f}' for key, value in sorted(metrics.items()) if isinst…）。
    tqdm.write(
        f"{label} "
        + " ".join(
            f"{key}={value:.3f}"
            for key, value in sorted(metrics.items())
            if isinstance(value, float)
            and (
                key.startswith(("warmup/", "actor/", "critic/abs", "td/store"))
            )
        )
        + f" loss={loss_text}"
    )


# 逐行讲解：定义函数 main：warm-up -> TD 两阶段训练闭环。
def main(args: argparse.Namespace) -> None:
    """warm-up -> TD 两阶段训练闭环。"""
    # 逐行讲解：调用 shuffled_games（shuffled_games(args.data_root, args.train_split, args.seed, max_games=args.max_train_games)），赋给 games。
    games = shuffled_games(
        args.data_root,
        args.train_split,
        args.seed,
        max_games=args.max_train_games,
    )
    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：调用 service_client.create_lora_training_client（service_client.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank, seed=args.seed)），赋给 training_client。
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    # 逐行讲解：调用 training_client.get_tokenizer（training_client.get_tokenizer()），赋给 tokenizer。
    tokenizer = training_client.get_tokenizer()

    # 逐行讲解：调用 RolloutConfig（RolloutConfig(branches=args.branches, macro_rounds=args.macro_rounds, max_episode_steps=args.max_episode_st…），赋给 rollout_config。
    rollout_config = RolloutConfig(
        branches=args.branches,
        macro_rounds=args.macro_rounds,
        max_episode_steps=args.max_episode_steps,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        include_admissible_actions=args.include_admissible_actions,
        environment_asynchronous=args.environment_asynchronous,
    )
    # 逐行讲解：调用 WarmupRunner（WarmupRunner(args, rollout_config)），赋给 warmup。
    warmup = WarmupRunner(args, rollout_config)
    # 逐行讲解：调用 TDRunner（TDRunner(args, rollout_config)），赋给 td。
    td = TDRunner(args, rollout_config)

    # 逐行讲解：计算 args.warmup_updates + args.td_updates，赋给 total_updates。
    total_updates = args.warmup_updates + args.td_updates
    # 逐行讲解：调用 swanlab.init（swanlab.init(project=args.swanlab_project, name=args.run_name, mode=args.swanlab_mode, config=serializable_…）。
    swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config=serializable_config(args),
        tags=["pytrio", "alfworld", "tempo", "macro-step"],
    )
    # 逐行讲解：取值 0，赋给 cursor。
    cursor = 0
    # 逐行讲解：取值 0，赋给 update_index。
    update_index = 0
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：进入上下文管理器 tqdm(total=total_updates, desc='TEMPO training', unit='update')，退出时执行其收尾逻辑。
        with tqdm(
            total=total_updates, desc="TEMPO training", unit="update"
        ) as progress:
            # 逐行讲解：遍历 range(args.warmup_updates)，将每项绑定到 _。
            for _ in range(args.warmup_updates):
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 started。
                started = perf_counter()
                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='sampler', refresh=True)）。
                progress.set_postfix(phase="sampler", refresh=True)
                # 逐行讲解：保存当前权重并创建对应采样客户端（training_client.save_weights_and_get_sampling_client()），赋给 sampling_client。
                sampling_client = (
                    training_client.save_weights_and_get_sampling_client()
                )
                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='warmup-rollout', refresh=True)）。
                progress.set_postfix(phase="warmup-rollout", refresh=True)
                # 逐行讲解：调用 warmup.run_update（warmup.run_update(training_client, sampling_client, tokenizer, games, cursor, update_index=update_index, se…），赋给 (metrics, used)。
                metrics, used = warmup.run_update(
                    training_client,
                    sampling_client,
                    tokenizer,
                    games,
                    cursor,
                    update_index=update_index,
                    seed_offset=update_index * 7919,
                )
                # 逐行讲解：更新 cursor：cursor += used。
                cursor += used
                # 逐行讲解：计算 perf_counter() - started，赋给 metrics['time/update_seconds']。
                metrics["time/update_seconds"] = perf_counter() - started
                # 逐行讲解：计算自然对数（swanlab.log(metrics, step=update_index)）。
                swanlab.log(metrics, step=update_index)
                # 逐行讲解：调用 _log_update（_log_update(progress, f'warmup={update_index + 1}', metrics, started)）。
                _log_update(progress, f"warmup={update_index + 1}", metrics, started)
                # 逐行讲解：更新 update_index：update_index += 1。
                update_index += 1

            # 逐行讲解：遍历 range(args.td_updates)，将每项绑定到 _。
            for _ in range(args.td_updates):
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 started。
                started = perf_counter()
                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='sampler', refresh=True)）。
                progress.set_postfix(phase="sampler", refresh=True)
                # 逐行讲解：保存当前权重并创建对应采样客户端（training_client.save_weights_and_get_sampling_client()），赋给 sampling_client。
                sampling_client = (
                    training_client.save_weights_and_get_sampling_client()
                )
                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='td-rollout', refresh=True)）。
                progress.set_postfix(phase="td-rollout", refresh=True)
                # 逐行讲解：调用 td.run_update（td.run_update(training_client, sampling_client, tokenizer, games, cursor, update_index=update_index, seed_o…），赋给 metrics。
                metrics = td.run_update(
                    training_client,
                    sampling_client,
                    tokenizer,
                    games,
                    cursor,
                    update_index=update_index,
                    seed_offset=update_index * 7919,
                )
                # 逐行讲解：更新 cursor：cursor += int(metrics['td/fresh_states'])。
                cursor += int(metrics["td/fresh_states"])
                # 逐行讲解：计算 perf_counter() - started，赋给 metrics['time/update_seconds']。
                metrics["time/update_seconds"] = perf_counter() - started
                # 逐行讲解：计算自然对数（swanlab.log(metrics, step=update_index)）。
                swanlab.log(metrics, step=update_index)
                # 逐行讲解：调用 _log_update（_log_update(progress, f'td={update_index + 1}', metrics, started)）。
                _log_update(progress, f"td={update_index + 1}", metrics, started)
                # 逐行讲解：更新 update_index：update_index += 1。
                update_index += 1

                # 逐行讲解：检查条件 args.save_every > 0 and update_index % args.save_every == 0；成立时执行下方分支。
                if args.save_every > 0 and update_index % args.save_every == 0:
                    # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='checkpoint', refresh=True)）。
                    progress.set_postfix(phase="checkpoint", refresh=True)
                    # 逐行讲解：调用 save_checkpoint（save_checkpoint(training_client, f'{args.run_name}-update-{update_index}')）。
                    save_checkpoint(
                        training_client,
                        f"{args.run_name}-update-{update_index}",
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
