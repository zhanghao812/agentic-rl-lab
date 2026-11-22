"""使用 PyTRIO 在 text-only ALFWorld 上训练逐步决策策略。

先下载环境数据，再从本目录启动一次最小训练：

uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/datasets/alfworld"

测试:
uv run --extra alfworld python train.py \
    --max-steps 10 \
    --games-per-batch 4 \
    --group-size 8 \
    --save-every 5 \
    --swanlab-mode disabled

正式训练：（预计需要12小时）
uv run --extra alfworld python train.py \
    --max-steps 80 \
    --games-per-batch 8 \
    --group-size 8 \
    --save-every 20 \
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
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
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

# 逐行讲解：从 data 导入 SPLIT_DIRECTORIES, default_data_root, shuffled_games, take_batch。
from data import SPLIT_DIRECTORIES, default_data_root, shuffled_games, take_batch
# 逐行讲解：从 rollout 导入 MAX_SEQUENCE_TOKENS, RolloutBatch, RolloutConfig, Trajectory, rollout_batch。
from rollout import (
    MAX_SEQUENCE_TOKENS,
    RolloutBatch,
    RolloutConfig,
    Trajectory,
    rollout_batch,
)


# 逐行讲解：取值 128，赋给 MAX_TRAIN_BATCH_SIZE。
MAX_TRAIN_BATCH_SIZE = 128


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
    # 逐行讲解：声明 game_id 的类型为 str。
    game_id: str
    # 逐行讲解：声明 group_index 的类型为 int。
    group_index: int
    # 逐行讲解：声明 step_count 的类型为 int。
    step_count: int


# 逐行讲解：定义函数 _path：封装下方逻辑，供调用方复用。
def _path(value: str | Path) -> Path:
    # 逐行讲解：返回结果：Path(os.path.expandvars(str(value))).expanduser().resolve()。
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


# 逐行讲解：定义函数 parse_args：解析训练、rollout、奖励和日志参数。
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析训练、rollout、奖励和日志参数。"""
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data-root', type=Path, default=default_data_root())）。
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--train-split', choices=list(SPLIT_DIRECTORIES), default='train')）。
    parser.add_argument(
        "--train-split",
        choices=list(SPLIT_DIRECTORIES),
        default="train",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-train-games', type=int, default=0, help='最多使用多少局；0 表示使用完整训练 split')）。
    parser.add_argument(
        "--max-train-games",
        type=int,
        default=0,
        help="最多使用多少局；0 表示使用完整训练 split",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-steps', type=int, default=200, help='最大训练迭代次数，与单条轨迹的 --max-episode-steps 不同')）。
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="最大训练迭代次数，与单条轨迹的 --max-episode-steps 不同",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--games-per-batch', type=int, default=8)）。
    parser.add_argument(
        "--games-per-batch",
        type=int,
        default=8,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--group-size', type=int, default=8)）。
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-episode-steps', type=int, default=50)）。
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=50,
    )
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
    parser.add_argument(
        "--max-trajectory-tokens",
        type=int,
        default=MAX_SEQUENCE_TOKENS,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-assistant-tokens', type=int, default=2048)）。
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=2048,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0)）。
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0)）。
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B')）。
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=32)）。
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=1e-06)）。
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta1', type=float, default=0.9)）。
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta2', type=float, default=0.95)）。
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--weight-decay', type=float, default=0.0)）。
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--grad-clip-norm', type=float, default=1.0)）。
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--ppo-clip-low', type=float, default=0.8)）。
    parser.add_argument(
        "--ppo-clip-low",
        type=float,
        default=0.8,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--ppo-clip-high', type=float, default=1.2)）。
    parser.add_argument(
        "--ppo-clip-high",
        type=float,
        default=1.2,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every', type=int, default=25, help='每隔多少次 update 保存 state 和 sampler weights；0 关…）。
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help=(
            "每隔多少次 update 保存 state 和 sampler weights；"
            "0 关闭中间保存，最终产物始终保存"
        ),
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--run-name', default='alfworld-agent-rl-qwen35-4b')）。
    parser.add_argument(
        "--run-name",
        default="alfworld-agent-rl-qwen35-4b",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-alfworld')）。
    parser.add_argument(
        "--swanlab-project",
        default="agentic-rl-lab-alfworld",
    )
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
    # 逐行讲解：构造字典 {'max_steps': args.max_steps, 'games_per_batch': args.games_per_batch, 'group_size': args.group_size, 'max_…，赋给 positive。
    positive = {
        "max_steps": args.max_steps,
        "games_per_batch": args.games_per_batch,
        "group_size": args.group_size,
        "max_episode_steps": args.max_episode_steps,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "max_assistant_tokens": args.max_assistant_tokens,
        "lora_rank": args.lora_rank,
    }
    # 逐行讲解：遍历 positive.items()，将每项绑定到 (name, value)。
    for name, value in positive.items():
        # 逐行讲解：检查条件 value < 1；成立时执行下方分支。
        if value < 1:
            # 逐行讲解：抛出异常：ValueError(f'{name} 必须大于等于 1')。
            raise ValueError(f"{name} 必须大于等于 1")
    # 逐行讲解：检查条件 not 0.0 < args.ppo_clip_low <= 1.0 <= args.ppo_clip_high；成立时执行下方分支。
    if not 0.0 < args.ppo_clip_low <= 1.0 <= args.ppo_clip_high:
        # 逐行讲解：抛出异常：ValueError('PPO clip 阈值必须满足 0 < low <= 1 <= high')。
        raise ValueError("PPO clip 阈值必须满足 0 < low <= 1 <= high")
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
    # 逐行讲解：计算 args.games_per_batch * args.group_size，赋给 rollout_batch_size。
    rollout_batch_size = args.games_per_batch * args.group_size
    # 逐行讲解：检查条件 rollout_batch_size > MAX_TRAIN_BATCH_SIZE；成立时执行下方分支。
    if rollout_batch_size > MAX_TRAIN_BATCH_SIZE:
        # 逐行讲解：抛出异常：ValueError(f'games_per_batch * group_size 不能超过平台 batch 上限 {MAX_TRAIN_BATCH_SIZE}')。
        raise ValueError(
            f"games_per_batch * group_size 不能超过平台 batch 上限 "
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


# 逐行讲解：定义函数 build_trajectory_datum：把完整工具轨迹构造成一个右移对齐的 PPO Datum。
def build_trajectory_datum(
    trajectory: Trajectory,
    *,
    max_sequence_tokens: int,
) -> TrainingDatum:
    """把完整工具轨迹构造成一个右移对齐的 PPO Datum。

    每轮 prompt 相对上一轮只增加 assistant 闭合符和 tool observation。
    这些环境 token 的 old logprob/advantage 均为 0；所有 assistant completion
    使用采样时保存的 old logprob 和整条轨迹的 group-relative advantage。
    """
    # 逐行讲解：检查条件 not trajectory.steps；成立时执行下方分支。
    if not trajectory.steps:
        # 逐行讲解：抛出异常：ValueError('不能用空轨迹构造 Datum')。
        raise ValueError("不能用空轨迹构造 Datum")

    # 逐行讲解：声明 full_tokens 的类型为 list[int]；构造列表 []。
    full_tokens: list[int] = []
    # 逐行讲解：声明 old_logprobs_by_token 的类型为 list[float]；构造列表 []。
    old_logprobs_by_token: list[float] = []
    # 逐行讲解：声明 advantages_by_token 的类型为 list[float]；构造列表 []。
    advantages_by_token: list[float] = []
    # 逐行讲解：取值 0，赋给 assistant_token_count。
    assistant_token_count = 0
    # 逐行讲解：取值 0，赋给 step_count。
    step_count = 0

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
            # 逐行讲解：取值 step.prompt_tokens，赋给 delta_environment。
            delta_environment = step.prompt_tokens
        # 逐行讲解：检查条件 step.prompt_tokens[:len(full_tokens)] == full_tokens；成立时执行下方分支。
        elif step.prompt_tokens[: len(full_tokens)] == full_tokens:
            # 逐行讲解：按索引或切片读取 step.prompt_tokens[len(full_tokens):]，赋给 delta_environment。
            delta_environment = step.prompt_tokens[len(full_tokens) :]
        else:
            # 逐行讲解：抛出异常：ValueError(f'第 {step.index} 步 prompt 不是已有轨迹的前缀扩展')。
            raise ValueError(
                f"第 {step.index} 步 prompt 不是已有轨迹的前缀扩展"
            )

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
        # 逐行讲解：更新 step_count：step_count += 1。
        step_count += 1

    # 最后一次工具返回后可能不再生成 assistant，但它仍属于完整轨迹，
    # 因此保留真实 token，并明确把训练信号设为 0。
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
        # 逐行讲解：抛出异常：ValueError('不能用没有 assistant token 的轨迹构造 Datum')。
        raise ValueError("不能用没有 assistant token 的轨迹构造 Datum")
    # 逐行讲解：检查条件 not len(full_tokens) == len(old_logprobs_by_token) == len(advantages_by_token)；成立时执行下方分支。
    if not (
        len(full_tokens)
        == len(old_logprobs_by_token)
        == len(advantages_by_token)
    ):
        # 逐行讲解：抛出异常：ValueError('完整轨迹的 token、logprob 和 advantage 长度不一致')。
        raise ValueError("完整轨迹的 token、logprob 和 advantage 长度不一致")
    # 逐行讲解：检查条件 len(full_tokens) > max_sequence_tokens；成立时执行下方分支。
    if len(full_tokens) > max_sequence_tokens:
        # 逐行讲解：抛出异常：ValueError(f'轨迹为 {len(full_tokens)} tokens，超过上限 {max_sequence_tokens}')。
        raise ValueError(
            f"轨迹为 {len(full_tokens)} tokens，超过上限 {max_sequence_tokens}"
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
    # 逐行讲解：调用 datum.validate_loss_inputs（datum.validate_loss_inputs('ppo')）。
    datum.validate_loss_inputs("ppo")
    # 逐行讲解：返回结果：TrainingDatum(datum=datum, num_tokens=len(input_tokens), loss_tokens=assistant_token_count, game_id=traject…。
    return TrainingDatum(
        datum=datum,
        num_tokens=len(input_tokens),
        loss_tokens=assistant_token_count,
        game_id=trajectory.example.id,
        group_index=trajectory.group_index,
        step_count=step_count,
    )


# 逐行讲解：定义函数 logical_step_count：统计有 completion 的全部 step；零 advantage 也属于逻辑 batch。
def logical_step_count(trajectories: list[Trajectory]) -> int:
    """统计有 completion 的全部 step；零 advantage 也属于逻辑 batch。"""
    # 逐行讲解：返回结果：sum((bool(step.completion_tokens) for trajectory in trajectories for step in trajectory.steps))。
    return sum(
        bool(step.completion_tokens)
        for trajectory in trajectories
        for step in trajectory.steps
    )


# 逐行讲解：定义函数 build_training_datums：每条轨迹构造一个 Datum，并跳过 advantage 为零的轨迹。
def build_training_datums(
    trajectories: list[Trajectory],
    *,
    max_sequence_tokens: int,
) -> list[TrainingDatum]:
    """每条轨迹构造一个 Datum，并跳过 advantage 为零的轨迹。"""
    # 逐行讲解：声明 datums 的类型为 list[TrainingDatum]；构造列表 []。
    datums: list[TrainingDatum] = []
    # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
    for trajectory in trajectories:
        # 逐行讲解：通过推导式逐项生成结果：[step for step in trajectory.steps if step.completion_tokens]，赋给 trainable_steps。
        trainable_steps = [
            step for step in trajectory.steps if step.completion_tokens
        ]
        # 逐行讲解：检查条件 not trainable_steps or abs(trajectory.advantage) <= 1e-12；成立时执行下方分支。
        if not trainable_steps or abs(trajectory.advantage) <= 1e-12:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：向列表末尾追加一个元素（datums.append(build_trajectory_datum(trajectory, max_sequence_tokens=max_sequence_tokens))）。
        datums.append(
            build_trajectory_datum(
                trajectory,
                max_sequence_tokens=max_sequence_tokens,
            )
        )
    # 逐行讲解：返回结果：datums。
    return datums


# 逐行讲解：定义函数 mean：封装下方逻辑，供调用方复用。
def mean(values: list[float]) -> float:
    # 逐行讲解：返回结果：sum(values) / len(values) if values else 0.0。
    return sum(values) / len(values) if values else 0.0


# 逐行讲解：定义函数 population_std：封装下方逻辑，供调用方复用。
def population_std(values: list[float]) -> float:
    # 逐行讲解：返回结果：float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0。
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0


# 逐行讲解：定义函数 rollout_metrics：汇总环境、终局奖励、advantage 与完整轨迹 Datum 指标。
def rollout_metrics(
    batch: RolloutBatch,
    datums: list[TrainingDatum],
    logical_steps: int,
) -> dict[str, float]:
    """汇总环境、终局奖励、advantage 与完整轨迹 Datum 指标。"""
    # 逐行讲解：取值 batch.trajectories，赋给 trajectories。
    trajectories = batch.trajectories
    # 逐行讲解：通过推导式逐项生成结果：[step for trajectory in trajectories for step in trajectory.steps]，赋给 steps。
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    # 逐行讲解：通过推导式逐项生成结果：[trajectory.reward for trajectory in trajectories]，赋给 rewards。
    rewards = [trajectory.reward for trajectory in trajectories]
    # 逐行讲解：通过推导式逐项生成结果：[trajectory.advantage for trajectory in trajectories]，赋给 advantages。
    advantages = [trajectory.advantage for trajectory in trajectories]
    # 逐行讲解：求和（sum((any((step.completion_tokens for step in trajectory.steps)) for trajectory in trajectories))），赋给 logical_trajectories。
    logical_trajectories = sum(
        any(step.completion_tokens for step in trajectory.steps)
        for trajectory in trajectories
    )
    # 逐行讲解：构造字典 {'reward/mean': mean(rewards), 'reward/success_rate': mean([float(item.won) for item in trajectories]), 'ro…，赋给 metrics。
    metrics = {
        "reward/mean": mean(rewards),
        "reward/success_rate": mean([float(item.won) for item in trajectories]),
        "rollout/steps_mean": mean(
            [float(len(trajectory.steps)) for trajectory in trajectories]
        ),
        "rollout/truncated_rate": mean(
            [float(item.truncated) for item in trajectories]
        ),
        "rollout/valid_tool_call_rate": mean(
            [float(step.valid_format) for step in steps]
        ),
        "rollout/admissible_action_rate": mean(
            [float(step.admissible) for step in steps]
        ),
        "rollout/valid_action_rate": mean(
            [float(step.valid_action) for step in steps]
        ),
        "rollout/invalid_actions_mean": mean(
            [float(item.invalid_action_count) for item in trajectories]
        ),
        "advantage/std": population_std(advantages),
        "advantage/groups": float(batch.advantage_stats.groups),
        "advantage/degenerate_group_rate": (
            batch.advantage_stats.degenerate_groups
            / max(batch.advantage_stats.groups, 1)
        ),
        "train/logical_steps": float(logical_steps),
        "train/logical_trajectories": float(logical_trajectories),
        "train/nonzero_trajectory_datums": float(len(datums)),
        "train/batch_size": float(len(datums)),
        "train/tokens": float(sum(item.num_tokens for item in datums)),
        "train/loss_tokens": float(sum(item.loss_tokens for item in datums)),
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
    # 逐行讲解：声明 merged 的类型为 dict[str, float]；构造字典 {}。
    merged: dict[str, float] = {}
    # 逐行讲解：遍历 values.items()，将每项绑定到 (key, items)。
    for key, items in values.items():
        # 逐行讲解：计算均值（mean(items)），赋给 merged[f'trainer/{key}']。
        merged[f"trainer/{key}"] = mean(items)
    # 逐行讲解：返回结果：merged。
    return merged


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


# 逐行讲解：定义函数 save_checkpoint：同时保存训练 state 和供 eval.py 使用的 sampler weights。
def save_checkpoint(training_client: Any, name: str) -> None:
    """同时保存训练 state 和供 eval.py 使用的 sampler weights。"""
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


# 逐行讲解：定义函数 main：运行 rollout -> advantage -> PPO backward -> optimizer 的训练闭环。
def main(args: argparse.Namespace) -> None:
    """运行 rollout -> advantage -> PPO backward -> optimizer 的训练闭环。"""
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

    # 逐行讲解：调用 RolloutConfig（RolloutConfig(group_size=args.group_size, max_steps=args.max_episode_steps, max_trajectory_tokens=args.max_…），赋给 rollout_config。
    rollout_config = RolloutConfig(
        group_size=args.group_size,
        max_steps=args.max_episode_steps,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
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
    # 逐行讲解：构造字典 {'clip_low_threshold': args.ppo_clip_low, 'clip_high_threshold': args.ppo_clip_high}，赋给 ppo_config。
    ppo_config = {
        "clip_low_threshold": args.ppo_clip_low,
        "clip_high_threshold": args.ppo_clip_high,
    }

    # 逐行讲解：调用 swanlab.init（swanlab.init(project=args.swanlab_project, name=args.run_name, mode=args.swanlab_mode, config=serializable_…）。
    swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config=serializable_config(args),
        tags=["pytrio", "alfworld", "grpo"],
    )
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：进入上下文管理器 tqdm(total=args.max_steps, desc='Training', unit='update')，退出时执行其收尾逻辑。
        with tqdm(total=args.max_steps, desc="Training", unit="update") as progress:
            # 逐行讲解：遍历 range(args.max_steps)，将每项绑定到 update_index。
            for update_index in range(args.max_steps):
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 started。
                started = perf_counter()
                # 逐行讲解：调用 take_batch（take_batch(games, update_index * args.games_per_batch, args.games_per_batch)），赋给 games_batch。
                games_batch = take_batch(
                    games,
                    update_index * args.games_per_batch,
                    args.games_per_batch,
                )
                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='sampler', refresh=True)）。
                progress.set_postfix(phase="sampler", refresh=True)
                # 逐行讲解：保存当前权重并创建对应采样客户端（training_client.save_weights_and_get_sampling_client()），赋给 sampling_client。
                sampling_client = training_client.save_weights_and_get_sampling_client()

                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='rollout', refresh=True)）。
                progress.set_postfix(phase="rollout", refresh=True)
                # 逐行讲解：进入上下文管理器 tqdm(total=len(games_batch) * args.group_size, desc=f'Update {update_index + 1} rollout', unit='trajectory'…，退出时执行其收尾逻辑。
                with tqdm(
                    total=len(games_batch) * args.group_size,
                    desc=f"Update {update_index + 1} rollout",
                    unit="trajectory",
                    position=1,
                    leave=False,
                ) as rollout_progress:
                    # 逐行讲解：调用 rollout_batch（rollout_batch(sampling_client, tokenizer, games_batch, rollout_config, progress_callback=rollout_progress.u…），赋给 batch。
                    batch = rollout_batch(
                        sampling_client,
                        tokenizer,
                        games_batch,
                        rollout_config,
                        progress_callback=rollout_progress.update,
                    )
                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='datums', refresh=True)）。
                progress.set_postfix(phase="datums", refresh=True)
                # 逐行讲解：调用 logical_step_count（logical_step_count(batch.trajectories)），赋给 logical_steps。
                logical_steps = logical_step_count(batch.trajectories)
                # 逐行讲解：调用 build_training_datums（build_training_datums(batch.trajectories, max_sequence_tokens=args.max_trajectory_tokens)），赋给 datums。
                datums = build_training_datums(
                    batch.trajectories,
                    max_sequence_tokens=args.max_trajectory_tokens,
                )
                # 逐行讲解：检查条件 len(datums) > MAX_TRAIN_BATCH_SIZE；成立时执行下方分支。
                if len(datums) > MAX_TRAIN_BATCH_SIZE:
                    # 逐行讲解：抛出异常：ValueError(f'训练 Datum 数 {len(datums)} 超过平台 batch 上限 {MAX_TRAIN_BATCH_SIZE}')。
                    raise ValueError(
                        f"训练 Datum 数 {len(datums)} 超过平台 batch 上限 "
                        f"{MAX_TRAIN_BATCH_SIZE}"
                    )

                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='backward', refresh=True)）。
                progress.set_postfix(phase="backward", refresh=True)
                # 逐行讲解：构造列表 []，赋给 trainer_results。
                trainer_results = []
                # 逐行讲解：检查条件 datums；成立时执行下方分支。
                if datums:
                    # 逐行讲解：调用 training_client.forward_backward([item.datum for item in datums], loss_fn='ppo', loss_fn_config=ppo_config)…（training_client.forward_backward([item.datum for item in datums], loss_fn='ppo', loss_fn_config=ppo_config)…），赋给 result。
                    result = training_client.forward_backward(
                        [item.datum for item in datums],
                        loss_fn="ppo",
                        loss_fn_config=ppo_config,
                    ).result()
                    # 逐行讲解：向列表末尾追加一个元素（trainer_results.append(result)）。
                    trainer_results.append(result)
                    # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='optimizer', refresh=True)）。
                    progress.set_postfix(phase="optimizer", refresh=True)
                    # 逐行讲解：等待任务完成并读取结果（training_client.optim_step(adam_params).result()）。
                    training_client.optim_step(adam_params).result()

                # 逐行讲解：检查条件 args.save_every > 0 and (update_index + 1) % args.save_every == 0；成立时执行下方分支。
                if args.save_every > 0 and (update_index + 1) % args.save_every == 0:
                    # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='checkpoint', refresh=True)）。
                    progress.set_postfix(phase="checkpoint", refresh=True)
                    # 逐行讲解：调用 save_checkpoint（save_checkpoint(training_client, f'{args.run_name}-update-{update_index + 1}')）。
                    save_checkpoint(
                        training_client,
                        f"{args.run_name}-update-{update_index + 1}",
                    )

                # 逐行讲解：调用 rollout_metrics（rollout_metrics(batch, datums, logical_steps)），赋给 metrics。
                metrics = rollout_metrics(
                    batch,
                    datums,
                    logical_steps,
                )
                # 逐行讲解：调用 metrics.update（metrics.update(merge_trainer_metrics(trainer_results))）。
                metrics.update(merge_trainer_metrics(trainer_results))
                # 逐行讲解：转换为浮点表示（float(not datums)），赋给 metrics['train/update_skipped']。
                metrics["train/update_skipped"] = float(not datums)
                # 逐行讲解：计算 perf_counter() - started，赋给 metrics['time/update_seconds']。
                metrics["time/update_seconds"] = perf_counter() - started
                # 逐行讲解：计算自然对数（swanlab.log(metrics, step=update_index)）。
                swanlab.log(metrics, step=update_index)

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
                    step_s=f"{metrics['time/update_seconds']:.1f}",
                    refresh=True,
                )
                # 逐行讲解：调用 tqdm.write（tqdm.write(f"update={update_index + 1}/{args.max_steps} success={metrics['reward/success_rate']:.3f} reward…）。
                tqdm.write(
                    f"update={update_index + 1}/{args.max_steps} "
                    f"success={metrics['reward/success_rate']:.3f} "
                    f"reward={metrics['reward/mean']:.3f} "
                    f"valid_action={metrics['rollout/valid_action_rate']:.3f} "
                    f"mean_steps={metrics['rollout/steps_mean']:.2f} "
                    f"loss={loss_text}"
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
