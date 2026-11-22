"""用完全相同的工具协议评估 ALFWorld base model 或 sampler weights。

从 experiments/alfworld 目录评估 base model：

uv run --extra alfworld python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --games-per-batch 16 \
    --output eval_results/base-qwen35-4b-eval.jsonl \
    --swanlab-mode disabled

评估训练 checkpoint（使用 train.py 输出的 ``Saved sampler weights`` 路径）：

uv run --extra alfworld python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://RUN_ID/sampler_weights/WEIGHTS_NAME' \
    --split all \
    --games-per-batch 16 \
    --output eval_results/checkpoint-eval.jsonl \
    --swanlab-mode disabled

"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：os。
import os
# 逐行讲解：从 collections 导入 defaultdict。
from collections import defaultdict
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any, Sequence。
from typing import Any, Sequence

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm

# 逐行讲解：从 data 导入 default_data_root, discover_games。
from data import default_data_root, discover_games
# 逐行讲解：从 rollout 导入 MAX_SEQUENCE_TOKENS, RolloutConfig, Trajectory, rollout_batch, trajectory_record。
from rollout import (
    MAX_SEQUENCE_TOKENS,
    RolloutConfig,
    Trajectory,
    rollout_batch,
    trajectory_record,
)


# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 BASE_DIR。
BASE_DIR = Path(__file__).resolve().parent


# 逐行讲解：定义函数 _path：封装下方逻辑，供调用方复用。
def _path(value: str | Path) -> Path:
    # 逐行讲解：返回结果：Path(os.path.expandvars(str(value))).expanduser().resolve()。
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


# 逐行讲解：定义函数 parse_args：解析模型、split、输出和 rollout 参数。
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析模型、split、输出和 rollout 参数。"""
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data-root', type=Path, default=default_data_root())）。
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--split', choices=['valid_seen', 'valid_unseen', 'all'], default='all')）。
    parser.add_argument(
        "--split",
        choices=["valid_seen", "valid_unseen", "all"],
        default="all",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--limit-per-split', type=int, default=0, help='每个 split 最多评估多少局；0 表示全部')）。
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="每个 split 最多评估多少局；0 表示全部",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--games-per-batch', type=int, default=8)）。
    parser.add_argument(
        "--games-per-batch",
        type=int,
        default=8,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B')）。
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--model-path', help='PyTRIO sampler weights 路径；留空即评估 base model')）。
    parser.add_argument(
        "--model-path",
        help="PyTRIO sampler weights 路径；留空即评估 base model",
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
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=0.01)）。
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0)）。
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, default=BASE_DIR / 'eval_results' / 'eval-results.jsonl')）。
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "eval_results" / "eval-results.jsonl",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--overwrite-output', action='store_true', help='显式允许覆盖已有评估 JSONL')）。
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="显式允许覆盖已有评估 JSONL",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-alfworld')）。
    parser.add_argument(
        "--swanlab-project",
        default="agentic-rl-lab-alfworld",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--run-name', default='alfworld-agent-rl-qwen35-4b-eval')）。
    parser.add_argument(
        "--run-name",
        default="alfworld-agent-rl-qwen35-4b-eval",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=['online', 'local', 'offline', 'disabled'], default='disabled')）。
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="disabled",
    )
    # 逐行讲解：解析实际命令行参数（parser.parse_args(argv)），赋给 args。
    args = parser.parse_args(argv)
    # 逐行讲解：调用 _path（_path(args.data_root)），赋给 args.data_root。
    args.data_root = _path(args.data_root)
    # 逐行讲解：调用 _path（_path(args.output)），赋给 args.output。
    args.output = _path(args.output)
    # 逐行讲解：检查条件 args.games_per_batch < 1 or args.max_episode_steps < 1 or args.max_trajectory_tokens < 1 or (args.max_assis…；成立时执行下方分支。
    if (
        args.games_per_batch < 1
        or args.max_episode_steps < 1
        or args.max_trajectory_tokens < 1
        or args.max_assistant_tokens < 1
    ):
        # 逐行讲解：抛出异常：ValueError('batch、episode step 和 token 上限必须大于等于 1')。
        raise ValueError("batch、episode step 和 token 上限必须大于等于 1")
    # 逐行讲解：检查条件 args.limit_per_split < 0；成立时执行下方分支。
    if args.limit_per_split < 0:
        # 逐行讲解：抛出异常：ValueError('limit_per_split 不能为负数')。
        raise ValueError("limit_per_split 不能为负数")
    # 逐行讲解：检查条件 args.max_trajectory_tokens > MAX_SEQUENCE_TOKENS；成立时执行下方分支。
    if args.max_trajectory_tokens > MAX_SEQUENCE_TOKENS:
        # 逐行讲解：抛出异常：ValueError(f'max_trajectory_tokens 不能超过平台单序列上限 {MAX_SEQUENCE_TOKENS}')。
        raise ValueError(
            f"max_trajectory_tokens 不能超过平台单序列上限 {MAX_SEQUENCE_TOKENS}"
        )
    # 逐行讲解：检查条件 args.temperature < 0.0 or not 0.0 < args.top_p <= 1.0；成立时执行下方分支。
    if args.temperature < 0.0 or not 0.0 < args.top_p <= 1.0:
        # 逐行讲解：抛出异常：ValueError('temperature 不能为负数，top_p 必须位于 (0, 1]')。
        raise ValueError("temperature 不能为负数，top_p 必须位于 (0, 1]")
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 chunks：封装下方逻辑，供调用方复用。
def chunks(values: list[Any], size: int) -> list[list[Any]]:
    # 逐行讲解：返回结果：[values[index:index + size] for index in range(0, len(values), size)]。
    return [values[index : index + size] for index in range(0, len(values), size)]


# 逐行讲解：定义函数 mean：封装下方逻辑，供调用方复用。
def mean(values: list[float]) -> float:
    # 逐行讲解：返回结果：sum(values) / len(values) if values else 0.0。
    return sum(values) / len(values) if values else 0.0


# 逐行讲解：定义函数 split_metrics：计算一个 split 的成功率和行为指标。
def split_metrics(trajectories: list[Trajectory], split: str) -> dict[str, float]:
    """计算一个 split 的成功率和行为指标。"""
    # 逐行讲解：通过推导式逐项生成结果：[step for trajectory in trajectories for step in trajectory.steps]，赋给 steps。
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    # 逐行讲解：构造字典 {f'eval/{split}/reward_mean': mean([trajectory.reward for trajectory in trajectories]), f'eval/{split}/succ…，赋给 metrics。
    metrics = {
        f"eval/{split}/reward_mean": mean(
            [trajectory.reward for trajectory in trajectories]
        ),
        f"eval/{split}/success_rate": mean(
            [float(trajectory.won) for trajectory in trajectories]
        ),
        f"eval/{split}/steps_mean": mean(
            [float(len(trajectory.steps)) for trajectory in trajectories]
        ),
        f"eval/{split}/truncated_rate": mean(
            [float(trajectory.truncated) for trajectory in trajectories]
        ),
        f"eval/{split}/valid_tool_call_rate": mean(
            [float(step.valid_format) for step in steps]
        ),
        f"eval/{split}/admissible_action_rate": mean(
            [float(step.admissible) for step in steps]
        ),
        f"eval/{split}/invalid_actions_mean": mean(
            [float(trajectory.invalid_action_count) for trajectory in trajectories]
        ),
        f"eval/{split}/games": float(len(trajectories)),
    }
    # 逐行讲解：声明 by_task 的类型为 dict[str, list[float]]；调用 defaultdict（defaultdict(list)）。
    by_task: dict[str, list[float]] = defaultdict(list)
    # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
    for trajectory in trajectories:
        # 逐行讲解：向列表末尾追加一个元素（by_task[trajectory.example.task_type].append(float(trajectory.won))）。
        by_task[trajectory.example.task_type].append(float(trajectory.won))
    # 逐行讲解：遍历 sorted(by_task.items())，将每项绑定到 (task_type, values)。
    for task_type, values in sorted(by_task.items()):
        # 逐行讲解：计算均值（mean(values)），赋给 metrics[f'eval/{split}/success_rate/{task_type}']。
        metrics[f"eval/{split}/success_rate/{task_type}"] = mean(values)
    # 逐行讲解：返回结果：metrics。
    return metrics


# 逐行讲解：定义函数 serializable_args：封装下方逻辑，供调用方复用。
def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    # 逐行讲解：返回结果：{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}。
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


# 逐行讲解：定义函数 main：依次评测 seen/unseen，并将逐轨迹结果与 summary 写入一个 JSONL。
def main(args: argparse.Namespace) -> None:
    """依次评测 seen/unseen，并将逐轨迹结果与 summary 写入一个 JSONL。"""
    # 逐行讲解：按条件选择结果：['valid_seen', 'valid_unseen'] if args.split == 'all' else [args.split]，赋给 selected_splits。
    selected_splits = (
        ["valid_seen", "valid_unseen"] if args.split == "all" else [args.split]
    )
    # 逐行讲解：构造字典 {}，赋给 games_by_split。
    games_by_split = {}
    # 逐行讲解：遍历 selected_splits，将每项绑定到 split。
    for split in selected_splits:
        # 逐行讲解：调用 discover_games（discover_games(args.data_root, split)），赋给 games。
        games = discover_games(args.data_root, split)
        # 逐行讲解：检查条件 args.limit_per_split > 0；成立时执行下方分支。
        if args.limit_per_split > 0:
            # 逐行讲解：按索引或切片读取 games[:args.limit_per_split]，赋给 games。
            games = games[: args.limit_per_split]
        # 逐行讲解：取值 games，赋给 games_by_split[split]。
        games_by_split[split] = games

    # 逐行讲解：创建目录（args.output.parent.mkdir(parents=True, exist_ok=True)）。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：检查条件 args.output.exists() and (not args.overwrite_output)；成立时执行下方分支。
    if args.output.exists() and not args.overwrite_output:
        # 逐行讲解：抛出异常：FileExistsError(f'评估输出已存在: {args.output}；如需覆盖请增加 --overwrite-output')。
        raise FileExistsError(
            f"评估输出已存在: {args.output}；如需覆盖请增加 --overwrite-output"
        )

    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：调用 service_client.create_sampling_client（service_client.create_sampling_client(base_model=args.base_model, model_path=args.model_path)），赋给 sampling_client。
    sampling_client = service_client.create_sampling_client(
        base_model=args.base_model,
        model_path=args.model_path,
    )
    # 逐行讲解：调用 sampling_client.get_tokenizer（sampling_client.get_tokenizer()），赋给 tokenizer。
    tokenizer = sampling_client.get_tokenizer()
    # 逐行讲解：调用 RolloutConfig（RolloutConfig(group_size=1, max_steps=args.max_episode_steps, max_trajectory_tokens=args.max_trajectory_tok…），赋给 rollout_config。
    rollout_config = RolloutConfig(
        group_size=1,
        max_steps=args.max_episode_steps,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        include_admissible_actions=args.include_admissible_actions,
        environment_asynchronous=args.environment_asynchronous,
    )

    # 逐行讲解：按条件选择结果：'w' if args.overwrite_output else 'x'，赋给 mode。
    mode = "w" if args.overwrite_output else "x"
    # 逐行讲解：调用 swanlab.init（swanlab.init(project=args.swanlab_project, name=args.run_name, mode=args.swanlab_mode, config=serializable_…）。
    swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config=serializable_args(args),
        tags=["alfworld", "evaluation", *selected_splits],
        job_type="eval",
    )
    # 逐行讲解：声明 all_metrics 的类型为 dict[str, float]；构造字典 {}。
    all_metrics: dict[str, float] = {}
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：进入上下文管理器 args.output.open(mode, encoding='utf-8')，退出时执行其收尾逻辑。
        with args.output.open(mode, encoding="utf-8") as file:
            # 逐行讲解：遍历 selected_splits，将每项绑定到 split。
            for split in selected_splits:
                # 逐行讲解：声明 split_trajectories 的类型为 list[Trajectory]；构造列表 []。
                split_trajectories: list[Trajectory] = []
                # 逐行讲解：按索引或切片读取 games_by_split[split]，赋给 games。
                games = games_by_split[split]
                # 逐行讲解：进入上下文管理器 tqdm(total=len(games), desc=f'Eval {split}', unit='game')，退出时执行其收尾逻辑。
                with tqdm(total=len(games), desc=f"Eval {split}", unit="game") as progress:
                    # 逐行讲解：遍历 chunks(games, args.games_per_batch)，将每项绑定到 games_batch。
                    for games_batch in chunks(games, args.games_per_batch):
                        # 逐行讲解：调用 rollout_batch（rollout_batch(sampling_client, tokenizer, games_batch, rollout_config, progress_callback=progress.update)），赋给 batch。
                        batch = rollout_batch(
                            sampling_client,
                            tokenizer,
                            games_batch,
                            rollout_config,
                            progress_callback=progress.update,
                        )
                        # 逐行讲解：把一组元素依次加入列表（split_trajectories.extend(batch.trajectories)）。
                        split_trajectories.extend(batch.trajectories)
                        # 逐行讲解：遍历 batch.trajectories，将每项绑定到 trajectory。
                        for trajectory in batch.trajectories:
                            # 逐行讲解：调用 trajectory_record（trajectory_record(trajectory)），赋给 record。
                            record = trajectory_record(trajectory)
                            # 逐行讲解：取值 'trajectory'，赋给 record['type']。
                            record["type"] = "trajectory"
                            # 逐行讲解：取值 split，赋给 record['evaluation_split']。
                            record["evaluation_split"] = split
                            # 逐行讲解：调用 file.write（file.write(json.dumps(record, ensure_ascii=False) + '\n')）。
                            file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        # 逐行讲解：调用 file.flush（file.flush()）。
                        file.flush()

                # 逐行讲解：调用 split_metrics（split_metrics(split_trajectories, split)），赋给 metrics。
                metrics = split_metrics(split_trajectories, split)
                # 逐行讲解：调用 all_metrics.update（all_metrics.update(metrics)）。
                all_metrics.update(metrics)

            # 逐行讲解：调用 file.write（file.write(json.dumps({'type': 'summary', 'base_model': args.base_model, 'model_path': args.model_path, 'me…）。
            file.write(
                json.dumps(
                    {
                        "type": "summary",
                        "base_model": args.base_model,
                        "model_path": args.model_path,
                        "metrics": all_metrics,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        # 逐行讲解：计算自然对数（swanlab.log(all_metrics, step=0)）。
        swanlab.log(all_metrics, step=0)
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

    # 逐行讲解：调用 print（print(json.dumps(all_metrics, ensure_ascii=False, indent=2, sort_keys=True))）。
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2, sort_keys=True))
    # 逐行讲解：调用 print（print(f'Saved evaluation: {args.output}')）。
    print(f"Saved evaluation: {args.output}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main(parse_args())）。
    main(parse_args())
