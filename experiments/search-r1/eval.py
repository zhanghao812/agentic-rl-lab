"""用同一套搜索环境评测 base model 或 Search-R1 checkpoint。

在 experiments/search-r1 目录下评测 Base Model：
uv run python eval.py \
    --batch-size 16 \
    --output eval_result/eval_results_base.jsonl

评测训练后的 checkpoint：
uv run python eval.py \
    --batch-size 16 \
    --model-path 'trio://runxxxxxxxxxx' \
    --output eval_result/eval_results_rl_step_20.jsonl
"""

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：从 collections 导入 defaultdict。
from collections import defaultdict
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm

# 逐行讲解：从 data 导入 SearchExample, load_examples。
from data import SearchExample, load_examples
# 逐行讲解：从 rollout 导入 RolloutConfig, Trajectory, rollout_batch。
from rollout import RolloutConfig, Trajectory, rollout_batch
# 逐行讲解：从 search 导入 DEEPSEEK_SEARCH_ENDPOINT, SEARCH_BACKENDS, SearchClient, create_search_client, resolve_search_concurrency, resolve_search_timeout。
from search import (
    DEEPSEEK_SEARCH_ENDPOINT,
    SEARCH_BACKENDS,
    SearchClient,
    create_search_client,
    resolve_search_concurrency,
    resolve_search_timeout,
)


# 逐行讲解：计算 Path(__file__).resolve().parent / 'datasets' / 'dev.jsonl'，赋给 EVAL_DATA_PATH。
EVAL_DATA_PATH = Path(__file__).resolve().parent / "datasets" / "dev.jsonl"
# 逐行讲解：计算 Path(__file__).resolve().parent / 'eval_result'，赋给 EVAL_RESULT_DIR。
EVAL_RESULT_DIR = Path(__file__).resolve().parent / "eval_result"
# 逐行讲解：取值 70，赋给 EXPECTED_EVAL_SIZE。
EXPECTED_EVAL_SIZE = 70


# 逐行讲解：定义函数 mean：计算列表均值，空列表返回零。
def mean(values: list[float]) -> float:
    """计算列表均值，空列表返回零。"""
    # 逐行讲解：返回结果：sum(values) / len(values) if values else 0.0。
    return sum(values) / len(values) if values else 0.0


# 逐行讲解：定义函数 chunks：把评测问题切成适合并发采样的小批次。
def chunks(examples: list[SearchExample], size: int) -> list[list[SearchExample]]:
    """把评测问题切成适合并发采样的小批次。"""
    # 逐行讲解：返回结果：[examples[index:index + size] for index in range(0, len(examples), size)]。
    return [examples[index : index + size] for index in range(0, len(examples), size)]


# 逐行讲解：定义函数 trajectory_record：把一条评测轨迹转成可复查的 JSON 记录。
def trajectory_record(trajectory: Trajectory) -> dict[str, Any]:
    """把一条评测轨迹转成可复查的 JSON 记录。"""
    # 逐行讲解：返回结果：{'type': 'trajectory', 'id': trajectory.example.id, 'question': trajectory.example.question, 'answers': tra…。
    return {
        "type": "trajectory",
        "id": trajectory.example.id,
        "question": trajectory.example.question,
        "answers": trajectory.example.answers,
        "data_source": trajectory.example.data_source,
        "final_text": trajectory.final_text,
        "reward": trajectory.reward,
        "exact_match": trajectory.exact_match,
        "valid_format": trajectory.valid_format,
        "search_calls": trajectory.search_calls,
        "assistant_turns": len(trajectory.turns),
    }


# 逐行讲解：定义函数 evaluation_metrics：计算各数据源 EM、宏平均和搜索辅助指标。
def evaluation_metrics(
    trajectories: list[Trajectory], search_client: SearchClient
) -> dict[str, float]:
    """计算各数据源 EM、宏平均和搜索辅助指标。"""
    # 逐行讲解：声明 by_source 的类型为 dict[str, list[Trajectory]]；调用 defaultdict（defaultdict(list)）。
    by_source: dict[str, list[Trajectory]] = defaultdict(list)
    # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
    for trajectory in trajectories:
        # 逐行讲解：向列表末尾追加一个元素（by_source[trajectory.example.data_source].append(trajectory)）。
        by_source[trajectory.example.data_source].append(trajectory)
    # 逐行讲解：声明 metrics 的类型为 dict[str, float]；构造字典 {}。
    metrics: dict[str, float] = {}
    # 逐行讲解：声明 source_scores 的类型为 list[float]；构造列表 []。
    source_scores: list[float] = []
    # 逐行讲解：遍历 sorted(by_source.items())，将每项绑定到 (source, items)。
    for source, items in sorted(by_source.items()):
        # 逐行讲解：计算均值（mean([float(item.exact_match) for item in items])），赋给 score。
        score = mean([float(item.exact_match) for item in items])
        # 逐行讲解：取值 score，赋给 metrics[f'em/{source}']。
        metrics[f"em/{source}"] = score
        # 逐行讲解：向列表末尾追加一个元素（source_scores.append(score)）。
        source_scores.append(score)
    # 逐行讲解：调用 metrics.update（metrics.update({'em/macro': mean(source_scores), 'format/rate': mean([float(item.valid_format) for item in …）。
    metrics.update(
        {
            "em/macro": mean(source_scores),
            "format/rate": mean([float(item.valid_format) for item in trajectories]),
            "rollout/search_calls": mean(
                [float(item.search_calls) for item in trajectories]
            ),
            "rollout/no_search_rate": mean(
                [float(item.search_calls == 0) for item in trajectories]
            ),
            "rollout/turns": mean([float(len(item.turns)) for item in trajectories]),
        }
    )
    # 逐行讲解：调用 metrics.update（metrics.update(search_client.stats.metrics())）。
    metrics.update(search_client.stats.metrics())
    # 逐行讲解：返回结果：metrics。
    return metrics


# 逐行讲解：定义函数 summary_record：把本次评测配置和最终汇总指标转成 JSONL 的末行记录。
def summary_record(
    args: argparse.Namespace,
    trajectories: list[Trajectory],
    metrics: dict[str, float],
) -> dict[str, Any]:
    """把本次评测配置和最终汇总指标转成 JSONL 的末行记录。"""
    # 逐行讲解：返回结果：{'type': 'summary', 'base_model': args.base_model, 'model_path': args.model_path, 'search_backend': args.se…。
    return {
        "type": "summary",
        "base_model": args.base_model,
        "model_path": args.model_path,
        "search_backend": args.search_backend,
        "search_concurrency": args.search_concurrency,
        "search_timeout": args.search_timeout,
        "search_api": (
            DEEPSEEK_SEARCH_ENDPOINT if args.search_backend == "deepseek" else None
        ),
        "search_count": args.search_count if args.search_backend == "deepseek" else None,
        "search_need_content": False if args.search_backend == "deepseek" else None,
        "search_text_field": "snippet" if args.search_backend == "deepseek" else None,
        "evaluated_examples": len(trajectories),
        "metrics": metrics,
    }


# 逐行讲解：定义函数 parse_args：解析评测模型、输出和 rollout 参数。
def parse_args() -> argparse.Namespace:
    """解析评测模型、输出和 rollout 参数。"""
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--limit', type=int, default=0, help='最多评测多少道题；0 表示评测固定评测集中的全部 70 道题')）。
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多评测多少道题；0 表示评测固定评测集中的全部 70 道题",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=8, help='每批同时推进的评测问题数；模型采样和不同轨迹的搜索均可并发')）。
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="每批同时推进的评测问题数；模型采样和不同轨迹的搜索均可并发",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B', help='创建 PyTRIO sampling client 使用的基础模型')）。
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="创建 PyTRIO sampling client 使用的基础模型",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--model-path', help='PyTRIO sampler weights 路径；留空即评测未训练的 base model')）。
    parser.add_argument(
        "--model-path",
        help="PyTRIO sampler weights 路径；留空即评测未训练的 base model",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, default=EVAL_RESULT_DIR / 'eval_results.jsonl', help='逐题评测结果的 JS…）。
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_RESULT_DIR / "eval_results.jsonl",
        help="逐题评测结果的 JSONL 输出路径；已有文件会被覆盖",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-search-calls', type=int, default=4, help='每条评测轨迹最多调用搜索工具的次数')）。
    parser.add_argument(
        "--max-search-calls",
        type=int,
        default=4,
        help="每条评测轨迹最多调用搜索工具的次数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--search-backend', choices=SEARCH_BACKENDS, default='deepseek', help='搜索后端：deepseek 官方…）。
    parser.add_argument(
        "--search-backend",
        choices=SEARCH_BACKENDS,
        default="deepseek",
        help="搜索后端：deepseek 官方 Search API，wikipedia 免费免密钥",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--search-concurrency', type=int, default=None, help='不同轨迹之间的搜索并发；默认 deepseek=16、wikipe…）。
    parser.add_argument(
        "--search-concurrency",
        type=int,
        default=None,
        help="不同轨迹之间的搜索并发；默认 deepseek=16、wikipedia=3",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--search-count', type=int, choices=range(1, 51), default=3, metavar='1-50', help='Deep…）。
    parser.add_argument(
        "--search-count",
        type=int,
        choices=range(1, 51),
        default=3,
        metavar="1-50",
        help="DeepSeek Search 返回的网页条数；wikipedia 后端固定返回 Top 3",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--search-timeout', type=float, default=None, help='单次搜索的超时秒数；默认 deepseek=60、wikipedia=…）。
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=None,
        help="单次搜索的超时秒数；默认 deepseek=60、wikipedia=15",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-assistant-turns', type=int, default=6, help='每条评测轨迹最多生成的 assistant 回合数')）。
    parser.add_argument(
        "--max-assistant-turns",
        type=int,
        default=6,
        help="每条评测轨迹最多生成的 assistant 回合数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-trajectory-tokens', type=int, default=8192, help='整条评测轨迹允许使用的最大 token 数')）。
    parser.add_argument(
        "--max-trajectory-tokens",
        type=int,
        default=8192,
        help="整条评测轨迹允许使用的最大 token 数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-assistant-tokens', type=int, default=1024, help='单个 assistant 回合最多生成的 token 数')）。
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=1024,
        help="单个 assistant 回合最多生成的 token 数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tool-response-tokens', type=int, default=1024, help='单次搜索结果最多保留的 token 数')）。
    parser.add_argument(
        "--max-tool-response-tokens",
        type=int,
        default=1024,
        help="单次搜索结果最多保留的 token 数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=0.0, help='评测采样温度；默认 0.0，尽量使用确定性生成')）。
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="评测采样温度；默认 0.0，尽量使用确定性生成",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0, help='评测核采样的累积概率阈值')）。
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="评测核采样的累积概率阈值",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42, help='评测 rollout 使用的随机种子')）。
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="评测 rollout 使用的随机种子",
    )
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 main：在固定 70 题上运行 base 和 checkpoint 共用的 evaluator。
def main(args: argparse.Namespace) -> None:
    """在固定 70 题上运行 base 和 checkpoint 共用的 evaluator。"""
    # 逐行讲解：调用 resolve_search_concurrency（resolve_search_concurrency(args.search_backend, args.search_concurrency)），赋给 args.search_concurrency。
    args.search_concurrency = resolve_search_concurrency(
        args.search_backend, args.search_concurrency
    )
    # 逐行讲解：调用 resolve_search_timeout（resolve_search_timeout(args.search_backend, args.search_timeout)），赋给 args.search_timeout。
    args.search_timeout = resolve_search_timeout(
        args.search_backend, args.search_timeout
    )
    # 逐行讲解：调用 create_search_client（create_search_client(args.search_backend, Path(__file__).resolve().parent / '.env', count=args.search_count…），赋给 search_client。
    search_client = create_search_client(
        args.search_backend,
        Path(__file__).resolve().parent / ".env",
        count=args.search_count,
        timeout=args.search_timeout,
    )

    # 先校验磁盘上的固定评测集仍是完整 70 题，再按 limit 截取少量题做试跑。
    examples = load_examples(EVAL_DATA_PATH)
    # 逐行讲解：检查条件 len(examples) != EXPECTED_EVAL_SIZE；成立时执行下方分支。
    if len(examples) != EXPECTED_EVAL_SIZE:
        # 逐行讲解：抛出异常：ValueError(f'固定评测集应包含 {EXPECTED_EVAL_SIZE} 条，实际为 {len(examples)} 条；请重新运行 prepare_data.py')。
        raise ValueError(
            f"固定评测集应包含 {EXPECTED_EVAL_SIZE} 条，实际为 {len(examples)} 条；"
            "请重新运行 prepare_data.py"
        )
    # 逐行讲解：检查条件 args.limit > 0；成立时执行下方分支。
    if args.limit > 0:
        # 逐行讲解：按索引或切片读取 examples[:args.limit]，赋给 examples。
        examples = examples[: args.limit]
    # 逐行讲解：检查条件 not examples；成立时执行下方分支。
    if not examples:
        # 逐行讲解：抛出异常：ValueError('评测数据为空，请先运行 prepare_data.py')。
        raise ValueError("评测数据为空，请先运行 prepare_data.py")

    # model_path 留空时评测 base model；传入 sampler weights 时评测训练后模型。
    service_client = trio.ServiceClient()
    # 逐行讲解：调用 service_client.create_sampling_client（service_client.create_sampling_client(base_model=args.base_model, model_path=args.model_path)），赋给 sampling_client。
    sampling_client = service_client.create_sampling_client(
        base_model=args.base_model,
        model_path=args.model_path,
    )

    # 评测沿用训练时相同的 tokenizer、搜索客户端和多轮轨迹限制。
    tokenizer = sampling_client.get_tokenizer()
    # 逐行讲解：调用 RolloutConfig（RolloutConfig(group_size=1, max_search_calls=args.max_search_calls, max_assistant_turns=args.max_assistant_…），赋给 config。
    config = RolloutConfig(
        # 每道题只生成一条确定的评测轨迹，不需要训练阶段的 GRPO group。
        group_size=1,
        max_search_calls=args.max_search_calls,
        max_assistant_turns=args.max_assistant_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        max_tool_response_tokens=args.max_tool_response_tokens,
        search_concurrency=args.search_concurrency,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    # 自动创建 eval_result 等父目录；每次运行都会覆盖指定的 JSONL 文件。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：声明 trajectories 的类型为 list[Trajectory]；构造列表 []。
    trajectories: list[Trajectory] = []
    # 逐行讲解：调用 chunks（chunks(examples, args.batch_size)），赋给 batches。
    batches = chunks(examples, args.batch_size)
    # 逐行讲解：进入上下文管理器 args.output.open('w', encoding='utf-8')，退出时执行其收尾逻辑。
    with args.output.open("w", encoding="utf-8") as file:
        # 外层显示整个评测集进度，内层显示当前 batch 内已经结束的轨迹数。
        with tqdm(
            total=len(examples),
            desc="Total eval",
            unit="question",
            position=0,
        ) as total_progress:
            # 逐行讲解：遍历 enumerate(batches, 1)，将每项绑定到 (batch_index, batch)。
            for batch_index, batch in enumerate(batches, 1):
                # 逐行讲解：进入上下文管理器 tqdm(total=len(batch), desc=f'Batch {batch_index}/{len(batches)}', unit='question', position=1, leave=False)，退出时执行其收尾逻辑。
                with tqdm(
                    total=len(batch),
                    desc=f"Batch {batch_index}/{len(batches)}",
                    unit="question",
                    position=1,
                    leave=False,
                ) as batch_progress:

                    # 逐行讲解：定义函数 update_progress：同时推进当前 batch 和整个评测集的进度条。
                    def update_progress(completed: int) -> None:
                        """同时推进当前 batch 和整个评测集的进度条。"""
                        # 逐行讲解：调用 batch_progress.update（batch_progress.update(completed)）。
                        batch_progress.update(completed)
                        # 逐行讲解：调用 total_progress.update（total_progress.update(completed)）。
                        total_progress.update(completed)

                    # 逐行讲解：调用 rollout_batch（rollout_batch(sampling_client, tokenizer, search_client, batch, config, progress_callback=update_progress)），赋给 batch_trajectories。
                    batch_trajectories = rollout_batch(
                        sampling_client,
                        tokenizer,
                        search_client,
                        batch,
                        config,
                        progress_callback=update_progress,
                    )
                # 逐行讲解：把一组元素依次加入列表（trajectories.extend(batch_trajectories)）。
                trajectories.extend(batch_trajectories)

                # 每完成一批就立即落盘逐题结果，便于中途检查已完成的轨迹。
                for trajectory in batch_trajectories:
                    # 逐行讲解：调用 file.write（file.write(json.dumps(trajectory_record(trajectory), ensure_ascii=False) + '\n')）。
                    file.write(
                        json.dumps(trajectory_record(trajectory), ensure_ascii=False)
                        + "\n"
                    )
                # 逐行讲解：调用 total_progress.set_postfix（total_progress.set_postfix(completed=len(trajectories))）。
                total_progress.set_postfix(completed=len(trajectories))

        # 最后一行保存本次评测的模型信息和全部汇总指标，便于单文件复查。
        metrics = evaluation_metrics(trajectories, search_client)
        # 逐行讲解：调用 file.write（file.write(json.dumps(summary_record(args, trajectories, metrics), ensure_ascii=False) + '\n')）。
        file.write(
            json.dumps(
                summary_record(args, trajectories, metrics),
                ensure_ascii=False,
            )
            + "\n"
        )

    # 终端也打印同一份汇总指标，方便运行结束后立即查看 EM 等结果。
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    # 逐行讲解：调用 print（print(f'Saved predictions: {args.output}')）。
    print(f"Saved predictions: {args.output}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main(parse_args())）。
    main(parse_args())
