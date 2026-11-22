"""用 PyTRIO 训练 Tournament-GRPO：淘汰赛奖励 + 格式奖励、组内标准化 advantage、作者的 PPO loss。

正式训练（仓库根目录）：
uv run python experiments/tournament-grpo/train.py \
    --max-steps 80 \
    --questions-per-batch 8 \
    --group-size 8 \
    --save-every 10 \
    --run-name tournament-grpo-qwen35-4b-80steps

单步冒烟测试（1 题 × 8 条候选，28 场比赛）：
uv run python experiments/tournament-grpo/train.py \
    --max-steps 1 \
    --questions-per-batch 1 \
    --run-name tournament-grpo-smoke

从 checkpoint 续训（--max-steps 是累计步数）：
uv run python experiments/tournament-grpo/train.py \
    --max-steps 80 \
    --resume experiments/tournament-grpo/outputs/<run-name>/checkpoint-step-40.json \
    --run-name <run-name>
"""

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：从 collections 导入 Counter, defaultdict。
from collections import Counter, defaultdict
# 逐行讲解：从 dataclasses 导入 asdict。
from dataclasses import asdict
# 逐行讲解：从 datetime 导入 datetime。
from datetime import datetime
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 time 导入 perf_counter。
from time import perf_counter
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab
# 逐行讲解：导入模块：torch。
import torch
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm
# 逐行讲解：从 transformers 导入 AutoTokenizer。
from transformers import AutoTokenizer

# 逐行讲解：从 data 导入 load_examples。
from data import load_examples
# 逐行讲解：从 judge 导入 JudgeClient。
from judge import JudgeClient
# 逐行讲解：从 protocol 导入 TOOL_BUDGET_NOTICE, transcript_text。
from protocol import TOOL_BUDGET_NOTICE, transcript_text
# 逐行讲解：从 reward 导入 score_group。
from reward import score_group
# 逐行讲解：从 rollout 导入 RolloutConfig, Trajectory, rollout_batch, trajectory_record。
from rollout import RolloutConfig, Trajectory, rollout_batch, trajectory_record
# 逐行讲解：从 search 导入 ResearchTools。
from search import ResearchTools


# 逐行讲解：定义函数 assign_rewards：逐题跑淘汰赛，再按作者的组内标准化计算 advantage（样本标准差 ddof=1，eps=1e-6）。
def assign_rewards(
    trajectories: list[Trajectory], tokenizer: Any, judge: JudgeClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """逐题跑淘汰赛，再按作者的组内标准化计算 advantage（样本标准差 ddof=1，eps=1e-6）。

    返回与 trajectories 对齐的奖励明细，以及每道题一条的 judge 日志。
    """
    # 逐行讲解：调用 defaultdict（defaultdict(list)），赋给 groups。
    groups = defaultdict(list)
    # 逐行讲解：遍历 enumerate(trajectories)，将每项绑定到 (index, trajectory)。
    for index, trajectory in enumerate(trajectories):
        # 逐行讲解：向列表末尾追加一个元素（groups[trajectory.question_index].append(index)）。
        groups[trajectory.question_index].append(index)
    # 逐行讲解：构造元组 ([{} for _ in trajectories], [])，赋给 (details, judge_logs)。
    details, judge_logs = [{} for _ in trajectories], []
    # 逐行讲解：遍历 groups.items()，将每项绑定到 (question_index, indices)。
    for question_index, indices in groups.items():
        # 逐行讲解：通过推导式逐项生成结果：[trajectories[index] for index in indices]，赋给 group。
        group = [trajectories[index] for index in indices]
        # 逐行讲解：通过推导式逐项生成结果：[transcript_text(tokenizer, t.prompt_tokens, t.response_tokens) for t in group]，赋给 solutions。
        solutions = [transcript_text(tokenizer, t.prompt_tokens, t.response_tokens) for t in group]
        # 逐行讲解：调用 score_group（score_group(group[0].example, solutions, judge)），赋给 (results, log)。
        results, log = score_group(group[0].example, solutions, judge)
        # 逐行讲解：调用 torch.tensor（torch.tensor([item['score'] for item in results])），赋给 rewards。
        rewards = torch.tensor([item["score"] for item in results])
        # 逐行讲解：计算 (rewards - rewards.mean()) / (rewards.std(correction=1) + 1e-06)，赋给 advantages。
        advantages = (rewards - rewards.mean()) / (rewards.std(correction=1) + 1e-6)
        # 逐行讲解：遍历 zip(indices, group, results, advantages, strict=True)，将每项绑定到 (index, trajectory, result, advantage)。
        for index, trajectory, result, advantage in zip(indices, group, results, advantages, strict=True):
            # 逐行讲解：构造元组 (result['score'], float(advantage))，赋给 (trajectory.reward, trajectory.advantage)。
            trajectory.reward, trajectory.advantage = result["score"], float(advantage)
            # 逐行讲解：取值 result，赋给 details[index]。
            details[index] = result
        # 逐行讲解：向列表末尾追加一个元素（judge_logs.append({'question_index': question_index, **log, 'advantages': advantages.tolist()})）。
        judge_logs.append({"question_index": question_index, **log, "advantages": advantages.tolist()})
    # 逐行讲解：返回结果：(details, judge_logs)。
    return details, judge_logs


# 逐行讲解：定义函数 build_datum：把 prompt + response 右移一位作为训练样本；mask 和 advantage 留在本地 loss 里使用。
def build_datum(trajectory: Trajectory) -> trio.Datum:
    """把 prompt + response 右移一位作为训练样本；mask 和 advantage 留在本地 loss 里使用。

    右移后第 i 个位置预测 tokens[i + 1]，所以 response 的 logprob 从 len(prompt) - 1 开始。
    """
    # 逐行讲解：计算 trajectory.prompt_tokens + trajectory.response_tokens，赋给 tokens。
    tokens = trajectory.prompt_tokens + trajectory.response_tokens
    # 逐行讲解：返回结果：trio.Datum(model_input=trio.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs={'target_tokens': np.asarray(…。
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": np.asarray(tokens[1:], dtype=np.int64)},
    )


# 逐行讲解：定义函数 ppo_objective：作者的 vanilla PPO：token 平均、clip 0.2、dual clip 3，不加 KL 和熵项。
def ppo_objective(
    current: torch.Tensor, old: torch.Tensor, advantages: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """作者的 vanilla PPO：token 平均、clip 0.2、dual clip 3，不加 KL 和熵项。"""
    # 逐行讲解：将数值限制在指定区间（(current - old).clamp(-20.0, 20.0)），赋给 log_ratio。
    log_ratio = (current - old).clamp(-20.0, 20.0)
    # 逐行讲解：通过指数运算将对数值转换回原尺度（log_ratio.exp()），赋给 ratio。
    ratio = log_ratio.exp()
    # 逐行讲解：计算 -advantages * ratio，赋给 losses1。
    losses1 = -advantages * ratio
    # 逐行讲解：计算 -advantages * ratio.clamp(0.8, 1.2)，赋给 losses2。
    losses2 = -advantages * ratio.clamp(0.8, 1.2)
    # 逐行讲解：调用 torch.maximum（torch.maximum(losses1, losses2)），赋给 clipped。
    clipped = torch.maximum(losses1, losses2)
    # 逐行讲解：计算 -advantages * 3.0，赋给 dual_bound。
    dual_bound = -advantages * 3.0
    # 逐行讲解：调用 torch.where（torch.where(advantages < 0, torch.minimum(dual_bound, clipped), clipped)），赋给 losses。
    losses = torch.where(advantages < 0, torch.minimum(dual_bound, clipped), clipped)
    # 逐行讲解：计算均值（losses.mean()），赋给 loss。
    loss = losses.mean()
    # 逐行讲解：返回结果：(loss, {'ppo/loss': float(loss.detach()), 'ppo/clipfrac': float((losses2 > losses1).float().mean()), 'ppo/c…。
    return loss, {
        "ppo/loss": float(loss.detach()),
        "ppo/clipfrac": float((losses2 > losses1).float().mean()),
        "ppo/clipfrac_lower": float(((clipped > dual_bound) & (advantages < 0)).float().mean()),
        "ppo/kl": float(-log_ratio.detach().mean()),
    }


# 逐行讲解：定义函数 make_ppo_loss_fn：每批只更新一次：复用训练端前向的 detached logprob 作为 old，采样 logprob 只作诊断。
def make_ppo_loss_fn(trajectories: list[Trajectory]):
    """每批只更新一次：复用训练端前向的 detached logprob 作为 old，采样 logprob 只作诊断。

    若改成同批多次更新，必须在首次更新前固定 old，不能在每次回调里重新取值。
    """
    # 逐行讲解：通过推导式逐项生成结果：[(len(t.prompt_tokens) - 1, t.response_mask, t.logprobs, t.advantage) for t in trajectories]，赋给 metas。
    metas = [
        (len(t.prompt_tokens) - 1, t.response_mask, t.logprobs, t.advantage)
        for t in trajectories
    ]

    # 逐行讲解：定义函数 loss_fn：logprobs 是训练端前向得到的逐 token logprob，顺序与 data 一致。
    def loss_fn(data: list[trio.Datum], logprobs: list[torch.Tensor]):
        """logprobs 是训练端前向得到的逐 token logprob，顺序与 data 一致。"""
        # 逐行讲解：构造元组 ([], [], [])，赋给 (current_values, sampling_values, advantage_values)。
        current_values, sampling_values, advantage_values = [], [], []
        # 逐行讲解：遍历 zip(logprobs, metas, strict=True)，将每项绑定到 (values, (start, mask, sampling, advantage))。
        for values, (start, mask, sampling, advantage) in zip(logprobs, metas, strict=True):
            # 逐行讲解：调整张量形状（values.float().reshape(-1)），赋给 current。
            current = values.float().reshape(-1)
            # 逐行讲解：调用 torch.tensor（torch.tensor(mask, dtype=torch.bool, device=current.device)），赋给 action_mask。
            action_mask = torch.tensor(mask, dtype=torch.bool, device=current.device)
            # 逐行讲解：按索引或切片读取 current[start:][action_mask]，赋给 actions。
            actions = current[start:][action_mask]
            # 逐行讲解：向列表末尾追加一个元素（current_values.append(actions)）。
            current_values.append(actions)
            # 逐行讲解：向列表末尾追加一个元素（sampling_values.append(torch.tensor(sampling, device=current.device)[action_mask])）。
            sampling_values.append(torch.tensor(sampling, device=current.device)[action_mask])
            # 逐行讲解：向列表末尾追加一个元素（advantage_values.append(torch.full_like(actions, advantage))）。
            advantage_values.append(torch.full_like(actions, advantage))
        # 逐行讲解：拼接张量（torch.cat(current_values)），赋给 current。
        current = torch.cat(current_values)
        # 逐行讲解：调用 ppo_objective（ppo_objective(current, current.detach(), torch.cat(advantage_values))），赋给 (loss, metrics)。
        loss, metrics = ppo_objective(current, current.detach(), torch.cat(advantage_values))
        # 逐行讲解：转换为浮点表示（float((current.detach() - torch.cat(sampling_values)).abs().mean())），赋给 metrics['ppo/sampling_logprob_absdiff']。
        metrics["ppo/sampling_logprob_absdiff"] = float((current.detach() - torch.cat(sampling_values)).abs().mean())
        # 逐行讲解：转换为浮点表示（float(current.numel())），赋给 metrics['ppo/action_tokens']。
        metrics["ppo/action_tokens"] = float(current.numel())
        # 逐行讲解：返回结果：(loss, metrics)。
        return loss, metrics

    # 逐行讲解：返回结果：loss_fn。
    return loss_fn


# 逐行讲解：定义函数 update_policy：每个 batch 只更新一次（1 个 epoch）；advantage 为 0 的组仍计入 token 平均。
def update_policy(trainer: Any, trajectories: list[Trajectory], adam: trio.AdamParams) -> dict[str, float]:
    """每个 batch 只更新一次（1 个 epoch）；advantage 为 0 的组仍计入 token 平均。"""
    # 逐行讲解：通过推导式逐项生成结果：[build_datum(t) for t in trajectories]，赋给 datums。
    datums = [build_datum(t) for t in trajectories]
    # 逐行讲解：调用 make_ppo_loss_fn（make_ppo_loss_fn(trajectories)），赋给 loss_fn。
    loss_fn = make_ppo_loss_fn(trajectories)
    # 逐行讲解：等待任务完成并读取结果（trainer.forward_backward_custom(datums, loss_fn).result()），赋给 result。
    result = trainer.forward_backward_custom(datums, loss_fn).result()
    # 逐行讲解：等待任务完成并读取结果（trainer.optim_step(adam).result()），赋给 optimizer。
    optimizer = trainer.optim_step(adam).result()
    # 逐行讲解：返回结果：{**{f'trainer/{key}': value for key, value in result.metrics.items()}, **{f'optimizer/{key}': value for key…。
    return {**{f"trainer/{key}": value for key, value in result.metrics.items()},
            **{f"optimizer/{key}": value for key, value in optimizer.metrics.items()}}


# 逐行讲解：定义函数 mean：封装下方逻辑，供调用方复用。
def mean(values: list[float]) -> float:
    # 逐行讲解：返回结果：sum(values) / len(values)。
    return sum(values) / len(values)


# 逐行讲解：定义函数 rollout_metrics：汇总奖励、judge、rollout 和工具指标，写入 train.jsonl 和 SwanLab。
def rollout_metrics(
    trajectories: list[Trajectory], details: list[dict[str, Any]], judge_logs: list[dict[str, Any]],
) -> dict[str, float]:
    """汇总奖励、judge、rollout 和工具指标，写入 train.jsonl 和 SwanLab。"""
    # 逐行讲解：通过推导式逐项生成结果：[match for log in judge_logs for match in log['matches']]，赋给 matches。
    matches = [match for log in judge_logs for match in log["matches"]]
    # 逐行讲解：通过推导式逐项生成结果：[match['output']['usage'] for match in matches if isinstance(match['output'], dict)]，赋给 usage。
    usage = [match["output"]["usage"] for match in matches if isinstance(match["output"], dict)]
    # 逐行讲解：求和（sum((len(t.tool_calls) for t in trajectories))），赋给 calls。
    calls = sum(len(t.tool_calls) for t in trajectories)
    # 逐行讲解：构造字典 {'reward/mean': mean([t.reward for t in trajectories]), 'reward/tournament': mean([d['tournament_reward'] f…，赋给 metrics。
    metrics = {
        "reward/mean": mean([t.reward for t in trajectories]),
        "reward/tournament": mean([d["tournament_reward"] for d in details]),
        "reward/format": mean([d["format_reward"] for d in details]),
        "reward/has_answer": mean([float(bool(d["rollout_answer"])) for d in details]),
        "reward/degenerate_group_rate": mean([float(all(a == 0 for a in log["advantages"])) for log in judge_logs]),
        "judge/comparisons": float(len(matches)),
        # 只有 Jev 返回 usage；chat 后端的原始输出是字符串，这两项为 0。
        "judge/input_tokens": float(sum(u.get("input_tokens", 0) for u in usage)),
        "judge/output_tokens": float(sum(u.get("output_tokens", 0) for u in usage)),
        "rollout/assistant_turns": mean([t.assistant_turns for t in trajectories]),
        "rollout/response_tokens": mean([len(t.response_tokens) for t in trajectories]),
        "rollout/search_calls": mean([sum(c.name == "google_search" for c in t.tool_calls) for t in trajectories]),
        "rollout/browse_calls": mean([sum(c.name == "browse_webpage" for c in t.tool_calls) for t in trajectories]),
        "tools/error_rate": sum(t.tool_errors for t in trajectories) / max(calls, 1),
        "train/action_tokens": float(sum(sum(t.response_mask) for t in trajectories)),
    }
    # 逐行讲解：调用 Counter（Counter((t.finish_reason for t in trajectories))），赋给 finishes。
    finishes = Counter(t.finish_reason for t in trajectories)
    # 逐行讲解：调用 metrics.update（metrics.update({f'rollout/finish_{reason}': finishes[reason] / len(trajectories) for reason in ('no_tool_ca…）。
    metrics.update({f"rollout/finish_{reason}": finishes[reason] / len(trajectories)
                    for reason in ("no_tool_call", "max_turns", "max_tool_calls", "max_response_tokens")})
    # 逐行讲解：调用 Counter（Counter((reason for detail in details for reason in detail['failure_reasons']))），赋给 failures。
    failures = Counter(reason for detail in details for reason in detail["failure_reasons"])
    # 逐行讲解：调用 metrics.update（metrics.update({f'format/{reason}': count / len(details) for reason, count in failures.items()})）。
    metrics.update({f"format/{reason}": count / len(details) for reason, count in failures.items()})
    # 逐行讲解：返回结果：metrics。
    return metrics


# 逐行讲解：定义函数 write_jsonl：封装下方逻辑，供调用方复用。
def write_jsonl(path: Path, records: list[dict[str, Any]], mode: str = "a") -> None:
    # 逐行讲解：进入上下文管理器 path.open(mode, encoding='utf-8')，退出时执行其收尾逻辑。
    with path.open(mode, encoding="utf-8") as file:
        # 逐行讲解：遍历 records，将每项绑定到 record。
        for record in records:
            # 逐行讲解：调用 file.write（file.write(json.dumps(record, ensure_ascii=False) + '\n')）。
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# 逐行讲解：定义函数 save_checkpoint：保存可续训的 state 和可采样的权重，路径写进 checkpoint-step-N.json。
def save_checkpoint(trainer: Any, output_dir: Path, step: int, config: dict[str, Any]) -> None:
    """保存可续训的 state 和可采样的权重，路径写进 checkpoint-step-N.json。"""
    # 逐行讲解：等待任务完成并读取结果（trainer.save_state(name=f'step-{step}-state').result()），赋给 state。
    state = trainer.save_state(name=f"step-{step}-state").result()
    # 逐行讲解：等待任务完成并读取结果（trainer.save_weights_for_sampler(name=f'step-{step}-weights').result()），赋给 weights。
    weights = trainer.save_weights_for_sampler(name=f"step-{step}-weights").result()
    # 逐行讲解：构造字典 {'step': step, 'state_path': state.path, 'sampler_path': weights.path, 'config': config}，赋给 checkpoint。
    checkpoint = {"step": step, "state_path": state.path, "sampler_path": weights.path, "config": config}
    # 逐行讲解：计算 output_dir / f'checkpoint-step-{step}.json'，赋给 path。
    path = output_dir / f"checkpoint-step-{step}.json"
    # 逐行讲解：将文本写入文件（path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')）。
    path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 逐行讲解：调用 tqdm.write（tqdm.write(f'Saved checkpoint: {path} ({weights.path})')）。
    tqdm.write(f"Saved checkpoint: {path} ({weights.path})")


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：取值 Path(__file__).resolve().parent，赋给 base。
    base = Path(__file__).resolve().parent
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-steps', type=int, required=True, help='累计训练步数')）。
    parser.add_argument("--max-steps", type=int, required=True, help="累计训练步数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data', type=Path, default=base / 'datasets/train.jsonl')）。
    parser.add_argument("--data", type=Path, default=base / "datasets/train.jsonl")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B')）。
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--questions-per-batch', type=int, default=8)）。
    parser.add_argument("--questions-per-batch", type=int, default=8)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--group-size', type=int, default=8)）。
    parser.add_argument("--group-size", type=int, default=8)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=32)）。
    parser.add_argument("--lora-rank", type=int, default=32)
    # 作者全参训练用 1e-6；本仓库的 LoRA 章节统一用 4e-5。
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every', type=int, default=10, help='每 N 步保存一次；最后一步总会保存')）。
    parser.add_argument("--save-every", type=int, default=10, help="每 N 步保存一次；最后一步总会保存")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--run-name', default='tournament-grpo-' + datetime.now().strftime('%Y%m%d-%H%M%S'))）。
    parser.add_argument("--run-name", default="tournament-grpo-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output-dir', type=Path, help='默认 experiments/tournament-grpo/outputs/<run-name>')）。
    parser.add_argument("--output-dir", type=Path, help="默认 experiments/tournament-grpo/outputs/<run-name>")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--resume', type=Path, help='checkpoint-step-N.json：恢复权重、优化器和数据位置')）。
    parser.add_argument("--resume", type=Path, help="checkpoint-step-N.json：恢复权重、优化器和数据位置")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='tournament-grpo')）。
    parser.add_argument("--swanlab-project", default="tournament-grpo")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=('online', 'local', 'offline', 'disabled'), default='online')）。
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline", "disabled"), default="online")
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()
    # 逐行讲解：取值 args.output_dir or base / 'outputs' / args.run_name，赋给 args.output_dir。
    args.output_dir = args.output_dir or base / "outputs" / args.run_name
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main(args: argparse.Namespace) -> None:
    # 逐行讲解：取值 Path(__file__).resolve().parent，赋给 base。
    base = Path(__file__).resolve().parent
    # judge 和工具在创建训练 client 前读取配置，缺 Key 时立即报错。
    judge = JudgeClient.from_env(base / ".env")
    # 逐行讲解：调用 ResearchTools.from_env（ResearchTools.from_env(base / '.env')），赋给 tools。
    tools = ResearchTools.from_env(base / ".env")
    # 逐行讲解：调用 load_examples（load_examples(args.data)），赋给 examples。
    examples = load_examples(args.data)
    # 作者 shuffle=False、drop_last=True：按文件顺序取完整 batch，一个 epoch 后从头开始。
    batches_per_epoch = len(examples) // args.questions_per_batch
    # 逐行讲解：按条件选择结果：json.loads(args.resume.read_text(encoding='utf-8')) if args.resume else None，赋给 checkpoint。
    checkpoint = json.loads(args.resume.read_text(encoding="utf-8")) if args.resume else None
    # 逐行讲解：按条件选择结果：checkpoint['step'] if checkpoint else 0，赋给 start_step。
    start_step = checkpoint["step"] if checkpoint else 0

    # 逐行讲解：调用 AutoTokenizer.from_pretrained（AutoTokenizer.from_pretrained(args.base_model)），赋给 tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service。
    service = trio.ServiceClient()
    # 逐行讲解：检查条件 checkpoint；成立时执行下方分支。
    if checkpoint:
        # state 同时包含 LoRA 权重和 AdamW 状态。
        trainer = service.create_training_client_from_state_with_optimizer(checkpoint["state_path"])
    else:
        # 逐行讲解：调用 service.create_lora_training_client（service.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank, seed=args.seed)），赋给 trainer。
        trainer = service.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank, seed=args.seed)
    # 逐行讲解：调用 trio.AdamParams（trio.AdamParams(learning_rate=args.learning_rate, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.01, gra…），赋给 adam。
    adam = trio.AdamParams(
        learning_rate=args.learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, grad_clip_norm=1.0,
    )
    # 写入 SwanLab 和 checkpoint，记录与作者不同的后端和实现选择。
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "judge_backend": judge.backend,
        "judge_model": judge.model,
        "judge_long_context": judge.long_context,
        "optimizer": adam.model_dump(),
        "jina_reader_base_url": tools.jina_reader_base_url,
        "rollout": asdict(RolloutConfig(group_size=args.group_size, seed=args.seed)),
        "search_count": tools.search_count,
        "ppo_old_logprobs": "custom_forward_detached",
        "judge_failure_policy": "raise_after_3_attempts",
        "tool_budget_notice": TOOL_BUDGET_NOTICE,
    }
    # 逐行讲解：创建目录（args.output_dir.mkdir(parents=True, exist_ok=True)）。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 swanlab.init（swanlab.init(project=args.swanlab_project, name=args.run_name, mode=args.swanlab_mode, config=config)）。
    swanlab.init(project=args.swanlab_project, name=args.run_name, mode=args.swanlab_mode, config=config)
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：进入上下文管理器 tqdm(total=args.max_steps, initial=start_step, desc='Training', unit='step')，退出时执行其收尾逻辑。
        with tqdm(total=args.max_steps, initial=start_step, desc="Training", unit="step") as progress:
            # 逐行讲解：遍历 range(start_step + 1, args.max_steps + 1)，将每项绑定到 step。
            for step in range(start_step + 1, args.max_steps + 1):
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 started。
                started = perf_counter()
                # 逐行讲解：计算 (step - 1) % batches_per_epoch * args.questions_per_batch，赋给 offset。
                offset = ((step - 1) % batches_per_epoch) * args.questions_per_batch
                # 逐行讲解：按索引或切片读取 examples[offset:offset + args.questions_per_batch]，赋给 batch。
                batch = examples[offset:offset + args.questions_per_batch]
                # 逐行讲解：调用 RolloutConfig（RolloutConfig(group_size=args.group_size, seed=args.seed + (step - 1) * args.questions_per_batch)），赋给 rollout_config。
                rollout_config = RolloutConfig(group_size=args.group_size, seed=args.seed + (step - 1) * args.questions_per_batch)

                # 1. 用当前权重采样：每道题 group_size 条多轮工具轨迹。
                progress.set_postfix(phase="rollout", refresh=True)
                # 逐行讲解：保存当前权重并创建对应采样客户端（trainer.save_weights_and_get_sampling_client()），赋给 sampler。
                sampler = trainer.save_weights_and_get_sampling_client()
                # 逐行讲解：进入上下文管理器 tqdm(total=len(batch) * args.group_size, desc=f'Step {step} rollout', leave=False)，退出时执行其收尾逻辑。
                with tqdm(total=len(batch) * args.group_size, desc=f"Step {step} rollout", leave=False) as sampling:
                    # 逐行讲解：调用 rollout_batch（rollout_batch(sampler, tokenizer, tools, batch, rollout_config, sampling.update)），赋给 trajectories。
                    trajectories = rollout_batch(sampler, tokenizer, tools, batch, rollout_config, sampling.update)
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 sampled。
                sampled = perf_counter()

                # 2. 同题淘汰赛 + 格式奖励 → 组内 advantage。每场比赛的输入和 judge 输出都写进 judge.jsonl。
                progress.set_postfix(phase="judge", refresh=True)
                # 逐行讲解：调用 assign_rewards（assign_rewards(trajectories, tokenizer, judge)），赋给 (details, judge_logs)。
                details, judge_logs = assign_rewards(trajectories, tokenizer, judge)
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 judged。
                judged = perf_counter()
                # 逐行讲解：调用 write_jsonl（write_jsonl(args.output_dir / 'judge.jsonl', [{'step': step, **log} for log in judge_logs])）。
                write_jsonl(args.output_dir / "judge.jsonl", [{"step": step, **log} for log in judge_logs])

                # 3. 一次 PPO 更新。
                progress.set_postfix(phase="update", refresh=True)
                # 逐行讲解：构造字典 {**rollout_metrics(trajectories, details, judge_logs), **update_policy(trainer, trajectories, adam)}，赋给 metrics。
                metrics = {**rollout_metrics(trajectories, details, judge_logs), **update_policy(trainer, trajectories, adam)}
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 updated。
                updated = perf_counter()

                # 4. 保存 checkpoint；第一步和每个 checkpoint 保存完整轨迹，用于复盘模型行为的变化。
                save = step % args.save_every == 0 or step == args.max_steps
                # 逐行讲解：检查条件 save；成立时执行下方分支。
                if save:
                    # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(phase='checkpoint', refresh=True)）。
                    progress.set_postfix(phase="checkpoint", refresh=True)
                    # 逐行讲解：调用 save_checkpoint（save_checkpoint(trainer, args.output_dir, step, config)）。
                    save_checkpoint(trainer, args.output_dir, step, config)
                # 逐行讲解：检查条件 step == start_step + 1 or save；成立时执行下方分支。
                if step == start_step + 1 or save:
                    # 逐行讲解：通过推导式逐项生成结果：[trajectory_record(trajectory, tokenizer, step=step, reward_details=detail, base_model=args.base_model, con…，赋给 records。
                    records = [
                        trajectory_record(trajectory, tokenizer, step=step, reward_details=detail,
                                          base_model=args.base_model, config=asdict(rollout_config))
                        for trajectory, detail in zip(trajectories, details, strict=True)
                    ]
                    # 逐行讲解：调用 write_jsonl（write_jsonl(args.output_dir / f'rollout-step-{step}.jsonl', records, mode='w')）。
                    write_jsonl(args.output_dir / f"rollout-step-{step}.jsonl", records, mode="w")

                # 逐行讲解：调用 metrics.update（metrics.update({'time/rollout_seconds': sampled - started, 'time/reward_seconds': judged - sampled, 'time/u…）。
                metrics.update({"time/rollout_seconds": sampled - started, "time/reward_seconds": judged - sampled,
                                "time/update_seconds": updated - judged, "time/step_seconds": perf_counter() - started})
                # 逐行讲解：调用 write_jsonl（write_jsonl(args.output_dir / 'train.jsonl', [{'step': step, 'metrics': metrics}])）。
                write_jsonl(args.output_dir / "train.jsonl", [{"step": step, "metrics": metrics}])
                # 逐行讲解：计算自然对数（swanlab.log(metrics, step=step)）。
                swanlab.log(metrics, step=step)
                # 逐行讲解：调用 progress.update（progress.update(1)）。
                progress.update(1)
                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(reward=f"{metrics['reward/mean']:.3f}", loss=f"{metrics['trainer/ppo/loss']:.4f}", ref…）。
                progress.set_postfix(reward=f"{metrics['reward/mean']:.3f}",
                                     loss=f"{metrics['trainer/ppo/loss']:.4f}", refresh=True)
    except KeyboardInterrupt:
        # 逐行讲解：调用 swanlab.finish（swanlab.finish(state='aborted')）。
        swanlab.finish(state="aborted")
        # 逐行讲解：抛出异常：重新抛出当前异常。
        raise
    except Exception as exc:
        # 逐行讲解：调用 swanlab.finish（swanlab.finish(state='crashed', error=str(exc))）。
        swanlab.finish(state="crashed", error=str(exc))
        # 逐行讲解：抛出异常：重新抛出当前异常。
        raise
    else:
        # 逐行讲解：调用 swanlab.finish（swanlab.finish()）。
        swanlab.finish()


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main(parse_args())）。
    main(parse_args())
