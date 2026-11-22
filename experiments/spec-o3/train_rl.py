r"""从 SFT state 权重开始，用多轮工具轨迹进行 GRPO 训练。

在 experiments/spec-o3/ 目录下运行，首次使用先执行 uv run trio login 和 uv run swanlab login：

uv run python train_rl.py --state-path '<PyTrio 权重控制台获取或训练终端打印的 state_path>'

uv run python train_rl.py \
    --state-path '<PyTrio 权重控制台获取或训练终端打印的 state_path>' \
    --epochs 1 \
    --batch-size 8 \
    --group-size 8 \
    --learning-rate 1e-6 \
    --save-every 50 \
    --swanlab-mode online
"""

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：导入模块：random。
import random
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 tempfile 导入 TemporaryDirectory。
from tempfile import TemporaryDirectory
# 逐行讲解：导入模块：time。
import time

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm
# 逐行讲解：从 transformers 导入 AutoImageProcessor, AutoTokenizer。
from transformers import AutoImageProcessor, AutoTokenizer

# 逐行讲解：从 protocol 导入 build_rl_datum, encode_messages。
from protocol import build_rl_datum, encode_messages
# 逐行讲解：从 rollout 导入 rollout。
from rollout import rollout


# 逐行讲解：定义异步函数 rollout_group：共享第一轮输入，采样出一组开头，再并发完成每条独立轨迹。
async def rollout_group(
    row,
    sampler,
    tokenizer,
    processor,
    output_dir,
    args,
    seed,
    progress,
):
    """共享第一轮输入，采样出一组开头，再并发完成每条独立轨迹。"""
    # 逐行讲解：等待异步操作 asyncio.to_thread(encode_messages, row['messages'], row['images'], tokenizer, processor, generation=True) 完成并取得结果，赋给 (prompt, _, _)。
    prompt, _, _ = await asyncio.to_thread(
        encode_messages,
        row["messages"],
        row["images"],
        tokenizer,
        processor,
        generation=True,
    )
    # 逐行讲解：等待异步操作 sampler.sample_async(prompt=prompt, num_samples=args.group_size, sampling_params=trio.SamplingParams(max_to… 完成并取得结果，赋给 first_turn。
    first_turn = await sampler.sample_async(
        prompt=prompt,
        num_samples=args.group_size,
        sampling_params=trio.SamplingParams(
            max_tokens=min(args.max_tokens, args.max_seq_len - len(prompt)),
            temperature=args.temperature,
            top_p=args.top_p,
            stop=[tokenizer.convert_tokens_to_ids("<|im_end|>")],
            seed=seed,
        ),
    )

    # 逐行讲解：定义异步函数 run_trajectory：封装下方逻辑，供调用方复用。
    async def run_trajectory(index, sequence):
        # 逐行讲解：等待异步操作 rollout(row, sampler, tokenizer, processor, output_dir / str(index), max_turns=args.max_turns, max_tokens=a… 完成并取得结果，赋给 trajectory。
        trajectory = await rollout(
            row,
            sampler,
            tokenizer,
            processor,
            output_dir / str(index),
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            max_seq_len=args.max_seq_len,
            temperature=args.temperature,
            seed=seed + (index + 1) * args.max_turns,
            top_p=args.top_p,
            first_sequence=sequence,
        )
        # 每条完整轨迹结束就推进一次，同一 step 的所有 group 共用进度条。
        progress.update(1)
        # 逐行讲解：返回结果：trajectory。
        return trajectory

    # 逐行讲解：返回结果：await asyncio.gather(*(run_trajectory(index, sequence) for index, sequence in enumerate(first_turn.sequence…。
    return await asyncio.gather(
        *(run_trajectory(index, sequence) for index, sequence in enumerate(first_turn.sequences))
    )


# 逐行讲解：定义函数 trajectory_reward：奖励 = 答案正确得 1 分 − 格式错误罚分；不奖励工具调用次数。
def trajectory_reward(trajectory, format_penalty):
    """奖励 = 答案正确得 1 分 − 格式错误罚分；不奖励工具调用次数。"""
    # 单独读取最终答案，允许“答案正确但缺少规定标签”得到部分分数。
    # 只识别思考结束后、回答开头的 YES/NO，不从思考或工具参数中猜答案。
    body = trajectory["turns"][-1]["text"].partition("</think>")[2].strip()
    # 逐行讲解：调用 re.match（re.match('(?:<answer>\\s*)?(?:\\\\boxed\\{)?(YES|NO)\\b', body)），赋给 answer。
    answer = re.match(r"(?:<answer>\s*)?(?:\\boxed\{)?(YES|NO)\b", body)
    # 逐行讲解：取值 answer is not None and answer[1] == trajectory['label']，赋给 correct。
    correct = answer is not None and answer[1] == trajectory["label"]
    # 逐行讲解：计算 float(correct) - format_penalty * (not trajectory['format_ok'])，赋给 reward。
    reward = float(correct) - format_penalty * (not trajectory["format_ok"])
    # 逐行讲解：返回结果：(reward, correct)。
    return reward, correct


# 逐行讲解：定义函数 build_training_samples：在完整 group 内计算 advantage，汇总当前 batch 的有效训练样本。
def build_training_samples(groups, format_penalty):
    """在完整 group 内计算 advantage，汇总当前 batch 的有效训练样本。"""
    # 逐行讲解：构造列表 []，赋给 samples。
    samples = []
    # 逐行讲解：构造列表 []，赋给 rewards。
    rewards = []
    # 逐行讲解：构造列表 []，赋给 correct。
    correct = []
    # 逐行讲解：构造列表 []，赋给 trajectories。
    trajectories = []
    # 逐行讲解：取值 0，赋给 degenerate_groups。
    degenerate_groups = 0

    # 逐行讲解：遍历 groups，将每项绑定到 group。
    for group in groups:
        # 逐行讲解：通过推导式逐项生成结果：[trajectory_reward(item, format_penalty) for item in group]，赋给 scores。
        scores = [trajectory_reward(item, format_penalty) for item in group]
        # 逐行讲解：构造指定类型的 NumPy 数组（np.asarray([score[0] for score in scores], dtype=np.float32)），赋给 group_rewards。
        group_rewards = np.asarray([score[0] for score in scores], dtype=np.float32)
        # 逐行讲解：把一组元素依次加入列表（rewards.extend(group_rewards.tolist())）。
        rewards.extend(group_rewards.tolist())
        # 逐行讲解：把一组元素依次加入列表（correct.extend((score[1] for score in scores))）。
        correct.extend(score[1] for score in scores)
        # 逐行讲解：把一组元素依次加入列表（trajectories.extend(group)）。
        trajectories.extend(group)

        # 同组奖励完全相同，没有相对学习信号；仍计入全部 rollout 指标。
        if np.ptp(group_rewards) == 0:
            # 逐行讲解：更新 degenerate_groups：degenerate_groups += 1。
            degenerate_groups += 1
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue

        # 沿用作者 VeRL 的样本标准差（ddof=1）及 GRPO 的 epsilon。
        advantages = (
            (group_rewards - group_rewards.mean())
            / (group_rewards.std(ddof=1) + 1e-6)
        )
        # 逐行讲解：遍历 zip(group, advantages)，将每项绑定到 (trajectory, advantage)。
        for trajectory, advantage in zip(group, advantages):
            # 逐行讲解：求和（sum(trajectory['action_mask'])），赋给 tokens。
            tokens = sum(trajectory["action_mask"])
            # 没有生成动作的轨迹无法提供策略梯度，奖励仍参与组内归一化。
            if tokens:
                # 逐行讲解：向列表末尾追加一个元素（samples.append((build_rl_datum(trajectory, advantage), tokens))）。
                samples.append((build_rl_datum(trajectory, advantage), tokens))

    # 逐行讲解：调用 len（len(trajectories)），赋给 count。
    count = len(trajectories)
    # 逐行讲解：构造字典 {'train/reward': float(np.mean(rewards)), 'train/accuracy': sum(correct) / count, 'train/format_rate': sum(…，赋给 metrics。
    metrics = {
        "train/reward": float(np.mean(rewards)),
        "train/accuracy": sum(correct) / count,
        "train/format_rate": sum(item["format_ok"] for item in trajectories) / count,
        "train/degenerate_fraction": degenerate_groups / len(groups),
        "train/trajectories": len(samples),
        "train/assistant_tokens": sum(tokens for _, tokens in samples),
        "rollout/mean_turns": sum(len(item["turns"]) for item in trajectories) / count,
        "rollout/mean_generated_tokens": (
            sum(sum(item["action_mask"]) for item in trajectories) / count
        ),
        "rollout/tool_calls": sum(
            "observation" in turn
            for item in trajectories
            for turn in item["turns"]
        ),
    }
    # 逐行讲解：返回结果：(samples, metrics)。
    return samples, metrics


# 逐行讲解：定义异步函数 save_checkpoint：分别保存可继续训练的 state 和可独立评测的 sampler 权重。
async def save_checkpoint(client, name):
    """分别保存可继续训练的 state 和可独立评测的 sampler 权重。"""
    # 逐行讲解：等待异步操作 client.save_state_async(name=f'{name}-state') 完成并取得结果，赋给 state_future。
    state_future = await client.save_state_async(
        name=f"{name}-state",
    )
    # 逐行讲解：等待异步操作 client.save_weights_for_sampler_async(name=f'{name}-sampler') 完成并取得结果，赋给 sampler_future。
    sampler_future = await client.save_weights_for_sampler_async(
        name=f"{name}-sampler",
    )
    # 逐行讲解：等待异步操作 asyncio.gather(state_future, sampler_future) 完成并取得结果，赋给 (state, sampler)。
    state, sampler = await asyncio.gather(state_future, sampler_future)
    # 逐行讲解：调用 tqdm.write（tqdm.write(f'{name} state_path（继续训练）：{state.path}')）。
    tqdm.write(f"{name} state_path（继续训练）：{state.path}")
    # 逐行讲解：调用 tqdm.write（tqdm.write(f'{name} sampler_path（推理评测）：{sampler.path}')）。
    tqdm.write(f"{name} sampler_path（推理评测）：{sampler.path}")


# 逐行讲解：定义异步函数 main：封装下方逻辑，供调用方复用。
async def main(args):
    # 1. 加载 RL 训练集，从 state 恢复基模、LoRA 配置和训练权重。
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service。
    service = trio.ServiceClient()
    # 逐行讲解：检查条件 args.resume_optimizer；成立时执行下方分支。
    if args.resume_optimizer:
        # 逐行讲解：等待异步操作 service.create_training_client_from_state_with_optimizer_async(args.state_path) 完成并取得结果，赋给 client。
        client = await service.create_training_client_from_state_with_optimizer_async(
            args.state_path,
        )
    else:
        # 逐行讲解：等待异步操作 service.create_training_client_from_state_async(args.state_path) 完成并取得结果，赋给 client。
        client = await service.create_training_client_from_state_async(args.state_path)

    # 逐行讲解：调用 AutoTokenizer.from_pretrained（AutoTokenizer.from_pretrained(client.model_id, revision=args.model_revision)），赋给 tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(
        client.model_id,
        revision=args.model_revision,
    )
    # 逐行讲解：调用 AutoImageProcessor.from_pretrained（AutoImageProcessor.from_pretrained(client.model_id, revision=args.model_revision, backend='pil')），赋给 processor。
    processor = AutoImageProcessor.from_pretrained(
        client.model_id,
        revision=args.model_revision,
        backend="pil",
    )
    # 逐行讲解：调用 trio.AdamParams（trio.AdamParams(learning_rate=args.learning_rate)），赋给 optimizer。
    optimizer = trio.AdamParams(learning_rate=args.learning_rate)
    # 逐行讲解：调用 swanlab.init（swanlab.init(project=args.project_name, experiment_name=args.run_name, config={**vars(args), 'data': str(ar…）。
    swanlab.init(
        project=args.project_name,
        experiment_name=args.run_name,
        config={
            **vars(args),
            "data": str(args.data),
            "base_model": client.model_id,
        },
        mode=args.swanlab_mode,
    )

    # 逐行讲解：调用 random.Random（random.Random(args.seed)），赋给 rng。
    rng = random.Random(args.seed)
    # 逐行讲解：取值 0，赋给 step。
    step = 0
    # 逐行讲解：取值 0，赋给 updates。
    updates = 0
    # 逐行讲解：计算 (len(rows) + args.batch_size - 1) // args.batch_size，赋给 steps_per_epoch。
    steps_per_epoch = (len(rows) + args.batch_size - 1) // args.batch_size
    # 逐行讲解：遍历 range(1, args.epochs + 1)，将每项绑定到 epoch。
    for epoch in range(1, args.epochs + 1):
        # 逐行讲解：打乱数据顺序（rng.shuffle(rows)）。
        rng.shuffle(rows)
        # 逐行讲解：进入上下文管理器 tqdm(total=len(rows) * args.group_size, desc=f'RL epoch {epoch}', unit='traj', position=0), tqdm(total=0, unit='traj', position=1, leave=False)，退出时执行其收尾逻辑。
        with (
            tqdm(
                total=len(rows) * args.group_size,
                desc=f"RL epoch {epoch}",
                unit="traj",
                position=0,
            ) as progress,
            tqdm(total=0, unit="traj", position=1, leave=False) as step_progress,
        ):
            # 逐行讲解：遍历 range(0, len(rows), args.batch_size)，将每项绑定到 start。
            for start in range(0, len(rows), args.batch_size):
                # 逐行讲解：调用 time.monotonic（time.monotonic()），赋给 started。
                started = time.monotonic()
                # 逐行讲解：按索引或切片读取 rows[start:start + args.batch_size]，赋给 batch。
                batch = rows[start : start + args.batch_size]
                # 逐行讲解：计算 start // args.batch_size + 1，赋给 step_in_epoch。
                step_in_epoch = start // args.batch_size + 1
                # 逐行讲解：调用 step_progress.reset（step_progress.reset(total=len(batch) * args.group_size)）。
                step_progress.reset(total=len(batch) * args.group_size)
                # 逐行讲解：调用 step_progress.set_description（step_progress.set_description(f'Step {step_in_epoch}/{steps_per_epoch}')）。
                step_progress.set_description(f"Step {step_in_epoch}/{steps_per_epoch}")
                # 逐行讲解：调用 step_progress.set_postfix_str（step_progress.set_postfix_str('准备采样')）。
                step_progress.set_postfix_str("准备采样")

                # 2. 当前 batch 共用一个固定版本的 sampler，完成全部组后才更新。
                sampler = await client.save_weights_and_get_sampling_client_async()
                # 逐行讲解：调用 step_progress.set_postfix_str（step_progress.set_postfix_str('采样中')）。
                step_progress.set_postfix_str("采样中")
                # 逐行讲解：进入上下文管理器 TemporaryDirectory(prefix='spec-o3-rl-')，退出时执行其收尾逻辑。
                with TemporaryDirectory(prefix="spec-o3-rl-") as work_dir:
                    # 逐行讲解：等待异步操作 asyncio.gather(*(rollout_group(row, sampler, tokenizer, processor, Path(work_dir) / str(index), args, seed=… 完成并取得结果，赋给 groups。
                    groups = await asyncio.gather(
                        *(
                            rollout_group(
                                row,
                                sampler,
                                tokenizer,
                                processor,
                                Path(work_dir) / str(index),
                                args,
                                seed=args.seed + (
                                    (step * args.batch_size + index)
                                    * (args.group_size + 1) * args.max_turns
                                ),
                                progress=step_progress,
                            )
                            for index, row in enumerate(batch)
                        )
                    )
                    # 逐行讲解：调用 build_training_samples（build_training_samples(groups, args.format_penalty)），赋给 (samples, metrics)。
                    samples, metrics = build_training_samples(groups, args.format_penalty)

                # Datum 内已经包含图片字节，后续更新不再依赖临时图片文件。
                if samples:
                    # 逐行讲解：调用 step_progress.set_postfix_str（step_progress.set_postfix_str('更新参数')）。
                    step_progress.set_postfix_str("更新参数")
                    # 逐行讲解：通过推导式逐项生成结果：[datum for datum, _ in samples]，赋给 data。
                    data = [datum for datum, _ in samples]
                    # 逐行讲解：求和（sum((count for _, count in samples))），赋给 tokens。
                    tokens = sum(count for _, count in samples)

                    # 3. 按整个 batch 的 assistant token 数缩放 advantage。
                    for datum in data:
                        # 逐行讲解：调用 datum.loss_fn_inputs['advantages'].to_numpy（datum.loss_fn_inputs['advantages'].to_numpy()），赋给 advantages。
                        advantages = datum.loss_fn_inputs["advantages"].to_numpy()
                        # 逐行讲解：调用 trio.types.TensorData.from_numpy（trio.types.TensorData.from_numpy(advantages / tokens)），赋给 datum.loss_fn_inputs['advantages']。
                        datum.loss_fn_inputs["advantages"] = trio.types.TensorData.from_numpy(
                            advantages / tokens,
                        )

                    # 整批交给 PyTRIO 自动拆分、累积梯度，只更新一次参数。
                    backward = await client.forward_backward_async(
                        data,
                        loss_fn="ppo",
                        loss_fn_config={
                            "clip_low_threshold": 1 - args.clip_ratio,
                            "clip_high_threshold": 1 + args.clip_ratio,
                        },
                    )
                    # 逐行讲解：等待异步操作 client.optim_step_async(optimizer) 完成并取得结果，赋给 update。
                    update = await client.optim_step_async(optimizer)
                    # 逐行讲解：等待异步操作 backward 完成并取得结果，赋给 result。
                    result = await backward
                    # 逐行讲解：等待异步操作 update 完成并取得结果。
                    await update
                    # 逐行讲解：更新 updates：updates += 1。
                    updates += 1
                    # 逐行讲解：调用 step_progress.set_postfix_str（step_progress.set_postfix_str('完成')）。
                    step_progress.set_postfix_str("完成")
                    # Future 返回整个 batch 聚合后的服务端指标。
                    metrics.update({
                        f"trainer/{key}": value
                        for key, value in result.metrics.items()
                    })
                else:
                    # 逐行讲解：调用 step_progress.set_postfix_str（step_progress.set_postfix_str('无有效训练样本，跳过更新')）。
                    step_progress.set_postfix_str("无有效训练样本，跳过更新")

                # 4. 记录所有采样的表现，包括没有相对学习信号的 group。
                step += 1
                # 逐行讲解：取值 epoch，赋给 metrics['train/epoch']。
                metrics["train/epoch"] = epoch
                # 逐行讲解：取值 updates，赋给 metrics['train/updates']。
                metrics["train/updates"] = updates
                # 逐行讲解：计算 time.monotonic() - started，赋给 metrics['time/batch_seconds']。
                metrics["time/batch_seconds"] = time.monotonic() - started
                # 逐行讲解：计算自然对数（swanlab.log(metrics, step=step)）。
                swanlab.log(metrics, step=step)
                # 逐行讲解：调用 progress.update（progress.update(sum((len(group) for group in groups)))）。
                progress.update(sum(len(group) for group in groups))
                # 逐行讲解：调用 progress.set_postfix（progress.set_postfix(reward=f"{metrics['train/reward']:.3f}", accuracy=f"{metrics['train/accuracy']:.1%}", …）。
                progress.set_postfix(
                    reward=f"{metrics['train/reward']:.3f}",
                    accuracy=f"{metrics['train/accuracy']:.1%}",
                    updates=updates,
                )

                # 5. 按累计 rollout step 保存，此时本批的参数更新已完成。
                if step % args.save_every == 0:
                    # 逐行讲解：调用 step_progress.set_postfix_str（step_progress.set_postfix_str('保存权重')）。
                    step_progress.set_postfix_str("保存权重")
                    # 逐行讲解：等待异步操作 save_checkpoint(client, f'{args.run_name}-step-{step}') 完成并取得结果。
                    await save_checkpoint(client, f"{args.run_name}-step-{step}")
                    # 逐行讲解：调用 step_progress.set_postfix_str（step_progress.set_postfix_str('已保存权重')）。
                    step_progress.set_postfix_str("已保存权重")

        # 当前 epoch 的全部参数更新完成后，分别保存两份权重。
        await save_checkpoint(client, f"{args.run_name}-epoch-{epoch}")

    # 逐行讲解：调用 swanlab.finish（swanlab.finish()）。
    swanlab.finish()


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)），赋给 parser。
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 从已有权重开始训练；模型与 rank 由 state 决定。
    parser.add_argument(
        "--state-path",
        required=True,
        help="PyTrio 权重控制台获取或训练终端打印的 state_path",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--resume-optimizer', action='store_true', help='同时恢复 RL 优化器状态')）。
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="同时恢复 RL 优化器状态",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--model-revision', default='main')）。
    parser.add_argument("--model-revision", default="main")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'datasets/rl/bench_rl_train.jsonl')）。
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "datasets/rl/bench_rl_train.jsonl",
    )

    # batch-size 为每批题目数，group-size 为每题采样的轨迹数。
    parser.add_argument("--epochs", type=int, default=1)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=8)）。
    parser.add_argument("--batch-size", type=int, default=8)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--group-size', type=int, default=8)）。
    parser.add_argument("--group-size", type=int, default=8)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=1e-06)）。
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--clip-ratio', type=float, default=0.2)）。
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--format-penalty', type=float, default=0.2)）。
    parser.add_argument("--format-penalty", type=float, default=0.2)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every', type=int, default=100, help='每多少个累计 rollout step 保存 state 和 sampler，默认 …）。
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="每多少个累计 rollout step 保存 state 和 sampler，默认 100",
    )

    # 整条轨迹的轮数、上下文预算，以及每次生成的采样参数。
    parser.add_argument("--max-turns", type=int, default=8)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=2048)）。
    parser.add_argument("--max-tokens", type=int, default=2048)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-seq-len', type=int, default=16384)）。
    parser.add_argument("--max-seq-len", type=int, default=16384)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0)）。
    parser.add_argument("--temperature", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0)）。
    parser.add_argument("--top-p", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--project-name', default='agentic-rl-lab-spec-o3')）。
    parser.add_argument("--project-name", default="agentic-rl-lab-spec-o3")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--run-name', default='spec-o3-rl')）。
    parser.add_argument("--run-name", default="spec-o3-rl")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=['online', 'local', 'disabled'], default='online')）。
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "disabled"],
        default="online",
    )
    # 逐行讲解：调用 asyncio.run（asyncio.run(main(parser.parse_args()))）。
    asyncio.run(main(parser.parse_args()))
