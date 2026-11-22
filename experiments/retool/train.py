"""使用 PyTRIO 和本地代码沙箱训练 Qwen3.5-4B ReTool。

小规模训练:
uv run python train.py \
    --max-steps 20 \
    --questions-per-batch 8 \
    --group-size 8 \
    --save-every 5 \
    --swanlab-mode online \
    --run-name retool-qwen35-4b-step20

小规模测试:
uv run python train.py \
    --max-steps 20 \
    --questions-per-batch 2 \
    --group-size 8 \
    --save-every 5 \
    --swanlab-mode disabled
"""

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：从 collections 导入 defaultdict。
from collections import defaultdict
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
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm

# 逐行讲解：从 data 导入 shuffled_examples, take_batch。
from data import shuffled_examples, take_batch
# 逐行讲解：从 rollout 导入 RolloutConfig, Trajectory, rollout_batch。
from rollout import RolloutConfig, Trajectory, rollout_batch
# 逐行讲解：从 sandbox 导入 LocalPythonSandbox。
from sandbox import LocalPythonSandbox


# 逐行讲解：取值 8192，赋给 MAX_TRAIN_CONTEXT_TOKENS。
MAX_TRAIN_CONTEXT_TOKENS = 8192  # 单个训练 Datum 允许的最大 token 数（已拍板 8k）。
# 逐行讲解：取值 32，赋给 MAX_MICRO_BATCH_ITEMS。
MAX_MICRO_BATCH_ITEMS = 32  # 单个 micro-batch 最多容纳的 Datum 数。
# 逐行讲解：取值 64000，赋给 MAX_MICRO_BATCH_PADDED_TOKENS。
MAX_MICRO_BATCH_PADDED_TOKENS = 64_000  # 单批 padding 矩形允许的最大 token 数。

# 与官方 recipe 对齐的 PPO-clip 阈值（ε_low=0.2 / ε_high=0.28 的绝对值形式）。
PPO_LOSS_CONFIG = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}


# 逐行讲解：定义类 TrainingDatum：把 PyTRIO Datum 和真实 token 数放在一起。
class TrainingDatum:
    """把 PyTRIO Datum 和真实 token 数放在一起。"""

    # 逐行讲解：定义函数 __init__：保存一个待装箱的训练样本。
    def __init__(self, datum: trio.Datum, num_tokens: int) -> None:
        """保存一个待装箱的训练样本。"""
        # 逐行讲解：取值 datum，赋给 self.datum。
        self.datum = datum
        # 逐行讲解：取值 num_tokens，赋给 self.num_tokens。
        self.num_tokens = num_tokens


# 逐行讲解：定义函数 build_datum：把一条前缀扩展的多轮轨迹合成一个右移对齐的训练 Datum。
def build_datum(trajectory: Trajectory) -> TrainingDatum:
    """把一条前缀扩展的多轮轨迹合成一个右移对齐的训练 Datum。

    observation（tool 返回）token 保留真实 target 但 advantage 置零，
    即官方要求的 interpreter feedback mask。
    """
    # 逐行讲解：检查条件 not trajectory.turns；成立时执行下方分支。
    if not trajectory.turns:
        # 逐行讲解：抛出异常：ValueError('不能用没有 assistant turn 的轨迹构造训练 Datum')。
        raise ValueError("不能用没有 assistant turn 的轨迹构造训练 Datum")

    # 逐行讲解：声明 full_tokens 的类型为 list[int]；构造列表 []。
    full_tokens: list[int] = []
    # 逐行讲解：声明 old_logprobs_by_token 的类型为 list[float]；构造列表 []。
    old_logprobs_by_token: list[float] = []
    # 逐行讲解：声明 advantages_by_token 的类型为 list[float]；构造列表 []。
    advantages_by_token: list[float] = []
    # 逐行讲解：取值 0，赋给 assistant_token_count。
    assistant_token_count = 0

    # 逐行讲解：遍历 enumerate(trajectory.turns)，将每项绑定到 (turn_index, turn)。
    for turn_index, turn in enumerate(trajectory.turns):
        # 逐行讲解：检查条件 len(turn.completion_tokens) != len(turn.logprobs)；成立时执行下方分支。
        if len(turn.completion_tokens) != len(turn.logprobs):
            # 逐行讲解：抛出异常：ValueError(f'第 {turn_index + 1} 个 assistant turn 的 token 与 logprob 长度不一致')。
            raise ValueError(
                f"第 {turn_index + 1} 个 assistant turn 的 token 与 logprob 长度不一致"
            )

        # 下一轮 prompt 应由已有轨迹加上新的 tool observation 和 assistant 前缀组成。
        if turn_index == 0:
            # 逐行讲解：取值 turn.prompt_tokens，赋给 delta_observation。
            delta_observation = turn.prompt_tokens
        # 逐行讲解：检查条件 turn.prompt_tokens[:len(full_tokens)] == full_tokens；成立时执行下方分支。
        elif turn.prompt_tokens[: len(full_tokens)] == full_tokens:
            # 逐行讲解：按索引或切片读取 turn.prompt_tokens[len(full_tokens):]，赋给 delta_observation。
            delta_observation = turn.prompt_tokens[len(full_tokens) :]
        else:
            # 逐行讲解：抛出异常：ValueError(f'第 {turn_index + 1} 个 assistant turn 的 prompt 不是已有轨迹的前缀扩展，无法安全对齐采样 logprob')。
            raise ValueError(
                f"第 {turn_index + 1} 个 assistant turn 的 prompt "
                "不是已有轨迹的前缀扩展，无法安全对齐采样 logprob"
            )

        # 逐行讲解：把一组元素依次加入列表（full_tokens.extend(delta_observation)）。
        full_tokens.extend(delta_observation)
        # 逐行讲解：把一组元素依次加入列表（full_tokens.extend(turn.completion_tokens)）。
        full_tokens.extend(turn.completion_tokens)
        # 逐行讲解：把一组元素依次加入列表（old_logprobs_by_token.extend([0.0] * len(delta_observation))）。
        old_logprobs_by_token.extend([0.0] * len(delta_observation))
        # 逐行讲解：把一组元素依次加入列表（old_logprobs_by_token.extend(turn.logprobs)）。
        old_logprobs_by_token.extend(turn.logprobs)
        # 逐行讲解：把一组元素依次加入列表（advantages_by_token.extend([0.0] * len(delta_observation))）。
        advantages_by_token.extend([0.0] * len(delta_observation))
        # 逐行讲解：把一组元素依次加入列表（advantages_by_token.extend([trajectory.advantage] * len(turn.completion_tokens))）。
        advantages_by_token.extend(
            [trajectory.advantage] * len(turn.completion_tokens)
        )
        # 逐行讲解：更新 assistant_token_count：assistant_token_count += len(turn.completion_tokens)。
        assistant_token_count += len(turn.completion_tokens)

    # 逐行讲解：检查条件 assistant_token_count == 0；成立时执行下方分支。
    if assistant_token_count == 0:
        # 逐行讲解：抛出异常：ValueError('不能用没有 assistant token 的轨迹构造训练 Datum')。
        raise ValueError("不能用没有 assistant token 的轨迹构造训练 Datum")
    # 逐行讲解：检查条件 not len(full_tokens) == len(old_logprobs_by_token) == len(advantages_by_token)；成立时执行下方分支。
    if not (
        len(full_tokens)
        == len(old_logprobs_by_token)
        == len(advantages_by_token)
    ):
        # 逐行讲解：抛出异常：ValueError('完整轨迹的 token、logprob 和 advantage 长度不一致')。
        raise ValueError("完整轨迹的 token、logprob 和 advantage 长度不一致")

    # 对完整序列统一右移；observation 保留真实 target token，但训练信号为零。
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
        # 逐行讲解：抛出异常：ValueError('Datum 的 input、target、logprobs 和 advantages 长度不一致')。
        raise ValueError("Datum 的 input、target、logprobs 和 advantages 长度不一致")
    # 逐行讲解：检查条件 len(input_tokens) > MAX_TRAIN_CONTEXT_TOKENS；成立时执行下方分支。
    if len(input_tokens) > MAX_TRAIN_CONTEXT_TOKENS:
        # 逐行讲解：抛出异常：ValueError(f'Datum 超过 {MAX_TRAIN_CONTEXT_TOKENS} token')。
        raise ValueError(f"Datum 超过 {MAX_TRAIN_CONTEXT_TOKENS} token")
    # 逐行讲解：调用 trio.Datum（trio.Datum(model_input=trio.ModelInput.from_ints(input_tokens), loss_fn_inputs={'target_tokens': np.asarray…），赋给 datum。
    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(old_logprobs, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
        },
    )
    # 逐行讲解：返回结果：TrainingDatum(datum, len(input_tokens))。
    return TrainingDatum(datum, len(input_tokens))


# 逐行讲解：定义函数 build_training_datums：为每条有训练信号的完整轨迹创建一个 Datum。
def build_training_datums(trajectories: list[Trajectory]) -> list[TrainingDatum]:
    """为每条有训练信号的完整轨迹创建一个 Datum。"""
    # 逐行讲解：声明 datums 的类型为 list[TrainingDatum]；构造列表 []。
    datums: list[TrainingDatum] = []
    # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
    for trajectory in trajectories:
        # 逐行讲解：检查条件 trajectory.advantage == 0.0；成立时执行下方分支。
        if trajectory.advantage == 0.0:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：检查条件 any((turn.completion_tokens for turn in trajectory.turns))；成立时执行下方分支。
        if any(turn.completion_tokens for turn in trajectory.turns):
            # 逐行讲解：向列表末尾追加一个元素（datums.append(build_datum(trajectory))）。
            datums.append(build_datum(trajectory))
    # 逐行讲解：返回结果：datums。
    return datums


# 逐行讲解：定义函数 datum_size：返回装箱排序使用的 Datum token 数。
def datum_size(item: TrainingDatum) -> int:
    """返回装箱排序使用的 Datum token 数。"""
    # 逐行讲解：返回结果：item.num_tokens。
    return item.num_tokens


# 逐行讲解：定义函数 datum_loss_token_count：统计一条 Datum 中 advantage 非零、实际参与 loss 的 token 数。
def datum_loss_token_count(item: TrainingDatum) -> int:
    """统计一条 Datum 中 advantage 非零、实际参与 loss 的 token 数。"""
    # 逐行讲解：调用 item.datum.loss_fn_inputs['advantages'].to_numpy（item.datum.loss_fn_inputs['advantages'].to_numpy()），赋给 advantages。
    advantages = item.datum.loss_fn_inputs["advantages"].to_numpy()
    # 逐行讲解：返回结果：int(np.count_nonzero(advantages))。
    return int(np.count_nonzero(advantages))


# 逐行讲解：定义函数 pack_micro_batches：按 padding 后的矩形面积使用 first-fit decreasing 动态装箱。
def pack_micro_batches(datums: list[TrainingDatum]) -> list[list[TrainingDatum]]:
    """按 padding 后的矩形面积使用 first-fit decreasing 动态装箱。"""
    # 逐行讲解：声明 batches 的类型为 list[list[TrainingDatum]]；构造列表 []。
    batches: list[list[TrainingDatum]] = []
    # 逐行讲解：声明 batch_max_tokens 的类型为 list[int]；构造列表 []。
    batch_max_tokens: list[int] = []
    # 逐行讲解：遍历 sorted(datums, key=datum_size, reverse=True)，将每项绑定到 item。
    for item in sorted(datums, key=datum_size, reverse=True):
        # 逐行讲解：遍历 enumerate(batches)，将每项绑定到 (index, batch)。
        for index, batch in enumerate(batches):
            # 逐行讲解：计算 len(batch) + 1，赋给 next_items。
            next_items = len(batch) + 1
            # 逐行讲解：调用 max（max(batch_max_tokens[index], item.num_tokens)），赋给 next_max_tokens。
            next_max_tokens = max(batch_max_tokens[index], item.num_tokens)
            # 逐行讲解：计算 next_items * next_max_tokens，赋给 next_padded_tokens。
            next_padded_tokens = next_items * next_max_tokens
            # 逐行讲解：取值 next_items <= MAX_MICRO_BATCH_ITEMS，赋给 fits_items。
            fits_items = next_items <= MAX_MICRO_BATCH_ITEMS
            # 逐行讲解：取值 next_padded_tokens <= MAX_MICRO_BATCH_PADDED_TOKENS，赋给 fits_tokens。
            fits_tokens = next_padded_tokens <= MAX_MICRO_BATCH_PADDED_TOKENS
            # 逐行讲解：检查条件 fits_items and fits_tokens；成立时执行下方分支。
            if fits_items and fits_tokens:
                # 逐行讲解：向列表末尾追加一个元素（batch.append(item)）。
                batch.append(item)
                # 逐行讲解：取值 next_max_tokens，赋给 batch_max_tokens[index]。
                batch_max_tokens[index] = next_max_tokens
                # 逐行讲解：结束当前循环。
                break
        else:
            # 逐行讲解：向列表末尾追加一个元素（batches.append([item])）。
            batches.append([item])
            # 逐行讲解：向列表末尾追加一个元素（batch_max_tokens.append(item.num_tokens)）。
            batch_max_tokens.append(item.num_tokens)
    # 逐行讲解：返回结果：batches。
    return batches


# 逐行讲解：定义函数 weight_micro_batch_for_global_mean：按样本占比缩放 advantage，使梯度累计等价于全局样本均值。
def weight_micro_batch_for_global_mean(
    micro_batch: list[TrainingDatum],
    total_samples: int,
) -> list[trio.Datum]:
    """按样本占比缩放 advantage，使梯度累计等价于全局样本均值。"""
    # 逐行讲解：检查条件 not micro_batch；成立时执行下方分支。
    if not micro_batch:
        # 逐行讲解：返回结果：[]。
        return []
    # 逐行讲解：检查条件 total_samples <= 0；成立时执行下方分支。
    if total_samples <= 0:
        # 逐行讲解：抛出异常：ValueError('全局样本数必须大于零')。
        raise ValueError("全局样本数必须大于零")
    # 逐行讲解：检查条件 len(micro_batch) > total_samples；成立时执行下方分支。
    if len(micro_batch) > total_samples:
        # 逐行讲解：抛出异常：ValueError('micro-batch 样本数不能超过全局样本数')。
        raise ValueError("micro-batch 样本数不能超过全局样本数")

    # 远端对每次 forward_backward 内的样本取 mean。多个大小不同的
    # micro-batch 直接累积会让小批次权重过大，因此将第 k 批乘以 n_k / N：
    # sum_k [n_k / N * mean(loss_k)] = mean(loss_global)。
    micro_batch_weight = np.float32(len(micro_batch) / total_samples)
    # 逐行讲解：声明 weighted_datums 的类型为 list[trio.Datum]；构造列表 []。
    weighted_datums: list[trio.Datum] = []
    # 逐行讲解：遍历 micro_batch，将每项绑定到 item。
    for item in micro_batch:
        # 逐行讲解：取值 item.datum.loss_fn_inputs，赋给 loss_inputs。
        loss_inputs = item.datum.loss_fn_inputs
        # 逐行讲解：向列表末尾追加一个元素（weighted_datums.append(trio.Datum(model_input=item.datum.model_input, loss_fn_inputs={'target_tokens': loss…）。
        weighted_datums.append(
            trio.Datum(
                model_input=item.datum.model_input,
                loss_fn_inputs={
                    "target_tokens": loss_inputs["target_tokens"].to_numpy(),
                    "logprobs": loss_inputs["logprobs"].to_numpy(),
                    "advantages": (
                        loss_inputs["advantages"].to_numpy() * micro_batch_weight
                    ),
                },
            )
        )
    # 逐行讲解：返回结果：weighted_datums。
    return weighted_datums


# 逐行讲解：定义函数 mean：计算列表均值，空列表返回零。
def mean(values: list[float]) -> float:
    """计算列表均值，空列表返回零。"""
    # 逐行讲解：返回结果：sum(values) / len(values) if values else 0.0。
    return sum(values) / len(values) if values else 0.0


# 逐行讲解：定义函数 degenerate_group_count：统计整组 advantage 都为零的问题数。
def degenerate_group_count(trajectories: list[Trajectory]) -> int:
    """统计整组 advantage 都为零的问题数。"""
    # 逐行讲解：声明 groups 的类型为 dict[int, list[float]]；调用 defaultdict（defaultdict(list)）。
    groups: dict[int, list[float]] = defaultdict(list)
    # 逐行讲解：遍历 trajectories，将每项绑定到 trajectory。
    for trajectory in trajectories:
        # 逐行讲解：向列表末尾追加一个元素（groups[trajectory.question_index].append(trajectory.advantage)）。
        groups[trajectory.question_index].append(trajectory.advantage)
    # 逐行讲解：返回结果：sum((all((advantage == 0.0 for advantage in values)) for values in groups.values()))。
    return sum(all(advantage == 0.0 for advantage in values) for values in groups.values())


# 逐行讲解：定义函数 rollout_metrics：汇总 reward、轨迹和动态 micro-batch 指标。
def rollout_metrics(
    trajectories: list[Trajectory],
    datums: list[TrainingDatum],
    micro_batches: list[list[TrainingDatum]],
    question_count: int,
) -> dict[str, float]:
    """汇总 reward、轨迹和动态 micro-batch 指标。"""
    # 逐行讲解：求和（sum(('<tool_call>' in turn.text for trajectory in trajectories for turn in trajectory.turns))），赋给 tool_attempts。
    tool_attempts = sum(
        "<tool_call>" in turn.text
        for trajectory in trajectories
        for turn in trajectory.turns
    )
    # 逐行讲解：求和（sum((trajectory.code_calls for trajectory in trajectories))），赋给 valid_tool_calls。
    valid_tool_calls = sum(trajectory.code_calls for trajectory in trajectories)
    # 逐行讲解：通过推导式逐项生成结果：[len(trajectory.turns[-1].prompt_tokens) + len(trajectory.turns[-1].completion_tokens) for trajectory in tr…，赋给 trajectory_lengths。
    trajectory_lengths = [
        len(trajectory.turns[-1].prompt_tokens)
        + len(trajectory.turns[-1].completion_tokens)
        for trajectory in trajectories
        if trajectory.turns
    ]
    # 逐行讲解：通过推导式逐项生成结果：[max((item.num_tokens for item in batch), default=0) * len(batch) for batch in micro_batches]，赋给 micro_batch_padded_tokens。
    micro_batch_padded_tokens = [
        max((item.num_tokens for item in batch), default=0) * len(batch)
        for batch in micro_batches
    ]
    # 逐行讲解：求和（sum((item.num_tokens for item in datums))），赋给 input_tokens。
    input_tokens = sum(item.num_tokens for item in datums)
    # 逐行讲解：求和（sum((datum_loss_token_count(item) for item in datums))），赋给 loss_tokens。
    loss_tokens = sum(datum_loss_token_count(item) for item in datums)
    # 逐行讲解：求和（sum(micro_batch_padded_tokens)），赋给 padded_tokens。
    padded_tokens = sum(micro_batch_padded_tokens)
    # 逐行讲解：返回结果：{'reward/mean': mean([trajectory.reward for trajectory in trajectories]), 'reward/correct': mean([float(tra…。
    return {
        "reward/mean": mean([trajectory.reward for trajectory in trajectories]),
        "reward/correct": mean([float(trajectory.correct) for trajectory in trajectories]),
        "reward/format": mean([float(trajectory.valid_format) for trajectory in trajectories]),
        "rollout/turns": mean([float(len(trajectory.turns)) for trajectory in trajectories]),
        "rollout/code_calls": mean(
            [float(trajectory.code_calls) for trajectory in trajectories]
        ),
        "rollout/trajectory_tokens": mean([float(value) for value in trajectory_lengths]),
        "rollout/valid_tool_call_rate": valid_tool_calls / max(tool_attempts, 1),
        "rollout/degenerate_group_rate": degenerate_group_count(trajectories)
        / max(question_count, 1),
        "train/datums_per_rollout_batch": float(len(datums)),
        "train/micro_batches_per_step": float(len(micro_batches)),
        "train/tokens_per_rollout_batch": float(input_tokens),
        "train/mean_datum_tokens": mean([float(item.num_tokens) for item in datums]),
        "train/loss_tokens_per_rollout_batch": float(loss_tokens),
        "train/padded_tokens_per_rollout_batch": float(padded_tokens),
        "train/max_micro_batch_padded_tokens": float(
            max(micro_batch_padded_tokens, default=0)
        ),
    }


# 逐行讲解：定义函数 merge_trainer_metrics：合并多个 micro-batch 指标；预加权 mean loss 需要求和。
def merge_trainer_metrics(results: list[Any]) -> dict[str, float]:
    """合并多个 micro-batch 指标；预加权 mean loss 需要求和。"""
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
        # 提交前的 advantage 已乘以 n_k / N，因此各 micro-batch 返回的
        # mean loss 相加才是整个 logical batch 的 global mean loss。
        if key in {"loss_mean", "loss/mean"}:
            # 逐行讲解：求和（sum(items)），赋给 merged[f'trainer/{key}']。
            merged[f"trainer/{key}"] = sum(items)
        else:
            # 逐行讲解：计算均值（mean(items)），赋给 merged[f'trainer/{key}']。
            merged[f"trainer/{key}"] = mean(items)
    # 逐行讲解：返回结果：merged。
    return merged


# 逐行讲解：定义函数 pick_mean_loss_metric：从 PyTRIO 指标中提取 mean loss，不混用总 loss。
def pick_mean_loss_metric(metrics: dict[str, float]) -> float | None:
    """从 PyTRIO 指标中提取 mean loss，不混用总 loss。"""
    # 逐行讲解：遍历 ('trainer/loss_mean', 'trainer/loss/mean')，将每项绑定到 key。
    for key in (
        "trainer/loss_mean",
        "trainer/loss/mean",
    ):
        # 逐行讲解：检查条件 key in metrics；成立时执行下方分支。
        if key in metrics:
            # 逐行讲解：返回结果：float(metrics[key])。
            return float(metrics[key])
    # 逐行讲解：返回结果：None。
    return None


# 逐行讲解：定义函数 serializable_config：把 Path 参数转换成 SwanLab 可记录的字符串。
def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    """把 Path 参数转换成 SwanLab 可记录的字符串。"""
    # 逐行讲解：返回结果：{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}。
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


# 逐行讲解：定义函数 save_checkpoint：同时保存断点 state（保险用）和推理 sampler weights（eval 用）。
def save_checkpoint(training_client: Any, name: str) -> None:
    """同时保存断点 state（保险用）和推理 sampler weights（eval 用）。"""
    # 逐行讲解：等待任务完成并读取结果（training_client.save_state(name=f'{name}-state').result()），赋给 state。
    state = training_client.save_state(name=f"{name}-state").result()
    # 逐行讲解：等待任务完成并读取结果（training_client.save_weights_for_sampler(name=f'{name}-weights').result()），赋给 weights。
    weights = training_client.save_weights_for_sampler(name=f"{name}-weights").result()
    # 逐行讲解：调用 print（print(f'Saved state: {state.path}')）。
    print(f"Saved state: {state.path}")
    # 逐行讲解：调用 print（print(f'Saved sampler weights: {weights.path}')）。
    print(f"Saved sampler weights: {weights.path}")


# 逐行讲解：定义函数 parse_args：解析训练超参数。
def parse_args() -> argparse.Namespace:
    """解析训练超参数。"""
    # 逐行讲解：取值 Path(__file__).resolve().parent，赋给 base_dir。
    base_dir = Path(__file__).resolve().parent
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-steps', type=int, required=True, help='最多执行多少个 GRPO 训练 step')）。
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help="最多执行多少个 GRPO 训练 step",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data', type=Path, default=base_dir / 'datasets' / 'train.jsonl', help='训练数据 JSONL 文件…）。
    parser.add_argument(
        "--data",
        type=Path,
        default=base_dir / "datasets" / "train.jsonl",
        help="训练数据 JSONL 文件路径",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-train-samples', type=int, default=0, help='最多使用多少条训练问题；0 表示使用全部数据')）。
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="最多使用多少条训练问题；0 表示使用全部数据",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B', help='创建 LoRA 训练客户端使用的基础模型')）。
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="创建 LoRA 训练客户端使用的基础模型",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=32, help='新建 LoRA 训练客户端时使用的 rank')）。
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="新建 LoRA 训练客户端时使用的 rank",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--questions-per-batch', type=int, default=8, help='每个训练 step 选取的问题数量')）。
    parser.add_argument(
        "--questions-per-batch",
        type=int,
        default=8,
        help="每个训练 step 选取的问题数量",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--group-size', type=int, default=8, help='同一道问题采样的轨迹数量，用于计算组内相对 advantage')）。
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="同一道问题采样的轨迹数量，用于计算组内相对 advantage",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-code-calls', type=int, default=4, help='每条轨迹最多调用代码解释器的次数')）。
    parser.add_argument(
        "--max-code-calls",
        type=int,
        default=4,
        help="每条轨迹最多调用代码解释器的次数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-assistant-turns', type=int, default=6, help='每条轨迹最多生成的 assistant 回合数')）。
    parser.add_argument(
        "--max-assistant-turns",
        type=int,
        default=6,
        help="每条轨迹最多生成的 assistant 回合数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-trajectory-tokens', type=int, default=8192, help='整条轨迹允许使用的最大 token 数')）。
    parser.add_argument(
        "--max-trajectory-tokens",
        type=int,
        default=8192,
        help="整条轨迹允许使用的最大 token 数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-assistant-tokens', type=int, default=1024, help='单个 assistant 回合最多生成的 token 数')）。
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=1024,
        help="单个 assistant 回合最多生成的 token 数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tool-response-tokens', type=int, default=512, help='单次代码执行结果最多保留的 token 数')）。
    parser.add_argument(
        "--max-tool-response-tokens",
        type=int,
        default=512,
        help="单次代码执行结果最多保留的 token 数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sandbox-timeout', type=float, default=30.0, help='单次代码执行的 wall-clock 超时秒数')）。
    parser.add_argument(
        "--sandbox-timeout",
        type=float,
        default=30.0,
        help="单次代码执行的 wall-clock 超时秒数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sandbox-workers', type=int, default=8, help='本地沙箱并发执行的进程数上限')）。
    parser.add_argument(
        "--sandbox-workers",
        type=int,
        default=8,
        help="本地沙箱并发执行的进程数上限",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0, help='rollout 采样温度；越高随机性越强')）。
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="rollout 采样温度；越高随机性越强",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0, help='rollout 核采样的累积概率阈值')）。
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="rollout 核采样的累积概率阈值",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=4e-05, help='Adam 优化器学习率')）。
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-5,
        help="Adam 优化器学习率",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta1', type=float, default=0.9, help='Adam 优化器的一阶动量系数')）。
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam 优化器的一阶动量系数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta2', type=float, default=0.95, help='Adam 优化器的二阶动量系数')）。
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="Adam 优化器的二阶动量系数",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every', type=int, default=25, help='每隔多少个 step 保存断点 state 和推理 sampler weights；0…）。
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help=(
            "每隔多少个 step 保存断点 state 和推理 sampler weights；"
            "0 表示只在训练结束时保存"
        ),
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42, help='数据打乱、LoRA 初始化和 rollout 采样使用的随机种子')）。
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="数据打乱、LoRA 初始化和 rollout 采样使用的随机种子",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--run-name', default='retool-qwen35-4b', help='SwanLab 实验名称，同时作为 checkpoint 名称前缀')）。
    parser.add_argument(
        "--run-name",
        default="retool-qwen35-4b",
        help="SwanLab 实验名称，同时作为 checkpoint 名称前缀",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-retool', help='SwanLab 项目名称')）。
    parser.add_argument(
        "--swanlab-project",
        default="agentic-rl-lab-retool",
        help="SwanLab 项目名称",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=['online', 'local', 'offline', 'disabled'], default='online',…）。
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="online",
        help="SwanLab 日志模式",
    )
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 main：运行同步训练循环，并在 rollout 采样与代码执行时使用 async。
def main(args: argparse.Namespace) -> None:
    """运行同步训练循环，并在 rollout 采样与代码执行时使用 async。"""
    # 先固定打乱训练问题；max_train_samples 只用于限制本次实际参与训练的数据量。
    examples = shuffled_examples(args.data, args.seed)
    # 逐行讲解：检查条件 args.max_train_samples > 0；成立时执行下方分支。
    if args.max_train_samples > 0:
        # 逐行讲解：按索引或切片读取 examples[:args.max_train_samples]，赋给 examples。
        examples = examples[: args.max_train_samples]
    # 逐行讲解：检查条件 not examples；成立时执行下方分支。
    if not examples:
        # 逐行讲解：抛出异常：ValueError('训练数据为空，请先运行 prepare_data.py')。
        raise ValueError("训练数据为空，请先运行 prepare_data.py")

    # 从基础模型新建 LoRA 训练客户端
    service_client = trio.ServiceClient()
    # 逐行讲解：调用 service_client.create_lora_training_client（service_client.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank, seed=args.seed)），赋给 training_client。
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )

    # tokenizer 来自训练客户端，保证 rollout、训练 Datum 使用完全相同的分词方式。
    tokenizer = training_client.get_tokenizer()
    # 逐行讲解：调用 LocalPythonSandbox（LocalPythonSandbox(timeout=args.sandbox_timeout, max_workers=args.sandbox_workers)），赋给 sandbox。
    sandbox = LocalPythonSandbox(
        timeout=args.sandbox_timeout,
        max_workers=args.sandbox_workers,
    )

    # 将命令行中的采样、代码调用次数和轨迹长度限制集中成 rollout 配置。
    rollout_config = RolloutConfig(
        group_size=args.group_size,
        max_code_calls=args.max_code_calls,
        max_assistant_turns=args.max_assistant_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        max_tool_response_tokens=args.max_tool_response_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    # 每个训练 step 的所有 micro-batch 共享同一组 Adam 参数，并只 optim_step 一次。
    adam_params = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    # 记录完整命令行配置，方便后续从 SwanLab 对照和复现实验。
    run = swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config=serializable_config(args),
    )

    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 外层进度条统计已完成的训练 step，并自动显示总耗时和预计剩余时间。
        with tqdm(
            total=args.max_steps,
            desc="Training",
            unit="step",
            position=0,
        ) as training_progress:
            # 逐行讲解：遍历 range(args.max_steps)，将每项绑定到 step。
            for step in range(args.max_steps):
                # 逐行讲解：调用 perf_counter（perf_counter()），赋给 step_started。
                step_started = perf_counter()

                # 按 questions_per_batch 循环取题；超过数据末尾时 take_batch 会回绕。
                batch = take_batch(
                    examples,
                    step * args.questions_per_batch,
                    args.questions_per_batch,
                )

                # 导出当前 LoRA 权重创建 sampler，确保本 step 的 rollout 来自当前策略。
                training_progress.set_postfix(phase="prepare sampler", refresh=True)
                # 逐行讲解：保存当前权重并创建对应采样客户端（training_client.save_weights_and_get_sampling_client()），赋给 sampling_client。
                sampling_client = training_client.save_weights_and_get_sampling_client()

                # 内层进度条显示当前 step 已完成多少条轨迹。
                training_progress.set_postfix(phase="rollout", refresh=True)
                # 逐行讲解：进入上下文管理器 tqdm(total=len(batch) * args.group_size, desc=f'Step {step + 1}/{args.max_steps} rollout', unit='trajectory…，退出时执行其收尾逻辑。
                with tqdm(
                    total=len(batch) * args.group_size,
                    desc=f"Step {step + 1}/{args.max_steps} rollout",
                    unit="trajectory",
                    position=1,
                    leave=False,
                ) as rollout_progress:
                    # 为每道题采样 group_size 条多轮代码交织轨迹，并计算 reward 和组内 advantage。
                    trajectories = rollout_batch(
                        sampling_client,
                        tokenizer,
                        sandbox,
                        batch,
                        rollout_config,
                        progress_callback=rollout_progress.update,
                    )

                # 每条有训练信号的完整轨迹只构造一个 Datum，再按 padding 矩形动态装箱。
                training_progress.set_postfix(phase="build datums", refresh=True)
                # 逐行讲解：调用 build_training_datums（build_training_datums(trajectories)），赋给 datums。
                datums = build_training_datums(trajectories)
                # 逐行讲解：调用 pack_micro_batches（pack_micro_batches(datums)），赋给 micro_batches。
                micro_batches = pack_micro_batches(datums)

                # 远端对每次请求按样本取 mean；按当前批次占全部 rollout 样本的比例
                # 缩放 advantage 后再累积，保证动态拆批不改变 global mean 梯度。
                training_progress.set_postfix(phase="backward", refresh=True)
                # 逐行讲解：构造列表 []，赋给 trainer_results。
                trainer_results = []
                # 逐行讲解：遍历 micro_batches，将每项绑定到 micro_batch。
                for micro_batch in micro_batches:
                    # 逐行讲解：调用 training_client.forward_backward(weight_micro_batch_for_global_mean(micro_batch, total_samples=len(trajecto…（training_client.forward_backward(weight_micro_batch_for_global_mean(micro_batch, total_samples=len(trajecto…），赋给 result。
                    result = training_client.forward_backward(
                        weight_micro_batch_for_global_mean(
                            micro_batch,
                            total_samples=len(trajectories),
                        ),
                        loss_fn="ppo",
                        loss_fn_config=PPO_LOSS_CONFIG,
                    ).result()
                    # 逐行讲解：向列表末尾追加一个元素（trainer_results.append(result)）。
                    trainer_results.append(result)

                # 整个 rollout batch 只更新一次参数；没有非零 advantage 时跳过更新。
                if micro_batches:
                    # 逐行讲解：调用 training_progress.set_postfix（training_progress.set_postfix(phase='optimizer', refresh=True)）。
                    training_progress.set_postfix(phase="optimizer", refresh=True)
                    # 逐行讲解：等待任务完成并读取结果（training_client.optim_step(adam_params).result()）。
                    training_client.optim_step(adam_params).result()

                # 汇总 rollout、沙箱和远程 trainer 指标。
                metrics = rollout_metrics(
                    trajectories,
                    datums,
                    micro_batches,
                    len(batch),
                )
                # 逐行讲解：调用 metrics.update（metrics.update(sandbox.stats.metrics())）。
                metrics.update(sandbox.stats.metrics())
                # 逐行讲解：调用 metrics.update（metrics.update(merge_trainer_metrics(trainer_results))）。
                metrics.update(merge_trainer_metrics(trainer_results))
                # 逐行讲解：调用 pick_mean_loss_metric（pick_mean_loss_metric(metrics)），赋给 mean_loss。
                mean_loss = pick_mean_loss_metric(metrics)

                # 按配置周期保存断点 state 和推理 sampler weights；保存耗时算在当前 step 内。
                if args.save_every > 0 and (step + 1) % args.save_every == 0:
                    # 逐行讲解：调用 training_progress.set_postfix（training_progress.set_postfix(phase='checkpoint', refresh=True)）。
                    training_progress.set_postfix(phase="checkpoint", refresh=True)
                    # 逐行讲解：调用 save_checkpoint（save_checkpoint(training_client, f'{args.run_name}-step-{step + 1}')）。
                    save_checkpoint(
                        training_client,
                        f"{args.run_name}-step-{step + 1}",
                    )

                # 记录完整 step 耗时，终端和 SwanLab 都能看到同一数值。
                step_seconds = perf_counter() - step_started
                # 逐行讲解：取值 step_seconds，赋给 metrics['time/step_seconds']。
                metrics["time/step_seconds"] = step_seconds
                # 逐行讲解：转换为浮点表示（float(not micro_batches)），赋给 metrics['train/update_skipped']。
                metrics["train/update_skipped"] = float(not micro_batches)
                # 逐行讲解：计算自然对数（swanlab.log(metrics, step=step)）。
                swanlab.log(metrics, step=step)

                # 逐行讲解：检查条件 not micro_batches；成立时执行下方分支。
                if not micro_batches:
                    # 逐行讲解：取值 'skipped'，赋给 loss_mean_text。
                    loss_mean_text = "skipped"
                # 逐行讲解：检查条件 mean_loss is None；成立时执行下方分支。
                elif mean_loss is None:
                    # 逐行讲解：取值 'missing'，赋给 loss_mean_text。
                    loss_mean_text = "missing"
                else:
                    # 逐行讲解：取值 f'{mean_loss:.4f}'，赋给 loss_mean_text。
                    loss_mean_text = f"{mean_loss:.4f}"
                # 逐行讲解：调用 training_progress.update（training_progress.update(1)）。
                training_progress.update(1)
                # 逐行讲解：调用 training_progress.set_postfix（training_progress.set_postfix(step_s=f'{step_seconds:.1f}', loss_mean=loss_mean_text, reward=f"{metrics['re…）。
                training_progress.set_postfix(
                    step_s=f"{step_seconds:.1f}",
                    loss_mean=loss_mean_text,
                    reward=f"{metrics['reward/mean']:.3f}",
                    refresh=True,
                )
                # 逐行讲解：调用 tqdm.write（tqdm.write(f"step={step + 1}/{args.max_steps} step_time={step_seconds:.1f}s loss_mean={loss_mean_text} rewa…）。
                tqdm.write(
                    f"step={step + 1}/{args.max_steps} "
                    f"step_time={step_seconds:.1f}s "
                    f"loss_mean={loss_mean_text} "
                    f"reward={metrics['reward/mean']:.3f} "
                    f"correct={metrics['reward/correct']:.3f} "
                    f"code_calls={metrics['rollout/code_calls']:.2f}"
                )

        # 无论周期保存频率如何，正常完成训练后始终保存一次最终 checkpoint。
        save_checkpoint(training_client, f"{args.run_name}-final")
    finally:
        # 即使训练中途抛出异常，也要结束 SwanLab run，避免实验一直显示为运行中。
        run.finish()


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main(parse_args())）。
    main(parse_args())
