"""GSPO 序列级裁剪 loss，对应论文 arXiv:2507.18071 式 (5) 和式 (7)。

每条 completion 的重要性比率为逐 token 概率比的几何平均；整条序列共享
同一个 advantage 和裁剪结果。PyTRIO 提供当前 logprob，sampling logprob 与
advantage 由 rollout 元数据传入。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：从 collections.abc 导入 Callable。
from collections.abc import Callable
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：torch。
import torch

# 逐行讲解：从 rollout 导入 RolloutSample。
from rollout import RolloutSample


# 逐行讲解：定义类 GSPOConfig：GSPO 序列级裁剪区间，默认值为论文原文取值。
@dataclass(frozen=True)
class GSPOConfig:
    """GSPO 序列级裁剪区间，默认值为论文原文取值。"""

    # 逐行讲解：声明 clip_ratio_low 的类型为 float；取值 0.0003。
    clip_ratio_low: float = 3e-4
    # 逐行讲解：声明 clip_ratio_high 的类型为 float；取值 0.0004。
    clip_ratio_high: float = 4e-4

    # 逐行讲解：定义函数 __post_init__：封装下方逻辑，供调用方复用。
    def __post_init__(self) -> None:
        # 逐行讲解：检查条件 self.clip_ratio_low < 0 or self.clip_ratio_high < 0；成立时执行下方分支。
        if self.clip_ratio_low < 0 or self.clip_ratio_high < 0:
            # 逐行讲解：抛出异常：ValueError('GSPO clip ratios must be >= 0')。
            raise ValueError("GSPO clip ratios must be >= 0")


# 逐行讲解：定义类 GSPOMeta：一条 RolloutSample 的本地 GSPO 元信息。
@dataclass(frozen=True)
class GSPOMeta:
    """一条 RolloutSample 的本地 GSPO 元信息。"""

    # 逐行讲解：声明 sampling_logprobs 的类型为 list[float]。
    sampling_logprobs: list[float]
    # 逐行讲解：声明 advantage 的类型为 float。
    advantage: float
    # 逐行讲解：声明 completion_tokens 的类型为 int。
    completion_tokens: int


# 逐行讲解：定义函数 build_datum：把 RolloutSample 转成右移对齐的 Datum 与 GSPO 元数据。
def build_datum(
    prompt_tokens: list[int],
    sample: RolloutSample,
) -> tuple[trio.Datum, GSPOMeta]:
    """把 RolloutSample 转成右移对齐的 Datum 与 GSPO 元数据。

    - model_input   = prompt_tokens + completion_tokens[:-1]
    - target_tokens = [0] * (len(prompt_tokens) - 1) + completion_tokens

    prompt 段 target 仅作占位，loss 只读取末尾 completion 区间。
    """
    # 逐行讲解：通过推导式逐项生成结果：[int(token) for token in prompt_tokens]，赋给 prompt_tokens。
    prompt_tokens = [int(token) for token in prompt_tokens]
    # 逐行讲解：通过推导式逐项生成结果：[int(token) for token in sample.completion_tokens]，赋给 completion_tokens。
    completion_tokens = [int(token) for token in sample.completion_tokens]
    # 逐行讲解：通过推导式逐项生成结果：[float(value) for value in sample.sampling_logprobs]，赋给 sampling_logprobs。
    sampling_logprobs = [
        float(value) for value in sample.sampling_logprobs
    ]
    # 逐行讲解：检查条件 not prompt_tokens；成立时执行下方分支。
    if not prompt_tokens:
        # 逐行讲解：抛出异常：ValueError('prompt_tokens must not be empty')。
        raise ValueError("prompt_tokens must not be empty")
    # 逐行讲解：检查条件 not completion_tokens；成立时执行下方分支。
    if not completion_tokens:
        # 逐行讲解：抛出异常：ValueError('completion_tokens must not be empty')。
        raise ValueError("completion_tokens must not be empty")
    # 逐行讲解：检查条件 len(completion_tokens) != len(sampling_logprobs)；成立时执行下方分支。
    if len(completion_tokens) != len(sampling_logprobs):
        # 逐行讲解：抛出异常：ValueError('completion token/logprob lengths must match')。
        raise ValueError("completion token/logprob lengths must match")

    # 逐行讲解：计算 prompt_tokens + completion_tokens[:-1]，赋给 input_tokens。
    input_tokens = prompt_tokens + completion_tokens[:-1]
    # 逐行讲解：计算 [0] * (len(prompt_tokens) - 1) + completion_tokens，赋给 target_tokens。
    target_tokens = [0] * (len(prompt_tokens) - 1) + completion_tokens
    # 逐行讲解：调用 trio.Datum（trio.Datum(model_input=trio.ModelInput.from_ints(input_tokens), loss_fn_inputs={'target_tokens': np.asarray…），赋给 datum。
    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
        },
    )
    # 逐行讲解：调用 GSPOMeta（GSPOMeta(sampling_logprobs=sampling_logprobs, advantage=float(sample.advantage), completion_tokens=len(comp…），赋给 meta。
    meta = GSPOMeta(
        sampling_logprobs=sampling_logprobs,
        advantage=float(sample.advantage),
        completion_tokens=len(completion_tokens),
    )
    # 逐行讲解：返回结果：(datum, meta)。
    return datum, meta


# 逐行讲解：定义函数 make_gspo_loss_fn：创建 PyTRIO forward_backward_custom 使用的本地 GSPO loss。
def make_gspo_loss_fn(
    metas: list[GSPOMeta],
    config: GSPOConfig,
    *,
    normalization_sequences: int | None = None,
    normalization_tokens: int | None = None,
) -> Callable[[list[trio.Datum], list[Any]], tuple[Any, dict[str, float]]]:
    """创建 PyTRIO forward_backward_custom 使用的本地 GSPO loss。"""
    # 逐行讲解：检查条件 not metas；成立时执行下方分支。
    if not metas:
        # 逐行讲解：抛出异常：ValueError('GSPO loss requires at least one datum')。
        raise ValueError("GSPO loss requires at least one datum")
    # 逐行讲解：检查条件 normalization_sequences is None；成立时执行下方分支。
    if normalization_sequences is None:
        # 逐行讲解：调用 len（len(metas)），赋给 normalization_sequences。
        normalization_sequences = len(metas)
    # 逐行讲解：检查条件 normalization_sequences < len(metas)；成立时执行下方分支。
    if normalization_sequences < len(metas):
        # 逐行讲解：抛出异常：ValueError('normalization_sequences must be >= the number of training sequences')。
        raise ValueError(
            "normalization_sequences must be >= the number of training sequences"
        )
    # 逐行讲解：求和（sum((meta.completion_tokens for meta in metas))），赋给 train_tokens。
    train_tokens = sum(meta.completion_tokens for meta in metas)
    # 逐行讲解：检查条件 normalization_tokens is None；成立时执行下方分支。
    if normalization_tokens is None:
        # 逐行讲解：取值 train_tokens，赋给 normalization_tokens。
        normalization_tokens = train_tokens
    # 逐行讲解：检查条件 normalization_tokens < train_tokens；成立时执行下方分支。
    if normalization_tokens < train_tokens:
        # 逐行讲解：抛出异常：ValueError('normalization_tokens must be >= the number of training tokens')。
        raise ValueError(
            "normalization_tokens must be >= the number of training tokens"
        )

    # 逐行讲解：定义函数 gspo_loss_fn：封装下方逻辑，供调用方复用。
    def gspo_loss_fn(
        data: list[trio.Datum],
        logprobs_list: list[Any],
    ) -> tuple[Any, dict[str, float]]:
        # 逐行讲解：检查条件 not len(data) == len(logprobs_list) == len(metas)；成立时执行下方分支。
        if not (len(data) == len(logprobs_list) == len(metas)):
            # 逐行讲解：抛出异常：ValueError('GSPO loss got mismatched batch lengths')。
            raise ValueError("GSPO loss got mismatched batch lengths")

        # 逐行讲解：声明 sequence_objectives 的类型为 list[torch.Tensor]；构造列表 []。
        sequence_objectives: list[torch.Tensor] = []
        # 逐行讲解：声明 detached_ratios 的类型为 list[torch.Tensor]；构造列表 []。
        detached_ratios: list[torch.Tensor] = []
        # 逐行讲解：声明 clip_flags 的类型为 list[torch.Tensor]；构造列表 []。
        clip_flags: list[torch.Tensor] = []

        # 逐行讲解：遍历 zip(metas, logprobs_list, strict=True)，将每项绑定到 (meta, current_values)。
        for meta, current_values in zip(metas, logprobs_list, strict=True):
            # 末尾 completion 区间与 sampling_logprobs 一一对应。
            current = current_values.float().reshape(-1)
            # 逐行讲解：检查条件 current.numel() < meta.completion_tokens；成立时执行下方分支。
            if current.numel() < meta.completion_tokens:
                # 逐行讲解：抛出异常：ValueError('logprob sequence shorter than completion')。
                raise ValueError("logprob sequence shorter than completion")
            # 逐行讲解：按索引或切片读取 current[-meta.completion_tokens:]，赋给 current_completion。
            current_completion = current[-meta.completion_tokens :]
            # 逐行讲解：把数据转换为 PyTorch 张量（torch.as_tensor(meta.sampling_logprobs, dtype=torch.float32, device=current.device)），赋给 sampling。
            sampling = torch.as_tensor(
                meta.sampling_logprobs,
                dtype=torch.float32,
                device=current.device,
            )
            # 逐行讲解：检查条件 sampling.numel() != meta.completion_tokens；成立时执行下方分支。
            if sampling.numel() != meta.completion_tokens:
                # 逐行讲解：抛出异常：ValueError('sampling logprobs must match completion_tokens')。
                raise ValueError(
                    "sampling logprobs must match completion_tokens"
                )

            # 式 (7)：逐 token 概率比的几何平均。
            log_ratio = (current_completion - sampling).mean()
            # 逐行讲解：通过指数运算将对数值转换回原尺度（torch.exp(log_ratio)），赋给 seq_ratio。
            seq_ratio = torch.exp(log_ratio)

            # 式 (5)：整条序列共享同一个裁剪比率。
            clipped_ratio = torch.clamp(
                seq_ratio,
                min=1.0 - config.clip_ratio_low,
                max=1.0 + config.clip_ratio_high,
            )
            # 逐行讲解：计算 seq_ratio * meta.advantage，赋给 unclipped_objective。
            unclipped_objective = seq_ratio * meta.advantage
            # 逐行讲解：计算 clipped_ratio * meta.advantage，赋给 clipped_objective。
            clipped_objective = clipped_ratio * meta.advantage
            # 逐行讲解：向列表末尾追加一个元素（sequence_objectives.append(torch.minimum(unclipped_objective, clipped_objective))）。
            sequence_objectives.append(
                torch.minimum(unclipped_objective, clipped_objective)
            )

            # 逐行讲解：向列表末尾追加一个元素（detached_ratios.append(seq_ratio.detach())）。
            detached_ratios.append(seq_ratio.detach())
            # 逐行讲解：向列表末尾追加一个元素（clip_flags.append((clipped_objective.detach() < unclipped_objective.detach()).float())）。
            clip_flags.append(
                (clipped_objective.detach() < unclipped_objective.detach())
                .float()
            )

        # 每条序列等权；退化组的零目标仍包含在原始分母中。
        gspo_loss = (
            -torch.stack(sequence_objectives).sum() / normalization_sequences
        )

        # 逐行讲解：增加一个维度并堆叠张量（torch.stack(detached_ratios)），赋给 ratios。
        ratios = torch.stack(detached_ratios)
        # 逐行讲解：增加一个维度并堆叠张量（torch.stack(clip_flags)），赋给 clips。
        clips = torch.stack(clip_flags)
        # 逐行讲解：把数据转换为 PyTorch 张量（torch.as_tensor([meta.completion_tokens for meta in metas], dtype=clips.dtype, device=clips.device)），赋给 completion_lengths。
        completion_lengths = torch.as_tensor(
            [meta.completion_tokens for meta in metas],
            dtype=clips.dtype,
            device=clips.device,
        )
        # 逐行讲解：构造字典 {'gspo/loss': float(gspo_loss.detach().item()), 'gspo/seq_ratio_mean': float(ratios.mean().item()), 'gspo/s…，赋给 metrics。
        metrics = {
            "gspo/loss": float(gspo_loss.detach().item()),
            "gspo/seq_ratio_mean": float(ratios.mean().item()),
            # 退化组计入原始分母，但不计入裁剪分子。
            "gspo/sequence_clip_fraction": float(
                (clips.sum() / normalization_sequences).item()
            ),
            "gspo/token_clip_fraction": float(
                ((clips * completion_lengths).sum() / normalization_tokens).item()
            ),
            "gspo/train_sequences": float(len(metas)),
            "gspo/normalization_sequences": float(normalization_sequences),
            "gspo/active_sequence_fraction": float(
                len(metas) / normalization_sequences
            ),
            "gspo/train_tokens": float(train_tokens),
            "gspo/normalization_tokens": float(normalization_tokens),
            "gspo/active_token_fraction": float(
                train_tokens / normalization_tokens
            ),
        }
        # 逐行讲解：返回结果：(gspo_loss, metrics)。
        return gspo_loss, metrics

    # 逐行讲解：返回结果：gspo_loss_fn。
    return gspo_loss_fn
