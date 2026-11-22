# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：从 collections.abc 导入 Callable。
from collections.abc import Callable
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：导入模块：math。
import math
# 逐行讲解：从 typing 导入 Any。
from typing import Any
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：从 data 导入 MathExample。
from data import MathExample
# 逐行讲解：从 reward 导入 RewardResult, score_answer。
from reward import RewardResult, score_answer


# 逐行讲解：取值 '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'，赋给 QUESTION_SUFFIX。
QUESTION_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)
# 逐行讲解：取值 1e-08，赋给 ADVANTAGE_EPSILON。
ADVANTAGE_EPSILON = 1e-8


# 逐行讲解：定义类 RolloutConfig：采样与 group 构造参数（GSPO 无候选倍率 / overlong 字段）。
@dataclass(frozen=True)
class RolloutConfig:
    """采样与 group 构造参数（GSPO 无候选倍率 / overlong 字段）。"""

    # 逐行讲解：声明 group_size 的类型为 int；取值 8。
    group_size: int = 8
    # 逐行讲解：声明 max_prompt_tokens 的类型为 int；取值 512。
    max_prompt_tokens: int = 512
    # 逐行讲解：声明 max_tokens 的类型为 int；取值 4096。
    max_tokens: int = 4096
    # 逐行讲解：声明 temperature 的类型为 float；取值 1.0。
    temperature: float = 1.0
    # 逐行讲解：声明 top_p 的类型为 float；取值 1.0。
    top_p: float = 1.0
    # 逐行讲解：声明 top_k 的类型为 int；取值 -1。
    top_k: int = -1
    # 逐行讲解：声明 seed 的类型为 int；取值 42。
    seed: int = 42

    # 逐行讲解：定义函数 validate：在发起远端采样前检查配置。
    def validate(self) -> None:
        """在发起远端采样前检查配置。"""
        # 逐行讲解：检查条件 self.group_size < 2；成立时执行下方分支。
        if self.group_size < 2:
            # 逐行讲解：抛出异常：ValueError('group_size must be >= 2')。
            raise ValueError("group_size must be >= 2")
        # 逐行讲解：检查条件 self.max_prompt_tokens < 1 or self.max_tokens < 1；成立时执行下方分支。
        if self.max_prompt_tokens < 1 or self.max_tokens < 1:
            # 逐行讲解：抛出异常：ValueError('max_prompt_tokens and max_tokens must be >= 1')。
            raise ValueError("max_prompt_tokens and max_tokens must be >= 1")
        # 逐行讲解：检查条件 self.temperature < 0；成立时执行下方分支。
        if self.temperature < 0:
            # 逐行讲解：抛出异常：ValueError('temperature must be >= 0')。
            raise ValueError("temperature must be >= 0")
        # 逐行讲解：检查条件 not 0 < self.top_p <= 1；成立时执行下方分支。
        if not 0 < self.top_p <= 1:
            # 逐行讲解：抛出异常：ValueError('top_p must be in (0, 1]')。
            raise ValueError("top_p must be in (0, 1]")


# 逐行讲解：定义类 RolloutSample：一条 completion 及其训练所需的旧策略信息。
@dataclass
class RolloutSample:
    """一条 completion 及其训练所需的旧策略信息。

    ``sampling_logprobs`` 是 sampler 对每个 completion token 返回的 logprob，
    GSPO 的序列级重要性比率以它为旧策略端点（见 loss.py）。
    """

    # 逐行讲解：声明 example 的类型为 MathExample。
    example: MathExample
    # 逐行讲解：声明 completion_tokens 的类型为 list[int]。
    completion_tokens: list[int]
    # 逐行讲解：声明 sampling_logprobs 的类型为 list[float]。
    sampling_logprobs: list[float]
    # 逐行讲解：声明 text 的类型为 str。
    text: str
    # 逐行讲解：声明 reward_result 的类型为 RewardResult。
    reward_result: RewardResult
    # 逐行讲解：声明 advantage 的类型为 float；取值 0.0。
    advantage: float = 0.0


# 逐行讲解：定义类 RolloutGroup：同一道题的一组 completion（一个 GSPO 组）。
@dataclass
class RolloutGroup:
    """同一道题的一组 completion（一个 GSPO 组）。"""

    # 逐行讲解：声明 example 的类型为 MathExample。
    example: MathExample
    # 逐行讲解：声明 prompt_tokens 的类型为 list[int]。
    prompt_tokens: list[int]
    # 逐行讲解：声明 samples 的类型为 list[RolloutSample]。
    samples: list[RolloutSample]
    # 逐行讲解：声明 expected_group_size 的类型为 int。
    expected_group_size: int

    # 逐行讲解：定义函数 correct_count：封装下方逻辑，供调用方复用。
    @property
    def correct_count(self) -> int:
        # 逐行讲解：返回结果：sum((int(sample.reward_result.correct) for sample in self.samples))。
        return sum(int(sample.reward_result.correct) for sample in self.samples)

    # 逐行讲解：定义函数 usable：完整返回且每条 completion 都有 token，才允许进入训练。
    @property
    def usable(self) -> bool:
        """完整返回且每条 completion 都有 token，才允许进入训练。"""
        # 逐行讲解：返回结果：len(self.samples) == self.expected_group_size and all((sample.completion_tokens for sample in self.samples))。
        return (
            len(self.samples) == self.expected_group_size
            and all(sample.completion_tokens for sample in self.samples)
        )

    # 逐行讲解：定义函数 degenerate：组内 reward 全相同的退化组：advantage 全 0，对梯度没有贡献。
    @property
    def degenerate(self) -> bool:
        """组内 reward 全相同的退化组：advantage 全 0，对梯度没有贡献。

        GSPO 论文未使用 Dynamic Sampling，退化组不补采。训练侧可跳过
        它们的远端计算，但会在 loss 的原始 batch 分母中保留其零目标。
        """
        # 逐行讲解：通过推导式逐项生成结果：{sample.reward_result.shaped_reward for sample in self.samples}，赋给 rewards。
        rewards = {sample.reward_result.shaped_reward for sample in self.samples}
        # 逐行讲解：返回结果：len(rewards) <= 1。
        return len(rewards) <= 1


# 逐行讲解：定义类 RolloutBatch：一个训练 step 的 rollout 结果。
@dataclass(frozen=True)
class RolloutBatch:
    """一个训练 step 的 rollout 结果。

    ``train_groups`` 包含退化组在内的全部采样组，是否剔除由调用方决定。
    """

    # 逐行讲解：声明 requested_groups 的类型为 int。
    requested_groups: int
    # 逐行讲解：声明 train_groups 的类型为 list[RolloutGroup]。
    train_groups: list[RolloutGroup]

    # 逐行讲解：定义函数 effective_groups：完整可用且非退化的组，即实际能产生梯度的组。
    @property
    def effective_groups(self) -> list[RolloutGroup]:
        """完整可用且非退化的组，即实际能产生梯度的组。"""
        # 逐行讲解：返回结果：[group for group in self.train_groups if group.usable and (not group.degenerate)]。
        return [
            group
            for group in self.train_groups
            if group.usable and not group.degenerate
        ]

    # 逐行讲解：定义函数 effective_group_ratio：封装下方逻辑，供调用方复用。
    @property
    def effective_group_ratio(self) -> float:
        # 逐行讲解：返回结果：len(self.effective_groups) / max(len(self.train_groups), 1)。
        return len(self.effective_groups) / max(len(self.train_groups), 1)


# 逐行讲解：定义函数 build_prompt_tokens：用模型 chat template 构造单轮数学题 prompt（模板与后缀同 experiments/dapo）。
def build_prompt_tokens(tokenizer: Any, question: str) -> list[int]:
    """用模型 chat template 构造单轮数学题 prompt（模板与后缀同 experiments/dapo）。

    长度上限不在此处检查，由 ``sample_group_async`` 按
    ``RolloutConfig.max_prompt_tokens`` 校验。
    """
    # 逐行讲解：计算 question.strip() + QUESTION_SUFFIX，赋给 content。
    content = question.strip() + QUESTION_SUFFIX
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt…），赋给 prompt。
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # 逐行讲解：调用 list（list(tokenizer.encode(prompt, add_special_tokens=False))），赋给 tokens。
    tokens = list(tokenizer.encode(prompt, add_special_tokens=False))
    # 逐行讲解：检查条件 not tokens；成立时执行下方分支。
    if not tokens:
        # 逐行讲解：抛出异常：ValueError('prompt tokens are empty')。
        raise ValueError("prompt tokens are empty")
    # 逐行讲解：返回结果：tokens。
    return tokens


# 逐行讲解：定义函数 stop_sequences：返回 chat 终止字符串。
def stop_sequences(tokenizer: Any) -> list[str]:
    """返回 chat 终止字符串。"""
    # 逐行讲解：返回结果：[tokenizer.eos_token] if tokenizer.eos_token else ['<|im_end|>']。
    return [tokenizer.eos_token] if tokenizer.eos_token else ["<|im_end|>"]


# 逐行讲解：定义函数 read_sequence：读取 PyTRIO completion，并严格校验 token/logprob 对齐。
def read_sequence(sequence: Any, tokenizer: Any) -> tuple[list[int], list[float], str]:
    """读取 PyTRIO completion，并严格校验 token/logprob 对齐。"""
    # 逐行讲解：通过推导式逐项生成结果：[int(token) for token in sequence.tokens]，赋给 tokens。
    tokens = [int(token) for token in sequence.tokens]
    # 逐行讲解：通过推导式逐项生成结果：[float(value) for value in sequence.logprobs]，赋给 logprobs。
    logprobs = [float(value) for value in sequence.logprobs]
    # 逐行讲解：检查条件 len(tokens) != len(logprobs)；成立时执行下方分支。
    if len(tokens) != len(logprobs):
        # 逐行讲解：抛出异常：ValueError(f'completion token/logprob mismatch: {len(tokens)} != {len(logprobs)}')。
        raise ValueError(
            f"completion token/logprob mismatch: {len(tokens)} != {len(logprobs)}"
        )
    # 逐行讲解：取值 sequence.text，赋给 text。
    text = sequence.text
    # 逐行讲解：检查条件 text is None；成立时执行下方分支。
    if text is None:
        # 逐行讲解：将 token 解码为文本（tokenizer.decode(tokens, skip_special_tokens=False)），赋给 text。
        text = tokenizer.decode(tokens, skip_special_tokens=False)
    # 逐行讲解：返回结果：(tokens, logprobs, str(text))。
    return tokens, logprobs, str(text)


# 逐行讲解：定义函数 assign_group_advantages：在组内对 reward 做标准化，得到每条序列共享的 advantage 标量。
def assign_group_advantages(
    group: RolloutGroup,
    *,
    epsilon: float = ADVANTAGE_EPSILON,
) -> None:
    """在组内对 reward 做标准化，得到每条序列共享的 advantage 标量。

    advantage = (r - mean) / (std + epsilon)，其中 std 使用 N-1 分母的
    样本标准差（对齐 verl 的 torch.std 默认行为）。组内 reward 全相同时
    std 为 0，整组 advantage 置 0，即退化组。
    """
    # 逐行讲解：检查条件 not group.samples；成立时执行下方分支。
    if not group.samples:
        # 逐行讲解：返回，结束当前函数。
        return
    # 逐行讲解：通过推导式逐项生成结果：[sample.reward_result.shaped_reward for sample in group.samples]，赋给 rewards。
    rewards = [sample.reward_result.shaped_reward for sample in group.samples]
    # 逐行讲解：计算 sum(rewards) / len(rewards)，赋给 reward_mean。
    reward_mean = sum(rewards) / len(rewards)
    # 逐行讲解：检查条件 len(rewards) == 1；成立时执行下方分支。
    if len(rewards) == 1:
        # 逐行讲解：取值 0.0，赋给 reward_std。
        reward_std = 0.0
    else:
        # 逐行讲解：计算 sum(((reward - reward_mean) ** 2 for reward in rewards)) / (len(rewards) - 1)，赋给 variance。
        variance = sum(
            (reward - reward_mean) ** 2 for reward in rewards
        ) / (len(rewards) - 1)
        # 逐行讲解：调用 math.sqrt（math.sqrt(variance)），赋给 reward_std。
        reward_std = math.sqrt(variance)
    # 逐行讲解：检查条件 reward_std == 0.0；成立时执行下方分支。
    if reward_std == 0.0:
        # 逐行讲解：遍历 group.samples，将每项绑定到 sample。
        for sample in group.samples:
            # 逐行讲解：取值 0.0，赋给 sample.advantage。
            sample.advantage = 0.0
        # 逐行讲解：返回，结束当前函数。
        return
    # 逐行讲解：计算 reward_std + epsilon，赋给 denominator。
    denominator = reward_std + epsilon
    # 逐行讲解：遍历 group.samples，将每项绑定到 sample。
    for sample in group.samples:
        # 逐行讲解：计算 (sample.reward_result.shaped_reward - reward_mean) / denominator，赋给 sample.advantage。
        sample.advantage = (
            sample.reward_result.shaped_reward - reward_mean
        ) / denominator


# 逐行讲解：定义异步函数 sample_group_async：为一道题采样一个完整 group 并计算 reward/advantage。
async def sample_group_async(
    sampling_client: Any,
    tokenizer: Any,
    example: MathExample,
    config: RolloutConfig,
    *,
    group_index: int,
) -> RolloutGroup:
    """为一道题采样一个完整 group 并计算 reward/advantage。"""
    # 逐行讲解：调用 build_prompt_tokens（build_prompt_tokens(tokenizer, example.question)），赋给 prompt_tokens。
    prompt_tokens = build_prompt_tokens(tokenizer, example.question)
    # 逐行讲解：检查条件 len(prompt_tokens) > config.max_prompt_tokens；成立时执行下方分支。
    if len(prompt_tokens) > config.max_prompt_tokens:
        # 逐行讲解：抛出异常：ValueError(f'prompt has {len(prompt_tokens)} tokens, exceeding {config.max_prompt_tokens}')。
        raise ValueError(
            f"prompt has {len(prompt_tokens)} tokens, "
            f"exceeding {config.max_prompt_tokens}"
        )
    # 逐行讲解：等待异步操作 sampling_client.sample_async(prompt=trio.ModelInput.from_ints(prompt_tokens), num_samples=config.group_size… 完成并取得结果，赋给 response。
    response = await sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=config.group_size,
        sampling_params=trio.SamplingParams(
            max_tokens=config.max_tokens,
            seed=config.seed + group_index,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop=stop_sequences(tokenizer),
        ),
        return_text=True,
    )

    # 逐行讲解：声明 samples 的类型为 list[RolloutSample]；构造列表 []。
    samples: list[RolloutSample] = []
    # 逐行讲解：遍历 response.sequences，将每项绑定到 sequence。
    for sequence in response.sequences:
        # 逐行讲解：调用 read_sequence（read_sequence(sequence, tokenizer)），赋给 (tokens, sampling_logprobs, text)。
        tokens, sampling_logprobs, text = read_sequence(sequence, tokenizer)
        # 逐行讲解：调用 score_answer（score_answer(text, example.answer, completion_tokens=len(tokens))），赋给 reward。
        reward = score_answer(
            text,
            example.answer,
            completion_tokens=len(tokens),
        )
        # 逐行讲解：向列表末尾追加一个元素（samples.append(RolloutSample(example=example, completion_tokens=tokens, sampling_logprobs=sampling_logprobs…）。
        samples.append(
            RolloutSample(
                example=example,
                completion_tokens=tokens,
                sampling_logprobs=sampling_logprobs,
                text=text,
                reward_result=reward,
            )
        )

    # 逐行讲解：调用 RolloutGroup（RolloutGroup(example=example, prompt_tokens=prompt_tokens, samples=samples, expected_group_size=config.grou…），赋给 group。
    group = RolloutGroup(
        example=example,
        prompt_tokens=prompt_tokens,
        samples=samples,
        expected_group_size=config.group_size,
    )
    # 逐行讲解：调用 assign_group_advantages（assign_group_advantages(group)）。
    assign_group_advantages(group)
    # 逐行讲解：返回结果：group。
    return group


# 逐行讲解：定义异步函数 collect_rollout_batch：对给定题目一次性并发采齐所有组，不做 Dynamic Sampling 补采。
async def collect_rollout_batch(
    sampling_client: Any,
    tokenizer: Any,
    examples: list[MathExample],
    config: RolloutConfig,
    *,
    concurrency: int,
    progress_callback: Callable[[int], None] | None = None,
) -> RolloutBatch:
    """对给定题目一次性并发采齐所有组，不做 Dynamic Sampling 补采。

    GSPO 论文未使用 Dynamic Sampling：每道题只采一轮 ``group_size`` 条，
    退化组（组内 reward 全同）保留在返回的 batch 里；训练侧跳过其远端
    计算，同时保留原始 batch 分母。
    并发度由调用方传入并用 semaphore 限流，返回顺序与输入题目顺序一致。
    """
    # 逐行讲解：调用 config.validate（config.validate()）。
    config.validate()
    # 逐行讲解：检查条件 not examples；成立时执行下方分支。
    if not examples:
        # 逐行讲解：抛出异常：ValueError('examples must not be empty')。
        raise ValueError("examples must not be empty")

    # 逐行讲解：调用 asyncio.Semaphore（asyncio.Semaphore(concurrency)），赋给 semaphore。
    semaphore = asyncio.Semaphore(concurrency)

    # 逐行讲解：定义异步函数 sample_one：封装下方逻辑，供调用方复用。
    async def sample_one(group_index: int, example: MathExample) -> RolloutGroup:
        # 逐行讲解：进入上下文管理器 semaphore，退出时执行其收尾逻辑。
        async with semaphore:
            # 逐行讲解：等待异步操作 sample_group_async(sampling_client, tokenizer, example, config, group_index=group_index) 完成并取得结果，赋给 group。
            group = await sample_group_async(
                sampling_client,
                tokenizer,
                example,
                config,
                group_index=group_index,
            )
            # 逐行讲解：检查条件 progress_callback is not None；成立时执行下方分支。
            if progress_callback is not None:
                # 逐行讲解：调用 progress_callback（progress_callback(1)）。
                progress_callback(1)
            # 逐行讲解：返回结果：group。
            return group

    # 逐行讲解：调用 list（list(await asyncio.gather(*(sample_one(group_index, example) for group_index, example in enumerate(examples…），赋给 groups。
    groups = list(
        await asyncio.gather(
            *(
                sample_one(group_index, example)
                for group_index, example in enumerate(examples)
            )
        )
    )
    # 逐行讲解：返回结果：RolloutBatch(requested_groups=len(examples), train_groups=groups)。
    return RolloutBatch(requested_groups=len(examples), train_groups=groups)
