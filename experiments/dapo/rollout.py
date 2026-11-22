"""GRPO/DAPO 共用的 group rollout 与 Dynamic Sampling。"""

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
# 逐行讲解：从 typing 导入 Any, Literal。
from typing import Any, Literal
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：从 data 导入 MathExample。
from data import MathExample
# 逐行讲解：从 reward 导入 RewardResult, score_answer。
from reward import RewardResult, score_answer


# 逐行讲解：按索引或切片读取 Literal['grpo', 'dapo']，赋给 Algorithm。
Algorithm = Literal["grpo", "dapo"]
# 逐行讲解：取值 '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'，赋给 QUESTION_SUFFIX。
QUESTION_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


# 逐行讲解：定义类 RolloutConfig：采样与 group 构造参数。
@dataclass(frozen=True)
class RolloutConfig:
    """采样与 group 构造参数。"""

    # 逐行讲解：声明 max_prompt_tokens 的类型为 int。
    max_prompt_tokens: int
    # 逐行讲解：声明 max_tokens 的类型为 int。
    max_tokens: int
    # 逐行讲解：声明 overlong_cache 的类型为 int。
    overlong_cache: int
    # 逐行讲解：声明 group_size 的类型为 int；取值 8。
    group_size: int = 8
    # 逐行讲解：声明 temperature 的类型为 float；取值 1.0。
    temperature: float = 1.0
    # 逐行讲解：声明 top_p 的类型为 float；取值 1.0。
    top_p: float = 1.0
    # 逐行讲解：声明 top_k 的类型为 int；取值 -1。
    top_k: int = -1
    # 逐行讲解：声明 concurrency 的类型为 int；取值 16。
    concurrency: int = 16
    # 逐行讲解：声明 seed 的类型为 int；取值 42。
    seed: int = 42
    # 逐行讲解：声明 advantage_epsilon 的类型为 float；取值 1e-08。
    advantage_epsilon: float = 1e-8

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
        # 逐行讲解：检查条件 not 1 <= self.overlong_cache <= self.max_tokens；成立时执行下方分支。
        if not 1 <= self.overlong_cache <= self.max_tokens:
            # 逐行讲解：抛出异常：ValueError('overlong_cache must be in [1, max_tokens]')。
            raise ValueError("overlong_cache must be in [1, max_tokens]")
        # 逐行讲解：检查条件 self.temperature < 0；成立时执行下方分支。
        if self.temperature < 0:
            # 逐行讲解：抛出异常：ValueError('temperature must be >= 0')。
            raise ValueError("temperature must be >= 0")
        # 逐行讲解：检查条件 not 0 < self.top_p <= 1；成立时执行下方分支。
        if not 0 < self.top_p <= 1:
            # 逐行讲解：抛出异常：ValueError('top_p must be in (0, 1]')。
            raise ValueError("top_p must be in (0, 1]")
        # 逐行讲解：检查条件 self.concurrency < 1；成立时执行下方分支。
        if self.concurrency < 1:
            # 逐行讲解：抛出异常：ValueError('concurrency must be >= 1')。
            raise ValueError("concurrency must be >= 1")
        # 逐行讲解：检查条件 self.advantage_epsilon <= 0；成立时执行下方分支。
        if self.advantage_epsilon <= 0:
            # 逐行讲解：抛出异常：ValueError('advantage_epsilon must be > 0')。
            raise ValueError("advantage_epsilon must be > 0")


# 逐行讲解：定义类 RolloutSample：一条 completion 及其训练所需的旧策略信息。
@dataclass
class RolloutSample:
    """一条 completion 及其训练所需的旧策略信息。"""

    # 逐行讲解：声明 tokens 的类型为 list[int]。
    tokens: list[int]
    # 逐行讲解：声明 logprobs 的类型为 list[float]。
    logprobs: list[float]
    # 逐行讲解：声明 text 的类型为 str。
    text: str
    # 逐行讲解：声明 reward 的类型为 RewardResult。
    reward: RewardResult
    # 逐行讲解：声明 stop_reason 的类型为 str | None。
    stop_reason: str | None
    # 逐行讲解：声明 advantage 的类型为 float；取值 0.0。
    advantage: float = 0.0


# 逐行讲解：定义类 RolloutGroup：同一道题的一组 completion。
@dataclass
class RolloutGroup:
    """同一道题的一组 completion。"""

    # 逐行讲解：声明 example 的类型为 MathExample。
    example: MathExample
    # 逐行讲解：声明 prompt_tokens 的类型为 list[int]。
    prompt_tokens: list[int]
    # 逐行讲解：声明 samples 的类型为 list[RolloutSample]。
    samples: list[RolloutSample]
    # 逐行讲解：声明 expected_group_size 的类型为 int。
    expected_group_size: int
    # 逐行讲解：声明 candidate_index 的类型为 int。
    candidate_index: int

    # 逐行讲解：定义函数 correct_count：封装下方逻辑，供调用方复用。
    @property
    def correct_count(self) -> int:
        # 逐行讲解：返回结果：sum((int(sample.reward.correct) for sample in self.samples))。
        return sum(int(sample.reward.correct) for sample in self.samples)

    # 逐行讲解：定义函数 usable：完整返回且每条 completion 都有 token，才允许进入训练。
    @property
    def usable(self) -> bool:
        """完整返回且每条 completion 都有 token，才允许进入训练。"""
        # 逐行讲解：返回结果：len(self.samples) == self.expected_group_size and all((sample.tokens for sample in self.samples))。
        return (
            len(self.samples) == self.expected_group_size
            and all(sample.tokens for sample in self.samples)
        )


# 逐行讲解：定义类 RolloutBatch：一个训练 step 的候选组和最终训练组。
@dataclass(frozen=True)
class RolloutBatch:
    """一个训练 step 的候选组和最终训练组。"""

    # 逐行讲解：声明 algorithm 的类型为 Algorithm。
    algorithm: Algorithm
    # 逐行讲解：声明 requested_groups 的类型为 int。
    requested_groups: int
    # 逐行讲解：声明 candidate_groups 的类型为 list[RolloutGroup]。
    candidate_groups: list[RolloutGroup]
    # 逐行讲解：声明 train_groups 的类型为 list[RolloutGroup]。
    train_groups: list[RolloutGroup]

    # 逐行讲解：定义函数 effective_group_ratio：封装下方逻辑，供调用方复用。
    @property
    def effective_group_ratio(self) -> float:
        # 逐行讲解：返回结果：len(self.train_groups) / max(len(self.candidate_groups), 1)。
        return len(self.train_groups) / max(len(self.candidate_groups), 1)

    # 逐行讲解：定义函数 effective_fill_ratio：实际有效组数量占目标训练组数量的比例。
    @property
    def effective_fill_ratio(self) -> float:
        """实际有效组数量占目标训练组数量的比例。"""
        # 逐行讲解：返回结果：len(self.train_groups) / max(self.requested_groups, 1)。
        return len(self.train_groups) / max(self.requested_groups, 1)

    # 逐行讲解：定义函数 oversample_ratio：封装下方逻辑，供调用方复用。
    @property
    def oversample_ratio(self) -> float:
        # 逐行讲解：返回结果：len(self.candidate_groups) / max(self.requested_groups, 1)。
        return len(self.candidate_groups) / max(self.requested_groups, 1)


# 逐行讲解：定义函数 build_prompt_tokens：用模型 chat template 构造单轮数学题 prompt。
def build_prompt_tokens(
    tokenizer: Any,
    question: str,
    *,
    max_prompt_tokens: int,
) -> list[int]:
    """用模型 chat template 构造单轮数学题 prompt。"""
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
    # 逐行讲解：检查条件 len(tokens) > max_prompt_tokens；成立时执行下方分支。
    if len(tokens) > max_prompt_tokens:
        # 逐行讲解：抛出异常：ValueError(f'prompt has {len(tokens)} tokens, exceeding {max_prompt_tokens}')。
        raise ValueError(
            f"prompt has {len(tokens)} tokens, exceeding {max_prompt_tokens}"
        )
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


# 逐行讲解：定义函数 assign_group_advantages：用 shaped reward 在完整 group 内计算标准化 advantage。
def assign_group_advantages(
    group: RolloutGroup,
    *,
    epsilon: float,
) -> None:
    """用 shaped reward 在完整 group 内计算标准化 advantage。"""
    # 逐行讲解：检查条件 not group.samples；成立时执行下方分支。
    if not group.samples:
        # 逐行讲解：返回，结束当前函数。
        return
    # 逐行讲解：通过推导式逐项生成结果：[sample.reward.shaped_reward for sample in group.samples]，赋给 rewards。
    rewards = [sample.reward.shaped_reward for sample in group.samples]
    # 逐行讲解：计算 sum(rewards) / len(rewards)，赋给 reward_mean。
    reward_mean = sum(rewards) / len(rewards)
    # 逐行讲解：检查条件 len(rewards) == 1；成立时执行下方分支。
    if len(rewards) == 1:
        # 逐行讲解：取值 0.0，赋给 reward_std。
        reward_std = 0.0
    else:
        # 对齐 verl 的 torch.std 默认行为：使用 N-1 分母的样本标准差。
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
        # 逐行讲解：计算 (sample.reward.shaped_reward - reward_mean) / denominator，赋给 sample.advantage。
        sample.advantage = (sample.reward.shaped_reward - reward_mean) / denominator


# 逐行讲解：定义函数 is_effective_group：Dynamic Sampling 只按原始正确性判断是否为非退化组。
def is_effective_group(group: RolloutGroup) -> bool:
    """Dynamic Sampling 只按原始正确性判断是否为非退化组。"""
    # 逐行讲解：返回结果：group.usable and 0 < group.correct_count < group.expected_group_size。
    return (
        group.usable
        and 0 < group.correct_count < group.expected_group_size
    )


# 逐行讲解：定义异步函数 sample_group_async：为一道题采样一个完整 group 并计算 reward/advantage。
async def sample_group_async(
    sampling_client: Any,
    tokenizer: Any,
    example: MathExample,
    config: RolloutConfig,
    *,
    candidate_index: int,
    enable_overlong: bool,
) -> RolloutGroup:
    """为一道题采样一个完整 group 并计算 reward/advantage。"""
    # 逐行讲解：调用 build_prompt_tokens（build_prompt_tokens(tokenizer, example.question, max_prompt_tokens=config.max_prompt_tokens)），赋给 prompt_tokens。
    prompt_tokens = build_prompt_tokens(
        tokenizer,
        example.question,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    # 逐行讲解：等待异步操作 sampling_client.sample_async(prompt=trio.ModelInput.from_ints(prompt_tokens), num_samples=config.group_size… 完成并取得结果，赋给 response。
    response = await sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=config.group_size,
        sampling_params=trio.SamplingParams(
            max_tokens=config.max_tokens,
            seed=config.seed + candidate_index,
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
        # 逐行讲解：调用 read_sequence（read_sequence(sequence, tokenizer)），赋给 (tokens, logprobs, text)。
        tokens, logprobs, text = read_sequence(sequence, tokenizer)
        # 逐行讲解：调用 score_answer（score_answer(text, example.answer, completion_tokens=len(tokens), max_tokens=config.max_tokens, overlong_ca…），赋给 reward。
        reward = score_answer(
            text,
            example.answer,
            completion_tokens=len(tokens),
            max_tokens=config.max_tokens,
            overlong_cache=config.overlong_cache,
            enable_overlong=enable_overlong,
        )
        # 逐行讲解：调用 getattr（getattr(sequence, 'stop_reason', None)），赋给 stop_reason。
        stop_reason = getattr(sequence, "stop_reason", None)
        # 逐行讲解：向列表末尾追加一个元素（samples.append(RolloutSample(tokens=tokens, logprobs=logprobs, text=text, reward=reward, stop_reason=str(st…）。
        samples.append(
            RolloutSample(
                tokens=tokens,
                logprobs=logprobs,
                text=text,
                reward=reward,
                stop_reason=str(stop_reason) if stop_reason is not None else None,
            )
        )

    # 逐行讲解：调用 RolloutGroup（RolloutGroup(example=example, prompt_tokens=prompt_tokens, samples=samples, expected_group_size=config.grou…），赋给 group。
    group = RolloutGroup(
        example=example,
        prompt_tokens=prompt_tokens,
        samples=samples,
        expected_group_size=config.group_size,
        candidate_index=candidate_index,
    )
    # 逐行讲解：调用 assign_group_advantages（assign_group_advantages(group, epsilon=config.advantage_epsilon)）。
    assign_group_advantages(group, epsilon=config.advantage_epsilon)
    # 逐行讲解：返回结果：group。
    return group


# 逐行讲解：定义异步函数 sample_groups_async：在设定并发度内采样多道题，并保持输入顺序。
async def sample_groups_async(
    sampling_client: Any,
    tokenizer: Any,
    examples: list[MathExample],
    config: RolloutConfig,
    *,
    candidate_start: int,
    enable_overlong: bool,
    progress_callback: Callable[[int], None] | None = None,
) -> list[RolloutGroup]:
    """在设定并发度内采样多道题，并保持输入顺序。"""
    # 逐行讲解：调用 asyncio.Semaphore（asyncio.Semaphore(config.concurrency)），赋给 semaphore。
    semaphore = asyncio.Semaphore(config.concurrency)

    # 逐行讲解：定义异步函数 sample_one：封装下方逻辑，供调用方复用。
    async def sample_one(offset: int, example: MathExample) -> RolloutGroup:
        # 逐行讲解：进入上下文管理器 semaphore，退出时执行其收尾逻辑。
        async with semaphore:
            # 逐行讲解：等待异步操作 sample_group_async(sampling_client, tokenizer, example, config, candidate_index=candidate_start + offset, e… 完成并取得结果，赋给 group。
            group = await sample_group_async(
                sampling_client,
                tokenizer,
                example,
                config,
                candidate_index=candidate_start + offset,
                enable_overlong=enable_overlong,
            )
            # 逐行讲解：检查条件 progress_callback is not None；成立时执行下方分支。
            if progress_callback is not None:
                # 逐行讲解：调用 progress_callback（progress_callback(1)）。
                progress_callback(1)
            # 逐行讲解：返回结果：group。
            return group

    # 逐行讲解：返回结果：list(await asyncio.gather(*(sample_one(offset, example) for offset, example in enumerate(examples))))。
    return list(
        await asyncio.gather(
            *(
                sample_one(offset, example)
                for offset, example in enumerate(examples)
            )
        )
    )


# 逐行讲解：定义异步函数 collect_rollout_batch：执行固定 GRPO batch，或在候选上限内补采 DAPO 有效组。
async def collect_rollout_batch(
    sampling_client: Any,
    tokenizer: Any,
    take_examples: Callable[[int], list[MathExample]],
    *,
    algorithm: Algorithm,
    requested_groups: int,
    config: RolloutConfig,
    max_candidate_groups: int,
    progress_callback: Callable[[int], None] | None = None,
    progress_total_callback: Callable[[int], None] | None = None,
) -> RolloutBatch:
    """执行固定 GRPO batch，或在候选上限内补采 DAPO 有效组。"""
    # 逐行讲解：调用 config.validate（config.validate()）。
    config.validate()
    # 逐行讲解：检查条件 algorithm not in {'grpo', 'dapo'}；成立时执行下方分支。
    if algorithm not in {"grpo", "dapo"}:
        # 逐行讲解：抛出异常：ValueError(f'unsupported algorithm: {algorithm}')。
        raise ValueError(f"unsupported algorithm: {algorithm}")
    # 逐行讲解：检查条件 requested_groups < 1；成立时执行下方分支。
    if requested_groups < 1:
        # 逐行讲解：抛出异常：ValueError('requested_groups must be >= 1')。
        raise ValueError("requested_groups must be >= 1")
    # 逐行讲解：检查条件 max_candidate_groups < requested_groups；成立时执行下方分支。
    if max_candidate_groups < requested_groups:
        # 逐行讲解：抛出异常：ValueError('max_candidate_groups must be >= requested_groups')。
        raise ValueError("max_candidate_groups must be >= requested_groups")

    # 逐行讲解：声明 candidate_groups 的类型为 list[RolloutGroup]；构造列表 []。
    candidate_groups: list[RolloutGroup] = []
    # 逐行讲解：声明 train_groups 的类型为 list[RolloutGroup]；构造列表 []。
    train_groups: list[RolloutGroup] = []
    # 逐行讲解：取值 algorithm == 'dapo'，赋给 enable_overlong。
    enable_overlong = algorithm == "dapo"

    # 逐行讲解：检查条件 algorithm == 'grpo'；成立时执行下方分支。
    if algorithm == "grpo":
        # 逐行讲解：调用 take_examples（take_examples(requested_groups)），赋给 examples。
        examples = take_examples(requested_groups)
        # 逐行讲解：检查条件 progress_total_callback is not None；成立时执行下方分支。
        if progress_total_callback is not None:
            # 逐行讲解：调用 progress_total_callback（progress_total_callback(len(examples))）。
            progress_total_callback(len(examples))
        # 逐行讲解：等待异步操作 sample_groups_async(sampling_client, tokenizer, examples, config, candidate_start=0, enable_overlong=False,… 完成并取得结果，赋给 candidate_groups。
        candidate_groups = await sample_groups_async(
            sampling_client,
            tokenizer,
            examples,
            config,
            candidate_start=0,
            enable_overlong=False,
            progress_callback=progress_callback,
        )
        # 逐行讲解：通过推导式逐项生成结果：[group for group in candidate_groups if is_effective_group(group)]，赋给 train_groups。
        train_groups = [
            group for group in candidate_groups if is_effective_group(group)
        ]
    else:
        # 逐行讲解：当 len(train_groups) < requested_groups 成立时继续循环。
        while len(train_groups) < requested_groups:
            # 逐行讲解：计算 max_candidate_groups - len(candidate_groups)，赋给 available。
            available = max_candidate_groups - len(candidate_groups)
            # 逐行讲解：检查条件 available <= 0；成立时执行下方分支。
            if available <= 0:
                # 候选预算耗尽后返回已经收集到的有效组：部分 batch 仍可训练，
                # 一个有效组都没有时由训练循环跳过本 step 的参数更新。
                break
            # 逐行讲解：调用 min（min(requested_groups - len(train_groups), available)），赋给 round_size。
            round_size = min(requested_groups - len(train_groups), available)
            # 逐行讲解：调用 take_examples（take_examples(round_size)），赋给 examples。
            examples = take_examples(round_size)
            # 逐行讲解：检查条件 progress_total_callback is not None；成立时执行下方分支。
            if progress_total_callback is not None:
                # 逐行讲解：调用 progress_total_callback（progress_total_callback(len(examples))）。
                progress_total_callback(len(examples))
            # 逐行讲解：等待异步操作 sample_groups_async(sampling_client, tokenizer, examples, config, candidate_start=len(candidate_groups), en… 完成并取得结果，赋给 new_groups。
            new_groups = await sample_groups_async(
                sampling_client,
                tokenizer,
                examples,
                config,
                candidate_start=len(candidate_groups),
                enable_overlong=enable_overlong,
                progress_callback=progress_callback,
            )
            # 逐行讲解：把一组元素依次加入列表（candidate_groups.extend(new_groups)）。
            candidate_groups.extend(new_groups)
            # 逐行讲解：把一组元素依次加入列表（train_groups.extend((group for group in new_groups if is_effective_group(group)))）。
            train_groups.extend(
                group for group in new_groups if is_effective_group(group)
            )

    # 逐行讲解：返回结果：RolloutBatch(algorithm=algorithm, requested_groups=requested_groups, candidate_groups=candidate_groups, tra…。
    return RolloutBatch(
        algorithm=algorithm,
        requested_groups=requested_groups,
        candidate_groups=candidate_groups,
        train_groups=train_groups,
    )
