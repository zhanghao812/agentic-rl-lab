"""PyTRIO 同步版 GRPO 教学脚本（只有 importance_sampling / ppo 两个内置 loss）。

GRPO 训练流程：
1. 从 GSM8K 取一批数学题；
2. 用当前 LoRA 权重保存出采样客户端；
3. 每道题同步采样 group_size 个答案；
4. 用 boxed answer reward 计算 group-relative advantage；
5. 用 PyTRIO 内置 loss 做一次优化。

两个 loss 的区别（都是 policy gradient + importance sampling 修正）：

- importance_sampling：
      loss = -ratio * advantage
  其中 ratio = exp(当前策略 logprob - 采样时旧策略 logprob)。
  不做任何截断，新旧策略偏差大时训练容易不稳定。

- ppo：
      loss = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
  在 importance_sampling 基础上把 ratio 截断到 [1-eps, 1+eps]，
  防止一次更新把策略推得太远。eps 等参数由 PyTRIO 内置 ppo loss 管理。

最小试跑：

uv run python 03-grpo.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled

正式训练：

uv run python 03-grpo.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode online
"""

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：导入模块：time。
import time
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：从 datasets 导入 Dataset, load_dataset。
from datasets import Dataset, load_dataset
# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm


# 逐行讲解：取值 'Provide a numerical answer without units, written inside \\boxed{}.'，赋给 QUESTION_SUFFIX。
QUESTION_SUFFIX = "Provide a numerical answer without units, written inside \\boxed{}."
# 逐行讲解：构造元组 ('importance_sampling', 'ppo')，赋给 LOSS_FNS。
LOSS_FNS = ("importance_sampling", "ppo")
# 逐行讲解：构造列表 [{'role': 'user', 'content': "How many r's are in strawberry?" + QUESTION_SUFFIX}, {'role': 'assistant', 'c…，赋给 FEWSHOT_PREFIX。
FEWSHOT_PREFIX = [
    {"role": "user", "content": "How many r's are in strawberry?" + QUESTION_SUFFIX},
    {
        "role": "assistant",
        "content": (
            "<think>\n\n</think>\n\n"
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. "
            "There are three r's. \\boxed{3}"
        ),
    },
]


# 逐行讲解：定义类 GRPOConfig：命令行参数解析后的训练配置。
@dataclass
class GRPOConfig:
    """命令行参数解析后的训练配置。"""

    # 逐行讲解：声明 base_model 的类型为 str。
    base_model: str
    # 逐行讲解：声明 lora_rank 的类型为 int。
    lora_rank: int
    # 逐行讲解：声明 steps 的类型为 int。
    steps: int
    # 逐行讲解：声明 all_data 的类型为 bool。
    all_data: bool
    # 逐行讲解：声明 batch_size 的类型为 int。
    batch_size: int
    # 逐行讲解：声明 group_size 的类型为 int。
    group_size: int
    # 逐行讲解：声明 max_tokens 的类型为 int。
    max_tokens: int
    # 逐行讲解：声明 temperature 的类型为 float。
    temperature: float
    # 逐行讲解：声明 top_p 的类型为 float。
    top_p: float
    # 逐行讲解：声明 seed 的类型为 int。
    seed: int
    # 逐行讲解：声明 learning_rate 的类型为 float。
    learning_rate: float
    # 逐行讲解：声明 beta1 的类型为 float。
    beta1: float
    # 逐行讲解：声明 beta2 的类型为 float。
    beta2: float
    # 逐行讲解：声明 loss_fn 的类型为 str。
    loss_fn: str
    # 逐行讲解：声明 swanlab_mode 的类型为 str。
    swanlab_mode: str
    # 逐行讲解：声明 swanlab_project 的类型为 str。
    swanlab_project: str


# 逐行讲解：定义类 RolloutSample：一条采样结果，以及构造 importance_sampling 所需的旧策略 logprobs。
@dataclass
class RolloutSample:
    """一条采样结果，以及构造 importance_sampling 所需的旧策略 logprobs。"""

    # 逐行讲解：声明 tokens 的类型为 list[int]。
    tokens: list[int]
    # 逐行讲解：声明 logprobs 的类型为 list[float]。
    logprobs: list[float]
    # 逐行讲解：声明 text 的类型为 str。
    text: str
    # 逐行讲解：声明 reward 的类型为 float。
    reward: float
    # 逐行讲解：声明 advantage 的类型为 float。
    advantage: float


# 逐行讲解：定义函数 parse_args：把训练配置集中到命令行参数，避免依赖环境变量。
def parse_args() -> GRPOConfig:
    """把训练配置集中到命令行参数，避免依赖环境变量。"""
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='PyTRIO 同步版 GRPO / GSM8K 教学脚本')），赋给 parser。
    parser = argparse.ArgumentParser(description="PyTRIO 同步版 GRPO / GSM8K 教学脚本")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B', help='PyTRIO 基础模型名')）。
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B", help="PyTRIO 基础模型名")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=32, help='LoRA rank')）。
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--steps', type=int, default=10, help='GRPO 优化步数；每步从 GSM8K 取 batch-size 道题做 rollout')）。
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="GRPO 优化步数；每步从 GSM8K 取 batch-size 道题做 rollout",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--all-data', action='store_true', help='使用 GSM8K train split 全量数据训练一遍；打开后忽略 --steps')）。
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="使用 GSM8K train split 全量数据训练一遍；打开后忽略 --steps",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=4, help='每个 step 的 GSM8K 题目数')）。
    parser.add_argument("--batch-size", type=int, default=4, help="每个 step 的 GSM8K 题目数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--group-size', type=int, default=4, help='每道题采样的 completion 数')）。
    parser.add_argument("--group-size", type=int, default=4, help="每道题采样的 completion 数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=1024, help='每次采样最多生成 token 数')）。
    parser.add_argument("--max-tokens", type=int, default=1024, help="每次采样最多生成 token 数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0, help='采样 temperature')）。
    parser.add_argument("--temperature", type=float, default=1.0, help="采样 temperature")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0, help='采样 top_p')）。
    parser.add_argument("--top-p", type=float, default=1.0, help="采样 top_p")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42, help='本地随机种子')）。
    parser.add_argument("--seed", type=int, default=42, help="本地随机种子")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=4e-05, help='Adam learning rate')）。
    parser.add_argument("--learning-rate", type=float, default=4e-5, help="Adam learning rate")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')）。
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')）。
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--loss-fn', choices=LOSS_FNS, default='importance_sampling', help='训练 loss：importance_…）。
    parser.add_argument(
        "--loss-fn",
        choices=LOSS_FNS,
        default="importance_sampling",
        help="训练 loss：importance_sampling / ppo",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=('online', 'disabled'), default='online', help='SwanLab 记录模式：…）。
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "disabled"),
        default="online",
        help="SwanLab 记录模式：online 上传到云端，disabled 完全关闭日志",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-grpo-tashan', help='SwanLab project')）。
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-grpo-tashan", help="SwanLab project")
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()

    # 逐行讲解：返回结果：GRPOConfig(base_model=args.base_model, lora_rank=args.lora_rank, steps=args.steps, all_data=args.all_data, …。
    return GRPOConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        steps=args.steps,
        all_data=args.all_data,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        loss_fn=args.loss_fn,
        swanlab_mode=args.swanlab_mode,
        swanlab_project=args.swanlab_project,
    )


# 逐行讲解：定义函数 extract_boxed：取最后一个 \boxed{...} 作为模型最终答案。
def extract_boxed(text: str) -> str | None:
    """取最后一个 \\boxed{...} 作为模型最终答案。"""
    # 逐行讲解：调用 re.findall（re.findall('\\\\boxed\\{([^}]+)\\}', text)），赋给 matches。
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    # 逐行讲解：检查条件 not matches；成立时执行下方分支。
    if not matches:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：matches[-1].strip()。
    return matches[-1].strip()


# 逐行讲解：定义函数 normalize_answer：GSM8K 答案只做轻量归一化，避免 1,000 和 1000 被判成不同。
def normalize_answer(text: str) -> str:
    """GSM8K 答案只做轻量归一化，避免 1,000 和 1000 被判成不同。"""
    # 逐行讲解：返回结果：text.replace(',', '').strip().rstrip('.')。
    return text.replace(",", "").strip().rstrip(".")


# 逐行讲解：定义函数 grade_answer：boxed answer 与标准答案完全一致时给 1，否则给 0。
def grade_answer(response: str, ground_truth: str) -> float:
    """boxed answer 与标准答案完全一致时给 1，否则给 0。"""
    # 逐行讲解：调用 extract_boxed（extract_boxed(response)），赋给 answer。
    answer = extract_boxed(response)
    # 逐行讲解：检查条件 answer is None；成立时执行下方分支。
    if answer is None:
        # 逐行讲解：返回结果：0.0。
        return 0.0
    # 逐行讲解：返回结果：1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0。
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0


# 逐行讲解：定义函数 extract_gsm8k_answer：GSM8K 的最终答案位于 `####` 后面。
def extract_gsm8k_answer(answer_text: str) -> str:
    """GSM8K 的最终答案位于 `####` 后面。"""
    # 逐行讲解：调用 re.search（re.search('####\\s*(.+)', answer_text)），赋给 match。
    match = re.search(r"####\s*(.+)", answer_text)
    # 逐行讲解：检查条件 match is None；成立时执行下方分支。
    if match is None:
        # 逐行讲解：抛出异常：ValueError(f'No GSM8K final answer found: {answer_text!r}')。
        raise ValueError(f"No GSM8K final answer found: {answer_text!r}")
    # 逐行讲解：返回结果：normalize_answer(match.group(1))。
    return normalize_answer(match.group(1))


# 逐行讲解：定义函数 build_prompt：把 few-shot + 当前题目渲染成模型输入 tokens。
def build_prompt(tokenizer: Any, question: str) -> list[int]:
    """把 few-shot + 当前题目渲染成模型输入 tokens。"""
    # 逐行讲解：构造列表 [*FEWSHOT_PREFIX, {'role': 'user', 'content': question + QUESTION_SUFFIX}]，赋给 messages。
    messages = [
        *FEWSHOT_PREFIX,
        {"role": "user", "content": question + QUESTION_SUFFIX},
    ]
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)），赋给 prompt_text。
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # 逐行讲解：将文本编码为 token（tokenizer.encode(prompt_text, add_special_tokens=False)），赋给 prompt_tokens。
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    # 逐行讲解：检查条件 not prompt_tokens；成立时执行下方分支。
    if not prompt_tokens:
        # 逐行讲解：抛出异常：ValueError('Prompt tokens are empty')。
        raise ValueError("Prompt tokens are empty")
    # 逐行讲解：返回结果：prompt_tokens。
    return prompt_tokens


# 逐行讲解：定义函数 load_gsm8k_train：加载 GSM8K 训练集；首次运行会由 datasets 下载缓存。
def load_gsm8k_train() -> Dataset:
    """加载 GSM8K 训练集；首次运行会由 datasets 下载缓存。"""
    # 逐行讲解：调用 load_dataset（load_dataset('openai/gsm8k', 'main', split='train')），赋给 dataset。
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    # 逐行讲解：检查条件 not isinstance(dataset, Dataset)；成立时执行下方分支。
    if not isinstance(dataset, Dataset):
        # 逐行讲解：抛出异常：TypeError(f'Expected Dataset, got {type(dataset)!r}')。
        raise TypeError(f"Expected Dataset, got {type(dataset)!r}")
    # 逐行讲解：返回结果：dataset。
    return dataset


# 逐行讲解：定义函数 get_stop_sequences：采样停止符尽量贴近 chat template，同时避免 None 和重复项。
def get_stop_sequences(tokenizer: Any) -> list[str]:
    """采样停止符尽量贴近 chat template，同时避免 None 和重复项。"""
    # 逐行讲解：构造列表 [tokenizer.eos_token, '<|im_end|>']，赋给 candidates。
    candidates = [tokenizer.eos_token, "<|im_end|>"]
    # 逐行讲解：返回结果：list(dict.fromkeys([token for token in candidates if token]))。
    return list(dict.fromkeys([token for token in candidates if token]))


# 逐行讲解：定义函数 run_rollout_group：同步采样一个 prompt 的 group_size 个 completion，并计算组内 advantage。
def run_rollout_group(
    sampling_client: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    ground_truth: str,
    sampling_params: trio.SamplingParams,
    group_size: int,
) -> list[RolloutSample]:
    """同步采样一个 prompt 的 group_size 个 completion，并计算组内 advantage。"""
    # 对同一个 prompt 一次采样 group_size 个回答；同步 API 返回 future，所以这里立刻 `.result()` 等待。
    result = sampling_client.sample(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=group_size,
        sampling_params=sampling_params,
        return_text=True,
    ).result()

    # 先把本组内每个回答的 token、旧策略 logprob、文本和 reward 收集起来。
    # advantage 需要等整组 reward 都算完后，减去组内均值才能得到。
    rewards: list[float] = []
    # 逐行讲解：声明 raw_samples 的类型为 list[tuple[list[int], list[float], str]]；构造列表 []。
    raw_samples: list[tuple[list[int], list[float], str]] = []

    # 逐行讲解：遍历 result.sequences，将每项绑定到 sequence。
    for sequence in result.sequences:
        # return_text=True 时通常会直接返回文本；如果没有文本，就用 tokenizer 从 token 解码。
        text = sequence.text
        # 逐行讲解：检查条件 text is None；成立时执行下方分支。
        if text is None:
            # 逐行讲解：将 token 解码为文本（tokenizer.decode(sequence.tokens, skip_special_tokens=True)），赋给 text。
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)

        # PyTRIO 采样返回 completion token 的 logprobs；
        # 后续 importance_sampling / ppo loss 需要用这组 logprobs 作为“采样时旧策略”的概率。
        tokens = list(sequence.tokens)
        # 逐行讲解：通过推导式逐项生成结果：[float(value) for value in sequence.logprobs]，赋给 logprobs。
        logprobs = [float(value) for value in sequence.logprobs]
        # 逐行讲解：检查条件 len(tokens) != len(logprobs)；成立时执行下方分支。
        if len(tokens) != len(logprobs):
            # 逐行讲解：抛出异常：ValueError(f'Generated token/logprob length mismatch: {len(tokens)} != {len(logprobs)}')。
            raise ValueError(
                f"Generated token/logprob length mismatch: {len(tokens)} != {len(logprobs)}"
            )

        # reward 只看模型回答里最后一个 \boxed{}，和 GSM8K 标准答案一致则为 1，否则为 0。
        reward = grade_answer(text, ground_truth)
        # 逐行讲解：向列表末尾追加一个元素（rewards.append(reward)）。
        rewards.append(reward)
        # 逐行讲解：向列表末尾追加一个元素（raw_samples.append((tokens, logprobs, text))）。
        raw_samples.append((tokens, logprobs, text))

    # GRPO 的核心是组内相对优势：同一道题里，比平均 reward 高的回答得到正 advantage。
    mean_reward = sum(rewards) / len(rewards)
    # 逐行讲解：返回结果：[RolloutSample(tokens=tokens, logprobs=logprobs, text=text, reward=reward, advantage=reward - mean_reward) …。
    return [
        RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            text=text,
            reward=reward,
            advantage=reward - mean_reward,
        )
        for (tokens, logprobs, text), reward in zip(raw_samples, rewards, strict=True)
    ]


# 逐行讲解：定义函数 build_grpo_datum：把单条 completion 转成 PyTRIO importance_sampling / ppo 所需的 Datum。
def build_grpo_datum(prompt_tokens: list[int], sample: RolloutSample) -> trio.Datum:
    """把单条 completion 转成 PyTRIO importance_sampling / ppo 所需的 Datum。"""
    # 逐行讲解：检查条件 not sample.tokens；成立时执行下方分支。
    if not sample.tokens:
        # 逐行讲解：抛出异常：ValueError('Cannot train on an empty completion')。
        raise ValueError("Cannot train on an empty completion")

    # 自回归对齐方式如下：
    # input = prompt + completion[:-1]
    # target 前 observation_len 个位置属于 prompt 内部预测，不训练，用 0 / 0.0 占位；
    # 从最后一个 prompt token 开始预测 completion 的每个 token。
    observation_len = len(prompt_tokens) - 1
    # 逐行讲解：计算 prompt_tokens + sample.tokens[:-1]，赋给 input_tokens。
    input_tokens = prompt_tokens + sample.tokens[:-1]
    # 逐行讲解：计算 [0] * observation_len + sample.tokens，赋给 target_tokens。
    target_tokens = [0] * observation_len + sample.tokens
    # 逐行讲解：计算 [0.0] * observation_len + sample.logprobs，赋给 padded_logprobs。
    padded_logprobs = [0.0] * observation_len + sample.logprobs
    # 逐行讲解：计算 [0.0] * observation_len + [sample.advantage] * len(sample.tokens)，赋给 padded_advantages。
    padded_advantages = [0.0] * observation_len + [sample.advantage] * len(sample.tokens)

    # 逐行讲解：检查条件 not len(input_tokens) == len(target_tokens) == len(padded_logprobs) == len(padded_advantages)；成立时执行下方分支。
    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        # 逐行讲解：抛出异常：ValueError('GRPO datum fields must have the same token length')。
        raise ValueError("GRPO datum fields must have the same token length")

    # 逐行讲解：返回结果：trio.Datum(model_input=trio.ModelInput.from_ints(input_tokens), loss_fn_inputs={'target_tokens': np.asarray…。
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )


# 逐行讲解：定义函数 get_num_steps：计算实际训练 step 数；all-data 模式会覆盖命令行里的 steps。
def get_num_steps(dataset: Dataset, config: GRPOConfig) -> int:
    """计算实际训练 step 数；all-data 模式会覆盖命令行里的 steps。"""
    # 逐行讲解：检查条件 config.all_data；成立时执行下方分支。
    if config.all_data:
        # 逐行讲解：返回结果：(len(dataset) + config.batch_size - 1) // config.batch_size。
        return (len(dataset) + config.batch_size - 1) // config.batch_size
    # 逐行讲解：返回结果：config.steps。
    return config.steps


# 逐行讲解：定义函数 model_slug：把模型名压成适合 experiment / weights name 的短名称。
def model_slug(base_model: str) -> str:
    """把模型名压成适合 experiment / weights name 的短名称。"""
    # 逐行讲解：调用 base_model.rsplit('/', 1)[-1].lower（base_model.rsplit('/', 1)[-1].lower()），赋给 name。
    name = base_model.rsplit("/", 1)[-1].lower()
    # 逐行讲解：调用 name.replace（name.replace('qwen3.5', 'qwen35')），赋给 name。
    name = name.replace("qwen3.5", "qwen35")
    # 逐行讲解：返回结果：re.sub('[^a-z0-9]+', '-', name).strip('-')。
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


# 逐行讲解：定义函数 build_run_name：SwanLab experiment name 和最终权重名共用同一个规则生成值。
def build_run_name(config: GRPOConfig, effective_steps: int) -> str:
    """SwanLab experiment name 和最终权重名共用同一个规则生成值。"""
    # 逐行讲解：调用 config.loss_fn.replace（config.loss_fn.replace('_', '-')），赋给 loss_slug。
    loss_slug = config.loss_fn.replace("_", "-")
    # 逐行讲解：返回结果：f'grpo-sync-{model_slug(config.base_model)}-gsm8k-{loss_slug}-steps{effective_steps}'。
    return (
        f"grpo-sync-{model_slug(config.base_model)}-gsm8k-"
        f"{loss_slug}-steps{effective_steps}"
    )


# 逐行讲解：定义函数 pick_batch：取当前 step 的 batch；all-data 模式下不回绕，确保每条样本最多用一次。
def pick_batch(dataset: Dataset, step: int, batch_size: int, all_data: bool) -> Dataset:
    """取当前 step 的 batch；all-data 模式下不回绕，确保每条样本最多用一次。"""
    # 逐行讲解：计算 step * batch_size，赋给 start。
    start = step * batch_size
    # 逐行讲解：检查条件 all_data；成立时执行下方分支。
    if all_data:
        # 逐行讲解：调用 min（min(start + batch_size, len(dataset))），赋给 end。
        end = min(start + batch_size, len(dataset))
        # 逐行讲解：调用 list（list(range(start, end))），赋给 indices。
        indices = list(range(start, end))
    else:
        # 非 all-data 模式保留原来的回绕逻辑，允许 steps 超过数据集可切出的完整 batch 数。
        indices = [(start + offset) % len(dataset) for offset in range(batch_size)]
    # 逐行讲解：返回结果：dataset.select(indices)。
    return dataset.select(indices)


# 逐行讲解：定义函数 init_swanlab_run：SwanLab 只记录关键 GRPO 指标，不影响主训练逻辑。
def init_swanlab_run(
    config: GRPOConfig,
    effective_steps: int,
    dataset_size: int,
    run_name: str,
) -> Any | None:
    """SwanLab 只记录关键 GRPO 指标，不影响主训练逻辑。"""
    # 逐行讲解：检查条件 config.swanlab_mode == 'disabled'；成立时执行下方分支。
    if config.swanlab_mode == "disabled":
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：swanlab.init(project=config.swanlab_project, name=run_name, mode=config.swanlab_mode, config={'base_model':…。
    return swanlab.init(
        project=config.swanlab_project,
        name=run_name,
        mode=config.swanlab_mode,
        config={
            "base_model": config.base_model,
            "run_name": run_name,
            "lora_rank": config.lora_rank,
            "steps": config.steps,
            "all_data": config.all_data,
            "effective_steps": effective_steps,
            "batch_size": config.batch_size,
            "dataset_size": dataset_size,
            "group_size": config.group_size,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "learning_rate": config.learning_rate,
            "beta1": config.beta1,
            "beta2": config.beta2,
            "loss_fn": config.loss_fn,
            "swanlab_mode": config.swanlab_mode,
            "seed": config.seed,
            "weights_name": run_name,
        },
        log_dir=str(Path(__file__).resolve().parent / "swanlog"),
    )


# 逐行讲解：定义函数 get_numeric_metric：按当前 PyTRIO 实际返回的 metric key 精确取值，不做模糊匹配。
def get_numeric_metric(loss_metrics: dict[str, Any], key: str) -> float | None:
    """按当前 PyTRIO 实际返回的 metric key 精确取值，不做模糊匹配。"""
    # 逐行讲解：查询指定键，缺失时使用默认值（loss_metrics.get(key)），赋给 value。
    value = loss_metrics.get(key)
    # 逐行讲解：检查条件 isinstance(value, int | float)；成立时执行下方分支。
    if isinstance(value, int | float):
        # 逐行讲解：返回结果：float(value)。
        return float(value)
    # 逐行讲解：返回结果：None。
    return None


# 逐行讲解：定义函数 count_train_tokens：统计非零 advantage 的 token 数，也就是实际参与 RL loss 的 completion token。
def count_train_tokens(datums: list[trio.Datum]) -> int:
    """统计非零 advantage 的 token 数，也就是实际参与 RL loss 的 completion token。"""
    # 逐行讲解：取值 0，赋给 total。
    total = 0
    # 逐行讲解：遍历 datums，将每项绑定到 datum。
    for datum in datums:
        # 逐行讲解：更新 total：total += sum((1 for value in datum.loss_fn_inputs['advantages'].data if float(value) != 0.0))。
        total += sum(
            1
            for value in datum.loss_fn_inputs["advantages"].data
            if float(value) != 0.0
        )
    # 逐行讲解：返回结果：total。
    return total


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main(config: GRPOConfig) -> None:
    # 逐行讲解：设置对应随机数生成器的种子（np.random.seed(config.seed)）。
    np.random.seed(config.seed)

    # 逐行讲解：调用 print（print('Loading GSM8K dataset...')）。
    print("Loading GSM8K dataset...")
    # 逐行讲解：调用 load_gsm8k_train（load_gsm8k_train()），赋给 train_data。
    train_data = load_gsm8k_train()
    # 逐行讲解：调用 print（print(f'Loaded {len(train_data)} GSM8K training examples')）。
    print(f"Loaded {len(train_data)} GSM8K training examples")
    # 逐行讲解：调用 get_num_steps（get_num_steps(train_data, config)），赋给 effective_steps。
    effective_steps = get_num_steps(train_data, config)
    # 逐行讲解：调用 build_run_name（build_run_name(config, effective_steps)），赋给 run_name。
    run_name = build_run_name(config, effective_steps)
    # 逐行讲解：调用 print（print(f'Run / weights name: {run_name}')）。
    print(f"Run / weights name: {run_name}")
    # 逐行讲解：检查条件 config.all_data；成立时执行下方分支。
    if config.all_data:
        # 逐行讲解：调用 print（print(f'All-data mode: {effective_steps} steps will cover {len(train_data)} examples once')）。
        print(
            f"All-data mode: {effective_steps} steps will cover "
            f"{len(train_data)} examples once"
        )

    # 逐行讲解：调用 print（print('Creating PyTRIO clients...')）。
    print("Creating PyTRIO clients...")
    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：调用 service_client.create_lora_training_client（service_client.create_lora_training_client(base_model=config.base_model, rank=config.lora_rank)），赋给 training_client。
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
    )
    # 逐行讲解：调用 training_client.get_tokenizer（training_client.get_tokenizer()），赋给 tokenizer。
    tokenizer = training_client.get_tokenizer()

    # 逐行讲解：调用 trio.SamplingParams（trio.SamplingParams(max_tokens=config.max_tokens, temperature=config.temperature, top_p=config.top_p, stop=…），赋给 sampling_params。
    sampling_params = trio.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=get_stop_sequences(tokenizer),
    )
    # 逐行讲解：调用 trio.AdamParams（trio.AdamParams(learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2)），赋给 adam_params。
    adam_params = trio.AdamParams(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
    )
    # 逐行讲解：调用 init_swanlab_run（init_swanlab_run(config=config, effective_steps=effective_steps, dataset_size=len(train_data), run_name=run…），赋给 swanlab_run。
    swanlab_run = init_swanlab_run(
        config=config,
        effective_steps=effective_steps,
        dataset_size=len(train_data),
        run_name=run_name,
    )

    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：遍历 range(effective_steps)，将每项绑定到 step。
        for step in range(effective_steps):
            # 逐行讲解：调用 pick_batch（pick_batch(train_data, step, config.batch_size, config.all_data)），赋给 batch_rows。
            batch_rows = pick_batch(
                train_data,
                step,
                config.batch_size,
                config.all_data,
            )

            # 采样必须使用当前策略，所以每个 step 先保存临时匿名 LoRA 权重并创建 sampler。
            sampling_client = training_client.save_weights_and_get_sampling_client()

            # 逐行讲解：声明 datums 的类型为 list[trio.Datum]；构造列表 []。
            datums: list[trio.Datum] = []
            # 逐行讲解：声明 prompt_mean_rewards 的类型为 list[float]；构造列表 []。
            prompt_mean_rewards: list[float] = []
            # 逐行讲解：声明 rollout_lengths 的类型为 list[int]；构造列表 []。
            rollout_lengths: list[int] = []
            # 逐行讲解：取值 0，赋给 n_degenerate。
            n_degenerate = 0

            # 逐行讲解：遍历 tqdm(batch_rows, desc=f'GRPO step {step}', unit='prompt')，将每项绑定到 row。
            for row in tqdm(batch_rows, desc=f"GRPO step {step}", unit="prompt"):
                # 逐行讲解：调用 build_prompt（build_prompt(tokenizer, row['question'])），赋给 prompt_tokens。
                prompt_tokens = build_prompt(tokenizer, row["question"])
                # 逐行讲解：调用 extract_gsm8k_answer（extract_gsm8k_answer(row['answer'])），赋给 ground_truth。
                ground_truth = extract_gsm8k_answer(row["answer"])
                # 逐行讲解：调用 run_rollout_group（run_rollout_group(sampling_client=sampling_client, tokenizer=tokenizer, prompt_tokens=prompt_tokens, ground…），赋给 rollout_samples。
                rollout_samples = run_rollout_group(
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens,
                    ground_truth=ground_truth,
                    sampling_params=sampling_params,
                    group_size=config.group_size,
                )

                # 逐行讲解：把一组元素依次加入列表（rollout_lengths.extend((len(sample.tokens) for sample in rollout_samples))）。
                rollout_lengths.extend(len(sample.tokens) for sample in rollout_samples)
                # 逐行讲解：通过推导式逐项生成结果：[sample.reward for sample in rollout_samples]，赋给 rewards。
                rewards = [sample.reward for sample in rollout_samples]
                # 逐行讲解：计算 sum(rewards) / len(rewards)，赋给 prompt_mean_reward。
                prompt_mean_reward = sum(rewards) / len(rewards)
                # 逐行讲解：向列表末尾追加一个元素（prompt_mean_rewards.append(prompt_mean_reward)）。
                prompt_mean_rewards.append(prompt_mean_reward)

                # 同一题 group 内 reward 完全一样时，advantage 全为 0，没有训练信号，直接跳过。
                if all(sample.advantage == 0.0 for sample in rollout_samples):
                    # 逐行讲解：更新 n_degenerate：n_degenerate += 1。
                    n_degenerate += 1
                    # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                    continue

                # 逐行讲解：遍历 rollout_samples，将每项绑定到 sample。
                for sample in rollout_samples:
                    # 逐行讲解：向列表末尾追加一个元素（datums.append(build_grpo_datum(prompt_tokens, sample))）。
                    datums.append(build_grpo_datum(prompt_tokens, sample))

            # 逐行讲解：检查条件 datums；成立时执行下方分支。
            if datums:
                # importance_sampling / ppo 都是 PyTRIO 内置 loss，
                # datum 只需要 target_tokens（当前模型 forward 用）+
                # logprobs（采样时旧策略）+ advantages（GRPO 组内相对优势）。
                fwd_bwd_future = training_client.forward_backward(
                    datums,
                    loss_fn=config.loss_fn,
                )

                # 同步版 PyTRIO：提交远程前向/反向和优化器更新后，显式 `.result()` 等待完成。
                optim_future = training_client.optim_step(adam_params)
                # 逐行讲解：等待任务完成并读取结果（fwd_bwd_future.result()），赋给 fwd_bwd_result。
                fwd_bwd_result = fwd_bwd_future.result()
                # 逐行讲解：等待任务完成并读取结果（optim_future.result()）。
                optim_future.result()
                # 逐行讲解：调用 dict（dict(fwd_bwd_result.metrics)），赋给 loss_metrics。
                loss_metrics = dict(fwd_bwd_result.metrics)
            else:
                # 逐行讲解：构造字典 {}，赋给 loss_metrics。
                loss_metrics = {}

            # 逐行讲解：调用 get_numeric_metric（get_numeric_metric(loss_metrics, 'loss_mean')），赋给 loss_mean。
            loss_mean = get_numeric_metric(loss_metrics, "loss_mean")
            # 逐行讲解：调用 count_train_tokens（count_train_tokens(datums)），赋给 train_tokens。
            train_tokens = count_train_tokens(datums)
            # 逐行讲解：计算 sum(prompt_mean_rewards) / len(prompt_mean_rewards)，赋给 mean_reward。
            mean_reward = sum(prompt_mean_rewards) / len(prompt_mean_rewards)
            # 逐行讲解：按条件选择结果：sum(rollout_lengths) / len(rollout_lengths) if rollout_lengths else 0.0，赋给 avg_gen_len。
            avg_gen_len = (
                sum(rollout_lengths) / len(rollout_lengths)
                if rollout_lengths
                else 0.0
            )
            # 退化 group 指同一道题的所有回答 reward 都一样，advantage 全为 0，没有相对优劣信号。
            # 这个比例越高，说明当前 batch 里真正用于 GRPO 学习的题目越少。
            frac_degenerate = n_degenerate / len(prompt_mean_rewards)
            # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
            if swanlab_run is not None:
                # 逐行讲解：构造字典 {'reward': mean_reward, 'frac_degenerate': frac_degenerate, 'rollout/avg_gen_len': avg_gen_len, 'datums': l…，赋给 log_payload。
                log_payload = {
                    "reward": mean_reward,
                    "frac_degenerate": frac_degenerate,
                    "rollout/avg_gen_len": avg_gen_len,
                    "datums": len(datums),
                    "train_tokens": train_tokens,
                    **{
                        f"trainer/{key}": value
                        for key, value in loss_metrics.items()
                        if key == "loss_mean"
                    },
                }
                # 逐行讲解：检查条件 loss_mean is not None；成立时执行下方分支。
                if loss_mean is not None:
                    # 逐行讲解：取值 loss_mean，赋给 log_payload['loss']。
                    log_payload["loss"] = loss_mean
                    # 逐行讲解：取值 loss_mean，赋给 log_payload['loss_mean']。
                    log_payload["loss_mean"] = loss_mean
                # 逐行讲解：计算自然对数（swanlab.log(log_payload, step=step)）。
                swanlab.log(log_payload, step=step)

            # 逐行讲解：按条件选择结果：'n/a' if loss_mean is None else f'{loss_mean:.4f}'，赋给 loss_mean_text。
            loss_mean_text = "n/a" if loss_mean is None else f"{loss_mean:.4f}"
            # 逐行讲解：调用 print（print(f'Step {step:2d} | reward: {mean_reward:.3f} | degenerate: {frac_degenerate:.0%} | datums: {len(datum…）。
            print(
                f"Step {step:2d} | reward: {mean_reward:.3f} | "
                f"degenerate: {frac_degenerate:.0%} | datums: {len(datums)} | "
                f"train_tokens: {train_tokens} | "
                f"avg_gen_len: {avg_gen_len:.1f} | "
                f"loss_mean: {loss_mean_text} | loss_fn: {config.loss_fn}"
            )

        # 逐行讲解：调用 print（print('Saving final LoRA weights for sampler...')）。
        print("Saving final LoRA weights for sampler...")
        # 逐行讲解：等待任务完成并读取结果（training_client.save_weights_for_sampler(name=run_name).result()），赋给 final_weights。
        final_weights = training_client.save_weights_for_sampler(
            name=run_name
        ).result()
        # 逐行讲解：调用 print（print(f'Saved weights name: {run_name}, path: {final_weights.path}')）。
        print(f"Saved weights name: {run_name}, path: {final_weights.path}")
    finally:
        # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
        if swanlab_run is not None:
            # 逐行讲解：调用 swanlab_run.finish（swanlab_run.finish()）。
            swanlab_run.finish()


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 cli_config。
    cli_config = parse_args()
    # 逐行讲解：调用 time.time（time.time()），赋给 start_main_time。
    start_main_time = time.time()
    # 逐行讲解：调用 main（main(cli_config)）。
    main(cli_config)
    # 逐行讲解：调用 time.time（time.time()），赋给 end_main_time。
    end_main_time = time.time()
    # 逐行讲解：调用 print（print('#' * 50)）。
    print("#" * 50)
    # 逐行讲解：调用 print（print('# all done')）。
    print("# all done")
    # 逐行讲解：调用 print（print(f'# train cost {end_main_time - start_main_time:.2f}s')）。
    print(f"# train cost {end_main_time - start_main_time:.2f}s")
    # 逐行讲解：调用 print（print('#' * 50)）。
    print("#" * 50)
