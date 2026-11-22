"""PyTRIO 同步版 OPSD：固定 base teacher + privileged reference solution。

这个版本按数据流顺序展开，便于理解：

1. student 只看 ``problem``，用当前 LoRA 策略采样 completion；
2. fixed teacher 使用同一 base model，但额外看见 ``solution``；
3. teacher 对 student 的同一条 completion 计算逐 token logprob；
4. ``advantage = teacher_logprob - student_logprob``；
5. PyTRIO ``importance_sampling`` 更新 student。

测试:
uv run python 01-opsd-sync.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 1 \
    --max-tokens 1024 \
    --sample-size 100 \
    --save-every-steps 10 \
    --swanlab-mode disabled

默认每 25 step 同时保存可续训 state 和可评测 sampler weights。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：导入模块：random。
import random
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：导入模块：time。
import time
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：从 datasets 导入 Dataset, DatasetDict, load_from_disk。
from datasets import Dataset, DatasetDict, load_from_disk
# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm


# 逐行讲解：调用 trio.configure（trio.configure(request_timeout_seconds=1800)）。
trio.configure(request_timeout_seconds=1800)

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets' / 'openthoughts_math_30k_opsd'，赋给 DEFAULT_DATASET_PATH。
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "openthoughts_math_30k_opsd"
# 逐行讲解：取值 29434，赋给 EXPECTED_DATASET_ROWS。
EXPECTED_DATASET_ROWS = 29_434

# 逐行讲解：取值 'Please reason step by step, and put your final answer within \\boxed{}.'，赋给 STUDENT_INSTRUCTION。
STUDENT_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
# 逐行讲解：取值 "After reading the reference solution above, make sure you truly understand the reasoning behind each step …，赋给 TEACHER_TRANSITION。
TEACHER_TRANSITION = (
    "After reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:"
)


# 逐行讲解：定义类 PromptRollout：封装下方逻辑，供调用方复用。
@dataclass(frozen=True)
class PromptRollout:
    # 逐行讲解：声明 datums 的类型为 list[trio.Datum]。
    datums: list[trio.Datum]
    # 逐行讲解：声明 reverse_kls 的类型为 list[float]。
    reverse_kls: list[float]
    # 逐行讲解：声明 student_logprobs 的类型为 list[float]。
    student_logprobs: list[float]
    # 逐行讲解：声明 teacher_logprobs 的类型为 list[float]。
    teacher_logprobs: list[float]
    # 逐行讲解：声明 completion_token_counts 的类型为 list[int]。
    completion_token_counts: list[int]
    # 逐行讲解：声明 sample_text 的类型为 str | None。
    sample_text: str | None


# 逐行讲解：定义函数 model_slug：把基础模型名称转换为适合放进实验名的短标识。
def model_slug(base_model: str) -> str:
    """把基础模型名称转换为适合放进实验名的短标识。"""
    # 逐行讲解：调用 base_model.rsplit('/', 1)[-1].lower().replace（base_model.rsplit('/', 1)[-1].lower().replace('qwen3.5', 'qwen35')），赋给 name。
    name = base_model.rsplit("/", 1)[-1].lower().replace("qwen3.5", "qwen35")
    # 逐行讲解：返回结果：re.sub('[^a-z0-9]+', '-', name).strip('-')。
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


# 逐行讲解：定义函数 default_run_name：根据模型、目标函数和步数生成默认实验名称。
def default_run_name(args: argparse.Namespace) -> str:
    """根据模型、目标函数和步数生成默认实验名称。"""
    # 逐行讲解：按条件选择结果：'full' if args.steps == 0 else f'steps{args.steps}'，赋给 steps。
    steps = "full" if args.steps == 0 else f"steps{args.steps}"
    # 逐行讲解：返回结果：f'opsd-sync-{model_slug(args.base_model)}-sampled-token-{steps}'。
    return f"opsd-sync-{model_slug(args.base_model)}-sampled-token-{steps}"


# 逐行讲解：定义函数 parse_args：解析并校验同步 OPSD 的训练、采样、保存和日志参数。
def parse_args() -> argparse.Namespace:
    """解析并校验同步 OPSD 的训练、采样、保存和日志参数。"""
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-path', type=Path, default=DEFAULT_DATASET_PATH, help='00-datasets.py 保存的 Ope…）。
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="00-datasets.py 保存的 Openthoughts_math_30k_opsd 目录",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sample-size', type=int, default=0, help='随机抽样题数；0 表示全量')）。
    parser.add_argument(
        "--sample-size", type=int, default=0, help="随机抽样题数；0 表示全量"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B')）。
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=64)）。
    parser.add_argument("--lora-rank", type=int, default=64)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--train-unembed', action=argparse.BooleanOptionalAction, default=False, help='LoRA 是否训…）。
    parser.add_argument(
        "--train-unembed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="LoRA 是否训练 unembed；论文只训练 attention + MLP，默认 False",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--steps', type=int, default=100, help='训练 step；0 表示遍历当前数据一次')）。
    parser.add_argument(
        "--steps", type=int, default=100, help="训练 step；0 表示遍历当前数据一次"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=4, help='每个 step 的 problem 数')）。
    parser.add_argument(
        "--batch-size", type=int, default=4, help="每个 step 的 problem 数"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--group-size', type=int, default=1, help='每个 problem 的 completion 数')）。
    parser.add_argument(
        "--group-size", type=int, default=1, help="每个 problem 的 completion 数"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=1024)）。
    parser.add_argument("--max-tokens", type=int, default=1024)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.1)）。
    parser.add_argument("--temperature", type=float, default=1.1)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=0.95)）。
    parser.add_argument("--top-p", type=float, default=0.95)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-k', type=int, default=20)）。
    parser.add_argument("--top-k", type=int, default=20)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--student-thinking', action=argparse.BooleanOptionalAction, default=False, help='stude…）。
    parser.add_argument(
        "--student-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="student rollout 是否启用 thinking；算法复现默认 False",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--teacher-thinking', action=argparse.BooleanOptionalAction, default=False, help='privi…）。
    parser.add_argument(
        "--teacher-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="privileged teacher prompt 是否启用 thinking；算法复现默认 False",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--kl-penalty-coef', type=float, default=1.0)）。
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=5e-06)）。
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta1', type=float, default=0.9)）。
    parser.add_argument("--beta1", type=float, default=0.9)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta2', type=float, default=0.95)）。
    parser.add_argument("--beta2", type=float, default=0.95)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sampler-refresh-steps', type=int, default=1, help='刷新 student sampler 的间隔；1 才是严格 on-…）。
    parser.add_argument(
        "--sampler-refresh-steps",
        type=int,
        default=1,
        help="刷新 student sampler 的间隔；1 才是严格 on-policy",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every-steps', type=int, default=25, help='每 N step 保存 state + sampler weights；0…）。
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=25,
        help="每 N step 保存 state + sampler weights；0 仅保存最终 checkpoint",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--log-sample-every-steps', type=int, default=10, help='每 N step 向 SwanLab 记录一个 student…）。
    parser.add_argument(
        "--log-sample-every-steps",
        type=int,
        default=10,
        help="每 N step 向 SwanLab 记录一个 student completion；0 表示不记录文本",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--run-name', default=None, help='SwanLab 和 TRIO checkpoint 名称前缀')）。
    parser.add_argument(
        "--run-name", default=None, help="SwanLab 和 TRIO checkpoint 名称前缀"
    )

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab', action=argparse.BooleanOptionalAction, default=True)）。
    parser.add_argument(
        "--swanlab", action=argparse.BooleanOptionalAction, default=True
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-opsd')）。
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-opsd")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-workspace', default=None)）。
    parser.add_argument("--swanlab-workspace", default=None)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=['online', 'local', 'offline', 'disabled'], default=None)）。
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default=None,
    )
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()

    # 逐行讲解：遍历 ('batch_size', 'group_size', 'max_tokens', 'sampler_refresh_steps')，将每项绑定到 name。
    for name in ("batch_size", "group_size", "max_tokens", "sampler_refresh_steps"):
        # 逐行讲解：检查条件 getattr(args, name) < 1；成立时执行下方分支。
        if getattr(args, name) < 1:
            # 逐行讲解：抛出异常：ValueError(f"--{name.replace('_', '-')} must be >= 1")。
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    # 逐行讲解：遍历 ('steps', 'sample_size', 'save_every_steps', 'log_sample_every_steps')，将每项绑定到 name。
    for name in ("steps", "sample_size", "save_every_steps", "log_sample_every_steps"):
        # 逐行讲解：检查条件 getattr(args, name) < 0；成立时执行下方分支。
        if getattr(args, name) < 0:
            # 逐行讲解：抛出异常：ValueError(f"--{name.replace('_', '-')} must be >= 0")。
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    # 逐行讲解：检查条件 not 4 <= args.lora_rank <= 64；成立时执行下方分支。
    if not 4 <= args.lora_rank <= 64:
        # 逐行讲解：抛出异常：ValueError('--lora-rank must be between 4 and 64')。
        raise ValueError("--lora-rank must be between 4 and 64")
    # 逐行讲解：检查条件 args.kl_penalty_coef <= 0；成立时执行下方分支。
    if args.kl_penalty_coef <= 0:
        # 逐行讲解：抛出异常：ValueError('--kl-penalty-coef must be > 0')。
        raise ValueError("--kl-penalty-coef must be > 0")
    # 逐行讲解：检查条件 args.learning_rate <= 0；成立时执行下方分支。
    if args.learning_rate <= 0:
        # 逐行讲解：抛出异常：ValueError('--learning-rate must be > 0')。
        raise ValueError("--learning-rate must be > 0")
    # 逐行讲解：检查条件 not 0 < args.top_p <= 1；成立时执行下方分支。
    if not 0 < args.top_p <= 1:
        # 逐行讲解：抛出异常：ValueError('--top-p must be in (0, 1]')。
        raise ValueError("--top-p must be in (0, 1]")
    # 逐行讲解：检查条件 args.temperature < 0；成立时执行下方分支。
    if args.temperature < 0:
        # 逐行讲解：抛出异常：ValueError('--temperature must be >= 0')。
        raise ValueError("--temperature must be >= 0")

    # 逐行讲解：取值 args.run_name or default_run_name(args)，赋给 args.run_name。
    args.run_name = args.run_name or default_run_name(args)
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 load_training_dataset：读取本地 OPSD 数据，校验字段和行数后打乱或抽样。
def load_training_dataset(args: argparse.Namespace) -> Dataset:
    """读取本地 OPSD 数据，校验字段和行数后打乱或抽样。"""
    # 逐行讲解：检查条件 not args.dataset_path.exists()；成立时执行下方分支。
    if not args.dataset_path.exists():
        # 逐行讲解：抛出异常：FileNotFoundError(f'找不到 OPSD 数据：{args.dataset_path}\n请先运行：uv run python experiments/opsd/00-datasets.py --o…。
        raise FileNotFoundError(
            f"找不到 OPSD 数据：{args.dataset_path}\n"
            "请先运行：uv run python experiments/opsd/00-datasets.py --only opsd"
        )
    # 逐行讲解：调用 load_from_disk（load_from_disk(str(args.dataset_path))），赋给 loaded。
    loaded = load_from_disk(str(args.dataset_path))
    # 逐行讲解：按条件选择结果：loaded['train'] if isinstance(loaded, DatasetDict) else loaded，赋给 dataset。
    dataset = loaded["train"] if isinstance(loaded, DatasetDict) else loaded
    # 逐行讲解：检查条件 not isinstance(dataset, Dataset)；成立时执行下方分支。
    if not isinstance(dataset, Dataset):
        # 逐行讲解：抛出异常：TypeError(f'期望 Dataset，实际得到 {type(dataset)!r}')。
        raise TypeError(f"期望 Dataset，实际得到 {type(dataset)!r}")
    # 逐行讲解：调用 sorted（sorted({'problem', 'solution'} - set(dataset.column_names))），赋给 missing。
    missing = sorted({"problem", "solution"} - set(dataset.column_names))
    # 逐行讲解：检查条件 missing；成立时执行下方分支。
    if missing:
        # 逐行讲解：抛出异常：ValueError(f'OPSD 数据缺少字段 {missing}；实际字段为 {dataset.column_names}')。
        raise ValueError(
            f"OPSD 数据缺少字段 {missing}；实际字段为 {dataset.column_names}"
        )
    # 逐行讲解：检查条件 len(dataset) != EXPECTED_DATASET_ROWS；成立时执行下方分支。
    if len(dataset) != EXPECTED_DATASET_ROWS:
        # 逐行讲解：抛出异常：ValueError(f'官方 OPSD 数据应有 {EXPECTED_DATASET_ROWS:,} 条，实际为 {len(dataset):,} 条')。
        raise ValueError(
            f"官方 OPSD 数据应有 {EXPECTED_DATASET_ROWS:,} 条，实际为 {len(dataset):,} 条"
        )

    # 逐行讲解：打乱数据顺序（dataset.shuffle(seed=args.seed)），赋给 dataset。
    dataset = dataset.shuffle(seed=args.seed)
    # 逐行讲解：检查条件 args.sample_size > 0；成立时执行下方分支。
    if args.sample_size > 0:
        # 逐行讲解：按索引选择数据行（dataset.select(range(min(args.sample_size, len(dataset))))），赋给 dataset。
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))
    # 逐行讲解：检查条件 not dataset；成立时执行下方分支。
    if not dataset:
        # 逐行讲解：抛出异常：ValueError('训练数据为空')。
        raise ValueError("训练数据为空")
    # 逐行讲解：返回结果：dataset。
    return dataset


# 逐行讲解：定义函数 batch_for_step：按当前 step 选择一个 batch，固定步数模式下允许循环取样。
def batch_for_step(
    dataset: Dataset,
    step: int,
    batch_size: int,
    full_dataset_run: bool,
) -> Dataset:
    """按当前 step 选择一个 batch，固定步数模式下允许循环取样。"""
    # 逐行讲解：计算 step * batch_size，赋给 start。
    start = step * batch_size
    # 逐行讲解：检查条件 full_dataset_run；成立时执行下方分支。
    if full_dataset_run:
        # 逐行讲解：调用 list（list(range(start, min(start + batch_size, len(dataset))))），赋给 indices。
        indices = list(range(start, min(start + batch_size, len(dataset))))
    else:
        # 逐行讲解：通过推导式逐项生成结果：[(start + offset) % len(dataset) for offset in range(batch_size)]，赋给 indices。
        indices = [(start + offset) % len(dataset) for offset in range(batch_size)]
    # 逐行讲解：返回结果：dataset.select(indices)。
    return dataset.select(indices)


# 逐行讲解：定义函数 render_chat_prompt：应用 chat template 并把用户消息编码为 prompt token。
def render_chat_prompt(
    tokenizer: Any,
    user_message: str,
    enable_thinking: bool,
) -> list[int]:
    """应用 chat template 并把用户消息编码为 prompt token。"""
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template([{'role': 'user', 'content': user_message}], tokenize=False, add_generation_p…），赋给 rendered。
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    # 逐行讲解：将文本编码为 token（tokenizer.encode(rendered, add_special_tokens=False)），赋给 prompt_ids。
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    # 逐行讲解：检查条件 not prompt_ids；成立时执行下方分支。
    if not prompt_ids:
        # 逐行讲解：抛出异常：ValueError('chat prompt token 为空')。
        raise ValueError("chat prompt token 为空")
    # 逐行讲解：返回结果：prompt_ids。
    return prompt_ids


# 逐行讲解：定义函数 build_student_prompt_ids：构造只包含题目、不包含参考解答的 Student prompt。
def build_student_prompt_ids(
    tokenizer: Any, problem: str, enable_thinking: bool
) -> list[int]:
    """构造只包含题目、不包含参考解答的 Student prompt。"""
    # 逐行讲解：取值 f'Problem: {problem.strip()}\n\n{STUDENT_INSTRUCTION}'，赋给 user_message。
    user_message = f"Problem: {problem.strip()}\n\n{STUDENT_INSTRUCTION}"
    # 逐行讲解：返回结果：render_chat_prompt(tokenizer, user_message, enable_thinking)。
    return render_chat_prompt(tokenizer, user_message, enable_thinking)


# 逐行讲解：定义函数 build_teacher_prompt_ids：构造包含题目和特权参考解答的 Teacher prompt。
def build_teacher_prompt_ids(
    tokenizer: Any,
    problem: str,
    solution: str,
    enable_thinking: bool,
) -> list[int]:
    """构造包含题目和特权参考解答的 Teacher prompt。"""
    # 逐行讲解：取值 f'Problem: {problem.strip()}\n\nHere is a reference solution to this problem:\n=== Reference Solution Begin…，赋给 user_message。
    user_message = (
        f"Problem: {problem.strip()}\n\n"
        "Here is a reference solution to this problem:\n"
        "=== Reference Solution Begin ===\n"
        f"{solution.strip()}\n"
        "=== Reference Solution End ===\n\n\n"
        f"{TEACHER_TRANSITION}\n\n"
        f"{STUDENT_INSTRUCTION}"
    )
    # 逐行讲解：返回结果：render_chat_prompt(tokenizer, user_message, enable_thinking)。
    return render_chat_prompt(tokenizer, user_message, enable_thinking)


# 逐行讲解：定义函数 teacher_completion_logprobs：计算 Teacher 对 Student 实际 completion 的逐 token logprob。
def teacher_completion_logprobs(
    teacher_client: Any,
    teacher_prompt_ids: list[int],
    completion_ids: list[int],
) -> list[float]:
    """计算 Teacher 对 Student 实际 completion 的逐 token logprob。"""
    # 逐行讲解：计算 teacher_prompt_ids + completion_ids，赋给 all_ids。
    all_ids = teacher_prompt_ids + completion_ids
    # 逐行讲解：等待任务完成并读取结果（teacher_client.compute_logprobs(trio.ModelInput.from_ints(all_ids)).result()），赋给 all_logprobs。
    all_logprobs = teacher_client.compute_logprobs(
        trio.ModelInput.from_ints(all_ids)
    ).result()
    # 逐行讲解：按索引或切片读取 all_logprobs[len(teacher_prompt_ids):]，赋给 completion_logprobs。
    completion_logprobs = all_logprobs[len(teacher_prompt_ids) :]
    # 逐行讲解：检查条件 len(completion_logprobs) != len(completion_ids)；成立时执行下方分支。
    if len(completion_logprobs) != len(completion_ids):
        # 逐行讲解：抛出异常：ValueError(f'Teacher token/logprob 长度不一致：{len(completion_ids)} != {len(completion_logprobs)}')。
        raise ValueError(
            "Teacher token/logprob 长度不一致："
            f"{len(completion_ids)} != {len(completion_logprobs)}"
        )
    # 逐行讲解：检查条件 any((value is None for value in completion_logprobs))；成立时执行下方分支。
    if any(value is None for value in completion_logprobs):
        # 逐行讲解：抛出异常：ValueError('Teacher completion logprob 中存在 None')。
        raise ValueError("Teacher completion logprob 中存在 None")
    # 逐行讲解：返回结果：[float(value) for value in completion_logprobs]。
    return [float(value) for value in completion_logprobs]


# 逐行讲解：定义函数 build_opd_datum：右移并对齐轨迹字段，构造 importance_sampling 所需 Datum。
def build_opd_datum(
    student_prompt_ids: list[int],
    completion_ids: list[int],
    old_logprobs: list[float],
    advantages: np.ndarray,
) -> trio.Datum:
    """右移并对齐轨迹字段，构造 importance_sampling 所需 Datum。"""
    # 逐行讲解：检查条件 not completion_ids；成立时执行下方分支。
    if not completion_ids:
        # 逐行讲解：抛出异常：ValueError('completion 不能为空')。
        raise ValueError("completion 不能为空")
    # 逐行讲解：检查条件 len(completion_ids) != len(old_logprobs) or len(completion_ids) != len(advantages)；成立时执行下方分支。
    if len(completion_ids) != len(old_logprobs) or len(completion_ids) != len(
        advantages
    ):
        # 逐行讲解：抛出异常：ValueError('completion、old_logprobs、advantages 长度必须一致')。
        raise ValueError("completion、old_logprobs、advantages 长度必须一致")

    # 逐行讲解：计算 len(student_prompt_ids) - 1，赋给 prompt_loss_len。
    prompt_loss_len = len(student_prompt_ids) - 1
    # 逐行讲解：计算 student_prompt_ids + completion_ids[:-1]，赋给 input_ids。
    input_ids = student_prompt_ids + completion_ids[:-1]
    # 逐行讲解：计算 [0] * prompt_loss_len + completion_ids，赋给 target_ids。
    target_ids = [0] * prompt_loss_len + completion_ids
    # 逐行讲解：计算 [0.0] * prompt_loss_len + old_logprobs，赋给 padded_logprobs。
    padded_logprobs = [0.0] * prompt_loss_len + old_logprobs
    # 逐行讲解：计算 [0.0] * prompt_loss_len + advantages.tolist()，赋给 padded_advantages。
    padded_advantages = [0.0] * prompt_loss_len + advantages.tolist()
    # 逐行讲解：检查条件 not len(input_ids) == len(target_ids) == len(padded_logprobs) == len(padded_advantages)；成立时执行下方分支。
    if not (
        len(input_ids)
        == len(target_ids)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        # 逐行讲解：抛出异常：ValueError('OPSD Datum 字段长度不一致')。
        raise ValueError("OPSD Datum 字段长度不一致")

    # 逐行讲解：返回结果：trio.Datum(model_input=trio.ModelInput.from_ints(input_ids), loss_fn_inputs={'target_tokens': np.asarray(ta…。
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_ids, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )


# 逐行讲解：定义函数 run_prompt_rollout：完成单题 Student 采样、Teacher 打分和 OPD Datum 构造。
def run_prompt_rollout(
    student_sampler: Any,
    teacher_client: Any,
    tokenizer: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    sampling_params: trio.SamplingParams,
) -> PromptRollout:
    """完成单题 Student 采样、Teacher 打分和 OPD Datum 构造。"""
    # 逐行讲解：调用 str(row['problem']).strip（str(row['problem']).strip()），赋给 problem。
    problem = str(row["problem"]).strip()
    # 逐行讲解：调用 str(row['solution']).strip（str(row['solution']).strip()），赋给 solution。
    solution = str(row["solution"]).strip()
    # 逐行讲解：检查条件 not problem or not solution；成立时执行下方分支。
    if not problem or not solution:
        # 逐行讲解：抛出异常：ValueError('OPSD row 的 problem/solution 不能为空')。
        raise ValueError("OPSD row 的 problem/solution 不能为空")

    # 逐行讲解：调用 build_student_prompt_ids（build_student_prompt_ids(tokenizer, problem, args.student_thinking)），赋给 student_prompt_ids。
    student_prompt_ids = build_student_prompt_ids(
        tokenizer,
        problem,
        args.student_thinking,
    )
    # 逐行讲解：调用 build_teacher_prompt_ids（build_teacher_prompt_ids(tokenizer, problem, solution, args.teacher_thinking)），赋给 teacher_prompt_ids。
    teacher_prompt_ids = build_teacher_prompt_ids(
        tokenizer,
        problem,
        solution,
        args.teacher_thinking,
    )
    # 逐行讲解：调用 student_sampler.sample(prompt=trio.ModelInput.from_ints(student_prompt_ids), num_samples=args.group_size, s…（student_sampler.sample(prompt=trio.ModelInput.from_ints(student_prompt_ids), num_samples=args.group_size, s…），赋给 sample_result。
    sample_result = student_sampler.sample(
        prompt=trio.ModelInput.from_ints(student_prompt_ids),
        num_samples=args.group_size,
        sampling_params=sampling_params,
        return_text=False,
    ).result()

    # 逐行讲解：声明 datums 的类型为 list[trio.Datum]；构造列表 []。
    datums: list[trio.Datum] = []
    # 逐行讲解：声明 reverse_kls 的类型为 list[float]；构造列表 []。
    reverse_kls: list[float] = []
    # 逐行讲解：声明 student_logprobs 的类型为 list[float]；构造列表 []。
    student_logprobs: list[float] = []
    # 逐行讲解：声明 teacher_logprobs 的类型为 list[float]；构造列表 []。
    teacher_logprobs: list[float] = []
    # 逐行讲解：声明 completion_token_counts 的类型为 list[int]；构造列表 []。
    completion_token_counts: list[int] = []
    # 逐行讲解：取值 None，赋给 sample_text。
    sample_text = None

    # 逐行讲解：遍历 sample_result.sequences，将每项绑定到 sequence。
    for sequence in sample_result.sequences:
        # 逐行讲解：调用 list（list(sequence.tokens)），赋给 completion_ids。
        completion_ids = list(sequence.tokens)
        # 逐行讲解：检查条件 not completion_ids；成立时执行下方分支。
        if not completion_ids:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：检查条件 len(sequence.logprobs) != len(completion_ids) or any((value is None for value in sequence.logprobs))；成立时执行下方分支。
        if len(sequence.logprobs) != len(completion_ids) or any(
            value is None for value in sequence.logprobs
        ):
            # 逐行讲解：抛出异常：ValueError('Student completion token/logprob 无法一一对齐')。
            raise ValueError("Student completion token/logprob 无法一一对齐")
        # 逐行讲解：通过推导式逐项生成结果：[float(value) for value in sequence.logprobs]，赋给 student_lps。
        student_lps = [float(value) for value in sequence.logprobs]
        # 逐行讲解：调用 teacher_completion_logprobs（teacher_completion_logprobs(teacher_client, teacher_prompt_ids, completion_ids)），赋给 teacher_lps。
        teacher_lps = teacher_completion_logprobs(
            teacher_client,
            teacher_prompt_ids,
            completion_ids,
        )

        # 这是 KL(pi_student || pi_teacher) 的 sampled-token Monte Carlo 项。
        reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)
        # 逐行讲解：计算 -args.kl_penalty_coef * reverse_kl，赋给 advantages。
        advantages = -args.kl_penalty_coef * reverse_kl
        # 逐行讲解：向列表末尾追加一个元素（datums.append(build_opd_datum(student_prompt_ids, completion_ids, student_lps, advantages))）。
        datums.append(
            build_opd_datum(
                student_prompt_ids,
                completion_ids,
                student_lps,
                advantages,
            )
        )
        # 逐行讲解：把一组元素依次加入列表（reverse_kls.extend(reverse_kl.tolist())）。
        reverse_kls.extend(reverse_kl.tolist())
        # 逐行讲解：把一组元素依次加入列表（student_logprobs.extend(student_lps)）。
        student_logprobs.extend(student_lps)
        # 逐行讲解：把一组元素依次加入列表（teacher_logprobs.extend(teacher_lps)）。
        teacher_logprobs.extend(teacher_lps)
        # 逐行讲解：向列表末尾追加一个元素（completion_token_counts.append(len(completion_ids))）。
        completion_token_counts.append(len(completion_ids))
        # 逐行讲解：检查条件 sample_text is None；成立时执行下方分支。
        if sample_text is None:
            # 逐行讲解：将 token 解码为文本（tokenizer.decode(completion_ids, skip_special_tokens=False)），赋给 sample_text。
            sample_text = tokenizer.decode(completion_ids, skip_special_tokens=False)

    # 逐行讲解：返回结果：PromptRollout(datums=datums, reverse_kls=reverse_kls, student_logprobs=student_logprobs, teacher_logprobs=t…。
    return PromptRollout(
        datums=datums,
        reverse_kls=reverse_kls,
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        completion_token_counts=completion_token_counts,
        sample_text=sample_text,
    )


# 逐行讲解：定义函数 numeric_trainer_metrics：筛选 PyTRIO 返回的数值指标并添加 trainer 命名空间。
def numeric_trainer_metrics(result: Any) -> dict[str, float]:
    """筛选 PyTRIO 返回的数值指标并添加 trainer 命名空间。"""
    # 逐行讲解：声明 metrics 的类型为 dict[str, float]；构造字典 {}。
    metrics: dict[str, float] = {}
    # 逐行讲解：遍历 dict(result.metrics).items()，将每项绑定到 (key, value)。
    for key, value in dict(result.metrics).items():
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：转换为浮点表示（float(value)），赋给 metrics[f'trainer/{key}']。
            metrics[f"trainer/{key}"] = float(value)
        except (TypeError, ValueError):
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
    # 逐行讲解：返回结果：metrics。
    return metrics


# 逐行讲解：定义函数 start_swanlab：按配置创建 SwanLab 实验，并记录本次训练的静态元信息。
def start_swanlab(
    args: argparse.Namespace,
    dataset_size: int,
    total_steps: int,
) -> Any | None:
    """按配置创建 SwanLab 实验，并记录本次训练的静态元信息。"""
    # 逐行讲解：检查条件 not args.swanlab；成立时执行下方分支。
    if not args.swanlab:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：调用 vars(args).copy（vars(args).copy()），赋给 config。
    config = vars(args).copy()
    # 逐行讲解：调用 str（str(args.dataset_path)），赋给 config['dataset_path']。
    config["dataset_path"] = str(args.dataset_path)
    # 逐行讲解：取值 dataset_size，赋给 config['dataset_size']。
    config["dataset_size"] = dataset_size
    # 逐行讲解：取值 total_steps，赋给 config['total_steps']。
    config["total_steps"] = total_steps
    # 逐行讲解：取值 'fixed base model with privileged solution'，赋给 config['teacher']。
    config["teacher"] = "fixed base model with privileged solution"
    # 逐行讲解：取值 'sampled-token reverse KL via importance_sampling'，赋给 config['objective']。
    config["objective"] = "sampled-token reverse KL via importance_sampling"
    # 逐行讲解：返回结果：swanlab.init(project=args.swanlab_project, name=args.run_name, workspace=args.swanlab_workspace, mode=args.…。
    return swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=["PyTRIO", "OPSD", "OpenThoughts", "fixed-teacher", "sync"],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


# 逐行讲解：定义函数 save_checkpoint：同时保存可续训 state 和可用于评测的 sampler weights。
def save_checkpoint(
    training_client: Any,
    swanlab_run: Any | None,
    run_name: str,
    completed_steps: int,
) -> tuple[str, str]:
    """同时保存可续训 state 和可用于评测的 sampler weights。"""
    # 逐行讲解：取值 f'step{completed_steps:06d}'，赋给 tag。
    tag = f"step{completed_steps:06d}"
    # 逐行讲解：取值 f'{run_name}-{tag}'，赋给 name。
    name = f"{run_name}-{tag}"

    # Train state 包含模型和优化器；sampler weights 用于推理与 AIME25 评测。
    state_result = training_client.save_state(name=name).result()
    # 逐行讲解：等待任务完成并读取结果（training_client.save_weights_for_sampler(name=name).result()），赋给 weights_result。
    weights_result = training_client.save_weights_for_sampler(name=name).result()
    # 逐行讲解：调用 print（print(f'Saved state [{tag}]: {state_result.path}')）。
    print(f"Saved state [{tag}]: {state_result.path}")
    # 逐行讲解：调用 print（print(f'Saved sampler weights [{tag}]: {weights_result.path}')）。
    print(f"Saved sampler weights [{tag}]: {weights_result.path}")
    # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
    if swanlab_run is not None:
        # 逐行讲解：计算自然对数（swanlab.log({'checkpoint/step': completed_steps, 'checkpoint/state_path': swanlab.Text(state_result.path), …）。
        swanlab.log(
            {
                "checkpoint/step": completed_steps,
                "checkpoint/state_path": swanlab.Text(state_result.path),
                "checkpoint/sampler_weights_path": swanlab.Text(weights_result.path),
            },
            step=completed_steps,
        )
    # 逐行讲解：返回结果：(state_result.path, weights_result.path)。
    return state_result.path, weights_result.path


# 逐行讲解：定义函数 finish_swanlab：根据正常结束、中断或异常设置 SwanLab 运行状态。
def finish_swanlab(swanlab_run: Any | None, error: BaseException | None) -> None:
    """根据正常结束、中断或异常设置 SwanLab 运行状态。"""
    # 逐行讲解：检查条件 swanlab_run is None；成立时执行下方分支。
    if swanlab_run is None:
        # 逐行讲解：返回，结束当前函数。
        return
    # 逐行讲解：检查条件 error is None；成立时执行下方分支。
    if error is None:
        # 逐行讲解：调用 swanlab.finish（swanlab.finish()）。
        swanlab.finish()
    # 逐行讲解：检查条件 isinstance(error, KeyboardInterrupt)；成立时执行下方分支。
    elif isinstance(error, KeyboardInterrupt):
        # 逐行讲解：调用 swanlab.finish（swanlab.finish(state='aborted')）。
        swanlab.finish(state="aborted")
    else:
        # 逐行讲解：调用 swanlab.finish（swanlab.finish(state='crashed', error=str(error))）。
        swanlab.finish(state="crashed", error=str(error))


# 逐行讲解：定义函数 train：执行同步 OPSD 主循环，并定期记录指标和保存双份权重。
def train(args: argparse.Namespace) -> None:
    """执行同步 OPSD 主循环，并定期记录指标和保存双份权重。"""
    # 固定本地随机源；数据集 shuffle、远程训练和采样也会复用同一个 seed。
    random.seed(args.seed)
    # 逐行讲解：设置对应随机数生成器的种子（np.random.seed(args.seed)）。
    np.random.seed(args.seed)
    # 逐行讲解：调用 load_training_dataset（load_training_dataset(args)），赋给 dataset。
    dataset = load_training_dataset(args)

    # steps=0 表示完整遍历一次数据，否则严格执行用户指定的 step 数。
    full_dataset_run = args.steps == 0
    # 逐行讲解：按条件选择结果：(len(dataset) + args.batch_size - 1) // args.batch_size if full_dataset_run else args.steps，赋给 total_steps。
    total_steps = (
        (len(dataset) + args.batch_size - 1) // args.batch_size
        if full_dataset_run
        else args.steps
    )
    # 逐行讲解：调用 print（print(f'Loaded OPSD examples: {len(dataset):,}')）。
    print(f"Loaded OPSD examples: {len(dataset):,}")
    # 逐行讲解：调用 print（print(f'Training steps: {total_steps}')）。
    print(f"Training steps: {total_steps}")
    # 逐行讲解：调用 print（print(f'Run name: {args.run_name}')）。
    print(f"Run name: {args.run_name}")
    # 逐行讲解：调用 print（print('Teacher: fixed base model; privileged field: solution')）。
    print("Teacher: fixed base model; privileged field: solution")
    # 逐行讲解：调用 print（print('Objective: sampled-token reverse KL (PyTRIO importance_sampling)')）。
    print("Objective: sampled-token reverse KL (PyTRIO importance_sampling)")

    # ServiceClient 是创建远程训练客户端和采样客户端的统一入口。
    service_client = trio.ServiceClient()
    # 逐行讲解：取值 None，赋给 swanlab_run。
    swanlab_run = None
    # 逐行讲解：声明 caught_error 的类型为 BaseException | None；取值 None。
    caught_error: BaseException | None = None
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # Student 只训练 LoRA；attention 和 MLP 默认参与训练，unembed 可选。
        training_client = service_client.create_lora_training_client(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
            train_attn=True,
            train_mlp=True,
            train_unembed=args.train_unembed,
        )
        # 不传 model_path，teacher 始终是 step-0 base policy，不随 student LoRA 更新。
        teacher_client = service_client.create_sampling_client(
            base_model=args.base_model
        )
        # 逐行讲解：调用 teacher_client.get_tokenizer（teacher_client.get_tokenizer()），赋给 tokenizer。
        tokenizer = teacher_client.get_tokenizer()

        # 所有 Student rollout 共用这一组采样参数。
        sampling_params = trio.SamplingParams(
            max_tokens=args.max_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop=list(
                dict.fromkeys(
                    token for token in (tokenizer.eos_token, "<|im_end|>") if token
                )
            ),
        )
        # 每个 step 的所有 Datum 聚合后，只执行一次 optimizer update。
        adam = trio.AdamParams(
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
        )
        # 逐行讲解：调用 start_swanlab（start_swanlab(args, len(dataset), total_steps)），赋给 swanlab_run。
        swanlab_run = start_swanlab(args, len(dataset), total_steps)
        # 逐行讲解：取值 None，赋给 student_sampler。
        student_sampler = None
        # 逐行讲解：声明 saved_steps 的类型为 set[int]；调用 set（set()）。
        saved_steps: set[int] = set()

        # 逐行讲解：遍历 range(total_steps)，将每项绑定到 step。
        for step in range(total_steps):
            # 逐行讲解：调用 time.time（time.time()），赋给 step_start。
            step_start = time.time()

            # 刷新 sampler 权重；间隔为 1 时，每步 rollout 都来自最新 Student。
            if student_sampler is None or step % args.sampler_refresh_steps == 0:
                # 逐行讲解：保存当前权重并创建对应采样客户端（training_client.save_weights_and_get_sampling_client()），赋给 student_sampler。
                student_sampler = training_client.save_weights_and_get_sampling_client()

            # 从训练开始时已经打乱的数据集中，按 step 取出当前 problem batch。
            batch = batch_for_step(
                dataset,
                step,
                args.batch_size,
                full_dataset_run,
            )

            # 同步版逐题执行：Student 采样 -> 固定 Teacher 打分 -> 构造 OPD Datum。
            rollouts = [
                run_prompt_rollout(
                    student_sampler,
                    teacher_client,
                    tokenizer,
                    row,
                    args,
                    sampling_params,
                )
                for row in tqdm(
                    batch, desc=f"OPSD sync step {step + 1}", unit="problem"
                )
            ]

            # 将题目级结果展平，组成一次远程训练更新及 SwanLab 聚合指标。
            datums = [datum for rollout in rollouts for datum in rollout.datums]
            # 逐行讲解：通过推导式逐项生成结果：[value for rollout in rollouts for value in rollout.reverse_kls]，赋给 reverse_kls。
            reverse_kls = [
                value for rollout in rollouts for value in rollout.reverse_kls
            ]
            # 逐行讲解：通过推导式逐项生成结果：[value for rollout in rollouts for value in rollout.student_logprobs]，赋给 student_lps。
            student_lps = [
                value for rollout in rollouts for value in rollout.student_logprobs
            ]
            # 逐行讲解：通过推导式逐项生成结果：[value for rollout in rollouts for value in rollout.teacher_logprobs]，赋给 teacher_lps。
            teacher_lps = [
                value for rollout in rollouts for value in rollout.teacher_logprobs
            ]
            # 逐行讲解：通过推导式逐项生成结果：[value for rollout in rollouts for value in rollout.completion_token_counts]，赋给 completion_counts。
            completion_counts = [
                value
                for rollout in rollouts
                for value in rollout.completion_token_counts
            ]
            # 逐行讲解：检查条件 not datums；成立时执行下方分支。
            if not datums:
                # 逐行讲解：抛出异常：RuntimeError('本 step 没有生成有效 OPSD Datum')。
                raise RuntimeError("本 step 没有生成有效 OPSD Datum")

            # 先提交 forward/backward 和 optimizer，再依次等待两个远程任务完成。
            fwd_bwd_future = training_client.forward_backward(
                datums,
                loss_fn="importance_sampling",
            )
            # 逐行讲解：提交优化器更新（training_client.optim_step(adam)），赋给 optim_future。
            optim_future = training_client.optim_step(adam)
            # 逐行讲解：等待任务完成并读取结果（fwd_bwd_future.result()），赋给 fwd_bwd_result。
            fwd_bwd_result = fwd_bwd_future.result()
            # 逐行讲解：等待任务完成并读取结果（optim_future.result()）。
            optim_future.result()

            # 汇总当前 step 的训练、OPSD 差异和整体耗时指标。
            completed_steps = step + 1
            # 逐行讲解：计算 time.time() - step_start，赋给 elapsed。
            elapsed = time.time() - step_start
            # 逐行讲解：调用 int（int(sum(completion_counts))），赋给 completion_tokens_total。
            completion_tokens_total = int(sum(completion_counts))
            # 逐行讲解：声明 metrics 的类型为 dict[str, Any]；构造字典 {'train/step': completed_steps, 'train/learning_rate': args.learning_rate, 'data/prompts': len(batch), 'dat…。
            metrics: dict[str, Any] = {
                "train/step": completed_steps,
                "train/learning_rate": args.learning_rate,
                "data/prompts": len(batch),
                "data/datums": len(datums),
                "data/completion_tokens_mean": float(np.mean(completion_counts)),
                "data/completion_tokens_total": completion_tokens_total,
                "opd/reverse_kl_mean": float(np.mean(reverse_kls)),
                "opd/reverse_kl_std": float(np.std(reverse_kls)),
                "opd/advantage_mean": float(
                    -args.kl_penalty_coef * np.mean(reverse_kls)
                ),
                "opd/student_logprob_mean": float(np.mean(student_lps)),
                "opd/teacher_logprob_mean": float(np.mean(teacher_lps)),
                "time/step_elapsed_time": elapsed,
            }
            # 逐行讲解：调用 metrics.update（metrics.update(numeric_trainer_metrics(fwd_bwd_result))）。
            metrics.update(numeric_trainer_metrics(fwd_bwd_result))

            # 按配置抽取一条 Student completion，避免每步上传大量文本。
            if (
                args.log_sample_every_steps > 0
                and completed_steps % args.log_sample_every_steps == 0
            ):
                # 逐行讲解：调用 next（next((rollout.sample_text for rollout in rollouts if rollout.sample_text), None)），赋给 sample_text。
                sample_text = next(
                    (
                        rollout.sample_text
                        for rollout in rollouts
                        if rollout.sample_text
                    ),
                    None,
                )
                # 逐行讲解：检查条件 sample_text is not None；成立时执行下方分支。
                if sample_text is not None:
                    # 逐行讲解：调用 swanlab.Text（swanlab.Text(sample_text)），赋给 metrics['sample/student_completion']。
                    metrics["sample/student_completion"] = swanlab.Text(sample_text)

            # SwanLab 写入放在远程训练完成之后，确保 step 和权重更新严格对齐。
            if swanlab_run is not None:
                # 逐行讲解：计算自然对数（swanlab.log(metrics, step=completed_steps)）。
                swanlab.log(metrics, step=completed_steps)

            # 逐行讲解：调用 print（print(f"step {completed_steps:03d}/{total_steps} | datums {len(datums)} | avg_tokens {metrics['data/complet…）。
            print(
                f"step {completed_steps:03d}/{total_steps} | datums {len(datums)} | "
                f"avg_tokens {metrics['data/completion_tokens_mean']:.1f} | "
                f"reverse_kl {metrics['opd/reverse_kl_mean']:.4f} | "
                f"time {elapsed:.2f}s"
            )

            # 定期同时保存可续训 state 和可直接评测的 sampler weights。
            if (
                args.save_every_steps > 0
                and completed_steps % args.save_every_steps == 0
            ):
                # 逐行讲解：调用 save_checkpoint（save_checkpoint(training_client, swanlab_run, args.run_name, completed_steps)）。
                save_checkpoint(
                    training_client,
                    swanlab_run,
                    args.run_name,
                    completed_steps,
                )
                # 逐行讲解：调用 saved_steps.add（saved_steps.add(completed_steps)）。
                saved_steps.add(completed_steps)

        # 如果最后一步不在定期保存点上，额外保存最终双份 checkpoint。
        if total_steps not in saved_steps:
            # 逐行讲解：调用 save_checkpoint（save_checkpoint(training_client, swanlab_run, args.run_name, total_steps)）。
            save_checkpoint(
                training_client,
                swanlab_run,
                args.run_name,
                total_steps,
            )
        # 逐行讲解：调用 print（print(f'Completed {total_steps} OPSD steps')）。
        print(f"Completed {total_steps} OPSD steps")
    except BaseException as error:
        # 逐行讲解：取值 error，赋给 caught_error。
        caught_error = error
        # 逐行讲解：抛出异常：重新抛出当前异常。
        raise
    finally:
        # 正常结束、中断或异常时都正确关闭 SwanLab run。
        finish_swanlab(swanlab_run, caught_error)


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 time.time（time.time()），赋给 started。
    started = time.time()
    # 逐行讲解：调用 train（train(parse_args())）。
    train(parse_args())
    # 逐行讲解：调用 print（print(f'All done in {time.time() - started:.2f}s')）。
    print(f"All done in {time.time() - started:.2f}s")
