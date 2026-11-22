"""异步版 OPD demo：ModelScope DeepMath-103K + PyTRIO + SwanLab。

核心逻辑是 on-policy distillation：
student 先采样，teacher 对 student 采样轨迹算 logprob，
reverse_kl = student_logprob - teacher_logprob，再用 -reverse_kl 做 advantage。

小成本试跑：
uv run python 02-demo-async.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 4 \
    --max-tokens 512 \
    --sample-size 100 \
    --swanlab-mode disabled
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：random。
import random
# 逐行讲解：导入模块：shutil。
import shutil
# 逐行讲解：导入模块：time。
import time
# 逐行讲解：导入模块：urllib.parse。
import urllib.parse
# 逐行讲解：导入模块：urllib.request。
import urllib.request
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


# 逐行讲解：调用 trio.configure（trio.configure(request_timeout_seconds=600)）。
trio.configure(
    request_timeout_seconds=600,
)


# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets' / 'DeepMath-103K'，赋给 DEFAULT_DATA_DIR。
DEFAULT_DATA_DIR = SCRIPT_DIR / "datasets" / "DeepMath-103K"
# 逐行讲解：取值 10，赋给 DEEPMATH_SHARDS。
DEEPMATH_SHARDS = 10


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='PyTRIO 异步版 OPD / DeepMath')），赋给 parser。
    parser = argparse.ArgumentParser(description="PyTRIO 异步版 OPD / DeepMath")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-repo', default='AI-ModelScope/DeepMath-103K')）。
    parser.add_argument("--dataset-repo", default="AI-ModelScope/DeepMath-103K")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-revision', default='master')）。
    parser.add_argument("--dataset-revision", default="master")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATA_DIR)）。
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA_DIR)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--num-shards', type=int, default=1, help='完整 DeepMath 为 10 个分片')）。
    parser.add_argument("--num-shards", type=int, default=1, help="完整 DeepMath 为 10 个分片")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--force-download', action='store_true')）。
    parser.add_argument("--force-download", action="store_true")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sample-size', type=int, default=1000, help='随机抽样数量；<=0 表示全用')）。
    parser.add_argument("--sample-size", type=int, default=1000, help="随机抽样数量；<=0 表示全用")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B')）。
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=32)）。
    parser.add_argument("--lora-rank", type=int, default=32)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--teacher-base-model', default='Qwen/Qwen3.6-27B', help='默认使用 27B teacher')）。
    parser.add_argument("--teacher-base-model", default="Qwen/Qwen3.6-27B", help="默认使用 27B teacher")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--teacher-model-path', default=None)）。
    parser.add_argument("--teacher-model-path", default=None)

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--steps', type=int, default=10)）。
    parser.add_argument("--steps", type=int, default=10)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=2)）。
    parser.add_argument("--batch-size", type=int, default=2)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--group-size', type=int, default=2)）。
    parser.add_argument("--group-size", type=int, default=2)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=512)）。
    parser.add_argument("--max-tokens", type=int, default=512)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0)）。
    parser.add_argument("--temperature", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0)）。
    parser.add_argument("--top-p", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-k', type=int, default=-1)）。
    parser.add_argument("--top-k", type=int, default=-1)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--question-suffix', default='Please solve the problem step by step and put the final a…）。
    parser.add_argument(
        "--question-suffix",
        default="Please solve the problem step by step and put the final answer in \\boxed{}.",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=False)）。
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--kl-penalty-coef', type=float, default=1.0)）。
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=4e-05)）。
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta1', type=float, default=0.9)）。
    parser.add_argument("--beta1", type=float, default=0.9)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta2', type=float, default=0.95)）。
    parser.add_argument("--beta2", type=float, default=0.95)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sampler-refresh-steps', type=int, default=1)）。
    parser.add_argument("--sampler-refresh-steps", type=int, default=1)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-weights-name', default='opd-deepmath-qwen35-4b-async')）。
    parser.add_argument("--save-weights-name", default="opd-deepmath-qwen35-4b-async")

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab', action=argparse.BooleanOptionalAction, default=True)）。
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=True)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-opd')）。
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-opd")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-name', default='opd-deepmath-qwen35-4b-async')）。
    parser.add_argument("--swanlab-name", default="opd-deepmath-qwen35-4b-async")
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

    # DeepMath-103K 在 ModelScope 上是 10 个 parquet 分片，这里允许只下载前 N 个分片试跑。
    if not 1 <= args.num_shards <= DEEPMATH_SHARDS:
        # 逐行讲解：抛出异常：ValueError(f'--num-shards must be between 1 and {DEEPMATH_SHARDS}')。
        raise ValueError(f"--num-shards must be between 1 and {DEEPMATH_SHARDS}")

    # 这些参数都会直接参与训练循环或取模逻辑，不能为 0 或负数。
    for name in ("steps", "batch_size", "group_size", "sampler_refresh_steps"):
        # 逐行讲解：检查条件 getattr(args, name) < 1；成立时执行下方分支。
        if getattr(args, name) < 1:
            # 逐行讲解：抛出异常：ValueError(f"--{name.replace('_', '-')} must be >= 1")。
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 shard_name：封装下方逻辑，供调用方复用。
def shard_name(index: int) -> str:
    # ModelScope 上的 DeepMath-103K 使用 HuggingFace parquet 分片命名格式。
    return f"data/train-{index:05d}-of-{DEEPMATH_SHARDS:05d}.parquet"


# 逐行讲解：定义函数 modelscope_file_url：封装下方逻辑，供调用方复用。
def modelscope_file_url(repo: str, revision: str, file_path: str) -> str:
    # repo 形如 "AI-ModelScope/DeepMath-103K"，拆成 namespace 和数据集名。
    namespace, name = repo.split("/", 1)

    # ModelScope 的 repo 文件接口要求把文件路径放在 FilePath 参数里。
    query = urllib.parse.urlencode(
        {
            "Source": "SDK",
            "Revision": revision,
            "FilePath": file_path,
            "View": "False",
        }
    )
    # 逐行讲解：返回结果：f'https://www.modelscope.cn/api/v1/datasets/{namespace}/{name}/repo?{query}'。
    return f"https://www.modelscope.cn/api/v1/datasets/{namespace}/{name}/repo?{query}"


# 逐行讲解：定义函数 download_if_needed：封装下方逻辑，供调用方复用。
def download_if_needed(url: str, local_path: Path, force: bool) -> None:
    # 已经下载过且没有要求强制重下时，直接复用本地分片。
    if local_path.exists() and not force:
        # 逐行讲解：返回，结束当前函数。
        return

    # 先写入 .tmp，下载完成后再替换，避免中断时留下半截 parquet。
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 local_path.with_suffix（local_path.with_suffix(local_path.suffix + '.tmp')），赋给 tmp_path。
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    # 逐行讲解：调用 print（print(f'Downloading {url}\n  -> {local_path}')）。
    print(f"Downloading {url}\n  -> {local_path}")

    # urllib 足够处理这个单文件下载场景，避免额外依赖 requests。
    request = urllib.request.Request(url, headers={"User-Agent": "trio-case-opd-demo"})
    # 逐行讲解：进入上下文管理器 urllib.request.urlopen(request), tmp_path.open('wb')，退出时执行其收尾逻辑。
    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as output:
        # 逐行讲解：调用 shutil.copyfileobj（shutil.copyfileobj(response, output)）。
        shutil.copyfileobj(response, output)
    # 逐行讲解：调用 tmp_path.replace（tmp_path.replace(local_path)）。
    tmp_path.replace(local_path)


# 逐行讲解：定义函数 load_deepmath：从 ModelScope 下载 parquet 到本地，然后用 datasets 库读取本地文件。
def load_deepmath(args: argparse.Namespace) -> Dataset:
    """从 ModelScope 下载 parquet 到本地，然后用 datasets 库读取本地文件。"""
    # 逐行讲解：构造列表 []，赋给 shard_paths。
    shard_paths = []

    # 只下载前 num_shards 个分片，方便先用小数据量做联调。
    for index in range(args.num_shards):
        # 逐行讲解：调用 shard_name（shard_name(index)），赋给 remote_path。
        remote_path = shard_name(index)
        # 逐行讲解：计算 args.dataset_dir / remote_path，赋给 local_path。
        local_path = args.dataset_dir / remote_path

        # 数据真正来自 ModelScope；后面的 load_dataset 只是读取这些本地 parquet。
        download_if_needed(
            modelscope_file_url(args.dataset_repo, args.dataset_revision, remote_path),
            local_path,
            args.force_download,
        )
        # 逐行讲解：向列表末尾追加一个元素（shard_paths.append(str(local_path))）。
        shard_paths.append(str(local_path))

    # 使用 HuggingFace datasets 的 parquet reader 读本地文件，不会从 HF Hub 拉数据。
    dataset = load_dataset(
        "parquet",
        data_files=shard_paths,
        split="train",
        cache_dir=str(args.dataset_dir / ".datasets_cache"),
    )
    # 逐行讲解：检查条件 not isinstance(dataset, Dataset)；成立时执行下方分支。
    if not isinstance(dataset, Dataset):
        # 逐行讲解：抛出异常：TypeError(f'Expected Dataset, got {type(dataset)!r}')。
        raise TypeError(f"Expected Dataset, got {type(dataset)!r}")

    # OPD 这里只需要 prompt；DeepMath 里对应字段是 question。
    if "question" not in dataset.column_names:
        # 逐行讲解：抛出异常：ValueError(f"DeepMath must contain 'question', got {dataset.column_names}")。
        raise ValueError(f"DeepMath must contain 'question', got {dataset.column_names}")

    # 先固定 seed 打乱，再截取 sample_size，保证每次试跑可复现。
    dataset = dataset.shuffle(seed=args.seed)
    # 逐行讲解：检查条件 args.sample_size > 0；成立时执行下方分支。
    if args.sample_size > 0:
        # 逐行讲解：按索引选择数据行（dataset.select(range(min(args.sample_size, len(dataset))))），赋给 dataset。
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))
    # 逐行讲解：返回结果：dataset。
    return dataset


# 逐行讲解：定义函数 build_prompt：把 DeepMath question 渲染成 chat prompt token。
def build_prompt(tokenizer: Any, question: str, suffix: str, enable_thinking: bool) -> list[int]:
    """把 DeepMath question 渲染成 chat prompt token。"""
    # 逐行讲解：按条件选择结果：question.strip() if not suffix else f'{question.strip()}\n\n{suffix}'，赋给 content。
    content = question.strip() if not suffix else f"{question.strip()}\n\n{suffix}"
    # 逐行讲解：构造列表 [{'role': 'user', 'content': content}]，赋给 messages。
    messages = [{"role": "user", "content": content}]
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_…），赋给 prompt。
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    # 逐行讲解：将文本编码为 token（tokenizer.encode(prompt, add_special_tokens=False)），赋给 prompt_ids。
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # 逐行讲解：检查条件 not prompt_ids；成立时执行下方分支。
    if not prompt_ids:
        # 逐行讲解：抛出异常：ValueError('Prompt tokens are empty')。
        raise ValueError("Prompt tokens are empty")
    # 逐行讲解：返回结果：prompt_ids。
    return prompt_ids


# 逐行讲解：定义异步函数 completion_teacher_logprobs_async：teacher 对 student 实际生成 completion 的逐 token logprob。
async def completion_teacher_logprobs_async(
    teacher_client: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> list[float]:
    """teacher 对 student 实际生成 completion 的逐 token logprob。"""
    # teacher 需要看到和 student rollout 完全一致的上下文：
    # prompt 是题目，completion 是 student 已经生成出来的答案 token。
    all_ids = prompt_ids + completion_ids

    # compute_logprobs_async 在当前 PyTRIO SDK 里 await 后直接返回 list[float | None]。
    # 第一个 token 通常没有前文，所以返回值里可能有 None；我们只取 completion 区间。
    all_logprobs = await teacher_client.compute_logprobs_async(
        trio.ModelInput.from_ints(all_ids)
    )

    # completion 的第一个 token 是在完整 prompt 后被预测出来的，
    # 因此从 len(prompt_ids) 开始到结尾就是 completion 的 logprobs。
    completion_logprobs = all_logprobs[len(prompt_ids) :]

    # completion 区间必须和 student 生成 token 一一对应，不能有 None，
    # 否则 reverse KL 的 token 对齐就不可靠。
    if len(completion_logprobs) != len(completion_ids) or any(v is None for v in completion_logprobs):
        # 逐行讲解：抛出异常：ValueError('Invalid teacher logprobs for completion tokens')。
        raise ValueError("Invalid teacher logprobs for completion tokens")
    # 逐行讲解：返回结果：[float(v) for v in completion_logprobs]。
    return [float(v) for v in completion_logprobs]


# 逐行讲解：定义函数 build_opd_datum：PyTRIO importance_sampling 需要右移后的 input/target/logprobs/advantages。
def build_opd_datum(
    prompt_ids: list[int],
    completion_ids: list[int],
    old_logprobs: list[float],
    advantages: list[float] | np.ndarray,
) -> trio.Datum:
    """PyTRIO importance_sampling 需要右移后的 input/target/logprobs/advantages。"""
    # 自回归训练需要右移：用当前位置 input token 预测下一个 target token。
    # prompt 内部的预测不是 OPD 训练目标，所以这部分 advantage 置 0。
    prompt_loss_len = len(prompt_ids) - 1

    # input 由完整 prompt 加 completion[:-1] 组成：
    # 最后一个 input 位置用来预测 completion 的最后一个 token。
    input_ids = prompt_ids + completion_ids[:-1]

    # target 的 prompt 区间用 0 占位；从最后一个 prompt token 位置开始预测 completion。
    target_ids = [0] * prompt_loss_len + completion_ids

    # old_logprobs 是 student rollout 时每个 completion token 的旧策略 logprob。
    # prompt 区间不训练，因此同样用 0.0 占位。
    padded_logprobs = [0.0] * prompt_loss_len + old_logprobs

    # advantages 是 -kl_penalty_coef * reverse_kl，只对 completion token 生效。
    padded_advantages = [0.0] * prompt_loss_len + list(advantages)

    # PyTRIO 的 importance_sampling 要求四个序列长度严格一致。
    if not (len(input_ids) == len(target_ids) == len(padded_logprobs) == len(padded_advantages)):
        # 逐行讲解：抛出异常：ValueError('OPD datum fields must have the same length')。
        raise ValueError("OPD datum fields must have the same length")

    # loss_fn_inputs 的三个字段对应 importance_sampling schema：
    # target_tokens 是要预测的 token，logprobs 是采样时旧策略概率，advantages 是训练信号。
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_ids, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )


# 逐行讲解：定义异步函数 run_prompt_rollout：异步处理单道题：student 采样，再让 teacher 给同一批 completion 打 logprob。
async def run_prompt_rollout(
    student_sampler: Any,
    teacher_client: Any,
    tokenizer: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    sampling_params: trio.SamplingParams,
) -> tuple[list[trio.Datum], list[float], list[int]]:
    """异步处理单道题：student 采样，再让 teacher 给同一批 completion 打 logprob。"""
    # DeepMath 这里只使用 question 字段构造 prompt。
    prompt_ids = build_prompt(
        tokenizer,
        row["question"],
        args.question_suffix,
        args.enable_thinking,
    )

    # 当前 student 采样出的 completion 就是 OPD 的 on-policy 轨迹。
    result = await student_sampler.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_ids),
        num_samples=args.group_size,
        sampling_params=sampling_params,
        return_text=False,
    )

    # 逐行讲解：通过推导式逐项生成结果：[seq for seq in result.sequences if seq.tokens]，赋给 sequences。
    sequences = [seq for seq in result.sequences if seq.tokens]

    # teacher 不是重新生成答案，而是并发给 student 已生成 token 打 logprob。
    teacher_tasks = [
        completion_teacher_logprobs_async(teacher_client, prompt_ids, seq.tokens)
        for seq in sequences
    ]
    # 逐行讲解：按条件选择结果：await asyncio.gather(*teacher_tasks) if teacher_tasks else []，赋给 teacher_logprobs_list。
    teacher_logprobs_list = await asyncio.gather(*teacher_tasks) if teacher_tasks else []

    # datums 会送进 PyTRIO forward_backward；
    # reverse_kls 和 completion_token_counts 只用于本 step 的日志统计。
    datums: list[trio.Datum] = []
    # 逐行讲解：声明 reverse_kls 的类型为 list[float]；构造列表 []。
    reverse_kls: list[float] = []
    # 逐行讲解：声明 completion_token_counts 的类型为 list[int]；构造列表 []。
    completion_token_counts: list[int] = []

    # 逐行讲解：遍历 zip(sequences, teacher_logprobs_list, strict=True)，将每项绑定到 (seq, teacher_lps)。
    for seq, teacher_lps in zip(sequences, teacher_logprobs_list, strict=True):
        # 逐行讲解：取值 seq.tokens，赋给 completion_ids。
        completion_ids = seq.tokens
        # 逐行讲解：通过推导式逐项生成结果：[float(value) for value in seq.logprobs]，赋给 student_lps。
        student_lps = [float(value) for value in seq.logprobs]

        # 逐行讲解：检查条件 len(student_lps) != len(completion_ids)；成立时执行下方分支。
        if len(student_lps) != len(completion_ids):
            # 逐行讲解：抛出异常：ValueError(f'Student token/logprob length mismatch: {len(completion_ids)} != {len(student_lps)}')。
            raise ValueError(
                f"Student token/logprob length mismatch: {len(completion_ids)} != {len(student_lps)}"
            )

        # OPD 信号：reverse_kl = student_logprob - teacher_logprob。
        reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)
        # 逐行讲解：计算 -args.kl_penalty_coef * reverse_kl，赋给 advantages。
        advantages = -args.kl_penalty_coef * reverse_kl

        # importance_sampling 使用旧策略 logprobs 和 KL advantage 更新 student。
        datums.append(build_opd_datum(prompt_ids, completion_ids, student_lps, advantages))
        # 逐行讲解：把一组元素依次加入列表（reverse_kls.extend(reverse_kl.tolist())）。
        reverse_kls.extend(reverse_kl.tolist())
        # 逐行讲解：向列表末尾追加一个元素（completion_token_counts.append(len(completion_ids))）。
        completion_token_counts.append(len(completion_ids))

    # 逐行讲解：返回结果：(datums, reverse_kls, completion_token_counts)。
    return datums, reverse_kls, completion_token_counts


# 逐行讲解：定义函数 start_swanlab：SwanLab 只做实验记录，不影响训练逻辑。
def start_swanlab(args: argparse.Namespace, dataset_size: int) -> Any | None:
    """SwanLab 只做实验记录，不影响训练逻辑。"""
    # 逐行讲解：检查条件 not args.swanlab；成立时执行下方分支。
    if not args.swanlab:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：调用 vars(args).copy（vars(args).copy()），赋给 config。
    config = vars(args).copy()
    # 逐行讲解：调用 str（str(args.dataset_dir)），赋给 config['dataset_dir']。
    config["dataset_dir"] = str(args.dataset_dir)
    # 逐行讲解：取值 dataset_size，赋给 config['dataset_size']。
    config["dataset_size"] = dataset_size
    # 逐行讲解：返回结果：swanlab.init(project=args.swanlab_project, name=args.swanlab_name, workspace=args.swanlab_workspace, mode=a…。
    return swanlab.init(
        project=args.swanlab_project,
        name=args.swanlab_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=["TRIO", "OPD", "DeepMath", "ModelScope", "async"],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


# 逐行讲解：定义异步函数 main：封装下方逻辑，供调用方复用。
async def main(args: argparse.Namespace) -> None:
    # 固定本地随机性：数据 shuffle 和部分采样参数都会用到 seed。
    random.seed(args.seed)
    # 逐行讲解：设置对应随机数生成器的种子（np.random.seed(args.seed)）。
    np.random.seed(args.seed)

    # DeepMath 在这里作为 prompt pool 使用，只取 question 字段。
    dataset = load_deepmath(args)
    # 逐行讲解：调用 print（print(f'Loaded {len(dataset)} DeepMath prompts')）。
    print(f"Loaded {len(dataset)} DeepMath prompts")

    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：取值 None，赋给 swanlab_run。
    swanlab_run = None

    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：调用 print（print('Creating PyTRIO clients...')）。
        print("Creating PyTRIO clients...")
        # student 是一个 LoRA training client，后续 forward/backward 和 optim 都作用在它上面。
        training_client = await service_client.create_lora_training_client_async(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
        )
        # 逐行讲解：调用 training_client.get_tokenizer（training_client.get_tokenizer()），赋给 tokenizer。
        tokenizer = training_client.get_tokenizer()

        # teacher 只负责给 student 采样轨迹打 logprob，不参与优化。
        teacher_client = await service_client.create_sampling_client_async(
            base_model=args.teacher_base_model or args.base_model,
            model_path=args.teacher_model_path,
        )

        # student rollout 的采样参数；这些回答会成为 OPD 的 on-policy 轨迹。
        sampling_params = trio.SamplingParams(
            max_tokens=args.max_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop=[x for x in [tokenizer.eos_token, "<|im_end|>"] if x],
        )

        # PyTRIO 的优化器参数；每步 forward_backward 后调用一次 optim_step。
        adam_params = trio.AdamParams(
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
        )

        # 逐行讲解：调用 start_swanlab（start_swanlab(args, len(dataset))），赋给 swanlab_run。
        swanlab_run = start_swanlab(args, len(dataset))
        # 逐行讲解：取值 None，赋给 student_sampler。
        student_sampler = None

        # 逐行讲解：遍历 range(args.steps)，将每项绑定到 step。
        for step in range(args.steps):
            # 逐行讲解：调用 time.time（time.time()），赋给 step_start。
            step_start = time.time()
            # 逐行讲解：检查条件 student_sampler is None or step % args.sampler_refresh_steps == 0；成立时执行下方分支。
            if student_sampler is None or step % args.sampler_refresh_steps == 0:
                # 严格 on-policy：用当前 student 权重采样。
                student_sampler = await training_client.save_weights_and_get_sampling_client_async()

            # dataset 已经提前随机打乱；这里按 step 循环取 batch，超出后回绕。
            indices = [(step * args.batch_size + i) % len(dataset) for i in range(args.batch_size)]
            # 逐行讲解：按索引选择数据行（dataset.select(indices)），赋给 batch_rows。
            batch_rows = dataset.select(indices)

            # batch 内每道题的 rollout 彼此独立，可以并发请求远端 sampler。
            rollout_tasks = [
                run_prompt_rollout(
                    student_sampler=student_sampler,
                    teacher_client=teacher_client,
                    tokenizer=tokenizer,
                    row=row,
                    args=args,
                    sampling_params=sampling_params,
                )
                for row in batch_rows
            ]
            # 逐行讲解：进入上下文管理器 tqdm(total=len(rollout_tasks), desc=f'OPD async step {step}', unit='prompt')，退出时执行其收尾逻辑。
            with tqdm(total=len(rollout_tasks), desc=f"OPD async step {step}", unit="prompt") as progress_bar:

                # 逐行讲解：定义异步函数 run_and_track：封装下方逻辑，供调用方复用。
                async def run_and_track(rollout_task: Any) -> tuple[list[trio.Datum], list[float], list[int]]:
                    # 逐行讲解：等待异步操作 rollout_task 完成并取得结果，赋给 result。
                    result = await rollout_task
                    # 逐行讲解：调用 progress_bar.update（progress_bar.update(1)）。
                    progress_bar.update(1)
                    # 逐行讲解：返回结果：result。
                    return result

                # 逐行讲解：等待异步操作 asyncio.gather(*(run_and_track(rollout_task) for rollout_task in rollout_tasks)) 完成并取得结果，赋给 rollout_results。
                rollout_results = await asyncio.gather(
                    *(run_and_track(rollout_task) for rollout_task in rollout_tasks)
                )

            # 一个 step 内会把 batch_size * group_size 条 completion 转成训练 datum。
            datums: list[trio.Datum] = []
            # 逐行讲解：声明 reverse_kls 的类型为 list[float]；构造列表 []。
            reverse_kls: list[float] = []
            # 逐行讲解：声明 completion_token_counts 的类型为 list[int]；构造列表 []。
            completion_token_counts: list[int] = []
            # 逐行讲解：遍历 rollout_results，将每项绑定到 (rollout_datums, rollout_reverse_kls, rollout_completion_counts)。
            for rollout_datums, rollout_reverse_kls, rollout_completion_counts in rollout_results:
                # 逐行讲解：把一组元素依次加入列表（datums.extend(rollout_datums)）。
                datums.extend(rollout_datums)
                # 逐行讲解：把一组元素依次加入列表（reverse_kls.extend(rollout_reverse_kls)）。
                reverse_kls.extend(rollout_reverse_kls)
                # 逐行讲解：把一组元素依次加入列表（completion_token_counts.extend(rollout_completion_counts)）。
                completion_token_counts.extend(rollout_completion_counts)

            # 逐行讲解：检查条件 not datums；成立时执行下方分支。
            if not datums:
                # 逐行讲解：抛出异常：RuntimeError('No OPD datums were built')。
                raise RuntimeError("No OPD datums were built")

            # 异步版 PyTRIO：先异步提交远程前向/反向和优化器更新，再 await 对应 future。
            fwd_bwd_future = await training_client.forward_backward_async(
                datums,
                loss_fn="importance_sampling",
            )
            # 逐行讲解：等待异步操作 training_client.optim_step_async(adam_params) 完成并取得结果，赋给 optim_future。
            optim_future = await training_client.optim_step_async(adam_params)
            # 逐行讲解：等待异步操作 fwd_bwd_future 完成并取得结果，赋给 fwd_bwd_result。
            fwd_bwd_result = await fwd_bwd_future
            # 逐行讲解：等待异步操作 optim_future 完成并取得结果。
            await optim_future

            # 逐行讲解：计算 time.time() - step_start，赋给 step_elapsed_time。
            step_elapsed_time = time.time() - step_start
            # 逐行讲解：调用 int（int(sum(completion_token_counts))），赋给 completion_tokens_total。
            completion_tokens_total = int(sum(completion_token_counts))

            # 记录最关键的 OPD 和训练指标，方便在 SwanLab 上看趋势。
            metrics = {
                "data/datums": len(datums),
                "data/completion_tokens_mean": float(np.mean(completion_token_counts)),
                "data/completion_tokens_total": completion_tokens_total,
                "data/completion_tokens_per_second": completion_tokens_total / step_elapsed_time,
                "opd/reverse_kl_mean": float(np.mean(reverse_kls)),
                "opd/reverse_kl_std": float(np.std(reverse_kls)),
                "train/learning_rate": args.learning_rate,
                "train/step": step,
                "time/step_elapsed_time": step_elapsed_time,
            }
            # 逐行讲解：调用 metrics.update（metrics.update({f'trainer/{k}': float(v) for k, v in dict(fwd_bwd_result.metrics).items()})）。
            metrics.update({f"trainer/{k}": float(v) for k, v in dict(fwd_bwd_result.metrics).items()})
            # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
            if swanlab_run is not None:
                # 逐行讲解：计算自然对数（swanlab.log(metrics, step=step)）。
                swanlab.log(metrics, step=step)
            # 逐行讲解：调用 tqdm.write（tqdm.write(f"step {step:03d}/{args.steps} | datums {len(datums)} | completion tokens mean {metrics['data/co…）。
            tqdm.write(
                f"step {step:03d}/{args.steps} | datums {len(datums)} | "
                f"completion tokens mean {metrics['data/completion_tokens_mean']:.1f} | "
                f"tokens/s {metrics['data/completion_tokens_per_second']:.1f} | "
                f"reverse_kl {metrics['opd/reverse_kl_mean']:.4f} | "
                f"time {metrics['time/step_elapsed_time']:.2f}s"
            )

        # 保存最终 LoRA 权重，后续可以用这个 path 创建 sampler 做推理。
        save_future = await training_client.save_weights_for_sampler_async(
            name=args.save_weights_name
        )
        # 逐行讲解：等待异步操作 save_future 完成并取得结果，赋给 save_result。
        save_result = await save_future
        # 逐行讲解：调用 print（print(f'Saved weights: {save_result.path}')）。
        print(f"Saved weights: {save_result.path}")
        # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
        if swanlab_run is not None:
            # 逐行讲解：计算自然对数（swanlab.log({'save/weights_path': swanlab.Text(save_result.path)}, step=args.steps)）。
            swanlab.log({"save/weights_path": swanlab.Text(save_result.path)}, step=args.steps)
    finally:
        # 无论中间是否报错，都尽量正常结束日志。
        if swanlab_run is not None:
            # 逐行讲解：调用 swanlab.finish（swanlab.finish()）。
            swanlab.finish()


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 time.time（time.time()），赋给 start。
    start = time.time()
    # 逐行讲解：调用 asyncio.run（asyncio.run(main(parse_args()))）。
    asyncio.run(main(parse_args()))
    # 逐行讲解：调用 print（print('#' * 50)）。
    print("#" * 50)
    # 逐行讲解：调用 print（print('# all done')）。
    print("# all done")
    # 逐行讲解：调用 print（print(f'# train cost {time.time() - start:.2f}s')）。
    print(f"# train cost {time.time() - start:.2f}s")
    # 逐行讲解：调用 print（print('#' * 50)）。
    print("#" * 50)
