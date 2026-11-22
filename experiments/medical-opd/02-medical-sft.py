"""中文医疗 reasoning SFT：medical-o1-reasoning-SFT-zh + PyTRIO async + SwanLab。

训练格式：
- prompt 只包含 system + user，并用 chat template 加 generation prompt。
- assistant completion 由脚本构造成 <think>{Complex_CoT}</think> + Response。
- prompt token 不参与 loss；assistant completion token 和 EOS 参与 loss。
- checkpoint 同时保存 training state 和 sampler weights；state 用于后续 OPD 初始化 student，sampler weights 用于评测。

小成本试跑：
uv run python 02-medical-sft.py \
    --sample-size 100 \
    --num-epochs 1 \
    --batch-size 2 \
    --max-length 2048 \
    --swanlab-mode disabled

正式训练：
uv run python 02-medical-sft.py \
    --num-epochs 3 \
    --batch-size 16 \
    --max-length 2048 \
    --swanlab-mode online
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：random。
import random
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

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm


# 逐行讲解：调用 trio.configure（trio.configure(request_timeout_seconds=600)）。
trio.configure(request_timeout_seconds=600)


# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets' / 'medical-o1-reasoning-SFT-zh' / 'train.jsonl'，赋给 DEFAULT_DATASET_PATH。
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "medical-o1-reasoning-SFT-zh" / "train.jsonl"
# 逐行讲解：取值 '你是一个中文医疗问答助手。请根据题目给出严谨的医学推理和最终答案。'，赋给 DEFAULT_SYSTEM_MESSAGE。
DEFAULT_SYSTEM_MESSAGE = "你是一个中文医疗问答助手。请根据题目给出严谨的医学推理和最终答案。"


# 逐行讲解：定义类 TokenizedExample：封装下方逻辑，供调用方复用。
@dataclass(frozen=True)
class TokenizedExample:
    # 逐行讲解：声明 tokens 的类型为 list[int]。
    tokens: list[int]
    # 逐行讲解：声明 weights 的类型为 list[float]。
    weights: list[float]
    # 逐行讲解：声明 prompt_len 的类型为 int。
    prompt_len: int
    # 逐行讲解：声明 truncated 的类型为 bool。
    truncated: bool


# 逐行讲解：定义类 ProcessedDatum：封装下方逻辑，供调用方复用。
@dataclass(frozen=True)
class ProcessedDatum:
    # 逐行讲解：声明 datum 的类型为 trio.Datum。
    datum: trio.Datum
    # 逐行讲解：声明 total_tokens 的类型为 int。
    total_tokens: int
    # 逐行讲解：声明 prompt_tokens 的类型为 int。
    prompt_tokens: int
    # 逐行讲解：声明 trainable_tokens 的类型为 int。
    trainable_tokens: int
    # 逐行讲解：声明 truncated 的类型为 bool。
    truncated: bool


# 逐行讲解：定义函数 model_slug：封装下方逻辑，供调用方复用。
def model_slug(base_model: str) -> str:
    # 逐行讲解：调用 base_model.rsplit('/', 1)[-1].lower（base_model.rsplit('/', 1)[-1].lower()），赋给 name。
    name = base_model.rsplit("/", 1)[-1].lower()
    # 逐行讲解：调用 name.replace（name.replace('qwen3.5', 'qwen35')），赋给 name。
    name = name.replace("qwen3.5", "qwen35")
    # 逐行讲解：返回结果：re.sub('[^a-z0-9]+', '-', name).strip('-')。
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


# 逐行讲解：定义函数 build_run_name：封装下方逻辑，供调用方复用。
def build_run_name(args: argparse.Namespace) -> str:
    # 逐行讲解：调用 args.dataset_path.parent.name.lower().replace（args.dataset_path.parent.name.lower().replace('_', '-')），赋给 dataset_slug。
    dataset_slug = args.dataset_path.parent.name.lower().replace("_", "-")
    # 逐行讲解：调用 re.sub('[^a-z0-9-]+', '-', dataset_slug).strip（re.sub('[^a-z0-9-]+', '-', dataset_slug).strip('-')），赋给 dataset_slug。
    dataset_slug = re.sub(r"[^a-z0-9-]+", "-", dataset_slug).strip("-")
    # 逐行讲解：按条件选择结果：f'sample{args.sample_size}' if args.sample_size > 0 else 'full'，赋给 sample_slug。
    sample_slug = f"sample{args.sample_size}" if args.sample_size > 0 else "full"
    # 逐行讲解：按条件选择结果：f'-steps{args.max_steps}' if args.max_steps > 0 else ''，赋给 steps_slug。
    steps_slug = f"-steps{args.max_steps}" if args.max_steps > 0 else ""
    # 逐行讲解：返回结果：f'sft-medical-{model_slug(args.base_model)}-{dataset_slug}-{sample_slug}-epochs{args.num_epochs}{steps_slug}'。
    return f"sft-medical-{model_slug(args.base_model)}-{dataset_slug}-{sample_slug}-epochs{args.num_epochs}{steps_slug}"


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='PyTRIO 异步版中文医疗 SFT')），赋给 parser。
    parser = argparse.ArgumentParser(description="PyTRIO 异步版中文医疗 SFT")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-path', type=Path, default=DEFAULT_DATASET_PATH, help='medical-o1 reasoning S…）。
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="medical-o1 reasoning SFT JSONL 路径")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sample-size', type=int, default=0, help='随机抽样数量；<=0 表示使用全量训练集')）。
    parser.add_argument("--sample-size", type=int, default=0, help="随机抽样数量；<=0 表示使用全量训练集")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42, help='随机种子')）。
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B', help='LoRA SFT 使用的基础模型')）。
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B", help="LoRA SFT 使用的基础模型")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=32, help='LoRA rank')）。
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--num-epochs', type=int, default=1, help='训练 epoch 数')）。
    parser.add_argument("--num-epochs", type=int, default=1, help="训练 epoch 数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=16, help='每个 step 的样本数')）。
    parser.add_argument("--batch-size", type=int, default=16, help="每个 step 的样本数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-steps', type=int, default=0, help='最多训练 step 数；<=0 表示不限制')）。
    parser.add_argument("--max-steps", type=int, default=0, help="最多训练 step 数；<=0 表示不限制")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-length', type=int, default=2048, help='单条样本 prompt+completion 最大 token 数')）。
    parser.add_argument("--max-length", type=int, default=2048, help="单条样本 prompt+completion 最大 token 数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=0.0001, help='Adam 学习率')）。
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam 学习率")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')）。
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')）。
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--system-message', default=DEFAULT_SYSTEM_MESSAGE, help='训练时插入到 chat template 的 system…）。
    parser.add_argument("--system-message", default=DEFAULT_SYSTEM_MESSAGE, help="训练时插入到 chat template 的 system message；传空字符串表示不加")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-each-epoch', action=argparse.BooleanOptionalAction, default=True, help='是否每个 ep…）。
    parser.add_argument("--save-each-epoch", action=argparse.BooleanOptionalAction, default=True, help="是否每个 epoch 结束同时保存 training state 和 sampler 权重")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every-steps', type=int, default=1000, help='每多少 step 同时保存 training state 和 samp…）。
    parser.add_argument("--save-every-steps", type=int, default=1000, help="每多少 step 同时保存 training state 和 sampler 权重；0 表示不按 step 保存")

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab', action=argparse.BooleanOptionalAction, default=True, help='是否启用 SwanLab 记录')）。
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=True, help="是否启用 SwanLab 记录")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-opd', help='SwanLab 项目名')）。
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-opd", help="SwanLab 项目名")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-workspace', default=None, help='SwanLab workspace；不传则使用默认 workspace')）。
    parser.add_argument("--swanlab-workspace", default=None, help="SwanLab workspace；不传则使用默认 workspace")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=['online', 'local', 'offline', 'disabled'], default=None, hel…）。
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default=None,
        help="SwanLab 运行模式",
    )
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()

    # 逐行讲解：遍历 ('num_epochs', 'batch_size', 'max_length')，将每项绑定到 name。
    for name in ("num_epochs", "batch_size", "max_length"):
        # 逐行讲解：检查条件 getattr(args, name) < 1；成立时执行下方分支。
        if getattr(args, name) < 1:
            # 逐行讲解：抛出异常：ValueError(f"--{name.replace('_', '-')} must be >= 1")。
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    # 逐行讲解：检查条件 args.sample_size < 0；成立时执行下方分支。
    if args.sample_size < 0:
        # 逐行讲解：抛出异常：ValueError('--sample-size must be >= 0')。
        raise ValueError("--sample-size must be >= 0")
    # 逐行讲解：检查条件 args.max_steps < 0；成立时执行下方分支。
    if args.max_steps < 0:
        # 逐行讲解：抛出异常：ValueError('--max-steps must be >= 0')。
        raise ValueError("--max-steps must be >= 0")
    # 逐行讲解：检查条件 args.save_every_steps < 0；成立时执行下方分支。
    if args.save_every_steps < 0:
        # 逐行讲解：抛出异常：ValueError('--save-every-steps must be >= 0')。
        raise ValueError("--save-every-steps must be >= 0")

    # 逐行讲解：调用 build_run_name（build_run_name(args)），赋给 run_name。
    run_name = build_run_name(args)
    # 逐行讲解：取值 run_name，赋给 args.swanlab_name。
    args.swanlab_name = run_name
    # 逐行讲解：取值 run_name，赋给 args.save_state_name。
    args.save_state_name = run_name
    # 逐行讲解：取值 run_name，赋给 args.save_weights_name。
    args.save_weights_name = run_name
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 build_assistant_text：封装下方逻辑，供调用方复用。
def build_assistant_text(complex_cot: str, response: str) -> str:
    # 逐行讲解：调用 complex_cot.strip（complex_cot.strip()），赋给 complex_cot。
    complex_cot = complex_cot.strip()
    # 逐行讲解：调用 response.strip（response.strip()），赋给 response。
    response = response.strip()
    # medical-o1 把推理过程和最终回答分成两个字段；
    # 训练 Qwen thinking 模型时，把推理过程包进 <think>...</think>。
    if complex_cot and response:
        # 逐行讲解：返回结果：f'<think>\n{complex_cot}\n</think>\n\n{response}'。
        return f"<think>\n{complex_cot}\n</think>\n\n{response}"
    # 逐行讲解：检查条件 complex_cot；成立时执行下方分支。
    if complex_cot:
        # 逐行讲解：返回结果：f'<think>\n{complex_cot}\n</think>'。
        return f"<think>\n{complex_cot}\n</think>"
    # 逐行讲解：返回结果：response。
    return response


# 逐行讲解：定义函数 normalize_row：封装下方逻辑，供调用方复用。
def normalize_row(row: dict[str, Any]) -> dict[str, str] | None:
    # 逐行讲解：调用 str(row.get('question', '')).strip（str(row.get('question', '')).strip()），赋给 question。
    question = str(row.get("question", "")).strip()
    # 逐行讲解：调用 str(row.get('complex_cot', '')).strip（str(row.get('complex_cot', '')).strip()），赋给 complex_cot。
    complex_cot = str(row.get("complex_cot", "")).strip()
    # 逐行讲解：调用 str(row.get('response', '')).strip（str(row.get('response', '')).strip()），赋给 response。
    response = str(row.get("response", "")).strip()
    # 逐行讲解：调用 build_assistant_text（build_assistant_text(complex_cot, response)），赋给 assistant。
    assistant = build_assistant_text(complex_cot, response)
    # 逐行讲解：检查条件 not question or not assistant；成立时执行下方分支。
    if not question or not assistant:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：{'question': question, 'assistant': assistant}。
    return {
        "question": question,
        "assistant": assistant,
    }


# 逐行讲解：定义函数 load_examples：封装下方逻辑，供调用方复用。
def load_examples(args: argparse.Namespace) -> list[dict[str, str]]:
    # 逐行讲解：检查条件 not args.dataset_path.exists()；成立时执行下方分支。
    if not args.dataset_path.exists():
        # 逐行讲解：抛出异常：FileNotFoundError(f'找不到 SFT 数据文件：{args.dataset_path}，请先运行 00-download-dataset.py')。
        raise FileNotFoundError(f"找不到 SFT 数据文件：{args.dataset_path}，请先运行 00-download-dataset.py")

    # 逐行讲解：构造列表 []，赋给 examples。
    examples = []
    # 逐行讲解：进入上下文管理器 args.dataset_path.open('r', encoding='utf-8')，退出时执行其收尾逻辑。
    with args.dataset_path.open("r", encoding="utf-8") as f:
        # 逐行讲解：遍历 f，将每项绑定到 line。
        for line in f:
            # 逐行讲解：检查条件 not line.strip()；成立时执行下方分支。
            if not line.strip():
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：调用 normalize_row（normalize_row(json.loads(line))），赋给 example。
            example = normalize_row(json.loads(line))
            # 逐行讲解：检查条件 example is not None；成立时执行下方分支。
            if example is not None:
                # 逐行讲解：向列表末尾追加一个元素（examples.append(example)）。
                examples.append(example)

    # 逐行讲解：检查条件 not examples；成立时执行下方分支。
    if not examples:
        # 逐行讲解：抛出异常：ValueError(f'没有在 {args.dataset_path} 中找到有效 SFT 样本')。
        raise ValueError(f"没有在 {args.dataset_path} 中找到有效 SFT 样本")

    # 逐行讲解：打乱数据顺序（random.Random(args.seed).shuffle(examples)）。
    random.Random(args.seed).shuffle(examples)
    # 逐行讲解：检查条件 args.sample_size > 0；成立时执行下方分支。
    if args.sample_size > 0:
        # 逐行讲解：按索引或切片读取 examples[:min(args.sample_size, len(examples))]，赋给 examples。
        examples = examples[: min(args.sample_size, len(examples))]
    # 逐行讲解：返回结果：examples。
    return examples


# 逐行讲解：定义函数 build_prompt_tokens：封装下方逻辑，供调用方复用。
def build_prompt_tokens(tokenizer: Any, question: str, system_message: str) -> list[int]:
    # 逐行讲解：构造列表 []，赋给 messages。
    messages = []
    # 逐行讲解：检查条件 system_message.strip()；成立时执行下方分支。
    if system_message.strip():
        # 逐行讲解：向列表末尾追加一个元素（messages.append({'role': 'system', 'content': system_message.strip()})）。
        messages.append({"role": "system", "content": system_message.strip()})
    # 逐行讲解：向列表末尾追加一个元素（messages.append({'role': 'user', 'content': question})）。
    messages.append({"role": "user", "content": question})

    # 这里只把 system + user 放进 chat template，并让模板补出 assistant 起始位置。
    # assistant 正文会在后面单独 tokenize 后拼接，不作为 messages 输入模板。
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # 逐行讲解：返回结果：tokenizer.encode(prompt_text, add_special_tokens=False)。
    return tokenizer.encode(prompt_text, add_special_tokens=False)


# 逐行讲解：定义函数 tokenize_example：封装下方逻辑，供调用方复用。
def tokenize_example(
    example: dict[str, str],
    tokenizer: Any,
    args: argparse.Namespace,
) -> TokenizedExample:
    # 逐行讲解：调用 build_prompt_tokens（build_prompt_tokens(tokenizer, example['question'], args.system_message)），赋给 prompt_tokens。
    prompt_tokens = build_prompt_tokens(tokenizer, example["question"], args.system_message)
    # 逐行讲解：将文本编码为 token（tokenizer.encode(example['assistant'], add_special_tokens=False)），赋给 assistant_tokens。
    assistant_tokens = tokenizer.encode(example["assistant"], add_special_tokens=False)

    # 让模型学习回答结束位置；EOS 也属于 assistant completion 的训练目标。
    if tokenizer.eos_token_id is not None:
        # 逐行讲解：向列表末尾追加一个元素（assistant_tokens.append(int(tokenizer.eos_token_id))）。
        assistant_tokens.append(int(tokenizer.eos_token_id))

    # 逐行讲解：计算 prompt_tokens + assistant_tokens，赋给 tokens。
    tokens = prompt_tokens + assistant_tokens

    # prompt 只是上下文，不参与 loss；assistant completion 和 EOS 参与 loss。
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(assistant_tokens)
    # 逐行讲解：取值 len(tokens) > args.max_length，赋给 truncated。
    truncated = len(tokens) > args.max_length
    # 逐行讲解：检查条件 truncated；成立时执行下方分支。
    if truncated:
        # 逐行讲解：按索引或切片读取 tokens[:args.max_length]，赋给 tokens。
        tokens = tokens[: args.max_length]
        # 逐行讲解：按索引或切片读取 weights[:args.max_length]，赋给 weights。
        weights = weights[: args.max_length]

    # 逐行讲解：调用 min（min(len(prompt_tokens), len(tokens))），赋给 prompt_len。
    prompt_len = min(len(prompt_tokens), len(tokens))
    # 逐行讲解：返回结果：TokenizedExample(tokens=tokens, weights=weights, prompt_len=prompt_len, truncated=truncated)。
    return TokenizedExample(
        tokens=tokens,
        weights=weights,
        prompt_len=prompt_len,
        truncated=truncated,
    )


# 逐行讲解：定义函数 build_datum：封装下方逻辑，供调用方复用。
def build_datum(
    example: dict[str, str],
    tokenizer: Any,
    args: argparse.Namespace,
) -> ProcessedDatum | None:
    # 逐行讲解：调用 tokenize_example（tokenize_example(example, tokenizer, args)），赋给 tokenized。
    tokenized = tokenize_example(example, tokenizer, args)
    # 逐行讲解：取值 tokenized.tokens，赋给 tokens。
    tokens = tokenized.tokens
    # 逐行讲解：取值 tokenized.weights，赋给 weights。
    weights = tokenized.weights

    # 逐行讲解：检查条件 len(tokens) < 2；成立时执行下方分支。
    if len(tokens) < 2:
        # 逐行讲解：返回结果：None。
        return None

    # 自回归右移：当前位置 input 预测下一个 target。
    # weights 也右移到 target 侧，因此第一个 assistant token 会被正确训练到。
    input_tokens = tokens[:-1]
    # 逐行讲解：按索引或切片读取 tokens[1:]，赋给 target_tokens。
    target_tokens = tokens[1:]
    # 逐行讲解：按索引或切片读取 weights[1:]，赋给 loss_weights。
    loss_weights = weights[1:]
    # 逐行讲解：检查条件 sum(loss_weights) <= 0；成立时执行下方分支。
    if sum(loss_weights) <= 0:
        # 逐行讲解：返回结果：None。
        return None

    # 逐行讲解：调用 trio.Datum（trio.Datum(model_input=trio.ModelInput.from_ints(input_tokens), loss_fn_inputs={'target_tokens': np.asarray…），赋给 datum。
    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int32),
            "weights": np.asarray(loss_weights, dtype=np.float32),
        },
    )
    # 逐行讲解：返回结果：ProcessedDatum(datum=datum, total_tokens=len(tokens), prompt_tokens=tokenized.prompt_len, trainable_tokens=…。
    return ProcessedDatum(
        datum=datum,
        total_tokens=len(tokens),
        prompt_tokens=tokenized.prompt_len,
        trainable_tokens=int(sum(loss_weights)),
        truncated=tokenized.truncated,
    )


# 逐行讲解：定义函数 build_processed_examples：封装下方逻辑，供调用方复用。
def build_processed_examples(
    examples: list[dict[str, str]],
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[ProcessedDatum]:
    # 逐行讲解：构造列表 []，赋给 processed_examples。
    processed_examples = []
    # 逐行讲解：遍历 tqdm(examples, desc='Build SFT datums', unit='sample')，将每项绑定到 example。
    for example in tqdm(examples, desc="Build SFT datums", unit="sample"):
        # 逐行讲解：调用 build_datum（build_datum(example, tokenizer, args)），赋给 built。
        built = build_datum(example, tokenizer, args)
        # 逐行讲解：检查条件 built is not None；成立时执行下方分支。
        if built is not None:
            # 逐行讲解：向列表末尾追加一个元素（processed_examples.append(built)）。
            processed_examples.append(built)

    # 逐行讲解：检查条件 not processed_examples；成立时执行下方分支。
    if not processed_examples:
        # 逐行讲解：抛出异常：RuntimeError('No tokenized SFT example has trainable assistant tokens')。
        raise RuntimeError("No tokenized SFT example has trainable assistant tokens")
    # 逐行讲解：返回结果：processed_examples。
    return processed_examples


# 逐行讲解：定义函数 iter_epoch_batches：封装下方逻辑，供调用方复用。
def iter_epoch_batches(
    examples: list[ProcessedDatum],
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[ProcessedDatum]]:
    # 逐行讲解：调用 list（list(range(len(examples)))），赋给 indices。
    indices = list(range(len(examples)))
    # 逐行讲解：打乱数据顺序（random.Random(seed + epoch).shuffle(indices)）。
    random.Random(seed + epoch).shuffle(indices)
    # 逐行讲解：返回结果：[[examples[index] for index in indices[start:start + batch_size]] for start in range(0, len(indices), batch…。
    return [
        [examples[index] for index in indices[start : start + batch_size]]
        for start in range(0, len(indices), batch_size)
    ]


# 逐行讲解：定义函数 to_float_array：封装下方逻辑，供调用方复用。
def to_float_array(value: Any) -> np.ndarray:
    # 逐行讲解：检查条件 hasattr(value, 'tolist')；成立时执行下方分支。
    if hasattr(value, "tolist"):
        # 逐行讲解：调用 value.tolist（value.tolist()），赋给 value。
        value = value.tolist()
    # 逐行讲解：返回结果：np.asarray(value, dtype=np.float32)。
    return np.asarray(value, dtype=np.float32)


# 逐行讲解：定义函数 weighted_loss：封装下方逻辑，供调用方复用。
def weighted_loss(fwdbwd_result: Any, datums: list[trio.Datum]) -> float:
    # TRIO 返回的是每个 target token 的 logprob；这里按 assistant-only weights 还原平均 CE loss。
    logprobs = np.concatenate(
        [to_float_array(output["logprobs"]) for output in fwdbwd_result.loss_fn_outputs]
    )
    # 逐行讲解：调用 np.concatenate（np.concatenate([to_float_array(datum.loss_fn_inputs['weights']) for datum in datums])），赋给 weights。
    weights = np.concatenate(
        [to_float_array(datum.loss_fn_inputs["weights"]) for datum in datums]
    )
    # 逐行讲解：转换为浮点表示（float(weights.sum())），赋给 weight_sum。
    weight_sum = float(weights.sum())
    # 逐行讲解：检查条件 weight_sum <= 0；成立时执行下方分支。
    if weight_sum <= 0:
        # 逐行讲解：返回结果：0.0。
        return 0.0
    # 逐行讲解：返回结果：float(-np.dot(logprobs, weights) / weight_sum)。
    return float(-np.dot(logprobs, weights) / weight_sum)


# 逐行讲解：定义函数 length_stats：封装下方逻辑，供调用方复用。
def length_stats(values: list[int], prefix: str) -> dict[str, float | int]:
    # 逐行讲解：构造指定类型的 NumPy 数组（np.asarray(values, dtype=np.float32)），赋给 arr。
    arr = np.asarray(values, dtype=np.float32)
    # 逐行讲解：返回结果：{f'{prefix}_mean': float(arr.mean()), f'{prefix}_p50': float(np.percentile(arr, 50)), f'{prefix}_p95': floa…。
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": int(arr.max()),
    }


# 逐行讲解：定义函数 analyze_dataset_tokens：封装下方逻辑，供调用方复用。
def analyze_dataset_tokens(
    processed_examples: list[ProcessedDatum],
    sample_count: int,
) -> dict[str, float | int]:
    # 这里直接复用训练前已经构造好的 Datum 统计，保证统计路径和训练路径完全一致。
    context_lengths = [example.total_tokens for example in processed_examples]
    # 逐行讲解：通过推导式逐项生成结果：[example.prompt_tokens for example in processed_examples]，赋给 prompt_lengths。
    prompt_lengths = [example.prompt_tokens for example in processed_examples]
    # 逐行讲解：通过推导式逐项生成结果：[example.trainable_tokens for example in processed_examples]，赋给 trainable_lengths。
    trainable_lengths = [example.trainable_tokens for example in processed_examples]
    # 逐行讲解：求和（sum((1 for example in processed_examples if example.truncated))），赋给 truncated_count。
    truncated_count = sum(1 for example in processed_examples if example.truncated)
    # 逐行讲解：计算 sample_count - len(processed_examples)，赋给 skipped_count。
    skipped_count = sample_count - len(processed_examples)
    # 逐行讲解：调用 len（len(context_lengths)），赋给 valid_count。
    valid_count = len(context_lengths)
    # 逐行讲解：声明 stats 的类型为 dict[str, float | int]；构造字典 {'sample_count': sample_count, 'valid_sample_count': valid_count, 'skipped_sample_count': skipped_count, 't…。
    stats: dict[str, float | int] = {
        "sample_count": sample_count,
        "valid_sample_count": valid_count,
        "skipped_sample_count": skipped_count,
        "truncated_sample_count": truncated_count,
        "truncated_sample_rate": truncated_count / valid_count,
    }
    # 逐行讲解：调用 stats.update（stats.update(length_stats(context_lengths, 'context_tokens'))）。
    stats.update(length_stats(context_lengths, "context_tokens"))
    # 逐行讲解：调用 stats.update（stats.update(length_stats(prompt_lengths, 'prompt_tokens'))）。
    stats.update(length_stats(prompt_lengths, "prompt_tokens"))
    # 逐行讲解：调用 stats.update（stats.update(length_stats(trainable_lengths, 'trainable_tokens'))）。
    stats.update(length_stats(trainable_lengths, "trainable_tokens"))
    # 逐行讲解：返回结果：stats。
    return stats


# 逐行讲解：定义函数 print_dataset_token_stats：封装下方逻辑，供调用方复用。
def print_dataset_token_stats(stats: dict[str, float | int]) -> None:
    # 逐行讲解：调用 print（print('=' * 25 + 'Dataset token stats' + '=' * 25)）。
    print("=" * 25 + "Dataset token stats" + "=" * 25)
    # 逐行讲解：调用 print（print(f"- 样本数: {stats['valid_sample_count']}/{stats['sample_count']} | 跳过 {stats['skipped_sample_count']} |…）。
    print(
        f"- 样本数: {stats['valid_sample_count']}/{stats['sample_count']} | "
        f"跳过 {stats['skipped_sample_count']} | "
        f"截断 {stats['truncated_sample_count']} ({stats['truncated_sample_rate']:.2%})"
    )
    # 逐行讲解：调用 print（print(f"- 总 context tokens(prompt+assistant+EOS): mean {stats['context_tokens_mean']:.1f} | p50 {stats['con…）。
    print(
        f"- 总 context tokens(prompt+assistant+EOS): mean {stats['context_tokens_mean']:.1f} | "
        f"p50 {stats['context_tokens_p50']:.1f} | p95 {stats['context_tokens_p95']:.1f} | "
        f"max {stats['context_tokens_max']}"
    )
    # 逐行讲解：调用 print（print(f"- prompt tokens(system+user): mean {stats['prompt_tokens_mean']:.1f} | p50 {stats['prompt_tokens_p5…）。
    print(
        f"- prompt tokens(system+user): mean {stats['prompt_tokens_mean']:.1f} | "
        f"p50 {stats['prompt_tokens_p50']:.1f} | p95 {stats['prompt_tokens_p95']:.1f} | "
        f"max {stats['prompt_tokens_max']}"
    )
    # 逐行讲解：调用 print（print(f"- 可训练 tokens(weights=1): mean {stats['trainable_tokens_mean']:.1f} | p50 {stats['trainable_tokens_p…）。
    print(
        f"- 可训练 tokens(weights=1): mean {stats['trainable_tokens_mean']:.1f} | "
        f"p50 {stats['trainable_tokens_p50']:.1f} | p95 {stats['trainable_tokens_p95']:.1f} | "
        f"max {stats['trainable_tokens_max']}"
    )
    # 逐行讲解：调用 print（print('=' * 22 + 'End of dataset token stats' + '=' * 21)）。
    print("=" * 22+ "End of dataset token stats" + "=" * 21)


# 逐行讲解：定义异步函数 log_step_result：封装下方逻辑，供调用方复用。
async def log_step_result(
    fwdbwd_future: Any,
    optim_future: Any,
    batch: list[ProcessedDatum],
    args: argparse.Namespace,
    swanlab_run: Any | None,
    progress_bar: tqdm,
    step_start: float,
    step: int,
    total_steps: int,
    epoch: int,
) -> None:
    # 逐行讲解：等待异步操作 fwdbwd_future 完成并取得结果，赋给 fwdbwd_result。
    fwdbwd_result = await fwdbwd_future
    # 逐行讲解：等待异步操作 optim_future 完成并取得结果。
    await optim_future

    # 逐行讲解：通过推导式逐项生成结果：[item.datum for item in batch]，赋给 datums。
    datums = [item.datum for item in batch]
    # 逐行讲解：调用 int（int(sum((item.total_tokens for item in batch)))），赋给 total_tokens。
    total_tokens = int(sum(item.total_tokens for item in batch))
    # 逐行讲解：调用 int（int(sum((1 for item in batch if item.truncated)))），赋给 truncated。
    truncated = int(sum(1 for item in batch if item.truncated))
    # 逐行讲解：计算 time.time() - step_start，赋给 step_elapsed。
    step_elapsed = time.time() - step_start
    # 逐行讲解：调用 weighted_loss（weighted_loss(fwdbwd_result, datums)），赋给 loss。
    loss = weighted_loss(fwdbwd_result, datums)

    # 逐行讲解：构造字典 {'train/loss': loss, 'train/learning_rate': args.learning_rate, 'train/epoch': epoch + 1, 'train/step': ste…，赋给 metrics。
    metrics = {
        "train/loss": loss,
        "train/learning_rate": args.learning_rate,
        "train/epoch": epoch + 1,
        "train/step": step,
        "data/batch_size": len(datums),
        "data/total_tokens": total_tokens,
        "data/truncated_examples": truncated,
        "time/step_elapsed_time": step_elapsed,
        "time/tokens_per_second": total_tokens / step_elapsed,
    }
    # 逐行讲解：调用 metrics.update（metrics.update({f'trainer/{key}': float(value) for key, value in dict(fwdbwd_result.metrics).items()})）。
    metrics.update({f"trainer/{key}": float(value) for key, value in dict(fwdbwd_result.metrics).items()})
    # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
    if swanlab_run is not None:
        # 逐行讲解：计算自然对数（swanlab.log(metrics, step=step)）。
        swanlab.log(metrics, step=step)

    # 逐行讲解：调用 progress_bar.update（progress_bar.update(1)）。
    progress_bar.update(1)
    # 逐行讲解：调用 progress_bar.set_postfix（progress_bar.set_postfix(loss=f'{loss:.4f}', epoch=f'{epoch + 1}/{args.num_epochs}')）。
    progress_bar.set_postfix(loss=f"{loss:.4f}", epoch=f"{epoch + 1}/{args.num_epochs}")
    # 逐行讲解：调用 tqdm.write（tqdm.write(f'step {step:04d}/{total_steps} | epoch {epoch + 1}/{args.num_epochs} | loss {loss:.4f} | batch …）。
    tqdm.write(
        f"step {step:04d}/{total_steps} | epoch {epoch + 1}/{args.num_epochs} | "
        f"loss {loss:.4f} | batch {len(datums)} | time {step_elapsed:.2f}s"
    )


# 逐行讲解：定义函数 start_swanlab：封装下方逻辑，供调用方复用。
def start_swanlab(
    args: argparse.Namespace,
    dataset_size: int,
    total_steps: int,
    token_stats: dict[str, float | int],
) -> Any | None:
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
    # 逐行讲解：取值 args.swanlab_name，赋给 config['run_name']。
    config["run_name"] = args.swanlab_name
    # 逐行讲解：取值 args.save_state_name，赋给 config['state_name']。
    config["state_name"] = args.save_state_name
    # 逐行讲解：取值 args.save_weights_name，赋给 config['weights_name']。
    config["weights_name"] = args.save_weights_name
    # 逐行讲解：取值 token_stats，赋给 config['dataset_token_stats']。
    config["dataset_token_stats"] = token_stats
    # 逐行讲解：返回结果：swanlab.init(project=args.swanlab_project, name=args.swanlab_name, workspace=args.swanlab_workspace, mode=a…。
    return swanlab.init(
        project=args.swanlab_project,
        name=args.swanlab_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=["PyTrio", "SFT", "medical", "async"],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


# 逐行讲解：定义异步函数 save_training_state：封装下方逻辑，供调用方复用。
async def save_training_state(
    training_client: Any,
    swanlab_run: Any | None,
    name: str,
    step: int,
    tag: str,
) -> str:
    # save_state 保存完整训练状态；后续 OPD 可用 create_training_client_from_state_async 初始化 student。
    save_future = await training_client.save_state_async(name=name)
    # 逐行讲解：等待异步操作 save_future 完成并取得结果，赋给 save_result。
    save_result = await save_future
    # 逐行讲解：调用 print（print(f'Saved state [{tag}]: {save_result.path}')）。
    print(f"Saved state [{tag}]: {save_result.path}")
    # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
    if swanlab_run is not None:
        # 逐行讲解：计算自然对数（swanlab.log({f'save/{tag}_state_path': swanlab.Text(save_result.path)}, step=step)）。
        swanlab.log({f"save/{tag}_state_path": swanlab.Text(save_result.path)}, step=step)
    # 逐行讲解：返回结果：save_result.path。
    return save_result.path


# 逐行讲解：定义异步函数 save_sampler_weights：封装下方逻辑，供调用方复用。
async def save_sampler_weights(
    training_client: Any,
    swanlab_run: Any | None,
    name: str,
    step: int,
    tag: str,
) -> str:
    # 逐行讲解：等待异步操作 training_client.save_weights_for_sampler_async(name=name) 完成并取得结果，赋给 save_future。
    save_future = await training_client.save_weights_for_sampler_async(name=name)
    # 逐行讲解：等待异步操作 save_future 完成并取得结果，赋给 save_result。
    save_result = await save_future
    # 逐行讲解：调用 print（print(f'Saved weights [{tag}]: {save_result.path}')）。
    print(f"Saved weights [{tag}]: {save_result.path}")
    # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
    if swanlab_run is not None:
        # 逐行讲解：计算自然对数（swanlab.log({f'save/{tag}_weights_path': swanlab.Text(save_result.path)}, step=step)）。
        swanlab.log({f"save/{tag}_weights_path": swanlab.Text(save_result.path)}, step=step)
    # 逐行讲解：返回结果：save_result.path。
    return save_result.path


# 逐行讲解：定义异步函数 save_checkpoint：封装下方逻辑，供调用方复用。
async def save_checkpoint(
    training_client: Any,
    swanlab_run: Any | None,
    state_name: str,
    weights_name: str,
    step: int,
    tag: str,
) -> tuple[str, str]:
    # 逐行讲解：等待异步操作 save_training_state(training_client=training_client, swanlab_run=swanlab_run, name=state_name, step=step, t… 完成并取得结果，赋给 state_path。
    state_path = await save_training_state(
        training_client=training_client,
        swanlab_run=swanlab_run,
        name=state_name,
        step=step,
        tag=tag,
    )
    # 逐行讲解：等待异步操作 save_sampler_weights(training_client=training_client, swanlab_run=swanlab_run, name=weights_name, step=step… 完成并取得结果，赋给 weights_path。
    weights_path = await save_sampler_weights(
        training_client=training_client,
        swanlab_run=swanlab_run,
        name=weights_name,
        step=step,
        tag=tag,
    )
    # 逐行讲解：返回结果：(state_path, weights_path)。
    return state_path, weights_path


# 逐行讲解：定义异步函数 train：封装下方逻辑，供调用方复用。
async def train(args: argparse.Namespace) -> None:
    # 逐行讲解：设置对应随机数生成器的种子（random.seed(args.seed)）。
    random.seed(args.seed)

    # 逐行讲解：调用 load_examples（load_examples(args)），赋给 examples。
    examples = load_examples(args)
    # 逐行讲解：调用 print（print(f'Loaded {len(examples)} medical SFT examples')）。
    print(f"Loaded {len(examples)} medical SFT examples")
    # 逐行讲解：调用 print（print(f'Run name: {args.swanlab_name}')）。
    print(f"Run name: {args.swanlab_name}")
    # 逐行讲解：调用 print（print(f'State name: {args.save_state_name}')）。
    print(f"State name: {args.save_state_name}")
    # 逐行讲解：调用 print（print(f'Sampler weights name: {args.save_weights_name}')）。
    print(f"Sampler weights name: {args.save_weights_name}")

    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：等待异步操作 service_client.create_lora_training_client_async(base_model=args.base_model, rank=args.lora_rank, seed=args… 完成并取得结果，赋给 training_client。
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    # 逐行讲解：调用 training_client.get_tokenizer（training_client.get_tokenizer()），赋给 tokenizer。
    tokenizer = training_client.get_tokenizer()
    # 逐行讲解：调用 build_processed_examples（build_processed_examples(examples, tokenizer, args)），赋给 processed_examples。
    processed_examples = build_processed_examples(examples, tokenizer, args)
    # 逐行讲解：调用 print（print(f'Prepared {len(processed_examples)} valid SFT datums')）。
    print(f"Prepared {len(processed_examples)} valid SFT datums")

    # 逐行讲解：计算 (len(processed_examples) + args.batch_size - 1) // args.batch_size，赋给 steps_per_epoch。
    steps_per_epoch = (len(processed_examples) + args.batch_size - 1) // args.batch_size
    # 逐行讲解：计算 args.num_epochs * steps_per_epoch，赋给 planned_steps。
    planned_steps = args.num_epochs * steps_per_epoch
    # 逐行讲解：按条件选择结果：min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps，赋给 total_steps。
    total_steps = min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps

    # 逐行讲解：调用 analyze_dataset_tokens（analyze_dataset_tokens(processed_examples, len(examples))），赋给 token_stats。
    token_stats = analyze_dataset_tokens(processed_examples, len(examples))
    # 逐行讲解：调用 print_dataset_token_stats（print_dataset_token_stats(token_stats)）。
    print_dataset_token_stats(token_stats)

    # 逐行讲解：调用 trio.AdamParams（trio.AdamParams(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2)），赋给 adam_params。
    adam_params = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    # 逐行讲解：调用 start_swanlab（start_swanlab(args, len(processed_examples), total_steps, token_stats)），赋给 swanlab_run。
    swanlab_run = start_swanlab(args, len(processed_examples), total_steps, token_stats)
    # 逐行讲解：取值 0，赋给 submitted_steps。
    submitted_steps = 0
    # 逐行讲解：取值 0，赋给 last_checkpoint_step。
    last_checkpoint_step = 0

    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：进入上下文管理器 tqdm(total=total_steps, desc='Medical SFT async', unit='step')，退出时执行其收尾逻辑。
        with tqdm(total=total_steps, desc="Medical SFT async", unit="step") as progress_bar:
            # 逐行讲解：遍历 range(args.num_epochs)，将每项绑定到 epoch。
            for epoch in range(args.num_epochs):
                # 逐行讲解：计算 total_steps - submitted_steps，赋给 remaining_steps。
                remaining_steps = total_steps - submitted_steps
                # 逐行讲解：检查条件 remaining_steps <= 0；成立时执行下方分支。
                if remaining_steps <= 0:
                    # 逐行讲解：结束当前循环。
                    break

                # 逐行讲解：调用 min（min(steps_per_epoch, remaining_steps)），赋给 epoch_submit_steps。
                epoch_submit_steps = min(steps_per_epoch, remaining_steps)
                # 逐行讲解：构造列表 []，赋给 log_tasks。
                log_tasks = []
                # 逐行讲解：调用 tqdm（tqdm(total=epoch_submit_steps, desc=f'Epoch {epoch + 1}/{args.num_epochs} submit', unit='step', leave=False)），赋给 submit_bar。
                submit_bar = tqdm(
                    total=epoch_submit_steps,
                    desc=f"Epoch {epoch + 1}/{args.num_epochs} submit",
                    unit="step",
                    leave=False,
                )

                # 逐行讲解：遍历 iter_epoch_batches(processed_examples, args.batch_size, args.seed, epoch)，将每项绑定到 batch。
                for batch in iter_epoch_batches(processed_examples, args.batch_size, args.seed, epoch):
                    # 逐行讲解：检查条件 submitted_steps >= total_steps；成立时执行下方分支。
                    if submitted_steps >= total_steps:
                        # 逐行讲解：结束当前循环。
                        break

                    # 逐行讲解：调用 time.time（time.time()），赋给 step_start。
                    step_start = time.time()
                    # 逐行讲解：取值 submitted_steps，赋给 step。
                    step = submitted_steps
                    # 逐行讲解：通过推导式逐项生成结果：[item.datum for item in batch]，赋给 datums。
                    datums = [item.datum for item in batch]

                    # 连续提交远程训练任务，loss/SwanLab 记录放到后台 task。
                    fwdbwd_future = await training_client.forward_backward_async(
                        datums,
                        loss_fn="cross_entropy",
                    )
                    # 逐行讲解：等待异步操作 training_client.optim_step_async(adam_params) 完成并取得结果，赋给 optim_future。
                    optim_future = await training_client.optim_step_async(adam_params)
                    # 逐行讲解：调用 submit_bar.update（submit_bar.update(1)）。
                    submit_bar.update(1)
                    # 逐行讲解：向列表末尾追加一个元素（log_tasks.append(asyncio.create_task(log_step_result(fwdbwd_future=fwdbwd_future, optim_future=optim_future…）。
                    log_tasks.append(
                        asyncio.create_task(
                            log_step_result(
                                fwdbwd_future=fwdbwd_future,
                                optim_future=optim_future,
                                batch=batch,
                                args=args,
                                swanlab_run=swanlab_run,
                                progress_bar=progress_bar,
                                step_start=step_start,
                                step=step,
                                total_steps=total_steps,
                                epoch=epoch,
                            )
                        )
                    )
                    # 逐行讲解：更新 submitted_steps：submitted_steps += 1。
                    submitted_steps += 1
                    # 逐行讲解：检查条件 args.save_every_steps > 0 and submitted_steps % args.save_every_steps == 0；成立时执行下方分支。
                    if args.save_every_steps > 0 and submitted_steps % args.save_every_steps == 0:
                        # 按 step 保存前，先等已提交任务完成，确保 checkpoint 对齐到明确的 step。
                        await asyncio.gather(*log_tasks)
                        # 逐行讲解：构造列表 []，赋给 log_tasks。
                        log_tasks = []
                        # 逐行讲解：等待异步操作 save_checkpoint(training_client=training_client, swanlab_run=swanlab_run, state_name=f'{args.save_state_nam… 完成并取得结果。
                        await save_checkpoint(
                            training_client=training_client,
                            swanlab_run=swanlab_run,
                            state_name=f"{args.save_state_name}-step{submitted_steps:06d}",
                            weights_name=f"{args.save_weights_name}-step{submitted_steps:06d}",
                            step=submitted_steps,
                            tag=f"step{submitted_steps:06d}",
                        )
                        # 逐行讲解：取值 submitted_steps，赋给 last_checkpoint_step。
                        last_checkpoint_step = submitted_steps

                # 逐行讲解：调用 submit_bar.close（submit_bar.close()）。
                submit_bar.close()
                # 逐行讲解：检查条件 log_tasks；成立时执行下方分支。
                if log_tasks:
                    # 逐行讲解：等待异步操作 asyncio.gather(*log_tasks) 完成并取得结果。
                    await asyncio.gather(*log_tasks)
                # 逐行讲解：检查条件 args.save_each_epoch and submitted_steps % steps_per_epoch == 0；成立时执行下方分支。
                if args.save_each_epoch and submitted_steps % steps_per_epoch == 0:
                    # 逐行讲解：等待异步操作 save_checkpoint(training_client=training_client, swanlab_run=swanlab_run, state_name=f'{args.save_state_nam… 完成并取得结果。
                    await save_checkpoint(
                        training_client=training_client,
                        swanlab_run=swanlab_run,
                        state_name=f"{args.save_state_name}-epoch{epoch + 1:03d}",
                        weights_name=f"{args.save_weights_name}-epoch{epoch + 1:03d}",
                        step=submitted_steps,
                        tag=f"epoch{epoch + 1:03d}",
                    )
                    # 逐行讲解：取值 submitted_steps，赋给 last_checkpoint_step。
                    last_checkpoint_step = submitted_steps

                # 逐行讲解：检查条件 submitted_steps >= total_steps；成立时执行下方分支。
                if submitted_steps >= total_steps:
                    # 逐行讲解：结束当前循环。
                    break

        # 逐行讲解：检查条件 submitted_steps == 0；成立时执行下方分支。
        if submitted_steps == 0:
            # 逐行讲解：抛出异常：RuntimeError('No SFT training step was completed')。
            raise RuntimeError("No SFT training step was completed")
        # 逐行讲解：检查条件 last_checkpoint_step != submitted_steps；成立时执行下方分支。
        if last_checkpoint_step != submitted_steps:
            # 逐行讲解：等待异步操作 save_checkpoint(training_client=training_client, swanlab_run=swanlab_run, state_name=f'{args.save_state_nam… 完成并取得结果。
            await save_checkpoint(
                training_client=training_client,
                swanlab_run=swanlab_run,
                state_name=f"{args.save_state_name}-step{submitted_steps:06d}",
                weights_name=f"{args.save_weights_name}-step{submitted_steps:06d}",
                step=submitted_steps,
                tag=f"step{submitted_steps:06d}",
            )
        # 逐行讲解：调用 print（print(f'Completed {submitted_steps} SFT steps')）。
        print(f"Completed {submitted_steps} SFT steps")
    finally:
        # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
        if swanlab_run is not None:
            # 逐行讲解：调用 swanlab.finish（swanlab.finish()）。
            swanlab.finish()


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：调用 time.time（time.time()），赋给 start。
    start = time.time()
    # 逐行讲解：调用 asyncio.run（asyncio.run(train(parse_args()))）。
    asyncio.run(train(parse_args()))
    # 逐行讲解：调用 print（print('#' * 50)）。
    print("#" * 50)
    # 逐行讲解：调用 print（print('# all done')）。
    print("# all done")
    # 逐行讲解：调用 print（print(f'# train cost {time.time() - start:.2f}s')）。
    print(f"# train cost {time.time() - start:.2f}s")
    # 逐行讲解：调用 print（print('#' * 50)）。
    print("#" * 50)


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
