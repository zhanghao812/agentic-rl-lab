"""异步版医疗 OPD：medical SFT 4B teacher -> fresh 4B base student。

本脚本和 02-medical-sft.py 使用同一份 medical-o1-reasoning-SFT-zh 数据：
student 只读取 question 并按当前策略采样，SFT teacher 对 student 的实际轨迹计算
逐 token logprob，再用 reverse KL advantage 更新 fresh LoRA student。

异步加速发生在同一 batch 的多个 student rollout，以及每条 rollout 的 teacher
logprob 计算。每一步仍会等待 optimizer 完成后再刷新 student sampler，保持 on-policy。

MedQA-ZH 和 C-Eval 都不进入训练。它们只用于评测 step checkpoint：
- MedQA-ZH @ max_tokens=1024 检查短输出预算下的医疗表现；
- C-Eval 检查通用能力是否下降。

小成本试跑：
uv run python 03-medical-opd-async.py \
    --teacher-model-path YOUR_SFT_SAMPLER_WEIGHTS_PATH \
    --steps 10 \
    --batch-size 2 \
    --group-size 2 \
    --sample-size 100 \
    --max-tokens 2048 \
    --save-every-steps 5 \
    --swanlab-mode disabled

第一轮正式试验：
uv run python 03-medical-opd-async.py \
    --teacher-model-path YOUR_SFT_SAMPLER_WEIGHTS_PATH \
    --steps 300 \
    --batch-size 4 \
    --group-size 4 \
    --sample-size 0 \
    --max-tokens 2048 \
    --learning-rate 4e-5 \
    --save-every-steps 300 \
    --swanlab-mode online

注意：`--max-tokens 2048` 是 completion 输出预算，不是总上下文长度。
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
DEFAULT_DATASET_PATH = (
    SCRIPT_DIR / "datasets" / "medical-o1-reasoning-SFT-zh" / "train.jsonl"
)
# 逐行讲解：取值 '你是一个中文医疗问答助手。请根据题目给出严谨的医学推理和最终答案。'，赋给 DEFAULT_SYSTEM_MESSAGE。
DEFAULT_SYSTEM_MESSAGE = "你是一个中文医疗问答助手。请根据题目给出严谨的医学推理和最终答案。"
# 逐行讲解：构造元组 ('importance_sampling', 'ppo')，赋给 LOSS_FNS。
LOSS_FNS = ("importance_sampling", "ppo")


# 逐行讲解：定义类 PromptRollout：封装下方逻辑，供调用方复用。
@dataclass(frozen=True)
class PromptRollout:
    # 逐行讲解：声明 datums 的类型为 list[trio.Datum]。
    datums: list[trio.Datum]
    # 逐行讲解：声明 reverse_kls 的类型为 list[float]。
    reverse_kls: list[float]
    # 逐行讲解：声明 completion_token_counts 的类型为 list[int]。
    completion_token_counts: list[int]


# 逐行讲解：定义类 ChineseArgumentParser：封装下方逻辑，供调用方复用。
class ChineseArgumentParser(argparse.ArgumentParser):
    # 逐行讲解：定义函数 format_usage：封装下方逻辑，供调用方复用。
    def format_usage(self) -> str:
        # 逐行讲解：返回结果：super().format_usage().replace('usage:', '用法:', 1)。
        return super().format_usage().replace("usage:", "用法:", 1)

    # 逐行讲解：定义函数 format_help：封装下方逻辑，供调用方复用。
    def format_help(self) -> str:
        # 逐行讲解：返回结果：super().format_help().replace('usage:', '用法:', 1)。
        return super().format_help().replace("usage:", "用法:", 1)


# 逐行讲解：定义函数 model_slug：封装下方逻辑，供调用方复用。
def model_slug(base_model: str) -> str:
    # 逐行讲解：调用 base_model.rsplit('/', 1)[-1].lower().replace（base_model.rsplit('/', 1)[-1].lower().replace('qwen3.5', 'qwen35')），赋给 name。
    name = base_model.rsplit("/", 1)[-1].lower().replace("qwen3.5", "qwen35")
    # 逐行讲解：返回结果：re.sub('[^a-z0-9]+', '-', name).strip('-')。
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


# 逐行讲解：定义函数 dataset_slug：封装下方逻辑，供调用方复用。
def dataset_slug(path: Path) -> str:
    # 逐行讲解：调用 path.parent.name.lower().replace（path.parent.name.lower().replace('_', '-')），赋给 name。
    name = path.parent.name.lower().replace("_", "-")
    # 逐行讲解：返回结果：re.sub('[^a-z0-9-]+', '-', name).strip('-')。
    return re.sub(r"[^a-z0-9-]+", "-", name).strip("-")


# 逐行讲解：定义函数 build_run_name：封装下方逻辑，供调用方复用。
def build_run_name(args: argparse.Namespace) -> str:
    # 逐行讲解：调用 args.loss_fn.replace（args.loss_fn.replace('_', '-')），赋给 loss_slug。
    loss_slug = args.loss_fn.replace("_", "-")
    # 逐行讲解：按条件选择结果：'full' if args.steps == 0 else f'steps{args.steps}'，赋给 steps_slug。
    steps_slug = "full" if args.steps == 0 else f"steps{args.steps}"
    # 逐行讲解：返回结果：f'opd-medical-async-{model_slug(args.base_model)}-{dataset_slug(args.dataset_path)}-{loss_slug}-{steps_slug}'。
    return (
        f"opd-medical-async-{model_slug(args.base_model)}-"
        f"{dataset_slug(args.dataset_path)}-{loss_slug}-{steps_slug}"
    )


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 ChineseArgumentParser（ChineseArgumentParser(description='PyTRIO 异步版医疗 OPD：SFT teacher -> fresh base student', add_help=False)），赋给 parser。
    parser = ChineseArgumentParser(
        description="PyTRIO 异步版医疗 OPD：SFT teacher -> fresh base student",
        add_help=False,
    )
    # 逐行讲解：取值 '选项'，赋给 parser._optionals.title。
    parser._optionals.title = "选项"
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('-h', '--help', action='help', help='显示帮助信息并退出')）。
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-path', type=Path, default=DEFAULT_DATASET_PATH, help='medical-o1-reasoning-S…）。
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="medical-o1-reasoning-SFT-zh train JSONL",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sample-size', type=int, default=0, help='随机抽样 prompt 数；0 表示全用')）。
    parser.add_argument("--sample-size", type=int, default=0, help="随机抽样 prompt 数；0 表示全用")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42, help='随机种子')）。
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B', help='fresh LoRA student 的基础模型')）。
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B", help="fresh LoRA student 的基础模型")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=16, help='fresh student 的 LoRA rank，范围 4-64')）。
    parser.add_argument("--lora-rank", type=int, default=16, help="fresh student 的 LoRA rank，范围 4-64")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--teacher-base-model', default=None, help='SFT teacher 的基础模型；默认和 student base 相同')）。
    parser.add_argument(
        "--teacher-base-model",
        default=None,
        help="SFT teacher 的基础模型；默认和 student base 相同",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--teacher-model-path', required=True, help='medical SFT 的 trio:// sampler_weights 路径')）。
    parser.add_argument(
        "--teacher-model-path",
        required=True,
        help="medical SFT 的 trio:// sampler_weights 路径",
    )

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--steps', type=int, default=300, help='训练 step 数；0 表示完整遍历当前数据一次')）。
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="训练 step 数；0 表示完整遍历当前数据一次",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=2, help='每个 step 使用的医疗 prompt 数')）。
    parser.add_argument("--batch-size", type=int, default=2, help="每个 step 使用的医疗 prompt 数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--group-size', type=int, default=4, help='每个 prompt 的 student completion 数')）。
    parser.add_argument("--group-size", type=int, default=4, help="每个 prompt 的 student completion 数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=2048, help='每条 completion 最多生成 token 数')）。
    parser.add_argument("--max-tokens", type=int, default=2048, help="每条 completion 最多生成 token 数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0, help='student rollout temperature')）。
    parser.add_argument("--temperature", type=float, default=1.0, help="student rollout temperature")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0, help='student rollout top_p')）。
    parser.add_argument("--top-p", type=float, default=1.0, help="student rollout top_p")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-k', type=int, default=-1, help='student rollout top_k；-1 表示不限制')）。
    parser.add_argument("--top-k", type=int, default=-1, help="student rollout top_k；-1 表示不限制")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--system-message', default=DEFAULT_SYSTEM_MESSAGE, help='与 medical SFT 一致的 system mess…）。
    parser.add_argument(
        "--system-message",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="与 medical SFT 一致的 system message；空字符串表示不加",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=True, help='chat te…）。
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="chat template 是否启用 thinking",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--kl-penalty-coef', type=float, default=1.0, help='reverse KL advantage 系数')）。
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0, help="reverse KL advantage 系数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=4e-05, help='Adam 学习率')）。
    parser.add_argument("--learning-rate", type=float, default=4e-5, help="Adam 学习率")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')）。
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')）。
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sampler-refresh-steps', type=int, default=1, help='每隔多少 step 用最新 student 权重刷新 sample…）。
    parser.add_argument(
        "--sampler-refresh-steps",
        type=int,
        default=1,
        help="每隔多少 step 用最新 student 权重刷新 sampler",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--loss-fn', choices=LOSS_FNS, default='ppo', help='PyTRIO 内置 RL loss')）。
    parser.add_argument("--loss-fn", choices=LOSS_FNS, default="ppo", help="PyTRIO 内置 RL loss")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every-steps', type=int, default=300, help='每多少 step 保存一次 state + sampler weight…）。
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=300,
        help="每多少 step 保存一次 state + sampler weights；0 表示不保存中间 checkpoint，最终 step 仍保存",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab', action=argparse.BooleanOptionalAction, default=True, help='是否启用 SwanLab')）。
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=True, help="是否启用 SwanLab")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-opd', help='SwanLab 项目名')）。
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-opd", help="SwanLab 项目名")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-workspace', default=None, help='SwanLab workspace')）。
    parser.add_argument("--swanlab-workspace", default=None, help="SwanLab workspace")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=['online', 'local', 'offline', 'disabled'], default=None, hel…）。
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default=None,
        help="SwanLab 运行模式",
    )
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()

    # 逐行讲解：遍历 ('batch_size', 'group_size', 'max_tokens', 'sampler_refresh_steps')，将每项绑定到 name。
    for name in ("batch_size", "group_size", "max_tokens", "sampler_refresh_steps"):
        # 逐行讲解：检查条件 getattr(args, name) < 1；成立时执行下方分支。
        if getattr(args, name) < 1:
            # 逐行讲解：抛出异常：ValueError(f"--{name.replace('_', '-')} must be >= 1")。
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    # 逐行讲解：检查条件 args.steps < 0；成立时执行下方分支。
    if args.steps < 0:
        # 逐行讲解：抛出异常：ValueError('--steps must be >= 0')。
        raise ValueError("--steps must be >= 0")
    # 逐行讲解：检查条件 args.sample_size < 0；成立时执行下方分支。
    if args.sample_size < 0:
        # 逐行讲解：抛出异常：ValueError('--sample-size must be >= 0')。
        raise ValueError("--sample-size must be >= 0")
    # 逐行讲解：检查条件 not 4 <= args.lora_rank <= 64；成立时执行下方分支。
    if not 4 <= args.lora_rank <= 64:
        # 逐行讲解：抛出异常：ValueError('--lora-rank must be between 4 and 64')。
        raise ValueError("--lora-rank must be between 4 and 64")
    # 逐行讲解：检查条件 args.kl_penalty_coef <= 0；成立时执行下方分支。
    if args.kl_penalty_coef <= 0:
        # 逐行讲解：抛出异常：ValueError('--kl-penalty-coef must be > 0')。
        raise ValueError("--kl-penalty-coef must be > 0")
    # 逐行讲解：检查条件 args.save_every_steps < 0；成立时执行下方分支。
    if args.save_every_steps < 0:
        # 逐行讲解：抛出异常：ValueError('--save-every-steps must be >= 0')。
        raise ValueError("--save-every-steps must be >= 0")

    # 逐行讲解：取值 args.teacher_base_model or args.base_model，赋给 args.teacher_base_model。
    args.teacher_base_model = args.teacher_base_model or args.base_model
    # 逐行讲解：调用 build_run_name（build_run_name(args)），赋给 args.run_name。
    args.run_name = build_run_name(args)
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 load_medical_prompts：封装下方逻辑，供调用方复用。
def load_medical_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    # 逐行讲解：检查条件 not args.dataset_path.exists()；成立时执行下方分支。
    if not args.dataset_path.exists():
        # 逐行讲解：抛出异常：FileNotFoundError(f'找不到 medical-o1 数据：{args.dataset_path}，请先运行 00-download-dataset.py')。
        raise FileNotFoundError(
            f"找不到 medical-o1 数据：{args.dataset_path}，请先运行 00-download-dataset.py"
        )

    # 逐行讲解：声明 rows 的类型为 list[dict[str, str]]；构造列表 []。
    rows: list[dict[str, str]] = []
    # 逐行讲解：进入上下文管理器 args.dataset_path.open('r', encoding='utf-8')，退出时执行其收尾逻辑。
    with args.dataset_path.open("r", encoding="utf-8") as f:
        # 逐行讲解：遍历 enumerate(f, start=1)，将每项绑定到 (line_number, line)。
        for line_number, line in enumerate(f, start=1):
            # 逐行讲解：检查条件 not line.strip()；成立时执行下方分支。
            if not line.strip():
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(line)），赋给 row。
            row = json.loads(line)
            # 逐行讲解：调用 str(row.get('question', '')).strip（str(row.get('question', '')).strip()），赋给 question。
            question = str(row.get("question", "")).strip()
            # 逐行讲解：检查条件 not question；成立时执行下方分支。
            if not question:
                # 逐行讲解：抛出异常：ValueError(f'medical-o1 第 {line_number} 行缺少 question')。
                raise ValueError(f"medical-o1 第 {line_number} 行缺少 question")
            # 逐行讲解：向列表末尾追加一个元素（rows.append({'question': question})）。
            rows.append({"question": question})

    # 逐行讲解：检查条件 not rows；成立时执行下方分支。
    if not rows:
        # 逐行讲解：抛出异常：ValueError(f'medical-o1 数据为空：{args.dataset_path}')。
        raise ValueError(f"medical-o1 数据为空：{args.dataset_path}")
    # 逐行讲解：打乱数据顺序（random.Random(args.seed).shuffle(rows)）。
    random.Random(args.seed).shuffle(rows)
    # 逐行讲解：检查条件 args.sample_size > 0；成立时执行下方分支。
    if args.sample_size > 0:
        # 逐行讲解：按索引或切片读取 rows[:min(args.sample_size, len(rows))]，赋给 rows。
        rows = rows[: min(args.sample_size, len(rows))]
    # 逐行讲解：返回结果：rows。
    return rows


# 逐行讲解：定义函数 build_prompt_tokens：封装下方逻辑，供调用方复用。
def build_prompt_tokens(
    tokenizer: Any,
    question: str,
    system_message: str,
    enable_thinking: bool,
) -> list[int]:
    # 逐行讲解：构造列表 []，赋给 messages。
    messages = []
    # 逐行讲解：检查条件 system_message.strip()；成立时执行下方分支。
    if system_message.strip():
        # 逐行讲解：向列表末尾追加一个元素（messages.append({'role': 'system', 'content': system_message.strip()})）。
        messages.append({"role": "system", "content": system_message.strip()})
    # 逐行讲解：向列表末尾追加一个元素（messages.append({'role': 'user', 'content': question.strip()})）。
    messages.append({"role": "user", "content": question.strip()})
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


# 逐行讲解：定义异步函数 completion_teacher_logprobs_async：封装下方逻辑，供调用方复用。
async def completion_teacher_logprobs_async(
    teacher_client: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> list[float]:
    # 逐行讲解：计算 prompt_ids + completion_ids，赋给 all_ids。
    all_ids = prompt_ids + completion_ids
    # 逐行讲解：等待异步操作 teacher_client.compute_logprobs_async(trio.ModelInput.from_ints(all_ids)) 完成并取得结果，赋给 all_logprobs。
    all_logprobs = await teacher_client.compute_logprobs_async(
        trio.ModelInput.from_ints(all_ids)
    )
    # 逐行讲解：按索引或切片读取 all_logprobs[len(prompt_ids):]，赋给 completion_logprobs。
    completion_logprobs = all_logprobs[len(prompt_ids) :]
    # 逐行讲解：检查条件 len(completion_logprobs) != len(completion_ids) or any((value is None for value in completion_logprobs))；成立时执行下方分支。
    if len(completion_logprobs) != len(completion_ids) or any(
        value is None for value in completion_logprobs
    ):
        # 逐行讲解：抛出异常：ValueError('Invalid teacher logprobs for completion tokens')。
        raise ValueError("Invalid teacher logprobs for completion tokens")
    # 逐行讲解：返回结果：[float(value) for value in completion_logprobs]。
    return [float(value) for value in completion_logprobs]


# 逐行讲解：定义函数 build_opd_datum：封装下方逻辑，供调用方复用。
def build_opd_datum(
    prompt_ids: list[int],
    completion_ids: list[int],
    old_logprobs: list[float],
    advantages: list[float] | np.ndarray,
) -> trio.Datum:
    # 逐行讲解：计算 len(prompt_ids) - 1，赋给 prompt_loss_len。
    prompt_loss_len = len(prompt_ids) - 1
    # 逐行讲解：计算 prompt_ids + completion_ids[:-1]，赋给 input_ids。
    input_ids = prompt_ids + completion_ids[:-1]
    # 逐行讲解：计算 [0] * prompt_loss_len + completion_ids，赋给 target_ids。
    target_ids = [0] * prompt_loss_len + completion_ids
    # 逐行讲解：计算 [0.0] * prompt_loss_len + old_logprobs，赋给 padded_logprobs。
    padded_logprobs = [0.0] * prompt_loss_len + old_logprobs
    # 逐行讲解：计算 [0.0] * prompt_loss_len + list(advantages)，赋给 padded_advantages。
    padded_advantages = [0.0] * prompt_loss_len + list(advantages)
    # 逐行讲解：检查条件 not len(input_ids) == len(target_ids) == len(padded_logprobs) == len(padded_advantages)；成立时执行下方分支。
    if not (
        len(input_ids)
        == len(target_ids)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        # 逐行讲解：抛出异常：ValueError('OPD datum fields must have the same length')。
        raise ValueError("OPD datum fields must have the same length")
    # 逐行讲解：返回结果：trio.Datum(model_input=trio.ModelInput.from_ints(input_ids), loss_fn_inputs={'target_tokens': np.asarray(ta…。
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_ids, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )


# 逐行讲解：定义异步函数 run_prompt_rollout_async：封装下方逻辑，供调用方复用。
async def run_prompt_rollout_async(
    student_sampler: Any,
    teacher_client: Any,
    tokenizer: Any,
    row: dict[str, str],
    args: argparse.Namespace,
    sampling_params: trio.SamplingParams,
) -> PromptRollout:
    # 逐行讲解：调用 build_prompt_tokens（build_prompt_tokens(tokenizer=tokenizer, question=row['question'], system_message=args.system_message, enab…），赋给 prompt_ids。
    prompt_ids = build_prompt_tokens(
        tokenizer=tokenizer,
        question=row["question"],
        system_message=args.system_message,
        enable_thinking=args.enable_thinking,
    )
    # 逐行讲解：等待异步操作 student_sampler.sample_async(prompt=trio.ModelInput.from_ints(prompt_ids), num_samples=args.group_size, sam… 完成并取得结果，赋给 sample_result。
    sample_result = await student_sampler.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_ids),
        num_samples=args.group_size,
        sampling_params=sampling_params,
        return_text=False,
    )
    # 逐行讲解：通过推导式逐项生成结果：[sequence for sequence in sample_result.sequences if sequence.tokens]，赋给 sequences。
    sequences = [sequence for sequence in sample_result.sequences if sequence.tokens]
    # 逐行讲解：等待异步操作 asyncio.gather(*(completion_teacher_logprobs_async(teacher_client=teacher_client, prompt_ids=prompt_ids, co… 完成并取得结果，赋给 teacher_logprobs_list。
    teacher_logprobs_list = await asyncio.gather(
        *(
            completion_teacher_logprobs_async(
                teacher_client=teacher_client,
                prompt_ids=prompt_ids,
                completion_ids=list(sequence.tokens),
            )
            for sequence in sequences
        )
    )

    # 逐行讲解：声明 datums 的类型为 list[trio.Datum]；构造列表 []。
    datums: list[trio.Datum] = []
    # 逐行讲解：声明 reverse_kls 的类型为 list[float]；构造列表 []。
    reverse_kls: list[float] = []
    # 逐行讲解：声明 completion_token_counts 的类型为 list[int]；构造列表 []。
    completion_token_counts: list[int] = []
    # 逐行讲解：遍历 zip(sequences, teacher_logprobs_list, strict=True)，将每项绑定到 (sequence, teacher_logprobs)。
    for sequence, teacher_logprobs in zip(
        sequences,
        teacher_logprobs_list,
        strict=True,
    ):
        # 逐行讲解：调用 list（list(sequence.tokens)），赋给 completion_ids。
        completion_ids = list(sequence.tokens)
        # 逐行讲解：通过推导式逐项生成结果：[float(value) for value in sequence.logprobs]，赋给 student_logprobs。
        student_logprobs = [float(value) for value in sequence.logprobs]
        # 逐行讲解：检查条件 len(student_logprobs) != len(completion_ids)；成立时执行下方分支。
        if len(student_logprobs) != len(completion_ids):
            # 逐行讲解：抛出异常：ValueError('Student token/logprob length mismatch')。
            raise ValueError("Student token/logprob length mismatch")
        # 逐行讲解：计算 np.asarray(student_logprobs) - np.asarray(teacher_logprobs)，赋给 reverse_kl。
        reverse_kl = np.asarray(student_logprobs) - np.asarray(teacher_logprobs)
        # 逐行讲解：计算 -args.kl_penalty_coef * reverse_kl，赋给 advantages。
        advantages = -args.kl_penalty_coef * reverse_kl
        # 逐行讲解：向列表末尾追加一个元素（datums.append(build_opd_datum(prompt_ids=prompt_ids, completion_ids=completion_ids, old_logprobs=student_lo…）。
        datums.append(
            build_opd_datum(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                old_logprobs=student_logprobs,
                advantages=advantages,
            )
        )
        # 逐行讲解：把一组元素依次加入列表（reverse_kls.extend(reverse_kl.tolist())）。
        reverse_kls.extend(reverse_kl.tolist())
        # 逐行讲解：向列表末尾追加一个元素（completion_token_counts.append(len(completion_ids))）。
        completion_token_counts.append(len(completion_ids))

    # 逐行讲解：返回结果：PromptRollout(datums=datums, reverse_kls=reverse_kls, completion_token_counts=completion_token_counts)。
    return PromptRollout(
        datums=datums,
        reverse_kls=reverse_kls,
        completion_token_counts=completion_token_counts,
    )


# 逐行讲解：定义函数 batch_for_step：封装下方逻辑，供调用方复用。
def batch_for_step(
    rows: list[dict[str, str]],
    step: int,
    batch_size: int,
    full_dataset_run: bool,
) -> list[dict[str, str]]:
    # 逐行讲解：检查条件 full_dataset_run；成立时执行下方分支。
    if full_dataset_run:
        # 逐行讲解：计算 step * batch_size，赋给 start。
        start = step * batch_size
        # 逐行讲解：返回结果：rows[start:start + batch_size]。
        return rows[start : start + batch_size]
    # 逐行讲解：返回结果：[rows[(step * batch_size + offset) % len(rows)] for offset in range(batch_size)]。
    return [rows[(step * batch_size + offset) % len(rows)] for offset in range(batch_size)]


# 逐行讲解：定义函数 pick_loss_metric：封装下方逻辑，供调用方复用。
def pick_loss_metric(metrics: dict[str, float]) -> float | None:
    # 逐行讲解：遍历 ('trainer/loss', 'trainer/loss_mean', 'trainer/loss/mean', 'trainer/policy_loss', 'trainer/ppo_loss')，将每项绑定到 key。
    for key in (
        "trainer/loss",
        "trainer/loss_mean",
        "trainer/loss/mean",
        "trainer/policy_loss",
        "trainer/ppo_loss",
    ):
        # 逐行讲解：检查条件 key in metrics；成立时执行下方分支。
        if key in metrics:
            # 逐行讲解：返回结果：float(metrics[key])。
            return float(metrics[key])
    # 逐行讲解：遍历 metrics.items()，将每项绑定到 (key, value)。
    for key, value in metrics.items():
        # 逐行讲解：检查条件 'loss' in key.lower()；成立时执行下方分支。
        if "loss" in key.lower():
            # 逐行讲解：返回结果：float(value)。
            return float(value)
    # 逐行讲解：返回结果：None。
    return None


# 逐行讲解：定义函数 start_swanlab：封装下方逻辑，供调用方复用。
def start_swanlab(
    args: argparse.Namespace,
    dataset_size: int,
    total_steps: int,
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
    # 逐行讲解：返回结果：swanlab.init(project=args.swanlab_project, name=args.run_name, workspace=args.swanlab_workspace, mode=args.…。
    return swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=["PyTrio", "OPD", "medical", "SFT-teacher", "async"],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


# 逐行讲解：定义异步函数 save_checkpoint_async：封装下方逻辑，供调用方复用。
async def save_checkpoint_async(
    training_client: Any,
    swanlab_run: Any | None,
    args: argparse.Namespace,
    completed_steps: int,
) -> tuple[str, str]:
    # 逐行讲解：取值 f'step{completed_steps:06d}'，赋给 tag。
    tag = f"step{completed_steps:06d}"
    # 逐行讲解：取值 f'{args.run_name}-{tag}'，赋给 state_name。
    state_name = f"{args.run_name}-{tag}"
    # 逐行讲解：取值 f'{args.run_name}-{tag}'，赋给 weights_name。
    weights_name = f"{args.run_name}-{tag}"
    # 逐行讲解：等待异步操作 training_client.save_state_async(name=state_name) 完成并取得结果，赋给 state_future。
    state_future = await training_client.save_state_async(name=state_name)
    # 逐行讲解：等待异步操作 state_future 完成并取得结果，赋给 state_result。
    state_result = await state_future
    # 逐行讲解：等待异步操作 training_client.save_weights_for_sampler_async(name=weights_name) 完成并取得结果，赋给 weights_future。
    weights_future = await training_client.save_weights_for_sampler_async(
        name=weights_name
    )
    # 逐行讲解：等待异步操作 weights_future 完成并取得结果，赋给 weights_result。
    weights_result = await weights_future
    # 逐行讲解：调用 print（print(f'Saved checkpoint [{tag}] state: {state_result.path}')）。
    print(f"Saved checkpoint [{tag}] state: {state_result.path}")
    # 逐行讲解：调用 print（print(f'Saved checkpoint [{tag}] weights: {weights_result.path}')）。
    print(f"Saved checkpoint [{tag}] weights: {weights_result.path}")
    # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
    if swanlab_run is not None:
        # 逐行讲解：计算自然对数（swanlab.log({f'save/{tag}_state_path': swanlab.Text(state_result.path), f'save/{tag}_weights_path': swanlab…）。
        swanlab.log(
            {
                f"save/{tag}_state_path": swanlab.Text(state_result.path),
                f"save/{tag}_weights_path": swanlab.Text(weights_result.path),
            },
            step=completed_steps,
        )
    # 逐行讲解：返回结果：(state_result.path, weights_result.path)。
    return state_result.path, weights_result.path


# 逐行讲解：定义异步函数 train：封装下方逻辑，供调用方复用。
async def train(args: argparse.Namespace) -> None:
    # 逐行讲解：设置对应随机数生成器的种子（random.seed(args.seed)）。
    random.seed(args.seed)
    # 逐行讲解：设置对应随机数生成器的种子（np.random.seed(args.seed)）。
    np.random.seed(args.seed)
    # 逐行讲解：调用 load_medical_prompts（load_medical_prompts(args)），赋给 rows。
    rows = load_medical_prompts(args)
    # 逐行讲解：取值 args.steps == 0，赋给 full_dataset_run。
    full_dataset_run = args.steps == 0
    # 逐行讲解：按条件选择结果：(len(rows) + args.batch_size - 1) // args.batch_size if full_dataset_run else args.steps，赋给 total_steps。
    total_steps = (
        (len(rows) + args.batch_size - 1) // args.batch_size
        if full_dataset_run
        else args.steps
    )
    # 逐行讲解：调用 print（print(f'Loaded medical-o1 prompts: {len(rows)}')）。
    print(f"Loaded medical-o1 prompts: {len(rows)}")
    # 逐行讲解：调用 print（print(f'Training steps: {total_steps}')）。
    print(f"Training steps: {total_steps}")
    # 逐行讲解：调用 print（print(f'Run name: {args.run_name}')）。
    print(f"Run name: {args.run_name}")
    # 逐行讲解：调用 print（print('C-Eval is evaluation-only and is not loaded by this training script.')）。
    print("C-Eval is evaluation-only and is not loaded by this training script.")

    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：取值 None，赋给 swanlab_run。
    swanlab_run = None
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：调用 print（print(f'Creating fresh LoRA student from base: {args.base_model}')）。
        print(f"Creating fresh LoRA student from base: {args.base_model}")
        # 逐行讲解：等待异步操作 service_client.create_lora_training_client_async(base_model=args.base_model, rank=args.lora_rank, seed=args… 完成并取得结果，赋给 training_client。
        training_client = await service_client.create_lora_training_client_async(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
        )
        # 逐行讲解：等待异步操作 service_client.create_sampling_client_async(base_model=args.teacher_base_model, model_path=args.teacher_mod… 完成并取得结果，赋给 teacher_client。
        teacher_client = await service_client.create_sampling_client_async(
            base_model=args.teacher_base_model,
            model_path=args.teacher_model_path,
        )
        # SamplingClient 按完整模型名加载 tokenizer，可避免 TrainingClient 在部分
        # SDK 版本里只识别内部模型别名的问题。
        tokenizer = teacher_client.get_tokenizer()
        # 逐行讲解：调用 trio.SamplingParams（trio.SamplingParams(max_tokens=args.max_tokens, seed=args.seed, temperature=args.temperature, top_p=args.to…），赋给 sampling_params。
        sampling_params = trio.SamplingParams(
            max_tokens=args.max_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop=list(
                dict.fromkeys(
                    token
                    for token in (tokenizer.eos_token, "<|im_end|>")
                    if token
                )
            ),
        )
        # 逐行讲解：调用 trio.AdamParams（trio.AdamParams(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2)），赋给 adam。
        adam = trio.AdamParams(
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
        )
        # 逐行讲解：调用 start_swanlab（start_swanlab(args, len(rows), total_steps)），赋给 swanlab_run。
        swanlab_run = start_swanlab(args, len(rows), total_steps)
        # 逐行讲解：取值 None，赋给 student_sampler。
        student_sampler = None

        # 逐行讲解：遍历 range(total_steps)，将每项绑定到 step。
        for step in range(total_steps):
            # 逐行讲解：调用 time.time（time.time()），赋给 step_start。
            step_start = time.time()
            # 逐行讲解：检查条件 student_sampler is None or step % args.sampler_refresh_steps == 0；成立时执行下方分支。
            if student_sampler is None or step % args.sampler_refresh_steps == 0:
                # 逐行讲解：等待异步操作 training_client.save_weights_and_get_sampling_client_async() 完成并取得结果，赋给 student_sampler。
                student_sampler = (
                    await training_client.save_weights_and_get_sampling_client_async()
                )

            # 逐行讲解：调用 batch_for_step（batch_for_step(rows, step, args.batch_size, full_dataset_run)），赋给 batch。
            batch = batch_for_step(
                rows,
                step,
                args.batch_size,
                full_dataset_run,
            )
            # 逐行讲解：进入上下文管理器 tqdm(total=len(batch), desc=f'Medical OPD async step {step}', unit='prompt')，退出时执行其收尾逻辑。
            with tqdm(
                total=len(batch),
                desc=f"Medical OPD async step {step}",
                unit="prompt",
            ) as progress_bar:

                # 逐行讲解：定义异步函数 run_and_track：封装下方逻辑，供调用方复用。
                async def run_and_track(row: dict[str, str]) -> PromptRollout:
                    # 逐行讲解：等待异步操作 run_prompt_rollout_async(student_sampler=student_sampler, teacher_client=teacher_client, tokenizer=tokenize… 完成并取得结果，赋给 rollout。
                    rollout = await run_prompt_rollout_async(
                        student_sampler=student_sampler,
                        teacher_client=teacher_client,
                        tokenizer=tokenizer,
                        row=row,
                        args=args,
                        sampling_params=sampling_params,
                    )
                    # 逐行讲解：调用 progress_bar.update（progress_bar.update(1)）。
                    progress_bar.update(1)
                    # 逐行讲解：返回结果：rollout。
                    return rollout

                # 逐行讲解：等待异步操作 asyncio.gather(*(run_and_track(row) for row in batch)) 完成并取得结果，赋给 rollouts。
                rollouts = await asyncio.gather(
                    *(run_and_track(row) for row in batch)
                )

            # 逐行讲解：通过推导式逐项生成结果：[datum for rollout in rollouts for datum in rollout.datums]，赋给 datums。
            datums = [datum for rollout in rollouts for datum in rollout.datums]
            # 逐行讲解：通过推导式逐项生成结果：[value for rollout in rollouts for value in rollout.reverse_kls]，赋给 reverse_kls。
            reverse_kls = [value for rollout in rollouts for value in rollout.reverse_kls]
            # 逐行讲解：通过推导式逐项生成结果：[value for rollout in rollouts for value in rollout.completion_token_counts]，赋给 completion_counts。
            completion_counts = [
                value
                for rollout in rollouts
                for value in rollout.completion_token_counts
            ]
            # 逐行讲解：检查条件 not datums；成立时执行下方分支。
            if not datums:
                # 逐行讲解：抛出异常：RuntimeError('No OPD datums were built')。
                raise RuntimeError("No OPD datums were built")

            # 逐行讲解：等待异步操作 training_client.forward_backward_async(datums, loss_fn=args.loss_fn) 完成并取得结果，赋给 fwd_bwd_future。
            fwd_bwd_future = await training_client.forward_backward_async(
                datums,
                loss_fn=args.loss_fn,
            )
            # 逐行讲解：等待异步操作 training_client.optim_step_async(adam) 完成并取得结果，赋给 optim_future。
            optim_future = await training_client.optim_step_async(adam)
            # 逐行讲解：等待异步操作 fwd_bwd_future 完成并取得结果，赋给 fwd_bwd_result。
            fwd_bwd_result = await fwd_bwd_future
            # 逐行讲解：等待异步操作 optim_future 完成并取得结果。
            await optim_future

            # 逐行讲解：计算 step + 1，赋给 completed_steps。
            completed_steps = step + 1
            # 逐行讲解：计算 time.time() - step_start，赋给 elapsed。
            elapsed = time.time() - step_start
            # 逐行讲解：调用 int（int(sum(completion_counts))），赋给 completion_tokens_total。
            completion_tokens_total = int(sum(completion_counts))
            # 逐行讲解：声明 metrics 的类型为 dict[str, float | int]；构造字典 {'data/prompts': len(batch), 'data/datums': len(datums), 'data/completion_tokens_mean': float(np.mean(compl…。
            metrics: dict[str, float | int] = {
                "data/prompts": len(batch),
                "data/datums": len(datums),
                "data/completion_tokens_mean": float(np.mean(completion_counts)),
                "data/completion_tokens_total": completion_tokens_total,
                "opd/reverse_kl_mean": float(np.mean(reverse_kls)),
                "opd/reverse_kl_std": float(np.std(reverse_kls)),
                "train/step": completed_steps,
                "train/learning_rate": args.learning_rate,
                "time/step_elapsed_time": elapsed,
                "step/completion_tokens_per_second": completion_tokens_total / elapsed,
            }
            # 逐行讲解：调用 metrics.update（metrics.update({f'trainer/{key}': float(value) for key, value in dict(fwd_bwd_result.metrics).items()})）。
            metrics.update(
                {
                    f"trainer/{key}": float(value)
                    for key, value in dict(fwd_bwd_result.metrics).items()
                }
            )
            # 逐行讲解：调用 pick_loss_metric（pick_loss_metric({key: float(value) for key, value in metrics.items()})），赋给 loss_value。
            loss_value = pick_loss_metric(
                {key: float(value) for key, value in metrics.items()}
            )
            # 逐行讲解：检查条件 loss_value is not None；成立时执行下方分支。
            if loss_value is not None:
                # 逐行讲解：取值 loss_value，赋给 metrics['train/loss']。
                metrics["train/loss"] = loss_value
            # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
            if swanlab_run is not None:
                # 逐行讲解：计算自然对数（swanlab.log(metrics, step=completed_steps)）。
                swanlab.log(metrics, step=completed_steps)

            # 逐行讲解：按条件选择结果：'n/a' if loss_value is None else f'{loss_value:.4f}'，赋给 loss_text。
            loss_text = "n/a" if loss_value is None else f"{loss_value:.4f}"
            # 逐行讲解：调用 tqdm.write（tqdm.write(f"step {completed_steps:03d}/{total_steps} | datums {len(datums)} | completion tokens mean {metr…）。
            tqdm.write(
                f"step {completed_steps:03d}/{total_steps} | datums {len(datums)} | "
                f"completion tokens mean {metrics['data/completion_tokens_mean']:.1f} | "
                f"reverse_kl {metrics['opd/reverse_kl_mean']:.4f} | "
                f"tokens/s {metrics['step/completion_tokens_per_second']:.1f} | "
                f"loss {loss_text} | time {elapsed:.2f}s"
            )

            # 逐行讲解：取值 completed_steps == total_steps or (args.save_every_steps > 0 and completed_steps % args.save_every_steps ==…，赋给 should_save。
            should_save = completed_steps == total_steps or (
                args.save_every_steps > 0
                and completed_steps % args.save_every_steps == 0
            )
            # 逐行讲解：检查条件 should_save；成立时执行下方分支。
            if should_save:
                # 逐行讲解：等待异步操作 save_checkpoint_async(training_client=training_client, swanlab_run=swanlab_run, args=args, completed_steps=… 完成并取得结果。
                await save_checkpoint_async(
                    training_client=training_client,
                    swanlab_run=swanlab_run,
                    args=args,
                    completed_steps=completed_steps,
                )
    finally:
        # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
        if swanlab_run is not None:
            # 逐行讲解：调用 swanlab.finish（swanlab.finish()）。
            swanlab.finish()


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：调用 time.time（time.time()），赋给 started_at。
    started_at = time.time()
    # 逐行讲解：调用 asyncio.run（asyncio.run(train(parse_args()))）。
    asyncio.run(train(parse_args()))
    # 逐行讲解：调用 print（print('#' * 50)）。
    print("#" * 50)
    # 逐行讲解：调用 print（print('# all done')）。
    print("# all done")
    # 逐行讲解：调用 print（print(f'# train cost {time.time() - started_at:.2f}s')）。
    print(f"# train cost {time.time() - started_at:.2f}s")
    # 逐行讲解：调用 print（print('#' * 50)）。
    print("#" * 50)


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
