"""异步版第三阶段 C-Eval OPD：4B base teacher -> Medical OPD student。

本阶段先把第二阶段 Medical OPD 的 Train state 加载到临时 TrainingClient，
保存一份 source-copy Train state。随后从 source-copy state 创建正式 TrainingClient，
并在任何 OPD update 之前，于正式 run 内保存 step000000 state + sampler weights。
训练时使用原始 Qwen3.5-4B base 作为通用能力锚点，在 C-Eval non-med
train pool 上让 student 自主采样，base teacher 对同一条 student completion
计算逐 token logprob，并通过 reverse KL advantage 更新 student。

训练只使用 C-Eval 的题目和选项，不使用 answer_idx。MedQA-ZH 和 C-Eval test
都不进入训练，只用于评测各 step checkpoint：
- MedQA-ZH @ max_tokens=1024 检查医疗能力是否保持；
- C-Eval non-med @ max_tokens=8192 检查通用能力是否恢复。

小成本试跑：
uv run python 04-ceval-opd-async.py \
    --student-state-path YOUR_MEDICAL_OPD_TRAIN_STATE_PATH \
    --steps 10 \
    --batch-size 2 \
    --group-size 2 \
    --sample-size 100 \
    --max-tokens 2048 \
    --learning-rate 5e-6 \
    --save-every-steps 5 \
    --swanlab-mode disabled

第三阶段正式试验：
uv run python 04-ceval-opd-async.py \
    --student-state-path YOUR_MEDICAL_OPD_TRAIN_STATE_PATH \
    --steps 300 \
    --batch-size 4 \
    --group-size 4 \
    --sample-size 0 \
    --max-tokens 2048 \
    --learning-rate 5e-6 \
    --save-every-steps 50 \
    --swanlab-mode online

注意：
- `--student-state-path` 必须是 `save_state` 生成的 Train state，不能传 sampler weights。
- 原始 Medical OPD state 只用于创建临时 client，不会直接进入第三阶段训练循环。
- 临时 client 只保存 source-copy state，用它创建正式 TrainingClient。
- 正式 client 会在训练前保存本 run 的 step000000 state + sampler weights。
- 两次加载都只加载模型权重，不继承第二阶段优化器状态。
- `--max-tokens 2048` 是训练 rollout 的 completion 预算，不是 C-Eval 评测预算。
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
# 逐行讲解：计算 SCRIPT_DIR / 'datasets' / 'ceval-non-med' / 'opd_train.jsonl'，赋给 DEFAULT_DATASET_PATH。
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "ceval-non-med" / "opd_train.jsonl"
# 逐行讲解：构造集合 {'test_sample.jsonl', 'test_pool.jsonl', 'all.jsonl'}，赋给 FORBIDDEN_EVAL_FILENAMES。
FORBIDDEN_EVAL_FILENAMES = {"test_sample.jsonl", "test_pool.jsonl", "all.jsonl"}
# 逐行讲解：取值 '你是中文单项选择题作答助手。请在内部完成必要推理，但最终回答只能包含 A、B、C、D 中的一个大写字母，不要输出推理过程、解释、标点或其他文字。'，赋给 DEFAULT_SYSTEM_MESSAGE。
DEFAULT_SYSTEM_MESSAGE = (
    "你是中文单项选择题作答助手。请在内部完成必要推理，"
    "但最终回答只能包含 A、B、C、D 中的一个大写字母，"
    "不要输出推理过程、解释、标点或其他文字。"
)
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
    # 逐行讲解：定义函数 format_usage：把 argparse 的 usage 标题替换成中文。
    def format_usage(self) -> str:
        """把 argparse 的 usage 标题替换成中文。"""
        # 逐行讲解：返回结果：super().format_usage().replace('usage:', '用法:', 1)。
        return super().format_usage().replace("usage:", "用法:", 1)

    # 逐行讲解：定义函数 format_help：把 argparse 帮助信息中的 usage 标题替换成中文。
    def format_help(self) -> str:
        """把 argparse 帮助信息中的 usage 标题替换成中文。"""
        # 逐行讲解：返回结果：super().format_help().replace('usage:', '用法:', 1)。
        return super().format_help().replace("usage:", "用法:", 1)


# 逐行讲解：定义函数 model_slug：把模型名转换成适合放进运行名的短字符串。
def model_slug(base_model: str) -> str:
    """把模型名转换成适合放进运行名的短字符串。"""
    # 逐行讲解：调用 base_model.rsplit('/', 1)[-1].lower().replace（base_model.rsplit('/', 1)[-1].lower().replace('qwen3.5', 'qwen35')），赋给 name。
    name = base_model.rsplit("/", 1)[-1].lower().replace("qwen3.5", "qwen35")
    # 逐行讲解：返回结果：re.sub('[^a-z0-9]+', '-', name).strip('-')。
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


# 逐行讲解：定义函数 build_run_name：根据 teacher、loss 和训练步数自动生成运行名。
def build_run_name(args: argparse.Namespace) -> str:
    """根据 teacher、loss 和训练步数自动生成运行名。"""
    # 逐行讲解：调用 args.loss_fn.replace（args.loss_fn.replace('_', '-')），赋给 loss_slug。
    loss_slug = args.loss_fn.replace("_", "-")
    # 逐行讲解：按条件选择结果：'full' if args.steps == 0 else f'steps{args.steps}'，赋给 steps_slug。
    steps_slug = "full" if args.steps == 0 else f"steps{args.steps}"
    # 逐行讲解：返回结果：f'opd-ceval-async-{model_slug(args.teacher_base_model)}-base-anchor-{loss_slug}-{steps_slug}'。
    return (
        f"opd-ceval-async-{model_slug(args.teacher_base_model)}-"
        f"base-anchor-{loss_slug}-{steps_slug}"
    )


# 逐行讲解：定义函数 parse_args：解析并校验第三阶段 C-Eval OPD 的命令行参数。
def parse_args() -> argparse.Namespace:
    """解析并校验第三阶段 C-Eval OPD 的命令行参数。"""
    # 逐行讲解：调用 ChineseArgumentParser（ChineseArgumentParser(description='PyTRIO 异步版第三阶段 C-Eval OPD：base teacher -> Medical OPD student', add_help…），赋给 parser。
    parser = ChineseArgumentParser(
        description=(
            "PyTRIO 异步版第三阶段 C-Eval OPD：base teacher -> Medical OPD student"
        ),
        add_help=False,
    )
    # 逐行讲解：取值 '选项'，赋给 parser._optionals.title。
    parser._optionals.title = "选项"
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('-h', '--help', action='help', help='显示帮助信息并退出')）。
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-path', type=Path, default=DEFAULT_DATASET_PATH, help='C-Eval non-med OPD tra…）。
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="C-Eval non-med OPD train JSONL；不能使用 test 文件",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sample-size', type=int, default=0, help='随机抽样 prompt 数；0 表示全用')）。
    parser.add_argument("--sample-size", type=int, default=0, help="随机抽样 prompt 数；0 表示全用")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42, help='随机种子')）。
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--student-state-path', required=True, help='Medical OPD 的 trio:// Train state 路径；不能传 s…）。
    parser.add_argument(
        "--student-state-path",
        required=True,
        help="Medical OPD 的 trio:// Train state 路径；不能传 sampler_weights",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--teacher-base-model', default='Qwen/Qwen3.5-4B', help='作为通用能力锚点的原始 base model')）。
    parser.add_argument(
        "--teacher-base-model",
        default="Qwen/Qwen3.5-4B",
        help="作为通用能力锚点的原始 base model",
    )

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--steps', type=int, default=300, help='训练 step 数；0 表示完整遍历当前数据一次')）。
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="训练 step 数；0 表示完整遍历当前数据一次",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=4, help='每个 step 使用的 C-Eval prompt 数')）。
    parser.add_argument("--batch-size", type=int, default=4, help="每个 step 使用的 C-Eval prompt 数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--group-size', type=int, default=4, help='每个 prompt 的 student completion 数')）。
    parser.add_argument("--group-size", type=int, default=4, help="每个 prompt 的 student completion 数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=2048, help='每条训练 completion 最多生成 token 数')）。
    parser.add_argument("--max-tokens", type=int, default=2048, help="每条训练 completion 最多生成 token 数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0, help='student rollout temperature')）。
    parser.add_argument("--temperature", type=float, default=1.0, help="student rollout temperature")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0, help='student rollout top_p')）。
    parser.add_argument("--top-p", type=float, default=1.0, help="student rollout top_p")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-k', type=int, default=-1, help='student rollout top_k；-1 表示不限制')）。
    parser.add_argument("--top-k", type=int, default=-1, help="student rollout top_k；-1 表示不限制")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--system-message', default=DEFAULT_SYSTEM_MESSAGE, help='与 C-Eval 评测一致的 system message…）。
    parser.add_argument(
        "--system-message",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="与 C-Eval 评测一致的 system message；空字符串表示不加",
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
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=5e-06, help='第三阶段 Adam 学习率')）。
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="第三阶段 Adam 学习率")
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
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every-steps', type=int, default=50, help='每多少 step 保存 state + sampler weights；0…）。
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=50,
        help="每多少 step 保存 state + sampler weights；0 表示只保存最终 checkpoint",
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
    # 逐行讲解：检查条件 args.kl_penalty_coef <= 0；成立时执行下方分支。
    if args.kl_penalty_coef <= 0:
        # 逐行讲解：抛出异常：ValueError('--kl-penalty-coef must be > 0')。
        raise ValueError("--kl-penalty-coef must be > 0")
    # 逐行讲解：检查条件 args.learning_rate <= 0；成立时执行下方分支。
    if args.learning_rate <= 0:
        # 逐行讲解：抛出异常：ValueError('--learning-rate must be > 0')。
        raise ValueError("--learning-rate must be > 0")
    # 逐行讲解：检查条件 args.temperature < 0；成立时执行下方分支。
    if args.temperature < 0:
        # 逐行讲解：抛出异常：ValueError('--temperature must be >= 0')。
        raise ValueError("--temperature must be >= 0")
    # 逐行讲解：检查条件 not 0 < args.top_p <= 1；成立时执行下方分支。
    if not 0 < args.top_p <= 1:
        # 逐行讲解：抛出异常：ValueError('--top-p must be in (0, 1]')。
        raise ValueError("--top-p must be in (0, 1]")
    # 逐行讲解：检查条件 args.save_every_steps < 0；成立时执行下方分支。
    if args.save_every_steps < 0:
        # 逐行讲解：抛出异常：ValueError('--save-every-steps must be >= 0')。
        raise ValueError("--save-every-steps must be >= 0")
    # 逐行讲解：检查条件 '/sampler_weights/' in args.student_state_path；成立时执行下方分支。
    if "/sampler_weights/" in args.student_state_path:
        # 逐行讲解：抛出异常：ValueError('--student-state-path 必须传 save_state 生成的 Train state，不能传 sampler_weights')。
        raise ValueError(
            "--student-state-path 必须传 save_state 生成的 Train state，不能传 sampler_weights"
        )

    # 逐行讲解：调用 build_run_name（build_run_name(args)），赋给 args.run_name。
    args.run_name = build_run_name(args)
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 build_ceval_prompt：把一条 C-Eval 题目和四个选项整理成统一训练 prompt。
def build_ceval_prompt(row: dict[str, Any], line_number: int) -> str:
    """把一条 C-Eval 题目和四个选项整理成统一训练 prompt。"""
    # 逐行讲解：调用 str(row.get('question', '')).strip（str(row.get('question', '')).strip()），赋给 question。
    question = str(row.get("question", "")).strip()
    # 逐行讲解：查询指定键，缺失时使用默认值（row.get('options')），赋给 options。
    options = row.get("options")
    # 逐行讲解：检查条件 not isinstance(options, dict)；成立时执行下方分支。
    if not isinstance(options, dict):
        # 逐行讲解：抛出异常：ValueError(f'C-Eval 第 {line_number} 行的 options 不是对象')。
        raise ValueError(f"C-Eval 第 {line_number} 行的 options 不是对象")
    # 逐行讲解：通过推导式逐项生成结果：{key: str(options.get(key, '')).strip() for key in ('A', 'B', 'C', 'D')}，赋给 normalized_options。
    normalized_options = {
        key: str(options.get(key, "")).strip() for key in ("A", "B", "C", "D")
    }
    # 逐行讲解：检查条件 not question or not all(normalized_options.values())；成立时执行下方分支。
    if not question or not all(normalized_options.values()):
        # 逐行讲解：抛出异常：ValueError(f'C-Eval 第 {line_number} 行缺少 question 或 A/B/C/D 选项')。
        raise ValueError(f"C-Eval 第 {line_number} 行缺少 question 或 A/B/C/D 选项")
    # 逐行讲解：返回结果：'\n'.join(['以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。', '', f'题目：{question}', f"A. {normalized_options['A']}", f"B…。
    return "\n".join(
        [
            "以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。",
            "",
            f"题目：{question}",
            f"A. {normalized_options['A']}",
            f"B. {normalized_options['B']}",
            f"C. {normalized_options['C']}",
            f"D. {normalized_options['D']}",
            "",
            "答案：",
        ]
    )


# 逐行讲解：定义函数 load_ceval_prompts：加载、校验并打乱 C-Eval OPD 训练 prompt。
def load_ceval_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    """加载、校验并打乱 C-Eval OPD 训练 prompt。"""
    # 逐行讲解：检查条件 not args.dataset_path.exists()；成立时执行下方分支。
    if not args.dataset_path.exists():
        # 逐行讲解：抛出异常：FileNotFoundError(f'找不到 C-Eval OPD 数据：{args.dataset_path}，请先运行 00-download-dataset.py')。
        raise FileNotFoundError(
            f"找不到 C-Eval OPD 数据：{args.dataset_path}，请先运行 00-download-dataset.py"
        )
    # 逐行讲解：检查条件 args.dataset_path.parent.name == 'ceval-non-med' and args.dataset_path.name in FORBIDDEN_EVAL_FILENAMES；成立时执行下方分支。
    if (
        args.dataset_path.parent.name == "ceval-non-med"
        and args.dataset_path.name in FORBIDDEN_EVAL_FILENAMES
    ):
        # 逐行讲解：抛出异常：ValueError(f'禁止使用 C-Eval 测试数据训练：{args.dataset_path.name}；请使用 opd_train.jsonl')。
        raise ValueError(
            f"禁止使用 C-Eval 测试数据训练：{args.dataset_path.name}；请使用 opd_train.jsonl"
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
            # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(line)），赋给 raw_row。
            raw_row = json.loads(line)
            # 逐行讲解：向列表末尾追加一个元素（rows.append({'row_id': str(raw_row.get('row_id', line_number)), 'subject': str(raw_row.get('subject', 'unkn…）。
            rows.append(
                {
                    "row_id": str(raw_row.get("row_id", line_number)),
                    "subject": str(raw_row.get("subject", "unknown")),
                    "prompt": build_ceval_prompt(raw_row, line_number),
                }
            )

    # 逐行讲解：检查条件 not rows；成立时执行下方分支。
    if not rows:
        # 逐行讲解：抛出异常：ValueError(f'C-Eval OPD 数据为空：{args.dataset_path}')。
        raise ValueError(f"C-Eval OPD 数据为空：{args.dataset_path}")
    # 逐行讲解：打乱数据顺序（random.Random(args.seed).shuffle(rows)）。
    random.Random(args.seed).shuffle(rows)
    # 逐行讲解：检查条件 args.sample_size > 0；成立时执行下方分支。
    if args.sample_size > 0:
        # 逐行讲解：按索引或切片读取 rows[:min(args.sample_size, len(rows))]，赋给 rows。
        rows = rows[: min(args.sample_size, len(rows))]
    # 逐行讲解：返回结果：rows。
    return rows


# 逐行讲解：定义函数 build_prompt_tokens：应用 chat template，把 system + C-Eval prompt 转成 token。
def build_prompt_tokens(
    tokenizer: Any,
    prompt: str,
    system_message: str,
    enable_thinking: bool,
) -> list[int]:
    """应用 chat template，把 system + C-Eval prompt 转成 token。"""
    # 逐行讲解：构造列表 []，赋给 messages。
    messages = []
    # 逐行讲解：检查条件 system_message.strip()；成立时执行下方分支。
    if system_message.strip():
        # 逐行讲解：向列表末尾追加一个元素（messages.append({'role': 'system', 'content': system_message.strip()})）。
        messages.append({"role": "system", "content": system_message.strip()})
    # 逐行讲解：向列表末尾追加一个元素（messages.append({'role': 'user', 'content': prompt})）。
    messages.append({"role": "user", "content": prompt})
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_…），赋给 prompt_text。
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    # 逐行讲解：将文本编码为 token（tokenizer.encode(prompt_text, add_special_tokens=False)），赋给 prompt_ids。
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    # 逐行讲解：检查条件 not prompt_ids；成立时执行下方分支。
    if not prompt_ids:
        # 逐行讲解：抛出异常：ValueError('Prompt tokens are empty')。
        raise ValueError("Prompt tokens are empty")
    # 逐行讲解：返回结果：prompt_ids。
    return prompt_ids


# 逐行讲解：定义异步函数 completion_teacher_logprobs_async：计算 base teacher 对 student completion 的逐 token logprob。
async def completion_teacher_logprobs_async(
    teacher_client: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> list[float]:
    """计算 base teacher 对 student completion 的逐 token logprob。"""
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


# 逐行讲解：定义函数 build_opd_datum：把一条 student 轨迹及其 advantage 组装成 PyTRIO OPD Datum。
def build_opd_datum(
    prompt_ids: list[int],
    completion_ids: list[int],
    old_logprobs: list[float],
    advantages: list[float] | np.ndarray,
) -> trio.Datum:
    """把一条 student 轨迹及其 advantage 组装成 PyTRIO OPD Datum。"""
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


# 逐行讲解：定义异步函数 run_prompt_rollout_async：完成一个 prompt 的 student 采样、teacher 打分和 OPD Datum 构造。
async def run_prompt_rollout_async(
    student_sampler: Any,
    teacher_client: Any,
    tokenizer: Any,
    row: dict[str, str],
    args: argparse.Namespace,
    sampling_params: trio.SamplingParams,
) -> PromptRollout:
    """完成一个 prompt 的 student 采样、teacher 打分和 OPD Datum 构造。"""
    # 逐行讲解：调用 build_prompt_tokens（build_prompt_tokens(tokenizer=tokenizer, prompt=row['prompt'], system_message=args.system_message, enable_t…），赋给 prompt_ids。
    prompt_ids = build_prompt_tokens(
        tokenizer=tokenizer,
        prompt=row["prompt"],
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
    # 逐行讲解：检查条件 not sequences；成立时执行下方分支。
    if not sequences:
        # 逐行讲解：抛出异常：RuntimeError(f"Student 没有为 C-Eval prompt 生成有效 completion：{row['row_id']}")。
        raise RuntimeError(f"Student 没有为 C-Eval prompt 生成有效 completion：{row['row_id']}")

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


# 逐行讲解：定义函数 batch_for_step：按当前 step 选择训练 batch，完整遍历时不重复最后一批。
def batch_for_step(
    rows: list[dict[str, str]],
    step: int,
    batch_size: int,
    full_dataset_run: bool,
) -> list[dict[str, str]]:
    """按当前 step 选择训练 batch，完整遍历时不重复最后一批。"""
    # 逐行讲解：检查条件 full_dataset_run；成立时执行下方分支。
    if full_dataset_run:
        # 逐行讲解：计算 step * batch_size，赋给 start。
        start = step * batch_size
        # 逐行讲解：返回结果：rows[start:start + batch_size]。
        return rows[start : start + batch_size]
    # 逐行讲解：返回结果：[rows[(step * batch_size + offset) % len(rows)] for offset in range(batch_size)]。
    return [rows[(step * batch_size + offset) % len(rows)] for offset in range(batch_size)]


# 逐行讲解：定义函数 pick_loss_metric：从 PyTRIO 返回的不同指标命名中提取 loss。
def pick_loss_metric(metrics: dict[str, float]) -> float | None:
    """从 PyTRIO 返回的不同指标命名中提取 loss。"""
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


# 逐行讲解：定义函数 start_swanlab：按命令行配置初始化 SwanLab 运行记录。
def start_swanlab(
    args: argparse.Namespace,
    dataset_size: int,
    total_steps: int,
) -> Any | None:
    """按命令行配置初始化 SwanLab 运行记录。"""
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
    # 逐行讲解：取值 False，赋给 config['optimizer_state_restored']。
    config["optimizer_state_restored"] = False
    # 逐行讲解：返回结果：swanlab.init(project=args.swanlab_project, name=args.run_name, workspace=args.swanlab_workspace, mode=args.…。
    return swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=["PyTrio", "OPD", "C-Eval", "base-anchor", "async"],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


# 逐行讲解：定义异步函数 save_checkpoint_async：同时保存可续训 state 和可评测 sampler weights。
async def save_checkpoint_async(
    training_client: Any,
    swanlab_run: Any | None,
    args: argparse.Namespace,
    completed_steps: int,
) -> tuple[str, str]:
    """同时保存可续训 state 和可评测 sampler weights。"""
    # 逐行讲解：取值 f'step{completed_steps:06d}'，赋给 tag。
    tag = f"step{completed_steps:06d}"
    # 逐行讲解：取值 f'{args.run_name}-{tag}'，赋给 name。
    name = f"{args.run_name}-{tag}"
    # 逐行讲解：等待异步操作 training_client.save_state_async(name=name) 完成并取得结果，赋给 state_future。
    state_future = await training_client.save_state_async(name=name)
    # 逐行讲解：等待异步操作 state_future 完成并取得结果，赋给 state_result。
    state_result = await state_future
    # 逐行讲解：等待异步操作 training_client.save_weights_for_sampler_async(name=name) 完成并取得结果，赋给 weights_future。
    weights_future = await training_client.save_weights_for_sampler_async(name=name)
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


# 逐行讲解：定义异步函数 save_source_state_copy_async：把原始 Medical OPD 权重保存成供正式 TrainingClient 加载的 state 副本。
async def save_source_state_copy_async(
    training_client: Any,
    swanlab_run: Any | None,
    args: argparse.Namespace,
) -> str:
    """把原始 Medical OPD 权重保存成供正式 TrainingClient 加载的 state 副本。"""
    # 逐行讲解：取值 f'{args.run_name}-source-copy'，赋给 name。
    name = f"{args.run_name}-source-copy"
    # 逐行讲解：等待异步操作 training_client.save_state_async(name=name) 完成并取得结果，赋给 state_future。
    state_future = await training_client.save_state_async(name=name)
    # 逐行讲解：等待异步操作 state_future 完成并取得结果，赋给 state_result。
    state_result = await state_future
    # 逐行讲解：调用 print（print(f'Saved source-copy state: {state_result.path}')）。
    print(f"Saved source-copy state: {state_result.path}")
    # 逐行讲解：检查条件 swanlab_run is not None；成立时执行下方分支。
    if swanlab_run is not None:
        # 逐行讲解：计算自然对数（swanlab.log({'save/source_copy_state_path': swanlab.Text(state_result.path)}, step=0)）。
        swanlab.log(
            {"save/source_copy_state_path": swanlab.Text(state_result.path)},
            step=0,
        )
    # 逐行讲解：返回结果：state_result.path。
    return state_result.path


# 逐行讲解：定义异步函数 train：克隆 Medical OPD state，并执行第三阶段异步 C-Eval OPD 训练。
async def train(args: argparse.Namespace) -> None:
    """克隆 Medical OPD state，并执行第三阶段异步 C-Eval OPD 训练。"""
    # 逐行讲解：设置对应随机数生成器的种子（random.seed(args.seed)）。
    random.seed(args.seed)
    # 逐行讲解：设置对应随机数生成器的种子（np.random.seed(args.seed)）。
    np.random.seed(args.seed)
    # 逐行讲解：调用 load_ceval_prompts（load_ceval_prompts(args)），赋给 rows。
    rows = load_ceval_prompts(args)
    # 逐行讲解：取值 args.steps == 0，赋给 full_dataset_run。
    full_dataset_run = args.steps == 0
    # 逐行讲解：按条件选择结果：(len(rows) + args.batch_size - 1) // args.batch_size if full_dataset_run else args.steps，赋给 total_steps。
    total_steps = (
        (len(rows) + args.batch_size - 1) // args.batch_size
        if full_dataset_run
        else args.steps
    )
    # 逐行讲解：调用 print（print(f'Loaded C-Eval non-med OPD prompts: {len(rows)}')）。
    print(f"Loaded C-Eval non-med OPD prompts: {len(rows)}")
    # 逐行讲解：调用 print（print(f'Training steps: {total_steps}')）。
    print(f"Training steps: {total_steps}")
    # 逐行讲解：调用 print（print(f'Run name: {args.run_name}')）。
    print(f"Run name: {args.run_name}")
    # 逐行讲解：调用 print（print(f'Student Train state: {args.student_state_path}')）。
    print(f"Student Train state: {args.student_state_path}")
    # 逐行讲解：调用 print（print(f'Base-anchor teacher: {args.teacher_base_model}')）。
    print(f"Base-anchor teacher: {args.teacher_base_model}")
    # 逐行讲解：调用 print（print('The source state will be cloned before any C-Eval OPD update.')）。
    print("The source state will be cloned before any C-Eval OPD update.")
    # 逐行讲解：调用 print（print('answer_idx is ignored; C-Eval test data are not loaded by this training script.')）。
    print("answer_idx is ignored; C-Eval test data are not loaded by this training script.")

    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：取值 None，赋给 swanlab_run。
    swanlab_run = None
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：调用 start_swanlab（start_swanlab(args, len(rows), total_steps)），赋给 swanlab_run。
        swanlab_run = start_swanlab(args, len(rows), total_steps)

        # 逐行讲解：调用 print（print('Loading Medical OPD weights into a temporary TrainingClient...')）。
        print("Loading Medical OPD weights into a temporary TrainingClient...")
        # 逐行讲解：等待异步操作 service_client.create_training_client_from_state_async(path=args.student_state_path) 完成并取得结果，赋给 source_client。
        source_client = await service_client.create_training_client_from_state_async(
            path=args.student_state_path
        )
        # 逐行讲解：等待异步操作 save_source_state_copy_async(training_client=source_client, swanlab_run=swanlab_run, args=args) 完成并取得结果，赋给 source_copy_state_path。
        source_copy_state_path = await save_source_state_copy_async(
            training_client=source_client,
            swanlab_run=swanlab_run,
            args=args,
        )
        # 逐行讲解：取值 None，赋给 source_client。
        source_client = None

        # 逐行讲解：调用 print（print('Creating the stage-3 TrainingClient from the source-copy state...')）。
        print("Creating the stage-3 TrainingClient from the source-copy state...")
        # 逐行讲解：等待异步操作 service_client.create_training_client_from_state_async(path=source_copy_state_path) 完成并取得结果，赋给 training_client。
        training_client = await service_client.create_training_client_from_state_async(
            path=source_copy_state_path
        )
        # 逐行讲解：等待异步操作 save_checkpoint_async(training_client=training_client, swanlab_run=swanlab_run, args=args, completed_steps=0) 完成并取得结果，赋给 (initial_state_path, initial_weights_path)。
        initial_state_path, initial_weights_path = await save_checkpoint_async(
            training_client=training_client,
            swanlab_run=swanlab_run,
            args=args,
            completed_steps=0,
        )
        # 逐行讲解：调用 print（print(f'Saved formal-run step000000 state: {initial_state_path}')）。
        print(f"Saved formal-run step000000 state: {initial_state_path}")
        # 逐行讲解：调用 print（print(f'Saved formal-run step000000 sampler weights: {initial_weights_path}')）。
        print(f"Saved formal-run step000000 sampler weights: {initial_weights_path}")
        # 逐行讲解：等待异步操作 service_client.create_sampling_client_async(base_model=args.teacher_base_model) 完成并取得结果，赋给 teacher_client。
        teacher_client = await service_client.create_sampling_client_async(
            base_model=args.teacher_base_model
        )
        # 逐行讲解：调用 teacher_client.get_tokenizer（teacher_client.get_tokenizer()），赋给 tokenizer。
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

            # 逐行讲解：调用 batch_for_step（batch_for_step(rows=rows, step=step, batch_size=args.batch_size, full_dataset_run=full_dataset_run)），赋给 batch。
            batch = batch_for_step(
                rows=rows,
                step=step,
                batch_size=args.batch_size,
                full_dataset_run=full_dataset_run,
            )
            # 逐行讲解：进入上下文管理器 tqdm(total=len(batch), desc=f'C-Eval OPD async step {step + 1}', unit='prompt')，退出时执行其收尾逻辑。
            with tqdm(
                total=len(batch),
                desc=f"C-Eval OPD async step {step + 1}",
                unit="prompt",
            ) as progress_bar:

                # 逐行讲解：定义异步函数 run_and_track：执行单题 rollout，并同步更新当前 step 的进度条。
                async def run_and_track(row: dict[str, str]) -> PromptRollout:
                    """执行单题 rollout，并同步更新当前 step 的进度条。"""
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
                # 逐行讲解：抛出异常：RuntimeError('No C-Eval OPD datums were built')。
                raise RuntimeError("No C-Eval OPD datums were built")

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


# 逐行讲解：定义函数 main：启动异步训练并打印总耗时。
def main() -> None:
    """启动异步训练并打印总耗时。"""
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
