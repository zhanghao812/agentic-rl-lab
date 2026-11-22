"""使用 PyTRIO 在 AIME 2025 上评测 ReTool（retool / text-only 两种模式）。

retool 模式（多轮代码工具 rollout，复用 rollout.py 状态机）：
uv run python eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --output eval-results/aime25-retool-base.jsonl

评测训练好的 sampler weights 时加：
uv run python eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --model-path trio://<your_sampler_weights_path> \
    --output eval-results/aime25-retool-stepxx.jsonl

脚本报告 Average@N（论文 pass@1 的估计方式）、Pass@N、boxed format rate
和代码调用统计。评测集复用 experiments/opsd/datasets/aime_2025（30 题，答案均为整数）。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：从 datasets 导入 Dataset, DatasetDict, load_from_disk。
from datasets import Dataset, DatasetDict, load_from_disk
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm

# 逐行讲解：从 data 导入 MathExample。
from data import MathExample
# 逐行讲解：从 rollout 导入 RolloutConfig, rollout_batch_async。
from rollout import RolloutConfig, rollout_batch_async
# 逐行讲解：从 sandbox 导入 LocalPythonSandbox。
from sandbox import LocalPythonSandbox


# 逐行讲解：调用 trio.configure（trio.configure(sampling_timeout_seconds=18000)）。
trio.configure(sampling_timeout_seconds=18000,)  # 5 小时，足够多轮评测 30 道题

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR.parent / 'opsd' / 'datasets' / 'aime_2025'，赋给 DEFAULT_DATASET_PATH。
DEFAULT_DATASET_PATH = SCRIPT_DIR.parent / "opsd" / "datasets" / "aime_2025"
# 逐行讲解：计算 SCRIPT_DIR / 'eval-results' / 'aime25.jsonl'，赋给 DEFAULT_OUTPUT_PATH。
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "eval-results" / "aime25.jsonl"
# 逐行讲解：取值 30，赋给 EXPECTED_ROWS。
EXPECTED_ROWS = 30


# 逐行讲解：定义函数 parse_args：解析并校验评测参数。
def parse_args() -> argparse.Namespace:
    """解析并校验评测参数。"""
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--mode', choices=['retool', 'text-only'], default='retool', help='retool=多轮代码工具 rollou…）。
    parser.add_argument(
        "--mode",
        choices=["retool", "text-only"],
        default="retool",
        help="retool=多轮代码工具 rollout；text-only=禁用工具的单轮推理（对照组）",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-path', type=Path, default=DEFAULT_DATASET_PATH, help='experiments/opsd 保存的 A…）。
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="experiments/opsd 保存的 AIME25 数据目录",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B', help='PyTRIO sampling client 的基础模型')）。
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="PyTRIO sampling client 的基础模型",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--model-path', default=None, help='save_weights_for_sampler 返回的 trio:// 路径；留空评测 base m…）。
    parser.add_argument(
        "--model-path",
        default=None,
        help="save_weights_for_sampler 返回的 trio:// 路径；留空评测 base model",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--val-n', type=int, default=12, help='每道题生成多少条轨迹')）。
    parser.add_argument("--val-n", type=int, default=12, help="每道题生成多少条轨迹")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--limit', type=int, default=0, help='只评测前 N 题；0 表示全部 30 题')）。
    parser.add_argument(
        "--limit", type=int, default=0, help="只评测前 N 题；0 表示全部 30 题"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--chunk-size', type=int, default=5, help='retool 模式每次 rollout_batch 处理的题目数（逐 chunk 落盘）')）。
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="retool 模式每次 rollout_batch 处理的题目数（逐 chunk 落盘）",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--concurrency', type=int, default=15, help='text-only 模式并发题目数')）。
    parser.add_argument("--concurrency", type=int, default=15, help="text-only 模式并发题目数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=8192, help='text-only 模式单答案最大 token 数')）。
    parser.add_argument(
        "--max-tokens", type=int, default=8192, help="text-only 模式单答案最大 token 数"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0)）。
    parser.add_argument("--temperature", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=0.7, help='对齐论文评测配置')）。
    parser.add_argument("--top-p", type=float, default=0.7, help="对齐论文评测配置")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-k', type=int, default=-1)）。
    parser.add_argument("--top-k", type=int, default=-1)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-code-calls', type=int, default=4)）。
    parser.add_argument("--max-code-calls", type=int, default=4)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-assistant-turns', type=int, default=6)）。
    parser.add_argument("--max-assistant-turns", type=int, default=6)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-trajectory-tokens', type=int, default=8192)）。
    parser.add_argument("--max-trajectory-tokens", type=int, default=8192)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-assistant-tokens', type=int, default=1024)）。
    parser.add_argument("--max-assistant-tokens", type=int, default=1024)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tool-response-tokens', type=int, default=512)）。
    parser.add_argument("--max-tool-response-tokens", type=int, default=512)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sandbox-timeout', type=float, default=30.0)）。
    parser.add_argument("--sandbox-timeout", type=float, default=30.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--sandbox-workers', type=int, default=8)）。
    parser.add_argument("--sandbox-workers", type=int, default=8)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_PATH, help='逐题 JSONL 输出；最后一行为 summary')）。
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="逐题 JSONL 输出；最后一行为 summary",
    )
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()

    # 逐行讲解：遍历 ('val_n', 'concurrency', 'max_tokens', 'chunk_size')，将每项绑定到 name。
    for name in ("val_n", "concurrency", "max_tokens", "chunk_size"):
        # 逐行讲解：检查条件 getattr(args, name) < 1；成立时执行下方分支。
        if getattr(args, name) < 1:
            # 逐行讲解：抛出异常：ValueError(f"--{name.replace('_', '-')} must be >= 1")。
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    # 逐行讲解：检查条件 args.limit < 0；成立时执行下方分支。
    if args.limit < 0:
        # 逐行讲解：抛出异常：ValueError('--limit must be >= 0')。
        raise ValueError("--limit must be >= 0")
    # 逐行讲解：检查条件 args.temperature < 0；成立时执行下方分支。
    if args.temperature < 0:
        # 逐行讲解：抛出异常：ValueError('--temperature must be >= 0')。
        raise ValueError("--temperature must be >= 0")
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 load_aime25：读取本地 AIME25，验证结构后按需截取少量题目。
def load_aime25(path: Path, limit: int) -> Dataset:
    """读取本地 AIME25，验证结构后按需截取少量题目。"""
    # 逐行讲解：检查条件 not path.exists()；成立时执行下方分支。
    if not path.exists():
        # 逐行讲解：抛出异常：FileNotFoundError(f'找不到 AIME25 数据：{path}\n请先运行：uv run python experiments/opsd/00-datasets.py --only aime25')。
        raise FileNotFoundError(
            f"找不到 AIME25 数据：{path}\n"
            "请先运行：uv run python experiments/opsd/00-datasets.py --only aime25"
        )
    # 逐行讲解：调用 load_from_disk（load_from_disk(str(path))），赋给 loaded。
    loaded = load_from_disk(str(path))
    # 逐行讲解：按条件选择结果：loaded['train'] if isinstance(loaded, DatasetDict) else loaded，赋给 dataset。
    dataset = loaded["train"] if isinstance(loaded, DatasetDict) else loaded
    # 逐行讲解：检查条件 not isinstance(dataset, Dataset)；成立时执行下方分支。
    if not isinstance(dataset, Dataset):
        # 逐行讲解：抛出异常：TypeError(f'期望 Dataset，实际得到 {type(dataset)!r}')。
        raise TypeError(f"期望 Dataset，实际得到 {type(dataset)!r}")
    # 逐行讲解：调用 sorted（sorted({'problem', 'answer'} - set(dataset.column_names))），赋给 missing。
    missing = sorted({"problem", "answer"} - set(dataset.column_names))
    # 逐行讲解：检查条件 missing；成立时执行下方分支。
    if missing:
        # 逐行讲解：抛出异常：ValueError(f'AIME25 缺少字段 {missing}，实际字段为 {dataset.column_names}')。
        raise ValueError(
            f"AIME25 缺少字段 {missing}，实际字段为 {dataset.column_names}"
        )
    # 逐行讲解：检查条件 len(dataset) != EXPECTED_ROWS；成立时执行下方分支。
    if len(dataset) != EXPECTED_ROWS:
        # 逐行讲解：抛出异常：ValueError(f'AIME25 应有 {EXPECTED_ROWS} 题，实际为 {len(dataset)} 题')。
        raise ValueError(f"AIME25 应有 {EXPECTED_ROWS} 题，实际为 {len(dataset)} 题")
    # 逐行讲解：检查条件 limit > 0；成立时执行下方分支。
    if limit > 0:
        # 逐行讲解：按索引选择数据行（dataset.select(range(min(limit, len(dataset))))），赋给 dataset。
        dataset = dataset.select(range(min(limit, len(dataset))))
    # 逐行讲解：返回结果：dataset。
    return dataset


# 逐行讲解：定义函数 extract_last_boxed：提取最后一个花括号完整闭合的 ``\boxed{...}``，支持嵌套花括号。
def extract_last_boxed(text: str) -> str | None:
    """提取最后一个花括号完整闭合的 ``\\boxed{...}``，支持嵌套花括号。"""
    # 逐行讲解：调用 len（len(text)），赋给 end。
    end = len(text)
    # 逐行讲解：当 True 成立时继续循环。
    while True:
        # 逐行讲解：调用 text.rfind（text.rfind('\\boxed', 0, end)），赋给 start。
        start = text.rfind("\\boxed", 0, end)
        # 逐行讲解：检查条件 start < 0；成立时执行下方分支。
        if start < 0:
            # 逐行讲解：返回结果：None。
            return None
        # 逐行讲解：调用 text.find（text.find('{', start)），赋给 left。
        left = text.find("{", start)
        # 逐行讲解：检查条件 left < 0；成立时执行下方分支。
        if left < 0:
            # 逐行讲解：取值 start，赋给 end。
            end = start
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：取值 0，赋给 depth。
        depth = 0
        # 逐行讲解：遍历 range(left, len(text))，将每项绑定到 index。
        for index in range(left, len(text)):
            # 逐行讲解：检查条件 text[index] == '{'；成立时执行下方分支。
            if text[index] == "{":
                # 逐行讲解：更新 depth：depth += 1。
                depth += 1
            # 逐行讲解：检查条件 text[index] == '}'；成立时执行下方分支。
            elif text[index] == "}":
                # 逐行讲解：更新 depth：depth -= 1。
                depth -= 1
                # 逐行讲解：检查条件 depth == 0；成立时执行下方分支。
                if depth == 0:
                    # 逐行讲解：返回结果：text[left + 1:index].strip()。
                    return text[left + 1 : index].strip()
        # 逐行讲解：取值 start，赋给 end。
        end = start


# 逐行讲解：定义函数 normalize_aime_answer：将 AIME 答案规范为无前导零的整数字符串。
def normalize_aime_answer(answer: str | None) -> str | None:
    """将 AIME 答案规范为无前导零的整数字符串。"""
    # 逐行讲解：检查条件 answer is None；成立时执行下方分支。
    if answer is None:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：调用 answer.strip().replace(',', '').replace（answer.strip().replace(',', '').replace('$', '')），赋给 cleaned。
    cleaned = answer.strip().replace(",", "").replace("$", "")
    # 逐行讲解：调用 re.sub（re.sub('\\\\(?:text|mathrm)\\s*\\{([^{}]*)\\}', '\\1', cleaned)），赋给 cleaned。
    cleaned = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", cleaned)
    # 逐行讲解：调用 re.fullmatch（re.fullmatch('\\s*([+-]?\\d+)\\s*', cleaned)），赋给 match。
    match = re.fullmatch(r"\s*([+-]?\d+)\s*", cleaned)
    # 逐行讲解：检查条件 match is None；成立时执行下方分支。
    if match is None:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：str(int(match.group(1)))。
    return str(int(match.group(1)))


# 逐行讲解：定义函数 build_text_prompt_ids：text-only 模式：与 experiments/opsd 评测相同的单轮推理 prompt。
def build_text_prompt_ids(tokenizer: Any, problem: str) -> list[int]:
    """text-only 模式：与 experiments/opsd 评测相同的单轮推理 prompt。"""
    # 逐行讲解：取值 f'{problem.strip()}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.'，赋给 content。
    content = (
        f"{problem.strip()}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt…），赋给 prompt。
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # 逐行讲解：将文本编码为 token（tokenizer.encode(prompt, add_special_tokens=False)），赋给 prompt_ids。
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # 逐行讲解：检查条件 not prompt_ids；成立时执行下方分支。
    if not prompt_ids:
        # 逐行讲解：抛出异常：ValueError('AIME25 prompt token 为空')。
        raise ValueError("AIME25 prompt token 为空")
    # 逐行讲解：返回结果：prompt_ids。
    return prompt_ids


# 逐行讲解：定义异步函数 evaluate_problem_text_only：text-only 模式：并发采样一道题的 N 个单轮答案。
async def evaluate_problem_text_only(
    index: int,
    row: dict[str, Any],
    sampling_client: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """text-only 模式：并发采样一道题的 N 个单轮答案。"""
    # 逐行讲解：调用 str(row['problem']).strip（str(row['problem']).strip()），赋给 problem。
    problem = str(row["problem"]).strip()
    # 逐行讲解：调用 normalize_aime_answer（normalize_aime_answer(str(row['answer']))），赋给 ground_truth。
    ground_truth = normalize_aime_answer(str(row["answer"]))
    # 逐行讲解：检查条件 ground_truth is None；成立时执行下方分支。
    if ground_truth is None:
        # 逐行讲解：抛出异常：ValueError(f"AIME25 第 {index} 题 ground truth 不是整数: {row['answer']!r}")。
        raise ValueError(f"AIME25 第 {index} 题 ground truth 不是整数: {row['answer']!r}")

    # 逐行讲解：调用 build_text_prompt_ids（build_text_prompt_ids(tokenizer, problem)），赋给 prompt_ids。
    prompt_ids = build_text_prompt_ids(tokenizer, problem)
    # 逐行讲解：调用 trio.SamplingParams（trio.SamplingParams(max_tokens=args.max_tokens, seed=args.seed + index, temperature=args.temperature, top_p…），赋给 params。
    params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        seed=args.seed + index,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=list(
            dict.fromkeys(
                token for token in (tokenizer.eos_token, "<|im_end|>") if token
            )
        ),
    )
    # 逐行讲解：进入上下文管理器 semaphore，退出时执行其收尾逻辑。
    async with semaphore:
        # 逐行讲解：等待异步操作 sampling_client.sample_async(prompt=trio.ModelInput.from_ints(prompt_ids), num_samples=args.val_n, sampling… 完成并取得结果，赋给 response。
        response = await sampling_client.sample_async(
            prompt=trio.ModelInput.from_ints(prompt_ids),
            num_samples=args.val_n,
            sampling_params=params,
            return_text=True,
        )
    # 逐行讲解：检查条件 len(response.sequences) != args.val_n；成立时执行下方分支。
    if len(response.sequences) != args.val_n:
        # 逐行讲解：抛出异常：RuntimeError(f'AIME25 第 {index} 题请求 {args.val_n} 条 completion，实际返回 {len(response.sequences)} 条')。
        raise RuntimeError(
            f"AIME25 第 {index} 题请求 {args.val_n} 条 completion，"
            f"实际返回 {len(response.sequences)} 条"
        )

    # 逐行讲解：构造列表 []，赋给 generations。
    generations = []
    # 逐行讲解：遍历 response.sequences，将每项绑定到 sequence。
    for sequence in response.sequences:
        # 逐行讲解：取值 sequence.text，赋给 text。
        text = sequence.text
        # 逐行讲解：检查条件 text is None；成立时执行下方分支。
        if text is None:
            # 逐行讲解：将 token 解码为文本（tokenizer.decode(sequence.tokens, skip_special_tokens=False)），赋给 text。
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
        # 逐行讲解：调用 extract_last_boxed（extract_last_boxed(text)），赋给 boxed。
        boxed = extract_last_boxed(text)
        # 逐行讲解：调用 normalize_aime_answer（normalize_aime_answer(boxed)），赋给 predicted。
        predicted = normalize_aime_answer(boxed)
        # 逐行讲解：向列表末尾追加一个元素（generations.append({'predicted_answer': predicted, 'boxed_answer': boxed, 'correct': predicted == ground_tr…）。
        generations.append(
            {
                "predicted_answer": predicted,
                "boxed_answer": boxed,
                "correct": predicted == ground_truth,
                "formatted": boxed is not None,
                "code_calls": 0,
                "turns": 1,
                "completion_tokens": len(sequence.tokens),
                "text": text,
            }
        )

    # 逐行讲解：返回结果：{'type': 'problem', 'problem_id': int(row.get('id', index)), 'problem': problem, 'ground_truth': ground_tru…。
    return {
        "type": "problem",
        "problem_id": int(row.get("id", index)),
        "problem": problem,
        "ground_truth": ground_truth,
        "val_n": args.val_n,
        "num_correct": sum(int(item["correct"]) for item in generations),
        "pass_at_n": any(item["correct"] for item in generations),
        "generations": generations,
    }


# 逐行讲解：定义异步函数 evaluate_chunk_retool：retool 模式：对一组题目跑多轮代码工具 rollout 并汇总逐题结果。
async def evaluate_chunk_retool(
    chunk: list[tuple[int, dict[str, Any]]],
    sampling_client: Any,
    tokenizer: Any,
    sandbox: LocalPythonSandbox,
    args: argparse.Namespace,
    progress: tqdm,
) -> list[dict[str, Any]]:
    """retool 模式：对一组题目跑多轮代码工具 rollout 并汇总逐题结果。"""
    # 逐行讲解：通过推导式逐项生成结果：[MathExample(id=str(row.get('id', index)), question=str(row['problem']).strip(), answer=str(row['answer']),…，赋给 examples。
    examples = [
        MathExample(
            id=str(row.get("id", index)),
            question=str(row["problem"]).strip(),
            answer=str(row["answer"]),
            data_source="aime_2025",
        )
        for index, row in chunk
    ]
    # 逐行讲解：调用 RolloutConfig（RolloutConfig(group_size=args.val_n, max_code_calls=args.max_code_calls, max_assistant_turns=args.max_assis…），赋给 config。
    config = RolloutConfig(
        group_size=args.val_n,
        max_code_calls=args.max_code_calls,
        max_assistant_turns=args.max_assistant_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        max_tool_response_tokens=args.max_tool_response_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    # 逐行讲解：等待异步操作 rollout_batch_async(sampling_client, tokenizer, sandbox, examples, config, progress_callback=progress.update) 完成并取得结果，赋给 trajectories。
    trajectories = await rollout_batch_async(
        sampling_client,
        tokenizer,
        sandbox,
        examples,
        config,
        progress_callback=progress.update,
    )

    # 逐行讲解：声明 results 的类型为 list[dict[str, Any]]；构造列表 []。
    results: list[dict[str, Any]] = []
    # 逐行讲解：遍历 enumerate(chunk)，将每项绑定到 (question_index, (index, row))。
    for question_index, (index, row) in enumerate(chunk):
        # 逐行讲解：通过推导式逐项生成结果：[trajectory for trajectory in trajectories if trajectory.question_index == question_index]，赋给 group。
        group = [
            trajectory
            for trajectory in trajectories
            if trajectory.question_index == question_index
        ]
        # 逐行讲解：调用 normalize_aime_answer（normalize_aime_answer(str(row['answer']))），赋给 ground_truth。
        ground_truth = normalize_aime_answer(str(row["answer"]))
        # 逐行讲解：通过推导式逐项生成结果：[{'predicted_answer': normalize_aime_answer(extract_last_boxed(trajectory.final_text)), 'boxed_answer': ext…，赋给 generations。
        generations = [
            {
                "predicted_answer": normalize_aime_answer(
                    extract_last_boxed(trajectory.final_text)
                ),
                "boxed_answer": extract_last_boxed(trajectory.final_text),
                "correct": trajectory.correct,
                "formatted": trajectory.valid_format,
                "code_calls": trajectory.code_calls,
                "turns": len(trajectory.turns),
                "completion_tokens": sum(
                    len(turn.completion_tokens) for turn in trajectory.turns
                ),
                "text": trajectory.final_text,
            }
            for trajectory in group
        ]
        # 逐行讲解：向列表末尾追加一个元素（results.append({'type': 'problem', 'problem_id': int(row.get('id', index)), 'problem': str(row['problem']).…）。
        results.append(
            {
                "type": "problem",
                "problem_id": int(row.get("id", index)),
                "problem": str(row["problem"]).strip(),
                "ground_truth": ground_truth,
                "val_n": args.val_n,
                "num_correct": sum(int(item["correct"]) for item in generations),
                "pass_at_n": any(item["correct"] for item in generations),
                "generations": generations,
            }
        )
    # 逐行讲解：返回结果：results。
    return results


# 逐行讲解：定义函数 summarize：聚合逐题结果，生成 Average@N、Pass@N、格式率和代码调用统计。
def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """聚合逐题结果，生成 Average@N、Pass@N、格式率和代码调用统计。"""
    # 逐行讲解：调用 len（len(results)），赋给 total_problems。
    total_problems = len(results)
    # 逐行讲解：通过推导式逐项生成结果：[generation for item in results for generation in item['generations']]，赋给 generations。
    generations = [
        generation for item in results for generation in item["generations"]
    ]
    # 逐行讲解：调用 len（len(generations)），赋给 total_generations。
    total_generations = len(generations)
    # 逐行讲解：求和（sum((item['num_correct'] for item in results))），赋给 total_correct。
    total_correct = sum(item["num_correct"] for item in results)
    # 逐行讲解：求和（sum((int(generation['formatted']) for generation in generations))），赋给 total_formatted。
    total_formatted = sum(int(generation["formatted"]) for generation in generations)
    # 逐行讲解：求和（sum((int(item['pass_at_n']) for item in results))），赋给 pass_count。
    pass_count = sum(int(item["pass_at_n"]) for item in results)
    # 逐行讲解：返回结果：{'type': 'summary', 'dataset': 'yentinglin/aime_2025', 'mode': args.mode, 'base_model': args.base_model, 'm…。
    return {
        "type": "summary",
        "dataset": "yentinglin/aime_2025",
        "mode": args.mode,
        "base_model": args.base_model,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "val_n": args.val_n,
        "problems": total_problems,
        "generations": total_generations,
        "average_at_n": total_correct / total_generations if total_generations else 0.0,
        "pass_at_n": pass_count / total_problems if total_problems else 0.0,
        "format_rate": total_formatted / total_generations if total_generations else 0.0,
        "mean_code_calls": (
            sum(float(generation["code_calls"]) for generation in generations)
            / total_generations
            if total_generations
            else 0.0
        ),
        "mean_turns": (
            sum(float(generation["turns"]) for generation in generations)
            / total_generations
            if total_generations
            else 0.0
        ),
        "correct_generations": total_correct,
        "passed_problems": pass_count,
    }


# 逐行讲解：定义函数 write_results：把逐题结果追加写入 JSONL 文件。
def write_results(
    results: list[dict[str, Any]], output: Path, *, append: bool
) -> None:
    """把逐题结果追加写入 JSONL 文件。"""
    # 逐行讲解：创建目录（output.parent.mkdir(parents=True, exist_ok=True)）。
    output.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：进入上下文管理器 output.open('a' if append else 'w', encoding='utf-8')，退出时执行其收尾逻辑。
    with output.open("a" if append else "w", encoding="utf-8") as file:
        # 逐行讲解：遍历 results，将每项绑定到 result。
        for result in results:
            # 逐行讲解：调用 file.write（file.write(json.dumps(result, ensure_ascii=False) + '\n')）。
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


# 逐行讲解：定义异步函数 evaluate：创建 PyTRIO 采样客户端，按模式完成评测和结果落盘。
async def evaluate(args: argparse.Namespace) -> None:
    """创建 PyTRIO 采样客户端，按模式完成评测和结果落盘。"""
    # 逐行讲解：调用 load_aime25（load_aime25(args.dataset_path, args.limit)），赋给 dataset。
    dataset = load_aime25(args.dataset_path, args.limit)
    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：等待异步操作 service_client.create_sampling_client_async(base_model=args.base_model, model_path=args.model_path) 完成并取得结果，赋给 sampling_client。
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.base_model,
        model_path=args.model_path,
    )
    # 逐行讲解：调用 sampling_client.get_tokenizer（sampling_client.get_tokenizer()），赋给 tokenizer。
    tokenizer = sampling_client.get_tokenizer()

    # 逐行讲解：声明 results 的类型为 list[dict[str, Any]]；构造列表 []。
    results: list[dict[str, Any]] = []
    # 逐行讲解：检查条件 args.mode == 'text-only'；成立时执行下方分支。
    if args.mode == "text-only":
        # 逐行讲解：调用 asyncio.Semaphore（asyncio.Semaphore(args.concurrency)），赋给 semaphore。
        semaphore = asyncio.Semaphore(args.concurrency)
        # 逐行讲解：进入上下文管理器 tqdm(total=len(dataset), desc='AIME25 text-only', unit='problem')，退出时执行其收尾逻辑。
        with tqdm(total=len(dataset), desc="AIME25 text-only", unit="problem") as progress:

            # 逐行讲解：定义异步函数 evaluate_and_track：封装下方逻辑，供调用方复用。
            async def evaluate_and_track(index: int, row: dict[str, Any]) -> dict[str, Any]:
                # 逐行讲解：等待异步操作 evaluate_problem_text_only(index, row, sampling_client, tokenizer, args, semaphore) 完成并取得结果，赋给 result。
                result = await evaluate_problem_text_only(
                    index, row, sampling_client, tokenizer, args, semaphore
                )
                # 逐行讲解：调用 progress.update（progress.update(1)）。
                progress.update(1)
                # 逐行讲解：返回结果：result。
                return result

            # 逐行讲解：调用 list（list(await asyncio.gather(*(evaluate_and_track(index, row) for index, row in enumerate(dataset))))），赋给 results。
            results = list(
                await asyncio.gather(
                    *(evaluate_and_track(index, row) for index, row in enumerate(dataset))
                )
            )
        # 逐行讲解：调用 write_results（write_results(results, args.output, append=False)）。
        write_results(results, args.output, append=False)
    else:
        # 逐行讲解：调用 LocalPythonSandbox（LocalPythonSandbox(timeout=args.sandbox_timeout, max_workers=args.sandbox_workers)），赋给 sandbox。
        sandbox = LocalPythonSandbox(
            timeout=args.sandbox_timeout,
            max_workers=args.sandbox_workers,
        )
        # 逐行讲解：调用 list（list(enumerate(dataset))），赋给 indexed_rows。
        indexed_rows = list(enumerate(dataset))
        # 逐行讲解：通过推导式逐项生成结果：[indexed_rows[offset:offset + args.chunk_size] for offset in range(0, len(indexed_rows), args.chunk_size)]，赋给 chunks。
        chunks = [
            indexed_rows[offset : offset + args.chunk_size]
            for offset in range(0, len(indexed_rows), args.chunk_size)
        ]
        # 逐行讲解：进入上下文管理器 tqdm(total=len(dataset) * args.val_n, desc='AIME25 retool', unit='trajectory')，退出时执行其收尾逻辑。
        with tqdm(
            total=len(dataset) * args.val_n,
            desc="AIME25 retool",
            unit="trajectory",
        ) as progress:
            # 逐行讲解：遍历 enumerate(chunks)，将每项绑定到 (chunk_number, chunk)。
            for chunk_number, chunk in enumerate(chunks):
                # 逐行讲解：等待异步操作 evaluate_chunk_retool(chunk, sampling_client, tokenizer, sandbox, args, progress) 完成并取得结果，赋给 chunk_results。
                chunk_results = await evaluate_chunk_retool(
                    chunk, sampling_client, tokenizer, sandbox, args, progress
                )
                # 逐行讲解：把一组元素依次加入列表（results.extend(chunk_results)）。
                results.extend(chunk_results)
                # 逐 chunk 落盘，长评测中途失败也保留已完成部分。
                write_results(chunk_results, args.output, append=chunk_number > 0)
                # 逐行讲解：调用 tqdm.write（tqdm.write(f"chunk {chunk_number + 1}/{len(chunks)}: correct={sum((r['num_correct'] for r in chunk_results)…）。
                tqdm.write(
                    f"chunk {chunk_number + 1}/{len(chunks)}: "
                    f"correct={sum(r['num_correct'] for r in chunk_results)}"
                    f"/{sum(len(r['generations']) for r in chunk_results)} "
                    f"sandbox={sandbox.stats.metrics()}"
                )

    # 逐行讲解：调用 summarize（summarize(results, args)），赋给 summary。
    summary = summarize(results, args)
    # 逐行讲解：进入上下文管理器 args.output.open('a', encoding='utf-8')，退出时执行其收尾逻辑。
    with args.output.open("a", encoding="utf-8") as file:
        # 逐行讲解：调用 file.write（file.write(json.dumps(summary, ensure_ascii=False) + '\n')）。
        file.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # 逐行讲解：调用 print（print(f"AIME25 [{args.mode}] Average@{args.val_n}: {summary['average_at_n']:.2%} | Pass@{args.val_n}: {summ…）。
    print(
        f"AIME25 [{args.mode}] Average@{args.val_n}: {summary['average_at_n']:.2%} | "
        f"Pass@{args.val_n}: {summary['pass_at_n']:.2%} | "
        f"Format: {summary['format_rate']:.2%} | "
        f"CodeCalls: {summary['mean_code_calls']:.2f}"
    )
    # 逐行讲解：调用 print（print(json.dumps(summary, ensure_ascii=False, indent=2))）。
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # 逐行讲解：调用 print（print(f'Results: {args.output}')）。
    print(f"Results: {args.output}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 asyncio.run（asyncio.run(evaluate(parse_args()))）。
    asyncio.run(evaluate(parse_args()))
