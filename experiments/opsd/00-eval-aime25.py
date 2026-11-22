"""使用 PyTRIO 评测 base model 或 OPSD sampler weights 在 AIME 2025 上的表现。

先下载评测集：

uv run python 00-datasets.py --only aime25


Base Model 评测：

uv run python 00-eval-aime25.py \
    --val-n 12 \
    --max-tokens 38912 \
    --temperature 1.0 \
    --enable-thinking false \
    --output eval-results/aime25-base.jsonl

Sample weights 评测：

uv run python 00-eval-aime25.py \
    --val-n 12 \
    --max-tokens 38912 \
    --temperature 1.0 \
    --enable-thinking false \
    --model-path trio://<your_sampler_weights_path> \
    --output eval-results/aime25-sampler-steps25.jsonl

脚本报告 Average@N、Pass@N 和 boxed format rate。AIME 答案均为整数，
因此使用整数精确匹配，不额外依赖 math_verify。
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


# 逐行讲解：调用 trio.configure（trio.configure(sampling_timeout_seconds=18000)）。
trio.configure(sampling_timeout_seconds=18000,)  # 5 小时，足够评测 30 道题

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets' / 'aime_2025'，赋给 DEFAULT_DATASET_PATH。
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "aime_2025"
# 逐行讲解：计算 SCRIPT_DIR / 'eval-results' / 'aime25.jsonl'，赋给 DEFAULT_OUTPUT_PATH。
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "eval-results" / "aime25.jsonl"
# 逐行讲解：取值 30，赋给 EXPECTED_ROWS。
EXPECTED_ROWS = 30


# 逐行讲解：定义函数 parse_bool：把命令行中的 True/False 文本转换为布尔值。
def parse_bool(value: str) -> bool:
    """把命令行中的 True/False 文本转换为布尔值。"""
    # 逐行讲解：调用 value.strip().lower（value.strip().lower()），赋给 normalized。
    normalized = value.strip().lower()
    # 逐行讲解：检查条件 normalized == 'true'；成立时执行下方分支。
    if normalized == "true":
        # 逐行讲解：返回结果：True。
        return True
    # 逐行讲解：检查条件 normalized == 'false'；成立时执行下方分支。
    if normalized == "false":
        # 逐行讲解：返回结果：False。
        return False
    # 逐行讲解：抛出异常：argparse.ArgumentTypeError('expected True or False')。
    raise argparse.ArgumentTypeError("expected True or False")


# 逐行讲解：定义函数 parse_args：解析并校验 AIME25 并发采样与输出相关参数。
def parse_args() -> argparse.Namespace:
    """解析并校验 AIME25 并发采样与输出相关参数。"""
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-path', type=Path, default=DEFAULT_DATASET_PATH, help='00-datasets.py 保存的 AIM…）。
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="00-datasets.py 保存的 AIME25 数据目录",
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
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--val-n', type=int, default=12, help='每道题生成多少个答案')）。
    parser.add_argument("--val-n", type=int, default=12, help="每道题生成多少个答案")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--limit', type=int, default=0, help='只评测前 N 题；0 表示全部 30 题')）。
    parser.add_argument(
        "--limit", type=int, default=0, help="只评测前 N 题；0 表示全部 30 题"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--concurrency', type=int, default=15, help='并发评测题目数')）。
    parser.add_argument("--concurrency", type=int, default=15, help="并发评测题目数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=38912, help='每个答案最大生成 token 数')）。
    parser.add_argument(
        "--max-tokens", type=int, default=38912, help="每个答案最大生成 token 数"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0)）。
    parser.add_argument("--temperature", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0)）。
    parser.add_argument("--top-p", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-k', type=int, default=-1)）。
    parser.add_argument("--top-k", type=int, default=-1)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--enable-thinking', type=parse_bool, default=False, metavar='{True,False}', help='是否启用…）。
    parser.add_argument(
        "--enable-thinking",
        type=parse_bool,
        default=False,
        metavar="{True,False}",
        help="是否启用模型 chat template 的 thinking 模式；默认 False",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_PATH, help='逐题 JSONL 输出；最后一行为 summary')）。
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="逐题 JSONL 输出；最后一行为 summary",
    )
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()

    # 逐行讲解：遍历 ('val_n', 'concurrency', 'max_tokens')，将每项绑定到 name。
    for name in ("val_n", "concurrency", "max_tokens"):
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
    """提取最后一个花括号完整闭合的 ``\\boxed{...}``，支持嵌套花括号。

    生成被 max_tokens 截断时，末尾可能只剩半个 ``\\boxed{``，
    此时向前回退到最近一个完整的 ``\\boxed{...}``。
    """
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


# 逐行讲解：定义函数 build_prompt_ids：套用模型 chat template，将一道题编码为采样 prompt token。
def build_prompt_ids(tokenizer: Any, problem: str, enable_thinking: bool) -> list[int]:
    """套用模型 chat template，将一道题编码为采样 prompt token。"""
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
        enable_thinking=enable_thinking,
    )
    # 逐行讲解：将文本编码为 token（tokenizer.encode(prompt, add_special_tokens=False)），赋给 prompt_ids。
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # 逐行讲解：检查条件 not prompt_ids；成立时执行下方分支。
    if not prompt_ids:
        # 逐行讲解：抛出异常：ValueError('AIME25 prompt token 为空')。
        raise ValueError("AIME25 prompt token 为空")
    # 逐行讲解：返回结果：prompt_ids。
    return prompt_ids


# 逐行讲解：定义异步函数 evaluate_problem：并发采样一道题的 N 个答案，并计算正确性与多数投票结果。
async def evaluate_problem(
    index: int,
    row: dict[str, Any],
    sampling_client: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """并发采样一道题的 N 个答案，并计算正确性与多数投票结果。"""
    # 逐行讲解：调用 str(row['problem']).strip（str(row['problem']).strip()），赋给 problem。
    problem = str(row["problem"]).strip()
    # 逐行讲解：调用 normalize_aime_answer（normalize_aime_answer(str(row['answer']))），赋给 ground_truth。
    ground_truth = normalize_aime_answer(str(row["answer"]))
    # 逐行讲解：检查条件 ground_truth is None；成立时执行下方分支。
    if ground_truth is None:
        # 逐行讲解：抛出异常：ValueError(f"AIME25 第 {index} 题 ground truth 不是整数: {row['answer']!r}")。
        raise ValueError(
            f"AIME25 第 {index} 题 ground truth 不是整数: {row['answer']!r}"
        )

    # 逐行讲解：调用 build_prompt_ids（build_prompt_ids(tokenizer, problem, args.enable_thinking)），赋给 prompt_ids。
    prompt_ids = build_prompt_ids(tokenizer, problem, args.enable_thinking)
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


# 逐行讲解：定义函数 summarize：聚合逐题结果，生成 Average@N、Pass@N 等总体指标。
def summarize(
    results: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    """聚合逐题结果，生成 Average@N、Pass@N 等总体指标。"""
    # 逐行讲解：调用 len（len(results)），赋给 total_problems。
    total_problems = len(results)
    # 逐行讲解：求和（sum((len(item['generations']) for item in results))），赋给 total_generations。
    total_generations = sum(len(item["generations"]) for item in results)
    # 逐行讲解：求和（sum((item['num_correct'] for item in results))），赋给 total_correct。
    total_correct = sum(item["num_correct"] for item in results)
    # 逐行讲解：求和（sum((int(generation['formatted']) for item in results for generation in item['generations']))），赋给 total_formatted。
    total_formatted = sum(
        int(generation["formatted"])
        for item in results
        for generation in item["generations"]
    )
    # 逐行讲解：求和（sum((int(item['pass_at_n']) for item in results))），赋给 pass_count。
    pass_count = sum(int(item["pass_at_n"]) for item in results)
    # 逐行讲解：返回结果：{'type': 'summary', 'dataset': 'yentinglin/aime_2025', 'base_model': args.base_model, 'model_path': args.mo…。
    return {
        "type": "summary",
        "dataset": "yentinglin/aime_2025",
        "base_model": args.base_model,
        "model_path": args.model_path,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "val_n": args.val_n,
        "problems": total_problems,
        "generations": total_generations,
        "average_at_n": total_correct / total_generations if total_generations else 0.0,
        "pass_at_n": pass_count / total_problems if total_problems else 0.0,
        "format_rate": total_formatted / total_generations
        if total_generations
        else 0.0,
        "correct_generations": total_correct,
        "passed_problems": pass_count,
    }


# 逐行讲解：定义异步函数 evaluate：创建 PyTRIO 采样客户端，并发完成评测和结果落盘。
async def evaluate(args: argparse.Namespace) -> None:
    """创建 PyTRIO 采样客户端，并发完成评测和结果落盘。"""
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
    # 逐行讲解：调用 asyncio.Semaphore（asyncio.Semaphore(args.concurrency)），赋给 semaphore。
    semaphore = asyncio.Semaphore(args.concurrency)

    # 逐行讲解：进入上下文管理器 tqdm(total=len(dataset), desc='AIME25', unit='problem')，退出时执行其收尾逻辑。
    with tqdm(total=len(dataset), desc="AIME25", unit="problem") as progress:

        # 逐行讲解：定义异步函数 evaluate_and_track：评测单题并在完成后安全更新题目级进度条。
        async def evaluate_and_track(index: int, row: dict[str, Any]) -> dict[str, Any]:
            """评测单题并在完成后安全更新题目级进度条。"""
            # 逐行讲解：等待异步操作 evaluate_problem(index, row, sampling_client, tokenizer, args, semaphore) 完成并取得结果，赋给 result。
            result = await evaluate_problem(
                index,
                row,
                sampling_client,
                tokenizer,
                args,
                semaphore,
            )
            # 逐行讲解：调用 progress.update（progress.update(1)）。
            progress.update(1)
            # 逐行讲解：返回结果：result。
            return result

        # 逐行讲解：等待异步操作 asyncio.gather(*(evaluate_and_track(index, row) for index, row in enumerate(dataset))) 完成并取得结果，赋给 results。
        results = await asyncio.gather(
            *(evaluate_and_track(index, row) for index, row in enumerate(dataset))
        )

    # 逐行讲解：调用 summarize（summarize(results, args)），赋给 summary。
    summary = summarize(results, args)
    # 逐行讲解：创建目录（args.output.parent.mkdir(parents=True, exist_ok=True)）。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：进入上下文管理器 args.output.open('w', encoding='utf-8')，退出时执行其收尾逻辑。
    with args.output.open("w", encoding="utf-8") as file:
        # 逐行讲解：遍历 results，将每项绑定到 result。
        for result in results:
            # 逐行讲解：调用 file.write（file.write(json.dumps(result, ensure_ascii=False) + '\n')）。
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
        # 逐行讲解：调用 file.write（file.write(json.dumps(summary, ensure_ascii=False) + '\n')）。
        file.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # 逐行讲解：调用 print（print(f"AIME25 Average@{args.val_n}: {summary['average_at_n']:.2%} | Pass@{args.val_n}: {summary['pass_at_n…）。
    print(
        f"AIME25 Average@{args.val_n}: {summary['average_at_n']:.2%} | "
        f"Pass@{args.val_n}: {summary['pass_at_n']:.2%} | "
        f"Format: {summary['format_rate']:.2%}"
    )
    # 逐行讲解：调用 print（print('Evaluation summary:')）。
    print("Evaluation summary:")
    # 逐行讲解：调用 print（print(json.dumps(summary, ensure_ascii=False, indent=2))）。
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # 逐行讲解：调用 print（print(f'Results: {args.output}')）。
    print(f"Results: {args.output}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 asyncio.run（asyncio.run(evaluate(parse_args()))）。
    asyncio.run(evaluate(parse_args()))
