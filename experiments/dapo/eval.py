"""在 AIME25 上评测 base model 或传入的 PyTRIO LoRA sampler weights。

Base model：

    uv run python eval.py

LoRA：

    uv run python eval.py \
        --model-path trio://run_xxx/sampler_weights/weights-name \
        --output eval-results/aime25-grpo-10steps.jsonl
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

# 逐行讲解：从 rollout 导入 build_prompt_tokens, stop_sequences。
from rollout import build_prompt_tokens, stop_sequences


# 逐行讲解：调用 trio.configure（trio.configure(sampling_timeout_seconds=18000)）。
trio.configure(sampling_timeout_seconds=18000)

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets' / 'aime_2025'，赋给 DEFAULT_DATASET_PATH。
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "aime_2025"
# 逐行讲解：计算 SCRIPT_DIR / 'eval-results' / 'aime25-base.jsonl'，赋给 DEFAULT_OUTPUT_PATH。
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "eval-results" / "aime25-base.jsonl"
# 逐行讲解：取值 30，赋给 EXPECTED_ROWS。
EXPECTED_ROWS = 30


# 逐行讲解：定义函数 parse_args：解析 AIME25 评测参数。
def parse_args() -> argparse.Namespace:
    """解析 AIME25 评测参数。"""
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-path', type=Path, default=DEFAULT_DATASET_PATH)）。
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B')）。
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--model-path', default=None, help='可选；save_weights_for_sampler 返回的 trio:// 路径，不传则评测 ba…）。
    parser.add_argument(
        "--model-path",
        default=None,
        help="可选；save_weights_for_sampler 返回的 trio:// 路径，不传则评测 base model",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--val-n', type=int, default=12)）。
    parser.add_argument("--val-n", type=int, default=12)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--limit', type=int, default=0)）。
    parser.add_argument("--limit", type=int, default=0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--concurrency', type=int, default=15)）。
    parser.add_argument("--concurrency", type=int, default=15)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-prompt-tokens', type=int, default=4095)）。
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=4095,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=8192)）。
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0)）。
    parser.add_argument("--temperature", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0)）。
    parser.add_argument("--top-p", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-k', type=int, default=-1)）。
    parser.add_argument("--top-k", type=int, default=-1)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, default=None, help='结果 JSONL 路径；base 默认写入 aime25-base.jsonl，LoRA…）。
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="结果 JSONL 路径；base 默认写入 aime25-base.jsonl，LoRA 必须手动指定",
    )
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()

    # 逐行讲解：遍历 ('val_n', 'concurrency', 'max_prompt_tokens', 'max_tokens')，将每项绑定到 name。
    for name in (
        "val_n",
        "concurrency",
        "max_prompt_tokens",
        "max_tokens",
    ):
        # 逐行讲解：检查条件 getattr(args, name) < 1；成立时执行下方分支。
        if getattr(args, name) < 1:
            # 逐行讲解：抛出异常：ValueError(f"--{name.replace('_', '-')} must be >= 1")。
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    # 逐行讲解：检查条件 args.limit < 0；成立时执行下方分支。
    if args.limit < 0:
        # 逐行讲解：抛出异常：ValueError('--limit must be >= 0')。
        raise ValueError("--limit must be >= 0")
    # 逐行讲解：检查条件 args.temperature < 0 or not 0 < args.top_p <= 1；成立时执行下方分支。
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        # 逐行讲解：抛出异常：ValueError('invalid sampling temperature/top-p')。
        raise ValueError("invalid sampling temperature/top-p")
    # 逐行讲解：检查条件 args.model_path is not None and args.output is None；成立时执行下方分支。
    if args.model_path is not None and args.output is None:
        # 逐行讲解：抛出异常：ValueError('--model-path requires --output')。
        raise ValueError("--model-path requires --output")
    # 逐行讲解：取值 args.output or DEFAULT_OUTPUT_PATH，赋给 args.output。
    args.output = args.output or DEFAULT_OUTPUT_PATH
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 load_aime25：读取本地固定版本 AIME25，并验证 30 题结构。
def load_aime25(path: Path, limit: int) -> Dataset:
    """读取本地固定版本 AIME25，并验证 30 题结构。"""
    # 逐行讲解：检查条件 not path.exists()；成立时执行下方分支。
    if not path.exists():
        # 逐行讲解：抛出异常：FileNotFoundError(f'找不到 AIME25 数据：{path}\n请先运行：uv run python experiments/dapo/prepare_data.py')。
        raise FileNotFoundError(
            f"找不到 AIME25 数据：{path}\n"
            "请先运行：uv run python experiments/dapo/prepare_data.py"
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
        # 逐行讲解：抛出异常：ValueError(f'AIME25 缺少字段 {missing}')。
        raise ValueError(f"AIME25 缺少字段 {missing}")
    # 逐行讲解：检查条件 len(dataset) != EXPECTED_ROWS；成立时执行下方分支。
    if len(dataset) != EXPECTED_ROWS:
        # 逐行讲解：抛出异常：ValueError(f'AIME25 应有 {EXPECTED_ROWS} 题，实际为 {len(dataset)} 题')。
        raise ValueError(
            f"AIME25 应有 {EXPECTED_ROWS} 题，实际为 {len(dataset)} 题"
        )
    # 逐行讲解：检查条件 limit > 0；成立时执行下方分支。
    if limit > 0:
        # 逐行讲解：按索引选择数据行（dataset.select(range(min(limit, len(dataset))))），赋给 dataset。
        dataset = dataset.select(range(min(limit, len(dataset))))
    # 逐行讲解：返回结果：dataset。
    return dataset


# 逐行讲解：定义函数 extract_last_boxed：提取最后一个完整 ``\boxed{...}``，支持嵌套花括号。
def extract_last_boxed(text: str) -> str | None:
    """提取最后一个完整 ``\\boxed{...}``，支持嵌套花括号。"""
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


# 逐行讲解：定义函数 normalize_aime_answer：把 AIME 答案规范为无前导零的整数字符串。
def normalize_aime_answer(answer: str | None) -> str | None:
    """把 AIME 答案规范为无前导零的整数字符串。"""
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
    # 逐行讲解：返回结果：str(int(match.group(1))) if match is not None else None。
    return str(int(match.group(1))) if match is not None else None


# 逐行讲解：定义异步函数 evaluate_problem：并发采样一道题的 N 个答案。
async def evaluate_problem(
    index: int,
    row: dict[str, Any],
    sampling_client: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """并发采样一道题的 N 个答案。"""
    # 逐行讲解：调用 str(row['problem']).strip（str(row['problem']).strip()），赋给 problem。
    problem = str(row["problem"]).strip()
    # 逐行讲解：调用 normalize_aime_answer（normalize_aime_answer(str(row['answer']))），赋给 ground_truth。
    ground_truth = normalize_aime_answer(str(row["answer"]))
    # 逐行讲解：检查条件 ground_truth is None；成立时执行下方分支。
    if ground_truth is None:
        # 逐行讲解：抛出异常：ValueError(f"AIME25 第 {index} 题 ground truth 不是整数：{row['answer']!r}")。
        raise ValueError(
            f"AIME25 第 {index} 题 ground truth 不是整数：{row['answer']!r}"
        )
    # 逐行讲解：调用 build_prompt_tokens（build_prompt_tokens(tokenizer, problem, max_prompt_tokens=args.max_prompt_tokens)），赋给 prompt_tokens。
    prompt_tokens = build_prompt_tokens(
        tokenizer,
        problem,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    # 逐行讲解：调用 trio.SamplingParams（trio.SamplingParams(max_tokens=args.max_tokens, seed=args.seed + index, temperature=args.temperature, top_p…），赋给 params。
    params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        seed=args.seed + index,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=stop_sequences(tokenizer),
    )
    # 逐行讲解：进入上下文管理器 semaphore，退出时执行其收尾逻辑。
    async with semaphore:
        # 逐行讲解：等待异步操作 sampling_client.sample_async(prompt=trio.ModelInput.from_ints(prompt_tokens), num_samples=args.val_n, sampl… 完成并取得结果，赋给 response。
        response = await sampling_client.sample_async(
            prompt=trio.ModelInput.from_ints(prompt_tokens),
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

    # 逐行讲解：声明 generations 的类型为 list[dict[str, Any]]；构造列表 []。
    generations: list[dict[str, Any]] = []
    # 逐行讲解：遍历 response.sequences，将每项绑定到 sequence。
    for sequence in response.sequences:
        # 逐行讲解：取值 sequence.text，赋给 text。
        text = sequence.text
        # 逐行讲解：检查条件 text is None；成立时执行下方分支。
        if text is None:
            # 逐行讲解：将 token 解码为文本（tokenizer.decode(sequence.tokens, skip_special_tokens=False)），赋给 text。
            text = tokenizer.decode(
                sequence.tokens,
                skip_special_tokens=False,
            )
        # 逐行讲解：调用 extract_last_boxed（extract_last_boxed(str(text))），赋给 boxed。
        boxed = extract_last_boxed(str(text))
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
                "stop_reason": str(getattr(sequence, "stop_reason", "")),
                "text": str(text),
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


# 逐行讲解：定义函数 summarize：聚合 Average@N、Pass@N、格式率与生成长度。
def summarize(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """聚合 Average@N、Pass@N、格式率与生成长度。"""
    # 逐行讲解：通过推导式逐项生成结果：[generation for result in results for generation in result['generations']]，赋给 generations。
    generations = [
        generation
        for result in results
        for generation in result["generations"]
    ]
    # 逐行讲解：调用 len（len(generations)），赋给 total_generations。
    total_generations = len(generations)
    # 逐行讲解：求和（sum((result['num_correct'] for result in results))），赋给 total_correct。
    total_correct = sum(result["num_correct"] for result in results)
    # 逐行讲解：求和（sum((int(result['pass_at_n']) for result in results))），赋给 passed_problems。
    passed_problems = sum(int(result["pass_at_n"]) for result in results)
    # 逐行讲解：返回结果：{'type': 'summary', 'dataset': 'yentinglin/aime_2025', 'base_model': args.base_model, 'model_path': args.mo…。
    return {
        "type": "summary",
        "dataset": "yentinglin/aime_2025",
        "base_model": args.base_model,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "val_n": args.val_n,
        "problems": len(results),
        "generations": total_generations,
        "average_at_n": (
            total_correct / total_generations if total_generations else 0.0
        ),
        "pass_at_n": (
            passed_problems / len(results) if results else 0.0
        ),
        "format_rate": (
            sum(int(item["formatted"]) for item in generations)
            / total_generations
            if total_generations
            else 0.0
        ),
        "mean_completion_tokens": (
            sum(int(item["completion_tokens"]) for item in generations)
            / total_generations
            if total_generations
            else 0.0
        ),
        "correct_generations": total_correct,
        "passed_problems": passed_problems,
    }


# 逐行讲解：定义异步函数 evaluate：创建采样 client，完成 AIME25 评测并落盘。
async def evaluate(args: argparse.Namespace) -> None:
    """创建采样 client，完成 AIME25 评测并落盘。"""
    # 逐行讲解：调用 load_aime25（load_aime25(args.dataset_path, args.limit)），赋给 dataset。
    dataset = load_aime25(args.dataset_path, args.limit)
    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：构造字典 {'base_model': args.base_model}，赋给 client_kwargs。
    client_kwargs = {"base_model": args.base_model}
    # 逐行讲解：检查条件 args.model_path is not None；成立时执行下方分支。
    if args.model_path is not None:
        # 逐行讲解：取值 args.model_path，赋给 client_kwargs['model_path']。
        client_kwargs["model_path"] = args.model_path
    # 逐行讲解：等待异步操作 service_client.create_sampling_client_async(**client_kwargs) 完成并取得结果，赋给 sampling_client。
    sampling_client = await service_client.create_sampling_client_async(
        **client_kwargs
    )
    # 逐行讲解：调用 sampling_client.get_tokenizer（sampling_client.get_tokenizer()），赋给 tokenizer。
    tokenizer = sampling_client.get_tokenizer()
    # 逐行讲解：调用 asyncio.Semaphore（asyncio.Semaphore(args.concurrency)），赋给 semaphore。
    semaphore = asyncio.Semaphore(args.concurrency)

    # 逐行讲解：进入上下文管理器 tqdm(total=len(dataset), desc='AIME25', unit='problem')，退出时执行其收尾逻辑。
    with tqdm(total=len(dataset), desc="AIME25", unit="problem") as progress:

        # 逐行讲解：定义异步函数 evaluate_and_track：封装下方逻辑，供调用方复用。
        async def evaluate_and_track(
            index: int,
            row: dict[str, Any],
        ) -> dict[str, Any]:
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

        # 逐行讲解：调用 list（list(await asyncio.gather(*(evaluate_and_track(index, row) for index, row in enumerate(dataset))))），赋给 results。
        results = list(
            await asyncio.gather(
                *(
                    evaluate_and_track(index, row)
                    for index, row in enumerate(dataset)
                )
            )
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

    # 逐行讲解：按条件选择结果：'base' if args.model_path is None else 'LoRA'，赋给 model_type。
    model_type = "base" if args.model_path is None else "LoRA"
    # 逐行讲解：调用 print（print(f"AIME25 [{model_type}] Average@{args.val_n}: {summary['average_at_n']:.2%} | Pass@{args.val_n}: {sum…）。
    print(
        f"AIME25 [{model_type}] Average@{args.val_n}: "
        f"{summary['average_at_n']:.2%} | "
        f"Pass@{args.val_n}: {summary['pass_at_n']:.2%} | "
        f"Format: {summary['format_rate']:.2%}"
    )
    # 逐行讲解：调用 print（print(json.dumps(summary, ensure_ascii=False, indent=2))）。
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # 逐行讲解：调用 print（print(f'Results: {args.output}')）。
    print(f"Results: {args.output}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 asyncio.run（asyncio.run(evaluate(parse_args()))）。
    asyncio.run(evaluate(parse_args()))
