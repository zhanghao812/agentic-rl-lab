"""MedQA-zh 四选一评测脚本。

使用 TRIO OpenAI-compatible API 评测模型。
`--model` 可以传基模名称，也可以传 trio:// sampler 权重路径。

启动命令：
uv run python 01-eval-medical.py \
    --model Qwen/Qwen3.5-4B \
    --concurrency 16
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：os。
import os
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：导入模块：time。
import time
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：从 dotenv 导入 load_dotenv。
from dotenv import load_dotenv
# 逐行讲解：从 openai 导入 AsyncOpenAI。
from openai import AsyncOpenAI
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm


# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets' / 'medqa-zh-4options' / 'test_sample.jsonl'，赋给 DEFAULT_DATASET_PATH。
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "medqa-zh-4options" / "test_sample.jsonl"
# 逐行讲解：计算 SCRIPT_DIR / 'eval-results'，赋给 DEFAULT_OUTPUT_DIR。
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval-results"
# 逐行讲解：取值 'https://pytrio.cn/api/openai/v1'，赋给 DEFAULT_BASE_URL。
DEFAULT_BASE_URL = "https://pytrio.cn/api/openai/v1"
# 逐行讲解：取值 '你是中文单项选择题作答助手。请在内部完成必要推理，但最终回答只能包含 A、B、C、D 中的一个大写字母，不要输出推理过程、解释、标点或其他文字。'，赋给 CHOICE_SYSTEM_MESSAGE。
CHOICE_SYSTEM_MESSAGE = (
    "你是中文单项选择题作答助手。请在内部完成必要推理，"
    "但最终回答只能包含 A、B、C、D 中的一个大写字母，"
    "不要输出推理过程、解释、标点或其他文字。"
)


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='MedQA-zh 四选一评测')），赋给 parser。
    parser = argparse.ArgumentParser(description="MedQA-zh 四选一评测")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-path', type=Path, default=DEFAULT_DATASET_PATH, help='MedQA-zh JSONL 文件')）。
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="MedQA-zh JSONL 文件")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output-path', type=Path, default=None, help='预测结果 JSONL 输出路径；不传则按 model 自动生成')）。
    parser.add_argument("--output-path", type=Path, default=None, help="预测结果 JSONL 输出路径；不传则按 model 自动生成")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--metrics-path', type=Path, default=None, help='准确率指标 JSON 输出路径；默认根据 output-path 自动生成')）。
    parser.add_argument("--metrics-path", type=Path, default=None, help="准确率指标 JSON 输出路径；默认根据 output-path 自动生成")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--limit', type=int, default=0, help='最多评测多少条；<=0 表示全量')）。
    parser.add_argument("--limit", type=int, default=0, help="最多评测多少条；<=0 表示全量")

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-url', default=DEFAULT_BASE_URL, help='OpenAI-compatible API base_url')）。
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base_url")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--api-key-env', default='PYTRIO_API_KEY', help='读取 API key 的环境变量名')）。
    parser.add_argument("--api-key-env", default="PYTRIO_API_KEY", help="读取 API key 的环境变量名")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--model', default='Qwen/Qwen3.5-4B', help='模型名或 trio:// sampler 权重路径')）。
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="模型名或 trio:// sampler 权重路径")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--system-message', default=CHOICE_SYSTEM_MESSAGE, help='发送给模型的 system message')）。
    parser.add_argument("--system-message", default=CHOICE_SYSTEM_MESSAGE, help="发送给模型的 system message")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=8192, help='每题最多生成 token 数')）。
    parser.add_argument("--max-tokens", type=int, default=8192, help="每题最多生成 token 数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=0.01, help='采样 temperature')）。
    parser.add_argument("--temperature", type=float, default=0.01, help="采样 temperature")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=0.9, help='采样 top_p')）。
    parser.add_argument("--top-p", type=float, default=0.9, help="采样 top_p")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--concurrency', type=int, default=4, help='并发请求数量')）。
    parser.add_argument("--concurrency", type=int, default=4, help="并发请求数量")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--timeout', type=float, default=120.0, help='单次请求超时时间，单位秒')）。
    parser.add_argument("--timeout", type=float, default=120.0, help="单次请求超时时间，单位秒")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--show-errors', type=int, default=5, help='终端最多展示多少条错误样例')）。
    parser.add_argument("--show-errors", type=int, default=5, help="终端最多展示多少条错误样例")
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()

    # 逐行讲解：检查条件 args.limit < 0；成立时执行下方分支。
    if args.limit < 0:
        # 逐行讲解：抛出异常：ValueError('--limit must be >= 0')。
        raise ValueError("--limit must be >= 0")
    # 逐行讲解：检查条件 args.max_tokens <= 0；成立时执行下方分支。
    if args.max_tokens <= 0:
        # 逐行讲解：抛出异常：ValueError('--max-tokens must be > 0')。
        raise ValueError("--max-tokens must be > 0")
    # 逐行讲解：检查条件 args.concurrency <= 0；成立时执行下方分支。
    if args.concurrency <= 0:
        # 逐行讲解：抛出异常：ValueError('--concurrency must be > 0')。
        raise ValueError("--concurrency must be > 0")
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 model_slug：封装下方逻辑，供调用方复用。
def model_slug(model: str) -> str:
    # 逐行讲解：调用 re.sub('[^A-Za-z0-9._-]+', '-', model.strip()).strip('-').lower（re.sub('[^A-Za-z0-9._-]+', '-', model.strip()).strip('-').lower()），赋给 slug。
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model.strip()).strip("-").lower()
    # 逐行讲解：返回结果：slug or 'model'。
    return slug or "model"


# 逐行讲解：定义函数 default_output_path：封装下方逻辑，供调用方复用。
def default_output_path(model: str) -> Path:
    # 逐行讲解：返回结果：DEFAULT_OUTPUT_DIR / f'medical_eval_{model_slug(model)}.jsonl'。
    return DEFAULT_OUTPUT_DIR / f"medical_eval_{model_slug(model)}.jsonl"


# 逐行讲解：定义函数 load_examples：封装下方逻辑，供调用方复用。
def load_examples(path: Path, limit: int) -> list[dict[str, Any]]:
    # 逐行讲解：检查条件 not path.exists()；成立时执行下方分支。
    if not path.exists():
        # 逐行讲解：抛出异常：FileNotFoundError(f'找不到评测文件：{path}，请先运行 00-download-dataset.py')。
        raise FileNotFoundError(f"找不到评测文件：{path}，请先运行 00-download-dataset.py")

    # 逐行讲解：声明 examples 的类型为 list[dict[str, Any]]；构造列表 []。
    examples: list[dict[str, Any]] = []
    # 逐行讲解：进入上下文管理器 path.open('r', encoding='utf-8')，退出时执行其收尾逻辑。
    with path.open("r", encoding="utf-8") as f:
        # 逐行讲解：遍历 f，将每项绑定到 line。
        for line in f:
            # 逐行讲解：检查条件 not line.strip()；成立时执行下方分支。
            if not line.strip():
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(line)），赋给 row。
            row = json.loads(line)
            # 逐行讲解：检查条件 'answer_idx' not in row；成立时执行下方分支。
            if "answer_idx" not in row:
                # 逐行讲解：抛出异常：ValueError(f'评测样本缺少 answer_idx 字段：{row.keys()}')。
                raise ValueError(f"评测样本缺少 answer_idx 字段：{row.keys()}")
            # 逐行讲解：向列表末尾追加一个元素（examples.append(row)）。
            examples.append(row)
            # 逐行讲解：检查条件 limit > 0 and len(examples) >= limit；成立时执行下方分支。
            if limit > 0 and len(examples) >= limit:
                # 逐行讲解：结束当前循环。
                break
    # 逐行讲解：返回结果：examples。
    return examples


# 逐行讲解：定义函数 build_prompt：封装下方逻辑，供调用方复用。
def build_prompt(row: dict[str, Any]) -> str:
    # 逐行讲解：调用 str(row.get('question', '')).strip（str(row.get('question', '')).strip()），赋给 question。
    question = str(row.get("question", "")).strip()
    # 逐行讲解：取值 row.get('options') or {}，赋给 options。
    options = row.get("options") or {}
    # 逐行讲解：检查条件 question and all((key in options for key in ('A', 'B', 'C', 'D')))；成立时执行下方分支。
    if question and all(key in options for key in ("A", "B", "C", "D")):
        # 逐行讲解：返回结果：'\n'.join(['以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。', '', f'题目：{question}', f"A. {options['A']}", f"B. {options[…。
        return "\n".join(
            [
                "以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。",
                "",
                f"题目：{question}",
                f"A. {options['A']}",
                f"B. {options['B']}",
                f"C. {options['C']}",
                f"D. {options['D']}",
                "",
                "答案：",
            ]
        )
    # 逐行讲解：检查条件 'prompt' in row；成立时执行下方分支。
    if "prompt" in row:
        # 逐行讲解：返回结果：str(row['prompt'])。
        return str(row["prompt"])
    # 逐行讲解：抛出异常：ValueError(f'评测样本缺少 question/options 或 prompt 字段：{row.keys()}')。
    raise ValueError(f"评测样本缺少 question/options 或 prompt 字段：{row.keys()}")


# 逐行讲解：定义函数 create_client：封装下方逻辑，供调用方复用。
def create_client(args: argparse.Namespace) -> AsyncOpenAI:
    # 逐行讲解：调用 load_dotenv（load_dotenv(SCRIPT_DIR / '.env', encoding='utf-8-sig', override=True)）。
    load_dotenv(SCRIPT_DIR / ".env", encoding="utf-8-sig", override=True)
    # 逐行讲解：调用 os.getenv（os.getenv(args.api_key_env)），赋给 api_key。
    api_key = os.getenv(args.api_key_env)
    # 逐行讲解：检查条件 not api_key；成立时执行下方分支。
    if not api_key:
        # 逐行讲解：抛出异常：RuntimeError(f'请在 experiments/medical-opd/.env 或环境变量中设置 {args.api_key_env}')。
        raise RuntimeError(f"请在 experiments/medical-opd/.env 或环境变量中设置 {args.api_key_env}")

    # 逐行讲解：返回结果：AsyncOpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout)。
    return AsyncOpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=args.timeout,
    )


# 逐行讲解：定义异步函数 infer_one：封装下方逻辑，供调用方复用。
async def infer_one(client: AsyncOpenAI, args: argparse.Namespace, prompt: str) -> str:
    # 逐行讲解：等待异步操作 client.chat.completions.create(model=args.model, messages=[{'role': 'system', 'content': args.system_messag… 完成并取得结果，赋给 response。
    response = await client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": args.system_message},
            {"role": "user", "content": prompt},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    # 逐行讲解：返回结果：response.choices[0].message.content or ''。
    return response.choices[0].message.content or ""


# 逐行讲解：定义函数 parse_choice：封装下方逻辑，供调用方复用。
def parse_choice(text: str) -> str | None:
    # 逐行讲解：调用 text.strip（text.strip()），赋给 raw_text。
    raw_text = text.strip()
    # 逐行讲解：检查条件 '</think>' in raw_text；成立时执行下方分支。
    if "</think>" in raw_text:
        # 逐行讲解：构造列表 [raw_text.rsplit('</think>', 1)[-1]]，赋给 segments。
        segments = [raw_text.rsplit("</think>", 1)[-1]]
    else:
        # 逐行讲解：构造列表 [raw_text]，赋给 segments。
        segments = [raw_text]

    # 逐行讲解：遍历 segments，将每项绑定到 segment。
    for segment in segments:
        # 逐行讲解：调用 segment.strip().upper（segment.strip().upper()），赋给 normalized。
        normalized = segment.strip().upper()
        # 逐行讲解：检查条件 not normalized；成立时执行下方分支。
        if not normalized:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：调用 re.fullmatch（re.fullmatch('[^A-Z]*([ABCD])[^A-Z]*', normalized)），赋给 exact_match。
        exact_match = re.fullmatch(r"[^A-Z]*([ABCD])[^A-Z]*", normalized)
        # 逐行讲解：检查条件 exact_match；成立时执行下方分支。
        if exact_match:
            # 逐行讲解：返回结果：exact_match.group(1)。
            return exact_match.group(1)

        # 逐行讲解：构造列表 ['(?:FINAL\\s*(?:ANSWER|OUTPUT)|ANSWER|OUTPUT|最终答案|正确答案|答案|选项|选择|应选|故选)\\s*(?:IS|是|为|:|：|->)?\\s*[\\(\\[【]?…，赋给 patterns。
        patterns = [
            r"(?:FINAL\s*(?:ANSWER|OUTPUT)|ANSWER|OUTPUT|最终答案|正确答案|答案|选项|选择|应选|故选)\s*(?:IS|是|为|:|：|->)?\s*[\(\[【]?\s*([ABCD])(?![A-Z])",
            r"[\(\[【]\s*([ABCD])\s*[\)\]】]",
        ]
        # 逐行讲解：声明 candidates 的类型为 list[str]；构造列表 []。
        candidates: list[str] = []
        # 逐行讲解：遍历 patterns，将每项绑定到 pattern。
        for pattern in patterns:
            # 逐行讲解：把一组元素依次加入列表（candidates.extend(re.findall(pattern, normalized))）。
            candidates.extend(re.findall(pattern, normalized))
        # 逐行讲解：检查条件 candidates；成立时执行下方分支。
        if candidates:
            # 逐行讲解：返回结果：candidates[-1]。
            return candidates[-1]

        # 逐行讲解：遍历 reversed(normalized.splitlines())，将每项绑定到 line。
        for line in reversed(normalized.splitlines()):
            # 逐行讲解：调用 re.fullmatch（re.fullmatch('\\s*([ABCD])\\s*[\\.。:：、]?\\s*', line)），赋给 line_match。
            line_match = re.fullmatch(r"\s*([ABCD])\s*[\.。:：、]?\s*", line)
            # 逐行讲解：检查条件 line_match；成立时执行下方分支。
            if line_match:
                # 逐行讲解：返回结果：line_match.group(1)。
                return line_match.group(1)

    # 逐行讲解：调用 re.findall（re.findall('(?<![A-Z])([ABCD])(?![A-Z])', segments[0].upper()[-1000:])），赋给 fallback。
    fallback = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", segments[0].upper()[-1000:])
    # 逐行讲解：检查条件 fallback；成立时执行下方分支。
    if fallback:
        # 逐行讲解：返回结果：fallback[-1]。
        return fallback[-1]
    # 逐行讲解：返回结果：None。
    return None


# 逐行讲解：定义异步函数 evaluate_one：封装下方逻辑，供调用方复用。
async def evaluate_one(
    index: int,
    row: dict[str, Any],
    client: AsyncOpenAI,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    # 逐行讲解：调用 build_prompt（build_prompt(row)），赋给 prompt。
    prompt = build_prompt(row)
    # 逐行讲解：进入上下文管理器 semaphore，退出时执行其收尾逻辑。
    async with semaphore:
        # 逐行讲解：等待异步操作 infer_one(client, args, prompt) 完成并取得结果，赋给 output。
        output = await infer_one(client, args, prompt)

    # 逐行讲解：调用 parse_choice（parse_choice(output)），赋给 prediction。
    prediction = parse_choice(output)
    # 逐行讲解：调用 str(row['answer_idx']).strip().upper（str(row['answer_idx']).strip().upper()），赋给 answer。
    answer = str(row["answer_idx"]).strip().upper()
    # 逐行讲解：取值 prediction == answer，赋给 is_correct。
    is_correct = prediction == answer
    # 逐行讲解：返回结果：{'index': index, 'question': row.get('question'), 'answer_idx': answer, 'prediction': prediction, 'correct'…。
    return {
        "index": index,
        "question": row.get("question"),
        "answer_idx": answer,
        "prediction": prediction,
        "correct": is_correct,
        "output": output,
        "prompt": prompt,
        "model": args.model,
    }


# 逐行讲解：定义异步函数 evaluate：封装下方逻辑，供调用方复用。
async def evaluate(args: argparse.Namespace) -> None:
    # 逐行讲解：调用 load_examples（load_examples(args.dataset_path, args.limit)），赋给 examples。
    examples = load_examples(args.dataset_path, args.limit)
    # 逐行讲解：检查条件 not examples；成立时执行下方分支。
    if not examples:
        # 逐行讲解：抛出异常：ValueError(f'评测文件为空：{args.dataset_path}')。
        raise ValueError(f"评测文件为空：{args.dataset_path}")

    # 逐行讲解：取值 args.output_path or default_output_path(args.model)，赋给 args.output_path。
    args.output_path = args.output_path or default_output_path(args.model)
    # 逐行讲解：调用 create_client（create_client(args)），赋给 client。
    client = create_client(args)
    # 逐行讲解：创建目录（args.output_path.parent.mkdir(parents=True, exist_ok=True)）。
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # 逐行讲解：调用 print（print(f'Loaded {len(examples)} MedQA-zh examples')）。
    print(f"Loaded {len(examples)} MedQA-zh examples")
    # 逐行讲解：调用 print（print(f'API base_url: {args.base_url}')）。
    print(f"API base_url: {args.base_url}")
    # 逐行讲解：调用 print（print(f'Model: {args.model}')）。
    print(f"Model: {args.model}")
    # 逐行讲解：调用 print（print(f'Concurrency: {args.concurrency}')）。
    print(f"Concurrency: {args.concurrency}")

    # 逐行讲解：调用 time.time（time.time()），赋给 start。
    start = time.time()
    # 逐行讲解：调用 asyncio.Semaphore（asyncio.Semaphore(args.concurrency)），赋给 semaphore。
    semaphore = asyncio.Semaphore(args.concurrency)
    # 逐行讲解：通过推导式逐项生成结果：[asyncio.create_task(evaluate_one(index, row, client, args, semaphore)) for index, row in enumerate(example…，赋给 tasks。
    tasks = [
        asyncio.create_task(evaluate_one(index, row, client, args, semaphore))
        for index, row in enumerate(examples)
    ]
    # 逐行讲解：声明 records 的类型为 list[dict[str, Any]]；构造列表 []。
    records: list[dict[str, Any]] = []
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：遍历 tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='MedQA eval', unit='item')，将每项绑定到 future。
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="MedQA eval", unit="item"):
            # 逐行讲解：向列表末尾追加一个元素（records.append(await future)）。
            records.append(await future)
    finally:
        # 逐行讲解：遍历 tasks，将每项绑定到 task。
        for task in tasks:
            # 逐行讲解：检查条件 not task.done()；成立时执行下方分支。
            if not task.done():
                # 逐行讲解：调用 task.cancel（task.cancel()）。
                task.cancel()
        # 逐行讲解：等待异步操作 asyncio.gather(*tasks, return_exceptions=True) 完成并取得结果。
        await asyncio.gather(*tasks, return_exceptions=True)
        # 逐行讲解：等待异步操作 client.close() 完成并取得结果。
        await client.close()

    # 逐行讲解：调用 records.sort（records.sort(key=lambda item: item['index'])）。
    records.sort(key=lambda item: item["index"])
    # 逐行讲解：求和（sum((1 for item in records if item['correct']))），赋给 correct。
    correct = sum(1 for item in records if item["correct"])
    # 逐行讲解：求和（sum((1 for item in records if item['prediction'] is None))），赋给 invalid。
    invalid = sum(1 for item in records if item["prediction"] is None)
    # 逐行讲解：按索引或切片读取 [item for item in records if not item['correct']][:args.show_errors]，赋给 errors。
    errors = [item for item in records if not item["correct"]][: args.show_errors]

    # 逐行讲解：进入上下文管理器 args.output_path.open('w', encoding='utf-8')，退出时执行其收尾逻辑。
    with args.output_path.open("w", encoding="utf-8") as out:
        # 逐行讲解：遍历 records，将每项绑定到 record。
        for record in records:
            # 逐行讲解：调用 out.write（out.write(json.dumps(record, ensure_ascii=False) + '\n')）。
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 逐行讲解：调用 len（len(examples)），赋给 total。
    total = len(examples)
    # 逐行讲解：按条件选择结果：correct / total if total else 0.0，赋给 accuracy。
    accuracy = correct / total if total else 0.0
    # 逐行讲解：计算 time.time() - start，赋给 elapsed。
    elapsed = time.time() - start
    # 逐行讲解：构造字典 {'accuracy': accuracy, 'correct': correct, 'total': total, 'invalid': invalid, 'elapsed_seconds': elapsed, …，赋给 metrics。
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "invalid": invalid,
        "elapsed_seconds": elapsed,
        "model": args.model,
        "concurrency": args.concurrency,
        "system_message": args.system_message,
        "dataset_path": str(args.dataset_path),
        "output_path": str(args.output_path),
    }
    # 逐行讲解：取值 args.metrics_path or args.output_path.with_suffix('.metrics.json')，赋给 metrics_path。
    metrics_path = args.metrics_path or args.output_path.with_suffix(".metrics.json")
    # 逐行讲解：创建目录（metrics_path.parent.mkdir(parents=True, exist_ok=True)）。
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：将文本写入文件（metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')）。
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 逐行讲解：调用 print（print(f'accuracy {accuracy:.4f} | correct {correct}/{total} | invalid {invalid} | time {elapsed:.2f}s')）。
    print(f"accuracy {accuracy:.4f} | correct {correct}/{total} | invalid {invalid} | time {elapsed:.2f}s")
    # 逐行讲解：调用 print（print(f'Saved predictions: {args.output_path}')）。
    print(f"Saved predictions: {args.output_path}")
    # 逐行讲解：调用 print（print(f'Saved metrics: {metrics_path}')）。
    print(f"Saved metrics: {metrics_path}")

    # 逐行讲解：检查条件 errors；成立时执行下方分支。
    if errors:
        # 逐行讲解：调用 print（print('\n错误样例：')）。
        print("\n错误样例：")
        # 逐行讲解：遍历 errors，将每项绑定到 item。
        for item in errors:
            # 逐行讲解：按索引或切片读取 str(item['question'])[:80]，赋给 question。
            question = str(item["question"])[:80]
            # 逐行讲解：调用 print（print(f"- #{item['index']} pred={item['prediction']} answer={item['answer_idx']} question={question}")）。
            print(f"- #{item['index']} pred={item['prediction']} answer={item['answer_idx']} question={question}")


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：调用 asyncio.run（asyncio.run(evaluate(parse_args()))）。
    asyncio.run(evaluate(parse_args()))


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
