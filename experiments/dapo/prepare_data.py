"""准备 DAPO-Math-17K 训练集和 AIME25 评测集。

DAPO-Math 的官方 parquet 是约 100 倍重复版本，不能直接拿来训练。本脚本按
规范化后的 question 去重，丢弃同题但 ground truth 冲突的组，再用固定 seed
切出 50 条 dev smoke-test 数据，其余作为训练集。同时下载固定版本的 AIME25，
供 ``eval.py`` 使用。

运行：

    uv run python experiments/dapo/prepare_data.py
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：导入模块：random。
import random
# 逐行讲解：导入模块：shutil。
import shutil
# 逐行讲解：从 typing 导入 Any, Iterable。
from typing import Any, Iterable

# 逐行讲解：从 datasets 导入 Dataset, DatasetDict, load_dataset, load_from_disk。
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


# 逐行讲解：取值 'BytedTsinghua-SIA/DAPO-Math-17k'，赋给 DAPO_DATASET_ID。
DAPO_DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
# 逐行讲解：取值 '65877096c24ffa7abc4e4fa5edb95cf3413a5674'，赋给 DAPO_REVISION。
DAPO_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
# 逐行讲解：取值 'yentinglin/aime_2025'，赋给 AIME25_DATASET_ID。
AIME25_DATASET_ID = "yentinglin/aime_2025"
# 逐行讲解：取值 '6f71d77b0b89b9dabe07ab466c51df33f514df7f'，赋给 AIME25_REVISION。
AIME25_REVISION = "6f71d77b0b89b9dabe07ab466c51df33f514df7f"
# 逐行讲解：取值 'aime_2025'，赋给 AIME25_OUTPUT_NAME。
AIME25_OUTPUT_NAME = "aime_2025"
# 逐行讲解：取值 30，赋给 AIME25_EXPECTED_ROWS。
AIME25_EXPECTED_ROWS = 30
# 逐行讲解：构造元组 ('problem', 'answer')，赋给 AIME25_REQUIRED_COLUMNS。
AIME25_REQUIRED_COLUMNS = ("problem", "answer")
# 逐行讲解：取值 'Solve the following math problem step by step. The last line of your response should be of the form Answer…，赋给 PROMPT_PREFIX。
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
# 逐行讲解：取值 '\n\nRemember to put your answer on its own line after "Answer:".'，赋给 PROMPT_SUFFIX。
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


# 逐行讲解：定义类 PreparationStats：数据清洗统计。
@dataclass(frozen=True)
class PreparationStats:
    """数据清洗统计。"""

    # 逐行讲解：声明 total_rows 的类型为 int。
    total_rows: int
    # 逐行讲解：声明 invalid_rows 的类型为 int。
    invalid_rows: int
    # 逐行讲解：声明 duplicate_rows 的类型为 int。
    duplicate_rows: int
    # 逐行讲解：声明 conflicting_questions 的类型为 int。
    conflicting_questions: int


# 逐行讲解：定义函数 strip_dapo_template：剥掉和 ``\boxed{}`` 输出要求冲突的 DAPO 外层模板。
def strip_dapo_template(question: str) -> str:
    """剥掉和 ``\\boxed{}`` 输出要求冲突的 DAPO 外层模板。"""
    # 逐行讲解：检查条件 question.startswith(PROMPT_PREFIX)；成立时执行下方分支。
    if question.startswith(PROMPT_PREFIX):
        # 逐行讲解：按索引或切片读取 question[len(PROMPT_PREFIX):]，赋给 question。
        question = question[len(PROMPT_PREFIX) :]
    # 逐行讲解：检查条件 question.endswith(PROMPT_SUFFIX)；成立时执行下方分支。
    if question.endswith(PROMPT_SUFFIX):
        # 逐行讲解：按索引或切片读取 question[:-len(PROMPT_SUFFIX)]，赋给 question。
        question = question[: -len(PROMPT_SUFFIX)]
    # 逐行讲解：返回结果：question.strip()。
    return question.strip()


# 逐行讲解：定义函数 normalize_question_key：生成只用于去重和切分的稳定题面 key。
def normalize_question_key(question: str) -> str:
    """生成只用于去重和切分的稳定题面 key。"""
    # 逐行讲解：返回结果：' '.join(strip_dapo_template(question).split())。
    return " ".join(strip_dapo_template(question).split())


# 逐行讲解：定义函数 normalize_row：把 verl 格式行转换为本项目的最小训练格式。
def normalize_row(index: int, row: dict[str, Any]) -> dict[str, str] | None:
    """把 verl 格式行转换为本项目的最小训练格式。"""
    # 逐行讲解：查询指定键，缺失时使用默认值（row.get('prompt')），赋给 prompt。
    prompt = row.get("prompt")
    # 逐行讲解：检查条件 not isinstance(prompt, list) or not prompt or (not isinstance(prompt[0], dict))；成立时执行下方分支。
    if not isinstance(prompt, list) or not prompt or not isinstance(prompt[0], dict):
        # 逐行讲解：返回结果：None。
        return None

    # 逐行讲解：调用 strip_dapo_template（strip_dapo_template(str(prompt[0].get('content') or ''))），赋给 question。
    question = strip_dapo_template(str(prompt[0].get("content") or ""))
    # 逐行讲解：查询指定键，缺失时使用默认值（row.get('reward_model')），赋给 reward_model。
    reward_model = row.get("reward_model")
    # 逐行讲解：按条件选择结果：reward_model.get('ground_truth') if isinstance(reward_model, dict) else None，赋给 ground_truth。
    ground_truth = (
        reward_model.get("ground_truth")
        if isinstance(reward_model, dict)
        else None
    )
    # 逐行讲解：检查条件 isinstance(ground_truth, list)；成立时执行下方分支。
    if isinstance(ground_truth, list):
        # 逐行讲解：按条件选择结果：ground_truth[0] if ground_truth else None，赋给 ground_truth。
        ground_truth = ground_truth[0] if ground_truth else None
    # 逐行讲解：调用 str(ground_truth or '').strip（str(ground_truth or '').strip()），赋给 answer。
    answer = str(ground_truth or "").strip()
    # 逐行讲解：检查条件 not question or not answer；成立时执行下方分支。
    if not question or not answer:
        # 逐行讲解：返回结果：None。
        return None

    # 逐行讲解：查询指定键，缺失时使用默认值（row.get('extra_info')），赋给 extra_info。
    extra_info = row.get("extra_info")
    # 逐行讲解：按条件选择结果：extra_info.get('index') if isinstance(extra_info, dict) else None，赋给 source_id。
    source_id = extra_info.get("index") if isinstance(extra_info, dict) else None
    # 逐行讲解：返回结果：{'id': str(source_id or index), 'question': question, 'answer': answer, 'data_source': str(row.get('data_so…。
    return {
        "id": str(source_id or index),
        "question": question,
        "answer": answer,
        "data_source": str(row.get("data_source") or "math_dapo"),
    }


# 逐行讲解：定义函数 deduplicate_rows：按规范化 question 去重，并完整删除 ground truth 冲突的题组。
def deduplicate_rows(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, str]], PreparationStats]:
    """按规范化 question 去重，并完整删除 ground truth 冲突的题组。"""
    # 逐行讲解：声明 unique 的类型为 dict[str, dict[str, str]]；构造字典 {}。
    unique: dict[str, dict[str, str]] = {}
    # 逐行讲解：声明 conflicts 的类型为 set[str]；调用 set（set()）。
    conflicts: set[str] = set()
    # 逐行讲解：取值 0，赋给 total_rows。
    total_rows = 0
    # 逐行讲解：取值 0，赋给 invalid_rows。
    invalid_rows = 0
    # 逐行讲解：取值 0，赋给 duplicate_rows。
    duplicate_rows = 0

    # 逐行讲解：遍历 enumerate(rows)，将每项绑定到 (index, row)。
    for index, row in enumerate(rows):
        # 逐行讲解：更新 total_rows：total_rows += 1。
        total_rows += 1
        # 逐行讲解：调用 normalize_row（normalize_row(index, row)），赋给 record。
        record = normalize_row(index, row)
        # 逐行讲解：检查条件 record is None；成立时执行下方分支。
        if record is None:
            # 逐行讲解：更新 invalid_rows：invalid_rows += 1。
            invalid_rows += 1
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue

        # 逐行讲解：调用 normalize_question_key（normalize_question_key(record['question'])），赋给 key。
        key = normalize_question_key(record["question"])
        # 逐行讲解：检查条件 key in conflicts；成立时执行下方分支。
        if key in conflicts:
            # 逐行讲解：更新 duplicate_rows：duplicate_rows += 1。
            duplicate_rows += 1
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue

        # 逐行讲解：查询指定键，缺失时使用默认值（unique.get(key)），赋给 previous。
        previous = unique.get(key)
        # 逐行讲解：检查条件 previous is None；成立时执行下方分支。
        if previous is None:
            # 逐行讲解：取值 record，赋给 unique[key]。
            unique[key] = record
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue

        # 逐行讲解：更新 duplicate_rows：duplicate_rows += 1。
        duplicate_rows += 1
        # 逐行讲解：检查条件 previous['answer'] != record['answer']；成立时执行下方分支。
        if previous["answer"] != record["answer"]:
            # 逐行讲解：调用 conflicts.add（conflicts.add(key)）。
            conflicts.add(key)
            # 逐行讲解：删除 unique[key]。
            del unique[key]
        # 逐行讲解：检查条件 record['id'] < previous['id']；成立时执行下方分支。
        elif record["id"] < previous["id"]:
            # 同题同答案时固定保留字典序最小的源 ID，避免依赖 parquet 行顺序。
            unique[key] = record

    # 逐行讲解：调用 sorted（sorted(unique.values(), key=lambda item: (normalize_question_key(item['question']), item['id']))），赋给 records。
    records = sorted(
        unique.values(),
        key=lambda item: (normalize_question_key(item["question"]), item["id"]),
    )
    # 逐行讲解：返回结果：(records, PreparationStats(total_rows=total_rows, invalid_rows=invalid_rows, duplicate_rows=duplicate_rows,…。
    return records, PreparationStats(
        total_rows=total_rows,
        invalid_rows=invalid_rows,
        duplicate_rows=duplicate_rows,
        conflicting_questions=len(conflicts),
    )


# 逐行讲解：定义函数 split_records：固定 seed 打乱后切分 train/dev，两个 split 不共享题面。
def split_records(
    records: list[dict[str, str]],
    *,
    dev_size: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """固定 seed 打乱后切分 train/dev，两个 split 不共享题面。"""
    # 逐行讲解：检查条件 dev_size < 1；成立时执行下方分支。
    if dev_size < 1:
        # 逐行讲解：抛出异常：ValueError('--dev-size must be >= 1')。
        raise ValueError("--dev-size must be >= 1")
    # 逐行讲解：检查条件 dev_size >= len(records)；成立时执行下方分支。
    if dev_size >= len(records):
        # 逐行讲解：抛出异常：ValueError('--dev-size must be smaller than the deduplicated dataset')。
        raise ValueError("--dev-size must be smaller than the deduplicated dataset")
    # 逐行讲解：调用 list（list(records)），赋给 shuffled。
    shuffled = list(records)
    # 逐行讲解：打乱数据顺序（random.Random(seed).shuffle(shuffled)）。
    random.Random(seed).shuffle(shuffled)
    # 逐行讲解：返回结果：(shuffled[dev_size:], shuffled[:dev_size])。
    return shuffled[dev_size:], shuffled[:dev_size]


# 逐行讲解：定义函数 write_jsonl：写入 UTF-8 JSONL。
def write_jsonl(records: Iterable[dict[str, str]], path: Path) -> None:
    """写入 UTF-8 JSONL。"""
    # 逐行讲解：创建目录（path.parent.mkdir(parents=True, exist_ok=True)）。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：进入上下文管理器 path.open('w', encoding='utf-8')，退出时执行其收尾逻辑。
    with path.open("w", encoding="utf-8") as file:
        # 逐行讲解：遍历 records，将每项绑定到 record。
        for record in records:
            # 逐行讲解：调用 file.write（file.write(json.dumps(record, ensure_ascii=False) + '\n')）。
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# 逐行讲解：定义函数 parse_args：解析数据准备参数。
def parse_args() -> argparse.Namespace:
    """解析数据准备参数。"""
    # 逐行讲解：取值 Path(__file__).resolve().parent，赋给 script_dir。
    script_dir = Path(__file__).resolve().parent
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--raw', type=Path, default=None, help='可选的本地 DAPO-Math parquet；默认从 Hugging Face Hub 读取')）。
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="可选的本地 DAPO-Math parquet；默认从 Hugging Face Hub 读取",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output-dir', type=Path, default=script_dir / 'datasets', help='训练集、dev 集和 AIME25 的保存…）。
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "datasets",
        help="训练集、dev 集和 AIME25 的保存目录。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dev-size', type=int, default=50, help='从去重后的训练题中切出的 dev 题目数。')）。
    parser.add_argument(
        "--dev-size",
        type=int,
        default=50,
        help="从去重后的训练题中切出的 dev 题目数。",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42, help='数据切分随机种子。')）。
    parser.add_argument("--seed", type=int, default=42, help="数据切分随机种子。")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--force-aime25', action='store_true', help='删除并重新下载已经存在的本地 AIME25。')）。
    parser.add_argument(
        "--force-aime25",
        action="store_true",
        help="删除并重新下载已经存在的本地 AIME25。",
    )
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 load_source：流式读取本地 parquet 或官方 Hub 数据，避免展开百万行到内存。
def load_source(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    """流式读取本地 parquet 或官方 Hub 数据，避免展开百万行到内存。"""
    # 逐行讲解：计算 args.output_dir / '.hf-cache'，赋给 cache_dir。
    cache_dir = args.output_dir / ".hf-cache"
    # 逐行讲解：检查条件 args.raw is not None；成立时执行下方分支。
    if args.raw is not None:
        # 逐行讲解：检查条件 not args.raw.is_file()；成立时执行下方分支。
        if not args.raw.is_file():
            # 逐行讲解：抛出异常：FileNotFoundError(f'找不到本地 DAPO-Math parquet：{args.raw}')。
            raise FileNotFoundError(f"找不到本地 DAPO-Math parquet：{args.raw}")
        # datasets 的 IterableDataset 会初始化 torch shared memory；数据准备不需要
        # PyTorch，直接按 Arrow record batch 读取更轻量，也不会复制一份 parquet cache。
        import pyarrow.parquet as parquet

        # 逐行讲解：定义函数 iter_parquet：封装下方逻辑，供调用方复用。
        def iter_parquet() -> Iterable[dict[str, Any]]:
            # 逐行讲解：构造列表 ['data_source', 'prompt', 'reward_model', 'extra_info']，赋给 columns。
            columns = [
                "data_source",
                "prompt",
                "reward_model",
                "extra_info",
            ]
            # 逐行讲解：遍历 parquet.ParquetFile(args.raw).iter_batches(batch_size=65536, columns=columns)，将每项绑定到 batch。
            for batch in parquet.ParquetFile(args.raw).iter_batches(
                batch_size=65536,
                columns=columns,
            ):
                # 逐行讲解：取值 (yield from batch.to_pylist())。
                yield from batch.to_pylist()

        # 逐行讲解：返回结果：iter_parquet()。
        return iter_parquet()
    # 逐行讲解：返回结果：load_dataset(DAPO_DATASET_ID, revision=DAPO_REVISION, split='train', cache_dir=str(cache_dir), streaming=Tr…。
    return load_dataset(
        DAPO_DATASET_ID,
        revision=DAPO_REVISION,
        split="train",
        cache_dir=str(cache_dir),
        streaming=True,
    )


# 逐行讲解：定义函数 as_dataset：把本地 AIME25 的 Dataset/单 split DatasetDict 统一为 Dataset。
def as_dataset(dataset: Dataset | DatasetDict) -> Dataset:
    """把本地 AIME25 的 Dataset/单 split DatasetDict 统一为 Dataset。"""
    # 逐行讲解：检查条件 isinstance(dataset, Dataset)；成立时执行下方分支。
    if isinstance(dataset, Dataset):
        # 逐行讲解：返回结果：dataset。
        return dataset
    # 逐行讲解：检查条件 'train' not in dataset；成立时执行下方分支。
    if "train" not in dataset:
        # 逐行讲解：抛出异常：ValueError(f'AIME25 缺少 train split：{list(dataset)}')。
        raise ValueError(f"AIME25 缺少 train split：{list(dataset)}")
    # 逐行讲解：返回结果：dataset['train']。
    return dataset["train"]


# 逐行讲解：定义函数 validate_aime25：校验 AIME25 的行数、字段和首尾样本。
def validate_aime25(dataset: Dataset, location: str) -> None:
    """校验 AIME25 的行数、字段和首尾样本。"""
    # 逐行讲解：调用 sorted（sorted(set(AIME25_REQUIRED_COLUMNS) - set(dataset.column_names))），赋给 missing。
    missing = sorted(
        set(AIME25_REQUIRED_COLUMNS) - set(dataset.column_names)
    )
    # 逐行讲解：检查条件 missing；成立时执行下方分支。
    if missing:
        # 逐行讲解：抛出异常：ValueError(f'{AIME25_DATASET_ID} 在 {location} 缺少字段 {missing}')。
        raise ValueError(
            f"{AIME25_DATASET_ID} 在 {location} 缺少字段 {missing}"
        )
    # 逐行讲解：检查条件 len(dataset) != AIME25_EXPECTED_ROWS；成立时执行下方分支。
    if len(dataset) != AIME25_EXPECTED_ROWS:
        # 逐行讲解：抛出异常：ValueError(f'{AIME25_DATASET_ID} 在 {location} 应有 {AIME25_EXPECTED_ROWS} 题，实际为 {len(dataset)} 题')。
        raise ValueError(
            f"{AIME25_DATASET_ID} 在 {location} 应有 "
            f"{AIME25_EXPECTED_ROWS} 题，实际为 {len(dataset)} 题"
        )
    # 逐行讲解：遍历 (0, len(dataset) - 1)，将每项绑定到 index。
    for index in (0, len(dataset) - 1):
        # 逐行讲解：通过推导式逐项生成结果：[column for column in AIME25_REQUIRED_COLUMNS if not str(dataset[index].get(column, '')).strip()]，赋给 empty。
        empty = [
            column
            for column in AIME25_REQUIRED_COLUMNS
            if not str(dataset[index].get(column, "")).strip()
        ]
        # 逐行讲解：检查条件 empty；成立时执行下方分支。
        if empty:
            # 逐行讲解：抛出异常：ValueError(f'AIME25 第 {index} 题存在空字段：{empty}')。
            raise ValueError(f"AIME25 第 {index} 题存在空字段：{empty}")


# 逐行讲解：定义函数 prepare_aime25：下载并校验本项目独立保存的固定版本 AIME25。
def prepare_aime25(
    output_dir: Path,
    *,
    force: bool,
) -> Path:
    """下载并校验本项目独立保存的固定版本 AIME25。"""
    # 逐行讲解：计算 output_dir / AIME25_OUTPUT_NAME，赋给 output_path。
    output_path = output_dir / AIME25_OUTPUT_NAME
    # 逐行讲解：检查条件 output_path.exists() and (not force)；成立时执行下方分支。
    if output_path.exists() and not force:
        # 逐行讲解：调用 as_dataset（as_dataset(load_from_disk(str(output_path)))），赋给 local。
        local = as_dataset(load_from_disk(str(output_path)))
        # 逐行讲解：调用 validate_aime25（validate_aime25(local, str(output_path))）。
        validate_aime25(local, str(output_path))
        # 逐行讲解：调用 print（print(f'[skip] AIME25: 本地数据已存在且校验通过 ({len(local)} rows) -> {output_path}')）。
        print(
            f"[skip] AIME25: 本地数据已存在且校验通过 "
            f"({len(local)} rows) -> {output_path}"
        )
        # 逐行讲解：返回结果：output_path。
        return output_path

    # 逐行讲解：检查条件 output_path.exists()；成立时执行下方分支。
    if output_path.exists():
        # 逐行讲解：调用 shutil.rmtree（shutil.rmtree(output_path)）。
        shutil.rmtree(output_path)

    # 逐行讲解：计算 output_dir / '.hf-cache'，赋给 cache_dir。
    cache_dir = output_dir / ".hf-cache"
    # 逐行讲解：创建目录（cache_dir.mkdir(parents=True, exist_ok=True)）。
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 print（print(f'[download] {AIME25_DATASET_ID}@{AIME25_REVISION} [train]')）。
    print(
        f"[download] {AIME25_DATASET_ID}@{AIME25_REVISION} [train]"
    )
    # 逐行讲解：调用 load_dataset（load_dataset(AIME25_DATASET_ID, revision=AIME25_REVISION, split='train', cache_dir=str(cache_dir))），赋给 dataset。
    dataset = load_dataset(
        AIME25_DATASET_ID,
        revision=AIME25_REVISION,
        split="train",
        cache_dir=str(cache_dir),
    )
    # 逐行讲解：检查条件 not isinstance(dataset, Dataset)；成立时执行下方分支。
    if not isinstance(dataset, Dataset):
        # 逐行讲解：抛出异常：TypeError(f'期望 Dataset，实际得到 {type(dataset)!r}')。
        raise TypeError(f"期望 Dataset，实际得到 {type(dataset)!r}")
    # 逐行讲解：调用 validate_aime25（validate_aime25(dataset, 'Hugging Face')）。
    validate_aime25(dataset, "Hugging Face")

    # 逐行讲解：创建目录（output_path.parent.mkdir(parents=True, exist_ok=True)）。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 dataset.save_to_disk（dataset.save_to_disk(str(output_path))）。
    dataset.save_to_disk(str(output_path))
    # 逐行讲解：调用 as_dataset（as_dataset(load_from_disk(str(output_path)))），赋给 saved。
    saved = as_dataset(load_from_disk(str(output_path)))
    # 逐行讲解：调用 validate_aime25（validate_aime25(saved, str(output_path))）。
    validate_aime25(saved, str(output_path))
    # 逐行讲解：调用 print（print(f'[done] AIME25: {len(saved)} rows -> {output_path}')）。
    print(f"[done] AIME25: {len(saved)} rows -> {output_path}")
    # 逐行讲解：返回结果：output_path。
    return output_path


# 逐行讲解：定义函数 main：生成训练/dev JSONL，并下载本项目自己的 AIME25。
def main() -> None:
    """生成训练/dev JSONL，并下载本项目自己的 AIME25。"""
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：创建目录（args.output_dir.mkdir(parents=True, exist_ok=True)）。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 prepare_aime25（prepare_aime25(args.output_dir, force=args.force_aime25)）。
    prepare_aime25(args.output_dir, force=args.force_aime25)

    # 逐行讲解：调用 deduplicate_rows（deduplicate_rows(load_source(args))），赋给 (records, stats)。
    records, stats = deduplicate_rows(load_source(args))
    # 逐行讲解：检查条件 not records；成立时执行下方分支。
    if not records:
        # 逐行讲解：抛出异常：ValueError('DAPO-Math 清洗去重后为空')。
        raise ValueError("DAPO-Math 清洗去重后为空")

    # 逐行讲解：调用 split_records（split_records(records, dev_size=args.dev_size, seed=args.seed)），赋给 (train_records, dev_records)。
    train_records, dev_records = split_records(
        records,
        dev_size=args.dev_size,
        seed=args.seed,
    )
    # 逐行讲解：调用 write_jsonl（write_jsonl(train_records, args.output_dir / 'train.jsonl')）。
    write_jsonl(train_records, args.output_dir / "train.jsonl")
    # 逐行讲解：调用 write_jsonl（write_jsonl(dev_records, args.output_dir / 'dev.jsonl')）。
    write_jsonl(dev_records, args.output_dir / "dev.jsonl")

    # 逐行讲解：调用 print（print(f'source rows: {stats.total_rows:,}')）。
    print(f"source rows: {stats.total_rows:,}")
    # 逐行讲解：调用 print（print(f'invalid rows: {stats.invalid_rows:,}')）。
    print(f"invalid rows: {stats.invalid_rows:,}")
    # 逐行讲解：调用 print（print(f'duplicate rows: {stats.duplicate_rows:,}')）。
    print(f"duplicate rows: {stats.duplicate_rows:,}")
    # 逐行讲解：调用 print（print(f'conflicting questions dropped: {stats.conflicting_questions:,}')）。
    print(f"conflicting questions dropped: {stats.conflicting_questions:,}")
    # 逐行讲解：调用 print（print(f'deduplicated questions: {len(records):,}')）。
    print(f"deduplicated questions: {len(records):,}")
    # 逐行讲解：调用 print（print(f'train: {len(train_records):,}')）。
    print(f"train: {len(train_records):,}")
    # 逐行讲解：调用 print（print(f'dev: {len(dev_records):,}')）。
    print(f"dev: {len(dev_records):,}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
