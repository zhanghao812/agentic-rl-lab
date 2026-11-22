"""从 ModelScope 下载并整理 Search-R1 数据集。

运行：
uv run python prepare_data.py
"""

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：random。
import random
# 逐行讲解：从 collections 导入 Counter, defaultdict。
from collections import Counter, defaultdict
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：从 datasets 导入 load_dataset。
from datasets import load_dataset
# 逐行讲解：从 modelscope 导入 dataset_snapshot_download。
from modelscope import dataset_snapshot_download


# 逐行讲解：取值 'zhuangzhuang2023/nq_hotpotqa_train'，赋给 DATASET_ID。
DATASET_ID = "zhuangzhuang2023/nq_hotpotqa_train"
# 逐行讲解：取值 'aa2da0496c1b1a50a66af7acabdf09c07a0cb79e'，赋给 DATASET_REVISION。
DATASET_REVISION = "aa2da0496c1b1a50a66af7acabdf09c07a0cb79e"
# 逐行讲解：构造列表 ['id', 'question', 'golden_answers', 'data_source', 'reward_model']，赋给 RAW_COLUMNS。
RAW_COLUMNS = ["id", "question", "golden_answers", "data_source", "reward_model"]


# 逐行讲解：定义函数 download_dataset：从 ModelScope 下载固定版本的两个 Parquet 文件。
def download_dataset(raw_dir: Path) -> Path:
    """从 ModelScope 下载固定版本的两个 Parquet 文件。"""
    # 逐行讲解：创建目录（raw_dir.mkdir(parents=True, exist_ok=True)）。
    raw_dir.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 dataset_snapshot_download（dataset_snapshot_download(DATASET_ID, revision=DATASET_REVISION, local_dir=str(raw_dir), allow_patterns=['t…），赋给 path。
    path = dataset_snapshot_download(
        DATASET_ID,
        revision=DATASET_REVISION,
        local_dir=str(raw_dir),
        allow_patterns=["train.parquet", "test.parquet"],
    )
    # 逐行讲解：返回结果：Path(path)。
    return Path(path)


# 逐行讲解：定义函数 clean_answers：清理答案列表并保持原有顺序。
def clean_answers(values: Any) -> list[str]:
    """清理答案列表并保持原有顺序。"""
    # 逐行讲解：检查条件 isinstance(values, str)；成立时执行下方分支。
    if isinstance(values, str):
        # 逐行讲解：构造列表 [values]，赋给 values。
        values = [values]
    # 逐行讲解：检查条件 not isinstance(values, (list, tuple))；成立时执行下方分支。
    if not isinstance(values, (list, tuple)):
        # 逐行讲解：返回结果：[]。
        return []
    # 逐行讲解：声明 answers 的类型为 list[str]；构造列表 []。
    answers: list[str] = []
    # 逐行讲解：遍历 values，将每项绑定到 value。
    for value in values:
        # 逐行讲解：调用 str(value).strip（str(value).strip()），赋给 answer。
        answer = str(value).strip()
        # 逐行讲解：检查条件 answer and answer not in answers；成立时执行下方分支。
        if answer and answer not in answers:
            # 逐行讲解：向列表末尾追加一个元素（answers.append(answer)）。
            answers.append(answer)
    # 逐行讲解：返回结果：answers。
    return answers


# 逐行讲解：定义函数 extract_answers：优先读取 golden_answers，并兼容 reward_model 字段。
def extract_answers(row: dict[str, Any]) -> list[str]:
    """优先读取 golden_answers，并兼容 reward_model 字段。"""
    # 逐行讲解：调用 clean_answers（clean_answers(row.get('golden_answers'))），赋给 answers。
    answers = clean_answers(row.get("golden_answers"))
    # 逐行讲解：检查条件 answers；成立时执行下方分支。
    if answers:
        # 逐行讲解：返回结果：answers。
        return answers
    # 逐行讲解：取值 row.get('reward_model') or {}，赋给 reward_model。
    reward_model = row.get("reward_model") or {}
    # 逐行讲解：按条件选择结果：reward_model.get('ground_truth') if isinstance(reward_model, dict) else None，赋给 ground_truth。
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
    # 逐行讲解：检查条件 isinstance(ground_truth, dict)；成立时执行下方分支。
    if isinstance(ground_truth, dict):
        # 逐行讲解：查询指定键，缺失时使用默认值（ground_truth.get('target')），赋给 ground_truth。
        ground_truth = ground_truth.get("target")
    # 逐行讲解：返回结果：clean_answers(ground_truth)。
    return clean_answers(ground_truth)


# 逐行讲解：定义函数 normalize_row：把原始样本转成训练和评测共用的简洁格式。
def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """把原始样本转成训练和评测共用的简洁格式。"""
    # 逐行讲解：调用 str(row.get('question') or '').strip（str(row.get('question') or '').strip()），赋给 question。
    question = str(row.get("question") or "").strip()
    # 逐行讲解：调用 extract_answers（extract_answers(row)），赋给 answers。
    answers = extract_answers(row)
    # 逐行讲解：检查条件 not question or not answers；成立时执行下方分支。
    if not question or not answers:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：{'id': str(row.get('id') or ''), 'question': question, 'answers': answers, 'data_source': str(row.get('data…。
    return {
        "id": str(row.get("id") or ""),
        "question": question,
        "answers": answers,
        "data_source": str(row.get("data_source") or "unknown"),
    }


# 逐行讲解：定义函数 prepare_split：逐条清洗一个 Parquet split 并写成 JSONL。
def prepare_split(
    parquet_path: Path,
    output_path: Path,
    collect_records: bool = False,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """逐条清洗一个 Parquet split 并写成 JSONL。"""
    # 逐行讲解：调用 load_dataset（load_dataset('parquet', data_files=str(parquet_path), split='train', columns=RAW_COLUMNS)），赋给 dataset。
    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
        columns=RAW_COLUMNS,
    )
    # 逐行讲解：声明 records 的类型为 list[dict[str, Any]]；构造列表 []。
    records: list[dict[str, Any]] = []
    # 逐行讲解：声明 counts 的类型为 Counter[str]；调用 Counter（Counter()）。
    counts: Counter[str] = Counter()
    # 逐行讲解：进入上下文管理器 output_path.open('w', encoding='utf-8')，退出时执行其收尾逻辑。
    with output_path.open("w", encoding="utf-8") as file:
        # 逐行讲解：遍历 dataset，将每项绑定到 row。
        for row in dataset:
            # 逐行讲解：调用 normalize_row（normalize_row(row)），赋给 record。
            record = normalize_row(row)
            # 逐行讲解：检查条件 record is None；成立时执行下方分支。
            if record is None:
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：调用 file.write（file.write(json.dumps(record, ensure_ascii=False) + '\n')）。
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            # 逐行讲解：检查条件 collect_records；成立时执行下方分支。
            if collect_records:
                # 逐行讲解：向列表末尾追加一个元素（records.append(record)）。
                records.append(record)
            # 逐行讲解：更新 counts[record['data_source']]：counts[record['data_source']] += 1。
            counts[record["data_source"]] += 1
    # 逐行讲解：返回结果：(records, counts)。
    return records, counts

# 按来源等量抽样；默认从 7 个 benchmark 各抽 10 条，共 70 条
def select_dev(records: list[dict[str, Any]], per_source: int, seed: int) -> list[dict[str, Any]]:
    """从每个测试来源固定抽取一小份开发集。"""
    # 逐行讲解：声明 groups 的类型为 dict[str, list[dict[str, Any]]]；调用 defaultdict（defaultdict(list)）。
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # 逐行讲解：遍历 records，将每项绑定到 record。
    for record in records:
        # 逐行讲解：向列表末尾追加一个元素（groups[record['data_source']].append(record)）。
        groups[record["data_source"]].append(record)
    # 逐行讲解：调用 random.Random（random.Random(seed)），赋给 rng。
    rng = random.Random(seed)
    # 逐行讲解：声明 selected 的类型为 list[dict[str, Any]]；构造列表 []。
    selected: list[dict[str, Any]] = []
    # 逐行讲解：遍历 sorted(groups)，将每项绑定到 source。
    for source in sorted(groups):
        # 逐行讲解：打乱数据顺序（rng.shuffle(groups[source])）。
        rng.shuffle(groups[source])
        # 逐行讲解：把一组元素依次加入列表（selected.extend(groups[source][:per_source])）。
        selected.extend(groups[source][:per_source])
    # 逐行讲解：返回结果：selected。
    return selected


# 逐行讲解：定义函数 write_jsonl：把处理后的样本写入 JSONL 文件。
def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """把处理后的样本写入 JSONL 文件。"""
    # 逐行讲解：进入上下文管理器 path.open('w', encoding='utf-8')，退出时执行其收尾逻辑。
    with path.open("w", encoding="utf-8") as file:
        # 逐行讲解：遍历 records，将每项绑定到 record。
        for record in records:
            # 逐行讲解：调用 file.write（file.write(json.dumps(record, ensure_ascii=False) + '\n')）。
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# 逐行讲解：定义函数 print_counts：打印一个 split 的样本总量和来源分布。
def print_counts(name: str, counts: Counter[str]) -> None:
    """打印一个 split 的样本总量和来源分布。"""
    # 逐行讲解：调用 ', '.join（', '.join((f'{source}={count}' for source, count in sorted(counts.items())))），赋给 details。
    details = ", ".join(f"{source}={count}" for source, count in sorted(counts.items()))
    # 逐行讲解：调用 print（print(f'{name}: total={sum(counts.values())}; {details}')）。
    print(f"{name}: total={sum(counts.values())}; {details}")


# 逐行讲解：定义函数 parse_args：解析数据准备命令行参数。
def parse_args() -> argparse.Namespace:
    """解析数据准备命令行参数。"""
    # 逐行讲解：取值 Path(__file__).resolve().parent，赋给 base_dir。
    base_dir = Path(__file__).resolve().parent
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--raw-dir', type=Path, default=base_dir / 'datasets' / 'raw')）。
    parser.add_argument("--raw-dir", type=Path, default=base_dir / "datasets" / "raw")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output-dir', type=Path, default=base_dir / 'datasets')）。
    parser.add_argument("--output-dir", type=Path, default=base_dir / "datasets")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dev-per-source', type=int, default=10)）。
    parser.add_argument("--dev-per-source", type=int, default=10)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 main：下载原始数据并生成训练集、固定评测集和完整评测池。
def main() -> None:
    """下载原始数据并生成训练集、固定评测集和完整评测池。"""
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：创建目录（args.output_dir.mkdir(parents=True, exist_ok=True)）。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 download_dataset（download_dataset(args.raw_dir)），赋给 raw_dir。
    raw_dir = download_dataset(args.raw_dir)

    # 逐行讲解：调用 prepare_split（prepare_split(raw_dir / 'train.parquet', args.output_dir / 'train.jsonl')），赋给 (_, train_counts)。
    _, train_counts = prepare_split(raw_dir / "train.parquet", args.output_dir / "train.jsonl")
    # 逐行讲解：调用 prepare_split（prepare_split(raw_dir / 'test.parquet', args.output_dir / 'test.jsonl', collect_records=True)），赋给 (test_records, test_counts)。
    test_records, test_counts = prepare_split(
        raw_dir / "test.parquet",
        args.output_dir / "test.jsonl",
        collect_records=True,
    )
    # 逐行讲解：调用 select_dev（select_dev(test_records, args.dev_per_source, args.seed)），赋给 dev_records。
    dev_records = select_dev(test_records, args.dev_per_source, args.seed)
    # 逐行讲解：调用 write_jsonl（write_jsonl(dev_records, args.output_dir / 'dev.jsonl')）。
    write_jsonl(dev_records, args.output_dir / "dev.jsonl")

    # 逐行讲解：调用 print_counts（print_counts('train', train_counts)）。
    print_counts("train", train_counts)
    # 逐行讲解：调用 print_counts（print_counts('test', test_counts)）。
    print_counts("test", test_counts)
    # 逐行讲解：调用 print_counts（print_counts('dev', Counter((record['data_source'] for record in dev_records)))）。
    print_counts("dev", Counter(record["data_source"] for record in dev_records))


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
