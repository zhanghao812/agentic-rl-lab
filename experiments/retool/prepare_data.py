"""下载并整理 ReTool 训练数据（DAPO-Math-17k）。

跟随官方 recipe 使用 BytedTsinghua-SIA/DAPO-Math-17k：
- question 取 prompt[0].content
- answer 取 reward_model.ground_truth
按固定 seed 打乱后切出 50 条 dev.jsonl（smoke test / 快速验证用，不参与训练），
其余写入 train.jsonl。

运行：
uv run python prepare_data.py
"""

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：random。
import random
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：从 datasets 导入 load_dataset。
from datasets import load_dataset


# 逐行讲解：取值 'BytedTsinghua-SIA/DAPO-Math-17k'，赋给 DATASET_ID。
DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"

# DAPO-Math-17k 的题目外面统一包了一层模板，要求用 "Answer: $Answer" 格式作答，
# 与我们 system prompt / reward 要求的 \boxed{} 格式直接冲突，清洗时剥掉。
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
# 逐行讲解：取值 '\n\nRemember to put your answer on its own line after "Answer:".'，赋给 PROMPT_SUFFIX。
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


# 逐行讲解：定义函数 strip_dapo_template：剥掉 DAPO 统一外层模板，只保留题目本体；模板不完整时原样保留。
def strip_dapo_template(question: str) -> str:
    """剥掉 DAPO 统一外层模板，只保留题目本体；模板不完整时原样保留。"""
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


# 逐行讲解：定义函数 normalize_row：把 verl 格式的原始样本转成简洁的训练格式。
def normalize_row(index: int, row: dict[str, Any]) -> dict[str, Any] | None:
    """把 verl 格式的原始样本转成简洁的训练格式。"""
    # 逐行讲解：查询指定键，缺失时使用默认值（row.get('prompt')），赋给 prompt。
    prompt = row.get("prompt")
    # 逐行讲解：检查条件 not isinstance(prompt, list) or not prompt；成立时执行下方分支。
    if not isinstance(prompt, list) or not prompt:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：调用 strip_dapo_template（strip_dapo_template(str(prompt[0].get('content') or ''))），赋给 question。
    question = strip_dapo_template(str(prompt[0].get("content") or ""))
    # 逐行讲解：取值 row.get('reward_model') or {}，赋给 reward_model。
    reward_model = row.get("reward_model") or {}
    # 逐行讲解：按条件选择结果：reward_model.get('ground_truth') if isinstance(reward_model, dict) else None，赋给 ground_truth。
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
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
    # 逐行讲解：返回结果：{'id': str(row.get('extra_info', {}).get('index') or index), 'question': question, 'answer': answer, 'data_…。
    return {
        "id": str(row.get("extra_info", {}).get("index") or index),
        "question": question,
        "answer": answer,
        "data_source": str(row.get("data_source") or "dapo_math"),
    }


# 逐行讲解：定义函数 write_jsonl：把处理后的样本写入 JSONL 文件。
def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """把处理后的样本写入 JSONL 文件。"""
    # 逐行讲解：进入上下文管理器 path.open('w', encoding='utf-8')，退出时执行其收尾逻辑。
    with path.open("w", encoding="utf-8") as file:
        # 逐行讲解：遍历 records，将每项绑定到 record。
        for record in records:
            # 逐行讲解：调用 file.write（file.write(json.dumps(record, ensure_ascii=False) + '\n')）。
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# 逐行讲解：定义函数 parse_args：解析数据准备命令行参数。
def parse_args() -> argparse.Namespace:
    """解析数据准备命令行参数。"""
    # 逐行讲解：取值 Path(__file__).resolve().parent，赋给 base_dir。
    base_dir = Path(__file__).resolve().parent
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--raw', type=Path, default=base_dir / 'datasets' / 'raw' / 'dapo-math-17k.parquet', he…）。
    parser.add_argument(
        "--raw",
        type=Path,
        default=base_dir / "datasets" / "raw" / "dapo-math-17k.parquet",
        help="已下载的原始 parquet；不存在时回退到从 Hub 加载",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output-dir', type=Path, default=base_dir / 'datasets')）。
    parser.add_argument("--output-dir", type=Path, default=base_dir / "datasets")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dev-size', type=int, default=50, help='切出多少条作为 dev 集')）。
    parser.add_argument("--dev-size", type=int, default=50, help="切出多少条作为 dev 集")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 main：下载 DAPO-Math-17k 并生成 train.jsonl 与 dev.jsonl。
def main() -> None:
    """下载 DAPO-Math-17k 并生成 train.jsonl 与 dev.jsonl。"""
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：创建目录（args.output_dir.mkdir(parents=True, exist_ok=True)）。
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 逐行讲解：检查条件 args.raw.exists()；成立时执行下方分支。
    if args.raw.exists():
        # 网络不稳定时先用 curl 下载到 datasets/raw/，再从本地 parquet 加载。
        dataset = load_dataset("parquet", data_files=str(args.raw), split="train")
    else:
        # 逐行讲解：调用 load_dataset（load_dataset(DATASET_ID, split='train')），赋给 dataset。
        dataset = load_dataset(DATASET_ID, split="train")
    # 逐行讲解：通过推导式逐项生成结果：[record for index, row in enumerate(dataset) if (record := normalize_row(index, row)) is not None]，赋给 records。
    records = [
        record
        for index, row in enumerate(dataset)
        if (record := normalize_row(index, row)) is not None
    ]
    # 逐行讲解：检查条件 not records；成立时执行下方分支。
    if not records:
        # 逐行讲解：抛出异常：ValueError('数据清洗后为空，请检查 DAPO-Math-17k 的字段结构')。
        raise ValueError("数据清洗后为空，请检查 DAPO-Math-17k 的字段结构")

    # 逐行讲解：打乱数据顺序（random.Random(args.seed).shuffle(records)）。
    random.Random(args.seed).shuffle(records)
    # 逐行讲解：按索引或切片读取 records[:args.dev_size]，赋给 dev_records。
    dev_records = records[: args.dev_size]
    # 逐行讲解：按索引或切片读取 records[args.dev_size:]，赋给 train_records。
    train_records = records[args.dev_size :]
    # 逐行讲解：调用 write_jsonl（write_jsonl(train_records, args.output_dir / 'train.jsonl')）。
    write_jsonl(train_records, args.output_dir / "train.jsonl")
    # 逐行讲解：调用 write_jsonl（write_jsonl(dev_records, args.output_dir / 'dev.jsonl')）。
    write_jsonl(dev_records, args.output_dir / "dev.jsonl")

    # 逐行讲解：调用 sorted（sorted((len(record['answer']) for record in records))），赋给 lengths。
    lengths = sorted(len(record["answer"]) for record in records)
    # 逐行讲解：调用 print（print(f'train: total={len(train_records)}')）。
    print(f"train: total={len(train_records)}")
    # 逐行讲解：调用 print（print(f'dev: total={len(dev_records)}')）。
    print(f"dev: total={len(dev_records)}")
    # 逐行讲解：调用 print（print(f'answer length: min={lengths[0]} median={lengths[len(lengths) // 2]} max={lengths[-1]}')）。
    print(
        "answer length: "
        f"min={lengths[0]} median={lengths[len(lengths) // 2]} max={lengths[-1]}"
    )
    # 逐行讲解：调用 print（print('sample:', json.dumps(train_records[0], ensure_ascii=False)[:300])）。
    print("sample:", json.dumps(train_records[0], ensure_ascii=False)[:300])


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
