"""下载作者发布的 Tournament-GRPO 数据，并按作者的 4096-token 规则筛长。

运行（仓库根目录）：
uv run python experiments/tournament-grpo/prepare_data.py
"""

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：urllib.request。
import urllib.request
# 逐行讲解：从 collections 导入 Counter。
from collections import Counter
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：导入模块：pyarrow.parquet，别名 pq。
import pyarrow.parquet as pq
# 逐行讲解：从 transformers 导入 AutoTokenizer。
from transformers import AutoTokenizer

# 逐行讲解：从 protocol 导入 MAX_PROMPT_LENGTH, TOOL_SCHEMAS, render_chat。
from protocol import MAX_PROMPT_LENGTH, TOOL_SCHEMAS, render_chat


# 固定作者仓库 commit，保证数据与论文代码一致。
SOURCE_REVISION = "eea826cf66705cac9ec81823f746d0ccd6074228"
# 逐行讲解：取值 f'https://raw.githubusercontent.com/PuffYang/Tournament_GRPO/{SOURCE_REVISION}'，赋给 SOURCE_URL。
SOURCE_URL = f"https://raw.githubusercontent.com/PuffYang/Tournament_GRPO/{SOURCE_REVISION}"
# 逐行讲解：构造字典 {'train': 'data/train_new_train_95.parquet', 'val': 'data/train_new_val_05.parquet'}，赋给 SPLIT_FILES。
SPLIT_FILES = {
    "train": "data/train_new_train_95.parquet",
    "val": "data/train_new_val_05.parquet",
}
# Deep Research Bench 评测资产，后续 eval.py 使用。
BENCH_FILES = {
    "query.jsonl": "deep_research_bench/data/prompt_data/query.jsonl",
    "criteria.jsonl": "deep_research_bench/data/criteria_data/criteria.jsonl",
    "reference.jsonl": "deep_research_bench/data/test_data/cleaned_data/reference.jsonl",
}


# 逐行讲解：定义函数 download：下载固定 commit 的文件；已存在时直接复用。
def download(remote: str, target: Path) -> Path:
    """下载固定 commit 的文件；已存在时直接复用。"""
    # 逐行讲解：检查条件 not target.exists()；成立时执行下方分支。
    if not target.exists():
        # 逐行讲解：创建目录（target.parent.mkdir(parents=True, exist_ok=True)）。
        target.parent.mkdir(parents=True, exist_ok=True)
        # 逐行讲解：进入上下文管理器 urllib.request.urlopen(f'{SOURCE_URL}/{remote}', timeout=60)，退出时执行其收尾逻辑。
        with urllib.request.urlopen(f"{SOURCE_URL}/{remote}", timeout=60) as response:
            # 逐行讲解：调用 target.write_bytes（target.write_bytes(response.read())）。
            target.write_bytes(response.read())
    # 逐行讲解：返回结果：target。
    return target


# 逐行讲解：定义函数 normalize_row：只保留训练需要的字段；作者的 query/rubrics 原文和顺序不做修改。
def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """只保留训练需要的字段；作者的 query/rubrics 原文和顺序不做修改。

    作者数据的 prompt 只有一条 user 消息，内容与 ground_truth.query 相同，所以只取 query。
    """
    # 逐行讲解：按索引或切片读取 row['reward_model']['ground_truth']，赋给 ground_truth。
    ground_truth = row["reward_model"]["ground_truth"]
    # 逐行讲解：返回结果：{'id': str(row['extra_info']['index']), 'query': ground_truth['query'], 'rubrics': ground_truth['rubrics'],…。
    return {
        "id": str(row["extra_info"]["index"]),
        "query": ground_truth["query"],
        "rubrics": ground_truth["rubrics"],
        "data_source": row["data_source"],
    }


# 逐行讲解：定义函数 prompt_length：复现 verl filter_overlong_prompts：原 user 消息 + 原生工具 schema，不含研究 system。
def prompt_length(tokenizer: Any, query: str) -> int:
    """复现 verl filter_overlong_prompts：原 user 消息 + 原生工具 schema，不含研究 system。"""
    # 逐行讲解：返回结果：len(render_chat(tokenizer, [{'role': 'user', 'content': query}], TOOL_SCHEMAS))。
    return len(render_chat(tokenizer, [{"role": "user", "content": query}], TOOL_SCHEMAS))


# 逐行讲解：定义函数 prepare_split：按原顺序筛长并写成 JSONL；重复题目照原数据保留。
def prepare_split(path: Path, output_path: Path, tokenizer: Any) -> tuple[Counter[str], list[str]]:
    """按原顺序筛长并写成 JSONL；重复题目照原数据保留。"""
    # 逐行讲解：声明 counts 的类型为 Counter[str]；调用 Counter（Counter()）。
    counts: Counter[str] = Counter()
    # 逐行讲解：声明 dropped 的类型为 list[str]；构造列表 []。
    dropped: list[str] = []
    # 逐行讲解：进入上下文管理器 output_path.open('w', encoding='utf-8')，退出时执行其收尾逻辑。
    with output_path.open("w", encoding="utf-8") as file:
        # 逐行讲解：遍历 pq.read_table(path).to_pylist()，将每项绑定到 row。
        for row in pq.read_table(path).to_pylist():
            # 逐行讲解：调用 normalize_row（normalize_row(row)），赋给 record。
            record = normalize_row(row)
            # 逐行讲解：检查条件 prompt_length(tokenizer, record['query']) > MAX_PROMPT_LENGTH；成立时执行下方分支。
            if prompt_length(tokenizer, record["query"]) > MAX_PROMPT_LENGTH:
                # 逐行讲解：向列表末尾追加一个元素（dropped.append(record['id'])）。
                dropped.append(record["id"])
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：调用 file.write（file.write(json.dumps(record, ensure_ascii=False) + '\n')）。
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            # 逐行讲解：更新 counts[record['data_source']]：counts[record['data_source']] += 1。
            counts[record["data_source"]] += 1
    # 逐行讲解：返回结果：(counts, dropped)。
    return counts, dropped


# 逐行讲解：定义函数 parse_args：解析数据准备命令行参数。
def parse_args() -> argparse.Namespace:
    """解析数据准备命令行参数。"""
    # 逐行讲解：取值 Path(__file__).resolve().parent，赋给 base_dir。
    base_dir = Path(__file__).resolve().parent
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output-dir', type=Path, default=base_dir / 'datasets')）。
    parser.add_argument("--output-dir", type=Path, default=base_dir / "datasets")
    # 筛长结果依赖 policy tokenizer，更换基座模型后需要重新运行。
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 main：下载原始数据，生成 train/val JSONL 和 Deep Research Bench 评测文件。
def main() -> None:
    """下载原始数据，生成 train/val JSONL 和 Deep Research Bench 评测文件。"""
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：调用 AutoTokenizer.from_pretrained（AutoTokenizer.from_pretrained(args.base_model)），赋给 tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # 逐行讲解：遍历 SPLIT_FILES.items()，将每项绑定到 (split, remote)。
    for split, remote in SPLIT_FILES.items():
        # 逐行讲解：调用 download（download(remote, args.output_dir / 'raw' / Path(remote).name)），赋给 raw_path。
        raw_path = download(remote, args.output_dir / "raw" / Path(remote).name)
        # 逐行讲解：调用 prepare_split（prepare_split(raw_path, args.output_dir / f'{split}.jsonl', tokenizer)），赋给 (counts, dropped)。
        counts, dropped = prepare_split(raw_path, args.output_dir / f"{split}.jsonl", tokenizer)
        # 逐行讲解：调用 ', '.join（', '.join((f'{source}={count}' for source, count in sorted(counts.items())))），赋给 details。
        details = ", ".join(f"{source}={count}" for source, count in sorted(counts.items()))
        # 逐行讲解：调用 print（print(f'{split}: total={sum(counts.values())}; {details}; overlong_dropped={dropped}')）。
        print(f"{split}: total={sum(counts.values())}; {details}; overlong_dropped={dropped}")
    # 逐行讲解：遍历 BENCH_FILES.items()，将每项绑定到 (name, remote)。
    for name, remote in BENCH_FILES.items():
        # 逐行讲解：调用 download（download(remote, args.output_dir / 'deep_research_bench' / name)）。
        download(remote, args.output_dir / "deep_research_bench" / name)
    # 逐行讲解：调用 print（print(f"deep_research_bench: {', '.join(BENCH_FILES)}")）。
    print(f"deep_research_bench: {', '.join(BENCH_FILES)}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
