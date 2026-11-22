"""下载并整理中文医疗 SFT / 医疗评测 / 通用 OPD 数据。

默认会准备三类数据：
1. FreedomIntelligence/medical-o1-reasoning-SFT zh：医疗 reasoning SFT。
2. bigbio/med_qa med_qa_zh_4options_source：中文 MedQA 四选一评测。
3. ceval/ceval-exam：精选 8 个中文通用能力 subset，构造 OPD train pool 和 held-out test。

脚本会先完整下载/缓存原始数据，再按固定 seed 导出训练和评测文件。

默认规则：
- medical SFT 使用全量训练数据。
- MedQA-zh 从完整 test 中抽 600 条做医疗 eval。
- C-Eval 精选 subset 按 80% / 20% 切分，80% 做 OPD，剩余部分抽 300 条做 test。

正式下载：
uv run python 00-download-dataset.py

小成本调试：
uv run python 00-download-dataset.py \
    --medical-sft-sample-size 100 \
    --medqa-sample-size 20 \
    --ceval-train-size 100 \
    --ceval-test-size 20
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：math。
import math
# 逐行讲解：导入模块：random。
import random
# 逐行讲解：导入模块：zipfile。
import zipfile
# 逐行讲解：从 collections.abc 导入 Iterable。
from collections.abc import Iterable
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：从 datasets 导入 Dataset, DatasetDict, load_dataset。
from datasets import Dataset, DatasetDict, load_dataset
# 逐行讲解：从 huggingface_hub 导入 hf_hub_download。
from huggingface_hub import hf_hub_download


# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets'，赋给 DEFAULT_DATA_DIR。
DEFAULT_DATA_DIR = SCRIPT_DIR / "datasets"

# 逐行讲解：取值 'FreedomIntelligence/medical-o1-reasoning-SFT'，赋给 MEDICAL_SFT_REPO。
MEDICAL_SFT_REPO = "FreedomIntelligence/medical-o1-reasoning-SFT"
# 逐行讲解：取值 'zh'，赋给 MEDICAL_SFT_CONFIG。
MEDICAL_SFT_CONFIG = "zh"

# 逐行讲解：取值 'bigbio/med_qa'，赋给 MEDQA_REPO。
MEDQA_REPO = "bigbio/med_qa"
# 逐行讲解：取值 'med_qa_zh_4options_source'，赋给 MEDQA_CONFIG。
MEDQA_CONFIG = "med_qa_zh_4options_source"

# 逐行讲解：取值 'ceval/ceval-exam'，赋给 CEVAL_REPO。
CEVAL_REPO = "ceval/ceval-exam"
# 逐行讲解：构造集合 {'basic_medicine', 'clinical_medicine', 'physician', 'veterinary_medicine', 'high_school_biology', 'middle_…，赋给 CEVAL_EXCLUDED_CONFIGS。
CEVAL_EXCLUDED_CONFIGS = {
    "basic_medicine",
    "clinical_medicine",
    "physician",
    "veterinary_medicine",
    "high_school_biology",
    "middle_school_biology",
}

# 逐行讲解：构造列表 ['computer_network', 'college_programming', 'advanced_mathematics', 'discrete_mathematics', 'college_physic…，赋给 CEVAL_DEFAULT_CONFIGS。
CEVAL_DEFAULT_CONFIGS = [
    "computer_network",
    "college_programming",
    "advanced_mathematics",
    "discrete_mathematics",
    "college_physics",
    "logic",
    "chinese_language_and_literature",
    "college_economics",
]

# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='下载中文医疗 SFT、MedQA-zh eval 和 C-Eval OPD/eval 数据')），赋给 parser。
    parser = argparse.ArgumentParser(description="下载中文医疗 SFT、MedQA-zh eval 和 C-Eval OPD/eval 数据")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR, help='数据输出目录')）。
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="数据输出目录")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--cache-dir', type=Path, default=None, help='HuggingFace datasets cache 目录')）。
    parser.add_argument("--cache-dir", type=Path, default=None, help="HuggingFace datasets cache 目录")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42, help='固定抽样和切分使用的随机种子')）。
    parser.add_argument("--seed", type=int, default=42, help="固定抽样和切分使用的随机种子")

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--skip-medical-sft', action='store_true', help='跳过 medical-o1 reasoning SFT 数据')）。
    parser.add_argument("--skip-medical-sft", action="store_true", help="跳过 medical-o1 reasoning SFT 数据")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--skip-medqa', action='store_true', help='跳过 MedQA-zh 评测数据')）。
    parser.add_argument("--skip-medqa", action="store_true", help="跳过 MedQA-zh 评测数据")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--skip-ceval', action='store_true', help='跳过 C-Eval 通用数据')）。
    parser.add_argument("--skip-ceval", action="store_true", help="跳过 C-Eval 通用数据")

    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--medical-sft-sample-size', type=int, default=0, help='调试用医疗 SFT sample 数量；<=0 表示不单独导出…）。
    parser.add_argument("--medical-sft-sample-size", type=int, default=0, help="调试用医疗 SFT sample 数量；<=0 表示不单独导出 sample，SFT 默认使用全量 train.jsonl")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--medqa-sample-size', type=int, default=600, help='从完整 MedQA-zh test 中导出的 eval sample …）。
    parser.add_argument("--medqa-sample-size", type=int, default=600, help="从完整 MedQA-zh test 中导出的 eval sample 数量；<=0 表示全量")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--ceval-test-size', '--ceval-eval-size', dest='ceval_test_size', type=int, default=300…）。
    parser.add_argument("--ceval-test-size", "--ceval-eval-size", dest="ceval_test_size", type=int, default=300, help="从 C-Eval held-out test pool 中导出的 test sample 数量")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--ceval-train-ratio', type=float, default=0.8, help='C-Eval 数据中用于 OPD train pool 的比例')）。
    parser.add_argument("--ceval-train-ratio", type=float, default=0.8, help="C-Eval 数据中用于 OPD train pool 的比例")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--ceval-train-size', type=int, default=0, help='从 C-Eval OPD train pool 中导出的 OPD train…）。
    parser.add_argument("--ceval-train-size", type=int, default=0, help="从 C-Eval OPD train pool 中导出的 OPD train 数量；<=0 表示使用完整 train pool")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--ceval-splits', nargs='+', default=['dev', 'val', 'test'], help='参与 C-Eval 80/20 切分的 …）。
    parser.add_argument(
        "--ceval-splits",
        nargs="+",
        default=["dev", "val", "test"],
        help="参与 C-Eval 80/20 切分的 split",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--ceval-configs', nargs='*', default=None, help='指定 C-Eval config；不传表示使用默认精选 8 个 subse…）。
    parser.add_argument("--ceval-configs", nargs="*", default=None, help="指定 C-Eval config；不传表示使用默认精选 8 个 subset")
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()
    # 逐行讲解：检查条件 not 0 < args.ceval_train_ratio < 1；成立时执行下方分支。
    if not 0 < args.ceval_train_ratio < 1:
        # 逐行讲解：抛出异常：ValueError('--ceval-train-ratio must be between 0 and 1')。
        raise ValueError("--ceval-train-ratio must be between 0 and 1")
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 cache_dir：封装下方逻辑，供调用方复用。
def cache_dir(args: argparse.Namespace) -> str:
    # 逐行讲解：取值 args.cache_dir or args.data_dir / '.hf_cache'，赋给 path。
    path = args.cache_dir or (args.data_dir / ".hf_cache")
    # 逐行讲解：创建目录（path.mkdir(parents=True, exist_ok=True)）。
    path.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：返回结果：str(path)。
    return str(path)


# 逐行讲解：定义函数 write_jsonl：封装下方逻辑，供调用方复用。
def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    # 逐行讲解：创建目录（path.parent.mkdir(parents=True, exist_ok=True)）。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：取值 0，赋给 count。
    count = 0
    # 逐行讲解：进入上下文管理器 path.open('w', encoding='utf-8')，退出时执行其收尾逻辑。
    with path.open("w", encoding="utf-8") as f:
        # 逐行讲解：遍历 rows，将每项绑定到 row。
        for row in rows:
            # 逐行讲解：调用 f.write（f.write(json.dumps(row, ensure_ascii=False) + '\n')）。
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            # 逐行讲解：更新 count：count += 1。
            count += 1
    # 逐行讲解：返回结果：count。
    return count


# 逐行讲解：定义函数 write_json：封装下方逻辑，供调用方复用。
def write_json(path: Path, data: dict[str, Any]) -> None:
    # 逐行讲解：创建目录（path.parent.mkdir(parents=True, exist_ok=True)）。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：将文本写入文件（path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')）。
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 逐行讲解：定义函数 shuffled_rows：封装下方逻辑，供调用方复用。
def shuffled_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    # 逐行讲解：调用 list（list(rows)），赋给 rows。
    rows = list(rows)
    # 逐行讲解：打乱数据顺序（random.Random(seed).shuffle(rows)）。
    random.Random(seed).shuffle(rows)
    # 逐行讲解：返回结果：rows。
    return rows


# 逐行讲解：定义函数 select_size：封装下方逻辑，供调用方复用。
def select_size(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    # 逐行讲解：检查条件 size <= 0；成立时执行下方分支。
    if size <= 0:
        # 逐行讲解：返回结果：rows。
        return rows
    # 逐行讲解：返回结果：rows[:min(size, len(rows))]。
    return rows[: min(size, len(rows))]


# 逐行讲解：定义函数 first_text：封装下方逻辑，供调用方复用。
def first_text(row: dict[str, Any], names: tuple[str, ...]) -> str:
    # 逐行讲解：遍历 names，将每项绑定到 name。
    for name in names:
        # 逐行讲解：查询指定键，缺失时使用默认值（row.get(name)），赋给 value。
        value = row.get(name)
        # 逐行讲解：检查条件 value is not None and str(value).strip()；成立时执行下方分支。
        if value is not None and str(value).strip():
            # 逐行讲解：返回结果：str(value).strip()。
            return str(value).strip()
    # 逐行讲解：返回结果：''。
    return ""


# 逐行讲解：定义函数 medical_sft_completion：封装下方逻辑，供调用方复用。
def medical_sft_completion(complex_cot: str, response: str) -> str:
    # 数据文件保留原始 reasoning + answer；模型专用的 <think> 标签在 SFT 脚本里构造。
    parts = [part.strip() for part in (complex_cot, response) if part and part.strip()]
    # 逐行讲解：返回结果：'\n\n'.join(parts)。
    return "\n\n".join(parts)


# 逐行讲解：定义函数 normalize_medical_sft_row：封装下方逻辑，供调用方复用。
def normalize_medical_sft_row(row: dict[str, Any]) -> dict[str, Any] | None:
    # 逐行讲解：调用 first_text（first_text(row, ('Question', 'question', 'instruction', 'prompt'))），赋给 question。
    question = first_text(row, ("Question", "question", "instruction", "prompt"))
    # 逐行讲解：调用 first_text（first_text(row, ('Complex_CoT', 'complex_cot', 'cot', 'reasoning'))），赋给 complex_cot。
    complex_cot = first_text(row, ("Complex_CoT", "complex_cot", "cot", "reasoning"))
    # 逐行讲解：调用 first_text（first_text(row, ('Response', 'response', 'answer', 'output'))），赋给 response。
    response = first_text(row, ("Response", "response", "answer", "output"))
    # 逐行讲解：调用 medical_sft_completion（medical_sft_completion(complex_cot, response)），赋给 completion。
    completion = medical_sft_completion(complex_cot, response)
    # 逐行讲解：检查条件 not question or not completion；成立时执行下方分支。
    if not question or not completion:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：{'question': question, 'complex_cot': complex_cot, 'response': response, 'completion': completion, 'message…。
    return {
        "question": question,
        "complex_cot": complex_cot,
        "response": response,
        "completion": completion,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": completion},
        ],
    }


# 逐行讲解：定义函数 prepare_medical_sft：封装下方逻辑，供调用方复用。
def prepare_medical_sft(args: argparse.Namespace, hf_cache_dir: str) -> None:
    # 逐行讲解：调用 print（print(f'Loading {MEDICAL_SFT_REPO}/{MEDICAL_SFT_CONFIG} ...')）。
    print(f"Loading {MEDICAL_SFT_REPO}/{MEDICAL_SFT_CONFIG} ...")
    # 逐行讲解：调用 load_dataset（load_dataset(MEDICAL_SFT_REPO, MEDICAL_SFT_CONFIG, cache_dir=hf_cache_dir)），赋给 data。
    data = load_dataset(MEDICAL_SFT_REPO, MEDICAL_SFT_CONFIG, cache_dir=hf_cache_dir)
    # 逐行讲解：检查条件 not isinstance(data, DatasetDict)；成立时执行下方分支。
    if not isinstance(data, DatasetDict):
        # 逐行讲解：调用 DatasetDict（DatasetDict({'train': data})），赋给 data。
        data = DatasetDict({"train": data})

    # 逐行讲解：计算 args.data_dir / 'medical-o1-reasoning-SFT-zh'，赋给 out_dir。
    out_dir = args.data_dir / "medical-o1-reasoning-SFT-zh"
    # 逐行讲解：遍历 data.items()，将每项绑定到 (split, dataset)。
    for split, dataset in data.items():
        # 逐行讲解：构造列表 []，赋给 rows。
        rows = []
        # 逐行讲解：遍历 dataset，将每项绑定到 row。
        for row in dataset:
            # 逐行讲解：调用 normalize_medical_sft_row（normalize_medical_sft_row(dict(row))），赋给 normalized。
            normalized = normalize_medical_sft_row(dict(row))
            # 逐行讲解：检查条件 normalized is not None；成立时执行下方分支。
            if normalized is not None:
                # 逐行讲解：取值 split，赋给 normalized['source_split']。
                normalized["source_split"] = split
                # 逐行讲解：向列表末尾追加一个元素（rows.append(normalized)）。
                rows.append(normalized)
        # 逐行讲解：调用 shuffled_rows（shuffled_rows(rows, args.seed)），赋给 rows。
        rows = shuffled_rows(rows, args.seed)
        # 逐行讲解：计算 out_dir / f'{split}_all.jsonl'，赋给 all_path。
        all_path = out_dir / f"{split}_all.jsonl"
        # 逐行讲解：调用 write_jsonl（write_jsonl(all_path, rows)），赋给 all_count。
        all_count = write_jsonl(all_path, rows)
        # 逐行讲解：调用 print（print(f'Saved medical SFT {split} all: {all_count} rows -> {all_path}')）。
        print(f"Saved medical SFT {split} all: {all_count} rows -> {all_path}")

        # 逐行讲解：检查条件 split == 'train'；成立时执行下方分支。
        if split == "train":
            # 逐行讲解：计算 out_dir / 'train.jsonl'，赋给 train_path。
            train_path = out_dir / "train.jsonl"
            # 逐行讲解：调用 write_jsonl（write_jsonl(train_path, rows)），赋给 train_count。
            train_count = write_jsonl(train_path, rows)
            # 逐行讲解：调用 print（print(f'Saved medical SFT train: {train_count} rows -> {train_path}')）。
            print(f"Saved medical SFT train: {train_count} rows -> {train_path}")
            # 逐行讲解：检查条件 args.medical_sft_sample_size > 0；成立时执行下方分支。
            if args.medical_sft_sample_size > 0:
                # 逐行讲解：调用 select_size（select_size(rows, args.medical_sft_sample_size)），赋给 sample_rows。
                sample_rows = select_size(rows, args.medical_sft_sample_size)
                # 逐行讲解：计算 out_dir / 'train_sample.jsonl'，赋给 sample_path。
                sample_path = out_dir / "train_sample.jsonl"
                # 逐行讲解：调用 write_jsonl（write_jsonl(sample_path, sample_rows)），赋给 sample_count。
                sample_count = write_jsonl(sample_path, sample_rows)
                # 逐行讲解：调用 print（print(f'Saved medical SFT train sample: {sample_count} rows -> {sample_path}')）。
                print(f"Saved medical SFT train sample: {sample_count} rows -> {sample_path}")


# 逐行讲解：定义函数 option_dict_from_any：封装下方逻辑，供调用方复用。
def option_dict_from_any(options: Any) -> dict[str, str]:
    # 逐行讲解：检查条件 isinstance(options, dict)；成立时执行下方分支。
    if isinstance(options, dict):
        # 逐行讲解：返回结果：{str(k).upper(): str(v).strip() for k, v in options.items() if str(v).strip()}。
        return {str(k).upper(): str(v).strip() for k, v in options.items() if str(v).strip()}
    # 逐行讲解：检查条件 isinstance(options, list)；成立时执行下方分支。
    if isinstance(options, list):
        # 逐行讲解：声明 result 的类型为 dict[str, str]；构造字典 {}。
        result: dict[str, str] = {}
        # 逐行讲解：遍历 enumerate(options)，将每项绑定到 (i, item)。
        for i, item in enumerate(options):
            # 逐行讲解：调用 chr（chr(ord('A') + i)），赋给 default_key。
            default_key = chr(ord("A") + i)
            # 逐行讲解：检查条件 isinstance(item, dict)；成立时执行下方分支。
            if isinstance(item, dict):
                # 逐行讲解：取值 item.get('key') or item.get('label') or item.get('id') or item.get('name') or default_key，赋给 key。
                key = item.get("key") or item.get("label") or item.get("id") or item.get("name") or default_key
                # 逐行讲解：取值 item.get('value') or item.get('text') or item.get('option') or item.get('content')，赋给 value。
                value = item.get("value") or item.get("text") or item.get("option") or item.get("content")
            else:
                # 逐行讲解：取值 default_key，赋给 key。
                key = default_key
                # 逐行讲解：取值 item，赋给 value。
                value = item
            # 逐行讲解：检查条件 value is not None and str(value).strip()；成立时执行下方分支。
            if value is not None and str(value).strip():
                # 逐行讲解：调用 str(value).strip（str(value).strip()），赋给 result[str(key).upper()]。
                result[str(key).upper()] = str(value).strip()
        # 逐行讲解：返回结果：result。
        return result
    # 逐行讲解：返回结果：{}。
    return {}


# 逐行讲解：定义函数 option_dict_from_abcd：封装下方逻辑，供调用方复用。
def option_dict_from_abcd(row: dict[str, Any]) -> dict[str, str]:
    # 逐行讲解：构造字典 {}，赋给 options。
    options = {}
    # 逐行讲解：遍历 ('A', 'B', 'C', 'D')，将每项绑定到 key。
    for key in ("A", "B", "C", "D"):
        # 逐行讲解：查询指定键，缺失时使用默认值（row.get(key)），赋给 value。
        value = row.get(key)
        # 逐行讲解：检查条件 value is not None and str(value).strip()；成立时执行下方分支。
        if value is not None and str(value).strip():
            # 逐行讲解：调用 str(value).strip（str(value).strip()），赋给 options[key]。
            options[key] = str(value).strip()
    # 逐行讲解：返回结果：options。
    return options


# 逐行讲解：定义函数 normalize_answer_idx：封装下方逻辑，供调用方复用。
def normalize_answer_idx(answer_idx: Any, options: dict[str, str], answer: str) -> str:
    # 逐行讲解：检查条件 answer_idx is not None and str(answer_idx).strip()；成立时执行下方分支。
    if answer_idx is not None and str(answer_idx).strip():
        # 逐行讲解：调用 str(answer_idx).strip().upper（str(answer_idx).strip().upper()），赋给 value。
        value = str(answer_idx).strip().upper()
        # 逐行讲解：检查条件 value in options；成立时执行下方分支。
        if value in options:
            # 逐行讲解：返回结果：value。
            return value
    # 逐行讲解：遍历 options.items()，将每项绑定到 (key, value)。
    for key, value in options.items():
        # 逐行讲解：检查条件 answer and value.strip() == answer.strip()；成立时执行下方分支。
        if answer and value.strip() == answer.strip():
            # 逐行讲解：返回结果：key。
            return key
    # 逐行讲解：返回结果：''。
    return ""


# 逐行讲解：定义函数 choice_prompt：封装下方逻辑，供调用方复用。
def choice_prompt(question: str, options: dict[str, str]) -> str:
    # 逐行讲解：通过推导式逐项生成结果：[f'{key}. {options[key]}' for key in ('A', 'B', 'C', 'D') if key in options]，赋给 option_lines。
    option_lines = [f"{key}. {options[key]}" for key in ("A", "B", "C", "D") if key in options]
    # 逐行讲解：返回结果：'\n'.join(['以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。', '', f'题目：{question}', *option_lines, '', '答案：'])。
    return "\n".join(
        [
            "以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。",
            "",
            f"题目：{question}",
            *option_lines,
            "",
            "答案：",
        ]
    )


# 逐行讲解：定义函数 normalize_medqa_row：封装下方逻辑，供调用方复用。
def normalize_medqa_row(row: dict[str, Any]) -> dict[str, Any] | None:
    # 逐行讲解：调用 first_text（first_text(row, ('question', 'Question'))），赋给 question。
    question = first_text(row, ("question", "Question"))
    # 逐行讲解：取值 option_dict_from_any(row.get('options')) or option_dict_from_abcd(row)，赋给 options。
    options = option_dict_from_any(row.get("options")) or option_dict_from_abcd(row)
    # 逐行讲解：调用 first_text（first_text(row, ('answer', 'Answer'))），赋给 answer。
    answer = first_text(row, ("answer", "Answer"))
    # 逐行讲解：调用 normalize_answer_idx（normalize_answer_idx(row.get('answer_idx') or row.get('answer_index'), options, answer)），赋给 answer_idx。
    answer_idx = normalize_answer_idx(row.get("answer_idx") or row.get("answer_index"), options, answer)
    # 逐行讲解：检查条件 not question or len(options) < 4 or (not answer_idx)；成立时执行下方分支。
    if not question or len(options) < 4 or not answer_idx:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：{'question': question, 'options': {key: options[key] for key in ('A', 'B', 'C', 'D') if key in options}, 'a…。
    return {
        "question": question,
        "options": {key: options[key] for key in ("A", "B", "C", "D") if key in options},
        "answer": answer,
        "answer_idx": answer_idx,
        "prompt": choice_prompt(question, options),
    }


# 逐行讲解：定义函数 prepare_medqa：封装下方逻辑，供调用方复用。
def prepare_medqa(args: argparse.Namespace, hf_cache_dir: str) -> None:
    # 逐行讲解：调用 print（print(f'Loading {MEDQA_REPO}/{MEDQA_CONFIG} test ...')）。
    print(f"Loading {MEDQA_REPO}/{MEDQA_CONFIG} test ...")
    # bigbio/med_qa 仍保留旧版 datasets loading script；新版 datasets 不再执行这类脚本。
    # 这里直接下载仓库里的 data_clean.zip，再读取中文 Mainland/4_options/test.jsonl。
    zip_path = hf_hub_download(
        repo_id=MEDQA_REPO,
        filename="data_clean.zip",
        repo_type="dataset",
        cache_dir=hf_cache_dir,
    )
    # 逐行讲解：取值 'data_clean/questions/Mainland/4_options/test.jsonl'，赋给 target_suffix。
    target_suffix = "data_clean/questions/Mainland/4_options/test.jsonl"
    # 逐行讲解：构造列表 []，赋给 rows。
    rows = []
    # 逐行讲解：进入上下文管理器 zipfile.ZipFile(zip_path)，退出时执行其收尾逻辑。
    with zipfile.ZipFile(zip_path) as zf:
        # 逐行讲解：通过推导式逐项生成结果：[name for name in zf.namelist() if name.endswith(target_suffix)]，赋给 matched_names。
        matched_names = [name for name in zf.namelist() if name.endswith(target_suffix)]
        # 逐行讲解：检查条件 not matched_names；成立时执行下方分支。
        if not matched_names:
            # 逐行讲解：抛出异常：FileNotFoundError(f'Cannot find {target_suffix} in {zip_path}')。
            raise FileNotFoundError(f"Cannot find {target_suffix} in {zip_path}")
        # 逐行讲解：进入上下文管理器 zf.open(matched_names[0])，退出时执行其收尾逻辑。
        with zf.open(matched_names[0]) as f:
            # 逐行讲解：遍历 f，将每项绑定到 line。
            for line in f:
                # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(line.decode('utf-8'))），赋给 row。
                row = json.loads(line.decode("utf-8"))
                # 逐行讲解：调用 normalize_medqa_row（normalize_medqa_row(row)），赋给 normalized。
                normalized = normalize_medqa_row(row)
                # 逐行讲解：检查条件 normalized is not None；成立时执行下方分支。
                if normalized is not None:
                    # 逐行讲解：向列表末尾追加一个元素（rows.append(normalized)）。
                    rows.append(normalized)
    # 逐行讲解：调用 shuffled_rows（shuffled_rows(rows, args.seed)），赋给 rows。
    rows = shuffled_rows(rows, args.seed)

    # 逐行讲解：计算 args.data_dir / 'medqa-zh-4options'，赋给 out_dir。
    out_dir = args.data_dir / "medqa-zh-4options"
    # 逐行讲解：计算 out_dir / 'test_all.jsonl'，赋给 all_path。
    all_path = out_dir / "test_all.jsonl"
    # 逐行讲解：调用 write_jsonl（write_jsonl(all_path, rows)），赋给 all_count。
    all_count = write_jsonl(all_path, rows)
    # 逐行讲解：调用 print（print(f'Saved MedQA-zh test all: {all_count} rows -> {all_path}')）。
    print(f"Saved MedQA-zh test all: {all_count} rows -> {all_path}")

    # 逐行讲解：调用 select_size（select_size(rows, args.medqa_sample_size)），赋给 sample_rows。
    sample_rows = select_size(rows, args.medqa_sample_size)
    # 逐行讲解：计算 out_dir / 'test_sample.jsonl'，赋给 sample_path。
    sample_path = out_dir / "test_sample.jsonl"
    # 逐行讲解：调用 write_jsonl（write_jsonl(sample_path, sample_rows)），赋给 sample_count。
    sample_count = write_jsonl(sample_path, sample_rows)
    # 逐行讲解：调用 print（print(f'Saved MedQA-zh eval sample: {sample_count} rows -> {sample_path}')）。
    print(f"Saved MedQA-zh eval sample: {sample_count} rows -> {sample_path}")


# 逐行讲解：定义函数 ceval_configs：封装下方逻辑，供调用方复用。
def ceval_configs(args: argparse.Namespace, hf_cache_dir: str) -> list[str]:
    # 逐行讲解：检查条件 args.ceval_configs；成立时执行下方分支。
    if args.ceval_configs:
        # 逐行讲解：调用 list（list(args.ceval_configs)），赋给 configs。
        configs = list(args.ceval_configs)
    else:
        # 逐行讲解：取值 CEVAL_DEFAULT_CONFIGS，赋给 configs。
        configs = CEVAL_DEFAULT_CONFIGS
    # 逐行讲解：返回结果：[name for name in configs if name not in CEVAL_EXCLUDED_CONFIGS]。
    return [name for name in configs if name not in CEVAL_EXCLUDED_CONFIGS]


# 逐行讲解：定义函数 normalize_ceval_row：封装下方逻辑，供调用方复用。
def normalize_ceval_row(row: dict[str, Any], subject: str, split: str, row_index: int) -> dict[str, Any] | None:
    # 逐行讲解：调用 first_text（first_text(row, ('question', 'Question'))），赋给 question。
    question = first_text(row, ("question", "Question"))
    # 逐行讲解：调用 option_dict_from_abcd（option_dict_from_abcd(row)），赋给 options。
    options = option_dict_from_abcd(row)
    # 逐行讲解：调用 first_text（first_text(row, ('answer', 'Answer'))），赋给 answer。
    answer = first_text(row, ("answer", "Answer"))
    # 逐行讲解：调用 normalize_answer_idx（normalize_answer_idx(answer, options, answer='')），赋给 answer_idx。
    answer_idx = normalize_answer_idx(answer, options, answer="")
    # 逐行讲解：检查条件 not question or len(options) < 4；成立时执行下方分支。
    if not question or len(options) < 4:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：返回结果：{'row_id': f'{subject}:{split}:{row_index}', 'subject': subject, 'source_split': split, 'question': questio…。
    return {
        "row_id": f"{subject}:{split}:{row_index}",
        "subject": subject,
        "source_split": split,
        "question": question,
        "options": options,
        "answer_idx": answer_idx,
        "prompt": choice_prompt(question, options),
    }


# 逐行讲解：定义函数 load_ceval_split：封装下方逻辑，供调用方复用。
def load_ceval_split(config: str, split: str, hf_cache_dir: str) -> Dataset | None:
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：返回结果：load_dataset(CEVAL_REPO, config, split=split, cache_dir=hf_cache_dir)。
        return load_dataset(CEVAL_REPO, config, split=split, cache_dir=hf_cache_dir)
    except Exception as exc:  # noqa: BLE001
        # 逐行讲解：调用 print（print(f'Skip C-Eval {config}/{split}: {exc}')）。
        print(f"Skip C-Eval {config}/{split}: {exc}")
        # 逐行讲解：返回结果：None。
        return None


# 逐行讲解：定义函数 prepare_ceval：封装下方逻辑，供调用方复用。
def prepare_ceval(args: argparse.Namespace, hf_cache_dir: str) -> None:
    # 逐行讲解：调用 ceval_configs（ceval_configs(args, hf_cache_dir)），赋给 configs。
    configs = ceval_configs(args, hf_cache_dir)
    # 逐行讲解：调用 print（print(f'Loading C-Eval non-med configs: {len(configs)} subjects')）。
    print(f"Loading C-Eval non-med configs: {len(configs)} subjects")

    # 逐行讲解：声明 all_rows 的类型为 list[dict[str, Any]]；构造列表 []。
    all_rows: list[dict[str, Any]] = []
    # 逐行讲解：声明 loaded_configs 的类型为 set[str]；调用 set（set()）。
    loaded_configs: set[str] = set()
    # 逐行讲解：声明 skipped_splits 的类型为 list[str]；构造列表 []。
    skipped_splits: list[str] = []
    # 逐行讲解：遍历 configs，将每项绑定到 config。
    for config in configs:
        # 逐行讲解：取值 False，赋给 config_loaded。
        config_loaded = False
        # 逐行讲解：遍历 args.ceval_splits，将每项绑定到 split。
        for split in args.ceval_splits:
            # 逐行讲解：调用 load_ceval_split（load_ceval_split(config, split, hf_cache_dir)），赋给 dataset。
            dataset = load_ceval_split(config, split, hf_cache_dir)
            # 逐行讲解：检查条件 dataset is None；成立时执行下方分支。
            if dataset is None:
                # 逐行讲解：向列表末尾追加一个元素（skipped_splits.append(f'{config}/{split}')）。
                skipped_splits.append(f"{config}/{split}")
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：取值 True，赋给 config_loaded。
            config_loaded = True
            # 逐行讲解：遍历 enumerate(dataset)，将每项绑定到 (row_index, row)。
            for row_index, row in enumerate(dataset):
                # 逐行讲解：调用 normalize_ceval_row（normalize_ceval_row(dict(row), config, split, row_index)），赋给 normalized。
                normalized = normalize_ceval_row(dict(row), config, split, row_index)
                # 逐行讲解：检查条件 normalized is not None；成立时执行下方分支。
                if normalized is not None:
                    # 逐行讲解：向列表末尾追加一个元素（all_rows.append(normalized)）。
                    all_rows.append(normalized)
        # 逐行讲解：检查条件 config_loaded；成立时执行下方分支。
        if config_loaded:
            # 逐行讲解：调用 loaded_configs.add（loaded_configs.add(config)）。
            loaded_configs.add(config)

    # 逐行讲解：调用 sorted（sorted(set(configs) - loaded_configs)），赋给 missing_configs。
    missing_configs = sorted(set(configs) - loaded_configs)
    # 逐行讲解：检查条件 missing_configs；成立时执行下方分支。
    if missing_configs:
        # 逐行讲解：抛出异常：RuntimeError(f'Some C-Eval configs were not loaded. Missing: {missing_configs}. Check network/cache, or pas…。
        raise RuntimeError(
            "Some C-Eval configs were not loaded. "
            f"Missing: {missing_configs}. "
            "Check network/cache, or pass --ceval-configs with only available configs."
        )

    # 逐行讲解：调用 shuffled_rows（shuffled_rows(all_rows, args.seed)），赋给 all_rows。
    all_rows = shuffled_rows(all_rows, args.seed)
    # 逐行讲解：通过推导式逐项生成结果：[row for row in all_rows if row['answer_idx']]，赋给 answer_rows。
    answer_rows = [row for row in all_rows if row["answer_idx"]]
    # 逐行讲解：通过推导式逐项生成结果：[row for row in all_rows if not row['answer_idx']]，赋给 no_answer_rows。
    no_answer_rows = [row for row in all_rows if not row["answer_idx"]]

    # 逐行讲解：调用 math.ceil（math.ceil(len(answer_rows) * (1 - args.ceval_train_ratio))），赋给 heldout_size。
    heldout_size = math.ceil(len(answer_rows) * (1 - args.ceval_train_ratio))
    # 逐行讲解：调用 min（min(heldout_size, len(answer_rows))），赋给 heldout_size。
    heldout_size = min(heldout_size, len(answer_rows))

    # 逐行讲解：按索引或切片读取 answer_rows[:heldout_size]，赋给 test_pool。
    test_pool = answer_rows[:heldout_size]
    # 逐行讲解：调用 shuffled_rows（shuffled_rows(answer_rows[heldout_size:] + no_answer_rows, args.seed + 1)），赋给 train_pool。
    train_pool = shuffled_rows(answer_rows[heldout_size:] + no_answer_rows, args.seed + 1)
    # 逐行讲解：调用 select_size（select_size(train_pool, args.ceval_train_size)），赋给 train_rows。
    train_rows = select_size(train_pool, args.ceval_train_size)
    # 逐行讲解：调用 select_size（select_size(test_pool, args.ceval_test_size)），赋给 test_rows。
    test_rows = select_size(test_pool, args.ceval_test_size)

    # 逐行讲解：计算 args.data_dir / 'ceval-non-med'，赋给 out_dir。
    out_dir = args.data_dir / "ceval-non-med"
    # 逐行讲解：调用 write_jsonl（write_jsonl(out_dir / 'all.jsonl', all_rows)），赋给 all_count。
    all_count = write_jsonl(out_dir / "all.jsonl", all_rows)
    # 逐行讲解：调用 write_jsonl（write_jsonl(out_dir / 'test_pool.jsonl', test_pool)），赋给 test_pool_count。
    test_pool_count = write_jsonl(out_dir / "test_pool.jsonl", test_pool)
    # 逐行讲解：调用 write_jsonl（write_jsonl(out_dir / 'train_pool.jsonl', train_pool)），赋给 train_pool_count。
    train_pool_count = write_jsonl(out_dir / "train_pool.jsonl", train_pool)
    # 逐行讲解：调用 write_jsonl（write_jsonl(out_dir / 'test_sample.jsonl', test_rows)），赋给 test_count。
    test_count = write_jsonl(out_dir / "test_sample.jsonl", test_rows)
    # 逐行讲解：调用 write_jsonl（write_jsonl(out_dir / 'opd_train.jsonl', train_rows)），赋给 train_count。
    train_count = write_jsonl(out_dir / "opd_train.jsonl", train_rows)
    # 逐行讲解：调用 write_json（write_json(out_dir / 'manifest.json', {'repo': CEVAL_REPO, 'excluded_configs': sorted(CEVAL_EXCLUDED_CONFIG…）。
    write_json(
        out_dir / "manifest.json",
        {
            "repo": CEVAL_REPO,
            "excluded_configs": sorted(CEVAL_EXCLUDED_CONFIGS),
            "configs": configs,
            "seed": args.seed,
            "splits": args.ceval_splits,
            "train_ratio": args.ceval_train_ratio,
            "loaded_configs": sorted(loaded_configs),
            "skipped_splits": sorted(skipped_splits),
            "all_rows": all_count,
            "test_pool_rows": test_pool_count,
            "train_pool_rows": train_pool_count,
            "test_sample_rows": test_count,
            "opd_train_rows": train_count,
        },
    )
    # 逐行讲解：调用 print（print(f"Saved C-Eval all rows: {all_count} rows -> {out_dir / 'all.jsonl'}")）。
    print(f"Saved C-Eval all rows: {all_count} rows -> {out_dir / 'all.jsonl'}")
    # 逐行讲解：调用 print（print(f"Saved C-Eval OPD train pool: {train_pool_count} rows -> {out_dir / 'train_pool.jsonl'}")）。
    print(f"Saved C-Eval OPD train pool: {train_pool_count} rows -> {out_dir / 'train_pool.jsonl'}")
    # 逐行讲解：调用 print（print(f"Saved C-Eval held-out test pool: {test_pool_count} rows -> {out_dir / 'test_pool.jsonl'}")）。
    print(f"Saved C-Eval held-out test pool: {test_pool_count} rows -> {out_dir / 'test_pool.jsonl'}")
    # 逐行讲解：调用 print（print(f"Saved C-Eval OPD train: {train_count} rows -> {out_dir / 'opd_train.jsonl'}")）。
    print(f"Saved C-Eval OPD train: {train_count} rows -> {out_dir / 'opd_train.jsonl'}")
    # 逐行讲解：调用 print（print(f"Saved C-Eval general test: {test_count} rows -> {out_dir / 'test_sample.jsonl'}")）。
    print(f"Saved C-Eval general test: {test_count} rows -> {out_dir / 'test_sample.jsonl'}")


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：创建目录（args.data_dir.mkdir(parents=True, exist_ok=True)）。
    args.data_dir.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 cache_dir（cache_dir(args)），赋给 hf_cache_dir。
    hf_cache_dir = cache_dir(args)

    # 逐行讲解：检查条件 not args.skip_medical_sft；成立时执行下方分支。
    if not args.skip_medical_sft:
        # 逐行讲解：调用 prepare_medical_sft（prepare_medical_sft(args, hf_cache_dir)）。
        prepare_medical_sft(args, hf_cache_dir)
    # 逐行讲解：检查条件 not args.skip_medqa；成立时执行下方分支。
    if not args.skip_medqa:
        # 逐行讲解：调用 prepare_medqa（prepare_medqa(args, hf_cache_dir)）。
        prepare_medqa(args, hf_cache_dir)
    # 逐行讲解：检查条件 not args.skip_ceval；成立时执行下方分支。
    if not args.skip_ceval:
        # 逐行讲解：调用 prepare_ceval（prepare_ceval(args, hf_cache_dir)）。
        prepare_ceval(args, hf_cache_dir)

    # 逐行讲解：调用 print（print('All requested datasets are ready.')）。
    print("All requested datasets are ready.")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
