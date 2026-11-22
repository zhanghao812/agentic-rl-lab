"""从 Hugging Face 下载 OPSD 训练集与 AIME 2025 评测集。

默认下载两份数据并保存为 Hugging Face ``save_to_disk`` 格式：

    uv run python 00-datasets.py

只下载其中一份：

    uv run python 00-datasets.py --only opsd
    uv run python 00-datasets.py --only aime25

数据 revision 被固定，避免上游更新导致训练集或 benchmark 静默变化。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：shutil。
import shutil
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：从 datasets 导入 Dataset, DatasetDict, load_dataset, load_from_disk。
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets'，赋给 DEFAULT_DATA_ROOT。
DEFAULT_DATA_ROOT = SCRIPT_DIR / "datasets"


# 逐行讲解：定义类 DatasetSpec：封装下方逻辑，供调用方复用。
@dataclass(frozen=True)
class DatasetSpec:
    # 逐行讲解：声明 key 的类型为 str。
    key: str
    # 逐行讲解：声明 repo_id 的类型为 str。
    repo_id: str
    # 逐行讲解：声明 revision 的类型为 str。
    revision: str
    # 逐行讲解：声明 split 的类型为 str。
    split: str
    # 逐行讲解：声明 output_name 的类型为 str。
    output_name: str
    # 逐行讲解：声明 expected_rows 的类型为 int。
    expected_rows: int
    # 逐行讲解：声明 required_columns 的类型为 tuple[str, ...]。
    required_columns: tuple[str, ...]


# 逐行讲解：构造字典 {'opsd': DatasetSpec(key='opsd', repo_id='siyanzhao/Openthoughts_math_30k_opsd', revision='1f33e9dc2e8a1c63…，赋给 DATASETS。
DATASETS = {
    "opsd": DatasetSpec(
        key="opsd",
        repo_id="siyanzhao/Openthoughts_math_30k_opsd",
        revision="1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62",
        split="train",
        output_name="openthoughts_math_30k_opsd",
        expected_rows=29_434,
        required_columns=("problem", "solution"),
    ),
    "aime25": DatasetSpec(
        key="aime25",
        repo_id="yentinglin/aime_2025",
        revision="6f71d77b0b89b9dabe07ab466c51df33f514df7f",
        split="train",
        output_name="aime_2025",
        expected_rows=30,
        required_columns=("problem", "answer"),
    ),
}


# 逐行讲解：定义函数 parse_args：解析数据集选择、缓存目录和强制重下等命令行参数。
def parse_args() -> argparse.Namespace:
    """解析数据集选择、缓存目录和强制重下等命令行参数。"""
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--only', choices=['all', *DATASETS], default='all', help='下载全部数据，或只下载 OPSD/AIME25；默认 a…）。
    parser.add_argument(
        "--only",
        choices=["all", *DATASETS],
        default="all",
        help="下载全部数据，或只下载 OPSD/AIME25；默认 all",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT, help='save_to_disk 数据保存目录')）。
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="save_to_disk 数据保存目录",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--cache-dir', type=Path, default=None, help='Hugging Face 下载缓存；默认使用 <data-root>/.cache')）。
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face 下载缓存；默认使用 <data-root>/.cache",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--force', action='store_true', help='删除并重新下载已经存在的目标数据目录')）。
    parser.add_argument(
        "--force",
        action="store_true",
        help="删除并重新下载已经存在的目标数据目录",
    )
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 as_single_split：把单个 Dataset 或 DatasetDict 统一转换为指定 split。
def as_single_split(dataset: Dataset | DatasetDict, split: str) -> Dataset:
    """把单个 Dataset 或 DatasetDict 统一转换为指定 split。"""
    # 逐行讲解：检查条件 isinstance(dataset, Dataset)；成立时执行下方分支。
    if isinstance(dataset, Dataset):
        # 逐行讲解：返回结果：dataset。
        return dataset
    # 逐行讲解：检查条件 split not in dataset；成立时执行下方分支。
    if split not in dataset:
        # 逐行讲解：抛出异常：ValueError(f'数据集中不存在 {split!r} split，可用 split: {list(dataset)}')。
        raise ValueError(f"数据集中不存在 {split!r} split，可用 split: {list(dataset)}")
    # 逐行讲解：返回结果：dataset[split]。
    return dataset[split]


# 逐行讲解：定义函数 validate_dataset：校验数据集的字段、行数以及首尾样本是否完整。
def validate_dataset(dataset: Dataset, spec: DatasetSpec, location: str) -> None:
    """校验数据集的字段、行数以及首尾样本是否完整。"""
    # 逐行讲解：调用 sorted（sorted(set(spec.required_columns) - set(dataset.column_names))），赋给 missing。
    missing = sorted(set(spec.required_columns) - set(dataset.column_names))
    # 逐行讲解：检查条件 missing；成立时执行下方分支。
    if missing:
        # 逐行讲解：抛出异常：ValueError(f'{spec.repo_id} 在 {location} 缺少字段 {missing}；实际字段为 {dataset.column_names}')。
        raise ValueError(
            f"{spec.repo_id} 在 {location} 缺少字段 {missing}；"
            f"实际字段为 {dataset.column_names}"
        )
    # 逐行讲解：检查条件 len(dataset) != spec.expected_rows；成立时执行下方分支。
    if len(dataset) != spec.expected_rows:
        # 逐行讲解：抛出异常：ValueError(f'{spec.repo_id} 在 {location} 应有 {spec.expected_rows} 条，实际为 {len(dataset)} 条')。
        raise ValueError(
            f"{spec.repo_id} 在 {location} 应有 {spec.expected_rows} 条，"
            f"实际为 {len(dataset)} 条"
        )

    # 首尾抽样既能发现错误 split，也不会为了检查而把 30k 条长解答全部读入内存。
    for index in (0, len(dataset) - 1):
        # 逐行讲解：按索引或切片读取 dataset[index]，赋给 row。
        row = dataset[index]
        # 逐行讲解：通过推导式逐项生成结果：[column for column in spec.required_columns if not str(row.get(column, '')).strip()]，赋给 empty。
        empty = [
            column
            for column in spec.required_columns
            if not str(row.get(column, "")).strip()
        ]
        # 逐行讲解：检查条件 empty；成立时执行下方分支。
        if empty:
            # 逐行讲解：抛出异常：ValueError(f'{spec.repo_id} 第 {index} 条存在空字段: {empty}')。
            raise ValueError(f"{spec.repo_id} 第 {index} 条存在空字段: {empty}")


# 逐行讲解：定义函数 download_one：下载一份固定 revision 的数据集，落盘后再次读取校验。
def download_one(
    spec: DatasetSpec, data_root: Path, cache_dir: Path, force: bool
) -> None:
    """下载一份固定 revision 的数据集，落盘后再次读取校验。"""
    # 逐行讲解：计算 data_root / spec.output_name，赋给 output_path。
    output_path = data_root / spec.output_name
    # 逐行讲解：检查条件 output_path.exists() and (not force)；成立时执行下方分支。
    if output_path.exists() and not force:
        # 逐行讲解：调用 as_single_split（as_single_split(load_from_disk(str(output_path)), spec.split)），赋给 local。
        local = as_single_split(load_from_disk(str(output_path)), spec.split)
        # 逐行讲解：调用 validate_dataset（validate_dataset(local, spec, str(output_path))）。
        validate_dataset(local, spec, str(output_path))
        # 逐行讲解：调用 print（print(f'[skip] {spec.key}: 本地数据已经存在且校验通过 ({len(local):,} rows) -> {output_path}')）。
        print(
            f"[skip] {spec.key}: 本地数据已经存在且校验通过 "
            f"({len(local):,} rows) -> {output_path}"
        )
        # 逐行讲解：返回，结束当前函数。
        return

    # 逐行讲解：检查条件 output_path.exists()；成立时执行下方分支。
    if output_path.exists():
        # 逐行讲解：调用 shutil.rmtree（shutil.rmtree(output_path)）。
        shutil.rmtree(output_path)

    # 逐行讲解：调用 print（print(f'[download] {spec.repo_id}@{spec.revision} [{spec.split}]')）。
    print(f"[download] {spec.repo_id}@{spec.revision} [{spec.split}]")
    # 逐行讲解：调用 load_dataset（load_dataset(spec.repo_id, revision=spec.revision, split=spec.split, cache_dir=str(cache_dir))），赋给 dataset。
    dataset = load_dataset(
        spec.repo_id,
        revision=spec.revision,
        split=spec.split,
        cache_dir=str(cache_dir),
    )
    # 逐行讲解：检查条件 not isinstance(dataset, Dataset)；成立时执行下方分支。
    if not isinstance(dataset, Dataset):
        # 逐行讲解：抛出异常：TypeError(f'期望 Dataset，实际得到 {type(dataset)!r}')。
        raise TypeError(f"期望 Dataset，实际得到 {type(dataset)!r}")
    # 逐行讲解：调用 validate_dataset（validate_dataset(dataset, spec, 'Hugging Face')）。
    validate_dataset(dataset, spec, "Hugging Face")

    # 逐行讲解：创建目录（output_path.parent.mkdir(parents=True, exist_ok=True)）。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 dataset.save_to_disk（dataset.save_to_disk(str(output_path))）。
    dataset.save_to_disk(str(output_path))

    # 逐行讲解：调用 as_single_split（as_single_split(load_from_disk(str(output_path)), spec.split)），赋给 saved。
    saved = as_single_split(load_from_disk(str(output_path)), spec.split)
    # 逐行讲解：调用 validate_dataset（validate_dataset(saved, spec, str(output_path))）。
    validate_dataset(saved, spec, str(output_path))
    # 逐行讲解：调用 print（print(f'[done] {spec.key}: {len(saved):,} rows, columns={saved.column_names} -> {output_path}')）。
    print(
        f"[done] {spec.key}: {len(saved):,} rows, "
        f"columns={saved.column_names} -> {output_path}"
    )


# 逐行讲解：定义函数 main：根据用户选择依次下载 OPSD 训练集和 AIME25 评测集。
def main(args: argparse.Namespace) -> None:
    """根据用户选择依次下载 OPSD 训练集和 AIME25 评测集。"""
    # 逐行讲解：调用 args.data_root.resolve（args.data_root.resolve()），赋给 data_root。
    data_root = args.data_root.resolve()
    # 逐行讲解：调用 (args.cache_dir or data_root / '.cache').resolve（(args.cache_dir or data_root / '.cache').resolve()），赋给 cache_dir。
    cache_dir = (args.cache_dir or data_root / ".cache").resolve()
    # 逐行讲解：创建目录（data_root.mkdir(parents=True, exist_ok=True)）。
    data_root.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：创建目录（cache_dir.mkdir(parents=True, exist_ok=True)）。
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 逐行讲解：按条件选择结果：DATASETS.values() if args.only == 'all' else (DATASETS[args.only],)，赋给 selected。
    selected = DATASETS.values() if args.only == "all" else (DATASETS[args.only],)
    # 逐行讲解：遍历 selected，将每项绑定到 spec。
    for spec in selected:
        # 逐行讲解：调用 download_one（download_one(spec, data_root, cache_dir, args.force)）。
        download_one(spec, data_root, cache_dir, args.force)


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main(parse_args())）。
    main(parse_args())
