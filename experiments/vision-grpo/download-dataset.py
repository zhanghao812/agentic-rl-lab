"""下载 GeoQA 数据集到当前案例的 datasets 目录。

运行（仓库根目录）：
    uv run python experiments/vision-grpo/download-dataset.py
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 tempfile 导入 TemporaryDirectory。
from tempfile import TemporaryDirectory

# 逐行讲解：从 datasets 导入 load_dataset。
from datasets import load_dataset
# 逐行讲解：从 huggingface_hub 导入 snapshot_download。
from huggingface_hub import snapshot_download

# 逐行讲解：取值 'hz2475/geoQA'，赋给 DATASET_ID。
DATASET_ID = "hz2475/geoQA"
# 逐行讲解：计算 Path(__file__).resolve().parent / 'datasets'，赋给 DEFAULT_OUTPUT_DIR。
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "datasets"
# 逐行讲解：取值 42，赋给 TEST_SEED。
TEST_SEED = 42
# 逐行讲解：取值 100，赋给 TEST_SIZE。
TEST_SIZE = 100


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='下载 GeoQA 数据集')），赋给 parser。
    parser = argparse.ArgumentParser(description="下载 GeoQA 数据集")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-id', default=DATASET_ID)）。
    parser.add_argument("--dataset-id", default=DATASET_ID)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)）。
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--revision', default='main')）。
    parser.add_argument("--revision", default="main")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--force', action='store_true', help='强制重新下载')）。
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：调用 args.output_dir.expanduser().resolve（args.output_dir.expanduser().resolve()），赋给 output_dir。
    output_dir = args.output_dir.expanduser().resolve()
    # 逐行讲解：创建目录（output_dir.mkdir(parents=True, exist_ok=True)）。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 逐行讲解：进入上下文管理器 TemporaryDirectory(prefix='geoqa-')，退出时执行其收尾逻辑。
    with TemporaryDirectory(prefix="geoqa-") as download_dir:
        # 逐行讲解：调用 snapshot_download（snapshot_download(repo_id=args.dataset_id, repo_type='dataset', revision=args.revision, local_dir=download_…）。
        snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=download_dir,
            allow_patterns=["data/*.parquet"],
            force_download=args.force,
        )
        # 逐行讲解：调用 sorted（sorted((Path(download_dir) / 'data').glob('*.parquet'))），赋给 parquet_files。
        parquet_files = sorted((Path(download_dir) / "data").glob("*.parquet"))
        # 逐行讲解：调用 load_dataset（load_dataset('parquet', data_files={'train': [str(path) for path in parquet_files]}, split='train')），赋给 dataset。
        dataset = load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in parquet_files]},
            split="train",
        )

        # 逐行讲解：调用 dataset.filter(lambda split: split == 'train', input_columns=['original_split'], desc='提取 GeoQA train').rem…（dataset.filter(lambda split: split == 'train', input_columns=['original_split'], desc='提取 GeoQA train').rem…），赋给 train_data。
        train_data = dataset.filter(
            lambda split: split == "train",
            input_columns=["original_split"],
            desc="提取 GeoQA train",
        ).remove_columns("original_split")
        # 逐行讲解：打乱数据顺序（dataset.filter(lambda split: split == 'test', input_columns=['original_split'], desc='提取 GeoQA test').shuff…），赋给 test_data。
        test_data = dataset.filter(
            lambda split: split == "test",
            input_columns=["original_split"],
            desc="提取 GeoQA test",
        ).shuffle(seed=TEST_SEED)
        # 逐行讲解：调用 test_data.select(range(len(test_data) - TEST_SIZE, len(test_data))).remove_columns（test_data.select(range(len(test_data) - TEST_SIZE, len(test_data))).remove_columns('original_split')），赋给 test_data。
        test_data = test_data.select(
            range(len(test_data) - TEST_SIZE, len(test_data))
        ).remove_columns("original_split")

        # 逐行讲解：计算 output_dir / 'train.parquet'，赋给 train_path。
        train_path = output_dir / "train.parquet"
        # 逐行讲解：计算 output_dir / 'test.parquet'，赋给 test_path。
        test_path = output_dir / "test.parquet"
        # 逐行讲解：调用 train_data.to_parquet（train_data.to_parquet(train_path)）。
        train_data.to_parquet(train_path)
        # 逐行讲解：调用 test_data.to_parquet（test_data.to_parquet(test_path)）。
        test_data.to_parquet(test_path)

    # 逐行讲解：调用 print（print(f'dataset_dir={output_dir}')）。
    print(f"dataset_dir={output_dir}")
    # 逐行讲解：调用 print（print(f'train_file={train_path} rows={len(train_data)}')）。
    print(f"train_file={train_path} rows={len(train_data)}")
    # 逐行讲解：调用 print（print(f'test_file={test_path} rows={len(test_data)}')）。
    print(f"test_file={test_path} rows={len(test_data)}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
