"""在固定的 100 条 GeoQA test 样本上评测单个模型。

评测 Base：
uv run python eval.py

评测训练后模型：
uv run python eval.py \
    --model-path trio://run_xxx/sampler_weights/xxx-step-100-sampler
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：time。
import time
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：从 datasets 导入 Dataset, load_dataset。
from datasets import Dataset, load_dataset
# 逐行讲解：从 tqdm.asyncio 导入 tqdm_asyncio。
from tqdm.asyncio import tqdm_asyncio
# 逐行讲解：从 transformers 导入 AutoImageProcessor。
from transformers import AutoImageProcessor

# 逐行讲解：从 train 导入 CHOICE_LETTERS, DEFAULT_DATASET_DIR, DEFAULT_MODEL, build_prompt_chunks, extract_choice。
from train import (
    CHOICE_LETTERS,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL,
    build_prompt_chunks,
    extract_choice,
)

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：取值 42，赋给 EVAL_SEED。
EVAL_SEED = 42
# 逐行讲解：取值 100，赋给 EVAL_SIZE。
EVAL_SIZE = 100
# 逐行讲解：计算 SCRIPT_DIR / 'eval-results.json'，赋给 DEFAULT_OUTPUT。
DEFAULT_OUTPUT = SCRIPT_DIR / "eval-results.json"


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='评测 GeoQA 多模态 GRPO')），赋给 parser。
    parser = argparse.ArgumentParser(description="评测 GeoQA 多模态 GRPO")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)）。
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default=DEFAULT_MODEL)）。
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--model-path', help='训练脚本输出的 Sampler 权重路径，不传则评测 Base 模型')）。
    parser.add_argument(
        "--model-path",
        help="训练脚本输出的 Sampler 权重路径，不传则评测 Base 模型",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=1024)）。
    parser.add_argument("--max-tokens", type=int, default=1024)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--limit', type=int, default=EVAL_SIZE, help='默认评测固定的 100 条；调试时可缩小')）。
    parser.add_argument(
        "--limit",
        type=int,
        default=EVAL_SIZE,
        help="默认评测固定的 100 条；调试时可缩小",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)）。
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 load_eval_data：读取下载阶段固定的 100 条测试数据。
def load_eval_data(dataset_dir: Path, limit: int) -> Dataset:
    """读取下载阶段固定的 100 条测试数据。"""
    # 逐行讲解：调用 load_dataset（load_dataset('parquet', data_files=str(dataset_dir / 'test.parquet'), split='train')），赋给 dataset。
    dataset = load_dataset(
        "parquet",
        data_files=str(dataset_dir / "test.parquet"),
        split="train",
    )
    # 逐行讲解：返回结果：dataset.select(range(min(limit, len(dataset))))。
    return dataset.select(range(min(limit, len(dataset))))


# 逐行讲解：定义函数 parse_response：读取单条采样结果中的文本和选项。
def parse_response(response: Any, tokenizer: Any) -> tuple[str, str | None]:
    """读取单条采样结果中的文本和选项。"""
    # 逐行讲解：按索引或切片读取 response.sequences[0]，赋给 sequence。
    sequence = response.sequences[0]
    # 逐行讲解：取值 sequence.text or tokenizer.decode(sequence.tokens, skip_special_tokens=True)，赋给 text。
    text = sequence.text or tokenizer.decode(sequence.tokens, skip_special_tokens=True)
    # 逐行讲解：返回结果：(text, extract_choice(text))。
    return text, extract_choice(text)


# 逐行讲解：定义异步函数 main：封装下方逻辑，供调用方复用。
async def main(args: argparse.Namespace) -> None:
    # 逐行讲解：调用 load_eval_data（load_eval_data(args.dataset_dir.expanduser().resolve(), args.limit)），赋给 eval_data。
    eval_data = load_eval_data(args.dataset_dir.expanduser().resolve(), args.limit)
    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：等待异步操作 service_client.create_sampling_client_async(base_model=args.base_model, model_path=args.model_path) 完成并取得结果，赋给 sampling_client。
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.base_model,
        model_path=args.model_path,
    )
    # 逐行讲解：调用 sampling_client.get_tokenizer（sampling_client.get_tokenizer()），赋给 tokenizer。
    tokenizer = sampling_client.get_tokenizer()
    # 逐行讲解：调用 AutoImageProcessor.from_pretrained（AutoImageProcessor.from_pretrained(args.base_model, backend='pil')），赋给 image_processor。
    image_processor = AutoImageProcessor.from_pretrained(
        args.base_model,
        backend="pil",
    )
    # 逐行讲解：调用 trio.SamplingParams（trio.SamplingParams(max_tokens=args.max_tokens, seed=EVAL_SEED, temperature=0.0, stop='<|im_end|>')），赋给 sampling_params。
    sampling_params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        seed=EVAL_SEED,
        temperature=0.0,
        stop="<|im_end|>",
    )

    # 逐行讲解：定义异步函数 evaluate_row：封装下方逻辑，供调用方复用。
    async def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
        # 逐行讲解：通过推导式逐项生成结果：[str(choice) for choice in row['choices']]，赋给 choices。
        choices = [str(choice) for choice in row["choices"]]
        # 逐行讲解：按索引或切片读取 CHOICE_LETTERS[int(row['label'])]，赋给 gold_choice。
        gold_choice = CHOICE_LETTERS[int(row["label"])]
        # 逐行讲解：调用 trio.ModelInput（trio.ModelInput(chunks=build_prompt_chunks(tokenizer, image_processor, row['image'], str(row['subject']), c…），赋给 prompt。
        prompt = trio.ModelInput(
            chunks=build_prompt_chunks(
                tokenizer,
                image_processor,
                row["image"],
                str(row["subject"]),
                choices,
            )
        )
        # 逐行讲解：等待异步操作 sampling_client.sample_async(prompt=prompt, num_samples=1, sampling_params=sampling_params, return_text=True) 完成并取得结果，赋给 response。
        response = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
            return_text=True,
        )
        # 逐行讲解：调用 parse_response（parse_response(response, tokenizer)），赋给 (text, predicted_choice)。
        text, predicted_choice = parse_response(response, tokenizer)
        # 逐行讲解：返回结果：{'id': int(row['id']), 'gold': gold_choice, 'prediction': predicted_choice, 'text': text}。
        return {
            "id": int(row["id"]),
            "gold": gold_choice,
            "prediction": predicted_choice,
            "text": text,
        }

    # 固定测试集共享同一个 sampler 并发评测。
    results = await tqdm_asyncio.gather(
        *(evaluate_row(row) for row in eval_data),
        desc="评测 GeoQA",
        unit="sample",
    )

    # 逐行讲解：调用 len（len(results)），赋给 total。
    total = len(results)
    # 逐行讲解：求和（sum((result['prediction'] == result['gold'] for result in results))），赋给 correct。
    correct = sum(result["prediction"] == result["gold"] for result in results)
    # 逐行讲解：求和（sum((result['prediction'] is not None for result in results))），赋给 formatted。
    formatted = sum(result["prediction"] is not None for result in results)
    # 逐行讲解：构造字典 {'accuracy': correct / total, 'format_rate': formatted / total}，赋给 metrics。
    metrics = {
        "accuracy": correct / total,
        "format_rate": formatted / total,
    }
    # 逐行讲解：调用 args.output.expanduser().resolve（args.output.expanduser().resolve()），赋给 output。
    output = args.output.expanduser().resolve()
    # 逐行讲解：创建目录（output.parent.mkdir(parents=True, exist_ok=True)）。
    output.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：将文本写入文件（output.write_text(json.dumps({'base_model': args.base_model, 'model_path': args.model_path, 'eval_seed': EV…）。
    output.write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "model_path": args.model_path,
                "eval_seed": EVAL_SEED,
                "eval_size": total,
                "metrics": metrics,
                "samples": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 逐行讲解：调用 print（print(f'模型：{args.model_path or args.base_model}')）。
    print(f"模型：{args.model_path or args.base_model}")
    # 逐行讲解：调用 print（print(f"Accuracy：{metrics['accuracy']:.1%}")）。
    print(f"Accuracy：{metrics['accuracy']:.1%}")
    # 逐行讲解：调用 print（print(f"Format rate：{metrics['format_rate']:.1%}")）。
    print(f"Format rate：{metrics['format_rate']:.1%}")
    # 逐行讲解：调用 print（print(f'评测结果：{output}')）。
    print(f"评测结果：{output}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 time.perf_counter（time.perf_counter()），赋给 start_time。
    start_time = time.perf_counter()
    # 逐行讲解：调用 asyncio.run（asyncio.run(main(parse_args()))）。
    asyncio.run(main(parse_args()))
    # 逐行讲解：调用 print（print(f'评测耗时：{time.perf_counter() - start_time:.2f}s')）。
    print(f"评测耗时：{time.perf_counter() - start_time:.2f}s")
