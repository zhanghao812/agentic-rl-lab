"""温度校准：在 calibration 划分上拟合标量温度 T，使候选分布的 NLL 最小。

p_T = softmax(logprob / T)，T 只改概率形状、不改 argmax。
搜索用 log T 上的黄金分割（NLL 关于 1/T 是凸的，这里直接做无导数搜索）。

用法：
uv run python experiments/pytrio-jev/calibrate.py --weights trio://...
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：math。
import math
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm

# 逐行讲解：从 data 导入 Encoder, compile_record, load_jsonl。
from data import Encoder, compile_record, load_jsonl
# 逐行讲解：从 inference 导入 candidate_logprobs_async。
from inference import candidate_logprobs_async

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets' / 'calibration.jsonl'，赋给 DEFAULT_DATA。
DEFAULT_DATA = SCRIPT_DIR / "datasets" / "calibration.jsonl"
# 逐行讲解：计算 SCRIPT_DIR / 'outputs' / 'calibration.json'，赋给 DEFAULT_OUTPUT。
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "calibration.json"
# 逐行讲解：取值 'Qwen/Qwen3.5-4B'，赋给 DEFAULT_MODEL。
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


# 逐行讲解：定义异步函数 collect_instances：逐字段收集 (候选 logprob 向量, gold 下标)。
async def collect_instances(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    records: list[dict],
    concurrency: int,
) -> list[tuple[np.ndarray, int]]:
    """逐字段收集 (候选 logprob 向量, gold 下标)。"""
    # 逐行讲解：调用 asyncio.Semaphore（asyncio.Semaphore(concurrency)），赋给 semaphore。
    semaphore = asyncio.Semaphore(concurrency)
    # 逐行讲解：声明 per_record 的类型为 list；计算 [None] * len(records)。
    per_record: list = [None] * len(records)

    # 逐行讲解：进入上下文管理器 tqdm(total=len(records), desc='收集校准预测', unit='条')，退出时执行其收尾逻辑。
    with tqdm(total=len(records), desc="收集校准预测", unit="条") as progress:

        # 逐行讲解：定义异步函数 run_one：封装下方逻辑，供调用方复用。
        async def run_one(index: int, record: dict) -> None:
            # 逐行讲解：按索引或切片读取 compile_record(record, include_targets=True)['targets']，赋给 targets。
            targets = compile_record(record, include_targets=True)["targets"]
            # 逐行讲解：进入上下文管理器 semaphore，退出时执行其收尾逻辑。
            async with semaphore:
                # 逐行讲解：等待异步操作 candidate_logprobs_async(sampling_client, encoder, record) 完成并取得结果，赋给 collected。
                collected = await candidate_logprobs_async(
                    sampling_client, encoder, record
                )
            # 逐行讲解：构造元组 (collected['encoded'], collected['logprobs'])，赋给 (encoded, logprobs)。
            encoded, logprobs = collected["encoded"], collected["logprobs"]
            # 逐行讲解：构造列表 []，赋给 instances。
            instances = []
            # 逐行讲解：遍历 encoded['options'].items()，将每项绑定到 (field, options)。
            for field, options in encoded["options"].items():
                # 逐行讲解：通过推导式逐项生成结果：[symbol for symbol, _, _ in options]，赋给 symbols。
                symbols = [symbol for symbol, _, _ in options]
                # 逐行讲解：调用 symbols.index（symbols.index(targets[field])），赋给 gold。
                gold = symbols.index(targets[field])
                # 逐行讲解：向列表末尾追加一个元素（instances.append((np.array([logprobs[field][s] for s in symbols]), gold))）。
                instances.append(
                    (np.array([logprobs[field][s] for s in symbols]), gold)
                )
            # 逐行讲解：取值 instances，赋给 per_record[index]。
            per_record[index] = instances
            # 逐行讲解：调用 progress.update（progress.update(1)）。
            progress.update(1)

        # 逐行讲解：等待异步操作 asyncio.gather(*(run_one(index, record) for index, record in enumerate(records))) 完成并取得结果。
        await asyncio.gather(
            *(run_one(index, record) for index, record in enumerate(records))
        )
    # 逐行讲解：返回结果：[instance for instances in per_record for instance in instances]。
    return [instance for instances in per_record for instance in instances]


# 逐行讲解：定义函数 mean_nll：封装下方逻辑，供调用方复用。
def mean_nll(instances: list[tuple[np.ndarray, int]], temperature: float) -> float:
    # 逐行讲解：取值 0.0，赋给 total。
    total = 0.0
    # 逐行讲解：遍历 instances，将每项绑定到 (logits, gold)。
    for logits, gold in instances:
        # 逐行讲解：计算 logits / temperature，赋给 z。
        z = logits / temperature
        # 逐行讲解：更新 total：total += float(np.log(np.exp(z - z.max()).sum()) + z.max() - z[gold])。
        total += float(np.log(np.exp(z - z.max()).sum()) + z.max() - z[gold])
    # 逐行讲解：返回结果：total / len(instances)。
    return total / len(instances)


# 逐行讲解：定义函数 fit_temperature：在 log T ∈ [log 0.01, log 100] 上做黄金分割搜索。
def fit_temperature(instances: list[tuple[np.ndarray, int]]) -> float:
    """在 log T ∈ [log 0.01, log 100] 上做黄金分割搜索。"""
    # 逐行讲解：构造元组 (math.log(0.01), math.log(100.0))，赋给 (lo, hi)。
    lo, hi = math.log(0.01), math.log(100.0)
    # 逐行讲解：计算 (math.sqrt(5) - 1) / 2，赋给 inv_phi。
    inv_phi = (math.sqrt(5) - 1) / 2
    # 逐行讲解：遍历 range(80)，将每项绑定到 _。
    for _ in range(80):
        # 逐行讲解：计算 hi - inv_phi * (hi - lo)，赋给 c。
        c = hi - inv_phi * (hi - lo)
        # 逐行讲解：计算 lo + inv_phi * (hi - lo)，赋给 d。
        d = lo + inv_phi * (hi - lo)
        # 逐行讲解：检查条件 mean_nll(instances, math.exp(c)) < mean_nll(instances, math.exp(d))；成立时执行下方分支。
        if mean_nll(instances, math.exp(c)) < mean_nll(instances, math.exp(d)):
            # 逐行讲解：取值 d，赋给 hi。
            hi = d
        else:
            # 逐行讲解：取值 c，赋给 lo。
            lo = c
    # 逐行讲解：返回结果：math.exp((lo + hi) / 2)。
    return math.exp((lo + hi) / 2)


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='温度校准')），赋给 parser。
    parser = argparse.ArgumentParser(description="温度校准")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default=DEFAULT_MODEL)）。
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--weights', required=True, help='trio:// LoRA 权重路径')）。
    parser.add_argument("--weights", required=True, help="trio:// LoRA 权重路径")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data', type=Path, default=DEFAULT_DATA)）。
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)）。
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--concurrency', type=int, default=15, help='并发收集条数')）。
    parser.add_argument("--concurrency", type=int, default=15, help="并发收集条数")
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义异步函数 main：封装下方逻辑，供调用方复用。
async def main(args: argparse.Namespace) -> None:
    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：等待异步操作 service_client.create_sampling_client_async(base_model=args.base_model, model_path=args.weights) 完成并取得结果，赋给 sampling_client。
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.base_model, model_path=args.weights
    )
    # 逐行讲解：调用 Encoder（Encoder(sampling_client.get_tokenizer())），赋给 encoder。
    encoder = Encoder(sampling_client.get_tokenizer())

    # 逐行讲解：调用 load_jsonl（load_jsonl(args.data)），赋给 records。
    records = load_jsonl(args.data)
    # 逐行讲解：等待异步操作 collect_instances(sampling_client, encoder, records, args.concurrency) 完成并取得结果，赋给 instances。
    instances = await collect_instances(
        sampling_client, encoder, records, args.concurrency
    )
    # 逐行讲解：调用 print（print(f'校准实例：{len(instances)} 个字段（来自 {len(records)} 条记录）')）。
    print(f"校准实例：{len(instances)} 个字段（来自 {len(records)} 条记录）")

    # 逐行讲解：调用 mean_nll（mean_nll(instances, 1.0)），赋给 nll_before。
    nll_before = mean_nll(instances, 1.0)
    # 逐行讲解：调用 fit_temperature（fit_temperature(instances)），赋给 temperature。
    temperature = fit_temperature(instances)
    # 逐行讲解：调用 mean_nll（mean_nll(instances, temperature)），赋给 nll_after。
    nll_after = mean_nll(instances, temperature)
    # 逐行讲解：调用 print（print(f'NLL: T=1 → {nll_before:.4f} | T={temperature:.6f} → {nll_after:.4f}')）。
    print(f"NLL: T=1 → {nll_before:.4f} | T={temperature:.6f} → {nll_after:.4f}")

    # 逐行讲解：创建目录（args.output.parent.mkdir(parents=True, exist_ok=True)）。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：将文本写入文件（args.output.write_text(json.dumps({'temperature': temperature, 'weights': args.weights, 'instances': len(in…）。
    args.output.write_text(
        json.dumps(
            {
                "temperature": temperature,
                "weights": args.weights,
                "instances": len(instances),
                "nll_t1": nll_before,
                "nll_fitted": nll_after,
            },
            indent=2,
        )
        + "\n"
    )
    # 逐行讲解：调用 print（print(f'写出 {args.output}')）。
    print(f"写出 {args.output}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 asyncio.run（asyncio.run(main(parse_args()))）。
    asyncio.run(main(parse_args()))
