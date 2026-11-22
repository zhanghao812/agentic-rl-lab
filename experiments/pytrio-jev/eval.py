"""五项基准评测：Jevbench Easy/Original/Hard + Typed Decision + ToolACE。

先选优、先校准，最后才读 test（防污染红线：评测集一律不进训练）。

用法：
uv run python eval.py                          # base 模型
uv run python eval.py --weights trio://... --calibration outputs/calibration.json
uv run python eval.py --weights trio://... --suite typed_decisions
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

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm

# 逐行讲解：从 data 导入 Encoder, load_jsonl, normalize_label。
from data import Encoder, load_jsonl, normalize_label
# 逐行讲解：从 inference 导入 decide_async。
from inference import decide_async

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets' / 'eval'，赋给 EVAL_DIR。
EVAL_DIR = SCRIPT_DIR / "datasets" / "eval"
# 逐行讲解：计算 SCRIPT_DIR / 'outputs'，赋给 OUTPUT_DIR。
OUTPUT_DIR = SCRIPT_DIR / "outputs"
# 逐行讲解：取值 'Qwen/Qwen3.5-4B'，赋给 DEFAULT_MODEL。
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

# 逐行讲解：构造字典 {'jevbench_easy': EVAL_DIR / 'jevbench_easy.jsonl', 'jevbench_original': EVAL_DIR / 'jevbench_original.json…，赋给 SUITES。
SUITES = {
    "jevbench_easy": EVAL_DIR / "jevbench_easy.jsonl",
    "jevbench_original": EVAL_DIR / "jevbench_original.jsonl",
    "jevbench_hard": EVAL_DIR / "jevbench_hard.jsonl",
    "typed_decisions": EVAL_DIR / "typed_decisions.jsonl",
    "toolace": EVAL_DIR / "toolace.jsonl",
}


# 逐行讲解：定义函数 adapt_jevbench：Jevbench 上游单问题格式 → canonical record。
def adapt_jevbench(row: dict) -> dict:
    """Jevbench 上游单问题格式 → canonical record。

    choice 的选项顺序以 labels 数组为准（criteria 在文件里是字母序）；noul 的
    criteria 键是 true/false、score 的 criteria 是等级列表，二者由 question_options
    直接展开，顺序与 labels 天然一致。
    """
    # 逐行讲解：按索引或切片读取 row['question']，赋给 question。
    question = row["question"]
    # 逐行讲解：按索引或切片读取 question['criteria']，赋给 criteria。
    criteria = question["criteria"]
    # 逐行讲解：检查条件 question['type'] == 'choice'；成立时执行下方分支。
    if question["type"] == "choice":
        # 逐行讲解：通过推导式逐项生成结果：{label: criteria[label] for label in row['labels']}，赋给 criteria。
        criteria = {label: criteria[label] for label in row["labels"]}
    # 逐行讲解：返回结果：{'id': row['id'], 'state': row['state'], 'questions': {'decision': {'type': question['type'], 'instructions…。
    return {
        "id": row["id"],
        "state": row["state"],
        "questions": {
            "decision": {
                "type": question["type"],
                "instructions": question["instructions"],
                "criteria": criteria,
            }
        },
        "targets": {"decision": {"label": row["expected"]}},
    }


# 逐行讲解：定义函数 load_suite：封装下方逻辑，供调用方复用。
def load_suite(name: str) -> list[dict]:
    # 逐行讲解：调用 load_jsonl（load_jsonl(SUITES[name])），赋给 rows。
    rows = load_jsonl(SUITES[name])
    # 逐行讲解：检查条件 name.startswith('jevbench')；成立时执行下方分支。
    if name.startswith("jevbench"):
        # 逐行讲解：返回结果：[adapt_jevbench(row) for row in rows]。
        return [adapt_jevbench(row) for row in rows]
    # 逐行讲解：返回结果：rows。
    return rows


# 逐行讲解：定义函数 ece10：10 个等宽 bin 的期望校准误差，置信度取 max(P)。
def ece10(confidences: np.ndarray, correct: np.ndarray) -> float:
    """10 个等宽 bin 的期望校准误差，置信度取 max(P)。"""
    # 逐行讲解：取值 0.0，赋给 value。
    value = 0.0
    # 逐行讲解：遍历 np.arange(0.0, 1.0, 0.1)，将每项绑定到 lo。
    for lo in np.arange(0.0, 1.0, 0.1):
        # 逐行讲解：计算 (confidences >= lo) & (confidences < lo + 0.1 if lo < 0.9 else confidences <= 1.0)，赋给 mask。
        mask = (confidences >= lo) & (confidences < lo + 0.1 if lo < 0.9 else confidences <= 1.0)
        # 逐行讲解：检查条件 mask.any()；成立时执行下方分支。
        if mask.any():
            # 逐行讲解：更新 value：value += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())。
            value += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    # 逐行讲解：返回结果：float(value)。
    return float(value)


# 逐行讲解：定义异步函数 evaluate_suite：封装下方逻辑，供调用方复用。
async def evaluate_suite(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    name: str,
    temperature: float,
    concurrency: int,
) -> dict:
    # 逐行讲解：调用 load_suite（load_suite(name)），赋给 records。
    records = load_suite(name)
    # 逐行讲解：调用 asyncio.Semaphore（asyncio.Semaphore(concurrency)），赋给 semaphore。
    semaphore = asyncio.Semaphore(concurrency)
    # 逐行讲解：声明 completed 的类型为 list；计算 [None] * len(records)。
    completed: list = [None] * len(records)

    # 逐行讲解：进入上下文管理器 tqdm(total=len(records), desc=name, unit='条')，退出时执行其收尾逻辑。
    with tqdm(total=len(records), desc=name, unit="条") as progress:

        # 逐行讲解：定义异步函数 run_one：封装下方逻辑，供调用方复用。
        async def run_one(index: int, record: dict) -> None:
            # 逐行讲解：进入上下文管理器 semaphore，退出时执行其收尾逻辑。
            async with semaphore:
                # 逐行讲解：等待异步操作 decide_async(sampling_client, encoder, record, temperature) 完成并取得结果，赋给 result。
                result = await decide_async(
                    sampling_client, encoder, record, temperature
                )
            # 逐行讲解：构造元组 (record, result)，赋给 completed[index]。
            completed[index] = (record, result)
            # 逐行讲解：调用 progress.update（progress.update(1)）。
            progress.update(1)

        # 逐行讲解：等待异步操作 asyncio.gather(*(run_one(index, record) for index, record in enumerate(records))) 完成并取得结果。
        await asyncio.gather(
            *(run_one(index, record) for index, record in enumerate(records))
        )

    # 逐行讲解：取值 0，赋给 n、correct。
    n = correct = 0
    # 逐行讲解：构造元组 ([], [], [])，赋给 (confidences, correctness, briers)。
    confidences, correctness, briers = [], [], []
    # 逐行讲解：遍历 completed，将每项绑定到 (record, result)。
    for record, result in completed:
        # 逐行讲解：遍历 result.items()，将每项绑定到 (field, field_result)。
        for field, field_result in result.items():
            # 逐行讲解：调用 normalize_label（normalize_label(record['questions'][field], record['targets'][field]['label'])），赋给 gold_label。
            gold_label = normalize_label(
                record["questions"][field], record["targets"][field]["label"]
            )
            # 逐行讲解：按索引或切片读取 field_result['probabilities']，赋给 probs。
            probs = field_result["probabilities"]
            # 逐行讲解：取值 field_result['prediction'] == gold_label，赋给 hit。
            hit = field_result["prediction"] == gold_label
            # 逐行讲解：更新 n：n += 1。
            n += 1
            # 逐行讲解：更新 correct：correct += hit。
            correct += hit
            # 逐行讲解：向列表末尾追加一个元素（confidences.append(max(probs.values()))）。
            confidences.append(max(probs.values()))
            # 逐行讲解：向列表末尾追加一个元素（correctness.append(hit)）。
            correctness.append(hit)
            # 逐行讲解：向列表末尾追加一个元素（briers.append(sum(((p - (1.0 if value == gold_label else 0.0)) ** 2 for value, p in probs.items())))）。
            briers.append(
                sum((p - (1.0 if value == gold_label else 0.0)) ** 2 for value, p in probs.items())
            )
    # 逐行讲解：调用 np.array（np.array(confidences)），赋给 confidences。
    confidences = np.array(confidences)
    # 逐行讲解：调用 np.array（np.array(correctness, dtype=float)），赋给 correctness。
    correctness = np.array(correctness, dtype=float)
    # 逐行讲解：返回结果：{'suite': name, 'records': len(records), 'decisions': n, 'accuracy': correct / n, 'ece': ece10(confidences,…。
    return {
        "suite": name,
        "records": len(records),
        "decisions": n,
        "accuracy": correct / n,
        "ece": ece10(confidences, correctness),
        "brier": float(np.mean(briers)),
    }


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='五项决策基准评测')），赋给 parser。
    parser = argparse.ArgumentParser(description="五项决策基准评测")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default=DEFAULT_MODEL)）。
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--weights', default=None, help='trio:// LoRA 权重路径，不传则评 base')）。
    parser.add_argument("--weights", default=None, help="trio:// LoRA 权重路径，不传则评 base")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--calibration', type=Path, default=None, help='calibration.json')）。
    parser.add_argument("--calibration", type=Path, default=None, help="calibration.json")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--suite', choices=list(SUITES) + ['all'], default='all')）。
    parser.add_argument("--suite", choices=list(SUITES) + ["all"], default="all")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--tag', default=None, help='输出文件名标记')）。
    parser.add_argument("--tag", default=None, help="输出文件名标记")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--concurrency', type=int, default=15, help='并发评测条数')）。
    parser.add_argument("--concurrency", type=int, default=15, help="并发评测条数")
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义异步函数 main：封装下方逻辑，供调用方复用。
async def main(args: argparse.Namespace) -> None:
    # 逐行讲解：取值 1.0，赋给 temperature。
    temperature = 1.0
    # 逐行讲解：检查条件 args.calibration；成立时执行下方分支。
    if args.calibration:
        # 逐行讲解：按索引或切片读取 json.loads(args.calibration.read_text())['temperature']，赋给 temperature。
        temperature = json.loads(args.calibration.read_text())["temperature"]

    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：等待异步操作 service_client.create_sampling_client_async(base_model=args.base_model, model_path=args.weights) 完成并取得结果，赋给 sampling_client。
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.base_model, model_path=args.weights
    )
    # 逐行讲解：调用 Encoder（Encoder(sampling_client.get_tokenizer())），赋给 encoder。
    encoder = Encoder(sampling_client.get_tokenizer())

    # 逐行讲解：按条件选择结果：list(SUITES) if args.suite == 'all' else [args.suite]，赋给 suite_names。
    suite_names = list(SUITES) if args.suite == "all" else [args.suite]
    # 逐行讲解：通过推导式逐项生成结果：[await evaluate_suite(sampling_client, encoder, name, temperature, args.concurrency) for name in suite_names]，赋给 results。
    results = [
        await evaluate_suite(
            sampling_client, encoder, name, temperature, args.concurrency
        )
        for name in suite_names
    ]

    # 逐行讲解：调用 print（print(f"\n{'suite':<20} {'decisions':>9} {'acc':>8} {'ece':>8} {'brier':>8}")）。
    print(f"\n{'suite':<20} {'decisions':>9} {'acc':>8} {'ece':>8} {'brier':>8}")
    # 逐行讲解：遍历 results，将每项绑定到 r。
    for r in results:
        # 逐行讲解：调用 print（print(f"{r['suite']:<20} {r['decisions']:>9} {r['accuracy']:>8.2%} {r['ece']:>8.4f} {r['brier']:>8.4f}")）。
        print(
            f"{r['suite']:<20} {r['decisions']:>9} "
            f"{r['accuracy']:>8.2%} {r['ece']:>8.4f} {r['brier']:>8.4f}"
        )
    # 逐行讲解：检查条件 len(results) > 1；成立时执行下方分支。
    if len(results) > 1:
        # 逐行讲解：计算均值（np.mean([r['accuracy'] for r in results])），赋给 mean_acc。
        mean_acc = np.mean([r["accuracy"] for r in results])
        # 逐行讲解：调用 print（print(f"{'average':<20} {'':>9} {mean_acc:>8.2%}")）。
        print(f"{'average':<20} {'':>9} {mean_acc:>8.2%}")

    # 逐行讲解：创建目录（OUTPUT_DIR.mkdir(parents=True, exist_ok=True)）。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：取值 args.tag or (f'trained-{int(time.time())}' if args.weights else 'base')，赋给 tag。
    tag = args.tag or (f"trained-{int(time.time())}" if args.weights else "base")
    # 逐行讲解：计算 OUTPUT_DIR / f'eval-{tag}.json'，赋给 output。
    output = OUTPUT_DIR / f"eval-{tag}.json"
    # 逐行讲解：将文本写入文件（output.write_text(json.dumps({'weights': args.weights, 'temperature': temperature, 'results': results}, ind…）。
    output.write_text(
        json.dumps(
            {
                "weights": args.weights,
                "temperature": temperature,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    # 逐行讲解：调用 print（print(f'写出 {output}')）。
    print(f"写出 {output}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 asyncio.run（asyncio.run(main(parse_args()))）。
    asyncio.run(main(parse_args()))
