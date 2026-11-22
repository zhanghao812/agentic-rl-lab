r"""用同一套工具与预算评测 Base、SFT 或后续 RL 的 sampler 权重。

在 experiments/spec-o3/ 目录下运行：

uv run python eval.py \
    --output outputs/base-dev \
    --max-tokens 6144 \
    --max-seq-len 16384

uv run python eval.py \
    --model-path '<训练终端打印的 sampler_path>' \
    --output outputs/sft-dev \
    --max-tokens 6144 \
    --max-seq-len 16384

uv run python eval.py \
    --base-model Qwen/Qwen3.8-27B \
    --model-path '<训练终端打印的 sampler_path>' \
    --output outputs/rl-27b-sft \
    --max-tokens 6144 \
    --max-seq-len 16384

默认使用 datasets/rl/bench_dev.jsonl；输出分类指标、完整轨迹和工具图片。
--base-model 默认 Qwen/Qwen3.5-4B，必须与 sampler 权重的基模一致。
未指定 --model-revision 时，4B 沿用固定版本，其他模型使用 main。
不传 --model-path 时，仅为 Base 追加最终答案格式提醒；SFT/RL 使用原始提示词。
分类答案取最后一个 </think> 后的最后一个完整 answer 块；格式合规率单独统计。
进度条按已完成的样本数统计，每批评测结束后更新。
"""

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：从 dotenv 导入 load_dotenv。
from dotenv import load_dotenv
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm
# 逐行讲解：从 transformers 导入 AutoImageProcessor, AutoTokenizer。
from transformers import AutoImageProcessor, AutoTokenizer

# 逐行讲解：从 protocol 导入 BASE_MODEL, MODEL_REVISION。
from protocol import BASE_MODEL, MODEL_REVISION
# 逐行讲解：从 rollout 导入 rollout。
from rollout import rollout


# 逐行讲解：定义函数 extract_answer：只读思考结束后的正文，取最后一个完整答案块中的 YES/NO。
def extract_answer(text):
    """只读思考结束后的正文，取最后一个完整答案块中的 YES/NO。"""
    # 逐行讲解：调用 text.rpartition（text.rpartition('</think>')），赋给 (_, thinking_end, body)。
    _, thinking_end, body = text.rpartition("</think>")
    # 逐行讲解：检查条件 not thinking_end；成立时执行下方分支。
    if not thinking_end:
        # 逐行讲解：返回结果：None。
        return None

    # 逐行讲解：调用 re.findall（re.findall('<answer>\\s*\\\\boxed\\{(YES|NO)\\}.*?</answer>', body, re.S)），赋给 answers。
    answers = re.findall(
        r"<answer>\s*\\boxed\{(YES|NO)\}.*?</answer>",
        body,
        re.S,
    )
    # 逐行讲解：返回结果：answers[-1] if answers else None。
    return answers[-1] if answers else None


# 逐行讲解：定义函数 classification_metrics：先计算各任务的二分类指标，再汇总整体表现与工具使用情况。
def classification_metrics(results):
    """先计算各任务的二分类指标，再汇总整体表现与工具使用情况。"""
    # 逐行讲解：构造字典 {}，赋给 metrics。
    metrics = {}

    # 逐行讲解：遍历 sorted({row['task'] for row in results})，将每项绑定到 task。
    for task in sorted({row["task"] for row in results}):
        # 逐行讲解：通过推导式逐项生成结果：[row for row in results if row['task'] == task]，赋给 rows。
        rows = [row for row in results if row["task"] == task]
        # 逐行讲解：求和（sum((row['label'] == 'YES' and row['prediction'] == 'YES' for row in rows))），赋给 tp。
        tp = sum(
            row["label"] == "YES" and row["prediction"] == "YES"
            for row in rows
        )
        # 逐行讲解：求和（sum((row['label'] == 'NO' and row['prediction'] == 'YES' for row in rows))），赋给 fp。
        fp = sum(
            row["label"] == "NO" and row["prediction"] == "YES"
            for row in rows
        )
        # 未给出有效答案时，正类样本也计入漏检。
        fn = sum(
            row["label"] == "YES" and row["prediction"] != "YES"
            for row in rows
        )

        # 没有预测正类时 precision 记 0；这是分类指标的 zero-division 定义。
        metrics[task] = {
            "count": len(rows),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy": sum(row["label"] == row["prediction"] for row in rows) / len(rows),
            "format_rate": sum(row["format_ok"] for row in rows) / len(rows),
        }

    # 逐行讲解：通过推导式逐项生成结果：[turn['observation'] for row in results for turn in row['turns'] if 'observation' in turn]，赋给 observations。
    observations = [
        turn["observation"]
        for row in results
        for turn in row["turns"]
        if "observation" in turn
    ]

    # 逐行讲解：返回结果：{'tasks': metrics, 'macro_f1': sum((item['f1'] for item in metrics.values())) / len(metrics), 'accuracy': s…。
    return {
        "tasks": metrics,
        "macro_f1": sum(item["f1"] for item in metrics.values()) / len(metrics),
        "accuracy": sum(row["label"] == row["prediction"] for row in results) / len(results),
        "format_rate": sum(row["format_ok"] for row in results) / len(results),
        "mean_turns": sum(len(row["turns"]) for row in results) / len(results),
        "tool_calls": len(observations),
        "tool_successes": sum(obs["ok"] for obs in observations),
    }


# 逐行讲解：定义异步函数 main：封装下方逻辑，供调用方复用。
async def main(args):
    # 逐行讲解：调用 load_dotenv（load_dotenv()）。
    load_dotenv()
    # 逐行讲解：检查条件 args.model_revision is None；成立时执行下方分支。
    if args.model_revision is None:
        # 逐行讲解：按条件选择结果：MODEL_REVISION if args.base_model == BASE_MODEL else 'main'，赋给 args.model_revision。
        args.model_revision = MODEL_REVISION if args.base_model == BASE_MODEL else "main"

    # 1. 读取评测集，创建与训练模板匹配的 tokenizer 和图像处理器。
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    # Base 额外强调最终答案格式；只修改本次评测的消息，不改数据文件。
    if args.model_path is None:
        # 逐行讲解：遍历 rows，将每项绑定到 row。
        for row in rows:
            # 逐行讲解：向列表末尾追加一个元素（row['messages'][-1]['content'].append({'type': 'text', 'text': '\n\nFor your final answer, after thinking, …）。
            row["messages"][-1]["content"].append({
                "type": "text",
                "text": (
                    "\n\nFor your final answer, after thinking, output only "
                    r"<answer>\boxed{YES}your justification</answer> or "
                    r"<answer>\boxed{NO}your justification</answer>, "
                    "with all justification inside the answer tags and no text outside them."
                ),
            })

    # 逐行讲解：调用 AutoTokenizer.from_pretrained（AutoTokenizer.from_pretrained(args.base_model, revision=args.model_revision)），赋给 tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.model_revision,
    )
    # 逐行讲解：调用 AutoImageProcessor.from_pretrained（AutoImageProcessor.from_pretrained(args.base_model, revision=args.model_revision, backend='pil')），赋给 processor。
    processor = AutoImageProcessor.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        backend="pil",
    )

    # 2. model_path 留空使用 Base；传入时加载对应 sampler 权重。
    service = trio.ServiceClient()
    # 逐行讲解：调用 service.create_sampling_client（service.create_sampling_client(base_model=args.base_model, model_path=args.model_path)），赋给 sampler。
    sampler = service.create_sampling_client(
        base_model=args.base_model,
        model_path=args.model_path,
    )
    # 逐行讲解：创建目录（args.output.mkdir(parents=True, exist_ok=True)）。
    args.output.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：构造列表 []，赋给 results。
    results = []

    # 3. 按 concurrency 分批并发执行完整工具交互，逐批写入轨迹。
    with (
        (args.output / "trajectories.jsonl").open("w") as file,
        tqdm(total=len(rows), desc="Evaluating", unit="sample") as progress,
    ):
        # 逐行讲解：遍历 range(0, len(rows), args.concurrency)，将每项绑定到 start。
        for start in range(0, len(rows), args.concurrency):
            # 逐行讲解：按索引或切片读取 rows[start:start + args.concurrency]，赋给 batch。
            batch = rows[start : start + args.concurrency]
            # 逐行讲解：等待异步操作 asyncio.gather(*(rollout(row, sampler, tokenizer, processor, args.output / 'images' / row['id'], max_turns=… 完成并取得结果，赋给 trajectories。
            trajectories = await asyncio.gather(
                *(
                    rollout(
                        row,
                        sampler,
                        tokenizer,
                        processor,
                        args.output / "images" / row["id"],
                        max_turns=args.max_turns,
                        max_tokens=args.max_tokens,
                        max_seq_len=args.max_seq_len,
                        temperature=args.temperature,
                        seed=args.seed + start + index,
                    )
                    for index, row in enumerate(batch)
                )
            )

            # 逐行讲解：遍历 trajectories，将每项绑定到 result。
            for result in trajectories:
                # PNG 单独保存；JSONL 保留逐轮文本、原始采样 token/logprob 和动作 mask。
                record = {
                    key: value
                    for key, value in result.items()
                    if key != "model_input"
                }
                # 分类读取最后一轮的答案；format_ok 仍记录严格动作格式。
                final_text = record["turns"][-1]["text"] if record["turns"] else ""
                # 逐行讲解：调用 extract_answer（extract_answer(final_text)），赋给 record['prediction']。
                record["prediction"] = extract_answer(final_text)
                # 逐行讲解：调用 file.write（file.write(json.dumps(record, ensure_ascii=False) + '\n')）。
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                # 逐行讲解：向列表末尾追加一个元素（results.append({key: record[key] for key in ('task', 'label', 'prediction', 'format_ok', 'turns')})）。
                results.append({
                    key: record[key]
                    for key in ("task", "label", "prediction", "format_ok", "turns")
                })

            # 逐行讲解：调用 file.flush（file.flush()）。
            file.flush()
            # 逐行讲解：调用 progress.update（progress.update(len(trajectories))）。
            progress.update(len(trajectories))

    # 4. 汇总各任务指标与整体指标，保存本次评测参数。
    metrics = classification_metrics(results)
    # 逐行讲解：构造字典 {**vars(args), 'data': str(args.data), 'output': str(args.output), 'base_format_reminder': args.model_path …，赋给 config。
    config = {
        **vars(args),
        "data": str(args.data),
        "output": str(args.output),
        "base_format_reminder": args.model_path is None,
        "answer_extraction": "last_answer_after_last_think",
    }
    # 逐行讲解：将文本写入文件（(args.output / 'metrics.json').write_text(json.dumps({'config': config, **metrics}, ensure_ascii=False, ind…）。
    (args.output / "metrics.json").write_text(
        json.dumps({"config": config, **metrics}, ensure_ascii=False, indent=2)
    )
    # 逐行讲解：调用 print（print(json.dumps(metrics, ensure_ascii=False, indent=2))）。
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)

    # 数据、模型和评测产物。
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "datasets/rl/bench_dev.jsonl",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default=BASE_MODEL, help='评测使用的 Qwen3.5 基座模型，须与 sampler 权重的基模一致')）。
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="评测使用的 Qwen3.5 基座模型，须与 sampler 权重的基模一致",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--model-revision', help='模型的 HF 文件版本（commit、tag 或分支）；默认 4B 固定版本，其他模型 main')）。
    parser.add_argument(
        "--model-revision",
        help="模型的 HF 文件版本（commit、tag 或分支）；默认 4B 固定版本，其他模型 main",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--model-path', help='留空评测 Base；SFT/RL 传入 sampler_path')）。
    parser.add_argument(
        "--model-path",
        help="留空评测 Base；SFT/RL 传入 sampler_path",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, required=True)）。
    parser.add_argument("--output", type=Path, required=True)

    # 工具交互预算与采样参数。
    parser.add_argument("--concurrency", type=int, default=16)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-turns', type=int, default=8)）。
    parser.add_argument("--max-turns", type=int, default=8)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=2048)）。
    parser.add_argument("--max-tokens", type=int, default=2048)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-seq-len', type=int, default=16384)）。
    parser.add_argument("--max-seq-len", type=int, default=16384)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=0.6)）。
    parser.add_argument("--temperature", type=float, default=0.6)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)

    # 逐行讲解：调用 asyncio.run（asyncio.run(main(parser.parse_args()))）。
    asyncio.run(main(parser.parse_args()))
