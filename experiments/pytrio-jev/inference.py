"""受限 softmax 推理：一次前向读出全部字段的候选分布。

不生成文本——把含 marker 骨架的完整 prompt 发给 sample()，靠 topk_prompt_logprobs
在每个 marker 位置读"下一个 token"的 top-k 分布，过滤到该字段候选符号后归一化。

用法：
uv run python experiments/pytrio-jev/inference.py --weights trio://...   # 不传则评 base
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio

# 逐行讲解：从 data 导入 Encoder, load_jsonl。
from data import Encoder, load_jsonl

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：取值 'Qwen/Qwen3.5-4B'，赋给 DEFAULT_MODEL。
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


# 逐行讲解：定义函数 _field_logprobs：从一次前向的 topk_prompt_logprobs 提取各字段候选符号的 logprob。
def _field_logprobs(encoder: Encoder, encoded: dict, response) -> dict:
    """从一次前向的 topk_prompt_logprobs 提取各字段候选符号的 logprob。

    topk_prompt_logprobs[m] 是"给定前 m 个 token，第 m 个 token 的分布"，
    marker 在第 m 位，所以该位置就是训练时监督的预测点。
    候选未进 top-k 时按 floor = min(topk) - 10 近似（约等于概率 0）。
    """
    # 逐行讲解：断言 response.input_tokens == len(encoded['ids'])；条件不成立时报告错误。
    assert response.input_tokens == len(encoded["ids"]), "本地/远端 prompt 长度不一致"
    # 逐行讲解：声明 fields 的类型为 dict[str, dict[str, float]]；构造字典 {}。
    fields: dict[str, dict[str, float]] = {}
    # 逐行讲解：遍历 encoded['markers'].items()，将每项绑定到 (field, position)。
    for field, position in encoded["markers"].items():
        # 逐行讲解：调用 dict（dict(response.topk_prompt_logprobs[position])），赋给 topk。
        topk = dict(response.topk_prompt_logprobs[position])
        # 逐行讲解：计算 min(topk.values()) - 10.0，赋给 floor。
        floor = min(topk.values()) - 10.0
        # 逐行讲解：通过推导式逐项生成结果：{symbol: topk.get(encoder.symbol_token_id(symbol), floor) for symbol, _, _ in encoded['options'][field]}，赋给 fields[field]。
        fields[field] = {
            symbol: topk.get(encoder.symbol_token_id(symbol), floor)
            for symbol, _, _ in encoded["options"][field]
        }
    # 逐行讲解：返回结果：{'encoded': encoded, 'logprobs': fields}。
    return {"encoded": encoded, "logprobs": fields}


# 逐行讲解：定义函数 candidate_logprobs：一次前向，返回 {field: {symbol: logprob}}（未归一、未加温度）。
def candidate_logprobs(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    record: dict,
) -> dict:
    """一次前向，返回 {field: {symbol: logprob}}（未归一、未加温度）。"""
    # 逐行讲解：将文本编码为 token（encoder.encode(record, include_targets=False)），赋给 encoded。
    encoded = encoder.encode(record, include_targets=False)
    # 逐行讲解：调用 sampling_client.sample(prompt=trio.ModelInput.from_ints(encoded['ids']), num_samples=1, sampling_params=tri…（sampling_client.sample(prompt=trio.ModelInput.from_ints(encoded['ids']), num_samples=1, sampling_params=tri…），赋给 response。
    response = sampling_client.sample(
        prompt=trio.ModelInput.from_ints(encoded["ids"]),
        num_samples=1,
        sampling_params=trio.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=max(len(opts) for opts in encoded["options"].values()),
    ).result()
    # 逐行讲解：返回结果：_field_logprobs(encoder, encoded, response)。
    return _field_logprobs(encoder, encoded, response)


# 逐行讲解：定义异步函数 candidate_logprobs_async：candidate_logprobs 的异步版，供并发评测/校准使用。
async def candidate_logprobs_async(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    record: dict,
) -> dict:
    """candidate_logprobs 的异步版，供并发评测/校准使用。"""
    # 逐行讲解：将文本编码为 token（encoder.encode(record, include_targets=False)），赋给 encoded。
    encoded = encoder.encode(record, include_targets=False)
    # 逐行讲解：等待异步操作 sampling_client.sample_async(prompt=trio.ModelInput.from_ints(encoded['ids']), num_samples=1, sampling_para… 完成并取得结果，赋给 response。
    response = await sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(encoded["ids"]),
        num_samples=1,
        sampling_params=trio.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=max(len(opts) for opts in encoded["options"].values()),
    )
    # 逐行讲解：返回结果：_field_logprobs(encoder, encoded, response)。
    return _field_logprobs(encoder, encoded, response)


# 逐行讲解：定义函数 softmax：封装下方逻辑，供调用方复用。
def softmax(logprobs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    # 逐行讲解：计算 logprobs / temperature，赋给 z。
    z = logprobs / temperature
    # 逐行讲解：计算 z - z.max()，赋给 z。
    z = z - z.max()
    # 逐行讲解：通过指数运算将对数值转换回原尺度（np.exp(z)），赋给 p。
    p = np.exp(z)
    # 逐行讲解：返回结果：p / p.sum()。
    return p / p.sum()


# 逐行讲解：定义函数 _build_decisions：候选 logprob → 各字段的分布与预测。
def _build_decisions(
    record: dict, encoded: dict, logprobs: dict, temperature: float
) -> dict[str, dict]:
    """候选 logprob → 各字段的分布与预测。

    choice → {value: prob} + argmax；noul → p_yes；score → 概率加权的期望分。
    """
    # 逐行讲解：声明 out 的类型为 dict[str, dict]；构造字典 {}。
    out: dict[str, dict] = {}
    # 逐行讲解：遍历 encoded['options'].items()，将每项绑定到 (field, options)。
    for field, options in encoded["options"].items():
        # 逐行讲解：通过推导式逐项生成结果：[symbol for symbol, _, _ in options]，赋给 symbols。
        symbols = [symbol for symbol, _, _ in options]
        # 逐行讲解：调用 softmax（softmax(np.array([logprobs[field][s] for s in symbols]), temperature)），赋给 probs。
        probs = softmax(
            np.array([logprobs[field][s] for s in symbols]), temperature
        )
        # 逐行讲解：通过推导式逐项生成结果：{value: float(p) for (_, value, _), p in zip(options, probs)}，赋给 by_value。
        by_value = {value: float(p) for (_, value, _), p in zip(options, probs)}
        # 逐行讲解：按索引或切片读取 record['questions'][field]，赋给 question。
        question = record["questions"][field]
        # 逐行讲解：按索引或切片读取 question['type']，赋给 kind。
        kind = question["type"]
        # 逐行讲解：声明 result 的类型为 dict；构造字典 {'type': kind, 'probabilities': by_value}。
        result: dict = {"type": kind, "probabilities": by_value}
        # 逐行讲解：调用 max（max(by_value, key=by_value.get)），赋给 best。
        best = max(by_value, key=by_value.get)
        # 逐行讲解：检查条件 kind == 'noul'；成立时执行下方分支。
        if kind == "noul":
            # 逐行讲解：按索引或切片读取 by_value['yes']，赋给 result['p_yes']。
            result["p_yes"] = by_value["yes"]
            # 逐行讲解：按条件选择结果：'yes' if by_value['yes'] >= by_value['no'] else 'no'，赋给 result['prediction']。
            result["prediction"] = "yes" if by_value["yes"] >= by_value["no"] else "no"
        # 逐行讲解：检查条件 kind == 'score'；成立时执行下方分支。
        elif kind == "score":
            # 逐行讲解：转换为浮点表示（float(sum((int(value) * p for value, p in by_value.items())))），赋给 result['expected_score']。
            result["expected_score"] = float(
                sum(int(value) * p for value, p in by_value.items())
            )
            # 逐行讲解：取值 best，赋给 result['prediction']。
            result["prediction"] = best
        else:
            # 逐行讲解：取值 best，赋给 result['prediction']。
            result["prediction"] = best
        # 逐行讲解：取值 result，赋给 out[field]。
        out[field] = result
    # 逐行讲解：返回结果：out。
    return out


# 逐行讲解：定义函数 decide：对一条记录输出各字段的分布与预测（同步版）。
def decide(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    record: dict,
    temperature: float = 1.0,
) -> dict[str, dict]:
    """对一条记录输出各字段的分布与预测（同步版）。"""
    # 逐行讲解：调用 candidate_logprobs（candidate_logprobs(sampling_client, encoder, record)），赋给 collected。
    collected = candidate_logprobs(sampling_client, encoder, record)
    # 逐行讲解：返回结果：_build_decisions(record, collected['encoded'], collected['logprobs'], temperature)。
    return _build_decisions(
        record, collected["encoded"], collected["logprobs"], temperature
    )


# 逐行讲解：定义异步函数 decide_async：对一条记录输出各字段的分布与预测（异步版）。
async def decide_async(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    record: dict,
    temperature: float = 1.0,
) -> dict[str, dict]:
    """对一条记录输出各字段的分布与预测（异步版）。"""
    # 逐行讲解：等待异步操作 candidate_logprobs_async(sampling_client, encoder, record) 完成并取得结果，赋给 collected。
    collected = await candidate_logprobs_async(sampling_client, encoder, record)
    # 逐行讲解：返回结果：_build_decisions(record, collected['encoded'], collected['logprobs'], temperature)。
    return _build_decisions(
        record, collected["encoded"], collected["logprobs"], temperature
    )


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='候选符号受限 softmax 推理')），赋给 parser。
    parser = argparse.ArgumentParser(description="候选符号受限 softmax 推理")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default=DEFAULT_MODEL)）。
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--weights', default=None, help='trio:// LoRA 权重路径，不传则评 base')）。
    parser.add_argument("--weights", default=None, help="trio:// LoRA 权重路径，不传则评 base")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0)）。
    parser.add_argument("--temperature", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--input', type=Path, default=None, help='canonical JSONL，默认用内置样例')）。
    parser.add_argument("--input", type=Path, default=None, help="canonical JSONL，默认用内置样例")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--limit', type=int, default=3)）。
    parser.add_argument("--limit", type=int, default=3)
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()

    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：调用 service_client.create_sampling_client（service_client.create_sampling_client(base_model=args.base_model, model_path=args.weights)），赋给 sampling_client。
    sampling_client = service_client.create_sampling_client(
        base_model=args.base_model, model_path=args.weights
    )
    # 逐行讲解：调用 Encoder（Encoder(sampling_client.get_tokenizer())），赋给 encoder。
    encoder = Encoder(sampling_client.get_tokenizer())

    # 逐行讲解：检查条件 args.input；成立时执行下方分支。
    if args.input:
        # 逐行讲解：按索引或切片读取 load_jsonl(args.input)[:args.limit]，赋给 records。
        records = load_jsonl(args.input)[: args.limit]
    else:
        # 逐行讲解：构造列表 [{'id': 'demo-1', 'state': {'customer_message': 'I bought headphones yesterday and noticed today that I was…，赋给 records。
        records = [
            {
                "id": "demo-1",
                "state": {
                    "customer_message": "I bought headphones yesterday and noticed "
                    "today that I was charged twice. Please refund the extra charge "
                    "as soon as possible."
                },
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle this ticket?",
                        "criteria": {
                            "billing": "Payments and refunds",
                            "delivery": "Shipping and delivery",
                            "technical": "Product technical support",
                        },
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "Choose the priority for handling this ticket.",
                        "criteria": [
                            "Low: general inquiry",
                            "Medium: an issue requiring follow-up",
                            "High: a financial issue such as a duplicate charge",
                        ],
                    },
                    "refund_requested": {
                        "type": "noul",
                        "instructions": "Does the customer explicitly request a refund?",
                    },
                },
            }
        ]

    # 逐行讲解：遍历 records，将每项绑定到 record。
    for record in records:
        # 逐行讲解：调用 decide（decide(sampling_client, encoder, record, args.temperature)），赋给 result。
        result = decide(sampling_client, encoder, record, args.temperature)
        # 逐行讲解：调用 print（print(f"\n=== {record['id']} ===")）。
        print(f"\n=== {record['id']} ===")
        # 逐行讲解：调用 print（print(json.dumps(result, ensure_ascii=False, indent=2))）。
        print(json.dumps(result, ensure_ascii=False, indent=2))


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
