"""DAPO 数学正确性奖励与 Soft Overlong Punishment。"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 math_verify 导入 parse, verify。
from math_verify import parse, verify


# 逐行讲解：取值 300，赋给 ANSWER_WINDOW_CHARS。
ANSWER_WINDOW_CHARS = 300


# 逐行讲解：定义类 RewardResult：一条 completion 的可解释奖励。
@dataclass(frozen=True)
class RewardResult:
    """一条 completion 的可解释奖励。"""

    # 逐行讲解：声明 base_reward 的类型为 float。
    base_reward: float
    # 逐行讲解：声明 length_penalty 的类型为 float。
    length_penalty: float
    # 逐行讲解：声明 shaped_reward 的类型为 float。
    shaped_reward: float
    # 逐行讲解：声明 correct 的类型为 bool。
    correct: bool
    # 逐行讲解：声明 valid_format 的类型为 bool。
    valid_format: bool
    # 逐行讲解：声明 answer 的类型为 str | None。
    answer: str | None
    # 逐行讲解：声明 completion_tokens 的类型为 int。
    completion_tokens: int


# 逐行讲解：定义函数 extract_last_boxed：从末尾向前找到最后一个完整的 ``\boxed{...}``，支持嵌套括号。
def extract_last_boxed(text: str) -> str | None:
    """从末尾向前找到最后一个完整的 ``\\boxed{...}``，支持嵌套括号。"""
    # 逐行讲解：调用 len（len(text)），赋给 end。
    end = len(text)
    # 逐行讲解：当 True 成立时继续循环。
    while True:
        # 逐行讲解：调用 text.rfind（text.rfind('\\boxed', 0, end)），赋给 marker。
        marker = text.rfind("\\boxed", 0, end)
        # 逐行讲解：检查条件 marker < 0；成立时执行下方分支。
        if marker < 0:
            # 逐行讲解：返回结果：None。
            return None
        # 逐行讲解：调用 text.find（text.find('{', marker)），赋给 left。
        left = text.find("{", marker)
        # 逐行讲解：检查条件 left < 0；成立时执行下方分支。
        if left < 0:
            # 逐行讲解：取值 marker，赋给 end。
            end = marker
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：取值 0，赋给 depth。
        depth = 0
        # 逐行讲解：遍历 range(left, len(text))，将每项绑定到 position。
        for position in range(left, len(text)):
            # 逐行讲解：检查条件 text[position] == '{'；成立时执行下方分支。
            if text[position] == "{":
                # 逐行讲解：更新 depth：depth += 1。
                depth += 1
            # 逐行讲解：检查条件 text[position] == '}'；成立时执行下方分支。
            elif text[position] == "}":
                # 逐行讲解：更新 depth：depth -= 1。
                depth -= 1
                # 逐行讲解：检查条件 depth == 0；成立时执行下方分支。
                if depth == 0:
                    # 逐行讲解：返回结果：text[left + 1:position].strip()。
                    return text[left + 1 : position].strip()
        # 逐行讲解：取值 marker，赋给 end。
        end = marker


# 逐行讲解：定义函数 answers_equivalent：使用 ``math_verify`` 判断两个数学答案是否等价。
def answers_equivalent(prediction: str, reference: str) -> bool:
    """使用 ``math_verify`` 判断两个数学答案是否等价。"""
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：返回结果：bool(verify(parse(f'${reference}$'), parse(f'${prediction}$')))。
        return bool(verify(parse(f"${reference}$"), parse(f"${prediction}$")))
    except Exception:
        # 逐行讲解：返回结果：False。
        return False


# 逐行讲解：定义函数 soft_overlong_penalty：按 DAPO 的分段线性规则返回 ``[-1, 0]`` 长度惩罚。
def soft_overlong_penalty(
    completion_tokens: int,
    *,
    max_tokens: int,
    overlong_cache: int,
) -> float:
    """按 DAPO 的分段线性规则返回 ``[-1, 0]`` 长度惩罚。"""
    # 逐行讲解：检查条件 completion_tokens < 0；成立时执行下方分支。
    if completion_tokens < 0:
        # 逐行讲解：抛出异常：ValueError('completion_tokens must be >= 0')。
        raise ValueError("completion_tokens must be >= 0")
    # 逐行讲解：检查条件 max_tokens < 1；成立时执行下方分支。
    if max_tokens < 1:
        # 逐行讲解：抛出异常：ValueError('max_tokens must be >= 1')。
        raise ValueError("max_tokens must be >= 1")
    # 逐行讲解：检查条件 not 1 <= overlong_cache <= max_tokens；成立时执行下方分支。
    if not 1 <= overlong_cache <= max_tokens:
        # 逐行讲解：抛出异常：ValueError('overlong_cache must be in [1, max_tokens]')。
        raise ValueError("overlong_cache must be in [1, max_tokens]")

    # 逐行讲解：计算 max_tokens - overlong_cache，赋给 penalty_start。
    penalty_start = max_tokens - overlong_cache
    # 逐行讲解：检查条件 completion_tokens <= penalty_start；成立时执行下方分支。
    if completion_tokens <= penalty_start:
        # 逐行讲解：返回结果：0.0。
        return 0.0
    # 逐行讲解：检查条件 completion_tokens <= max_tokens；成立时执行下方分支。
    if completion_tokens <= max_tokens:
        # 逐行讲解：返回结果：-(completion_tokens - penalty_start) / overlong_cache。
        return -(completion_tokens - penalty_start) / overlong_cache
    # 逐行讲解：返回结果：-1.0。
    return -1.0


# 逐行讲解：定义函数 score_answer：计算 ±1 正确性奖励，并按 preset 可选叠加长度惩罚。
def score_answer(
    text: str,
    reference: str,
    *,
    completion_tokens: int,
    max_tokens: int,
    overlong_cache: int,
    enable_overlong: bool,
) -> RewardResult:
    """计算 ±1 正确性奖励，并按 preset 可选叠加长度惩罚。"""
    # 逐行讲解：调用 extract_last_boxed（extract_last_boxed(text[-ANSWER_WINDOW_CHARS:])），赋给 answer。
    answer = extract_last_boxed(text[-ANSWER_WINDOW_CHARS:])
    # 逐行讲解：取值 answer is not None and answers_equivalent(answer.strip(), reference.strip())，赋给 correct。
    correct = (
        answer is not None
        and answers_equivalent(answer.strip(), reference.strip())
    )
    # 逐行讲解：按条件选择结果：1.0 if correct else -1.0，赋给 base_reward。
    base_reward = 1.0 if correct else -1.0
    # 逐行讲解：按条件选择结果：soft_overlong_penalty(completion_tokens, max_tokens=max_tokens, overlong_cache=overlong_cache) if enable_ov…，赋给 length_penalty。
    length_penalty = (
        soft_overlong_penalty(
            completion_tokens,
            max_tokens=max_tokens,
            overlong_cache=overlong_cache,
        )
        if enable_overlong
        else 0.0
    )
    # 逐行讲解：返回结果：RewardResult(base_reward=base_reward, length_penalty=length_penalty, shaped_reward=base_reward + length_pen…。
    return RewardResult(
        base_reward=base_reward,
        length_penalty=length_penalty,
        shaped_reward=base_reward + length_penalty,
        correct=correct,
        valid_format=answer is not None,
        answer=answer,
        completion_tokens=completion_tokens,
    )
