"""计算 ReTool 的结果奖励：抽取最后一个 \\boxed{} 并与参考答案做数学等价判定。

对齐官方 recipe 的 math_dapo.compute_score(strict_box_verify=True)：
只取回答末尾 300 个字符、提取最后一个 \\boxed{}，对 +1 / 错 −1（格式非法也算错）。
等价判定使用 math_verify（verl 生态实际使用的库）。
"""

# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass

# 逐行讲解：从 math_verify 导入 parse, verify。
from math_verify import parse, verify


# 逐行讲解：取值 300，赋给 ANSWER_WINDOW_CHARS。
ANSWER_WINDOW_CHARS = 300  # 官方实现只检查回答末尾 300 个字符。


# 逐行讲解：定义类 RewardResult：保存最终 reward 及其判定细节。
@dataclass(frozen=True)
class RewardResult:
    """保存最终 reward 及其判定细节。"""

    # 逐行讲解：声明 reward 的类型为 float。
    reward: float  # +1.0（答案正确）或 -1.0（错误、含格式非法）。
    # 逐行讲解：声明 correct 的类型为 bool。
    correct: bool  # 答案是否通过数学等价判定。
    # 逐行讲解：声明 valid_format 的类型为 bool。
    valid_format: bool  # 是否找到合法的 \boxed{} 答案。
    # 逐行讲解：声明 answer 的类型为 str | None。
    answer: str | None  # 抽取到的答案原文，用于记录和调试。


# 逐行讲解：定义函数 extract_last_boxed：提取最后一个 \boxed{...} 的内容，按花括号配平。
def extract_last_boxed(text: str) -> str | None:
    """提取最后一个 \\boxed{...} 的内容，按花括号配平。"""
    # 逐行讲解：取值 '\\boxed{'，赋给 marker。
    marker = "\\boxed{"
    # 逐行讲解：调用 text.rfind（text.rfind(marker)），赋给 index。
    index = text.rfind(marker)
    # 逐行讲解：检查条件 index < 0；成立时执行下方分支。
    if index < 0:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：计算 index + len(marker)，赋给 start。
    start = index + len(marker)
    # 逐行讲解：取值 1，赋给 depth。
    depth = 1
    # 逐行讲解：遍历 range(start, len(text))，将每项绑定到 position。
    for position in range(start, len(text)):
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
                # 逐行讲解：返回结果：text[start:position]。
                return text[start:position]
    # 逐行讲解：返回结果：None。
    return None


# 逐行讲解：定义函数 answers_equivalent：用 math_verify 判定预测答案是否与参考答案数学等价。
def answers_equivalent(prediction: str, reference: str) -> bool:
    """用 math_verify 判定预测答案是否与参考答案数学等价。"""
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：返回结果：bool(verify(parse(f'${reference}$'), parse(f'${prediction}$')))。
        return bool(verify(parse(f"${reference}$"), parse(f"${prediction}$")))
    except Exception:  # math_verify 对怪异输入可能抛异常，一律按不等价处理
        # 逐行讲解：返回结果：False。
        return False


# 逐行讲解：定义函数 score_answer：按官方 ±1 规则给整条轨迹的最终文本打分。
def score_answer(text: str, reference: str) -> RewardResult:
    """按官方 ±1 规则给整条轨迹的最终文本打分。"""
    # 逐行讲解：调用 extract_last_boxed（extract_last_boxed(text[-ANSWER_WINDOW_CHARS:])），赋给 answer。
    answer = extract_last_boxed(text[-ANSWER_WINDOW_CHARS:])
    # 逐行讲解：检查条件 answer is None；成立时执行下方分支。
    if answer is None:
        # 逐行讲解：返回结果：RewardResult(-1.0, False, False, None)。
        return RewardResult(-1.0, False, False, None)
    # 逐行讲解：调用 answers_equivalent（answers_equivalent(answer.strip(), reference.strip())），赋给 correct。
    correct = answers_equivalent(answer.strip(), reference.strip())
    # 逐行讲解：返回结果：RewardResult(1.0 if correct else -1.0, correct, True, answer)。
    return RewardResult(1.0 if correct else -1.0, correct, True, answer)
