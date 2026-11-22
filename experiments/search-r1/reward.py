"""计算 Search-R1 的格式奖励和答案精确匹配奖励。"""

# 逐行讲解：导入模块：re。
import re
# 逐行讲解：导入模块：unicodedata。
import unicodedata
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass


# 逐行讲解：调用 re.compile（re.compile('^\\s*Answer:\\s*(.*?)\\s*$', re.IGNORECASE | re.MULTILINE)），赋给 ANSWER_PATTERN。
ANSWER_PATTERN = re.compile(r"^\s*Answer:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
# 逐行讲解：调用 re.compile（re.compile('\\b(a|an|the)\\b', re.IGNORECASE)），赋给 ARTICLE_PATTERN。
ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


# 逐行讲解：定义类 RewardResult：保存最终 reward 及其判定细节。
@dataclass(frozen=True)
class RewardResult:
    """保存最终 reward 及其判定细节。"""

    # 逐行讲解：声明 reward 的类型为 float。
    reward: float
    # 逐行讲解：声明 valid_format 的类型为 bool。
    valid_format: bool
    # 逐行讲解：声明 exact_match 的类型为 bool。
    exact_match: bool
    # 逐行讲解：声明 answer 的类型为 str | None。
    answer: str | None


# 逐行讲解：定义函数 normalize_answer：统一答案格式：转小写、去除标点和英文冠词，并合并多余空格。
def normalize_answer(text: str) -> str:
    """统一答案格式：转小写、去除标点和英文冠词，并合并多余空格。"""
    # 逐行讲解：调用 text.lower（text.lower()），赋给 lowered。
    lowered = text.lower()
    # 逐行讲解：调用 ''.join（''.join((char for char in lowered if not unicodedata.category(char).startswith('P')))），赋给 without_punctuation。
    without_punctuation = "".join(
        char for char in lowered if not unicodedata.category(char).startswith("P")
    )
    # 逐行讲解：调用 ARTICLE_PATTERN.sub（ARTICLE_PATTERN.sub(' ', without_punctuation)），赋给 without_articles。
    without_articles = ARTICLE_PATTERN.sub(" ", without_punctuation)
    # 逐行讲解：返回结果：' '.join(without_articles.split())。
    return " ".join(without_articles.split())


# 逐行讲解：定义函数 extract_answer：只接受恰好一行非空的 Answer: 最终答案。
def extract_answer(text: str) -> str | None:
    """只接受恰好一行非空的 Answer: 最终答案。"""
    # 逐行讲解：调用 ANSWER_PATTERN.findall（ANSWER_PATTERN.findall(text)），赋给 matches。
    matches = ANSWER_PATTERN.findall(text)
    # 逐行讲解：检查条件 len(matches) != 1；成立时执行下方分支。
    if len(matches) != 1:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：调用 matches[0].strip（matches[0].strip()），赋给 answer。
    answer = matches[0].strip()
    # 逐行讲解：返回结果：answer or None。
    return answer or None


# 逐行讲解：定义函数 score_answer：按 1.0、0.0、-0.1 三档规则计算轨迹奖励。
def score_answer(text: str, references: list[str]) -> RewardResult:
    """按 1.0、0.0、-0.1 三档规则计算轨迹奖励。"""
    # 逐行讲解：调用 extract_answer（extract_answer(text)），赋给 answer。
    answer = extract_answer(text)
    # 逐行讲解：检查条件 answer is None；成立时执行下方分支。
    if answer is None:
        # 逐行讲解：返回结果：RewardResult(-0.1, False, False, None)。
        return RewardResult(-0.1, False, False, None)
    # 逐行讲解：调用 normalize_answer（normalize_answer(answer)），赋给 normalized。
    normalized = normalize_answer(answer)
    # 逐行讲解：调用 any（any((normalized == normalize_answer(reference) for reference in references))），赋给 exact_match。
    exact_match = any(normalized == normalize_answer(reference) for reference in references)
    # 逐行讲解：返回结果：RewardResult(float(exact_match), True, exact_match, answer)。
    return RewardResult(float(exact_match), True, exact_match, answer)
