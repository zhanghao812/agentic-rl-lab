"""读取 prepare_data.py 生成的本地 Search-R1 数据。"""

# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：random。
import random
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path


# 逐行讲解：定义类 SearchExample：保存一道搜索问答题及其标准答案。
@dataclass(frozen=True)
class SearchExample:
    """保存一道搜索问答题及其标准答案。"""

    # 逐行讲解：声明 id 的类型为 str。
    id: str
    # 逐行讲解：声明 question 的类型为 str。
    question: str
    # 逐行讲解：声明 answers 的类型为 list[str]。
    answers: list[str]
    # 逐行讲解：声明 data_source 的类型为 str。
    data_source: str


# 逐行讲解：定义函数 load_examples：从本地 JSONL 文件读取全部样本。
def load_examples(path: str | Path) -> list[SearchExample]:
    """从本地 JSONL 文件读取全部样本。"""
    # 逐行讲解：声明 examples 的类型为 list[SearchExample]；构造列表 []。
    examples: list[SearchExample] = []
    # 逐行讲解：进入上下文管理器 Path(path).open(encoding='utf-8')，退出时执行其收尾逻辑。
    with Path(path).open(encoding="utf-8") as file:
        # 逐行讲解：遍历 file，将每项绑定到 line。
        for line in file:
            # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(line)），赋给 row。
            row = json.loads(line)
            # 逐行讲解：向列表末尾追加一个元素（examples.append(SearchExample(id=str(row['id']), question=row['question'], answers=list(row['answers']), da…）。
            examples.append(
                SearchExample(
                    id=str(row["id"]),
                    question=row["question"],
                    answers=list(row["answers"]),
                    data_source=row["data_source"],
                )
            )
    # 逐行讲解：返回结果：examples。
    return examples


# 逐行讲解：定义函数 shuffled_examples：读取并按固定随机种子打乱训练样本。
def shuffled_examples(path: str | Path, seed: int) -> list[SearchExample]:
    """读取并按固定随机种子打乱训练样本。"""
    # 逐行讲解：调用 load_examples（load_examples(path)），赋给 examples。
    examples = load_examples(path)
    # 逐行讲解：打乱数据顺序（random.Random(seed).shuffle(examples)）。
    random.Random(seed).shuffle(examples)
    # 逐行讲解：返回结果：examples。
    return examples


# 逐行讲解：定义函数 take_batch：循环取出一个固定大小的训练问题批次。
def take_batch(examples: list[SearchExample], start: int, batch_size: int) -> list[SearchExample]:
    """循环取出一个固定大小的训练问题批次。"""
    # 逐行讲解：检查条件 not examples；成立时执行下方分支。
    if not examples:
        # 逐行讲解：返回结果：[]。
        return []
    # 逐行讲解：返回结果：[examples[(start + offset) % len(examples)] for offset in range(batch_size)]。
    return [examples[(start + offset) % len(examples)] for offset in range(batch_size)]
