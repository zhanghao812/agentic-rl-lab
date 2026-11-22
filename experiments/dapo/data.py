"""读取并循环消费 ``prepare_data.py`` 生成的 DAPO-Math 数据。"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：导入模块：random。
import random


# 逐行讲解：定义类 MathExample：一道数学训练题。
@dataclass(frozen=True)
class MathExample:
    """一道数学训练题。"""

    # 逐行讲解：声明 id 的类型为 str。
    id: str
    # 逐行讲解：声明 question 的类型为 str。
    question: str
    # 逐行讲解：声明 answer 的类型为 str。
    answer: str
    # 逐行讲解：声明 data_source 的类型为 str。
    data_source: str


# 逐行讲解：定义函数 load_examples：读取 JSONL，并对缺字段或空数据快速失败。
def load_examples(path: str | Path) -> list[MathExample]:
    """读取 JSONL，并对缺字段或空数据快速失败。"""
    # 逐行讲解：调用 Path（Path(path)），赋给 source。
    source = Path(path)
    # 逐行讲解：检查条件 not source.is_file()；成立时执行下方分支。
    if not source.is_file():
        # 逐行讲解：抛出异常：FileNotFoundError(f'找不到训练数据：{source}\n请先运行：uv run python experiments/dapo/prepare_data.py')。
        raise FileNotFoundError(
            f"找不到训练数据：{source}\n"
            "请先运行：uv run python experiments/dapo/prepare_data.py"
        )

    # 逐行讲解：声明 examples 的类型为 list[MathExample]；构造列表 []。
    examples: list[MathExample] = []
    # 逐行讲解：进入上下文管理器 source.open(encoding='utf-8')，退出时执行其收尾逻辑。
    with source.open(encoding="utf-8") as file:
        # 逐行讲解：遍历 enumerate(file, start=1)，将每项绑定到 (line_number, line)。
        for line_number, line in enumerate(file, start=1):
            # 逐行讲解：检查条件 not line.strip()；成立时执行下方分支。
            if not line.strip():
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
            try:
                # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(line)），赋给 row。
                row = json.loads(line)
                # 逐行讲解：调用 MathExample（MathExample(id=str(row['id']), question=str(row['question']).strip(), answer=str(row['answer']).strip(), da…），赋给 example。
                example = MathExample(
                    id=str(row["id"]),
                    question=str(row["question"]).strip(),
                    answer=str(row["answer"]).strip(),
                    data_source=str(row.get("data_source") or "math_dapo"),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                # 逐行讲解：抛出异常：ValueError(f'无效数据行：{source}:{line_number}')。
                raise ValueError(f"无效数据行：{source}:{line_number}") from error
            # 逐行讲解：检查条件 not example.question or not example.answer；成立时执行下方分支。
            if not example.question or not example.answer:
                # 逐行讲解：抛出异常：ValueError(f'题目或答案为空：{source}:{line_number}')。
                raise ValueError(f"题目或答案为空：{source}:{line_number}")
            # 逐行讲解：向列表末尾追加一个元素（examples.append(example)）。
            examples.append(example)

    # 逐行讲解：检查条件 not examples；成立时执行下方分支。
    if not examples:
        # 逐行讲解：抛出异常：ValueError(f'数据文件为空：{source}')。
        raise ValueError(f"数据文件为空：{source}")
    # 逐行讲解：返回结果：examples。
    return examples


# 逐行讲解：定义函数 shuffled_examples：读取并按固定 seed 打乱。
def shuffled_examples(path: str | Path, seed: int) -> list[MathExample]:
    """读取并按固定 seed 打乱。"""
    # 逐行讲解：调用 load_examples（load_examples(path)），赋给 examples。
    examples = load_examples(path)
    # 逐行讲解：打乱数据顺序（random.Random(seed).shuffle(examples)）。
    random.Random(seed).shuffle(examples)
    # 逐行讲解：返回结果：examples。
    return examples


# 逐行讲解：定义类 ExampleCursor：可回绕的数据游标，用于 Dynamic Sampling 持续补采新题。
class ExampleCursor:
    """可回绕的数据游标，用于 Dynamic Sampling 持续补采新题。"""

    # 逐行讲解：定义函数 __init__：封装下方逻辑，供调用方复用。
    def __init__(self, examples: list[MathExample]) -> None:
        # 逐行讲解：检查条件 not examples；成立时执行下方分支。
        if not examples:
            # 逐行讲解：抛出异常：ValueError('ExampleCursor requires at least one example')。
            raise ValueError("ExampleCursor requires at least one example")
        # 逐行讲解：取值 examples，赋给 self._examples。
        self._examples = examples
        # 逐行讲解：取值 0，赋给 self.consumed。
        self.consumed = 0

    # 逐行讲解：定义函数 position：下一个样本在打乱后数据中的位置。
    @property
    def position(self) -> int:
        """下一个样本在打乱后数据中的位置。"""
        # 逐行讲解：返回结果：self.consumed % len(self._examples)。
        return self.consumed % len(self._examples)

    # 逐行讲解：定义函数 take：取 ``count`` 道题并推进游标，越过末尾时循环。
    def take(self, count: int) -> list[MathExample]:
        """取 ``count`` 道题并推进游标，越过末尾时循环。"""
        # 逐行讲解：检查条件 count < 1；成立时执行下方分支。
        if count < 1:
            # 逐行讲解：抛出异常：ValueError('count must be >= 1')。
            raise ValueError("count must be >= 1")
        # 逐行讲解：通过推导式逐项生成结果：[self._examples[(self.consumed + offset) % len(self._examples)] for offset in range(count)]，赋给 batch。
        batch = [
            self._examples[(self.consumed + offset) % len(self._examples)]
            for offset in range(count)
        ]
        # 逐行讲解：更新 self.consumed：self.consumed += count。
        self.consumed += count
        # 逐行讲解：返回结果：batch。
        return batch
