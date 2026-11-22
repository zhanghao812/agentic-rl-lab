"""读取 prepare_data.py 生成的本地 Tournament-GRPO 数据。"""

# 逐行讲解：导入模块：json。
import json
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path


# 逐行讲解：定义类 ResearchExample：保存一道开放式研究题及作者提供的评分 rubric。
@dataclass(frozen=True)
class ResearchExample:
    """保存一道开放式研究题及作者提供的评分 rubric。"""

    # 逐行讲解：声明 id 的类型为 str。
    id: str
    # 逐行讲解：声明 query 的类型为 str。
    query: str
    # 逐行讲解：声明 rubrics 的类型为 list[dict[str, str]]。
    rubrics: list[dict[str, str]]
    # 逐行讲解：声明 data_source 的类型为 str。
    data_source: str


# 逐行讲解：定义函数 load_examples：按文件原顺序读取全部样本；作者训练使用 shuffle=False。
def load_examples(path: str | Path) -> list[ResearchExample]:
    """按文件原顺序读取全部样本；作者训练使用 shuffle=False。"""
    # 逐行讲解：进入上下文管理器 Path(path).open(encoding='utf-8')，退出时执行其收尾逻辑。
    with Path(path).open(encoding="utf-8") as file:
        # 逐行讲解：返回结果：[ResearchExample(**json.loads(line)) for line in file]。
        return [ResearchExample(**json.loads(line)) for line in file]
