"""加载只提供给 Teacher 的固定 ALFWorld SkillBank。"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：json。
import json
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path


# 逐行讲解：计算 Path(__file__).resolve().parent / 'skills' / 'alfworld'，赋给 DEFAULT_SKILLS_DIR。
DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent / "skills" / "alfworld"
# 逐行讲解：取值 'https://github.com/ZethWang/AgentOPSD'，赋给 SKILL_SOURCE_REPOSITORY。
SKILL_SOURCE_REPOSITORY = "https://github.com/ZethWang/AgentOPSD"
# 逐行讲解：取值 '0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b'，赋给 SKILL_SOURCE_COMMIT。
SKILL_SOURCE_COMMIT = "0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b"

# 本地 ALFWorld 数据集将单物体任务命名为 ``pick_and_place_simple``，
# 上游 SkillBank 使用的名称则是 ``pick_and_place``。
TASK_TYPE_ALIASES = {
    "pick_and_place_simple": "pick_and_place",
}


# 逐行讲解：定义类 SkillSelection：Resolved audit key and the exact text injected into the Teacher.
@dataclass(frozen=True)
class SkillSelection:
    """Resolved audit key and the exact text injected into the Teacher."""

    # 逐行讲解：声明 task_type 的类型为 str。
    task_type: str
    # 逐行讲解：声明 skill_name 的类型为 str。
    skill_name: str
    # 逐行讲解：声明 key 的类型为 str。
    key: str
    # 逐行讲解：声明 text 的类型为 str。
    text: str


# 逐行讲解：定义类 SkillProvider：Deterministically map ALFWorld task metadata to pinned skill text.
class SkillProvider:
    """Deterministically map ALFWorld task metadata to pinned skill text."""

    # 逐行讲解：定义函数 __init__：封装下方逻辑，供调用方复用。
    def __init__(self, skills_dir: str | Path = DEFAULT_SKILLS_DIR) -> None:
        # 逐行讲解：调用 Path(skills_dir).expanduser().resolve（Path(skills_dir).expanduser().resolve()），赋给 self.skills_dir。
        self.skills_dir = Path(skills_dir).expanduser().resolve()
        # 逐行讲解：计算 self.skills_dir / 'skill_mapping.json'，赋给 mapping_path。
        mapping_path = self.skills_dir / "skill_mapping.json"
        # 逐行讲解：进入上下文管理器 mapping_path.open(encoding='utf-8')，退出时执行其收尾逻辑。
        with mapping_path.open(encoding="utf-8") as file:
            # 逐行讲解：调用 json.load（json.load(file)），赋给 mapping。
            mapping = json.load(file)

        # 逐行讲解：按索引或切片读取 mapping['skill_files']，赋给 skill_files。
        skill_files = mapping["skill_files"]
        # 逐行讲解：按索引或切片读取 mapping['task_to_skill']，赋给 task_to_skill。
        task_to_skill = mapping["task_to_skill"]
        # 逐行讲解：通过推导式逐项生成结果：{str(task_type): str(skill_name) for task_type, skill_name in task_to_skill.items()}，赋给 self.task_to_skill。
        self.task_to_skill = {
            str(task_type): str(skill_name)
            for task_type, skill_name in task_to_skill.items()
        }
        # 逐行讲解：声明 self.skill_contents 的类型为 dict[str, str]；构造字典 {}。
        self.skill_contents: dict[str, str] = {}
        # 逐行讲解：遍历 skill_files.items()，将每项绑定到 (skill_name, filename)。
        for skill_name, filename in skill_files.items():
            # 逐行讲解：计算 self.skills_dir / str(filename)，赋给 path。
            path = self.skills_dir / str(filename)
            # 逐行讲解：调用 path.read_text(encoding='utf-8').strip（path.read_text(encoding='utf-8').strip()），赋给 content。
            content = path.read_text(encoding="utf-8").strip()
            # 逐行讲解：取值 content，赋给 self.skill_contents[str(skill_name)]。
            self.skill_contents[str(skill_name)] = content

    # 逐行讲解：定义函数 canonical_task_type：封装下方逻辑，供调用方复用。
    @staticmethod
    def canonical_task_type(task_type: str) -> str:
        # 逐行讲解：调用 str(task_type).strip（str(task_type).strip()），赋给 value。
        value = str(task_type).strip()
        # 逐行讲解：返回结果：TASK_TYPE_ALIASES.get(value, value)。
        return TASK_TYPE_ALIASES.get(value, value)

    # 逐行讲解：定义函数 resolve：Return ``general + task-specific`` text for an explicit task type.
    def resolve(self, task_type: str) -> SkillSelection:
        """Return ``general + task-specific`` text for an explicit task type."""

        # 逐行讲解：调用 self.canonical_task_type（self.canonical_task_type(task_type)），赋给 canonical。
        canonical = self.canonical_task_type(task_type)
        # 逐行讲解：按索引或切片读取 self.task_to_skill[canonical]，赋给 skill_name。
        skill_name = self.task_to_skill[canonical]
        # 逐行讲解：按索引或切片读取 self.skill_contents[skill_name]，赋给 task_skill。
        task_skill = self.skill_contents[skill_name]
        # 逐行讲解：按索引或切片读取 self.skill_contents['general_skills']，赋给 general。
        general = self.skill_contents["general_skills"]
        # 逐行讲解：返回结果：SkillSelection(task_type=str(task_type), skill_name=skill_name, key=f'general_skills+{skill_name}', text=f'…。
        return SkillSelection(
            task_type=str(task_type),
            skill_name=skill_name,
            key=f"general_skills+{skill_name}",
            text=f"{general}\n\n{task_skill}",
        )
