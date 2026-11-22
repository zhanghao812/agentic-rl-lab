"""发现并按固定顺序读取 ALFWorld 的 text-only 游戏。

在 experiments/tempo/ 目录下载数据：

uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/datasets/alfworld"
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：random。
import random
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 functools 导入 lru_cache。
from functools import lru_cache
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path


# 逐行讲解：计算 Path(__file__).resolve().parent / 'datasets' / 'alfworld'，赋给 DEFAULT_DATA_ROOT。
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "datasets" / "alfworld"

# 逐行讲解：构造字典 {'train': 'train', 'valid_seen': 'valid_seen', 'valid_unseen': 'valid_unseen'}，赋给 SPLIT_DIRECTORIES。
SPLIT_DIRECTORIES = {
    "train": "train",
    "valid_seen": "valid_seen",
    "valid_unseen": "valid_unseen",
}

# 逐行讲解：调用 frozenset（frozenset({'pick_and_place_simple', 'look_at_obj_in_light', 'pick_clean_then_place_in_recep', 'pick_heat_th…），赋给 SUPPORTED_TASK_TYPES。
SUPPORTED_TASK_TYPES = frozenset(
    {
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    }
)


# 逐行讲解：定义类 GameExample：一局可复现的 ALFWorld 游戏。
@dataclass(frozen=True)
class GameExample:
    """一局可复现的 ALFWorld 游戏。"""

    # 逐行讲解：声明 id 的类型为 str。
    id: str
    # 逐行讲解：声明 split 的类型为 str。
    split: str
    # 逐行讲解：声明 game_file 的类型为 Path。
    game_file: Path
    # 逐行讲解：声明 task_type 的类型为 str。
    task_type: str


# 逐行讲解：定义函数 default_data_root：返回项目内数据目录；其他位置请通过 ``--data-root`` 显式传入。
def default_data_root() -> Path:
    """返回项目内数据目录；其他位置请通过 ``--data-root`` 显式传入。"""
    # 逐行讲解：返回结果：DEFAULT_DATA_ROOT.resolve()。
    return DEFAULT_DATA_ROOT.resolve()


# 逐行讲解：定义函数 _dataset_root：兼容传入缓存根目录、json_2.1.1 目录或某个 split 目录。
def _dataset_root(data_root: str | Path) -> Path:
    """兼容传入缓存根目录、json_2.1.1 目录或某个 split 目录。"""
    # 逐行讲解：调用 Path(data_root).expanduser().resolve（Path(data_root).expanduser().resolve()），赋给 root。
    root = Path(data_root).expanduser().resolve()
    # 逐行讲解：检查条件 root.name in SPLIT_DIRECTORIES.values()；成立时执行下方分支。
    if root.name in SPLIT_DIRECTORIES.values():
        # 逐行讲解：返回结果：root.parent。
        return root.parent
    # 逐行讲解：计算 root / 'json_2.1.1'，赋给 nested。
    nested = root / "json_2.1.1"
    # 逐行讲解：返回结果：nested if nested.is_dir() else root。
    return nested if nested.is_dir() else root


# 逐行讲解：定义函数 split_directory：解析一个公开 split 对应的磁盘目录。
def split_directory(data_root: str | Path, split: str) -> Path:
    """解析一个公开 split 对应的磁盘目录。"""
    # 逐行讲解：检查条件 split not in SPLIT_DIRECTORIES；成立时执行下方分支。
    if split not in SPLIT_DIRECTORIES:
        # 逐行讲解：调用 ', '.join（', '.join(SPLIT_DIRECTORIES)），赋给 choices。
        choices = ", ".join(SPLIT_DIRECTORIES)
        # 逐行讲解：抛出异常：ValueError(f'未知 split: {split!r}；可选值为 {choices}')。
        raise ValueError(f"未知 split: {split!r}；可选值为 {choices}")
    # 逐行讲解：调用 _dataset_root（_dataset_root(data_root)），赋给 root。
    root = _dataset_root(data_root)
    # 逐行讲解：检查条件 root.name == SPLIT_DIRECTORIES[split]；成立时执行下方分支。
    if root.name == SPLIT_DIRECTORIES[split]:
        # 逐行讲解：返回结果：root。
        return root
    # 逐行讲解：返回结果：root / SPLIT_DIRECTORIES[split]。
    return root / SPLIT_DIRECTORIES[split]


# 逐行讲解：定义函数 _load_json：封装下方逻辑，供调用方复用。
def _load_json(path: Path) -> dict:
    # 逐行讲解：进入上下文管理器 path.open(encoding='utf-8')，退出时执行其收尾逻辑。
    with path.open(encoding="utf-8") as file:
        # 逐行讲解：调用 json.load（json.load(file)），赋给 value。
        value = json.load(file)
    # 逐行讲解：检查条件 not isinstance(value, dict)；成立时执行下方分支。
    if not isinstance(value, dict):
        # 逐行讲解：抛出异常：ValueError(f'JSON 顶层应为对象: {path}')。
        raise ValueError(f"JSON 顶层应为对象: {path}")
    # 逐行讲解：返回结果：value。
    return value


# 逐行讲解：定义函数 discover_games：发现指定 split 中所有受支持且标记为 solvable 的游戏。
def discover_games(
    data_root: str | Path,
    split: str,
    *,
    task_types: set[str] | frozenset[str] | None = None,
) -> list[GameExample]:
    """发现指定 split 中所有受支持且标记为 solvable 的游戏。"""
    # 逐行讲解：调用 split_directory（split_directory(data_root, split)），赋给 directory。
    directory = split_directory(data_root, split)
    # 逐行讲解：检查条件 not directory.is_dir()；成立时执行下方分支。
    if not directory.is_dir():
        # 逐行讲解：抛出异常：FileNotFoundError(f'找不到 ALFWorld 数据目录: {directory}\n请按 data.py 文件开头的命令下载数据，或显式传入 --data-root')。
        raise FileNotFoundError(
            f"找不到 ALFWorld 数据目录: {directory}\n"
            "请按 data.py 文件开头的命令下载数据，或显式传入 --data-root"
        )

    # 逐行讲解：按条件选择结果：SUPPORTED_TASK_TYPES if task_types is None else frozenset(task_types)，赋给 allowed。
    allowed = SUPPORTED_TASK_TYPES if task_types is None else frozenset(task_types)
    # 逐行讲解：声明 examples 的类型为 list[GameExample]；构造列表 []。
    examples: list[GameExample] = []
    # 逐行讲解：遍历 sorted(directory.rglob('game.tw-pddl'))，将每项绑定到 game_file。
    for game_file in sorted(directory.rglob("game.tw-pddl")):
        # 逐行讲解：调用 str（str(game_file)），赋给 path_text。
        path_text = str(game_file)
        # 逐行讲解：检查条件 'movable' in path_text or 'Sliced' in path_text；成立时执行下方分支。
        if "movable" in path_text or "Sliced" in path_text:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue

        # 逐行讲解：调用 game_file.with_name（game_file.with_name('traj_data.json')），赋给 trajectory_file。
        trajectory_file = game_file.with_name("traj_data.json")
        # 逐行讲解：检查条件 not trajectory_file.is_file()；成立时执行下方分支。
        if not trajectory_file.is_file():
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：调用 _load_json（_load_json(trajectory_file)），赋给 trajectory。
        trajectory = _load_json(trajectory_file)
        # 逐行讲解：调用 str（str(trajectory.get('task_type', ''))），赋给 task_type。
        task_type = str(trajectory.get("task_type", ""))
        # 逐行讲解：检查条件 task_type not in allowed；成立时执行下方分支。
        if task_type not in allowed:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue

        # 逐行讲解：调用 _load_json（_load_json(game_file)），赋给 game_data。
        game_data = _load_json(game_file)
        # 逐行讲解：检查条件 not bool(game_data.get('solvable', False))；成立时执行下方分支。
        if not bool(game_data.get("solvable", False)):
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue

        # 逐行讲解：调用 game_file.relative_to（game_file.relative_to(directory)），赋给 relative。
        relative = game_file.relative_to(directory)
        # 逐行讲解：向列表末尾追加一个元素（examples.append(GameExample(id=f'{split}:{relative.parent.as_posix()}', split=split, game_file=game_file.re…）。
        examples.append(
            GameExample(
                id=f"{split}:{relative.parent.as_posix()}",
                split=split,
                game_file=game_file.resolve(),
                task_type=task_type,
            )
        )

    # 逐行讲解：检查条件 not examples；成立时执行下方分支。
    if not examples:
        # 逐行讲解：抛出异常：ValueError(f'{directory} 中没有找到可用的 game.tw-pddl；请检查 alfworld-download 是否完整')。
        raise ValueError(
            f"{directory} 中没有找到可用的 game.tw-pddl；"
            "请检查 alfworld-download 是否完整"
        )
    # 逐行讲解：返回结果：examples。
    return examples


# 逐行讲解：定义函数 shuffled_games：发现游戏后按固定种子打乱，并可限制本次使用数量。
def shuffled_games(
    data_root: str | Path,
    split: str,
    seed: int,
    *,
    max_games: int = 0,
) -> list[GameExample]:
    """发现游戏后按固定种子打乱，并可限制本次使用数量。"""
    # 逐行讲解：调用 discover_games（discover_games(data_root, split)），赋给 games。
    games = discover_games(data_root, split)
    # 逐行讲解：打乱数据顺序（random.Random(seed).shuffle(games)）。
    random.Random(seed).shuffle(games)
    # 逐行讲解：检查条件 max_games > 0；成立时执行下方分支。
    if max_games > 0:
        # 逐行讲解：按索引或切片读取 games[:max_games]，赋给 games。
        games = games[:max_games]
    # 逐行讲解：返回结果：games。
    return games


# 逐行讲解：定义函数 take_batch：从固定游戏序列循环取一个 batch。
def take_batch(
    games: list[GameExample],
    start: int,
    batch_size: int,
) -> list[GameExample]:
    """从固定游戏序列循环取一个 batch。"""
    # 逐行讲解：检查条件 batch_size < 1；成立时执行下方分支。
    if batch_size < 1:
        # 逐行讲解：抛出异常：ValueError('batch_size 必须大于等于 1')。
        raise ValueError("batch_size 必须大于等于 1")
    # 逐行讲解：检查条件 not games；成立时执行下方分支。
    if not games:
        # 逐行讲解：返回结果：[]。
        return []
    # 逐行讲解：返回结果：[games[(start + offset) % len(games)] for offset in range(batch_size)]。
    return [games[(start + offset) % len(games)] for offset in range(batch_size)]


# 逐行讲解：定义函数 load_walkthrough：读取 game.tw-pddl 自带的专家动作序列，作为 critic 的特权信息。
@lru_cache(maxsize=4096)
def load_walkthrough(game_file: Path | str) -> tuple[str, ...]:
    """读取 game.tw-pddl 自带的专家动作序列，作为 critic 的特权信息。

    actor 看不到这份 walkthrough；只有 critic prompt 会引用它。
    """
    # 逐行讲解：调用 _load_json（_load_json(Path(game_file))），赋给 game_data。
    game_data = _load_json(Path(game_file))
    # 逐行讲解：通过推导式逐项生成结果：[str(step).strip() for step in game_data.get('walkthrough', [])]，赋给 walkthrough。
    walkthrough = [str(step).strip() for step in game_data.get("walkthrough", [])]
    # 逐行讲解：返回结果：tuple(walkthrough)。
    return tuple(walkthrough)
