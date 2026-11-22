"""Plot AgentOPSD ALFWorld evaluation results for Base, Step 40, and Step 80.

Run from the repository root:

    uv run python experiments/agentopsd/analysis.py

The default output is ``experiments/agentopsd/images/agentopsd_checkpoint_evaluation.png``.
The script reads the final ``type=summary`` record from each JSONL file and
also verifies that all three evaluations contain the same game IDs.
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：math。
import math
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：导入模块：matplotlib.pyplot，别名 plt。
import matplotlib.pyplot as plt
# 逐行讲解：从 matplotlib.ticker 导入 PercentFormatter。
from matplotlib.ticker import PercentFormatter


# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'eval_results'，赋给 DEFAULT_RESULT_DIR。
DEFAULT_RESULT_DIR = SCRIPT_DIR / "eval_results"
# 逐行讲解：计算 SCRIPT_DIR / 'images' / 'agentopsd_checkpoint_evaluation.png'，赋给 DEFAULT_OUTPUT。
DEFAULT_OUTPUT = SCRIPT_DIR / "images" / "agentopsd_checkpoint_evaluation.png"

# 逐行讲解：构造元组 (('Base', 'base-qwen35-4b-eval.jsonl'), ('Step 40', 'checkpoint-eval-step40.jsonl'), ('Step 80', 'checkpoin…，赋给 CHECKPOINTS。
CHECKPOINTS = (
    ("Base", "base-qwen35-4b-eval.jsonl"),
    ("Step 40", "checkpoint-eval-step40.jsonl"),
    ("Step 80", "checkpoint-eval-step80.jsonl"),
)
# 逐行讲解：构造元组 ('valid_seen', 'valid_unseen')，赋给 EVALUATION_SPLITS。
EVALUATION_SPLITS = ("valid_seen", "valid_unseen")


# 逐行讲解：定义类 CheckpointMetrics：Metrics used in the two figure panels.
@dataclass(frozen=True)
class CheckpointMetrics:
    """Metrics used in the two figure panels."""

    # 逐行讲解：声明 label 的类型为 str。
    label: str
    # 逐行讲解：声明 seen_success_rate 的类型为 float。
    seen_success_rate: float
    # 逐行讲解：声明 unseen_success_rate 的类型为 float。
    unseen_success_rate: float
    # 逐行讲解：声明 overall_success_rate 的类型为 float。
    overall_success_rate: float
    # 逐行讲解：声明 mean_invalid_actions 的类型为 float。
    mean_invalid_actions: float
    # 逐行讲解：声明 seen_games 的类型为 int。
    seen_games: int
    # 逐行讲解：声明 unseen_games 的类型为 int。
    unseen_games: int

    # 逐行讲解：定义函数 total_games：封装下方逻辑，供调用方复用。
    @property
    def total_games(self) -> int:
        # 逐行讲解：返回结果：self.seen_games + self.unseen_games。
        return self.seen_games + self.unseen_games


# 逐行讲解：定义类 EvaluationFile：One parsed evaluation file plus its trajectory identity set.
@dataclass(frozen=True)
class EvaluationFile:
    """One parsed evaluation file plus its trajectory identity set."""

    # 逐行讲解：声明 summary 的类型为 dict[str, Any]。
    summary: dict[str, Any]
    # 逐行讲解：声明 game_ids 的类型为 frozenset[tuple[str, str]]。
    game_ids: frozenset[tuple[str, str]]


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--result-dir', type=Path, default=DEFAULT_RESULT_DIR, help='Directory containing the t…）。
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing the three evaluation JSONL files",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Output image path; its extension s…）。
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path; its extension selects the export format",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dpi', type=int, default=300, help='Raster export resolution')）。
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster export resolution",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--show', action='store_true', help='Open an interactive preview after saving')）。
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive preview after saving",
    )
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 load_evaluation：Parse one JSONL file and collect its summary and trajectory IDs.
def load_evaluation(path: Path) -> EvaluationFile:
    """Parse one JSONL file and collect its summary and trajectory IDs."""
    # 逐行讲解：声明 summary 的类型为 dict[str, Any] | None；取值 None。
    summary: dict[str, Any] | None = None
    # 逐行讲解：声明 game_ids 的类型为 set[tuple[str, str]]；调用 set（set()）。
    game_ids: set[tuple[str, str]] = set()

    # 逐行讲解：进入上下文管理器 path.open(encoding='utf-8')，退出时执行其收尾逻辑。
    with path.open(encoding="utf-8") as file:
        # 逐行讲解：遍历 enumerate(file, start=1)，将每项绑定到 (line_number, line)。
        for line_number, line in enumerate(file, start=1):
            # 逐行讲解：检查条件 not line.strip()；成立时执行下方分支。
            if not line.strip():
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
            try:
                # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(line)），赋给 record。
                record = json.loads(line)
            except json.JSONDecodeError as error:
                # 逐行讲解：抛出异常：ValueError(f'Invalid JSON in {path}:{line_number}')。
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error

            # 逐行讲解：查询指定键，缺失时使用默认值（record.get('type')），赋给 record_type。
            record_type = record.get("type")
            # 逐行讲解：检查条件 record_type == 'summary'；成立时执行下方分支。
            if record_type == "summary":
                # 逐行讲解：检查条件 summary is not None；成立时执行下方分支。
                if summary is not None:
                    # 逐行讲解：抛出异常：ValueError(f'Multiple summary records found in {path}')。
                    raise ValueError(f"Multiple summary records found in {path}")
                # 逐行讲解：取值 record，赋给 summary。
                summary = record
            # 逐行讲解：检查条件 record_type == 'trajectory'；成立时执行下方分支。
            elif record_type == "trajectory":
                # 逐行讲解：查询指定键，缺失时使用默认值（record.get('evaluation_split', record.get('split'))），赋给 split。
                split = record.get("evaluation_split", record.get("split"))
                # 逐行讲解：查询指定键，缺失时使用默认值（record.get('game_id')），赋给 game_id。
                game_id = record.get("game_id")
                # 逐行讲解：检查条件 split not in EVALUATION_SPLITS；成立时执行下方分支。
                if split not in EVALUATION_SPLITS:
                    # 逐行讲解：抛出异常：ValueError(f'Unexpected evaluation split in {path}:{line_number}: {split!r}')。
                    raise ValueError(
                        f"Unexpected evaluation split in {path}:{line_number}: {split!r}"
                    )
                # 逐行讲解：检查条件 not isinstance(game_id, str) or not game_id；成立时执行下方分支。
                if not isinstance(game_id, str) or not game_id:
                    # 逐行讲解：抛出异常：ValueError(f'Missing game_id in {path}:{line_number}')。
                    raise ValueError(f"Missing game_id in {path}:{line_number}")
                # 逐行讲解：构造元组 (split, game_id)，赋给 identity。
                identity = (split, game_id)
                # 逐行讲解：检查条件 identity in game_ids；成立时执行下方分支。
                if identity in game_ids:
                    # 逐行讲解：抛出异常：ValueError(f'Duplicate trajectory {identity!r} in {path}')。
                    raise ValueError(f"Duplicate trajectory {identity!r} in {path}")
                # 逐行讲解：调用 game_ids.add（game_ids.add(identity)）。
                game_ids.add(identity)

    # 逐行讲解：检查条件 summary is None；成立时执行下方分支。
    if summary is None:
        # 逐行讲解：抛出异常：ValueError(f'No type=summary record found in {path}')。
        raise ValueError(f"No type=summary record found in {path}")
    # 逐行讲解：检查条件 not game_ids；成立时执行下方分支。
    if not game_ids:
        # 逐行讲解：抛出异常：ValueError(f'No trajectory records found in {path}')。
        raise ValueError(f"No trajectory records found in {path}")
    # 逐行讲解：返回结果：EvaluationFile(summary=summary, game_ids=frozenset(game_ids))。
    return EvaluationFile(summary=summary, game_ids=frozenset(game_ids))


# 逐行讲解：定义函数 numeric_metric：Read one finite numeric metric with a useful failure message.
def numeric_metric(metrics: dict[str, Any], key: str, path: Path) -> float:
    """Read one finite numeric metric with a useful failure message."""
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：转换为浮点表示（float(metrics[key])），赋给 value。
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError) as error:
        # 逐行讲解：抛出异常：ValueError(f'Missing or non-numeric metric {key!r} in {path}')。
        raise ValueError(f"Missing or non-numeric metric {key!r} in {path}") from error
    # 逐行讲解：检查条件 not math.isfinite(value)；成立时执行下方分支。
    if not math.isfinite(value):
        # 逐行讲解：抛出异常：ValueError(f'Non-finite metric {key!r} in {path}: {value}')。
        raise ValueError(f"Non-finite metric {key!r} in {path}: {value}")
    # 逐行讲解：返回结果：value。
    return value


# 逐行讲解：定义函数 integer_metric：封装下方逻辑，供调用方复用。
def integer_metric(metrics: dict[str, Any], key: str, path: Path) -> int:
    # 逐行讲解：调用 numeric_metric（numeric_metric(metrics, key, path)），赋给 value。
    value = numeric_metric(metrics, key, path)
    # 逐行讲解：检查条件 value < 0 or not value.is_integer()；成立时执行下方分支。
    if value < 0 or not value.is_integer():
        # 逐行讲解：抛出异常：ValueError(f'Expected a non-negative integer metric {key!r} in {path}')。
        raise ValueError(f"Expected a non-negative integer metric {key!r} in {path}")
    # 逐行讲解：返回结果：int(value)。
    return int(value)


# 逐行讲解：定义函数 describe_game_set_difference：封装下方逻辑，供调用方复用。
def describe_game_set_difference(
    expected: frozenset[tuple[str, str]],
    actual: frozenset[tuple[str, str]],
) -> str:
    # 逐行讲解：调用 sorted（sorted(expected - actual)），赋给 missing。
    missing = sorted(expected - actual)
    # 逐行讲解：调用 sorted（sorted(actual - expected)），赋给 extra。
    extra = sorted(actual - expected)
    # 逐行讲解：构造列表 [f'missing={len(missing)}', f'extra={len(extra)}']，赋给 details。
    details = [f"missing={len(missing)}", f"extra={len(extra)}"]
    # 逐行讲解：检查条件 missing；成立时执行下方分支。
    if missing:
        # 逐行讲解：向列表末尾追加一个元素（details.append(f'first missing={missing[0]!r}')）。
        details.append(f"first missing={missing[0]!r}")
    # 逐行讲解：检查条件 extra；成立时执行下方分支。
    if extra:
        # 逐行讲解：向列表末尾追加一个元素（details.append(f'first extra={extra[0]!r}')）。
        details.append(f"first extra={extra[0]!r}")
    # 逐行讲解：返回结果：', '.join(details)。
    return ", ".join(details)


# 逐行讲解：定义函数 load_metrics：Load metrics and enforce a like-for-like checkpoint comparison.
def load_metrics(
    result_dir: Path,
) -> tuple[str, bool, list[CheckpointMetrics]]:
    """Load metrics and enforce a like-for-like checkpoint comparison."""
    # 逐行讲解：声明 base_model 的类型为 str | None；取值 None。
    base_model: str | None = None
    # 逐行讲解：声明 skill_free 的类型为 bool | None；取值 None。
    skill_free: bool | None = None
    # 逐行讲解：声明 expected_game_ids 的类型为 frozenset[tuple[str, str]] | None；取值 None。
    expected_game_ids: frozenset[tuple[str, str]] | None = None
    # 逐行讲解：声明 checkpoints 的类型为 list[CheckpointMetrics]；构造列表 []。
    checkpoints: list[CheckpointMetrics] = []

    # 逐行讲解：遍历 CHECKPOINTS，将每项绑定到 (label, filename)。
    for label, filename in CHECKPOINTS:
        # 逐行讲解：计算 result_dir / filename，赋给 path。
        path = result_dir / filename
        # 逐行讲解：检查条件 not path.is_file()；成立时执行下方分支。
        if not path.is_file():
            # 逐行讲解：抛出异常：FileNotFoundError(f'Missing evaluation result: {path}')。
            raise FileNotFoundError(f"Missing evaluation result: {path}")

        # 逐行讲解：调用 load_evaluation（load_evaluation(path)），赋给 evaluation。
        evaluation = load_evaluation(path)
        # 逐行讲解：取值 evaluation.summary，赋给 summary。
        summary = evaluation.summary

        # 逐行讲解：查询指定键，缺失时使用默认值（summary.get('base_model')），赋给 model_name。
        model_name = summary.get("base_model")
        # 逐行讲解：检查条件 not isinstance(model_name, str) or not model_name；成立时执行下方分支。
        if not isinstance(model_name, str) or not model_name:
            # 逐行讲解：抛出异常：ValueError(f'Missing base_model in {path}')。
            raise ValueError(f"Missing base_model in {path}")
        # 逐行讲解：检查条件 base_model is None；成立时执行下方分支。
        if base_model is None:
            # 逐行讲解：取值 model_name，赋给 base_model。
            base_model = model_name
        # 逐行讲解：检查条件 model_name != base_model；成立时执行下方分支。
        elif model_name != base_model:
            # 逐行讲解：抛出异常：ValueError(f'Inconsistent base_model in {path}: {model_name!r} != {base_model!r}')。
            raise ValueError(
                f"Inconsistent base_model in {path}: {model_name!r} != {base_model!r}"
            )

        # 逐行讲解：查询指定键，缺失时使用默认值（summary.get('skill_free')），赋给 run_skill_free。
        run_skill_free = summary.get("skill_free")
        # 逐行讲解：检查条件 not isinstance(run_skill_free, bool)；成立时执行下方分支。
        if not isinstance(run_skill_free, bool):
            # 逐行讲解：抛出异常：ValueError(f'Missing boolean skill_free field in {path}')。
            raise ValueError(f"Missing boolean skill_free field in {path}")
        # 逐行讲解：检查条件 skill_free is None；成立时执行下方分支。
        if skill_free is None:
            # 逐行讲解：取值 run_skill_free，赋给 skill_free。
            skill_free = run_skill_free
        # 逐行讲解：检查条件 run_skill_free != skill_free；成立时执行下方分支。
        elif run_skill_free != skill_free:
            # 逐行讲解：抛出异常：ValueError(f'Inconsistent skill_free setting in {path}')。
            raise ValueError(f"Inconsistent skill_free setting in {path}")

        # 逐行讲解：检查条件 expected_game_ids is None；成立时执行下方分支。
        if expected_game_ids is None:
            # 逐行讲解：取值 evaluation.game_ids，赋给 expected_game_ids。
            expected_game_ids = evaluation.game_ids
        # 逐行讲解：检查条件 evaluation.game_ids != expected_game_ids；成立时执行下方分支。
        elif evaluation.game_ids != expected_game_ids:
            # 逐行讲解：调用 describe_game_set_difference（describe_game_set_difference(expected_game_ids, evaluation.game_ids)），赋给 difference。
            difference = describe_game_set_difference(
                expected_game_ids, evaluation.game_ids
            )
            # 逐行讲解：抛出异常：ValueError(f'Evaluation games differ in {path}: {difference}')。
            raise ValueError(f"Evaluation games differ in {path}: {difference}")

        # 逐行讲解：查询指定键，缺失时使用默认值（summary.get('metrics')），赋给 metrics。
        metrics = summary.get("metrics")
        # 逐行讲解：检查条件 not isinstance(metrics, dict)；成立时执行下方分支。
        if not isinstance(metrics, dict):
            # 逐行讲解：抛出异常：ValueError(f'Missing metrics object in {path}')。
            raise ValueError(f"Missing metrics object in {path}")

        # 逐行讲解：声明 split_values 的类型为 dict[str, tuple[int, float, float]]；构造字典 {}。
        split_values: dict[str, tuple[int, float, float]] = {}
        # 逐行讲解：遍历 EVALUATION_SPLITS，将每项绑定到 split。
        for split in EVALUATION_SPLITS:
            # 逐行讲解：调用 integer_metric（integer_metric(metrics, f'eval/{split}/games', path)），赋给 games。
            games = integer_metric(metrics, f"eval/{split}/games", path)
            # 逐行讲解：调用 numeric_metric（numeric_metric(metrics, f'eval/{split}/success_rate', path)），赋给 success_rate。
            success_rate = numeric_metric(
                metrics, f"eval/{split}/success_rate", path
            )
            # 逐行讲解：调用 numeric_metric（numeric_metric(metrics, f'eval/{split}/invalid_actions_mean', path)），赋给 invalid_actions。
            invalid_actions = numeric_metric(
                metrics, f"eval/{split}/invalid_actions_mean", path
            )
            # 逐行讲解：检查条件 not 0.0 <= success_rate <= 1.0；成立时执行下方分支。
            if not 0.0 <= success_rate <= 1.0:
                # 逐行讲解：抛出异常：ValueError(f'Success rate for {split} is outside [0, 1] in {path}')。
                raise ValueError(
                    f"Success rate for {split} is outside [0, 1] in {path}"
                )
            # 逐行讲解：检查条件 invalid_actions < 0.0；成立时执行下方分支。
            if invalid_actions < 0.0:
                # 逐行讲解：抛出异常：ValueError(f'Negative invalid-action mean for {split} in {path}')。
                raise ValueError(f"Negative invalid-action mean for {split} in {path}")
            # 逐行讲解：构造元组 (games, success_rate, invalid_actions)，赋给 split_values[split]。
            split_values[split] = (games, success_rate, invalid_actions)

        # 逐行讲解：按索引或切片读取 split_values['valid_seen']，赋给 (seen_games, seen_success, seen_invalid)。
        seen_games, seen_success, seen_invalid = split_values["valid_seen"]
        # 逐行讲解：按索引或切片读取 split_values['valid_unseen']，赋给 (unseen_games, unseen_success, unseen_invalid)。
        unseen_games, unseen_success, unseen_invalid = split_values["valid_unseen"]
        # 逐行讲解：检查条件 seen_games + unseen_games != len(evaluation.game_ids)；成立时执行下方分支。
        if seen_games + unseen_games != len(evaluation.game_ids):
            # 逐行讲解：抛出异常：ValueError(f'Summary reports {seen_games + unseen_games} games but {len(evaluation.game_ids)} trajectories …。
            raise ValueError(
                f"Summary reports {seen_games + unseen_games} games but "
                f"{len(evaluation.game_ids)} trajectories were found in {path}"
            )

        # 逐行讲解：计算 seen_games + unseen_games，赋给 total_games。
        total_games = seen_games + unseen_games
        # 逐行讲解：检查条件 total_games == 0；成立时执行下方分支。
        if total_games == 0:
            # 逐行讲解：抛出异常：ValueError(f'Evaluation contains no games: {path}')。
            raise ValueError(f"Evaluation contains no games: {path}")

        # 逐行讲解：向列表末尾追加一个元素（checkpoints.append(CheckpointMetrics(label=label, seen_success_rate=seen_success, unseen_success_rate=unsee…）。
        checkpoints.append(
            CheckpointMetrics(
                label=label,
                seen_success_rate=seen_success,
                unseen_success_rate=unseen_success,
                overall_success_rate=(
                    seen_success * seen_games + unseen_success * unseen_games
                )
                / total_games,
                mean_invalid_actions=(
                    seen_invalid * seen_games + unseen_invalid * unseen_games
                )
                / total_games,
                seen_games=seen_games,
                unseen_games=unseen_games,
            )
        )

    # 逐行讲解：断言 base_model is not None；条件不成立时报告错误。
    assert base_model is not None
    # 逐行讲解：断言 skill_free is not None；条件不成立时报告错误。
    assert skill_free is not None
    # 逐行讲解：返回结果：(base_model, skill_free, checkpoints)。
    return base_model, skill_free, checkpoints


# 逐行讲解：定义函数 configure_style：Reuse the restrained paper-style theme of nearby experiment figures.
def configure_style() -> None:
    """Reuse the restrained paper-style theme of nearby experiment figures."""
    # 逐行讲解：调用 plt.rcParams.update（plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'f…）。
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": "#202020",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


# 逐行讲解：定义函数 add_percent_labels：封装下方逻辑，供调用方复用。
def add_percent_labels(
    axis: plt.Axes,
    x_positions: list[float],
    values: list[float],
    *,
    offset: float,
) -> None:
    # 逐行讲解：遍历 zip(x_positions, values, strict=True)，将每项绑定到 (x_position, value)。
    for x_position, value in zip(x_positions, values, strict=True):
        # 逐行讲解：调用 axis.text（axis.text(x_position, value + offset, f'{value:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bol…）。
        axis.text(
            x_position,
            value + offset,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#202020",
        )


# 逐行讲解：定义函数 make_figure：封装下方逻辑，供调用方复用。
def make_figure(
    base_model: str,
    skill_free: bool,
    checkpoints: list[CheckpointMetrics],
) -> plt.Figure:
    # 逐行讲解：调用 configure_style（configure_style()）。
    configure_style()

    # 逐行讲解：通过推导式逐项生成结果：[checkpoint.label for checkpoint in checkpoints]，赋给 labels。
    labels = [checkpoint.label for checkpoint in checkpoints]
    # 逐行讲解：调用 list（list(range(len(checkpoints)))），赋给 x_positions。
    x_positions = list(range(len(checkpoints)))
    # 逐行讲解：通过推导式逐项生成结果：[checkpoint.seen_success_rate for checkpoint in checkpoints]，赋给 seen_success。
    seen_success = [checkpoint.seen_success_rate for checkpoint in checkpoints]
    # 逐行讲解：通过推导式逐项生成结果：[checkpoint.unseen_success_rate for checkpoint in checkpoints]，赋给 unseen_success。
    unseen_success = [checkpoint.unseen_success_rate for checkpoint in checkpoints]
    # 逐行讲解：通过推导式逐项生成结果：[checkpoint.overall_success_rate for checkpoint in checkpoints]，赋给 overall_success。
    overall_success = [checkpoint.overall_success_rate for checkpoint in checkpoints]
    # 逐行讲解：通过推导式逐项生成结果：[checkpoint.mean_invalid_actions for checkpoint in checkpoints]，赋给 invalid_actions。
    invalid_actions = [checkpoint.mean_invalid_actions for checkpoint in checkpoints]

    # 逐行讲解：调用 plt.subplots（plt.subplots(1, 2, figsize=(13.2, 5.5), gridspec_kw={'wspace': 0.18})），赋给 (figure, (split_axis, trend_axis))。
    figure, (split_axis, trend_axis) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.5),
        gridspec_kw={"wspace": 0.18},
    )

    # Panel (a): absolute split success rates, always shown from zero.
    bar_width = 0.34
    # 逐行讲解：通过推导式逐项生成结果：[x - bar_width / 2 for x in x_positions]，赋给 seen_positions。
    seen_positions = [x - bar_width / 2 for x in x_positions]
    # 逐行讲解：通过推导式逐项生成结果：[x + bar_width / 2 for x in x_positions]，赋给 unseen_positions。
    unseen_positions = [x + bar_width / 2 for x in x_positions]
    # 逐行讲解：调用 split_axis.bar（split_axis.bar(seen_positions, seen_success, width=bar_width, color='#D94A4A', edgecolor='#202020', linewid…）。
    split_axis.bar(
        seen_positions,
        seen_success,
        width=bar_width,
        color="#D94A4A",
        edgecolor="#202020",
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
        label="Valid Seen",
    )
    # 逐行讲解：调用 split_axis.bar（split_axis.bar(unseen_positions, unseen_success, width=bar_width, color='#3388B8', edgecolor='#202020', lin…）。
    split_axis.bar(
        unseen_positions,
        unseen_success,
        width=bar_width,
        color="#3388B8",
        edgecolor="#202020",
        linewidth=1.0,
        hatch="///",
        alpha=0.92,
        zorder=3,
        label="Valid Unseen",
    )
    # 逐行讲解：调用 add_percent_labels（add_percent_labels(split_axis, seen_positions, seen_success, offset=0.012)）。
    add_percent_labels(split_axis, seen_positions, seen_success, offset=0.012)
    # 逐行讲解：调用 add_percent_labels（add_percent_labels(split_axis, unseen_positions, unseen_success, offset=0.012)）。
    add_percent_labels(split_axis, unseen_positions, unseen_success, offset=0.012)
    # 逐行讲解：调用 split_axis.set_title（split_axis.set_title('(a) Success Rate by Evaluation Split', pad=14)）。
    split_axis.set_title("(a) Success Rate by Evaluation Split", pad=14)
    # 逐行讲解：调用 split_axis.set_ylabel（split_axis.set_ylabel('Success rate')）。
    split_axis.set_ylabel("Success rate")
    # 逐行讲解：调用 split_axis.set_xticks（split_axis.set_xticks(x_positions, labels)）。
    split_axis.set_xticks(x_positions, labels)
    # 逐行讲解：调用 split_axis.set_xlabel（split_axis.set_xlabel('Evaluated checkpoint', labelpad=10)）。
    split_axis.set_xlabel("Evaluated checkpoint", labelpad=10)
    # 逐行讲解：调用 split_axis.set_ylim（split_axis.set_ylim(0.0, 0.75)）。
    split_axis.set_ylim(0.0, 0.75)
    # 逐行讲解：调用 split_axis.yaxis.set_major_formatter（split_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))）。
    split_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    # 逐行讲解：调用 split_axis.grid（split_axis.grid(axis='y', color='#C7C7C7', linestyle='-.', linewidth=0.8, alpha=0.75, zorder=0)）。
    split_axis.grid(
        axis="y",
        color="#C7C7C7",
        linestyle="-.",
        linewidth=0.8,
        alpha=0.75,
        zorder=0,
    )
    # 逐行讲解：调用 split_axis.legend（split_axis.legend(loc='upper left', frameon=False, fontsize=10, handlelength=1.5)）。
    split_axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=10,
        handlelength=1.5,
    )

    # Panel (b): three ordered, discrete evaluations on focused axes.
    success_color = "#F28E2B"
    # 逐行讲解：取值 '#596A7D'，赋给 invalid_color。
    invalid_color = "#596A7D"
    # 逐行讲解：按索引或切片读取 trend_axis.plot(x_positions, overall_success, color=success_color, marker='s', markersize=7, markerfacecolo…，赋给 success_line。
    success_line = trend_axis.plot(
        x_positions,
        overall_success,
        color=success_color,
        marker="s",
        markersize=7,
        markerfacecolor="#F2B36B",
        markeredgecolor="#202020",
        markeredgewidth=0.9,
        linewidth=2.2,
        zorder=4,
        label="Overall success rate",
    )[0]
    # 逐行讲解：调用 trend_axis.twinx（trend_axis.twinx()），赋给 invalid_axis。
    invalid_axis = trend_axis.twinx()
    # 逐行讲解：按索引或切片读取 invalid_axis.plot(x_positions, invalid_actions, color=invalid_color, marker='o', markersize=7, markerfaceco…，赋给 invalid_line。
    invalid_line = invalid_axis.plot(
        x_positions,
        invalid_actions,
        color=invalid_color,
        marker="o",
        markersize=7,
        markerfacecolor="#A7B4C2",
        markeredgecolor="#202020",
        markeredgewidth=0.9,
        linewidth=2.0,
        linestyle="--",
        zorder=4,
        label="Mean invalid actions",
    )[0]

    # 逐行讲解：计算 max(overall_success) - min(overall_success)，赋给 success_span。
    success_span = max(overall_success) - min(overall_success)
    # 逐行讲解：计算 max(invalid_actions) - min(invalid_actions)，赋给 invalid_span。
    invalid_span = max(invalid_actions) - min(invalid_actions)
    # 逐行讲解：计算 max(success_span, 0.01) * 0.55，赋给 success_padding。
    success_padding = max(success_span, 0.01) * 0.55
    # 逐行讲解：计算 max(invalid_span, 0.2) * 0.28，赋给 invalid_padding。
    invalid_padding = max(invalid_span, 0.20) * 0.28
    # 逐行讲解：调用 trend_axis.set_ylim（trend_axis.set_ylim(min(overall_success) - success_padding, max(overall_success) + success_padding)）。
    trend_axis.set_ylim(
        min(overall_success) - success_padding,
        max(overall_success) + success_padding,
    )
    # 逐行讲解：调用 invalid_axis.set_ylim（invalid_axis.set_ylim(min(invalid_actions) - invalid_padding, max(invalid_actions) + invalid_padding)）。
    invalid_axis.set_ylim(
        min(invalid_actions) - invalid_padding,
        max(invalid_actions) + invalid_padding,
    )

    # 逐行讲解：遍历 zip(x_positions, overall_success, strict=True)，将每项绑定到 (x_position, value)。
    for x_position, value in zip(x_positions, overall_success, strict=True):
        # 逐行讲解：调用 trend_axis.annotate（trend_axis.annotate(f'{value:.1%}', (x_position, value), xytext=(0, 10), textcoords='offset points', ha='ce…）。
        trend_axis.annotate(
            f"{value:.1%}",
            (x_position, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=success_color,
        )
    # 逐行讲解：遍历 zip(x_positions, invalid_actions, strict=True)，将每项绑定到 (x_position, value)。
    for x_position, value in zip(x_positions, invalid_actions, strict=True):
        # 逐行讲解：调用 invalid_axis.annotate（invalid_axis.annotate(f'{value:.2f}', (x_position, value), xytext=(0, -13), textcoords='offset points', ha=…）。
        invalid_axis.annotate(
            f"{value:.2f}",
            (x_position, value),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=invalid_color,
        )

    # 逐行讲解：计算 overall_success[-1] - overall_success[0]，赋给 success_gain。
    success_gain = overall_success[-1] - overall_success[0]
    # 逐行讲解：计算 invalid_actions[0] - invalid_actions[-1]，赋给 invalid_reduction。
    invalid_reduction = invalid_actions[0] - invalid_actions[-1]
    # 逐行讲解：调用 trend_axis.text（trend_axis.text(0.5, 0.04, f'Step 80 vs Base: +{success_gain * 100:.1f} pp success  ·  {invalid_reduction:.…）。
    trend_axis.text(
        0.5,
        0.04,
        (
            f"Step 80 vs Base: +{success_gain * 100:.1f} pp success  ·  "
            f"{invalid_reduction:.2f} fewer invalid actions / game"
        ),
        transform=trend_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#303030",
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "#FFF8EE",
            "edgecolor": "#E1B678",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
        zorder=5,
    )

    # 逐行讲解：调用 trend_axis.set_title（trend_axis.set_title('(b) Overall Success & Invalid Actions', pad=14)）。
    trend_axis.set_title("(b) Overall Success & Invalid Actions", pad=14)
    # 逐行讲解：调用 trend_axis.set_xticks（trend_axis.set_xticks(x_positions, labels)）。
    trend_axis.set_xticks(x_positions, labels)
    # 逐行讲解：调用 trend_axis.set_xlabel（trend_axis.set_xlabel('Evaluated checkpoint', labelpad=10)）。
    trend_axis.set_xlabel("Evaluated checkpoint", labelpad=10)
    # 逐行讲解：调用 trend_axis.set_ylabel（trend_axis.set_ylabel('Overall success rate', color=success_color)）。
    trend_axis.set_ylabel("Overall success rate", color=success_color)
    # 逐行讲解：调用 invalid_axis.set_ylabel（invalid_axis.set_ylabel('Mean invalid actions', color=invalid_color)）。
    invalid_axis.set_ylabel("Mean invalid actions", color=invalid_color)
    # 逐行讲解：调用 trend_axis.yaxis.set_major_formatter（trend_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))）。
    trend_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    # 逐行讲解：调用 trend_axis.tick_params（trend_axis.tick_params(axis='y', colors=success_color)）。
    trend_axis.tick_params(axis="y", colors=success_color)
    # 逐行讲解：调用 invalid_axis.tick_params（invalid_axis.tick_params(axis='y', colors=invalid_color)）。
    invalid_axis.tick_params(axis="y", colors=invalid_color)
    # 逐行讲解：调用 trend_axis.spines['left'].set_color（trend_axis.spines['left'].set_color(success_color)）。
    trend_axis.spines["left"].set_color(success_color)
    # 逐行讲解：调用 invalid_axis.spines['right'].set_color（invalid_axis.spines['right'].set_color(invalid_color)）。
    invalid_axis.spines["right"].set_color(invalid_color)
    # 逐行讲解：调用 trend_axis.grid（trend_axis.grid(axis='y', color='#C7C7C7', linestyle='-.', linewidth=0.8, alpha=0.65, zorder=0)）。
    trend_axis.grid(
        axis="y",
        color="#C7C7C7",
        linestyle="-.",
        linewidth=0.8,
        alpha=0.65,
        zorder=0,
    )
    # 逐行讲解：调用 trend_axis.legend（trend_axis.legend([success_line, invalid_line], [success_line.get_label(), invalid_line.get_label()], loc='…）。
    trend_axis.legend(
        [success_line, invalid_line],
        [success_line.get_label(), invalid_line.get_label()],
        loc="upper center",
        frameon=False,
        fontsize=9,
        handlelength=2.2,
    )

    # 逐行讲解：遍历 (split_axis, trend_axis)，将每项绑定到 axis。
    for axis in (split_axis, trend_axis):
        # 逐行讲解：调用 axis.tick_params（axis.tick_params(axis='x', direction='in', top=True)）。
        axis.tick_params(axis="x", direction="in", top=True)

    # 逐行讲解：按索引或切片读取 checkpoints[0]，赋给 first。
    first = checkpoints[0]
    # 逐行讲解：按索引或切片读取 base_model.rsplit('/', maxsplit=1)[-1]，赋给 model_display_name。
    model_display_name = base_model.rsplit("/", maxsplit=1)[-1]
    # 逐行讲解：按条件选择结果：'skill-free evaluation' if skill_free else 'evaluation'，赋给 evaluation_mode。
    evaluation_mode = "skill-free evaluation" if skill_free else "evaluation"
    # 逐行讲解：调用 figure.suptitle（figure.suptitle('AgentOPSD · ALFWorld Checkpoint Evaluation', fontsize=17, fontweight='bold', y=0.975)）。
    figure.suptitle(
        "AgentOPSD · ALFWorld Checkpoint Evaluation",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.915, f'{model_display_name} · same {first.total_games} games at every checkpoint ({first…）。
    figure.text(
        0.5,
        0.915,
        (
            f"{model_display_name} · same {first.total_games} games at every checkpoint "
            f"({first.seen_games} seen + {first.unseen_games} unseen) · "
            f"{evaluation_mode}"
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        color="#555555",
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.025, 'Panel (a) starts at zero; panel (b) uses focused y-axes and connects only the thre…）。
    figure.text(
        0.5,
        0.025,
        (
            "Panel (a) starts at zero; panel (b) uses focused y-axes and "
            "connects only the three evaluated checkpoints."
        ),
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
        style="italic",
    )
    # 逐行讲解：调用 figure.subplots_adjust（figure.subplots_adjust(left=0.075, right=0.925, bottom=0.16, top=0.81)）。
    figure.subplots_adjust(left=0.075, right=0.925, bottom=0.16, top=0.81)
    # 逐行讲解：返回结果：figure。
    return figure


# 逐行讲解：定义函数 print_metrics：封装下方逻辑，供调用方复用。
def print_metrics(checkpoints: list[CheckpointMetrics]) -> None:
    # 逐行讲解：调用 print（print(f"{'Checkpoint':<12} {'Seen':>8} {'Unseen':>8} {'Overall':>8} {'Invalid':>9}")）。
    print(
        f"{'Checkpoint':<12} {'Seen':>8} {'Unseen':>8} "
        f"{'Overall':>8} {'Invalid':>9}"
    )
    # 逐行讲解：遍历 checkpoints，将每项绑定到 checkpoint。
    for checkpoint in checkpoints:
        # 逐行讲解：调用 print（print(f'{checkpoint.label:<12} {checkpoint.seen_success_rate:>7.1%} {checkpoint.unseen_success_rate:>7.1%} …）。
        print(
            f"{checkpoint.label:<12} "
            f"{checkpoint.seen_success_rate:>7.1%} "
            f"{checkpoint.unseen_success_rate:>7.1%} "
            f"{checkpoint.overall_success_rate:>7.1%} "
            f"{checkpoint.mean_invalid_actions:>9.2f}"
        )


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main(args: argparse.Namespace) -> None:
    # 逐行讲解：调用 load_metrics（load_metrics(args.result_dir)），赋给 (base_model, skill_free, checkpoints)。
    base_model, skill_free, checkpoints = load_metrics(args.result_dir)
    # 逐行讲解：调用 make_figure（make_figure(base_model, skill_free, checkpoints)），赋给 figure。
    figure = make_figure(base_model, skill_free, checkpoints)

    # 逐行讲解：创建目录（args.output.parent.mkdir(parents=True, exist_ok=True)）。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 figure.savefig（figure.savefig(args.output, dpi=args.dpi, bbox_inches='tight')）。
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    # 逐行讲解：调用 print_metrics（print_metrics(checkpoints)）。
    print_metrics(checkpoints)
    # 逐行讲解：调用 print（print(f'Verified identical game IDs across all checkpoints: {checkpoints[0].total_games} games')）。
    print(
        f"Verified identical game IDs across all checkpoints: "
        f"{checkpoints[0].total_games} games"
    )
    # 逐行讲解：调用 print（print(f'Saved figure: {args.output.resolve()}')）。
    print(f"Saved figure: {args.output.resolve()}")

    # 逐行讲解：检查条件 args.show；成立时执行下方分支。
    if args.show:
        # 逐行讲解：调用 plt.show（plt.show()）。
        plt.show()
    else:
        # 逐行讲解：调用 plt.close（plt.close(figure)）。
        plt.close(figure)


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main(parse_args())）。
    main(parse_args())
