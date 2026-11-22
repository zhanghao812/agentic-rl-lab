"""Plot AIME25 metrics across GSPO checkpoints.

Run from the repository root:

    uv run python experiments/gspo/analyse.py

The figure is written to ``experiments/gspo/images/aime25-gspo-progress.png``.
Metrics are read from the final ``type=summary`` record of each evaluation
JSONL in ``experiments/gspo/eval-results``.
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：math。
import math
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：导入模块：matplotlib.pyplot，别名 plt。
import matplotlib.pyplot as plt
# 逐行讲解：从 matplotlib.ticker 导入 FuncFormatter, PercentFormatter。
from matplotlib.ticker import FuncFormatter, PercentFormatter


# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'eval-results'，赋给 DEFAULT_RESULT_DIR。
DEFAULT_RESULT_DIR = SCRIPT_DIR / "eval-results"
# 逐行讲解：计算 SCRIPT_DIR / 'images' / 'aime25-gspo-progress.png'，赋给 DEFAULT_OUTPUT。
DEFAULT_OUTPUT = SCRIPT_DIR / "images" / "aime25-gspo-progress.png"

# 逐行讲解：取值 '#D94A4A'，赋给 COLOR_AVERAGE。
COLOR_AVERAGE = "#D94A4A"
# 逐行讲解：取值 '#3388B8'，赋给 COLOR_PASS。
COLOR_PASS = "#3388B8"
# 逐行讲解：取值 '#F28E2B'，赋给 COLOR_FORMAT。
COLOR_FORMAT = "#F28E2B"
# 逐行讲解：取值 '#F2B36B'，赋给 COLOR_FORMAT_MARKER。
COLOR_FORMAT_MARKER = "#F2B36B"
# 逐行讲解：取值 '#596A7D'，赋给 COLOR_TOKENS。
COLOR_TOKENS = "#596A7D"
# 逐行讲解：取值 '#A7B4C2'，赋给 COLOR_TOKENS_MARKER。
COLOR_TOKENS_MARKER = "#A7B4C2"
# 逐行讲解：取值 '#202020'，赋给 COLOR_EDGE。
COLOR_EDGE = "#202020"
# 逐行讲解：取值 '#555555'，赋给 COLOR_NOTE。
COLOR_NOTE = "#555555"

# 逐行讲解：调用 re.compile（re.compile('^aime25-gspo-step(\\d+)\\.jsonl$')），赋给 STEP_PATTERN。
STEP_PATTERN = re.compile(r"^aime25-gspo-step(\d+)\.jsonl$")


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--result-dir', type=Path, default=DEFAULT_RESULT_DIR, help='Directory containing AIME2…）。
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing AIME25 evaluation JSONL files",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Output image path; the extension s…）。
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path; the extension selects the export format",
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
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()
    # 逐行讲解：检查条件 args.dpi < 1；成立时执行下方分支。
    if args.dpi < 1:
        # 逐行讲解：抛出异常：ValueError('--dpi must be >= 1')。
        raise ValueError("--dpi must be >= 1")
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 load_summary：Return the final summary record in an evaluation JSONL file.
def load_summary(path: Path) -> dict[str, Any]:
    """Return the final summary record in an evaluation JSONL file."""
    # 逐行讲解：声明 summary 的类型为 dict[str, Any] | None；取值 None。
    summary: dict[str, Any] | None = None
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
            # 逐行讲解：检查条件 record.get('type') == 'summary'；成立时执行下方分支。
            if record.get("type") == "summary":
                # 逐行讲解：取值 record，赋给 summary。
                summary = record

    # 逐行讲解：检查条件 summary is None；成立时执行下方分支。
    if summary is None:
        # 逐行讲解：抛出异常：ValueError(f'No type=summary record found in {path}')。
        raise ValueError(f"No type=summary record found in {path}")
    # 逐行讲解：返回结果：summary。
    return summary


# 逐行讲解：定义函数 discover_checkpoints：Find the persisted Base run and numerically sorted GSPO checkpoints.
def discover_checkpoints(result_dir: Path) -> list[tuple[int, str, Path]]:
    """Find the persisted Base run and numerically sorted GSPO checkpoints."""
    # 逐行讲解：计算 result_dir / 'aime25-base.jsonl'，赋给 base_path。
    base_path = result_dir / "aime25-base.jsonl"
    # 逐行讲解：检查条件 not base_path.is_file()；成立时执行下方分支。
    if not base_path.is_file():
        # 逐行讲解：抛出异常：FileNotFoundError(f'Missing evaluation result: {base_path}')。
        raise FileNotFoundError(f"Missing evaluation result: {base_path}")

    # 逐行讲解：声明 checkpoints 的类型为 list[tuple[int, str, Path]]；构造列表 [(0, 'Base', base_path)]。
    checkpoints: list[tuple[int, str, Path]] = [(0, "Base", base_path)]
    # 逐行讲解：遍历 result_dir.glob('aime25-gspo-step*.jsonl')，将每项绑定到 path。
    for path in result_dir.glob("aime25-gspo-step*.jsonl"):
        # 逐行讲解：调用 STEP_PATTERN.fullmatch（STEP_PATTERN.fullmatch(path.name)），赋给 match。
        match = STEP_PATTERN.fullmatch(path.name)
        # 逐行讲解：检查条件 match is None；成立时执行下方分支。
        if match is None:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：调用 int（int(match.group(1))），赋给 step。
        step = int(match.group(1))
        # 逐行讲解：向列表末尾追加一个元素（checkpoints.append((step, f'Step {step}', path))）。
        checkpoints.append((step, f"Step {step}", path))

    # 逐行讲解：调用 checkpoints.sort（checkpoints.sort(key=lambda item: item[0])）。
    checkpoints.sort(key=lambda item: item[0])
    # 逐行讲解：检查条件 len(checkpoints) == 1；成立时执行下方分支。
    if len(checkpoints) == 1:
        # 逐行讲解：抛出异常：FileNotFoundError(f'No aime25-gspo-step*.jsonl evaluation results found in {result_dir}')。
        raise FileNotFoundError(
            f"No aime25-gspo-step*.jsonl evaluation results found in {result_dir}"
        )
    # 逐行讲解：返回结果：checkpoints。
    return checkpoints


# 逐行讲解：定义函数 load_metrics：Load evaluation metrics from persisted summaries.
def load_metrics(
    result_dir: Path,
) -> tuple[
    list[str],
    int,
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Load evaluation metrics from persisted summaries."""
    # 逐行讲解：声明 labels 的类型为 list[str]；构造列表 []。
    labels: list[str] = []
    # 逐行讲解：声明 average_at_n 的类型为 list[float]；构造列表 []。
    average_at_n: list[float] = []
    # 逐行讲解：声明 pass_at_n 的类型为 list[float]；构造列表 []。
    pass_at_n: list[float] = []
    # 逐行讲解：声明 format_rate 的类型为 list[float]；构造列表 []。
    format_rate: list[float] = []
    # 逐行讲解：声明 mean_completion_tokens 的类型为 list[float]；构造列表 []。
    mean_completion_tokens: list[float] = []
    # 逐行讲解：声明 val_n 的类型为 int | None；取值 None。
    val_n: int | None = None

    # 逐行讲解：遍历 discover_checkpoints(result_dir)，将每项绑定到 (_, label, path)。
    for _, label, path in discover_checkpoints(result_dir):
        # 逐行讲解：调用 load_summary（load_summary(path)），赋给 summary。
        summary = load_summary(path)
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：调用 int（int(summary['val_n'])），赋给 current_val_n。
            current_val_n = int(summary["val_n"])
            # 逐行讲解：转换为浮点表示（float(summary['average_at_n'])），赋给 average_value。
            average_value = float(summary["average_at_n"])
            # 逐行讲解：转换为浮点表示（float(summary['pass_at_n'])），赋给 pass_value。
            pass_value = float(summary["pass_at_n"])
            # 逐行讲解：转换为浮点表示（float(summary['format_rate'])），赋给 format_value。
            format_value = float(summary["format_rate"])
            # 逐行讲解：转换为浮点表示（float(summary['mean_completion_tokens'])），赋给 token_value。
            token_value = float(summary["mean_completion_tokens"])
        except (KeyError, TypeError, ValueError) as error:
            # 逐行讲解：抛出异常：ValueError(f'Expected numeric val_n, average_at_n, pass_at_n, and format_rate, and mean_completion_tokens m…。
            raise ValueError(
                "Expected numeric val_n, average_at_n, pass_at_n, and "
                f"format_rate, and mean_completion_tokens metrics in {path}"
            ) from error

        # 逐行讲解：检查条件 val_n is None；成立时执行下方分支。
        if val_n is None:
            # 逐行讲解：取值 current_val_n，赋给 val_n。
            val_n = current_val_n
        # 逐行讲解：检查条件 current_val_n != val_n；成立时执行下方分支。
        elif current_val_n != val_n:
            # 逐行讲解：抛出异常：ValueError(f'Inconsistent val_n in {path}: {current_val_n} != {val_n}')。
            raise ValueError(
                f"Inconsistent val_n in {path}: {current_val_n} != {val_n}"
            )

        # 逐行讲解：遍历 (('average_at_n', average_value), ('pass_at_n', pass_value), ('format_rate', format_value))，将每项绑定到 (metric_name, value)。
        for metric_name, value in (
            ("average_at_n", average_value),
            ("pass_at_n", pass_value),
            ("format_rate", format_value),
        ):
            # 逐行讲解：检查条件 not 0.0 <= value <= 1.0；成立时执行下方分支。
            if not 0.0 <= value <= 1.0:
                # 逐行讲解：抛出异常：ValueError(f'{metric_name} in {path} is outside [0, 1]')。
                raise ValueError(f"{metric_name} in {path} is outside [0, 1]")
        # 逐行讲解：检查条件 token_value <= 0.0；成立时执行下方分支。
        if token_value <= 0.0:
            # 逐行讲解：抛出异常：ValueError(f'mean_completion_tokens in {path} must be > 0')。
            raise ValueError(f"mean_completion_tokens in {path} must be > 0")

        # 逐行讲解：向列表末尾追加一个元素（labels.append(label)）。
        labels.append(label)
        # 逐行讲解：向列表末尾追加一个元素（average_at_n.append(average_value)）。
        average_at_n.append(average_value)
        # 逐行讲解：向列表末尾追加一个元素（pass_at_n.append(pass_value)）。
        pass_at_n.append(pass_value)
        # 逐行讲解：向列表末尾追加一个元素（format_rate.append(format_value)）。
        format_rate.append(format_value)
        # 逐行讲解：向列表末尾追加一个元素（mean_completion_tokens.append(token_value)）。
        mean_completion_tokens.append(token_value)

    # 逐行讲解：断言 val_n is not None；条件不成立时报告错误。
    assert val_n is not None
    # 逐行讲解：返回结果：(labels, val_n, average_at_n, pass_at_n, format_rate, mean_completion_tokens)。
    return (
        labels,
        val_n,
        average_at_n,
        pass_at_n,
        format_rate,
        mean_completion_tokens,
    )


# 逐行讲解：定义函数 configure_style：Apply the paper-style theme shared by the other subprojects.
def configure_style() -> None:
    """Apply the paper-style theme shared by the other subprojects."""
    # 逐行讲解：调用 plt.rcParams.update（plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'f…）。
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": COLOR_EDGE,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


# 逐行讲解：定义函数 add_value_labels：封装下方逻辑，供调用方复用。
def add_value_labels(
    axis: plt.Axes,
    x_positions: list[float] | list[int],
    values: list[float],
    *,
    offset: float,
    fontsize: int = 9,
) -> None:
    # 逐行讲解：遍历 zip(x_positions, values, strict=True)，将每项绑定到 (x_position, value)。
    for x_position, value in zip(x_positions, values, strict=True):
        # 逐行讲解：调用 axis.text（axis.text(x_position, value + offset, f'{value:.1%}', ha='center', va='bottom', fontsize=fontsize, color=CO…）。
        axis.text(
            x_position,
            value + offset,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=COLOR_EDGE,
        )


# 逐行讲解：定义函数 make_figure：封装下方逻辑，供调用方复用。
def make_figure(
    labels: list[str],
    val_n: int,
    average_at_n: list[float],
    pass_at_n: list[float],
    format_rate: list[float],
    mean_completion_tokens: list[float],
) -> plt.Figure:
    # 逐行讲解：调用 configure_style（configure_style()）。
    configure_style()

    # 逐行讲解：调用 list（list(range(len(labels)))），赋给 x_positions。
    x_positions = list(range(len(labels)))
    # 逐行讲解：调用 plt.subplots（plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True, gridspec_kw={'wspace': 0.08})），赋给 (figure, axes)。
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    # 逐行讲解：取值 axes，赋给 (accuracy_axis, format_axis)。
    accuracy_axis, format_axis = axes

    # 逐行讲解：调用 list（list(range(1, len(labels)))），赋给 trained_indices。
    trained_indices = list(range(1, len(labels)))
    # 逐行讲解：调用 max（max(trained_indices, key=average_at_n.__getitem__)），赋给 best_index。
    best_index = max(trained_indices, key=average_at_n.__getitem__)

    # 逐行讲解：取值 0.36，赋给 bar_width。
    bar_width = 0.36
    # 逐行讲解：通过推导式逐项生成结果：[x - bar_width / 2 for x in x_positions]，赋给 average_positions。
    average_positions = [x - bar_width / 2 for x in x_positions]
    # 逐行讲解：通过推导式逐项生成结果：[x + bar_width / 2 for x in x_positions]，赋给 pass_positions。
    pass_positions = [x + bar_width / 2 for x in x_positions]
    # 逐行讲解：调用 accuracy_axis.bar（accuracy_axis.bar(average_positions, average_at_n, width=bar_width, color=COLOR_AVERAGE, edgecolor=COLOR_ED…），赋给 average_bars。
    average_bars = accuracy_axis.bar(
        average_positions,
        average_at_n,
        width=bar_width,
        color=COLOR_AVERAGE,
        edgecolor=COLOR_EDGE,
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
        label=f"Average@{val_n}",
    )
    # 逐行讲解：调用 accuracy_axis.bar（accuracy_axis.bar(pass_positions, pass_at_n, width=bar_width, color=COLOR_PASS, edgecolor=COLOR_EDGE, linew…）。
    accuracy_axis.bar(
        pass_positions,
        pass_at_n,
        width=bar_width,
        color=COLOR_PASS,
        edgecolor=COLOR_EDGE,
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
        label=f"Pass@{val_n}",
    )
    # 逐行讲解：调用 average_bars[best_index].set_color（average_bars[best_index].set_color(COLOR_FORMAT)）。
    average_bars[best_index].set_color(COLOR_FORMAT)
    # 逐行讲解：调用 average_bars[best_index].set_hatch（average_bars[best_index].set_hatch('///')）。
    average_bars[best_index].set_hatch("///")

    # 逐行讲解：调用 add_value_labels（add_value_labels(accuracy_axis, average_positions, average_at_n, offset=0.02, fontsize=8)）。
    add_value_labels(
        accuracy_axis,
        average_positions,
        average_at_n,
        offset=0.02,
        fontsize=8,
    )
    # 逐行讲解：调用 add_value_labels（add_value_labels(accuracy_axis, pass_positions, pass_at_n, offset=0.02, fontsize=8)）。
    add_value_labels(
        accuracy_axis,
        pass_positions,
        pass_at_n,
        offset=0.02,
        fontsize=8,
    )
    # 逐行讲解：调用 accuracy_axis.set_title（accuracy_axis.set_title(f'(a) Average@{val_n} & Pass@{val_n}', pad=14)）。
    accuracy_axis.set_title(f"(a) Average@{val_n} & Pass@{val_n}", pad=14)
    # 逐行讲解：调用 accuracy_axis.set_ylabel（accuracy_axis.set_ylabel('Score')）。
    accuracy_axis.set_ylabel("Score")
    # 逐行讲解：调用 accuracy_axis.legend（accuracy_axis.legend(loc='upper left', frameon=False, fontsize=10, handlelength=1.4)）。
    accuracy_axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=10,
        handlelength=1.4,
    )

    # 逐行讲解：按索引或切片读取 format_axis.plot(x_positions, format_rate, color=COLOR_FORMAT, marker='s', markersize=7, markerfacecolor=CO…，赋给 format_line。
    format_line = format_axis.plot(
        x_positions,
        format_rate,
        color=COLOR_FORMAT,
        marker="s",
        markersize=7,
        markerfacecolor=COLOR_FORMAT_MARKER,
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.9,
        linewidth=2.1,
        zorder=4,
        label="Format rate",
    )[0]
    # 逐行讲解：调用 add_value_labels（add_value_labels(format_axis, x_positions, format_rate, offset=0.025)）。
    add_value_labels(format_axis, x_positions, format_rate, offset=0.025)

    # 逐行讲解：调用 format_axis.twinx（format_axis.twinx()），赋给 token_axis。
    token_axis = format_axis.twinx()
    # 逐行讲解：按索引或切片读取 token_axis.plot(x_positions, mean_completion_tokens, color=COLOR_TOKENS, marker='o', markersize=6.5, marker…，赋给 token_line。
    token_line = token_axis.plot(
        x_positions,
        mean_completion_tokens,
        color=COLOR_TOKENS,
        marker="o",
        markersize=6.5,
        markerfacecolor=COLOR_TOKENS_MARKER,
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.9,
        linewidth=2.0,
        linestyle="--",
        zorder=3,
        label="Mean completion tokens",
    )[0]
    # 逐行讲解：遍历 zip(x_positions, mean_completion_tokens, strict=True)，将每项绑定到 (x_position, value)。
    for x_position, value in zip(
        x_positions,
        mean_completion_tokens,
        strict=True,
    ):
        # 逐行讲解：调用 token_axis.annotate（token_axis.annotate(f'{value:,.0f}', (x_position, value), xytext=(0, -16), textcoords='offset points', ha='…）。
        token_axis.annotate(
            f"{value:,.0f}",
            (x_position, value),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            color=COLOR_TOKENS,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 1.0,
            },
            zorder=5,
        )

    # 逐行讲解：计算 math.ceil(max(mean_completion_tokens) / 1000.0) * 1000.0，赋给 token_ceiling。
    token_ceiling = math.ceil(max(mean_completion_tokens) / 1000.0) * 1000.0
    # 逐行讲解：调用 token_axis.set_ylim（token_axis.set_ylim(0.0, token_ceiling)）。
    token_axis.set_ylim(0.0, token_ceiling)
    # 逐行讲解：调用 token_axis.set_ylabel（token_axis.set_ylabel('Mean completion tokens', color=COLOR_TOKENS, labelpad=10)）。
    token_axis.set_ylabel(
        "Mean completion tokens",
        color=COLOR_TOKENS,
        labelpad=10,
    )
    # 逐行讲解：调用 token_axis.yaxis.set_major_formatter（token_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1000.0:.0f}k'))）。
    token_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000.0:.0f}k")
    )
    # 逐行讲解：调用 token_axis.tick_params（token_axis.tick_params(axis='y', which='major', direction='in', right=True, length=5, width=0.9, colors=COL…）。
    token_axis.tick_params(
        axis="y",
        which="major",
        direction="in",
        right=True,
        length=5,
        width=0.9,
        colors=COLOR_TOKENS,
    )
    # 逐行讲解：调用 token_axis.spines['right'].set_color（token_axis.spines['right'].set_color(COLOR_EDGE)）。
    token_axis.spines["right"].set_color(COLOR_EDGE)
    # 逐行讲解：调用 token_axis.spines['right'].set_linewidth（token_axis.spines['right'].set_linewidth(1.0)）。
    token_axis.spines["right"].set_linewidth(1.0)
    # 逐行讲解：调用 token_axis.grid（token_axis.grid(False)）。
    token_axis.grid(False)

    # 逐行讲解：调用 format_axis.legend（format_axis.legend(handles=[format_line, token_line], loc='lower left', frameon=False, fontsize=9.5, handle…）。
    format_axis.legend(
        handles=[format_line, token_line],
        loc="lower left",
        frameon=False,
        fontsize=9.5,
        handlelength=2.2,
    )
    # 逐行讲解：调用 format_axis.set_title（format_axis.set_title('(b) Format Rate & Completion Length', pad=14)）。
    format_axis.set_title("(b) Format Rate & Completion Length", pad=14)

    # 逐行讲解：遍历 axes，将每项绑定到 axis。
    for axis in axes:
        # 逐行讲解：调用 axis.set_xticks（axis.set_xticks(x_positions, labels)）。
        axis.set_xticks(x_positions, labels)
        # 逐行讲解：调用 axis.set_xlabel（axis.set_xlabel('Model / Checkpoint', labelpad=10)）。
        axis.set_xlabel("Model / Checkpoint", labelpad=10)
        # 逐行讲解：调用 axis.set_ylim（axis.set_ylim(0.0, 1.08)）。
        axis.set_ylim(0.0, 1.08)
        # 逐行讲解：调用 axis.yaxis.set_major_formatter（axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))）。
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        # 逐行讲解：调用 axis.grid（axis.grid(axis='y', color='#C7C7C7', linestyle='-.', linewidth=0.8, alpha=0.75, zorder=0)）。
        axis.grid(
            axis="y",
            color="#C7C7C7",
            linestyle="-.",
            linewidth=0.8,
            alpha=0.75,
            zorder=0,
        )
        # 逐行讲解：调用 axis.tick_params（axis.tick_params(axis='both', which='major', direction='in', top=True, right=axis is accuracy_axis, length=…）。
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=axis is accuracy_axis,
            length=5,
            width=0.9,
        )
        # 逐行讲解：调用 axis.margins（axis.margins(x=0.06)）。
        axis.margins(x=0.06)

    # 逐行讲解：调用 figure.suptitle（figure.suptitle('GSPO AIME25 Checkpoint Evaluation', fontsize=17, y=0.995)）。
    figure.suptitle(
        "GSPO AIME25 Checkpoint Evaluation",
        fontsize=17,
        y=0.995,
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.935, f'AIME 2025 (30 problems) · val-n {val_n} · temperature 1.0', ha='center', va='top'…）。
    figure.text(
        0.5,
        0.935,
        f"AIME 2025 (30 problems) · val-n {val_n} · temperature 1.0",
        ha="center",
        va="top",
        fontsize=10,
        color=COLOR_NOTE,
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.015, f'Orange hatching marks the highest observed trained Average@{val_n} ({labels[best_…）。
    figure.text(
        0.5,
        0.015,
        "Orange hatching marks the highest observed trained "
        f"Average@{val_n} ({labels[best_index]}; descriptive only). "
        "The token axis starts at zero; Base is the persisted evaluation run.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLOR_NOTE,
    )
    # 逐行讲解：调用 figure.subplots_adjust（figure.subplots_adjust(left=0.075, right=0.94, bottom=0.15, top=0.84)）。
    figure.subplots_adjust(left=0.075, right=0.94, bottom=0.15, top=0.84)
    # 逐行讲解：返回结果：figure。
    return figure


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：调用 load_metrics（load_metrics(args.result_dir)），赋给 (labels, val_n, average_at_n, pass_at_n, format_rate, mean_completion_tokens)。
    (
        labels,
        val_n,
        average_at_n,
        pass_at_n,
        format_rate,
        mean_completion_tokens,
    ) = load_metrics(args.result_dir)
    # 逐行讲解：调用 make_figure（make_figure(labels, val_n, average_at_n, pass_at_n, format_rate, mean_completion_tokens)），赋给 figure。
    figure = make_figure(
        labels,
        val_n,
        average_at_n,
        pass_at_n,
        format_rate,
        mean_completion_tokens,
    )

    # 逐行讲解：创建目录（args.output.parent.mkdir(parents=True, exist_ok=True)）。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 figure.savefig（figure.savefig(args.output, dpi=args.dpi, bbox_inches='tight')）。
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    # 逐行讲解：调用 print（print(f'Saved figure: {args.output}')）。
    print(f"Saved figure: {args.output}")

    # 逐行讲解：检查条件 args.show；成立时执行下方分支。
    if args.show:
        # 逐行讲解：调用 plt.show（plt.show()）。
        plt.show()
    else:
        # 逐行讲解：调用 plt.close（plt.close(figure)）。
        plt.close(figure)


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
