"""Plot Macro EM and format rate for Search-R1 evaluations.

Run from the repository root:

    uv run python experiments/search-r1/analyse.py
    uv run python experiments/search-r1/analyse.py --preset deepseek
    uv run python experiments/search-r1/analyse.py --preset deepseek-api

The Zhihu checkpoint figure is written to
``experiments/search-r1/eval_result/checkpoint_em_format.png`` by default. The DeepSeek
Base-vs-Step-20 figure is written to
``experiments/search-r1/images/deepseek_checkpoint_em_format.png``. Each metric is read
from the final ``type=summary`` record of the corresponding evaluation JSONL.
The official Search API Base-vs-Step-30 figure is written to
``experiments/search-r1/images/deepseek_api_base_vs_step30.png``.
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
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
# 逐行讲解：计算 SCRIPT_DIR / 'eval_result'，赋给 DEFAULT_RESULT_DIR。
DEFAULT_RESULT_DIR = SCRIPT_DIR / "eval_result"
# 逐行讲解：计算 DEFAULT_RESULT_DIR / 'checkpoint_em_format.png'，赋给 DEFAULT_OUTPUT。
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "checkpoint_em_format.png"
# 逐行讲解：计算 SCRIPT_DIR / 'images' / 'deepseek_checkpoint_em_format.png'，赋给 DEFAULT_DEEPSEEK_OUTPUT。
DEFAULT_DEEPSEEK_OUTPUT = SCRIPT_DIR / "images" / "deepseek_checkpoint_em_format.png"
# 逐行讲解：计算 SCRIPT_DIR / 'images' / 'deepseek_api_base_vs_step30.png'，赋给 DEFAULT_DEEPSEEK_API_OUTPUT。
DEFAULT_DEEPSEEK_API_OUTPUT = SCRIPT_DIR / "images" / "deepseek_api_base_vs_step30.png"

# 逐行讲解：构造元组 (('Base', 'eval_results.jsonl'), ('Step 20', 'eval_results_rl_step_20.jsonl'), ('Step 50', 'eval_results_rl…，赋给 ZHIHU_CHECKPOINTS。
ZHIHU_CHECKPOINTS = (
    ("Base", "eval_results.jsonl"),
    ("Step 20", "eval_results_rl_step_20.jsonl"),
    ("Step 50", "eval_results_rl_step_50.jsonl"),
    ("Step 100", "eval_results_rl_step_100.jsonl"),
    ("Step 150", "eval_results_rl_step_150.jsonl"),
    ("Step 200", "eval_results_rl_step_200.jsonl"),
)
# 逐行讲解：构造元组 (('Base', 'eval_results_base_deepseek_search.jsonl'), ('Step 20', 'eval_results_rl_step_20_deepseek_search.…，赋给 DEEPSEEK_CHECKPOINTS。
DEEPSEEK_CHECKPOINTS = (
    ("Base", "eval_results_base_deepseek_search.jsonl"),
    ("Step 20", "eval_results_rl_step_20_deepseek_search.jsonl"),
)
# 逐行讲解：构造元组 (('Base', 'eval_results_base.jsonl'), ('Step 30', 'eval_results_rl_step_30.jsonl'))，赋给 DEEPSEEK_API_CHECKPOINTS。
DEEPSEEK_API_CHECKPOINTS = (
    ("Base", "eval_results_base.jsonl"),
    ("Step 30", "eval_results_rl_step_30.jsonl"),
)


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--preset', choices=('zhihu', 'deepseek', 'deepseek-api'), default='zhihu', help='Evalu…）。
    parser.add_argument(
        "--preset",
        choices=("zhihu", "deepseek", "deepseek-api"),
        default="zhihu",
        help="Evaluation set to plot; the default preserves the original Zhihu figure",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--result-dir', type=Path, default=DEFAULT_RESULT_DIR, help='Directory containing eval_…）。
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing eval_results*.jsonl",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, default=None, help='Output image path; each preset has its own d…）。
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path; each preset has its own default",
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


# 逐行讲解：定义函数 load_summary：Return the last summary record in an evaluation JSONL file.
def load_summary(path: Path) -> dict[str, Any]:
    """Return the last summary record in an evaluation JSONL file."""
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


# 逐行讲解：定义函数 load_metrics：封装下方逻辑，供调用方复用。
def load_metrics(
    result_dir: Path,
    checkpoints: tuple[tuple[str, str], ...],
    *,
    expected_backend: str | None = None,
) -> tuple[list[str], list[float], list[float]]:
    # 逐行讲解：声明 labels 的类型为 list[str]；构造列表 []。
    labels: list[str] = []
    # 逐行讲解：声明 macro_em 的类型为 list[float]；构造列表 []。
    macro_em: list[float] = []
    # 逐行讲解：声明 format_rate 的类型为 list[float]；构造列表 []。
    format_rate: list[float] = []
    # 逐行讲解：声明 evaluation_sizes 的类型为 set[int]；调用 set（set()）。
    evaluation_sizes: set[int] = set()
    # 逐行讲解：声明 search_configs 的类型为 set[tuple[str, int, float, str]]；调用 set（set()）。
    search_configs: set[tuple[str, int, float, str]] = set()

    # 逐行讲解：遍历 checkpoints，将每项绑定到 (label, filename)。
    for label, filename in checkpoints:
        # 逐行讲解：计算 result_dir / filename，赋给 path。
        path = result_dir / filename
        # 逐行讲解：检查条件 not path.is_file()；成立时执行下方分支。
        if not path.is_file():
            # 逐行讲解：抛出异常：FileNotFoundError(f'Missing evaluation result: {path}')。
            raise FileNotFoundError(f"Missing evaluation result: {path}")

        # 逐行讲解：调用 load_summary（load_summary(path)），赋给 summary。
        summary = load_summary(path)
        # 逐行讲解：检查条件 expected_backend is not None；成立时执行下方分支。
        if expected_backend is not None:
            # 逐行讲解：查询指定键，缺失时使用默认值（summary.get('search_backend')），赋给 backend。
            backend = summary.get("search_backend")
            # 逐行讲解：检查条件 backend != expected_backend；成立时执行下方分支。
            if backend != expected_backend:
                # 逐行讲解：抛出异常：ValueError(f'Expected search_backend={expected_backend!r} in {path}, got {backend!r}')。
                raise ValueError(
                    f"Expected search_backend={expected_backend!r} in {path}, got {backend!r}"
                )
            # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
            try:
                # 逐行讲解：调用 evaluation_sizes.add（evaluation_sizes.add(int(summary['evaluated_examples']))）。
                evaluation_sizes.add(int(summary["evaluated_examples"]))
                # 旧 Evidence 实验仍可绘图，但不能与官方 Search API 混为同一环境。
                if summary.get("search_api"):
                    # 逐行讲解：构造字典 {'api': summary['search_api'], 'count': int(summary['search_count']), 'need_content': summary['search_need_…，赋给 search_details。
                    search_details = {
                        "api": summary["search_api"],
                        "count": int(summary["search_count"]),
                        "need_content": summary["search_need_content"],
                        "text_field": summary["search_text_field"],
                    }
                else:
                    # 逐行讲解：构造字典 {'legacy_evidence_model': summary['search_model']}，赋给 search_details。
                    search_details = {"legacy_evidence_model": summary["search_model"]}
                # 逐行讲解：调用 search_configs.add（search_configs.add((str(backend), int(summary['search_concurrency']), float(summary['search_timeout']), jso…）。
                search_configs.add(
                    (
                        str(backend),
                        int(summary["search_concurrency"]),
                        float(summary["search_timeout"]),
                        json.dumps(search_details, sort_keys=True),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                # 逐行讲解：抛出异常：ValueError(f'Search provenance is incomplete in {path}')。
                raise ValueError(f"Search provenance is incomplete in {path}") from error

        # 逐行讲解：查询指定键，缺失时使用默认值（summary.get('metrics')），赋给 metrics。
        metrics = summary.get("metrics")
        # 逐行讲解：检查条件 not isinstance(metrics, dict)；成立时执行下方分支。
        if not isinstance(metrics, dict):
            # 逐行讲解：抛出异常：ValueError(f'Summary metrics are missing in {path}')。
            raise ValueError(f"Summary metrics are missing in {path}")

        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：转换为浮点表示（float(metrics['em/macro'])），赋给 em_value。
            em_value = float(metrics["em/macro"])
            # 逐行讲解：转换为浮点表示（float(metrics['format/rate'])），赋给 format_value。
            format_value = float(metrics["format/rate"])
        except (KeyError, TypeError, ValueError) as error:
            # 逐行讲解：抛出异常：ValueError(f'Expected numeric em/macro and format/rate metrics in {path}')。
            raise ValueError(
                f"Expected numeric em/macro and format/rate metrics in {path}"
            ) from error

        # 逐行讲解：向列表末尾追加一个元素（labels.append(label)）。
        labels.append(label)
        # 逐行讲解：向列表末尾追加一个元素（macro_em.append(em_value)）。
        macro_em.append(em_value)
        # 逐行讲解：向列表末尾追加一个元素（format_rate.append(format_value)）。
        format_rate.append(format_value)

    # 逐行讲解：检查条件 expected_backend is not None；成立时执行下方分支。
    if expected_backend is not None:
        # 逐行讲解：检查条件 evaluation_sizes != {70}；成立时执行下方分支。
        if evaluation_sizes != {70}:
            # 逐行讲解：抛出异常：ValueError(f'DeepSeek comparison requires the same fixed 70-question set, got {evaluation_sizes}')。
            raise ValueError(
                f"DeepSeek comparison requires the same fixed 70-question set, got {evaluation_sizes}"
            )
        # 逐行讲解：检查条件 len(search_configs) != 1；成立时执行下方分支。
        if len(search_configs) != 1:
            # 逐行讲解：抛出异常：ValueError(f'DeepSeek comparison requires identical search settings, got {search_configs}')。
            raise ValueError(
                f"DeepSeek comparison requires identical search settings, got {search_configs}"
            )

    # 逐行讲解：返回结果：(labels, macro_em, format_rate)。
    return labels, macro_em, format_rate


# 逐行讲解：定义函数 configure_style：Apply a restrained paper-style Matplotlib theme.
def configure_style() -> None:
    """Apply a restrained paper-style Matplotlib theme."""
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


# 逐行讲解：定义函数 add_value_labels：封装下方逻辑，供调用方复用。
def add_value_labels(
    axis: plt.Axes,
    x_positions: list[int],
    values: list[float],
    *,
    offset: float,
) -> None:
    # 逐行讲解：遍历 zip(x_positions, values, strict=True)，将每项绑定到 (x_position, value)。
    for x_position, value in zip(x_positions, values, strict=True):
        # 逐行讲解：调用 axis.text（axis.text(x_position, value + offset, f'{value:.1%}', ha='center', va='bottom', fontsize=9, color='#202020')）。
        axis.text(
            x_position,
            value + offset,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#202020",
        )


# 逐行讲解：定义函数 make_figure：封装下方逻辑，供调用方复用。
def make_figure(
    labels: list[str],
    macro_em: list[float],
    format_rate: list[float],
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
    # 逐行讲解：取值 axes，赋给 (em_axis, format_axis)。
    em_axis, format_axis = axes

    # Panel (a): absolute checkpoint comparison.
    em_colors = ["#D94A4A"] * len(labels)
    # 逐行讲解：取值 '#F28E2B'，赋给 em_colors[2]。
    em_colors[2] = "#F28E2B"
    # 逐行讲解：调用 em_axis.bar（em_axis.bar(x_positions, macro_em, width=0.66, color=em_colors, edgecolor='#202020', linewidth=1.0, alpha=0…），赋给 bars。
    bars = em_axis.bar(
        x_positions,
        macro_em,
        width=0.66,
        color=em_colors,
        edgecolor="#202020",
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
    )
    # 逐行讲解：调用 bars[2].set_hatch（bars[2].set_hatch('///')）。
    bars[2].set_hatch("///")
    # 逐行讲解：调用 add_value_labels（add_value_labels(em_axis, x_positions, macro_em, offset=0.025)）。
    add_value_labels(em_axis, x_positions, macro_em, offset=0.025)
    # 逐行讲解：调用 em_axis.set_title（em_axis.set_title('(a) Macro Exact Match', pad=14)）。
    em_axis.set_title("(a) Macro Exact Match", pad=14)
    # 逐行讲解：调用 em_axis.set_ylabel（em_axis.set_ylabel('Score')）。
    em_axis.set_ylabel("Score")

    # Panel (b): ordered checkpoint trend.
    format_axis.plot(
        x_positions,
        format_rate,
        color="#3388B8",
        marker="s",
        markersize=7,
        markerfacecolor="#57B8D2",
        markeredgecolor="#202020",
        markeredgewidth=0.9,
        linewidth=2.1,
        zorder=4,
    )
    # 逐行讲解：调用 format_axis.scatter（format_axis.scatter([x_positions[2]], [format_rate[2]], s=90, facecolor='#F28E2B', edgecolor='#202020', lin…）。
    format_axis.scatter(
        [x_positions[2]],
        [format_rate[2]],
        s=90,
        facecolor="#F28E2B",
        edgecolor="#202020",
        linewidth=1.0,
        zorder=5,
    )
    # 逐行讲解：调用 add_value_labels（add_value_labels(format_axis, x_positions, format_rate, offset=0.025)）。
    add_value_labels(format_axis, x_positions, format_rate, offset=0.025)
    # 逐行讲解：调用 format_axis.set_title（format_axis.set_title('(b) Valid Answer Format Rate', pad=14)）。
    format_axis.set_title("(b) Valid Answer Format Rate", pad=14)

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
        # 逐行讲解：调用 axis.tick_params（axis.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=5, width=0.9)）。
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=5,
            width=0.9,
        )
        # 逐行讲解：调用 axis.margins（axis.margins(x=0.06)）。
        axis.margins(x=0.06)

    # 逐行讲解：调用 figure.suptitle（figure.suptitle('Search-R1 Checkpoint Evaluation', fontsize=17, y=0.995)）。
    figure.suptitle(
        "Search-R1 Checkpoint Evaluation",
        fontsize=17,
        y=0.995,
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.935, 'Fixed 70-question evaluation set · values read from persisted JSONL summaries', ha…）。
    figure.text(
        0.5,
        0.935,
        "Fixed 70-question evaluation set · values read from persisted JSONL summaries",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.015, 'Orange indicates the best observed Macro EM checkpoint (Step 50).', ha='center', v…）。
    figure.text(
        0.5,
        0.015,
        "Orange indicates the best observed Macro EM checkpoint (Step 50).",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    # 逐行讲解：调用 figure.subplots_adjust（figure.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.84)）。
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.84)
    # 逐行讲解：返回结果：figure。
    return figure


# 逐行讲解：定义函数 make_deepseek_figure：Compare Base and a checkpoint under the same DeepSeek Search environment.
def make_deepseek_figure(
    labels: list[str],
    macro_em: list[float],
    format_rate: list[float],
    *,
    y_limits: tuple[tuple[float, float], tuple[float, float]] = (
        (0.42, 0.52), (0.86, 1.00),
    ),
) -> plt.Figure:
    """Compare Base and a checkpoint under the same DeepSeek Search environment."""
    # 逐行讲解：调用 configure_style（configure_style()）。
    configure_style()

    # 逐行讲解：调用 list（list(range(len(labels)))），赋给 x_positions。
    x_positions = list(range(len(labels)))
    # 逐行讲解：调用 plt.subplots（plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=False, gridspec_kw={'wspace': 0.08})），赋给 (figure, axes)。
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        sharey=False,
        gridspec_kw={"wspace": 0.08},
    )
    # 逐行讲解：构造元组 ((axes[0], macro_em, '(a) Macro Exact Match', y_limits[0]), (axes[1], format_rate, '(b) Valid Answer Format…，赋给 panels。
    panels = (
        (axes[0], macro_em, "(a) Macro Exact Match", y_limits[0]),
        (axes[1], format_rate, "(b) Valid Answer Format Rate", y_limits[1]),
    )

    # 逐行讲解：遍历 panels，将每项绑定到 (axis, values, title, y_limits)。
    for axis, values, title, y_limits in panels:
        # 逐行讲解：通过推导式逐项生成结果：[value - y_limits[0] for value in values]，赋给 visible_heights。
        visible_heights = [value - y_limits[0] for value in values]
        # 逐行讲解：调用 axis.bar（axis.bar(x_positions, visible_heights, bottom=y_limits[0], width=0.48, color=['#A9AFB8', '#3B82F6'], edgeco…），赋给 bars。
        bars = axis.bar(
            x_positions,
            visible_heights,
            bottom=y_limits[0],
            width=0.48,
            color=["#A9AFB8", "#3B82F6"],
            edgecolor="#202020",
            linewidth=1.0,
            alpha=0.96,
            zorder=3,
        )
        # 逐行讲解：调用 bars[1].set_hatch（bars[1].set_hatch('///')）。
        bars[1].set_hatch("///")
        # 逐行讲解：计算 (y_limits[1] - y_limits[0]) * 0.045，赋给 label_offset。
        label_offset = (y_limits[1] - y_limits[0]) * 0.045
        # 逐行讲解：调用 add_value_labels（add_value_labels(axis, x_positions, values, offset=label_offset)）。
        add_value_labels(axis, x_positions, values, offset=label_offset)
        # 逐行讲解：计算 values[1] - values[0]，赋给 gain。
        gain = values[1] - values[0]
        # 逐行讲解：调用 axis.text（axis.text(0.5, 0.16, f'{gain * 100:+.1f} pp', transform=axis.transAxes, ha='center', va='center', fontsize=…）。
        axis.text(
            0.5,
            0.16,
            f"{gain * 100:+.1f} pp",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#2563EB",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#93C5FD",
                "linewidth": 0.8,
                "alpha": 0.96,
            },
            zorder=5,
        )
        # 逐行讲解：调用 axis.set_title（axis.set_title(title, pad=14)）。
        axis.set_title(title, pad=14)
        # 逐行讲解：调用 axis.set_xticks（axis.set_xticks(x_positions, labels)）。
        axis.set_xticks(x_positions, labels)
        # 逐行讲解：调用 axis.set_xlabel（axis.set_xlabel('Model / Checkpoint', labelpad=10)）。
        axis.set_xlabel("Model / Checkpoint", labelpad=10)
        # 逐行讲解：调用 axis.set_ylim（axis.set_ylim(*y_limits)）。
        axis.set_ylim(*y_limits)
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
        # 逐行讲解：调用 axis.tick_params（axis.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=5, width=0.9)）。
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=5,
            width=0.9,
        )
        # 逐行讲解：调用 axis.margins（axis.margins(x=0.28)）。
        axis.margins(x=0.28)
        # 逐行讲解：调用 axis.text（axis.text(0.02, 0.035, 'focused y-axis', transform=axis.transAxes, ha='left', va='bottom', fontsize=8, colo…）。
        axis.text(
            0.02,
            0.035,
            "focused y-axis",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#777777",
            style="italic",
        )

    # 逐行讲解：调用 axes[0].set_ylabel（axes[0].set_ylabel('Score')）。
    axes[0].set_ylabel("Score")
    # 逐行讲解：计算 macro_em[1] - macro_em[0]，赋给 em_gain。
    em_gain = macro_em[1] - macro_em[0]
    # 逐行讲解：计算 format_rate[1] - format_rate[0]，赋给 format_gain。
    format_gain = format_rate[1] - format_rate[0]

    # 逐行讲解：调用 figure.suptitle（figure.suptitle('Search-R1 · DeepSeek Search Evaluation', fontsize=17, y=0.995)）。
    figure.suptitle(
        "Search-R1 · DeepSeek Search Evaluation",
        fontsize=17,
        y=0.995,
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.935, 'Fixed 70-question evaluation set · same DeepSeek Search backend · independent focu…）。
    figure.text(
        0.5,
        0.935,
        (
            "Fixed 70-question evaluation set · same DeepSeek Search backend · "
            "independent focused y-axes"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.015, f"{labels[1]} change: Macro EM {em_gain * 100:+.1f} pp · valid answer format {forma…）。
    figure.text(
        0.5,
        0.015,
        (
            f"{labels[1]} change: Macro EM {em_gain * 100:+.1f} pp · "
            f"valid answer format {format_gain * 100:+.1f} pp · "
            f"{labels[1].removeprefix('Step ')} training steps · "
            "search success 100% in both evaluations."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    # 逐行讲解：调用 figure.subplots_adjust（figure.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.84)）。
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.84)
    # 逐行讲解：返回结果：figure。
    return figure


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：检查条件 args.preset == 'deepseek'；成立时执行下方分支。
    if args.preset == "deepseek":
        # 逐行讲解：调用 load_metrics（load_metrics(args.result_dir, DEEPSEEK_CHECKPOINTS, expected_backend='deepseek')），赋给 (labels, macro_em, format_rate)。
        labels, macro_em, format_rate = load_metrics(
            args.result_dir,
            DEEPSEEK_CHECKPOINTS,
            expected_backend="deepseek",
        )
        # 逐行讲解：调用 make_deepseek_figure（make_deepseek_figure(labels, macro_em, format_rate)），赋给 figure。
        figure = make_deepseek_figure(labels, macro_em, format_rate)
        # 逐行讲解：取值 args.output or DEFAULT_DEEPSEEK_OUTPUT，赋给 output。
        output = args.output or DEFAULT_DEEPSEEK_OUTPUT
    # 逐行讲解：检查条件 args.preset == 'deepseek-api'；成立时执行下方分支。
    elif args.preset == "deepseek-api":
        # 逐行讲解：调用 load_metrics（load_metrics(args.result_dir, DEEPSEEK_API_CHECKPOINTS, expected_backend='deepseek')），赋给 (labels, macro_em, format_rate)。
        labels, macro_em, format_rate = load_metrics(
            args.result_dir,
            DEEPSEEK_API_CHECKPOINTS,
            expected_backend="deepseek",
        )
        # 逐行讲解：调用 make_deepseek_figure（make_deepseek_figure(labels, macro_em, format_rate, y_limits=((0.38, 0.46), (0.8, 0.96)))），赋给 figure。
        figure = make_deepseek_figure(
            labels, macro_em, format_rate,
            y_limits=((0.38, 0.46), (0.80, 0.96)),
        )
        # 逐行讲解：取值 args.output or DEFAULT_DEEPSEEK_API_OUTPUT，赋给 output。
        output = args.output or DEFAULT_DEEPSEEK_API_OUTPUT
    else:
        # 逐行讲解：调用 load_metrics（load_metrics(args.result_dir, ZHIHU_CHECKPOINTS)），赋给 (labels, macro_em, format_rate)。
        labels, macro_em, format_rate = load_metrics(
            args.result_dir,
            ZHIHU_CHECKPOINTS,
        )
        # 逐行讲解：调用 make_figure（make_figure(labels, macro_em, format_rate)），赋给 figure。
        figure = make_figure(labels, macro_em, format_rate)
        # 逐行讲解：取值 args.output or DEFAULT_OUTPUT，赋给 output。
        output = args.output or DEFAULT_OUTPUT

    # 逐行讲解：创建目录（output.parent.mkdir(parents=True, exist_ok=True)）。
    output.parent.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 figure.savefig（figure.savefig(output, dpi=args.dpi, bbox_inches='tight')）。
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    # 逐行讲解：调用 print（print(f'Saved figure: {output}')）。
    print(f"Saved figure: {output}")

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
