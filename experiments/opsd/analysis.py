"""Plot AIME25 Average@12 / Pass@12 across OPSD checkpoints.

Run from the repository root:

    uv run python experiments/opsd/analysis.py

The figure is written to ``experiments/opsd/images/aime25-opsd-progress.png``.
Metrics are read from the ``type=summary`` record of each evaluation JSONL
in ``experiments/opsd/eval-results/``; new ``aime25-sampler-steps*.jsonl`` files are
picked up automatically.
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：re。
import re
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
# 逐行讲解：计算 SCRIPT_DIR / 'eval-results'，赋给 DEFAULT_RESULT_DIR。
DEFAULT_RESULT_DIR = SCRIPT_DIR / "eval-results"
# 逐行讲解：计算 SCRIPT_DIR / 'images' / 'aime25-opsd-progress.png'，赋给 DEFAULT_OUTPUT。
DEFAULT_OUTPUT = SCRIPT_DIR / "images" / "aime25-opsd-progress.png"

# 逐行讲解：取值 '#D94A4A'，赋给 COLOR_AVG。
COLOR_AVG = "#D94A4A"
# 逐行讲解：取值 '#3388B8'，赋给 COLOR_PASS。
COLOR_PASS = "#3388B8"
# 逐行讲解：取值 '#F28E2B'，赋给 COLOR_BEST。
COLOR_BEST = "#F28E2B"
# 逐行讲解：取值 '#202020'，赋给 COLOR_EDGE。
COLOR_EDGE = "#202020"


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--result-dir', type=Path, default=DEFAULT_RESULT_DIR, help='Directory containing aime2…）。
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing aime25*.jsonl evaluation results",
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


# 逐行讲解：定义函数 load_metrics：Collect (label, Average@12, Pass@12) per checkpoint, sorted by step.
def load_metrics(result_dir: Path) -> tuple[list[str], list[float], list[float]]:
    """Collect (label, Average@12, Pass@12) per checkpoint, sorted by step."""
    # 逐行讲解：声明 checkpoints 的类型为 list[tuple[int, str, Path]]；构造列表 [(0, 'Base', result_dir / 'aime25-base.jsonl')]。
    checkpoints: list[tuple[int, str, Path]] = [(0, "Base", result_dir / "aime25-base.jsonl")]
    # 逐行讲解：遍历 result_dir.glob('aime25-sampler-steps*.jsonl')，将每项绑定到 path。
    for path in result_dir.glob("aime25-sampler-steps*.jsonl"):
        # 逐行讲解：调用 re.search（re.search('steps(\\d+)', path.name)），赋给 match。
        match = re.search(r"steps(\d+)", path.name)
        # 逐行讲解：检查条件 match；成立时执行下方分支。
        if match:
            # 逐行讲解：向列表末尾追加一个元素（checkpoints.append((int(match.group(1)), f'Step {match.group(1)}', path))）。
            checkpoints.append((int(match.group(1)), f"Step {match.group(1)}", path))
    # 逐行讲解：调用 checkpoints.sort（checkpoints.sort()）。
    checkpoints.sort()

    # 逐行讲解：声明 labels 的类型为 list[str]；构造列表 []。
    labels: list[str] = []
    # 逐行讲解：声明 average_at_n 的类型为 list[float]；构造列表 []。
    average_at_n: list[float] = []
    # 逐行讲解：声明 pass_at_n 的类型为 list[float]；构造列表 []。
    pass_at_n: list[float] = []
    # 逐行讲解：遍历 checkpoints，将每项绑定到 (_, label, path)。
    for _, label, path in checkpoints:
        # 逐行讲解：检查条件 not path.is_file()；成立时执行下方分支。
        if not path.is_file():
            # 逐行讲解：抛出异常：FileNotFoundError(f'Missing evaluation result: {path}')。
            raise FileNotFoundError(f"Missing evaluation result: {path}")
        # 逐行讲解：调用 load_summary（load_summary(path)），赋给 summary。
        summary = load_summary(path)
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：转换为浮点表示（float(summary['average_at_n'])），赋给 avg_value。
            avg_value = float(summary["average_at_n"])
            # 逐行讲解：转换为浮点表示（float(summary['pass_at_n'])），赋给 pass_value。
            pass_value = float(summary["pass_at_n"])
        except (KeyError, TypeError, ValueError) as error:
            # 逐行讲解：抛出异常：ValueError(f'Expected numeric average_at_n and pass_at_n metrics in {path}')。
            raise ValueError(
                f"Expected numeric average_at_n and pass_at_n metrics in {path}"
            ) from error
        # 逐行讲解：向列表末尾追加一个元素（labels.append(label)）。
        labels.append(label)
        # 逐行讲解：向列表末尾追加一个元素（average_at_n.append(avg_value)）。
        average_at_n.append(avg_value)
        # 逐行讲解：向列表末尾追加一个元素（pass_at_n.append(pass_value)）。
        pass_at_n.append(pass_value)

    # 逐行讲解：返回结果：(labels, average_at_n, pass_at_n)。
    return labels, average_at_n, pass_at_n


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
            "axes.edgecolor": COLOR_EDGE,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


# 逐行讲解：定义函数 make_figure：封装下方逻辑，供调用方复用。
def make_figure(
    labels: list[str],
    average_at_n: list[float],
    pass_at_n: list[float],
) -> plt.Figure:
    # 逐行讲解：调用 configure_style（configure_style()）。
    configure_style()

    # 逐行讲解：调用 list（list(range(len(labels)))），赋给 x_positions。
    x_positions = list(range(len(labels)))
    # 逐行讲解：调用 plt.subplots（plt.subplots(1, 2, figsize=(13.0, 5.4), gridspec_kw={'wspace': 0.16})），赋给 (figure, axes)。
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        gridspec_kw={"wspace": 0.16},
    )
    # 逐行讲解：取值 axes，赋给 (avg_axis, pass_axis)。
    avg_axis, pass_axis = axes

    # 逐行讲解：定义函数 bar_panel：封装下方逻辑，供调用方复用。
    def bar_panel(axis: plt.Axes, values: list[float], color: str, title: str) -> Any:
        # 逐行讲解：调用 axis.bar（axis.bar(x_positions, values, 0.62, color=color, edgecolor=COLOR_EDGE, linewidth=1.0, alpha=0.96, zorder=3)），赋给 bars。
        bars = axis.bar(
            x_positions,
            values,
            0.62,
            color=color,
            edgecolor=COLOR_EDGE,
            linewidth=1.0,
            alpha=0.96,
            zorder=3,
        )
        # 逐行讲解：计算 min(values) - 0.04，赋给 low。
        low = min(values) - 0.04
        # 逐行讲解：计算 max(values) + 0.05，赋给 high。
        high = max(values) + 0.05
        # 逐行讲解：调用 axis.set_ylim（axis.set_ylim(low, high)）。
        axis.set_ylim(low, high)
        # 逐行讲解：遍历 bars，将每项绑定到 bar。
        for bar in bars:
            # 逐行讲解：调用 axis.text（axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (high - low) * 0.015, f'{bar.get_height():.…）。
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (high - low) * 0.015,
                f"{bar.get_height():.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLOR_EDGE,
            )
        # 逐行讲解：调用 axis.set_title（axis.set_title(title, pad=14)）。
        axis.set_title(title, pad=14)
        # 逐行讲解：返回结果：bars。
        return bars

    # Panel (a): Average@12 柱状图，橙色 + 斜纹标出最佳 checkpoint（与 experiments/search-r1 图例一致）。
    avg_bars = bar_panel(avg_axis, average_at_n, COLOR_AVG, "(a) Average@12")
    # 逐行讲解：调用 max（max(range(len(average_at_n)), key=average_at_n.__getitem__)），赋给 best_index。
    best_index = max(range(len(average_at_n)), key=average_at_n.__getitem__)
    # 逐行讲解：调用 avg_bars[best_index].set_color（avg_bars[best_index].set_color(COLOR_BEST)）。
    avg_bars[best_index].set_color(COLOR_BEST)
    # 逐行讲解：调用 avg_bars[best_index].set_hatch（avg_bars[best_index].set_hatch('///')）。
    avg_bars[best_index].set_hatch("///")
    # 逐行讲解：调用 avg_axis.set_ylabel（avg_axis.set_ylabel('Accuracy')）。
    avg_axis.set_ylabel("Accuracy")

    # Panel (b): Pass@12 折线图，橙色圆点同步标出最佳 Average@12 对应的 checkpoint。
    low = min(pass_at_n) - 0.04
    # 逐行讲解：计算 max(pass_at_n) + 0.05，赋给 high。
    high = max(pass_at_n) + 0.05
    # 逐行讲解：调用 pass_axis.set_ylim（pass_axis.set_ylim(low, high)）。
    pass_axis.set_ylim(low, high)
    # 逐行讲解：调用 pass_axis.plot（pass_axis.plot(x_positions, pass_at_n, color=COLOR_PASS, marker='s', markersize=7, markerfacecolor='#57B8D2…）。
    pass_axis.plot(
        x_positions,
        pass_at_n,
        color=COLOR_PASS,
        marker="s",
        markersize=7,
        markerfacecolor="#57B8D2",
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.9,
        linewidth=2.1,
        zorder=4,
    )
    # 逐行讲解：调用 pass_axis.scatter（pass_axis.scatter([x_positions[best_index]], [pass_at_n[best_index]], s=90, facecolor=COLOR_BEST, edgecolor…）。
    pass_axis.scatter(
        [x_positions[best_index]],
        [pass_at_n[best_index]],
        s=90,
        facecolor=COLOR_BEST,
        edgecolor=COLOR_EDGE,
        linewidth=1.0,
        zorder=5,
    )
    # 逐行讲解：遍历 zip(x_positions, pass_at_n, strict=True)，将每项绑定到 (x, value)。
    for x, value in zip(x_positions, pass_at_n, strict=True):
        # 逐行讲解：调用 pass_axis.text（pass_axis.text(x, value + (high - low) * 0.015, f'{value:.1%}', ha='center', va='bottom', fontsize=9, color…）。
        pass_axis.text(
            x,
            value + (high - low) * 0.015,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLOR_EDGE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2},
            zorder=6,
        )
    # 逐行讲解：调用 pass_axis.set_title（pass_axis.set_title('(b) Pass@12', pad=14)）。
    pass_axis.set_title("(b) Pass@12", pad=14)

    # 逐行讲解：遍历 axes，将每项绑定到 axis。
    for axis in axes:
        # 逐行讲解：调用 axis.set_xticks（axis.set_xticks(x_positions, labels)）。
        axis.set_xticks(x_positions, labels)
        # 逐行讲解：调用 axis.set_xlabel（axis.set_xlabel('Model / Checkpoint', labelpad=10)）。
        axis.set_xlabel("Model / Checkpoint", labelpad=10)
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

    # 逐行讲解：调用 figure.suptitle（figure.suptitle('OPSD AIME25 Checkpoint Evaluation', fontsize=17, y=0.98)）。
    figure.suptitle(
        "OPSD AIME25 Checkpoint Evaluation",
        fontsize=17,
        y=0.98,
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.915, '30 problems × 12 samples per checkpoint · values read from persisted JSONL summari…）。
    figure.text(
        0.5,
        0.915,
        "30 problems × 12 samples per checkpoint · values read from persisted JSONL summaries",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.015, f'Orange indicates the best observed Average@12 checkpoint ({labels[best_index]}). …）。
    figure.text(
        0.5,
        0.015,
        f"Orange indicates the best observed Average@12 checkpoint ({labels[best_index]}). "
        "Y-axes are truncated to highlight trends.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    # 逐行讲解：调用 figure.subplots_adjust（figure.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.83)）。
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.83)
    # 逐行讲解：返回结果：figure。
    return figure


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：调用 load_metrics（load_metrics(args.result_dir)），赋给 (labels, average_at_n, pass_at_n)。
    labels, average_at_n, pass_at_n = load_metrics(args.result_dir)
    # 逐行讲解：调用 make_figure（make_figure(labels, average_at_n, pass_at_n)），赋给 figure。
    figure = make_figure(labels, average_at_n, pass_at_n)

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
