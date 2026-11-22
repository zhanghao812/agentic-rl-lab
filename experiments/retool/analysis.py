"""Plot Average@n / Pass@n and format rate across ReTool checkpoints.

Run from the repository root:

    uv run python experiments/retool/analysis.py

The figure is written to ``experiments/retool/images/checkpoint_avg_pass_format.png``
by default. Each metric is read from the final ``type=summary`` record of the
corresponding evaluation JSONL file.
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
# 逐行讲解：计算 SCRIPT_DIR / 'eval-results'，赋给 DEFAULT_RESULT_DIR。
DEFAULT_RESULT_DIR = SCRIPT_DIR / "eval-results"
# 逐行讲解：计算 SCRIPT_DIR / 'images' / 'checkpoint_avg_pass_format.png'，赋给 DEFAULT_OUTPUT。
DEFAULT_OUTPUT = SCRIPT_DIR / "images" / "checkpoint_avg_pass_format.png"

# 逐行讲解：构造元组 (('Base', 'aime25-retool-base.jsonl'), ('Step 20', 'aime25-retool-step20.jsonl'), ('Step 50', 'aime25-retoo…，赋给 CHECKPOINTS。
CHECKPOINTS = (
    ("Base", "aime25-retool-base.jsonl"),
    ("Step 20", "aime25-retool-step20.jsonl"),
    ("Step 50", "aime25-retool-step50.jsonl"),
    ("Step 100", "aime25-retool-step100.jsonl"),
    ("Step 150", "aime25-retool-step150.jsonl"),
    ("Step 200", "aime25-retool-step200.jsonl"),
)


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--result-dir', type=Path, default=DEFAULT_RESULT_DIR, help='Directory containing aime2…）。
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing aime25-retool-*.jsonl",
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


# 逐行讲解：定义函数 load_metrics：封装下方逻辑，供调用方复用。
def load_metrics(
    result_dir: Path,
) -> tuple[list[str], int, list[float], list[float], list[float]]:
    # 逐行讲解：声明 labels 的类型为 list[str]；构造列表 []。
    labels: list[str] = []
    # 逐行讲解：声明 val_n 的类型为 int | None；取值 None。
    val_n: int | None = None
    # 逐行讲解：声明 average_at_n 的类型为 list[float]；构造列表 []。
    average_at_n: list[float] = []
    # 逐行讲解：声明 pass_at_n 的类型为 list[float]；构造列表 []。
    pass_at_n: list[float] = []
    # 逐行讲解：声明 format_rate 的类型为 list[float]；构造列表 []。
    format_rate: list[float] = []

    # 逐行讲解：遍历 CHECKPOINTS，将每项绑定到 (label, filename)。
    for label, filename in CHECKPOINTS:
        # 逐行讲解：计算 result_dir / filename，赋给 path。
        path = result_dir / filename
        # 逐行讲解：检查条件 not path.is_file()；成立时执行下方分支。
        if not path.is_file():
            # 逐行讲解：抛出异常：FileNotFoundError(f'Missing evaluation result: {path}')。
            raise FileNotFoundError(f"Missing evaluation result: {path}")

        # 逐行讲解：调用 load_summary（load_summary(path)），赋给 summary。
        summary = load_summary(path)
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：调用 int（int(summary['val_n'])），赋给 n_value。
            n_value = int(summary["val_n"])
            # 逐行讲解：转换为浮点表示（float(summary['average_at_n'])），赋给 average_value。
            average_value = float(summary["average_at_n"])
            # 逐行讲解：转换为浮点表示（float(summary['pass_at_n'])），赋给 pass_value。
            pass_value = float(summary["pass_at_n"])
            # 逐行讲解：转换为浮点表示（float(summary['format_rate'])），赋给 format_value。
            format_value = float(summary["format_rate"])
        except (KeyError, TypeError, ValueError) as error:
            # 逐行讲解：抛出异常：ValueError(f'Expected numeric val_n / average_at_n / pass_at_n / format_rate metrics in {path}')。
            raise ValueError(
                "Expected numeric val_n / average_at_n / pass_at_n / format_rate "
                f"metrics in {path}"
            ) from error

        # 逐行讲解：检查条件 val_n is None；成立时执行下方分支。
        if val_n is None:
            # 逐行讲解：取值 n_value，赋给 val_n。
            val_n = n_value
        # 逐行讲解：检查条件 val_n != n_value；成立时执行下方分支。
        elif val_n != n_value:
            # 逐行讲解：抛出异常：ValueError(f'Inconsistent val_n in {path}: {n_value} != {val_n}')。
            raise ValueError(
                f"Inconsistent val_n in {path}: {n_value} != {val_n}"
            )

        # 逐行讲解：向列表末尾追加一个元素（labels.append(label)）。
        labels.append(label)
        # 逐行讲解：向列表末尾追加一个元素（average_at_n.append(average_value)）。
        average_at_n.append(average_value)
        # 逐行讲解：向列表末尾追加一个元素（pass_at_n.append(pass_value)）。
        pass_at_n.append(pass_value)
        # 逐行讲解：向列表末尾追加一个元素（format_rate.append(format_value)）。
        format_rate.append(format_value)

    # 逐行讲解：断言 val_n is not None；条件不成立时报告错误。
    assert val_n is not None
    # 逐行讲解：返回结果：(labels, val_n, average_at_n, pass_at_n, format_rate)。
    return labels, val_n, average_at_n, pass_at_n, format_rate


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
    x_positions: list[float],
    values: list[float],
    *,
    offset: float,
    fontsize: int = 9,
) -> None:
    # 逐行讲解：遍历 zip(x_positions, values, strict=True)，将每项绑定到 (x_position, value)。
    for x_position, value in zip(x_positions, values, strict=True):
        # 逐行讲解：调用 axis.text（axis.text(x_position, value + offset, f'{value:.1%}', ha='center', va='bottom', fontsize=fontsize, color='#…）。
        axis.text(
            x_position,
            value + offset,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#202020",
        )


# 逐行讲解：定义函数 make_figure：封装下方逻辑，供调用方复用。
def make_figure(
    labels: list[str],
    val_n: int,
    average_at_n: list[float],
    pass_at_n: list[float],
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
    # 逐行讲解：取值 axes，赋给 (accuracy_axis, format_axis)。
    accuracy_axis, format_axis = axes

    # Panel (a): grouped bars for Average@n and Pass@n.
    bar_width = 0.36
    # 逐行讲解：通过推导式逐项生成结果：[x - bar_width / 2 for x in x_positions]，赋给 average_positions。
    average_positions = [x - bar_width / 2 for x in x_positions]
    # 逐行讲解：通过推导式逐项生成结果：[x + bar_width / 2 for x in x_positions]，赋给 pass_positions。
    pass_positions = [x + bar_width / 2 for x in x_positions]
    # 逐行讲解：调用 accuracy_axis.bar（accuracy_axis.bar(average_positions, average_at_n, width=bar_width, color='#D94A4A', edgecolor='#202020', l…）。
    accuracy_axis.bar(
        average_positions,
        average_at_n,
        width=bar_width,
        color="#D94A4A",
        edgecolor="#202020",
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
        label=f"Average@{val_n}",
    )
    # 逐行讲解：调用 accuracy_axis.bar（accuracy_axis.bar(pass_positions, pass_at_n, width=bar_width, color='#3388B8', edgecolor='#202020', linewid…）。
    accuracy_axis.bar(
        pass_positions,
        pass_at_n,
        width=bar_width,
        color="#3388B8",
        edgecolor="#202020",
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
        label=f"Pass@{val_n}",
    )
    # 逐行讲解：调用 add_value_labels（add_value_labels(accuracy_axis, average_positions, average_at_n, offset=0.02, fontsize=8)）。
    add_value_labels(
        accuracy_axis, average_positions, average_at_n, offset=0.02, fontsize=8
    )
    # 逐行讲解：调用 add_value_labels（add_value_labels(accuracy_axis, pass_positions, pass_at_n, offset=0.02, fontsize=8)）。
    add_value_labels(
        accuracy_axis, pass_positions, pass_at_n, offset=0.02, fontsize=8
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

    # Panel (b): ordered checkpoint trend of the format rate.
    format_axis.plot(
        x_positions,
        format_rate,
        color="#F28E2B",
        marker="s",
        markersize=7,
        markerfacecolor="#F2B36B",
        markeredgecolor="#202020",
        markeredgewidth=0.9,
        linewidth=2.1,
        zorder=4,
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

    # 逐行讲解：调用 figure.suptitle（figure.suptitle('ReTool Checkpoint Evaluation', fontsize=17, y=0.995)）。
    figure.suptitle(
        "ReTool Checkpoint Evaluation",
        fontsize=17,
        y=0.995,
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.935, 'AIME 2025 (30 problems) · val-n 12 · temperature 1.0', ha='center', va='top', font…）。
    figure.text(
        0.5,
        0.935,
        "AIME 2025 (30 problems) · val-n 12 · temperature 1.0",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    # 逐行讲解：调用 figure.subplots_adjust（figure.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.84)）。
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.84)
    # 逐行讲解：返回结果：figure。
    return figure


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：调用 load_metrics（load_metrics(args.result_dir)），赋给 (labels, val_n, average_at_n, pass_at_n, format_rate)。
    labels, val_n, average_at_n, pass_at_n, format_rate = load_metrics(
        args.result_dir
    )
    # 逐行讲解：调用 make_figure（make_figure(labels, val_n, average_at_n, pass_at_n, format_rate)），赋给 figure。
    figure = make_figure(labels, val_n, average_at_n, pass_at_n, format_rate)

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
