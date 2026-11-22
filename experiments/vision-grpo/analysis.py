"""Plot the focused GeoQA evaluation comparison before and after GRPO."""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：导入模块：matplotlib，别名 mpl。
import matplotlib as mpl

# 逐行讲解：调用 mpl.use（mpl.use('Agg')）。
mpl.use("Agg")

# 逐行讲解：导入模块：matplotlib.pyplot，别名 plt。
import matplotlib.pyplot as plt
# 逐行讲解：从 matplotlib.axes 导入 Axes。
from matplotlib.axes import Axes
# 逐行讲解：从 matplotlib.ticker 导入 PercentFormatter。
from matplotlib.ticker import PercentFormatter


# 逐行讲解：构造字典 {'accuracy': 71.0, 'format_rate': 75.0}，赋给 BASE_RESULTS。
BASE_RESULTS = {
    "accuracy": 71.0,
    "format_rate": 75.0,
}
# 逐行讲解：构造字典 {'accuracy': 87.0, 'format_rate': 91.0}，赋给 GRPO_RESULTS。
GRPO_RESULTS = {
    "accuracy": 87.0,
    "format_rate": 91.0,
}

# 逐行讲解：取值 '#AEB4BE'，赋给 BASE_COLOR。
BASE_COLOR = "#AEB4BE"
# 逐行讲解：取值 '#4A86E8'，赋给 GRPO_COLOR。
GRPO_COLOR = "#4A86E8"
# 逐行讲解：取值 '#252525'，赋给 INK。
INK = "#252525"
# 逐行讲解：取值 '#777777'，赋给 MUTED_INK。
MUTED_INK = "#777777"
# 逐行讲解：取值 '#D6D9DE'，赋给 GRID_COLOR。
GRID_COLOR = "#D6D9DE"

# 逐行讲解：计算 Path(__file__).resolve().parent / 'images' / 'eval-comparison.png'，赋给 DEFAULT_OUTPUT。
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "images" / "eval-comparison.png"


# 逐行讲解：定义函数 configure_style：Configure the serif typography and restrained paper-figure palette.
def configure_style() -> None:
    """Configure the serif typography and restrained paper-figure palette."""
    # 逐行讲解：调用 mpl.rcParams.update（mpl.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.labe…）。
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "hatch.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


# 逐行讲解：定义函数 style_axis：Match the framed, inward-tick axes used in the reference figure.
def style_axis(ax: Axes) -> None:
    """Match the framed, inward-tick axes used in the reference figure."""
    # 逐行讲解：调用 ax.set_axisbelow（ax.set_axisbelow(True)）。
    ax.set_axisbelow(True)
    # 逐行讲解：调用 ax.grid（ax.grid(axis='y', color=GRID_COLOR, linestyle='-.', linewidth=0.65)）。
    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linestyle="-.",
        linewidth=0.65,
    )
    # 逐行讲解：遍历 ax.spines.values()，将每项绑定到 spine。
    for spine in ax.spines.values():
        # 逐行讲解：调用 spine.set_visible（spine.set_visible(True)）。
        spine.set_visible(True)
        # 逐行讲解：调用 spine.set_color（spine.set_color(INK)）。
        spine.set_color(INK)
        # 逐行讲解：调用 spine.set_linewidth（spine.set_linewidth(0.8)）。
        spine.set_linewidth(0.8)
    # 逐行讲解：调用 ax.tick_params（ax.tick_params(top=True, right=True)）。
    ax.tick_params(top=True, right=True)
    # 逐行讲解：调用 ax.yaxis.set_major_formatter（ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))）。
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


# 逐行讲解：定义函数 plot_metric：Plot one Base-versus-checkpoint comparison on a focused y-axis.
def plot_metric(
    ax: Axes,
    *,
    title: str,
    base_value: float,
    grpo_value: float,
    y_limits: tuple[float, float],
    y_ticks: list[float],
    show_ylabel: bool,
) -> None:
    """Plot one Base-versus-checkpoint comparison on a focused y-axis."""
    # 逐行讲解：构造列表 [0, 1]，赋给 x_positions。
    x_positions = [0, 1]
    # 逐行讲解：调用 ax.bar（ax.bar(x_positions, [base_value, grpo_value], width=0.48, color=[BASE_COLOR, GRPO_COLOR], edgecolor=INK, li…），赋给 bars。
    bars = ax.bar(
        x_positions,
        [base_value, grpo_value],
        width=0.48,
        color=[BASE_COLOR, GRPO_COLOR],
        edgecolor=INK,
        linewidth=0.85,
        zorder=3,
    )
    # 逐行讲解：调用 bars[1].set_hatch（bars[1].set_hatch('///')）。
    bars[1].set_hatch("///")

    # 逐行讲解：计算 (y_limits[1] - y_limits[0]) * 0.04，赋给 value_offset。
    value_offset = (y_limits[1] - y_limits[0]) * 0.04
    # 逐行讲解：遍历 zip(x_positions, [base_value, grpo_value], strict=True)，将每项绑定到 (x_position, value)。
    for x_position, value in zip(
        x_positions,
        [base_value, grpo_value],
        strict=True,
    ):
        # 逐行讲解：调用 ax.text（ax.text(x_position, value + value_offset, f'{value:.1f}%', ha='center', va='bottom', color=INK, fontsize=8.0)）。
        ax.text(
            x_position,
            value + value_offset,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=8.0,
        )

    # 逐行讲解：计算 grpo_value - base_value，赋给 delta。
    delta = grpo_value - base_value
    # 逐行讲解：计算 (base_value + grpo_value) / 2 + 1.0，赋给 delta_y。
    delta_y = (base_value + grpo_value) / 2 + 1.0
    # 逐行讲解：调用 ax.text（ax.text(0.5, delta_y, f'+{delta:.1f} pp', ha='center', va='center', color='#2563C4', fontsize=8.0, fontweig…）。
    ax.text(
        0.5,
        delta_y,
        f"+{delta:.1f} pp",
        ha="center",
        va="center",
        color="#2563C4",
        fontsize=8.0,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.23",
            "facecolor": "white",
            "edgecolor": "#7DB4FF",
            "linewidth": 0.9,
        },
        zorder=5,
    )

    # 逐行讲解：调用 ax.text（ax.text(0.02, 0.035, 'focused y-axis', transform=ax.transAxes, ha='left', va='bottom', color='#9A9A9A', fon…）。
    ax.text(
        0.02,
        0.035,
        "focused y-axis",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#9A9A9A",
        fontsize=6.7,
        fontstyle="italic",
    )
    # 逐行讲解：调用 ax.set_title（ax.set_title(title, pad=10)）。
    ax.set_title(title, pad=10)
    # 逐行讲解：调用 ax.set_xlabel（ax.set_xlabel('Model / Checkpoint', labelpad=8)）。
    ax.set_xlabel("Model / Checkpoint", labelpad=8)
    # 逐行讲解：检查条件 show_ylabel；成立时执行下方分支。
    if show_ylabel:
        # 逐行讲解：调用 ax.set_ylabel（ax.set_ylabel('Score')）。
        ax.set_ylabel("Score")
    # 逐行讲解：调用 ax.set_xticks（ax.set_xticks(x_positions, ['Base', 'Step 100'])）。
    ax.set_xticks(x_positions, ["Base", "Step 100"])
    # 逐行讲解：调用 ax.set_xlim（ax.set_xlim(-0.72, 1.72)）。
    ax.set_xlim(-0.72, 1.72)
    # 逐行讲解：调用 ax.set_ylim（ax.set_ylim(*y_limits)）。
    ax.set_ylim(*y_limits)
    # 逐行讲解：调用 ax.set_yticks（ax.set_yticks(y_ticks)）。
    ax.set_yticks(y_ticks)
    # 逐行讲解：调用 style_axis（style_axis(ax)）。
    style_axis(ax)


# 逐行讲解：定义函数 build_figure：Build the two-panel publication-style comparison figure.
def build_figure() -> plt.Figure:
    """Build the two-panel publication-style comparison figure."""
    # 逐行讲解：调用 configure_style（configure_style()）。
    configure_style()
    # 逐行讲解：调用 plt.subplots（plt.subplots(1, 2, figsize=(7.2, 3.0))），赋给 (figure, axes)。
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # 逐行讲解：调用 plot_metric（plot_metric(axes[0], title='(a) Accuracy', base_value=BASE_RESULTS['accuracy'], grpo_value=GRPO_RESULTS['ac…）。
    plot_metric(
        axes[0],
        title="(a) Accuracy",
        base_value=BASE_RESULTS["accuracy"],
        grpo_value=GRPO_RESULTS["accuracy"],
        y_limits=(68, 92),
        y_ticks=[68, 72, 76, 80, 84, 88, 92],
        show_ylabel=True,
    )
    # 逐行讲解：调用 plot_metric（plot_metric(axes[1], title='(b) Format Rate', base_value=BASE_RESULTS['format_rate'], grpo_value=GRPO_RESUL…）。
    plot_metric(
        axes[1],
        title="(b) Format Rate",
        base_value=BASE_RESULTS["format_rate"],
        grpo_value=GRPO_RESULTS["format_rate"],
        y_limits=(72, 96),
        y_ticks=[72, 76, 80, 84, 88, 92, 96],
        show_ylabel=False,
    )

    # 逐行讲解：调用 figure.suptitle（figure.suptitle('GeoQA · Vision GRPO Evaluation', x=0.5, y=0.965, ha='center', fontsize=12.0, fontweight='n…）。
    figure.suptitle(
        "GeoQA · Vision GRPO Evaluation",
        x=0.5,
        y=0.965,
        ha="center",
        fontsize=12.0,
        fontweight="normal",
        color=INK,
    )
    # 逐行讲解：调用 figure.text（figure.text(0.5, 0.875, 'Fixed 100-question evaluation set · Qwen3.5-4B · independent focused y-axes', ha='…）。
    figure.text(
        0.5,
        0.875,
        "Fixed 100-question evaluation set · Qwen3.5-4B · independent focused y-axes",
        ha="center",
        va="center",
        fontsize=8.0,
        color=MUTED_INK,
    )
    # 逐行讲解：调用 figure.subplots_adjust（figure.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.72, wspace=0.22)）。
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.72,
        wspace=0.22,
    )
    # 逐行讲解：返回结果：figure。
    return figure


# 逐行讲解：定义函数 save_figure：Save the comparison as a 600 DPI PNG.
def save_figure(output: Path) -> Path:
    """Save the comparison as a 600 DPI PNG."""
    # 逐行讲解：按条件选择结果：output if output.suffix.lower() == '.png' else output.with_suffix('.png')，赋给 png_path。
    png_path = output if output.suffix.lower() == ".png" else output.with_suffix(".png")
    # 逐行讲解：创建目录（png_path.parent.mkdir(parents=True, exist_ok=True)）。
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # 逐行讲解：调用 build_figure（build_figure()），赋给 figure。
    figure = build_figure()
    # 逐行讲解：调用 figure.savefig（figure.savefig(png_path, dpi=600)）。
    figure.savefig(png_path, dpi=600)
    # 逐行讲解：调用 plt.close（plt.close(figure)）。
    plt.close(figure)
    # 逐行讲解：返回结果：png_path。
    return png_path


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='Plot the Base vs. GRPO step-100 GeoQA evaluation results.')），赋给 parser。
    parser = argparse.ArgumentParser(
        description="Plot the Base vs. GRPO step-100 GeoQA evaluation results."
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='PNG output path.')）。
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path.",
    )
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：调用 save_figure（save_figure(args.output)），赋给 png_path。
    png_path = save_figure(args.output)
    # 逐行讲解：调用 print（print(f'Saved {png_path}')）。
    print(f"Saved {png_path}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
