"""Visualize the final SAR-OPD and IDT-OPD benchmark results.

The script creates two publication-style figures. Each figure contains two
subplots and is exported as both a 300-DPI PNG and a vector PDF.

Usage:
    uv run python experiments/medical-opd/analysis.py
    uv run python experiments/medical-opd/analysis.py --output-dir experiments/medical-opd/images --dpi 300
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：导入模块：matplotlib.pyplot，别名 plt。
import matplotlib.pyplot as plt
# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：从 matplotlib.lines 导入 Line2D。
from matplotlib.lines import Line2D
# 逐行讲解：从 matplotlib.patches 导入 Patch。
from matplotlib.patches import Patch


# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'images'，赋给 DEFAULT_OUTPUT_DIR。
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "images"

# 逐行讲解：取值 '#DB4B4B'，赋给 MEDICAL_COLOR。
MEDICAL_COLOR = "#DB4B4B"
# 逐行讲解：取值 '#4B93C3'，赋给 GENERAL_COLOR。
GENERAL_COLOR = "#4B93C3"
# 逐行讲解：取值 '#22252A'，赋给 INK_COLOR。
INK_COLOR = "#22252A"
# 逐行讲解：取值 '#C8CDD2'，赋给 GRID_COLOR。
GRID_COLOR = "#C8CDD2"
# 逐行讲解：取值 '#8A9199'，赋给 NEUTRAL_COLOR。
NEUTRAL_COLOR = "#8A9199"
# 逐行讲解：取值 '#E6E9EC'，赋给 LIGHT_NEUTRAL。
LIGHT_NEUTRAL = "#E6E9EC"

# 逐行讲解：取值 72.17，赋给 BASE_MEDQA。
BASE_MEDQA = 72.17
# 逐行讲解：取值 81.67，赋给 BASE_CEVAL。
BASE_CEVAL = 81.67


# 逐行讲解：定义函数 parse_args：Parse output settings without coupling plotting to the training code.
def parse_args() -> argparse.Namespace:
    """Parse output settings without coupling plotting to the training code."""

    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='Plot SAR-OPD and IDT-OPD results')），赋给 parser。
    parser = argparse.ArgumentParser(description="Plot SAR-OPD and IDT-OPD results")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory for PNG and PDF …）。
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG and PDF figures",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dpi', type=int, default=300, help='PNG resolution')）。
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()
    # 逐行讲解：检查条件 args.dpi <= 0；成立时执行下方分支。
    if args.dpi <= 0:
        # 逐行讲解：抛出异常：ValueError('--dpi must be > 0')。
        raise ValueError("--dpi must be > 0")
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义函数 configure_style：Apply one restrained, reference-inspired visual system.
def configure_style() -> None:
    """Apply one restrained, reference-inspired visual system."""

    # 逐行讲解：调用 plt.rcParams.update（plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize'…）。
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": INK_COLOR,
            "axes.linewidth": 1.15,
            "axes.titleweight": "bold",
            "text.color": INK_COLOR,
            "axes.labelcolor": INK_COLOR,
            "xtick.color": INK_COLOR,
            "ytick.color": INK_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# 逐行讲解：定义函数 style_axes：Match the quiet grid, dark frame, and inward ticks of the reference.
def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    """Match the quiet grid, dark frame, and inward ticks of the reference."""

    # 逐行讲解：调用 ax.set_axisbelow（ax.set_axisbelow(True)）。
    ax.set_axisbelow(True)
    # 逐行讲解：调用 ax.grid（ax.grid(axis=grid_axis, color=GRID_COLOR, linestyle='-.', linewidth=0.8, alpha=0.8)）。
    ax.grid(
        axis=grid_axis,
        color=GRID_COLOR,
        linestyle="-.",
        linewidth=0.8,
        alpha=0.8,
    )
    # 逐行讲解：调用 ax.tick_params（ax.tick_params(direction='in', top=True, right=True, length=5, width=1.0)）。
    ax.tick_params(direction="in", top=True, right=True, length=5, width=1.0)
    # 逐行讲解：遍历 ax.spines.values()，将每项绑定到 spine。
    for spine in ax.spines.values():
        # 逐行讲解：调用 spine.set_color（spine.set_color(INK_COLOR)）。
        spine.set_color(INK_COLOR)
        # 逐行讲解：调用 spine.set_linewidth（spine.set_linewidth(1.15)）。
        spine.set_linewidth(1.15)


# 逐行讲解：定义函数 add_bar_labels：Place exact percentage labels above grouped bars.
def add_bar_labels(ax: plt.Axes, bars: list, values: list[float]) -> None:
    """Place exact percentage labels above grouped bars."""

    # 逐行讲解：遍历 zip(bars, values, strict=True)，将每项绑定到 (bar, value)。
    for bar, value in zip(bars, values, strict=True):
        # 逐行讲解：调用 ax.text（ax.text(bar.get_x() + bar.get_width() / 2, value + 1.25, f'{value:.2f}', ha='center', va='bottom', fontsize…）。
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.25,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=INK_COLOR,
        )


# 逐行讲解：定义函数 draw_grouped_bars：Draw a zero-based grouped comparison with color and hatch redundancy.
def draw_grouped_bars(
    ax: plt.Axes,
    labels: list[str],
    medqa: list[float],
    ceval: list[float],
    *,
    highlighted_indices: set[int],
) -> None:
    """Draw a zero-based grouped comparison with color and hatch redundancy."""

    # 逐行讲解：调用 np.arange（np.arange(len(labels), dtype=float)），赋给 x。
    x = np.arange(len(labels), dtype=float)
    # 逐行讲解：取值 0.34，赋给 width。
    width = 0.34
    # 逐行讲解：调用 ax.bar（ax.bar(x - width / 2, medqa, width, color=MEDICAL_COLOR, edgecolor=INK_COLOR, linewidth=1.15, alpha=0.95, l…），赋给 med_bars。
    med_bars = ax.bar(
        x - width / 2,
        medqa,
        width,
        color=MEDICAL_COLOR,
        edgecolor=INK_COLOR,
        linewidth=1.15,
        alpha=0.95,
        label="MedQA-zh 1k",
        zorder=3,
    )
    # 逐行讲解：调用 ax.bar（ax.bar(x + width / 2, ceval, width, color=GENERAL_COLOR, edgecolor=INK_COLOR, linewidth=1.15, hatch='///', …），赋给 ceval_bars。
    ceval_bars = ax.bar(
        x + width / 2,
        ceval,
        width,
        color=GENERAL_COLOR,
        edgecolor=INK_COLOR,
        linewidth=1.15,
        hatch="///",
        alpha=0.92,
        label="C-Eval non-med 8k",
        zorder=3,
    )
    # 逐行讲解：遍历 highlighted_indices，将每项绑定到 index。
    for index in highlighted_indices:
        # 逐行讲解：调用 med_bars[index].set_linewidth（med_bars[index].set_linewidth(2.3)）。
        med_bars[index].set_linewidth(2.3)
        # 逐行讲解：调用 ceval_bars[index].set_linewidth（ceval_bars[index].set_linewidth(2.3)）。
        ceval_bars[index].set_linewidth(2.3)

    # 逐行讲解：调用 add_bar_labels（add_bar_labels(ax, list(med_bars), medqa)）。
    add_bar_labels(ax, list(med_bars), medqa)
    # 逐行讲解：调用 add_bar_labels（add_bar_labels(ax, list(ceval_bars), ceval)）。
    add_bar_labels(ax, list(ceval_bars), ceval)
    # 逐行讲解：调用 ax.set_xticks（ax.set_xticks(x, labels)）。
    ax.set_xticks(x, labels)
    # 逐行讲解：调用 ax.set_ylabel（ax.set_ylabel('Accuracy (%)')）。
    ax.set_ylabel("Accuracy (%)")
    # 逐行讲解：调用 ax.set_ylim（ax.set_ylim(0, 100)）。
    ax.set_ylim(0, 100)
    # 逐行讲解：调用 ax.set_yticks（ax.set_yticks(np.arange(0, 101, 20))）。
    ax.set_yticks(np.arange(0, 101, 20))
    # 逐行讲解：调用 ax.margins（ax.margins(x=0.06)）。
    ax.margins(x=0.06)
    # 逐行讲解：调用 style_axes（style_axes(ax)）。
    style_axes(ax)


# 逐行讲解：定义函数 metric_legend_handles：Return consistent metric handles for both figures.
def metric_legend_handles() -> list[Patch]:
    """Return consistent metric handles for both figures."""

    # 逐行讲解：返回结果：[Patch(facecolor=MEDICAL_COLOR, edgecolor=INK_COLOR, linewidth=1.1, label='MedQA-zh 1k'), Patch(facecolor=G…。
    return [
        Patch(
            facecolor=MEDICAL_COLOR,
            edgecolor=INK_COLOR,
            linewidth=1.1,
            label="MedQA-zh 1k",
        ),
        Patch(
            facecolor=GENERAL_COLOR,
            edgecolor=INK_COLOR,
            linewidth=1.1,
            hatch="///",
            label="C-Eval non-med 8k",
        ),
    ]


# 逐行讲解：定义函数 export_figure：Export a raster preview and a vector publication artifact.
def export_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    """Export a raster preview and a vector publication artifact."""

    # 逐行讲解：创建目录（output_dir.mkdir(parents=True, exist_ok=True)）。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：计算 output_dir / f'{stem}.png'，赋给 png_path。
    png_path = output_dir / f"{stem}.png"
    # 逐行讲解：计算 output_dir / f'{stem}.pdf'，赋给 pdf_path。
    pdf_path = output_dir / f"{stem}.pdf"
    # 逐行讲解：调用 fig.savefig（fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')）。
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    # 逐行讲解：调用 fig.savefig（fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')）。
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    # 逐行讲解：调用 print（print(f'Saved: {png_path}')）。
    print(f"Saved: {png_path}")
    # 逐行讲解：调用 print（print(f'Saved: {pdf_path}')）。
    print(f"Saved: {pdf_path}")


# 逐行讲解：定义函数 plot_sar_opd：Plot SAR-OPD stage comparison and base-anchor recovery trajectory.
def plot_sar_opd(output_dir: Path, dpi: int) -> None:
    """Plot SAR-OPD stage comparison and base-anchor recovery trajectory."""

    # 逐行讲解：构造列表 ['4B Base', 'Medical SFT\n(epoch 3)', 'Medical OPD\n(step 300)', 'SAR-OPD\n(step 300)']，赋给 stage_labels。
    stage_labels = [
        "4B Base",
        "Medical SFT\n(epoch 3)",
        "Medical OPD\n(step 300)",
        "SAR-OPD\n(step 300)",
    ]
    # 逐行讲解：构造列表 [72.17, 79.0, 80.0, 84.33]，赋给 stage_medqa。
    stage_medqa = [72.17, 79.00, 80.00, 84.33]
    # 逐行讲解：构造列表 [81.67, 69.33, 73.0, 84.67]，赋给 stage_ceval。
    stage_ceval = [81.67, 69.33, 73.00, 84.67]

    # 逐行讲解：构造列表 [0, 50, 100, 150, 200, 250, 300]，赋给 recovery_steps。
    recovery_steps = [0, 50, 100, 150, 200, 250, 300]
    # 逐行讲解：构造列表 [80.0, 84.17, 82.5, 83.0, 82.17, 83.33, 84.33]，赋给 recovery_medqa。
    recovery_medqa = [80.00, 84.17, 82.50, 83.00, 82.17, 83.33, 84.33]
    # 逐行讲解：构造列表 [73.0, 76.67, 79.33, 80.33, 83.67, 82.67, 84.67]，赋给 recovery_ceval。
    recovery_ceval = [73.00, 76.67, 79.33, 80.33, 83.67, 82.67, 84.67]

    # 逐行讲解：调用 plt.subplots（plt.subplots(1, 2, figsize=(14.5, 6.35))），赋给 (fig, axes)。
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.35))
    # 逐行讲解：调用 fig.suptitle（fig.suptitle('SAR-OPD: Staged Anchor-Restoration OPD', fontsize=17, fontweight='bold', y=0.985)）。
    fig.suptitle(
        "SAR-OPD: Staged Anchor-Restoration OPD",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    # 逐行讲解：调用 fig.text（fig.text(0.5, 0.92, 'Accuracy on MedQA-zh (600 questions) and C-Eval non-med (300 questions)', ha='center',…）。
    fig.text(
        0.5,
        0.920,
        "Accuracy on MedQA-zh (600 questions) and C-Eval non-med (300 questions)",
        ha="center",
        fontsize=10.5,
        color="#555B63",
    )

    # 逐行讲解：取值 axes，赋给 (left, right)。
    left, right = axes
    # 逐行讲解：调用 draw_grouped_bars（draw_grouped_bars(left, stage_labels, stage_medqa, stage_ceval, highlighted_indices={3})）。
    draw_grouped_bars(
        left,
        stage_labels,
        stage_medqa,
        stage_ceval,
        highlighted_indices={3},
    )
    # 逐行讲解：调用 left.set_title（left.set_title('(a) Stage-wise accuracy')）。
    left.set_title("(a) Stage-wise accuracy")

    # 逐行讲解：调用 right.plot（right.plot(recovery_steps, recovery_medqa, color=MEDICAL_COLOR, marker='o', markersize=6.5, markeredgecolor…）。
    right.plot(
        recovery_steps,
        recovery_medqa,
        color=MEDICAL_COLOR,
        marker="o",
        markersize=6.5,
        markeredgecolor=INK_COLOR,
        linewidth=2.2,
        label="MedQA-zh 1k",
        zorder=4,
    )
    # 逐行讲解：调用 right.plot（right.plot(recovery_steps, recovery_ceval, color=GENERAL_COLOR, marker='s', markersize=6.2, markerfacecolor…）。
    right.plot(
        recovery_steps,
        recovery_ceval,
        color=GENERAL_COLOR,
        marker="s",
        markersize=6.2,
        markerfacecolor="white",
        markeredgecolor=GENERAL_COLOR,
        markeredgewidth=1.7,
        linewidth=2.2,
        linestyle="--",
        label="C-Eval non-med 8k",
        zorder=4,
    )
    # 逐行讲解：调用 right.axhline（right.axhline(BASE_MEDQA, color=MEDICAL_COLOR, linestyle=':', linewidth=1.5, alpha=0.75, label='4B Base: Me…）。
    right.axhline(
        BASE_MEDQA,
        color=MEDICAL_COLOR,
        linestyle=":",
        linewidth=1.5,
        alpha=0.75,
        label="4B Base: MedQA",
        zorder=2,
    )
    # 逐行讲解：调用 right.axhline（right.axhline(BASE_CEVAL, color=GENERAL_COLOR, linestyle=':', linewidth=1.5, alpha=0.75, label='4B Base: C-…）。
    right.axhline(
        BASE_CEVAL,
        color=GENERAL_COLOR,
        linestyle=":",
        linewidth=1.5,
        alpha=0.75,
        label="4B Base: C-Eval",
        zorder=2,
    )
    # 逐行讲解：构造字典 {200: (0, -16)}，赋给 medqa_label_offsets。
    medqa_label_offsets = {200: (0, -16)}
    # 逐行讲解：构造字典 {200: (0, 10)}，赋给 ceval_label_offsets。
    ceval_label_offsets = {200: (0, 10)}
    # 逐行讲解：遍历 zip(recovery_steps, recovery_medqa, recovery_ceval, strict=True)，将每项绑定到 (step, med_value, ceval_value)。
    for step, med_value, ceval_value in zip(
        recovery_steps,
        recovery_medqa,
        recovery_ceval,
        strict=True,
    ):
        # 逐行讲解：调用 right.annotate（right.annotate(f'{med_value:.2f}', (step, med_value), xytext=medqa_label_offsets.get(step, (0, 9)), textcoo…）。
        right.annotate(
            f"{med_value:.2f}",
            (step, med_value),
            xytext=medqa_label_offsets.get(step, (0, 9)),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
            color=MEDICAL_COLOR,
        )
        # 逐行讲解：调用 right.annotate（right.annotate(f'{ceval_value:.2f}', (step, ceval_value), xytext=ceval_label_offsets.get(step, (0, -14)), t…）。
        right.annotate(
            f"{ceval_value:.2f}",
            (step, ceval_value),
            xytext=ceval_label_offsets.get(step, (0, -14)),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
            color=GENERAL_COLOR,
        )
    # 逐行讲解：调用 right.scatter（right.scatter([300, 300], [recovery_medqa[-1], recovery_ceval[-1]], s=115, facecolors='none', edgecolors=IN…）。
    right.scatter(
        [300, 300],
        [recovery_medqa[-1], recovery_ceval[-1]],
        s=115,
        facecolors="none",
        edgecolors=INK_COLOR,
        linewidths=1.8,
        zorder=5,
    )
    # 逐行讲解：调用 right.set_title（right.set_title('(b) Base-anchor recovery trajectory (focused scale)')）。
    right.set_title("(b) Base-anchor recovery trajectory (focused scale)")
    # 逐行讲解：调用 right.set_xlabel（right.set_xlabel('Base-anchor C-Eval OPD step')）。
    right.set_xlabel("Base-anchor C-Eval OPD step")
    # 逐行讲解：调用 right.set_ylabel（right.set_ylabel('Accuracy (%)')）。
    right.set_ylabel("Accuracy (%)")
    # 逐行讲解：调用 right.set_xticks（right.set_xticks(recovery_steps)）。
    right.set_xticks(recovery_steps)
    # 逐行讲解：调用 right.set_ylim（right.set_ylim(68, 87)）。
    right.set_ylim(68, 87)
    # 逐行讲解：调用 right.set_yticks（right.set_yticks(np.arange(70, 88, 2.5))）。
    right.set_yticks(np.arange(70, 88, 2.5))
    # 逐行讲解：调用 right.margins（right.margins(x=0.04)）。
    right.margins(x=0.04)
    # 逐行讲解：调用 style_axes（style_axes(right)）。
    style_axes(right)

    # 逐行讲解：调用 fig.legend（fig.legend(handles=[Line2D([0], [0], color=MEDICAL_COLOR, marker='o', markeredgecolor=INK_COLOR, linewidth=…）。
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=MEDICAL_COLOR,
                marker="o",
                markeredgecolor=INK_COLOR,
                linewidth=2.2,
                label="MedQA-zh 1k",
            ),
            Line2D(
                [0],
                [0],
                color=GENERAL_COLOR,
                marker="s",
                markerfacecolor="white",
                linestyle="--",
                linewidth=2.2,
                label="C-Eval non-med 8k",
            ),
            Line2D(
                [0],
                [0],
                color=INK_COLOR,
                marker="o",
                markerfacecolor="none",
                linewidth=0,
                markersize=9,
                label="Selected checkpoint",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.870),
        ncol=3,
        frameon=True,
        fancybox=True,
        edgecolor=INK_COLOR,
        framealpha=0.96,
    )
    # 逐行讲解：调用 fig.subplots_adjust（fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.72, wspace=0.25)）。
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.72, wspace=0.25)
    # 逐行讲解：调用 export_figure（export_figure(fig, output_dir, 'sar_opd_analysis', dpi)）。
    export_figure(fig, output_dir, "sar_opd_analysis", dpi)
    # 逐行讲解：调用 plt.close（plt.close(fig)）。
    plt.close(fig)


# 逐行讲解：定义函数 plot_idt_opd：Plot IDT-OPD key checkpoints and the medical/general Pareto frontier.
def plot_idt_opd(output_dir: Path, dpi: int) -> None:
    """Plot IDT-OPD key checkpoints and the medical/general Pareto frontier."""

    # 逐行讲解：构造列表 ['4B Base', 'Medical SFT\nTeacher', 'IDT-OPD\nstep 100', 'IDT-OPD\nstep 200', 'IDT-OPD\nstep 600']，赋给 comparison_labels。
    comparison_labels = [
        "4B Base",
        "Medical SFT\nTeacher",
        "IDT-OPD\nstep 100",
        "IDT-OPD\nstep 200",
        "IDT-OPD\nstep 600",
    ]
    # 逐行讲解：构造列表 [72.17, 79.0, 79.17, 76.5, 75.0]，赋给 comparison_medqa。
    comparison_medqa = [72.17, 79.00, 79.17, 76.50, 75.00]
    # 逐行讲解：构造列表 [81.67, 69.33, 81.67, 86.0, 82.0]，赋给 comparison_ceval。
    comparison_ceval = [81.67, 69.33, 81.67, 86.00, 82.00]

    # 逐行讲解：构造列表 [100, 200, 300, 400, 500, 600]，赋给 checkpoint_steps。
    checkpoint_steps = [100, 200, 300, 400, 500, 600]
    # 逐行讲解：构造列表 [79.17, 76.5, 76.33, 75.17, 75.33, 75.0]，赋给 checkpoint_medqa。
    checkpoint_medqa = [79.17, 76.50, 76.33, 75.17, 75.33, 75.00]
    # 逐行讲解：构造列表 [81.67, 86.0, 84.33, 81.33, 83.67, 82.0]，赋给 checkpoint_ceval。
    checkpoint_ceval = [81.67, 86.00, 84.33, 81.33, 83.67, 82.00]

    # 逐行讲解：调用 plt.subplots（plt.subplots(1, 2, figsize=(14.5, 6.35))），赋给 (fig, axes)。
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.35))
    # 逐行讲解：调用 fig.suptitle（fig.suptitle('IDT-OPD: Interleaved Dual-Teacher OPD', fontsize=17, fontweight='bold', y=0.985)）。
    fig.suptitle(
        "IDT-OPD: Interleaved Dual-Teacher OPD",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    # 逐行讲解：调用 fig.text（fig.text(0.5, 0.92, '1:1 Medical/C-Eval scheduling; accuracy on fixed 600- and 300-question test sets', ha=…）。
    fig.text(
        0.5,
        0.920,
        "1:1 Medical/C-Eval scheduling; accuracy on fixed 600- and 300-question test sets",
        ha="center",
        fontsize=10.5,
        color="#555B63",
    )

    # 逐行讲解：取值 axes，赋给 (left, right)。
    left, right = axes
    # 逐行讲解：调用 draw_grouped_bars（draw_grouped_bars(left, comparison_labels, comparison_medqa, comparison_ceval, highlighted_indices={2, 3})）。
    draw_grouped_bars(
        left,
        comparison_labels,
        comparison_medqa,
        comparison_ceval,
        highlighted_indices={2, 3},
    )
    # 逐行讲解：调用 left.set_title（left.set_title('(a) Key checkpoint comparison')）。
    left.set_title("(a) Key checkpoint comparison")

    # 逐行讲解：调用 right.plot（right.plot(checkpoint_ceval, checkpoint_medqa, color=NEUTRAL_COLOR, linewidth=1.4, linestyle='--', alpha=0.…）。
    right.plot(
        checkpoint_ceval,
        checkpoint_medqa,
        color=NEUTRAL_COLOR,
        linewidth=1.4,
        linestyle="--",
        alpha=0.85,
        zorder=2,
    )
    # 逐行讲解：调用 right.scatter（right.scatter(checkpoint_ceval, checkpoint_medqa, s=72, color=GENERAL_COLOR, edgecolors=INK_COLOR, linewidt…）。
    right.scatter(
        checkpoint_ceval,
        checkpoint_medqa,
        s=72,
        color=GENERAL_COLOR,
        edgecolors=INK_COLOR,
        linewidths=1.0,
        zorder=4,
    )
    # 逐行讲解：调用 right.scatter（right.scatter([BASE_CEVAL, 69.33], [BASE_MEDQA, 79.0], s=82, facecolors='white', edgecolors=NEUTRAL_COLOR, …）。
    right.scatter(
        [BASE_CEVAL, 69.33],
        [BASE_MEDQA, 79.00],
        s=82,
        facecolors="white",
        edgecolors=NEUTRAL_COLOR,
        linewidths=1.8,
        marker="D",
        zorder=4,
    )
    # 逐行讲解：构造列表 [checkpoint_ceval[0], checkpoint_ceval[1]]，赋给 pareto_x。
    pareto_x = [checkpoint_ceval[0], checkpoint_ceval[1]]
    # 逐行讲解：构造列表 [checkpoint_medqa[0], checkpoint_medqa[1]]，赋给 pareto_y。
    pareto_y = [checkpoint_medqa[0], checkpoint_medqa[1]]
    # 逐行讲解：调用 right.plot（right.plot(pareto_x, pareto_y, color=MEDICAL_COLOR, linewidth=2.2, linestyle=':', zorder=3)）。
    right.plot(
        pareto_x,
        pareto_y,
        color=MEDICAL_COLOR,
        linewidth=2.2,
        linestyle=":",
        zorder=3,
    )
    # 逐行讲解：调用 right.scatter（right.scatter(pareto_x, pareto_y, s=150, facecolors='none', edgecolors=MEDICAL_COLOR, linewidths=2.4, zorde…）。
    right.scatter(
        pareto_x,
        pareto_y,
        s=150,
        facecolors="none",
        edgecolors=MEDICAL_COLOR,
        linewidths=2.4,
        zorder=5,
    )

    # 逐行讲解：构造列表 [(BASE_CEVAL, BASE_MEDQA, '4B Base', (-8, -16)), (69.33, 79.0, 'Medical SFT', (7, 8))]，赋给 reference_points。
    reference_points = [
        (BASE_CEVAL, BASE_MEDQA, "4B Base", (-8, -16)),
        (69.33, 79.00, "Medical SFT", (7, 8)),
    ]
    # 逐行讲解：遍历 reference_points，将每项绑定到 (x_value, y_value, label, offset)。
    for x_value, y_value, label, offset in reference_points:
        # 逐行讲解：调用 right.annotate（right.annotate(label, (x_value, y_value), xytext=offset, textcoords='offset points', fontsize=8.5, color='#…）。
        right.annotate(
            label,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color="#5F666E",
        )

    # 逐行讲解：构造字典 {100: (-18, 10), 200: (-18, 9), 300: (-8, -16), 400: (-19, -16), 500: (7, -2), 600: (-2, 9)}，赋给 offsets。
    offsets = {
        100: (-18, 10),
        200: (-18, 9),
        300: (-8, -16),
        400: (-19, -16),
        500: (7, -2),
        600: (-2, 9),
    }
    # 逐行讲解：遍历 zip(checkpoint_steps, checkpoint_ceval, checkpoint_medqa, strict=True)，将每项绑定到 (step, x_value, y_value)。
    for step, x_value, y_value in zip(
        checkpoint_steps,
        checkpoint_ceval,
        checkpoint_medqa,
        strict=True,
    ):
        # 逐行讲解：调用 right.annotate（right.annotate(f'step {step}', (x_value, y_value), xytext=offsets[step], textcoords='offset points', fontsi…）。
        right.annotate(
            f"step {step}",
            (x_value, y_value),
            xytext=offsets[step],
            textcoords="offset points",
            fontsize=8.3,
            fontweight="bold" if step in {100, 200} else "normal",
            color=INK_COLOR,
        )

    # 逐行讲解：调用 right.set_title（right.set_title('(b) Medical-general Pareto frontier (focused scale)')）。
    right.set_title("(b) Medical-general Pareto frontier (focused scale)")
    # 逐行讲解：调用 right.set_xlabel（right.set_xlabel('C-Eval non-med 8k accuracy (%)  →')）。
    right.set_xlabel("C-Eval non-med 8k accuracy (%)  →")
    # 逐行讲解：调用 right.set_ylabel（right.set_ylabel('MedQA-zh 1k accuracy (%)  →')）。
    right.set_ylabel("MedQA-zh 1k accuracy (%)  →")
    # 逐行讲解：调用 right.set_xlim（right.set_xlim(68, 87.2)）。
    right.set_xlim(68, 87.2)
    # 逐行讲解：调用 right.set_ylim（right.set_ylim(71, 80.4)）。
    right.set_ylim(71, 80.4)
    # 逐行讲解：调用 right.set_xticks（right.set_xticks(np.arange(70, 88, 2.5))）。
    right.set_xticks(np.arange(70, 88, 2.5))
    # 逐行讲解：调用 right.set_yticks（right.set_yticks(np.arange(72, 81, 1.5))）。
    right.set_yticks(np.arange(72, 81, 1.5))
    # 逐行讲解：调用 style_axes（style_axes(right, grid_axis='both')）。
    style_axes(right, grid_axis="both")
    # 逐行讲解：调用 right.legend（right.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor=GENERAL_COLOR, markeredgec…）。
    right.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=GENERAL_COLOR,
                markeredgecolor=INK_COLOR,
                markersize=7,
                label="IDT checkpoints",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="white",
                markeredgecolor=NEUTRAL_COLOR,
                markersize=7,
                label="References",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=MEDICAL_COLOR,
                markerfacecolor="none",
                markeredgecolor=MEDICAL_COLOR,
                linestyle=":",
                markersize=9,
                label="Best trade-offs (Pareto)",
            ),
        ],
        loc="lower left",
        frameon=True,
        fancybox=True,
        edgecolor=INK_COLOR,
        framealpha=0.95,
        fontsize=8.5,
    )

    # 逐行讲解：调用 fig.legend（fig.legend(handles=metric_legend_handles(), loc='upper center', bbox_to_anchor=(0.5, 0.87), ncol=2, frameon…）。
    fig.legend(
        handles=metric_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.870),
        ncol=2,
        frameon=True,
        fancybox=True,
        edgecolor=INK_COLOR,
        framealpha=0.96,
    )
    # 逐行讲解：调用 fig.subplots_adjust（fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.72, wspace=0.25)）。
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.72, wspace=0.25)
    # 逐行讲解：调用 export_figure（export_figure(fig, output_dir, 'idt_opd_analysis', dpi)）。
    export_figure(fig, output_dir, "idt_opd_analysis", dpi)
    # 逐行讲解：调用 plt.close（plt.close(fig)）。
    plt.close(fig)


# 逐行讲解：定义函数 main：Generate exactly two figures, one for each training scheme.
def main() -> None:
    """Generate exactly two figures, one for each training scheme."""

    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：调用 configure_style（configure_style()）。
    configure_style()
    # 逐行讲解：调用 plot_sar_opd（plot_sar_opd(args.output_dir, args.dpi)）。
    plot_sar_opd(args.output_dir, args.dpi)
    # 逐行讲解：调用 plot_idt_opd（plot_idt_opd(args.output_dir, args.dpi)）。
    plot_idt_opd(args.output_dir, args.dpi)


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
