"""绘制 Qwen3.5-4B / 9B 的五组开发集结果。

在 experiments/spec-o3/ 目录下运行：

uv run python analysis.py

数据固定为 readme.md 的结果表，无需加载模型或重新评测。
四个面板分别展示 Accuracy、Macro F1、严格格式合规率与答对题数。
输出 images/results_comparison.png（300 DPI）和同名矢量 PDF。
"""

# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：导入模块：matplotlib。
import matplotlib

# 逐行讲解：调用 matplotlib.use（matplotlib.use('Agg')）。
matplotlib.use("Agg")
# 逐行讲解：导入模块：matplotlib.pyplot，别名 plt。
import matplotlib.pyplot as plt
# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：从 matplotlib.ticker 导入 PercentFormatter。
from matplotlib.ticker import PercentFormatter


# 固定结果快照：Base/SFT 使用当前答案提取规则重算，三组 RL 使用原 metrics.json。
# 来源依次为 outputs/{base-dev,sft-dev}/metrics-reparsed.json，以及
# outputs/{sft-rl,rl-4b-e3,rl-9b-e3}/metrics.json；百分比与 readme.md 一致。
MODELS = ["4B\nBase", "4B\nSFT epoch 2", "4B\nRL epoch 1", "4B\nRL epoch 3", "9B\nRL epoch 3"]
# 逐行讲解：构造列表 ['Macro F1', 'Accuracy', 'Format']，赋给 METRICS。
METRICS = ["Macro F1", "Accuracy", "Format"]
# 逐行讲解：调用 np.array（np.array([[53.99, 38.67, 0.39], [63.05, 53.91, 85.94], [71.68, 67.97, 96.09], [70.93, 68.36, 98.05], [75.94…），赋给 SCORES。
SCORES = np.array([
    [53.99, 38.67, 0.39],
    [63.05, 53.91, 85.94],
    [71.68, 67.97, 96.09],
    [70.93, 68.36, 98.05],
    [75.94, 74.61, 99.61],
])
# 逐行讲解：构造列表 [99, 138, 174, 175, 191]，赋给 CORRECT_ANSWERS。
CORRECT_ANSWERS = [99, 138, 174, 175, 191]
# 逐行讲解：取值 256，赋给 TOTAL_QUESTIONS。
TOTAL_QUESTIONS = 256
# 逐行讲解：构造列表 ['#D94A4A', '#3388B8']，赋给 BAR_COLORS。
BAR_COLORS = ["#D94A4A", "#3388B8"]
# 逐行讲解：取值 '#F28E2B'，赋给 FORMAT_COLOR。
FORMAT_COLOR = "#F28E2B"
# 逐行讲解：取值 '#596A7D'，赋给 COUNT_COLOR。
COUNT_COLOR = "#596A7D"
# 逐行讲解：计算 Path(__file__).parent / 'images'，赋给 OUTPUT_DIR。
OUTPUT_DIR = Path(__file__).parent / "images"


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main():
    # 逐行讲解：调用 plt.rcParams.update（plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'f…）。
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.labelcolor": "#202020",
        "text.color": "#202020",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "#202020",
        "axes.linewidth": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    # 逐行讲解：调用 plt.subplots（plt.subplots(2, 2, figsize=(13.4, 8.6))），赋给 (fig, axes)。
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6))
    # 逐行讲解：调用 fig.subplots_adjust（fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.81, hspace=0.55, wspace=0.18)）。
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.81, hspace=0.55, wspace=0.18)
    # 逐行讲解：调用 np.arange（np.arange(len(MODELS))），赋给 positions。
    positions = np.arange(len(MODELS))
    # 逐行讲解：构造列表 [('(a) Accuracy', SCORES[:, 1], BAR_COLORS[1]), ('(b) Macro F1', SCORES[:, 0], BAR_COLORS[0]), ('(c) Strict…，赋给 panels。
    panels = [
        ("(a) Accuracy", SCORES[:, 1], BAR_COLORS[1]),
        ("(b) Macro F1", SCORES[:, 0], BAR_COLORS[0]),
        ("(c) Strict Format Compliance", SCORES[:, 2], FORMAT_COLOR),
        ("(d) Correct Answers", np.asarray(CORRECT_ANSWERS), COUNT_COLOR),
    ]

    # checkpoint 为离散类别；各面板使用独立柱形，不把 4B 与 9B 连成训练曲线。
    for panel_index, (ax, (title, values, color)) in enumerate(zip(axes.flat, panels)):
        # 逐行讲解：取值 panel_index == 3，赋给 is_count。
        is_count = panel_index == 3
        # 逐行讲解：调用 ax.set_axisbelow（ax.set_axisbelow(True)）。
        ax.set_axisbelow(True)
        # 逐行讲解：调用 ax.grid（ax.grid(axis='y', color='#C7C7C7', linestyle='-.', linewidth=0.8, alpha=0.75)）。
        ax.grid(axis="y", color="#C7C7C7", linestyle="-.", linewidth=0.8, alpha=0.75)
        # 逐行讲解：调用 ax.bar（ax.bar(positions, values, width=0.62, color=color, edgecolor='#202020', linewidth=0.9, zorder=3)），赋给 bars。
        bars = ax.bar(positions, values, width=0.62, color=color, edgecolor="#202020", linewidth=0.9, zorder=3)
        # 逐行讲解：通过推导式逐项生成结果：[f'{int(value)} / {TOTAL_QUESTIONS}' if is_count else f'{value:.2f}%' for value in values]，赋给 labels。
        labels = [f"{int(value)} / {TOTAL_QUESTIONS}" if is_count else f"{value:.2f}%" for value in values]
        # 逐行讲解：调用 ax.bar_label（ax.bar_label(bars, labels=labels, padding=4, fontsize=10, fontweight='bold')）。
        ax.bar_label(bars, labels=labels, padding=4, fontsize=10, fontweight="bold")
        # 逐行讲解：调用 ax.set_xticks（ax.set_xticks(positions, MODELS)）。
        ax.set_xticks(positions, MODELS)
        # 逐行讲解：调用 ax.set_xlim（ax.set_xlim(-0.6, len(MODELS) - 0.4)）。
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        # 逐行讲解：调用 ax.tick_params（ax.tick_params(axis='x', length=0, pad=7)）。
        ax.tick_params(axis="x", length=0, pad=7)
        # 逐行讲解：调用 ax.set_title（ax.set_title(title, pad=12)）。
        ax.set_title(title, pad=12)
        # 逐行讲解：检查条件 is_count；成立时执行下方分支。
        if is_count:
            # 逐行讲解：调用 ax.set_ylim（ax.set_ylim(0, 280)）。
            ax.set_ylim(0, 280)
            # 逐行讲解：调用 ax.set_yticks（ax.set_yticks([0, 64, 128, 192, 256])）。
            ax.set_yticks([0, 64, 128, 192, 256])
            # 逐行讲解：调用 ax.set_ylabel（ax.set_ylabel('Correct answers (out of 256)')）。
            ax.set_ylabel("Correct answers (out of 256)")
        else:
            # 逐行讲解：调用 ax.set_ylim（ax.set_ylim(0, 112)）。
            ax.set_ylim(0, 112)
            # 逐行讲解：调用 ax.set_yticks（ax.set_yticks(np.arange(0, 101, 20))）。
            ax.set_yticks(np.arange(0, 101, 20))
            # 逐行讲解：调用 ax.yaxis.set_major_formatter（ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))）。
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            # 逐行讲解：调用 ax.set_ylabel（ax.set_ylabel('Score')）。
            ax.set_ylabel("Score")

    # 逐行讲解：调用 fig.suptitle（fig.suptitle('Spec-o3 · Qwen3.5-4B / 9B Evaluation', fontsize=18, fontweight='bold', y=0.98)）。
    fig.suptitle("Spec-o3 · Qwen3.5-4B / 9B Evaluation", fontsize=18, fontweight="bold", y=0.98)
    # 逐行讲解：调用 fig.text（fig.text(0.5, 0.934, 'Same 256 dev examples · 4B Base / SFT epoch 2 / RL epochs 1 & 3 · 9B RL epoch 3', ha=…）。
    fig.text(
        0.5, 0.934,
        "Same 256 dev examples · 4B Base / SFT epoch 2 / RL epochs 1 & 3 · 9B RL epoch 3",
        ha="center", va="center", fontsize=11, color="#555555",
    )
    # 逐行讲解：计算 CORRECT_ANSWERS[-1] - CORRECT_ANSWERS[-2]，赋给 correct_gain。
    correct_gain = CORRECT_ANSWERS[-1] - CORRECT_ANSWERS[-2]
    # 逐行讲解：计算 SCORES[-1, 1] - SCORES[-2, 1]，赋给 accuracy_gain。
    accuracy_gain = SCORES[-1, 1] - SCORES[-2, 1]
    # 逐行讲解：计算 SCORES[-1, 0] - SCORES[-2, 0]，赋给 f1_gain。
    f1_gain = SCORES[-1, 0] - SCORES[-2, 0]
    # 逐行讲解：调用 fig.text（fig.text(0.5, 0.89, f'9B vs 4B at RL epoch 3: +{correct_gain} correct · +{accuracy_gain:.2f} pp accuracy · …）。
    fig.text(
        0.5, 0.89,
        f"9B vs 4B at RL epoch 3: +{correct_gain} correct · +{accuracy_gain:.2f} pp accuracy · +{f1_gain:.2f} pp Macro F1",
        ha="center", va="center", fontsize=11, color="#303030",
    )
    # 逐行讲解：调用 fig.text（fig.text(0.5, 0.037, 'Generation caps: 4B Base 6,132; 4B SFT 2,048; all RL checkpoints 6,144 tokens per tur…）。
    fig.text(
        0.5, 0.037,
        "Generation caps: 4B Base 6,132; 4B SFT 2,048; all RL checkpoints 6,144 tokens per turn.\n"
        "Base alone uses an extra format reminder. All runs use the same answer extraction and a 16,384-token context cap.",
        ha="center", va="center", fontsize=9, color="#666666", style="italic", linespacing=1.6,
    )

    # 逐行讲解：创建目录（OUTPUT_DIR.mkdir(parents=True, exist_ok=True)）。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：遍历 ('png', 'pdf')，将每项绑定到 extension。
    for extension in ("png", "pdf"):
        # 逐行讲解：计算 OUTPUT_DIR / f'results_comparison.{extension}'，赋给 path。
        path = OUTPUT_DIR / f"results_comparison.{extension}"
        # 逐行讲解：调用 fig.savefig（fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.16)）。
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.16)
        # 逐行讲解：调用 print（print(f'Saved: {path}')）。
        print(f"Saved: {path}")
    # 逐行讲解：调用 plt.close（plt.close(fig)）。
    plt.close(fig)


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
