"""从原始光谱重绘局部波段，作为下一轮的图片观察。"""

# 逐行讲解：导入模块：matplotlib。
import matplotlib

# 逐行讲解：调用 matplotlib.use（matplotlib.use('Agg')）。
matplotlib.use("Agg")
# 逐行讲解：导入模块：matplotlib.pyplot，别名 plt。
import matplotlib.pyplot as plt
# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np


# 逐行讲解：定义函数 spectral_visualization_tool：在静止系波长上选取窗口，返回绘图结果及可直接回填的工具消息。
def spectral_visualization_tool(
    wavelength,
    flux,
    redshift,
    wavelength_range,
    label,
    output_path,
):
    """在静止系波长上选取窗口，返回绘图结果及可直接回填的工具消息。"""
    # 1. 校正红移，再按模型指定的波长区间选取数据。
    wavelength = np.asarray(wavelength) / (1 + redshift)
    # 逐行讲解：构造指定类型的 NumPy 数组（np.asarray(flux)），赋给 flux。
    flux = np.asarray(flux)
    # 逐行讲解：取值 wavelength_range，赋给 (low, high)。
    low, high = wavelength_range
    # 逐行讲解：计算 (wavelength >= low) & (wavelength <= high)，赋给 selected。
    selected = (wavelength >= low) & (wavelength <= high)

    # 空窗口是工具动作的一种结果，返回文字让模型继续判断。
    if selected.sum() < 2:
        # 逐行讲解：返回结果：{'content': [{'type': 'text', 'text': f'No spectrum data in wavelength range ({low:g}, {high:g}) Å.'}], 'im…。
        return {
            "content": [{
                "type": "text",
                "text": f"No spectrum data in wavelength range ({low:g}, {high:g}) Å.",
            }],
            "images": [],
            "ok": False,
        }

    # 2. 根据窗口宽度选择画布，绘制局部光谱。
    wave, values = wavelength[selected], flux[selected]
    # 逐行讲解：计算 high - low，赋给 width。
    width = high - low
    # 逐行讲解：检查条件 width < 50；成立时执行下方分支。
    if width < 50:
        # 逐行讲解：构造元组 (6, 4)，赋给 figsize。
        figsize = (6, 4)
    # 逐行讲解：检查条件 width < 200；成立时执行下方分支。
    elif width < 200:
        # 逐行讲解：构造元组 (7, 4.5)，赋给 figsize。
        figsize = (7, 4.5)
    # 逐行讲解：检查条件 width < 1000；成立时执行下方分支。
    elif width < 1000:
        # 逐行讲解：构造元组 (8, 5)，赋给 figsize。
        figsize = (8, 5)
    else:
        # 逐行讲解：构造元组 (9, 5.5)，赋给 figsize。
        figsize = (9, 5.5)

    # 逐行讲解：调用 plt.subplots（plt.subplots(figsize=figsize, dpi=80)），赋给 (fig, ax)。
    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    # 逐行讲解：调用 ax.plot（ax.plot(wave, values, color='blue', linewidth=1, alpha=0.8)）。
    ax.plot(wave, values, color="blue", linewidth=1, alpha=0.8)
    # 逐行讲解：调用 ax.set_xlabel（ax.set_xlabel('Wavelength (Å)', fontsize=12, fontweight='bold')）。
    ax.set_xlabel("Wavelength (Å)", fontsize=12, fontweight="bold")
    # 逐行讲解：调用 ax.set_ylabel（ax.set_ylabel('Flux', fontsize=12, fontweight='bold')）。
    ax.set_ylabel("Flux", fontsize=12, fontweight="bold")
    # 逐行讲解：调用 ax.set_title（ax.set_title('Spectrum', fontsize=14, fontweight='bold', pad=20)）。
    ax.set_title("Spectrum", fontsize=14, fontweight="bold", pad=20)
    # 逐行讲解：调用 ax.grid（ax.grid(True, alpha=0.3)）。
    ax.grid(True, alpha=0.3)
    # 逐行讲解：调用 ax.minorticks_on（ax.minorticks_on()）。
    ax.minorticks_on()

    # 3. 标注窗口内的数据范围，保存供模型继续观察的 PNG。
    ax.text(
        0.98,
        0.98,
        f"Points: {len(wave):,}\n"
        f"λ: {wave.min():.1f}–{wave.max():.1f} Å\n"
        f"Flux: {values.min():.2f}–{values.max():.2f}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
        },
    )
    # 逐行讲解：调用 fig.tight_layout（fig.tight_layout()）。
    fig.tight_layout()
    # 逐行讲解：调用 fig.savefig（fig.savefig(output_path, format='png', dpi=80, bbox_inches='tight')）。
    fig.savefig(output_path, format="png", dpi=80, bbox_inches="tight")
    # 逐行讲解：调用 plt.close（plt.close(fig)）。
    plt.close(fig)

    # 4. 图片与文字说明一起作为下一轮的 tool observation。
    text = f"Generated spectral visualization for wavelength range ({low:g}, {high:g}) Å"
    # 逐行讲解：检查条件 label；成立时执行下方分支。
    if label:
        # 逐行讲解：更新 text：text += f' ({label})'。
        text += f" ({label})"
    # 逐行讲解：返回结果：{'content': [{'type': 'image', 'image': str(output_path)}, {'type': 'text', 'text': text}], 'images': [str(…。
    return {
        "content": [
            {"type": "image", "image": str(output_path)},
            {"type": "text", "text": text},
        ],
        "images": [str(output_path)],
        "ok": True,
    }
