"""汇总评测结果，与参考基线对照。

用法：
uv run python experiments/pytrio-jev/analysis.py [outputs/eval-xxx.json ...]
不传参数时汇总 outputs/ 下全部 eval-*.json。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：sys。
import sys
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'outputs'，赋给 OUTPUT_DIR。
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# 参考基线准确率（百分数）。
REFERENCE_4B = {
    "jevbench_easy": 100.00,
    "jevbench_original": 98.61,
    "jevbench_hard": 73.87,
    "typed_decisions": 80.55,
    "toolace": 96.45,
}


# 逐行讲解：定义函数 render：封装下方逻辑，供调用方复用。
def render(path: Path) -> list[str]:
    # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(path.read_text())），赋给 report。
    report = json.loads(path.read_text())
    # 逐行讲解：取值 path.stem，赋给 label。
    label = path.stem
    # 逐行讲解：构造列表 [f"\n### {label}（T={report['temperature']:.4f}）", '']，赋给 lines。
    lines = [f"\n### {label}（T={report['temperature']:.4f}）", ""]
    # 逐行讲解：向列表末尾追加一个元素（lines.append('| suite | decisions | acc | reference | ece | brier |')）。
    lines.append("| suite | decisions | acc | reference | ece | brier |")
    # 逐行讲解：向列表末尾追加一个元素（lines.append('|---|---|---|---|---|---|')）。
    lines.append("|---|---|---|---|---|---|")
    # 逐行讲解：遍历 report['results']，将每项绑定到 r。
    for r in report["results"]:
        # 逐行讲解：查询指定键，缺失时使用默认值（REFERENCE_4B.get(r['suite'])），赋给 ref。
        ref = REFERENCE_4B.get(r["suite"])
        # 逐行讲解：按条件选择结果：f'{ref:.2f}%' if ref is not None else '-'，赋给 ref_text。
        ref_text = f"{ref:.2f}%" if ref is not None else "-"
        # 逐行讲解：向列表末尾追加一个元素（lines.append(f"| {r['suite']} | {r['decisions']} | {r['accuracy']:.2%} | {ref_text} | {r['ece']:.4f} | {r['…）。
        lines.append(
            f"| {r['suite']} | {r['decisions']} | {r['accuracy']:.2%} | "
            f"{ref_text} | {r['ece']:.4f} | {r['brier']:.4f} |"
        )
    # 逐行讲解：通过推导式逐项生成结果：[r['accuracy'] for r in report['results']]，赋给 accuracies。
    accuracies = [r["accuracy"] for r in report["results"]]
    # 逐行讲解：检查条件 len(accuracies) > 1；成立时执行下方分支。
    if len(accuracies) > 1:
        # 逐行讲解：向列表末尾追加一个元素（lines.append(f'| **average** | | **{sum(accuracies) / len(accuracies):.2%}** | | | |')）。
        lines.append(f"| **average** | | **{sum(accuracies) / len(accuracies):.2%}** | | | |")
    # 逐行讲解：返回结果：lines。
    return lines


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：取值 [Path(p) for p in sys.argv[1:]] or sorted(OUTPUT_DIR.glob('eval-*.json'))，赋给 paths。
    paths = [Path(p) for p in sys.argv[1:]] or sorted(OUTPUT_DIR.glob("eval-*.json"))
    # 逐行讲解：检查条件 not paths；成立时执行下方分支。
    if not paths:
        # 逐行讲解：调用 print（print(f'{OUTPUT_DIR} 下没有 eval-*.json，先跑 eval.py')）。
        print(f"{OUTPUT_DIR} 下没有 eval-*.json，先跑 eval.py")
        # 逐行讲解：返回，结束当前函数。
        return
    # 逐行讲解：声明 lines 的类型为 list[str]；构造列表 []。
    lines: list[str] = []
    # 逐行讲解：遍历 paths，将每项绑定到 path。
    for path in paths:
        # 逐行讲解：把一组元素依次加入列表（lines.extend(render(path))）。
        lines.extend(render(path))
    # 逐行讲解：调用 '\n'.join（'\n'.join(lines)），赋给 text。
    text = "\n".join(lines)
    # 逐行讲解：调用 print（print(text)）。
    print(text)
    # 逐行讲解：计算 OUTPUT_DIR / 'summary.md'，赋给 summary。
    summary = OUTPUT_DIR / "summary.md"
    # 逐行讲解：将文本写入文件（summary.write_text(text + '\n', encoding='utf-8')）。
    summary.write_text(text + "\n", encoding="utf-8")
    # 逐行讲解：调用 print（print(f'\n写出 {summary}')）。
    print(f"\n写出 {summary}")


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
