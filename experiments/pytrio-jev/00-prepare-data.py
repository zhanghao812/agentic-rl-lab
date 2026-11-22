"""下载并准备训练/评测数据。

用法（仓库根目录）：
uv run python experiments/pytrio-jev/00-prepare-data.py

产物（experiments/pytrio-jev/datasets/）：
- train.jsonl       typed-decisions train 切出 960 案例 + ToolACE 1200 条
- dev.jsonl         typed-decisions 案例级切分 120 案例（选优用）
- calibration.jsonl typed-decisions 案例级切分 120 案例（温度拟合用）
- eval/*.jsonl      五个评测基准文件
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：hashlib。
import hashlib
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：random。
import random
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：导入模块：urllib.request。
import urllib.request
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：从 datasets 导入 load_dataset。
from datasets import load_dataset

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets'，赋给 DATASET_DIR。
DATASET_DIR = SCRIPT_DIR / "datasets"
# 逐行讲解：计算 DATASET_DIR / 'eval'，赋给 EVAL_DIR。
EVAL_DIR = DATASET_DIR / "eval"

# 逐行讲解：取值 'LocalLLaMA/typed-decisions'，赋给 TYPED_DECISIONS_REPO。
TYPED_DECISIONS_REPO = "LocalLLaMA/typed-decisions"
# 逐行讲解：取值 'Team-ACE/ToolACE'，赋给 TOOLACE_REPO。
TOOLACE_REPO = "Team-ACE/ToolACE"
# 逐行讲解：取值 1200，赋给 TOOLACE_TRAIN_SIZE。
TOOLACE_TRAIN_SIZE = 1200
# 逐行讲解：取值 'pytrio-jev'，赋给 SPLIT_SEED。
SPLIT_SEED = "pytrio-jev"
# 逐行讲解：取值 42，赋给 SAMPLE_SEED。
SAMPLE_SEED = 42

# 逐行讲解：取值 'https://raw.githubusercontent.com/InternLM/Intern-Decision/main/benchmarks/accuracy-v1'，赋给 EVAL_BASE。
EVAL_BASE = (
    "https://raw.githubusercontent.com/InternLM/Intern-Decision/"
    "main/benchmarks/accuracy-v1"
)
# 逐行讲解：构造字典 {'jevbench_easy.jsonl': 'jevbench/easy.jsonl', 'jevbench_original.jsonl': 'jevbench/original.jsonl', 'jevbe…，赋给 EVAL_FILES。
EVAL_FILES = {
    "jevbench_easy.jsonl": "jevbench/easy.jsonl",
    "jevbench_original.jsonl": "jevbench/original.jsonl",
    "jevbench_hard.jsonl": "jevbench/hard.jsonl",
    "typed_decisions.jsonl": "typed_decisions/test.jsonl",
    "toolace.jsonl": "toolace/test.jsonl",
}


# 逐行讲解：定义函数 download_eval_files：封装下方逻辑，供调用方复用。
def download_eval_files() -> None:
    # 逐行讲解：创建目录（EVAL_DIR.mkdir(parents=True, exist_ok=True)）。
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：遍历 EVAL_FILES.items()，将每项绑定到 (name, remote)。
    for name, remote in EVAL_FILES.items():
        # 逐行讲解：计算 EVAL_DIR / name，赋给 target。
        target = EVAL_DIR / name
        # 逐行讲解：检查条件 target.exists()；成立时执行下方分支。
        if target.exists():
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：进入上下文管理器 urllib.request.urlopen(f'{EVAL_BASE}/{remote}')，退出时执行其收尾逻辑。
        with urllib.request.urlopen(f"{EVAL_BASE}/{remote}") as response:
            # 逐行讲解：调用 target.write_bytes（target.write_bytes(response.read())）。
            target.write_bytes(response.read())
        # 逐行讲解：调用 print（print(f'下载 {name}: {sum((1 for _ in open(target)))} 行')）。
        print(f"下载 {name}: {sum(1 for _ in open(target))} 行")


# 逐行讲解：定义函数 convert_typed_decisions：HF 行 → canonical record；gold 取硬标签（软分布在训练里不用）。
def convert_typed_decisions(row: dict) -> dict:
    """HF 行 → canonical record；gold 取硬标签（软分布在训练里不用）。"""
    # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(row['questions'])），赋给 questions。
    questions = json.loads(row["questions"])
    # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(row['gold'])），赋给 gold。
    gold = json.loads(row["gold"])
    # 逐行讲解：返回结果：{'id': row['id'], 'state': json.loads(row['state']), 'questions': questions, 'targets': {field: {'label': g…。
    return {
        "id": row["id"],
        "state": json.loads(row["state"]),
        "questions": questions,
        "targets": {field: {"label": gold[field]["label"]} for field in questions},
    }


# 逐行讲解：定义函数 split_typed_decisions：案例级确定性切分：每个 workflow 240 train / 30 dev / 30 calibration。
def split_typed_decisions() -> tuple[list[dict], list[dict], list[dict]]:
    """案例级确定性切分：每个 workflow 240 train / 30 dev / 30 calibration。"""
    # 逐行讲解：调用 load_dataset（load_dataset(TYPED_DECISIONS_REPO, 'all', split='train')），赋给 dataset。
    dataset = load_dataset(TYPED_DECISIONS_REPO, "all", split="train")
    # 逐行讲解：调用 print（print(f'typed-decisions train: {len(dataset)} 案例')）。
    print(f"typed-decisions train: {len(dataset)} 案例")
    # 逐行讲解：声明 by_workflow 的类型为 dict[str, list[dict]]；构造字典 {}。
    by_workflow: dict[str, list[dict]] = {}
    # 逐行讲解：遍历 dataset，将每项绑定到 row。
    for row in dataset:
        # 逐行讲解：向列表末尾追加一个元素（by_workflow.setdefault(row['workflow'], []).append(row)）。
        by_workflow.setdefault(row["workflow"], []).append(row)

    # 逐行讲解：构造元组 ([], [], [])，赋给 (train, dev, calibration)。
    train, dev, calibration = [], [], []
    # 逐行讲解：遍历 sorted(by_workflow.items())，将每项绑定到 (workflow, rows)。
    for workflow, rows in sorted(by_workflow.items()):
        # 逐行讲解：调用 sorted（sorted(rows, key=lambda r: hashlib.sha256(f"{SPLIT_SEED}:{r['id']}".encode()).hexdigest())），赋给 rows。
        rows = sorted(
            rows,
            key=lambda r: hashlib.sha256(f"{SPLIT_SEED}:{r['id']}".encode()).hexdigest(),
        )
        # 逐行讲解：把一组元素依次加入列表（train.extend((convert_typed_decisions(r) for r in rows[:240]))）。
        train.extend(convert_typed_decisions(r) for r in rows[:240])
        # 逐行讲解：把一组元素依次加入列表（dev.extend((convert_typed_decisions(r) for r in rows[240:270]))）。
        dev.extend(convert_typed_decisions(r) for r in rows[240:270])
        # 逐行讲解：把一组元素依次加入列表（calibration.extend((convert_typed_decisions(r) for r in rows[270:300]))）。
        calibration.extend(convert_typed_decisions(r) for r in rows[270:300])
        # 逐行讲解：调用 print（print(f'  {workflow}: {len(rows)} → 240 train / 30 dev / 30 calibration')）。
        print(f"  {workflow}: {len(rows)} → 240 train / 30 dev / 30 calibration")
    # 逐行讲解：返回结果：(train, dev, calibration)。
    return train, dev, calibration


# 逐行讲解：定义函数 load_eval_request_keys：ToolACE 310 条 eval 行的请求文本指纹，训练采样要避开它们。
def load_eval_request_keys() -> set[str]:
    """ToolACE 310 条 eval 行的请求文本指纹，训练采样要避开它们。"""
    # 逐行讲解：调用 set（set()），赋给 keys。
    keys = set()
    # 逐行讲解：进入上下文管理器 open(EVAL_DIR / 'toolace.jsonl', encoding='utf-8')，退出时执行其收尾逻辑。
    with open(EVAL_DIR / "toolace.jsonl", encoding="utf-8") as f:
        # 逐行讲解：遍历 f，将每项绑定到 line。
        for line in f:
            # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(line)），赋给 row。
            row = json.loads(line)
            # 逐行讲解：调用 keys.add（keys.add(' '.join(row['state']['request'].lower().split()))）。
            keys.add(" ".join(row["state"]["request"].lower().split()))
    # 逐行讲解：返回结果：keys。
    return keys


# 逐行讲解：定义函数 convert_toolace：raw ToolACE 行 → canonical record；只保留标准格式且单工具调用的样本。
def convert_toolace(row: dict) -> dict | None:
    """raw ToolACE 行 → canonical record；只保留标准格式且单工具调用的样本。"""
    # 逐行讲解：按索引或切片读取 row['system']，赋给 system。
    system = row["system"]
    # 逐行讲解：取值 'Here is a list of functions in JSON format that you can invoke:\n'，赋给 marker。
    marker = "Here is a list of functions in JSON format that you can invoke:\n"
    # 逐行讲解：检查条件 marker not in system；成立时执行下方分支。
    if marker not in system:
        # 逐行讲解：返回结果：None。
        return None
    # 逐行讲解：调用 json.JSONDecoder().raw_decode（json.JSONDecoder().raw_decode(system.split(marker, 1)[1].strip())），赋给 (tools, _)。
    tools, _ = json.JSONDecoder().raw_decode(system.split(marker, 1)[1].strip())
    # 逐行讲解：通过推导式逐项生成结果：[tool['name'] for tool in tools]，赋给 tool_names。
    tool_names = [tool["name"] for tool in tools]
    # 逐行讲解：检查条件 len(tool_names) < 2；成立时执行下方分支。
    if len(tool_names) < 2:
        # 逐行讲解：返回结果：None。
        return None

    # 逐行讲解：按索引或切片读取 row['conversations']，赋给 conversations。
    conversations = row["conversations"]
    # 逐行讲解：遍历 enumerate(conversations)，将每项绑定到 (index, message)。
    for index, message in enumerate(conversations):
        # 逐行讲解：检查条件 message['from'] != 'user' or index + 1 >= len(conversations)；成立时执行下方分支。
        if message["from"] != "user" or index + 1 >= len(conversations):
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：按索引或切片读取 conversations[index + 1]，赋给 reply。
        reply = conversations[index + 1]
        # 逐行讲解：检查条件 reply['from'] != 'assistant'；成立时执行下方分支。
        if reply["from"] != "assistant":
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：通过推导式逐项生成结果：[name for name in tool_names if re.search('(?:^|\\[|, )' + re.escape(name) + '\\(', reply['value'])]，赋给 called。
        called = [
            name
            for name in tool_names
            if re.search(r"(?:^|\[|, )" + re.escape(name) + r"\(", reply["value"])
        ]
        # 逐行讲解：检查条件 len(called) != 1；成立时执行下方分支。
        if len(called) != 1:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：按索引或切片读取 message['value']，赋给 request。
        request = message["value"]
        # 逐行讲解：返回结果：{'id': '', 'state': {'request': request, 'tools': [{'name': t['name'], 'description': t.get('description', …。
        return {
            "id": "",
            "state": {
                "request": request,
                "tools": [
                    {"name": t["name"], "description": t.get("description", "")}
                    for t in tools
                ],
            },
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Which single tool from the catalog should be called for this request?",
                    "criteria": {
                        t["name"]: t.get("description", "") for t in tools
                    },
                }
            },
            "targets": {"decision": {"label": called[0]}},
            "_request_key": " ".join(request.lower().split()),
        }
    # 逐行讲解：返回结果：None。
    return None


# 逐行讲解：定义函数 sample_toolace：封装下方逻辑，供调用方复用。
def sample_toolace() -> list[dict]:
    # 逐行讲解：调用 load_eval_request_keys（load_eval_request_keys()），赋给 eval_keys。
    eval_keys = load_eval_request_keys()
    # 逐行讲解：调用 load_dataset（load_dataset(TOOLACE_REPO, split='train')），赋给 dataset。
    dataset = load_dataset(TOOLACE_REPO, split="train")
    # 逐行讲解：调用 print（print(f'ToolACE train: {len(dataset)} 行')）。
    print(f"ToolACE train: {len(dataset)} 行")
    # 逐行讲解：构造列表 []，赋给 pool。
    pool = []
    # 逐行讲解：遍历 dataset，将每项绑定到 row。
    for row in dataset:
        # 逐行讲解：调用 convert_toolace（convert_toolace(row)），赋给 record。
        record = convert_toolace(row)
        # 逐行讲解：检查条件 record is None or record['_request_key'] in eval_keys；成立时执行下方分支。
        if record is None or record["_request_key"] in eval_keys:
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：向列表末尾追加一个元素（pool.append(record)）。
        pool.append(record)
    # 逐行讲解：调用 print（print(f'可转换且不与 eval 重叠: {len(pool)} 条')）。
    print(f"可转换且不与 eval 重叠: {len(pool)} 条")
    # 逐行讲解：打乱数据顺序（random.Random(SAMPLE_SEED).shuffle(pool)）。
    random.Random(SAMPLE_SEED).shuffle(pool)
    # 逐行讲解：按索引或切片读取 pool[:TOOLACE_TRAIN_SIZE]，赋给 selected。
    selected = pool[:TOOLACE_TRAIN_SIZE]
    # 逐行讲解：遍历 enumerate(selected)，将每项绑定到 (index, record)。
    for index, record in enumerate(selected):
        # 逐行讲解：取值 f'toolace-{index:05d}'，赋给 record['id']。
        record["id"] = f"toolace-{index:05d}"
        # 逐行讲解：删除 record['_request_key']。
        del record["_request_key"]
    # 逐行讲解：返回结果：selected。
    return selected


# 逐行讲解：定义函数 write_jsonl：封装下方逻辑，供调用方复用。
def write_jsonl(path: Path, rows: list[dict]) -> None:
    # 逐行讲解：进入上下文管理器 open(path, 'w', encoding='utf-8')，退出时执行其收尾逻辑。
    with open(path, "w", encoding="utf-8") as f:
        # 逐行讲解：遍历 rows，将每项绑定到 row。
        for row in rows:
            # 逐行讲解：调用 f.write（f.write(json.dumps(row, ensure_ascii=False) + '\n')）。
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # 逐行讲解：调用 print（print(f'写出 {path.name}: {len(rows)} 行')）。
    print(f"写出 {path.name}: {len(rows)} 行")


# 逐行讲解：定义函数 main：封装下方逻辑，供调用方复用。
def main() -> None:
    # 逐行讲解：创建目录（DATASET_DIR.mkdir(parents=True, exist_ok=True)）。
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 download_eval_files（download_eval_files()）。
    download_eval_files()

    # 逐行讲解：调用 split_typed_decisions（split_typed_decisions()），赋给 (train_td, dev, calibration)。
    train_td, dev, calibration = split_typed_decisions()
    # 逐行讲解：调用 sample_toolace（sample_toolace()），赋给 train_toolace。
    train_toolace = sample_toolace()

    # 逐行讲解：计算 train_td + train_toolace，赋给 train。
    train = train_td + train_toolace
    # 逐行讲解：打乱数据顺序（random.Random(SAMPLE_SEED).shuffle(train)）。
    random.Random(SAMPLE_SEED).shuffle(train)
    # 逐行讲解：调用 write_jsonl（write_jsonl(DATASET_DIR / 'train.jsonl', train)）。
    write_jsonl(DATASET_DIR / "train.jsonl", train)
    # 逐行讲解：调用 write_jsonl（write_jsonl(DATASET_DIR / 'dev.jsonl', dev)）。
    write_jsonl(DATASET_DIR / "dev.jsonl", dev)
    # 逐行讲解：调用 write_jsonl（write_jsonl(DATASET_DIR / 'calibration.jsonl', calibration)）。
    write_jsonl(DATASET_DIR / "calibration.jsonl", calibration)


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
