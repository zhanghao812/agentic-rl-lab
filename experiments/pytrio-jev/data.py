"""判别决策格式编译：canonical record → chat messages → tokens/weights。

- 候选符号表为 62 个单 token 符号（A-Z a-z 0-9），choice 按插入序、score 按等级序、
  noul 固定 [no, yes]；
- assistant 内容是 JSON 骨架，值的位置放占位符 marker。不加 special token（TRIO 服务端
  词表不可扩展），用已有单 token "decision"（Qwen3.5 tokenizer id 61781）；
- labels 只落在 marker 位置（右移后 = marker 前一位置预测答案符号），其余全 0。
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：json。
import json
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio

# 逐行讲解：取值 'You are a careful decision assistant. Use the state and decision schema in the user message to make the re…，赋给 SYSTEM_PROMPT。
SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in "
    "the user message to make the requested decisions. For every field, choose "
    "exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
    "return one valid JSON object mapping each field name to its chosen symbol. "
    "Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text."
)
# 逐行讲解：取值 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'，赋给 ANSWER_SYMBOLS。
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
# 逐行讲解：取值 'decision'，赋给 MARKER。
MARKER = "decision"
# 逐行讲解：取值 'The answer is yes (affirmative, or align with the claim).'，赋给 NOUL_YES。
NOUL_YES = "The answer is yes (affirmative, or align with the claim)."
# 逐行讲解：取值 'The answer is no (negative, or disagree with the claim).'，赋给 NOUL_NO。
NOUL_NO = "The answer is no (negative, or disagree with the claim)."
# 逐行讲解：计算 Path(__file__).resolve().parent / 'templates' / 'qwen3_5_jev.jinja'，赋给 TEMPLATE_PATH。
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "qwen3_5_jev.jinja"


# 逐行讲解：定义函数 question_options：按题型展开为有序的 (value, description) 列表。
def question_options(question: dict) -> list[tuple[str, str]]:
    """按题型展开为有序的 (value, description) 列表。"""
    # 逐行讲解：按索引或切片读取 question['type']，赋给 kind。
    kind = question["type"]
    # 逐行讲解：查询指定键，缺失时使用默认值（question.get('criteria')），赋给 criteria。
    criteria = question.get("criteria")
    # 逐行讲解：检查条件 kind == 'choice'；成立时执行下方分支。
    if kind == "choice":
        # 逐行讲解：返回结果：[(str(k), str(v)) for k, v in criteria.items()]。
        return [(str(k), str(v)) for k, v in criteria.items()]
    # 逐行讲解：检查条件 kind == 'score'；成立时执行下方分支。
    if kind == "score":
        # 逐行讲解：检查条件 isinstance(criteria, list)；成立时执行下方分支。
        if isinstance(criteria, list):
            # 逐行讲解：返回结果：[(str(i), str(v)) for i, v in enumerate(criteria)]。
            return [(str(i), str(v)) for i, v in enumerate(criteria)]
        # 逐行讲解：返回结果：[(str(k), str(v)) for k, v in criteria.items()]。
        return [(str(k), str(v)) for k, v in criteria.items()]
    # 逐行讲解：检查条件 kind == 'noul'；成立时执行下方分支。
    if kind == "noul":
        # 逐行讲解：取值 criteria or {}，赋给 descriptions。
        descriptions = criteria or {}
        # 逐行讲解：调用 next（next((str(v) for k, v in descriptions.items() if str(k).lower() in {'yes', 'true', '1'}), NOUL_YES)），赋给 yes。
        yes = next(
            (str(v) for k, v in descriptions.items() if str(k).lower() in {"yes", "true", "1"}),
            NOUL_YES,
        )
        # 逐行讲解：调用 next（next((str(v) for k, v in descriptions.items() if str(k).lower() in {'no', 'false', '0'}), NOUL_NO)），赋给 no。
        no = next(
            (str(v) for k, v in descriptions.items() if str(k).lower() in {"no", "false", "0"}),
            NOUL_NO,
        )
        # 逐行讲解：返回结果：[('no', no), ('yes', yes)]。
        return [("no", no), ("yes", yes)]
    # 逐行讲解：抛出异常：ValueError(f'unsupported question type: {kind!r}')。
    raise ValueError(f"unsupported question type: {kind!r}")


# 逐行讲解：定义函数 normalize_label：noul 的 true/false/1/0 归一到 yes/no，其余题型转字符串。
def normalize_label(question: dict, label: object) -> str:
    """noul 的 true/false/1/0 归一到 yes/no，其余题型转字符串。"""
    # 逐行讲解：检查条件 question['type'] == 'noul'；成立时执行下方分支。
    if question["type"] == "noul":
        # 逐行讲解：返回结果：'yes' if str(label).lower() in {'yes', 'true', '1'} else 'no'。
        return "yes" if str(label).lower() in {"yes", "true", "1"} else "no"
    # 逐行讲解：返回结果：str(label)。
    return str(label)


# 逐行讲解：定义函数 compile_record：把一条 canonical record 编译成 messages 和符号映射。
def compile_record(record: dict, include_targets: bool = True) -> dict:
    """把一条 canonical record 编译成 messages 和符号映射。

    返回 fields / options（field → [(symbol, value, description)]）/ targets（field → 符号）。
    """
    # 逐行讲解：按索引或切片读取 record['questions']，赋给 questions。
    questions = record["questions"]
    # 逐行讲解：声明 fields 的类型为 list[str]；构造列表 []。
    fields: list[str] = []
    # 逐行讲解：声明 options 的类型为 dict[str, list[tuple[str, str, str]]]；构造字典 {}。
    options: dict[str, list[tuple[str, str, str]]] = {}
    # 逐行讲解：声明 targets 的类型为 dict[str, str]；构造字典 {}。
    targets: dict[str, str] = {}
    # 逐行讲解：声明 schema_lines 的类型为 list[str]；构造列表 []。
    schema_lines: list[str] = []
    # 逐行讲解：遍历 questions.items()，将每项绑定到 (field, question)。
    for field, question in questions.items():
        # 逐行讲解：调用 question_options（question_options(question)），赋给 opts。
        opts = question_options(question)
        # 逐行讲解：调用 str（str(field)），赋给 field。
        field = str(field)
        # 逐行讲解：向列表末尾追加一个元素（fields.append(field)）。
        fields.append(field)
        # 逐行讲解：按索引或切片读取 ANSWER_SYMBOLS[:len(opts)]，赋给 symbols。
        symbols = ANSWER_SYMBOLS[: len(opts)]
        # 逐行讲解：通过推导式逐项生成结果：[(symbol, value, description) for symbol, (value, description) in zip(symbols, opts)]，赋给 options[field]。
        options[field] = [
            (symbol, value, description)
            for symbol, (value, description) in zip(symbols, opts)
        ]
        # 逐行讲解：向列表末尾追加一个元素（schema_lines.append(f"{field}: {question['instructions']}")）。
        schema_lines.append(f"{field}: {question['instructions']}")
        # 逐行讲解：遍历 options[field]，将每项绑定到 (symbol, value, description)。
        for symbol, value, description in options[field]:
            # 逐行讲解：向列表末尾追加一个元素（schema_lines.append(f'    {symbol} = {value}: {description}')）。
            schema_lines.append(f"    {symbol} = {value}: {description}")
        # 逐行讲解：检查条件 include_targets；成立时执行下方分支。
        if include_targets:
            # 逐行讲解：调用 normalize_label（normalize_label(question, record['targets'][field]['label'])），赋给 label。
            label = normalize_label(question, record["targets"][field]["label"])
            # 逐行讲解：通过推导式逐项生成结果：[value for _, value, _ in options[field]]，赋给 values。
            values = [value for _, value, _ in options[field]]
            # 逐行讲解：按索引或切片读取 symbols[values.index(label)]，赋给 targets[field]。
            targets[field] = symbols[values.index(label)]

    # 逐行讲解：将对象序列化为 JSON 字符串（json.dumps(record.get('state'), ensure_ascii=False, indent=2)），赋给 state。
    state = json.dumps(record.get("state"), ensure_ascii=False, indent=2)
    # 逐行讲解：计算 f'Return one answer for every field using the supplied answer symbols.\n\n## State\n{state}\n## Decision sc…，赋给 user_text。
    user_text = (
        "Return one answer for every field using the supplied answer symbols.\n\n"
        f"## State\n{state}\n"
        "## Decision schema\n" + "\n".join(schema_lines)
    )
    # 逐行讲解：将对象序列化为 JSON 字符串（json.dumps(dict.fromkeys(fields, MARKER), ensure_ascii=False, indent=4)），赋给 skeleton。
    skeleton = json.dumps(dict.fromkeys(fields, MARKER), ensure_ascii=False, indent=4)
    # 逐行讲解：构造列表 [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user_text}, {'role': 'assistant'…，赋给 messages。
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": skeleton},
    ]
    # 逐行讲解：返回结果：{'messages': messages, 'fields': fields, 'options': options, 'targets': targets if include_targets else None}。
    return {
        "messages": messages,
        "fields": fields,
        "options": options,
        "targets": targets if include_targets else None,
    }


# 逐行讲解：定义类 Encoder：绑定 tokenizer 与钉死的 chat template，负责 tokenization 和 marker 定位。
class Encoder:
    """绑定 tokenizer 与钉死的 chat template，负责 tokenization 和 marker 定位。"""

    # 逐行讲解：定义函数 __init__：封装下方逻辑，供调用方复用。
    def __init__(self, tokenizer):
        # 逐行讲解：取值 tokenizer，赋给 self.tokenizer。
        self.tokenizer = tokenizer
        # 逐行讲解：读取文件文本（TEMPLATE_PATH.read_text()），赋给 self.template。
        self.template = TEMPLATE_PATH.read_text()
        # 逐行讲解：将文本编码为 token（tokenizer.encode(MARKER, add_special_tokens=False)），赋给 marker_ids。
        marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
        # 逐行讲解：断言 len(marker_ids) == 1；条件不成立时报告错误。
        assert len(marker_ids) == 1, "marker 必须是单 token"
        # 逐行讲解：按索引或切片读取 marker_ids[0]，赋给 self.marker_id。
        self.marker_id = marker_ids[0]

    # 逐行讲解：定义函数 encode：渲染完整训练/推理文本，返回 token ids 与各字段的 marker 索引。
    def encode(self, record: dict, include_targets: bool = True) -> dict:
        """渲染完整训练/推理文本，返回 token ids 与各字段的 marker 索引。

        marker 索引靠"prefix 之外、骨架之内"定位，不扫全文，因此 instructions 或
        state 里出现单词 decision 也不会误判。
        """
        # 逐行讲解：调用 compile_record（compile_record(record, include_targets)），赋给 compiled。
        compiled = compile_record(record, include_targets)
        # 逐行讲解：按索引或切片读取 compiled['messages']，赋给 messages。
        messages = compiled["messages"]
        # 逐行讲解：将文本编码为 token（self.tokenizer.encode(self.tokenizer.apply_chat_template(messages, chat_template=self.template, tokenize=Fa…），赋给 ids。
        ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(
                messages, chat_template=self.template, tokenize=False
            ),
            add_special_tokens=False,
        )
        # 逐行讲解：将文本编码为 token（self.tokenizer.encode(self.tokenizer.apply_chat_template(messages[:-1], chat_template=self.template, tokeni…），赋给 prefix_ids。
        prefix_ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(
                messages[:-1],
                chat_template=self.template,
                tokenize=False,
                add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )
        # 逐行讲解：断言 ids[:len(prefix_ids)] == prefix_ids；条件不成立时报告错误。
        assert ids[: len(prefix_ids)] == prefix_ids, "prefix/skeleton 边界 tokenization 不稳定"
        # 逐行讲解：按索引或切片读取 ids[len(prefix_ids):]，赋给 skeleton_ids。
        skeleton_ids = ids[len(prefix_ids) :]
        # 骨架里 marker token 的命中包含字段名中的出现（如 ToolACE 的字段就叫 "decision"，
        # 骨架为 {"decision": "decision"}），按字段顺序先跳过 key 里的命中，再取 value。
        hits = [
            i for i, token_id in enumerate(skeleton_ids) if token_id == self.marker_id
        ]
        # 逐行讲解：声明 markers 的类型为 dict[str, int]；构造字典 {}。
        markers: dict[str, int] = {}
        # 逐行讲解：取值 0，赋给 cursor。
        cursor = 0
        # 逐行讲解：遍历 compiled['fields']，将每项绑定到 field。
        for field in compiled["fields"]:
            # 逐行讲解：更新 cursor：cursor += self.tokenizer.encode(field, add_special_tokens=False).count(self.marker_id)。
            cursor += self.tokenizer.encode(field, add_special_tokens=False).count(
                self.marker_id
            )
            # 逐行讲解：计算 len(prefix_ids) + hits[cursor]，赋给 markers[field]。
            markers[field] = len(prefix_ids) + hits[cursor]
            # 逐行讲解：更新 cursor：cursor += 1。
            cursor += 1
        # 逐行讲解：断言 cursor == len(hits)；条件不成立时报告错误。
        assert cursor == len(hits), "骨架中 marker 数与字段数不一致"
        # 逐行讲解：返回结果：{**compiled, 'ids': ids, 'markers': markers}。
        return {
            **compiled,
            "ids": ids,
            "markers": markers,
        }

    # 逐行讲解：定义函数 symbol_token_id：封装下方逻辑，供调用方复用。
    def symbol_token_id(self, symbol: str) -> int:
        # 逐行讲解：返回结果：self.tokenizer.encode(symbol, add_special_tokens=False)[0]。
        return self.tokenizer.encode(symbol, add_special_tokens=False)[0]

    # 逐行讲解：定义函数 build_sft_datum：构造 masked SFT 的 Datum：只在 marker 前一位置监督答案符号。
    def build_sft_datum(self, record: dict) -> trio.Datum:
        """构造 masked SFT 的 Datum：只在 marker 前一位置监督答案符号。"""
        # 逐行讲解：将文本编码为 token（self.encode(record)），赋给 encoded。
        encoded = self.encode(record)
        # 逐行讲解：按索引或切片读取 encoded['ids']，赋给 ids。
        ids = encoded["ids"]
        # 逐行讲解：调用 list（list(ids[1:])），赋给 target_tokens。
        target_tokens = list(ids[1:])
        # 逐行讲解：计算 [0.0] * len(target_tokens)，赋给 weights。
        weights = [0.0] * len(target_tokens)
        # 逐行讲解：遍历 encoded['targets'].items()，将每项绑定到 (field, symbol)。
        for field, symbol in encoded["targets"].items():
            # 逐行讲解：计算 encoded['markers'][field] - 1，赋给 position。
            position = encoded["markers"][field] - 1
            # 逐行讲解：调用 self.symbol_token_id（self.symbol_token_id(symbol)），赋给 target_tokens[position]。
            target_tokens[position] = self.symbol_token_id(symbol)
            # 逐行讲解：取值 1.0，赋给 weights[position]。
            weights[position] = 1.0
        # 逐行讲解：返回结果：trio.Datum(model_input=trio.ModelInput.from_ints(ids[:-1]), loss_fn_inputs={'target_tokens': np.asarray(tar…。
        return trio.Datum(
            model_input=trio.ModelInput.from_ints(ids[:-1]),
            loss_fn_inputs={
                "target_tokens": np.asarray(target_tokens, dtype=np.int64),
                "weights": np.asarray(weights, dtype=np.float32),
            },
        )


# 逐行讲解：定义函数 load_jsonl：封装下方逻辑，供调用方复用。
def load_jsonl(path: str | Path) -> list[dict]:
    # 逐行讲解：进入上下文管理器 open(path, encoding='utf-8')，退出时执行其收尾逻辑。
    with open(path, encoding="utf-8") as f:
        # 逐行讲解：返回结果：[json.loads(line) for line in f if line.strip()]。
        return [json.loads(line) for line in f if line.strip()]
