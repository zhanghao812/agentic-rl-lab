"""Qwen3.5 的消息模板、图文 chunks、SFT mask 和动作协议。"""

# 逐行讲解：导入模块：io。
import io
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：从 PIL 导入 Image。
from PIL import Image

# 逐行讲解：取值 'Qwen/Qwen3.5-4B'，赋给 BASE_MODEL。
BASE_MODEL = "Qwen/Qwen3.5-4B"
# 逐行讲解：取值 '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'，赋给 MODEL_REVISION。
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# 逐行讲解：读取文件文本（(Path(__file__).parent / 'templates/qwen3_5_spec_o3.jinja').read_text()），赋给 TEMPLATE。
TEMPLATE = (Path(__file__).parent / "templates/qwen3_5_spec_o3.jinja").read_text()
# 逐行讲解：取值 '<|image_pad|>'，赋给 IMAGE_PAD。
IMAGE_PAD = "<|image_pad|>"
# 逐行讲解：构造列表 [{'type': 'function', 'function': {'name': 'spectral_visualization_tool', 'description': 'Visualize a wavel…，赋给 TOOLS。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "spectral_visualization_tool",
            "description": "Visualize a wavelength window from the current spectrum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wavelength_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Wavelength interval [minimum, maximum] in Angstroms",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional spectral feature label",
                    },
                },
                "required": ["wavelength_range"],
            },
        },
    }
]
# 逐行讲解：取值 '-?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?'，赋给 NUMBER。
NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


# 逐行讲解：定义函数 render_messages：渲染可阅读文本；generation=True 时追加下一轮 assistant 思考前缀。
def render_messages(messages, tokenizer, generation=False):
    """渲染可阅读文本；generation=True 时追加下一轮 assistant 思考前缀。"""
    # 逐行讲解：返回结果：tokenizer.apply_chat_template(messages, tools=TOOLS, chat_template=TEMPLATE, tokenize=False, add_generation…。
    return tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        chat_template=TEMPLATE,
        tokenize=False,
        add_generation_prompt=generation,
        enable_thinking=True,
    )


# 逐行讲解：定义函数 image_chunk：把一张图编码为 PNG，并用匹配的处理器计算视觉 token 数。
def image_chunk(path, image_processor):
    """把一张图编码为 PNG，并用匹配的处理器计算视觉 token 数。"""
    # 原图中透明背景合成到白色；传入 processor 和服务端的是同一张 RGB 图。
    with Image.open(path) as source:
        # 逐行讲解：调用 source.convert（source.convert('RGBA')），赋给 rgba。
        rgba = source.convert("RGBA")
        # 逐行讲解：调用 Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert（Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')），赋给 image。
        image = Image.alpha_composite(
            Image.new("RGBA", rgba.size, "white"),
            rgba,
        ).convert("RGB")

    # 逐行讲解：调用 io.BytesIO（io.BytesIO()），赋给 buffer。
    buffer = io.BytesIO()
    # 逐行讲解：调用 image.save（image.save(buffer, format='PNG')）。
    image.save(buffer, format="PNG")
    # 逐行讲解：调用 image_processor.get_number_of_image_patches（image_processor.get_number_of_image_patches(image.height, image.width, images_kwargs={})），赋给 patches。
    patches = image_processor.get_number_of_image_patches(
        image.height,
        image.width,
        images_kwargs={},
    )
    # 逐行讲解：返回结果：trio.ImageChunk(data=buffer.getvalue(), format='png', expected_tokens=patches // image_processor.merge_size…。
    return trio.ImageChunk(
        data=buffer.getvalue(),
        format="png",
        expected_tokens=patches // image_processor.merge_size ** 2,
    )


# 逐行讲解：定义函数 pack_chunks：image_pad 展开成视觉 token；target 和 mask 随之扩展。
def pack_chunks(ids, masks, image_paths, tokenizer, image_processor):
    """image_pad 展开成视觉 token；target 和 mask 随之扩展。"""
    # 逐行讲解：调用 tokenizer.convert_tokens_to_ids（tokenizer.convert_tokens_to_ids(IMAGE_PAD)），赋给 image_id。
    image_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    # 逐行讲解：调用 iter（iter(image_paths)），赋给 paths。
    paths = iter(image_paths)
    # 逐行讲解：构造元组 ([], [], [])，赋给 (chunks, targets, weights)。
    chunks, targets, weights = [], [], []
    # 逐行讲解：取值 0，赋给 start。
    start = 0

    # 逐行讲解：遍历 enumerate(ids)，将每项绑定到 (index, token)。
    for index, token in enumerate(ids):
        # 逐行讲解：检查条件 token == image_id；成立时执行下方分支。
        if token == image_id:
            # 先保留图片前面的文本，以及模板给出的 assistant 监督范围。
            chunks.append(trio.types.EncodedTextChunk(tokens=ids[start:index]))
            # 逐行讲解：把一组元素依次加入列表（targets.extend(ids[start:index])）。
            targets.extend(ids[start:index])
            # 逐行讲解：把一组元素依次加入列表（weights.extend(masks[start:index])）。
            weights.extend(masks[start:index])

            # 一枚 image_pad 替换为实际图片；所有视觉位置只提供上下文。
            chunk = image_chunk(next(paths), image_processor)
            # 逐行讲解：向列表末尾追加一个元素（chunks.append(chunk)）。
            chunks.append(chunk)
            # 逐行讲解：把一组元素依次加入列表（targets.extend([0] * chunk.expected_tokens)）。
            targets.extend([0] * chunk.expected_tokens)
            # 逐行讲解：把一组元素依次加入列表（weights.extend([0.0] * chunk.expected_tokens)）。
            weights.extend([0.0] * chunk.expected_tokens)
            # 逐行讲解：计算 index + 1，赋给 start。
            start = index + 1

    # 最后一张图片后的文本也要进入输入和监督序列。
    chunks.append(trio.types.EncodedTextChunk(tokens=ids[start:]))
    # 逐行讲解：把一组元素依次加入列表（targets.extend(ids[start:])）。
    targets.extend(ids[start:])
    # 逐行讲解：把一组元素依次加入列表（weights.extend(masks[start:])）。
    weights.extend(masks[start:])
    # 逐行讲解：返回结果：(trio.ModelInput(chunks=chunks), np.asarray(targets, dtype=np.int64), np.asarray(weights, dtype=np.float32))。
    return (
        trio.ModelInput(chunks=chunks),
        np.asarray(targets, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


# 逐行讲解：定义函数 encode_messages：从模板取得 token 与 assistant mask，再把图片占位符展开为图文 chunks。
def encode_messages(messages, image_paths, tokenizer, image_processor, generation=False):
    """从模板取得 token 与 assistant mask，再把图片占位符展开为图文 chunks。"""
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template(messages, tools=TOOLS, chat_template=TEMPLATE, tokenize=True, return_dict=Tru…），赋给 encoded。
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        chat_template=TEMPLATE,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=generation,
        enable_thinking=True,
    )
    # 逐行讲解：返回结果：pack_chunks(encoded['input_ids'], encoded['assistant_masks'], image_paths, tokenizer, image_processor)。
    return pack_chunks(
        encoded["input_ids"],
        encoded["assistant_masks"],
        image_paths,
        tokenizer,
        image_processor,
    )


# 逐行讲解：定义函数 build_sft_datum：构造完整多轮轨迹的 SFT 输入，仅监督 assistant 生成的位置。
def build_sft_datum(row, tokenizer, image_processor):
    """构造完整多轮轨迹的 SFT 输入，仅监督 assistant 生成的位置。"""
    # 逐行讲解：调用 encode_messages（encode_messages(row['messages'], row['images'], tokenizer, image_processor)），赋给 (model_input, targets, weights)。
    model_input, targets, weights = encode_messages(
        row["messages"],
        row["images"],
        tokenizer,
        image_processor,
    )

    # 最后一个文本 token 只作为 target，所有序列统一右移一次。
    chunks = model_input.chunks[:-1] + [
        trio.types.EncodedTextChunk(tokens=model_input.chunks[-1].tokens[:-1]),
    ]
    # 逐行讲解：取值 0，赋给 targets[weights == 0]。
    targets[weights == 0] = 0
    # 逐行讲解：返回结果：trio.Datum(model_input=trio.ModelInput(chunks=chunks), loss_fn_inputs={'target_tokens': targets[1:], 'weigh…。
    return trio.Datum(
        model_input=trio.ModelInput(chunks=chunks),
        loss_fn_inputs={
            "target_tokens": targets[1:],
            "weights": weights[1:],
        },
    )


# 逐行讲解：定义函数 build_rl_datum：将实际采样轨迹右移一位，只在 assistant 动作位置计算 PPO loss。
def build_rl_datum(trajectory, advantage):
    """将实际采样轨迹右移一位，只在 assistant 动作位置计算 PPO loss。"""
    # 逐行讲解：取值 trajectory['model_input'].chunks，赋给 chunks。
    chunks = trajectory["model_input"].chunks
    # 逐行讲解：调用 trio.ModelInput（trio.ModelInput(chunks=[*chunks[:-1], trio.types.EncodedTextChunk(tokens=chunks[-1].tokens[:-1])])），赋给 model_input。
    model_input = trio.ModelInput(chunks=[
        *chunks[:-1],
        trio.types.EncodedTextChunk(tokens=chunks[-1].tokens[:-1]),
    ])
    # 逐行讲解：按索引或切片读取 np.asarray(trajectory['action_mask'], dtype=np.float32)[1:]，赋给 action_mask。
    action_mask = np.asarray(trajectory["action_mask"], dtype=np.float32)[1:]

    # 逐行讲解：返回结果：trio.Datum(model_input=model_input, loss_fn_inputs={'target_tokens': np.asarray(trajectory['target_tokens']…。
    return trio.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": np.asarray(trajectory["target_tokens"], dtype=np.int64)[1:],
            "logprobs": np.asarray(trajectory["logprobs"], dtype=np.float32)[1:],
            "advantages": action_mask * advantage,
        },
    )


# 逐行讲解：定义函数 tool_continuation：只编码模板中新追加的 tool 返回和 assistant 前缀，保留旧采样 token。
def tool_continuation(messages, image_paths, tokenizer, image_processor):
    """只编码模板中新追加的 tool 返回和 assistant 前缀，保留旧采样 token。"""
    # 逐行讲解：调用 render_messages（render_messages(messages, tokenizer, generation=True)），赋给 rendered。
    rendered = render_messages(messages, tokenizer, generation=True)
    # 逐行讲解：按索引或切片读取 rendered[rendered.rindex('<|im_start|>user\n<tool_response>'):]，赋给 suffix。
    suffix = rendered[rendered.rindex("<|im_start|>user\n<tool_response>") :]
    # 逐行讲解：将文本编码为 token（tokenizer.encode(suffix, add_special_tokens=False)），赋给 ids。
    ids = tokenizer.encode(suffix, add_special_tokens=False)
    # 逐行讲解：返回结果：pack_chunks(ids, [0] * len(ids), image_paths, tokenizer, image_processor)[0]。
    return pack_chunks(
        ids,
        [0] * len(ids),
        image_paths,
        tokenizer,
        image_processor,
    )[0]


# 逐行讲解：定义函数 parse_action：动作语法属于环境规则；非法动作结束该轨迹，不重试或补写。
def parse_action(text):
    """动作语法属于环境规则；非法动作结束该轨迹，不重试或补写。"""
    # 逐行讲解：调用 text.partition('</think>')[2].strip().removesuffix('<|im_end|>').strip（text.partition('</think>')[2].strip().removesuffix('<|im_end|>').strip()），赋给 body。
    body = (
        text.partition("</think>")[2]
        .strip()
        .removesuffix("<|im_end|>")
        .strip()
    )

    # 最终回答必须使用约定的 answer / boxed 格式。
    answer = re.fullmatch(r"<answer>\s*\\boxed\{(YES|NO)\}(.*?)</answer>", body, re.S)
    # 逐行讲解：检查条件 answer；成立时执行下方分支。
    if answer:
        # 逐行讲解：返回结果：{'kind': 'answer', 'prediction': answer[1]}。
        return {"kind": "answer", "prediction": answer[1]}

    # 工具动作使用 Qwen3.5 原生 XML，从参数中解析波长区间和可选标签。
    call = re.fullmatch(
        r"<tool_call>\s*<function=spectral_visualization_tool>"
        r"\s*(.*?)\s*</function>\s*</tool_call>",
        body,
        re.S,
    )
    # 逐行讲解：检查条件 call；成立时执行下方分支。
    if call:
        # 逐行讲解：调用 dict（dict(re.findall('<parameter=(\\w+)>\\s*(.*?)\\s*</parameter>', call[1], re.S))），赋给 parameters。
        parameters = dict(
            re.findall(
                r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
                call[1],
                re.S,
            )
        )
        # 逐行讲解：调用 re.fullmatch（re.fullmatch(f'\\[\\s*({NUMBER})\\s*,\\s*({NUMBER})\\s*\\]', parameters.get('wavelength_range', ''))），赋给 interval。
        interval = re.fullmatch(
            rf"\[\s*({NUMBER})\s*,\s*({NUMBER})\s*\]",
            parameters.get("wavelength_range", ""),
        )
        # 逐行讲解：检查条件 interval；成立时执行下方分支。
        if interval:
            # 逐行讲解：返回结果：{'kind': 'tool', 'wavelength_range': [float(interval[1]), float(interval[2])], 'label': parameters.get('lab…。
            return {
                "kind": "tool",
                "wavelength_range": [float(interval[1]), float(interval[2])],
                "label": parameters.get("label", ""),
            }

    # 逐行讲解：返回结果：{'kind': 'invalid'}。
    return {"kind": "invalid"}
