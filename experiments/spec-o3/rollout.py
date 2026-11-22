"""共用的多轮光谱审核循环：采样 → 绘图 → 回填观察 → 再采样。"""

# 逐行讲解：从 copy 导入 deepcopy。
from copy import deepcopy

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio

# 逐行讲解：从 protocol 导入 encode_messages, parse_action, tool_continuation。
from protocol import encode_messages, parse_action, tool_continuation
# 逐行讲解：从 tools 导入 spectral_visualization_tool。
from tools import spectral_visualization_tool


# 逐行讲解：定义异步函数 rollout：执行一条光谱审核轨迹，保留每轮生成结果和用于后续 RL 的 token 信息。
async def rollout(
    row,
    sampler,
    tokenizer,
    image_processor,
    output_dir,
    max_turns=8,
    max_tokens=2048,
    max_seq_len=16384,
    temperature=0.6,
    seed=42,
    top_p=0.95,
    first_sequence=None,
):
    """执行一条光谱审核轨迹，保留每轮生成结果和用于后续 RL 的 token 信息。"""
    # 1. 构造初始图文输入，并加载可供工具重绘的原始光谱。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：调用 deepcopy（deepcopy(row['messages'])），赋给 messages。
    messages = deepcopy(row["messages"])
    # 逐行讲解：调用 encode_messages（encode_messages(messages, row['images'], tokenizer, image_processor, generation=True)），赋给 (prompt, _, _)。
    prompt, _, _ = encode_messages(
        messages,
        row["images"],
        tokenizer,
        image_processor,
        generation=True,
    )
    # 逐行讲解：进入上下文管理器 np.load(row['spectrum'])，退出时执行其收尾逻辑。
    with np.load(row["spectrum"]) as spectrum:
        # 逐行讲解：构造元组 (spectrum['wavelength'], spectrum['flux'], float(spectrum['redshift']))，赋给 (wavelength, flux, redshift)。
        wavelength, flux, redshift = (
            spectrum["wavelength"],
            spectrum["flux"],
            float(spectrum["redshift"]),
        )

    # 初始 prompt 只提供上下文；生成 token 和工具观察在下面逐轮追加。
    target_tokens = [0] * len(prompt)
    # 逐行讲解：计算 [0.0] * len(prompt)，赋给 logprobs。
    logprobs = [0.0] * len(prompt)
    # 逐行讲解：计算 [0.0] * len(prompt)，赋给 action_mask。
    action_mask = [0.0] * len(prompt)
    # 逐行讲解：构造列表 []，赋给 turns。
    turns = []
    # 逐行讲解：取值 None，赋给 prediction。
    prediction = None
    # 逐行讲解：取值 'max_turns'，赋给 finish_reason。
    finish_reason = "max_turns"
    # 逐行讲解：调用 tokenizer.convert_tokens_to_ids（tokenizer.convert_tokens_to_ids('<|im_end|>')），赋给 im_end。
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # 逐行讲解：遍历 range(max_turns)，将每项绑定到 turn。
    for turn in range(max_turns):
        # 2. 在剩余上下文预算内，生成下一段思考和动作。
        remaining = max_seq_len - len(prompt)
        # 逐行讲解：检查条件 remaining <= 0；成立时执行下方分支。
        if remaining <= 0:
            # 逐行讲解：取值 'max_seq_len'，赋给 finish_reason。
            finish_reason = "max_seq_len"
            # 逐行讲解：结束当前循环。
            break

        # GRPO 第一轮已按 num_samples 成组采样；后续沿各自的上下文继续。
        if turn == 0 and first_sequence is not None:
            # 逐行讲解：取值 first_sequence，赋给 sequence。
            sequence = first_sequence
        else:
            # 逐行讲解：等待异步操作 sampler.sample_async(prompt=prompt, num_samples=1, sampling_params=trio.SamplingParams(max_tokens=min(max_t… 完成并取得结果，赋给 response。
            response = await sampler.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=trio.SamplingParams(
                    max_tokens=min(max_tokens, remaining),
                    temperature=temperature,
                    top_p=top_p,
                    stop=[im_end],
                    seed=seed + turn,
                ),
            )
            # 逐行讲解：按索引或切片读取 response.sequences[0]，赋给 sequence。
            sequence = response.sequences[0]
        # 逐行讲解：将 token 解码为文本（tokenizer.decode(sequence.tokens, skip_special_tokens=False)），赋给 text。
        text = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
        # 逐行讲解：调用 parse_action（parse_action(text)），赋给 action。
        action = parse_action(text)

        # 追加服务实际生成的 token，不重新编码 assistant 历史。
        if sequence.tokens:
            # 逐行讲解：调用 trio.ModelInput（trio.ModelInput(chunks=[*prompt.chunks, trio.types.EncodedTextChunk(tokens=sequence.tokens)])），赋给 prompt。
            prompt = trio.ModelInput(chunks=[
                *prompt.chunks,
                trio.types.EncodedTextChunk(tokens=sequence.tokens),
            ])
        # 逐行讲解：把一组元素依次加入列表（target_tokens.extend(sequence.tokens)）。
        target_tokens.extend(sequence.tokens)
        # 逐行讲解：把一组元素依次加入列表（logprobs.extend(sequence.logprobs)）。
        logprobs.extend(sequence.logprobs)
        # 逐行讲解：把一组元素依次加入列表（action_mask.extend([1.0] * len(sequence.tokens))）。
        action_mask.extend([1.0] * len(sequence.tokens))
        # 逐行讲解：构造字典 {'turn': turn, 'text': text, 'tokens': sequence.tokens, 'logprobs': sequence.logprobs, 'stop_reason': seque…，赋给 record。
        record = {
            "turn": turn,
            "text": text,
            "tokens": sequence.tokens,
            "logprobs": sequence.logprobs,
            "stop_reason": sequence.stop_reason,
            "action": action,
        }
        # 逐行讲解：向列表末尾追加一个元素（turns.append(record)）。
        turns.append(record)

        # 3. 最终答案、非法动作和轮数耗尽都会结束当前轨迹。
        if action["kind"] == "answer":
            # 逐行讲解：按索引或切片读取 action['prediction']，赋给 prediction。
            prediction = action["prediction"]
            # 逐行讲解：取值 'answer'，赋给 finish_reason。
            finish_reason = "answer"
            # 逐行讲解：结束当前循环。
            break
        # 逐行讲解：检查条件 action['kind'] == 'invalid'；成立时执行下方分支。
        if action["kind"] == "invalid":
            # 逐行讲解：取值 'invalid_action'，赋给 finish_reason。
            finish_reason = "invalid_action"
            # 逐行讲解：结束当前循环。
            break
        # 逐行讲解：检查条件 turn + 1 == max_turns；成立时执行下方分支。
        if turn + 1 == max_turns:
            # 逐行讲解：结束当前循环。
            break

        # 4. 按模型选择的波段绘图，再追加 assistant 消息和工具观察。
        observation = spectral_visualization_tool(
            wavelength,
            flux,
            redshift,
            action["wavelength_range"],
            action["label"],
            output_dir / f"turn-{turn}.png",
        )
        # 逐行讲解：取值 observation，赋给 record['observation']。
        record["observation"] = observation
        # 逐行讲解：调用 text.partition（text.partition('</think>')），赋给 (reasoning, _, content)。
        reasoning, _, content = text.partition("</think>")
        # 逐行讲解：向列表末尾追加一个元素（messages.append({'role': 'assistant', 'reasoning_content': reasoning, 'content': content.removesuffix('<|im…）。
        messages.append({
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": content.removesuffix("<|im_end|>").strip(),
            "tool_calls": [],
        })
        # 逐行讲解：向列表末尾追加一个元素（messages.append({'role': 'tool', 'content': observation['content']})）。
        messages.append({"role": "tool", "content": observation["content"]})

        # 5. 只编码新增的观察和下一轮 assistant 前缀。
        continuation = tool_continuation(
            messages,
            observation["images"],
            tokenizer,
            image_processor,
        )
        # stop token 若被 sampler 排除，补到上下文；它没有采样 logprob，也不参与 loss。
        separator = "\n" if sequence.tokens[-1] == im_end else "<|im_end|>\n"
        # 逐行讲解：调用 trio.ModelInput（trio.ModelInput(chunks=[trio.types.EncodedTextChunk(tokens=tokenizer.encode(separator, add_special_tokens=F…），赋给 context。
        context = trio.ModelInput(chunks=[
            trio.types.EncodedTextChunk(
                tokens=tokenizer.encode(separator, add_special_tokens=False),
            ),
            *continuation.chunks,
        ])
        # 逐行讲解：取值 len(prompt) + len(context) < max_seq_len，赋给 record['observation_delivered']。
        record["observation_delivered"] = len(prompt) + len(context) < max_seq_len
        # 逐行讲解：检查条件 not record['observation_delivered']；成立时执行下方分支。
        if not record["observation_delivered"]:
            # 逐行讲解：取值 'max_seq_len'，赋给 finish_reason。
            finish_reason = "max_seq_len"
            # 逐行讲解：结束当前循环。
            break

        # 工具观察属于上下文，target、logprob 和动作 mask 都填零。
        prompt = trio.ModelInput(chunks=[*prompt.chunks, *context.chunks])
        # 逐行讲解：把一组元素依次加入列表（target_tokens.extend([0] * len(context))）。
        target_tokens.extend([0] * len(context))
        # 逐行讲解：把一组元素依次加入列表（logprobs.extend([0.0] * len(context))）。
        logprobs.extend([0.0] * len(context))
        # 逐行讲解：把一组元素依次加入列表（action_mask.extend([0.0] * len(context))）。
        action_mask.extend([0.0] * len(context))

    # 逐行讲解：返回结果：{'id': row['id'], 'task': row['task'], 'label': row['label'], 'prediction': prediction, 'format_ok': finish…。
    return {
        "id": row["id"],
        "task": row["task"],
        "label": row["label"],
        "prediction": prediction,
        "format_ok": finish_reason == "answer",
        "finish_reason": finish_reason,
        "turns": turns,
        "initial_messages": row["messages"],
        "initial_images": row["images"],
        "model_input": prompt,
        "target_tokens": target_tokens,
        "logprobs": logprobs,
        "action_mask": action_mask,
    }
