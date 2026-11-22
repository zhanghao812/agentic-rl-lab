"""使用 PyTRIO 在 GeoQA 上运行多模态 GRPO。

准备数据（仓库根目录）：
uv run python download-dataset.py

小规模测试：（5 元人民币）
uv run python train.py \
    --steps 20 \
    --batch-size 8 \
    --group-size 8 \
    --max-tokens 1024 \
    --save-every 10 \
    --swanlab-mode online
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：io。
import io
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：导入模块：time。
import time
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 importlib.metadata 导入 version。
from importlib.metadata import version
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：导入模块：numpy，别名 np。
import numpy as np
# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab
# 逐行讲解：从 datasets 导入 Dataset, load_dataset。
from datasets import Dataset, load_dataset
# 逐行讲解：从 PIL 导入 Image。
from PIL import Image
# 逐行讲解：从 tqdm.asyncio 导入 tqdm_asyncio。
from tqdm.asyncio import tqdm_asyncio
# 逐行讲解：从 transformers 导入 AutoImageProcessor。
from transformers import AutoImageProcessor

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets'，赋给 DEFAULT_DATASET_DIR。
DEFAULT_DATASET_DIR = SCRIPT_DIR / "datasets"
# 逐行讲解：取值 'Qwen/Qwen3.5-4B'，赋给 DEFAULT_MODEL。
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
# 逐行讲解：取值 '<|image_pad|>'，赋给 IMAGE_PAD_TOKEN。
IMAGE_PAD_TOKEN = "<|image_pad|>"
# 逐行讲解：取值 'ABCD'，赋给 CHOICE_LETTERS。
CHOICE_LETTERS = "ABCD"
# 逐行讲解：调用 re.compile（re.compile('\\\\boxed\\s*\\{\\s*([A-D])\\s*\\}', re.IGNORECASE)），赋给 BOXED_CHOICE_PATTERN。
BOXED_CHOICE_PATTERN = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)


# 逐行讲解：定义类 RolloutSample：封装下方逻辑，供调用方复用。
@dataclass(frozen=True)
class RolloutSample:
    # 逐行讲解：声明 tokens 的类型为 list[int]。
    tokens: list[int]
    # 逐行讲解：声明 logprobs 的类型为 list[float]。
    logprobs: list[float]
    # 逐行讲解：声明 text 的类型为 str。
    text: str
    # 逐行讲解：声明 predicted_choice 的类型为 str | None。
    predicted_choice: str | None
    # 逐行讲解：声明 reward 的类型为 float。
    reward: float
    # 逐行讲解：声明 advantage 的类型为 float。
    advantage: float


# 逐行讲解：定义类 RolloutGroup：封装下方逻辑，供调用方复用。
@dataclass(frozen=True)
class RolloutGroup:
    # 逐行讲解：声明 prompt_chunks 的类型为 list[Any]。
    prompt_chunks: list[Any]
    # 逐行讲解：声明 prompt_length 的类型为 int。
    prompt_length: int
    # 逐行讲解：声明 samples 的类型为 list[RolloutSample]。
    samples: list[RolloutSample]


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='GeoQA 多模态 GRPO')），赋给 parser。
    parser = argparse.ArgumentParser(description="GeoQA 多模态 GRPO")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)）。
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default=DEFAULT_MODEL)）。
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=32)）。
    parser.add_argument("--lora-rank", type=int, default=32)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--steps', type=int, default=10)）。
    parser.add_argument("--steps", type=int, default=10)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=4)）。
    parser.add_argument("--batch-size", type=int, default=4)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--group-size', type=int, default=4)）。
    parser.add_argument("--group-size", type=int, default=4)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-samples', type=int, default=0, help='0 表示使用全部训练集')）。
    parser.add_argument(
        "--max-samples", type=int, default=0, help="0 表示使用全部训练集"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-tokens', type=int, default=64)）。
    parser.add_argument("--max-tokens", type=int, default=64)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--temperature', type=float, default=1.0)）。
    parser.add_argument("--temperature", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--top-p', type=float, default=1.0)）。
    parser.add_argument("--top-p", type=float, default=1.0)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=4e-05)）。
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=('online', 'local', 'offline', 'disabled'), default='online')）。
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
        default="online",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-vision-grpo')）。
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-vision-grpo")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--experiment-name', default='vision-grpo-qwen35-4b-geoqa')）。
    parser.add_argument(
        "--experiment-name",
        default="vision-grpo-qwen35-4b-geoqa",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--weights-name', default='vision-grpo-qwen35-4b-geoqa')）。
    parser.add_argument(
        "--weights-name",
        default="vision-grpo-qwen35-4b-geoqa",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every', type=int, default=10, help='每隔多少个 step 保存一次，0 表示只保存最终 checkpoint')）。
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="每隔多少个 step 保存一次，0 表示只保存最终 checkpoint",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-weights', action=argparse.BooleanOptionalAction, default=True)）。
    parser.add_argument(
        "--save-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--show-samples', action='store_true')）。
    parser.add_argument("--show-samples", action="store_true")
    # 逐行讲解：返回结果：parser.parse_args()。
    return parser.parse_args()


# 逐行讲解：定义函数 load_geoqa_train：读取并打乱训练数据。
def load_geoqa_train(dataset_dir: Path, seed: int, max_samples: int) -> Dataset:
    """读取并打乱训练数据。"""
    # 逐行讲解：打乱数据顺序（load_dataset('parquet', data_files=str(dataset_dir / 'train.parquet'), split='train').shuffle(seed=seed)），赋给 dataset。
    dataset = load_dataset(
        "parquet",
        data_files=str(dataset_dir / "train.parquet"),
        split="train",
    ).shuffle(seed=seed)
    # 逐行讲解：检查条件 max_samples > 0；成立时执行下方分支。
    if max_samples > 0:
        # 逐行讲解：按索引选择数据行（dataset.select(range(min(max_samples, len(dataset))))），赋给 dataset。
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    # 逐行讲解：返回结果：dataset。
    return dataset


# 逐行讲解：定义函数 pick_batch：按 step 顺序取 batch，走完数据后从头继续。
def pick_batch(dataset: Dataset, step: int, batch_size: int) -> Dataset:
    """按 step 顺序取 batch，走完数据后从头继续。"""
    # 逐行讲解：计算 step * batch_size，赋给 start。
    start = step * batch_size
    # 逐行讲解：通过推导式逐项生成结果：[(start + offset) % len(dataset) for offset in range(batch_size)]，赋给 indices。
    indices = [(start + offset) % len(dataset) for offset in range(batch_size)]
    # 逐行讲解：返回结果：dataset.select(indices)。
    return dataset.select(indices)


# 逐行讲解：定义函数 encode_image：将图片编码成 PyTRIO chunk，并计算视觉 token 数。
def encode_image(image: Image.Image, image_processor: Any) -> trio.ImageChunk:
    """将图片编码成 PyTRIO chunk，并计算视觉 token 数。"""
    # 逐行讲解：按条件选择结果：'jpeg' if image.format in {'JPG', 'JPEG'} else 'png'，赋给 chunk_format。
    chunk_format = "jpeg" if image.format in {"JPG", "JPEG"} else "png"
    # 逐行讲解：调用 image.convert（image.convert('RGBA')），赋给 rgba。
    rgba = image.convert("RGBA")
    # 逐行讲解：调用 Image.new（Image.new('RGBA', rgba.size, (255, 255, 255, 255))），赋给 background。
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    # 逐行讲解：调用 Image.alpha_composite(background, rgba).convert（Image.alpha_composite(background, rgba).convert('RGB')），赋给 image。
    image = Image.alpha_composite(background, rgba).convert("RGB")

    # 逐行讲解：调用 io.BytesIO（io.BytesIO()），赋给 buffer。
    buffer = io.BytesIO()
    # 逐行讲解：调用 image.save（image.save(buffer, format=chunk_format.upper())）。
    image.save(buffer, format=chunk_format.upper())
    # 逐行讲解：调用 image_processor.get_number_of_image_patches（image_processor.get_number_of_image_patches(image.height, image.width, images_kwargs={})），赋给 patches。
    patches = image_processor.get_number_of_image_patches(
        image.height,
        image.width,
        images_kwargs={},
    )
    # 逐行讲解：计算 patches // int(image_processor.merge_size) ** 2，赋给 expected_tokens。
    expected_tokens = patches // int(image_processor.merge_size) ** 2
    # 逐行讲解：返回结果：trio.ImageChunk(data=buffer.getvalue(), format=chunk_format, expected_tokens=expected_tokens)。
    return trio.ImageChunk(
        data=buffer.getvalue(),
        format=chunk_format,
        expected_tokens=expected_tokens,
    )


# 逐行讲解：定义函数 format_question：将题目和四个选项整理成模型指令。
def format_question(subject: str, choices: list[str]) -> str:
    """将题目和四个选项整理成模型指令。"""
    # 逐行讲解：调用 '\n'.join（'\n'.join((f'{letter}. {choice}' for letter, choice in zip(CHOICE_LETTERS, choices, strict=True)))），赋给 choice_lines。
    choice_lines = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(CHOICE_LETTERS, choices, strict=True)
    )
    # 逐行讲解：返回结果：f'请根据图片解答下面的几何选择题。\n题目：{subject.strip()}\n选项：\n{choice_lines}\n请先进行简单逻辑推理思考，再给出最终答案。最终选项格式必须是 \\boxed{{A}}、…。
    return (
        "请根据图片解答下面的几何选择题。\n"
        f"题目：{subject.strip()}\n"
        f"选项：\n{choice_lines}\n"
        "请先进行简单逻辑推理思考，再给出最终答案。"
        "最终选项格式必须是 \\boxed{A}、\\boxed{B}、\\boxed{C} 或 \\boxed{D}。"
    )


# 逐行讲解：定义函数 build_prompt_chunks：先用 chat template 格式化 messages，再拆成图文 chunks。
def build_prompt_chunks(
    tokenizer: Any,
    image_processor: Any,
    image: Image.Image,
    subject: str,
    choices: list[str],
) -> list[Any]:
    """先用 chat template 格式化 messages，再拆成图文 chunks。"""
    # 逐行讲解：构造列表 [{'role': 'user', 'content': [{'type': 'text', 'text': format_question(subject, choices)}, {'type': 'image'…，赋给 messages。
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": format_question(subject, choices)},
                {"type": "image", "image": "geoqa"},
            ],
        }
    ]
    # 逐行讲解：按模型聊天模板组织消息（tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)），赋给 prompt。
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # 逐行讲解：调用 prompt.split（prompt.split(IMAGE_PAD_TOKEN)），赋给 (before_image, after_image)。
    before_image, after_image = prompt.split(IMAGE_PAD_TOKEN)
    # 逐行讲解：返回结果：[trio.types.EncodedTextChunk(tokens=tokenizer.encode(before_image, add_special_tokens=False)), encode_image…。
    return [
        trio.types.EncodedTextChunk(
            tokens=tokenizer.encode(before_image, add_special_tokens=False)
        ),
        encode_image(image, image_processor),
        trio.types.EncodedTextChunk(
            tokens=tokenizer.encode(after_image, add_special_tokens=False)
        ),
    ]


# 逐行讲解：定义函数 extract_choice：提取回答中最后一个 boxed 选项。
def extract_choice(text: str) -> str | None:
    """提取回答中最后一个 boxed 选项。"""
    # 逐行讲解：调用 BOXED_CHOICE_PATTERN.findall（BOXED_CHOICE_PATTERN.findall(text)），赋给 matches。
    matches = BOXED_CHOICE_PATTERN.findall(text)
    # 逐行讲解：返回结果：matches[-1].upper() if matches else None。
    return matches[-1].upper() if matches else None


# 逐行讲解：定义异步函数 run_rollout_group：异步采样同一道题的一组回答，并计算组内 advantage。
async def run_rollout_group(
    sampling_client: Any,
    tokenizer: Any,
    prompt_chunks: list[Any],
    gold_choice: str,
    sampling_params: trio.SamplingParams,
    group_size: int,
) -> RolloutGroup:
    """异步采样同一道题的一组回答，并计算组内 advantage。"""
    # 逐行讲解：调用 trio.ModelInput（trio.ModelInput(chunks=prompt_chunks)），赋给 prompt。
    prompt = trio.ModelInput(chunks=prompt_chunks)
    # 逐行讲解：调用 len（len(prompt)），赋给 prompt_length。
    prompt_length = len(prompt)
    # 逐行讲解：等待异步操作 sampling_client.sample_async(prompt=prompt, num_samples=group_size, sampling_params=sampling_params, return… 完成并取得结果，赋给 response。
    response = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=group_size,
        sampling_params=sampling_params,
        return_text=True,
    )
    # 逐行讲解：检查条件 response.input_tokens != prompt_length；成立时执行下方分支。
    if response.input_tokens != prompt_length:
        # 逐行讲解：抛出异常：ValueError(f'图文 prompt 长度不一致：local={prompt_length}, remote={response.input_tokens}')。
        raise ValueError(
            f"图文 prompt 长度不一致：local={prompt_length}, remote={response.input_tokens}"
        )

    # 逐行讲解：声明 raw_samples 的类型为 list[tuple[list[int], list[float], str, str | None, float]]；构造列表 []。
    raw_samples: list[tuple[list[int], list[float], str, str | None, float]] = []
    # 逐行讲解：声明 rewards 的类型为 list[float]；构造列表 []。
    rewards: list[float] = []
    # 逐行讲解：遍历 response.sequences，将每项绑定到 sequence。
    for sequence in response.sequences:
        # 逐行讲解：调用 list（list(sequence.tokens)），赋给 tokens。
        tokens = list(sequence.tokens)
        # 逐行讲解：通过推导式逐项生成结果：[float(value) for value in sequence.logprobs]，赋给 logprobs。
        logprobs = [float(value) for value in sequence.logprobs]
        # 逐行讲解：检查条件 len(tokens) != len(logprobs)；成立时执行下方分支。
        if len(tokens) != len(logprobs):
            # 逐行讲解：抛出异常：ValueError('生成 token 与 logprob 长度不一致')。
            raise ValueError("生成 token 与 logprob 长度不一致")
        # 逐行讲解：取值 sequence.text or tokenizer.decode(tokens, skip_special_tokens=True)，赋给 text。
        text = sequence.text or tokenizer.decode(tokens, skip_special_tokens=True)
        # 逐行讲解：调用 extract_choice（extract_choice(text)），赋给 predicted_choice。
        predicted_choice = extract_choice(text)
        # 逐行讲解：转换为浮点表示（float(predicted_choice == gold_choice)），赋给 reward。
        reward = float(predicted_choice == gold_choice)
        # 逐行讲解：向列表末尾追加一个元素（rewards.append(reward)）。
        rewards.append(reward)
        # 逐行讲解：向列表末尾追加一个元素（raw_samples.append((tokens, logprobs, text, predicted_choice, reward))）。
        raw_samples.append((tokens, logprobs, text, predicted_choice, reward))

    # 逐行讲解：计算 sum(rewards) / len(rewards)，赋给 mean_reward。
    mean_reward = sum(rewards) / len(rewards)
    # 逐行讲解：通过推导式逐项生成结果：[RolloutSample(tokens=tokens, logprobs=logprobs, text=text, predicted_choice=predicted_choice, reward=rewar…，赋给 samples。
    samples = [
        RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            text=text,
            predicted_choice=predicted_choice,
            reward=reward,
            advantage=reward - mean_reward,
        )
        for tokens, logprobs, text, predicted_choice, reward in raw_samples
    ]
    # 逐行讲解：返回结果：RolloutGroup(prompt_chunks, prompt_length, samples)。
    return RolloutGroup(prompt_chunks, prompt_length, samples)


# 逐行讲解：定义函数 build_grpo_datum：把图文 prompt chunk 和 completion 拼成 GRPO Datum。
def build_grpo_datum(group: RolloutGroup, sample: RolloutSample) -> trio.Datum:
    """把图文 prompt chunk 和 completion 拼成 GRPO Datum。"""
    # 逐行讲解：调用 trio.ModelInput（trio.ModelInput(chunks=[*group.prompt_chunks, trio.types.EncodedTextChunk(tokens=sample.tokens[:-1])])），赋给 model_input。
    model_input = trio.ModelInput(
        chunks=[
            *group.prompt_chunks,
            trio.types.EncodedTextChunk(tokens=sample.tokens[:-1]),
        ]
    )
    # 逐行讲解：计算 group.prompt_length - 1，赋给 observation_length。
    observation_length = group.prompt_length - 1
    # 逐行讲解：返回结果：trio.Datum(model_input=model_input, loss_fn_inputs={'target_tokens': np.asarray([0] * observation_length + …。
    return trio.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": np.asarray(
                [0] * observation_length + sample.tokens,
                dtype=np.int64,
            ),
            "logprobs": np.asarray(
                [0.0] * observation_length + sample.logprobs,
                dtype=np.float32,
            ),
            "advantages": np.asarray(
                [0.0] * observation_length + [sample.advantage] * len(sample.tokens),
                dtype=np.float32,
            ),
        },
    )


# 逐行讲解：定义函数 init_swanlab：初始化训练日志。
def init_swanlab(args: argparse.Namespace, dataset_size: int) -> Any:
    """初始化训练日志。"""
    # 逐行讲解：返回结果：swanlab.init(mode=args.swanlab_mode, project=args.swanlab_project, experiment_name=args.experiment_name, co…。
    return swanlab.init(
        mode=args.swanlab_mode,
        project=args.swanlab_project,
        experiment_name=args.experiment_name,
        config={
            "algorithm": "vision-grpo",
            "dataset": "hz2475/geoQA",
            "dataset_size": dataset_size,
            "base_model": args.base_model,
            "pytrio_version": version("pytrio"),
            "enable_thinking": False,
            "lora_rank": args.lora_rank,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "save_every": args.save_every,
        },
    )


# 逐行讲解：定义异步函数 save_checkpoint：同时保存推理权重和完整训练状态。
async def save_checkpoint(
    training_client: trio.TrainingClient,
    weights_name: str,
    step: int,
) -> None:
    """同时保存推理权重和完整训练状态。"""
    # 逐行讲解：取值 f'{weights_name}-step-{step}'，赋给 prefix。
    prefix = f"{weights_name}-step-{step}"
    # 逐行讲解：等待异步操作 training_client.save_weights_for_sampler_async(name=f'{prefix}-sampler') 完成并取得结果，赋给 sampler_future。
    sampler_future = await training_client.save_weights_for_sampler_async(
        name=f"{prefix}-sampler"
    )
    # 逐行讲解：等待异步操作 training_client.save_state_async(name=f'{prefix}-state') 完成并取得结果，赋给 state_future。
    state_future = await training_client.save_state_async(name=f"{prefix}-state")
    # 逐行讲解：等待异步操作 asyncio.gather(sampler_future, state_future) 完成并取得结果，赋给 (sampler_weights, training_state)。
    sampler_weights, training_state = await asyncio.gather(
        sampler_future,
        state_future,
    )
    # 逐行讲解：调用 print（print(f'Sampler 权重：{sampler_weights.path}')）。
    print(f"Sampler 权重：{sampler_weights.path}")
    # 逐行讲解：调用 print（print(f'State 权重：{training_state.path}')）。
    print(f"State 权重：{training_state.path}")


# 逐行讲解：定义异步函数 main：封装下方逻辑，供调用方复用。
async def main(args: argparse.Namespace) -> None:
    # 逐行讲解：调用 load_geoqa_train（load_geoqa_train(args.dataset_dir.expanduser().resolve(), args.seed, args.max_samples)），赋给 train_data。
    train_data = load_geoqa_train(
        args.dataset_dir.expanduser().resolve(),
        args.seed,
        args.max_samples,
    )
    # 逐行讲解：调用 print（print(f'加载 GeoQA train 数据：{len(train_data)} 条')）。
    print(f"加载 GeoQA train 数据：{len(train_data)} 条")
    # 逐行讲解：调用 print（print(f"PyTRIO：{version('pytrio')}")）。
    print(f"PyTRIO：{version('pytrio')}")

    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：等待异步操作 service_client.create_lora_training_client_async(base_model=args.base_model, rank=args.lora_rank, seed=args… 完成并取得结果，赋给 training_client。
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    # 逐行讲解：调用 training_client.get_tokenizer（training_client.get_tokenizer()），赋给 tokenizer。
    tokenizer = training_client.get_tokenizer()
    # 逐行讲解：调用 AutoImageProcessor.from_pretrained（AutoImageProcessor.from_pretrained(args.base_model, backend='pil')），赋给 image_processor。
    image_processor = AutoImageProcessor.from_pretrained(
        args.base_model,
        backend="pil",
    )
    # 逐行讲解：调用 trio.SamplingParams（trio.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, stop='<|im_…），赋给 sampling_params。
    sampling_params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop="<|im_end|>",
    )
    # 逐行讲解：调用 trio.AdamParams（trio.AdamParams(learning_rate=args.learning_rate)），赋给 adam_params。
    adam_params = trio.AdamParams(learning_rate=args.learning_rate)
    # 逐行讲解：调用 init_swanlab（init_swanlab(args, len(train_data))），赋给 swanlab_run。
    swanlab_run = init_swanlab(args, len(train_data))
    # 逐行讲解：取值 0，赋给 last_saved_step。
    last_saved_step = 0

    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：遍历 range(args.steps)，将每项绑定到 step。
        for step in range(args.steps):
            # 逐行讲解：调用 list（list(pick_batch(train_data, step, args.batch_size))），赋给 batch_rows。
            batch_rows = list(pick_batch(train_data, step, args.batch_size))
            # 逐行讲解：等待异步操作 training_client.save_weights_and_get_sampling_client_async() 完成并取得结果，赋给 sampling_client。
            sampling_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )
            # 逐行讲解：声明 datums 的类型为 list[trio.Datum]；构造列表 []。
            datums: list[trio.Datum] = []
            # 逐行讲解：声明 all_samples 的类型为 list[RolloutSample]；构造列表 []。
            all_samples: list[RolloutSample] = []
            # 逐行讲解：声明 prompt_rewards 的类型为 list[float]；构造列表 []。
            prompt_rewards: list[float] = []
            # 逐行讲解：取值 0，赋给 degenerate_groups。
            degenerate_groups = 0

            # 一个 step 内的不同题目使用同一版 sampler 并发 rollout。
            rollout_groups = await tqdm_asyncio.gather(
                *(
                    run_rollout_group(
                        sampling_client,
                        tokenizer,
                        build_prompt_chunks(
                            tokenizer,
                            image_processor,
                            row["image"],
                            str(row["subject"]),
                            [str(choice) for choice in row["choices"]],
                        ),
                        CHOICE_LETTERS[int(row["label"])],
                        sampling_params,
                        args.group_size,
                    )
                    for row in batch_rows
                ),
                desc=f"Step {step + 1}/{args.steps} rollout",
                unit="题",
            )

            # 逐行讲解：遍历 zip(batch_rows, rollout_groups, strict=True)，将每项绑定到 (row, group)。
            for row, group in zip(batch_rows, rollout_groups, strict=True):
                # 逐行讲解：按索引或切片读取 CHOICE_LETTERS[int(row['label'])]，赋给 gold_choice。
                gold_choice = CHOICE_LETTERS[int(row["label"])]
                # 逐行讲解：把一组元素依次加入列表（all_samples.extend(group.samples)）。
                all_samples.extend(group.samples)
                # 逐行讲解：通过推导式逐项生成结果：[sample.reward for sample in group.samples]，赋给 rewards。
                rewards = [sample.reward for sample in group.samples]
                # 逐行讲解：向列表末尾追加一个元素（prompt_rewards.append(sum(rewards) / len(rewards))）。
                prompt_rewards.append(sum(rewards) / len(rewards))

                # 逐行讲解：检查条件 args.show_samples；成立时执行下方分支。
                if args.show_samples:
                    # 逐行讲解：调用 print（print(f"\nGeoQA id={row['id']} gold={gold_choice}")）。
                    print(f"\nGeoQA id={row['id']} gold={gold_choice}")
                    # 逐行讲解：遍历 enumerate(group.samples)，将每项绑定到 (index, sample)。
                    for index, sample in enumerate(group.samples):
                        # 逐行讲解：调用 print（print(f'  sample={index} predicted={sample.predicted_choice} reward={sample.reward:.0f} text={sample.text!r…）。
                        print(
                            f"  sample={index} predicted={sample.predicted_choice} "
                            f"reward={sample.reward:.0f} text={sample.text!r}"
                        )

                # 整组 reward 相同时没有相对优势，不参与更新。
                if len(set(rewards)) == 1:
                    # 逐行讲解：更新 degenerate_groups：degenerate_groups += 1。
                    degenerate_groups += 1
                    # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                    continue
                # 逐行讲解：把一组元素依次加入列表（datums.extend((build_grpo_datum(group, sample) for sample in group.samples if sample.tokens))）。
                datums.extend(
                    build_grpo_datum(group, sample)
                    for sample in group.samples
                    if sample.tokens
                )

            # 逐行讲解：计算 sum((len(sample.tokens) for sample in all_samples)) / len(all_samples)，赋给 mean_output_tokens。
            mean_output_tokens = sum(
                len(sample.tokens) for sample in all_samples
            ) / len(all_samples)
            # 逐行讲解：调用 tqdm_asyncio.write（tqdm_asyncio.write(f'本 batch 平均输出长度：{mean_output_tokens:.1f} tokens')）。
            tqdm_asyncio.write(
                f"本 batch 平均输出长度：{mean_output_tokens:.1f} tokens"
            )

            # 逐行讲解：声明 trainer_metrics 的类型为 dict[str, float]；构造字典 {}。
            trainer_metrics: dict[str, float] = {}
            # 逐行讲解：检查条件 datums；成立时执行下方分支。
            if datums:
                # 逐行讲解：等待异步操作 training_client.forward_backward_async(datums, loss_fn='importance_sampling') 完成并取得结果，赋给 forward_backward。
                forward_backward = await training_client.forward_backward_async(
                    datums,
                    loss_fn="importance_sampling",
                )
                # 逐行讲解：等待异步操作 training_client.optim_step_async(adam_params) 完成并取得结果，赋给 optim_step。
                optim_step = await training_client.optim_step_async(adam_params)
                # 逐行讲解：等待异步操作 forward_backward 完成并取得结果，赋给 result。
                result = await forward_backward
                # 逐行讲解：等待异步操作 optim_step 完成并取得结果。
                await optim_step
                # 逐行讲解：通过推导式逐项生成结果：{key: float(value) for key, value in result.metrics.items()}，赋给 trainer_metrics。
                trainer_metrics = {
                    key: float(value) for key, value in result.metrics.items()
                }

            # 逐行讲解：计算 sum(prompt_rewards) / len(prompt_rewards)，赋给 mean_reward。
            mean_reward = sum(prompt_rewards) / len(prompt_rewards)
            # 逐行讲解：计算 sum((sample.predicted_choice is not None for sample in all_samples)) / len(all_samples)，赋给 format_rate。
            format_rate = sum(
                sample.predicted_choice is not None for sample in all_samples
            ) / len(all_samples)
            # 逐行讲解：计算 degenerate_groups / len(prompt_rewards)，赋给 degenerate_fraction。
            degenerate_fraction = degenerate_groups / len(prompt_rewards)
            # 逐行讲解：构造字典 {'reward': mean_reward, 'format_rate': format_rate, 'degenerate_fraction': degenerate_fraction, 'train_datu…，赋给 metrics。
            metrics = {
                "reward": mean_reward,
                "format_rate": format_rate,
                "degenerate_fraction": degenerate_fraction,
                "train_datums": len(datums),
                "rollout/completion_tokens_mean": mean_output_tokens,
                **{f"trainer/{key}": value for key, value in trainer_metrics.items()},
            }
            # 逐行讲解：计算自然对数（swanlab.log(metrics, step=step)）。
            swanlab.log(metrics, step=step)

            # 逐行讲解：查询指定键，缺失时使用默认值（trainer_metrics.get('loss_mean')），赋给 loss_mean。
            loss_mean = trainer_metrics.get("loss_mean")
            # 逐行讲解：按条件选择结果：'n/a' if loss_mean is None else f'{loss_mean:.4f}'，赋给 loss_text。
            loss_text = "n/a" if loss_mean is None else f"{loss_mean:.4f}"
            # 逐行讲解：调用 print（print(f'Step {step + 1}/{args.steps} | reward={mean_reward:.3f} | format={format_rate:.1%} | degenerate={de…）。
            print(
                f"Step {step + 1}/{args.steps} | reward={mean_reward:.3f} | "
                f"format={format_rate:.1%} | degenerate={degenerate_fraction:.1%} | "
                f"datums={len(datums)} | loss_mean={loss_text}",
                flush=True,
            )

            # 逐行讲解：计算 step + 1，赋给 current_step。
            current_step = step + 1
            # 逐行讲解：检查条件 args.save_weights and args.save_every > 0 and (current_step % args.save_every == 0)；成立时执行下方分支。
            if (
                args.save_weights
                and args.save_every > 0
                and current_step % args.save_every == 0
            ):
                # 逐行讲解：等待异步操作 save_checkpoint(training_client, args.weights_name, current_step) 完成并取得结果。
                await save_checkpoint(training_client, args.weights_name, current_step)
                # 逐行讲解：取值 current_step，赋给 last_saved_step。
                last_saved_step = current_step

        # 逐行讲解：检查条件 args.save_weights and last_saved_step != args.steps；成立时执行下方分支。
        if args.save_weights and last_saved_step != args.steps:
            # 逐行讲解：等待异步操作 save_checkpoint(training_client, args.weights_name, args.steps) 完成并取得结果。
            await save_checkpoint(training_client, args.weights_name, args.steps)
    finally:
        # 逐行讲解：调用 swanlab_run.finish（swanlab_run.finish()）。
        swanlab_run.finish()


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 time.perf_counter（time.perf_counter()），赋给 start_time。
    start_time = time.perf_counter()
    # 逐行讲解：调用 asyncio.run（asyncio.run(main(parse_args()))）。
    asyncio.run(main(parse_args()))
    # 逐行讲解：调用 print（print(f'训练耗时：{time.perf_counter() - start_time:.2f}s')）。
    print(f"训练耗时：{time.perf_counter() - start_time:.2f}s")
