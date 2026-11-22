r"""以完整交错思考轨迹异步训练 Qwen3.5 LoRA，默认使用 Qwen3.5-4B。

在 experiments/spec-o3/ 目录下运行：

# 首次运行前登录；已登录可跳过。SwanLab 云端记录需要 SwanLab 登录。
uv run trio login
uv run swanlab login

uv run python train_sft.py

uv run python train_sft.py \
    --base-model Qwen/Qwen3.5-4B \
    --epochs 2 \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --data-dir datasets/sft \
    --swanlab-mode online \

默认读取 datasets/sft/sft_train.jsonl 和 datasets/sft/sft_val.jsonl，
每个 epoch 在远程分别保存 state（继续训练）和 sampler（推理评测）两份权重。
两份权重的路径与验证 loss 打印到终端。
--base-model 同时选择 tokenizer、图像处理器和远程训练基模。
--model-revision 指定所选模型的 HF 文件版本，默认 main；模板沿用 Qwen3.5。
--project-name 指定 SwanLab 项目，--run-name 指定实验名称和远程权重名称前缀。
训练循环连续提交 batch，后台等待结果并记录 loss；每轮结束后验证并保存权重。
脚本使用已保存的登录信息，不读取 .env 文件。
--swanlab-mode local 或 disabled 无需登录 SwanLab。
"""

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：导入模块：random。
import random

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab
# 逐行讲解：从 transformers 导入 AutoImageProcessor, AutoTokenizer。
from transformers import AutoImageProcessor, AutoTokenizer

# 逐行讲解：从 protocol 导入 BASE_MODEL, build_sft_datum。
from protocol import BASE_MODEL, build_sft_datum


# 逐行讲解：定义函数 loss_totals：累计加权负对数似然和权重，用于计算参与监督位置的平均 loss。
def loss_totals(result, data):
    """累计加权负对数似然和权重，用于计算参与监督位置的平均 loss。"""
    # 逐行讲解：构造元组 (0.0, 0.0)，赋给 (nll, token_weight)。
    nll, token_weight = 0.0, 0.0

    # 逐行讲解：遍历 zip(result.loss_fn_outputs, data)，将每项绑定到 (output, datum)。
    for output, datum in zip(result.loss_fn_outputs, data):
        # 逐行讲解：调用 datum.loss_fn_inputs['weights'].to_numpy（datum.loss_fn_inputs['weights'].to_numpy()），赋给 weights。
        weights = datum.loss_fn_inputs["weights"].to_numpy()
        # 逐行讲解：更新 nll：nll -= float((output['logprobs'].to_numpy() * weights).sum())。
        nll -= float((output["logprobs"].to_numpy() * weights).sum())
        # 逐行讲解：更新 token_weight：token_weight += float(weights.sum())。
        token_weight += float(weights.sum())

    # 逐行讲解：返回结果：(nll, token_weight)。
    return nll, token_weight


# 逐行讲解：定义函数 build_training_batch：构造图文 Datum，并按整个 batch 的 assistant token 数归一化权重。
def build_training_batch(rows, tokenizer, processor):
    """构造图文 Datum，并按整个 batch 的 assistant token 数归一化权重。"""
    # 逐行讲解：通过推导式逐项生成结果：[build_sft_datum(row, tokenizer, processor) for row in rows]，赋给 data。
    data = [build_sft_datum(row, tokenizer, processor) for row in rows]
    # 逐行讲解：求和（sum((float(datum.loss_fn_inputs['weights'].to_numpy().sum()) for datum in data))），赋给 tokens。
    tokens = sum(
        float(datum.loss_fn_inputs["weights"].to_numpy().sum())
        for datum in data
    )

    # cross_entropy 在服务端求和；归一化后得到按监督 token 平均的梯度。
    for datum in data:
        # 逐行讲解：调用 trio.types.TensorData.from_numpy（trio.types.TensorData.from_numpy(datum.loss_fn_inputs['weights'].to_numpy() / tokens)），赋给 datum.loss_fn_inputs['weights']。
        datum.loss_fn_inputs["weights"] = trio.types.TensorData.from_numpy(
            datum.loss_fn_inputs["weights"].to_numpy() / tokens,
        )

    # 逐行讲解：返回结果：(data, tokens)。
    return data, tokens


# 逐行讲解：定义异步函数 log_training_step：后台等待计算与更新完成，并按提交顺序记录 loss。
async def log_training_step(backward, update, data, tokens, epoch, step, log_lock):
    """后台等待计算与更新完成，并按提交顺序记录 loss。"""
    # 日志任务依次进入这个锁，保证 SwanLab 的 step 递增。
    # 主循环提交后续 batch 不需要获取这个锁。
    async with log_lock:
        # 逐行讲解：等待异步操作 backward 完成并取得结果，赋给 result。
        result = await backward
        # 逐行讲解：等待异步操作 update 完成并取得结果。
        await update

        # 逐行讲解：调用 loss_totals（loss_totals(result, data)），赋给 (nll, weight)。
        nll, weight = loss_totals(result, data)
        # 逐行讲解：构造字典 {'train/loss': nll / weight, 'train/assistant_tokens': tokens, 'train/epoch': epoch}，赋给 metrics。
        metrics = {
            "train/loss": nll / weight,
            "train/assistant_tokens": tokens,
            "train/epoch": epoch,
        }
        # 逐行讲解：计算自然对数（swanlab.log(metrics, step=step)）。
        swanlab.log(metrics, step=step)
        # 逐行讲解：调用 print（print(f"epoch={epoch} step={step} loss={metrics['train/loss']:.4f} assistant_tokens={tokens:.0f}")）。
        print(
            f"epoch={epoch} step={step} "
            f"loss={metrics['train/loss']:.4f} assistant_tokens={tokens:.0f}"
        )


# 逐行讲解：定义异步函数 validation_loss：验证集只做前向计算，按所有监督 token 汇总平均 loss。
async def validation_loss(client, rows, tokenizer, processor, batch_size):
    """验证集只做前向计算，按所有监督 token 汇总平均 loss。"""
    # 逐行讲解：构造元组 (0.0, 0.0)，赋给 (nll, tokens)。
    nll, tokens = 0.0, 0.0

    # 逐行讲解：遍历 range(0, len(rows), batch_size)，将每项绑定到 start。
    for start in range(0, len(rows), batch_size):
        # 逐行讲解：通过推导式逐项生成结果：[build_sft_datum(row, tokenizer, processor) for row in rows[start:start + batch_size]]，赋给 data。
        data = [
            build_sft_datum(row, tokenizer, processor)
            for row in rows[start : start + batch_size]
        ]

        # 逐行讲解：等待异步操作 client.forward_async(data, loss_fn='cross_entropy') 完成并取得结果，赋给 future。
        future = await client.forward_async(data, loss_fn="cross_entropy")
        # 逐行讲解：调用 loss_totals（loss_totals(await future, data)），赋给 (batch_nll, batch_tokens)。
        batch_nll, batch_tokens = loss_totals(await future, data)

        # 逐行讲解：更新 nll：nll += batch_nll。
        nll += batch_nll
        # 逐行讲解：更新 tokens：tokens += batch_tokens。
        tokens += batch_tokens

    # 逐行讲解：返回结果：nll / tokens。
    return nll / tokens


# 逐行讲解：定义异步函数 main：封装下方逻辑，供调用方复用。
async def main(args):
    # 1. 读取完整多轮轨迹；每条样本的图片路径已经写在 JSONL 中。
    train = [
        json.loads(line)
        for line in (args.data_dir / "sft_train.jsonl").read_text().splitlines()
    ]
    # 逐行讲解：通过推导式逐项生成结果：[json.loads(line) for line in (args.data_dir / 'sft_val.jsonl').read_text().splitlines()]，赋给 val。
    val = [
        json.loads(line)
        for line in (args.data_dir / "sft_val.jsonl").read_text().splitlines()
    ]

    # 2. 文本和图像处理器使用同一个基模、同一个 HF 文件版本。
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.model_revision,
    )
    # 逐行讲解：调用 AutoImageProcessor.from_pretrained（AutoImageProcessor.from_pretrained(args.base_model, revision=args.model_revision, backend='pil')），赋给 processor。
    processor = AutoImageProcessor.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        backend="pil",
    )

    # 3. 初始化实验记录和远程 LoRA 训练任务。
    config = {
        **vars(args),
        "data_dir": str(args.data_dir),
    }
    # 逐行讲解：调用 swanlab.init（swanlab.init(project=args.project_name, experiment_name=args.run_name, config=config, mode=args.swanlab_mode)）。
    swanlab.init(
        project=args.project_name,
        experiment_name=args.run_name,
        config=config,
        mode=args.swanlab_mode,
    )

    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service。
    service = trio.ServiceClient()
    # 逐行讲解：等待异步操作 service.create_lora_training_client_async(base_model=args.base_model, rank=args.rank, seed=args.seed) 完成并取得结果，赋给 client。
    client = await service.create_lora_training_client_async(
        base_model=args.base_model,
        rank=args.rank,
        seed=args.seed,
    )
    # 逐行讲解：调用 trio.AdamParams（trio.AdamParams(learning_rate=args.learning_rate)），赋给 optimizer。
    optimizer = trio.AdamParams(learning_rate=args.learning_rate)

    # 逐行讲解：调用 random.Random（random.Random(args.seed)），赋给 rng。
    rng = random.Random(args.seed)
    # 逐行讲解：取值 0，赋给 step。
    step = 0

    # 4. 每个 epoch 依次执行训练、验证和保存。
    for epoch in range(1, args.epochs + 1):
        # 逐行讲解：打乱数据顺序（rng.shuffle(train)）。
        rng.shuffle(train)
        # 逐行讲解：构造列表 []，赋给 log_tasks。
        log_tasks = []
        # 逐行讲解：调用 asyncio.Lock（asyncio.Lock()），赋给 log_lock。
        log_lock = asyncio.Lock()

        # 逐行讲解：遍历 range(0, len(train), args.batch_size)，将每项绑定到 start。
        for start in range(0, len(train), args.batch_size):
            # 图片读取和 token 构造放到线程中，让事件循环继续处理远程结果。
            data, tokens = await asyncio.to_thread(
                build_training_batch,
                train[start : start + args.batch_size],
                tokenizer,
                processor,
            )

            # 两次调用都先取得 Future，远程按 FB → 更新 → 下一批 FB 的顺序执行。
            backward = await client.forward_backward_async(
                data,
                loss_fn="cross_entropy",
            )
            # 逐行讲解：等待异步操作 client.optim_step_async(optimizer) 完成并取得结果，赋给 update。
            update = await client.optim_step_async(optimizer)

            # 逐行讲解：更新 step：step += 1。
            step += 1
            # 逐行讲解：向列表末尾追加一个元素（log_tasks.append(asyncio.create_task(log_training_step(backward, update, data, tokens, epoch, step, log_loc…）。
            log_tasks.append(
                asyncio.create_task(
                    log_training_step(
                        backward,
                        update,
                        data,
                        tokens,
                        epoch,
                        step,
                        log_lock,
                    )
                )
            )

        # 等待这一轮所有更新和日志完成，再验证并保存同一轮的权重。
        await asyncio.gather(*log_tasks)
        # 逐行讲解：等待异步操作 validation_loss(client, val, tokenizer, processor, args.batch_size) 完成并取得结果，赋给 val_loss。
        val_loss = await validation_loss(
            client,
            val,
            tokenizer,
            processor,
            args.batch_size,
        )
        # 逐行讲解：计算自然对数（swanlab.log({'val/loss': val_loss}, step=step)）。
        swanlab.log({"val/loss": val_loss}, step=step)

        # state 用于继续训练，sampler 权重用于后续工具交互评测。
        state_future = await client.save_state_async(
            name=f"{args.run_name}-epoch-{epoch}-state",
        )
        # 逐行讲解：等待异步操作 client.save_weights_for_sampler_async(name=f'{args.run_name}-epoch-{epoch}-sampler') 完成并取得结果，赋给 sampler_future。
        sampler_future = await client.save_weights_for_sampler_async(
            name=f"{args.run_name}-epoch-{epoch}-sampler",
        )
        # 逐行讲解：等待异步操作 asyncio.gather(state_future, sampler_future) 完成并取得结果，赋给 (state, sampler)。
        state, sampler = await asyncio.gather(state_future, sampler_future)

        # 逐行讲解：调用 print（print(f'epoch={epoch} val_loss={val_loss:.4f}')）。
        print(f"epoch={epoch} val_loss={val_loss:.4f}")
        # 逐行讲解：调用 print（print(f'state_path（继续训练）：{state.path}')）。
        print(f"state_path（继续训练）：{state.path}")
        # 逐行讲解：调用 print（print(f'sampler_path（推理评测）：{sampler.path}')）。
        print(f"sampler_path（推理评测）：{sampler.path}")

    # 逐行讲解：调用 swanlab.finish（swanlab.finish()）。
    swanlab.finish()


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)），赋给 parser。
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 模型和数据路径。
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="用于训练的 Qwen3.5 基座模型名称",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--model-revision', default='main', help='所选模型的 Hugging Face 文件版本（commit、tag 或分支）')）。
    parser.add_argument(
        "--model-revision",
        default="main",
        help="所选模型的 Hugging Face 文件版本（commit、tag 或分支）",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'datasets/sft')）。
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "datasets/sft",
    )

    # 训练超参数。
    parser.add_argument("--epochs", type=int, default=1)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=4)）。
    parser.add_argument("--batch-size", type=int, default=4)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--rank', type=int, default=32)）。
    parser.add_argument("--rank", type=int, default=32)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=2e-05)）。
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)

    # 实验记录。
    parser.add_argument(
        "--project-name", type=str, default="agentic-rl-lab-spec-o3"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--run-name', type=str, default='spec-o3-sft')）。
    parser.add_argument("--run-name", type=str, default="spec-o3-sft")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=['online', 'local', 'disabled'], default='online')）。
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "disabled"],
        default="online",
    )

    # 逐行讲解：调用 asyncio.run（asyncio.run(main(parser.parse_args()))）。
    asyncio.run(main(parser.parse_args()))
