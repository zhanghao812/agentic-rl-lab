"""判别决策模型的异步 LoRA masked SFT 训练：只在 marker 前一位置监督答案符号。

准备数据（仓库根目录）：
uv run python 00-prepare-data.py

小规模测试：
uv run python train.py \
    --max-samples 128 \
    --batch-size 16 \
    --epochs 1 \
    --save-every 0 \
    --swanlab-mode disabled

正式训练：
uv run python train.py \
    --epochs 2 \
    --batch-size 16 \
    --save-every 100 \
    --swanlab-mode online


换基座模型（experiment/weights 名字自动跟随）：
uv run python train.py --base-model Qwen/Qwen3.5-9B --epochs 2
"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：random。
import random
# 逐行讲解：导入模块：time。
import time
# 逐行讲解：从 importlib.metadata 导入 version。
from importlib.metadata import version
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab

# 逐行讲解：从 data 导入 Encoder, load_jsonl。
from data import Encoder, load_jsonl

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 SCRIPT_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent
# 逐行讲解：计算 SCRIPT_DIR / 'datasets' / 'train.jsonl'，赋给 DEFAULT_DATA。
DEFAULT_DATA = SCRIPT_DIR / "datasets" / "train.jsonl"
# 逐行讲解：取值 'Qwen/Qwen3.5-4B'，赋给 DEFAULT_MODEL。
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args() -> argparse.Namespace:
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description='异步 LoRA masked SFT 训练')），赋给 parser。
    parser = argparse.ArgumentParser(description="异步 LoRA masked SFT 训练")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--data', type=Path, default=DEFAULT_DATA)）。
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default=DEFAULT_MODEL)）。
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--lora-rank', type=int, default=32)）。
    parser.add_argument("--lora-rank", type=int, default=32)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--epochs', type=int, default=2)）。
    parser.add_argument("--epochs", type=int, default=2)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=16)）。
    parser.add_argument("--batch-size", type=int, default=16)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--learning-rate', type=float, default=0.0001)）。
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--max-samples', type=int, default=0, help='0 表示使用全部训练集')）。
    parser.add_argument(
        "--max-samples", type=int, default=0, help="0 表示使用全部训练集"
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--save-every', type=int, default=60, help='每隔多少个 step 保存一次推理权重；每个 epoch 结束总会保存，0 表示不做 …）。
    parser.add_argument(
        "--save-every",
        type=int,
        default=60,
        help="每隔多少个 step 保存一次推理权重；每个 epoch 结束总会保存，0 表示不做 step 级保存",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=('online', 'local', 'offline', 'disabled'), default='online')）。
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
        default="online",
    )
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-project', default='agentic-rl-lab-pytrio-jev')）。
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-pytrio-jev")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--experiment-name', default=None, help='默认跟随 --base-model')）。
    parser.add_argument("--experiment-name", default=None, help="默认跟随 --base-model")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--weights-name', default=None, help='默认跟随 --base-model')）。
    parser.add_argument("--weights-name", default=None, help="默认跟随 --base-model")
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()
    # 逐行讲解：调用 args.base_model.split('/')[-1].lower().replace（args.base_model.split('/')[-1].lower().replace('.', '')），赋给 model_slug。
    model_slug = args.base_model.split("/")[-1].lower().replace(".", "")
    # 逐行讲解：检查条件 args.experiment_name is None；成立时执行下方分支。
    if args.experiment_name is None:
        # 逐行讲解：取值 f'pytrio-jev-{model_slug}'，赋给 args.experiment_name。
        args.experiment_name = f"pytrio-jev-{model_slug}"
    # 逐行讲解：检查条件 args.weights_name is None；成立时执行下方分支。
    if args.weights_name is None:
        # 逐行讲解：取值 f'pytrio-jev-{model_slug}'，赋给 args.weights_name。
        args.weights_name = f"pytrio-jev-{model_slug}"
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：定义异步函数 save_weights：封装下方逻辑，供调用方复用。
async def save_weights(
    training_client: trio.TrainingClient, weights_name: str, step: int
) -> str:
    # 逐行讲解：取值 f'{weights_name}-step-{step}'，赋给 name。
    name = f"{weights_name}-step-{step}"
    # 逐行讲解：等待异步操作 training_client.save_weights_for_sampler_async(name=name) 完成并取得结果，赋给 future。
    future = await training_client.save_weights_for_sampler_async(name=name)
    # 逐行讲解：取值 (await future).path，赋给 path。
    path = (await future).path
    # 逐行讲解：调用 print（print(f'保存权重 {name}: {path}')）。
    print(f"保存权重 {name}: {path}")
    # 逐行讲解：返回结果：path。
    return path


# 逐行讲解：定义异步函数 main：封装下方逻辑，供调用方复用。
async def main(args: argparse.Namespace) -> None:
    # 逐行讲解：调用 load_jsonl（load_jsonl(args.data)），赋给 records。
    records = load_jsonl(args.data)
    # 逐行讲解：检查条件 args.max_samples > 0；成立时执行下方分支。
    if args.max_samples > 0:
        # 逐行讲解：按索引或切片读取 records[:args.max_samples]，赋给 records。
        records = records[: args.max_samples]
    # 逐行讲解：调用 print（print(f"训练数据：{len(records)} 条 | PyTRIO：{version('pytrio')}")）。
    print(f"训练数据：{len(records)} 条 | PyTRIO：{version('pytrio')}")

    # 逐行讲解：调用 trio.ServiceClient（trio.ServiceClient()），赋给 service_client。
    service_client = trio.ServiceClient()
    # 逐行讲解：等待异步操作 service_client.create_lora_training_client_async(base_model=args.base_model, rank=args.lora_rank, seed=args… 完成并取得结果，赋给 training_client。
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    # 逐行讲解：调用 Encoder（Encoder(training_client.get_tokenizer())），赋给 encoder。
    encoder = Encoder(training_client.get_tokenizer())
    # 逐行讲解：调用 trio.AdamParams（trio.AdamParams(learning_rate=args.learning_rate)），赋给 adam_params。
    adam_params = trio.AdamParams(learning_rate=args.learning_rate)

    # 逐行讲解：调用 swanlab.init（swanlab.init(mode=args.swanlab_mode, project=args.swanlab_project, experiment_name=args.experiment_name, co…），赋给 swanlab_run。
    swanlab_run = swanlab.init(
        mode=args.swanlab_mode,
        project=args.swanlab_project,
        experiment_name=args.experiment_name,
        config={
            "algorithm": "masked-sft",
            "dataset": "typed-decisions+toolace",
            "dataset_size": len(records),
            "base_model": args.base_model,
            "pytrio_version": version("pytrio"),
            "lora_rank": args.lora_rank,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "save_every": args.save_every,
        },
    )

    # 逐行讲解：取值 0，赋给 global_step。
    global_step = 0
    # 逐行讲解：取值 0，赋给 last_saved_step。
    last_saved_step = 0
    # 逐行讲解：计算 args.epochs * (len(records) // args.batch_size)，赋给 total_steps。
    total_steps = args.epochs * (len(records) // args.batch_size)
    # 逐行讲解：声明 pending 的类型为 list[asyncio.Task]；构造列表 []。
    pending: list[asyncio.Task] = []
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：遍历 range(args.epochs)，将每项绑定到 epoch。
        for epoch in range(args.epochs):
            # 逐行讲解：打乱数据顺序（random.Random(args.seed + epoch).shuffle(records)）。
            random.Random(args.seed + epoch).shuffle(records)
            # 逐行讲解：遍历 range(0, len(records) - args.batch_size + 1, args.batch_size)，将每项绑定到 start。
            for start in range(0, len(records) - args.batch_size + 1, args.batch_size):
                # 逐行讲解：按索引或切片读取 records[start:start + args.batch_size]，赋给 batch。
                batch = records[start : start + args.batch_size]
                # 逐行讲解：通过推导式逐项生成结果：[encoder.build_sft_datum(record) for record in batch]，赋给 datums。
                datums = [encoder.build_sft_datum(record) for record in batch]
                # 逐行讲解：求和（sum((sum(d.loss_fn_inputs['weights'].data) for d in datums))），赋给 supervised。
                supervised = sum(
                    sum(d.loss_fn_inputs["weights"].data) for d in datums
                )

                # 提交后不等待，loss 计算和 SwanLab 上报交给后台任务，训练流水线不断流。
                fwdbwd_future = await training_client.forward_backward_async(
                    datums, "cross_entropy"
                )
                # 逐行讲解：等待异步操作 training_client.optim_step_async(adam_params) 完成并取得结果，赋给 optim_future。
                optim_future = await training_client.optim_step_async(adam_params)
                # 逐行讲解：更新 global_step：global_step += 1。
                global_step += 1

                # 逐行讲解：定义异步函数 log_step：封装下方逻辑，供调用方复用。
                async def log_step(fwdbwd_future, optim_future, supervised, epoch, step):
                    # 逐行讲解：等待异步操作 fwdbwd_future 完成并取得结果，赋给 result。
                    result = await fwdbwd_future
                    # 逐行讲解：等待异步操作 optim_future 完成并取得结果。
                    await optim_future
                    # 逐行讲解：通过推导式逐项生成结果：{key: float(value) for key, value in result.metrics.items()}，赋给 trainer_metrics。
                    trainer_metrics = {
                        key: float(value) for key, value in result.metrics.items()
                    }
                    # 逐行讲解：计算 trainer_metrics['loss_sum'] / supervised，赋给 loss。
                    loss = trainer_metrics["loss_sum"] / supervised
                    # 逐行讲解：计算自然对数（swanlab.log({'loss': loss, 'epoch': epoch, 'supervised_tokens': supervised, **{f'trainer/{key}': value for …）。
                    swanlab.log(
                        {
                            "loss": loss,
                            "epoch": epoch,
                            "supervised_tokens": supervised,
                            **{
                                f"trainer/{key}": value
                                for key, value in trainer_metrics.items()
                            },
                        },
                        step=step,
                    )
                    # 逐行讲解：调用 print（print(f'Step {step}/{total_steps} (epoch {epoch}) | loss={loss:.4f} | supervised_tokens={supervised:.0f}', …）。
                    print(
                        f"Step {step}/{total_steps} (epoch {epoch}) | "
                        f"loss={loss:.4f} | supervised_tokens={supervised:.0f}",
                        flush=True,
                    )

                # 逐行讲解：向列表末尾追加一个元素（pending.append(asyncio.create_task(log_step(fwdbwd_future, optim_future, supervised, epoch, global_step)))）。
                pending.append(
                    asyncio.create_task(
                        log_step(fwdbwd_future, optim_future, supervised, epoch, global_step)
                    )
                )

                # 逐行讲解：检查条件 args.save_every > 0 and global_step % args.save_every == 0；成立时执行下方分支。
                if args.save_every > 0 and global_step % args.save_every == 0:
                    # 逐行讲解：等待异步操作 asyncio.gather(*pending) 完成并取得结果。
                    await asyncio.gather(*pending)
                    # 逐行讲解：调用 pending.clear（pending.clear()）。
                    pending.clear()
                    # 逐行讲解：等待异步操作 save_weights(training_client, args.weights_name, global_step) 完成并取得结果。
                    await save_weights(training_client, args.weights_name, global_step)
                    # 逐行讲解：取值 global_step，赋给 last_saved_step。
                    last_saved_step = global_step

            # 每个 epoch 结束都存一份；与 step 存档点重合时跳过，避免重复保存。
            if last_saved_step != global_step:
                # 逐行讲解：等待异步操作 asyncio.gather(*pending) 完成并取得结果。
                await asyncio.gather(*pending)
                # 逐行讲解：调用 pending.clear（pending.clear()）。
                pending.clear()
                # 逐行讲解：等待异步操作 save_weights(training_client, args.weights_name, global_step) 完成并取得结果。
                await save_weights(training_client, args.weights_name, global_step)
                # 逐行讲解：取值 global_step，赋给 last_saved_step。
                last_saved_step = global_step

        # 逐行讲解：等待异步操作 asyncio.gather(*pending) 完成并取得结果。
        await asyncio.gather(*pending)
        # 逐行讲解：调用 pending.clear（pending.clear()）。
        pending.clear()
        # 逐行讲解：检查条件 last_saved_step != global_step；成立时执行下方分支。
        if last_saved_step != global_step:
            # 逐行讲解：等待异步操作 save_weights(training_client, args.weights_name, global_step) 完成并取得结果。
            await save_weights(training_client, args.weights_name, global_step)
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
