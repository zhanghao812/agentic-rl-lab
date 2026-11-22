"""Deep Research Bench 评测：先生成研究报告，再用 RACE 对照参考报告评分。

uv run python experiments/tournament-grpo/eval.py generate
uv run python experiments/tournament-grpo/eval.py score

两个阶段都逐题写入 jsonl；同一命令重跑时会跳过已完成的题目。
"""

# 逐行讲解：导入模块：argparse。
import argparse
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 collections 导入 Counter。
from collections import Counter
# 逐行讲解：从 concurrent.futures 导入 ThreadPoolExecutor, as_completed。
from concurrent.futures import ThreadPoolExecutor, as_completed
# 逐行讲解：从 dataclasses 导入 asdict, replace。
from dataclasses import asdict, replace
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 time 导入 perf_counter。
from time import perf_counter

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio
# 逐行讲解：导入模块：swanlab。
import swanlab
# 逐行讲解：从 tqdm 导入 tqdm。
from tqdm import tqdm

# 逐行讲解：从 data 导入 ResearchExample。
from data import ResearchExample
# 逐行讲解：从 protocol 导入 transcript_text。
from protocol import transcript_text
# 逐行讲解：从 race 导入 RACE_DIMS, RaceJudge。
from race import RACE_DIMS, RaceJudge
# 逐行讲解：从 reward 导入 analyze_format。
from reward import analyze_format
# 逐行讲解：从 rollout 导入 RolloutConfig, rollout_batch。
from rollout import RolloutConfig, rollout_batch
# 逐行讲解：从 search 导入 ResearchTools。
from search import ResearchTools

# 逐行讲解：取值 Path(__file__).resolve().parent，赋给 BASE_DIR。
BASE_DIR = Path(__file__).resolve().parent
# 逐行讲解：构造元组 ('no_tool_call', 'max_turns', 'max_tool_calls', 'max_response_tokens')，赋给 FINISH_REASONS。
FINISH_REASONS = ("no_tool_call", "max_turns", "max_tool_calls", "max_response_tokens")


# 逐行讲解：定义函数 load_jsonl：封装下方逻辑，供调用方复用。
def load_jsonl(path: Path) -> list[dict]:
    # 逐行讲解：检查条件 not path.exists()；成立时执行下方分支。
    if not path.exists():
        # 逐行讲解：返回结果：[]。
        return []
    # 逐行讲解：返回结果：[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]。
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# 逐行讲解：定义函数 append_jsonl：封装下方逻辑，供调用方复用。
def append_jsonl(file, record: dict) -> None:
    # 逐行讲解：调用 file.write（file.write(json.dumps(record, ensure_ascii=False) + '\n')）。
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    # 逐行讲解：调用 file.flush（file.flush()）。
    file.flush()


# 逐行讲解：定义函数 write_json：封装下方逻辑，供调用方复用。
def write_json(path: Path, value: dict) -> None:
    # 逐行讲解：将文本写入文件（path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')）。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 逐行讲解：定义函数 path_name：把模型名或远程权重路径转换成单层目录名。
def path_name(value: str) -> str:
    """把模型名或远程权重路径转换成单层目录名。"""
    # 逐行讲解：返回结果：re.sub('[^A-Za-z0-9._-]+', '--', value).strip('-')。
    return re.sub(r"[^A-Za-z0-9._-]+", "--", value).strip("-")


# 逐行讲解：定义函数 generation_metrics：封装下方逻辑，供调用方复用。
def generation_metrics(records: list[dict]) -> dict[str, float]:
    # 逐行讲解：调用 len（len(records)），赋给 count。
    count = len(records)
    # 逐行讲解：求和（sum((len(r['tool_calls']) for r in records))），赋给 calls。
    calls = sum(len(r["tool_calls"]) for r in records)
    # 逐行讲解：调用 Counter（Counter((r['finish_reason'] for r in records))），赋给 finishes。
    finishes = Counter(r["finish_reason"] for r in records)
    # 逐行讲解：返回结果：{'eval/questions': count, 'eval/answer_rate': sum((bool(r['article']) for r in records)) / count, 'eval/for…。
    return {
        "eval/questions": count,
        "eval/answer_rate": sum(bool(r["article"]) for r in records) / count,
        "eval/format_rate": sum(r["format"]["format_reward"] for r in records) / count,
        "eval/assistant_turns": sum(r["assistant_turns"] for r in records) / count,
        "eval/response_tokens": sum(len(r["response_tokens"]) for r in records) / count,
        "eval/action_tokens": sum(sum(r["response_mask"]) for r in records) / count,
        "eval/tool_calls": calls / count,
        "eval/tool_error_rate": sum(r["tool_errors"] for r in records) / max(calls, 1),
        **{f"eval/finish_{reason}": finishes[reason] / count for reason in FINISH_REASONS},
    }


# 逐行讲解：定义函数 generate：每题贪心生成 1 条带工具调用的轨迹，最终 <answer> 就是待评分的报告。
def generate(args):
    """每题贪心生成 1 条带工具调用的轨迹，最终 <answer> 就是待评分的报告。"""
    # 逐行讲解：调用 load_jsonl（load_jsonl(args.bench_dir / 'query.jsonl')），赋给 tasks。
    tasks = load_jsonl(args.bench_dir / "query.jsonl")
    # 逐行讲解：检查条件 args.limit；成立时执行下方分支。
    if args.limit:
        # 逐行讲解：按索引或切片读取 tasks[:args.limit]，赋给 tasks。
        tasks = tasks[:args.limit]
    # 逐行讲解：调用 ResearchTools.from_env（ResearchTools.from_env(BASE_DIR / '.env')），赋给 tools。
    tools = ResearchTools.from_env(BASE_DIR / ".env")
    # 逐行讲解：调用 RolloutConfig（RolloutConfig(group_size=1, temperature=0.0, seed=args.seed)），赋给 rollout。
    rollout = RolloutConfig(group_size=1, temperature=0.0, seed=args.seed)
    # 逐行讲解：构造字典 {'questions': len(tasks), 'base_model': args.base_model, 'model_path': args.model_path, 'batch_size': args.…，赋给 config。
    config = {"questions": len(tasks), "base_model": args.base_model, "model_path": args.model_path,
              "batch_size": args.batch_size, "rollout": asdict(rollout), "search_count": tools.search_count}
    # 逐行讲解：调用 write_json（write_json(args.output_dir / 'config.json', config)）。
    write_json(args.output_dir / "config.json", config)

    # 逐行讲解：调用 trio.ServiceClient().create_sampling_client（trio.ServiceClient().create_sampling_client(base_model=args.base_model, model_path=args.model_path)），赋给 sampler。
    sampler = trio.ServiceClient().create_sampling_client(base_model=args.base_model, model_path=args.model_path)
    # 逐行讲解：调用 sampler.get_tokenizer（sampler.get_tokenizer()），赋给 tokenizer。
    tokenizer = sampler.get_tokenizer()
    # 逐行讲解：计算 args.output_dir / 'trajectories.jsonl'，赋给 path。
    path = args.output_dir / "trajectories.jsonl"
    # 逐行讲解：调用 load_jsonl（load_jsonl(path)），赋给 records。
    records = load_jsonl(path)
    # 逐行讲解：通过推导式逐项生成结果：{r['id'] for r in records}，赋给 done。
    done = {r["id"] for r in records}

    # 逐行讲解：调用 swanlab.init（swanlab.init(project='tournament-grpo', name=args.run_name, config=config, mode=args.swanlab_mode)）。
    swanlab.init(project="tournament-grpo", name=args.run_name, config=config, mode=args.swanlab_mode)
    # 逐行讲解：调用 perf_counter（perf_counter()），赋给 started。
    started = perf_counter()
    # 逐行讲解：进入上下文管理器 path.open('a', encoding='utf-8'), tqdm(total=len(tasks), initial=len(done), desc='Generate', unit='report')，退出时执行其收尾逻辑。
    with path.open("a", encoding="utf-8") as file, \
            tqdm(total=len(tasks), initial=len(done), desc="Generate", unit="report") as progress:
        # 逐行讲解：遍历 range(0, len(tasks), args.batch_size)，将每项绑定到 offset。
        for offset in range(0, len(tasks), args.batch_size):
            # 逐行讲解：通过推导式逐项生成结果：[t for t in tasks[offset:offset + args.batch_size] if str(t['id']) not in done]，赋给 batch。
            batch = [t for t in tasks[offset:offset + args.batch_size] if str(t["id"]) not in done]
            # 逐行讲解：检查条件 not batch；成立时执行下方分支。
            if not batch:
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：通过推导式逐项生成结果：[ResearchExample(str(t['id']), t['prompt'], [], 'deep_research_bench') for t in batch]，赋给 examples。
            examples = [ResearchExample(str(t["id"]), t["prompt"], [], "deep_research_bench") for t in batch]
            # 逐行讲解：调用 rollout_batch（rollout_batch(sampler, tokenizer, tools, examples, replace(rollout, seed=args.seed + offset), lambda n: None)），赋给 trajectories。
            trajectories = rollout_batch(sampler, tokenizer, tools, examples,
                                         replace(rollout, seed=args.seed + offset), lambda n: None)
            # 逐行讲解：遍历 zip(batch, trajectories, strict=True)，将每项绑定到 (task, trajectory)。
            for task, trajectory in zip(batch, trajectories, strict=True):
                # 逐行讲解：调用 transcript_text（transcript_text(tokenizer, trajectory.prompt_tokens, trajectory.response_tokens)），赋给 text。
                text = transcript_text(tokenizer, trajectory.prompt_tokens, trajectory.response_tokens)
                # 逐行讲解：调用 analyze_format（analyze_format(text)），赋给 info。
                info = analyze_format(text)
                # 没有合法 <answer> 时 article 为空字符串，这道题仍然送去评分。
                record = {**asdict(trajectory), "id": str(task["id"]), "prompt": task["prompt"],
                          "language": task["language"], "article": info["rollout_answer"],
                          "response_text": text, "format": info}
                # 逐行讲解：调用 append_jsonl（append_jsonl(file, record)）。
                append_jsonl(file, record)
                # 逐行讲解：向列表末尾追加一个元素（records.append(record)）。
                records.append(record)
                # 逐行讲解：调用 progress.update（progress.update(1)）。
                progress.update(1)
            # 逐行讲解：计算自然对数（swanlab.log(generation_metrics(records), step=len(records))）。
            swanlab.log(generation_metrics(records), step=len(records))

    # 逐行讲解：进入上下文管理器 (args.output_dir / 'articles.jsonl').open('w', encoding='utf-8')，退出时执行其收尾逻辑。
    with (args.output_dir / "articles.jsonl").open("w", encoding="utf-8") as file:
        # 逐行讲解：遍历 records，将每项绑定到 record。
        for record in records:
            # 逐行讲解：调用 append_jsonl（append_jsonl(file, {key: record[key] for key in ('id', 'prompt', 'language', 'article')})）。
            append_jsonl(file, {key: record[key] for key in ("id", "prompt", "language", "article")})
    # 逐行讲解：构造字典 {**generation_metrics(records), 'eval/seconds_this_run': perf_counter() - started}，赋给 summary。
    summary = {**generation_metrics(records), "eval/seconds_this_run": perf_counter() - started}
    # 逐行讲解：调用 write_json（write_json(args.output_dir / 'generation-summary.json', summary)）。
    write_json(args.output_dir / "generation-summary.json", summary)
    # 逐行讲解：计算自然对数（swanlab.log(summary, step=len(records) + 1)）。
    swanlab.log(summary, step=len(records) + 1)
    # 逐行讲解：调用 print（print(json.dumps(summary, indent=2))）。
    print(json.dumps(summary, indent=2))


# 逐行讲解：定义函数 score：每份报告和同题参考报告一起交给裁判，按作者的 RACE 规则算出相对分数。
def score(args):
    """每份报告和同题参考报告一起交给裁判，按作者的 RACE 规则算出相对分数。"""
    # 逐行讲解：调用 load_jsonl（load_jsonl(args.output_dir / 'articles.jsonl')），赋给 articles。
    articles = load_jsonl(args.output_dir / "articles.jsonl")
    # 逐行讲解：通过推导式逐项生成结果：{r['prompt']: r for r in load_jsonl(args.bench_dir / 'reference.jsonl')}，赋给 references。
    references = {r["prompt"]: r for r in load_jsonl(args.bench_dir / "reference.jsonl")}
    # 逐行讲解：通过推导式逐项生成结果：{r['prompt']: r for r in load_jsonl(args.bench_dir / 'criteria.jsonl')}，赋给 criteria。
    criteria = {r["prompt"]: r for r in load_jsonl(args.bench_dir / "criteria.jsonl")}
    # 逐行讲解：调用 RaceJudge.from_env（RaceJudge.from_env(BASE_DIR / '.env')），赋给 judge。
    judge = RaceJudge.from_env(BASE_DIR / ".env")
    # 逐行讲解：计算 args.output_dir / 'race' / path_name(judge.model)，赋给 score_dir。
    score_dir = args.output_dir / "race" / path_name(judge.model)
    # 逐行讲解：创建目录（score_dir.mkdir(parents=True, exist_ok=True)）。
    score_dir.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：构造字典 {'judge_url': judge.api_url, 'judge_model': judge.model, 'judge_concurrency': args.judge_concurrency}，赋给 config。
    config = {"judge_url": judge.api_url, "judge_model": judge.model, "judge_concurrency": args.judge_concurrency}
    # 逐行讲解：调用 write_json（write_json(score_dir / 'score-config.json', config)）。
    write_json(score_dir / "score-config.json", config)

    # 逐行讲解：计算 score_dir / 'race-results.jsonl'，赋给 path。
    path = score_dir / "race-results.jsonl"
    # 逐行讲解：调用 load_jsonl（load_jsonl(path)），赋给 results。
    results = load_jsonl(path)
    # 逐行讲解：通过推导式逐项生成结果：{r['id'] for r in results}，赋给 done。
    done = {r["id"] for r in results}
    # 逐行讲解：通过推导式逐项生成结果：[a for a in articles if a['id'] not in done]，赋给 todo。
    todo = [a for a in articles if a["id"] not in done]

    # 逐行讲解：调用 swanlab.init（swanlab.init(project='tournament-grpo', name=f'{args.run_name}-{path_name(judge.model)}', config=config, mo…）。
    swanlab.init(project="tournament-grpo", name=f"{args.run_name}-{path_name(judge.model)}",
                 config=config, mode=args.swanlab_mode)
    # 逐行讲解：进入上下文管理器 ThreadPoolExecutor(max_workers=args.judge_concurrency), path.open('a', encoding='utf-8'), tqdm(total=len(articles), initial=len(done), desc='RACE', unit='report')，退出时执行其收尾逻辑。
    with ThreadPoolExecutor(max_workers=args.judge_concurrency) as pool, \
            path.open("a", encoding="utf-8") as file, \
            tqdm(total=len(articles), initial=len(done), desc="RACE", unit="report") as progress:
        # 逐行讲解：通过推导式逐项生成结果：[pool.submit(judge.score, a, references[a['prompt']], criteria[a['prompt']]) for a in todo]，赋给 futures。
        futures = [pool.submit(judge.score, a, references[a["prompt"]], criteria[a["prompt"]]) for a in todo]
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：遍历 as_completed(futures)，将每项绑定到 future。
            for future in as_completed(futures):
                # 逐行讲解：等待任务完成并读取结果（future.result()），赋给 result。
                result = future.result()
                # 逐行讲解：调用 append_jsonl（append_jsonl(file, result)）。
                append_jsonl(file, result)
                # 逐行讲解：向列表末尾追加一个元素（results.append(result)）。
                results.append(result)
                # 逐行讲解：调用 progress.update（progress.update(1)）。
                progress.update(1)
                # 逐行讲解：计算自然对数（swanlab.log({'race/completed': len(results)}, step=len(results))）。
                swanlab.log({"race/completed": len(results)}, step=len(results))
        except Exception:
            # 某题重试 3 次仍失败：取消还在排队的请求，已写入的结果下次重跑时跳过。
            pool.shutdown(cancel_futures=True)
            # 逐行讲解：抛出异常：重新抛出当前异常。
            raise

    # 逐行讲解：通过推导式逐项生成结果：{f'race/{dim}': 100 * sum((r[dim] for r in results)) / len(results) for dim in (*RACE_DIMS, 'overall_score')}，赋给 summary。
    summary = {f"race/{dim}": 100 * sum(r[dim] for r in results) / len(results)
               for dim in (*RACE_DIMS, "overall_score")}
    # 逐行讲解：调用 len（len(results)），赋给 summary['race/questions']。
    summary["race/questions"] = len(results)
    # 逐行讲解：调用 write_json（write_json(score_dir / 'race-summary.json', summary)）。
    write_json(score_dir / "race-summary.json", summary)
    # 逐行讲解：计算自然对数（swanlab.log(summary, step=len(results) + 1)）。
    swanlab.log(summary, step=len(results) + 1)
    # 逐行讲解：调用 print（print(json.dumps(summary, indent=2))）。
    print(json.dumps(summary, indent=2))


# 逐行讲解：定义函数 parse_args：封装下方逻辑，供调用方复用。
def parse_args():
    # 逐行讲解：调用 argparse.ArgumentParser（argparse.ArgumentParser(description=__doc__)），赋给 parser。
    parser = argparse.ArgumentParser(description=__doc__)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('stage', choices=('generate', 'score'))）。
    parser.add_argument("stage", choices=("generate", "score"))
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--bench-dir', type=Path, default=BASE_DIR / 'datasets/deep_research_bench')）。
    parser.add_argument("--bench-dir", type=Path, default=BASE_DIR / "datasets/deep_research_bench")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--output-dir', type=Path, default=BASE_DIR / 'outputs/drb', help='输出根目录，自动追加 policy 模型…）。
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs/drb",
                        help="输出根目录，自动追加 policy 模型名和权重标识")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--base-model', default='Qwen/Qwen3.5-4B')）。
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--checkpoint', type=Path, help='训练保存的 checkpoint-step-N.json')）。
    parser.add_argument("--checkpoint", type=Path, help="训练保存的 checkpoint-step-N.json")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--limit', type=int, default=0, help='只生成前 N 题；0 为全部 100 题')）。
    parser.add_argument("--limit", type=int, default=0, help="只生成前 N 题；0 为全部 100 题")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--seed', type=int, default=42)）。
    parser.add_argument("--seed", type=int, default=42)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--batch-size', type=int, default=2)）。
    parser.add_argument("--batch-size", type=int, default=2)
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--judge-concurrency', type=int, default=8, help='同时进行的 RACE 评分请求数')）。
    parser.add_argument("--judge-concurrency", type=int, default=8, help="同时进行的 RACE 评分请求数")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--run-name', help='SwanLab 运行名，默认包含 policy 模型、权重和阶段')）。
    parser.add_argument("--run-name", help="SwanLab 运行名，默认包含 policy 模型、权重和阶段")
    # 逐行讲解：注册命令行选项及其默认值或约束（parser.add_argument('--swanlab-mode', choices=('online', 'offline', 'disabled'), default='online')）。
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "disabled"), default="online")
    # 逐行讲解：解析实际命令行参数（parser.parse_args()），赋给 args。
    args = parser.parse_args()
    # 逐行讲解：取值 None，赋给 args.model_path。
    args.model_path = None
    # 逐行讲解：检查条件 args.checkpoint；成立时执行下方分支。
    if args.checkpoint:
        # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(args.checkpoint.read_text())），赋给 checkpoint。
        checkpoint = json.loads(args.checkpoint.read_text())
        # 逐行讲解：按索引或切片读取 checkpoint['config']['base_model']，赋给 args.base_model。
        args.base_model = checkpoint["config"]["base_model"]
        # 逐行讲解：按索引或切片读取 checkpoint['sampler_path']，赋给 args.model_path。
        args.model_path = checkpoint["sampler_path"]
    # 逐行讲解：按条件选择结果：path_name(args.model_path) if args.model_path else 'base'，赋给 weights。
    weights = path_name(args.model_path) if args.model_path else "base"
    # 逐行讲解：调用 path_name（path_name(args.base_model)），赋给 model。
    model = path_name(args.base_model)
    # 逐行讲解：计算 args.output_dir / model / weights，赋给 args.output_dir。
    args.output_dir = args.output_dir / model / weights
    # 逐行讲解：取值 args.run_name or f'{model}-{weights}-{args.stage}'，赋给 args.run_name。
    args.run_name = args.run_name or f"{model}-{weights}-{args.stage}"
    # 逐行讲解：返回结果：args。
    return args


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：解析实际命令行参数（parse_args()），赋给 args。
    args = parse_args()
    # 逐行讲解：创建目录（args.output_dir.mkdir(parents=True, exist_ok=True)）。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 逐行讲解：检查条件 args.stage == 'generate'；成立时执行下方分支。
    if args.stage == "generate":
        # 逐行讲解：调用 generate（generate(args)）。
        generate(args)
    else:
        # 逐行讲解：调用 score（score(args)）。
        score(args)
    # 逐行讲解：调用 swanlab.finish（swanlab.finish()）。
    swanlab.finish()
