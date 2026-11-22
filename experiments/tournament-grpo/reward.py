"""Tournament-GRPO 的二值格式奖励与淘汰赛奖励。

Adapted from PuffYang/Tournament_GRPO (eea826cf), Apache-2.0.
格式解析和淘汰赛流程沿用作者 tournament_grpo_reward.py，去掉了 verl 适配层。
"""

# 逐行讲解：导入模块：random。
import random
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：从 data 导入 ResearchExample。
from data import ResearchExample
# 逐行讲解：从 judge 导入 JudgeClient。
from judge import JudgeClient


# 逐行讲解：取值 120000，赋给 MAX_ROLLOUT_CHARS。
MAX_ROLLOUT_CHARS = 120000  # 作者 max_rollout_chars：送给 judge 的答案最多保留的字符数
# 逐行讲解：取值 4，赋给 TOURNAMENT_REPEATS。
TOURNAMENT_REPEATS = 4  # 整场淘汰赛重复次数 M；每次重新抽签
# 逐行讲解：取值 2，赋给 PLAYERS_PER_MATCH。
PLAYERS_PER_MATCH = 2  # 每场比赛人数 G；每场选 1 个胜者，胜者 +1 分

# 逐行讲解：调用 re.compile（re.compile('<(/?)([A-Za-z_][A-Za-z0-9_]*)([^>]*)>', re.DOTALL)），赋给 TAG_PATTERN。
TAG_PATTERN = re.compile(r"<(/?)([A-Za-z_][A-Za-z0-9_]*)([^>]*)>", re.DOTALL)
# 逐行讲解：调用 re.compile（re.compile('<answer>(.*?)</answer>', re.DOTALL)），赋给 ANSWER_PATTERN。
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# 逐行讲解：调用 re.compile（re.compile('\\s+([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')），赋给 ATTRIBUTE_PATTERN。
ATTRIBUTE_PATTERN = re.compile(r'\s+([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
# 逐行讲解：构造集合 {'think', 'call_tool', 'tool_output', 'snippet', 'webpage', 'answer'}，赋给 ALLOWED_TAGS。
ALLOWED_TAGS = {"think", "call_tool", "tool_output", "snippet", "webpage", "answer"}
# 逐行讲解：构造集合 {'google_search', 'browse_webpage'}，赋给 TOOL_NAMES。
TOOL_NAMES = {"google_search", "browse_webpage"}
# 标签层面的错误，按作者的固定顺序写入 failure_reasons。
TAG_ERRORS = ("tag:unknown_tags", "tag:invalid_attrs", "tag:broken_xml", "structure:stray_text_outside_tags")


# ---------------------------------------------------------------------------
# 格式奖励
# ---------------------------------------------------------------------------


# 逐行讲解：定义类 TranscriptBlock：一个顶层标签块，例如 <think>...</think>；嵌套的子标签留在 content 里。
@dataclass(frozen=True)
class TranscriptBlock:
    """一个顶层标签块，例如 <think>...</think>；嵌套的子标签留在 content 里。"""

    # 逐行讲解：声明 tag 的类型为 str。
    tag: str
    # 逐行讲解：声明 attrs 的类型为 dict[str, str]。
    attrs: dict[str, str]
    # 逐行讲解：声明 content 的类型为 str。
    content: str


# 逐行讲解：定义函数 parse_attrs：解析标签属性；只有 call_tool 允许且必须带 name。
def parse_attrs(tag: str, text: str) -> tuple[dict[str, str], bool]:
    """解析标签属性；只有 call_tool 允许且必须带 name。"""
    # 逐行讲解：构造元组 ({}, 0)，赋给 (attrs, cursor)。
    attrs, cursor = {}, 0
    # 逐行讲解：遍历 ATTRIBUTE_PATTERN.finditer(text)，将每项绑定到 match。
    for match in ATTRIBUTE_PATTERN.finditer(text):
        # 逐行讲解：调用 match.groups（match.groups()），赋给 (key, value)。
        key, value = match.groups()
        # 逐行讲解：检查条件 text[cursor:match.start()].strip() or key in attrs；成立时执行下方分支。
        if text[cursor:match.start()].strip() or key in attrs:
            # 逐行讲解：返回结果：(attrs, False)。
            return attrs, False
        # 逐行讲解：取值 value，赋给 attrs[key]。
        attrs[key] = value
        # 逐行讲解：调用 match.end（match.end()），赋给 cursor。
        cursor = match.end()
    # 逐行讲解：按条件选择结果：{'name'} if tag == 'call_tool' else set()，赋给 allowed。
    allowed = {"name"} if tag == "call_tool" else set()
    # 逐行讲解：取值 not text[cursor:].strip() and (not attrs.keys() - allowed)，赋给 valid。
    valid = not text[cursor:].strip() and not (attrs.keys() - allowed)
    # 逐行讲解：返回结果：(attrs, valid and (tag != 'call_tool' or 'name' in attrs))。
    return attrs, valid and (tag != "call_tool" or "name" in attrs)


# 逐行讲解：定义函数 parse_transcript：把完整轨迹切成顶层标签块，并收集标签错误。
def parse_transcript(text: str) -> tuple[list[TranscriptBlock], set[str]]:
    """把完整轨迹切成顶层标签块，并收集标签错误。

    沿用作者的栈式恢复：observation 里的标签同样参与解析；关闭标签与栈顶不匹配时，
    回退到栈里最近的同名标签，中间未闭合的标签直接丢弃。
    """
    # 逐行讲解：构造元组 ([], [], set())，赋给 (blocks, stack, errors)。
    blocks, stack, errors = [], [], set()
    # 逐行讲解：取值 0，赋给 cursor。
    cursor = 0  # 上一个标签的结束位置，用来发现标签外的杂文
    # 逐行讲解：遍历 TAG_PATTERN.finditer(text)，将每项绑定到 match。
    for match in TAG_PATTERN.finditer(text):
        # 逐行讲解：检查条件 not stack and text[cursor:match.start()].strip()；成立时执行下方分支。
        if not stack and text[cursor:match.start()].strip():
            # 逐行讲解：调用 errors.add（errors.add('structure:stray_text_outside_tags')）。
            errors.add("structure:stray_text_outside_tags")
        # 逐行讲解：调用 match.groups（match.groups()），赋给 (closing, tag, attributes)。
        closing, tag, attributes = match.groups()
        # 逐行讲解：调用 match.end（match.end()），赋给 cursor。
        cursor = match.end()
        # 逐行讲解：检查条件 tag not in ALLOWED_TAGS；成立时执行下方分支。
        if tag not in ALLOWED_TAGS:
            # 逐行讲解：调用 errors.add（errors.add('tag:unknown_tags')）。
            errors.add("tag:unknown_tags")

        # 逐行讲解：检查条件 not closing；成立时执行下方分支。
        if not closing:
            # 逐行讲解：调用 parse_attrs（parse_attrs(tag, attributes)），赋给 (attrs, valid)。
            attrs, valid = parse_attrs(tag, attributes)
            # 逐行讲解：检查条件 not valid；成立时执行下方分支。
            if not valid:
                # 逐行讲解：调用 errors.add（errors.add('tag:invalid_attrs')）。
                errors.add("tag:invalid_attrs")
            # 逐行讲解：向列表末尾追加一个元素（stack.append((tag, attrs, match.end()))）。
            stack.append((tag, attrs, match.end()))
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue

        # 逐行讲解：检查条件 attributes.strip()；成立时执行下方分支。
        if attributes.strip():
            # 逐行讲解：调用 errors.add（errors.add('tag:invalid_attrs')）。
            errors.add("tag:invalid_attrs")
        # 逐行讲解：检查条件 stack and stack[-1][0] == tag；成立时执行下方分支。
        if stack and stack[-1][0] == tag:
            # 逐行讲解：调用 stack.pop（stack.pop()），赋给 opened。
            opened = stack.pop()
        else:
            # 栈顶不匹配：回退到最近的同名开标签，中间未闭合的标签一起丢弃。
            errors.add("tag:broken_xml")
            # 逐行讲解：调用 max（max((i for i, (open_tag, _, _) in enumerate(stack) if open_tag == tag), default=None)），赋给 index。
            index = max((i for i, (open_tag, _, _) in enumerate(stack) if open_tag == tag), default=None)
            # 逐行讲解：检查条件 index is None；成立时执行下方分支。
            if index is None:
                # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                continue
            # 逐行讲解：构造元组 (stack[index], stack[:index])，赋给 (opened, stack)。
            opened, stack = stack[index], stack[:index]
        # 逐行讲解：检查条件 not stack；成立时执行下方分支。
        if not stack:
            # 逐行讲解：取值 opened，赋给 (open_tag, attrs, start)。
            open_tag, attrs, start = opened
            # 逐行讲解：向列表末尾追加一个元素（blocks.append(TranscriptBlock(open_tag, attrs, text[start:match.start()]))）。
            blocks.append(TranscriptBlock(open_tag, attrs, text[start:match.start()]))

    # 逐行讲解：检查条件 not stack and text[cursor:].strip()；成立时执行下方分支。
    if not stack and text[cursor:].strip():
        # 逐行讲解：调用 errors.add（errors.add('structure:stray_text_outside_tags')）。
        errors.add("structure:stray_text_outside_tags")
    # 逐行讲解：检查条件 stack；成立时执行下方分支。
    if stack:
        # 逐行讲解：调用 errors.add（errors.add('tag:broken_xml')）。
        errors.add("tag:broken_xml")
    # 逐行讲解：返回结果：(blocks, errors)。
    return blocks, errors


# 逐行讲解：定义函数 extract_answer：只接受唯一、非空、不含标签且位于末尾的 <answer>。
def extract_answer(text: str) -> str:
    """只接受唯一、非空、不含标签且位于末尾的 <answer>。"""
    # 逐行讲解：调用 list（list(ANSWER_PATTERN.finditer(text))），赋给 matches。
    matches = list(ANSWER_PATTERN.finditer(text))
    # 逐行讲解：检查条件 len(matches) != 1；成立时执行下方分支。
    if len(matches) != 1:
        # 逐行讲解：返回结果：''。
        return ""
    # 逐行讲解：调用 matches[0].group(1).strip（matches[0].group(1).strip()），赋给 answer。
    answer = matches[0].group(1).strip()
    # 逐行讲解：检查条件 not answer or TAG_PATTERN.search(answer) or text[matches[0].end():].strip()；成立时执行下方分支。
    if not answer or TAG_PATTERN.search(answer) or text[matches[0].end():].strip():
        # 逐行讲解：返回结果：''。
        return ""
    # 逐行讲解：返回结果：answer。
    return answer


# 逐行讲解：定义函数 tool_name：封装下方逻辑，供调用方复用。
def tool_name(block: TranscriptBlock) -> str:
    # 逐行讲解：返回结果：block.attrs.get('name', '').strip()。
    return block.attrs.get("name", "").strip()


# 逐行讲解：定义函数 is_tool_cycle：blocks[index:index+3] 是否是一轮非空的 call_tool(name) → tool_output → think。
def is_tool_cycle(blocks: list[TranscriptBlock], index: int, name: str) -> bool:
    """blocks[index:index+3] 是否是一轮非空的 call_tool(name) → tool_output → think。"""
    # 逐行讲解：检查条件 index + 2 >= len(blocks)；成立时执行下方分支。
    if index + 2 >= len(blocks):
        # 逐行讲解：返回结果：False。
        return False
    # 逐行讲解：按索引或切片读取 blocks[index:index + 3]，赋给 (call, observation, think)。
    call, observation, think = blocks[index:index + 3]
    # 逐行讲解：返回结果：call.tag == 'call_tool' and tool_name(call) == name and bool(call.content.strip()) and (observation.tag == …。
    return (call.tag == "call_tool" and tool_name(call) == name and bool(call.content.strip())
            and observation.tag == "tool_output" and bool(observation.content.strip())
            and think.tag == "think" and bool(think.content.strip()))


# 逐行讲解：定义函数 check_structure：作者的轨迹结构：think → 若干研究段 → answer；返回第一条失败原因，合法时返回 None。
def check_structure(blocks: list[TranscriptBlock]) -> str | None:
    """作者的轨迹结构：think → 若干研究段 → answer；返回第一条失败原因，合法时返回 None。

    每个研究段 = 至少 1 轮 google_search 循环 + 任意轮 browse_webpage 循环。
    只在标签无错误且 extract_answer 成功时调用；此时唯一的 answer 必然是最后一个顶层块，
    所以作者的 must_end_with_answer、invalid_tool_cycle_order 两条检查不会触发，这里省略。
    """
    # 逐行讲解：检查条件 blocks[0].tag != 'think' or not blocks[0].content.strip()；成立时执行下方分支。
    if blocks[0].tag != "think" or not blocks[0].content.strip():
        # 逐行讲解：返回结果：'structure:must_start_with_non_empty_think'。
        return "structure:must_start_with_non_empty_think"
    # 逐行讲解：检查条件 len(blocks) <= 2；成立时执行下方分支。
    if len(blocks) <= 2:
        # 逐行讲解：返回结果：'structure:must_have_at_least_one_tool_cycle'。
        return "structure:must_have_at_least_one_tool_cycle"
    # 逐行讲解：取值 1，赋给 index。
    index = 1
    # 逐行讲解：当 index < len(blocks) - 1 成立时继续循环。
    while index < len(blocks) - 1:
        # 逐行讲解：检查条件 not is_tool_cycle(blocks, index, 'google_search')；成立时执行下方分支。
        if not is_tool_cycle(blocks, index, "google_search"):
            # 逐行讲解：返回结果：'structure:each_segment_must_start_with_google_search_cycle'。
            return "structure:each_segment_must_start_with_google_search_cycle"
        # 逐行讲解：当 is_tool_cycle(blocks, index, 'google_search') 成立时继续循环。
        while is_tool_cycle(blocks, index, "google_search"):
            # 逐行讲解：更新 index：index += 3。
            index += 3
        # 逐行讲解：当 is_tool_cycle(blocks, index, 'browse_webpage') 成立时继续循环。
        while is_tool_cycle(blocks, index, "browse_webpage"):
            # 逐行讲解：更新 index：index += 3。
            index += 3
    # 逐行讲解：返回结果：None。
    return None


# 逐行讲解：定义函数 analyze_format：格式奖励为 0 或 1：没有任何失败原因时得 1。同时取出送给 judge 的最终答案。
def analyze_format(text: str) -> dict[str, Any]:
    """格式奖励为 0 或 1：没有任何失败原因时得 1。同时取出送给 judge 的最终答案。"""
    # 逐行讲解：调用 parse_transcript（parse_transcript(text)），赋给 (blocks, errors)。
    blocks, errors = parse_transcript(text)
    # 逐行讲解：通过推导式逐项生成结果：[reason for reason in TAG_ERRORS if reason in errors]，赋给 failures。
    failures = [reason for reason in TAG_ERRORS if reason in errors]
    # 逐行讲解：检查条件 any((block.tag == 'call_tool' and tool_name(block) not in TOOL_NAMES for block in blocks))；成立时执行下方分支。
    if any(block.tag == "call_tool" and tool_name(block) not in TOOL_NAMES for block in blocks):
        # 逐行讲解：向列表末尾追加一个元素（failures.append('tool:unknown_tool_name')）。
        failures.append("tool:unknown_tool_name")
    # 逐行讲解：调用 extract_answer（extract_answer(text)），赋给 answer。
    answer = extract_answer(text)
    # 逐行讲解：检查条件 not answer；成立时执行下方分支。
    if not answer:
        # 逐行讲解：向列表末尾追加一个元素（failures.append('answer:invalid_or_missing_final_answer')）。
        failures.append("answer:invalid_or_missing_final_answer")
    # 逐行讲解：检查条件 not failures；成立时执行下方分支。
    if not failures:
        # 逐行讲解：调用 check_structure（check_structure(blocks)），赋给 structure_failure。
        structure_failure = check_structure(blocks)
        # 逐行讲解：检查条件 structure_failure；成立时执行下方分支。
        if structure_failure:
            # 逐行讲解：向列表末尾追加一个元素（failures.append(structure_failure)）。
            failures.append(structure_failure)
    # 逐行讲解：返回结果：{'format_reward': float(not failures), 'failure_reasons': failures, 'rollout_answer': answer}。
    return {"format_reward": float(not failures), "failure_reasons": failures, "rollout_answer": answer}


# ---------------------------------------------------------------------------
# 淘汰赛奖励
# ---------------------------------------------------------------------------


# 逐行讲解：定义函数 normalize_scores：组内 min-max 归一化；全部同分时每个候选记 0.5。
def normalize_scores(scores: list[float]) -> list[float]:
    """组内 min-max 归一化；全部同分时每个候选记 0.5。"""
    # 逐行讲解：构造元组 (min(scores), max(scores))，赋给 (low, high)。
    low, high = min(scores), max(scores)
    # 逐行讲解：返回结果：[0.5] * len(scores) if high == low else [(value - low) / (high - low) for value in scores]。
    return [0.5] * len(scores) if high == low else [(value - low) / (high - low) for value in scores]


# 逐行讲解：定义函数 run_tournament：作者默认设置：每组 2 人、每组 1 个胜者、整场重复 4 次、每胜一场 +1 分。
def run_tournament(
    query: str, rubrics: list[dict[str, str]], answers: list[str], judge: JudgeClient,
) -> dict[str, Any]:
    """作者默认设置：每组 2 人、每组 1 个胜者、整场重复 4 次、每胜一场 +1 分。

    K=8 时每次 4 + 2 + 1 场比赛，4 次共 28 场。
    """
    # 逐行讲解：计算 [0.0] * len(answers)，赋给 scores。
    scores = [0.0] * len(answers)
    # 逐行讲解：构造列表 []，赋给 matches。
    matches = []
    # 逐行讲解：调用 random.Random（random.Random(judge.seed)），赋给 rng。
    rng = random.Random(judge.seed)
    # 逐行讲解：遍历 range(1, TOURNAMENT_REPEATS + 1)，将每项绑定到 repeat。
    for repeat in range(1, TOURNAMENT_REPEATS + 1):
        # 逐行讲解：调用 list（list(range(len(answers)))），赋给 candidates。
        candidates = list(range(len(answers)))
        # 逐行讲解：取值 0，赋给 round_index。
        round_index = 0
        # 逐行讲解：当 len(candidates) > 1 成立时继续循环。
        while len(candidates) > 1:
            # 逐行讲解：更新 round_index：round_index += 1。
            round_index += 1
            # 逐行讲解：打乱数据顺序（rng.shuffle(candidates)）。
            rng.shuffle(candidates)
            # 作者按轮转方式分组：第 i 个候选进第 i % num_groups 组，人数为奇数时会出现 3 人组。
            num_groups = len(candidates) // PLAYERS_PER_MATCH
            # 逐行讲解：通过推导式逐项生成结果：[candidates[i::num_groups] for i in range(num_groups)]，赋给 groups。
            groups = [candidates[i::num_groups] for i in range(num_groups)]
            # 逐行讲解：构造列表 []，赋给 winners。
            winners = []
            # 逐行讲解：遍历 groups，将每项绑定到 group。
            for group in groups:
                # 逐行讲解：调用 judge.judge（judge.judge(query, rubrics, [answers[i] for i in group])），赋给 (local_index, output)。
                local_index, output = judge.judge(query, rubrics, [answers[i] for i in group])
                # 逐行讲解：按索引或切片读取 group[local_index]，赋给 winner。
                winner = group[local_index]
                # 逐行讲解：更新 scores[winner]：scores[winner] += 1.0。
                scores[winner] += 1.0
                # 逐行讲解：向列表末尾追加一个元素（winners.append(winner)）。
                winners.append(winner)
                # candidates 是 answers 的下标，可以据此还原每场比赛的 judge 输入。
                matches.append({"repeat": repeat, "round": round_index, "candidates": group,
                                "winner": winner, "output": output})
            # 逐行讲解：取值 winners，赋给 candidates。
            candidates = winners
    # 逐行讲解：返回结果：{'raw_scores': scores, 'rewards': normalize_scores(scores), 'matches': matches}。
    return {"raw_scores": scores, "rewards": normalize_scores(scores), "matches": matches}


# 逐行讲解：定义函数 score_group：给同一道题的一整组轨迹打分：总奖励 = 淘汰赛奖励 + 格式奖励（0–2）。
def score_group(
    example: ResearchExample, solutions: list[str], judge: JudgeClient,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """给同一道题的一整组轨迹打分：总奖励 = 淘汰赛奖励 + 格式奖励（0–2）。

    judge 只看提取出的最终答案。没有答案的候选仍以空字符串参赛，归一化之后再把淘汰赛奖励置 0。
    返回每个候选的奖励明细和本组的 judge 日志。
    """
    # 逐行讲解：通过推导式逐项生成结果：[analyze_format(text) for text in solutions]，赋给 formats。
    formats = [analyze_format(text) for text in solutions]
    # 逐行讲解：通过推导式逐项生成结果：[info['rollout_answer'][:MAX_ROLLOUT_CHARS] for info in formats]，赋给 answers。
    answers = [info["rollout_answer"][:MAX_ROLLOUT_CHARS] for info in formats]
    # 逐行讲解：调用 run_tournament（run_tournament(example.query.strip(), example.rubrics, answers, judge)），赋给 tournament。
    tournament = run_tournament(example.query.strip(), example.rubrics, answers, judge)
    # 逐行讲解：构造列表 []，赋给 results。
    results = []
    # 逐行讲解：遍历 zip(formats, answers, tournament['rewards'], tournament['raw_scores'], strict=True)，将每项绑定到 (info, answer, reward, raw)。
    for info, answer, reward, raw in zip(formats, answers, tournament["rewards"], tournament["raw_scores"], strict=True):
        # 逐行讲解：按条件选择结果：reward if answer else 0.0，赋给 reward。
        reward = reward if answer else 0.0
        # 逐行讲解：向列表末尾追加一个元素（results.append({**info, 'rollout_answer': answer, 'score': reward + info['format_reward'], 'tournament_rewa…）。
        results.append({**info, "rollout_answer": answer, "score": reward + info["format_reward"],
                        "tournament_reward": reward, "tournament_raw_score": raw})
    # 逐行讲解：构造字典 {'id': example.id, 'judge_backend': judge.backend, 'judge_model': judge.model, 'judge_long_context': judge.…，赋给 log。
    log = {
        "id": example.id, "judge_backend": judge.backend, "judge_model": judge.model,
        "judge_long_context": judge.long_context, "answers": answers,
        "raw_scores": tournament["raw_scores"], "tournament_rewards": [r["tournament_reward"] for r in results],
        "format_rewards": [r["format_reward"] for r in results], "matches": tournament["matches"],
    }
    # 逐行讲解：返回结果：(results, log)。
    return results, log
