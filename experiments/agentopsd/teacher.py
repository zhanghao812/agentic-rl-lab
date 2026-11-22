"""构造 Teacher 专用技能提示，并对齐整条轨迹的对数概率。"""

# 逐行讲解：从 __future__ 导入 annotations。
from __future__ import annotations

# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：从 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 逐行讲解：从 time 导入 perf_counter。
from time import perf_counter
# 逐行讲解：从 typing 导入 Any, Sequence。
from typing import Any, Sequence

# 逐行讲解：导入模块：pytrio，别名 trio。
import pytrio as trio

# 逐行讲解：从 protocol 导入 build_prompt, teacher_initial_messages。
from protocol import build_prompt, teacher_initial_messages
# 逐行讲解：从 rollout 导入 Trajectory。
from rollout import Trajectory
# 逐行讲解：从 skills 导入 SkillProvider, SkillSelection。
from skills import SkillProvider, SkillSelection


# 逐行讲解：定义类 TeacherTrajectoryScore：一条 Teacher 请求的打分结果。
@dataclass(frozen=True)
class TeacherTrajectoryScore:
    """一条 Teacher 请求的打分结果。"""

    # 逐行讲解：声明 group_id 的类型为 str。
    group_id: str
    # 逐行讲解：声明 group_index 的类型为 int。
    group_index: int
    # 逐行讲解：声明 policy_snapshot_id 的类型为 str。
    policy_snapshot_id: str
    # 逐行讲解：声明 skill_key 的类型为 str。
    skill_key: str
    # 逐行讲解：声明 teacher_input_tokens 的类型为 int。
    teacher_input_tokens: int
    # 逐行讲解：声明 action_tokens 的类型为 int。
    action_tokens: int
    # 逐行讲解：声明 evidences 的类型为 tuple[float, ...]。
    evidences: tuple[float, ...]


# 逐行讲解：定义类 TeacherBatchStats：一次更新中的 Teacher 请求与 token 统计。
@dataclass(frozen=True)
class TeacherBatchStats:
    """一次更新中的 Teacher 请求与 token 统计。"""

    # 逐行讲解：声明 requests 的类型为 int。
    requests: int
    # 逐行讲解：声明 teacher_input_tokens 的类型为 int。
    teacher_input_tokens: int
    # 逐行讲解：声明 action_tokens 的类型为 int。
    action_tokens: int
    # 逐行讲解：声明 elapsed_seconds 的类型为 float。
    elapsed_seconds: float


# 逐行讲解：定义函数 build_teacher_trajectory_tokens：构造带技能的 Teacher 轨迹 token。
def build_teacher_trajectory_tokens(
    tokenizer: Any,
    trajectory: Trajectory,
    selection: SkillSelection,
) -> tuple[list[int], int]:
    """构造带技能的 Teacher 轨迹 token。"""

    # 逐行讲解：取值 trajectory.steps[0].admissible_actions，赋给 initial_actions。
    initial_actions = trajectory.steps[0].admissible_actions
    # 逐行讲解：调用 teacher_initial_messages（teacher_initial_messages(skill_text=selection.text, task=trajectory.task, observation=trajectory.initial_ob…），赋给 teacher_messages。
    teacher_messages = teacher_initial_messages(
        skill_text=selection.text,
        task=trajectory.task,
        observation=trajectory.initial_observation,
        admissible_actions=initial_actions,
        include_admissible_actions=trajectory.include_admissible_actions,
    )
    # 逐行讲解：调用 build_prompt（build_prompt(tokenizer, teacher_messages)），赋给 teacher_initial_tokens。
    teacher_initial_tokens = build_prompt(tokenizer, teacher_messages)
    # 逐行讲解：按索引或切片读取 trajectory.full_student_tokens[trajectory.student_initial_length:]，赋给 student_suffix。
    student_suffix = trajectory.full_student_tokens[
        trajectory.student_initial_length :
    ]
    # 逐行讲解：构造列表 [*teacher_initial_tokens, *student_suffix]，赋给 teacher_full_tokens。
    teacher_full_tokens = [*teacher_initial_tokens, *student_suffix]
    # 逐行讲解：返回结果：(teacher_full_tokens, len(teacher_initial_tokens))。
    return teacher_full_tokens, len(teacher_initial_tokens)


# 逐行讲解：定义函数 _score_turns：计算每一轮的 Teacher 证据。
def _score_turns(
    trajectory: Trajectory,
    teacher_logprobs: Sequence[float | None],
    teacher_initial_length: int,
) -> tuple[tuple[float, ...], int]:
    """计算每一轮的 Teacher 证据。"""

    # 逐行讲解：声明 evidences 的类型为 list[float]；构造列表 []。
    evidences: list[float] = []
    # 逐行讲解：取值 0，赋给 action_tokens。
    action_tokens = 0
    # 逐行讲解：计算 teacher_initial_length - trajectory.student_initial_length，赋给 prefix_offset。
    prefix_offset = teacher_initial_length - trajectory.student_initial_length

    # 逐行讲解：遍历 trajectory.steps，将每项绑定到 step。
    for step in trajectory.steps:
        # 逐行讲解：计算 step.action_start + prefix_offset，赋给 teacher_start。
        teacher_start = step.action_start + prefix_offset
        # 逐行讲解：计算 step.action_end + prefix_offset，赋给 teacher_end。
        teacher_end = step.action_end + prefix_offset
        # 逐行讲解：按索引或切片读取 teacher_logprobs[teacher_start:teacher_end]，赋给 values。
        values = teacher_logprobs[teacher_start:teacher_end]
        # 逐行讲解：通过推导式逐项生成结果：[float(value) for value in values]，赋给 numeric。
        numeric = [float(value) for value in values]
        # 逐行讲解：求和（sum((teacher - student for teacher, student in zip(numeric, step.logprobs)))），赋给 evidence。
        evidence = sum(
            teacher - student
            for teacher, student in zip(numeric, step.logprobs)
        )
        # 逐行讲解：取值 numeric，赋给 step.teacher_logprobs。
        step.teacher_logprobs = numeric
        # 逐行讲解：转换为浮点表示（float(evidence)），赋给 step.teacher_evidence。
        step.teacher_evidence = float(evidence)
        # 逐行讲解：向列表末尾追加一个元素（evidences.append(float(evidence))）。
        evidences.append(float(evidence))
        # 逐行讲解：更新 action_tokens：action_tokens += len(numeric)。
        action_tokens += len(numeric)
    # 逐行讲解：返回结果：(tuple(evidences), action_tokens)。
    return tuple(evidences), action_tokens


# 逐行讲解：定义异步函数 score_trajectory_async：使用 Teacher 为一条轨迹打分。
async def score_trajectory_async(
    sampling_client: Any,
    tokenizer: Any,
    trajectory: Trajectory,
    selection: SkillSelection,
    semaphore: asyncio.Semaphore,
) -> TeacherTrajectoryScore:
    """使用 Teacher 为一条轨迹打分。"""

    # 逐行讲解：调用 build_teacher_trajectory_tokens（build_teacher_trajectory_tokens(tokenizer, trajectory, selection)），赋给 (teacher_tokens, teacher_initial_length)。
    teacher_tokens, teacher_initial_length = build_teacher_trajectory_tokens(
        tokenizer,
        trajectory,
        selection,
    )
    # 逐行讲解：进入上下文管理器 semaphore，退出时执行其收尾逻辑。
    async with semaphore:
        # 逐行讲解：等待异步操作 sampling_client.compute_logprobs_async(trio.ModelInput.from_ints(teacher_tokens)) 完成并取得结果，赋给 teacher_logprobs。
        teacher_logprobs = await sampling_client.compute_logprobs_async(
            trio.ModelInput.from_ints(teacher_tokens)
        )
    # 逐行讲解：调用 _score_turns（_score_turns(trajectory, teacher_logprobs, teacher_initial_length)），赋给 (evidences, action_tokens)。
    evidences, action_tokens = _score_turns(
        trajectory,
        teacher_logprobs,
        teacher_initial_length,
    )
    # 逐行讲解：取值 selection.key，赋给 trajectory.skill_key。
    trajectory.skill_key = selection.key
    # 逐行讲解：调用 list（list(evidences)），赋给 trajectory.turn_evidences。
    trajectory.turn_evidences = list(evidences)
    # 逐行讲解：调用 len（len(teacher_tokens)），赋给 trajectory.teacher_scored_tokens。
    trajectory.teacher_scored_tokens = len(teacher_tokens)
    # 逐行讲解：返回结果：TeacherTrajectoryScore(group_id=trajectory.group_id, group_index=trajectory.group_index, policy_snapshot_id…。
    return TeacherTrajectoryScore(
        group_id=trajectory.group_id,
        group_index=trajectory.group_index,
        policy_snapshot_id=trajectory.policy_snapshot_id,
        skill_key=selection.key,
        teacher_input_tokens=len(teacher_tokens),
        action_tokens=action_tokens,
        evidences=evidences,
    )


# 逐行讲解：定义异步函数 score_trajectories_async：并发为多条轨迹进行 Teacher 打分。
async def score_trajectories_async(
    sampling_client: Any,
    tokenizer: Any,
    trajectories: Sequence[Trajectory],
    skill_provider: SkillProvider,
    *,
    max_concurrency: int = 16,
) -> tuple[list[TeacherTrajectoryScore], TeacherBatchStats]:
    """并发为多条轨迹进行 Teacher 打分。"""

    # 逐行讲解：调用 perf_counter（perf_counter()），赋给 started。
    started = perf_counter()
    # 逐行讲解：调用 asyncio.Semaphore（asyncio.Semaphore(max_concurrency)），赋给 semaphore。
    semaphore = asyncio.Semaphore(max_concurrency)
    # 逐行讲解：通过推导式逐项生成结果：[trajectory for trajectory in trajectories if trajectory.steps]，赋给 scoreable。
    scoreable = [trajectory for trajectory in trajectories if trajectory.steps]
    # 逐行讲解：调用 list（list(await asyncio.gather(*(score_trajectory_async(sampling_client, tokenizer, trajectory, skill_provider.r…），赋给 scores。
    scores = list(
        await asyncio.gather(
            *(
                score_trajectory_async(
                    sampling_client,
                    tokenizer,
                    trajectory,
                    skill_provider.resolve(trajectory.example.task_type),
                    semaphore,
                )
                for trajectory in scoreable
            )
        )
    )
    # 逐行讲解：调用 TeacherBatchStats（TeacherBatchStats(requests=len(scores), teacher_input_tokens=sum((item.teacher_input_tokens for item in sco…），赋给 stats。
    stats = TeacherBatchStats(
        requests=len(scores),
        teacher_input_tokens=sum(item.teacher_input_tokens for item in scores),
        action_tokens=sum(item.action_tokens for item in scores),
        elapsed_seconds=perf_counter() - started,
    )
    # 逐行讲解：返回结果：(scores, stats)。
    return scores, stats
