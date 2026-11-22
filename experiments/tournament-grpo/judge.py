"""Tournament judge：作者的 OpenAI 兼容 chat 协议，以及本复现使用的 Jev（System One API）。

Adapted from PuffYang/Tournament_GRPO (eea826cf), Apache-2.0.

检查已保存的 judge 配置（仓库根目录）：
uv run python experiments/tournament-grpo/judge.py
"""

# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：os。
import os
# 逐行讲解：导入模块：re。
import re
# 逐行讲解：导入模块：time。
import time
# 逐行讲解：从 dataclasses 导入 dataclass, field。
from dataclasses import dataclass, field
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 typing 导入 Any。
from typing import Any

# 逐行讲解：导入模块：httpx。
import httpx
# 逐行讲解：从 dotenv 导入 load_dotenv。
from dotenv import load_dotenv


# 逐行讲解：取值 'https://api.typesafe.ai/v1/systemone'，赋给 JEV_API_URL。
JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
# 逐行讲解：取值 3，赋给 MAX_ATTEMPTS。
MAX_ATTEMPTS = 3  # 每场比赛的请求次数上限（含首次）
# 作者 judge prompt 的两段原文，chat 和 Jev 共用。
JUDGE_ROLE = "You are an expert judge for deep-research quality.\n"
# 逐行讲解：取值 'Selection rule:\n- Select exactly 1 winner(s).\n- Prefer responses that are accurate, complete, well-suppo…，赋给 SELECTION_RULE。
SELECTION_RULE = (
    "Selection rule:\n"
    "- Select exactly 1 winner(s).\n"
    "- Prefer responses that are accurate, complete, well-supported, and directly answer the query.\n"
)
# Jev 把比较对象放在 state、选择规则放在 instructions，所以不需要 JSON 输出要求。
JEV_INSTRUCTIONS = (
    "You are a strict deep-research tournament judge.\n" + JUDGE_ROLE
    + "Compare the candidate responses to the same query using the provided rubric.\n\n"
    + SELECTION_RULE.rstrip("\n")
)


# 逐行讲解：定义函数 build_judge_context：按作者格式拼出题目、rubric 和候选答案。
def build_judge_context(query: str, rubrics: list[dict[str, str]], answers: list[str]) -> str:
    """按作者格式拼出题目、rubric 和候选答案。"""
    # 逐行讲解：通过推导式逐项生成结果：[f"{i}. description: {item['description'].strip()}\n   dimension: {item['dimension'].strip()}\n   importanc…，赋给 rubric_lines。
    rubric_lines = [
        f"{i}. description: {item['description'].strip()}\n"
        f"   dimension: {item['dimension'].strip()}\n"
        f"   importance: {item['importance'].strip()}\n"
        f"   title: {item['title'].strip()}"
        for i, item in enumerate(rubrics, 1)
    ]
    # 逐行讲解：按条件选择结果：'\n'.join(rubric_lines) if rubric_lines else 'No rubrics provided.'，赋给 rubric_block。
    rubric_block = "\n".join(rubric_lines) if rubric_lines else "No rubrics provided."
    # 逐行讲解：调用 '\n\n'.join（'\n\n'.join((f'Candidate {i}:\n{text}' for i, text in enumerate(answers, 1)))），赋给 candidates。
    candidates = "\n\n".join(f"Candidate {i}:\n{text}" for i, text in enumerate(answers, 1))
    # 逐行讲解：返回结果：f'Query:\n{query}\n\nRubric (description, dimension, importance, title):\n{rubric_block}\n\nCandidates:\n{c…。
    return (f"Query:\n{query}\n\nRubric (description, dimension, importance, title):\n"
            f"{rubric_block}\n\nCandidates:\n{candidates}\n\n")


# 逐行讲解：定义函数 build_chat_prompt：作者的 chat judge prompt 原文。
def build_chat_prompt(context: str) -> str:
    """作者的 chat judge prompt 原文。"""
    # 逐行讲解：返回结果：JUDGE_ROLE + 'Compare the candidate responses to the same query using the rubric below.\n\n' + context + SE…。
    return (
        JUDGE_ROLE + "Compare the candidate responses to the same query using the rubric below.\n\n"
        + context + SELECTION_RULE
        + "- Return JSON only, without markdown or explanations.\n"
        + '- Use 1-based candidate numbers with this schema: {"winners": [1]}\n'
    )


# 逐行讲解：定义函数 parse_winner：沿用作者宽松的解析：优先读 JSON 里的 winners，否则取文本中的数字；候选编号从 1 开始。
def parse_winner(content: str, num_candidates: int) -> int:
    """沿用作者宽松的解析：优先读 JSON 里的 winners，否则取文本中的数字；候选编号从 1 开始。"""
    # 逐行讲解：调用 content.strip（content.strip()），赋给 cleaned。
    cleaned = content.strip()
    # 逐行讲解：检查条件 cleaned.startswith('```')；成立时执行下方分支。
    if cleaned.startswith("```"):
        # 逐行讲解：调用 cleaned.strip('`').removeprefix('json').strip（cleaned.strip('`').removeprefix('json').strip()），赋给 cleaned。
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
    try:
        # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(cleaned)），赋给 parsed。
        parsed = json.loads(cleaned)
    except ValueError:
        # 逐行讲解：取值 None，赋给 parsed。
        parsed = None
    # 逐行讲解：检查条件 isinstance(parsed, dict)；成立时执行下方分支。
    if isinstance(parsed, dict):
        # 逐行讲解：查询指定键，缺失时使用默认值（parsed.get('winners', parsed.get('winner', []))），赋给 winners。
        winners = parsed.get("winners", parsed.get("winner", []))
    # 逐行讲解：检查条件 isinstance(parsed, list)；成立时执行下方分支。
    elif isinstance(parsed, list):
        # 逐行讲解：取值 parsed，赋给 winners。
        winners = parsed
    # 逐行讲解：检查条件 isinstance(parsed, (int, float))；成立时执行下方分支。
    elif isinstance(parsed, (int, float)):
        # 逐行讲解：构造列表 [parsed]，赋给 winners。
        winners = [parsed]
    else:
        # 逐行讲解：调用 re.findall（re.findall('\\d+', cleaned)），赋给 winners。
        winners = re.findall(r"\d+", cleaned)
    # 逐行讲解：检查条件 not isinstance(winners, list)；成立时执行下方分支。
    if not isinstance(winners, list):
        # 逐行讲解：构造列表 [winners]，赋给 winners。
        winners = [winners]
    # 逐行讲解：遍历 winners，将每项绑定到 value。
    for value in winners:
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：计算 int(value) - 1，赋给 index。
            index = int(value) - 1
        except (ValueError, TypeError, OverflowError):
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：检查条件 0 <= index < num_candidates；成立时执行下方分支。
        if 0 <= index < num_candidates:
            # 逐行讲解：返回结果：index。
            return index
    # 逐行讲解：抛出异常：ValueError(f'Tournament judge returned invalid winners: {cleaned}')。
    raise ValueError(f"Tournament judge returned invalid winners: {cleaned}")


# 逐行讲解：定义类 JudgeClient：每场比赛选 1 个胜者。
@dataclass
class JudgeClient:
    """每场比赛选 1 个胜者。"""

    # 逐行讲解：声明 api_url 的类型为 str。
    api_url: str
    # 逐行讲解：声明 model 的类型为 str。
    model: str
    # 逐行讲解：声明 api_key 的类型为 str；调用 field（field(default='', repr=False)）。
    api_key: str = field(default="", repr=False)
    # 逐行讲解：声明 backend 的类型为 str；取值 'chat'。
    backend: str = "chat"  # chat 是作者的接口；本复现用 jev
    # 逐行讲解：声明 long_context 的类型为 bool；取值 False。
    long_context: bool = False
    # 逐行讲解：声明 seed 的类型为 int；取值 0。
    seed: int = 0  # 作者用同一个 seed 控制 judge 采样和淘汰赛分组
    # 逐行讲解：声明 timeout 的类型为 float；取值 240.0。
    timeout: float = 240.0

    # 逐行讲解：定义函数 from_env：封装下方逻辑，供调用方复用。
    @classmethod
    def from_env(cls, env_path: str | Path) -> "JudgeClient":
        # 逐行讲解：调用 load_dotenv（load_dotenv(env_path)）。
        load_dotenv(env_path)
        # 逐行讲解：调用 os.getenv（os.getenv('TOURNAMENT_JUDGE_BACKEND', 'chat')），赋给 backend。
        backend = os.getenv("TOURNAMENT_JUDGE_BACKEND", "chat")
        # 逐行讲解：取值 backend == 'jev'，赋给 jev。
        jev = backend == "jev"
        # 逐行讲解：调用 cls（cls(api_url=os.getenv('TOURNAMENT_JUDGE_API_URL') or (JEV_API_URL if jev else ''), model=os.getenv('TOURNAM…），赋给 judge。
        judge = cls(
            api_url=os.getenv("TOURNAMENT_JUDGE_API_URL") or (JEV_API_URL if jev else ""),
            model=os.getenv("TOURNAMENT_JUDGE_MODEL_NAME") or ("jev-1.14.0" if jev else "Qwen2.5-72B-Instruct"),
            api_key=os.getenv("TOURNAMENT_JUDGE_API_KEY", ""),
            backend=backend,
            long_context=os.getenv("TOURNAMENT_JUDGE_LONG_CONTEXT", "false").lower() == "true",
        )
        # 启动前检查配置，避免采样完成后才发现 judge 不可用。
        if not judge.api_url or (jev and not judge.api_key):
            # 逐行讲解：抛出异常：ValueError('judge 未配置：请在 experiments/tournament-grpo/.env 中填写 TOURNAMENT_JUDGE_*')。
            raise ValueError("judge 未配置：请在 experiments/tournament-grpo/.env 中填写 TOURNAMENT_JUDGE_*")
        # 逐行讲解：返回结果：judge。
        return judge

    # 逐行讲解：定义函数 judge：返回 (组内胜者下标, judge 原始输出)。
    def judge(self, query: str, rubrics: list[dict[str, str]], answers: list[str]) -> tuple[int, Any]:
        """返回 (组内胜者下标, judge 原始输出)。

        作者在重试失败后随机选胜者；本实现直接抛出异常停止训练，避免用错误的奖励更新。
        """
        # 逐行讲解：调用 build_judge_context（build_judge_context(query, rubrics, answers)），赋给 context。
        context = build_judge_context(query, rubrics, answers)
        # 逐行讲解：遍历 range(1, MAX_ATTEMPTS + 1)，将每项绑定到 attempt。
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
            try:
                # 逐行讲解：检查条件 self.backend == 'jev'；成立时执行下方分支。
                if self.backend == "jev":
                    # 逐行讲解：返回结果：self.ask_jev(context, len(answers))。
                    return self.ask_jev(context, len(answers))
                # 逐行讲解：返回结果：self.ask_chat(build_chat_prompt(context), len(answers))。
                return self.ask_chat(build_chat_prompt(context), len(answers))
            except Exception as exc:
                # 逐行讲解：调用 self.describe_error（self.describe_error(exc)），赋给 error。
                error = self.describe_error(exc)
                # 逐行讲解：检查条件 attempt < MAX_ATTEMPTS；成立时执行下方分支。
                if attempt < MAX_ATTEMPTS:
                    # 逐行讲解：调用 time.sleep（time.sleep(1.5 * attempt)）。
                    time.sleep(1.5 * attempt)
        # 逐行讲解：抛出异常：RuntimeError(f'Tournament judge failed after {MAX_ATTEMPTS} attempts; stopping training. error={error}')。
        raise RuntimeError(
            f"Tournament judge failed after {MAX_ATTEMPTS} attempts; stopping training. error={error}"
        ) from None

    # 逐行讲解：定义函数 describe_error：保留 HTTP 状态码和响应正文，并去掉其中可能出现的 Key。
    def describe_error(self, exc: Exception) -> str:
        """保留 HTTP 状态码和响应正文，并去掉其中可能出现的 Key。"""
        # 逐行讲解：检查条件 isinstance(exc, httpx.HTTPStatusError)；成立时执行下方分支。
        if isinstance(exc, httpx.HTTPStatusError):
            # 逐行讲解：取值 f'HTTP {exc.response.status_code}: {exc.response.text[:1500]}'，赋给 error。
            error = f"HTTP {exc.response.status_code}: {exc.response.text[:1500]}"
        else:
            # 逐行讲解：调用 str（str(exc)），赋给 error。
            error = str(exc)
        # 逐行讲解：返回结果：error.replace(self.api_key, '[REDACTED]') if self.api_key else error。
        return error.replace(self.api_key, "[REDACTED]") if self.api_key else error

    # 逐行讲解：定义函数 headers：封装下方逻辑，供调用方复用。
    def headers(self) -> dict[str, str]:
        # 逐行讲解：构造字典 {'Content-Type': 'application/json'}，赋给 headers。
        headers = {"Content-Type": "application/json"}
        # 逐行讲解：检查条件 self.api_key；成立时执行下方分支。
        if self.api_key:
            # 逐行讲解：取值 f'Bearer {self.api_key}'，赋给 headers['Authorization']。
            headers["Authorization"] = f"Bearer {self.api_key}"
        # 逐行讲解：检查条件 self.backend == 'jev' and self.long_context；成立时执行下方分支。
        if self.backend == "jev" and self.long_context:
            # 逐行讲解：取值 'true'，赋给 headers['TypeSafe-Danger-Zone-Long-Context']。
            headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
        # 逐行讲解：返回结果：headers。
        return headers

    # 逐行讲解：定义函数 ask_chat：作者的 chat 请求：temperature 0、max_tokens 1024。
    def ask_chat(self, prompt: str, num_candidates: int) -> tuple[int, str]:
        """作者的 chat 请求：temperature 0、max_tokens 1024。"""
        # 逐行讲解：构造字典 {'model': self.model, 'messages': [{'role': 'system', 'content': 'You are a strict deep-research tournament…，赋给 payload。
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict deep-research tournament judge. Return only valid JSON without markdown."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0, "max_tokens": 1024, "top_p": 0.95, "seed": self.seed,
        }
        # 逐行讲解：调用 httpx.post（httpx.post(self.api_url, headers=self.headers(), json=payload, timeout=self.timeout, trust_env=False)），赋给 response。
        response = httpx.post(self.api_url, headers=self.headers(), json=payload, timeout=self.timeout, trust_env=False)
        # 逐行讲解：调用 response.raise_for_status（response.raise_for_status()）。
        response.raise_for_status()
        # 逐行讲解：调用 (response.json()['choices'][0]['message'].get('content') or '').strip（(response.json()['choices'][0]['message'].get('content') or '').strip()），赋给 content。
        content = (response.json()["choices"][0]["message"].get("content") or "").strip()
        # 逐行讲解：返回结果：(parse_winner(content, num_candidates), content)。
        return parse_winner(content, num_candidates), content

    # 逐行讲解：定义函数 ask_jev：Jev 的 choice 问题直接返回一个候选编号，并附带 token usage。
    def ask_jev(self, context: str, num_candidates: int) -> tuple[int, dict[str, Any]]:
        """Jev 的 choice 问题直接返回一个候选编号，并附带 token usage。"""
        # 逐行讲解：构造字典 {'model': self.model, 'state': context, 'questions': {'winner': {'type': 'choice', 'instructions': JEV_INST…，赋给 payload。
        payload = {
            "model": self.model, "state": context,
            "questions": {"winner": {
                "type": "choice", "instructions": JEV_INSTRUCTIONS,
                "criteria": {str(i): f"Candidate {i}" for i in range(1, num_candidates + 1)},
            }},
        }
        # 逐行讲解：调用 httpx.post（httpx.post(self.api_url, headers=self.headers(), json=payload, timeout=self.timeout, trust_env=False)），赋给 response。
        response = httpx.post(self.api_url, headers=self.headers(), json=payload, timeout=self.timeout, trust_env=False)
        # 逐行讲解：调用 response.raise_for_status（response.raise_for_status()）。
        response.raise_for_status()
        # 逐行讲解：调用 response.json（response.json()），赋给 body。
        body = response.json()
        # 逐行讲解：按索引或切片读取 body['answers']['winner']，赋给 answer。
        answer = body["answers"]["winner"]
        # 逐行讲解：计算 int(answer['choice']) - 1，赋给 index。
        index = int(answer["choice"]) - 1
        # 越界编号会被 Python 负下标悄悄接受，必须当作失败重试。
        if not 0 <= index < num_candidates:
            # 逐行讲解：抛出异常：ValueError(f"Jev returned invalid choice: {answer['choice']}")。
            raise ValueError(f"Jev returned invalid choice: {answer['choice']}")
        # 逐行讲解：返回结果：(index, {'model': body['model'], 'answer': answer, 'usage': body.get('usage', {})})。
        return index, {"model": body["model"], "answer": answer, "usage": body.get("usage", {})}


# 逐行讲解：定义函数 main：用一道二选一的小题发一次真实请求，确认鉴权、模型和返回解析正常。
def main() -> None:
    """用一道二选一的小题发一次真实请求，确认鉴权、模型和返回解析正常。"""
    # 逐行讲解：调用 JudgeClient.from_env（JudgeClient.from_env(Path(__file__).resolve().parent / '.env')），赋给 judge。
    judge = JudgeClient.from_env(Path(__file__).resolve().parent / ".env")
    # 逐行讲解：调用 judge.judge（judge.judge('What is 2 + 2?', [], ['4', '5'])），赋给 (winner, output)。
    winner, output = judge.judge("What is 2 + 2?", [], ["4", "5"])
    # 逐行讲解：调用 print（print(f'{judge.backend} / {judge.model} 调用成功，选中 Candidate {winner + 1}。')）。
    print(f"{judge.backend} / {judge.model} 调用成功，选中 Candidate {winner + 1}。")
    # 逐行讲解：调用 print（print(json.dumps(output, ensure_ascii=False, indent=2))）。
    print(json.dumps(output, ensure_ascii=False, indent=2))


# 逐行讲解：检查条件 __name__ == '__main__'；成立时执行下方分支。
if __name__ == "__main__":
    # 逐行讲解：调用 main（main()）。
    main()
