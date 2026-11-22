"""Deep Research Bench RACE：作者评分提示词、权重和相对参考报告的归一化。

Adapted from PuffYang/Tournament_GRPO (eea826cf), deep_research_bench/
ood_race_eval.py and utils/score_calculator.py. Prompt license: race_prompts/LICENSE.
"""

# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：math。
import math
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

# 逐行讲解：导入模块：httpx。
import httpx
# 逐行讲解：从 dotenv 导入 load_dotenv。
from dotenv import load_dotenv


# 逐行讲解：构造元组 ('comprehensiveness', 'insight', 'instruction_following', 'readability')，赋给 RACE_DIMS。
RACE_DIMS = ("comprehensiveness", "insight", "instruction_following", "readability")
# 逐行讲解：计算 Path(__file__).resolve().parent / 'race_prompts'，赋给 PROMPT_DIR。
PROMPT_DIR = Path(__file__).resolve().parent / "race_prompts"


# 逐行讲解：定义类 ThinkingEnabledError：The judge returned reasoning despite disabled thinking.
class ThinkingEnabledError(RuntimeError):
    """The judge returned reasoning despite disabled thinking."""


# 逐行讲解：定义函数 build_score_prompt：作者原始中英文模板；裁判看细则描述，权重只在本地聚合时使用。
def build_score_prompt(article: dict, reference: dict, criteria: dict) -> str:
    """作者原始中英文模板；裁判看细则描述，权重只在本地聚合时使用。"""
    # 逐行讲解：读取文件文本（(PROMPT_DIR / ('zh.txt' if article['language'] == 'zh' else 'en.txt')).read_text()），赋给 template。
    template = (PROMPT_DIR / ("zh.txt" if article["language"] == "zh" else "en.txt")).read_text()
    # 逐行讲解：通过推导式逐项生成结果：{dim: [{'criterion': c['criterion'], 'explanation': c['explanation']} for c in items] for dim, items in cri…，赋给 descriptions。
    descriptions = {dim: [{"criterion": c["criterion"], "explanation": c["explanation"]}
                          for c in items] for dim, items in criteria["criterions"].items()}
    # 逐行讲解：返回结果：template.format(task_prompt=article['prompt'], article_1=article['article'], article_2=reference['article']…。
    return template.format(task_prompt=article["prompt"], article_1=article["article"],
                           article_2=reference["article"],
                           criteria_list=json.dumps(descriptions, ensure_ascii=False, indent=2))


# 逐行讲解：定义函数 normalized_scores：先按细则加权，再按维度加权；Overall 不是四项归一化分数的简单平均。
def normalized_scores(output: dict, criteria: dict) -> dict[str, float]:
    """先按细则加权，再按维度加权；Overall 不是四项归一化分数的简单平均。"""
    # 逐行讲解：构造列表 [0.0, 0.0]，赋给 totals。
    totals = [0.0, 0.0]
    # 逐行讲解：构造字典 {}，赋给 scores。
    scores = {}
    # 逐行讲解：遍历 RACE_DIMS，将每项绑定到 dim。
    for dim in RACE_DIMS:
        # 逐行讲解：按索引或切片读取 output[dim]，赋给 items。
        items = output[dim]
        # 逐行讲解：检查条件 not items or len(items) != len(criteria['criterions'][dim])；成立时执行下方分支。
        if not items or len(items) != len(criteria["criterions"][dim]):
            # 逐行讲解：抛出异常：ValueError(f'Incomplete RACE criteria: {dim}')。
            raise ValueError(f"Incomplete RACE criteria: {dim}")
        # 逐行讲解：通过推导式逐项生成结果：{c['criterion']: c['weight'] for c in criteria['criterions'][dim]}，赋给 weights。
        weights = {c["criterion"]: c["weight"] for c in criteria["criterions"][dim]}
        # 逐行讲解：构造元组 ([0.0, 0.0], 0.0)，赋给 (sums, weight_sum)。
        sums, weight_sum = [0.0, 0.0], 0.0
        # 逐行讲解：遍历 items，将每项绑定到 item。
        for item in items:
            # 逐行讲解：调用 item['criterion'].strip（item['criterion'].strip()），赋给 label。
            label = item["criterion"].strip()
            # 作者的匹配顺序：原文、忽略大小写、子串，最后用该维度平均权重。
            weight = weights.get(label)
            # 逐行讲解：检查条件 weight is None；成立时执行下方分支。
            if weight is None:
                # 逐行讲解：调用 next（next((v for k, v in weights.items() if k.lower() == label.lower()), None)），赋给 weight。
                weight = next((v for k, v in weights.items() if k.lower() == label.lower()), None)
            # 逐行讲解：检查条件 weight is None；成立时执行下方分支。
            if weight is None:
                # 逐行讲解：调用 next（next((v for k, v in weights.items() if label.lower() in k.lower() or k.lower() in label.lower()), None)），赋给 weight。
                weight = next((v for k, v in weights.items()
                               if label.lower() in k.lower() or k.lower() in label.lower()), None)
            # 逐行讲解：检查条件 weight is None；成立时执行下方分支。
            if weight is None:
                # 逐行讲解：计算 sum(weights.values()) / len(weights)，赋给 weight。
                weight = sum(weights.values()) / len(weights)
            # 逐行讲解：遍历 range(2)，将每项绑定到 i。
            for i in range(2):
                # 逐行讲解：转换为浮点表示（float(item[f'article_{i + 1}_score'])），赋给 value。
                value = float(item[f"article_{i + 1}_score"])
                # 逐行讲解：检查条件 not math.isfinite(value) or not 0 <= value <= 10；成立时执行下方分支。
                if not math.isfinite(value) or not 0 <= value <= 10:
                    # 逐行讲解：抛出异常：ValueError(f'Invalid RACE score: {dim}: {value}')。
                    raise ValueError(f"Invalid RACE score: {dim}: {value}")
                # 逐行讲解：更新 sums[i]：sums[i] += value * weight。
                sums[i] += value * weight
            # 逐行讲解：更新 weight_sum：weight_sum += weight。
            weight_sum += weight
        # 逐行讲解：通过推导式逐项生成结果：[value / weight_sum for value in sums]，赋给 averages。
        averages = [value / weight_sum for value in sums]
        # 逐行讲解：按条件选择结果：averages[0] / sum(averages) if sum(averages) > 0 else 0.0，赋给 scores[dim]。
        scores[dim] = averages[0] / sum(averages) if sum(averages) > 0 else 0.0
        # 逐行讲解：遍历 range(2)，将每项绑定到 i。
        for i in range(2):
            # 逐行讲解：更新 totals[i]：totals[i] += averages[i] * criteria['dimension_weight'][dim]。
            totals[i] += averages[i] * criteria["dimension_weight"][dim]
    # 逐行讲解：按条件选择结果：totals[0] / sum(totals) if sum(totals) > 0 else 0.0，赋给 scores['overall_score']。
    scores["overall_score"] = totals[0] / sum(totals) if sum(totals) > 0 else 0.0
    # 逐行讲解：返回结果：scores。
    return scores


# 逐行讲解：定义类 RaceJudge：独立的 OpenAI 兼容评分接口，不使用 tournament 的胜者选择接口。
@dataclass
class RaceJudge:
    """独立的 OpenAI 兼容评分接口，不使用 tournament 的胜者选择接口。"""

    # 逐行讲解：声明 api_url 的类型为 str。
    api_url: str
    # 逐行讲解：声明 model 的类型为 str。
    model: str
    # 逐行讲解：声明 api_key 的类型为 str；调用 field（field(default='', repr=False)）。
    api_key: str = field(default="", repr=False)
    # 逐行讲解：声明 timeout 的类型为 float；取值 600.0。
    timeout: float = 600.0
    # 逐行讲解：声明 max_tokens 的类型为 int；取值 8192。
    max_tokens: int = 8192

    # 逐行讲解：定义函数 from_env：封装下方逻辑，供调用方复用。
    @classmethod
    def from_env(cls, path: Path):
        # 逐行讲解：调用 load_dotenv（load_dotenv(path)）。
        load_dotenv(path)
        # 逐行讲解：构造元组 (os.getenv('RACE_JUDGE_API_URL', ''), os.getenv('RACE_JUDGE_MODEL_NAME', ''))，赋给 (url, model)。
        url, model = os.getenv("RACE_JUDGE_API_URL", ""), os.getenv("RACE_JUDGE_MODEL_NAME", "")
        # 逐行讲解：检查条件 not url or not model；成立时执行下方分支。
        if not url or not model:
            # 逐行讲解：抛出异常：ValueError('请配置 RACE_JUDGE_API_URL 和 RACE_JUDGE_MODEL_NAME；报告生成不需要裁判')。
            raise ValueError("请配置 RACE_JUDGE_API_URL 和 RACE_JUDGE_MODEL_NAME；报告生成不需要裁判")
        # 逐行讲解：返回结果：cls(url, model, os.getenv('RACE_JUDGE_API_KEY', ''))。
        return cls(url, model, os.getenv("RACE_JUDGE_API_KEY", ""))

    # 逐行讲解：定义函数 score：封装下方逻辑，供调用方复用。
    def score(self, article: dict, reference: dict, criteria: dict) -> dict:
        # 逐行讲解：调用 build_score_prompt（build_score_prompt(article, reference, criteria)），赋给 prompt。
        prompt = build_score_prompt(article, reference, criteria)
        # 逐行讲解：调用 self.api_url.rstrip('/').endswith（self.api_url.rstrip('/').endswith('/responses')），赋给 responses。
        responses = self.api_url.rstrip("/").endswith("/responses")
        # 逐行讲解：检查条件 responses；成立时执行下方分支。
        if responses:
            # 逐行讲解：构造字典 {'model': self.model, 'input': prompt, 'max_output_tokens': self.max_tokens, 'reasoning': {'effort': 'none'…，赋给 payload。
            payload = {"model": self.model, "input": prompt, "max_output_tokens": self.max_tokens,
                       "reasoning": {"effort": "none"}, "temperature": 0.0, "top_p": 0.95}
        else:
            # 逐行讲解：构造字典 {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.0, 'top_p': 0.95,…，赋给 payload。
            payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "top_p": 0.95, "seed": 0, "max_tokens": self.max_tokens}
            # 逐行讲解：检查条件 'deepseek' in self.model.lower()；成立时执行下方分支。
            if "deepseek" in self.model.lower():
                # 逐行讲解：调用 payload.update（payload.update(thinking={'type': 'disabled'}, reasoning_effort='none')）。
                payload.update(thinking={"type": "disabled"}, reasoning_effort="none")
        # 逐行讲解：按条件选择结果：{'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}，赋给 headers。
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # 逐行讲解：遍历 range(3)，将每项绑定到 attempt。
        for attempt in range(3):
            # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
            try:
                # 逐行讲解：调用 httpx.post（httpx.post(self.api_url, json=payload, headers=headers, timeout=self.timeout, trust_env=False)），赋给 response。
                response = httpx.post(self.api_url, json=payload, headers=headers,
                                      timeout=self.timeout, trust_env=False)
                # 逐行讲解：调用 response.raise_for_status（response.raise_for_status()）。
                response.raise_for_status()
                # 逐行讲解：调用 response.json（response.json()），赋给 body。
                body = response.json()
                # 逐行讲解：检查条件 responses；成立时执行下方分支。
                if responses:
                    # 逐行讲解：通过推导式逐项生成结果：[item for item in body.get('output', []) if item['type'] == 'reasoning']，赋给 reasoning。
                    reasoning = [item for item in body.get("output", []) if item["type"] == "reasoning"]
                    # 逐行讲解：调用 any（any((part.get('text') for item in reasoning for part in item.get('summary', []) + item.get('content', [])))），赋给 thinking。
                    thinking = any(part.get("text") for item in reasoning
                                   for part in item.get("summary", []) + item.get("content", []))
                    # 逐行讲解：更新 thinking：thinking |= bool(body.get('usage', {}).get('output_tokens_details', {}).get('reasoning_tokens', 0))。
                    thinking |= bool(body.get("usage", {}).get("output_tokens_details", {}).get("reasoning_tokens", 0))
                    # 逐行讲解：检查条件 thinking；成立时执行下方分支。
                    if thinking:
                        # 逐行讲解：抛出异常：ThinkingEnabledError('RACE judge returned reasoning with thinking disabled')。
                        raise ThinkingEnabledError("RACE judge returned reasoning with thinking disabled")
                    # 逐行讲解：检查条件 body.get('status') != 'completed'；成立时执行下方分支。
                    if body.get("status") != "completed":
                        # 逐行讲解：抛出异常：ValueError(f"RACE response incomplete: {body.get('incomplete_details')}")。
                        raise ValueError(f"RACE response incomplete: {body.get('incomplete_details')}")
                    # 逐行讲解：调用 ''.join（''.join((part['text'] for item in body['output'] if item['type'] == 'message' for part in item['content'] i…），赋给 text。
                    text = "".join(part["text"] for item in body["output"] if item["type"] == "message"
                                   for part in item["content"] if part["type"] == "output_text")
                    # 逐行讲解：按索引或切片读取 body['status']，赋给 finish_reason。
                    finish_reason = body["status"]
                else:
                    # 逐行讲解：按索引或切片读取 body['choices'][0]，赋给 choice。
                    choice = body["choices"][0]
                    # 逐行讲解：检查条件 choice['message'].get('reasoning_content')；成立时执行下方分支。
                    if choice["message"].get("reasoning_content"):
                        # 逐行讲解：抛出异常：ThinkingEnabledError('RACE judge returned reasoning with thinking disabled')。
                        raise ThinkingEnabledError("RACE judge returned reasoning with thinking disabled")
                    # 逐行讲解：查询指定键，缺失时使用默认值（choice.get('finish_reason')），赋给 finish_reason。
                    finish_reason = choice.get("finish_reason")
                    # 逐行讲解：检查条件 finish_reason in {'length', 'MAX_TOKENS'}；成立时执行下方分支。
                    if finish_reason in {"length", "MAX_TOKENS"}:
                        # 逐行讲解：抛出异常：ValueError('RACE judge output truncated; no score recorded')。
                        raise ValueError("RACE judge output truncated; no score recorded")
                    # 逐行讲解：按索引或切片读取 choice['message']['content']，赋给 text。
                    text = choice["message"]["content"]
                # 逐行讲解：调用 re.search（re.search('```(?:json)?\\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)），赋给 fenced。
                fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
                # 逐行讲解：按条件选择结果：fenced.group(1) if fenced else text[text.find('{'):text.rfind('}') + 1]，赋给 content。
                content = fenced.group(1) if fenced else text[text.find("{"):text.rfind("}") + 1]
                # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(content)），赋给 parsed。
                parsed = json.loads(content)
                # 逐行讲解：调用 normalized_scores（normalized_scores(parsed, criteria)），赋给 scores。
                scores = normalized_scores(parsed, criteria)
                # 逐行讲解：返回结果：{'id': article['id'], 'prompt': article['prompt'], **scores, 'judge_output': parsed, 'judge_raw_text': text…。
                return {"id": article["id"], "prompt": article["prompt"], **scores,
                        "judge_output": parsed, "judge_raw_text": text,
                        "judge_model": body.get("model", self.model), "judge_usage": body.get("usage", {}),
                        "judge_finish_reason": finish_reason, "judge_thinking": "disabled"}
            except ThinkingEnabledError:
                # 逐行讲解：抛出异常：重新抛出当前异常。
                raise
            except Exception as exc:
                # 逐行讲解：按条件选择结果：str(exc).replace(self.api_key, '[REDACTED]') if self.api_key else str(exc)，赋给 error。
                error = str(exc).replace(self.api_key, "[REDACTED]") if self.api_key else str(exc)
                # 逐行讲解：检查条件 attempt < 2；成立时执行下方分支。
                if attempt < 2:
                    # 逐行讲解：调用 time.sleep（time.sleep(1.5 * (attempt + 1))）。
                    time.sleep(1.5 * (attempt + 1))
        # 逐行讲解：抛出异常：RuntimeError(f'RACE judge failed after 3 attempts: {error}')。
        raise RuntimeError(f"RACE judge failed after 3 attempts: {error}") from None
