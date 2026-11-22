"""提供 DeepSeek 官方 Search API 和 Wikipedia 两个搜索后端。"""

# 逐行讲解：导入模块：gzip。
import gzip
# 逐行讲解：导入模块：json。
import json
# 逐行讲解：导入模块：os。
import os
# 逐行讲解：导入模块：socket。
import socket
# 逐行讲解：导入模块：threading。
import threading
# 逐行讲解：导入模块：time。
import time
# 逐行讲解：导入模块：urllib.error。
import urllib.error
# 逐行讲解：导入模块：urllib.parse。
import urllib.parse
# 逐行讲解：导入模块：urllib.request。
import urllib.request
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


# 逐行讲解：构造元组 ('deepseek', 'wikipedia')，赋给 SEARCH_BACKENDS。
SEARCH_BACKENDS = ("deepseek", "wikipedia")
# 逐行讲解：取值 'https://api.deepseek.com/v1/tools/search'，赋给 DEEPSEEK_SEARCH_ENDPOINT。
DEEPSEEK_SEARCH_ENDPOINT = "https://api.deepseek.com/v1/tools/search"
# 逐行讲解：取值 'https://en.wikipedia.org/w/api.php'，赋给 WIKIPEDIA_SEARCH_ENDPOINT。
WIKIPEDIA_SEARCH_ENDPOINT = "https://en.wikipedia.org/w/api.php"
# 逐行讲解：取值 'agentic-rl-lab/0.1 (https://github.com/datawhalechina/agentic-rl-lab)'，赋给 WIKIPEDIA_USER_AGENT。
WIKIPEDIA_USER_AGENT = (
    "agentic-rl-lab/0.1 "
    "(https://github.com/datawhalechina/agentic-rl-lab)"
)
# 逐行讲解：构造字典 {'deepseek': 16, 'wikipedia': 3}，赋给 DEFAULT_SEARCH_CONCURRENCY。
DEFAULT_SEARCH_CONCURRENCY = {"deepseek": 16, "wikipedia": 3}
# 逐行讲解：构造字典 {'deepseek': 60.0, 'wikipedia': 15.0}，赋给 DEFAULT_SEARCH_TIMEOUT。
DEFAULT_SEARCH_TIMEOUT = {"deepseek": 60.0, "wikipedia": 15.0}


# 逐行讲解：定义类 SearchItem：保存一条搜索证据及其可选来源信息。
@dataclass(frozen=True)
class SearchItem:
    """保存一条搜索证据及其可选来源信息。"""

    # 逐行讲解：声明 title 的类型为 str。
    title: str
    # 逐行讲解：声明 content 的类型为 str。
    content: str
    # 逐行讲解：声明 source 的类型为 str | None；取值 None。
    source: str | None = None
    # 逐行讲解：声明 url 的类型为 str | None；取值 None。
    url: str | None = None


# 逐行讲解：定义类 SearchResult：保存一次搜索的证据或错误信息。
@dataclass(frozen=True)
class SearchResult:
    """保存一次搜索的证据或错误信息。"""

    # 逐行讲解：声明 ok 的类型为 bool。
    ok: bool
    # 逐行讲解：声明 items 的类型为 list[SearchItem]。
    items: list[SearchItem]
    # 逐行讲解：声明 latency 的类型为 float。
    latency: float
    # 逐行讲解：声明 status 的类型为 int | None；取值 None。
    status: int | None = None
    # 逐行讲解：声明 error 的类型为 str | None；取值 None。
    error: str | None = None


# 逐行讲解：定义类 SearchStats：累计并发搜索请求的运行指标。
@dataclass
class SearchStats:
    """累计并发搜索请求的运行指标。"""

    # 逐行讲解：声明 backend 的类型为 str。
    backend: str
    # 逐行讲解：声明 requests 的类型为 int；取值 0。
    requests: int = 0
    # 逐行讲解：声明 successes 的类型为 int；取值 0。
    successes: int = 0
    # 逐行讲解：声明 timeouts 的类型为 int；取值 0。
    timeouts: int = 0
    # 逐行讲解：声明 rate_limits 的类型为 int；取值 0。
    rate_limits: int = 0
    # 逐行讲解：声明 errors 的类型为 int；取值 0。
    errors: int = 0
    # 逐行讲解：声明 latency_total 的类型为 float；取值 0.0。
    latency_total: float = 0.0
    # 逐行讲解：声明 api_requests 的类型为 int；取值 0。
    api_requests: int = 0
    # 逐行讲解：声明 billable_requests 的类型为 int；取值 0。
    billable_requests: int = 0
    # 逐行讲解：声明 result_count 的类型为 int；取值 0。
    result_count: int = 0
    # 逐行讲解：声明 _lock 的类型为 threading.Lock；调用 field（field(default_factory=threading.Lock, repr=False)）。
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # 逐行讲解：定义函数 metrics：把累计计数转换成便于 SwanLab 记录的单次请求均值。
    def metrics(self) -> dict[str, float]:
        """把累计计数转换成便于 SwanLab 记录的单次请求均值。"""
        # 逐行讲解：进入上下文管理器 self._lock，退出时执行其收尾逻辑。
        with self._lock:
            # 逐行讲解：调用 max（max(self.requests, 1)），赋给 denominator。
            denominator = max(self.requests, 1)
            # 逐行讲解：构造字典 {'search/success_rate': self.successes / denominator, 'search/timeout_rate': self.timeouts / denominator, '…，赋给 metrics。
            metrics = {
                "search/success_rate": self.successes / denominator,
                "search/timeout_rate": self.timeouts / denominator,
                "search/429_rate": self.rate_limits / denominator,
                "search/error_rate": self.errors / denominator,
                "search/latency": self.latency_total / denominator,
                "search/results": self.result_count / denominator,
            }
            # 逐行讲解：检查条件 self.backend == 'deepseek'；成立时执行下方分支。
            if self.backend == "deepseek":
                # 逐行讲解：调用 metrics.update（metrics.update({'search/api_requests': self.api_requests / denominator, 'search/billable_requests': self.bi…）。
                metrics.update(
                    {
                        "search/api_requests": self.api_requests / denominator,
                        "search/billable_requests": (
                            self.billable_requests / denominator
                        ),
                    }
                )
            # 逐行讲解：返回结果：metrics。
            return metrics


# 逐行讲解：定义类 DeepSeekSearchClient：直接调用官方 Search API，把网页摘要和来源转成工具证据。
@dataclass
class DeepSeekSearchClient:
    """直接调用官方 Search API，把网页摘要和来源转成工具证据。"""

    # 逐行讲解：声明 api_key 的类型为 str | None；调用 field（field(default=None, repr=False)）。
    api_key: str | None = field(default=None, repr=False)
    # 逐行讲解：声明 count 的类型为 int；取值 3。
    count: int = 3
    # 逐行讲解：声明 timeout 的类型为 float；按索引或切片读取 DEFAULT_SEARCH_TIMEOUT['deepseek']。
    timeout: float = DEFAULT_SEARCH_TIMEOUT["deepseek"]
    # 逐行讲解：声明 max_retries 的类型为 int；取值 1。
    max_retries: int = 1
    # 逐行讲解：声明 retry_delay 的类型为 float；取值 1.0。
    retry_delay: float = 1.0
    # 逐行讲解：声明 stats 的类型为 SearchStats；调用 field（field(default_factory=lambda: SearchStats(backend='deepseek'))）。
    stats: SearchStats = field(
        default_factory=lambda: SearchStats(backend="deepseek")
    )

    # 逐行讲解：定义函数 __post_init__：封装下方逻辑，供调用方复用。
    def __post_init__(self) -> None:
        # 逐行讲解：取值 self.api_key or os.getenv('DEEPSEEK_API_KEY') or os.getenv('DEEPSEEK_KEY')，赋给 self.api_key。
        self.api_key = (
            self.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")
        )
        # 逐行讲解：检查条件 not self.api_key or not self.api_key.strip()；成立时执行下方分支。
        if not self.api_key or not self.api_key.strip():
            # 逐行讲解：抛出异常：ValueError('请在 experiments/search-r1/.env 或环境变量中设置 DEEPSEEK_API_KEY')。
            raise ValueError("请在 experiments/search-r1/.env 或环境变量中设置 DEEPSEEK_API_KEY")
        # 逐行讲解：调用 self.api_key.strip（self.api_key.strip()），赋给 self.api_key。
        self.api_key = self.api_key.strip()
        # 逐行讲解：检查条件 not 1 <= self.count <= 50；成立时执行下方分支。
        if not 1 <= self.count <= 50:
            # 逐行讲解：抛出异常：ValueError('search_count 必须在 1 到 50 之间')。
            raise ValueError("search_count 必须在 1 到 50 之间")

    # 逐行讲解：定义函数 from_env：读取项目 .env；已有的进程环境变量优先。
    @classmethod
    def from_env(
        cls, env_path: str | Path | None = None, **kwargs: Any
    ) -> "DeepSeekSearchClient":
        """读取项目 .env；已有的进程环境变量优先。"""
        # 逐行讲解：检查条件 env_path；成立时执行下方分支。
        if env_path:
            # 逐行讲解：调用 load_dotenv（load_dotenv(env_path)）。
            load_dotenv(env_path)
        # 逐行讲解：返回结果：cls(**kwargs)。
        return cls(**kwargs)

    # 逐行讲解：定义函数 search：搜索 Top N 网页；超时、408、429 和 5xx 最多有限重试。
    def search(self, query: str) -> SearchResult:
        """搜索 Top N 网页；超时、408、429 和 5xx 最多有限重试。"""
        # 逐行讲解：调用 time.perf_counter（time.perf_counter()），赋给 started。
        started = time.perf_counter()
        # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
        with self.stats._lock:
            # 逐行讲解：更新 self.stats.requests：self.stats.requests += 1。
            self.stats.requests += 1

        # 逐行讲解：调用 query.strip（query.strip()），赋给 query。
        query = query.strip()
        # 逐行讲解：检查条件 not query or len(query) > 128；成立时执行下方分支。
        if not query or len(query) > 128:
            # 逐行讲解：返回结果：self._error_result(started, 'query must contain 1 to 128 Unicode characters', 400)。
            return self._error_result(
                started, "query must contain 1 to 128 Unicode characters", 400
            )

        # 逐行讲解：取值 False，赋给 saw_timeout。
        saw_timeout = False
        # 逐行讲解：取值 False，赋给 saw_rate_limit。
        saw_rate_limit = False
        # 逐行讲解：取值 0，赋给 attempt。
        attempt = 0
        # 逐行讲解：当 True 成立时继续循环。
        while True:
            # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
            try:
                # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
                with self.stats._lock:
                    # 逐行讲解：更新 self.stats.api_requests：self.stats.api_requests += 1。
                    self.stats.api_requests += 1
                # 逐行讲解：调用 httpx.post（httpx.post(DEEPSEEK_SEARCH_ENDPOINT, headers={'Authorization': f'Bearer {self.api_key}'}, json={'query': qu…），赋给 response。
                response = httpx.post(
                    DEEPSEEK_SEARCH_ENDPOINT,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"query": query, "count": self.count, "need_content": False},
                    timeout=self.timeout,
                )
                # 逐行讲解：调用 response.raise_for_status（response.raise_for_status()）。
                response.raise_for_status()
                # 逐行讲解：调用 response.json（response.json()），赋给 payload。
                payload = response.json()
                # 逐行讲解：检查条件 not isinstance(payload, dict) or not isinstance(payload.get('results'), list)；成立时执行下方分支。
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("results"), list
                ):
                    # 逐行讲解：抛出异常：ValueError('invalid results')。
                    raise ValueError("invalid results")
                # 逐行讲解：按索引或切片读取 payload['results']，赋给 results。
                results = payload["results"]
                # 逐行讲解：声明 items 的类型为 list[SearchItem]；构造列表 []。
                items: list[SearchItem] = []
                # 逐行讲解：遍历 results[:self.count]，将每项绑定到 item。
                for item in results[: self.count]:
                    # 逐行讲解：检查条件 not isinstance(item, dict)；成立时执行下方分支。
                    if not isinstance(item, dict):
                        # 逐行讲解：抛出异常：ValueError('invalid result item')。
                        raise ValueError("invalid result item")
                    # 优先使用 query 相关的短摘要，便于在小工具预算内读取多条证据。
                    content = str(item.get("snippet") or "").strip() or str(
                        item.get("summary") or ""
                    ).strip()
                    # 逐行讲解：检查条件 content；成立时执行下方分支。
                    if content:
                        # 逐行讲解：向列表末尾追加一个元素（items.append(SearchItem(title=str(item.get('title') or 'Untitled').strip(), content=content, source=str(ite…）。
                        items.append(
                            SearchItem(
                                title=str(item.get("title") or "Untitled").strip(),
                                content=content,
                                source=str(item.get("site_name") or "").strip() or None,
                                url=str(item.get("url") or "").strip() or None,
                            )
                        )
                # 逐行讲解：计算 time.perf_counter() - started，赋给 latency。
                latency = time.perf_counter() - started
                # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
                with self.stats._lock:
                    # 逐行讲解：更新 self.stats.successes：self.stats.successes += 1。
                    self.stats.successes += 1
                    # 逐行讲解：更新 self.stats.latency_total：self.stats.latency_total += latency。
                    self.stats.latency_total += latency
                    # 逐行讲解：更新 self.stats.billable_requests：self.stats.billable_requests += int(bool(results))。
                    self.stats.billable_requests += int(bool(results))
                    # 逐行讲解：更新 self.stats.result_count：self.stats.result_count += len(items)。
                    self.stats.result_count += len(items)
                # 逐行讲解：返回结果：SearchResult(ok=True, items=items, latency=latency, status=200)。
                return SearchResult(ok=True, items=items, latency=latency, status=200)
            except httpx.HTTPStatusError as error:
                # 逐行讲解：取值 error.response.status_code，赋给 status。
                status = error.response.status_code
                # 逐行讲解：取值 status in (408, 429) or status >= 500，赋给 retryable。
                retryable = status in (408, 429) or status >= 500
                # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
                with self.stats._lock:
                    # 逐行讲解：检查条件 status == 408 and (not saw_timeout)；成立时执行下方分支。
                    if status == 408 and not saw_timeout:
                        # 逐行讲解：更新 self.stats.timeouts：self.stats.timeouts += 1。
                        self.stats.timeouts += 1
                        # 逐行讲解：取值 True，赋给 saw_timeout。
                        saw_timeout = True
                    # 逐行讲解：检查条件 status == 429 and (not saw_rate_limit)；成立时执行下方分支。
                    if status == 429 and not saw_rate_limit:
                        # 逐行讲解：更新 self.stats.rate_limits：self.stats.rate_limits += 1。
                        self.stats.rate_limits += 1
                        # 逐行讲解：取值 True，赋给 saw_rate_limit。
                        saw_rate_limit = True
                # 逐行讲解：检查条件 retryable and attempt < self.max_retries；成立时执行下方分支。
                if retryable and attempt < self.max_retries:
                    # 逐行讲解：调用 time.sleep（time.sleep(self.retry_delay * 2 ** attempt)）。
                    time.sleep(self.retry_delay * (2**attempt))
                    # 逐行讲解：更新 attempt：attempt += 1。
                    attempt += 1
                    # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                    continue
                # 逐行讲解：返回结果：self._error_result(started, f'HTTP {status}', status)。
                return self._error_result(started, f"HTTP {status}", status)
            except httpx.TimeoutException:
                # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
                with self.stats._lock:
                    # 逐行讲解：检查条件 not saw_timeout；成立时执行下方分支。
                    if not saw_timeout:
                        # 逐行讲解：更新 self.stats.timeouts：self.stats.timeouts += 1。
                        self.stats.timeouts += 1
                        # 逐行讲解：取值 True，赋给 saw_timeout。
                        saw_timeout = True
                # 逐行讲解：检查条件 attempt < self.max_retries；成立时执行下方分支。
                if attempt < self.max_retries:
                    # 逐行讲解：调用 time.sleep（time.sleep(self.retry_delay * 2 ** attempt)）。
                    time.sleep(self.retry_delay * (2**attempt))
                    # 逐行讲解：更新 attempt：attempt += 1。
                    attempt += 1
                    # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                    continue
                # 逐行讲解：返回结果：self._error_result(started, 'request timeout')。
                return self._error_result(started, "request timeout")
            except httpx.HTTPError as error:
                # 逐行讲解：检查条件 attempt < self.max_retries；成立时执行下方分支。
                if attempt < self.max_retries:
                    # 逐行讲解：调用 time.sleep（time.sleep(self.retry_delay * 2 ** attempt)）。
                    time.sleep(self.retry_delay * (2**attempt))
                    # 逐行讲解：更新 attempt：attempt += 1。
                    attempt += 1
                    # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                    continue
                # 逐行讲解：返回结果：self._error_result(started, type(error).__name__)。
                return self._error_result(started, type(error).__name__)
            except (ValueError, TypeError):
                # 逐行讲解：返回结果：self._error_result(started, 'invalid search response', 200)。
                return self._error_result(started, "invalid search response", 200)

    # 逐行讲解：定义函数 _error_result：把请求异常转换为不会泄露密钥的工具结果。
    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        """把请求异常转换为不会泄露密钥的工具结果。"""
        # 逐行讲解：计算 time.perf_counter() - started，赋给 latency。
        latency = time.perf_counter() - started
        # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
        with self.stats._lock:
            # 逐行讲解：更新 self.stats.errors：self.stats.errors += 1。
            self.stats.errors += 1
            # 逐行讲解：更新 self.stats.latency_total：self.stats.latency_total += latency。
            self.stats.latency_total += latency
        # 逐行讲解：返回结果：SearchResult(False, [], latency, status=status, error=message)。
        return SearchResult(False, [], latency, status=status, error=message)


# 逐行讲解：定义类 WikipediaSearchClient：使用 Wikimedia Action API 搜索英文 Wikipedia 并返回页面正文片段。
@dataclass
class WikipediaSearchClient:
    """使用 Wikimedia Action API 搜索英文 Wikipedia 并返回页面正文片段。"""

    # 逐行讲解：声明 timeout 的类型为 float；按索引或切片读取 DEFAULT_SEARCH_TIMEOUT['wikipedia']。
    timeout: float = DEFAULT_SEARCH_TIMEOUT["wikipedia"]
    # 逐行讲解：声明 max_retries 的类型为 int；取值 2。
    max_retries: int = 2
    # 逐行讲解：声明 retry_delay 的类型为 float；取值 1.0。
    retry_delay: float = 1.0
    # 逐行讲解：声明 min_request_interval 的类型为 float；取值 0.31。
    min_request_interval: float = 0.31
    # 逐行讲解：声明 stats 的类型为 SearchStats；调用 field（field(default_factory=lambda: SearchStats(backend='wikipedia'))）。
    stats: SearchStats = field(
        default_factory=lambda: SearchStats(backend="wikipedia")
    )
    # 逐行讲解：声明 _rate_lock 的类型为 threading.Lock；调用 field（field(default_factory=threading.Lock, init=False, repr=False)）。
    _rate_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    # 逐行讲解：声明 _next_request_time 的类型为 float；调用 field（field(default=0.0, init=False, repr=False)）。
    _next_request_time: float = field(default=0.0, init=False, repr=False)

    # 逐行讲解：定义函数 search：搜索 Top 3 页面；所有并发调用共享约 200 RPM 的启动限速。
    def search(self, query: str) -> SearchResult:
        """搜索 Top 3 页面；所有并发调用共享约 200 RPM 的启动限速。"""
        # 逐行讲解：调用 time.perf_counter（time.perf_counter()），赋给 started。
        started = time.perf_counter()
        # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
        with self.stats._lock:
            # 逐行讲解：更新 self.stats.requests：self.stats.requests += 1。
            self.stats.requests += 1

        # 逐行讲解：取值 False，赋给 saw_timeout。
        saw_timeout = False
        # 逐行讲解：取值 False，赋给 saw_rate_limit。
        saw_rate_limit = False
        # 逐行讲解：取值 0，赋给 attempt。
        attempt = 0
        # 逐行讲解：当 True 成立时继续循环。
        while True:
            # 逐行讲解：调用 self._wait_for_rate_slot（self._wait_for_rate_slot()）。
            self._wait_for_rate_slot()
            # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
            try:
                # 逐行讲解：调用 self._request（self._request(query, started)），赋给 result。
                result = self._request(query, started)
                # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
                with self.stats._lock:
                    # 逐行讲解：更新 self.stats.successes：self.stats.successes += 1。
                    self.stats.successes += 1
                    # 逐行讲解：更新 self.stats.latency_total：self.stats.latency_total += result.latency。
                    self.stats.latency_total += result.latency
                    # 逐行讲解：更新 self.stats.result_count：self.stats.result_count += len(result.items)。
                    self.stats.result_count += len(result.items)
                # 逐行讲解：返回结果：result。
                return result
            except urllib.error.HTTPError as error:
                # 逐行讲解：检查条件 error.code == 429 and (not saw_rate_limit)；成立时执行下方分支。
                if error.code == 429 and not saw_rate_limit:
                    # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
                    with self.stats._lock:
                        # 逐行讲解：更新 self.stats.rate_limits：self.stats.rate_limits += 1。
                        self.stats.rate_limits += 1
                    # 逐行讲解：取值 True，赋给 saw_rate_limit。
                    saw_rate_limit = True
                # 逐行讲解：取值 error.code == 429 or error.code >= 500，赋给 retryable。
                retryable = error.code == 429 or error.code >= 500
                # 逐行讲解：检查条件 retryable and attempt < self.max_retries；成立时执行下方分支。
                if retryable and attempt < self.max_retries:
                    # 逐行讲解：查询指定键，缺失时使用默认值（error.headers.get('Retry-After')），赋给 retry_after。
                    retry_after = error.headers.get("Retry-After")
                    # 逐行讲解：按条件选择结果：float(retry_after) if retry_after and retry_after.isdigit() else self.retry_delay * 2 ** attempt，赋给 delay。
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else self.retry_delay * (2**attempt)
                    )
                    # 逐行讲解：调用 time.sleep（time.sleep(delay)）。
                    time.sleep(delay)
                    # 逐行讲解：更新 attempt：attempt += 1。
                    attempt += 1
                    # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                    continue
                # 逐行讲解：返回结果：self._error_result(started, f'HTTP {error.code}', error.code)。
                return self._error_result(started, f"HTTP {error.code}", error.code)
            except (TimeoutError, socket.timeout):
                # 逐行讲解：检查条件 not saw_timeout；成立时执行下方分支。
                if not saw_timeout:
                    # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
                    with self.stats._lock:
                        # 逐行讲解：更新 self.stats.timeouts：self.stats.timeouts += 1。
                        self.stats.timeouts += 1
                    # 逐行讲解：取值 True，赋给 saw_timeout。
                    saw_timeout = True
                # 逐行讲解：检查条件 attempt < self.max_retries；成立时执行下方分支。
                if attempt < self.max_retries:
                    # 逐行讲解：调用 time.sleep（time.sleep(self.retry_delay * 2 ** attempt)）。
                    time.sleep(self.retry_delay * (2**attempt))
                    # 逐行讲解：更新 attempt：attempt += 1。
                    attempt += 1
                    # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                    continue
                # 逐行讲解：返回结果：self._error_result(started, 'request timeout')。
                return self._error_result(started, "request timeout")
            except urllib.error.URLError as error:
                # 逐行讲解：检查条件 isinstance(error.reason, (TimeoutError, socket.timeout))；成立时执行下方分支。
                if isinstance(error.reason, (TimeoutError, socket.timeout)):
                    # 逐行讲解：检查条件 not saw_timeout；成立时执行下方分支。
                    if not saw_timeout:
                        # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
                        with self.stats._lock:
                            # 逐行讲解：更新 self.stats.timeouts：self.stats.timeouts += 1。
                            self.stats.timeouts += 1
                        # 逐行讲解：取值 True，赋给 saw_timeout。
                        saw_timeout = True
                    # 逐行讲解：检查条件 attempt < self.max_retries；成立时执行下方分支。
                    if attempt < self.max_retries:
                        # 逐行讲解：调用 time.sleep（time.sleep(self.retry_delay * 2 ** attempt)）。
                        time.sleep(self.retry_delay * (2**attempt))
                        # 逐行讲解：更新 attempt：attempt += 1。
                        attempt += 1
                        # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                        continue
                    # 逐行讲解：返回结果：self._error_result(started, 'request timeout')。
                    return self._error_result(started, "request timeout")
                # 逐行讲解：返回结果：self._error_result(started, type(error).__name__)。
                return self._error_result(started, type(error).__name__)
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                # 逐行讲解：返回结果：self._error_result(started, type(error).__name__)。
                return self._error_result(started, type(error).__name__)

    # 逐行讲解：定义函数 _wait_for_rate_slot：序列化请求启动时间，避免超过 Wikimedia 的识别客户端分钟限额。
    def _wait_for_rate_slot(self) -> None:
        """序列化请求启动时间，避免超过 Wikimedia 的识别客户端分钟限额。"""
        # 逐行讲解：进入上下文管理器 self._rate_lock，退出时执行其收尾逻辑。
        with self._rate_lock:
            # 逐行讲解：调用 time.monotonic（time.monotonic()），赋给 now。
            now = time.monotonic()
            # 逐行讲解：计算 self._next_request_time - now，赋给 delay。
            delay = self._next_request_time - now
            # 逐行讲解：检查条件 delay > 0；成立时执行下方分支。
            if delay > 0:
                # 逐行讲解：调用 time.sleep（time.sleep(delay)）。
                time.sleep(delay)
                # 逐行讲解：调用 time.monotonic（time.monotonic()），赋给 now。
                now = time.monotonic()
            # 逐行讲解：计算 now + self.min_request_interval，赋给 self._next_request_time。
            self._next_request_time = now + self.min_request_interval

    # 逐行讲解：定义函数 _request：一次请求同时执行全文搜索并取得前三个页面的纯文本摘要。
    def _request(self, query: str, started: float) -> SearchResult:
        """一次请求同时执行全文搜索并取得前三个页面的纯文本摘要。"""
        # 逐行讲解：调用 urllib.parse.urlencode（urllib.parse.urlencode({'action': 'query', 'format': 'json', 'formatversion': 2, 'generator': 'search', 'gs…），赋给 params。
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": 3,
                "prop": "extracts|info",
                "explaintext": 1,
                "exintro": 1,
                "exchars": 1200,
                "inprop": "url",
                "redirects": 1,
                "utf8": 1,
            }
        )
        # 逐行讲解：调用 urllib.request.Request（urllib.request.Request(f'{WIKIPEDIA_SEARCH_ENDPOINT}?{params}', headers={'Accept': 'application/json', 'Acc…），赋给 request。
        request = urllib.request.Request(
            f"{WIKIPEDIA_SEARCH_ENDPOINT}?{params}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": WIKIPEDIA_USER_AGENT,
            },
        )
        # 逐行讲解：进入上下文管理器 urllib.request.urlopen(request, timeout=self.timeout)，退出时执行其收尾逻辑。
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            # 逐行讲解：调用 response.read（response.read()），赋给 body。
            body = response.read()
            # 逐行讲解：检查条件 response.headers.get('Content-Encoding') == 'gzip'；成立时执行下方分支。
            if response.headers.get("Content-Encoding") == "gzip":
                # 逐行讲解：调用 gzip.decompress（gzip.decompress(body)），赋给 body。
                body = gzip.decompress(body)
            # 逐行讲解：将 JSON 字符串解析为 Python 对象（json.loads(body.decode('utf-8'))），赋给 payload。
            payload = json.loads(body.decode("utf-8"))
            # 逐行讲解：查询指定键，缺失时使用默认值（payload.get('query', {}).get('pages', [])），赋给 pages。
            pages = payload.get("query", {}).get("pages", [])
            # 逐行讲解：检查条件 not isinstance(pages, list)；成立时执行下方分支。
            if not isinstance(pages, list):
                # 逐行讲解：抛出异常：TypeError('Wikipedia 搜索响应 pages 不是列表')。
                raise TypeError("Wikipedia 搜索响应 pages 不是列表")
            # 逐行讲解：检查条件 any((not isinstance(page, dict) for page in pages))；成立时执行下方分支。
            if any(not isinstance(page, dict) for page in pages):
                # 逐行讲解：抛出异常：TypeError('Wikipedia 搜索响应 page 不是对象')。
                raise TypeError("Wikipedia 搜索响应 page 不是对象")
            # 逐行讲解：调用 sorted（sorted(pages, key=lambda page: int(page.get('index', 1000000)))），赋给 ordered_pages。
            ordered_pages = sorted(
                pages,
                key=lambda page: int(page.get("index", 1_000_000)),
            )
            # 逐行讲解：通过推导式逐项生成结果：[SearchItem(title=str(page.get('title') or 'Untitled').strip(), content=str(page.get('extract') or '').stri…，赋给 items。
            items = [
                SearchItem(
                    title=str(page.get("title") or "Untitled").strip(),
                    content=str(page.get("extract") or "").strip(),
                    source="Wikipedia",
                    url=str(page.get("fullurl") or "").strip(),
                )
                for page in ordered_pages
                if str(page.get("extract") or "").strip()
            ]
            # 逐行讲解：返回结果：SearchResult(ok=True, items=items, latency=time.perf_counter() - started, status=response.status)。
            return SearchResult(
                ok=True,
                items=items,
                latency=time.perf_counter() - started,
                status=response.status,
            )

    # 逐行讲解：定义函数 _error_result：把请求异常转换成 rollout 可观察、不中断训练的搜索结果。
    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        """把请求异常转换成 rollout 可观察、不中断训练的搜索结果。"""
        # 逐行讲解：计算 time.perf_counter() - started，赋给 latency。
        latency = time.perf_counter() - started
        # 逐行讲解：进入上下文管理器 self.stats._lock，退出时执行其收尾逻辑。
        with self.stats._lock:
            # 逐行讲解：更新 self.stats.errors：self.stats.errors += 1。
            self.stats.errors += 1
            # 逐行讲解：更新 self.stats.latency_total：self.stats.latency_total += latency。
            self.stats.latency_total += latency
        # 逐行讲解：返回结果：SearchResult(False, [], latency, status=status, error=message)。
        return SearchResult(False, [], latency, status=status, error=message)


# 逐行讲解：计算 DeepSeekSearchClient | WikipediaSearchClient，赋给 SearchClient。
SearchClient = DeepSeekSearchClient | WikipediaSearchClient


# 逐行讲解：定义函数 resolve_search_concurrency：返回用户设置或当前后端的默认搜索并发。
def resolve_search_concurrency(backend: str, value: int | None) -> int:
    """返回用户设置或当前后端的默认搜索并发。"""
    # 逐行讲解：检查条件 backend not in SEARCH_BACKENDS；成立时执行下方分支。
    if backend not in SEARCH_BACKENDS:
        # 逐行讲解：抛出异常：ValueError(f'不支持的搜索后端: {backend}')。
        raise ValueError(f"不支持的搜索后端: {backend}")
    # 逐行讲解：按条件选择结果：DEFAULT_SEARCH_CONCURRENCY[backend] if value is None else value，赋给 concurrency。
    concurrency = DEFAULT_SEARCH_CONCURRENCY[backend] if value is None else value
    # 逐行讲解：检查条件 concurrency < 1；成立时执行下方分支。
    if concurrency < 1:
        # 逐行讲解：抛出异常：ValueError('search_concurrency 必须大于等于 1')。
        raise ValueError("search_concurrency 必须大于等于 1")
    # 逐行讲解：返回结果：concurrency。
    return concurrency


# 逐行讲解：定义函数 resolve_search_timeout：返回用户设置或当前后端的默认请求超时。
def resolve_search_timeout(backend: str, value: float | None) -> float:
    """返回用户设置或当前后端的默认请求超时。"""
    # 逐行讲解：检查条件 backend not in SEARCH_BACKENDS；成立时执行下方分支。
    if backend not in SEARCH_BACKENDS:
        # 逐行讲解：抛出异常：ValueError(f'不支持的搜索后端: {backend}')。
        raise ValueError(f"不支持的搜索后端: {backend}")
    # 逐行讲解：按条件选择结果：DEFAULT_SEARCH_TIMEOUT[backend] if value is None else value，赋给 timeout。
    timeout = DEFAULT_SEARCH_TIMEOUT[backend] if value is None else value
    # 逐行讲解：检查条件 timeout <= 0；成立时执行下方分支。
    if timeout <= 0:
        # 逐行讲解：抛出异常：ValueError('search_timeout 必须大于 0')。
        raise ValueError("search_timeout 必须大于 0")
    # 逐行讲解：返回结果：timeout。
    return timeout


# 逐行讲解：定义函数 create_search_client：按名称创建搜索后端；两个后端共用同一套 rollout 接口。
def create_search_client(
    backend: str,
    env_path: str | Path | None = None,
    *,
    count: int = 3,
    timeout: float | None = None,
) -> SearchClient:
    """按名称创建搜索后端；两个后端共用同一套 rollout 接口。"""
    # 逐行讲解：调用 resolve_search_timeout（resolve_search_timeout(backend, timeout)），赋给 resolved_timeout。
    resolved_timeout = resolve_search_timeout(backend, timeout)
    # 逐行讲解：检查条件 backend == 'deepseek'；成立时执行下方分支。
    if backend == "deepseek":
        # 逐行讲解：返回结果：DeepSeekSearchClient.from_env(env_path, count=count, timeout=resolved_timeout)。
        return DeepSeekSearchClient.from_env(
            env_path,
            count=count,
            timeout=resolved_timeout,
        )
    # 逐行讲解：检查条件 backend == 'wikipedia'；成立时执行下方分支。
    if backend == "wikipedia":
        # 逐行讲解：返回结果：WikipediaSearchClient(timeout=resolved_timeout)。
        return WikipediaSearchClient(timeout=resolved_timeout)
    # 逐行讲解：抛出异常：ValueError(f'不支持的搜索后端: {backend}')。
    raise ValueError(f"不支持的搜索后端: {backend}")


# 逐行讲解：定义函数 format_item：按后端提供的信息格式化一条完整工具证据。
def format_item(item: SearchItem, index: int) -> str:
    """按后端提供的信息格式化一条完整工具证据。"""
    # 逐行讲解：构造列表 [f'[{index}] Title: {item.title}', f'    Content: {item.content}']，赋给 lines。
    lines = [
        f"[{index}] Title: {item.title}",
        f"    Content: {item.content}",
    ]
    # 逐行讲解：检查条件 item.source；成立时执行下方分支。
    if item.source:
        # 逐行讲解：向列表末尾追加一个元素（lines.append(f'    Source: {item.source}')）。
        lines.append(f"    Source: {item.source}")
    # 逐行讲解：检查条件 item.url；成立时执行下方分支。
    if item.url:
        # 逐行讲解：向列表末尾追加一个元素（lines.append(f'    URL: {item.url}')）。
        lines.append(f"    URL: {item.url}")
    # 逐行讲解：返回结果：'\n'.join(lines)。
    return "\n".join(lines)
