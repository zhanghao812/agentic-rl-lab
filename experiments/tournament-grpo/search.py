"""作者的两个 XML 工具：google_search 用 DeepSeek Search API，browse_webpage 用 Jina Reader。

工具失败时不抛异常，而是把错误文本作为 observation 返回给模型，和作者一致。

Adapted from experiments/search-r1 and PuffYang/Tournament_GRPO (eea826cf), Apache-2.0.
"""

# 逐行讲解：导入模块：fcntl。
import fcntl
# 逐行讲解：导入模块：os。
import os
# 逐行讲解：导入模块：tempfile。
import tempfile
# 逐行讲解：导入模块：time。
import time
# 逐行讲解：从 dataclasses 导入 dataclass, field。
from dataclasses import dataclass, field
# 逐行讲解：从 pathlib 导入 Path。
from pathlib import Path
# 逐行讲解：从 urllib.parse 导入 urlparse。
from urllib.parse import urlparse

# 逐行讲解：导入模块：httpx。
import httpx
# 逐行讲解：从 dotenv 导入 load_dotenv。
from dotenv import load_dotenv

# 逐行讲解：从 protocol 导入 ToolCall, truncate_tool_text。
from protocol import ToolCall, truncate_tool_text


# 逐行讲解：取值 'https://api.deepseek.com/v1/tools/search'，赋给 DEEPSEEK_SEARCH_URL。
DEEPSEEK_SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
# 逐行讲解：取值 128，赋给 DEEPSEEK_MAX_QUERY_CHARS。
DEEPSEEK_MAX_QUERY_CHARS = 128
# 逐行讲解：计算 Path(tempfile.gettempdir()) / 'tournament_grpo_jina_global_rate_limit.state'，赋给 JINA_RATE_LIMIT_FILE。
JINA_RATE_LIMIT_FILE = Path(tempfile.gettempdir()) / "tournament_grpo_jina_global_rate_limit.state"


# 逐行讲解：定义类 ToolResult：封装下方逻辑，供调用方复用。
@dataclass(frozen=True)
class ToolResult:
    # 逐行讲解：声明 text 的类型为 str。
    text: str
    # 逐行讲解：声明 ok 的类型为 bool。
    ok: bool  # 只用于统计工具错误率；模型只看到 text


# 逐行讲解：定义函数 proxy_from_env：按作者的变量优先级读取代理；只写 host:port 时补 http://。
def proxy_from_env(*names: str) -> str | None:
    """按作者的变量优先级读取代理；只写 host:port 时补 http://。"""
    # 逐行讲解：遍历 names，将每项绑定到 name。
    for name in names:
        # 逐行讲解：调用 os.getenv(name, '').strip（os.getenv(name, '').strip()），赋给 value。
        value = os.getenv(name, "").strip()
        # 逐行讲解：检查条件 value；成立时执行下方分支。
        if value:
            # 逐行讲解：返回结果：value if '://' in value else 'http://' + value。
            return value if "://" in value else "http://" + value
    # 逐行讲解：返回结果：None。
    return None


# 逐行讲解：定义函数 wait_for_jina：作者的跨进程限速：所有进程共享一个时间戳文件，默认 20 RPM（无 Key 的额度）。
def wait_for_jina() -> None:
    """作者的跨进程限速：所有进程共享一个时间戳文件，默认 20 RPM（无 Key 的额度）。

    文件里记着最近一次预约的请求时间，新请求排在它之后 interval 秒。
    """
    # 逐行讲解：计算 60.0 / int(os.getenv('JINA_RATE_LIMIT_RPM', '20'))，赋给 interval。
    interval = 60.0 / int(os.getenv("JINA_RATE_LIMIT_RPM", "20"))
    # 逐行讲解：进入上下文管理器 JINA_RATE_LIMIT_FILE.open('a+')，退出时执行其收尾逻辑。
    with JINA_RATE_LIMIT_FILE.open("a+") as file:
        # 逐行讲解：调用 fcntl.flock（fcntl.flock(file, fcntl.LOCK_EX)）。
        fcntl.flock(file, fcntl.LOCK_EX)  # 关闭文件时自动解锁
        # 逐行讲解：调用 file.seek（file.seek(0)）。
        file.seek(0)
        # 逐行讲解：调用 max（max(time.time(), float(file.read() or 0) + interval)），赋给 reserved。
        reserved = max(time.time(), float(file.read() or 0) + interval)
        # 逐行讲解：调用 file.seek（file.seek(0)）。
        file.seek(0)
        # 逐行讲解：调用 file.truncate（file.truncate()）。
        file.truncate()
        # 逐行讲解：调用 file.write（file.write(str(reserved))）。
        file.write(str(reserved))
    # 逐行讲解：调用 time.sleep（time.sleep(max(0.0, reserved - time.time()))）。
    time.sleep(max(0.0, reserved - time.time()))


# 逐行讲解：定义函数 read_page：返回 (title, url, content)。
def read_page(response: httpx.Response, url: str) -> tuple[str, str, str]:
    """返回 (title, url, content)。

    Jina 返回 JSON 时按作者 _extract_webpage_payload 的字段顺序取正文，否则直接使用纯文本。
    """
    # 逐行讲解：检查条件 'json' in response.headers.get('content-type', '').lower()；成立时执行下方分支。
    if "json" in response.headers.get("content-type", "").lower():
        # 逐行讲解：调用 response.json（response.json()），赋给 payload。
        payload = response.json()
        # 逐行讲解：查询指定键，缺失时使用默认值（payload.get('data', payload)），赋给 data。
        data = payload.get("data", payload)
        # 逐行讲解：调用 (data.get('content') or data.get('text') or data.get('markdown') or data.get('body') or data.get('result') …（(data.get('content') or data.get('text') or data.get('markdown') or data.get('body') or data.get('result') …），赋给 content。
        content = (data.get("content") or data.get("text") or data.get("markdown")
                   or data.get("body") or data.get("result") or "").strip()
        # 逐行讲解：检查条件 content；成立时执行下方分支。
        if content:
            # 逐行讲解：返回结果：((data.get('title') or '').strip(), (data.get('url') or '').strip() or url, content)。
            return (data.get("title") or "").strip(), (data.get("url") or "").strip() or url, content
    # 逐行讲解：返回结果：('', url, response.text.strip())。
    return "", url, response.text.strip()


# 逐行讲解：定义类 ResearchTools：封装下方逻辑，供调用方复用。
@dataclass
class ResearchTools:
    # 逐行讲解：声明 deepseek_api_key 的类型为 str；调用 field（field(repr=False)）。
    deepseek_api_key: str = field(repr=False)
    # 逐行讲解：声明 jina_api_key 的类型为 str；调用 field（field(default='', repr=False)）。
    jina_api_key: str = field(default="", repr=False)
    # 逐行讲解：声明 search_proxy 的类型为 str | None；取值 None。
    search_proxy: str | None = None
    # 逐行讲解：声明 jina_proxy 的类型为 str | None；取值 None。
    jina_proxy: str | None = None
    # 逐行讲解：声明 jina_reader_base_url 的类型为 str；取值 'https://r.jina.ai'。
    jina_reader_base_url: str = "https://r.jina.ai"
    # 逐行讲解：声明 timeout 的类型为 float；取值 30.0。
    timeout: float = 30.0
    # 逐行讲解：声明 search_count 的类型为 int；取值 3。
    search_count: int = 3  # 作者为 Top 10，本复现降为 Top 3

    # 逐行讲解：定义函数 from_env：封装下方逻辑，供调用方复用。
    @classmethod
    def from_env(cls, env_path: str | Path) -> "ResearchTools":
        # 逐行讲解：调用 load_dotenv（load_dotenv(env_path)）。
        load_dotenv(env_path)
        # 逐行讲解：调用 os.getenv('DEEPSEEK_API_KEY', '').strip（os.getenv('DEEPSEEK_API_KEY', '').strip()），赋给 key。
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        # 逐行讲解：检查条件 not key；成立时执行下方分支。
        if not key:
            # 逐行讲解：抛出异常：ValueError('请在 experiments/tournament-grpo/.env 中设置 DEEPSEEK_API_KEY')。
            raise ValueError("请在 experiments/tournament-grpo/.env 中设置 DEEPSEEK_API_KEY")
        # 逐行讲解：返回结果：cls(key, os.getenv('JINA_API_KEY', '').strip(), search_proxy=proxy_from_env('DEEPSEEK_SEARCH_PROXY'), jina_…。
        return cls(
            key, os.getenv("JINA_API_KEY", "").strip(),
            search_proxy=proxy_from_env("DEEPSEEK_SEARCH_PROXY"),
            jina_proxy=proxy_from_env("JINA_READER_HTTPS_PROXY", "JINA_READER_PROXY", "JINA_HTTPS_PROXY"),
            jina_reader_base_url=os.getenv("JINA_READER_BASE_URL", "https://r.jina.ai"),
        )

    # 逐行讲解：定义函数 search：每条搜索结果按作者的 <snippet> 格式输出。
    def search(self, query: str) -> ToolResult:
        """每条搜索结果按作者的 <snippet> 格式输出。"""
        # DeepSeek 限制 query 最多 128 字符；轨迹里仍保留模型生成的原始 query。
        query = query.strip()[:DEEPSEEK_MAX_QUERY_CHARS]
        # 逐行讲解：检查条件 not query；成立时执行下方分支。
        if not query:
            # 逐行讲解：返回结果：ToolResult('Error: query is required.', False)。
            return ToolResult("Error: query is required.", False)
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：调用 httpx.post（httpx.post(DEEPSEEK_SEARCH_URL, headers={'Authorization': f'Bearer {self.deepseek_api_key}'}, json={'query'…），赋给 response。
            response = httpx.post(
                DEEPSEEK_SEARCH_URL,
                headers={"Authorization": f"Bearer {self.deepseek_api_key}"},
                # 搜索只给摘要，模型要读正文时再调用 browse_webpage。
                json={"query": query, "count": self.search_count, "need_content": False},
                timeout=self.timeout, proxy=self.search_proxy, trust_env=False,
            )
            # 逐行讲解：调用 response.raise_for_status（response.raise_for_status()）。
            response.raise_for_status()
            # 逐行讲解：构造列表 []，赋给 blocks。
            blocks = []
            # 逐行讲解：遍历 response.json()['results'][:self.search_count]，将每项绑定到 item。
            for item in response.json()["results"][:self.search_count]:
                # 逐行讲解：调用 (item.get('title') or '').strip（(item.get('title') or '').strip()），赋给 title。
                title = (item.get("title") or "").strip()
                # 逐行讲解：调用 (item.get('url') or '').strip（(item.get('url') or '').strip()），赋给 url。
                url = (item.get("url") or "").strip()
                # 逐行讲解：调用 (item.get('snippet') or item.get('summary') or '').strip（(item.get('snippet') or item.get('summary') or '').strip()），赋给 snippet。
                snippet = (item.get("snippet") or item.get("summary") or "").strip()
                # 逐行讲解：调用 (item.get('date') or '').strip（(item.get('date') or '').strip()），赋给 date。
                date = (item.get("date") or "").strip()
                # 逐行讲解：检查条件 not (title or url or snippet)；成立时执行下方分支。
                if not (title or url or snippet):
                    # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
                    continue
                # 逐行讲解：构造列表 [f'Title: {title}', f'URL: {url}', f'Search Snippet: {snippet}']，赋给 lines。
                lines = [f"Title: {title}", f"URL: {url}", f"Search Snippet: {snippet}"]
                # 逐行讲解：检查条件 date；成立时执行下方分支。
                if date:
                    # 逐行讲解：向列表末尾追加一个元素（lines.append(f'Date: {date}')）。
                    lines.append(f"Date: {date}")
                # 逐行讲解：向列表末尾追加一个元素（blocks.append('<snippet>\n' + '\n'.join(lines) + '\n</snippet>')）。
                blocks.append("<snippet>\n" + "\n".join(lines) + "\n</snippet>")
            # 逐行讲解：取值 '\n'.join(blocks) or '<snippet>\nNo search results found.\n</snippet>'，赋给 content。
            content = "\n".join(blocks) or "<snippet>\nNo search results found.\n</snippet>"
            # 逐行讲解：返回结果：ToolResult(f'<tool_output>\n{content}\n</tool_output>', True)。
            return ToolResult(f"<tool_output>\n{content}\n</tool_output>", True)
        except Exception as exc:
            # 逐行讲解：返回结果：ToolResult(f'Error performing google_search: {exc}', False)。
            return ToolResult(f"Error performing google_search: {exc}", False)

    # 逐行讲解：定义函数 browse：调用 Jina Reader；429 时最多尝试 3 次。正文先截断一次，execute 再截断整个 observation。
    def browse(self, url: str) -> ToolResult:
        """调用 Jina Reader；429 时最多尝试 3 次。正文先截断一次，execute 再截断整个 observation。"""
        # 逐行讲解：调用 url.strip（url.strip()），赋给 url。
        url = url.strip()
        # 逐行讲解：检查条件 not url；成立时执行下方分支。
        if not url:
            # 逐行讲解：返回结果：ToolResult('Error: url is required.', False)。
            return ToolResult("Error: url is required.", False)
        # 逐行讲解：检查条件 '://' not in url；成立时执行下方分支。
        if "://" not in url:
            # 逐行讲解：计算 'https://' + url.lstrip('/')，赋给 url。
            url = "https://" + url.lstrip("/")
        # 和作者一样在 try 外校验 URL，由 execute 返回通用错误文本。
        if not urlparse(url).netloc:
            # 逐行讲解：抛出异常：ValueError(f'Invalid URL: {url}')。
            raise ValueError(f"Invalid URL: {url}")
        # 逐行讲解：按条件选择结果：{'Authorization': f'Bearer {self.jina_api_key}'} if self.jina_api_key else {}，赋给 headers。
        headers = {"Authorization": f"Bearer {self.jina_api_key}"} if self.jina_api_key else {}
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：遍历 range(3)，将每项绑定到 attempt。
            for attempt in range(3):
                # 逐行讲解：调用 wait_for_jina（wait_for_jina()）。
                wait_for_jina()
                # 逐行讲解：查询指定键，缺失时使用默认值（httpx.get(f"{self.jina_reader_base_url.rstrip('/')}/{url}", headers=headers, timeout=self.timeout, follow_r…），赋给 response。
                response = httpx.get(
                    f"{self.jina_reader_base_url.rstrip('/')}/{url}", headers=headers,
                    timeout=self.timeout, follow_redirects=True, proxy=self.jina_proxy, trust_env=False,
                )
                # 逐行讲解：检查条件 response.status_code != 429；成立时执行下方分支。
                if response.status_code != 429:
                    # 逐行讲解：结束当前循环。
                    break
                # 逐行讲解：调用 time.sleep（time.sleep(2 * 2 ** attempt)）。
                time.sleep(2 * 2**attempt)
            # 逐行讲解：调用 response.raise_for_status（response.raise_for_status()）。
            response.raise_for_status()
            # 逐行讲解：调用 read_page（read_page(response, url)），赋给 (title, url, content)。
            title, url, content = read_page(response, url)
            # 逐行讲解：检查条件 not content；成立时执行下方分支。
            if not content:
                # 逐行讲解：抛出异常：ValueError('Jina Reader response did not include readable page content')。
                raise ValueError("Jina Reader response did not include readable page content")
            # 逐行讲解：取值 f"<tool_output>\n<webpage>\nTitle: {title or 'No title available'}\nURL: {url}\nContent: {truncate_tool_tex…，赋给 text。
            text = (f"<tool_output>\n<webpage>\nTitle: {title or 'No title available'}\n"
                    f"URL: {url}\nContent: {truncate_tool_text(content)}\n</webpage>\n</tool_output>")
            # 逐行讲解：返回结果：ToolResult(text, True)。
            return ToolResult(text, True)
        except Exception as exc:
            # 逐行讲解：返回结果：ToolResult(f'Error performing browse_webpage: {exc}', False)。
            return ToolResult(f"Error performing browse_webpage: {exc}", False)

    # 逐行讲解：定义函数 execute：分发工具调用；未知工具和非法 URL 用作者的通用错误文本，最后整体再截断一次。
    def execute(self, call: ToolCall) -> ToolResult:
        """分发工具调用；未知工具和非法 URL 用作者的通用错误文本，最后整体再截断一次。"""
        # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
        try:
            # 逐行讲解：检查条件 call.name == 'google_search'；成立时执行下方分支。
            if call.name == "google_search":
                # 逐行讲解：调用 self.search（self.search(call.arguments.get('query', ''))），赋给 result。
                result = self.search(call.arguments.get("query", ""))
            # 逐行讲解：检查条件 call.name == 'browse_webpage'；成立时执行下方分支。
            elif call.name == "browse_webpage":
                # 逐行讲解：调用 self.browse（self.browse(call.arguments.get('url', ''))），赋给 result。
                result = self.browse(call.arguments.get("url", ""))
            else:
                # 逐行讲解：抛出异常：KeyError(call.name)。
                raise KeyError(call.name)
        except Exception as exc:
            # 逐行讲解：返回结果：ToolResult(f'Error when executing tool: {exc}', False)。
            return ToolResult(f"Error when executing tool: {exc}", False)
        # 逐行讲解：返回结果：ToolResult(truncate_tool_text(result.text), result.ok)。
        return ToolResult(truncate_tool_text(result.text), result.ok)
