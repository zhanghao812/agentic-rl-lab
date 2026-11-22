"""本地 Python 代码执行沙箱：ReTool rollout 的 code_interpreter 工具后端。

设计原理
========

执行模型：模型在 <tool_call> 中给出的 Python 代码，用
`python -B -c <bootstrap>` 起全新子进程执行。list 形式 argv 不经过 shell，
代码经 repr() 内嵌进引导串——不落脚本文件、无注入面；每次调用一个全新
进程，天然满足官方"每次执行独立无状态"的语义。

四道保险：

1. 限并发：threading.BoundedSemaphore（默认 8）。不用 asyncio.Semaphore，
   因为它会绑定到单个事件循环，而训练循环每个 step/round 都会新建 loop，
   跨 loop 复用会直接报错（smoke test 实测踩到）。
2. 限时间：wall-clock 超时（默认 30s，对齐官方 recipe）后 os.killpg 杀死
   整个进程组（start_new_session=True 才能整组杀，防止 fork 出孤儿进程）；
   另有子进程引导串里的 RLIMIT_CPU 作为内核级 CPU 时间兜底。
3. 限输出：stdout/stderr 写入匿名临时文件而不是 PIPE，防止 print 洪水撑爆
   父进程内存；读回时只取文件尾部并按 4096 字符截断——执行结果和
   traceback 的关键信息都在末尾。
4. 限线程：子进程环境把 OMP/OpenBLAS/MKL/VECLIB 线程数压到 1，防止多个
   并发沙箱各自拉起一整组 BLAS 线程把机器打满。

结果哲学：成功回 stdout（没有 print 则为空串，官方语义）；失败回原始
stderr（完整 traceback）；超时回超时说明。报错不修饰、原样回喂——
论文中"自我纠错"行为正是从看到错误反馈再改代码中涌现的。
唯一处理：tool 文本中的模板特殊标记（`<|im_end|>`、`</tool_response>` 等）
会替换成无害写法，防止污染 observation 结构。

已验证：正常输出、auto-print
补全、语法错误、运行时异常、超时杀进程、print 洪水截断、numpy 导入、
空输出语义、4 路异步并发，共 11 项全部通过。
"""

# 逐行讲解：导入模块：asyncio。
import asyncio
# 逐行讲解：导入模块：os。
import os
# 逐行讲解：导入模块：signal。
import signal
# 逐行讲解：导入模块：subprocess。
import subprocess
# 逐行讲解：导入模块：sys。
import sys
# 逐行讲解：导入模块：tempfile。
import tempfile
# 逐行讲解：导入模块：threading。
import threading
# 逐行讲解：导入模块：time。
import time
# 逐行讲解：从 dataclasses 导入 dataclass, field。
from dataclasses import dataclass, field
# 逐行讲解：从 typing 导入 Any。
from typing import Any


# 逐行讲解：取值 30.0，赋给 DEFAULT_TIMEOUT。
DEFAULT_TIMEOUT = 30.0  # 单次代码调用允许的最大 wall-clock 秒数。
# 逐行讲解：取值 4096，赋给 MAX_OUTPUT_CHARS。
MAX_OUTPUT_CHARS = 4096  # stdout/stderr 各自允许保留的最大字符数。
# 逐行讲解：计算 MAX_OUTPUT_CHARS * 4，赋给 READ_CAP_BYTES。
READ_CAP_BYTES = MAX_OUTPUT_CHARS * 4  # 从输出文件读取的最大字节数（预留多字节字符）。

# 子进程引导代码：先限制 CPU 时间，再 exec 模型代码。
# 用 repr 内嵌代码，避免任何转义问题；__name__ 保持 "__main__" 语义。
_BOOTSTRAP = (
    "import resource\n"
    "resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))\n"
    "exec(compile({code}, '<sandbox>', 'exec'))\n"
)

# 子进程环境：保留基础环境，但把常见数值库的线程数压到 1，
# 防止多个并发沙箱各自拉起一整组 BLAS 线程把机器打满。
_CHILD_ENV_OVERRIDES = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


# 逐行讲解：定义类 ExecResult：保存一次代码执行的结果。
@dataclass(frozen=True)
class ExecResult:
    """保存一次代码执行的结果。"""

    # 逐行讲解：声明 ok 的类型为 bool。
    ok: bool  # 进程正常结束且 returncode 为 0。
    # 逐行讲解：声明 stdout 的类型为 str。
    stdout: str  # 截断后的标准输出。
    # 逐行讲解：声明 stderr 的类型为 str。
    stderr: str  # 截断后的标准错误。
    # 逐行讲解：声明 returncode 的类型为 int | None。
    returncode: int | None  # 进程退出码；未正常退出时为 None。
    # 逐行讲解：声明 timed_out 的类型为 bool。
    timed_out: bool  # 是否因 wall-clock 超时被杀死。
    # 逐行讲解：声明 latency 的类型为 float。
    latency: float  # 从启动到回收的耗时（秒）。


# 逐行讲解：定义类 SandboxStats：累计沙箱调用的运行指标。
@dataclass
class SandboxStats:
    """累计沙箱调用的运行指标。"""

    # 逐行讲解：声明 calls 的类型为 int；取值 0。
    calls: int = 0
    # 逐行讲解：声明 successes 的类型为 int；取值 0。
    successes: int = 0
    # 逐行讲解：声明 errors 的类型为 int；取值 0。
    errors: int = 0  # 进程返回非零退出码（语法错误、异常等）。
    # 逐行讲解：声明 timeouts 的类型为 int；取值 0。
    timeouts: int = 0
    # 逐行讲解：声明 latency_total 的类型为 float；取值 0.0。
    latency_total: float = 0.0

    # 逐行讲解：定义函数 metrics：把累计计数转换成便于 SwanLab 记录的比例。
    def metrics(self) -> dict[str, float]:
        """把累计计数转换成便于 SwanLab 记录的比例。"""
        # 逐行讲解：调用 max（max(self.calls, 1)），赋给 denominator。
        denominator = max(self.calls, 1)
        # 逐行讲解：返回结果：{'sandbox/success_rate': self.successes / denominator, 'sandbox/error_rate': self.errors / denominator, 'sa…。
        return {
            "sandbox/success_rate": self.successes / denominator,
            "sandbox/error_rate": self.errors / denominator,
            "sandbox/timeout_rate": self.timeouts / denominator,
            "sandbox/latency": self.latency_total / denominator,
        }


# 逐行讲解：定义函数 ensure_trailing_print：给最后一行非空行补 print()，与官方 recipe 的 auto-print 技巧一致。
def ensure_trailing_print(code: str) -> str:
    """给最后一行非空行补 print()，与官方 recipe 的 auto-print 技巧一致。"""
    # 逐行讲解：调用 code.split（code.split('\n')），赋给 lines。
    lines = code.split("\n")
    # 逐行讲解：遍历 range(len(lines) - 1, -1, -1)，将每项绑定到 index。
    for index in range(len(lines) - 1, -1, -1):
        # 逐行讲解：检查条件 lines[index] == ''；成立时执行下方分支。
        if lines[index] == "":
            # 逐行讲解：跳过本轮剩余操作，进入下一轮循环。
            continue
        # 逐行讲解：检查条件 not lines[index].startswith('print')；成立时执行下方分支。
        if not lines[index].startswith("print"):
            # 逐行讲解：取值 f'print({lines[index]})'，赋给 lines[index]。
            lines[index] = f"print({lines[index]})"
        # 逐行讲解：结束当前循环。
        break
    # 逐行讲解：返回结果：'\n'.join(lines)。
    return "\n".join(lines)


# 逐行讲解：定义函数 _read_capped：只从输出文件尾部读取有限字节，避免 print 洪水撑爆内存。
def _read_capped(file: Any, limit: int = READ_CAP_BYTES) -> str:
    """只从输出文件尾部读取有限字节，避免 print 洪水撑爆内存。"""
    # 逐行讲解：调用 file.seek（file.seek(0, os.SEEK_END)）。
    file.seek(0, os.SEEK_END)
    # 逐行讲解：调用 file.tell（file.tell()），赋给 size。
    size = file.tell()
    # 逐行讲解：调用 file.seek（file.seek(max(0, size - limit))）。
    file.seek(max(0, size - limit))
    # 逐行讲解：返回结果：file.read().decode('utf-8', errors='replace')。
    return file.read().decode("utf-8", errors="replace")


# 逐行讲解：定义函数 _truncate_tail：按尾部保留输出；模型的关键结果通常打在末尾。
def _truncate_tail(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """按尾部保留输出；模型的关键结果通常打在末尾。"""
    # 逐行讲解：检查条件 len(text) <= limit；成立时执行下方分支。
    if len(text) <= limit:
        # 逐行讲解：返回结果：text。
        return text
    # 逐行讲解：计算 len(text) - limit，赋给 omitted。
    omitted = len(text) - limit
    # 逐行讲解：返回结果：f'[... truncated {omitted} chars ...]\n{text[-limit:]}'。
    return f"[... truncated {omitted} chars ...]\n{text[-limit:]}"


# 工具返回文本里的模板特殊标记会污染 observation 结构
# （chat template 渲染消息内容时不做转义），统一替换成无害写法。
_SANITIZE_REPLACEMENTS = {
    "<|im_start|>": "< im_start >",
    "<|im_end|>": "< im_end >",
    "<tool_response>": "< tool_response >",
    "</tool_response>": "< /tool_response>",
}


# 逐行讲解：定义函数 sanitize_tool_content：把 tool 返回文本中的模板特殊标记替换成无害写法。
def sanitize_tool_content(text: str) -> str:
    """把 tool 返回文本中的模板特殊标记替换成无害写法。"""
    # 逐行讲解：遍历 _SANITIZE_REPLACEMENTS.items()，将每项绑定到 (marker, replacement)。
    for marker, replacement in _SANITIZE_REPLACEMENTS.items():
        # 逐行讲解：调用 text.replace（text.replace(marker, replacement)），赋给 text。
        text = text.replace(marker, replacement)
    # 逐行讲解：返回结果：text。
    return text


# 逐行讲解：定义类 LocalPythonSandbox：用本机 Python 解释器执行模型生成的代码，带超时和资源限制。
class LocalPythonSandbox:
    """用本机 Python 解释器执行模型生成的代码，带超时和资源限制。"""

    # 逐行讲解：定义函数 __init__：保存执行参数，并创建并发信号量与统计器。
    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_workers: int = 8,
        python: str = sys.executable,
    ) -> None:
        """保存执行参数，并创建并发信号量与统计器。"""
        # 逐行讲解：取值 timeout，赋给 self.timeout。
        self.timeout = timeout
        # 逐行讲解：取值 python，赋给 self.python。
        self.python = python
        # 逐行讲解：调用 SandboxStats（SandboxStats()），赋给 self.stats。
        self.stats = SandboxStats()
        # 用线程信号量而不是 asyncio.Semaphore：后者会绑定到单个事件循环，
        # 而训练循环每个 step、每个 round 都会新建 loop，跨 loop 复用会报错。
        self._semaphore = threading.BoundedSemaphore(max_workers)

    # 逐行讲解：定义函数 run_code：同步执行一段代码，返回截断后的输出与执行状态。
    def run_code(self, code: str) -> ExecResult:
        """同步执行一段代码，返回截断后的输出与执行状态。"""
        # 逐行讲解：调用 time.perf_counter（time.perf_counter()），赋给 started。
        started = time.perf_counter()
        # 逐行讲解：更新 self.stats.calls：self.stats.calls += 1。
        self.stats.calls += 1
        # 逐行讲解：调用 _BOOTSTRAP.format（_BOOTSTRAP.format(cpu=int(self.timeout) + 5, code=repr(code))），赋给 bootstrap。
        bootstrap = _BOOTSTRAP.format(cpu=int(self.timeout) + 5, code=repr(code))
        # 逐行讲解：构造字典 {**os.environ, **_CHILD_ENV_OVERRIDES}，赋给 env。
        env = {**os.environ, **_CHILD_ENV_OVERRIDES}
        # 逐行讲解：进入上下文管理器 self._semaphore，退出时执行其收尾逻辑。
        with self._semaphore:
            # 输出写入匿名临时文件而不是 PIPE，防止子进程输出洪水撑爆父进程内存。
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                # 逐行讲解：调用 subprocess.Popen（subprocess.Popen([self.python, '-B', '-c', bootstrap], stdout=stdout_file, stderr=stderr_file, env=env, sta…），赋给 process。
                process = subprocess.Popen(
                    [self.python, "-B", "-c", bootstrap],
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=env,
                    start_new_session=True,  # 独立进程组，超时才能整组杀死
                )
                # 逐行讲解：取值 False，赋给 timed_out。
                timed_out = False
                # 逐行讲解：执行可能失败的操作，异常处理及收尾逻辑见下方分支。
                try:
                    # 逐行讲解：调用 process.wait（process.wait(timeout=self.timeout)），赋给 returncode。
                    returncode = process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    # 逐行讲解：取值 True，赋给 timed_out。
                    timed_out = True
                    # 逐行讲解：调用 os.killpg（os.killpg(process.pid, signal.SIGKILL)）。
                    os.killpg(process.pid, signal.SIGKILL)
                    # 逐行讲解：取值 None，赋给 returncode。
                    returncode = None
                    # 逐行讲解：调用 process.wait（process.wait()）。
                    process.wait()  # 回收僵尸进程
                # 逐行讲解：调用 _truncate_tail（_truncate_tail(_read_capped(stdout_file))），赋给 stdout。
                stdout = _truncate_tail(_read_capped(stdout_file))
                # 逐行讲解：调用 _truncate_tail（_truncate_tail(_read_capped(stderr_file))），赋给 stderr。
                stderr = _truncate_tail(_read_capped(stderr_file))

        # 逐行讲解：计算 time.perf_counter() - started，赋给 latency。
        latency = time.perf_counter() - started
        # 逐行讲解：取值 not timed_out and returncode == 0，赋给 ok。
        ok = not timed_out and returncode == 0
        # 逐行讲解：检查条件 timed_out；成立时执行下方分支。
        if timed_out:
            # 逐行讲解：更新 self.stats.timeouts：self.stats.timeouts += 1。
            self.stats.timeouts += 1
        # 逐行讲解：检查条件 ok；成立时执行下方分支。
        elif ok:
            # 逐行讲解：更新 self.stats.successes：self.stats.successes += 1。
            self.stats.successes += 1
        else:
            # 逐行讲解：更新 self.stats.errors：self.stats.errors += 1。
            self.stats.errors += 1
        # 逐行讲解：更新 self.stats.latency_total：self.stats.latency_total += latency。
        self.stats.latency_total += latency
        # 逐行讲解：返回结果：ExecResult(ok, stdout, stderr, returncode, timed_out, latency)。
        return ExecResult(ok, stdout, stderr, returncode, timed_out, latency)

    # 逐行讲解：定义异步函数 arun_code：在线程池中异步执行代码（并发由 run_code 内的线程信号量限流）。
    async def arun_code(self, code: str) -> ExecResult:
        """在线程池中异步执行代码（并发由 run_code 内的线程信号量限流）。"""
        # 逐行讲解：返回结果：await asyncio.to_thread(self.run_code, code)。
        return await asyncio.to_thread(self.run_code, code)

    # 逐行讲解：定义函数 format_tool_content：把执行结果格式化成喂给模型的 tool 消息内容（特殊标记已无害化）。
    def format_tool_content(self, result: ExecResult) -> str:
        """把执行结果格式化成喂给模型的 tool 消息内容（特殊标记已无害化）。"""
        # 逐行讲解：检查条件 result.timed_out；成立时执行下方分支。
        if result.timed_out:
            # 逐行讲解：返回结果：f'Error: execution timed out after {self.timeout:.0f} seconds.'。
            return f"Error: execution timed out after {self.timeout:.0f} seconds."
        # 逐行讲解：检查条件 result.ok；成立时执行下方分支。
        if result.ok:
            # 逐行讲解：返回结果：sanitize_tool_content(result.stdout)。
            return sanitize_tool_content(result.stdout)  # 无 print 时为空字符串（官方语义）
        # 逐行讲解：检查条件 result.stderr；成立时执行下方分支。
        if result.stderr:
            # 逐行讲解：返回结果：sanitize_tool_content(result.stderr)。
            return sanitize_tool_content(result.stderr)
        # 逐行讲解：返回结果：f'Error: process exited with code {result.returncode}.'。
        return f"Error: process exited with code {result.returncode}."
