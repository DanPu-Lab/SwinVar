import time
from pathlib import Path
from typing import Callable, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel

from swinvar.preprocess.utils import check_directory


@dataclass
class TaskResult:
    """任务结果数据类"""

    success: bool
    file_name: str
    result: Any = None
    error: Optional[str] = None


class Parallelizer:
    """并行任务处理器类

    该类提供了一个高级的并行任务处理接口，支持进度显示、
    错误处理和日志记录功能。

    Attributes:
        console (Console): Rich控制台对象
        results (List[TaskResult]): 任务结果列表
    """

    def __init__(self, console: Optional[Console] = None):
        """初始化并行任务处理器

        Args:
            console: Rich控制台对象，如果为None则创建新的控制台对象
        """
        self.console = console or Console()
        self.results: List[TaskResult] = []

    def _format_time(self, seconds: float) -> str:
        """格式化时间显示

        Args:
            seconds: 秒数

        Returns:
            格式化的时间字符串
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:.0f}h{minutes:.0f}m{seconds:.0f}s"

    def _print_start_banner(
        self, func_name: str, total_tasks: int, max_workers: int
    ) -> None:
        """打印开始横幅

        Args:
            func_name: 函数名
            total_tasks: 总任务数
            max_workers: 最大工作进程数
        """
        banner = Panel(
            f"[bold blue]Parallel Task Processor[/bold blue]\n\n"
            f"Function: [green]{func_name}[/green]\n"
            f"Total Tasks: [yellow]{total_tasks}[/yellow]\n"
            f"Worker Processes: [cyan]{max_workers}[/cyan]",
            title="Task Started",
            border_style="blue",
        )
        self.console.print(banner)

    def _print_summary(self, func_name: str, total_time: float) -> None:
        """打印执行摘要

        Args:
            func_name: 函数名
            total_time: 总执行时间
        """
        # 统计成功和失败的任务
        successful_tasks = sum(1 for result in self.results if result.success)
        failed_tasks = len(self.results) - successful_tasks

        # 创建摘要表格
        table = Table(title=f"Execution Summary - {func_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Tasks", str(len(self.results)))
        table.add_row("Successful Tasks", f"[green]{successful_tasks}[/green]")
        table.add_row("Failed Tasks", f"[red]{failed_tasks}[/red]")
        table.add_row("Execution Time", f"[yellow]{self._format_time(total_time)}[/yellow]")
        table.add_row(
            "Success Rate",
            (
                f"[green]{successful_tasks/len(self.results)*100:.1f}%[/green]"
                if self.results
                else "0%"
            ),
        )

        self.console.print(table)

        # 如果有失败的任务，显示失败详情
        if failed_tasks > 0:
            self.console.print("\n[bold red]Failed Mission Details:[/bold red]")
            for result in self.results:
                if not result.success:
                    self.console.print(
                        f"  • [red]{result.file_name}[/red]: {result.error}"
                    )

    def _process_task(
        self,
        future,
        args: Tuple,
        func_name: str,
        pool_fn_index: int,
        progress: Optional[Progress] = None,
    ) -> TaskResult:
        """处理单个任务

        Args:
            future: Future对象
            args: 任务参数
            func_name: 函数名
            pool_fn_index: 文件名在参数中的索引
            progress: 进度条对象

        Returns:
            任务结果
        """
        try:
            result = future.result()
            file_name = Path(args[pool_fn_index]).stem

            if progress:
                progress.console.print(f"✅ [green]{file_name}[/green] Completed")

            return TaskResult(success=True, file_name=file_name, result=result)

        except Exception as e:
            file_name = Path(args[pool_fn_index]).stem

            # 打印完整错误
            import traceback
            error_msg = traceback.format_exc()

            if progress:
                progress.console.print(f"❌ [red]{file_name}[/red] Failure: {str(error_msg)}")

            return TaskResult(success=False, file_name=file_name, error=str(e))

    def execute(
        self,
        func: Callable,
        args_list: List[Tuple],
        pool_fn_index: int,
        max_workers: int = 4,
        show_progress: bool = True,
        use_threads: bool = False,
    ) -> List[TaskResult]:
        """执行并行任务

        Args:
            func: 要执行的函数
            args_list: 参数列表
            pool_fn_index: 文件名在参数中的索引
            max_workers: 最大工作进程数，0表示使用所有可用CPU
            show_progress: 是否显示进度条

        Returns:
            任务结果列表
        """
        start_time = time.time()
        func_name = func.__name__
        total_tasks = len(args_list)

        # 清空之前的结果
        self.results = []

        # 打印开始横幅
        self._print_start_banner(func_name, total_tasks, max_workers)

        # 执行并行任务
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        with Executor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(func, *args): args for args in args_list}

            # 创建进度条
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=self.console,
                ) as progress:
                    task_progress = progress.add_task(
                        f"[green]执行 {func_name}", total=total_tasks
                    )

                    # 处理完成的任务
                    for future in as_completed(futures):
                        result = self._process_task(
                            future, futures[future], func_name, pool_fn_index, progress
                        )
                        self.results.append(result)
                        progress.update(task_progress, advance=1)
            else:
                # 不显示进度条
                for future in as_completed(futures):
                    result = self._process_task(
                        future, futures[future], func_name, pool_fn_index
                    )
                    self.results.append(result)

        # 计算总执行时间
        end_time = time.time()
        total_time = end_time - start_time

        # 打印摘要
        self._print_summary(func_name, total_time)

        return self.results

    def get_successful_results(self) -> List[TaskResult]:
        """获取成功的结果

        Returns:
            成功的任务结果列表
        """
        return [result for result in self.results if result.success]

    def get_failed_results(self) -> List[TaskResult]:
        """获取失败的结果

        Returns:
            失败的任务结果列表
        """
        return [result for result in self.results if not result.success]

    def clear_results(self) -> None:
        """清空结果列表"""
        self.results = []
