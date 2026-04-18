from typing import Union
from pathlib import Path
import subprocess
import logging
import shlex
import sys


def check_directory(directory: Union[str, Path]) -> bool:
    """检查并创建目录

    Args:
        directory: 目录路径

    Returns:
        目录是否成功创建或已存在

    Raises:
        OSError: 当目录创建失败时
    """
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            directory_path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        raise OSError(f"创建目录失败 {directory}: {e}")


def execute_cmd(
    command,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=sys.stderr,
    bufsize=10 * 1024 * 1024,
):
    return subprocess.Popen(
        shlex.split(command),
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        bufsize=bufsize,
        universal_newlines=True,
    )


def setup_logger(filename, mode="w", level=logging.INFO, format="%(message)s"):
    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(filename, mode=mode)
    file_handler.setLevel(level)

    formatter = logging.Formatter(format)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
