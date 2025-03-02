import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(name, level=None):
    """
    设置并返回一个配置好的日志记录器

    Args:
        name (str): 日志记录器名称
        level (str, optional): 日志级别，可以是'debug', 'info', 'warning', 'error', 'critical'
                              如果未指定，则从环境变量LOG_LEVEL读取，默认为'info'

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "info").upper()

    # 创建日志记录器
    logger = logging.getLogger(name)

    # 如果日志记录器已经有处理器，则不重复配置
    if logger.handlers:
        return logger

    # 设置日志级别
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 创建文件处理器 (每个文件最大10MB，保留10个备份文件)
    file_handler = RotatingFileHandler(
        log_dir / f"{name.replace('.', '_')}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# 创建默认的日志记录器
logger = setup_logger("agentfy_comment_api")