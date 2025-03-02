from fastapi import Depends, Header, HTTPException, Request
from typing import Optional
from datetime import datetime
import time

from app.config import settings
from agents.customer_agent import CustomerAgent
from app.utils.logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

# 全局CustomerAgent实例
_customer_agent = None


async def get_customer_agent() -> CustomerAgent:
    """
    获取CustomerAgent实例（单例模式）

    Returns:
        CustomerAgent: CustomerAgent实例
    """
    global _customer_agent

    if _customer_agent is None:
        _customer_agent = CustomerAgent()

    return _customer_agent


async def verify_api_key(
        x_api_key: Optional[str] = Header(None, description="API密钥")
) -> None:
    """
    验证API密钥

    Args:
        x_api_key: 请求头中的API密钥

    Returns:
        None

    Raises:
        HTTPException: 当API密钥无效时
    """
    # 如果未设置API_KEY_REQUIRED或为False，则不需要验证API密钥
    if not settings.API_KEY_REQUIRED:
        return

    # 验证API密钥
    valid_api_keys = settings.API_KEYS

    if not x_api_key:
        logger.warning("请求未提供API密钥")
        raise HTTPException(
            status_code=401,
            detail="缺少API密钥",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if x_api_key not in valid_api_keys:
        logger.warning(f"使用无效的API密钥: {x_api_key[:5]}...")
        raise HTTPException(
            status_code=401,
            detail="API密钥无效",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def log_request_middleware(request: Request, call_next):
    """
    请求日志中间件，记录请求信息和处理时间

    Args:
        request: FastAPI请求对象
        call_next: 下一个中间件处理函数

    Returns:
        响应对象
    """
    start_time = time.time()

    # 获取客户端IP
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(",")[0] if forwarded_for else request.client.host

    # 记录请求信息
    logger.info(
        f"开始请求: {request.method} {request.url.path} - "
        f"客户端: {client_ip}, "
        f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}"
    )

    # 处理请求
    response = await call_next(request)

    # 计算处理时间
    process_time = time.time() - start_time

    # 添加处理时间到响应头
    response.headers["X-Process-Time"] = str(process_time)

    # 记录响应信息
    logger.info(
        f"完成请求: {request.method} {request.url.path} - "
        f"状态码: {response.status_code}, "
        f"处理时间: {process_time:.4f}秒"
    )

    return response