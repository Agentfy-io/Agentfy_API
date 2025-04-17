import aiohttp
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


async def verify_tikhub_api_key(request: Request):
    """验证TikHub API Key并返回有效的Key"""
    authorization = request.headers.get("authorization")
    logger.info(request.headers)

    if not authorization:
        raise HTTPException(status_code=401, detail="请提供有效的TikHub API密钥，格式: YOUR_API_KEY")

    api_key = authorization.replace("Bearer ", "")

    # 验证API密钥是否有效
    base_url = "https://api.tikhub.io"  # 或从设置中获取
    test_url = f"{base_url}/api/v1/tikhub/user/get_user_info"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(test_url, headers=headers) as response:
                if response.status == 200:
                    return api_key
                elif response.status == 401:
                    logger.warning(f"TikHub API密钥无效: {api_key[:5]}...")
                    raise HTTPException(status_code=401, detail="TikHub API密钥无效")
                else:
                    logger.warning(f"TikHub API验证请求返回状态码: {response.status}")
                    # 暂时允许其他状态码通过，可能是TikHub API的临时问题
                    return api_key
    except Exception as e:
        logger.error(f"验证TikHub API密钥时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"验证TikHub API密钥时发生错误: {str(e)}")

async def verify_openai_api_key(request: Request):
    """验证OpenAI API Key并返回有效的Key"""
    authorization = request.headers.get("openai-authorization")

    if not authorization:
        raise HTTPException(status_code=401, detail="请提供有效的OpenAI API密钥，格式: YOUR_API_KEY")

    api_key = authorization.replace("Bearer ", "")

    return api_key

async def verify_claude_api_key(request: Request):
    """验证Claude API Key并返回有效的Key"""
    authorization = request.headers.get("claude-authorization")

    if not authorization:
        raise HTTPException(status_code=401, detail="请提供有效的Claude API密钥，格式: YOUR_API_KEY")

    api_key = authorization.replace("Bearer ", "")

    return api_key


async def verify_lemonfox_api_key(request: Request):
    """验证LemonFox API Key并返回有效的Key"""
    authorization = request.headers.get("lemonfox-authorization")

    if not authorization:
        raise HTTPException(status_code=401, detail="请提供有效的LemonFox API密钥，格式: YOUR_API_KEY")

    api_key = authorization.replace("Bearer ", "")

    return api_key

async def verify_elevenlabs_api_key(request: Request):
    """验证ElevenLabs API Key并返回有效的Key"""
    authorization = request.headers.get("elevenlabs-authorization")

    if not authorization:
        raise HTTPException(status_code=401, detail="请提供有效的ElevenLabs API密钥，格式: YOUR_API_KEY")

    api_key = authorization.replace("Bearer ", "")

    return api_key

