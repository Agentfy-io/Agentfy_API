import uuid
import time
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

from fastapi import Request, Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.utils.logger import setup_logger
from app.core.exceptions import AuthorizationError
from app.config import settings

# 设置日志记录器
logger = setup_logger(__name__)

# 创建API密钥头
api_key_header = APIKeyHeader(name="X-Session-ID", auto_error=False)

# 用户会话存储（实际应用中应使用Redis或数据库）
user_sessions = {}


class AuthService:
    """认证服务，管理用户会话和TikHub API密钥"""

    @staticmethod
    async def verify_tikhub_api_key(api_key: str, base_url: str) -> bool:
        """
        验证TikHub API密钥是否有效

        Args:
            api_key: TikHub API密钥
            base_url: TikHub API基础URL

        Returns:
            bool: 密钥是否有效
        """
        try:
            # 构建测试接口URL（使用一个简单的TikHub API端点）
            test_url = f"{base_url.rstrip('/')}/api/v1/tiktok/check_health"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(test_url, headers=headers) as response:
                    if response.status == 200:
                        return True
                    elif response.status == 401:
                        logger.warning(f"TikHub API密钥无效: {api_key[:5]}...")
                        return False
                    else:
                        logger.warning(f"TikHub API验证请求返回状态码: {response.status}")
                        # 暂时允许其他状态码通过，可能是TikHub API的临时问题
                        return True
        except Exception as e:
            logger.error(f"验证TikHub API密钥时出错: {str(e)}")
            # 在无法验证的情况下，仍然允许用户使用该密钥
            return True

    @staticmethod
    async def create_session(api_key: str, base_url: str) -> Tuple[str, datetime]:
        """
        创建新的用户会话

        Args:
            api_key: TikHub API密钥
            base_url: TikHub API基础URL

        Returns:
            Tuple[str, datetime]: 会话ID和过期时间
        """
        # 生成会话ID
        session_id = str(uuid.uuid4())

        # 设置过期时间（24小时后）
        expires_at = datetime.now() + timedelta(hours=24)

        # 存储会话信息
        user_sessions[session_id] = {
            "tikhub_api_key": api_key,
            "tikhub_base_url": base_url,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat()
        }

        logger.info(f"已创建新会话: {session_id[:8]}...")
        return session_id, expires_at

    @staticmethod
    def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            Optional[Dict[str, Any]]: 会话信息或None
        """
        session = user_sessions.get(session_id)

        if not session:
            return None

        # 检查会话是否已过期
        expires_at = datetime.fromisoformat(session["expires_at"])
        if expires_at < datetime.now():
            # 删除过期会话
            AuthService.remove_session(session_id)
            return None

        return session

    @staticmethod
    def remove_session(session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            bool: 是否成功删除
        """
        if session_id in user_sessions:
            del user_sessions[session_id]
            logger.info(f"已删除会话: {session_id[:8]}...")
            return True
        return False

    @staticmethod
    def clean_expired_sessions() -> int:
        """
        清理所有过期会话

        Returns:
            int: 清理的会话数量
        """
        now = datetime.now()
        expired_count = 0

        for session_id in list(user_sessions.keys()):
            expires_at = datetime.fromisoformat(user_sessions[session_id]["expires_at"])
            if expires_at < now:
                del user_sessions[session_id]
                expired_count += 1

        logger.info(f"已清理 {expired_count} 个过期会话")
        return expired_count


async def get_current_user_api_keys(
        request: Request,
        session_id: str = Depends(api_key_header)
) -> Dict[str, str]:
    """
    获取当前用户的API密钥，用作依赖项

    Args:
        request: FastAPI请求对象
        session_id: 会话ID（从请求头中获取）

    Returns:
        Dict[str, str]: 包含tikhub_api_key和tikhub_base_url的字典

    Raises:
        AuthorizationError: 当会话ID无效或过期时
    """
    if not session_id:
        # 检查请求体中是否有会话ID
        try:
            body = await request.json()
            session_id = body.get("session_id")
        except:
            pass

        # 检查查询参数中是否有会话ID
        if not session_id:
            session_id = request.query_params.get("session_id")

    if not session_id:
        raise AuthorizationError(detail="未提供会话ID，请先认证")

    session = AuthService.get_session(session_id)

    if not session:
        raise AuthorizationError(detail="会话无效或已过期，请重新认证")

    return {
        "tikhub_api_key": session["tikhub_api_key"],
        "tikhub_base_url": session["tikhub_base_url"]
    }