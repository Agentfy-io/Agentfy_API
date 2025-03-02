from fastapi import APIRouter, Depends, Form, HTTPException, Request
from typing import Dict, Any
import time
from datetime import datetime

from app.api.models.auth import UserAuth, AuthResponse
from app.api.models.responses import create_response
from services.auth.auth_service import AuthService
from app.core.exceptions import ValidationError, AuthorizationError
from app.utils.logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/auth")


@router.post(
    "/login",
    summary="用户认证",
    description="使用TikHub API密钥进行认证并创建会话",
    response_model_exclude_none=True,
    tags=["认证"]
)
async def login(
        request: Request,
        tikhub_api_key: str = Form(..., description="TikHub API密钥"),
        tikhub_base_url: str = Form("https://api.tikhub.io", description="TikHub API基础URL")
):
    """
    用户认证

    - **tikhub_api_key**: TikHub API密钥 (必填)
    - **tikhub_base_url**: TikHub API基础URL (可选，默认为https://api.tikhub.io)

    返回会话信息，包含会话ID和过期时间
    """
    start_time = time.time()

    try:
        # 验证用户输入
        user_auth = UserAuth(
            tikhub_api_key=tikhub_api_key,
            tikhub_base_url=tikhub_base_url
        )

        # 验证TikHub API密钥
        is_valid = await AuthService.verify_tikhub_api_key(
            user_auth.tikhub_api_key,
            user_auth.tikhub_base_url
        )

        if not is_valid:
            raise ValidationError(detail="TikHub API密钥无效", field="tikhub_api_key")

        # 创建会话
        session_id, expires_at = await AuthService.create_session(
            user_auth.tikhub_api_key,
            user_auth.tikhub_base_url
        )

        # 记录认证成功
        logger.info(f"用户认证成功，会话ID: {session_id[:8]}...")

        processing_time = time.time() - start_time

        # 返回认证信息
        auth_response = AuthResponse(
            success=True,
            message="认证成功",
            session_id=session_id,
            expires_at=expires_at
        )

        return create_response(
            data=auth_response.dict(),
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )

    except ValidationError as e:
        logger.error(f"认证验证错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"认证过程中发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/logout",
    summary="用户登出",
    description="销毁当前会话",
    response_model_exclude_none=True,
    tags=["认证"]
)
async def logout(
        request: Request,
        session_id: str = Form(..., description="会话ID")
):
    """
    用户登出

    - **session_id**: 会话ID (必填)

    返回登出结果
    """
    start_time = time.time()

    try:
        # 删除会话
        success = AuthService.remove_session(session_id)

        processing_time = time.time() - start_time

        if success:
            logger.info(f"用户成功登出，会话ID: {session_id[:8]}...")
            message = "登出成功"
        else:
            logger.warning(f"尝试登出无效会话: {session_id[:8]}...")
            message = "会话不存在或已过期"

        return create_response(
            data={"success": success, "message": message},
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )

    except Exception as e:
        logger.error(f"登出过程中发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.get(
    "/status",
    summary="会话状态",
    description="检查会话状态和剩余有效期",
    response_model_exclude_none=True,
    tags=["认证"]
)
async def session_status(
        request: Request,
        session_id: str
):
    """
    检查会话状态

    - **session_id**: 会话ID (必填)

    返回会话状态信息
    """
    start_time = time.time()

    try:
        # 获取会话
        session = AuthService.get_session(session_id)

        processing_time = time.time() - start_time

        if not session:
            return create_response(
                data={"valid": False, "message": "会话不存在或已过期"},
                success=True,
                processing_time_ms=round(processing_time * 1000, 2)
            )

        # 计算剩余有效期
        expires_at = datetime.fromisoformat(session["expires_at"])
        remaining_seconds = (expires_at - datetime.now()).total_seconds()
        remaining_hours = round(remaining_seconds / 3600, 1)

        return create_response(
            data={
                "valid": True,
                "message": "会话有效",
                "expires_at": session["expires_at"],
                "remaining_hours": remaining_hours,
                "created_at": session["created_at"]
            },
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )

    except Exception as e:
        logger.error(f"检查会话状态时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")