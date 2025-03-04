from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime


class UserAuth(BaseModel):
    """用户认证模型"""
    tikhub_api_key: str = Field(..., description="TikHub API密钥")

    @validator('tikhub_api_key')
    def validate_tikhub_api_key(cls, v):
        if not v or not isinstance(v, str) or len(v) < 8:
            raise ValueError("TikHub API密钥必须是有效的字符串且长度至少为8位")
        return v

class AuthResponse(BaseModel):
    """认证响应模型"""
    success: bool = Field(..., description="是否认证成功")
    message: str = Field(..., description="认证结果消息")
    session_id: Optional[str] = Field(None, description="会话ID")
    expires_at: Optional[datetime] = Field(None, description="过期时间")