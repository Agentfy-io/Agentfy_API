from datetime import datetime
from typing import Optional, Dict, Any, Union
from fastapi import HTTPException


class CommentAPIException(HTTPException):
    """自定义API异常基类"""

    def __init__(
            self,
            status_code: int,
            detail: str,
            error_type: str = "api_error",
            headers: Optional[Dict[str, Any]] = None,
            extra_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_type = error_type
        self.timestamp = datetime.now()
        self.extra_data = extra_data or {}


class ValidationError(CommentAPIException):
    """输入验证错误"""

    def __init__(
            self,
            detail: str = "Invalid input parameters",
            field: Optional[str] = None,
            headers: Optional[Dict[str, Any]] = None
    ):
        extra_data = {"field": field} if field else {}
        super().__init__(
            status_code=400,
            detail=detail,
            error_type="validation_error",
            headers=headers,
            extra_data=extra_data
        )


class NotFoundError(CommentAPIException):
    """资源未找到错误"""

    def __init__(
            self,
            detail: str = "The requested resource was not found",
            resource_type: Optional[str] = None,
            resource_id: Optional[Union[str, int]] = None,
            headers: Optional[Dict[str, Any]] = None
    ):
        extra_data = {}
        if resource_type:
            extra_data["resource_type"] = resource_type
        if resource_id:
            extra_data["resource_id"] = resource_id

        super().__init__(
            status_code=404,
            detail=detail,
            error_type="not_found_error",
            headers=headers,
            extra_data=extra_data
        )


class ExternalAPIError(CommentAPIException):
    """外部API错误"""

    def __init__(
            self,
            detail: str = "Error communicating with external API",
            service: Optional[str] = None,
            status_code: int = 502,
            headers: Optional[Dict[str, Any]] = None,
            original_error: Optional[Exception] = None
    ):
        extra_data = {}
        if service:
            extra_data["service"] = service
        if original_error:
            extra_data["original_error"] = str(original_error)

        super().__init__(
            status_code=status_code,
            detail=detail,
            error_type="external_api_error",
            headers=headers,
            extra_data=extra_data
        )


class AuthorizationError(CommentAPIException):
    """授权错误"""

    def __init__(
            self,
            detail: str = "Missing or invalid API key",
            headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=401,
            detail=detail,
            error_type="authorization_error",
            headers=headers
        )


class RateLimitError(CommentAPIException):
    """速率限制错误"""

    def __init__(
            self,
            detail: str = "Rate limit exceeded",
            retry_after: Optional[int] = None,
            headers: Optional[Dict[str, Any]] = None
    ):
        extra_data = {}
        if retry_after:
            extra_data["retry_after"] = retry_after

        super().__init__(
            status_code=429,
            detail=detail,
            error_type="rate_limit_error",
            headers=headers,
            extra_data=extra_data
        )


class InternalServerError(CommentAPIException):
    """内部服务器错误"""

    def __init__(
            self,
            detail: str = "An unexpected error occurred",
            headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=500,
            detail=detail,
            error_type="internal_server_error",
            headers=headers
        )