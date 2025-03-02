from typing import Generic, TypeVar, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

# 泛型类型变量
T = TypeVar('T')


class ErrorDetail(BaseModel):
    """错误详情模型"""
    code: int = Field(..., description="HTTP状态码")
    message: str = Field(..., description="错误详细描述")
    type: str = Field(..., description="错误类型")


class ResponseMetadata(BaseModel):
    """响应元数据模型"""
    request_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = Field(None, description="请求处理时间（毫秒）")
    pagination: Optional[Dict[str, Any]] = Field(None, description="分页信息")
    rate_limit: Optional[Dict[str, Any]] = Field(None, description="速率限制信息")
    version: str = Field("1.0.0", description="API版本")


class APIResponse(BaseModel, Generic[T]):
    """通用API响应模型，使用泛型支持不同的数据类型"""
    success: bool = Field(..., description="操作是否成功")
    data: Optional[T] = Field(None, description="响应数据")
    error: Optional[ErrorDetail] = Field(None, description="错误详情，仅在success=False时存在")
    meta: ResponseMetadata = Field(default_factory=ResponseMetadata)

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {},
                "error": None,
                "meta": {
                    "request_id": "20240228123456789",
                    "timestamp": "2024-02-28T12:34:56.789Z",
                    "processing_time_ms": 234.5,
                    "pagination": None,
                    "rate_limit": None,
                    "version": "1.0.0"
                }
            }
        }


class PaginatedResponseMetadata(ResponseMetadata):
    """带分页的响应元数据模型"""
    pagination: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total": 0,
            "page": 1,
            "page_size": 10,
            "total_pages": 0
        },
        description="分页信息"
    )


class PaginatedResponse(APIResponse, Generic[T]):
    """分页响应模型"""
    meta: PaginatedResponseMetadata = Field(default_factory=PaginatedResponseMetadata)
    data: List[T] = Field(default_factory=list)


def create_response(
        data: Any = None,
        success: bool = True,
        error: Optional[Dict[str, Any]] = None,
        processing_time_ms: Optional[float] = None,
        pagination: Optional[Dict[str, Any]] = None,
        rate_limit: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建一个标准的API响应字典

    Args:
        data: 响应数据
        success: 操作是否成功
        error: 错误详情
        processing_time_ms: 处理时间（毫秒）
        pagination: 分页信息
        rate_limit: 速率限制信息
        request_id: 请求ID

    Returns:
        标准格式的响应字典
    """
    meta = {
        "request_id": request_id or datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

    if processing_time_ms is not None:
        meta["processing_time_ms"] = processing_time_ms

    if pagination is not None:
        meta["pagination"] = pagination

    if rate_limit is not None:
        meta["rate_limit"] = rate_limit

    response = {
        "success": success,
        "data": data,
        "error": error,
        "meta": meta
    }

    return response