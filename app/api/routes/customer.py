from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

from app.api.models.customer import (
    VideoCommentsRequest,
    VideoCommentsResponse,
    PurchaseIntentRequest,
    PurchaseIntentAnalysis,
    PotentialCustomersRequest,
    PotentialCustomersAnalysis
)
from app.api.models.responses import create_response
from agents.customer_agent import CustomerAgent
from app.core.exceptions import (
    ValidationError,
    ExternalAPIError,
    InternalServerError,
    NotFoundError
)
from app.utils.logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/customers")


# 依赖项：获取CommentAgent实例
async def get_customer_agent():
    return CustomerAgent()

@router.get(
    "/health",
    summary="健康检查",
    description="检查API是否正常运行",
    tags=["健康检查"]
)
async def health_check():
    """简单的健康检查端点"""
    return create_response(
        data={"status": "healthy", "timestamp": datetime.now().isoformat()},
        success=True
    )


@router.post(
    "/fetch_video_comments",
    summary="获取指定视频关键评论数据",
    description=f"""
获取指定视频评论数据，并返回清洗后的评论列表，包括评论ID，评论内容、点赞数、回复数，评论者用户名、评论者安全用户ID(SecUid)、评论语言、评论者国家、Instagram ID、Twitter ID、创建时间。内置分页功能，默认每页30条评论。内置并发处理功能，默认并发数为5。
- **aweme_id**: TikTok视频ID
    """,
    response_model_exclude_none=True,
)
async def fetch_video_comments(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    获取指定TikTok视频的评论数据

    - **aweme_id**: TikTok视频ID

    返回清理后的评论列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取视频 {aweme_id} 的评论")

        comments_data = await customer_agent.fetch_video_comments(aweme_id)

        processing_time = time.time() - start_time

        return create_response(
            data=comments_data,
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )

    except ValidationError as e:
        logger.error(f"验证错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"获取视频评论时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/fetch_purchase_intent_stats",
    summary="分析指定视频购买意图",
    description=f"""分析指定TikTok视频评论中的购买意图，计算购买意图比率、各兴趣水平的购买意图数量. 
- **aweme_id**: TikTok视频ID
- **batch_size**: 每批处理的评论数量，默认为30
- **concurency**: ai处理并发数，默认为5, 最大为10
""",
    response_model_exclude_none=True,
)

async def analyze_purchase_intent(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size : int = Query(30, description="每批处理的评论数量"),
        concurency: int = Query(5, description="ai处理并发数"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    分析指定TikTok视频评论中的购买意图

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每批处理的评论数量，默认为30
    - **concurency**: ai处理并发数，默认为5, 最大为10

    返回购买意图分析结果
    """
    start_time = time.time()

    try:
        logger.info(f"分析视频 {aweme_id} 的购买意图")

        result = await customer_agent.get_purchase_intent_stats(
            aweme_id,
            batch_size,
            concurency
        )

        processing_time = time.time() - start_time

        return create_response(
            data=result,
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )

    except ValidationError as e:
        logger.error(f"验证错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except InternalServerError as e:
        logger.error(f"内部服务器错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"分析购买意图时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/fetch_potential_customers",
    summary="获取指定视频有购买需求客户信息",
    description=f"""根据评论识别TikTok视频的购买意向的潜在客户.返回用户名、用户ID、评论内容、评论者国家、Instagram ID、Twitter ID、创建时间、购买意向分数。未设置的社交媒体ID将返回None。
- batch_size: 每批处理的评论数量，默认为30
- min_score: 最小参与度分数，范围0-100，默认为50
- max_score: 最大参与度分数，范围1-100，默认为100
    """,
    response_model_exclude_none=True,
)
async def identify_potential_customers(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每批处理的评论数量"),
        min_score: int = Query(1, description="最小参与度分数，范围0-100"),
        max_score: int = Query(100, description="最大参与度分数，范围1-100"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    根据评论识别TikTok视频的潜在客户

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每批处理的评论数量，默认为30
    - **min_score**: 最小参与度分数，范围0-100，默认为50
    - **max_score**: 最大参与度分数，范围0-100，默认为100

    返回潜在客户分析结果
    """
    start_time = time.time()

    try:
        logger.info(f"识别视频 {aweme_id} 的潜在客户")

        result = await customer_agent.get_potential_customers(
            aweme_id,
            batch_size,
            min_score,
            max_score
        )

        processing_time = time.time() - start_time

        return create_response(
            data=result,
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )

    except ValidationError as e:
        logger.error(f"验证错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except InternalServerError as e:
        logger.error(f"内部服务器错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"识别潜在客户时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.get(
    "/{aweme_id}",
    summary="根据ID获取视频评论",
    description="使用GET方法获取指定TikTok视频的评论数据",
    response_model_exclude_none=True,
)
async def get_video_comments(
        request: Request,
        aweme_id: str = Path(..., description="TikTok视频ID"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    使用GET方法获取指定TikTok视频的评论数据

    - **aweme_id**: TikTok视频ID（路径参数）

    返回清理后的评论列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取视频 {aweme_id} 的评论")

        if not aweme_id:
            raise ValidationError(detail="视频ID不能为空", field="aweme_id")

        comments_data = await customer_agent.fetch_video_comments(aweme_id)

        processing_time = time.time() - start_time

        return create_response(
            data=comments_data,
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )

    except ValidationError as e:
        logger.error(f"验证错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"获取视频评论时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")