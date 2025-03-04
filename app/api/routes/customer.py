# -*- coding: utf-8 -*-
"""
@file: customer.py
@desc: FastAPI 客户端路由
@auth: Callmeiks
"""
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
from app.dependencies import verify_tikhub_api_key  # 从dependencies.py导入验证函数
from app.config import settings

# 设置日志记录器
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/customers")


# 依赖项：获取CustomerAgent实例
async def get_customer_agent(tikhub_api_key: str = Depends(verify_tikhub_api_key)):
    """使用验证后的TikHub API Key创建CustomerAgent实例"""
    return CustomerAgent(tikhub_api_key=tikhub_api_key, tikhub_base_url=settings.TIKHUB_BASE_URL)


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
    description="""
用途:
   * 获取TikTok视频评论数据，返回清洗后的评论列表
参数:
   * aweme_id: TikTok视频ID
返回:
   * 评论列表（包括评论ID、评论内容、点赞数、回复数、评论者用户名、评论者安全用户ID(SecUid)、评论语言、评论者国家、Instagram ID、Twitter ID、创建时间）
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
    description="""
用途:
   * 分析TikTok视频评论中的购买意图，计算购买意图比率及各兴趣水平的购买意图数量
参数:
   * aweme_id: TikTok视频ID
   * batch_size: 每批处理评论数量，默认30
   * concurency: ai处理并发数，默认5，最大10
返回:
   * 视频评论购买意图统计数据
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
    description="""
用途:
   * 根据评论识别TikTok视频的购买意向潜在客户
参数:
   * aweme_id: TikTok视频ID
   * batch_size: 每批处理评论数量，默认30
   * min_score: 最小购买意向分数，范围0-100，默认50
   * max_score: 最大购买意向分数，范围1-100，默认100
返回:
   * 潜在客户信息（用户名、ID、评论内容、国家、社交媒体ID、创建时间、购买意向分数）
""",
    response_model_exclude_none=True,
)
async def identify_potential_customers(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每批处理的评论数量"),
        min_score: int = Query(50, description="最小参与度分数，范围0-100"),
        max_score: int = Query(100, description="最大参与度分数，范围1-100"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    根据评论识别TikTok视频的潜在客户

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每批处理的评论数量，默认为30
    - **min_score**: 最小参与度分数，范围0-100，默认为50
    - **max_score**: 最大参与度分数，范围1-100，默认为100

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


@router.post(
    "/fetch_keyword_potential_customers",
    summary="获取关键词搜索视频中的购买需求客户信息",
    description="""
用途:
   * 根据关键词搜索视频并识别其中具有购买意向的潜在客户
参数:
   * keyword: 搜索关键词
   * batch_size: 每批处理评论数量，默认30
   * video_concurrency: 视频处理并发数，默认5
   * ai_concurrency: AI处理并发数，默认5
   * min_score: 最小购买意向分数，范围0-100，默认50
   * max_score: 最大购买意向分数，范围1-100，默认100
   * ins_filter: 是否过滤Instagram为空的用户，默认False
   * twitter_filter: 是否过滤Twitter为空的用户，默认False
   * region_filter: 按地区过滤用户，默认不过滤
   * max_customers: 最大返回客户数量，默认不限制
返回:
   * 潜在客户信息（关键词、客户列表、统计数据）
""",
    response_model_exclude_none=True,
)
async def identify_keyword_potential_customers(
        request: Request,
        keyword: str = Query(..., description="搜索关键词"),
        batch_size: int = Query(30, description="每批处理评论数量"),
        video_concurrency: int = Query(5, description="视频处理并发数"),
        ai_concurrency: int = Query(5, description="AI处理并发数"),
        min_score: float = Query(50.0, description="最小购买意向分数，范围0-100"),
        max_score: float = Query(100.0, description="最大购买意向分数，范围1-100"),
        ins_filter: bool = Query(False, description="是否过滤Instagram为空的用户"),
        twitter_filter: bool = Query(False, description="是否过滤Twitter为空的用户"),
        region_filter: Optional[str] = Query(None, description="按地区过滤用户"),
        max_customers: Optional[int] = Query(None, description="最大返回客户数量"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    根据关键词搜索视频并识别其中具有购买意向的潜在客户

    - **keyword**: 搜索关键词
    - **batch_size**: 每批处理的评论数量，默认为30
    - **video_concurrency**: 视频处理并发数，默认为5
    - **ai_concurrency**: AI处理并发数，默认为5
    - **min_score**: 最小购买意向分数，范围0-100，默认为50
    - **max_score**: 最大购买意向分数，范围1-100，默认为100
    - **ins_filter**: 是否过滤Instagram为空的用户，默认为False
    - **twitter_filter**: 是否过滤Twitter为空的用户，默认为False
    - **region_filter**: 按地区过滤用户，默认不过滤
    - **max_customers**: 最大返回客户数量，默认不限制

    返回关键词搜索的潜在客户分析结果
    """
    start_time = time.time()

    try:
        logger.info(f"识别关键词 '{keyword}' 搜索结果中的潜在客户")

        result = await customer_agent.get_keyword_potential_customers(
            keyword=keyword,
            batch_size=batch_size,
            video_concurrency=video_concurrency,
            ai_concurrency=ai_concurrency,
            min_score=min_score,
            max_score=max_score,
            ins_filter=ins_filter,
            twitter_filter=twitter_filter,
            region_filter=region_filter,
            max_customers=max_customers
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

    except ValueError as e:
        logger.error(f"参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        logger.error(f"运行时错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"识别关键词潜在客户时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")