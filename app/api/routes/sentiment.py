# -*- coding: utf-8 -*-
"""
@file: sentiment.py
@desc: FastAPI 客户端路由
@auth: Callmeiks
"""
import json

from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request, Body
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
from app.api.models.responses import create_response
from agents.sentiment_agent import SentimentAgent
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
router = APIRouter(prefix="/sentiment")


# 依赖项：获取CustomerAgent实例
async def get_sentiment_agent(tikhub_api_key: str = Depends(verify_tikhub_api_key)):
    """使用验证后的TikHub API Key创建CustomerAgent实例"""
    return SentimentAgent(tikhub_api_key=tikhub_api_key, tikhub_base_url=settings.TIKHUB_BASE_URL)


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
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取指定TikTok视频的评论数据

    - **aweme_id**: TikTok视频ID

    返回清理后的评论列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取视频 {aweme_id} 的评论")

        comments_data = await sentiment_agent.fetch_video_comments(aweme_id)

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
    "/analyze_sentiment",
    summary="获取评论舆情分析结果",
    description="""
用途:
    * 获取评论的情感分析结果
参数:
    * comments: 评论列表
返回:
    * 情感分析结果列表（包括评论ID、评论内容、情感分析结果、情感分析置信度）
""",
    response_model_exclude_none=True,
)
async def analyze_sentiment(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(50, description="每次处理的评论数量"),
        concurrency: int = Query(5, description="并发请求数"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取评论的情感分析结果

    - **comments**: 评论列表

    返回情感分析结果列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取评论情感分析结果")

        sentiment_data = await sentiment_agent.analyze_sentiment(aweme_id, batch_size, concurrency)

        processing_time = time.time() - start_time

        return create_response(
            data=sentiment_data,
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
        logger.error(f"获取评论情感分析结果时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@router.post(
    "/analyze_relationship",
    summary="获取观众与UP主的关系分析结果",
    description="""
用途:
    * 获取评论区观众与UP主的关系分析结果
参数:
    * aweme_id: TikTok视频ID
返回:
    * 关系分析结果列表（包括评论ID、评论内容、评论者用户名、评论者安全用户ID(SecUid)、关系分析结果、关系分析置信度）
""",
    response_model_exclude_none=True,
)
async def analyze_relationship(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每次处理的评论数量"),
        concurrency: int = Query(5, description="并发请求数"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取评论关系分析结果

    - **aweme_id**: TikTok视频ID

    返回关系分析结果列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取评论关系分析结果")

        relationship_data = await sentiment_agent.analyze_relationship(aweme_id, batch_size, concurrency)

        processing_time = time.time() - start_time

        return create_response(
            data=relationship_data,
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
        logger.error(f"获取评论关系分析结果时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@router.post(
    "/fetch_audience_info",
    summary="获取观众粉丝信息",
    description="""
用途:
    * 获取评论区各类观众粉丝信息
参数:
    * aweme_id: TikTok视频ID
返回:
    * 观众粉丝信息列表（包括评论ID、评论内容、评论者用户名、评论者安全用户ID(SecUid)、粉丝信息）
""",
    response_model_exclude_none=True,
)
async def fetch_audience_info(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每次处理的评论数量"),
        concurrency: int = Query(5, description="并发请求数"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取评论区各类粉丝信息

    - **aweme_id**: TikTok视频ID

    返回粉丝信息列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取评论区各类粉丝信息")

        audience_data = await sentiment_agent.fetch_audience_info(aweme_id, batch_size, concurrency)

        processing_time = time.time() - start_time

        return create_response(
            data=audience_data,
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
        logger.error(f"获取评论区各类粉丝信息时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@router.post(
    "/analyze_toxicity",
    summary="获取评论区黑评/差评分析结果",
    description="""
用途:
    * 获取评论区黑评/差评分析结果
参数:
    * aweme_id: TikTok视频ID
返回:
    * 黑评/差评分析结果列表（包括黑评/差评分析结果、黑评/差评置信度）
""",
    response_model_exclude_none=True,
)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")