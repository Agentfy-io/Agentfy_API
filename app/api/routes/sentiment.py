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


# 依赖项：获取SentimentAgent实例
async def get_sentiment_agent(tikhub_api_key: str = Depends(verify_tikhub_api_key)):
    """使用验证后的TikHub API Key创建SentimentAgent实例"""
    return SentimentAgent(tikhub_api_key=tikhub_api_key, tikhub_base_url=settings.TIKHUB_BASE_URL)


@router.post(
    "/fetch_video_comments",
    summary="【闪电抓取】一键获取指定视频关键评论数据",
    description="""
用途:
  * 快速获取并清洗TikTok视频评论数据
  * 评论列表（包括评论ID、评论内容、点赞数、回复数、评论者用户名、评论者安全用户ID(SecUid)、评论语言、评论者国家、Instagram ID、Twitter ID、创建时间）

参数:
  * aweme_id: TikTok视频ID

（让您一览评论全貌，洞察用户反馈！）
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
    summary="【深度舆情洞察】获取评论的情感分析统计数据",
    description="""
用途:
  * 深度挖掘评论的正负面情感与热点话题
  * 生成舆情分布图数据，观众情绪规律，互动规律，讨论主题等
  * 适用于内容创作者分析同赛道创作者的优劣势，以及用户对内容的反馈

参数:
  * aweme_id: TikTok视频ID
  * batch_size: 每批处理评论数量，默认30
  * concurrency: 并发请求数，默认5，最大不超过10

（让您瞬间捕捉用户情绪脉搏！）
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
    summary="【粉丝关系统计】获取观众与UP主关系分析统计",
    description="""
用途:
  * 挖掘评论区观众与UP主的互动关系和忠诚度
  * 忠诚度分析，语气分析，观众种类分析，粉丝分析等

参数:
  * aweme_id: TikTok视频ID
  * batch_size: 每批处理评论数量，默认30
  * concurrency：并发请求数，默认5，最大不超过10

（让创作者牢牢把握粉丝心声！）
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
    "/fetch_quality_audience_info",
    summary="【精英粉丝发掘】获取优质观众信息数据",
    description="""
用途:
  * 精准识别评论区核心粉丝、多社交账号互动者等优质受众
  * 返回观众粉丝信息列表（包括评论ID、评论内容、评论者用户名、评论者安全用户ID(SecUid)、粉丝信息）
参数:
  * aweme_id: TikTok视频ID
  * batch_size: 每批处理评论数量，默认30
  * concurrency: 并发请求数

（深度挖掘潜在KOC/KOL，与他们高效互动！）
""",
    response_model_exclude_none=True,
)
async def fetch_quality_audience_info(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每次处理的评论数量"),
        concurrency: int = Query(5, description="并发请求数"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取评论区各类优质观众粉丝信息

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每次处理的评论数量
    - **concurrency**: 并发请求数

    返回粉丝信息列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取评论区各类粉丝信息")

        audience_data = await sentiment_agent.fetch_quality_audience_info(aweme_id, batch_size, concurrency)

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
    summary="【毒性扫描】获取评论区负面评论分析统计结果",
    description="""
用途:
  * 快速定位黑评、差评、冲突性评论，支持高效舆情处置
  * 黑评/差评分析结果列表（包括黑评/差评分析结果、黑评/差评置信度）
  * 识别多种语言，多种场景类型

参数:
  * aweme_id: TikTok视频ID
  * batch_size: 每批处理评论数量，默认30
  * concurrency: 并发请求数，默认5，最大不超过10

（让风险预警立竿见影，形象维护从此事半功倍！）
""",
    response_model_exclude_none=True,
)
async def analyze_toxicity(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每次处理的评论数量"),
        concurrency: int = Query(5, description="并发请求数"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取评论区黑评/差评分析结果

    - **aweme_id**: TikTok视频ID

    返回黑评/差评分析结果列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取评论区黑评/差评分析结果")

        toxicity_data = await sentiment_agent.analyze_toxicity(aweme_id, batch_size, concurrency)

        processing_time = time.time() - start_time

        return create_response(
            data=toxicity_data,
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
        logger.error(f"获取评论区黑评/差评分析结果时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/fetch_negative_shop_reviews",
    summary="【店铺差评追踪】获取商品负面评价用户信息",
    description="""
用途:
  * 集中收集对商品、服务或店铺的负面评论与关联用户信息
  * 识别多种语言，多种场景类型
  * 返回差评评论者信息列表（包括评论ID、评论内容、评论者用户名、评论者安全用户ID(SecUid)）

参数:
  * aweme_id: TikTok视频ID
  * batch_size: 每批处理评论数量，默认30
  * concurrency: 并发请求数，默认5，最大不超过10

（助力及时解决售后问题，提升店铺好评率！）
""",
    response_model_exclude_none=True,
)
async def fetch_negative_shop_reviews(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每次处理的评论数量"),
        concurrency: int = Query(5, description="并发请求数"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取评论区商品差评的评论者信息

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每次处理的评论数量
    - **concurrency**: 并发请求数

    返回差评评论者信息列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取评论区商品差评的评论者信息")

        negative_reviews_data = await sentiment_agent.fetch_negative_shop_reviews(aweme_id, batch_size, concurrency)

        processing_time = time.time() - start_time

        return create_response(
            data=negative_reviews_data,
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
        logger.error(f"获取评论区商品差评的评论者信息时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/fetch_hate_speech",
    summary="【恶意言论狙击】获取评论区恶意言论用户信息",
    description="""
用途:
  * 追踪仇恨、攻击性言论的用户信息
  * 帮助社交媒体达人及时发现并处理恶意评论，避免声誉损害，维护粉丝群体和谐，提升自己的社交媒体形象
  * 返回恶意评论者信息列表（包括评论ID、评论内容、评论者用户名、评论者安全用户ID(SecUid)）

参数:
  * aweme_id: TikTok视频ID
  * batch_size: 每批处理评论数量，默认30
  * concurrency: 并发请求数，默认5，最大不超过10

（捍卫社区和谐，营造友好互动环境！）
""",
    response_model_exclude_none=True,
)
async def fetch_hate_speech(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每次处理的评论数量"),
        concurrency: int = Query(5, description="并发请求数"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取评论区恶意评论者信息

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每次处理的评论数量
    - **concurrency**: 并发请求数

    返回恶意评论者信息列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取评论区恶意评论者信息")

        hate_comments_data = await sentiment_agent.fetch_hate_speech(aweme_id, batch_size, concurrency)

        processing_time = time.time() - start_time

        return create_response(
            data=hate_comments_data,
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
        logger.error(f"获取评论区恶意评论者信息时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/fetch_spam_content",
    summary="【垃圾内容过滤】获取评论区滥发言论用户信息",
    description="""
用途:
  * 筛查评论区群发广告、诱导、欺诈等垃圾评论信息
  * 返回垃圾评论者信息列表（包括评论ID、评论内容、评论者用户名、评论者安全用户ID(SecUid)）

参数:
  * aweme_id: TikTok视频ID
  * batch_size: 每批处理评论数量，默认30
  * concurrency: 并发请求数，默认5，最大不超过10

（高效甄别无效信息，全面提升内容质量！）
""",
    response_model_exclude_none=True,
)
async def fetch_spam_content(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每次处理的评论数量"),
        concurrency: int = Query(5, description="并发请求数"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取评论区垃圾评论者信息

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每次处理的评论数量
    - **concurrency**: 并发请求数

    返回垃圾评论者信息列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取评论区垃圾评论者信息")

        spam_comments_data = await sentiment_agent.fetch_spam_content(aweme_id, batch_size, concurrency)

        processing_time = time.time() - start_time

        return create_response(
            data=spam_comments_data,
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
        logger.error(f"获取评论区垃圾评论者信息时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
