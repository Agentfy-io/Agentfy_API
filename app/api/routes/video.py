# -*- coding: utf-8 -*-
"""
@file: video.py
@desc: FastAPI 客户端路由
@auth: Callmeiks
"""
import json

from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request, Body
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
from app.api.models.responses import create_response
from agents.video_agent import VideoAgent
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
router = APIRouter(prefix="/video")


# 依赖项：获取VideoAgent实例
async def get_video_agent(tikhub_api_key: str = Depends(verify_tikhub_api_key)):
    """使用验证后的TikHub API Key创建VideoAgent实例"""
    return VideoAgent(tikhub_api_key=tikhub_api_key, tikhub_base_url=settings.TIKHUB_BASE_URL)


@router.post(
    "/fetch_single_video_data",
    summary="获取/清洗 视频关键数据",
    description="""
用途:
   * 获取TikTok视频数据，返回清洗后的app端视频数据
参数:
   * aweme_id: TikTok视频ID
返回: 视频点赞，评论，转发等数据。相关音乐，视频创作者，视频描述，视频设置等数据
   * 
""",
    response_model_exclude_none=True,
)
async def fetch_single_video_data(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        video_agent: VideoAgent = Depends(get_video_agent)
):
    """
    获取指定TikTok视频的\数据

    - **aweme_id**: TikTok视频ID

    返回清理后`app`端的视频数据
    """
    start_time = time.time()

    try:
        logger.info(f"获取视频 {aweme_id} 的评论")

        video_data = video_agent.fetch_video_data(aweme_id)

        processing_time = time.time() - start_time

        return create_response(
            data=video_data,
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
    "/analyze_video_info",
    summary="分析/生成 视频统计数据报告",
    description="""
用途:
    * 分析TikTok视频数据，返回一个视频报告
参数:
    * aweme_id: TikTok视频ID
返回: 
    * 视频数据分析报告
""",
    response_model_exclude_none=True,
)
async def analyze_video_info(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        video_agent: VideoAgent = Depends(get_video_agent)
):
    """
    分析TikTok视频数据

    - **video_data**: TikTok视频数据

    返回视频数据分析报告
    """
    start_time = time.time()

    try:
        logger.info(f"分析视频统计数据")

        video_report = video_agent.analyze_video_info(aweme_id)

        processing_time = time.time() - start_time

        return create_response(
            data=video_report,
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
        logger.error(f"分析视频数据时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@router.post(
    "/fetch_video_transcript",
    summary="分析/转录 视频音频",
    description="""
用途:
    * 分析TikTok视频字幕
参数:
    * aweme_id: TikTok视频ID
返回:
    * 视频字幕分析报告
""",
    response_model_exclude_none=True,
)
async def fetch_video_transcript(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        video_agent: VideoAgent = Depends(get_video_agent)
):
    """
    分析TikTok视频字幕

    - **video_data**: TikTok视频数据

    返回视频字幕分析报告
    """
    start_time = time.time()

    try:
        logger.info(f"分析视频字幕")

        video_transcript = video_agent.fetch_video_transcript(aweme_id)

        processing_time = time.time() - start_time

        return create_response(
            data=video_transcript,
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
        logger.error(f"分析视频字幕时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@router.post(
    "/analyze_video_frames",
    summary="分析/获取 视频关键帧内容",
    description="""
用途:
    * 分析TikTok视频帧
参数:
    * aweme_id: TikTok视频ID
    * frame_interval: 分析帧之间的间隔（秒）
返回:
    * 视频帧分析报告
""",
    response_model_exclude_none=True,
)
async def analyze_video_frames(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        frame_interval: float = Query(2.0, description="分析帧之间的间隔（秒）"),
        video_agent: VideoAgent = Depends(get_video_agent)
):
    """
    分析TikTok视频帧

    - **video_data**: TikTok视频数据

    返回视频帧分析报告
    """
    start_time = time.time()

    try:
        logger.info(f"分析视频帧")

        video_frames = video_agent.analyze_video_frames(aweme_id, frame_interval)

        processing_time = time.time() - start_time

        return create_response(
            data=video_frames,
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
        logger.error(f"分析视频帧时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@router.post(
    "/fetch_invideo_text",
    summary="分析/提取 视频内文字",
    description="""
用途:
    * 提取TikTok视频内文字
参数:
    * aweme_id: TikTok视频ID
    * frame_interval: 分析帧之间的间隔（秒）
    * confidence_threshold: 文字识别置信度阈值
返回:
    * 视频内文字提取报告
""",
    response_model_exclude_none=True,
)
async def fetch_invideo_text(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        frame_interval: int = Query(90, description="分析帧之间的间隔（秒）"),
        confidence_threshold: float = Query(0.6, description="文字识别置信度阈值"),
        video_agent: VideoAgent = Depends(get_video_agent)
):
    """
    提取TikTok视频内文字

    - **video_data**: TikTok视频数据

    返回视频内文字提取报告
    """
    start_time = time.time()

    try:
        logger.info(f"提取视频内文字")

        invideo_text = video_agent.fetch_invideo_text(aweme_id, frame_interval, confidence_threshold)

        processing_time = time.time() - start_time

        return create_response(
            data=invideo_text,
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
        logger.error(f"提取视频内文字时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

