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
    summary="【一键取数】快速获取/清洗 视频关键数据",
    description="""
用途:
  * 一键获取并清洗TikTok视频的核心数据（如点赞、评论、转发数，以及创作者信息、视频设置等）
  * 点赞、评论、转发等数据，以及相关音乐、创作者信息、视频描述、视频设置等

参数:
  * aweme_id: TikTok视频ID


（从此无需手动爬取，秒级呈现视频核心！）
""",
    response_model_exclude_none=True,
)
async def fetch_single_video_data(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        video_agent: VideoAgent = Depends(get_video_agent)
):
    """
    获取指定TikTok视频的数据

    - **aweme_id**: TikTok视频ID

    返回清理后`app`端的视频数据
    """
    start_time = time.time()

    try:
        logger.info(f"获取视频 {aweme_id} 的信息")

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
        logger.error(f"获取视频信息时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/analyze_video_info",
    summary="【数据报告】深度分析视频统计信息",
    description="""
用途:
  * 针对TikTok视频的各项指标进行深度解读，自动生成数据报告
  * 返回视频数据分析报告markdown string（含播放、互动、受众画像等要点）

参数:
  * aweme_id: TikTok视频ID

（高效洞察视频价值，一眼看穿数据背后的真相！）
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

    - **aweme_id**: TikTok视频ID

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
    summary="【音频转录】快速分析/提取 视频字幕内容",
    description="""
用途:
  * 获取TikTok视频的音频字幕或语言文本
  * Optional: 可以指定提取的音频的语言

参数:
  * aweme_id: TikTok视频ID
  

（精准提炼视频主旨，为视频内容理解与创意编排提供支持！）
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

    - **aweme_id**: TikTok视频ID

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
    summary="【画面识别】洞悉视频关键帧内容",
    description="""
用途:
  * 逐帧采样并识别TikTok视频中的关键视觉元素
  * 根据自定义的分析帧间隔，提取画面核心信息：场景、人物、动作等

参数:
  * aweme_id: TikTok视频ID
  * frame_interval: 分析帧之间的间隔（秒）

（图像识别省时省力，为视频精修与创意编排助力！）
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

    - **aweme_id**: TikTok视频ID

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
    summary="【文字扫描】自动提取 视频内文字内容",
    description="""
用途:
  * 高精度检测并识别视频画面中的文字
  * 针对于产品讲解或者信息类标识的视频，提取产品名字，价格等信息
  * 识别多种语言，多种场景类型，可后期用于配音字幕等
  * 返回视频内文字提取报告 （时间戳，帧数，文字内容）

参数:
  * aweme_id: TikTok视频ID
  * frame_interval: 分析帧之间的间隔（秒）
  * confidence_threshold: 文字识别置信度阈值

（让视频画面中的所有文字无所遁形，彻底捕捉宣传与信息点！）
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

    - **aweme_id**: TikTok视频ID
    - **frame_interval**: 分析帧之间的间隔（秒）
    - **confidence_threshold**: 文字识别置信度阈值

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
