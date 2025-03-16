# -*- coding: utf-8 -*-
from imaplib import IMAP4

from fastapi import APIRouter, File, UploadFile, Query, Depends, Form, HTTPException, Request
from typing import List, Optional, Dict, Any, Union
from app.dependencies import verify_tikhub_api_key
from app.config import settings
from pydantic import BaseModel, Field
import uuid
from app.utils.job_manager import JobManager, JobStatus
import time
from agents.video_subtitles_agent import VideoSubtitlesAgent
from app.utils.logger import setup_logger
from app.api.models.responses import create_response

from app.core.exceptions import (
    ValidationError,
    ExternalAPIError,
    InternalServerError,
    NotFoundError
)


# 设置日志记录器
logger = setup_logger(__name__)

router = APIRouter(prefix="/video_subtitles")

job_manager = JobManager()

# 定义数据模型
class SubtitleRequest(BaseModel):
    file_path: Optional[str] = None
    aweme_id: Optional[str] = None
    source_language: Optional[str] = "auto"
    target_language: str
    subtitle_format: str = "srt"

class BatchSubtitleRequest(BaseModel):
    video_sources: List[Dict[str, Any]]
    source_language: Optional[str] = "auto"
    target_language: str
    subtitle_format: str = "srt"

class SubtitleExtractRequest(BaseModel):
    file_path: Optional[str] = None
    aweme_id: Optional[str] = None

class SubtitleRemovalRequest(BaseModel):
    file_path: Optional[str] = None
    aweme_id: Optional[str] = None

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

# 依赖项：获取audio_generator实例
async def video_subtitles_generator(tikhub_api_key: str = Depends(verify_tikhub_api_key)):
    """使用验证后的TikHub API Key创建AudioGeneratorAgent实例"""
    return VideoSubtitlesAgent(tikhub_api_key=tikhub_api_key, tikhub_base_url=settings.TIKHUB_BASE_URL)


@router.post("/subtitles_generate", response_model=JobResponse)
async def create_subtitles(
        request: Request,
        file: Optional[UploadFile] = File(None,description="本地路径"),
        aweme_id: Optional[str] = Form(None,description="TikTok视频ID"),
        source_language: Optional[str] = Form("auto"),
        target_language: str = Form(...),
        subtitle_format: str = Form("srt"),
        video_subtitles_agent: VideoSubtitlesAgent = Depends(video_subtitles_generator)
):
    """
    为视频生成字幕
    - 支持上传视频文件或提供TikTok视频ID
    - 支持多语言转换
    - 返回带字幕的视频链接和SRT文件
    """


    # 创建作业
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id)

    start_time = time.time()


    try:
        logger.info(f"获取视频 {file}{aweme_id} 的字幕")

        video_subtitles_data = await video_subtitles_agent.process_video(file,aweme_id, source_language, target_language, subtitle_format,job_id)

        processing_time = time.time() - start_time
        JobResponse_dict = {"job_id": job_id, "status": "queued", "message": "任务已加入队列"}
        video_subtitles_result = create_response(
            data=video_subtitles_data,
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )

        video_subtitles_result.update(JobResponse_dict)


        return video_subtitles_result

    except ValidationError as e:
        logger.error(f"验证错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"获取视频评论时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post("/subtitles_batch", response_model=JobResponse)
async def batch_generate_subtitles(
        request: BatchSubtitleRequest,
        video_subtitles_agent: VideoSubtitlesAgent = Depends(video_subtitles_generator)

):
    """
    批量为多个视频生成字幕
    - 支持多视频并行处理
    - 支持批量作业状态追踪
    """
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id)

    start_time = time.time()
    JobResponse_dict = {"job_id": job_id, "status": "queued", "message": "任务已加入队列"}

    try:
        logger.info(f"批量获取视频 {request.video_sources} 的字幕")

        video_batch_subtitles_data = await video_subtitles_agent.process_batch_videos(
            request.video_sources,
            request.source_language,
            request.target_language,
            request.subtitle_format,
            job_id)

        processing_time = time.time() - start_time

        video_batch_subtitles_result = create_response(
            data=video_batch_subtitles_data,
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )
        video_batch_subtitles_result.update(JobResponse_dict)

        return video_batch_subtitles_result

    except ValidationError as e:
        logger.error(f"验证错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"获取视频评论时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post("/subtitles_extract", response_model=JobResponse)
async def extract_subtitles_route(
        file: Optional[UploadFile] = File(None),
        aweme_id: Optional[str] = Form(None),
        video_subtitles_agent: VideoSubtitlesAgent = Depends(video_subtitles_generator)
):
    """
    从视频中提取字幕
    - 支持上传视频文件或提供TikTok视频ID
    - 返回提取到的字幕文本和SRT文件
    """
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id)
    JobResponse_dict = {"job_id": job_id, "status": "queued", "message": "任务已加入队列"}
    start_time = time.time()

    try:
        logger.info(f"批量获取视频 {file}{aweme_id} 的字幕")

        video_extract_subtitles_data = await video_subtitles_agent.extract_video_subtitles(
            file, aweme_id,job_id)

        processing_time = time.time() - start_time

        video_extract_subtitles_result = create_response(
            data=video_extract_subtitles_data,
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )
        video_extract_subtitles_result.update(JobResponse_dict)
        return video_extract_subtitles_result

    except ValidationError as e:
        logger.error(f"验证错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"获取视频评论时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post("/subtitles_remove", response_model=JobResponse)
async def remove_subtitles_route(
        file: Optional[UploadFile] = File(None),
        aweme_id: Optional[str] = Form(None),
        video_subtitles_agent: VideoSubtitlesAgent = Depends(video_subtitles_generator)
):
    """
    从视频中移除硬编码字幕
    - 支持上传视频文件或提供TikTok视频ID
    - 返回移除字幕后的视频链接
    """
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id)
    JobResponse_dict = {"job_id": job_id, "status": "queued", "message": "任务已加入队列"}
    start_time = time.time()

    try:
        logger.info(f"批量获取视频  {file}{aweme_id} 的字幕")

        video_remove_subtitles_data = await video_subtitles_agent.remove_video_subtitles(
            file, aweme_id,job_id)

        processing_time = time.time() - start_time

        video_remove_subtitles_result = create_response(
            data=video_remove_subtitles_data,
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )

        video_remove_subtitles_result.update(JobResponse_dict)
        return video_remove_subtitles_result

    except ValidationError as e:
        logger.error(f"验证错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"获取视频评论时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")




