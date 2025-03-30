# -*- coding: utf-8 -*-
from imaplib import IMAP4

from fastapi import APIRouter, File, UploadFile, Query, Depends, Form, HTTPException, Request, BackgroundTasks
from typing import List, Optional, Dict, Any, Union

from multipart import file_path

from app.dependencies import verify_tikhub_api_key
from app.config import settings
from pydantic import BaseModel, Field
import uuid
import time
from agents.video_subtitles_agent import VideoSubtitlesAgent, JobManager
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
        background_tasks: BackgroundTasks,
        file: Optional[UploadFile] = File(None,description="本地路径"),
        aweme_id: Optional[str] = Query(None,description="TikTok视频ID"),
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
    # 参数验证
    if not file and not aweme_id:
        raise HTTPException(status_code=400, detail="必须提供视频文件或TikTok视频ID")

    file_path = None
    if file:
        file_path = await video_subtitles_agent.save_upload_file(file)

    # 创建作业
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id)

    # 异步处理视频
    background_tasks.add_task(
        video_subtitles_agent.process_video,
        file_path=file_path,
        aweme_id=aweme_id,
        source_language=source_language,
        target_language=target_language,
        subtitle_format=subtitle_format,
        job_id=job_id
    )

    return {"job_id": job_id, "status": "queued", "message": "任务已加入队列"}


@router.post("/subtitles_batch", response_model=JobResponse)
async def batch_generate_subtitles(
        background_tasks: BackgroundTasks,
        request: BatchSubtitleRequest,
        video_subtitles_agent: VideoSubtitlesAgent = Depends(video_subtitles_generator)

):
    """
    批量为多个视频生成字幕
    - 支持多视频并行处理
    - 支持批量作业状态追踪
    """
    # 参数验证

    if not request.video_sources:
        raise HTTPException(status_code=400, detail="必须提供至少一个视频源")

    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id)

    # 异步处理批量视频
    background_tasks.add_task(
        video_subtitles_agent.process_batch_videos,
        videos_data=request.video_sources,
        source_language=request.source_language,
        target_language=request.target_language,
        subtitle_format=request.subtitle_format,

        job_id=job_id
    )

    return {"job_id": job_id, "status": "queued", "message": "批量任务已加入队列"}



@router.post("/subtitles_extract", response_model=JobResponse)
async def extract_subtitles_route(
        background_tasks: BackgroundTasks,
        file: Optional[UploadFile] = File(None),
        aweme_id: Optional[str] = Query(None,description="TikTok视频ID"),
        video_subtitles_agent: VideoSubtitlesAgent = Depends(video_subtitles_generator)
):
    """
    从视频中提取字幕
    - 支持上传视频文件或提供TikTok视频ID
    - 返回提取到的字幕文本和SRT文件
    """
    # 参数验证
    if not file and not aweme_id:
        raise HTTPException(status_code=400, detail="必须提供视频文件或TikTok视频ID")

    file_path = None
    if file:
        file_path = await video_subtitles_agent.save_upload_file(file)

    # 创建作业
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id)

    # 异步提取字幕
    background_tasks.add_task(
        video_subtitles_agent.extract_video_subtitles,
        file_path=file_path,
        aweme_id=aweme_id,
        job_id=job_id
    )

    return {"job_id": job_id, "status": "queued", "message": "字幕提取任务已加入队列"}




@router.post("/subtitles_remove", response_model=JobResponse)
async def remove_subtitles_route(
        background_tasks: BackgroundTasks,
        file: Optional[UploadFile] = File(None),
        aweme_id: Optional[str] = Query(None,description="TikTok视频ID"),
        video_subtitles_agent: VideoSubtitlesAgent = Depends(video_subtitles_generator)
):
    """
    从视频中移除硬编码字幕
    - 支持上传视频文件或提供TikTok视频ID
    - 返回移除字幕后的视频链接
    """
    # 参数验证
    if not file and not aweme_id:
        raise HTTPException(status_code=400, detail="必须提供视频文件或TikTok视频ID")

    file_path = None
    if file:
        file_path = await video_subtitles_agent.save_upload_file(file)

    # 创建作业
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id)

    # 异步移除字幕
    background_tasks.add_task(
        video_subtitles_agent.remove_video_subtitles,
        file_path=file_path,
        aweme_id=aweme_id,
        job_id=job_id
    )

    return {"job_id": job_id, "status": "queued", "message": "字幕移除任务已加入队列"}





