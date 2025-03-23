# -*- coding: utf-8 -*-
"""
@file: video.py
@desc: FastAPI 客户端路由
@auth: Callmeiks
"""
import random
import string
import time
from datetime import datetime
from typing import Callable

from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request, BackgroundTasks

from app.api.models.responses import create_response
from agents.video_agent import VideoAgent
from app.core.exceptions import (
    ValidationError,
    ExternalAPIError,
)
from app.utils.logger import setup_logger
from app.dependencies import verify_tikhub_api_key  # 从dependencies.py导入验证函数

# 设置日志记录器
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/video")

# 用于存储后台任务结果的字典
task_results = {}


# 依赖项：获取VideoAgent实例
async def get_video_agent(tikhub_api_key: str = Depends(verify_tikhub_api_key)):
    """使用验证后的TikHub API Key创建VideoAgent实例"""
    return VideoAgent(tikhub_api_key=tikhub_api_key)


# 生成唯一任务ID的辅助函数
def generate_task_id(prefix: str) -> str:
    """
    生成唯一的任务ID

    Args:
        prefix: 任务ID前缀

    Returns:
        生成的任务ID
    """
    random_str = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
    timestamp = int(time.time())
    return f"{prefix}_{random_str}_{timestamp}"


# 通用任务处理函数
async def process_video_task(task_id: str, analysis_method: Callable, **kwargs):
    """
    处理视频分析任务

    Args:
        task_id: 任务ID
        analysis_method: 异步生成器方法
        **kwargs: 传递给方法的参数
    """
    try:
        # 设置初始状态
        task_results[task_id]["status"] = "in_progress"
        task_results[task_id]["message"] = "任务已创建，正在启动..."

        # 使用异步生成器获取进度和结果
        async for result in analysis_method(**kwargs):
            # 复制所有字段到任务结果
            for key, value in result.items():
                if key != "is_complete":  # 不复制is_complete标志
                    task_results[task_id][key] = value

            # 根据结果更新任务状态
            if "error" in result:
                task_results[task_id]["status"] = "failed"
                break

            if result.get("is_complete", False):
                task_results[task_id]["status"] = "completed"
                break
            else:
                task_results[task_id]["status"] = "in_progress"

    except ValidationError as e:
        logger.error(f"验证错误: {str(e)}")
        task_results[task_id]["status"] = "failed"
        task_results[task_id]["message"] = f"验证错误: {str(e)}"
        task_results[task_id]["error"] = str(e)

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {str(e)}")
        task_results[task_id]["status"] = "failed"
        task_results[task_id]["message"] = f"外部API错误: {str(e)}"
        task_results[task_id]["error"] = str(e)

    except Exception as e:
        logger.error(f"任务处理过程中发生未预期错误: {str(e)}")
        task_results[task_id]["status"] = "failed"
        task_results[task_id]["message"] = f"内部服务器错误: {str(e)}"
        task_results[task_id]["error"] = str(e)


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
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        video_agent: VideoAgent = Depends(get_video_agent)
):
    """
    获取指定TikTok视频的数据
    """
    # 生成任务ID
    task_id = generate_task_id("video_data")

    # 初始化任务状态
    task_results[task_id] = {
        "status": "created",
        "message": "任务已创建，等待启动",
        "aweme_id": aweme_id
    }

    # 添加后台任务
    background_tasks.add_task(
        process_video_task,
        task_id=task_id,
        analysis_method=video_agent.fetch_video_data,
        aweme_id=aweme_id
    )

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "created",
            "message": "任务已创建，正在启动",
        },
        success=True
    )


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
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        video_agent: VideoAgent = Depends(get_video_agent)
):
    """
    分析TikTok视频数据
    """
    # 生成任务ID
    task_id = generate_task_id("video_info")

    # 初始化任务状态
    task_results[task_id] = {
        "status": "created",
        "message": "任务已创建，等待启动",
        "aweme_id": aweme_id,
        "llm_processing_cost": 0
    }

    # 添加后台任务
    background_tasks.add_task(
        process_video_task,
        task_id=task_id,
        analysis_method=video_agent.analyze_video_info,
        aweme_id=aweme_id
    )

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "created",
            "message": "任务已创建，正在启动",
        },
        success=True
    )


@router.post(
    "/fetch_video_transcript",
    summary="【音频转录】快速分析/提取 视频音频内容",
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
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        video_agent: VideoAgent = Depends(get_video_agent)
):
    """
    分析TikTok视频字幕
    """
    # 生成任务ID
    task_id = generate_task_id("transcript")

    # 初始化任务状态
    task_results[task_id] = {
        "status": "created",
        "message": "任务已创建，等待启动",
        "aweme_id": aweme_id
    }

    # 添加后台任务
    background_tasks.add_task(
        process_video_task,
        task_id=task_id,
        analysis_method=video_agent.fetch_video_transcript,
        aweme_id=aweme_id
    )

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "created",
            "message": "任务已创建，正在启动",
        },
        success=True
    )


@router.post(
    "/analyze_video_frames",
    summary="【画面识别】洞悉视频关键帧内容",
    description="""
用途:
  * 逐帧采样并识别TikTok视频中的关键视觉元素
  * 根据自定义的分析帧间隔，提取画面核心信息：场景、人物、动作等

参数:
  * aweme_id: TikTok视频ID
  * time_interval: 分析帧之间的间隔（秒）

（图像识别省时省力，为视频精修与创意编排助力！）
""",
    response_model_exclude_none=True,
)
async def analyze_video_frames(
        request: Request,
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        time_interval: float = Query(2.0, description="分析帧之间的间隔（秒）"),
        video_agent: VideoAgent = Depends(get_video_agent)
):
    """
    分析TikTok视频帧
    """
    # 生成任务ID
    task_id = generate_task_id("video_frames")

    # 初始化任务状态
    task_results[task_id] = {
        "status": "created",
        "message": "任务已创建，等待启动",
        "aweme_id": aweme_id,
        "time_interval": time_interval
    }

    # 添加后台任务
    background_tasks.add_task(
        process_video_task,
        task_id=task_id,
        analysis_method=video_agent.analyze_video_frames,
        aweme_id=aweme_id,
        time_interval=time_interval
    )

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "created",
            "message": "任务已创建，正在启动",
        },
        success=True
    )


@router.post(
    "/fetch_invideo_text",
    summary="【文字扫描】自动提取 视频内文字内容",
    description="""
用途:
  * 高精度检测并识别视频画面中的文字
  * 针对于产品讲解或者信息类标识的视频，提取产品名字，价格等信息
  * 识别多种语言，多种场景类型，可后期用于配音字幕等
  * 返回视频内文字提取报告 （时间戳，帧数，文字内容）
  * （如视频中没有内置文字，该接口会返回空值或者错误信息！）

参数:
  * aweme_id: TikTok视频ID
  * time_interval: 分析帧之间的间隔（秒）
  * confidence_threshold: 文字识别置信度阈值

（让视频画面中的所有文字无所遁形，彻底捕捉宣传与信息点！）
""",
    response_model_exclude_none=True,
)
async def fetch_invideo_text(
        request: Request,
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        time_interval: int = Query(3, description="分析帧之间的间隔（秒）"),
        confidence_threshold: float = Query(0.5, description="文字识别置信度阈值"),
        video_agent: VideoAgent = Depends(get_video_agent)
):
    """
    提取TikTok视频内文字
    """
    # 生成任务ID
    task_id = generate_task_id("invideo_text")

    # 初始化任务状态
    task_results[task_id] = {
        "status": "created",
        "message": "任务已创建，等待启动",
        "aweme_id": aweme_id,
        "time_interval": time_interval,
        "confidence_threshold": confidence_threshold
    }

    # 添加后台任务
    background_tasks.add_task(
        process_video_task,
        task_id=task_id,
        analysis_method=video_agent.fetch_invideo_text,
        aweme_id=aweme_id,
        time_interval=time_interval,
        confidence_threshold=confidence_threshold
    )

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "created",
            "message": "任务已创建，正在启动",
        },
        success=True
    )


@router.get(
    "/tasks/{task_id}",
    summary="【任务查询】获取任务状态与结果",
    description="""
用途:
  * 查询视频分析任务的状态和结果
  * 适用于长时间运行的视频分析任务
  * 返回任务状态、进度信息和分析结果

参数:
  * task_id: 任务ID

（随时掌握任务进度，高效管理视频分析流程！）
""",
    response_model_exclude_none=True,
)
async def get_task_status(
        request: Request,
        task_id: str = Path(..., description="任务ID")
):
    """
    获取任务状态和结果
    """
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="任务不存在")

    return create_response(
        data=task_results[task_id],
        success=True
    )