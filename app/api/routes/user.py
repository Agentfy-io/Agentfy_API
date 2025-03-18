# -*- coding: utf-8 -*-
"""
@file: user.py
@desc: FastAPI 客户端路由
@auth: Callmeiks
"""
import json
import random
import string

from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request, Body, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
from app.api.models.responses import create_response
from agents.user_agent import UserAgent
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
router = APIRouter(prefix="/user")

# 用于存储后台任务结果的字典
task_results = {}


# 依赖项：获取UserAgent实例
async def get_user_agent(tikhub_api_key: str = Depends(verify_tikhub_api_key)):
    """使用验证后的TikHub API Key创建userAgent实例"""
    return UserAgent(tikhub_api_key=tikhub_api_key)


@router.post(
    "/fetch_user_profile_analysis",
    summary="【一键直达】快速分析TikTok用户/达人基础信息",
    description="""
用途:
  * 后台创建指定TikTok用户/达人基础信息分析任务
  * 分析用户/达人的基础资料和数据指标
  * 生成分析报告并提供访问链接
  * 包括用户唯一ID、原始数据、报告URL、处理时间戳等信息

参数:
  * url: TikTok用户主页URL，格式为https://tiktok.com/@username

（精准KOL分析，助您高效把握达人特征与价值！）
""",
    response_model_exclude_none=True,
)
async def fetch_user_profile_analysis(
        request: Request,
        background_tasks: BackgroundTasks,
        url: str = Query(..., description="TikTok用户主页URL"),
        user_agent: UserAgent = Depends(get_user_agent)
):
    """
    分析TikTok用户/达人的基础信息

    返回任务ID和初始状态
    """

    # 生成任务ID
    task_id = f"profile_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "pending",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "user_profile_url": url,
        "profile_raw_data": {}
    }

    async def process_user_profile():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "processing"
            task_results[task_id]["message"] = "正在分析用户/达人基础信息...请过10秒+后再查看"

            async for result in user_agent.fetch_user_profile_analysis(url=url):
                task_results[task_id]["timestamp"] = result['timestamp']
                task_results[task_id]["user_profile_url"] = result['user_profile_url']
                task_results[task_id]["uniqueId"] = result['uniqueId']
                task_results[task_id]["analysis_report"] = result['analysis_report']
                task_results[task_id]["profile_raw_data"] = result['profile_raw_data']
                task_results[task_id]["message"] = result['message']
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    break
                elif result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break
                else:
                    task_results[task_id]["status"] = "pending"
        except Exception as e:
            logger.error(f"后台任务处理用户 '{url}' 分析时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_user_profile)

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "pending",
            "message": "任务已创建，正在启动",
            "timestamp": datetime.now().isoformat()
        },
        success=True
    )



@router.get(
    "/tasks/{task_id}",
    summary="【任务查询】获取后台任务状态与结果",
    description="""
用途:
  * 查询后台任务的状态和结果
  * 适用于长时间运行的大规模数据挖掘任务
  * 返回任务状态、进度信息和已获取的结果

参数:
  * task_id: 任务ID

（随时掌握任务进度，高效管理数据获取流程！）
""",
    response_model_exclude_none=True,
)
async def get_task_status(
        request: Request,
        task_id: str = Path(..., description="任务ID")
):
    """
    获取任务状态和结果

    返回任务的当前状态、进度和部分结果
    """
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 复制任务状态，避免返回过大的结果集
    task_info = dict(task_results[task_id])

    # 如果结果过多，只返回前100个
    if "results" in task_info and len(task_info["results"]) > 100:
        task_info["results"] = task_info["results"][:100]
        task_info["results_truncated"] = True
        task_info["total_results"] = len(task_results[task_id]["results"])

    return create_response(
        data=task_info,
        success=True
    )
