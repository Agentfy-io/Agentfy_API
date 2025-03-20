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
    summary="快速分析TikTok用户/达人基础信息",
    description="""
用途:
  * 后台创建指定TikTok用户/达人基础信息分析任务
  * 获取指定用户/达人的基础信息，包括昵称、简介、粉丝数，点赞数，发帖数，公司信息等
  * 分析用户/达人的基础资料和数据指标
  * 生成分析报告并提供访问链接

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
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "user_profile_url": url,
    }

    async def process_user_profile():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在分析用户/达人基础信息...请过10秒+后再查看"

            async for result in user_agent.fetch_user_profile_analysis(url=url):
                task_results[task_id]["user_profile_url"] = result['user_profile_url']
                task_results[task_id]["message"] = result['message']
                task_results[task_id]["uniqueId"] = result['uniqueId']
                if 'report_url' in result:
                    task_results[task_id]["report_url"] = result['report_url']
                task_results[task_id]["llm_processing_cost"] = result['llm_processing_cost']
                task_results[task_id]["profile_data"] = result['profile_data']
                task_results[task_id]["timestamp"] = result['timestamp']
                task_results[task_id]["processing_time"] = result.get('processing_time', 0)

                if 'report_url' in result:
                    task_results[task_id]["report_url"] = result['report_url']

                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    break
                elif result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"
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
            "status": "in_progress",
            "message": "任务已创建，正在启动",
            "timestamp": datetime.now().isoformat()
        },
        success=True
    )


@router.post(
    "/fetch_user_posts_stats",
    summary="全面分析TikTok用户/达人发布作品统计",
    description="""
用途:
  * 后台创建指定TikTok用户/达人作品统计分析任务
  * 采集并分析用户/达人的所有发布作品数据，按照最新发布时间排序
  * 计算关键统计指标，包括平均互动数据、最高表现视频、发布频率等20多项指标
  * 生成详细的作品统计报告

参数:
  * url: TikTok用户主页URL，格式为https://tiktok.com/@username
  * max_post: （可选）最多分析的作品数量，默认分析全部作品

（深度内容分析，助您全面了解创作者表现！）
""",
    response_model_exclude_none=True,
)
async def fetch_user_posts_stats(
        request: Request,
        background_tasks: BackgroundTasks,
        url: str = Query(..., description="TikTok用户主页URL"),
        max_post: Optional[int] = Query(description="最多分析的作品数量，默认分析全部作品"),
        user_agent: UserAgent = Depends(get_user_agent)
):
    """
    分析TikTok用户/达人的发布作品统计

    返回任务ID和初始状态
    """

    # 生成任务ID
    task_id = f"posts_stats_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "url": url,
    }

    async def process_user_posts_stats():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在分析用户/达人发布作品统计...请过10秒+后再查看"

            # 直接调用提供的方法进行数据采集和分析
            async for result in user_agent.fetch_user_posts_stats(url, max_post):
                task_results[task_id]["message"] = result['message']
                task_results[task_id]["llm_processing_cost"] = result['llm_processing_cost']
                task_results[task_id]["total_collected_posts"] = result['total_collected_posts']
                task_results[task_id]["posts_stats"] = result['posts_stats']
                # task_results[task_id]["posts_data"] = result['posts_data']
                task_results[task_id]["timestamp"] = datetime.now().isoformat()
                task_results[task_id]["processing_time"] = result.get('processing_time', 0)
                if 'report_url' in result:
                    task_results[task_id]["report_url"] = result['report_url']

                # 处理进度更新
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    break
                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"
        except Exception as e:
            logger.error(f"后台任务处理用户 '{url}' 发布作品统计时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()


    # 添加后台任务
    background_tasks.add_task(process_user_posts_stats)

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "in_progress",
            "message": "任务已创建，正在启动",
            "timestamp": datetime.now().isoformat()
        },
        success=True
    )

@router.post(
    "/fetch_user_posts_trend",
    summary="全面分析TikTok用户/达人发布作品趋势",
    description="""
用途:
  * 后台创建指定TikTok用户/达人作品趋势分析任务
  * 采集并分析用户/达人的发布作品数据随时间的变化
  * 计算时间段内的发布频率和互动数据变化趋势 （每天发布数量、点赞量，评论量等4种趋势）
  * 支持自定义分析时间区间

参数:
  * url: TikTok用户主页URL，格式为https://tiktok.com/@username
  * time_interval: 分析的时间区间，如"90D"表示90天，默认为90天,

（深度趋势分析，助您掌握创作者成长轨迹！）
    """,
    response_model_exclude_none=True,
)
async def fetch_user_posts_trend(
        request: Request,
        background_tasks: BackgroundTasks,
        url: str = Query(..., description="TikTok用户主页URL"),
        time_interval: str = Query("90D", description="分析的时间区间，例如'90D'表示90天"),
        user_agent: UserAgent = Depends(get_user_agent)
    ):
    """
    分析TikTok用户/达人的发布作品趋势

    返回任务ID和初始状态
    """

    # 生成任务ID
    task_id = f"posts_trend_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "url": url,
        "time_interval": time_interval,
    }

    async def process_user_posts_trend():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在分析用户/达人发布作品趋势...请过10秒+后再查看"

            # 直接调用提供的方法进行数据采集和分析
            async for result in user_agent.fetch_user_posts_trend(url, time_interval):
                task_results[task_id]["message"] = result['message']
                task_results[task_id]["llm_processing_cost"] = result['llm_processing_cost']
                task_results[task_id]["total_collected_posts"] = result['total_collected_posts']
                # task_results[task_id]["posts_data"] = result['posts_data']
                task_results[task_id]["timestamp"] = datetime.now().isoformat()
                task_results[task_id]["processing_time"] = result.get('processing_time', 0)

                if 'report_url' in result:
                    task_results[task_id]["report_url"] = result['report_url']

                if 'trends_data' in result:
                    task_results[task_id]["trends_data"] = result['trends_data']

                # 处理进度更新
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    break
                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"
        except Exception as e:
            logger.error(f"后台任务处理用户 '{url}' 发布作品趋势时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_user_posts_trend)

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "in_progress",
            "message": "任务已创建，正在启动",
            "timestamp": datetime.now().isoformat()
        },
        success=True
    )


@router.post(
    "/fetch_post_duration_and_time_distribution",
    summary="分析TikTok用户/达人发布作品时长与时间分布",
    description="""
用途:
  * 后台创建指定TikTok用户/达人作品时长与发布时间分析任务
  * 采集并分析用户/达人作品的时长分布情况（0-15秒、15-30秒等区间）
  * 分析用户/达人发布作品的时间段分布（凌晨、上午、下午、晚上）
  * 生成详细的分布报告，了解创作者的内容模式

参数:
  * url: TikTok用户主页URL，格式为https://tiktok.com/@username

（内容规律分析，助您洞察创作者行为模式！）
""",
    response_model_exclude_none=True,
)
async def fetch_post_duration_and_time_distribution(
        request: Request,
        background_tasks: BackgroundTasks,
        url: str = Query(..., description="TikTok用户主页URL"),
        user_agent: UserAgent = Depends(get_user_agent)
):
    """
    分析TikTok用户/达人的发布作品时长与时间分布

    返回任务ID和初始状态
    """

    # 生成任务ID
    task_id = f"duration_time_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "url": url,
    }

    async def process_duration_and_time_distribution():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在分析用户/达人发布作品时长与时间分布...请过10秒+后再查看"

            # 直接调用提供的方法进行数据采集和分析
            async for result in user_agent.fetch_post_duration_and_time_distribution(url):
                task_results[task_id]["message"] = result['message']
                task_results[task_id]["llm_processing_cost"] = result['llm_processing_cost']
                task_results[task_id]["total_collected_posts"] = result['total_collected_posts']
                task_results[task_id]["duration_distribution"] = result['duration_distribution']
                task_results[task_id]["time_distribution"] = result.get('time_distribution', {})
                # task_results[task_id]["posts_data"] = result['posts_data']
                task_results[task_id]["timestamp"] = datetime.now().isoformat()
                task_results[task_id]["processing_time"] = result.get('processing_time', 0)

                if 'report_url' in result:
                    task_results[task_id]["report_url"] = result['report_url']

                # 处理进度更新
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    break
                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"
        except Exception as e:
            logger.error(f"后台任务处理用户 '{url}' 发布作品时长与时间分布时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_duration_and_time_distribution)

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "in_progress",
            "message": "任务已创建，正在启动",
            "timestamp": datetime.now().isoformat()
        },
        success=True
    )

@router.post(
    "/fetch_post_hashtags",
    summary="分析TikTok用户/达人使用的热门话题标签",
    description="""
用途:
  * 后台创建指定TikTok用户/达人话题标签分析任务
  * 采集并分析用户/达人所有作品中使用的话题标签
  * 统计话题标签使用频率，获取最常用的热门标签
  * 生成详细的标签使用报告, 提供产品分析和赛道分析

参数:
  * url: TikTok用户主页URL，格式为https://tiktok.com/@username
  * max_hashtags: 返回的热门话题标签数量，默认为所有标签

（标签策略分析，助您掌握创作者话题选择！）
""",
    response_model_exclude_none=True,
)
async def fetch_post_hashtags(
        request: Request,
        background_tasks: BackgroundTasks,
        url: str = Query(..., description="TikTok用户主页URL"),
        max_hashtags: int = Query(10, description="返回的热门话题标签数量"),
        user_agent: UserAgent = Depends(get_user_agent)
):
    """
    分析TikTok用户/达人使用的热门话题标签

    返回任务ID和初始状态
    """

    # 生成任务ID
    task_id = f"hashtags_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "url": url,
        "total_posts": 0,
        "top_hashtags": {}
    }

    async def process_post_hashtags():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在分析用户/达人使用的热门话题标签...请过10秒+后再查看"

            # 直接调用提供的方法进行数据采集和分析
            async for result in user_agent.fetch_post_hashtags(url, max_hashtags):
                task_results[task_id]["message"] = result['message']
                task_results[task_id]["llm_processing_cost"] = result['llm_processing_cost']
                if 'report_url' in result:
                    task_results[task_id]["report_url"] = result['report_url']
                task_results[task_id]["total_collected_posts"] = result['total_collected_posts']
                task_results[task_id]["top_hashtags"] = result.get('top_hashtags', {})
                # task_results[task_id]["posts_data"] = result['posts_data']
                task_results[task_id]["timestamp"] = datetime.now().isoformat()
                task_results[task_id]["processing_time"] = result.get('processing_time', 0)

                # 处理进度更新
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    break
                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"
        except Exception as e:
            logger.error(f"后台任务处理用户 '{url}' 热门话题标签分析时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_post_hashtags)

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "in_progress",
            "message": "任务已创建，正在启动",
            "timestamp": datetime.now().isoformat()
        },
        success=True
    )


@router.post(
    "/fetch_post_creator_analysis",
    summary="全面分析TikTok创作者视频内容特征",
    description="""
用途:
  * 后台创建指定TikTok用户/达人视频内容特征分析任务
  * 综合分析创作者的视频内容，包括多个维度的内容特征
  * 识别热门视频、广告/带货视频、AI/VR生成视频、风险视频等
  * 生成详细的内容特征报告

参数:
  * url: TikTok用户主页URL，格式为https://tiktok.com/@username

（深度内容特征分析，助您全面了解创作者内容策略！）
""",
    response_model_exclude_none=True,
)
async def fetch_post_creator_analysis(
        request: Request,
        background_tasks: BackgroundTasks,
        url: str = Query(..., description="TikTok用户主页URL"),
        user_agent: UserAgent = Depends(get_user_agent)
):
    """
    全面分析TikTok创作者视频内容特征

    返回任务ID和初始状态
    """

    # 生成任务ID
    task_id = f"creator_analysis_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "url": url,
    }

    async def process_post_creator_analysis():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在分析创作者视频内容特征...请过10秒+后再查看"

            # 直接调用提供的方法进行数据采集和分析
            async for result in user_agent.fetch_post_creator_analysis(url):
                task_results[task_id]["message"] = result['message']
                task_results[task_id]["llm_processing_cost"] = result['llm_processing_cost']
                if 'report_url' in result:
                    task_results[task_id]["report_url"] = result['report_url']
                task_results[task_id]["total_collected_posts"] = result['total_collected_posts']
                task_results[task_id]["analysis_results"] = result.get('analysis_results', {})
                # task_results[task_id]["posts_data"] = result['posts_data']
                task_results[task_id]["timestamp"] = datetime.now().isoformat()
                task_results[task_id]["processing_time"] = result.get('processing_time', 0)

                # 处理进度更新
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    break
                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"
        except Exception as e:
            logger.error(f"后台任务处理用户 '{url}' 创作者视频内容特征分析时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_post_creator_analysis)

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "in_progress",
            "message": "任务已创建，正在启动",
            "timestamp": datetime.now().isoformat()
        },
        success=True
    )

@router.post(
    "/fetch_user_fans",
    summary="获取TikTok用户/达人粉丝",
    description="""
用途:
  * 后台创建指定TikTok用户/达人粉丝采集任务
  * 采集用户/达人的粉丝列表和基本信息
  * 最大采集粉丝数量为10000，超过部分不采集

参数:
  * url: TikTok用户主页URL，格式为https://tiktok.com/@username
  * max_fans: 最多采集的粉丝数量，默认为10000

（粉丝洞察分析，助您了解目标受众特征！）
""",
    response_model_exclude_none=True,
)
async def fetch_user_fans(
        request: Request,
        background_tasks: BackgroundTasks,
        url: str = Query(..., description="TikTok用户主页URL"),
        max_fans: int = Query(10000, description="最多采集的粉丝数量"),
        user_agent: UserAgent = Depends(get_user_agent)
):
    """
    获取TikTok用户/达人的粉丝画像

    返回任务ID和初始状态
    """

    # 生成任务ID
    task_id = f"user_fans_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "url": url,
    }

    async def process_user_fans():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在获取用户粉丝数据...请过10秒+后再查看"

            # 直接调用提供的方法进行数据采集和分析
            async for result in user_agent.fetch_user_fans(url, max_fans):
                task_results[task_id]["message"] = result['message']
                task_results[task_id]["total_collected_fans"] = result['total_collected_fans']
                task_results[task_id]["fans"] = result['fans']
                task_results[task_id]["timestamp"] = datetime.now().isoformat()
                task_results[task_id]["processing_time"] = result.get('processing_time', 0)

                # 处理进度更新
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    break
                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"
        except Exception as e:
            logger.error(f"后台任务处理用户 '{url}' 粉丝采集时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_user_fans)

    # 返回任务信息
    return create_response(
        data={
            "task_id": task_id,
            "status": "in_progress",
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
