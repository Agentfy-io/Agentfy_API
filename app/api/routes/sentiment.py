# -*- coding: utf-8 -*-
"""
@file: sentiment.py
@desc: FastAPI 客户端路由
@auth: Callmeiks
"""
import json
import random
import string

from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request, Body, BackgroundTasks
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

# 用于存储后台任务结果的字典
task_results = {}


# 依赖项：获取SentimentAgent实例
async def get_sentiment_agent(tikhub_api_key: str = Depends(verify_tikhub_api_key)):
    """使用验证后的TikHub API Key创建SentimentAgent实例"""
    return SentimentAgent(tikhub_api_key=tikhub_api_key)


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
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取指定TikTok视频的评论数据

    - **aweme_id**: TikTok视频ID

    返回任务ID和初始状态
    """
    # 生成任务ID
    task_id = f"comments_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "aweme_id": aweme_id,
    }

    async def process_video_comments():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在获取视频评论数据...请过10秒+后再查看"

            start_time = time.time()

            try:
                logger.info(f"获取视频 {aweme_id} 的评论")

                comments_data = await sentiment_agent.fetch_video_comments(aweme_id)

                processing_time = time.time() - start_time

                task_results[task_id]["status"] = "completed"
                task_results[task_id]["message"] = "成功获取视频评论"
                task_results[task_id]["data"] = comments_data
                task_results[task_id]["processing_time_ms"] = round(processing_time * 1000, 2)
                task_results[task_id]["timestamp"] = datetime.now().isoformat()

            except ValidationError as e:
                logger.error(f"验证错误: {e.detail}")
                task_results[task_id]["status"] = "failed"
                task_results[task_id]["message"] = f"验证错误: {e.detail}"
                task_results[task_id]["timestamp"] = datetime.now().isoformat()

            except ExternalAPIError as e:
                logger.error(f"外部API错误: {e.detail}")
                task_results[task_id]["status"] = "failed"
                task_results[task_id]["message"] = f"外部API错误: {e.detail}"
                task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"获取视频评论时发生未预期错误: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"内部服务器错误: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_video_comments)

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
    "/fetch_sentiment_analysis",
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
async def fetch_sentiment_analysis(
        request: Request,
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(50, description="每次处理的评论数量"),
        concurrency: int = Query(5, description="并发请求数"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取评论的情感分析结果

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每次处理的评论数量
    - **concurrency**: 并发请求数

    返回任务ID和初始状态
    """
    # 生成任务ID
    task_id = f"sentiment_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "aweme_id": aweme_id,
        "batch_size": batch_size,
        "concurrency": concurrency
    }

    async def process_sentiment_analysis():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在获取评论情感分析结果...请过10秒+后再查看"

            # 使用AsyncGenerator获取进度和结果
            async for result in sentiment_agent.fetch_sentiment_analysis(aweme_id, batch_size, concurrency):
                # 更新任务状态
                task_results[task_id]["message"] = result["message"]
                task_results[task_id]["llm_processing_cost"] = result["llm_processing_cost"]
                task_results[task_id]["total_collected_comments"] = result["total_collected_comments"]
                task_results[task_id]["total_analyzed_comments"] = result["total_analyzed_comments"]
                task_results[task_id]["timestamp"] = result.get("timestamp", datetime.now().isoformat())
                task_results[task_id]["processing_time_ms"] = result.get("processing_time_ms", 0)

                # 如果有报告URL，添加到结果中
                if "report_url" in result:
                    task_results[task_id]["report_url"] = result["report_url"]

                # 如果有分析摘要，添加到结果中
                if "analysis_summary" in result:
                    task_results[task_id]["analysis_summary"] = result["analysis_summary"]

                # 处理错误
                if "error" in result:
                    task_results[task_id]["status"] = "failed"
                    task_results[task_id]["error"] = result["error"]
                    break

                # 处理完成状态
                if result["is_complete"]:
                    task_results[task_id]["status"] = "completed"
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"

        except ValidationError as e:
            logger.error(f"验证错误: {e.detail}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"验证错误: {e.detail}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except ExternalAPIError as e:
            logger.error(f"外部API错误: {e.detail}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"外部API错误: {e.detail}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"获取评论情感分析结果时发生未预期错误: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"内部服务器错误: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_sentiment_analysis)

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
    "/fetch_relationship_analysis",
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
async def fetch_relationship_analysis(
        request: Request,
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每次处理的评论数量"),
        concurrency: int = Query(5, description="并发请求数"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取评论关系分析结果

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每次处理的评论数量
    - **concurrency**: 并发请求数

    返回任务ID和初始状态
    """
    # 生成任务ID
    task_id = f"relationship_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "aweme_id": aweme_id,
        "batch_size": batch_size,
        "concurrency": concurrency
    }

    async def process_relationship_analysis():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在获取评论关系分析结果...请过10秒+后再查看"

            # 使用AsyncGenerator获取进度和结果
            async for result in sentiment_agent.fetch_relationship_analysis(aweme_id, batch_size, concurrency):
                # 更新任务状态
                task_results[task_id]["message"] = result["message"]
                task_results[task_id]["llm_processing_cost"] = result["llm_processing_cost"]
                task_results[task_id]["total_collected_comments"] = result["total_collected_comments"]
                task_results[task_id]["total_analyzed_comments"] = result["total_analyzed_comments"]
                task_results[task_id]["timestamp"] = result.get("timestamp", datetime.now().isoformat())
                task_results[task_id]["processing_time_ms"] = result.get("processing_time_ms", 0)

                # 如果有报告URL，添加到结果中
                if "report_url" in result:
                    task_results[task_id]["report_url"] = result["report_url"]

                # 如果有分析摘要，添加到结果中
                if "analysis_summary" in result:
                    task_results[task_id]["analysis_summary"] = result["analysis_summary"]

                # 处理错误
                if "error" in result:
                    task_results[task_id]["status"] = "failed"
                    task_results[task_id]["error"] = result["error"]
                    break

                # 处理完成状态
                if result["is_complete"]:
                    task_results[task_id]["status"] = "completed"
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"

        except ValidationError as e:
            logger.error(f"验证错误: {e.detail}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"验证错误: {e.detail}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except ExternalAPIError as e:
            logger.error(f"外部API错误: {e.detail}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"外部API错误: {e.detail}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"获取评论关系分析结果时发生未预期错误: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"内部服务器错误: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_relationship_analysis)

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
    "/fetch_toxicity_analysis",
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
async def fetch_toxicity_analysis(
        request: Request,
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每次处理的评论数量"),
        concurrency: int = Query(5, description="并发请求数"),
        sentiment_agent: SentimentAgent = Depends(get_sentiment_agent)
):
    """
    获取评论区黑评/差评分析结果

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每次处理的评论数量
    - **concurrency**: 并发请求数

    返回任务ID和初始状态
    """
    # 生成任务ID
    task_id = f"toxicity_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "aweme_id": aweme_id,
        "batch_size": batch_size,
        "concurrency": concurrency
    }

    async def process_toxicity_analysis():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在获取评论区负面评论分析统计结果...请过10秒+后再查看"

            # 使用AsyncGenerator获取进度和结果
            async for result in sentiment_agent.fetch_toxicity_analysis(aweme_id, batch_size, concurrency):
                # 更新任务状态
                task_results[task_id]["message"] = result["message"]
                task_results[task_id]["llm_processing_cost"] = result["llm_processing_cost"]
                task_results[task_id]["total_collected_comments"] = result["total_collected_comments"]
                task_results[task_id]["total_analyzed_comments"] = result["total_analyzed_comments"]
                task_results[task_id]["timestamp"] = result.get("timestamp", datetime.now().isoformat())
                task_results[task_id]["processing_time_ms"] = result.get("processing_time_ms", 0)

                # 如果有报告URL，添加到结果中
                if "report_url" in result:
                    task_results[task_id]["report_url"] = result["report_url"]

                # 如果有分析摘要，添加到结果中
                if "analysis_summary" in result:
                    task_results[task_id]["analysis_summary"] = result["analysis_summary"]

                # 处理错误
                if "error" in result:
                    task_results[task_id]["status"] = "failed"
                    task_results[task_id]["error"] = result["error"]
                    break

                # 处理完成状态
                if result["is_complete"]:
                    task_results[task_id]["status"] = "completed"
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"

        except ValidationError as e:
            logger.error(f"验证错误: {e.detail}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"验证错误: {e.detail}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except ExternalAPIError as e:
            logger.error(f"外部API错误: {e.detail}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"外部API错误: {e.detail}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"获取评论区黑评/差评分析结果时发生未预期错误: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"内部服务器错误: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_toxicity_analysis)

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
        background_tasks: BackgroundTasks,
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

    返回任务ID和初始状态
    """
    # 生成任务ID
    task_id = f"negative_reviews_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "aweme_id": aweme_id,
        "batch_size": batch_size,
        "concurrency": concurrency
    }

    async def process_negative_shop_reviews():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在获取评论区商品差评的评论者信息...请过10秒+后再查看"

            # 使用AsyncGenerator获取进度和结果
            async for result in sentiment_agent.fetch_negative_shop_reviews(aweme_id, batch_size, concurrency):
                # 更新任务状态
                task_results[task_id]["message"] = result["message"]
                task_results[task_id]["llm_processing_cost"] = result["llm_processing_cost"]
                task_results[task_id]["total_collected_comments"] = result["total_collected_comments"]
                task_results[task_id]["total_analyzed_comments"] = result["total_analyzed_comments"]
                task_results[task_id]["timestamp"] = result.get("timestamp", datetime.now().isoformat())
                task_results[task_id]["processing_time_ms"] = result.get("processing_time_ms", 0)

                # 如果有负面商店评论，添加到结果中
                if "negative_shop_reviews" in result:
                    task_results[task_id]["negative_shop_reviews"] = result["negative_shop_reviews"]

                # 处理错误
                if "error" in result:
                    task_results[task_id]["status"] = "failed"
                    task_results[task_id]["error"] = result["error"]
                    break

                # 处理完成状态
                if result["is_complete"]:
                    task_results[task_id]["status"] = "completed"
                    if "meta" in result:
                        task_results[task_id]["meta"] = result["meta"]
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"

        except ValidationError as e:
            logger.error(f"验证错误: {e.detail}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"验证错误: {e.detail}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except ExternalAPIError as e:
            logger.error(f"外部API错误: {e.detail}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"外部API错误: {e.detail}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"获取评论区商品差评的评论者信息时发生未预期错误: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"内部服务器错误: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_negative_shop_reviews)

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
    "/fetch_hate_spam_speech",
    summary="【恶意言论狙击】获取评论区恶意言论用户信息",
    description="""
用途:
  * 追踪仇恨、攻击性言论,垃圾评论等恶意评论
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
async def fetch_hate_spam_speech(
        request: Request,
        background_tasks: BackgroundTasks,
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

    返回任务ID和初始状态
    """
    # 生成任务ID
    task_id = f"hate_speech_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "in_progress",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "aweme_id": aweme_id,
        "batch_size": batch_size,
        "concurrency": concurrency
    }

    async def process_hate_speech():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "正在获取评论区恶意评论者信息...请过10秒+后再查看"

            # 使用AsyncGenerator获取进度和结果
            async for result in sentiment_agent.fetch_hate_spam_speech(aweme_id, batch_size, concurrency):
                # 更新任务状态
                task_results[task_id]["message"] = result["message"]
                task_results[task_id]["llm_processing_cost"] = result["llm_processing_cost"]
                task_results[task_id]["total_collected_comments"] = result["total_collected_comments"]
                task_results[task_id]["total_analyzed_comments"] = result["total_analyzed_comments"]
                task_results[task_id]["timestamp"] = result.get("timestamp", datetime.now().isoformat())
                task_results[task_id]["processing_time_ms"] = result.get("processing_time_ms", 0)

                # 如果有恶意言论评论，添加到结果中
                if "hate_comments" in result:
                    task_results[task_id]["hate_comments"] = result["hate_comments"]

                if "spam_comments" in result:
                    task_results[task_id]["spam_comments"] = result["spam_comments"]

                # 处理错误
                if "error" in result:
                    task_results[task_id]["status"] = "failed"
                    task_results[task_id]["error"] = result["error"]
                    break

                # 处理完成状态
                if result["is_complete"]:
                    task_results[task_id]["status"] = "completed"
                    if "meta" in result:
                        task_results[task_id]["meta"] = result["meta"]
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"

        except ValidationError as e:
            logger.error(f"验证错误: {e.detail}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"验证错误: {e.detail}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except ExternalAPIError as e:
            logger.error(f"外部API错误: {e.detail}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"外部API错误: {e.detail}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"获取评论区恶意评论者信息时发生未预期错误: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"内部服务器错误: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_hate_speech)

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
