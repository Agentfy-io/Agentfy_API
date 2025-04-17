# -*- coding: utf-8 -*-
"""
@file: customer.py
@desc: FastAPI 客户端路由
@auth: Callmeiks
"""
import json
import random
import string

from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request, Body, BackgroundTasks, Header
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
from app.api.models.responses import create_response
from agents.customer_agent import CustomerAgent
from app.core.exceptions import (
    ValidationError,
)
from app.utils.logger import setup_logger
from app.dependencies import verify_tikhub_api_key, verify_openai_api_key
from app.config import settings

# 设置日志记录器
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/customers")

# 用于存储后台任务结果的字典
task_results = {}


# 依赖项：获取CustomerAgent实例
async def get_customer_agent(
    tikhub_api_key: str = Depends(verify_tikhub_api_key),
    openai_api_key: str = Depends(verify_openai_api_key)
):
    """使用验证后的TikHub和OpenAI API Key创建CustomerAgent实例"""
    return CustomerAgent(
        tikhub_api_key=tikhub_api_key,
        openai_api_key=openai_api_key.replace("Bearer ", "").strip()  # 防止有人也加了 Bearer
    )



@router.post(
    "/fetch_video_comments",
    summary="【一键直达】快速获取指定视频评论数据",
    description="""
用途:
  * 后台创建指定Tk视频评论数据获取任务
  * 获取TikTok视频评论数据，返回清洗后的评论列表
  * 根据自定义过滤器过滤Instagram或Twitter为空的用户，或按地区过滤用户
  * 包括评论ID、评论内容、点赞数、回复数、评论者用户名、评论者安全用户ID(SecUid)、评论语言、评论者国家、Instagram ID、Twitter ID、创建时间

参数:
  * aweme_id: TikTok视频ID
  * ins_filter: 是否过滤Instagram为空的用户，默认False
  * twitter_filter: 是否过滤Twitter为空的用户，默认False
  * region_filter: 按地区过滤用户，默认不过滤

（超高效舆情分析，助您精准捕捉热点！）
""",
    response_model_exclude_none=True,
    deprecated=True,
)
async def fetch_video_comments(
        request: Request,
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        ins_filter: Optional[bool] = Query(False, description="是否过滤Instagram为空的用户"),
        twitter_filter: Optional[bool] = Query(False, description="是否过滤Twitter为空的用户"),
        region_filter: Optional[str] = Query(None, description="按地区过滤用户"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    获取指定TikTok视频的评论数据

    返回任务ID和初始状态
    """

    # 生成任务ID
    task_id = f"comment_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "pending",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "aweme_id": aweme_id,
    }

    async def process_video_comments():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "processing"
            task_results[task_id]["message"] = "正在获取视频评论数据...请过10秒+后再查看"

            async for result in customer_agent.fetch_video_comments(
                    aweme_id=aweme_id,
                    ins_filter=ins_filter,
                    twitter_filter=twitter_filter,
                    region_filter=region_filter
            ):
                task_results[task_id]["aweme_id"] = result["aweme_id"]
                task_results[task_id]["message"] = result['message']
                task_results[task_id]["total_collected_comments"] = result["total_collected_comments"]
                task_results[task_id]["timestamp"] = datetime.now().isoformat()
                task_results[task_id]["processing_time_ms"] = result['processing_time_ms']

                # 检查是否出错
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    task_results[task_id]["comments"] = result['comments']
                    return

                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    task_results[task_id]["comments"] = result['comments']
                    break
                else:
                    task_results[task_id]["status"] = "in_progress"
                    task_results[task_id]["current_batch_count"] = result['current_batch_count']
                    task_results[task_id]["current_batch_comments"] = result['current_batch_comments']

        except Exception as e:
            logger.error(f"后台任务处理视频 '{aweme_id}' 潜在客户时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_video_comments)

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


@router.post(
    "/stream_potential_customers",
    summary="【实时挖掘】流式获取视频潜在客户",
    description="""
用途:
  * 后台创建视频潜在客户挖掘任务
  * 适用于大型数据挖掘需求，无需等待API响应
  * 返回任务ID，可通过任务ID查询进度和结果
  * 返回潜在客户信息流（用户名、ID、评论内容、国家、社交媒体ID、参与度分数）

参数:
  * aweme_id: 视频ID
  * customer_count: 最大返回客户数量，默认100
  * min_score: 最小参与度分数，范围0-100，默认50
  * max_score: 最大参与度分数，范围1-100，默认100
  * ins_filter: 是否过滤Instagram为空的用户，默认False
  * twitter_filter: 是否过滤Twitter为空的用户，默认False
  * region_filter: 按地区过滤用户，默认不过滤

（后台处理视频评论，挖掘高价值潜在客户，提升转化率！）
""",
    response_model_exclude_none=True,
)
async def stream_potential_customers(
        request: Request,
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="视频ID"),
        customer_count: int = Query(100, description="最大返回客户数量"),
        min_score: float = Query(50.0, description="最小参与度分数，范围0-100"),
        max_score: float = Query(100.0, description="最大参与度分数，范围1-100"),
        ins_filter: bool = Query(False, description="是否过滤Instagram为空的用户"),
        twitter_filter: bool = Query(False, description="是否过滤Twitter为空的用户"),
        region_filter: Optional[str] = Query(None, description="按地区过滤用户"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    创建后台任务获取视频潜在客户

    返回任务ID和初始状态
    """
    # 生成任务ID
    task_id = f"customer_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "pending",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "aweme_id": aweme_id,
        "potential_customers": []
    }

    logger.info(f"API key: {request.headers.get('Authorization')}")

    # 定义后台任务
    async def process_video_customers():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "processing"
            task_results[task_id]["message"] = "正在获取视频评论数据...请过10秒+后再查看"

            # 使用流式API
            async for result in customer_agent.stream_potential_customers(
                    aweme_id=aweme_id,
                    customer_count=customer_count,
                    min_score=min_score,
                    max_score=max_score,
                    ins_filter=ins_filter,
                    twitter_filter=twitter_filter,
                    region_filter=region_filter
            ):
                task_results[task_id]["aweme_id"] = result["aweme_id"]
                task_results[task_id]["message"] = result['message']
                task_results[task_id]["llm_processing_cost"] = result['llm_processing_cost']
                task_results[task_id]["customer_count"] = result["customer_count"]
                task_results[task_id]["potential_customers"] = result["potential_customers"]
                task_results[task_id]["processing_time_ms"] = result['processing_time_ms']
                task_results[task_id]["timestamp"] = result["timestamp"]
                # 检查是否出错
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    return
                # 处理批量结果并检查是否完成
                if not result['is_complete']:
                    task_results[task_id]["status"] = "in_progress"

                # 检查任务是否完成
                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break

        except Exception as e:
            logger.error(f"后台任务处理视频 '{aweme_id}' 潜在客户时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_video_customers)

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

@router.post(
    "/stream_keyword_potential_customers",
    summary="【关键词流挖】实时获取自定义赛道潜在客户",
    description="""
用途:
  * 后台创建关键词/指定赛道潜在客户挖掘任务
  * 适用于大型数据挖掘需求，无需等待API响应
  * 支持高并发处理多个视频
  * 流式分析潜在客户信息（关键词、客户列表、视频来源、统计数据）
  * 返回任务ID，可通过任务ID查询进度和结果

参数:
  * keyword: 搜索关键词
  * customer_count: 最大返回客户数量，默认100
  * video_concurrency: 视频处理并发数，默认5
  * min_score: 最小购买意向分数，范围0-100，默认50
  * max_score: 最大购买意向分数，范围1-100，默认100
  * ins_filter: 是否过滤Instagram为空的用户，默认False
  * twitter_filter: 是否过滤Twitter为空的用户，默认False
  * region_filter: 按地区过滤用户，默认不过滤

（后台悄悄工作，挖掘潜在客户，提升营销效率！）
""",
    response_model_exclude_none=True,
)
async def stream_keyword_potential_customers(
        request: Request,
        background_tasks: BackgroundTasks,
        keyword: str = Query(..., description="搜索关键词"),
        customer_count: int = Query(100, description="最大返回客户数量"),
        min_score: float = Query(50.0, description="最小购买意向分数，范围0-100"),
        max_score: float = Query(100.0, description="最大购买意向分数，范围1-100"),
        ins_filter: bool = Query(False, description="是否过滤Instagram为空的用户"),
        twitter_filter: bool = Query(False, description="是否过滤Twitter为空的用户"),
        region_filter: Optional[str] = Query(None, description="按地区过滤用户"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    创建后台任务获取关键词潜在客户

    返回任务ID和初始状态
    """
    # 生成任务ID
    task_id = f"customer_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "pending",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "keyword": keyword,
    }

    # 定义后台任务
    async def process_keyword_customers():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "processing"
            task_results[task_id]["message"] = "正在获取关键词视频列表...请过10秒+后再查看"

            # 使用流式API
            async for result in customer_agent.stream_keyword_potential_customers(
                    keyword=keyword,
                    customer_count=customer_count,
                    min_score=min_score,
                    max_score=max_score,
                    ins_filter=ins_filter,
                    twitter_filter=twitter_filter,
                    region_filter=region_filter
            ):
                task_results[task_id]["keyword"] = result["keyword"]
                task_results[task_id]["message"] = result['message']
                task_results[task_id]["llm_processing_cost"] = result['llm_processing_cost']
                task_results[task_id]["total_collected_customers"] = result['customer_count']
                task_results[task_id]["potential_customers"] = result['potential_customers']
                task_results[task_id]["processing_time_ms"] = result['processing_time_ms']
                task_results[task_id]["timestamp"] = result['timestamp']
                # 检查是否出错
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    return

                # 处理最终结果
                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break

                # 处理批量结果
                if not result['is_complete']:
                    task_results[task_id]["status"] = "in_progress"


        except Exception as e:
            logger.error(f"后台任务处理关键词 '{keyword}' 潜在客户时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["end_time"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_keyword_customers)

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


@router.post(
    "/fetch_purchase_intent_analysis",
    summary="【智能剖析】指定视频观众购买意图与统计",
    description="""
用途:
  * 全面洞察视频评论中的购买意向，挖掘潜在商机
  * 返回购买意图统计数据 (舆情分布图，兴趣等级，购买意向统计）
  * 返回购买意图分析报告链接（report_url)

参数:
  * aweme_id: TikTok视频ID
  * batch_size: 每批处理评论数量，默认30
  * concurrency: ai处理并发数，默认5，最大10

（助力精确营销，抢占商机先机！）
""",
    response_model_exclude_none=True,
)
async def fetch_purchase_intent_analysis(
        request: Request,
        background_tasks: BackgroundTasks,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每批处理的评论数量"),
        concurrency: int = Query(5, description="ai处理并发数"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    创建后台任务分析指定TikTok视频评论中的购买意图

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每批处理的评论数量，默认为30
    - **concurrency**: ai处理并发数，默认为5, 最大为10

    返回购买意图分析结果
    """
    # 生成任务ID
    task_id = f"purchase_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "pending",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "aweme_id": aweme_id,
    }

    # 定义后台任务
    async def process_purchase_intent():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "processing"
            task_results[task_id]["message"] = "正在获取视频评论数据...请过10秒+后再查看"

            # 获取购买意图统计数据
            async for result in customer_agent.fetch_purchase_intent_analysis(
                aweme_id=aweme_id,
                batch_size=batch_size,
                concurrency=concurrency
            ):
                task_results[task_id]["aweme_id"] = result["aweme_id"]
                task_results[task_id]["message"] = result["message"]
                task_results[task_id]["llm_processing_cost"] = result['llm_processing_cost']
                task_results[task_id]["total_collected_comments"] = result["total_collected_comments"]
                task_results[task_id]["total_analyzed_comments"] = result["total_analyzed_comments"]
                task_results[task_id]["analysis_summary"] = result["analysis_summary"],
                task_results[task_id]["timestamp"] = result["timestamp"]
                task_results[task_id]["processing_time_ms"] = result['processing_time_ms']
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    return
                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    task_results[task_id]["report_url"] = result["report_url"]
                    break
                elif not result['is_complete']:
                    task_results[task_id]["status"] = "in_progress"

            # 更新任务状态
            task_results[task_id]["status"] = "completed"
            task_results[task_id]["message"] = "任务完成，已获取购买意图统计数据"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"后台任务处理视频 '{aweme_id}' 购买意图时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_purchase_intent)

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


@router.post(
    "/generate_single_reply",
    summary="【AI智答】生成单条客户消息",
    description="""
用途:
  * 后台创建生成单个客户消息回复任务
  * 根据自定义店铺信息为单个客户消息智能生成个性化回复
  * 可适用于私信回复、客服回复，评论区回复等场景
  * 内置强大创意与语言处理能力，支持多语言回复，多种场景类型。

参数:
  * shop_info: 自定义店铺信息
  * customer_id: 客户uniqueID
  * customer_message: 客户消息

（巧妙应对各种询问，让服务升级到"贴心+1"！）
""",
    response_model_exclude_none=True,
)
async def generate_single_reply(
        request: Request,
        background_tasks: BackgroundTasks,
        shop_info: str = Query(..., description="店铺信息"),
        customer_id: str = Query(..., description="客户uniqueID"),
        customer_message: str = Query(..., description="客户消息"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    生成单条客户回复消息

    - **shop_info**: 店铺信息
    - **customer_id**: 客户uniqueID
    - **customer_message**: 客户消息

    返回生成的客户回复消息
    """
    # 生成任务ID
    task_id = f"reply_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "pending",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "customer_id": customer_id,
        "results": []
    }

    # 定义后台任务
    async def process_single_reply():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "processing"
            task_results[task_id]["message"] = "正在生成客户回复消息...请过10秒+后再查看"

            # 生成客户回复消息
            async for result in customer_agent.generate_single_reply_message(
                shop_info=shop_info,
                customer_id=customer_id,
                customer_message=customer_message
            ):
                task_results[task_id]["customer_id"] = result["customer_id"]
                task_results[task_id]["llm_processing_cost"] = result['llm_processing_cost']
                task_results[task_id]["message"] = result["message"]
                task_results[task_id]["reply_message"] = result["reply_message"]
                task_results[task_id]["timestamp"] = datetime.now().isoformat()
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    return
                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break
                elif not result['is_complete']:
                    task_results[task_id]["status"] = "in_progress"

            # 更新任务状态
            task_results[task_id]["status"] = "completed"
            task_results[task_id]["message"] = "任务完成，已生成客户回复消息"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"后台任务生成客户 '{customer_id}' 回复消息时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_single_reply)

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

@router.post(
    "/generate_batch_replies",
    summary="【批量智答】一键生成多条客户回复",
    description="""
用途:
  * 后台创建批量生成客户回复任务
  * 根据自定义店铺信息为多个客户消息批量生成个性化回复
  * 可适用于私信回复、客服回复，评论区回复等场景
  * 内置强大创意与语言处理能力，支持多语言回复，多种场景类型。
  * 返回生成的回复消息列表

参数:
  * shop_info: 店铺信息
  * customer_messages: 客户消息列表，每个包含commenter_uniqueId,text
  * batch_size: 每批处理消息数量，默认5

（让沟通效率倍增，从此告别重复回复！）
""",
    response_model_exclude_none=True,
)
async def generate_batch_replies(
        request: Request,
        background_tasks: BackgroundTasks,
        shop_info: str = Query(..., description="店铺信息"),
        batch_size: int = Query(5, description="每批处理消息数量"),
        customer_messages: Dict[str, Any] = Body(...,
                                                 description="客户消息列表，每个包含commenter_uniqueId, comment_id, text",
                                                 examples=[{
                                                     "jessica1h": "请问这款气垫粉底适合干皮吗？我皮肤比较干，担心会起皮。",
                                                     # 中文
                                                     "adam_123": "Do you ship internationally? I'd like to order some items to Canada.",
                                                     # 英文
                                                     "yuki_kawaii": "この美容マスクは本当に素晴らしいです！肌がとても潤いました。また購入します！",
                                                     # 日语
                                                     "k_beauty_fan": "이 제품에 알코올이 포함되어 있나요? 제가 알코올에 민감해서요.",  # 韩语
                                                 }]
                                                 ),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    批量生成客户回复消息

    - **shop_info**: 店铺信息
    - **customer_messages**: 客户消息列表，每个包含commenter_uniqueId, comment_id, text
    - **batch_size**: 每批处理消息数量，默认为5

    返回生成的客户回复消息列表
    """
    # 生成任务ID
    task_id = f"batch_reply_{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))}_{int(time.time())}"

    # 初始化任务状态
    task_results[task_id] = {
        "status": "pending",
        "message": "任务已创建，正在启动",
        "timestamp": datetime.now().isoformat(),
        "results": []
    }

    # 定义后台任务
    async def process_batch_replies():
        try:
            # 更新任务状态
            task_results[task_id]["status"] = "processing"
            task_results[task_id]["message"] = "正在生成客户回复消息...请过10秒+后再查看"

            # 生成客户回复消息
            async for result in customer_agent.generate_batch_reply_messages(
                shop_info=shop_info,
                customer_messages=customer_messages,
                batch_size=batch_size
            ):
                task_results[task_id]["message"] = result["message"]
                task_results[task_id]["llm_processing_cost"] = result['llm_processing_cost']
                task_results[task_id]["total_replies_count"] = result["total_replies_count"]
                task_results[task_id]["replies"] = result["replies"]
                task_results[task_id]["timestamp"] = datetime.now().isoformat()
                if 'error' in result:
                    task_results[task_id]["status"] = "failed"
                    return
                if result['is_complete']:
                    task_results[task_id]["status"] = "completed"
                    break
                elif not result['is_complete']:
                    task_results[task_id]["status"] = "in_progress"

            # 更新任务状态
            task_results[task_id]["status"] = "completed"
            task_results[task_id]["message"] = "任务完成，已生成客户回复消息"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"后台任务生成客户回复消息时出错: {str(e)}")
            task_results[task_id]["status"] = "failed"
            task_results[task_id]["message"] = f"任务处理出错: {str(e)}"
            task_results[task_id]["timestamp"] = datetime.now().isoformat()

    # 添加后台任务
    background_tasks.add_task(process_batch_replies)

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






@router.post(
    "/generate_single_reply",
    summary="【AI智答】生成单条客户消息",
    description="""
用途:
  * 根据自定义店铺信息为单个客户消息智能生成个性化回复
  * 可适用于私信回复、客服回复，评论区回复等场景
  * 内置强大创意与语言处理能力，支持多语言回复，多种场景类型。

参数:
  * shop_info: 自定义店铺信息
  * customer_id: 客户uniqueID
  * customer_message: 客户消息

（巧妙应对各种询问，让服务升级到"贴心+1"！）
""",
    response_model_exclude_none=True,
)
async def generate_single_reply(
        request: Request,
        shop_info: str = Query(..., description="店铺信息"),
        customer_id: str = Query(..., description="客户uniqueID"),
        customer_message: str = Query(..., description="客户消息"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    生成单条客户回复消息

    - **shop_info**: 店铺信息
    - **customer_id**: 客户uniqueID
    - **customer_message**: 客户消息

    返回生成的客户回复消息
    """
    start_time = time.time()

    try:
        logger.info(f"为客户 '{customer_id}' 生成回复消息")

        result = await customer_agent.generate_single_reply_message(
            shop_info=shop_info,
            customer_id=customer_id,
            customer_message=customer_message
        )

        processing_time = time.time() - start_time

        return create_response(
            data=result,
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )

    except ValidationError as e:
        logger.error(f"验证错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except ValueError as e:
        logger.error(f"参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        logger.error(f"运行时错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"生成客户回复消息时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/generate_batch_replies",
    summary="【批量智答】一键生成多条客户回复",
    description="""
用途:
  * 根据自定义店铺信息为多个客户消息批量生成个性化回复
  * 可适用于私信回复、客服回复，评论区回复等场景
  * 内置强大创意与语言处理能力，支持多语言回复，多种场景类型。
  * 返回生成的回复消息列表

参数:
  * shop_info: 店铺信息
  * customer_messages: 客户消息列表，每个包含commenter_uniqueId,text
  * batch_size: 每批处理消息数量，默认5

（让沟通效率倍增，从此告别重复回复！）
""",
    response_model_exclude_none=True,
)
async def generate_batch_replies(
        request: Request,
        shop_info: str = Query(..., description="店铺信息"),
        batch_size: int = Query(5, description="每批处理消息数量"),
        customer_messages: Dict[str, Any] = Body(...,
                                                 description="客户消息列表，每个包含commenter_uniqueId, comment_id, text",
                                                 examples=[{
                                                     "jessica1h": "请问这款气垫粉底适合干皮吗？我皮肤比较干，担心会起皮。",
                                                     # 中文
                                                     "adam_123": "Do you ship internationally? I'd like to order some items to Canada.",
                                                     # 英文
                                                     "yuki_kawaii": "この美容マスクは本当に素晴らしいです！肌がとても潤いました。また購入します！",
                                                     # 日语
                                                     "k_beauty_fan": "이 제품에 알코올이 포함되어 있나요? 제가 알코올에 민감해서요.",  # 韩语
                                                 }]
                                                 ),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    批量生成客户回复消息

    - **shop_info**: 店铺信息
    - **customer_messages**: 客户消息列表，每个包含commenter_uniqueId, comment_id, text
    - **batch_size**: 每批处理消息数量，默认为5

    返回生成的客户回复消息列表
    """
    start_time = time.time()

    try:
        logger.info(f"批量生成客户回复消息，共 {len(customer_messages)} 条")

        result = await customer_agent.generate_customer_reply_messages(
            shop_info=shop_info,
            customer_messages=customer_messages,
            batch_size=batch_size
        )

        processing_time = time.time() - start_time

        return create_response(
            data=result,
            success=True,
            processing_time_ms=round(processing_time * 1000, 2)
        )

    except ValidationError as e:
        logger.error(f"验证错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except ValueError as e:
        logger.error(f"参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        logger.error(f"运行时错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"批量生成客户回复消息时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")