# -*- coding: utf-8 -*-
"""
@file: customer.py
@desc: FastAPI 客户端路由
@auth: Callmeiks
"""
import json

from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request, Body
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
from app.api.models.responses import create_response
from agents.customer_agent import CustomerAgent
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
router = APIRouter(prefix="/customers")


# 依赖项：获取CustomerAgent实例
async def get_customer_agent(tikhub_api_key: str = Depends(verify_tikhub_api_key)):
    """使用验证后的TikHub API Key创建CustomerAgent实例"""
    return CustomerAgent(tikhub_api_key=tikhub_api_key)


@router.post(
    "/fetch_video_comments",
    summary="【一键直达】快速获取指定视频评论数据",
    description="""
用途:
  * 获取TikTok视频评论数据，返回清洗后的评论列表
  * 包括评论ID、评论内容、点赞数、回复数、评论者用户名、评论者安全用户ID(SecUid)、评论语言、评论者国家、Instagram ID、Twitter ID、创建时间

参数:
  * aweme_id: TikTok视频ID
  
（超高效舆情分析，助您精准捕捉热点！）
""",
    response_model_exclude_none=True,
)
async def fetch_video_comments(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        ins_filter: Optional[bool] = Query(False, description="是否过滤Instagram为空的用户"),
        twitter_filter: Optional[bool] = Query(False, description="是否过滤Twitter为空的用户"),
        region_filter: Optional[str] = Query(None, description="按地区过滤用户"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    获取指定TikTok视频的评论数据

    - **aweme_id**: TikTok视频ID
    - **ins_filter**: 是否过滤Instagram为空的用户，默认为False
    - **twitter_filter**: 是否过滤Twitter为空的用户，默认为False
    - **region_filter**: 按地区过滤用户，默认不过滤

    返回清理后的评论列表
    """
    start_time = time.time()

    try:
        logger.info(f"获取视频 {aweme_id} 的评论")

        comments_data = await customer_agent.fetch_video_comments(aweme_id, ins_filter, twitter_filter, region_filter)

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
    "/fetch_purchase_intent_stats",
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
async def analyze_purchase_intent(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="每批处理的评论数量"),
        concurrency: int = Query(5, description="ai处理并发数"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    分析指定TikTok视频评论中的购买意图

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每批处理的评论数量，默认为30
    - **concurrency**: ai处理并发数，默认为5, 最大为10

    返回购买意图分析结果
    """
    start_time = time.time()

    try:
        logger.info(f"分析视频 {aweme_id} 的购买意图")

        result = await customer_agent.get_purchase_intent_stats(
            aweme_id,
            batch_size,
            concurrency
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

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except InternalServerError as e:
        logger.error(f"内部服务器错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"分析购买意图时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/get_potential_customers",
    summary="【深度挖掘】指定视频中潜在客户信息",
    description="""
用途:
  * 根据评论识别TikTok视频的购买意向潜在客户
  * 自动识别/抓取竞争对手的潜在客户
  * 识别多种语言，多种场景类型
  * 返回潜在客户信息（用户名、ID、评论内容、国家、社交媒体ID、创建时间、购买意向分数）

参数:
  * aweme_id: TikTok视频ID
  * batch_size: 每批处理评论数量，默认30
  * customer_count: 最大返回客户数量，默认100
  * min_score: 最小购买意向分数，范围0-100，默认50
  * max_score: 最大购买意向分数，范围1-100，默认100
  * ins_filter: 是否过滤Instagram为空的用户，默认False
  * twitter_filter: 是否过滤Twitter为空的用户，默认False
  * region_filter: 按地区过滤用户，默认不过滤

（一键找出可能为您买单的优质用户！）
""",
    response_model_exclude_none=True,
)
async def get_potential_customers(
        request: Request,
        aweme_id: str = Query(..., description="TikTok视频ID"),
        batch_size: int = Query(30, description="ai每批处理评论数量"),
        customer_count: int = Query(100, description="最大返回客户数量"),
        concurrency: int = Query(5, description="ai处理并发数"),
        min_score: Optional[int] = Query(50, description="最小参与度分数，范围0-100"),
        max_score: Optional[int] = Query(100, description="最大参与度分数，范围1-100"),
        ins_filter: Optional[bool] = Query(False, description="是否过滤Instagram为空的用户"),
        twitter_filter: Optional[bool] = Query(False, description="是否过滤Twitter为空的用户"),
        region_filter: Optional[str] = Query(None, description="按地区过滤用户"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    根据评论识别TikTok视频的潜在客户

    - **aweme_id**: TikTok视频ID
    - **batch_size**: 每批处理的评论数量，默认为30
    - **min_score**: 最小参与度分数，范围0-100，默认为50
    - **max_score**: 最大参与度分数，范围1-100，默认为100

    返回潜在客户分析结果
    """
    start_time = time.time()

    try:
        logger.info(f"识别视频 {aweme_id} 的潜在客户")

        result = await customer_agent.get_potential_customers(
            aweme_id,
            batch_size,
            customer_count,
            concurrency,
            min_score,
            max_score,
            ins_filter,
            twitter_filter,
            region_filter
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

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except InternalServerError as e:
        logger.error(f"内部服务器错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"识别潜在客户时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/get_keyword_potential_customers",
    summary="【关键词搜索】高效挖掘细分赛道潜在买家",
    description="""
用途:
  * 根据关键词搜索视频并识别其中具有购买意向的潜在客户
  * 自动识别/抓取同赛道竞争对手的潜在客户
  * 高效采集指定赛道的潜在买家信息
  * 返回潜在客户信息（关键词、客户列表、统计数据）

参数:
  * keyword: 搜索关键词
  * batch_size: 每批处理评论数量，默认30
  * customer_count: 最大返回客户数量，默认100
  * video_concurrency: 视频处理并发数，默认5
  * ai_concurrency: AI处理并发数，默认5
  * min_score: 最小购买意向分数，范围0-100，默认50
  * max_score: 最大购买意向分数，范围1-100，默认100
  * ins_filter: 是否过滤Instagram为空的用户，默认False
  * twitter_filter: 是否过滤Twitter为空的用户，默认False
  * region_filter: 按地区过滤用户，默认不过滤

（让流量精准变现，从海量视频中锁定目标客户！）
""",
    response_model_exclude_none=True,
)
async def get_keyword_potential_customers(
        request: Request,
        keyword: str = Query(..., description="搜索关键词"),
        batch_size: int = Query(30, description="每批处理评论数量"),
        customer_count: int = Query(100, description="最大返回客户数量"),
        video_concurrency: int = Query(5, description="视频处理并发数"),
        ai_concurrency: int = Query(5, description="AI处理并发数"),
        min_score: float = Query(50.0, description="最小购买意向分数，范围0-100"),
        max_score: float = Query(100.0, description="最大购买意向分数，范围1-100"),
        ins_filter: bool = Query(False, description="是否过滤Instagram为空的用户"),
        twitter_filter: bool = Query(False, description="是否过滤Twitter为空的用户"),
        region_filter: Optional[str] = Query(None, description="按地区过滤用户"),
        customer_agent: CustomerAgent = Depends(get_customer_agent)
):
    """
    根据关键词搜索视频并识别其中具有购买意向的潜在客户

    - **keyword**: 搜索关键词
    - **batch_size**: 每批处理的评论数量，默认为30
    - **video_concurrency**: 视频处理并发数，默认为5
    - **ai_concurrency**: AI处理并发数，默认为5
    - **min_score**: 最小购买意向分数，范围0-100，默认为50
    - **max_score**: 最大购买意向分数，范围1-100，默认为100
    - **ins_filter**: 是否过滤Instagram为空的用户，默认为False
    - **twitter_filter**: 是否过滤Twitter为空的用户，默认为False
    - **region_filter**: 按地区过滤用户，默认不过滤
    - **max_customers**: 最大返回客户数量，默认不限制

    返回关键词搜索的潜在客户分析结果
    """
    start_time = time.time()

    try:
        logger.info(f"识别关键词 '{keyword}' 搜索结果中的潜在客户")

        result = await customer_agent.get_keyword_potential_customers(
            keyword=keyword,
            batch_size=batch_size,
            customer_count=customer_count,
            video_concurrency=video_concurrency,
            ai_concurrency=ai_concurrency,
            min_score=min_score,
            max_score=max_score,
            ins_filter=ins_filter,
            twitter_filter=twitter_filter,
            region_filter=region_filter,
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

    except ExternalAPIError as e:
        logger.error(f"外部API错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except InternalServerError as e:
        logger.error(f"内部服务器错误: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except ValueError as e:
        logger.error(f"参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        logger.error(f"运行时错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"识别关键词潜在客户时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


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

（巧妙应对各种询问，让服务升级到“贴心+1”！）
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
