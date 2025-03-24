# -*- coding: utf-8 -*-
"""
@file: xhs.py
@desc: FastAPI 小红书相关路由
@auth: Callmeiks
"""
import random
import string
import time
from typing import Dict, Any, AsyncGenerator

from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request, BackgroundTasks

from app.api.models.responses import create_response
from agents.xhs_agent import XHSAgent  # 假设你有这个代理类
from app.core.exceptions import (
    ValidationError,
    ExternalAPIError,
    InternalServerError
)
from app.utils.logger import setup_logger
from app.dependencies import verify_tikhub_api_key  # 从dependencies.py导入验证函数
import pandas as pd

# 设置日志记录器
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/xhs")

# 用于存储后台任务结果的字典
task_results = {}


# 依赖项：获取XHSAgent实例
async def get_xhs_agent(tikhub_api_key: str = Depends(verify_tikhub_api_key)):
    """使用验证后的TikHub API Key创建XHSAgent实例"""
    return XHSAgent(api_key=tikhub_api_key)


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


@router.post(
    "/url_to_xhs",
    summary="【平台转换】抖音链接转小红书文案",
    description="""
用途:
  * 将抖音视频内容转换为小红书风格的文案
  * 根据目标人群特征优化转换内容
  * 保留原视频核心信息，同时调整为目标平台表达习惯

参数:
  * item_url: 抖音视频分享链接
  * source_platform: 源平台（默认为抖音）
  * target_platform: 目标平台（默认为小红书）
  * target_gender: 目标性别（默认为女性）
  * target_age: 目标年龄段（默认为18-30岁）

（一键适配不同平台文案风格，高效实现内容转换！）
""",
    response_model_exclude_none=True,
)
async def url_to_xhs(
        request: Request,
        item_url: str = Query("https://v.douyin.com/ifsy1fVo/", description="抖音视频分享链接"),
        source_platform: str = Query("抖音", description="源平台"),
        target_platform: str = Query("小红书", description="目标平台"),
        target_gender: str = Query("女性", description="目标性别"),
        target_age: str = Query("18-30岁", description="目标年龄段"),
        xhs_agent: XHSAgent = Depends(get_xhs_agent)
):
    """
    将抖音视频转换为小红书风格
    """
    try:
        # 直接调用XHSAgent的方法
        result = await xhs_agent.url_to_xhs(
            item_url=item_url,
            source_platform=source_platform,
            target_platform=target_platform,
            target_gender=target_gender,
            target_age=target_age
        )

        # 返回结果
        return create_response(
            data=result,
            success=True
        )
    except ValidationError as e:
        logger.error(f"验证错误: {str(e)}")
        return create_response(
            data={"error": str(e)},
            message=f"验证错误: {str(e)}",
            success=False
        )
    except ExternalAPIError as e:
        logger.error(f"外部API错误: {str(e)}")
        return create_response(
            data={"error": str(e)},
            message=f"外部API错误: {str(e)}",
            success=False
        )
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        return create_response(
            data={"error": str(e)},
            message=f"内部服务器错误: {str(e)}",
            success=False
        )


@router.post(
    "/keyword_to_xhs",
    summary="【关键词搜索】抖音关键词搜索结果转小红书文案",
    description="""
用途:
  * 搜索抖音上特定关键词的视频并转换为小红书风格
  * 自动筛选点赞最高的相关视频进行内容转换
  * 适用于竞品研究、爆款内容追踪等场景

参数:
  * keyword: 搜索关键词
  * source_platform: 源平台（默认为抖音）
  * target_platform: 目标平台（默认为小红书）
  * target_gender: 目标性别（默认为女性）
  * target_age: 目标年龄段（默认为18-30岁）

（爆款关键词一键转换，轻松发现并改编热门内容！）
""",
    response_model_exclude_none=True,
)
async def keyword_to_xhs(
        request: Request,
        background_tasks: BackgroundTasks,
        keyword: str = Query(..., description="搜索关键词"),
        source_platform: str = Query("抖音", description="源平台"),
        target_platform: str = Query("小红书", description="目标平台"),
        target_gender: str = Query("女性", description="目标性别"),
        target_age: str = Query("18-30岁", description="目标年龄段"),
        xhs_agent: XHSAgent = Depends(get_xhs_agent)
):
    """
    将抖音关键词搜索结果转换为小红书风格
    """
    # 生成任务ID
    task_id = generate_task_id("keyword_xhs")

    # 初始化任务状态
    task_results[task_id] = {
        "status": "created",
        "message": "任务已创建，等待启动",
        "keyword": keyword,
    }

    # 创建处理函数
    async def process_keyword_to_xhs_task(task_id: str):
        try:
            # 设置初始状态
            task_results[task_id]["status"] = "in_progress"
            task_results[task_id]["message"] = "任务已创建，正在启动..."

            # 调用异步生成器方法
            async for result in xhs_agent.keyword_to_xhs(
                    keyword=keyword,
                    source_platform=source_platform,
                    target_platform=target_platform,
                    target_gender=target_gender,
                    target_age=target_age
            ):
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

    # 添加后台任务
    background_tasks.add_task(process_keyword_to_xhs_task, task_id)

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
  * 查询转换任务的状态和结果
  * 适用于长时间运行的批量转换任务
  * 返回任务状态、进度信息和转换结果

参数:
  * task_id: 任务ID

（随时掌握任务进度，高效管理内容转换流程！）
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