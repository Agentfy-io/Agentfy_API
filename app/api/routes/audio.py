# -*- coding: utf-8 -*-
"""
@file: audio.py
@desc: FastAPI 客户端路由
@auth: Callmeiks
"""
import json

from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request, Body, Form
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
from app.api.models.responses import create_response
from agents.audio_generator import AudioGeneratorAgent
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
router = APIRouter(prefix="/audio")


# 依赖项：获取audio_generator实例
async def get_audio_generator(tikhub_api_key: str = Depends(verify_tikhub_api_key)):
    """使用验证后的TikHub API Key创建AudioGeneratorAgent实例"""
    return AudioGeneratorAgent(tikhub_api_key=tikhub_api_key, tikhub_base_url=settings.TIKHUB_BASE_URL)


@router.post(
    "/text_to_script",
    summary="【一键爆款】基于关键词生成创意短视频脚本与音频文稿",
    description="""
用途:
  * 输入简单关键词，即可自动生成高吸引力的短视频脚本或音频文稿
  * 支持多种场景类型（故事、产品推广、科普等）与多语言选择（中文、英文等）
  * 内置强大创意与语言处理能力，可一键爆发您的创作灵感。

参数:
  * text: 用户输入的关键词/文本
  * scenarioType：场景类型（如故事类型、产品展示类型等）
  * language：语言（如zh-CN、en-US等）

（此接口能让您在最短时间内轻松制作炫酷短视频或音频内容，让您的灵感一触即发！）
""",
    response_model_exclude_none=True,
)
async def text_to_script(
        request: Request,
        prompt: str = Query(..., description="用户输入的关键词/文本"),
        scenarioType: str = Query(..., description="场景类型（如：故事类型、产品展示类型等）"),
        language: str = Query(..., description="目标语言（如：zh-CN、en-US等）"),
        audio_generator: AudioGeneratorAgent = Depends(get_audio_generator)
):
    # -------------------------------------------------------------------------------------
    # 【简要说明】
    # 1. 根据用户提供的关键词、场景类型及语言要求，自动生成短视频脚本或音频稿。
    # 2. 内置强大创意与语言处理能力，可一键爆发您的创作灵感。
    # 3. 若发生任何验证错误、外部API错误或未知异常，会进行异常捕获并返回友好提示。
    # -------------------------------------------------------------------------------------

    start_time = time.time()

    try:
        script = await audio_generator.text_to_script(prompt, scenarioType, language)
        processing_time = time.time() - start_time
        return create_response(
            data=script,
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
        logger.error(f"生成短视频脚本或音频稿时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/script_to_audio",
    summary="【一键生成Tiktok音频】根据自定义文本生成音频",
    description="""
用途:
  * 输入自定义文本 text、语言、性别、年龄和语速，一键生成对应的 Tiktok 音频
  * 可根据不同发音人自动选择并合成音频

参数:
  * text: 待转换的文本
  * language: 语言（如 zh-CN, en-US 等）
  * gender: 性别（male / female）
  * age: 年龄（child, teen, young, middle, elderly 等）
  * speed: 语速（默认 1 表示正常语速，数值越大越快）

（此接口让您迅速合成多元化配音效果，为短视频或音频内容赋予更多创意！）
""",
    response_model_exclude_none=True,
)
async def script_to_audio_endpoint(
        request: Request,
        text: str = Form(..., description="待转换的文本"),
        language: str = Form(..., description="目标语言（如：zh-CN、en-US 等）"),
        gender: str = Form(..., description="说话人性别（如：male、female）"),
        age: str = Form(..., description="说话人年龄段（如：child, young, middle, elderly）"),
        speed: int = Form(1, description="语速，越大越快，默认=1"),
        audio_generator: Any = Depends(get_audio_generator),
):
    """
    根据自定义文本生成 Tiktok 音频
    """
    start_time = time.time()

    try:
        logger.info("开始生成音频")
        audio_summary = await audio_generator.script_to_audio(text, language, gender, age, speed)
        processing_time = time.time() - start_time

        return create_response(
            data=audio_summary,
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
        logger.error(f"未知错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.post(
    "/text_to_audio",
    summary="【一键生成短视频音频】根据用户关键词生成脚本并合成音频",
    description="""
用途:
  * 输入关键词 prompt、场景类型 scenarioType 及语言等信息，自动生成短视频脚本文本，再合成为 Tiktok 音频
  * 同时可指定性别、年龄和语速以获得不同风格的配音效果

参数:
  * prompt: 用户关键词/文本
  * scenarioType: 场景类型（如：故事类型、产品展示类型等）
  * language: 语言（如：zh-CN、en-US 等）
  * gender: 性别（male / female）
  * age: 年龄（child, teen, young, middle, elderly 等）
  * speed: 语速（默认 1）

（此接口整合脚本生成与语音合成，一站式满足短视频或音频内容需求！）
""",
    response_model_exclude_none=True,
)
async def text_to_audio_endpoint(
        request: Request,
        prompt: str = Query(..., description="用户输入的关键词/文本"),
        scenarioType: str = Query(..., description="场景类型（如：故事类型、产品展示类型等）"),
        language: str = Query(..., description="目标语言（如：zh-CN、en-US 等）"),
        gender: str = Query(..., description="说话人性别（如：male、female）"),
        age: str = Query(..., description="说话人年龄段（如：child, young, middle, elderly）"),
        speed: int = Query(1, description="语速，越大越快，默认=1"),
        audio_generator: Any = Depends(get_audio_generator),
):
    """
    根据用户提供的关键词生成脚本，然后合成为音频
    """
    start_time = time.time()

    try:
        # 首先生成文本脚本
        logger.info("开始生成脚本")
        script_result = await audio_generator.text_to_script(prompt, scenarioType, language)

        # 从脚本结果里获取文案内容
        transcript = script_result.get('transcript', '')
        if not transcript:
            logger.error("脚本生成结果为空")
            raise ValidationError(detail="脚本生成失败，内容为空")

        # 调用合成音频
        logger.info("脚本生成成功，开始合成音频")
        audio_result = await audio_generator.script_to_audio(transcript, language, gender, age, speed)

        processing_time = time.time() - start_time
        return create_response(
            data=audio_result,
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
        logger.error(f"生成音频时发生未预期错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")