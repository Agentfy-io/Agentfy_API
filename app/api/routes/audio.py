# -*- coding: utf-8 -*-
"""
@file: audio.py
@desc: FastAPI 客户端路由 - 音频生成
@auth: Callmeiks
"""
import asyncio
import random
import shutil
import string
import os
import time
from typing import List, Optional, Callable

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Path,
    HTTPException,
    Request,
    Body,
    BackgroundTasks,
    File,
    UploadFile,
    Form
)

from app.api.models.responses import create_response
from agents.audio_generator import AudioGeneratorAgent
from app.core.exceptions import (
    ValidationError,
    ExternalAPIError,
    InternalServerError,
)
from app.utils.logger import setup_logger
from app.dependencies import verify_tikhub_api_key, verify_openai_api_key, verify_lemonfox_api_key, verify_elevenlabs_api_key
from app.config import settings

# 设置日志记录器
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/audio")

# 用于存储后台任务结果的字典
task_results = {}


# 依赖项：获取AudioGeneratorAgent实例
async def get_audio_agent(tikhub_api_key: str = Depends(verify_tikhub_api_key),
                          openai_api_key: str = Depends(verify_openai_api_key),
                          lemonfox_api_key: str = Depends(verify_lemonfox_api_key),
                          elevenlabs_api_key: str = Depends(verify_elevenlabs_api_key)):
    """使用验证后的TikHub API Key创建AudioGeneratorAgent实例"""
    return AudioGeneratorAgent(tikhub_api_key=tikhub_api_key,
                               openai_api_key=openai_api_key,
                               lemonfox_api_key=lemonfox_api_key,
                               elevenlabs_api_key=elevenlabs_api_key)


# 通用任务处理函数
async def process_audio_task(task_id: str, analysis_method: Callable, **kwargs):
    """
    处理音频生成任务

    Args:
        task_id: 任务ID
        analysis_method: 异步方法
        **kwargs: 传递给方法的参数
    """
    try:
        # 设置初始状态
        task_results[task_id]["status"] = "in_progress"
        task_results[task_id]["message"] = "任务已创建，正在启动..."

        # 执行方法并获取结果
        result = await analysis_method(**kwargs)

        # 更新任务结果
        task_results[task_id].update(result)
        task_results[task_id]["status"] = "completed"
        task_results[task_id]["message"] = "任务已完成"

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


# 清理临时文件的函数
async def cleanup_temp_files(task_id: str, temp_dir: str, saved_files: List[str]):
    """
    清理上传文件的临时文件和目录

    Args:
        task_id: 任务ID
        temp_dir: 临时目录路径
        saved_files: 保存的文件路径列表
    """
    # 等待任务完成
    while task_id in task_results and task_results[task_id]["status"] in ["created", "processing"]:
        await asyncio.sleep(1)

    # 清理文件
    for file_path in saved_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info("清理临时文件: %s", file_path)
        except Exception as e:
            logger.error(f"清理临时文件 {file_path} 失败: {str(e)}")

    # 清理目录
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("清理临时目录: %s", temp_dir)
    except Exception as e:
        logger.error(f"清理临时目录 {temp_dir} 失败: {str(e)}")


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
    "/keyword_to_script",
    summary="【脚本生成】根据关键词/主题生成语音文本",
    description="""
用途:
  * 根据用户输入关键词/主题，生成适合语音朗读的脚本
  * 支持多种场景类型和语言选择
  * 返回生成的脚本文本及相关元数据

参数:
  * prompt: 用户输入关键词/主题
  * scenarioType: 场景类型 (Storytelling, Product Showcase, Educational/Explainer,Tutorial/How-To, Promotional/Marketing, Testimonial/Review, Comedy/Entertainment, Lifestyle/Vlog, Inspirational/Motivational, Documentary-Style, News/Current Events, Case Study, Interview or Q&A, Health & Wellness, Cooking/Recipe,Fashion/Beauty, Travel/Adventure, Tech Tips/Hacks, Challenge or Game)
  * language: 语言 (en, zh)

（快速生成专业脚本，为您的创作提供高质量内容支持！）
""",
    response_model_exclude_none=True,
)
async def text_to_script(
        request: Request,
        background_tasks: BackgroundTasks,
        prompt: str = Query(..., description="用户输入关键词/主题"),
        scenarioType: str = Query(..., description="场景类型"),
        language: str = Query(..., description="语言"),
        audio_agent: AudioGeneratorAgent = Depends(get_audio_agent)
):
    """
    根据用户输入关键词生成语音文本
    """
    # 生成任务ID
    task_id = generate_task_id("text_script")

    # 初始化任务状态
    task_results[task_id] = {
        "status": "created",
        "message": "任务已创建，等待启动",
    }

    # 添加后台任务
    background_tasks.add_task(
        process_audio_task,
        task_id=task_id,
        analysis_method=audio_agent.text_to_script,
        prompt=prompt,
        scenarioType=scenarioType,
        language=language
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
    "/script_to_audio",
    summary="【音频生成】将文本转换为TTS音频",
    description="""
用途:
  * 将提供的文本转换为高质量TTS音频
  * 支持多种语言、性别和年龄选择
  * 支持自定义发音人（可选）
  * 返回生成的音频URL及相关元数据

参数:
  * text: 待转换的文本
  * language: 语言 (en, zh)
  * gender: 性别 (male, female) 
  * age: 年龄 (young, middle-aged, old)
  * voice_id: 发音人ID（可选）

（一键生成自然流畅的语音，为您的内容注入生命力！）
""",
    response_model_exclude_none=True,
)
async def script_to_audio(
        request: Request,
        background_tasks: BackgroundTasks,
        text: str = Body(
            "TikTok is a dynamic, multi-purpose digital platform that seamlessly blends innovative technology with user-friendly interfaces. Originally developed as a solution for cross-platform data integration, it has evolved into a comprehensive ecosystem serving industries from healthcare to finance. Its distinctive approach to secure data handling and real-time analytics has earned recognition among both tech enthusiasts and enterprise users.",
            description="待转换的文本"
        ),
        language: str = Body("en", description="语言"),
        gender: str = Body("male", description="性别"),
        age: str = Body("middle-aged", description="年龄"),
        voice_id: Optional[str] = Body("", description="发音人ID（可选）"),
        audio_agent: AudioGeneratorAgent = Depends(get_audio_agent)
):
    """
    根据文本生成音频
    """
    # 生成任务ID
    task_id = generate_task_id("audio_gen")

    # 初始化任务状态
    task_results[task_id] = {
        "status": "created",
        "message": "任务已创建，等待启动",
    }

    # 添加后台任务
    background_tasks.add_task(
        process_audio_task,
        task_id=task_id,
        analysis_method=audio_agent.script_to_audio,
        text=text,
        language=language,
        gender=gender,
        age=age,
        voice_id=voice_id
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
    "/keyword_to_audio",
    summary="【一键生成】关键词直接生成音频",
    description="""
用途:
  * 一站式服务：直接从关键词/主题生成音频
  * 自动完成文本生成和TTS转换
  * 支持多种场景类型、语言、性别和年龄选择
  * 支持自定义发音人（可选）
  * 返回生成的音频URL及相关元数据

参数:
  * prompt: 用户输入关键词/主题
  * scenarioType: 场景类型 (Storytelling, Product Showcase, Educational/Explainer,Tutorial/How-To, Promotional/Marketing, Testimonial/Review, Comedy/Entertainment, Lifestyle/Vlog, Inspirational/Motivational, Documentary-Style, News/Current Events, Case Study, Interview or Q&A, Health & Wellness, Cooking/Recipe,Fashion/Beauty, Travel/Adventure, Tech Tips/Hacks, Challenge or Game)
  * language: 语言 (en, zh)
  * gender: 性别 (male, female) 
  * age: 年龄 (young, middle-aged, old)
  * voice_id: 发音人ID（可选）

（从创意到成品，一键搞定，让创作更加高效便捷！）
""",
    response_model_exclude_none=True,
)
async def text_to_audio(
        request: Request,
        background_tasks: BackgroundTasks,
        prompt: str = Body('Difference between Graff and Chaumet diamond rings', description="用户输入关键词/主题"),
        scenarioType: str = Body("Storytelling", description="场景类型"),
        language: str = Body('en', description="语言"),
        gender: str = Body("male", description="性别"),
        age: str = Body("middle-aged", description="年龄"),
        voice_id: Optional[str] = Body("", description="发音人ID（可选）"),
        audio_agent: AudioGeneratorAgent = Depends(get_audio_agent)
):
    """
    从关键词直接生成音频
    """
    # 生成任务ID
    task_id = generate_task_id("full_audio")

    # 初始化任务状态
    task_results[task_id] = {
        "status": "created",
        "message": "任务已创建，等待启动",
    }

    # 添加后台任务
    background_tasks.add_task(
        process_audio_task,
        task_id=task_id,
        analysis_method=audio_agent.text_to_audio,
        prompt=prompt,
        scenarioType=scenarioType,
        language=language,
        gender=gender,
        age=age,
        voice_id=voice_id
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
    "/create_voice",
    summary="【声音克隆】添加自定义声音",
    description="""
用途:
  * 上传音频样本文件，添加自定义声音
  * 支持多种音频格式（MP3, MP4, webm, amr）
  * 支持添加声音名称、描述和标签

参数:
  * name: 声音名称
  * files: 上传的音频样本文件列表
  * description: 声音描述（可选）
  * labels: 标签JSON字符串（可选）
""",
    response_model_exclude_none=True,
)
async def create_voice(
        request: Request,
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File([], description="上传的音频样本文件"),
        name: str = Form("Lucy's Voice", description="声音名称"),
        description: Optional[str] = Form("This is Lucy's voice", description="声音描述"),
        labels: Optional[str] = Form("", description="标签JSON字符串"),
        audio_agent: AudioGeneratorAgent = Depends(get_audio_agent)
):
    """添加新声音（声音克隆）"""
    # 验证输入
    if not files:
        raise ValidationError("请至少上传一个MP3文件或提供一个视频URL")

    # 检查上传的文件格式
    valid_extensions = (".mp3", ".mp4", ".webm", ".amr")
    for file in files:
        if not file.filename.lower().endswith(valid_extensions):
            raise ValidationError(
                f"文件 {file.filename} 不是有效的MP3, MP4, webm, amr格式"
            )

    # 生成任务ID
    task_id = generate_task_id("create_voice")

    # 创建临时目录
    temp_dir = os.path.join("temp_files", task_id)
    os.makedirs(temp_dir, exist_ok=True)

    saved_file_paths = []
    try:
        # 保存上传的文件
        for file in files:
            # 创建安全的文件名
            safe_filename = os.path.basename(file.filename)
            file_path = os.path.join(temp_dir, safe_filename)

            # 读取并保存文件内容
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_file_paths.append(file_path)

        # 初始化任务状态
        task_results[task_id] = {
            "status": "created",
            "message": "任务已创建，等待启动",
        }

        # 添加后台任务
        background_tasks.add_task(
            process_audio_task,
            task_id=task_id,
            analysis_method=audio_agent.create_voice,
            files=saved_file_paths,  # 传递保存的文件路径
            name=name,
            description=description,
            labels=labels
        )

        # 添加后台任务完成后的清理任务
        background_tasks.add_task(
            cleanup_temp_files,
            task_id=task_id,
            temp_dir=temp_dir,
            saved_files=saved_file_paths
        )

        # 返回任务信息
        return create_response(
            data={
                "task_id": task_id,
                "status": "created",
                "message": "任务已创建，正在启动",
                "files_count": len(saved_file_paths),
                "name": name,
                "description": description
            },
            success=True
        )
    except Exception as e:
        # 清理已创建的文件和目录
        for path in saved_file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass
        logger.error(f"处理上传文件时出错: {str(e)}")
        raise InternalServerError(f"处理上传文件时出错: {str(e)}")


@router.get(
    "/tasks/{task_id}",
    summary="【任务查询】获取任务状态与结果",
    description="""
用途:
  * 查询音频生成任务的状态和结果
  * 适用于长时间运行的音频生成任务
  * 返回任务状态、进度信息和结果数据

参数:
  * task_id: 任务ID

（随时掌握任务进度，高效管理音频生成流程！）
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