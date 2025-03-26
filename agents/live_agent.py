# -*- coding: utf-8 -*-
"""
@file: live_agent.py
@desc: 生成 AI 直播内容
@auth: Callmeiks
"""

import json
import re
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, AsyncGenerator

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from app.utils.logger import setup_logger
from services.ai_models.chatgpt import ChatGPT
from services.ai_models.claude import Claude
from services.ai_models.whisper import WhisperLemonFox
from services.ai_models.genny import Genny
from services.ai_models.elevenLabs import ElevenLabsClient
from app.config import settings
from app.core.exceptions import ValidationError, ExternalAPIError, InternalServerError
from services.auth.auth_service import user_sessions
from agents.audio_generator import AudioGeneratorAgent


# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()


class LiveAgent:
    """
    为创作者生成音频文件的代理类，提供文本到语音转换功能，支持多种语言和声音效果
    """

    def __init__(self, tikhub_api_key: Optional[str] = None):
        """
        初始化音频生成代理类

        Args:
            tikhub_api_key: TikHub API密钥
        """
        # 初始化AI模型客户端
        self.chatgpt = ChatGPT()
        self.claude = Claude()
        self.whisper = WhisperLemonFox()
        self.genny = Genny()
        self.elevenLabs = ElevenLabsClient()

        # 初始化内置agent
        self.audio_generator = AudioGeneratorAgent()

        # 保存TikHub API配置
        self.tikhub_api_key = tikhub_api_key
        self.tikhub_base_url = settings.TIKHUB_BASE_URL

        # 如果没有提供TikHub API密钥，记录警告
        if not self.tikhub_api_key:
            logger.warning("未提供TikHub API密钥，某些功能可能不可用")

        # 加载系统和用户提示
        self._load_system_prompts()

    def _load_system_prompts(self) -> None:
        """加载系统提示用于不同的评论分析类型"""
        self.system_prompts = {
        }

    async def keyword_to_livescript(self, keyword: str, language: str = 'en', files: List[str] = None,
                                    live_duration: int = 1)-> AsyncGenerator[Dict[str, Any], None]:
        """
        将关键字转换为直播脚本，使用异步生成器逐段生成脚本和音频

        Args:
            keyword: 关键字
            language: 语言, 默认为英语
            files: 音频文件, 默认为None
            live_duration: 直播时长, 默认为1小时

        Returns:
            异步生成器，逐段返回生成的脚本和音频信息
        """
        try:
            logger.info(f"开始生成直播脚本，关键字: {keyword}, 语言: {language}, 时长: {live_duration}小时")

            # 1. 生成自定义voice id
            voice_id = None
            if files:
                logger.info(f"使用上传的音频文件创建自定义语音: {files}")
                voice_id = await self.audio_generator.create_voice(name=f"LiveVoice_{int(time.time())}", files=files)
                logger.info(f"创建的自定义语音ID: {voice_id}")

            # 2. 根据时长计算需要生成的内容段数(每段大约5分钟)
            segments_count = int(live_duration * 60 / 5)  # 将小时转为分钟，每5分钟一段
            logger.info(f"计划生成 {segments_count} 段内容")

            # 构建系统提示
            system_prompt = f"""你是一个专业的直播脚本生成器。根据关键词'{keyword}'，生成一段有趣、吸引人的直播内容。
            内容应该自然流畅，像真人直播一样有互动感。避免过于正式的语言，使用更口语化的表达。
            包含一些与观众的互动，例如问候、回应虚拟提问等。语言为：{language}。
            每段内容应该在400-600字之间，能够朗读约3-5分钟。
            """

            # 3. 使用异步生成器逐段生成内容
            for segment_index in range(segments_count):
                # 为每个段落定制提示
                if segment_index == 0:
                    segment_prompt = f"这是直播的开始部分，热情地向观众打招呼并介绍今天直播的主题：{keyword}"
                elif segment_index == segments_count - 1:
                    segment_prompt = f"这是直播的结束部分，总结今天讨论的内容({keyword})，感谢观众的观看，并告别"
                else:
                    segment_prompt = f"继续讨论关于'{keyword}'的内容，加入一些互动元素和有趣的观点"

                # 使用ChatGPT或Claude生成内容
                try:
                    response = await self.chatgpt.chat(
                        system_prompt=system_prompt,
                        user_prompt=segment_prompt,
                    )
                    text_content = response['response']["choices"][0]["message"]["content"].strip()

                    logger.info(f"成功生成第 {segment_index + 1}/{segments_count} 段文本")


                    # 转换文本到音频
                    audio_result = await self.audio_generator.script_to_audio(
                        text=text_content,
                        voice_id="onwK4e9ZLuTAKqWW03F9",
                        language=language
                    )

                    logger.info(f"成功生成第 {segment_index + 1}/{segments_count} 段音频")

                    # 使用yield返回当前段落的内容
                    yield {
                        "segment_index": segment_index,
                        "total_segments": segments_count,
                        "text_content": text_content,
                        "audio_url": audio_result.get("audio_url"),
                        "audio_duration": audio_result.get("duration", 0),
                        "voice_id": voice_id,
                        "timestamp": datetime.now().isoformat()
                    }

                    # 添加短暂延迟避免API限流
                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"生成第 {segment_index + 1} 段内容时出错: {str(e)}")
                    # 返回错误信息
                    yield {
                        "segment_index": segment_index,
                        "total_segments": segments_count,
                        "error": str(e)
                    }

            # 最后返回一个完成标记
            yield {
                "status": "completed",
                "total_segments": segments_count,
                "keyword": keyword,
                "language": language,
                "voice_id": voice_id,
                "duration": live_duration,
                "completed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"生成直播脚本过程中发生错误: {str(e)}")
            yield {
                "status": "error",
                "error": str(e)
            }


async def generate_and_use_live_script():
    """使用异步生成器示例"""
    agent = LiveAgent(tikhub_api_key="your_api_key")

    # 收集所有生成的段落
    all_segments = []

    # 使用异步迭代器接收生成的内容
    async for segment in agent.keyword_to_livescript(
            keyword="滚轮微针",
            files=["recording.webm"],
            live_duration=1  # 30分钟直播
    ):
        # 处理每个返回的段落
        if "error" in segment:
            print(f"段落 {segment.get('segment_index', '未知')} 生成错误: {segment['error']}")
            continue

        if segment.get("status") == "completed":
            print(f"直播脚本生成完成，共 {segment['total_segments']} 段内容")
            break

        print(f"已生成段落 {segment['segment_index'] + 1}/{segment['total_segments']}")
        print(f"文本长度: {len(segment['text_content'])} 字符")
        print(f"音频URL: {segment['audio_url']}")
        print(f"音频时长: {segment['audio_duration']} 秒")
        print("-" * 50)

        # 保存段落
        all_segments.append(segment)

    # 所有段落生成完毕后，可以进一步处理
    total_duration = sum(segment.get('audio_duration', 0) for segment in all_segments)
    print(f"总音频时长: {total_duration} 秒")

    # 可以返回完整的结果
    return {
        "keyword": "人工智能在日常生活中的应用",
        "language": "zh",
        "segments": all_segments,
        "total_duration": total_duration
    }


# 运行示例
if __name__ == "__main__":
    result = asyncio.run(generate_and_use_live_script())
    print(f"生成了 {len(result['segments'])} 个段落，总时长 {result['total_duration']} 秒")