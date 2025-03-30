# -*- coding: utf-8 -*-
"""
@file: audio_generator.py
@desc: 为创作者生成音频文件的代理类，提供文本到语音转换功能，支持多种语言和声音效果
@auth: Callmeiks
"""

import json
import re
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

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

# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()


class AudioGeneratorAgent:
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
            "text_to_script": """You are a professional short-video script creation assistant. When given a brief request from the user (e.g., "Create a short story about Nezha"), you should automatically infer and decide on the following five elements:           
                    1. language:
                       - The language to be used in the script. 
                       - Must be returned using an ISO language-region format (e.g., "zh-CN", "en-US"). 
                       - If the user does not specify a language, you should automatically match the user's input or default to an appropriate language.           
                    2. context:
                       - The general context or theme (Mythology & Folklore, Historical Events & Biographies, Science & Technology, Educational/Academic, Art & Literature, Culture & Heritage, Travel & Exploration, Food & Culinary, Product Promotion/Commercial, Lifestyle, Entertainment & Pop Culture, Inspirational & Motivational, Health & Wellness, Personal Development & Self-Help, Philosophy & Ethics, Finance & Economics, Business & Entrepreneurship, Marketing & Advertising, Environmental & Sustainability, Social Issues & Awareness, Comedy & Satire, Parenting & Family, Relationships & Dating, Case Studies/Testimonials, Professional Skills & Career, Language Learning, Hobbies & Crafts, Gaming, News & Current Events, Sports & Fitness, Music & Performing Arts, Film & TV)                  
                    3. scenarioType:
                       - The scenario type. (Storytelling,Product Showcase, Educational/Explainer,Tutorial/How-To, Promotional/Marketing, Testimonial/Review, Comedy/Entertainment, Lifestyle/Vlog, Inspirational/Motivational, Documentary-Style, News/Current Events, Case Study, Interview or Q&A, Health & Wellness, Cooking/Recipe,Fashion/Beauty, Travel/Adventure, Tech Tips/Hacks, Challenge or Game.)              
                    4. tone:
                       - The overall style or mood (friendly, serious, playful, comedic, dramatic, authoritative, casual, formal, lively, motivational, urgent, enthusiastic, calm, warm, uplifting, confident, nostalgic, encouraging, soothing, empathetic, matter-of-fact, neutral, romantic, humorous). 
                       - If the user does not specify, default to a neutral/friendly tone.           
                    5. duration:
                       - The target reading duration for the script. If the user does not specify a duration, please aim for approximately 30 seconds of spoken content.          
                    **Your task**:
                    - When the user provides only a brief request, infer as many details as possible for these eight elements. 
                    - Create a short script suitable for platforms like TikTok or any short-video format, including an engaging opening, core content, and a concise conclusion or call-to-action. 
                    - Use textual cues to indicate pacing or tempo if needed (e.g., ellipses for dramatic pauses).
                    - Ensure the script is engaging, informative, and suitable for the specified context and language.
                    - you may pair up multiple tones with different scenarios, but make sure the tone is appropriate for the context.
                    - Return **only** the script in text (with no extra text or formatting):            

                    **Important**:
                    - If any user requirement is unclear or missing, assume reasonable defaults. 
                    - Always ensure the final script matches the language and context inferred or specified by the user.           
                    - Remember that this transcript will be converted to speech, so it should sound natural when read aloud. Use appropriate pacing, transitions, and conversational elements to ensure a smooth listening experience in the specified language.
                    """
        }

    async def text_to_script(
            self,
            prompt: str,
            scenarioType: str,
            language: str
    ) -> Dict[str, Any]:
        """
        根据用户输入关键词生成语音文本

        Args:
            prompt: 用户提示
            scenarioType: 场景类型
            language: 语言

        Returns:
            转换后的文本及元数据
        """
        start_time = time.time()

        try:
            # 参数验证
            if not prompt or not scenarioType or not language:
                raise ValidationError("缺少必要参数")

            logger.info(f"开始生成语音文本")

            # 生成文本
            sys_prompt = self.system_prompts.get("text_to_script")
            user_prompt = (
                f"Here is the request from the user: {prompt}\n\n"
                f"The scenario type is {scenarioType}, the language is {language}"
            )

            # 调用ChatGPT生成文本
            response = await self.chatgpt.chat(
                system_prompt=sys_prompt,
                user_prompt=user_prompt
            )

            # 解析ChatGPT返回的结果
            transcript = response['response']["choices"][0]["message"]["content"].strip()

            # 记录成功信息
            processing_time = time.time() - start_time
            logger.info(f"生成语音文本成功，耗时: {processing_time:.2f}秒")

            # 返回结果
            return {
                "transcript": transcript,
                "metadata": {
                    "prompt": prompt,
                    "language": language,
                    "scenarioType": scenarioType,
                    "llm_processing_cost": response['cost'],
                    'generated_at': datetime.now().isoformat(),
                    'processing_time': processing_time
                },
            }
        except ValidationError:
            # 直接向上传递验证错误
            raise
        except ExternalAPIError:
            # 直接向上传递API错误
            raise
        except Exception as e:
            logger.error(f"生成语音文本时发生未预期错误: {str(e)}")
            raise InternalServerError(f"生成语音文本时发生未预期错误: {str(e)}")

    async def script_to_audio(
            self,
            text: str,
            language: str = "en",
            gender: str = "male",
            age: str = "middle-aged",
            voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        根据自定义text生成音频

        Args:
            text: 待转换的文本
            language: 语言
            gender: 性别
            age: 年龄
            voice_id: 特定声音ID（可选）

        Returns:
            生成的音频文件信息
        """
        start_time = time.time()
        selected_voice_id = ""

        try:
            logger.info(f"开始生成音频")

            if voice_id:
                # 使用特定声音ID
                selected_voice_id = voice_id

            elif not voice_id:
                # 获取发音人列表
                voices = await self.elevenLabs.get_voices(language, gender, age)

                if not voices:
                    error_msg = "该语言和性别没有可用的发音人, 请尝试其他配置"
                    logger.error(error_msg)
                    raise ExternalAPIError(error_msg)

                # 选择第一个发音人
                selected_voice_id = voices[0]

            # 生成音频
            audio_url = await self.elevenLabs.text_to_speech(selected_voice_id, text)

            # 计算处理时间
            processing_time = time.time() - start_time

            # 构建结果
            audio_summary = {
                "audio_url": audio_url,
                "text": text,
                "meta": {
                    "voice_id": selected_voice_id,
                    "language": language,
                    "gender": gender,
                    "age": age,
                    "generated_at": datetime.now().isoformat(),
                    "processing_time": processing_time
                }
            }

            logger.info(f"生成音频成功，耗时: {processing_time:.2f}秒")

            return audio_summary

        except ExternalAPIError as e:
            logger.error(f"无法生成音频: {str(e)}")
            raise ExternalAPIError(f"无法生成音频: {str(e)}")
        except Exception as e:
            logger.error(f"生成音频时发生未知错误: {str(e)}")
            raise InternalServerError(f"生成音频时发生未知错误: {str(e)}")

    async def text_to_audio(
            self,
            prompt: str,
            scenarioType: str,
            language: str,
            gender: Optional[str],
            age: Optional[str],
            voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        根据用户输入关键词生成音频（一键生成）

        Args:
            prompt: 用户提示
            scenarioType: 场景类型
            language: 语言
            gender: 性别
            age: 年龄
            voice_id: 自定义发音人ID（可选）

        Returns:
            生成的音频及文本信息
        """
        start_time = time.time()

        try:
            # 第一步：生成文本脚本
            script_result = await self.text_to_script(prompt, scenarioType, language)
            script_text = script_result['transcript']
            script_metadata = script_result['metadata']

            # 第二步：生成音频
            audio_result = await self.script_to_audio(
                text=script_text,
                language=language,
                gender=gender,
                age=age,
                voice_id=voice_id
            )

            # 计算总处理时间
            total_processing_time = time.time() - start_time

            # 构建完整结果
            return {
                "audio_url": audio_result['audio_url'],
                "transcript": script_text,
                "metadata": {
                    "prompt": prompt,
                    "llm_processing_cost": script_metadata['llm_processing_cost'],
                    "language": language,
                    "scenarioType": scenarioType,
                    "gender": gender,
                    "age": age,
                    "voice_id": audio_result['meta']['voice_id'],
                    'generated_at': datetime.now().isoformat(),
                    'processing_time': total_processing_time
                }
            }
        except ValidationError:
            # 直接向上传递验证错误
            raise
        except ExternalAPIError:
            # 直接向上传递API错误
            raise
        except Exception as e:
            logger.error(f"一键生成音频时发生未预期错误: {str(e)}")
            raise InternalServerError(f"一键生成音频时发生未预期错误: {str(e)}")

    async def create_voice(
            self,
            name: str,
            files: List[str] = None,
            description: Optional[str] = None,
            labels: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        添加新声音（声音克隆）

        Args:
            name: 声音名称
            files: 本地音频文件路径列表
            description: 声音描述（可选）
            labels: 标签JSON字符串（可选）

        Returns:
            创建的声音ID及元数据
        """
        start_time = time.time()

        try:
            file_count = len(files) if files else 0
            logger.info(f"开始创建新声音，资源数量：{file_count}")

            # 确保至少有一个资源
            if not files or file_count == 0:
                raise ValidationError("创建声音至少需要一个音频或视频资源")

            # 验证文件路径
            for file_path in files:
                if not os.path.exists(file_path):
                    raise ValidationError(f"文件不存在: {file_path}")

            # 解析JSON标签（如果提供）
            parsed_labels = None
            if labels and isinstance(labels, str):
                try:
                    parsed_labels = json.loads(labels)
                except json.JSONDecodeError:
                    raise ValidationError("提供的标签不是有效的JSON格式")

            # 使用SDK创建声音
            voice_id = await self.elevenLabs.add_voice(
                name=name,
                files=files,
                description=description,
                labels=parsed_labels
            )

            # 计算处理时间
            processing_time = time.time() - start_time

            logger.info(f"成功创建新声音，ID: {voice_id}")

            # 返回结果
            return {
                "voice_id": voice_id,
                "meta": {
                    "name": name,
                    "files": files,
                    "description": description,
                    "labels": parsed_labels or labels,
                    "created_at": datetime.now().isoformat(),
                    "processing_time": processing_time
                }
            }
        except ExternalAPIError as e:
            logger.error(f"添加ElevenLabs声音失败: {str(e)}")
            raise ExternalAPIError(f"添加ElevenLabs声音失败: {str(e)}")
        except ValidationError as e:
            logger.error(f"验证错误: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"创建声音时发生未知错误: {str(e)}")
            raise InternalServerError(f"创建声音时发生未知错误: {str(e)}")


async def run_test():
    """测试函数"""
    agent = AudioGeneratorAgent()
    files = ['recording.mp3']
    result = await agent.create_voice(name="Test Voice", files=files)
    print(result)


if __name__ == "__main__":
    asyncio.run(run_test())